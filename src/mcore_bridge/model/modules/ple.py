# Copyright (c) ModelScope Contributors. All rights reserved.
import copy
import math
import torch
import torch.nn.functional as F
from megatron.core import parallel_state
from megatron.core.dist_checkpointing.mapping import ShardedTensor
from megatron.core.extensions.transformer_engine import TELinear
from megatron.core.ssm.mamba_context_parallel import _undo_attention_load_balancing
from megatron.core.tensor_parallel.layers import set_tensor_model_parallel_attributes
from megatron.core.tensor_parallel.mappings import all_to_all, gather_from_sequence_parallel_region
from megatron.core.tensor_parallel.random import get_cuda_rng_tracker, get_expert_parallel_rng_tracker_name
from megatron.core.transformer.utils import (ensure_metadata_has_dp_cp_group, make_sharded_tensors_for_checkpoint,
                                            sharded_state_dict_default)
from torch import nn
from transformers.utils import is_torch_npu_available
from typing import List, Optional

from ...utils import get_env_args, get_logger
from ...utils.constants import EXPORT_CHUNK_BYTES
from ...utils.megatron_utils import (_thd_partitioned_indices_torch, _undo_attention_load_balancing_thd_torch,
                                    get_num_samples, reconstruct_tensor_cp)
from .hyper_connection_gated import Qwen4ExpTextGroupedRMSNorm
from .kernels import gather_ple_rows, ple_gate_conv_triton

_MASK64 = (1 << 64) - 1
# Cap the per-all_reduce GPU buffer when exporting the (potentially huge) ngram
# table.
_EXPORT_CHUNK_BYTES = EXPORT_CHUNK_BYTES
_SPLITMIX_GAMMA = 0x9E3779B97F4A7C15
_SPLITMIX_M1 = 0xBF58476D1CE4E5B9
_SPLITMIX_M2 = 0x94D049BB133111EB
_PRIME_1 = 10007


def use_ple_cpu_offload() -> bool:
    return get_env_args('PLE_CPU_OFFLOAD', bool, False)


def use_ple_fused_kernel() -> bool:
    # On by default; PLE_FUSED_KERNEL=0 forces the eager chain.
    return get_env_args('PLE_FUSED_KERNEL', bool, True)


def _splitmix64(value: int) -> int:
    value = (value + _SPLITMIX_GAMMA) & _MASK64
    value = ((value ^ (value >> 30)) * _SPLITMIX_M1) & _MASK64
    value = ((value ^ (value >> 27)) * _SPLITMIX_M2) & _MASK64
    return (value ^ (value >> 31)) & _MASK64


def _build_layer_multipliers(unigram_vocab_size: int, ngram_size: int, ple_layer_index: int, seed: int) -> List[int]:
    max_long = (1 << 63) - 1
    multiplier_max = max_long // max(unigram_vocab_size, 1)
    half_bound = max(1, multiplier_max // 2)
    base_seed = seed + _PRIME_1 * ple_layer_index
    multipliers = []
    for index in range(ngram_size):
        value = (base_seed + _SPLITMIX_GAMMA * (index + 1)) & _MASK64
        multipliers.append(2 * (_splitmix64(value) % half_bound) + 1)
    return multipliers


def _is_prime_64(value: int) -> bool:
    if value < 2:
        return False
    for prime in (2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37):
        if value % prime == 0:
            return value == prime
    exponent = value - 1
    shifts = 0
    while exponent % 2 == 0:
        exponent //= 2
        shifts += 1
    for base in (2, 325, 9375, 28178, 450775, 9780504, 1795265022):
        if base % value == 0:
            continue
        witness = pow(base, exponent, value)
        if witness in (1, value - 1):
            continue
        for _ in range(shifts - 1):
            witness = pow(witness, 2, value)
            if witness == value - 1:
                break
        else:
            return False
    return True


def _nth_prime_after(start: int, count: int) -> int:
    # Mirrors transformers `_find_nth_prime_after`.
    prime = int(start)
    for _ in range(count):
        candidate = prime + 1
        if candidate <= 2:
            prime = 2
            continue
        if candidate % 2 == 0:
            candidate += 1
        while not _is_prime_64(candidate):
            candidate += 2
        prime = candidate
    return prime


class _EPRowParallelEmbedding(nn.Module):
    """Disjoint table rows on EP; replicas and optimizer states use expert-DP."""

    def __init__(self, config, vocab_size: int, head_dim: int):
        super().__init__()
        if config.expert_tensor_parallel_size != 1:
            raise ValueError('PLE EP row sharding requires expert_tensor_parallel_size=1.')
        self.ep_group = parallel_state.get_expert_model_parallel_group()
        self.ep_rank = self.ep_group.rank()
        self.ep_size = self.ep_group.size()
        if vocab_size % self.ep_size:
            raise ValueError(f'PLE padded vocab size {vocab_size} must be divisible by EP size {self.ep_size}.')
        rows_per_rank = vocab_size // self.ep_size
        self.vocab_start_index = self.ep_rank * rows_per_rank
        self.vocab_end_index = self.vocab_start_index + rows_per_rank
        device = 'cpu' if config.use_cpu_initialization else torch.cuda.current_device()
        self.weight = nn.Parameter(torch.empty(rows_per_rank, head_dim, dtype=config.params_dtype, device=device))
        # This also prevents gnorm from treating distinct EP shards as dense-TP replicas.
        # Checkpoint offsets below explicitly describe EP, independently of this flag.
        set_tensor_model_parallel_attributes(self.weight, is_parallel=True, dim=0, stride=1)
        self.weight.allreduce = False
        if config.perform_initialization:
            with get_cuda_rng_tracker().fork(get_expert_parallel_rng_tracker_name()):
                if self.weight.device.type == 'cpu':
                    # Initialize only the local shard, without consuming the CPU/default RNG.
                    initialized = torch.empty_like(self.weight, device=torch.cuda.current_device())
                    nn.init.normal_(initialized)
                    with torch.no_grad():
                        self.weight.copy_(initialized)
                else:
                    nn.init.normal_(self.weight)

    def forward(self, local_rows: torch.Tensor) -> torch.Tensor:
        # Empty owner input must still connect the parameter to the response all-to-all.
        return F.embedding(local_rows, self.weight)

    def sharded_state_dict(self, prefix='', sharded_offsets=(), metadata=None):
        axis = len(sharded_offsets)
        return {
            f'{prefix}weight': ShardedTensor.from_rank_offsets(
                f'{prefix}weight',
                self.weight,
                *sharded_offsets,
                (axis, self.ep_rank, self.ep_size),
                prepend_axis_num=axis,
                replica_id=(0, 0, parallel_state.get_expert_data_parallel_rank()),
            )
        }


class Qwen4ExpTextNGramEmbedding(nn.Module):

    def __init__(self, config, ple_layer_index: int):
        super().__init__()
        self.ngram_size = config.ngram_size
        self.context_len = self.ngram_size - 1
        self.heads_per_ngram = config.heads_per_ngram
        self.ngram_heads = (self.ngram_size - 1) * self.heads_per_ngram
        self.ple_layer_index = ple_layer_index
        # (transformers also stores unigram_vocab_size/ngram_vocab_size_base/
        # seed as attributes, consumed only by its deferred _init_weights
        # buffer population; here the buffers are registered at construction,
        # so no copies are kept.)
        head_dim_per_ngram = config.ple_embed_dim // self.ngram_heads
        # mcore-specific fail-loud: these drive the hash math and the checkpoint
        # shard layout, so they must come from the model config (the parser
        # validates them for qwen4_exp); a silently substituted default would
        # corrupt the weight conversion. (transformers reads eos_token_id
        # directly and has no split_ngram_parts: its table is replicated.)
        eos_token_id = config.eos_token_id
        split_ngram_parts = config.split_ngram_parts
        ple_seed = getattr(config, 'ple_seed', None)
        if ple_seed is None:
            raise ValueError('ple_seed must be provided by the model config (the parser derives it from '
                             'text_config.seed); a silently substituted default would desynchronize the '
                             'n-gram hash multipliers from transformers.')
        if eos_token_id is None or split_ngram_parts is None:
            raise ValueError(f'eos_token_id/split_ngram_parts must be provided by the model '
                             f'config (got {eos_token_id!r}/{split_ngram_parts!r}).')
        self.eos_token_id = int(eos_token_id)
        self.split_ngram_parts = int(split_ngram_parts)
        self.head_dim = head_dim_per_ngram  # mcore-specific: the bridge weight conversion reads it off the module.

        # Multipliers (splitmix64 derived, checkpoint-persistent). The seed comes
        # from the config (parser defaults it to Qwen4ExpTextConfig.seed's 1234);
        # the multipliers must stay bit-identical to transformers or the n-gram
        # lookups desynchronize.
        multipliers = _build_layer_multipliers(config.padded_vocab_size, self.ngram_size, self.ple_layer_index,
                                               int(ple_seed))
        self.register_buffer('layer_multipliers', torch.tensor(multipliers, dtype=torch.long), persistent=True)

        # Per-head prime table sizes/offsets (checkpoint-persistent), named as
        # in transformers.
        self.head_vocab_sizes = []
        self.head_offsets = []
        self.total_vocab_size = 0
        for head_idx in range(self.ngram_heads):
            global_head_idx = self.ple_layer_index * self.ngram_heads + head_idx
            size = _nth_prime_after(config.ngram_vocab_size_base - 1, global_head_idx + 1)
            self.head_vocab_sizes.append(size)
            self.head_offsets.append(self.total_vocab_size)
            self.total_vocab_size += size
        self.register_buffer(
            'ngram_heads_vocab_sizes', torch.tensor(self.head_vocab_sizes, dtype=torch.long), persistent=True)
        self.register_buffer('ngram_heads_offsets', torch.tensor(self.head_offsets, dtype=torch.long), persistent=True)
        ngram_vocab_divisor = config.make_ngram_vocab_size_divisible_by
        padded_vocab_size = math.ceil(self.total_vocab_size / ngram_vocab_divisor) * ngram_vocab_divisor
        self.padded_vocab_size = padded_vocab_size
        self.cpu_offload = use_ple_cpu_offload()
        if self.cpu_offload:
            raise NotImplementedError('PLE EP training requires PLE_CPU_OFFLOAD=0; optimizer CPU offload is supported.')
        self.ngram_embedding = _EPRowParallelEmbedding(config, padded_vocab_size, head_dim_per_ngram)

    def sharded_state_dict(self, prefix='', sharded_offsets=(), metadata=None):
        metadata = ensure_metadata_has_dp_cp_group(metadata)
        tp_group = parallel_state.get_tensor_model_parallel_group()
        # Hash buffers keep their dense replica layout; the table describes its own EP offsets.
        state_dict = {}
        self._save_to_state_dict(state_dict, '', keep_vars=True)
        state_dict = make_sharded_tensors_for_checkpoint(
            state_dict, prefix, sharded_offsets=sharded_offsets, tp_group=tp_group, dp_cp_group=metadata['dp_cp_group'])
        for name, module in self.named_children():
            state_dict.update(
                sharded_state_dict_default(
                    module, f'{prefix}{name}.', sharded_offsets, metadata, tp_group=tp_group))
        return state_dict

    # The PLE ngram embedding table is stored in the checkpoint as F8_E4M3 shards
    # plus a single scalar `weight_scale` (unlike experts, which use blockwise
    # `weight_scale_inv`). The true values are `weight * scale`.
    _NGRAM_SCALE_KEY = 'ple.ple_embedding.ngram_embedding.weight_scale'

    def fill_table_from_hf(self, hf_state_dict):
        """Populate the local row partition of the n-gram table from the HF
        checkpoint shards at load time. The embedding exposes its actual EP row range.
        """
        total = self.padded_vocab_size
        parts = self.split_ngram_parts
        shard_size = (total + parts - 1) // parts
        if self.cpu_offload:
            local_start, local_end = self.vocab_start, self.vocab_end
            dtype = self.host_table.dtype
        else:
            emb = self.ngram_embedding
            local_start = emb.vocab_start_index
            local_end = emb.vocab_end_index
            dtype = emb.weight.dtype
        # The fp8 shards must be multiplied by the scalar `weight_scale` before
        # being written into the bf16 table. Checkpoints that store the table in
        # bf16 directly have no such key; keep the raw values then (with a
        # warning). Stashed so export_table_to_hf can re-quantize with it.
        scale = None
        if self._NGRAM_SCALE_KEY in hf_state_dict:
            scale = hf_state_dict[self._NGRAM_SCALE_KEY].load().to(torch.float32)
        else:
            get_logger().warning(f'`{self._NGRAM_SCALE_KEY}` not found in the checkpoint; assuming the PLE ngram '
                                 'embedding is already dequantized and loading it as-is.')
        self._ngram_weight_scale = scale
        for i in range(parts):
            key = f'ple.ple_embedding.ngram_embedding.shard_{i}.weight'
            if key not in hf_state_dict:
                continue
            cs, ce = i * shard_size, min((i + 1) * shard_size, total)
            s, e = max(cs, local_start), min(ce, local_end)
            if s >= e:
                continue
            weight = hf_state_dict[key].load()
            if scale is not None:
                weight = weight.to(torch.float32) * scale
            if self.cpu_offload:
                self.host_table[s - local_start:e - local_start] = weight[s - cs:e - cs].to(
                    dtype=dtype, device=self.host_table.device)
            else:
                emb.weight.data[s - local_start:e - local_start] = weight[s - cs:e - cs].to(dtype)

    @torch.no_grad()
    def iter_export_table_to_hf(self, prefix=''):
        """Generator variant of ``export_table_to_hf``: produce one shard at a
        time so the 100GB-scale table never accumulates on the exporting rank --
        the consumer (the safetensors writer) drives the pace and each shard can
        be released once written. Every EP rank yields one item per shard;
        non-leaders use None so callers advance collectives in the same order.
        """
        with torch.no_grad():
            total = self.padded_vocab_size
            parts = self.split_ngram_parts
            shard_size = (total + parts - 1) // parts
            if self.cpu_offload:
                table = self.host_table
                local_start, local_end = self.vocab_start, self.vocab_end
                export_group = parallel_state.get_tensor_model_parallel_group()
                device = torch.cuda.current_device()
            else:
                emb = self.ngram_embedding
                table = emb.weight
                local_start = emb.vocab_start_index
                local_end = emb.vocab_end_index
                export_group = emb.ep_group
                device = table.device
            export_rank = export_group.rank()
            export_size = export_group.size()
            # Inverse of fill_table_from_hf: the checkpoint format is fp8 shards +
            # scalar `weight_scale`, so divide by the scale stashed during loading
            # and cast back to fp8. Without a known scale the values cannot be
            # represented as fp8 + scale; keep the current dtype and warn.
            # Updated BF16 parameters cannot be losslessly requantized with the
            # original checkpoint scale. Only immutable host tables reuse it.
            scale = getattr(self, '_ngram_weight_scale', None) if self.cpu_offload else None
            if scale is None and self.cpu_offload:
                get_logger().warning(f'`{self._NGRAM_SCALE_KEY}` was not seen during loading; exporting the PLE ngram '
                                     'embedding without re-quantizing to fp8.')
            # Reduce on GPU: NCCL has no CPU backend, and the host table is pinned
            # CPU. Each rank scatters its owned rows into shard chunks, sums across
            # the table's shard group (rows are disjoint, so sum == gather), then rank 0 assembles the
            # shard on CPU chunk by chunk. Chunking bounds the GPU transient: a
            # shard can be far larger than the free VRAM left while the training
            # state is still resident.
            elem_size = torch.tensor([], dtype=table.dtype).element_size()
            chunk_rows = max(1, _EXPORT_CHUNK_BYTES // max(1, table.shape[-1] * elem_size))
            out_dtype = torch.float8_e4m3fn if scale is not None else table.dtype
            for i in range(parts):
                cs, ce = i * shard_size, min((i + 1) * shard_size, total)
                # Accumulate the shard on CPU chunk by chunk; assigning inside the
                # loop would keep only the last chunk of each shard.
                shard = None
                if export_rank == 0:
                    shard = torch.zeros(ce - cs, table.shape[-1], dtype=out_dtype, device='cpu')
                for start in range(cs, ce, chunk_rows):
                    end = min(ce, start + chunk_rows)
                    local = torch.zeros(end - start, table.shape[-1], dtype=table.dtype, device=device)
                    s, e = max(start, local_start), min(end, local_end)
                    if s < e:
                        local[s - start:e - start] = table[s - local_start:e - local_start].to(device)
                    if export_size > 1:
                        torch.distributed.all_reduce(local, group=export_group)
                    if export_rank == 0:
                        # Re-quantize after the all_reduce: fp8 is not a valid accumulation
                        # dtype for NCCL, and the sum must happen in the loaded dtype.
                        if scale is not None:
                            local = (local.to(torch.float32) / scale.to(local.device)).to(torch.float8_e4m3fn)
                        shard[start - cs:end - cs] = local.cpu()
                yield f'{prefix}ple.ple_embedding.ngram_embedding.shard_{i}.weight', shard
            if export_rank == 0 and scale is not None:
                yield f'{prefix}{self._NGRAM_SCALE_KEY}', scale.reshape(())

    @torch.no_grad()
    def export_table_to_hf(self, hf_state_dict, prefix=''):
        """Reverse of ``fill_table_from_hf``: write the current table back as HF
        shards so a full-parameter checkpoint is self-contained. The trainable
        table is EP-sharded and is exported in its trained dtype. Bulk API: consumes
        :meth:`iter_export_table_to_hf` into ``hf_state_dict``.
        """
        for key, tensor in self.iter_export_table_to_hf(prefix):
            if tensor is not None:
                hf_state_dict[key] = tensor

    def _shift_right_ignore_eos(self, token_ids: torch.Tensor, shift: int) -> torch.Tensor:
        # Mirrors transformers `_shift_right_ignore_eos`: segment-aware shift,
        # segments are reset after every eos token (request boundary logic).
        if shift == 0:
            return token_ids
        batch_size, seq_len = token_ids.shape
        positions = torch.arange(seq_len, device=token_ids.device, dtype=torch.long)
        eos_positions = torch.where(token_ids == self.eos_token_id, positions, -1)
        previous_eos_inclusive = torch.cummax(eos_positions, dim=1).values
        previous_eos = torch.cat([eos_positions.new_full((batch_size, 1), -1), previous_eos_inclusive[:, :-1]], dim=1)
        segment_start = previous_eos + 1
        position_in_segment = positions.unsqueeze(0) - segment_start
        source_positions = positions - shift
        gather_positions = source_positions.clamp_min(0).unsqueeze(0).expand(batch_size, -1)
        shifted = token_ids.gather(dim=1, index=gather_positions)
        valid = (position_in_segment >= shift) & (source_positions.unsqueeze(0) >= 0)
        return torch.where(valid, shifted, token_ids.new_full((), self.eos_token_id))

    def _build_ngram_ids(self, input_ids: torch.Tensor, packed_seq_lens: Optional[List[int]] = None) -> torch.Tensor:
        """Construct global row IDs before selecting CP/SP-local requests.

        Training variant of transformers ``forward(input_ids, past_key_values,
        layer_idx)``: unpacked inputs prepend eos history for a fresh request;
        packed inputs reset that history at each segment boundary.
        """
        input_ids = input_ids.long()
        # mcore-specific: training has no conv-state cache; a fresh request
        # starts from eos context, so prepend it (the no-previous-state branch
        # of transformers `forward` returns exactly this). The latest HF
        # constructor also takes a cache-only `layer_idx`, which has no
        # training counterpart.
        if packed_seq_lens is not None:
            # Reset history at packed boundaries as well as after EOS tokens.
            # Concatenate the histories before hashing so lookup communicates once.
            segments = input_ids.split(packed_seq_lens, dim=1)
            shifted_tokens = [
                torch.cat([self._shift_right_ignore_eos(segment, shift) for segment in segments], dim=1)
                for shift in range(self.ngram_size)
            ]
        else:
            previous_context = input_ids.new_full((input_ids.shape[0], self.context_len), self.eos_token_id)
            token_history = torch.cat([previous_context, input_ids], dim=-1)
            shifted_tokens = [self._shift_right_ignore_eos(token_history, shift) for shift in range(self.ngram_size)]

        blocks = []
        for ngram in range(2, self.ngram_size + 1):
            start_idx = (ngram - 2) * self.heads_per_ngram
            end_idx = start_idx + self.heads_per_ngram
            mixed_ids = shifted_tokens[0] * self.layer_multipliers[0]
            for position in range(1, ngram):
                mixed_ids = torch.bitwise_xor(
                    mixed_ids,
                    shifted_tokens[position] * self.layer_multipliers[position],
                )
            head_vocab_sizes = self.ngram_heads_vocab_sizes[start_idx:end_idx]
            head_offsets = self.ngram_heads_offsets[start_idx:end_idx]
            ngram_ids = torch.remainder(mixed_ids.unsqueeze(-1), head_vocab_sizes.view(1, 1, -1))
            blocks.append(ngram_ids + head_offsets.view(1, 1, -1))

        return torch.cat(blocks, dim=-1)[:, -input_ids.shape[1]:]

    def _ep_lookup(self, ngram_ids: torch.Tensor) -> torch.Tensor:
        """Route IDs to row owners and return differentiable vectors in request order."""
        emb = self.ngram_embedding
        if emb.ep_size == 1:
            return emb(ngram_ids).flatten(-2)
        ids = ngram_ids.reshape(-1)
        rows_per_rank = emb.weight.shape[0]
        owners = ids // rows_per_rank
        permutation = torch.argsort(owners)
        send_counts = torch.bincount(owners, minlength=emb.ep_size)
        recv_counts = torch.empty_like(send_counts)
        torch.distributed.all_to_all_single(recv_counts, send_counts, group=emb.ep_group)
        # One host transfer for both split lists, never one .item() per peer.
        send_counts, recv_counts = torch.stack((send_counts, recv_counts)).cpu().tolist()
        received_ids = ids.new_empty(sum(recv_counts))
        torch.distributed.all_to_all_single(
            received_ids,
            ids.index_select(0, permutation),
            output_split_sizes=recv_counts,
            input_split_sizes=send_counts,
            group=emb.ep_group,
        )
        owner_values = emb(received_ids - emb.vocab_start_index)
        # Response splits reverse the ID exchange; backward sends gradients to owners.
        returned = all_to_all(emb.ep_group, owner_values, send_counts, recv_counts)
        inverse = torch.empty_like(permutation)
        inverse[permutation] = torch.arange(ids.numel(), device=ids.device)
        return returned.index_select(0, inverse).reshape(*ngram_ids.shape, self.head_dim).flatten(-2)

    def forward(self,
                input_ids: torch.Tensor,
                packed_seq_lens: Optional[List[int]] = None,
                *,
                local_token_indices: Optional[torch.Tensor] = None) -> torch.Tensor:
        real_end = sum(packed_seq_lens) if packed_seq_lens is not None else input_ids.shape[1]
        ngram_ids = self._build_ngram_ids(input_ids[:, :real_end], packed_seq_lens)
        if local_token_indices is None:
            return self._ep_lookup(ngram_ids)
        local_positions = torch.nonzero(local_token_indices < real_end, as_tuple=True)[0]
        selected = local_token_indices.index_select(0, local_positions)
        embeddings = self._ep_lookup(ngram_ids.index_select(1, selected))
        output = embeddings.new_zeros(input_ids.shape[0], local_token_indices.numel(), self.ngram_heads * self.head_dim)
        if local_positions.numel() == 0:
            # No scatter is needed for an empty source. A standalone zero tensor
            # would skip response all-to-all backward, so retain that dependency.
            # Include a zero element to avoid an empty reduction on the backend.
            dependency = torch.cat((embeddings.reshape(-1), embeddings.new_zeros(1))).sum() * 0
            return output + dependency
        return output.index_copy(1, local_positions, embeddings)

    def _init_host_table(self, config, padded_vocab_size, head_dim):
        tp_size = parallel_state.get_tensor_model_parallel_world_size()
        tp_rank = parallel_state.get_tensor_model_parallel_rank()
        per_partition = (padded_vocab_size + tp_size - 1) // tp_size
        self.vocab_start = tp_rank * per_partition
        self.vocab_end = min(padded_vocab_size, self.vocab_start + per_partition)
        n_local = max(0, self.vocab_end - self.vocab_start)
        self.host_table = torch.empty(n_local, head_dim, dtype=config.params_dtype, pin_memory=True)
        self._tp_size = tp_size
        self._tp_group = parallel_state.get_tensor_model_parallel_group()

    def _host_lookup(self, ngram_ids):
        # ngram_ids: [rows, L, nH] global ids in [0, padded_vocab_size).
        # Triton fast path reads rows straight out of the pinned host table;
        # otherwise fall back to a torch gather. Either way, all-reduce across
        # TP (partitions are disjoint, so the sum is the full embedding).
        rows = gather_ple_rows(self.host_table, ngram_ids.reshape(-1), self.vocab_start, self.vocab_end)
        if rows is not None:
            rows = rows.view(*ngram_ids.shape, self.host_table.shape[-1])
            if self._tp_size > 1:
                torch.distributed.all_reduce(rows, group=self._tp_group)
            return rows.flatten(-2)
        ids_cpu = ngram_ids.detach().to('cpu')
        local_mask = (ids_cpu >= self.vocab_start) & (ids_cpu < self.vocab_end)
        local_ids = (ids_cpu - self.vocab_start).clamp(0, max(self.host_table.shape[0] - 1, 0))
        rows = self.host_table[local_ids]  # [rows, L, nH, head_dim]
        rows = rows * local_mask.unsqueeze(-1).to(rows.dtype)
        rows = rows.to(ngram_ids.device, non_blocking=True)
        if self._tp_size > 1:
            torch.distributed.all_reduce(rows, group=self._tp_group)
        return rows.flatten(-2)


class Qwen4ExpTextPLELayer(nn.Module):
    """Inject hashed n-gram features into every hyper-connection stream;
    mirrors transformers ``Qwen4ExpTextPLELayer`` (training variant without
    the inference cache and ``conv_mask``).

    PLE projects each token's concatenated n-gram embedding to a shared value
    and one key per residual stream. The normalized stream activations gate
    those values, then a dilated depthwise convolution adds local lexical
    context. The returned tensor has width ``hc_count * hidden_size``.

    Checkpoint names under the layer prefix ``ple.``:
        ple_embedding.{layer_multipliers,ngram_heads_offsets,ngram_heads_vocab_sizes}
        ple_embedding.ngram_embedding.shard_{i}.weight
        key_proj.weight / value_proj.weight
        norm_key.weight / norm_query.weight / norm_conv.weight
        conv1d.weight
    """

    def sharded_state_dict(self, prefix='', sharded_offsets=(), metadata=None):
        # Visit the embedding's sharded_state_dict instead of flattening its weights.
        tp_group = parallel_state.get_tensor_model_parallel_group()
        state_dict = {}
        for name, module in self.named_children():
            state_dict.update(
                sharded_state_dict_default(module, f'{prefix}{name}.', sharded_offsets, metadata, tp_group=tp_group))
        return state_dict

    def __init__(self, config, ple_layer_index: int, pg_collection=None):
        super().__init__()
        self.config = config
        self.pg_collection = pg_collection
        self.tp_group = pg_collection.tp if pg_collection is not None else parallel_state.get_tensor_model_parallel_group()
        self.cp_group = parallel_state.get_context_parallel_group()
        if self.tp_group.size() > 1 and not config.sequence_parallel:
            raise ValueError('PLE EP training requires sequence_parallel=True when TP>1.')
        self.hidden_size = int(config.hidden_size)
        self.hc_count = int(config.hc_count)
        ple_embed_dim = int(config.ple_embed_dim)
        hc_hidden_size = self.hidden_size * self.hc_count
        self.ple_embedding = Qwen4ExpTextNGramEmbedding(config, ple_layer_index)
        conv_kernel_size = int(config.ple_conv_kernel_size)
        conv_dilation = int(config.ngram_size)
        self.short_conv_state_len = (conv_kernel_size - 1) * conv_dilation
        # Replicated projections in params_dtype.
        self.key_proj = TELinear(
            input_size=ple_embed_dim,
            output_size=hc_hidden_size,
            parallel_mode='duplicated',
            config=config,
            init_method=config.init_method,
            bias=False,
            skip_bias_add=True,
            skip_weight_param_allocation=False,
        )
        self.value_proj = TELinear(
            input_size=ple_embed_dim,
            output_size=self.hidden_size,
            parallel_mode='duplicated',
            config=config,
            init_method=config.init_method,
            bias=False,
            skip_bias_add=True,
            skip_weight_param_allocation=False,
        )
        # mcore's config field layernorm_epsilon corresponds to HF's rms_norm_eps;
        # the grouped norm is the mcore subclass adding dtype/SP-flag construction.
        self.norm_key = Qwen4ExpTextGroupedRMSNorm(
            hc_hidden_size,
            group_size=self.hidden_size,
            eps=config.layernorm_epsilon,
            dtype=config.params_dtype,
            sequence_parallel=config.sequence_parallel)
        self.norm_query = Qwen4ExpTextGroupedRMSNorm(
            hc_hidden_size,
            group_size=self.hidden_size,
            eps=config.layernorm_epsilon,
            dtype=config.params_dtype,
            sequence_parallel=config.sequence_parallel)
        self.norm_conv = Qwen4ExpTextGroupedRMSNorm(
            hc_hidden_size,
            group_size=self.hidden_size,
            eps=config.layernorm_epsilon,
            dtype=config.params_dtype,
            sequence_parallel=config.sequence_parallel)
        self.conv1d = nn.Conv1d(
            hc_hidden_size,
            hc_hidden_size,
            kernel_size=conv_kernel_size,
            groups=hc_hidden_size,
            dilation=conv_dilation,
            bias=False,
            dtype=config.params_dtype,
        )
        nn.init.zeros_(self.conv1d.weight)
        # Duplicated TELinear and the three norms already carry this flag. Their
        # local-token gradients, and the local-output conv gradients, require TP SUM.
        self.conv1d.weight.sequence_parallel = config.sequence_parallel

    def _short_conv(self, hidden_states: torch.Tensor, packed_seq_lens: Optional[List[int]] = None) -> torch.Tensor:
        if packed_seq_lens is not None:
            # Only participating samples are present; alignment padding never
            # launches its own convolution when real-sample metadata is available.
            return torch.cat(
                [self._short_conv(segment) for segment in hidden_states.split(packed_seq_lens, dim=1)], dim=1)
        # Training variant of transformers `_short_conv` (no conv state cache):
        # causal alignment is achieved by left-padding the conv input with
        # zeros (equivalent to trimming the padded conv output), which is a
        # no-op on fresh-request context.
        hidden_states = hidden_states.transpose(1, 2)
        sequence_length = hidden_states.shape[-1]
        conv_input = F.pad(hidden_states, (self.short_conv_state_len, 0))
        conv_input = conv_input[..., -(self.short_conv_state_len + sequence_length):]
        return F.silu(self.conv1d(conv_input)).transpose(1, 2)

    def compute(self, hidden_states: torch.Tensor, embeddings: torch.Tensor):
        """Local [B,L,...] projections/norms/gate; communication precedes convolution."""
        key_normed = self.norm_key(self.key_proj(embeddings)[0]).unflatten(-1, (self.hc_count, self.hidden_size))
        value = self.value_proj(embeddings)[0]
        query_normed = self.norm_query(hidden_states).unflatten(-1, (self.hc_count, self.hidden_size))
        gate = (key_normed * query_normed).sum(dim=-1, keepdim=True) / math.sqrt(self.hidden_size)
        gate = gate.abs().clamp_min(1e-6).sqrt() * gate.sign()
        gated_value = torch.sigmoid(gate) * value.unsqueeze(-2)
        gated_value_normed = self.norm_conv(gated_value.flatten(-2))
        gated_value = gated_value.flatten(-2)
        return gated_value, gated_value_normed

    def _compute_fused(self, hidden_states: torch.Tensor, embeddings: torch.Tensor):
        """Fused triton chain (gate + norm_conv + causal conv + SiLU + residual);
        fp32 accumulation, output dtype = params_dtype. Returns None when the
        kernel path is unavailable so the caller falls back to torch."""
        rows, seq_len = embeddings.shape[:2]
        key = self.key_proj(embeddings)[0].reshape(rows * seq_len, -1)
        value = self.value_proj(embeddings)[0].reshape(rows * seq_len, -1)
        query = hidden_states.reshape(rows * seq_len, -1)
        out = ple_gate_conv_triton(
            query,
            key,
            value,
            self.norm_key.weight,
            self.norm_query.weight,
            self.norm_conv.weight,
            self.conv1d.weight,
            self.hc_count,
            self.config.layernorm_epsilon,
            int(self.config.ngram_size),
            seq_len,
        )
        if out is None:
            return None
        return out.view(rows, seq_len, -1)

    @staticmethod
    def _normalize_cu_seqlens(cu: Optional[torch.Tensor], total: int) -> Optional[torch.Tensor]:
        """Align a (possibly padded/offset) cu_seqlens against the gathered full length.

        Mirrors GatedDeltaNet._resolve_cu_seqlens for the SP-only case: accept a
        global cumulative layout (possibly missing its leading 0 or carrying a
        per-batch offset), and return a cu whose last entry equals ``total`` so it
        indexes the gathered full-sequence tensor directly.
        """
        if cu is None:
            return None
        cu = cu.reshape(-1)
        if cu.numel() > 0 and int(cu[0]) != 0:
            if int(cu[-1]) == total:
                cu = torch.cat([cu.new_zeros(1), cu])
            elif int(cu[-1]) - int(cu[0]) == total:
                cu = cu - cu[0]
        if not (cu.numel() > 0 and int(cu[-1]) == total):
            raise ValueError(f'PLE cannot align cu_seqlens (last={int(cu[-1]) if cu.numel() else None}, '
                             f'first={int(cu[0]) if cu.numel() else None}) with the gathered sequence '
                             f'length {total} under sequence parallelism.')
        return cu

    def _prepare_local_layout(self, hidden_states, input_ids, packed_seq_params):
        if packed_seq_params is None or getattr(packed_seq_params, 'qkv_format', None) != 'thd':
            raise NotImplementedError('PLE EP local-sequence computation currently supports THD inputs only.')
        if (hidden_states.ndim != 3 or hidden_states.shape[1] != 1 or input_ids.ndim != 2
                or input_ids.shape[0] != 1):
            raise ValueError('PLE THD expects local hidden_states [L,1,H] and input_ids [1,T] or [1,T/CP].')
        sp_size = self.tp_group.size() if self.config.sequence_parallel else 1
        cp_size = self.cp_group.size()
        total = hidden_states.shape[0] * sp_size * cp_size
        # Keep dynamically attached num_samples/seq_lens while aligning both boundary views.
        psp = copy.copy(packed_seq_params)
        psp.cu_seqlens_q = self._normalize_cu_seqlens(psp.cu_seqlens_q, total)
        psp.cu_seqlens_q_padded = self._normalize_cu_seqlens(getattr(psp, 'cu_seqlens_q_padded', None), total)
        packed_seq_lens = self._get_packed_seq_lens(psp, total)
        if input_ids.shape[1] != total:
            if cp_size > 1 and input_ids.shape[1] == total // cp_size:
                input_ids = reconstruct_tensor_cp(input_ids, psp, dim=1)
            else:
                raise ValueError(f'PLE input_ids length {input_ids.shape[1]} does not match full length {total} '
                                 f'or CP-local length {total // cp_size}.')
        if cp_size > 1:
            indices = _thd_partitioned_indices_torch(
                psp.cu_seqlens_q, cp_size, self.cp_group.rank(), hidden_states.device)
        else:
            indices = torch.arange(total, device=hidden_states.device)
        if sp_size > 1:
            indices = indices.narrow(0, self.tp_group.rank() * hidden_states.shape[0], hidden_states.shape[0])
        return input_ids, psp, packed_seq_lens, indices

    def forward(self, hidden_states: torch.Tensor, input_ids: torch.Tensor, packed_seq_params=None) -> torch.Tensor:
        """THD: keep hidden/projections local; gather only normalized conv inputs."""
        input_ids, psp, packed_seq_lens, local_indices = self._prepare_local_layout(
            hidden_states, input_ids, packed_seq_params)
        real_end = sum(packed_seq_lens)
        total = input_ids.shape[1]
        embeddings = self.ple_embedding(input_ids, packed_seq_lens, local_token_indices=local_indices)
        gated_value, gated_value_normed = self.compute(hidden_states.transpose(0, 1), embeddings)
        # Megatron gathers DIMENSION ZERO: never pass batch-first [B,L,H] here.
        full_normed = gated_value_normed.transpose(0, 1).contiguous()
        if self.config.sequence_parallel and self.tp_group.size() > 1:
            full_normed = gather_from_sequence_parallel_region(
                full_normed, tensor_parallel_output_grad=True, group=self.tp_group)
        if self.cp_group.size() > 1:
            full_normed = gather_from_sequence_parallel_region(
                full_normed, tensor_parallel_output_grad=True, group=self.cp_group)
            if is_torch_npu_available():
                full_normed = _undo_attention_load_balancing_thd_torch(full_normed, self.cp_group.size(), psp)
            else:
                full_normed = _undo_attention_load_balancing(full_normed, self.cp_group.size(), psp)
        # Every rank executes the conv branch, including ranks with no local real
        # outputs. Indexing/masking must retain both gather backwards in that case.
        full_conv = self._short_conv(
            full_normed.transpose(0, 1).contiguous()[:, :real_end], packed_seq_lens=packed_seq_lens)
        if real_end < total:
            full_conv = F.pad(full_conv, (0, 0, 0, total - real_end))
        local_conv = full_conv.index_select(1, local_indices)
        output = gated_value + local_conv
        output = output.masked_fill((local_indices >= real_end).view(1, -1, 1), 0)
        return output.transpose(0, 1).contiguous()

    @staticmethod
    def _get_packed_seq_lens(packed_seq_params, total: int) -> List[int]:
        """Resolve participating segments in the gathered, original-order layout.

        With real-sample metadata, trailing alignment segments are excluded.
        Otherwise get_num_samples keeps its existing all-segments fallback.
        """
        cu = getattr(packed_seq_params, 'cu_seqlens_q', None)
        if cu is None:
            raise ValueError('PLE THD computation requires cu_seqlens_q.')
        cu_padded = getattr(packed_seq_params, 'cu_seqlens_q_padded', None)
        if cu_padded is not None and cu_padded is not cu and not torch.equal(cu.reshape(-1), cu_padded.reshape(-1)):
            raise ValueError('PLE THD computation requires cu_seqlens_q and cu_seqlens_q_padded '
                             'to describe the same storage layout.')
        # Transfer boundaries once; embedding and convolution reuse Python lengths.
        boundaries = cu.detach().reshape(-1).cpu().tolist()
        if (len(boundaries) < 2 or boundaries[0] != 0 or boundaries[-1] != total
                or any(end <= start for start, end in zip(boundaries[:-1], boundaries[1:]))):
            raise ValueError(f'PLE THD boundaries must start at 0, increase strictly, and end at {total}.')
        num_samples = get_num_samples(packed_seq_params)
        if not 1 <= num_samples < len(boundaries):
            raise ValueError(f'PLE THD segment count {num_samples} is invalid for {len(boundaries) - 1} segments.')
        return [boundaries[i + 1] - boundaries[i] for i in range(num_samples)]
