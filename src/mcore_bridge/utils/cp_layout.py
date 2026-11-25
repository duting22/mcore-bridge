# Copyright (c) ModelScope Contributors. All rights reserved.
"""THD context-parallel layout algorithms; PackedMetadata owns cached instances."""
import torch

from .packed_metadata import get_packed_metadata


def _thd_partitioned_indices_torch(cu_seqlens, cp_size, cp_rank, device, total_tokens=None, host_bounds=None):
    """Build all ranks in one batch when cp_rank is None, otherwise return one rank.

    Host validation and the explicit repeat_interleave output size avoid the
    repeated device scalar reads and dynamic output-size inference of the old
    per-rank implementation. NPU callers share the microbatch's host boundaries.
    """
    if cp_size < 1 or (cp_rank is not None and not 0 <= cp_rank < cp_size):
        raise ValueError('Invalid THD context-parallel size or rank.')
    bounds = host_bounds if host_bounds is not None else cu_seqlens.detach().reshape(-1).cpu().tolist()
    if bounds and bounds[0] != 0 and total_tokens is not None:
        if bounds[-1] == total_tokens:
            bounds = [0, *bounds]
        elif bounds[-1] - bounds[0] == total_tokens:
            first = bounds[0]
            bounds = [value - first for value in bounds]
    if len(bounds) < 2 or bounds[0] != 0:
        raise RuntimeError('THD CP boundaries must start at zero and contain at least one sequence.')
    lengths = [end - start for start, end in zip(bounds, bounds[1:])]
    if any(length < 0 for length in lengths):
        raise RuntimeError(f'THD CP requires nondecreasing boundaries, got {bounds}.')
    total = bounds[-1]
    if total_tokens is not None and total != total_tokens:
        raise RuntimeError(f'THD CP metadata ends at {total}, but the tensor has {total_tokens} tokens.')
    if cp_size == 1:
        positions = torch.arange(total, dtype=torch.long, device=device).unsqueeze(0)
        return positions if cp_rank is None else positions[cp_rank]
    divisor = 2 * cp_size
    if any(length % divisor for length in lengths):
        raise RuntimeError(f'THD CP requires lengths divisible by 2*cp_size={divisor}, got {lengths}.')
    local_tokens = total // cp_size
    cu = torch.tensor(bounds, dtype=torch.long, device=device)
    local_lengths = (cu[1:] - cu[:-1]) // cp_size
    token_doc = torch.repeat_interleave(
        torch.arange(len(lengths), device=device), local_lengths, output_size=local_tokens)
    starts = cu[:-1].index_select(0, token_doc)
    half = ((cu[1:] - cu[:-1]) // divisor).index_select(0, token_doc)
    offset = torch.arange(local_tokens, device=device) - starts // cp_size
    ranks = torch.arange(cp_size, device=device).unsqueeze(1)
    positions = starts.unsqueeze(0) + torch.where(
        offset.unsqueeze(0) < half.unsqueeze(0),
        ranks * half.unsqueeze(0) + offset.unsqueeze(0),
        (divisor - ranks - 1) * half.unsqueeze(0) + offset.unsqueeze(0) - half.unsqueeze(0),
    )
    return positions if cp_rank is None else positions[cp_rank]


class THDCPLayout:
    """Read-only device index tensors shared by consumers of one physical layout."""

    def __init__(self, positions):
        self.positions = positions
        self.total_tokens = positions.numel()
        self.restore_order = torch.empty(self.total_tokens, dtype=positions.dtype, device=positions.device)
        self.restore_order.scatter_(
            0, positions.reshape(-1),
            torch.arange(self.total_tokens, dtype=positions.dtype, device=positions.device))
        self._local_indices = {}

    def local_indices(self, cp_rank, sp_size=1, sp_rank=0):
        if not 0 <= cp_rank < self.positions.shape[0] or sp_size < 1 or not 0 <= sp_rank < sp_size:
            raise ValueError('Invalid CP/SP size or rank.')
        key = (cp_rank, sp_size, sp_rank)
        if key not in self._local_indices:
            with torch.inference_mode(False), torch.no_grad():
                positions = self.positions[cp_rank]
                if positions.numel() % sp_size:
                    raise ValueError('THD CP-local token count must be divisible by SP size.')
                local_tokens = positions.numel() // sp_size
                self._local_indices[key] = positions.narrow(0, sp_rank * local_tokens, local_tokens)
        return self._local_indices[key]


def get_thd_cp_layout(packed_seq_params, cp_size, device, *, padded=True, total_tokens=None,
                      index_dtype=torch.long):
    """Query the shared packed cache; retain the existing CP normalization contract."""
    if index_dtype != torch.long:
        # scatter/index_select consumers in this path use int64 indices.
        raise ValueError('THD CP layout indices must have dtype torch.long.')
    cu = getattr(packed_seq_params, 'cu_seqlens_q_padded', None) if padded else None
    if cu is None:
        cu = packed_seq_params.cu_seqlens_q
    if cu is None:
        raise RuntimeError('THD CP requires cu_seqlens_q or cu_seqlens_q_padded.')
    # Keep source boundaries unnormalized here. CP applies total_tokens itself,
    # so known-length and eager queries retain their original sharing behavior.
    metadata = get_packed_metadata(packed_seq_params, padded=padded)
    return metadata.cp_layout(cp_size, device, total_tokens=total_tokens, index_dtype=index_dtype)


def prepare_thd_cp_layout(packed_seq_params, config, device):
    """Warm the NPU THD layout before the model's first CP reconstruction/layer."""
    if (packed_seq_params is None or getattr(packed_seq_params, 'qkv_format', None) != 'thd'
            or config.context_parallel_size <= 1 or device is None
            or torch.device(device).type != 'npu'
            or getattr(config, 'cp_partition_mode', 'zigzag') != 'zigzag'):
        return
    get_thd_cp_layout(packed_seq_params, config.context_parallel_size, device)
