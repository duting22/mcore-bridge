# Copyright (c) ModelScope Contributors. All rights reserved.
"""Microbatch-owned THD context-parallel index layouts; no activation caching."""
import torch


def _boundary_key(cu):
    if cu is None:
        return None
    try:
        version = cu._version
    except RuntimeError:
        # Inference tensors have no version counter. Do not reuse a layout whose
        # source can change without us being able to detect it.
        return None
    shape = (cu.numel(),) if cu.is_contiguous() else (tuple(cu.shape), tuple(cu.stride()))
    return (cu.untyped_storage()._cdata, cu.storage_offset(), shape, cu.dtype, cu.device, version)


def _thd_partitioned_indices_torch(cu_seqlens, cp_size, cp_rank, device, total_tokens=None):
    """Build all ranks in one batch when cp_rank is None, otherwise return one rank.

    Host validation and the explicit repeat_interleave output size avoid the
    repeated device scalar reads and dynamic output-size inference of the old
    per-rank implementation. Host boundaries are temporary, not cached.
    """
    if cp_size < 1 or (cp_rank is not None and not 0 <= cp_rank < cp_size):
        raise ValueError('Invalid THD context-parallel size or rank.')
    bounds = cu_seqlens.detach().reshape(-1).cpu().tolist()
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


class _LayoutCache:

    def __init__(self, sources, signature):
        # Keep sources alive so storage identities cannot be recycled while
        # entries are alive. Shallow PackedSeqParams copies share this object.
        self.sources = sources
        self.signature = signature
        self.layouts = {}


def get_thd_cp_layout(packed_seq_params, cp_size, device, *, padded=True, total_tokens=None,
                      index_dtype=torch.long):
    """Return the microbatch's cached CP layout, rebuilding after source mutation.

    Equivalent contiguous boundary views share a storage/version key. Actual
    and padded boundaries are separate unless they alias the same source.
    """
    if index_dtype != torch.long:
        # scatter/index_select consumers in this path use int64 indices.
        raise ValueError('THD CP layout indices must have dtype torch.long.')
    actual = packed_seq_params.cu_seqlens_q
    padded_cu = getattr(packed_seq_params, 'cu_seqlens_q_padded', None)
    sources = (actual, padded_cu)
    cu = padded_cu if padded and padded_cu is not None else actual
    if cu is None:
        raise RuntimeError('THD CP requires cu_seqlens_q or cu_seqlens_q_padded.')
    signature = tuple(_boundary_key(source) for source in sources)
    reusable = all(source is None or key is not None for source, key in zip(sources, signature))
    cache = getattr(packed_seq_params, '_bridge_thd_cp_cache', None)
    if not reusable or cache is None or cache.signature != signature:
        cache = _LayoutCache(sources, signature)
        packed_seq_params._bridge_thd_cp_cache = cache
    device = torch.device(device)
    key = (_boundary_key(cu), cp_size, device, index_dtype, total_tokens)
    # Eager preparation and consumers with a known tensor length share an entry.
    if key not in cache.layouts:
        for existing_key, layout in cache.layouts.items():
            if existing_key[:4] == key[:4] and (total_tokens is None or layout.total_tokens == total_tokens):
                return layout
    if key not in cache.layouts:
        # Cached indices must remain usable by autograd even when first built
        # in a no-grad checkpoint forward or an inference_mode context.
        with torch.inference_mode(False), torch.no_grad():
            positions = _thd_partitioned_indices_torch(cu, cp_size, None, device, total_tokens)
            cache.layouts[key] = THDCPLayout(positions)
    return cache.layouts[key]


def prepare_thd_cp_layout(packed_seq_params, config, device):
    """Warm the NPU THD layout before the model's first CP reconstruction/layer."""
    if (packed_seq_params is None or getattr(packed_seq_params, 'qkv_format', None) != 'thd'
            or config.context_parallel_size <= 1 or device is None
            or torch.device(device).type != 'npu'
            or getattr(config, 'cp_partition_mode', 'zigzag') != 'zigzag'):
        return
    get_thd_cp_layout(packed_seq_params, config.context_parallel_size, device)
