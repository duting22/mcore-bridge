# Copyright (c) ModelScope Contributors. All rights reserved.
"""Microbatch-owned packed layouts. Never cache activations or selections."""
import torch


def boundary_key(cu):
    if cu is None:
        return None
    try:
        version = cu._version
    except RuntimeError:
        return None
    shape = (cu.numel(),) if cu.is_contiguous() else (tuple(cu.shape), tuple(cu.stride()))
    return (cu.untyped_storage()._cdata, cu.storage_offset(), shape, cu.dtype, cu.device, version)


class PackedMetadata:
    """Own host boundaries and all lazy CP/GDN/QSA layouts for one boundary source."""

    def __init__(self, cu, bounds, total_tokens=None):
        self.cu = cu.reshape(-1)
        self.was_normalized = False
        bounds = tuple(bounds)
        if bounds and bounds[0] != 0 and total_tokens is not None:
            if bounds[-1] == total_tokens:
                bounds = (0, *bounds)
                self.cu = torch.cat((self.cu.new_zeros(1), self.cu))
                self.was_normalized = True
            elif bounds[-1] - bounds[0] == total_tokens:
                first = bounds[0]
                bounds = tuple(value - first for value in bounds)
                self.cu = self.cu - first
                self.was_normalized = True
        self.bounds = bounds
        self.total = bounds[-1] if bounds else 0
        self.lengths = tuple(end - start for start, end in zip(bounds, bounds[1:]))
        self.slices = tuple(zip(bounds, bounds[1:]))
        self._cp_layouts = {}
        self._local_slices = {}
        self._qsa = {}

    def validate(self, total, divisor=1):
        return (len(self.bounds) >= 2 and self.bounds[0] == 0 and self.total == total
                and all(length >= 0 and length % divisor == 0 for length in self.lengths))

    def local_slices(self, cp_size):
        if cp_size not in self._local_slices:
            self._local_slices[cp_size] = tuple((start // cp_size, end // cp_size)
                                                for start, end in self.slices)
        return self._local_slices[cp_size]

    def cp_layout(self, cp_size, device, *, total_tokens=None, index_dtype=torch.long):
        """Share CP/SP indices, including eager calls without a known token count."""
        if index_dtype != torch.long:
            raise ValueError('THD CP layout indices must have dtype torch.long.')
        device = torch.device(device)
        key = (cp_size, device, index_dtype, total_tokens)
        if key not in self._cp_layouts:
            for existing_key, layout in self._cp_layouts.items():
                if existing_key[:3] == key[:3] and (total_tokens is None or layout.total_tokens == total_tokens):
                    return layout
            # Import only when needed: cp_layout owns the algorithms, not the cache.
            from .cp_layout import THDCPLayout, _thd_partitioned_indices_torch

            # The first caller may run under checkpoint no_grad or inference_mode.
            with torch.inference_mode(False), torch.no_grad():
                positions = _thd_partitioned_indices_torch(
                    self.cu, cp_size, None, device, total_tokens, host_bounds=self.bounds)
                self._cp_layouts[key] = THDCPLayout(positions)
        return self._cp_layouts[key]

    def qsa(self, compress_ratio, device):
        key = (compress_ratio, torch.device(device))
        if key not in self._qsa:
            with torch.inference_mode(False), torch.no_grad():
                self._qsa[key] = QSAMetadata(self, compress_ratio, device)
        return self._qsa[key]


class QSAMetadata:

    def __init__(self, packed, ratio, device):
        # Keep immutable host data without a reference back to the owning cache.
        self.bounds = packed.bounds
        self.slices = packed.slices
        self.ratio = ratio
        self.counts = tuple(length // ratio for length in packed.lengths)
        offsets = [0]
        for count in self.counts:
            offsets.append(offsets[-1] + count)
        self.offsets = tuple(offsets)
        self.num_blocks = offsets[-1]
        self.max_blocks = max(self.counts, default=0)
        self.full_blocks = torch.tensor(self.counts, dtype=torch.long, device=device)
        block_ids = torch.arange(self.max_blocks, device=device)
        self.block_ids = tuple(block_ids[:count] for count in self.counts)
        lengths = torch.tensor(packed.lengths, dtype=torch.long, device=device)
        docs = torch.arange(len(packed.lengths), device=device)
        self.token_doc = torch.repeat_interleave(docs, lengths, output_size=packed.total)
        cu = packed.cu.to(device=device, dtype=torch.long)
        self.pos_in_doc = torch.arange(packed.total, device=device) - cu.index_select(0, self.token_doc)
        self.visible_blocks = (self.pos_in_doc + 1) // ratio
        self.first_pack = torch.tensor(
            [start + block * ratio for (start, _), count in zip(packed.slices, self.counts)
             for block in range(count)], dtype=torch.long, device=device)
        # Avoid dynamic boolean indexing of raw_keys on every QSA layer.
        flat = torch.arange(self.num_blocks * ratio, device=device)
        self.full_block_ids = flat // ratio
        self.full_rows = self.first_pack.index_select(0, self.full_block_ids) + flat % ratio
        self._lightning = {}

    def lightning(self, topk):
        if topk not in self._lightning:
            with torch.inference_mode(False), torch.no_grad():
                self._lightning[topk] = self._build_lightning(topk)
        return self._lightning[topk]

    def _build_lightning(self, topk):
        docs = tuple(doc for doc, count in enumerate(self.counts) if count > topk)
        cuq, cuk, residuals, rows = [0], [0], [], []
        for doc in docs:
            start, end = self.slices[doc]
            cuq.append(cuq[-1] + end - start)
            cuk.append(cuk[-1] + self.counts[doc])
            residuals.append((end - start) % self.ratio)
            rows.append(torch.arange(start, end, device=self.full_blocks.device))
        device = self.full_blocks.device
        return dict(
            docs=docs, cuq=tuple(cuq), cuk=tuple(cuk),
            rows=torch.cat(rows) if rows else torch.empty(0, dtype=torch.long, device=device),
            cuq_t=torch.tensor(cuq, dtype=torch.int32, device=device),
            cuk_t=torch.tensor(cuk, dtype=torch.int32, device=device),
            residual_t=torch.tensor(residuals, dtype=torch.int32, device=device),
            max_q=max((b - a for a, b in zip(cuq, cuq[1:])), default=0),
            max_k=max((self.counts[doc] for doc in docs), default=0))


class _PackedCache:

    def __init__(self, sources, signature):
        self.sources = sources  # Keep storage identities alive without a global cache.
        self.signature = signature
        self.host_bounds = {}
        self.entries = {}


def get_packed_metadata(packed_seq_params, total_tokens=None, *, padded=True, cu_seqlens=None):
    """Share boundaries across CP reconstruction, GDN, QSA and recomputation.

    Equivalent contiguous views share a key. Shallow PackedSeqParams copies keep
    the cache reference; changed sources replace only the current owner's cache.
    Inference tensors without version counters deliberately bypass reuse.
    """
    if packed_seq_params is None:
        if cu_seqlens is None:
            raise ValueError('Packed metadata requires sequence boundaries.')
        with torch.inference_mode(False), torch.no_grad():
            return PackedMetadata(cu_seqlens, cu_seqlens.detach().reshape(-1).cpu().tolist(), total_tokens)
    actual = packed_seq_params.cu_seqlens_q
    padded_cu = getattr(packed_seq_params, 'cu_seqlens_q_padded', None)
    sources = (actual, padded_cu)
    cu = padded_cu if padded and padded_cu is not None else actual
    if cu is None:
        raise ValueError('Packed metadata requires cu_seqlens_q or cu_seqlens_q_padded.')
    signature = tuple(boundary_key(source) for source in sources)
    reusable = all(source is None or key is not None for source, key in zip(sources, signature))
    cache = getattr(packed_seq_params, '_bridge_packed_metadata_cache', None)
    if not reusable or cache is None or cache.signature != signature:
        cache = _PackedCache(sources, signature)
        packed_seq_params._bridge_packed_metadata_cache = cache
    source_key = boundary_key(cu)
    key = (source_key, total_tokens)
    if key not in cache.entries:
        for existing_key, entry in cache.entries.items():
            if existing_key[0] == source_key and entry.bounds and entry.bounds[0] == 0 \
                    and (total_tokens is not None or not entry.was_normalized) \
                    and (total_tokens is None or entry.total == total_tokens):
                return entry
        if source_key not in cache.host_bounds:
            cache.host_bounds[source_key] = tuple(cu.detach().reshape(-1).cpu().tolist())
        with torch.inference_mode(False), torch.no_grad():
            cache.entries[key] = PackedMetadata(cu, cache.host_bounds[source_key], total_tokens)
    return cache.entries[key]
