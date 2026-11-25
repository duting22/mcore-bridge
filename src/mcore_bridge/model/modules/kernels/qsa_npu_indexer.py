# Copyright (c) ModelScope Contributors. All rights reserved.
"""Ascend LightningIndexerV2 scoring for frozen QSA compressed key blocks."""
import math

import torch


def _torch_topk_rows(query, block_keys, row_ids, cu_seqlens, full_blocks,
                     visible_blocks, topk, chunk_bytes, packed_metadata=None):
    """Rescore ambiguous rows per document, preferring earlier blocks on ties.

    Match QSAIndexer's packed Torch backend: both the candidate space and stable
    ordering must follow the current baseline, rather than the old global TopK.
    Callers supply only pruning rows, which have more than ``topk`` visible blocks.
    """
    result = torch.empty((row_ids.numel(), topk), dtype=torch.long, device=query.device)
    bounds = packed_metadata.bounds if packed_metadata is not None else cu_seqlens.tolist()
    counts = packed_metadata.counts if packed_metadata is not None else full_blocks.tolist()
    block_start = 0
    for doc, ((start, end), count) in enumerate(zip(zip(bounds, bounds[1:]), counts)):
        positions = ((row_ids >= start) & (row_ids < end)).nonzero().flatten()
        if positions.numel():
            doc_rows = row_ids.index_select(0, positions)
            keys = block_keys[block_start:block_start + count].float()
            block_ids = (packed_metadata.block_ids[doc] if packed_metadata is not None
                         else torch.arange(count, device=query.device))
            chunk = max(1, chunk_bytes // max(1, query.shape[1] * count * 4))
            for offset in range(0, doc_rows.numel(), chunk):
                rows = doc_rows[offset:offset + chunk]
                scores = torch.einsum('thd,kd->thk', query[rows].float(), keys)
                scores = scores.relu().sum(1) / math.sqrt(query.shape[-1])
                valid = block_ids[None, :] < visible_blocks[rows, None]
                scores.masked_fill_(~valid, -torch.inf)
                selected = torch.sort(scores, dim=-1, descending=True, stable=True).indices[:, :topk]
                result[positions[offset:offset + rows.numel()]] = selected + block_start
        block_start += count
    return result


@torch.no_grad()
def lightning_topk_blocks(query, block_keys, cu_seqlens, full_blocks,
                          visible_blocks, topk, compress_ratio, chunk_bytes, packed_metadata=None):
    """Return pack-global block IDs and keep flags for the packed QSA selector.

    Pooling, normalization, and RoPE are performed by QSAIndexer before this call.
    One extra candidate exposes the TopK boundary. Ambiguous pruning rows use the
    same per-document FP32/stable selection as the current Torch backend.
    """
    if query.device.type != 'npu' or query.dtype not in (torch.bfloat16, torch.float16):
        raise ValueError('Ascend QSA indexer requires BF16/FP16 NPU query/key tensors.')
    if query.shape[-1] != 128 or not 0 <= topk < 2048:
        raise ValueError('Ascend QSA indexer currently requires head_dim=128 and 0 <= block_topk < 2048.')
    selected = torch.zeros((query.shape[0], topk), dtype=torch.long, device=query.device)
    keep = torch.zeros_like(selected, dtype=torch.bool)
    if topk == 0:
        return selected, keep
    try:
        from cann_ops_transformer.ops import lightning_indexer, lightning_indexer_metadata
    except ImportError as error:
        raise ImportError('Add $ASCEND_HOME_PATH/python/site-packages to PYTHONPATH for '
                          'QSA_INDEXER_BACKEND=ascendc.') from error

    if packed_metadata is not None:
        bounds, counts = packed_metadata.bounds, packed_metadata.counts
        block_offsets = packed_metadata.offsets
        plan = packed_metadata.lightning(topk) if packed_metadata.max_blocks > topk else None
        long_docs = plan['docs'] if plan is not None else ()
    else:
        bounds, counts = cu_seqlens.tolist(), full_blocks.tolist()
        block_offsets = [0]
        for count in counts:
            block_offsets.append(block_offsets[-1] + count)
        long_docs = tuple(doc for doc, count in enumerate(counts) if count > topk)
        plan = None
    for doc, (start, end) in enumerate(zip(bounds, bounds[1:])):
        count = counts[doc]
        if 0 < count <= topk:
            local_blocks = (packed_metadata.block_ids[doc] if packed_metadata is not None
                            else torch.arange(count, device=query.device))
            selected[start:end, :count] = block_offsets[doc] + local_blocks
            keep[start:end, :count] = local_blocks[None, :] < visible_blocks[start:end, None]
    if not long_docs:
        return selected, keep

    q = torch.cat([query[bounds[doc]:bounds[doc + 1]] for doc in long_docs]).contiguous()
    k = torch.cat([block_keys[block_offsets[doc]:block_offsets[doc + 1]]
                   for doc in long_docs]).unsqueeze(1).contiguous()
    if plan is None:
        row_ids, cuq, cuk, residuals = [], [0], [0], []
        for doc in long_docs:
            start, end = bounds[doc], bounds[doc + 1]
            row_ids.append(torch.arange(start, end, device=query.device))
            cuq.append(cuq[-1] + end - start)
            cuk.append(cuk[-1] + counts[doc])
            residuals.append((end - start) % compress_ratio)
        rows = torch.cat(row_ids)
        cuq_t = torch.tensor(cuq, device=query.device, dtype=torch.int32)
        cuk_t = torch.tensor(cuk, device=query.device, dtype=torch.int32)
        residual_t = torch.tensor(residuals, device=query.device, dtype=torch.int32)
    else:
        cuq, cuk = plan['cuq'], plan['cuk']
        rows, cuq_t, cuk_t, residual_t = plan['rows'], plan['cuq_t'], plan['cuk_t'], plan['residual_t']
    weights = torch.full(q.shape[:2], 1 / math.sqrt(query.shape[-1]),
                         device=query.device, dtype=torch.float32)
    native_topk = topk + 1
    metadata = lightning_indexer_metadata(
        q.shape[1], 1, q.shape[-1], native_topk,
        cu_seqlens_q=cuq_t, cu_seqlens_k=cuk_t, cmp_residual_k=residual_t,
        batch_size=len(long_docs), max_seqlen_q=plan['max_q'] if plan is not None
        else max(b - a for a, b in zip(cuq, cuq[1:])),
        max_seqlen_k=plan['max_k'] if plan is not None else max(counts[d] for d in long_docs),
        layout_q='TND', layout_k='TND',
        mask_mode=3, cmp_ratio=compress_ratio)
    native_ids, native_values = lightning_indexer(
        q, k, weights, native_topk, cu_seqlens_q=cuq_t, cu_seqlens_k=cuk_t,
        cmp_residual_k=residual_t, metadata=metadata, layout_q='TND', layout_k='TND',
        mask_mode=3, cmp_ratio=compress_ratio, return_value=1)
    native_ids = native_ids.squeeze(1).long()
    native_values = native_values.squeeze(1)
    for i, doc in enumerate(long_docs):
        start, end = bounds[doc], bounds[doc + 1]
        local = native_ids[cuq[i]:cuq[i + 1], :topk]
        valid = (local >= 0) & (local < visible_blocks[start:end, None])
        selected[start:end] = torch.where(valid, local + block_offsets[doc], 0)
        keep[start:end] = valid

    # Preserve stable Torch selection when rounding or ties can change the set.
    pruning = visible_blocks[rows] > topk
    left, right = native_values[:, topk - 1], native_values[:, topk]
    tolerance = 1e-4 * (1.0 + torch.maximum(left.abs(), right.abs()))
    fallback_rows = rows[pruning & ((left - right) <= tolerance)]
    if fallback_rows.numel():
        selected[fallback_rows] = _torch_topk_rows(
            query, block_keys, fallback_rows, cu_seqlens, full_blocks,
            visible_blocks, topk, chunk_bytes, packed_metadata)
        keep[fallback_rows] = True
    return selected, keep
