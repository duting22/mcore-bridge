# Copyright (c) ModelScope Contributors. All rights reserved.
"""Token-exact QSA masks on the native Ascend training FlashAttention operator.

Each packed document gets its own call: the TND API's single shared SS mask
cannot encode different content-dependent selections for multiple documents.
"""
import torch


def packed_document_boundaries(packed_seq_params, total):
    """Use the same physical token space as Qwen4ExpLayer's packed selector."""
    cu = getattr(packed_seq_params, 'cu_seqlens_q_padded', None)
    if cu is None:
        cu = getattr(packed_seq_params, 'cu_seqlens_q', None)
    if cu is None:
        raise ValueError('Packed QSA needs cu_seqlens_q or cu_seqlens_q_padded.')
    bounds = cu.reshape(-1).tolist() if isinstance(cu, torch.Tensor) else list(cu)
    if bounds and bounds[0] != 0:
        if bounds[-1] == total:
            bounds.insert(0, 0)
        elif bounds[-1] - bounds[0] == total:
            first = bounds[0]
            bounds = [x - first for x in bounds]
    if (not bounds or bounds[0] != 0 or bounds[-1] != total
            or any(b < a for a, b in zip(bounds, bounds[1:]))):
        raise ValueError(f'Cannot align packed QSA boundaries {bounds} with T={total}.')
    return bounds


@torch.no_grad()
def qsa_indices_to_npu_mask(indices, document_start, document_length, query_positions=None):
    """[Lq,K] pack token IDs -> [Lq,Lkv] bool mask, True means excluded.

    A separate sentinel column absorbs -1 slots. In particular, holes before
    valid tail tokens must not mask a real key or terminate the selection.
    """
    if indices.ndim != 2:
        raise ValueError('QSA indices must have one row per document query.')
    if query_positions is None:
        if indices.shape[0] != document_length:
            raise ValueError('Unequal Q/KV lengths need global query positions.')
        query_positions = torch.arange(document_length, device=indices.device) + document_start
    elif query_positions.ndim != 1 or query_positions.numel() != indices.shape[0]:
        raise ValueError('QSA needs one global position for each local query.')
    local = indices - document_start
    rows = (query_positions - document_start).unsqueeze(1)
    valid = (local >= 0) & (local < document_length) & (local <= rows)
    destinations = torch.where(valid, local, document_length).long()
    mask = torch.ones((indices.shape[0], document_length + 1), device=indices.device, dtype=torch.bool)
    mask.scatter_(1, destinations, False)
    return mask[:, :document_length].contiguous()


def _document_attention(query, key, value, indices, document_start, scale, dropout, query_positions=None):
    import torch_npu

    mask = qsa_indices_to_npu_mask(indices, document_start, key.shape[0], query_positions)
    # The original main Q/K have already received QK norm and partial RoPE.
    # Native GQA accepts the local TP ratio, e.g. 3 Q heads : 1 KV head.
    output = torch_npu.npu_fusion_attention(
        query.unsqueeze(0).contiguous(),
        key.unsqueeze(0).contiguous(),
        value.unsqueeze(0).contiguous(),
        head_num=query.shape[1],
        input_layout='BSND',
        atten_mask=mask,
        sparse_mode=1,
        scale=scale,
        keep_prob=1.0 - dropout,
    )[0].squeeze(0)
    return output


def qsa_npu_fusion_attention(query, key, value, indices, scale, packed_seq_params=None, dropout=0.0,
                             query_positions=None, cp_size=1):
    """Local Q attends full K/V with explicit QSA selections and global positions.

    CP callers gather K/V with autograd before entering here. Packed local Q is
    still document-major, with each document's front and mirrored back chunks.
    No zero-filled full-length Q buffer is necessary for native FA.
    """
    if query.device.type != 'npu':
        raise ValueError('QSA_ATTN_BACKEND=npu_fa requires Ascend NPU tensors.')
    if key.shape != value.shape or cp_size < 1 or query.shape[0] * cp_size != key.shape[0]:
        raise ValueError('NPU QSA requires local Q length * CP size == full K/V length.')
    if cp_size > 1 and (query_positions is None or query_positions.numel() != query.shape[0]):
        raise ValueError('CP QSA requires the global position of every local query.')
    if query.ndim == 4:
        # Unpacked sbhd indices use sample-local token IDs.
        if indices.shape[:2] != (query.shape[1], query.shape[0]):
            raise ValueError('Unpacked QSA indices must have shape [batch, sequence, K].')
        outputs = [
            _document_attention(query[:, b], key[:, b], value[:, b], indices[b], 0, scale, dropout,
                                query_positions)
            for b in range(query.shape[1])
        ]
        return torch.stack(outputs, dim=1)
    if query.ndim != 3 or indices.ndim != 2 or indices.shape[0] != query.shape[0]:
        raise ValueError('Packed QSA expects Q [T,N,D] and indices [T,K].')
    if packed_seq_params is None:
        bounds = [0, key.shape[0]]
    else:
        bounds = packed_document_boundaries(packed_seq_params, key.shape[0])
    outputs, local_start = [], 0
    for start, end in zip(bounds, bounds[1:]):
        length = end - start
        if cp_size > 1 and length % (2 * cp_size):
            raise ValueError('Packed CP QSA requires each physical document padded to 2 * CP size.')
        local_end = local_start + length // cp_size
        if length:
            positions = None if query_positions is None else query_positions[local_start:local_end]
            outputs.append(_document_attention(
                query[local_start:local_end], key[start:end], value[start:end],
                indices[local_start:local_end], start, scale, dropout, positions))
        local_start = local_end
    return torch.cat(outputs, dim=0)
