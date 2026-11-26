# Copyright (c) ModelScope Contributors. All rights reserved.
"""Differentiable exchange of PLE history between adjacent sequence shards."""
import torch
import torch.distributed as dist


def _exchange(send, send_peer, receive, receive_peer, group):
    """Finish matched P2P operations before consuming or releasing their buffers."""
    operations = []
    if send_peer is not None:
        operations.append(dist.P2POp(dist.isend, send, dist.get_global_rank(group, send_peer), group))
    if receive_peer is not None:
        operations.append(dist.P2POp(dist.irecv, receive, dist.get_global_rank(group, receive_peer), group))
    # Match the alternating order used by the HCCL P2P paths. Do not rely on
    # message tags: forward/recompute/backward calls complete in TP group order.
    if dist.get_rank(group) % 2:
        operations.reverse()
    if operations:
        for request in dist.batch_isend_irecv(operations):
            request.wait()


class _ExchangeLeftHalo(torch.autograd.Function):

    @staticmethod
    def forward(ctx, tail, group):
        ctx.group = group
        ctx.rank = dist.get_rank(group)
        ctx.size = dist.get_world_size(group)
        tail = tail.contiguous()
        history = torch.zeros_like(tail)
        _exchange(
            tail, ctx.rank + 1 if ctx.rank + 1 < ctx.size else None,
            history, ctx.rank - 1 if ctx.rank else None, group)
        return history

    @staticmethod
    def backward(ctx, grad_history):
        grad_history = grad_history.contiguous()
        grad_tail = torch.zeros_like(grad_history)
        _exchange(
            grad_history, ctx.rank - 1 if ctx.rank else None,
            grad_tail, ctx.rank + 1 if ctx.rank + 1 < ctx.size else None, ctx.group)
        return grad_tail, None


def exchange_left_halo(tail, group):
    """Send this shard's tail right; receive history left, reversing it in backward.

    Every TP rank calls this, including ranks with only padding or no usable
    left history. Callers keep a zero dependency when the received halo is unused.
    """
    return _ExchangeLeftHalo.apply(tail, group)
