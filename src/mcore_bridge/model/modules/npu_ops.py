# Copyright (c) ModelScope Contributors. All rights reserved.
import inspect
import torch
from contextlib import contextmanager


def npu_causal_conv1d(x, weight, bias=None, activation='silu', cu_seqlens=None):
    """Training convolution through the verified MindSpeed/fla-npu packed path.

    Inputs use FLA's [batch, sequence, channels] and [channels, width] layouts,
    but importing or installing FLA is not required. Dense batches are packed
    into independent sequences to avoid the legacy wrapper's BSH backward.
    """
    from fla_npu.ops import ascendc  # noqa: F401
    from mindspeed.core.ssm.ops.npu_causal_conv1d import causal_conv1d

    if x.ndim != 3 or weight.ndim != 2:
        raise ValueError('NPU convolution expects x [B,S,D] and weight [D,W].')
    batch, length, channels = x.shape
    if batch < 1 or length < 1 or channels != weight.shape[0]:
        raise ValueError('NPU convolution requires nonempty input with matching weight channels.')
    if cu_seqlens is None:
        cu_seqlens = torch.arange(batch + 1, dtype=torch.long) * length
        packed_x = x.reshape(1, batch * length, channels)
    else:
        if batch != 1:
            raise ValueError('Packed NPU convolution requires batch dimension 1.')
        packed_x = x
    output, _ = causal_conv1d(
        x=packed_x.contiguous(),
        weight=weight.contiguous(),
        bias=bias,
        initial_state=None,
        output_final_state=False,
        activation=activation,
        cu_seqlens=cu_seqlens,
    )
    return output.reshape(batch, length, channels), None


def get_npu_gdn_ops():
    """Resolve NPU training operators independently of FLA/Swift monkey patches."""
    from fla_npu.ops import ascendc  # noqa: F401
    from mindspeed.core.ssm.ops.flash_gated_delta_rule import flash_gated_delta_rule
    from mindspeed.ops.triton.l2norm import l2norm

    return l2norm, flash_gated_delta_rule


@contextmanager
def npu_gdn_init_context(base_class):
    """Let the legacy GDN constructor use an available NPU backend without FLA.

    Megatron/MindSpeed 0.16 gates construction on HAVE_FLA even though this
    bridge overrides forward. Restore the constructor's module after init.
    """
    l2norm, gated_delta_rule = get_npu_gdn_ops()
    namespace = inspect.unwrap(base_class.__init__).__globals__
    if namespace.get('HAVE_FLA', True):
        yield
        return
    replacements = {'HAVE_FLA': True, 'chunk_gated_delta_rule': gated_delta_rule, 'l2norm': l2norm}
    missing = object()
    original = {key: namespace.get(key, missing) for key in replacements}
    namespace.update(replacements)
    try:
        yield
    finally:
        for key, value in original.items():
            if value is missing:
                namespace.pop(key, None)
            else:
                namespace[key] = value
