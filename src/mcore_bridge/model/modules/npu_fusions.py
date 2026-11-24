# Copyright (c) ModelScope Contributors. All rights reserved.
"""Opt-in QHC NPU dispatch. Importing this module never imports Triton."""
from functools import lru_cache

import torch

from mcore_bridge.utils import get_env_args, get_logger

logger = get_logger()


@lru_cache(maxsize=1)
def fusion_enabled(name):
    # Environment switches are fixed for the lifetime of a training worker.
    return get_env_args(name, bool, False)


def _eligible(name, x):
    if not fusion_enabled(name) or x.device.type not in ('npu', 'privateuseone'):
        return False
    if x.numel() == 0:
        return False
    if x.dtype != torch.bfloat16:
        _fallback(name, 'the model fast path requires BF16 activations')
        return False
    return True


def _fallback(name, reason):
    logger.warning_once(f'{name}: using the original implementation because {reason}.')


def _hooks(module):
    # A bypass must not skip parameter-gather, adapter or user forward hooks.
    return bool(module._forward_pre_hooks or module._forward_hooks or module._backward_hooks
                or getattr(module, '_backward_pre_hooks', None))


def try_qhc(module, hyper_input):
    name = 'NPU_QHC_FUSED'
    if not _eligible(name, hyper_input):
        return None
    config = getattr(module.input_mix_weight_down, 'config', None)
    if (getattr(config, 'gradient_accumulation_fusion', False)
            or any(getattr(config, key, False) for key in ('fp8', 'fp4', 'fp8_param'))):
        _fallback(name, 'quantization or fused weight-gradient accumulation is enabled')
        return None
    projections = [module.input_mix_weight_down, module.input_mix_weight_up]
    if module.block_inject_weight is not None:
        projections.append(module.block_inject_weight)
    if any(type(p).__module__ != 'mindspeed.te.pytorch.module.linear' or type(p).__name__ != 'TELinear'
           or _hooks(p) or p.bias is not None for p in projections) or _hooks(module.hc_norm):
        _fallback(name, 'the HC projections/norm have wrappers, forward hooks or a nonstandard implementation')
        return None
    weights = [p.weight for p in projections]
    norm_weight = module.hc_norm.weight
    if (any(w.device != hyper_input.device or w.dtype != hyper_input.dtype for w in weights)
            or norm_weight.device != hyper_input.device
            or norm_weight.dtype not in (torch.bfloat16, torch.float32)):
        _fallback(name, 'the HC weight dtype/device is unsupported')
        return None
    from .kernels.qhc_npu import qhc_triton

    with torch.npu.device(hyper_input.device):
        output = qhc_triton(
            hyper_input, norm_weight, weights[0], weights[1],
            weights[2] if module.block_inject_weight is not None else None,
            module.hc_count, module.hidden_size, module.hc_norm.eps)
    logger.info_once('NPU_QHC_FUSED: using vendored Triton QHC with native GEMM projections.')
    if module.block_inject_weight is None:
        return output
    mixed, injection = output
    return mixed, hyper_input, injection
