# Copyright (c) ModelScope Contributors. All rights reserved.
"""Device helpers for the vendored Triton kernels; no upstream package bootstrap."""
from functools import lru_cache

import torch


def _index(device=None):
    return torch.npu.current_device() if device is None or device.index is None else device.index


@lru_cache(maxsize=None)
def _properties(index):
    from triton.runtime import driver

    return driver.active.utils.get_device_properties(index)


def device_properties(device=None):
    return _properties(_index(device))


@lru_cache(maxsize=None)
def _arch35(index):
    name = torch.npu.get_device_name(index)
    return 'Ascend950' in name or 'Ascend910_95' in name


def is_arch35(device=None):
    return _arch35(_index(device))


def qhc_launch_options(device):
    # Bound A3's UB usage, including compiler-created double buffers.
    return {} if is_arch35(device) else dict(num_stages=1, multibuffer=False)
