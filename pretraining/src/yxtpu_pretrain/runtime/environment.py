"""Process environment derived from a hardware profile."""

from __future__ import annotations

import os

from yxtpu_pretrain.config import HardwareProfile


def apply_hardware_environment(profile: HardwareProfile) -> None:
    """Sets compiler defaults before JAX is imported.

    An explicit user value wins. This function does not call any cloud API or
    create, resize, or delete resources.
    """
    flags = " ".join(profile.libtpu_init_args)
    if flags:
        os.environ.setdefault("LIBTPU_INIT_ARGS", flags)

