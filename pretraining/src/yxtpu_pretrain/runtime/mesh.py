"""MaxText-compatible logical mesh construction from owned hardware profiles."""

from __future__ import annotations

import math

import jax
import numpy as np
from jax.sharding import Mesh

from yxtpu_pretrain.config import HardwareProfile

MESH_AXES = (
    "diloco",
    "data",
    "stage",
    "fsdp",
    "fsdp_transpose",
    "context",
    "context_autoregressive",
    "tensor",
    "tensor_sequence",
    "expert",
    "autoregressive",
)


def create_mesh(profile: HardwareProfile, *, allow_device_mismatch: bool = False) -> Mesh:
    """Creates the full MaxText leaf-component mesh without provisioning devices."""
    devices = np.asarray(jax.devices(), dtype=object)
    if devices.size != profile.device_count and not allow_device_mismatch:
        raise RuntimeError(
            f"hardware profile {profile.name} requires {profile.device_count} JAX devices; "
            f"found {devices.size}"
        )
    if allow_device_mismatch:
        data = devices.size
        fsdp = tensor = sequence = 1
    else:
        data = profile.mesh.data
        fsdp = profile.mesh.fsdp
        tensor = profile.mesh.tensor
        sequence = profile.mesh.sequence
    requested = data * fsdp * tensor * sequence
    if requested != devices.size:
        raise RuntimeError(f"mesh shape requires {requested} devices; found {devices.size}")
    shape = (1, data, 1, fsdp, 1, sequence, 1, tensor, 1, 1, 1)
    return Mesh(devices.reshape(shape), MESH_AXES)


def mesh_size(mesh: Mesh) -> int:
    return math.prod(mesh.shape.values())

