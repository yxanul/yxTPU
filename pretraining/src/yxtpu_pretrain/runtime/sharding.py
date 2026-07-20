"""Logical mesh contexts shared by standalone runtime entry points."""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from contextlib import contextmanager

import jax
from flax.linen import partitioning as nn_partitioning
from jax.sharding import Mesh


@contextmanager
def logical_mesh_context(
    mesh: Mesh,
    logical_axis_rules: Sequence[tuple[str, object]],
) -> Iterator[None]:
    """Installs the same mesh and logical-axis contexts used by MaxText.

    Logical constraints are resolved while JAX traces a function. Callers must
    therefore enter this context both when constructing state and when invoking
    a lazily compiled function for the first time.
    """
    with jax.set_mesh(mesh), nn_partitioning.axis_rules(logical_axis_rules):
        yield
