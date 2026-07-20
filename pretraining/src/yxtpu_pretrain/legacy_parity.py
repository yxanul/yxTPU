"""One-way parameter adapter from the validated legacy MaxText KDA model."""

from __future__ import annotations

import jax.numpy as jnp
from flax import nnx

Path = tuple[str | int, ...]


def legacy_path_for(standalone_path: Path) -> tuple[Path, bool]:
    """Returns `(legacy_path, add_one)` for one standalone parameter path.

    The only value transform is Qwen3-Next RMSNorm: legacy stores a zero-based
    scale and applies `1 + scale`, while the standalone ordinary RMSNorm stores
    the effective scale directly.
    """
    path = list(standalone_path)
    add_one = False
    if path[0] == "token_embedding":
        path[0] = "token_embedder"
        return tuple(path), add_one
    if path[0] == "final_norm":
        return ("decoder", "decoder_norm", *path[1:]), add_one
    if path[0] == "logits":
        return ("decoder", "logits_dense", *path[1:]), add_one
    if path[0] != "cycles":
        raise KeyError(f"no legacy mapping for standalone path {standalone_path}")

    path[0] = "layers"
    if "input_norm" in path:
        path[path.index("input_norm")] = "input_layernorm"
        add_one = True
    if "post_mixer_norm" in path:
        path[path.index("post_mixer_norm")] = "post_attention_layernorm"
        add_one = True
    if "mixer" in path:
        mixer_index = path.index("mixer")
        layer_name = str(path[1])
        if layer_name == "layer_3":
            path[mixer_index : mixer_index + 1] = ["attention", "attention"]
            if "out_proj" in path:
                path[path.index("out_proj")] = "out"
        else:
            path[mixer_index] = "attention"
    return ("decoder", *path), add_one


def export_legacy_parameters(
    legacy_parameters,
    standalone_template,
):
    """Converts a legacy NNX parameter state and fails on every mismatch."""
    legacy_flat = dict(nnx.to_flat_state(legacy_parameters))
    standalone_flat = nnx.to_flat_state(standalone_template)
    converted = []
    used: set[Path] = set()
    for standalone_path, variable in standalone_flat:
        legacy_path, add_one = legacy_path_for(standalone_path)
        if legacy_path not in legacy_flat:
            raise KeyError(
                f"legacy parameter {legacy_path} required by {standalone_path} was not found"
            )
        legacy_variable = legacy_flat[legacy_path]
        value = legacy_variable.get_value()
        if add_one:
            value = value + jnp.asarray(1.0, dtype=value.dtype)
        if value.shape != variable.get_value().shape:
            raise ValueError(
                f"shape mismatch for {standalone_path}: standalone "
                f"{variable.get_value().shape}, legacy {value.shape}"
            )
        converted.append((standalone_path, variable.replace(value=value)))
        used.add(legacy_path)
    unused = sorted(path for path in legacy_flat if path not in used)
    if unused:
        raise ValueError(
            f"legacy export left {len(unused)} trainable parameters unused: {unused[:8]}"
        )
    return nnx.from_flat_state(converted)


def parity_errors(reference_logits, actual_logits) -> dict[str, float]:
    difference = actual_logits.astype(jnp.float32) - reference_logits.astype(jnp.float32)
    denominator = jnp.linalg.norm(reference_logits.astype(jnp.float32)) + 1.0e-30
    return {
        "max_abs": float(jnp.max(jnp.abs(difference))),
        "relative_l2": float(jnp.linalg.norm(difference) / denominator),
    }
