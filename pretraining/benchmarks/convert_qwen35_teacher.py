# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Converts a dense Qwen3.5 checkpoint into the MaxText parameter layout.

Forward-only teacher weights for GOLD. Three things make this more than a
rename, and each is a place to get it silently wrong:

**The GDN input projections are laid out differently.** MaxText inherits
Qwen3-Next's fused `in_proj_qkvz`, which reshapes to
`(H_k, 2*D_k + 2*D_v*V_per_K)` - grouped *per key head*, ordered
`[q | k | v | z]` within each group (``models/qwen3.py`` split_indices_qkvz).
Qwen3.5 instead ships four separate, each-contiguous tensors: `in_proj_qkv`
splitting `[query | key | value]` at `[key_dim, key_dim, value_dim]`, plus
`in_proj_z`, `in_proj_b`, `in_proj_a`. So converting is an interleave, not
a concatenation. Same story for `in_proj_ba`: `[b | a]` per key head, from
two flat `[H_v]` tensors. With 4B's asymmetric 16 key / 32 value heads,
V_per_K is 2, so each key-head group carries two value heads' worth of v
and z.

**The published checkpoint is a VLM wrapper.** Text weights live under
`model.language_model.*`; `model.visual.*` and the single `mtp.*` block are
dropped. Embeddings are tied, so there is no separate lm_head.

**MaxText stacks layers into the scan axis.** Every per-layer tensor is
written into a `[..., layer, ...]` slice, and the transposes differ per
tensor kind. Shapes are asserted on every write rather than trusted.

  # shape-only check, no download
  python benchmarks/convert_qwen35_teacher.py --dry-run

  # real conversion (run on a worker: needs ~20GB disk and the HF token)
  python benchmarks/convert_qwen35_teacher.py \
      --output /home/a1111/yxtpu_ckpts/qwen35-4b-teacher
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

TEXT = "model.language_model"


def geometry(config):
    """Derives every dimension the layout depends on."""
    text = config.get("text_config", config)
    key_heads = text["linear_num_key_heads"]
    value_heads = text["linear_num_value_heads"]
    head_k = text["linear_key_head_dim"]
    head_v = text["linear_value_head_dim"]
    return {
        "emb": text["hidden_size"],
        "layers": text["num_hidden_layers"],
        "vocab": text["vocab_size"],
        "mlp": text["intermediate_size"],
        "interval": text["full_attention_interval"],
        "q_heads": text["num_attention_heads"],
        "kv_heads": text["num_key_value_heads"],
        "head_dim": text["head_dim"],
        "key_heads": key_heads,
        "value_heads": value_heads,
        "head_k": head_k,
        "head_v": head_v,
        "key_dim": key_heads * head_k,
        "value_dim": value_heads * head_v,
        "v_per_k": value_heads // key_heads,
        "conv_kernel": text["linear_conv_kernel_dim"],
    }


def fuse_qkvz(qkv, z, g):
    """Interleaves Qwen3.5's contiguous q/k/v/z into MaxText's per-head fusion.

    ``qkv`` is ``[key_dim*2 + value_dim, emb]`` split contiguously as
    ``[query | key | value]``; ``z`` is ``[value_dim, emb]``. The result is
    ``[key_dim*2 + value_dim*2, emb]``, ordered key head by key head as
    ``[q_h | k_h | v_h (V_per_K heads) | z_h (V_per_K heads)]``.
    """
    key_dim, value_dim = g["key_dim"], g["value_dim"]
    head_k, head_v, v_per_k = g["head_k"], g["head_v"], g["v_per_k"]
    query, key, value = np.split(qkv, [key_dim, 2 * key_dim], axis=0)
    blocks = []
    for head in range(g["key_heads"]):
        k_lo, k_hi = head * head_k, (head + 1) * head_k
        v_lo, v_hi = head * v_per_k * head_v, (head + 1) * v_per_k * head_v
        blocks.append(np.concatenate(
            [query[k_lo:k_hi], key[k_lo:k_hi], value[v_lo:v_hi], z[v_lo:v_hi]],
            axis=0,
        ))
    fused = np.concatenate(blocks, axis=0)
    assert fused.shape[0] == 2 * key_dim + 2 * value_dim, fused.shape
    return fused


def fuse_ba(b, a, g):
    """Interleaves the two ``[H_v]`` gates into MaxText's ``[H_k, 2*V_per_K]``."""
    v_per_k = g["v_per_k"]
    blocks = []
    for head in range(g["key_heads"]):
        lo, hi = head * v_per_k, (head + 1) * v_per_k
        blocks.append(np.concatenate([b[lo:hi], a[lo:hi]], axis=0))
    fused = np.concatenate(blocks, axis=0)
    assert fused.shape[0] == 2 * g["value_heads"], fused.shape
    return fused


def expected_hf_shapes(g):
    """Every text tensor we consume, and the shape it must have."""
    shapes = {
        f"{TEXT}.embed_tokens.weight": (g["vocab"], g["emb"]),
        f"{TEXT}.norm.weight": (g["emb"],),
    }
    for layer in range(g["layers"]):
        prefix = f"{TEXT}.layers.{layer}"
        shapes[f"{prefix}.input_layernorm.weight"] = (g["emb"],)
        shapes[f"{prefix}.post_attention_layernorm.weight"] = (g["emb"],)
        shapes[f"{prefix}.mlp.gate_proj.weight"] = (g["mlp"], g["emb"])
        shapes[f"{prefix}.mlp.up_proj.weight"] = (g["mlp"], g["emb"])
        shapes[f"{prefix}.mlp.down_proj.weight"] = (g["emb"], g["mlp"])
        if (layer + 1) % g["interval"] == 0:      # full attention
            shapes[f"{prefix}.self_attn.q_proj.weight"] = (
                g["q_heads"] * g["head_dim"], g["emb"])
            shapes[f"{prefix}.self_attn.k_proj.weight"] = (
                g["kv_heads"] * g["head_dim"], g["emb"])
            shapes[f"{prefix}.self_attn.v_proj.weight"] = (
                g["kv_heads"] * g["head_dim"], g["emb"])
            shapes[f"{prefix}.self_attn.o_proj.weight"] = (
                g["emb"], g["q_heads"] * g["head_dim"])
            shapes[f"{prefix}.self_attn.q_norm.weight"] = (g["head_dim"],)
            shapes[f"{prefix}.self_attn.k_norm.weight"] = (g["head_dim"],)
        else:                                      # gated delta net
            shapes[f"{prefix}.linear_attn.in_proj_qkv.weight"] = (
                2 * g["key_dim"] + g["value_dim"], g["emb"])
            shapes[f"{prefix}.linear_attn.in_proj_z.weight"] = (
                g["value_dim"], g["emb"])
            shapes[f"{prefix}.linear_attn.in_proj_b.weight"] = (
                g["value_heads"], g["emb"])
            shapes[f"{prefix}.linear_attn.in_proj_a.weight"] = (
                g["value_heads"], g["emb"])
            shapes[f"{prefix}.linear_attn.out_proj.weight"] = (
                g["emb"], g["value_dim"])
            shapes[f"{prefix}.linear_attn.conv1d.weight"] = (
                2 * g["key_dim"] + g["value_dim"], 1, g["conv_kernel"])
            shapes[f"{prefix}.linear_attn.A_log"] = (g["value_heads"],)
            shapes[f"{prefix}.linear_attn.dt_bias"] = (g["value_heads"],)
            shapes[f"{prefix}.linear_attn.norm.weight"] = (g["head_v"],)
    return shapes


def note_q_proj_carries_the_gate(g, actual):
    """q_proj is twice the query width because it also carries the gate.

    Qwen3NextFullAttention splits query and sigmoid gate from one
    projection, which is why the released q_proj is 2x q_heads*head_dim.
    Worth asserting: getting this wrong halves the head count silently.
    """
    expected_gated = 2 * g["q_heads"] * g["head_dim"]
    return actual == (expected_gated, g["emb"])


def dry_run(repo):
    from huggingface_hub import get_safetensors_metadata, hf_hub_download

    config = json.load(open(hf_hub_download(repo, "config.json")))
    g = geometry(config)
    print(json.dumps(g, indent=1))

    present = {}
    for meta in get_safetensors_metadata(repo).files_metadata.values():
        for name, tensor in meta.tensors.items():
            present[name] = tuple(tensor.shape)

    wanted = expected_hf_shapes(g)
    missing, mismatched, gated = [], [], []
    for name, shape in wanted.items():
        if name not in present:
            missing.append(name)
        elif present[name] != shape:
            # The one legitimate divergence: q_proj carries query+gate.
            if name.endswith("self_attn.q_proj.weight") and \
                    note_q_proj_carries_the_gate(g, present[name]):
                gated.append(name)
            else:
                mismatched.append((name, shape, present[name]))

    text_keys = {k for k in present
                 if k.startswith(TEXT) and not k.startswith("mtp")}
    unconsumed = sorted(text_keys - set(wanted))
    dropped = sorted({k.split(".")[0] + "." + k.split(".")[1]
                      for k in present if k not in text_keys})

    print(f"\nconsumed {len(wanted) - len(missing):,}/{len(wanted):,} text tensors")
    print(f"q_proj carries query+gate (2x width): {len(gated)} layers")
    print(f"unconsumed text tensors: {unconsumed or 'none'}")
    print(f"dropped families: {dropped}")
    if missing:
        print(f"\nMISSING ({len(missing)}):")
        for name in missing[:10]:
            print(f"  {name}")
    if mismatched:
        print(f"\nSHAPE MISMATCH ({len(mismatched)}):")
        for name, want, got in mismatched[:10]:
            print(f"  {name}\n    want {want}  got {got}")
    ok = not missing and not mismatched
    print(f"\n{'OK - mapping covers the text decoder' if ok else 'FAILED'}")
    return 0 if ok else 1


def selftest_layout():
    """The interleave must be invertible by MaxText's own split indices."""
    g = {"emb": 4, "key_heads": 2, "value_heads": 4, "head_k": 3, "head_v": 5,
         "key_dim": 6, "value_dim": 20, "v_per_k": 2}
    # Tag every row with an identifiable value.
    qkv = np.concatenate([
        np.full((g["key_dim"], g["emb"]), 1.0),      # query
        np.full((g["key_dim"], g["emb"]), 2.0),      # key
        np.full((g["value_dim"], g["emb"]), 3.0),    # value
    ])
    for head in range(g["key_heads"]):               # make heads distinct
        qkv[head * g["head_k"]:(head + 1) * g["head_k"]] += 0.1 * head
    z = np.full((g["value_dim"], g["emb"]), 4.0)
    fused = fuse_qkvz(qkv, z, g)

    # Replay MaxText's reshape + split on the fused kernel.
    per_head = 2 * g["head_k"] + 2 * g["head_v"] * g["v_per_k"]
    grouped = fused.reshape(g["key_heads"], per_head, g["emb"])
    idx = [g["head_k"], 2 * g["head_k"],
           2 * g["head_k"] + g["v_per_k"] * g["head_v"]]
    query, key, value, zed = np.split(grouped, idx, axis=1)
    assert np.allclose(query[0], 1.0) and np.allclose(query[1], 1.1)
    assert np.allclose(key[0], 2.0) and np.allclose(key[1], 2.0)
    assert np.allclose(value, 3.0) and np.allclose(zed, 4.0)

    b = np.arange(g["value_heads"], dtype=np.float32)
    a = 100 + np.arange(g["value_heads"], dtype=np.float32)
    fused_ba = fuse_ba(b[:, None], a[:, None], g).reshape(
        g["key_heads"], 2 * g["v_per_k"])
    b_out, a_out = np.split(fused_ba, [g["v_per_k"]], axis=1)
    assert np.array_equal(b_out.reshape(-1), b)
    assert np.array_equal(a_out.reshape(-1), a)
    print("layout self-test OK: interleave inverts MaxText's split")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", default="Qwen/Qwen3.5-4B")
    parser.add_argument("--dry-run", action="store_true",
                        help="validate the mapping from safetensors headers only")
    parser.add_argument("--selftest", action="store_true")
    parser.add_argument("--output", default=None)
    arguments = parser.parse_args()

    if arguments.selftest:
        selftest_layout()
        return 0
    if arguments.dry_run or not arguments.output:
        selftest_layout()
        return dry_run(arguments.repo)
    raise SystemExit("materialization not implemented yet - see --dry-run")


if __name__ == "__main__":
    raise SystemExit(main())
