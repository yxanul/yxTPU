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

"""Pure-XLA chunked linear cross entropy for the 128k tied vocabulary head.

On v4 there is no memory-efficient loss: tokamax's mosaic_tpu kernel raises
NotImplementedError, and the standard path materializes [B, T, V] logits
(~4 GB of compiled temporaries at the 337M/PDB-8 shape, growing linearly with
batch — the binding memory term for larger per-device batches and the 1B
scale-up). This implementation scans over sequence blocks and, Liger-style,
computes the loss together with both gradients inside the same block: the
upstream cotangent of a scalar loss is a scalar, so dX and dW are computed in
the forward pass at full fidelity and merely rescaled in the backward.
FLOP-neutral with the standard path (three head GEMMs: logits, dX, dW);
peak logits memory drops from [B, T, V] to [B, block, V].

Numerics: block matmuls take the model's bf16 operands with fp32
accumulation (preferred_element_type), like the fused reference. The
standard path instead rounds its logits through bf16 before the fp32 cast,
so trajectories against it overlay at bf16-rounding noise rather than
bitwise; the gate is the standard fp-noise overlay A/B.

Sharding: plain jnp on [B, T, ...] arrays — the batch axis keeps its data
sharding, sequence-block slicing is shard-local, and the scalar reductions
lower to the usual cross-shard collectives. No shard_map is required, which
is what keeps this implementation v4-safe.
"""

from __future__ import annotations

from functools import partial

import jax
import jax.numpy as jnp
import numpy as np


def _block_loss_and_grads(x_block, labels_block, mask_block, weights):
    """Loss sum plus both gradients for one [B, block] token slab, in fp32."""
    logits = jnp.einsum(
        "bth,hv->btv",
        x_block,
        weights,
        preferred_element_type=jnp.float32,
    )
    maxima = jax.lax.stop_gradient(jnp.max(logits, axis=-1, keepdims=True))
    shifted = logits - maxima
    sum_exp = jnp.sum(jnp.exp(shifted), axis=-1, keepdims=True)
    log_sum_exp = jnp.log(sum_exp) + maxima
    safe_labels = jnp.where(mask_block > 0, labels_block, 0).astype(jnp.int32)
    target_logits = jnp.take_along_axis(logits, safe_labels[..., None], axis=-1)
    per_token = (log_sum_exp - target_logits)[..., 0]
    loss_sum = jnp.sum(per_token * mask_block, dtype=jnp.float32)

    # dL/dlogits for the masked SUM: (softmax - onehot) * mask.
    probabilities = jnp.exp(shifted) / sum_exp
    vocab_iota = jax.lax.broadcasted_iota(jnp.int32, probabilities.shape, 2)
    one_hot = (vocab_iota == safe_labels[..., None]).astype(jnp.float32)
    dlogits = (probabilities - one_hot) * mask_block[..., None]
    dx_block = jnp.einsum(
        "btv,hv->bth",
        dlogits.astype(x_block.dtype),
        weights,
        preferred_element_type=jnp.float32,
    )
    dweights_block = jnp.einsum(
        "bth,btv->hv",
        x_block,
        dlogits.astype(x_block.dtype),
        preferred_element_type=jnp.float32,
    )
    return loss_sum, dx_block, dweights_block


@partial(jax.custom_vjp, nondiff_argnums=(4,))
def _chunked_sum(x, labels, loss_mask, weights, block_tokens):
    return _chunked_sum_fwd(x, labels, loss_mask, weights, block_tokens)[0]


def _chunked_sum_fwd(x, labels, loss_mask, weights, block_tokens):
    sequence_length = x.shape[1]
    if sequence_length % block_tokens:
        raise ValueError(
            f"sequence length {sequence_length} must be divisible by "
            f"loss.block_tokens {block_tokens}"
        )
    num_blocks = sequence_length // block_tokens

    def scan_body(carry, block_index):
        loss_sum, dweights = carry
        start = block_index * block_tokens
        x_block = jax.lax.dynamic_slice_in_dim(x, start, block_tokens, axis=1)
        labels_block = jax.lax.dynamic_slice_in_dim(
            labels, start, block_tokens, axis=1
        )
        mask_block = jax.lax.dynamic_slice_in_dim(
            loss_mask, start, block_tokens, axis=1
        )
        block_sum, dx_block, dweights_block = _block_loss_and_grads(
            x_block, labels_block, mask_block, weights
        )
        return (loss_sum + block_sum, dweights + dweights_block), dx_block

    (loss_sum, dweights), dx_blocks = jax.lax.scan(
        scan_body,
        (jnp.zeros((), jnp.float32), jnp.zeros(weights.shape, jnp.float32)),
        jnp.arange(num_blocks),
    )
    # [num_blocks, B, block, H] -> [B, T, H].
    dx = jnp.transpose(dx_blocks, (1, 0, 2, 3)).reshape(x.shape)
    # Zero-size tokens carry the primal dtypes into the backward pass, where
    # the fp32 gradient accumulators are rescaled and cast exactly once.
    residuals = (
        dx,
        dweights,
        jnp.zeros((0,), x.dtype),
        jnp.zeros((0,), weights.dtype),
        labels,
        loss_mask,
    )
    return loss_sum, residuals


def _chunked_sum_bwd(block_tokens, residuals, loss_cotangent):
    del block_tokens
    dx, dweights, x_token, weights_token, labels, loss_mask = residuals
    return (
        (loss_cotangent * dx).astype(x_token.dtype),
        np.zeros(labels.shape, dtype=jax.dtypes.float0),
        jnp.zeros_like(loss_mask),
        (loss_cotangent * dweights).astype(weights_token.dtype),
    )


_chunked_sum.defvjp(_chunked_sum_fwd, _chunked_sum_bwd)


def chunked_linear_cross_entropy(
    hidden_states: jax.Array,
    labels: jax.Array,
    loss_mask: jax.Array,
    weights: jax.Array,
    *,
    block_tokens: int,
) -> tuple[jax.Array, jax.Array]:
    """Returns the globally normalized mean loss and the valid-token count.

    ``hidden_states`` is [B, T, H], ``weights`` the [H, V] output projection
    (for the tied head, the scaled transposed embedding compute copy — its
    cotangent flows back to the fp32 master through the caller's autodiff).
    ``loss_mask`` must be binary, matching the owned data pipelines.
    """
    if hidden_states.ndim != 3 or weights.ndim != 2:
        raise ValueError("chunked loss expects hidden_states[B,T,H] and weights[H,V]")
    if hidden_states.shape[-1] != weights.shape[0]:
        raise ValueError("hidden dimension does not match the output projection")
    if labels.shape != hidden_states.shape[:2] or loss_mask.shape != labels.shape:
        raise ValueError("labels and loss_mask must be [B, T]")
    mask = loss_mask.astype(jnp.float32)
    loss_sum = _chunked_sum(hidden_states, labels, mask, weights, block_tokens)
    token_count = jnp.sum(mask, dtype=jnp.float32)
    return loss_sum / jnp.maximum(token_count, 1.0), token_count
