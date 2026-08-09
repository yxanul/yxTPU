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

"""Native from-scratch vision tower.

The design follows the stability diet that makes from-scratch joint
optimization work (MoonViT-V2's recipe, which is also this codebase's
house style): RMSNorm everywhere, no bias anywhere, and the same
optimizer regime as the language model from step zero. Pipeline:

  images -> patch embed (+ learned 2D positions) -> encoder blocks
         -> 2x2 pixel shuffle -> MLP projector -> embedding stream

``encoder_layers: 0`` skips the encoder entirely (the encoder-free arm):
patch features go straight through the shuffle and projector, and all
visual mixing is left to the backbone's KDA/GQA layers.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
from flax import nnx
from maxtext.common.common_types import MODEL_MODE_TRAIN
from maxtext.layers.linears import DenseGeneral, MlpBlock
from maxtext.layers.normalizations import RMSNorm

from yxtpu_pretrain.config import VisionConfig
from yxtpu_pretrain.layers.roles import (
    ParamRole,
    declare_dense_kernel,
    declare_norm,
    declare_parameter,
)


def splice_visual_embeddings(text_embeddings, token_ids, visual, placeholder_id):
    """Replaces placeholder positions' embeddings with visual features.

    ``visual`` is ``[batch, slots, dim]`` in stream order; the k-th
    placeholder of a row takes the row's k-th visual slot. Rows with fewer
    placeholders than slots simply leave the surplus unread, which is how
    image-free (padded) rows cost nothing. Fully static shapes."""
    mask = token_ids == placeholder_id
    order = jnp.cumsum(mask.astype(jnp.int32), axis=1) - 1
    order = jnp.clip(order, 0, visual.shape[1] - 1)
    gathered = jnp.take_along_axis(visual, order[..., None], axis=1)
    return jnp.where(mask[..., None], gathered.astype(text_embeddings.dtype), text_embeddings)


class VisionBlock(nnx.Module):
    """Pre-norm bidirectional attention + fused SwiGLU, bias-free."""

    def __init__(self, vision: VisionConfig, *, leaf_config, mesh, rngs: nnx.Rngs):
        dim, heads = vision.encoder_dim, vision.encoder_heads
        self.heads = heads
        self.head_dim = dim // heads
        self.attn_norm = RMSNorm(
            num_features=dim,
            epsilon=1.0e-5,
            dtype=leaf_config.dtype,
            weight_dtype=leaf_config.weight_dtype,
            kernel_axes=("norm",),
            rngs=rngs,
        )
        declare_norm(self.attn_norm)
        self.qkv = DenseGeneral(
            in_features_shape=dim,
            out_features_shape=(3, heads, self.head_dim),
            dtype=leaf_config.dtype,
            weight_dtype=leaf_config.weight_dtype,
            kernel_axes=("embed", "qkv", "heads", "kv"),
            matmul_precision="default",
            use_bias=False,
            rngs=rngs,
        )
        declare_dense_kernel(self.qkv, ParamRole.VIT_MATRIX)
        self.out = DenseGeneral(
            in_features_shape=(heads, self.head_dim),
            out_features_shape=dim,
            axis=(-2, -1),
            dtype=leaf_config.dtype,
            weight_dtype=leaf_config.weight_dtype,
            kernel_axes=("heads", "kv", "embed"),
            matmul_precision="default",
            use_bias=False,
            rngs=rngs,
        )
        declare_dense_kernel(self.out, ParamRole.VIT_MATRIX, in_axes=(0, 1), out_axes=(2,))
        self.mlp_norm = RMSNorm(
            num_features=dim,
            epsilon=1.0e-5,
            dtype=leaf_config.dtype,
            weight_dtype=leaf_config.weight_dtype,
            kernel_axes=("norm",),
            rngs=rngs,
        )
        declare_norm(self.mlp_norm)
        self.mlp = MlpBlock(
            config=leaf_config,
            mesh=mesh,
            in_features=dim,
            intermediate_dim=vision.encoder_mlp_dim,
            activations=("silu", "linear"),
            intermediate_dropout_rate=0.0,
            dtype=leaf_config.dtype,
            weight_dtype=leaf_config.weight_dtype,
            model_mode=MODEL_MODE_TRAIN,
            rngs=rngs,
        )
        declare_dense_kernel(self.mlp.wi, ParamRole.VIT_MATRIX)
        declare_dense_kernel(self.mlp.wo, ParamRole.VIT_MATRIX)

    def __call__(self, tokens):
        # tokens: [rows, patches, dim]; bidirectional attention, fp32 softmax.
        normalized = self.attn_norm(tokens)
        qkv = self.qkv(normalized)
        query, key, value = (
            qkv[:, :, 0],
            qkv[:, :, 1],
            qkv[:, :, 2],
        )
        scores = jnp.einsum(
            "bqhd,bkhd->bhqk", query, key, preferred_element_type=jnp.float32
        ) / jnp.sqrt(jnp.asarray(self.head_dim, jnp.float32))
        weights = jax.nn.softmax(scores, axis=-1).astype(value.dtype)
        context = jnp.einsum("bhqk,bkhd->bqhd", weights, value)
        tokens = tokens + self.out(context)
        return tokens + self.mlp(self.mlp_norm(tokens), deterministic=True)


class VisionTower(nnx.Module):
    """Patch embed -> encoder -> pixel shuffle -> projector."""

    def __init__(self, vision: VisionConfig, emb_dim: int, *, leaf_config, mesh, rngs: nnx.Rngs):
        self.vision = vision
        self.dtype = leaf_config.dtype
        patch_features = vision.patch_size * vision.patch_size * 3
        self.patch_embed = DenseGeneral(
            in_features_shape=patch_features,
            out_features_shape=vision.encoder_dim,
            dtype=leaf_config.dtype,
            weight_dtype=leaf_config.weight_dtype,
            kernel_axes=("qkv", "embed"),
            matmul_precision="default",
            use_bias=False,
            rngs=rngs,
        )
        self.patch_embed.kernel = declare_parameter(
            self.patch_embed.kernel, ParamRole.VIT_EMBED
        )
        positions = vision.patch_grid**2
        self.position_embedding = declare_parameter(
            nnx.Param(
                jax.random.normal(rngs.params(), (positions, vision.encoder_dim))
                * 0.02
            ),
            ParamRole.VIT_EMBED,
        )
        self.num_blocks = vision.encoder_layers
        for index in range(vision.encoder_layers):
            setattr(
                self,
                f"block_{index}",
                VisionBlock(vision, leaf_config=leaf_config, mesh=mesh, rngs=rngs.fork()),
            )
        shuffled = vision.encoder_dim * vision.pixel_shuffle**2
        self.projector_norm = RMSNorm(
            num_features=shuffled,
            epsilon=1.0e-5,
            dtype=leaf_config.dtype,
            weight_dtype=leaf_config.weight_dtype,
            kernel_axes=("norm",),
            rngs=rngs,
        )
        declare_norm(self.projector_norm)
        self.projector_in = DenseGeneral(
            in_features_shape=shuffled,
            out_features_shape=emb_dim,
            dtype=leaf_config.dtype,
            weight_dtype=leaf_config.weight_dtype,
            kernel_axes=("qkv", "embed"),
            matmul_precision="default",
            use_bias=False,
            rngs=rngs,
        )
        declare_dense_kernel(self.projector_in, ParamRole.VIT_MATRIX)
        self.projector_out = DenseGeneral(
            in_features_shape=emb_dim,
            out_features_shape=emb_dim,
            dtype=leaf_config.dtype,
            weight_dtype=leaf_config.weight_dtype,
            kernel_axes=("embed", "mlp"),
            matmul_precision="default",
            use_bias=False,
            rngs=rngs,
        )
        declare_dense_kernel(self.projector_out, ParamRole.VIT_MATRIX)

    def _patchify(self, images):
        """[rows, H, W, 3] -> [rows, patches, patch_size^2 * 3]."""
        rows = images.shape[0]
        grid, patch = self.vision.patch_grid, self.vision.patch_size
        patches = images.reshape(rows, grid, patch, grid, patch, 3)
        patches = patches.transpose(0, 1, 3, 2, 4, 5)
        return patches.reshape(rows, grid * grid, patch * patch * 3)

    def _pixel_shuffle(self, tokens):
        """[rows, grid^2, dim] -> [rows, (grid/s)^2, dim * s^2]."""
        rows = tokens.shape[0]
        grid, s = self.vision.patch_grid, self.vision.pixel_shuffle
        dim = tokens.shape[-1]
        tokens = tokens.reshape(rows, grid // s, s, grid // s, s, dim)
        tokens = tokens.transpose(0, 1, 3, 2, 4, 5)
        return tokens.reshape(rows, (grid // s) ** 2, s * s * dim)

    def __call__(self, images):
        """[batch, images, H, W, 3] float -> [batch, images * tokens, emb]."""
        batch, per_row = images.shape[0], images.shape[1]
        flat = images.reshape(batch * per_row, *images.shape[2:])
        tokens = self.patch_embed(self._patchify(flat).astype(self.dtype))
        tokens = tokens + self.position_embedding.get_value().astype(tokens.dtype)
        for index in range(self.num_blocks):
            tokens = getattr(self, f"block_{index}")(tokens)
        tokens = self._pixel_shuffle(tokens)
        projected = self.projector_in(self.projector_norm(tokens))
        projected = self.projector_out(jax.nn.silu(projected))
        return projected.reshape(batch, per_row * projected.shape[1], projected.shape[-1])
