import jax
import jax.numpy as jnp

from yxtpu_pretrain.config import load_config
from yxtpu_pretrain.losses import (
    data_parallel_linear_cross_entropy,
    local_linear_cross_entropy_sum,
)
from yxtpu_pretrain.runtime.mesh import create_mesh


def _manual_loss(x, labels, mask, weights):
    logits = jax.lax.dot_general(
        x,
        weights,
        dimension_numbers=(((1,), (0,)), ((), ())),
        preferred_element_type=jnp.float32,
    )
    safe_labels = jnp.where(mask > 0, labels, 0)
    targets = jnp.take_along_axis(logits, safe_labels[:, None], axis=-1)[:, 0]
    return jnp.sum(mask * (jax.nn.logsumexp(logits, axis=-1) - targets)) / jnp.sum(mask)


def test_reference_loss_and_all_gradients_match_materialized_formula():
    x = jax.random.normal(jax.random.key(1), (8, 16), dtype=jnp.float32)
    weights = jax.random.normal(jax.random.key(2), (16, 32), dtype=jnp.float32) * 0.1
    labels = jnp.asarray([0, 31, 2, 9, 4, 17, 12, 1], dtype=jnp.int32)
    mask = jnp.asarray([1, 1, 0, 1, 0, 1, 1, 0], dtype=jnp.float32)

    config = load_config(
        model="kda_hybrid_273m",
        optimizer="adamw",
        data="synthetic",
        hardware="v6e-8",
        experiment="selected",
    )
    mesh = create_mesh(config.hardware, allow_device_mismatch=True)

    def actual(x_value, weight_value):
        loss, tokens = data_parallel_linear_cross_entropy(
            x_value,
            labels,
            mask,
            weight_value,
            mesh=mesh,
            implementation="reference",
        )
        return loss, tokens

    (actual_loss, tokens), actual_grads = jax.value_and_grad(
        actual, argnums=(0, 1), has_aux=True
    )(x, weights)
    expected_loss, expected_grads = jax.value_and_grad(
        _manual_loss, argnums=(0, 3)
    )(x, labels, mask, weights)

    assert tokens == 5
    assert jnp.allclose(actual_loss, expected_loss, rtol=1.0e-6, atol=1.0e-6)
    assert jnp.allclose(actual_grads[0], expected_grads[0], rtol=1.0e-6, atol=1.0e-6)
    assert jnp.allclose(actual_grads[1], expected_grads[1], rtol=1.0e-6, atol=1.0e-6)
    assert jnp.array_equal(actual_grads[0][mask == 0], jnp.zeros((3, 16)))


def test_local_loss_rejects_mismatched_flattened_shapes():
    with jax.disable_jit():
        try:
            local_linear_cross_entropy_sum(
                jnp.zeros((8, 16)),
                jnp.zeros((7,), dtype=jnp.int32),
                jnp.ones((8,)),
                jnp.zeros((16, 32)),
                implementation="reference",
            )
        except ValueError as error:
            assert "flattened token" in str(error)
        else:
            raise AssertionError("mismatched labels must be rejected")
