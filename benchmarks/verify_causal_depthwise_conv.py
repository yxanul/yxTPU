"""Check the shifted depthwise conv matches Flax's CAUSAL conv exactly."""
import jax, jax.numpy as jnp, numpy as np
from flax import nnx
from maxtext.layers.kimi_delta_attention import _causal_depthwise_conv

B, T, C, K = 2, 32, 48, 4
x = jax.random.normal(jax.random.key(0), (B, T, C), dtype=jnp.float32)
conv = nnx.Conv(
    in_features=C, out_features=C, kernel_size=(K,),
    feature_group_count=C, padding="CAUSAL", use_bias=False,
    dtype=jnp.float32, param_dtype=jnp.float32, rngs=nnx.Rngs(1),
)
# Original path: pre-pad K-1, CAUSAL conv, keep the last T outputs.
ref = conv(jnp.pad(x, ((0, 0), (K - 1, 0), (0, 0))))[:, -T:]
got = _causal_depthwise_conv(x, conv.kernel.value)
print("kernel shape", conv.kernel.value.shape)
print("max abs diff", float(jnp.max(jnp.abs(ref - got))))
np.testing.assert_allclose(np.asarray(got), np.asarray(ref), rtol=1e-6, atol=1e-6)

# Causality: a change at time t must not affect outputs before t.
x2 = x.at[:, T // 2].add(100.0)
d = jnp.max(jnp.abs(_causal_depthwise_conv(x2, conv.kernel.value) - got), axis=(0, 2))
assert float(jnp.max(d[: T // 2])) == 0.0, "leaked information backwards"
print("causal: no leakage before t; first affected index", int(jnp.argmax(d > 0)))
print("EQUIVALENT")
