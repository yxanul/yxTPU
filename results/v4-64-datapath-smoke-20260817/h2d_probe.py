"""Empirical probe of the training loop's host->device batch path on the v4-64.

Replicates train._device_batch (jnp.asarray -> host_local_array_to_global_array)
against variants, with the devices idle and with a heavy computation in flight
(with and without HBM ballast), and reports on-device sizes of the uint8 pixel
block. Read-only: no files written, no training state touched.
"""
import time
import numpy as np
import jax, jax.numpy as jnp
from jax.sharding import Mesh, NamedSharding, PartitionSpec
from jax.experimental import multihost_utils

jax.distributed.initialize()
pi = jax.process_index()
devs = jax.devices()
ldevs = jax.local_devices()
mesh = Mesh(np.asarray(devs).reshape(-1), ("data",))
P = PartitionSpec("data", None)
NS = NamedSharding(mesh, P)
local = len(ldevs)
PDB, T, IMG, S = 4, 4096, 4, 448
B = PDB * local

def say(*a):
    if pi == 0:
        print(*a, flush=True)

rng = np.random.default_rng(0)
def host_batch(patchify=False):
    b = {
        "input_ids": rng.integers(0, 49152, (B, T), dtype=np.int32),
        "labels": rng.integers(0, 49152, (B, T), dtype=np.int32),
        "loss_mask": np.ones((B, T), np.float32),
        "segment_ids": np.ones((B, T), np.int32),
        "positions": np.tile(np.arange(T, dtype=np.int32), (B, 1)),
        "vision_mask": np.zeros((B, T), np.float32),
    }
    img = rng.integers(0, 255, (B, IMG, S, S, 3), dtype=np.uint8)
    if patchify:
        g, p = S // 16, 16
        img = np.ascontiguousarray(img.reshape(B * IMG, g, p, g, p, 3).transpose(0, 1, 3, 2, 4, 5).reshape(B, IMG, g * g, p * p * 3))
    b["images"] = img
    return b

def db_current(batch):   # exact copy of train._device_batch (multi-process branch)
    return {k: multihost_utils.host_local_array_to_global_array(jnp.asarray(v), mesh, P) for k, v in batch.items()}
def db_numpy(batch):     # same call, numpy in (no device-0 staging)
    return {k: multihost_utils.host_local_array_to_global_array(np.asarray(v), mesh, P) for k, v in batch.items()}
def db_mpld(batch):      # jax.make_array_from_process_local_data
    return {k: jax.make_array_from_process_local_data(NS, np.asarray(v)) for k, v in batch.items()}

X = jax.jit(lambda: jnp.ones((len(devs) * 4096, 4096), jnp.float32), out_shardings=NS)()
@jax.jit
def burn(x, n):
    x = x.reshape(-1, 4096, 4096)
    def body(i, x):
        return jnp.tanh(jnp.einsum("bij,bjk->bik", x, x, preferred_element_type=jnp.float32)) * 0.5 + x
    return jax.lax.fori_loop(0, n, body, x).reshape(-1, 4096)

burn(X, 4).block_until_ready()  # compile
t = time.perf_counter(); burn(X, 64).block_until_ready(); dt = time.perf_counter() - t
say(f"burn(64) = {dt:.3f}s")
n_iters = max(2, int(64 * 1.6 / max(dt, 1e-3)))
t = time.perf_counter(); burn(X, n_iters).block_until_ready(); dt = time.perf_counter() - t
say(f"emulated step: burn({n_iters}) = {dt:.3f}s")

def mem():
    s = ldevs[0].memory_stats()
    return {k: round(s[k] / 1e9, 2) for k in ("bytes_in_use", "bytes_limit", "peak_bytes_in_use") if k in s}
say("memory (GB) before ballast:", mem())

# on-device size of the pixel block
hb = host_batch()
a = jax.device_put(hb["images"][:PDB], ldevs[0]); a.block_until_ready()
try:
    say(f"pixel block/device: logical {a.nbytes/1e6:.2f} MB, on-device {a.on_device_size_in_bytes()/1e6:.2f} MB, layout {a.format if hasattr(a,'format') else a.layout}")
except Exception as e:
    say("on_device_size probe failed:", repr(e))
del a
hbp = host_batch(patchify=True)
a = jax.device_put(hbp["images"][:PDB], ldevs[0]); a.block_until_ready()
try:
    say(f"patchified block/device: logical {a.nbytes/1e6:.2f} MB, on-device {a.on_device_size_in_bytes()/1e6:.2f} MB")
except Exception as e:
    say("on_device_size probe failed:", repr(e))
del a
a = jax.device_put(hb["input_ids"][:PDB], ldevs[0]); a.block_until_ready()
say(f"token array/device: logical {a.nbytes/1e6:.3f} MB, on-device {a.on_device_size_in_bytes()/1e6:.3f} MB"); del a

def timeit(name, fn, batch, busy):
    multihost_utils.sync_global_devices("probe")
    if busy:
        yb = burn(X, n_iters)
        time.sleep(0.05)
    t0 = time.perf_counter()
    out = fn(batch)
    t1 = time.perf_counter()
    jax.block_until_ready(out)
    t2 = time.perf_counter()
    if busy:
        yb.block_until_ready()
    t3 = time.perf_counter()
    call_ms = (t1 - t0) * 1e3; ready_ms = (t2 - t0) * 1e3
    allc = multihost_utils.process_allgather(np.asarray([call_ms, ready_ms], np.float64))
    say(f"  {name:22s} busy={busy!s:5s} call {call_ms:8.1f} ms (max host {allc[:,0].max():8.1f})  ready {ready_ms:8.1f} ms (max host {allc[:,1].max():8.1f})  end {(t3-t0)*1e3:8.1f} ms")
    del out

def suite(tag):
    say(f"== {tag}")
    for busy in (False, True):
        timeit("current(jnp+h2g)", db_current, host_batch(), busy)
        timeit("numpy+h2g", db_numpy, host_batch(), busy)
        timeit("numpy+mpld", db_mpld, host_batch(), busy)
        timeit("patchified numpy+h2g", db_numpy, host_batch(patchify=True), busy)
        timeit("patchified current", db_current, host_batch(patchify=True), busy)

suite("no ballast, pass 1")

# HBM ballast: leave ~1.5 GB free per device (the burn's temporaries need < 1 GB)
s = ldevs[0].memory_stats()
free = s["bytes_limit"] - s["bytes_in_use"]
per_dev = max(0, (free - int(1.5e9)) // 4)
if per_dev > 0:
    ballast = jax.jit(lambda: jnp.zeros((len(devs) * per_dev,), jnp.float32), out_shardings=NamedSharding(mesh, PartitionSpec("data")))()
    ballast.block_until_ready()
    say("memory (GB) with ballast:", mem())
    suite("ballast (~1.5 GB free), pass 1")
    del ballast
say("done")
