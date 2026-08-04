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

"""The top-K compressed GOLD path: loss, store, packing, train step.

The quiet failures here are alignment ones again - a store row attached to
the wrong packed offset, or shifted by one against the labels, still
trains. So the packing test builds targets whose top-1 id IS the next
token and asserts the identity through the whole iterator, and the
train-step test drives the real optimizer wiring end to end.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from yxtpu_pretrain.distillation.gold_loss import (
    gold_position_loss,
    gold_topk_position_loss,
    project_teacher_logits,
    topk_teacher_targets,
)

BATCH, TIME, STUDENT_VOCAB, TEACHER_VOCAB = 2, 6, 16, 40


def _projected_teacher(seed=2):
    teacher_logits = jax.random.normal(
        jax.random.key(seed), (BATCH, TIME, TEACHER_VOCAB)) * 2.0
    mapping = jnp.arange(STUDENT_VOCAB, dtype=jnp.int32) * 2
    return project_teacher_logits(teacher_logits, mapping, block=8)


def _mask():
    return jnp.asarray([[0, 1, 1, 1, 1, 0], [1, 1, 1, 1, 0, 0]], jnp.float32)


# ---------------------------------------------------------------- the loss


def test_topk_at_full_width_reproduces_the_full_loss():
    """At K = V the compression is exact: the tail equals the residual and
    the divergence must match ``gold_position_loss`` on the same batch."""
    matched, residual = _projected_teacher()
    ids, logprobs, rest = topk_teacher_targets(matched, residual, STUDENT_VOCAB)
    np.testing.assert_allclose(np.asarray(rest), np.asarray(residual),
                               atol=1e-5)
    student_logits = jax.random.normal(
        jax.random.key(7), (BATCH, TIME, STUDENT_VOCAB))
    full, _ = gold_position_loss(
        student_logits, matched, residual, _mask(), beta=0.0)
    compressed, _ = gold_topk_position_loss(
        student_logits, ids, logprobs, rest, _mask(), beta=0.0)
    assert float(compressed) == pytest.approx(float(full), rel=1e-5)


def test_topk_zero_at_match_and_the_coverage_penalty_is_log_kept():
    """Zero exactly when the student matches the renormalized teacher on K
    and carries no mass outside; moving half the student's mass off the K
    set must cost -log(1/2) - the coverage term, measured."""
    k = 4
    matched, residual = _projected_teacher(seed=11)
    ids, logprobs, rest = topk_teacher_targets(matched, residual, k)
    log_kept = np.log1p(-np.asarray(rest))
    target_logprobs = np.asarray(logprobs) - log_kept[..., None]

    concentrated = np.full((BATCH, TIME, STUDENT_VOCAB), -1e9, np.float32)
    np.put_along_axis(concentrated, np.asarray(ids), target_logprobs, axis=-1)
    loss, _ = gold_topk_position_loss(
        jnp.asarray(concentrated), ids, logprobs, rest, _mask(), beta=0.0)
    assert float(loss) == pytest.approx(0.0, abs=1e-5)

    outside = STUDENT_VOCAB - k
    spread = np.full(
        (BATCH, TIME, STUDENT_VOCAB),
        np.log(0.5 / outside), np.float32)
    np.put_along_axis(
        spread, np.asarray(ids), target_logprobs + np.log(0.5), axis=-1)
    loss, _ = gold_topk_position_loss(
        jnp.asarray(spread), ids, logprobs, rest, _mask(), beta=0.0)
    assert float(loss) == pytest.approx(np.log(2.0), rel=1e-3)


@pytest.mark.parametrize("beta", [0.5, 1.0])
def test_topk_interior_betas_are_zero_at_a_match_and_positive_off_it(beta):
    k = 4
    matched, residual = _projected_teacher(seed=13)
    ids, logprobs, rest = topk_teacher_targets(matched, residual, k)
    target_logprobs = np.asarray(logprobs) - np.log1p(
        -np.asarray(rest))[..., None]
    student = np.full((BATCH, TIME, STUDENT_VOCAB), -1e9, np.float32)
    np.put_along_axis(student, np.asarray(ids), target_logprobs, axis=-1)
    at_match, _ = gold_topk_position_loss(
        jnp.asarray(student), ids, logprobs, rest, _mask(), beta=beta)
    assert float(at_match) == pytest.approx(0.0, abs=1e-5)
    perturbed = student.copy()
    perturbed[..., 0] += 2.5
    off_match, _ = gold_topk_position_loss(
        jnp.asarray(perturbed), ids, logprobs, rest, _mask(), beta=beta)
    assert float(off_match) > 1e-4


def test_topk_masked_positions_receive_no_gradient():
    k = 4
    matched, residual = _projected_teacher(seed=17)
    ids, logprobs, rest = topk_teacher_targets(matched, residual, k)
    mask = jnp.asarray([[1, 0, 0, 1, 0, 0], [0, 0, 1, 0, 0, 1]], jnp.float32)

    def loss_of(student_logits):
        loss, _ = gold_topk_position_loss(
            student_logits, ids, logprobs, rest, mask, beta=0.0)
        return loss

    gradient = np.abs(np.asarray(jax.grad(loss_of)(
        jnp.zeros((BATCH, TIME, STUDENT_VOCAB))))).sum(-1)
    np.testing.assert_array_equal(gradient > 0, np.asarray(mask) > 0)


# --------------------------------------------------------------- the store


def test_store_roundtrips_and_reports_missing_keys(tmp_path):
    from yxtpu_pretrain.distillation.store import (
        GoldTargetStore,
        GoldTargetWriter,
    )

    k = 4
    writer = GoldTargetWriter(tmp_path, k=k, shard_examples=2)
    examples = []
    rng = np.random.default_rng(0)
    for index in range(5):
        length = 3 + index
        ids = rng.integers(0, 49152, size=length).tolist()
        topk_ids = rng.integers(0, 49152, size=(length, k))
        logprobs = -rng.random((length, k)).astype(np.float32) * 5
        rest = rng.random(length).astype(np.float32) * 0.1
        writer.add(ids, topk_ids, logprobs, rest)
        examples.append((ids, topk_ids, logprobs, rest))
    summary = writer.close()
    assert summary["examples"] == 5
    assert summary["shards"] >= 2

    store = GoldTargetStore(tmp_path)
    assert len(store) == 5
    assert store.k == k
    for ids, topk_ids, logprobs, rest in examples:
        got = store.lookup(ids)
        assert got is not None
        np.testing.assert_array_equal(got[0], topk_ids)
        # float16 storage: relative rounding, not exactness.
        np.testing.assert_allclose(got[1], logprobs, rtol=2e-3)
        np.testing.assert_allclose(got[2], rest, rtol=2e-3, atol=1e-4)
    assert store.lookup([1, 2, 3, 4, 5, 6, 7]) is None


# ------------------------------------------------- packing and the trainer


SYSTEM = ("You are a helpful assistant. Answer the user's question "
          "accurately, clearly, and concisely.")


def _record(question, answer):
    return {"messages": [{"role": "user", "content": question},
                         {"role": "assistant", "content": answer}],
            "system": SYSTEM, "uid": question}


class _FakeStream:
    def __init__(self, rows):
        self._rows = list(rows)

    def take(self, n):
        return _FakeStream(self._rows[:n])

    def shuffle(self, seed=0, buffer_size=None):
        import random

        rows = list(self._rows)
        random.Random(seed).shuffle(rows)
        return _FakeStream(rows)

    def __iter__(self):
        return iter(self._rows)


def test_store_targets_ride_packing_aligned_with_the_labels(
        tmp_path, monkeypatch):
    """The end-to-end alignment identity. Targets are built so that store
    position i's top-1 id IS the example's token i+1; after rendering,
    lookup, packing, shifting and batching, the teacher's top-1 column must
    therefore equal ``labels`` at every trained position. Any off-by-one in
    the store attachment, the packing offset or the input/label shift
    breaks the identity."""
    import datasets
    from transformers import AutoTokenizer

    from yxtpu_pretrain.distillation.store import (
        GoldTargetStore,
        GoldTargetWriter,
    )
    from yxtpu_pretrain.sft.mephisto import MephistoIterator, render_example

    tokenizer = AutoTokenizer.from_pretrained("tokenizers/yx49k",
                                              use_fast=True)
    rows = [_record(f"Q{i}?", f"Answer {i} indeed.") for i in range(8)]
    k = 4
    writer = GoldTargetWriter(tmp_path, k=k)
    for record in rows:
        ids, _ = render_example(tokenizer, record, system=SYSTEM)
        length = len(ids)
        topk_ids = np.zeros((length, k), np.int64)
        topk_ids[:length - 1, 0] = ids[1:]
        logprobs = np.full((length, k), -9.0, np.float32)
        logprobs[:, 0] = -0.05
        rest = np.full(length, 0.01, np.float32)
        writer.add(ids, topk_ids, logprobs, rest)
    writer.close()

    monkeypatch.setattr(
        datasets, "load_dataset",
        lambda spec, split, streaming: _FakeStream(rows), raising=False)
    iterator = MephistoIterator(
        tokenizer, datasets=["repo/only"], sequence_length=128,
        process_batch=2, process_index=0, process_count=1, epochs=1,
        system=SYSTEM, targets=GoldTargetStore(tmp_path),
    )
    batch = next(iter(iterator))
    assert iterator.rows_missing_targets == 0
    trained = np.asarray(batch["loss_mask"]) > 0
    assert trained.any()
    np.testing.assert_array_equal(
        np.asarray(batch["teacher_topk_ids"])[..., 0][trained],
        np.asarray(batch["labels"])[trained],
    )
    # Untrained positions carry zeroed targets and a zero tail, never NaN.
    assert np.isfinite(np.asarray(batch["teacher_topk_logprobs"])).all()


def test_the_train_step_optimizes_the_gold_objective():
    """The optimizer wiring, end to end on the tiny CPU model: the batch
    carries a synthetic teacher triple, the swapped loss_fn drives
    nnx.value_and_grad, and repeating the step on one batch must reduce the
    distillation term - the metric that proves GOLD, not CE, is what
    trains."""
    from flax import nnx
    from maxtext.common.train_state_nnx import TrainStateNNX

    from yxtpu_pretrain.config import load_config
    from yxtpu_pretrain.distillation.objective import make_gold_model_loss
    from yxtpu_pretrain.model import HybridLanguageModel
    from yxtpu_pretrain.optimizers import build_optimizer
    from yxtpu_pretrain.runtime.leaf_config import make_leaf_config
    from yxtpu_pretrain.runtime.mesh import create_mesh
    from yxtpu_pretrain.runtime.sharding import logical_mesh_context
    from yxtpu_pretrain.train import _make_train_step

    config = load_config(
        model="kda_hybrid_128k", optimizer="adamw", data="synthetic",
        hardware="v6e-8", experiment="selected",
        overrides=[
            "model.emb_dim=128", "model.mlp_dim=256", "model.num_layers=4",
            "model.num_cycles=1", "model.kda.num_heads=1",
            "model.kda.precision=full_fp32",
            "model.attention.num_query_heads=1",
            "model.attention.num_kv_heads=1", "model.vocab_size=256",
            "model.dtype=float32", "model.remat_policy=full",
            "data.sequence_length=16", "data.per_device_batch_size=2",
        ],
    )
    rows, length, vocab, k = 2, 16, 256, 8
    key = jax.random.key(3)
    tokens = jax.random.randint(key, (rows, length + 1), 1, vocab)
    labels = tokens[:, 1:]
    # Teacher's top-1 is the label; remaining K-1 columns spread over ids
    # offset from it, log-probabilities normalized over K with a 2% tail.
    topk_ids = (labels[..., None] + jnp.arange(k)) % vocab
    raw = jax.random.normal(jax.random.key(5), (rows, length, k))
    raw = raw.at[..., 0].add(3.0)
    topk_logprobs = jax.nn.log_softmax(raw, axis=-1) + jnp.log(0.98)
    batch = {
        "input_ids": tokens[:, :-1],
        "labels": labels,
        "loss_mask": jnp.ones((rows, length), jnp.float32),
        "segment_ids": jnp.ones((rows, length), jnp.int32),
        "positions": jnp.broadcast_to(
            jnp.arange(length, dtype=jnp.int32), (rows, length)),
        "teacher_topk_ids": topk_ids.astype(jnp.int32),
        "teacher_topk_logprobs": topk_logprobs,
        "teacher_rest_mass": jnp.full((rows, length), 0.02, jnp.float32),
    }

    mesh = create_mesh(config.hardware, allow_device_mismatch=True)
    rules = make_leaf_config(config).logical_axis_rules
    with logical_mesh_context(mesh, rules):
        model = HybridLanguageModel(config, mesh, rngs=nnx.Rngs(0))
        transform, _ = build_optimizer(model, config.optimizer)
        state = TrainStateNNX(
            model, nnx.Optimizer(model, transform, wrt=nnx.Param))
        step = _make_train_step(
            config,
            loss_fn=make_gold_model_loss(
                beta=0.0, distill_weight=1.0, ce_weight=0.0),
        )
        history = []
        for _ in range(6):
            metrics = step(state, batch)
            history.append(jax.tree.map(float, metrics))

    first, last = history[0], history[-1]
    for key_name in ("distill", "ce", "teacher_rest_mass",
                     "teacher_top1_is_label", "student_top1_is_label"):
        assert key_name in first, f"missing metric {key_name}"
    assert np.isfinite(first["distill"]) and np.isfinite(first["ce"])
    assert first["teacher_top1_is_label"] == pytest.approx(1.0)
    assert last["distill"] < first["distill"], (
        f"distill did not fall: {first['distill']} -> {last['distill']}")
    assert first["teacher_rest_mass"] == pytest.approx(0.02, rel=1e-3)
