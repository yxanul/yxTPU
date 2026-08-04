"""The frozen Qwen3.5 teacher: sharded, forward-only, already projected.

Everything downstream of ``score`` sees the teacher only as constants on
the student's vocabulary, which is what keeps the objective unable to
train it.

Sharding is not an optimization here, it is the difference between running
and not. Replicated at bfloat16 the teacher needs ~16.8G on one chip -
MaxText's layer scan carries the whole parameter set in as an input and
out as scanned state - against a v4 chip's 30.75G, which leaves too little
for a useful batch. The parameters carry logical axis names
(``sharding_names``, e.g. ``('embed', 'layers', 'gdn_head')``) that only
mean something against MaxText's own multi-axis mesh, so the mesh is built
from the config rather than hand-rolled, and each array is placed through
``logical_to_mesh_axes``. Arrays with no logical names - ``A_log``,
``dt_bias``, ``conv1d`` - are tiny and replicate.
"""

from __future__ import annotations


import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx

from yxtpu_pretrain.distillation.alignment import (
    direct_teacher_ids,
    validate_student_to_teacher,
)
from yxtpu_pretrain.distillation.gold_loss import project_teacher_logits


def _find_weights(node, depth=0):
    """Locates the parameter root by shape, not by wrapper depth."""
    if not isinstance(node, dict):
        raise ValueError("no parameter root in the restored checkpoint")
    if "decoder" in node and "token_embedder" in node:
        return node
    if depth > 4:
        raise ValueError(f"no parameter root within 4 levels; keys {list(node)}")
    for value in node.values():
        if isinstance(value, dict):
            try:
                return _find_weights(value, depth + 1)
            except ValueError:
                continue
    raise ValueError(f"no parameter root under keys {list(node)}")


def shard_params(params, mesh, rules):
    """Places every parameter per its own logical axes.

    Returns the count of sharded vs replicated arrays so a silently
    all-replicated load - which still runs, just out of memory later - is
    visible at startup.
    """
    from flax.linen.spmd import logical_to_mesh_axes
    from jax.sharding import NamedSharding, PartitionSpec

    sharded = replicated = 0
    for _, variable in nnx.to_flat_state(params):
        names = getattr(variable, "sharding_names", None)
        value = variable.value if hasattr(variable, "value") else variable
        if names:
            spec = PartitionSpec(*logical_to_mesh_axes(names, rules))
            if any(axis is not None for axis in spec):
                sharded += 1
            else:
                replicated += 1
        else:
            spec = PartitionSpec()
            replicated += 1
        variable.value = jax.device_put(value, NamedSharding(mesh, spec))
    return sharded, replicated


class Qwen35Teacher:
    """Scores student token sequences and projects onto the student vocab."""

    def __init__(
        self,
        checkpoint,
        *,
        student_to_teacher,
        base="../maxtext/src/maxtext/configs/base.yml",
        model_name="qwen3.5-4b",
        sequence=2048,
        batch=1,
        tensor_parallelism=4,
        valid_vocab=None,
        attention="dot_product",
    ):
        import orbax.checkpoint as ocp
        from jax.sharding import Mesh

        from maxtext import pyconfig
        from maxtext.common.common_types import MODEL_MODE_TRAIN
        from maxtext.models.models import Transformer
        from maxtext.utils import maxtext_utils

        mapping = np.asarray(student_to_teacher)
        validate_student_to_teacher(mapping, teacher_vocab=valid_vocab)
        self.mapping = jnp.asarray(mapping)
        self.valid_vocab = valid_vocab

        config = pyconfig.initialize([
            "gold_teacher", base, f"model_name={model_name}",
            "run_name=gold_teacher", f"per_device_batch_size={batch}",
            f"max_target_length={sequence}",
            "skip_jax_distributed_system=true", "enable_checkpointing=false",
            "scan_layers=true",
            # All three, explicitly. ici_data_parallelism defaults to -1 and
            # auto-fills every spare device, so asking only for tensor
            # parallelism silently yields data=4/tensor=1 - which then tries
            # to split a batch of 1 across 4 devices and raises
            # IndivisibleError deep inside an activation sharding
            # constraint. The teacher is scored batch-small and
            # parameter-large, so tensor parallelism is what we want.
            "ici_data_parallelism=1",
            "ici_fsdp_parallelism=1",
            f"ici_tensor_parallelism={tensor_parallelism}",
            # Splash attention rejects this sharding ("the sharding must
            # divide the mask blocks evenly between devices"). Only 8 of 32
            # layers use full attention and this is forward-only scoring, so
            # dot_product is the cheap way out: at seq 1024 with 16 heads
            # the materialized matrix is ~32MB, transient.
            f"attention={attention}",
        ])
        self.config = config
        mesh = Mesh(maxtext_utils.create_device_mesh(config), config.mesh_axes)
        self.mesh = mesh

        model = Transformer(config=config, mesh=mesh, quant=None,
                            rngs=nnx.Rngs(0), model_mode=MODEL_MODE_TRAIN)
        graphdef, params, rest = nnx.split(model, nnx.Param, ...)

        manager = ocp.CheckpointManager(
            checkpoint, options=ocp.CheckpointManagerOptions())
        step = manager.latest_step()
        restored = manager.restore(
            step, args=ocp.args.Composite(items=ocp.args.PyTreeRestore()))
        nnx.replace_by_pure_dict(params, _find_weights(restored["items"]))

        count = shard_params(params, mesh, config.logical_axis_rules)
        print(f"teacher: {count[0]} arrays sharded, {count[1]} replicated",
              flush=True)
        self.model = nnx.merge(graphdef, params, rest)
        self._scorer = None
        self._topk_scorer = None

    def score(self, student_input_ids, positions, segment_ids):
        """Teacher distribution over the student's vocabulary, per position.

        ``student_input_ids`` is the batch the *student* consumes. Position
        i of the result is the teacher's distribution over the token that
        follows ``student_input_ids[:i+1]`` - the same prefix the student
        reads, in the same segmentation - so it pairs with student position
        i directly, with no further shift.
        """
        if self._scorer is None:
            # nnx.jit, not jax.jit: MaxText's decoder reassigns self.layers
            # inside the layer scan (nnx_decoders.py:1771), and mutating a
            # module built outside the trace raises TraceContextError.
            # nnx.jit splits and remerges the module across the boundary so
            # the write lands at the right trace level.
            self._scorer = nnx.jit(_score, static_argnums=(5,))
        return self._scorer(self.model, student_input_ids, positions,
                            segment_ids, self.mapping, self.valid_vocab)


    def score_topk(self, student_input_ids, positions, segment_ids, k: int):
        """``score`` compressed on device to (ids, logprobs, rest) at top-K.

        Only K columns per position ever leave the device, which is what
        makes precomputing targets for a whole dataset practical.
        """
        if self._topk_scorer is None:
            self._topk_scorer = nnx.jit(_score_topk, static_argnums=(5, 6))
        return self._topk_scorer(self.model, student_input_ids, positions,
                                 segment_ids, self.mapping, self.valid_vocab,
                                 int(k))


def _project(model, student_input_ids, positions, segment_ids, mapping,
             valid_vocab):
    from maxtext.common.common_types import MODEL_MODE_TRAIN

    teacher_ids = direct_teacher_ids(student_input_ids, mapping)
    logits = model(teacher_ids, positions, segment_ids,
                   enable_dropout=False, model_mode=MODEL_MODE_TRAIN)
    if isinstance(logits, tuple):
        logits = logits[0]
    return project_teacher_logits(logits, mapping, valid_vocab=valid_vocab)


def _score(model, student_input_ids, positions, segment_ids, mapping,
           valid_vocab):
    matched, residual = _project(
        model, student_input_ids, positions, segment_ids, mapping, valid_vocab)
    # Constants downstream: the objective cannot train what it cannot
    # differentiate, and this makes that structural rather than incidental.
    return jax.lax.stop_gradient(matched), jax.lax.stop_gradient(residual)


def _score_topk(model, student_input_ids, positions, segment_ids, mapping,
                valid_vocab, k):
    from yxtpu_pretrain.distillation.gold_loss import topk_teacher_targets

    matched, residual = _project(
        model, student_input_ids, positions, segment_ids, mapping, valid_vocab)
    top_ids, top_logprobs, rest = topk_teacher_targets(matched, residual, k)
    return (jax.lax.stop_gradient(top_ids),
            jax.lax.stop_gradient(top_logprobs),
            jax.lax.stop_gradient(rest))
