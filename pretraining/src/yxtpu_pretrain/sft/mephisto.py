"""SFT on the Mephisto distillation sets, in the teacher's own chat format.

The yx49k tokenizer ships the Qwen3.5 chat template verbatim, so rendering
goes through ``apply_chat_template`` rather than a hand-written scheme:
the student is trained on byte-identical text to what the teacher emits,
which is what keeps GOLD's later on-policy alignment at its measured
~99% 1:1 rate.

Two conventions this pins down, both verified against
``Qwen/Qwen3.5-4B``'s tokenizer:

* Non-thinking mode puts an EMPTY think block in the *generation prompt*
  (``<|im_start|>assistant\\n<think>\\n\\n</think>\\n\\n``), not in the
  assistant's output. It is therefore prompt - masked - and the student
  learns to continue straight into the answer, exactly as the teacher was
  sampled.
* The trained span is ``answer + "<|im_end|>\\n"``; prompt and completion
  tokenize disjointly (asserted at render time), so the loss mask is exact
  rather than approximate.
"""

from __future__ import annotations

import numpy as np

# The yx49k ids for the Qwen chat specials (teacher ids 248044-248046 map
# onto these through student_to_teacher.npy).
ENDOFTEXT = 49119
IM_START = 49120
IM_END = 49121
# Untrained by pretraining: ClimbMix never emits a chat special. The
# separator (49119) is excluded - it IS trained as the document boundary.
UNTRAINED_SPECIAL_RANGE = (49120, 49152)

# Published row counts, used as interleaving weights when a source is
# taken whole.
_DEFAULT_SIZES = {
    "Yxanul/Mephisto-IF_172k": 172_000,
    "Yxanul/Mephisto-Knowledge_538k": 538_861,
}


def render_example(tokenizer, record, *, system: str | None = None):
    """Returns (token ids, loss mask) for one Mephisto row.

    ``system`` overrides the row's own system column when given. The mask
    trains the assistant answer and its ``<|im_end|>`` only.
    """
    messages = list(record["messages"])
    prompt_messages = [m for m in messages if m["role"] != "assistant"]
    system_text = system if system is not None else record.get("system")
    if system_text and not any(m["role"] == "system" for m in prompt_messages):
        prompt_messages = [{"role": "system", "content": system_text}, *prompt_messages]
    answers = [m for m in messages if m["role"] == "assistant"]
    if len(answers) != 1:
        raise ValueError("expected exactly one assistant turn")

    prompt_text = tokenizer.apply_chat_template(
        prompt_messages, tokenize=False, add_generation_prompt=True
    )
    completion_text = answers[0]["content"] + "<|im_end|>\n"
    prompt_ids = tokenizer.encode(prompt_text, add_special_tokens=False)
    completion_ids = tokenizer.encode(completion_text, add_special_tokens=False)
    ids = prompt_ids + completion_ids
    mask = [0] * len(prompt_ids) + [1] * len(completion_ids)
    return ids, mask


class MephistoIterator:
    """Streams one or more Mephisto repos, renders, and packs whole rows.

    Each repo is streamed, optionally capped to ``limit`` rows, sharded
    across processes by node, then merged by probability-weighted
    interleaving so the mixture is stationary for the whole epoch rather
    than draining one source at a time. Examples are short - so
    rows hold complete examples and pad tails are masked out and excluded
    from attention via segment 0, never splitting an example.
    """

    def __init__(
        self,
        tokenizer,
        *,
        datasets,
        sequence_length,
        process_batch,
        process_index,
        process_count,
        epochs=1,
        system=None,
        max_render_tokens=None,
        buffer_rows=256,
        shuffle_buffer=10_000,
        seed=0,
        targets=None,
    ):
        self._tok = tokenizer
        # Each spec is "repo" or "repo:limit"; the limit caps rows taken
        # from that source and also sets its interleaving weight.
        self._specs = []
        for entry in datasets:
            if isinstance(entry, (tuple, list)):
                repo, limit = entry
            elif ":" in entry and not entry.endswith(":"):
                repo, _, raw = entry.rpartition(":")
                limit = int(raw) if raw.isdigit() else None
                if limit is None:
                    repo, limit = entry, None
            else:
                repo, limit = entry, None
            self._specs.append((repo, limit))
        self._shuffle_buffer = shuffle_buffer
        self._seed = seed
        self._epochs = epochs
        self._system = system
        self._pi, self._pc = process_index, process_count
        self._B, self._T = process_batch, sequence_length
        self._max_render = max_render_tokens or sequence_length + 1
        self._buffer_rows = buffer_rows
        self._epoch = 0
        self._sources = None
        self._pool: list[tuple[np.ndarray, np.ndarray]] = []
        self._exhausted = False
        self._positions = np.tile(
            np.arange(sequence_length, dtype=np.int32), (process_batch, 1)
        )
        # A GoldTargetStore of precomputed teacher targets. When present,
        # every rendered example is looked up by its token-ids hash and the
        # top-K triple rides through packing next to the loss mask; rows the
        # store does not know are dropped and counted - a nonzero count on
        # a full store means the render drifted from the precompute.
        self._targets = targets
        self.rows_consumed = 0
        self.rows_dropped_oversize = 0
        self.rows_missing_targets = 0
        self.pad_tokens = 0
        self.epochs_started = 0
        self.metadata = {"datasets": [list(s) for s in self._specs],
                         "epochs": epochs, "streaming": True,
                         "shuffle_buffer": shuffle_buffer,
                         "gold_targets": bool(targets)}

    def _open_epoch(self):
        """Builds one uniformly-mixed stream over all sources.

        Round-robin draining put every source's tail at the end of the
        epoch: with IF (172k) against Knowledge (538k) the back half of
        each epoch was Knowledge-only, and the model collapsed onto that
        format. Probability-weighted interleaving proportional to the
        (possibly capped) source sizes keeps the mixture stationary from
        the first step to the last, and a shuffle buffer on top
        decorrelates neighbouring rows.
        """
        from datasets import interleave_datasets, load_dataset
        from datasets.distributed import split_dataset_by_node

        streams, weights = [], []
        for spec, limit in self._specs:
            if spec.startswith("/"):
                # A local JSONL, already sharded per host (the /dev/shm
                # pattern: download once, global-shuffle with shuf, split
                # round-robin, one file per worker). No node split - the
                # file IS this node's share - and interleaving weight 1
                # unless a limit says otherwise.
                stream = load_dataset(
                    "json", data_files=spec, split="train", streaming=True
                )
                if limit:
                    stream = stream.take(limit)
                streams.append(stream)
                weights.append(float(limit or 1))
                continue
            stream = load_dataset(spec, split="train", streaming=True)
            if limit:
                stream = stream.take(limit)
            if self._pc > 1:
                stream = split_dataset_by_node(
                    stream, rank=self._pi, world_size=self._pc
                )
            streams.append(stream)
            weights.append(float(limit or _DEFAULT_SIZES.get(spec, 1)))
        total = sum(weights)
        seed = self._seed + self._epoch
        if len(streams) > 1:
            merged = interleave_datasets(
                streams,
                probabilities=[w / total for w in weights],
                seed=seed,
                stopping_strategy="all_exhausted",
            )
        else:
            merged = streams[0]
        if self._shuffle_buffer:
            merged = merged.shuffle(seed=seed, buffer_size=self._shuffle_buffer)
        self._sources = iter(merged)
        self.epochs_started += 1

    def _draw(self):
        """Next row from the mixed stream; None when the epoch is dry."""
        if self._sources is None:
            self._open_epoch()
        try:
            return next(self._sources)
        except StopIteration:
            return None

    def _refill(self) -> bool:
        rendered = []
        while len(rendered) < self._buffer_rows:
            record = self._draw()
            if record is None:
                if self._epoch + 1 < self._epochs:
                    self._epoch += 1
                    self._open_epoch()
                    continue
                break
            try:
                ids, mask = render_example(
                    self._tok, record, system=self._system
                )
            except (KeyError, ValueError):
                continue
            if len(ids) > self._max_render:
                self.rows_dropped_oversize += 1
                continue
            targets = None
            if self._targets is not None:
                targets = self._targets.lookup(ids)
                if targets is None:
                    self.rows_missing_targets += 1
                    continue
            rendered.append(
                (np.asarray(ids, np.int32), np.asarray(mask, np.float32),
                 targets)
            )
            self.rows_consumed += 1
        if not rendered:
            return False
        self._pool.extend(rendered)
        return True

    def _assemble_row(self):
        capacity = self._T + 1
        row_ids, row_mask, row_targets, filled = [], [], [], 0
        while filled < capacity:
            picked = None
            for position, (ids, *_) in enumerate(self._pool):
                if len(ids) <= capacity - filled:
                    picked = position
                    break
            if picked is None:
                if not self._pool and not self._exhausted and self._refill():
                    continue
                if not self._pool:
                    self._exhausted = True
                break
            ids, mask, targets = self._pool.pop(picked)
            row_ids.append(ids)
            row_mask.append(mask)
            row_targets.append(targets)
            filled += len(ids)
        if not filled:
            return None
        # One segment id per packed example, positions restarting at each.
        # Previously every real token shared segment 1 and positions ran
        # straight through the row, so example k+1 attended into example k
        # and read continuing positions. Student and teacher saw the same
        # polluted prefix, so the GOLD pairing stayed sound - but the
        # teacher's distribution at the start of a packed example was then
        # conditioned on an unrelated preceding conversation, which is not
        # the conditioning it generated under. Its targets were answering a
        # question it was never asked.
        segments = np.zeros(capacity, np.int32)
        positions = np.zeros(capacity, np.int32)
        offset = 0
        for index, ids in enumerate(row_ids):
            segments[offset:offset + len(ids)] = index + 1
            positions[offset:offset + len(ids)] = np.arange(len(ids))
            offset += len(ids)
        packed_targets = None
        if self._targets is not None:
            # Store position i of an example is the teacher's distribution
            # over its token i+1, so it lands at the same token index the
            # example occupies in the row: input position offset+i then
            # supervises label row[offset+i+1], exactly the pairing the
            # objective assumes. Pad and the row tail stay zero and are
            # never trained (their loss_mask is zero).
            k = self._targets.k
            topk_ids = np.zeros((capacity, k), np.int32)
            topk_logprobs = np.zeros((capacity, k), np.float32)
            rest_mass = np.zeros(capacity, np.float32)
            offset = 0
            for ids, targets in zip(row_ids, row_targets):
                length = len(ids)
                topk_ids[offset:offset + length] = targets[0]
                topk_logprobs[offset:offset + length] = targets[1]
                rest_mass[offset:offset + length] = targets[2]
                offset += length
            packed_targets = (topk_ids[:self._T], topk_logprobs[:self._T],
                              rest_mass[:self._T])
        pad = capacity - filled
        if pad:
            row_ids.append(np.full(pad, ENDOFTEXT, np.int32))
            row_mask.append(np.zeros(pad, np.float32))
            self.pad_tokens += pad
            # Pad keeps segment 0, which excludes it from attention.
        return (np.concatenate(row_ids), np.concatenate(row_mask),
                segments[:self._T], positions[:self._T], packed_targets)

    @property
    def stats(self) -> dict[str, float]:
        report = {
            "rows_consumed": self.rows_consumed,
            "rows_dropped_oversize": self.rows_dropped_oversize,
            "pad_tokens": self.pad_tokens,
            "epochs_started": self.epochs_started,
            "pool_rows": len(self._pool),
        }
        if self._targets is not None:
            report["rows_missing_targets"] = self.rows_missing_targets
        return report

    def get_state(self) -> dict[str, int]:
        return {"rows_consumed": int(self.rows_consumed)}

    def set_state(self, payload) -> None:
        raise RuntimeError("streaming SFT data is not resumable")

    def __iter__(self):
        return self

    def __next__(self):
        rows = []
        while len(rows) < self._B:
            row = self._assemble_row()
            if row is None:
                raise StopIteration
            rows.append(row)
        ids = np.stack([row[0] for row in rows])
        mask = np.stack([row[1] for row in rows])
        segments = np.stack([row[2] for row in rows])
        positions = np.stack([row[3] for row in rows])
        batch = {
            "input_ids": ids[:, :-1],
            "labels": ids[:, 1:],
            "loss_mask": mask[:, 1:],
            "segment_ids": segments,
            "positions": positions,
        }
        if self._targets is not None:
            batch["teacher_topk_ids"] = np.stack([row[4][0] for row in rows])
            batch["teacher_topk_logprobs"] = np.stack(
                [row[4][1] for row in rows])
            batch["teacher_rest_mass"] = np.stack([row[4][2] for row in rows])
        return batch
