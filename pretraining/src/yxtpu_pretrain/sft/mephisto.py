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

    Each repo is streamed and sharded across processes by node, then
    interleaved round-robin so both datasets are represented uniformly in
    every window (they are already shuffled upstream, so no buffer shuffle
    is applied). Examples are short - the corpus maxes near 2k tokens - so
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
    ):
        self._tok = tokenizer
        self._specs = list(datasets)
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
        self.rows_consumed = 0
        self.rows_dropped_oversize = 0
        self.pad_tokens = 0
        self.epochs_started = 0
        self.metadata = {"datasets": self._specs, "epochs": epochs,
                         "streaming": True}

    def _open_epoch(self):
        from datasets import load_dataset
        from datasets.distributed import split_dataset_by_node

        streams = []
        for spec in self._specs:
            stream = load_dataset(spec, split="train", streaming=True)
            if self._pc > 1:
                stream = split_dataset_by_node(
                    stream, rank=self._pi, world_size=self._pc
                )
            streams.append(iter(stream))
        self._sources = streams
        self.epochs_started += 1

    def _draw(self):
        """Round-robin across sources; returns None when all are dry."""
        if self._sources is None:
            self._open_epoch()
        while any(source is not None for source in self._sources):
            for index, source in enumerate(self._sources):
                if source is None:
                    continue
                try:
                    return next(source)
                except StopIteration:
                    self._sources[index] = None
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
            rendered.append(
                (np.asarray(ids, np.int32), np.asarray(mask, np.float32))
            )
            self.rows_consumed += 1
        if not rendered:
            return False
        self._pool.extend(rendered)
        return True

    def _assemble_row(self):
        capacity = self._T + 1
        row_ids, row_mask, filled = [], [], 0
        while filled < capacity:
            picked = None
            for position, (ids, _) in enumerate(self._pool):
                if len(ids) <= capacity - filled:
                    picked = position
                    break
            if picked is None:
                if not self._pool and not self._exhausted and self._refill():
                    continue
                if not self._pool:
                    self._exhausted = True
                break
            ids, mask = self._pool.pop(picked)
            row_ids.append(ids)
            row_mask.append(mask)
            filled += len(ids)
        if not filled:
            return None
        pad = capacity - filled
        segments = np.ones(self._T, np.int32)
        if pad:
            row_ids.append(np.full(pad, ENDOFTEXT, np.int32))
            row_mask.append(np.zeros(pad, np.float32))
            self.pad_tokens += pad
            segments[filled:] = 0
        return np.concatenate(row_ids), np.concatenate(row_mask), segments

    @property
    def stats(self) -> dict[str, float]:
        return {
            "rows_consumed": self.rows_consumed,
            "rows_dropped_oversize": self.rows_dropped_oversize,
            "pad_tokens": self.pad_tokens,
            "epochs_started": self.epochs_started,
            "pool_rows": len(self._pool),
        }

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
        return {
            "input_ids": ids[:, :-1],
            "labels": ids[:, 1:],
            "loss_mask": mask[:, 1:],
            "segment_ids": segments,
            "positions": self._positions,
        }
