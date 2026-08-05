"""Precomputed GOLD teacher targets, keyed by the rendered token ids.

The lambda=0 stage scores a FIXED dataset, so the teacher's 5.3x-per-token
forward tax can be paid once instead of once per epoch per sweep. Each
example is scored in isolation - positions from zero, its own segment -
which is exactly the conditioning the teacher generated under, and
sidesteps both packing pollution and the recurrent-state segment leak
entirely on the teacher side.

Keying is by a hash of the example's rendered token ids, not by stream
order: the training iterator shuffles and interleaves, and any change to
batch size, packing or process topology reorders consumption. The hash
makes the store order-independent and doubles as a render-consistency
check - a tokenizer or system-prompt drift between precompute and
training shows up as missing targets (counted, and fatal in spirit),
never as silently wrong supervision.

Layout: sharded ``<prefix>shard_NNNNN.npz`` files plus a
``<prefix>manifest.json``. Multiple prefixes may share a directory - the
parallel precompute writes one prefix per host shard and the reader
merges every manifest it finds. Shards hold per-example top-K ids
(uint16 - the yx49k vocabulary fits), logprobs and tail mass (float16;
the values are logs of top-20-sampled probabilities, well inside range),
concatenated along positions with an offsets table.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np


def example_key(ids) -> str:
    """Stable 16-hex-digit key for one rendered example."""
    payload = np.asarray(ids, np.int32).tobytes()
    return hashlib.blake2b(payload, digest_size=8).hexdigest()


class GoldTargetWriter:
    """Accumulates per-example targets into sharded npz files."""

    def __init__(self, directory, *, k, prefix="", shard_examples=1024,
                 metadata=None):
        self._dir = Path(directory)
        self._dir.mkdir(parents=True, exist_ok=True)
        self.k = int(k)
        self._prefix = prefix
        self._shard_examples = shard_examples
        self._metadata = dict(metadata or {})
        self._entries: dict[str, list] = {}
        self._pending: list[tuple[str, np.ndarray, np.ndarray, np.ndarray]] = []
        self._shard_index = 0
        self.examples = 0
        self.positions = 0
        self.collisions = 0

    def add(self, ids, topk_ids, topk_logprobs, rest_mass) -> None:
        length = len(ids)
        topk_ids = np.asarray(topk_ids)
        if topk_ids.shape != (length, self.k):
            raise ValueError(
                f"expected [{length}, {self.k}] targets, got {topk_ids.shape}"
            )
        key = example_key(ids)
        if key in self._entries or any(k == key for k, *_ in self._pending):
            # Duplicate rendered examples exist in the data; first one wins,
            # and the count is reported so a systematic dupe is visible.
            self.collisions += 1
            return
        self._pending.append((
            key,
            topk_ids.astype(np.uint16),
            np.asarray(topk_logprobs).astype(np.float16),
            np.asarray(rest_mass).astype(np.float16),
        ))
        self.examples += 1
        self.positions += length
        if len(self._pending) >= self._shard_examples:
            self._flush()

    def _flush(self) -> None:
        if not self._pending:
            return
        name = f"{self._prefix}shard_{self._shard_index:05d}.npz"
        lengths = np.asarray([row[1].shape[0] for row in self._pending], np.int64)
        offsets = np.concatenate([[0], np.cumsum(lengths)])
        np.savez_compressed(
            self._dir / name,
            offsets=offsets,
            topk_ids=np.concatenate([row[1] for row in self._pending]),
            topk_logprobs=np.concatenate([row[2] for row in self._pending]),
            rest_mass=np.concatenate([row[3] for row in self._pending]),
        )
        for row_index, (key, *_rest) in enumerate(self._pending):
            self._entries[key] = [name, row_index]
        self._pending.clear()
        self._shard_index += 1

    def close(self) -> dict:
        self._flush()
        manifest = {
            "k": self.k,
            "examples": self.examples,
            "positions": self.positions,
            "collisions": self.collisions,
            "metadata": self._metadata,
            "entries": self._entries,
        }
        path = self._dir / f"{self._prefix}manifest.json"
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(manifest, handle)
        return {"manifest": str(path), "examples": self.examples,
                "positions": self.positions, "collisions": self.collisions,
                "shards": self._shard_index}


class GoldTargetStore:
    """Reads targets back by example ids; caches decoded shards.

    The cache is unbounded - a full Mephisto-scale store is tens of GB
    decoded, which the training hosts' RAM absorbs; revisit with an LRU if
    that stops being true.
    """

    def __init__(self, directory):
        self._dir = Path(directory)
        manifests = sorted(self._dir.glob("*manifest.json"))
        if not manifests:
            raise FileNotFoundError(f"no *manifest.json under {self._dir}")
        self.k = None
        self.metadata = []
        self._entries: dict[str, list] = {}
        for path in manifests:
            with open(path, encoding="utf-8") as handle:
                manifest = json.load(handle)
            if self.k is None:
                self.k = int(manifest["k"])
            elif self.k != int(manifest["k"]):
                raise ValueError(
                    f"manifests disagree on k: {self.k} vs {manifest['k']}"
                )
            self.metadata.append(manifest.get("metadata", {}))
            self._entries.update(manifest["entries"])
        self._shards: dict[str, dict] = {}

    def __len__(self) -> int:
        return len(self._entries)

    def _shard(self, name: str) -> dict:
        if name not in self._shards:
            with np.load(self._dir / name) as archive:
                self._shards[name] = {key: archive[key] for key in archive}
        return self._shards[name]

    def lookup(self, ids):
        """(topk_ids [L,K] i32, logprobs [L,K] f32, rest [L] f32), or None."""
        entry = self._entries.get(example_key(ids))
        if entry is None:
            return None
        shard = self._shard(entry[0])
        start = int(shard["offsets"][entry[1]])
        stop = int(shard["offsets"][entry[1] + 1])
        if stop - start != len(ids):
            raise ValueError(
                f"stored length {stop - start} != example length {len(ids)}"
            )
        return (
            shard["topk_ids"][start:stop].astype(np.int32),
            shard["topk_logprobs"][start:stop].astype(np.float32),
            shard["rest_mass"][start:stop].astype(np.float32),
        )
