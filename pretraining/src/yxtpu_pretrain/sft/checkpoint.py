"""Process-0 SFT checkpoint artifacts: full replicated state, no orbax.

Under pure data parallelism every host holds a complete copy of the train
state, so one host's pickle is a complete artifact. Saves are host-side
only (no barriers); other processes simply stall at their next collective
while process 0 writes. These are export/eval artifacts - resuming a
multi-host run from them would require copying the file to every host.
"""

from __future__ import annotations

import json
import pickle
from pathlib import Path

import jax
import numpy as np

from yxtpu_pretrain.runtime.checkpoints import _persistent_state, checkpoint_metadata


def save_sft_checkpoint(directory, step, train_state, iterator, config) -> None:
    if jax.process_index() != 0:
        return
    root = Path(directory) / str(step)
    tmp = root.with_name(root.name + ".tmp")
    if tmp.exists():
        for old in tmp.iterdir():
            old.unlink()
    tmp.mkdir(parents=True, exist_ok=True)
    pure = _persistent_state(train_state).to_pure_dict()
    host_tree = jax.tree.map(lambda leaf: np.asarray(jax.device_get(leaf)), pure)
    with open(tmp / "state.pkl", "wb") as handle:
        pickle.dump(host_tree, handle, protocol=5)
    (tmp / "iterator.json").write_text(json.dumps(iterator.get_state()))
    (tmp / "metadata.json").write_text(
        json.dumps(
            checkpoint_metadata(config, tokenizer=config.data.tokenizer)
            | {"step": step},
            default=str,
        )
    )
    if root.exists():
        for old in root.iterdir():
            old.unlink()
        root.rmdir()
    tmp.rename(root)


def load_sft_state(directory, step):
    with open(Path(directory) / str(step) / "state.pkl", "rb") as handle:
        return pickle.load(handle)
