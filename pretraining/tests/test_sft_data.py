import numpy as np

from yxtpu_pretrain.sft.data import SFTIterator, build_packed_dataset
from yxtpu_pretrain.sft.tokens import load_sft_tokenizer, render_conversation


class _FakeStream:
    """Stand-in for the HF stream: three tiny conversations."""

    def __iter__(self):
        for i in range(3):
            yield {"conversations": [
                {"from": "human", "value": f"question {i}?"},
                {"from": "gpt", "value": f"<think>\nwork {i}\n</think>\n\nanswer {i}"},
            ]}


def test_pack_alignment_shifts_mask_with_labels(monkeypatch):
    tokenizer = load_sft_tokenizer(
        "alisawuffles/superbpe-tokenizer-128k", padded_vocab_size=128256
    )
    import yxtpu_pretrain.sft.data as data_module
    monkeypatch.setattr(
        "datasets.load_dataset", lambda *a, **k: _FakeStream(), raising=False
    )
    inputs, labels, loss_mask = build_packed_dataset(
        tokenizer, dataset="x", subset="y", rows=3, sequence_length=32
    )
    assert inputs.shape == labels.shape == loss_mask.shape
    # labels are inputs shifted by one within the flat stream
    flat_in, flat_lab = inputs.reshape(-1), labels.reshape(-1)
    assert np.array_equal(flat_in[1:], flat_lab[:-1])
    # every trained position predicts assistant content or its <|im_end|>;
    # reconstruct the render masks to verify the shift exactly
    stream_ids, stream_mask = [], []
    for record in _FakeStream():
        ids, mask = render_conversation(tokenizer, [
            {"role": "user" if t["from"] == "human" else "assistant",
             "content": t["value"]} for t in record["conversations"]])
        stream_ids += ids
        stream_mask += mask
    n = loss_mask.size
    assert np.array_equal(loss_mask.reshape(-1), np.asarray(stream_mask[1 : n + 1], np.float32))
    trained = loss_mask.reshape(-1) > 0
    assert trained.any() and not trained.all()

    iterator = SFTIterator(inputs, labels, loss_mask, process_batch=1,
                           epochs=2, seed=7, process_index=0, process_count=1)
    batches = list(iterator)
    assert len(batches) == 2 * len(inputs)
    assert batches[0]["input_ids"].shape == (1, 32)
    assert set(batches[0]) == {"input_ids", "labels", "loss_mask", "segment_ids", "positions"}
