"""EleutherAI lm-evaluation-harness adapter for the live NNX model."""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx
from jax.experimental import multihost_utils
from jax.sharding import NamedSharding, PartitionSpec
from lm_eval.api.model import TemplateLM

from yxtpu_pretrain.config import ResolvedConfig
from yxtpu_pretrain.runtime.data import load_fast_tokenizer
from yxtpu_pretrain.runtime.sharding import logical_mesh_context

@dataclasses.dataclass(frozen=True)
class GenerationSettings:
    """Everything ``generate_until`` needs beyond the model itself.

    Present only when generative tasks are wanted: its absence keeps
    scoring-only suites from silently sampling.
    """

    sampling: Any
    max_gen_toks: int = 512
    end_token: int = 128005  # <|im_end|>
    pad_token: int = 128000  # <|endoftext|>
    seed: int = 0
    apply_chat_template: bool = True
    strip_reasoning: bool = True


_PRIMARY_METRICS = {
    "hellaswag": "acc_norm",
    "piqa": "acc_norm",
    "arc_easy": "acc_norm",
    "arc_challenge": "acc_norm",
    "openbookqa": "acc_norm",
    "sciq": "acc",
    "boolq": "acc",
    "copa": "acc",
    "commonsense_qa": "acc",
    "lambada_openai": "acc",
}


def _score_step():
    @nnx.jit
    def score(model, batch):
        logits = model(
            batch["input_ids"],
            decoder_segment_ids=batch["segment_ids"],
            decoder_positions=batch["positions"],
        )
        # log p(label) = logit[label] - logsumexp(logits). The reduction fuses
        # over the vocabulary axis, so no second [batch, sequence, vocab]
        # log-probability tensor is ever materialized; with the FP32 logits it
        # is exactly the log_softmax gather it replaces.
        logits = logits.astype(jnp.float32)
        log_normalizer = jax.nn.logsumexp(logits, axis=-1)
        selected = jnp.take_along_axis(logits, batch["labels"][..., None], axis=-1)[
            ..., 0
        ]
        mask = batch["score_mask"].astype(jnp.float32)
        loglikelihood = jnp.sum((selected - log_normalizer) * mask, axis=-1)
        greedy = jnp.all(
            (jnp.argmax(logits, axis=-1) == batch["labels"]) | (mask == 0),
            axis=-1,
        )
        return loglikelihood, greedy

    return score


class JaxHarnessLM(TemplateLM):
    """Scores lm-eval requests on all local TPU devices without model export."""

    backend = "causal"

    def __init__(
        self,
        config: ResolvedConfig,
        model,
        mesh,
        logical_axis_rules,
        *,
        generation=None,
    ):
        super().__init__()
        # Multi-host: every process runs the identical request stream
        # (simple_evaluate is fully seeded, so ordering is deterministic),
        # feeds its shard of each global batch, and allgathers the small
        # score vectors. A divergence guard in _loglikelihood_tokens turns
        # any cross-host request mismatch into an immediate error instead of
        # a silent hang.
        if config.data.tokenizer is None:
            raise ValueError("lm-eval requires data.tokenizer")
        self.config = config
        self.model = model
        self.mesh = mesh
        self.logical_axis_rules = logical_axis_rules
        self.max_length = config.data.sequence_length
        self.batch_size = (
            config.experiment.harness_eval.batch_size_per_device * jax.device_count()
        )
        self.tokenizer = load_fast_tokenizer(
            config.data.tokenizer,
            padded_vocab_size=config.model.vocab_size,
        )
        self._score = _score_step()
        self._data_matrix = NamedSharding(mesh, PartitionSpec("data", None))
        self.generation = generation
        if generation is not None:
            # Generative tasks need the chat tokenizer's specials (the SFT
            # <|im_end|> is the stop token, and chat-templated prompts encode
            # role markers), so the scoring tokenizer is replaced wholesale
            # rather than extended per call.
            from yxtpu_pretrain.sft.tokens import load_sft_tokenizer

            self.tokenizer = load_sft_tokenizer(
                config.data.tokenizer,
                padded_vocab_size=config.model.vocab_size,
            )

    @property
    def eot_token_id(self) -> int:
        return int(self.tokenizer.eos_token_id)

    @property
    def tokenizer_name(self) -> str:
        return str(self.config.data.tokenizer)

    def tok_encode(
        self,
        string: str,
        add_special_tokens: bool | None = None,
        **kwargs,
    ) -> list[int]:
        del kwargs
        return self.tokenizer.encode(
            string,
            add_special_tokens=bool(add_special_tokens),
        )

    def _prepare_request(
        self,
        context_tokens: list[int],
        continuation_tokens: list[int],
    ) -> dict[str, np.ndarray]:
        if not continuation_tokens:
            raise ValueError("lm-eval continuation must contain at least one token")
        joined = (context_tokens + continuation_tokens)[-(self.max_length + 1) :]
        continuation_length = min(len(continuation_tokens), len(joined) - 1)
        input_tokens = joined[:-1]
        labels = joined[1:]
        valid_length = len(input_tokens)
        padded_inputs = np.full(self.max_length, self.eot_token_id, dtype=np.int32)
        padded_labels = np.full(self.max_length, self.eot_token_id, dtype=np.int32)
        score_mask = np.zeros(self.max_length, dtype=np.float32)
        segment_ids = np.zeros(self.max_length, dtype=np.int32)
        padded_inputs[:valid_length] = input_tokens
        padded_labels[:valid_length] = labels
        segment_ids[:valid_length] = 1
        score_mask[valid_length - continuation_length : valid_length] = 1.0
        return {
            "input_ids": padded_inputs,
            "labels": padded_labels,
            "score_mask": score_mask,
            "segment_ids": segment_ids,
            "positions": np.arange(self.max_length, dtype=np.int32),
        }

    def _run_batch(self, examples: list[dict[str, np.ndarray]]):
        real_size = len(examples)
        if real_size < self.batch_size:
            padding = self._prepare_request([self.eot_token_id], [self.eot_token_id])
            padding["score_mask"].fill(0)
            examples = [*examples, *([padding] * (self.batch_size - real_size))]
        host_batch = {
            key: np.stack([example[key] for example in examples])
            for key in examples[0]
        }
        def globalize(value):
            array = np.asarray(value)
            if jax.process_count() == 1:
                return jax.device_put(jnp.asarray(array), self._data_matrix)
            return jax.make_array_from_callback(
                array.shape, self._data_matrix, lambda index, a=array: a[index]
            )

        device_batch = {key: globalize(value) for key, value in host_batch.items()}
        with logical_mesh_context(self.mesh, self.logical_axis_rules):
            scores, greedy = self._score(self.model, device_batch)
        if jax.process_count() > 1:
            scores, greedy = multihost_utils.global_array_to_host_local_array(
                (scores, greedy), self.mesh, (PartitionSpec(), PartitionSpec())
            )
        scores, greedy = jax.device_get((scores, greedy))
        return scores[:real_size], greedy[:real_size]

    def _loglikelihood_tokens(self, requests, **kwargs) -> list[tuple[float, bool]]:
        del kwargs
        examples = [
            self._prepare_request(context_tokens, continuation_tokens)
            for _, context_tokens, continuation_tokens in requests
        ]
        if jax.process_count() > 1:
            multihost_utils.assert_equal(
                np.int64(len(examples)),
                "lm-eval request streams diverged across hosts",
            )
        results: list[tuple[float, bool]] = []
        for start in range(0, len(examples), self.batch_size):
            scores, greedy = self._run_batch(examples[start : start + self.batch_size])
            results.extend(
                (float(score), bool(is_greedy))
                for score, is_greedy in zip(scores, greedy, strict=True)
            )
        return results

    def loglikelihood_rolling(self, requests, disable_tqdm: bool = False) -> list[float]:
        del disable_tqdm
        from lm_eval import utils

        results = []
        for request in requests:
            (text,) = request.args
            windows = list(
                map(
                    utils.make_disjoint_window,
                    utils.get_rolling_token_windows(
                        token_list=self.tok_encode(text),
                        prefix_token=self.prefix_token_id,
                        max_seq_len=self.max_length,
                        context_len=1,
                    ),
                )
            )
            token_requests = [
                ((text, ""), context, continuation)
                for context, continuation in windows
            ]
            results.append(sum(score for score, _ in self._loglikelihood_tokens(token_requests)))
        return results

    def generate_until(self, requests, disable_tqdm: bool = False) -> list[str]:
        """Serves generative tasks (IFEval, TriviaQA, BBH) from the cached
        decoder.

        Requests are grouped into fixed-size batches whose prompts share one
        decode loop; each row stops on the chat end token, and the decoded
        text is then cut at the first task-supplied stop string. Enabled only
        when the adapter was built with a generation config, so scoring-only
        suites keep failing loudly rather than silently sampling.
        """
        del disable_tqdm
        if self.generation is None:
            raise NotImplementedError(
                "this adapter was built for loglikelihood scoring only; pass "
                "generation=GenerationSettings(...) to serve generative tasks"
            )
        from yxtpu_pretrain.decode import generate

        settings = self.generation
        end_token = settings.end_token
        prepared = []
        for request in requests:
            context, arguments = request.args
            arguments = dict(arguments or {})
            until = arguments.get("until") or []
            if isinstance(until, str):
                until = [until]
            budget = int(arguments.get("max_gen_toks", settings.max_gen_toks))
            prepared.append((context, until, budget))

        outputs: list[str] = []
        width = self.batch_size
        for start in range(0, len(prepared), width):
            chunk = prepared[start : start + width]
            budget = max(item[2] for item in chunk)
            rendered = [
                self._render_generation_prompt(context) for context, _, _ in chunk
            ]
            # Pad the final chunk by repeating its last prompt: the decoder
            # takes a fixed batch, and the extra rows are discarded below.
            while len(rendered) < width:
                rendered.append(rendered[-1])
            longest = max(len(row) for row in rendered)
            keep = self.max_length - budget - 8
            if longest > keep:
                rendered = [row[-keep:] if len(row) > keep else row for row in rendered]
                longest = max(len(row) for row in rendered)
            lengths = np.asarray([len(row) for row in rendered], np.int32)
            padded = np.full((width, longest), settings.pad_token, np.int32)
            for index, row in enumerate(rendered):
                padded[index, : len(row)] = row
            with logical_mesh_context(self.mesh, self.logical_axis_rules):
                samples, _ = generate(
                    self.model,
                    jnp.asarray(padded),
                    jnp.asarray(lengths),
                    jax.random.key(settings.seed + start),
                    max_new_tokens=budget,
                    sampling=settings.sampling,
                    end_token=end_token,
                    max_length=longest + budget + 8,
                )
            samples = np.asarray(samples)
            for index, (_, until, _) in enumerate(chunk):
                row = list(samples[index, int(lengths[index]) - 1 :])
                stops = [i for i, token in enumerate(row) if token == end_token]
                generated = row[: stops[0]] if stops else row
                text = self.tokenizer.decode(generated)
                text = self._strip_reasoning(text)
                for stop in until:
                    if stop and stop in text:
                        text = text.split(stop)[0]
                outputs.append(text)
        return outputs

    def _render_generation_prompt(self, context: str) -> list[int]:
        """Chat-templates the prompt for an instruction-tuned checkpoint.

        A model trained to answer inside a role-tagged template is being
        asked the wrong question if the raw context is fed to it, so the
        template is applied unless explicitly disabled for a base model.
        """
        from yxtpu_pretrain.sft.tokens import (
            DOCUMENT_SEPARATOR,
            IM_END,
            IM_MIDDLE,
            ROLE_TOKENS,
        )

        encode = lambda text: self.tokenizer.encode(text, add_special_tokens=False)
        if not self.generation.apply_chat_template:
            return [DOCUMENT_SEPARATOR, *encode(context)]
        return [
            DOCUMENT_SEPARATOR,
            ROLE_TOKENS["user"], *encode("user"), IM_MIDDLE,
            *encode(context), IM_END,
            ROLE_TOKENS["assistant"], *encode("assistant"), IM_MIDDLE,
        ]

    def _strip_reasoning(self, text: str) -> str:
        """Drops the think block so tasks score the answer, not the trace.

        A reply whose think block never closed has no answer to score, and
        returning the trace would let a runaway generation accidentally
        satisfy a substring check; those score as empty instead.
        """
        if not self.generation.strip_reasoning:
            return text
        if "</think>" in text:
            return text.split("</think>", 1)[1].lstrip()
        if "<think>" in text:
            return ""
        return text


def flatten_harness_metrics(results: dict[str, Any]) -> dict[str, float]:
    """Flattens numeric metrics and makes the requested normalized metric primary."""
    flattened: dict[str, float] = {}
    task_results = results.get("results", {})
    for task, metrics in task_results.items():
        by_base_name: dict[str, float] = {}
        for raw_name, value in metrics.items():
            if not isinstance(value, (int, float)):
                continue
            metric = raw_name.split(",", 1)[0]
            flattened[f"{task}/{metric}"] = float(value)
            by_base_name[metric] = float(value)
        primary = _PRIMARY_METRICS.get(task)
        if primary in by_base_name:
            flattened[f"{task}/primary"] = by_base_name[primary]
    easy = flattened.get("arc_easy/primary")
    challenge = flattened.get("arc_challenge/primary")
    if easy is not None and challenge is not None:
        flattened["arc_easy_challenge_gap"] = easy - challenge
    return flattened


def _json_safe(value: Any):
    """Preserves harness provenance without trying to serialize task callables."""
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    if callable(value):
        module = getattr(value, "__module__", type(value).__module__)
        name = getattr(value, "__qualname__", getattr(value, "__name__", type(value).__name__))
        return {"callable": f"{module}.{name}"}
    return str(value)


def run_harness_evaluation(
    adapter: JaxHarnessLM,
    config: ResolvedConfig,
    *,
    run_dir: Path,
    step: int,
) -> tuple[dict[str, float], Path]:
    """Runs the pinned harness and persists its complete comparable result."""
    import lm_eval
    harness = config.experiment.harness_eval
    results = lm_eval.simple_evaluate(
        model=adapter,
        tasks=list(harness.tasks),
        num_fewshot=harness.num_fewshot,
        limit=harness.limit,
        cache_requests=harness.use_cache,
        bootstrap_iters=0,
        log_samples=False,
        random_seed=config.experiment.seed,
        numpy_random_seed=config.experiment.seed,
        torch_random_seed=config.experiment.seed,
        fewshot_random_seed=config.experiment.seed,
    )
    if results is None:
        raise RuntimeError("lm-eval returned no process-zero results")
    output_dir = run_dir / "lm_eval"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"step_{step:08d}.json"
    output_path.write_text(
        json.dumps(_json_safe(results), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return flatten_harness_metrics(results), output_path
