"""Typed layered configuration for standalone pretraining."""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

PACKAGE_ROOT = Path(__file__).resolve().parents[2]
CONFIG_ROOT = PACKAGE_ROOT / "configs"


class StrictModel(BaseModel):
    """Base model that rejects misspelled configuration keys."""

    model_config = ConfigDict(extra="forbid")


class KDAConfig(StrictModel):
    chunk_size: int = 64
    num_heads: int = 8
    key_head_dim: int = 128
    value_head_dim: int = 128
    conv_kernel_size: int = 4
    gate_rank: int = 128
    qk_norm: bool = True
    safe_gate: bool = True
    gate_lower_bound: float = -5.0
    precision: Literal["guarded_fp32", "full_fp32"] = "guarded_fp32"

    @model_validator(mode="after")
    def validate_production_shape(self) -> KDAConfig:
        if self.chunk_size != 64:
            raise ValueError("the production KDA kernel is specialized to chunk_size=64")
        if self.key_head_dim != 128 or self.value_head_dim != 128:
            raise ValueError("the production KDA kernel requires a 128x128 recurrent state")
        return self


class AttentionConfig(StrictModel):
    implementation: Literal["tokamax_splash"] = "tokamax_splash"
    num_query_heads: int = 8
    num_kv_heads: int = 2
    head_dim: int = 128
    block_q: int = 1024
    block_q_dkv: int = 2048
    fused_qkv: bool = True
    rope: bool = False

    @model_validator(mode="after")
    def validate_gqa(self) -> AttentionConfig:
        if self.num_query_heads % self.num_kv_heads:
            raise ValueError("num_query_heads must be divisible by num_kv_heads")
        return self


class LossConfig(StrictModel):
    implementation: Literal["standard", "tokamax_fused"] = "standard"


class ModelConfig(StrictModel):
    name: str
    vocab_size: int = 32768
    emb_dim: int = 1024
    num_layers: int = 16
    cycle: tuple[Literal["kda", "gqa"], ...] = ("kda", "kda", "kda", "gqa")
    num_cycles: int = 4
    mlp_dim: int = 2816
    fused_mlp: bool = True
    rms_norm_epsilon: float = 1.0e-5
    dtype: Literal["bfloat16", "float32"] = "bfloat16"
    weight_dtype: Literal["float32"] = "float32"
    param_scan_axis: int = 1
    remat_policy: Literal["minimal", "minimal_with_context", "save_dot_except_mlp", "full"] = (
        "minimal_with_context"
    )
    residual_policy: Literal["standard", "block_attnres"] = "standard"
    logits_via_embedding: bool = False
    dropout_rate: float = 0.0
    kda: KDAConfig = Field(default_factory=KDAConfig)
    attention: AttentionConfig = Field(default_factory=AttentionConfig)
    loss: LossConfig = Field(default_factory=LossConfig)

    @model_validator(mode="after")
    def validate_layout(self) -> ModelConfig:
        if self.num_layers != self.num_cycles * len(self.cycle):
            raise ValueError("num_layers must equal num_cycles * len(cycle)")
        if tuple(self.cycle) != ("kda", "kda", "kda", "gqa"):
            raise ValueError("the certified baseline requires [KDA,KDA,KDA,NoPE-GQA] cycles")
        if self.residual_policy == "block_attnres":
            raise ValueError(
                "block_attnres is reserved but disabled until its separate quality experiment"
            )
        return self


class OptimizerConfig(StrictModel):
    name: Literal["adamw", "muon", "muonclip"]
    learning_rate: float = 3.0e-4
    beta1: float = 0.9
    beta2: float = 0.95
    epsilon: float = 1.0e-8
    weight_decay: float = 0.1
    gradient_clip_norm: float = 1.0
    warmup_steps: int = 3
    schedule_steps: int = 30
    final_learning_rate_fraction: float = 0.1
    muon_beta: float = 0.95
    muon_epsilon: float = 1.0e-8
    muon_ns_steps: int = 5
    qk_clip_tau: float = 100.0
    qk_clip_epsilon: float = 1.0e-6

    @model_validator(mode="after")
    def validate_schedule(self) -> OptimizerConfig:
        if self.warmup_steps < 0:
            raise ValueError("warmup_steps must be non-negative")
        if self.schedule_steps <= self.warmup_steps:
            raise ValueError("schedule_steps must be greater than warmup_steps")
        if not 0.0 <= self.final_learning_rate_fraction <= 1.0:
            raise ValueError("final_learning_rate_fraction must be in [0, 1]")
        return self


class DataConfig(StrictModel):
    name: str
    type: Literal["synthetic", "huggingface", "grain"]
    sequence_length: int = 2048
    per_device_batch_size: int = 8
    eval_interval: int = 0
    eval_steps: int = 0
    # Materialize the first evaluation pass's host batches once and reuse them
    # at every later evaluation. This makes the held-out loss a comparable
    # curve over one fixed set instead of a rolling sample, and for streaming
    # sources it stops re-scanning roughly 1/validation_fraction documents per
    # evaluation batch after the first pass.
    eval_fixed_batches: bool = True
    dataset_name: str | None = None
    dataset_path: str | None = None
    tokenizer: str | None = None
    split: str = "train"
    eval_split: str = "validation"
    shuffle_seed: int = 42
    reuse_example_batch: bool = True
    streaming: bool = False
    validation_fraction: float = 0.0
    validation_seed: int = 17
    shuffle_buffer_size: int = 10_000
    tokenize_batch_size: int = 256
    prefetch_batches: int = 0
    append_eos: bool = True
    text_field: str = "text"

    @model_validator(mode="after")
    def validate_streaming(self) -> DataConfig:
        if not 0.0 <= self.validation_fraction < 1.0:
            raise ValueError("validation_fraction must be in [0, 1)")
        if self.streaming and self.type != "huggingface":
            raise ValueError("streaming is currently supported only for Hugging Face data")
        if self.streaming and not self.dataset_name:
            raise ValueError("streaming Hugging Face data requires dataset_name")
        if self.validation_fraction and not self.streaming:
            raise ValueError("validation_fraction is reserved for streaming datasets")
        if self.prefetch_batches < 0:
            raise ValueError("prefetch_batches must be non-negative")
        if self.shuffle_buffer_size < 1 or self.tokenize_batch_size < 1:
            raise ValueError("shuffle and tokenize batch sizes must be positive")
        return self


class MeshConfig(StrictModel):
    data: int
    fsdp: int = 1
    tensor: int = 1
    sequence: int = 1

    @property
    def size(self) -> int:
        return self.data * self.fsdp * self.tensor * self.sequence


class HardwareProfile(StrictModel):
    name: str
    accelerator: Literal["v6e-8", "v6e-64", "v5litepod-16", "v5litepod-64", "v4-32"]
    device_count: int
    chips: int
    hosts: int
    mesh: MeshConfig
    libtpu_init_args: tuple[str, ...] = ()
    multi_host: bool
    performance_verified: bool = False
    notes: str = ""

    @model_validator(mode="after")
    def validate_mesh(self) -> HardwareProfile:
        if self.mesh.size != self.device_count:
            raise ValueError(
                f"mesh contains {self.mesh.size} devices but profile requires {self.device_count}"
            )
        if self.multi_host != (self.hosts > 1):
            raise ValueError("multi_host must agree with hosts")
        return self


class CheckpointConfig(StrictModel):
    enabled: bool = False
    destination: str | None = None
    save_interval: int = 0
    async_save: bool = False
    keep: int = 2
    resume: bool = True

    @model_validator(mode="after")
    def validate_destination(self) -> CheckpointConfig:
        if self.enabled and not self.destination:
            raise ValueError("checkpoint destination is required when checkpointing is enabled")
        return self


class WandbConfig(StrictModel):
    enabled: bool = False
    project: str = "yxtpu-pretrain"
    entity: str | None = None
    group: str | None = None
    tags: tuple[str, ...] = ()
    mode: Literal["online", "offline", "disabled"] = "online"


class DiagnosticsConfig(StrictModel):
    enabled: bool = False
    interval: int = 0

    @model_validator(mode="after")
    def validate_interval(self) -> DiagnosticsConfig:
        if self.enabled and self.interval <= 0:
            raise ValueError("enabled diagnostics require a positive interval")
        return self


class HarnessEvalConfig(StrictModel):
    enabled: bool = False
    interval: int = 0
    tasks: tuple[str, ...] = ()
    batch_size_per_device: int = 1
    num_fewshot: int = 0
    limit: int | float | None = None
    use_cache: bool = True

    @model_validator(mode="after")
    def validate_harness(self) -> HarnessEvalConfig:
        if self.enabled and (self.interval <= 0 or not self.tasks):
            raise ValueError("enabled lm-eval requires a positive interval and at least one task")
        if self.batch_size_per_device < 1:
            raise ValueError("lm-eval batch_size_per_device must be positive")
        return self


class ExperimentConfig(StrictModel):
    name: str
    steps: int = 30
    gradient_accumulation_steps: int = 1
    run_dir: str = "runs"
    seed: int = 42
    log_interval: int = 1
    profile_steps: tuple[int, ...] = ()
    benchmark: bool = True
    token_budget: int | None = None
    acknowledge_no_checkpoint: bool = False
    checkpoint: CheckpointConfig = Field(default_factory=CheckpointConfig)
    wandb: WandbConfig = Field(default_factory=WandbConfig)
    diagnostics: DiagnosticsConfig = Field(default_factory=DiagnosticsConfig)
    harness_eval: HarnessEvalConfig = Field(default_factory=HarnessEvalConfig)
    model_overrides: dict[str, Any] = Field(default_factory=dict)
    data_overrides: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_checkpoint_policy(self) -> ExperimentConfig:
        if self.benchmark and self.checkpoint.enabled:
            raise ValueError("benchmark profiles must keep checkpointing disabled")
        if (
            not self.benchmark
            and not self.checkpoint.enabled
            and not self.acknowledge_no_checkpoint
        ):
            raise ValueError(
                "real-training profiles require a checkpoint destination or an explicit "
                "acknowledge_no_checkpoint=true"
            )
        if self.checkpoint.enabled and self.acknowledge_no_checkpoint:
            raise ValueError("checkpointing and acknowledge_no_checkpoint are mutually exclusive")
        if self.token_budget is not None and self.token_budget <= 0:
            raise ValueError("token_budget must be positive")
        return self


class ResolvedConfig(StrictModel):
    model: ModelConfig
    optimizer: OptimizerConfig
    data: DataConfig
    hardware: HardwareProfile
    experiment: ExperimentConfig

    @model_validator(mode="after")
    def apply_profile_overrides(self) -> ResolvedConfig:
        # Overrides are applied before validation by load_config. They remain in the
        # resolved document as provenance, rather than being silently discarded.
        if self.model.loss.implementation == "tokamax_fused":
            mesh = self.hardware.mesh
            if mesh.fsdp != 1 or mesh.tensor != 1 or mesh.sequence != 1:
                raise ValueError(
                    "tokamax_fused currently requires pure data parallelism "
                    "(fsdp=tensor=sequence=1); vocabulary parallelism needs explicit "
                    "global softmax collectives"
                )
        if self.data.streaming and self.experiment.checkpoint.enabled:
            raise ValueError(
                "the streaming packed iterator is not checkpointable; disable prefetch/streaming "
                "only after implementing an exact stream-state restore"
            )
        if self.data.streaming and self.data.eval_interval and not self.data.validation_fraction:
            raise ValueError("streaming validation requires a nonzero validation_fraction")
        diagnostics = self.experiment.diagnostics
        if diagnostics.enabled:
            if not self.data.eval_interval:
                raise ValueError("diagnostics require validation batches")
            if diagnostics.interval % self.data.eval_interval:
                raise ValueError("diagnostics interval must be a multiple of data.eval_interval")
        if self.experiment.token_budget is not None:
            tokens_available = (
                self.experiment.steps
                * self.data.sequence_length
                * self.data.per_device_batch_size
                * self.experiment.gradient_accumulation_steps
                * self.hardware.device_count
            )
            if tokens_available < self.experiment.token_budget:
                raise ValueError("configured steps cannot reach the requested token_budget")
        return self

    def as_dict(self) -> dict[str, Any]:
        return self.model_dump(mode="json")

    def to_yaml(self) -> str:
        return yaml.safe_dump(self.as_dict(), sort_keys=False)


def _read_yaml(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"configuration profile not found: {path}")
    with path.open(encoding="utf-8") as handle:
        value = yaml.safe_load(handle) or {}
    if not isinstance(value, dict):
        raise TypeError(f"configuration profile must be a mapping: {path}")
    return value


def _deep_merge(base: dict[str, Any], update: dict[str, Any]) -> dict[str, Any]:
    result = copy.deepcopy(base)
    for key, value in update.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


def _parse_override(raw: str) -> tuple[list[str], Any]:
    if "=" not in raw:
        raise ValueError(f"override must use dotted.path=value syntax: {raw!r}")
    path, raw_value = raw.split("=", 1)
    keys = [part for part in path.split(".") if part]
    if not keys:
        raise ValueError(f"override path is empty: {raw!r}")
    return keys, yaml.safe_load(raw_value)


def _set_nested(config: dict[str, Any], keys: list[str], value: Any) -> None:
    cursor = config
    for key in keys[:-1]:
        child = cursor.get(key)
        if child is None:
            child = {}
            cursor[key] = child
        if not isinstance(child, dict):
            raise ValueError(f"cannot set nested key below non-mapping {key!r}")
        cursor = child
    cursor[keys[-1]] = value


def profile_path(kind: str, name: str) -> Path:
    filename = name if name.endswith((".yml", ".yaml")) else f"{name}.yml"
    return CONFIG_ROOT / kind / filename


def load_config(
    *,
    model: str,
    optimizer: str,
    data: str,
    hardware: str,
    experiment: str,
    overrides: list[str] | tuple[str, ...] = (),
) -> ResolvedConfig:
    """Loads, composes, overrides, and validates a pretraining configuration."""
    raw: dict[str, Any] = {
        "model": _read_yaml(profile_path("models", model)),
        "optimizer": _read_yaml(profile_path("optimizers", optimizer)),
        "data": _read_yaml(profile_path("data", data)),
        "hardware": _read_yaml(profile_path("hardware", hardware)),
        "experiment": _read_yaml(profile_path("experiments", experiment)),
    }

    model_overrides = raw["experiment"].get("model_overrides", {})
    data_overrides = raw["experiment"].get("data_overrides", {})
    raw["model"] = _deep_merge(raw["model"], model_overrides)
    raw["data"] = _deep_merge(raw["data"], data_overrides)

    for override in overrides:
        keys, value = _parse_override(override)
        # `train.steps` was used in the original command proposal. Preserve that
        # friendly alias while keeping the typed section named `experiment`.
        if keys[0] == "train":
            keys[0] = "experiment"
        _set_nested(raw, keys, value)
    return ResolvedConfig.model_validate(raw)


def dump_json(config: ResolvedConfig) -> str:
    return json.dumps(config.as_dict(), indent=2, sort_keys=True)
