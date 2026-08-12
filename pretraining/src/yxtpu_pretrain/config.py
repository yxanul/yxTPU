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
    # Merge in_proj_qkv, decay_down, beta_proj, and output_gate_down into one
    # [embed, 3HD + rank + heads + rank] GEMM that reads hidden_states once.
    # Initialization is distribution-identical (fan_in-only initializer, all
    # blocks share fan_in = embed); AdamW dynamics are element-wise identical.
    fused_in_proj: bool = False

    @model_validator(mode="after")
    def validate_production_shape(self) -> KDAConfig:
        if self.chunk_size != 64:
            raise ValueError("the production KDA kernel is specialized to chunk_size=64")
        if self.key_head_dim != 128 or self.value_head_dim != 128:
            raise ValueError("the production KDA kernel requires a 128x128 recurrent state")
        # The gate bound is a kernel compute parameter, not only a stability
        # knob (Kimi K3 §2.1.1): the fused kernel's 16-row pairwise tiling is
        # budgeted for |log decay| <= 5 per token, which keeps the one-sided
        # rescaling factor below e^80, inside the fp32/bf16 exponent range.
        # A lower bound would silently halve the safe tile size.
        if self.precision == "guarded_fp32" and not self.safe_gate:
            raise ValueError(
                "the fused KDA kernel requires safe_gate: its pairwise tiling "
                "factors decay into per-block rescalings whose exponent range "
                "is only bounded when the log decay is"
            )
        if self.safe_gate and not -5.0 <= self.gate_lower_bound < 0.0:
            raise ValueError(
                "gate_lower_bound must lie in [-5, 0): the fused kernel's "
                "pairwise tiling budget (kda_fused_pallas_v4) is derived from "
                "this bound and must be revisited together with it"
            )
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
    # Head-specific sigmoid output gate (G1 of arXiv:2505.06708; the design
    # K3 §2.1.2 adopts for its global-attention layers): the per-head SDPA
    # output is gated elementwise by sigmoid(W_g x) BEFORE the output
    # projection, x being the post-pre-norm hidden state that also feeds
    # QKV. Architecture change (new parameters, checkpoint-incompatible
    # across the flag); with the flag off the module is not constructed, so
    # today's model is reproduced bit-identically. Costs q_heads*head_dim*
    # emb_dim params per GQA layer (+4.2M at 337M, +18.9M at 1b_deep).
    output_gate: bool = False

    @model_validator(mode="after")
    def validate_gqa(self) -> AttentionConfig:
        if self.num_query_heads % self.num_kv_heads:
            raise ValueError("num_query_heads must be divisible by num_kv_heads")
        return self


class TextSourceConfig(StrictModel):
    """One text stream of the mixed vision+text pipeline.

    ``weight`` is the per-DRAW probability mass within the text side of the
    mix; with uniform ``row_tokens`` across sources it approximates the
    loss-token share, and the realized share is measured per source as
    ``data/<name>_loss_token_share``. ``format: repo`` renders
    Stack-v3-style rows (one repository per row with a ``files[]`` array)
    as ``# path\\ncontent`` blocks; ``plain`` reads ``field`` directly."""

    name: str
    dataset: str
    subset: str | None = None
    weight: float = 1.0
    field: str = "text"
    format: Literal["plain", "repo"] = "plain"
    # Per-source row length override; None uses vision.text_row_tokens.
    row_tokens: int | None = None

    @model_validator(mode="after")
    def validate_source(self) -> TextSourceConfig:
        if self.weight <= 0:
            raise ValueError("text source weight must be positive")
        if self.row_tokens is not None and self.row_tokens < 8:
            raise ValueError("row_tokens must be at least 8")
        return self


class VisionConfig(StrictModel):
    """Native vision pathway: a from-scratch encoder trained jointly under
    the next-token objective (no contrastive stage, no pretrained ViT).

    Images enter as runs of ``placeholder_token_id`` in the token stream;
    the tower's projected features replace those positions' embeddings and
    the loss never lands on them. ``encoder_layers: 0`` selects the
    encoder-free arm: patches -> pixel shuffle -> projector, leaving all
    visual mixing to the backbone."""

    enabled: bool = False
    encoder_layers: int = 12
    encoder_dim: int = 384
    encoder_heads: int = 6
    encoder_mlp_dim: int = 1536
    patch_size: int = 16
    image_size: int = 448
    # Space-to-depth factor applied before the projector (K3-style): an s x s
    # neighborhood of patch features folds into one visual token of s^2 * dim
    # channels, cutting visual tokens per image by s^2.
    pixel_shuffle: int = 2
    placeholder_token_id: int = 49150
    max_images_per_sequence: int = 1
    # Mixed vision+text streaming, consumed by the main training loop when
    # ``enabled`` and ``dataset_name`` are both set: the loop then streams
    # this vision corpus interleaved with ``data``'s text corpus through the
    # packed mixed pipeline instead of ``create_data_iterator``. The stream
    # position is not resumable, so checkpoints degrade to weights+optimizer
    # exactly like text streaming (``allow_weights_only_resume``).
    dataset_name: str | None = None
    # Probability a packed row is a plain-text document. This is a row-level
    # Bernoulli, NOT the supervision mix: a vision row carries mostly
    # loss-free placeholder tokens while a text row is almost fully
    # supervised. The realized mix is measured, not assumed - tune this
    # against data/vision_loss_token_share.
    p_text: float = 0.3
    text_row_tokens: int = 1024
    min_visual_dependency: int = 0
    # Weighted multi-source text mix. Empty means the single legacy source:
    # ``data``'s corpus at weight 1. Non-empty REPLACES it - list the
    # pretraining corpus explicitly alongside the extra sources.
    text_datasets: tuple[TextSourceConfig, ...] = ()
    # Keep at 1: fsspec's cached HTTP filesystem is not thread-safe, and one
    # producer per host sustains 1B-scale step times.
    producer_threads: int = 1
    # Stream framing ids for the packed contract (yx49k: pad doubles as eos).
    pad_token_id: int = 49119
    eos_token_id: int = 49119

    @property
    def patch_grid(self) -> int:
        return self.image_size // self.patch_size

    @property
    def token_grid(self) -> int:
        return self.patch_grid // self.pixel_shuffle

    @property
    def visual_tokens_per_image(self) -> int:
        return self.token_grid**2

    @model_validator(mode="after")
    def validate_geometry(self) -> VisionConfig:
        if self.image_size % (self.patch_size * self.pixel_shuffle):
            raise ValueError(
                "image_size must be divisible by patch_size * pixel_shuffle "
                "so the token grid is exact"
            )
        if self.encoder_layers < 0:
            raise ValueError("encoder_layers must be non-negative (0 = encoder-free)")
        if self.encoder_layers and self.encoder_dim % self.encoder_heads:
            raise ValueError("encoder_dim must be divisible by encoder_heads")
        if self.max_images_per_sequence < 1:
            raise ValueError("max_images_per_sequence must be positive")
        if not 0.0 <= self.p_text <= 1.0:
            raise ValueError("vision.p_text must be in [0, 1]")
        if self.producer_threads < 1:
            raise ValueError("vision.producer_threads must be positive")
        return self


class LossConfig(StrictModel):
    # "standard" materializes [B, T, V] logits (~4 GB compiled temporaries at
    # 337M/PDB-8, scaling with batch); "tokamax_fused" is hard-blocked on v4;
    # "chunked" is the pure-XLA Liger-style implementation — FLOP-neutral,
    # peak logits memory [B, block_tokens, V]. MEASURED 2026-07-28 on v4-64
    # at 337M: chunked is numerics-clean (bf16-rounding-class overlay) and
    # saves 2.14 GB at block 256, but the scan costs +42.5 ms/step at PDB 8
    # (+24.5 at block 1024 — the fp32 dW accumulator round-trips HBM every
    # block), and PDB 12, which only chunked fits, nets 1.096M tok/s against
    # 1.156M at PDB 8 standard. Keep "standard" on v4 at this scale; chunked
    # is for configs where the standard logits do not fit at the desired
    # batch (the 1B campaign shapes).
    implementation: Literal["standard", "tokamax_fused", "chunked"] = "standard"
    # Sequence-block size for the chunked implementation; must divide the
    # sequence length. 256 keeps the block logits at ~1 GB fp32 at PDB 8;
    # larger blocks trade memory for fewer dW-accumulator round trips.
    block_tokens: int = 256

    @model_validator(mode="after")
    def validate_block(self) -> LossConfig:
        if self.block_tokens <= 0:
            raise ValueError("loss.block_tokens must be positive")
        return self


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
    remat_policy: Literal[
        "minimal", "minimal_with_context", "save_dot_except_mlp",
        "save_dot_context", "full"
    ] = "minimal_with_context"
    # Unroll factor for the cycle scan. 1 is the classic while-loop; higher
    # values trade compile time and code size for static per-cycle parameter
    # slices and schedulable gradient writes (the while-loop's per-iteration
    # dynamic-slice/DUS bookkeeping measured ~380 ms/step of small copies at
    # 1B/4096). num_cycles need not be divisible.
    scan_unroll: int = 1
    # Save the KDA kernel's output and state history across the cycle remat so
    # the sequential fused forward never re-runs in the backward pass. Costs
    # ~2.0 GB resident HBM at 2048-seq/batch-8 (state history scales with
    # sequence length; disable for memory-tight long-sequence runs).
    remat_save_kda_residuals: bool = True
    residual_policy: Literal["standard", "block_attnres"] = "standard"
    logits_via_embedding: bool = False
    dropout_rate: float = 0.0
    kda: KDAConfig = Field(default_factory=KDAConfig)
    attention: AttentionConfig = Field(default_factory=AttentionConfig)
    loss: LossConfig = Field(default_factory=LossConfig)
    vision: VisionConfig = Field(default_factory=VisionConfig)

    @model_validator(mode="after")
    def validate_layout(self) -> ModelConfig:
        if self.scan_unroll < 1:
            raise ValueError("scan_unroll must be at least 1")
        if self.vision.enabled and not 0 <= self.vision.placeholder_token_id < self.vocab_size:
            raise ValueError("vision.placeholder_token_id must lie inside the vocabulary")
        if self.num_layers != self.num_cycles * len(self.cycle):
            raise ValueError("num_layers must equal num_cycles * len(cycle)")
        allowed_cycles = (
            ("kda", "kda", "kda", "gqa"),
            ("gqa",),
            ("gqa", "gqa", "gqa", "gqa"),
        )
        if tuple(self.cycle) not in allowed_cycles:
            raise ValueError(
                "supported cycles are the certified [KDA,KDA,KDA,NoPE-GQA] hybrid "
                "or a pure-GQA transformer baseline"
            )
        if self.residual_policy == "block_attnres" and not self.cycle:
            raise ValueError("block_attnres requires a hybrid cycle")
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
    # When set, the schedule holds the peak learning rate constant after
    # warmup and applies the cosine decay only over the final decay_steps
    # of schedule_steps (terminal anneal). None keeps the classic
    # warmup-then-cosine shape over the whole schedule.
    decay_steps: int | None = None
    muon_beta: float = 0.95
    muon_epsilon: float = 1.0e-8
    muon_ns_steps: int = 5
    # Moonshot's "Muon is Scalable" RMS matching: scale each orthogonalized
    # update by sqrt(max(fan_in, fan_out)) * this value so Muon shares the
    # AdamW learning rate. optax's default width-transfer scaling leaves the
    # update RMS ~0.02-0.03 for our shapes, an effective 6-10x LR undershoot.
    muon_consistent_rms: float = 0.2
    # Run Muon's momentum + Newton-Schulz in bf16 (modded-nanogpt lineage):
    # a masked post-clip gradient cast plus mu_dtype=bf16. Note mu_dtype is a
    # shared knob inside optax.contrib.muon, so the muonclip arm's
    # AdamW-routed params also store their first moment in bf16 (their
    # update math stays fp32). Checkpoint note: toggling this changes the
    # optimizer-state pytree, so checkpoints are not cross-resumable across
    # the flag.
    muon_ns_bf16: bool = False
    # Per-Head Muon (Kimi K3 §2.5): orthogonalize the attention QKV
    # projections one [embed, head_dim] block per head slot. QKV only — the
    # KDA out_proj matricization has its own flag below so the two trajectory
    # effects can be gated separately. Beware the consistent-rms side effect:
    # the shape rule ties update scale to the matricization, so per-head
    # blocks receive smaller updates than the joint matrix (KDA in_proj
    # 1.73x, GQA qkv 1.22x at the 337M shape) unless
    # muon_per_head_scale_compensation cancels it. Optimizer state shapes are
    # unchanged, so checkpoints stay layout-compatible across the flag (the
    # update trajectory is not comparable across it).
    muon_per_head: bool = False
    # Switch the KDA out_proj from its historical Muon matricization —
    # reduction over the heads axis alone, an [8, 131072] matricization the
    # 50B run was validated with — to the whole-matrix (heads*dim -> embed)
    # form matching the GQA out_proj declaration. Note this also shrinks the
    # consistent-rms update scale of those matrices 11.31x
    # (sqrt(131072)/sqrt(1024) -> sqrt(1024) at the 337M shape): the flag
    # tests the matricization fix *including* its scale consequence.
    muon_kda_out_proj_whole: bool = False
    # Cancel the consistent-rms shape-rule artifact on per-head-matricized
    # QKV updates: each is multiplied by sqrt(max fans_joint) /
    # sqrt(max fans_per_head) inside the Muon chain, before weight decay, so
    # per-head orthogonalization is compared at the joint matricization's
    # update scale. No global muon_consistent_rms retune can do this — the
    # shift is per-parameter (1.22x-1.73x here).
    muon_per_head_scale_compensation: bool = False
    # The falsifying control for per-head Muon: KEEP the joint matricization
    # and multiply each QKV update by 1/ratio — i.e. apply exactly the
    # per-head run's update scale without its update direction. If this
    # reproduces the per-head run's trajectory, that run's gain was a
    # per-parameter LR effect (the QKV projections were over-stepped), not
    # cross-head equalization; ship the scale, not the matricization.
    muon_per_head_scale_only: bool = False
    # Distribute Muon's Newton-Schulz across the data axis (the
    # state-replicated first stage of K3 §5.2.2's design): every NS problem
    # is computed on exactly one chip and all-gathered back, instead of
    # replicated on all 32. Optimizer state and updates stay replicated, so
    # the checkpoint layout is unchanged and per-matrix numerics match the
    # replicated path. MEASURED 2026-07-28 on v4-64 at 337M: numerics pass
    # (fp-noise loss overlay) but p10 step time REGRESSES 463.4 -> 497.4 ms —
    # the update all-gather (2.45x padding inflation across 9 collectives)
    # costs more than the ~16 ms of replicated NS it replaces (NS runs at
    # ~94% MXU efficiency, measured via the per-head delta). Kept for the 1B
    # config where NS grows as d^3; do not enable on v4 at this scale.
    muon_distributed_ns: bool = False
    qk_clip_tau: float = 100.0
    qk_clip_epsilon: float = 1.0e-6

    @model_validator(mode="after")
    def validate_muon_flags(self) -> OptimizerConfig:
        if self.muon_per_head_scale_compensation and not self.muon_per_head:
            raise ValueError(
                "muon_per_head_scale_compensation compensates the per-head "
                "matricization's shape-rule scale shift and requires "
                "muon_per_head"
            )
        if self.muon_per_head_scale_only and self.muon_per_head:
            raise ValueError(
                "muon_per_head_scale_only is the control FOR muon_per_head "
                "(joint direction at per-head scale); the flags are mutually "
                "exclusive"
            )
        if self.muon_distributed_ns and (
            self.muon_per_head_scale_compensation or self.muon_per_head_scale_only
        ):
            raise ValueError(
                "update-scale stages and muon_distributed_ns compose "
                "different Muon chains; the combination is not implemented"
            )
        return self

    @model_validator(mode="after")
    def validate_schedule(self) -> OptimizerConfig:
        if self.warmup_steps < 0:
            raise ValueError("warmup_steps must be non-negative")
        if self.schedule_steps <= self.warmup_steps:
            raise ValueError("schedule_steps must be greater than warmup_steps")
        if not 0.0 <= self.final_learning_rate_fraction <= 1.0:
            raise ValueError("final_learning_rate_fraction must be in [0, 1]")
        if self.decay_steps is not None:
            if self.decay_steps <= 0:
                raise ValueError("decay_steps must be positive when set")
            if self.warmup_steps + self.decay_steps > self.schedule_steps:
                raise ValueError(
                    "warmup_steps + decay_steps must not exceed schedule_steps"
                )
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
    accelerator: Literal[
        "v6e-8", "v6e-16", "v6e-32", "v6e-64", "v5litepod-16", "v5litepod-64", "v4-32",
        "v4-64"
    ]
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
    # Streaming data profiles cannot serialize their stream position; this
    # opt-in accepts checkpoints whose restore keeps weights and optimizer
    # state while the stream restarts (see runtime/checkpoints.py stub).
    allow_weights_only_resume: bool = False

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
    # Total host batches staged ahead of the device: 1 is the classic
    # single-batch staging inside the step loop; values above 1 add a
    # background thread keeping (prefetch_batches - 1) further host batches
    # queued, masking the episodic >1 s stream-shard/flush stalls. Batch
    # order is preserved, so losses are bitwise identical to depth 1. Depths
    # above 1 are rejected when persisted iterator state must stay exact
    # (checkpointing without allow_weights_only_resume), because the
    # background thread necessarily runs the iterator ahead of the last
    # trained step.
    prefetch_batches: int = 1
    # Warm-start: when set and no own checkpoint exists, restore WEIGHTS
    # from this other run's latest checkpoint (same model tree required)
    # and train from step 0 with a fresh optimizer and schedule - the
    # continuation-after-anneal pattern (re-warmup handles the LR jump;
    # Muon momentum rebuilds within tens of steps).
    init_from_run: str | None = None
    init_from_destination: str | None = None
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
        if self.prefetch_batches < 1:
            raise ValueError("prefetch_batches must be at least 1")
        if (
            self.prefetch_batches > 1
            and self.checkpoint.enabled
            and not self.checkpoint.allow_weights_only_resume
        ):
            raise ValueError(
                "prefetch_batches > 1 runs the data iterator ahead of the "
                "last trained step; it requires checkpointing disabled or "
                "allow_weights_only_resume"
            )
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
        if self.model.kda.fused_in_proj and self.optimizer.name in ("muon", "muonclip"):
            raise ValueError(
                "kda.fused_in_proj fuses four KDA_MATRIX parameters that Muon "
                "currently orthogonalizes separately; blocked Muon routing for "
                "the fused weight is not implemented yet, so use adamw or "
                "disable the fusion"
            )
        if (
            self.data.streaming
            and self.experiment.checkpoint.enabled
            and not self.experiment.checkpoint.allow_weights_only_resume
        ):
            raise ValueError(
                "the streaming packed iterator cannot serialize its position: "
                "checkpoints of streaming runs restore weights and optimizer "
                "state but restart the stream. Set "
                "checkpoint.allow_weights_only_resume=true to accept that, or "
                "use a resumable data profile"
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
