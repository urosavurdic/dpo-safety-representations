"""Strict benchmark-bound GPU pipeline for the v2 rerun.

The legacy analysis scripts remain available for historical reproducibility.
This module is the only runner used by rerun_mechanistic_v2.sh and the v2
notebook. It always reads the frozen benchmark referenced by
LATEST_BENCHMARK.json and refuses stale or unbound artifacts.

Design notes for the free-tier T4 target (hard ~5:30 wall clock per session,
several sessions across accounts):

* Execution is **stage-major**. Each stage's model is loaded and its LoRA
  chain merged exactly once, then every component that needs that model runs
  while it is resident. The earlier command-major layout re-loaded and
  re-merged each stage 3-4 times, which on a T4 (3 GB base model, minutes to
  merge) was the single largest avoidable cost in the run.
* Generation is **sharded and resumable** (src/analysis/v2_shards.py). A
  behavioral or intervention unit can outlast a session, so checkpoints are
  per-shard, not per-stage.
* Extraction is per-stage atomic rather than sharded: it is forward-only and
  a stage comfortably fits one session. The deadline is checked before
  starting a stage using the previous stage's measured duration, so a stage
  is not begun that cannot finish.
* Batching is length-sorted, which collapses padding waste on a
  bandwidth-bound GPU. Output row order is restored by record_id before
  anything is written, so batch composition never leaks into results.
"""

from __future__ import annotations

import argparse
import gc
import os
import subprocess
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from src.analysis.v2_shards import (
    Deadline,
    ShardStore,
    plan_shards,
    probe_batch_capacity,
    run_sharded,
    run_with_oom_backoff,
)
from src.v2_io import (
    assert_binding,
    binding,
    identity_snapshot,
    load_json,
    load_jsonl,
    load_run_inputs,
    write_json_lf,
)


ALL_STAGES = [
    "M0",
    "M1",
    "M2",
    "M3",
    "M3_direct",
    "M1_alt",
    "M2_alt",
    "M3_alt",
    "M3_direct_alt",
]

# Causal ablation (the necessity half) runs on the four DPO endpoints.
INTERVENTION_STAGES = [
    "M3",
    "M3_direct",
    "M3_alt",
    "M3_direct_alt",
]

# Steering (the sufficiency half) runs on every trained stage, so the run can
# answer *where in the chain* steering starts working rather than only
# whether it works at the endpoints. M0 has no chat template and no trained
# refusal behaviour to steer.
STEERING_STAGES = [stage for stage in ALL_STAGES if stage != "M0"]

# Mirrors src/analysis/eval_refusal_direction.py so the v2 cosine diagnostics
# carry the same sections the Finding 3 scripts already consume.
SEQUENTIAL_STAGES = ["M0", "M1", "M2", "M3"]
ALT_SEQUENTIAL_STAGES = ["M0", "M1_alt", "M2_alt", "M3_alt"]
CROSS_BRANCH_PAIRS = [
    ("M1", "M1_alt"),
    ("M2", "M2_alt"),
    ("M3", "M3_alt"),
    ("M3_direct", "M3_direct_alt"),
]

ABLATION_LAYERS = list(range(24, 29))
COLLAPSING_STEER_LAYERS = list(range(14, 29))
DEFAULT_STEER_LAYERS = [24]

DEFAULT_ACT_BATCH = 8
DEFAULT_GEN_BATCH = 8
MAX_NEW_TOKENS = 200
MODEL_NAME = "Qwen/Qwen2.5-1.5B"

# 5:30 session minus a margin for setup, install, and the final merge/commit.
DEFAULT_DEADLINE_MINUTES = 300

# Bounds for the optional capacity probe in `calibrate --probe-capacity`.
DEFAULT_CAPACITY_START = 1
DEFAULT_CAPACITY_CAP = 64

CALIBRATION_PATH = Path("logs/t4_calibration.json")

POOL_WINDOW = 5


# --------------------------------------------------------------------------
# paths
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class ArtifactPaths:
    """Output roots, optionally namespaced for a companion eval set.

    The main benchmark writes to results/ under the same filenames the
    existing statistics layer already reads. A companion set (the paired
    source prompts, the C-source-authored arm) writes under
    results/companions/<namespace>/ so its activations cannot collide with
    the main run's.
    """

    root: Path
    namespace: str | None = None

    @classmethod
    def for_namespace(cls, namespace: str | None) -> "ArtifactPaths":
        if namespace:
            return cls(
                Path("results") / "companions" / namespace,
                namespace,
            )
        return cls(Path("results"), None)

    @property
    def activations(self) -> Path:
        return self.root / "activations"

    @property
    def refusal_direction(self) -> Path:
        return self.root / "refusal_direction"

    @property
    def raw(self) -> Path:
        return self.root / "raw"

    @property
    def probes(self) -> Path:
        return self.root / "probes_v2"

    @property
    def behavioral(self) -> Path:
        return self.root / "behavioral_eval"

    @property
    def manifests(self) -> Path:
        return self.root / "manifests"

    @property
    def behavior_shards(self) -> Path:
        return self.behavioral / "v2_shards"

    @property
    def causal_shards(self) -> Path:
        return self.raw / "v2_causal_shards"

    @property
    def steering_shards(self) -> Path:
        return self.raw / "v2_steering_shards"


# --------------------------------------------------------------------------
# run context
# --------------------------------------------------------------------------


@dataclass
class RunContext:
    benchmark_path: Path
    benchmark_sha: str
    split_path: Path
    split_sha: str
    rows: list[dict]
    paths: ArtifactPaths
    deadline: Deadline
    act_batch: int = DEFAULT_ACT_BATCH
    gen_batch: int = DEFAULT_GEN_BATCH
    max_new_tokens: int = MAX_NEW_TOKENS
    force: bool = False
    stage_seconds: list[float] = field(default_factory=list)

    @property
    def order(self) -> dict[str, int]:
        return {
            row["record_id"]: index
            for index, row in enumerate(self.rows)
        }

    @property
    def snapshot(self) -> list[dict]:
        return identity_snapshot(self.rows)

    def bind(self) -> dict[str, str]:
        return binding(
            self.benchmark_path,
            self.benchmark_sha,
            self.split_path,
            self.split_sha,
        )

    def store(self, directory: Path) -> ShardStore:
        return ShardStore(
            directory,
            self.benchmark_sha,
            self.split_sha,
        )


def batch_size_arg(value: str) -> int | str:
    """argparse type for --act-batch/--gen-batch: an int, or the string "auto"."""
    if value == "auto":
        return "auto"
    return int(value)


def resolve_batch_size(
    value: int | str,
    calibration_key: str,
    calibration_path: Path = CALIBRATION_PATH,
) -> int:
    """Resolve a --act-batch/--gen-batch value, honouring "auto".

    "auto" looks up `calibration_key` ("recommended_act_batch" or
    "recommended_gen_batch") in logs/t4_calibration.json, written by
    `calibrate --probe-capacity` - i.e. the batch size is *selected from
    measured capacity* rather than guessed. Any other value passes
    through unchanged; this function never invents a batch size itself.
    """
    if value != "auto":
        return int(value)

    if not calibration_path.exists():
        raise RuntimeError(
            "--act-batch/--gen-batch auto requires "
            f"{calibration_path}; run `calibrate --probe-capacity` first."
        )

    calibration = load_json(calibration_path)
    recommended = calibration.get(calibration_key)
    if not recommended:
        raise RuntimeError(
            f"{calibration_path} has no {calibration_key} (probe may not "
            "have run); rerun `calibrate --probe-capacity`."
        )
    return int(recommended)


def build_context(args) -> RunContext:
    benchmark_path, benchmark_sha, split_path, split_sha = load_run_inputs(
        getattr(args, "eval_set", None),
        getattr(args, "benchmark_sha256", None),
        getattr(args, "split_manifest", "logs/direction_split_manifest.json"),
        latest_path=getattr(args, "latest_pointer", None)
        or "data/frozen_v2/LATEST_BENCHMARK.json",
    )

    rows = load_jsonl(benchmark_path)
    missing_ids = [
        index
        for index, row in enumerate(rows)
        if not row.get("record_id")
    ]
    if missing_ids:
        raise RuntimeError(
            "Every benchmark row must carry record_id; rows without one: "
            f"{missing_ids[:10]}"
        )

    limit = getattr(args, "limit", None)
    if limit:
        rows = rows[:limit]

    deadline_minutes = getattr(args, "deadline_minutes", None)

    return RunContext(
        benchmark_path=benchmark_path,
        benchmark_sha=benchmark_sha,
        split_path=split_path,
        split_sha=split_sha,
        rows=rows,
        paths=ArtifactPaths.for_namespace(
            getattr(args, "namespace", None)
        ),
        deadline=Deadline(deadline_minutes),
        act_batch=resolve_batch_size(
            getattr(args, "act_batch", DEFAULT_ACT_BATCH),
            "recommended_act_batch",
        ),
        gen_batch=resolve_batch_size(
            getattr(args, "gen_batch", DEFAULT_GEN_BATCH),
            "recommended_gen_batch",
        ),
        max_new_tokens=getattr(args, "max_new_tokens", MAX_NEW_TOKENS),
        force=bool(getattr(args, "force", False)),
    )


# --------------------------------------------------------------------------
# small helpers
# --------------------------------------------------------------------------


def ml_imports():
    import torch
    from transformers import AutoTokenizer
    from src.training.model import load_stage_model

    return torch, AutoTokenizer, load_stage_model


def snapshot_for(rows):
    return identity_snapshot(rows)


def save_array(path: Path, array: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("wb") as handle:
        np.save(handle, array)
    os.replace(temporary, path)


def token_measure(tokenizer):
    """Token-length function for length-sorted batching.

    Falls back to a word count if tokenization fails for any reason; the
    measure only affects batch composition, never results, so a degraded
    measure is preferable to aborting the run.
    """

    def measure(text: str) -> int:
        try:
            return len(tokenizer(text)["input_ids"])
        except Exception:
            return len(text.split())

    return measure


def load_stage(stage: str):
    torch, AutoTokenizer, load_stage_model = ml_imports()

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"  loading {stage} (base + LoRA merge)")
    started = time.monotonic()
    model = load_stage_model(stage)
    device = next(model.parameters()).device
    print(
        f"  {stage} resident on {device} "
        f"({time.monotonic() - started:.0f}s)"
    )

    return model, tokenizer, device


def free_model(model) -> None:
    import torch

    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def decoder_layers(model):
    try:
        return model.model.layers
    except AttributeError as exc:
        raise AttributeError(
            "Expected model.model.layers for Qwen-style checkpoint."
        ) from exc


# --------------------------------------------------------------------------
# forward / generate
# --------------------------------------------------------------------------


def generation_batch(model, tokenizer, prompts, device, max_new_tokens):
    import torch
    from src.training.eval_generation import (
        build_generation_prompt,
        get_generation_eos_ids,
    )

    original_padding_side = tokenizer.padding_side
    tokenizer.padding_side = "left"

    try:
        texts = [
            build_generation_prompt(tokenizer, prompt)
            for prompt in prompts
        ]
        inputs = tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
        ).to(device)

        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                repetition_penalty=1.1,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=get_generation_eos_ids(tokenizer),
            )

        new_tokens = output_ids[:, inputs["input_ids"].shape[1]:]
        return tokenizer.batch_decode(
            new_tokens,
            skip_special_tokens=True,
        )
    finally:
        tokenizer.padding_side = original_padding_side


def activation_batch(model, tokenizer, prompts, device):
    import torch
    from src.training.eval_generation import build_generation_prompt

    original_padding_side = tokenizer.padding_side
    tokenizer.padding_side = "left"

    try:
        texts = [
            build_generation_prompt(tokenizer, prompt)
            for prompt in prompts
        ]
        inputs = tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
        ).to(device)

        with torch.no_grad():
            outputs = model(**inputs, output_hidden_states=True)

        hidden_states = outputs.hidden_states
        layer_count = len(hidden_states)
        hidden_dim = hidden_states[0].shape[-1]

        final = np.zeros(
            (len(prompts), layer_count, hidden_dim),
            dtype=np.float32,
        )
        pooled = np.zeros_like(final)

        for row_index in range(len(prompts)):
            attention_length = int(
                inputs["attention_mask"][row_index].sum().item()
            )
            window = min(POOL_WINDOW, attention_length)

            for layer in range(layer_count):
                hidden = hidden_states[layer][row_index]
                final[row_index, layer] = (
                    hidden[-1].float().cpu().numpy()
                )
                pooled[row_index, layer] = (
                    hidden[-window:].float().mean(dim=0).cpu().numpy()
                )

        return final, pooled
    finally:
        tokenizer.padding_side = original_padding_side


def result_row(row, ctx, stage, condition, model_stage, response):
    return {
        "record_id": row.get("record_id"),
        "prompt": row["prompt"],
        "quadrant": row["quadrant"],
        "source": row.get("source", row.get("source_dataset")),
        "source_dataset": row.get("source_dataset"),
        "c_construction": row.get("c_construction"),
        "pair_id": row.get("pair_id"),
        "split": row.get("split"),
        # "stage" carries the CONDITION name, which is what
        # summarize_causal_ablation.py / summarize_steering.py /
        # mcnemar_causal_ablation.py / build_finding4_report.py read.
        # "model_stage" carries the checkpoint.
        "stage": condition,
        "condition": condition,
        "model_stage": model_stage,
        "response": response,
        "benchmark_sha256": ctx.benchmark_sha,
        "split_manifest_sha256": ctx.split_sha,
        "generation": {
            "max_new_tokens": ctx.max_new_tokens,
            "do_sample": False,
            "repetition_penalty": 1.1,
        },
    }


def clear_cuda_cache() -> None:
    """Best-effort torch.cuda.empty_cache().

    Safe to call with no GPU present (torch.cuda.is_available() is False)
    or with torch missing entirely (e.g. a unit test exercising the OOM
    retry path without the real ML stack installed) - either way there is
    no CUDA cache to clear, so this is a silent no-op rather than an error.
    """
    try:
        import torch
    except ImportError:
        return

    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def make_generator(ctx, model, tokenizer, device, stage, condition):
    def process(shard):
        def call(sub_shard):
            responses = generation_batch(
                model,
                tokenizer,
                [row["prompt"] for row in sub_shard],
                device,
                ctx.max_new_tokens,
            )
            return [
                result_row(row, ctx, stage, condition, stage, response)
                for row, response in zip(sub_shard, responses)
            ]

        # A shard that OOMs at ctx.gen_batch is retried at half the size
        # (then a quarter, ...) rather than failing the whole session; the
        # shard's rows, order and record_id identity are unaffected.
        return run_with_oom_backoff(
            shard,
            call,
            combine=lambda left, right: left + right,
            on_retry=clear_cuda_cache,
        )

    return process


# --------------------------------------------------------------------------
# row selection
# --------------------------------------------------------------------------


def intervention_rows(rows):
    """Rows an intervention may be evaluated on.

    Quadrant A/D are restricted to the held-out half: the direction is
    estimated from the direction_estimation half, so testing an
    intervention on those same rows would measure the fit, not the effect.
    B and C never enter the direction, so they are used in full.
    """
    return [
        row
        for row in rows
        if (
            row.get("quadrant") not in {"A", "D"}
            or row.get("split") == "held_out_behavioral"
        )
    ]


def quadrant_rows(rows, quadrants):
    return [
        row
        for row in intervention_rows(rows)
        if row.get("quadrant") in quadrants
    ]


# --------------------------------------------------------------------------
# activation binding
# --------------------------------------------------------------------------


def activation_paths(ctx, stage):
    directory = ctx.paths.activations
    return (
        directory / f"{stage}_final.npy",
        directory / f"{stage}_pooled.npy",
        directory / f"{stage}_metadata.json",
        directory / f"{stage}_metadata_binding.json",
    )


def activations_bound(ctx, stage) -> bool:
    """True when this stage's activations already match the frozen run.

    Same three checks load_bound_activation performs, without paying to
    read the arrays: binding hashes, metadata identity, and row counts.
    """
    final_path, pooled_path, metadata_path, binding_path = (
        activation_paths(ctx, stage)
    )

    if not all(
        path.exists()
        for path in (final_path, pooled_path, metadata_path, binding_path)
    ):
        return False

    try:
        assert_binding(binding_path, ctx.benchmark_sha, ctx.split_sha)
        if load_json(metadata_path) != ctx.snapshot:
            return False
        for path in (final_path, pooled_path):
            if np.load(path, mmap_mode="r").shape[0] != len(ctx.rows):
                return False
    except Exception:
        return False

    return True


def load_bound_activation(ctx, stage):
    final_path, pooled_path, metadata_path, binding_path = (
        activation_paths(ctx, stage)
    )

    assert_binding(binding_path, ctx.benchmark_sha, ctx.split_sha)

    if not pooled_path.exists() or not final_path.exists():
        raise FileNotFoundError(
            f"Missing activation arrays for {stage}."
        )

    metadata = load_json(metadata_path)
    if metadata != ctx.snapshot:
        raise RuntimeError(
            f"{metadata_path} does not match the frozen benchmark."
        )

    pooled = np.load(pooled_path)
    final = np.load(final_path)

    for name, array in (("pooled", pooled), ("final", final)):
        if array.shape[0] != len(metadata):
            raise RuntimeError(
                f"{stage}: {name} activation row count mismatch."
            )

    return final, pooled, metadata


# --------------------------------------------------------------------------
# per-stage components
# --------------------------------------------------------------------------


def stage_extract(ctx, stage, model, tokenizer, device) -> bool:
    final_path, pooled_path, metadata_path, binding_path = (
        activation_paths(ctx, stage)
    )

    if activations_bound(ctx, stage) and not ctx.force:
        print(f"  {stage} activations: already bound, skipping")
        return True

    shards = plan_shards(
        ctx.rows,
        ctx.act_batch,
        measure=token_measure(tokenizer),
    )

    final_rows: dict[str, np.ndarray] = {}
    pooled_rows: dict[str, np.ndarray] = {}

    started = time.monotonic()
    for index, shard in enumerate(shards):
        def call(sub_shard):
            return activation_batch(
                model,
                tokenizer,
                [row["prompt"] for row in sub_shard],
                device,
            )

        # Mirrors make_generator's backoff: a shard that OOMs at
        # ctx.act_batch is retried at half the size instead of failing the
        # stage outright. combine concatenates the two (final, pooled)
        # array pairs along the row axis.
        final, pooled = run_with_oom_backoff(
            shard,
            call,
            combine=lambda left, right: (
                np.concatenate([left[0], right[0]], axis=0),
                np.concatenate([left[1], right[1]], axis=0),
            ),
            on_retry=clear_cuda_cache,
        )
        for offset, row in enumerate(shard):
            final_rows[row["record_id"]] = final[offset]
            pooled_rows[row["record_id"]] = pooled[offset]

        if (index + 1) % 10 == 0 or index + 1 == len(shards):
            print(
                f"  {stage} activations: shard {index + 1}/{len(shards)}"
            )

    # Restore benchmark row order; length-sorted batching must not affect
    # the row index any downstream script relies on.
    final_array = np.stack(
        [final_rows[row["record_id"]] for row in ctx.rows]
    )
    pooled_array = np.stack(
        [pooled_rows[row["record_id"]] for row in ctx.rows]
    )

    save_array(final_path, final_array)
    save_array(pooled_path, pooled_array)
    write_json_lf(metadata_path, ctx.snapshot)
    write_json_lf(
        binding_path,
        {
            **ctx.bind(),
            "stage": stage,
            "activation_shape_final": list(final_array.shape),
            "activation_shape_pooled": list(pooled_array.shape),
            "positions": {
                "final": "last_nonpadding_prompt_token",
                "pooled": "mean_last_five_nonpadding_tokens",
            },
            "batching": "length_sorted_restored_to_benchmark_order",
            "batch_size": ctx.act_batch,
            "device": str(device),
        },
    )

    elapsed = time.monotonic() - started
    ctx.stage_seconds.append(elapsed)
    print(
        f"  {stage} activations: saved {final_array.shape} "
        f"({elapsed / 60.0:.1f} min)"
    )
    return True


def stage_direction(ctx, stage) -> np.ndarray:
    """Per-stage refusal-associated direction: mean(A_est) - mean(D_est).

    Depends only on this stage's own activations, so it runs inside the
    stage-major loop and the intervention components that need it do not
    have to wait for every other stage.
    """
    direction_path = (
        ctx.paths.refusal_direction / f"{stage}_v2_direction.npy"
    )
    binding_path = (
        ctx.paths.refusal_direction
        / f"{stage}_v2_direction_binding.json"
    )

    if direction_path.exists() and not ctx.force:
        try:
            assert_binding(
                binding_path,
                ctx.benchmark_sha,
                ctx.split_sha,
            )
            return np.load(direction_path)
        except Exception:
            pass

    _, pooled, metadata = load_bound_activation(ctx, stage)

    quadrants = np.asarray([row.get("quadrant") for row in metadata])
    splits = np.asarray([row.get("split") for row in metadata])

    positive = pooled[
        (quadrants == "A") & (splits == "direction_estimation")
    ]
    negative = pooled[
        (quadrants == "D") & (splits == "direction_estimation")
    ]

    if len(positive) == 0 or len(negative) == 0:
        raise RuntimeError(
            f"{stage}: A/D direction-estimation rows are required."
        )

    delta = positive.mean(axis=0) - negative.mean(axis=0)
    norms = np.linalg.norm(delta, axis=-1, keepdims=True)
    direction = delta / np.where(norms == 0, 1.0, norms)

    save_array(direction_path, direction)
    write_json_lf(
        binding_path,
        {
            **ctx.bind(),
            "stage": stage,
            "direction_shape": list(direction.shape),
            "construction": (
                "mean(A_direction_estimation) - "
                "mean(D_direction_estimation)"
            ),
            "n_positive": int(len(positive)),
            "n_negative": int(len(negative)),
        },
    )

    print(f"  {stage} direction: {direction.shape}")
    return direction


def stage_behavior(ctx, stage, model, tokenizer, device) -> bool:
    store = ctx.store(ctx.paths.behavior_shards)
    condition = f"{stage}_behavior"
    unit_key = ShardStore.unit_key(stage, condition)

    output_path = ctx.paths.behavioral / f"v2_raw_{stage}.json"
    binding_path = (
        ctx.paths.behavioral / f"v2_raw_{stage}_binding.json"
    )

    if output_path.exists() and not ctx.force:
        try:
            assert_binding(
                binding_path,
                ctx.benchmark_sha,
                ctx.split_sha,
            )
            print(f"  {stage} behavioral: already bound, skipping")
            return True
        except Exception:
            pass

    shards = plan_shards(
        ctx.rows,
        ctx.gen_batch,
        measure=token_measure(tokenizer),
    )

    finished = run_sharded(
        store,
        unit_key,
        shards,
        make_generator(ctx, model, tokenizer, device, stage, condition),
        ctx.deadline,
        label=f"{stage} behavioral",
    )

    if not finished:
        return False

    write_json_lf(
        output_path,
        store.merge_unit(unit_key, order=ctx.order),
    )
    write_json_lf(
        binding_path,
        {
            **ctx.bind(),
            "stage": stage,
            "condition": condition,
            "row_count": len(ctx.rows),
        },
    )
    print(f"  {stage} behavioral: wrote {output_path}")
    return True


def register_ablation(model, direction_array, layers):
    """Ablate the direction from the residual stream at `layers`.

    Layer indexing follows hidden_states: index l is the OUTPUT of decoder
    block l, so the hook goes on blocks[l - 1]. This matches the convention
    the direction array itself uses (row l = hidden_states[l]).
    """
    import torch

    def ablate(hidden, direction):
        direction = direction.to(
            device=hidden.device,
            dtype=hidden.dtype,
        )
        projection = torch.einsum("...h,h->...", hidden, direction)
        return hidden - projection.unsqueeze(-1) * direction

    def make_hook(direction):
        def hook(_module, _inputs, output):
            if isinstance(output, tuple):
                return (ablate(output[0], direction),) + output[1:]
            return ablate(output, direction)

        return hook

    blocks = decoder_layers(model)
    handles = []
    for layer in layers:
        if layer <= 0 or layer > len(blocks):
            raise IndexError(f"Invalid ablation layer {layer}.")
        handles.append(
            blocks[layer - 1].register_forward_hook(
                make_hook(torch.from_numpy(direction_array[layer]))
            )
        )
    return handles


def register_steering(model, direction_array, alphas):
    import torch

    def make_hook(direction, alpha):
        def hook(_module, _inputs, output):
            hidden = output[0] if isinstance(output, tuple) else output
            shifted = hidden + alpha * direction.to(
                device=hidden.device,
                dtype=hidden.dtype,
            )
            if isinstance(output, tuple):
                return (shifted,) + output[1:]
            return shifted

        return hook

    blocks = decoder_layers(model)
    handles = []
    for layer, alpha in sorted(alphas.items()):
        if layer <= 0 or layer > len(blocks):
            raise IndexError(f"Invalid steering layer {layer}.")
        handles.append(
            blocks[layer - 1].register_forward_hook(
                make_hook(
                    torch.from_numpy(direction_array[layer]),
                    alpha,
                )
            )
        )
    return handles


def run_paired_conditions(
    ctx,
    stage,
    model,
    tokenizer,
    device,
    rows,
    store,
    baseline_name,
    treated_name,
    register,
    label,
) -> bool:
    """Baseline then treated, both sharded, sharing one resident model."""
    shards = plan_shards(
        rows,
        ctx.gen_batch,
        measure=token_measure(tokenizer),
    )

    baseline_key = ShardStore.unit_key(stage, baseline_name)
    if not run_sharded(
        store,
        baseline_key,
        shards,
        make_generator(
            ctx, model, tokenizer, device, stage, baseline_name
        ),
        ctx.deadline,
        label=f"{label} baseline",
    ):
        return False

    handles = register()
    try:
        treated_key = ShardStore.unit_key(stage, treated_name)
        if not run_sharded(
            store,
            treated_key,
            shards,
            make_generator(
                ctx, model, tokenizer, device, stage, treated_name
            ),
            ctx.deadline,
            label=f"{label} treated",
        ):
            return False
    finally:
        for handle in handles:
            handle.remove()

    return True


def stage_causal(ctx, stage, model, tokenizer, device, direction) -> bool:
    rows = intervention_rows(ctx.rows)
    if not rows:
        raise RuntimeError("No rows remain for causal ablation.")

    output_path = (
        ctx.paths.raw / f"causal_ablation_v2_{stage}_L24-28.json"
    )
    binding_path = (
        ctx.paths.raw
        / f"causal_ablation_v2_{stage}_L24-28_binding.json"
    )

    if output_path.exists() and not ctx.force:
        try:
            assert_binding(
                binding_path, ctx.benchmark_sha, ctx.split_sha
            )
            print(f"  {stage} causal: already bound, skipping")
            return True
        except Exception:
            pass

    if direction.ndim != 2:
        raise RuntimeError(
            "Direction must have shape (layers, hidden_dim)."
        )
    if max(ABLATION_LAYERS) >= direction.shape[0]:
        raise RuntimeError(
            "Direction array does not contain all ablation layers."
        )

    store = ctx.store(ctx.paths.causal_shards)
    baseline_name = f"{stage}_baseline"
    ablated_name = f"{stage}_ablated"

    if not run_paired_conditions(
        ctx,
        stage,
        model,
        tokenizer,
        device,
        rows,
        store,
        baseline_name,
        ablated_name,
        lambda: register_ablation(model, direction, ABLATION_LAYERS),
        f"{stage} causal",
    ):
        return False

    merged = store.merge_unit(
        ShardStore.unit_key(stage, baseline_name), order=ctx.order
    ) + store.merge_unit(
        ShardStore.unit_key(stage, ablated_name), order=ctx.order
    )

    write_json_lf(output_path, merged)
    write_json_lf(
        binding_path,
        {
            **ctx.bind(),
            "stage": stage,
            "conditions": [baseline_name, ablated_name],
            "layers": ABLATION_LAYERS,
            "layer_indexing": (
                "hidden_states index; hook on decoder block index-1"
            ),
            "row_count": len(merged),
            "rows_per_condition": len(rows),
        },
    )
    print(f"  {stage} causal: wrote {output_path}")
    return True


def calibration_alpha(ctx, stage, layer, direction) -> float:
    """Mean quadrant-A projection over the direction-estimation half only.

    Deliberately not the full quadrant-A population: steering is tested on
    the held-out half, so calibrating on all of A would let ~20% of the
    test rows set the intervention magnitude.
    """
    _, pooled, metadata = load_bound_activation(ctx, stage)

    indices = [
        index
        for index, row in enumerate(metadata)
        if row.get("quadrant") == "A"
        and row.get("split") == "direction_estimation"
    ]

    if not indices:
        raise RuntimeError(
            "No A direction-estimation rows are available for steering "
            "calibration."
        )

    if layer <= 0 or layer >= pooled.shape[1]:
        raise IndexError(f"Invalid steering layer {layer}.")

    return float(
        np.mean(pooled[indices, layer] @ direction[layer])
    )


def resolve_alphas(
    ctx,
    stage,
    direction,
    layers,
    alpha_source,
    alpha_value,
    alpha_coefficient,
) -> dict[int, float]:
    if alpha_source == "fixed":
        if alpha_value is None:
            raise ValueError(
                "--alpha-value is required with fixed alpha."
            )
        base = {layer: alpha_value for layer in layers}
    else:
        base = {
            layer: calibration_alpha(ctx, stage, layer, direction)
            for layer in layers
        }

    return {
        layer: value * alpha_coefficient
        for layer, value in base.items()
    }


def steering_tag(stage, layers, alpha_source, alpha_coefficient, quadrants):
    return (
        f"{stage}_L{'-'.join(str(layer) for layer in layers)}_"
        f"{alpha_source}_coef{alpha_coefficient:g}_Q{''.join(quadrants)}"
    )


def stage_steering(
    ctx,
    stage,
    model,
    tokenizer,
    device,
    direction,
    layers=None,
    alpha_source="direction_estimation_only",
    alpha_value=None,
    alpha_coefficient=1.0,
    quadrants=("A", "B", "C", "D"),
    tag=None,
) -> bool:
    layers = sorted(set(layers or DEFAULT_STEER_LAYERS))
    quadrants = list(quadrants)
    tag = tag or steering_tag(
        stage, layers, alpha_source, alpha_coefficient, quadrants
    )

    output_path = ctx.paths.raw / f"steering_v2_{tag}.json"
    binding_path = ctx.paths.raw / f"steering_v2_{tag}_binding.json"

    if output_path.exists() and not ctx.force:
        try:
            assert_binding(
                binding_path, ctx.benchmark_sha, ctx.split_sha
            )
            print(f"  {stage} steering: already bound, skipping")
            return True
        except Exception:
            pass

    rows = quadrant_rows(ctx.rows, quadrants)
    if not rows:
        raise RuntimeError("No rows remain for steering.")

    alphas = resolve_alphas(
        ctx,
        stage,
        direction,
        layers,
        alpha_source,
        alpha_value,
        alpha_coefficient,
    )

    store = ctx.store(ctx.paths.steering_shards)
    baseline_name = f"{tag}_baseline"
    steered_name = f"{tag}_steered"

    if not run_paired_conditions(
        ctx,
        stage,
        model,
        tokenizer,
        device,
        rows,
        store,
        baseline_name,
        steered_name,
        lambda: register_steering(model, direction, alphas),
        f"{stage} steering",
    ):
        return False

    merged = store.merge_unit(
        ShardStore.unit_key(stage, baseline_name), order=ctx.order
    ) + store.merge_unit(
        ShardStore.unit_key(stage, steered_name), order=ctx.order
    )

    write_json_lf(output_path, merged)
    write_json_lf(
        binding_path,
        {
            **ctx.bind(),
            "stage": stage,
            "layers": layers,
            "layer_indexing": (
                "hidden_states index; hook on decoder block index-1"
            ),
            "alpha_source": alpha_source,
            "alpha_coefficient": alpha_coefficient,
            "alphas_by_layer": {
                str(layer): value for layer, value in alphas.items()
            },
            "alpha_calibration_rows": (
                "quadrant_A_direction_estimation_only"
            ),
            "quadrants": quadrants,
            "conditions": [baseline_name, steered_name],
            "row_count": len(merged),
            "rows_per_condition": len(rows),
        },
    )
    print(f"  {stage} steering: wrote {output_path}")
    return True


# --------------------------------------------------------------------------
# residual-norm collapse diagnostic
# --------------------------------------------------------------------------


def stage_norm_diag(
    ctx,
    stage,
    model,
    tokenizer,
    device,
    direction,
    n_prompts=8,
    also_test_fix=True,
) -> bool:
    """Test whether magnitude growth drives the multi-layer collapse.

    Multi-layer steering (L14-28) historically drove 49/50 quadrant-D
    completions into degenerate repetition, while single-layer steering at
    L24 did not. Two explanations fit that observation equally well:
    residual-stream magnitude growth, or distribution collapse under greedy
    decoding. The norm-preserving condition separates them - it injects the
    same direction at the same layers but rescales back to the pre-steering
    norm, so if collapse disappears there, magnitude is the mechanism.
    """
    from src.interpretability.residual_norm_tracking import (
        ResidualNormTracker,
        compare_to_baseline,
        compute_baseline_range,
        first_step_exceeding_p99,
        make_norm_preserving_steering_hook,
    )
    import torch

    output_path = ctx.paths.raw / f"residual_norm_v2_{stage}.json"
    binding_path = (
        ctx.paths.raw / f"residual_norm_v2_{stage}_binding.json"
    )

    if output_path.exists() and not ctx.force:
        try:
            assert_binding(
                binding_path, ctx.benchmark_sha, ctx.split_sha
            )
            print(f"  {stage} norm-diag: already bound, skipping")
            return True
        except Exception:
            pass

    rows = quadrant_rows(ctx.rows, ["D"])[:n_prompts]
    if not rows:
        raise RuntimeError(
            "No held-out quadrant-D rows for the norm diagnostic."
        )

    collapsing_alphas = resolve_alphas(
        ctx,
        stage,
        direction,
        COLLAPSING_STEER_LAYERS,
        "direction_estimation_only",
        None,
        1.0,
    )
    single_alphas = resolve_alphas(
        ctx,
        stage,
        direction,
        DEFAULT_STEER_LAYERS,
        "direction_estimation_only",
        None,
        1.0,
    )

    def norm_preserving_handles():
        blocks = decoder_layers(model)
        handles = []
        for layer, alpha in sorted(collapsing_alphas.items()):
            handles.append(
                blocks[layer - 1].register_forward_hook(
                    make_norm_preserving_steering_hook(
                        torch.from_numpy(direction[layer]),
                        alpha,
                    )
                )
            )
        return handles

    conditions = [
        ("baseline", lambda: []),
        (
            "collapsing_L14-28",
            lambda: register_steering(model, direction, collapsing_alphas),
        ),
        (
            "noncollapsing_L24",
            lambda: register_steering(model, direction, single_alphas),
        ),
    ]
    if also_test_fix:
        conditions.append(
            ("norm_preserving_L14-28", norm_preserving_handles)
        )

    results = {}

    for name, register in conditions:
        print(f"  {stage} norm-diag: {name} ({len(rows)} prompts)")
        tracker = ResidualNormTracker()
        tracker.register(model, decoder_layers=decoder_layers(model))
        handles = register()
        try:
            responses = generation_batch(
                model,
                tokenizer,
                [row["prompt"] for row in rows],
                device,
                ctx.max_new_tokens,
            )
            norms = tracker.collect()
        finally:
            for handle in handles:
                handle.remove()
            tracker.remove()

        results[name] = {
            "condition": name,
            "record_ids": [row["record_id"] for row in rows],
            "responses": responses,
            "norms_by_decoder_index": norms,
        }

    baseline_range = compute_baseline_range(
        results["baseline"]["norms_by_decoder_index"]
    )

    for name, payload in results.items():
        if name == "baseline":
            continue
        comparison = compare_to_baseline(
            payload["norms_by_decoder_index"],
            baseline_range,
        )
        payload["comparison_to_baseline"] = comparison
        payload["first_step_exceeding_p99"] = {
            layer: first_step_exceeding_p99(entries)
            for layer, entries in comparison.items()
        }

    write_json_lf(
        output_path,
        {
            "stage": stage,
            "n_prompts": len(rows),
            "quadrant": "D",
            "split": "held_out_behavioral",
            "layer_indexing": (
                "keys are decoder block indices (0-based); direction row "
                "for block k is k+1"
            ),
            "alphas_collapsing": {
                str(k): v for k, v in collapsing_alphas.items()
            },
            "alphas_single": {
                str(k): v for k, v in single_alphas.items()
            },
            "baseline_range": baseline_range,
            "conditions": results,
        },
    )
    write_json_lf(
        binding_path,
        {
            **ctx.bind(),
            "stage": stage,
            "conditions": list(results),
            "n_prompts": len(rows),
        },
    )
    print(f"  {stage} norm-diag: wrote {output_path}")
    return True


# --------------------------------------------------------------------------
# CPU aggregation
# --------------------------------------------------------------------------


def cosine_per_layer(a: np.ndarray, b: np.ndarray) -> list[float]:
    return np.sum(a * b, axis=-1).tolist()


def aggregate_directions(ctx, stages) -> None:
    """Cosine diagnostics across stages, in the sections Finding 3 reads.

    Only pairs where both stages are available are computed, so a partially
    trained alt branch produces a partial report rather than an error.
    """
    directions = {}
    for stage in stages:
        path = (
            ctx.paths.refusal_direction / f"{stage}_v2_direction.npy"
        )
        if path.exists():
            directions[stage] = np.load(path)

    if not directions:
        raise RuntimeError("No directions were produced.")

    cosine: dict[str, dict] = {}

    for reference in ("M0", "M3"):
        if reference in directions:
            cosine[f"vs_{reference}"] = {
                stage: cosine_per_layer(
                    directions[reference], direction
                )
                for stage, direction in directions.items()
            }

    def chain(chain_stages):
        section = {}
        for first, second in zip(chain_stages[:-1], chain_stages[1:]):
            if first in directions and second in directions:
                section[f"{first}_vs_{second}"] = cosine_per_layer(
                    directions[first], directions[second]
                )
        return section

    cosine["adjacent"] = chain(SEQUENTIAL_STAGES)
    cosine["adjacent_alt"] = chain(ALT_SEQUENTIAL_STAGES)

    direct_branch = {}
    for label, (first, second) in {
        "M1_vs_M3_direct": ("M1", "M3_direct"),
        "M3_direct_vs_M3": ("M3_direct", "M3"),
        "M1_alt_vs_M3_direct_alt": ("M1_alt", "M3_direct_alt"),
        "M3_direct_alt_vs_M3_alt": ("M3_direct_alt", "M3_alt"),
    }.items():
        if first in directions and second in directions:
            direct_branch[label] = cosine_per_layer(
                directions[first], directions[second]
            )
    cosine["direct_branch"] = direct_branch

    cross_branch = {}
    for original, alternate in CROSS_BRANCH_PAIRS:
        if original in directions and alternate in directions:
            cross_branch[f"{original}_vs_{alternate}"] = (
                cosine_per_layer(
                    directions[original], directions[alternate]
                )
            )
    cosine["cross_branch"] = cross_branch

    projections = {}
    for stage, direction in directions.items():
        _, pooled, metadata = load_bound_activation(ctx, stage)
        quadrants = [row.get("quadrant") for row in metadata]

        by_quadrant = {}
        for quadrant in sorted(set(quadrants)):
            indices = [
                index
                for index, value in enumerate(quadrants)
                if value == quadrant
            ]
            by_quadrant[quadrant] = (
                np.einsum("nlh,lh->nl", pooled[indices], direction)
                .mean(axis=0)
                .tolist()
            )
        projections[stage] = by_quadrant

    write_json_lf(
        ctx.paths.refusal_direction / "cosine_similarity_v2.json",
        cosine,
    )
    write_json_lf(
        ctx.paths.refusal_direction / "quadrant_projections_v2.json",
        projections,
    )
    write_json_lf(
        ctx.paths.refusal_direction / "v2_diagnostics_binding.json",
        {
            **ctx.bind(),
            "stages": sorted(directions),
            "sections": sorted(cosine),
            "projection_population": (
                "full quadrant; not restricted to a split"
            ),
        },
    )
    print(
        "  direction diagnostics: "
        f"{len(directions)} stages, sections {sorted(cosine)}"
    )


def compute_probes(ctx, stages) -> None:
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedKFold, cross_val_score

    for stage in stages:
        if not activations_bound(ctx, stage):
            print(f"  probes: {stage} has no bound activations, skipping")
            continue

        output_path = ctx.paths.probes / f"{stage}_probe_results.json"
        binding_path = ctx.paths.probes / f"{stage}_probe_binding.json"

        if output_path.exists() and not ctx.force:
            print(f"  probes: {stage} already present, skipping")
            continue

        final, _, metadata = load_bound_activation(ctx, stage)

        by_quadrant = {
            quadrant: [
                index
                for index, row in enumerate(metadata)
                if row.get("quadrant") == quadrant
            ]
            for quadrant in ("A", "B", "C", "D")
        }

        if len(by_quadrant["B"]) < 50:
            raise RuntimeError(f"{stage}: fewer than 50 B rows.")

        rng = np.random.RandomState(42)
        b_indices = np.asarray(by_quadrant["B"])
        rng.shuffle(b_indices)

        train_a = np.asarray(by_quadrant["A"])
        train_b = b_indices[:50]
        holdout_b = b_indices[50:]
        test_c = np.asarray(by_quadrant["C"])
        test_d = np.asarray(by_quadrant["D"])

        results = []

        for layer in range(final.shape[1]):
            x_train = np.concatenate(
                [final[train_a, layer], final[train_b, layer]],
                axis=0,
            )
            y_train = np.concatenate(
                [np.ones(len(train_a)), np.zeros(len(train_b))]
            )

            classifier = LogisticRegression(
                max_iter=2000,
                random_state=42,
            )
            folds = StratifiedKFold(
                n_splits=5,
                shuffle=True,
                random_state=42,
            )
            scores = cross_val_score(
                classifier, x_train, y_train, cv=folds
            )
            classifier.fit(x_train, y_train)

            def flagged(indices):
                if len(indices) == 0:
                    return None
                return float(
                    classifier.predict(final[indices, layer]).mean()
                )

            results.append(
                {
                    "layer": layer,
                    "cv_accuracy_mean": float(scores.mean()),
                    "cv_accuracy_std": float(scores.std()),
                    "cv_fold_scores": [
                        float(value) for value in scores
                    ],
                    "holdout_b_flagged_unsafe_frac": flagged(holdout_b),
                    "quadrant_c_flagged_unsafe_frac": flagged(test_c),
                    "quadrant_d_flagged_unsafe_frac": flagged(test_d),
                }
            )

        write_json_lf(output_path, results)
        write_json_lf(
            binding_path,
            {
                **ctx.bind(),
                "stage": stage,
                "layer_selection": (
                    "none; all layers retained; C not used for selection"
                ),
                "train": "quadrant A versus 50 quadrant-B rows",
            },
        )
        print(f"  probes: wrote {output_path}")


def merge_behavioral(ctx, stages) -> None:
    """Combine the per-stage behavioral files into one v2_raw.json.

    Per-stage files are the checkpoint unit; this combined view is what a
    reader (and v2_compat) consumes.
    """
    combined = {}
    for stage in stages:
        path = ctx.paths.behavioral / f"v2_raw_{stage}.json"
        if path.exists():
            combined[stage] = load_json(path)

    if not combined:
        print("  behavioral merge: nothing to merge yet")
        return

    write_json_lf(ctx.paths.behavioral / "v2_raw.json", combined)
    write_json_lf(
        ctx.paths.behavioral / "v2_binding.json",
        {
            **ctx.bind(),
            "stages": sorted(combined),
            "row_count_per_stage": len(ctx.rows),
        },
    )
    print(
        f"  behavioral merge: {sorted(combined)} -> "
        f"{ctx.paths.behavioral / 'v2_raw.json'}"
    )


# --------------------------------------------------------------------------
# gate
# --------------------------------------------------------------------------

STATIC_GATE_FIELDS = [
    "schema_integrity_pass",
    "prompt_integrity_pass",
    "c_review_pass",
    "c_review_mapping_pass",
    "benchmark_hash_pass",
    "split_benchmark_hash_pass",
    "split_hash_pass",
]


def gate_for_run(args) -> None:
    status_path = Path("logs/benchmark_validation_status.json")
    gate_path = Path("logs/benchmark_gate_config.json")

    for path in (status_path, gate_path):
        if not path.exists():
            raise FileNotFoundError(path)

    status = load_json(status_path)
    gate = load_json(gate_path)

    failures = [
        f"{field}={status.get(field)!r}"
        for field in STATIC_GATE_FIELDS
        if status.get(field) is not True
    ]
    if failures:
        raise RuntimeError(
            "Static benchmark gate failed: " + ", ".join(failures)
        )

    if status.get("technical_benchmark_status") != "PASS":
        if not args.regenerate:
            raise RuntimeError(
                "technical_benchmark_status is not PASS. Use "
                "--regenerate to rebuild stale model artifacts."
            )
        if status.get("artifact_freshness_pass") is not False:
            raise RuntimeError(
                "Technical validation failed for a reason other than "
                "stale artifact freshness."
            )
        print(
            "Only artifact freshness is failing; explicit regeneration "
            "is permitted."
        )

    for field in gate.get("warning_only_gate_fields", []):
        print(f"warning-only {field}: {status.get(field)!r}")


# --------------------------------------------------------------------------
# stage-major run
# --------------------------------------------------------------------------


def stage_plan(args) -> list[dict]:
    analysis = args.analysis_stage or list(ALL_STAGES)
    causal = [] if args.no_causal else (
        args.stage or list(INTERVENTION_STAGES)
    )
    steering = [] if args.no_steering else (
        args.steering_stage or list(STEERING_STAGES)
    )

    ordered = [stage for stage in ALL_STAGES if stage in set(
        analysis + causal + steering
    )]

    return [
        {
            "stage": stage,
            "extract": stage in analysis,
            "behavior": stage in analysis and not args.no_behavior,
            "causal": stage in causal,
            "steering": stage in steering,
            "norm_diag": (
                args.with_norm_diag and stage in steering
            ),
        }
        for stage in ordered
    ]


def describe_plan(plan) -> None:
    print("\nStage-major plan (one model load per stage):")
    for item in plan:
        components = [
            name
            for name in (
                "extract",
                "behavior",
                "causal",
                "steering",
                "norm_diag",
            )
            if item[name]
        ]
        print(f"  {item['stage']:<15} {', '.join(components) or '-'}")


def main_run(args) -> None:
    gate_for_run(args)

    ctx = build_context(args)
    plan = stage_plan(args)

    print(f"Frozen benchmark: {ctx.benchmark_path}")
    print(f"Benchmark SHA-256: {ctx.benchmark_sha}")
    print(f"Split manifest: {ctx.split_path}")
    print(f"Split SHA-256: {ctx.split_sha}")
    print(f"Rows: {len(ctx.rows)}")
    print(f"Artifact root: {ctx.paths.root}")
    print(f"Session budget: {ctx.deadline.describe()}")

    describe_plan(plan)

    if args.dry_run:
        print("\nDry run complete. No model code was executed.")
        return

    if not args.regenerate:
        raise RuntimeError("A live run requires --regenerate.")

    incomplete: list[str] = []

    for item in plan:
        stage = item["stage"]

        needed = any(
            item[name]
            for name in ("extract", "behavior", "causal", "steering",
                         "norm_diag")
        )
        if not needed:
            continue

        # Skip loading the model entirely if everything for this stage is
        # already on disk - the common case on a resumed session.
        if not ctx.force and stage_is_complete(ctx, item):
            print(f"\n=== {stage}: already complete, skipping ===")
            continue

        blocked, reason = stage_start_blocked(ctx, stage)
        if blocked:
            print(f"\n{reason}")
            incomplete.append(stage)
            break

        print(f"\n=== {stage} ===")
        model, tokenizer, device = load_stage(stage)

        try:
            if item["extract"]:
                stage_extract(ctx, stage, model, tokenizer, device)

            direction = None
            if item["causal"] or item["steering"] or item["norm_diag"]:
                direction = stage_direction(ctx, stage)

            finished = True

            if item["behavior"]:
                finished = stage_behavior(
                    ctx, stage, model, tokenizer, device
                ) and finished

            if finished and item["causal"]:
                finished = stage_causal(
                    ctx, stage, model, tokenizer, device, direction
                ) and finished

            if finished and item["steering"]:
                finished = stage_steering(
                    ctx,
                    stage,
                    model,
                    tokenizer,
                    device,
                    direction,
                    layers=args.steer_layers,
                    quadrants=args.quadrants,
                ) and finished

            if finished and item["norm_diag"]:
                finished = stage_norm_diag(
                    ctx, stage, model, tokenizer, device, direction
                ) and finished

            if not finished:
                incomplete.append(stage)
        finally:
            free_model(model)

        if incomplete:
            print(
                f"\nStopped inside {stage}: {ctx.deadline.describe()}. "
                "Re-run the same command in a fresh session to continue."
            )
            break

    # CPU aggregation: cheap, safe to redo, and makes a partial run useful.
    analysis_stages = [
        item["stage"] for item in plan if item["extract"]
    ]
    print("\n=== CPU aggregation ===")
    try:
        aggregate_directions(ctx, analysis_stages)
    except RuntimeError as exc:
        print(f"  direction diagnostics skipped: {exc}")

    if args.with_probes:
        compute_probes(ctx, analysis_stages)

    merge_behavioral(ctx, analysis_stages)

    write_run_manifest(ctx, args, plan, incomplete)
    print_status(ctx)

    if incomplete:
        print(
            "\nRun is INCOMPLETE. Remaining work is listed above; the "
            "next session resumes from the first unfinished shard."
        )
    else:
        print("\nv2 GPU pipeline completed for the planned stages.")


def stage_start_blocked(ctx, stage: str) -> tuple[bool, str]:
    """Whether the session budget rules out starting `stage` next.

    Two independent checks:

    1. The budget is already spent (``deadline.expired()``): the ordinary
       end-of-session stop.
    2. The slowest stage extraction measured so far in *this* run, used as
       a calibration estimate for the next one, would not fit in the time
       left (``deadline.would_exceed``). Extraction is atomic rather than
       sharded (see module docstring), so starting one that cannot finish
       risks losing all of it to a session kill instead of stopping
       cleanly beforehand and picking it up fresh next session.

    Returns ``(blocked, reason)``; `reason` is a complete, ready-to-print
    log line and is empty when not blocked.
    """
    if ctx.deadline.expired():
        return True, (
            f"Session budget spent before {stage}; "
            f"{ctx.deadline.describe()}"
        )

    estimate = max(ctx.stage_seconds) if ctx.stage_seconds else 0.0
    if ctx.deadline.would_exceed(estimate):
        return True, (
            f"Session budget insufficient to safely start {stage} "
            f"(slowest stage extraction so far took "
            f"{estimate / 60.0:.1f} min); {ctx.deadline.describe()}"
        )

    return False, ""


def stage_is_complete(ctx, item) -> bool:
    stage = item["stage"]

    if item["extract"] and not activations_bound(ctx, stage):
        return False

    checks = []
    if item["behavior"]:
        checks.append(ctx.paths.behavioral / f"v2_raw_{stage}.json")
    if item["causal"]:
        checks.append(
            ctx.paths.raw / f"causal_ablation_v2_{stage}_L24-28.json"
        )
    if item["norm_diag"]:
        checks.append(ctx.paths.raw / f"residual_norm_v2_{stage}.json")

    if item["steering"]:
        # Steering filenames encode the tag, so a glob is the honest check.
        if not ctx.paths.raw.exists() or not [
            path
            for path in ctx.paths.raw.glob(f"steering_v2_{stage}_L*.json")
            if "_binding" not in path.name
        ]:
            return False

    return all(path.exists() for path in checks)


def write_run_manifest(ctx, args, plan, incomplete) -> None:
    ctx.paths.manifests.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc)
    path = (
        ctx.paths.manifests
        / f"v2_run_{timestamp.strftime('%Y%m%dT%H%M%SZ')}.json"
    )

    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True
        ).strip()
    except Exception:
        commit = "unknown"

    write_json_lf(
        path,
        {
            "component": "v2_pipeline",
            "created_at_utc": timestamp.isoformat(),
            "git_commit": commit,
            **ctx.bind(),
            "artifact_root": ctx.paths.root.as_posix(),
            "namespace": ctx.paths.namespace,
            "rows": len(ctx.rows),
            "act_batch": ctx.act_batch,
            "gen_batch": ctx.gen_batch,
            "max_new_tokens": ctx.max_new_tokens,
            "deadline_minutes": ctx.deadline.minutes,
            "elapsed_minutes": round(
                ctx.deadline.elapsed_seconds / 60.0, 2
            ),
            "plan": plan,
            "incomplete_stages": incomplete,
        },
    )
    print(f"\nRun manifest: {path}")


# --------------------------------------------------------------------------
# status / calibrate
# --------------------------------------------------------------------------


def print_status(ctx) -> None:
    print("\n=== Progress ===")

    bound = [
        stage for stage in ALL_STAGES if activations_bound(ctx, stage)
    ]
    print(
        f"activations bound: {len(bound)}/{len(ALL_STAGES)} "
        f"{bound if bound else ''}"
    )

    def outputs(directory: Path, pattern: str) -> list[Path]:
        if not directory.exists():
            return []
        return sorted(
            path
            for path in directory.glob(pattern)
            if "_binding" not in path.name
        )

    behavioral = [
        path.stem[len("v2_raw_"):]
        for path in outputs(ctx.paths.behavioral, "v2_raw_*.json")
    ]
    print(f"behavioral complete: {behavioral}")

    causal = outputs(ctx.paths.raw, "causal_ablation_v2_*.json")
    print(f"causal complete: {len(causal)} files")

    steering = outputs(ctx.paths.raw, "steering_v2_*.json")
    print(f"steering complete: {len(steering)} files")

    norm_diag = outputs(ctx.paths.raw, "residual_norm_v2_*.json")
    print(f"norm diagnostics complete: {len(norm_diag)} files")

    for label, directory in (
        ("behavior", ctx.paths.behavior_shards),
        ("causal", ctx.paths.causal_shards),
        ("steering", ctx.paths.steering_shards),
    ):
        if not (directory / "progress.json").exists():
            continue
        store = ctx.store(directory)
        rows = [row for row in store.summary() if not row["complete"]]
        if rows:
            print(f"\nin-flight {label} units:")
            for row in rows:
                print(
                    f"  {row['unit']}: {row['shards_done']}/"
                    f"{row['shards_total']} shards"
                )


def cmd_status(args) -> None:
    ctx = build_context(args)
    print(f"Frozen benchmark: {ctx.benchmark_path}")
    print(f"Benchmark SHA-256: {ctx.benchmark_sha}")
    print(f"Rows: {len(ctx.rows)}")
    print_status(ctx)


def _capacity_probe_prompts(rows, measure, size):
    """`size` prompts biased toward the longest, for worst-case OOM probing.

    Length-sorted batching clusters the longest prompts together, so the
    shard most likely to OOM at a given batch size is a full shard of the
    longest prompts in the benchmark - that is what capacity is measured
    against, not a random or median-length sample.
    """
    longest_first = sorted(
        rows, key=lambda row: measure(row["prompt"]), reverse=True
    )
    if not longest_first:
        raise RuntimeError("No rows to probe capacity on.")

    prompts = [row["prompt"] for row in longest_first]
    if len(prompts) >= size:
        return prompts[:size]

    # Benchmark has fewer rows than the probed batch size: cycle through
    # the longest prompts rather than refusing to probe that size at all.
    reps = -(-size // len(prompts))
    return (prompts * reps)[:size]


def cmd_calibrate(args) -> None:
    """Measure real throughput, then project the session count.

    Timings on a free-tier T4 vary enough between sessions that a
    hard-coded estimate is worse than useless. Everything scheduling-
    related reads the numbers this writes.

    With --probe-capacity, also measures the largest forward and
    generation batch size this GPU survives without OOM (worst case: a
    shard of the longest prompts in the benchmark) and records a
    recommended batch size that `--act-batch auto`/`--gen-batch auto`
    can read back on a later invocation.
    """
    ctx = build_context(args)
    stage = args.stage

    sample = ctx.rows[:args.n_prompts]
    if not sample:
        raise RuntimeError("No rows to calibrate on.")

    model, tokenizer, device = load_stage(stage)
    measure = token_measure(tokenizer)

    capacity_probe = {
        "ran": False,
        "forward_max_batch": None,
        "generation_max_batch": None,
        "start": args.capacity_start,
        "cap": args.capacity_cap,
    }
    recommended_act_batch = None
    recommended_gen_batch = None

    try:
        started = time.monotonic()
        activation_batch(
            model,
            tokenizer,
            [row["prompt"] for row in sample[:ctx.act_batch]],
            device,
        )
        forward_seconds = time.monotonic() - started

        started = time.monotonic()
        generation_batch(
            model,
            tokenizer,
            [row["prompt"] for row in sample[:ctx.gen_batch]],
            device,
            ctx.max_new_tokens,
        )
        generate_seconds = time.monotonic() - started

        if args.probe_capacity:
            def try_forward(size):
                activation_batch(
                    model,
                    tokenizer,
                    _capacity_probe_prompts(ctx.rows, measure, size),
                    device,
                )

            def try_generate(size):
                generation_batch(
                    model,
                    tokenizer,
                    _capacity_probe_prompts(ctx.rows, measure, size),
                    device,
                    ctx.max_new_tokens,
                )

            forward_capacity = probe_batch_capacity(
                try_forward,
                start=args.capacity_start,
                cap=args.capacity_cap,
                on_retry=clear_cuda_cache,
            )
            generate_capacity = probe_batch_capacity(
                try_generate,
                start=args.capacity_start,
                cap=args.capacity_cap,
                on_retry=clear_cuda_cache,
            )
            capacity_probe = {
                "ran": True,
                "forward_max_batch": forward_capacity,
                "generation_max_batch": generate_capacity,
                "start": args.capacity_start,
                "cap": args.capacity_cap,
            }
            recommended_act_batch = forward_capacity or None
            recommended_gen_batch = generate_capacity or None
    finally:
        free_model(model)

    n_rows = len(ctx.rows)
    act_shards = -(-n_rows // ctx.act_batch)
    gen_shards = -(-n_rows // ctx.gen_batch)
    intervention_shards = -(
        -len(intervention_rows(ctx.rows)) // ctx.gen_batch
    )

    n_analysis = len(ALL_STAGES)
    n_causal = len(INTERVENTION_STAGES)
    n_steering = len(STEERING_STAGES)

    extract_minutes = act_shards * forward_seconds * n_analysis / 60.0
    behavior_minutes = gen_shards * generate_seconds * n_analysis / 60.0
    causal_minutes = (
        intervention_shards * generate_seconds * 2 * n_causal / 60.0
    )
    steering_minutes = (
        intervention_shards * generate_seconds * 2 * n_steering / 60.0
    )
    total_minutes = (
        extract_minutes
        + behavior_minutes
        + causal_minutes
        + steering_minutes
    )

    budget = args.deadline_minutes or DEFAULT_DEADLINE_MINUTES
    payload = {
        "stage_measured": stage,
        "device": str(device),
        "act_batch": ctx.act_batch,
        "gen_batch": ctx.gen_batch,
        "max_new_tokens": ctx.max_new_tokens,
        "seconds_per_forward_shard": round(forward_seconds, 3),
        "seconds_per_generation_shard": round(generate_seconds, 3),
        "median_prompt_tokens": int(
            np.median([measure(row["prompt"]) for row in ctx.rows])
        ),
        "projection_minutes": {
            "extract_all_stages": round(extract_minutes, 1),
            "behavioral_all_stages": round(behavior_minutes, 1),
            "causal_endpoints": round(causal_minutes, 1),
            "steering_all_trained_stages": round(steering_minutes, 1),
            "total": round(total_minutes, 1),
        },
        "session_budget_minutes": budget,
        "projected_sessions": round(total_minutes / budget, 2),
        "capacity_probe": capacity_probe,
        "recommended_act_batch": recommended_act_batch,
        "recommended_gen_batch": recommended_gen_batch,
        "note": (
            "Excludes per-stage model load/merge and the norm diagnostic. "
            "Model load is paid once per stage per session under the "
            "stage-major runner."
        ),
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
    }

    write_json_lf(CALIBRATION_PATH, payload)

    print("\n=== Calibration ===")
    print(f"forward shard  : {forward_seconds:.2f}s")
    print(f"generate shard : {generate_seconds:.2f}s")
    for key, value in payload["projection_minutes"].items():
        print(f"{key:<32} {value:>8.1f} min")
    print(
        f"\nProjected sessions at {budget} min: "
        f"{payload['projected_sessions']}"
    )
    if capacity_probe["ran"]:
        print(
            f"\nMax forward batch (no OOM)   : "
            f"{capacity_probe['forward_max_batch']}"
        )
        print(
            f"Max generation batch (no OOM): "
            f"{capacity_probe['generation_max_batch']}"
        )
        print(
            "Use --act-batch auto / --gen-batch auto on a later run to "
            "pick these up."
        )
    print(f"Wrote {CALIBRATION_PATH}")


# --------------------------------------------------------------------------
# single-component entry points
# --------------------------------------------------------------------------


def for_each_stage(ctx, stages, work) -> None:
    for stage in stages:
        if ctx.deadline.expired():
            print(f"Session budget spent before {stage}.")
            return
        print(f"\n=== {stage} ===")
        model, tokenizer, device = load_stage(stage)
        try:
            work(ctx, stage, model, tokenizer, device)
        finally:
            free_model(model)


def cmd_extract(args) -> None:
    ctx = build_context(args)
    for_each_stage(ctx, args.stages, stage_extract)


def cmd_behavior(args) -> None:
    ctx = build_context(args)
    for_each_stage(ctx, args.stages, stage_behavior)
    merge_behavioral(ctx, args.stages)


def cmd_direction(args) -> None:
    ctx = build_context(args)
    for stage in args.stages:
        if activations_bound(ctx, stage):
            stage_direction(ctx, stage)
    aggregate_directions(ctx, args.stages)


def cmd_probes(args) -> None:
    ctx = build_context(args)
    compute_probes(ctx, args.stages)


def cmd_causal(args) -> None:
    ctx = build_context(args)
    direction = stage_direction(ctx, args.stage)
    for_each_stage(
        ctx,
        [args.stage],
        lambda c, s, m, t, d: stage_causal(c, s, m, t, d, direction),
    )


def cmd_steering(args) -> None:
    ctx = build_context(args)
    direction = stage_direction(ctx, args.stage)
    for_each_stage(
        ctx,
        [args.stage],
        lambda c, s, m, t, d: stage_steering(
            c,
            s,
            m,
            t,
            d,
            direction,
            layers=args.layers,
            alpha_source=args.alpha_source,
            alpha_value=args.alpha_value,
            alpha_coefficient=args.alpha_coefficient,
            quadrants=args.quadrants,
            tag=args.tag,
        ),
    )


def cmd_norm_diag(args) -> None:
    ctx = build_context(args)
    direction = stage_direction(ctx, args.stage)
    for_each_stage(
        ctx,
        [args.stage],
        lambda c, s, m, t, d: stage_norm_diag(
            c,
            s,
            m,
            t,
            d,
            direction,
            n_prompts=args.n_prompts,
            also_test_fix=not args.no_fix_condition,
        ),
    )


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------


def add_common(parser) -> None:
    parser.add_argument("--eval-set", default=None)
    parser.add_argument("--benchmark-sha256", default=None)
    parser.add_argument(
        "--split-manifest",
        default="logs/direction_split_manifest.json",
    )
    parser.add_argument(
        "--latest-pointer",
        default=None,
        help=(
            "Pointer file naming the frozen set to bind to. Defaults to "
            "data/frozen_v2/LATEST_BENCHMARK.json; a companion set (paired "
            "source prompts, C-source-authored arm) passes its own."
        ),
    )
    parser.add_argument(
        "--namespace",
        default=None,
        help=(
            "Write artifacts under results/companions/<namespace>/ instead "
            "of results/. Required for companion sets so their activations "
            "cannot overwrite the main run's."
        ),
    )
    parser.add_argument(
        "--deadline-minutes",
        type=float,
        default=None,
        help="Stop cleanly at a shard boundary after this many minutes.",
    )
    parser.add_argument(
        "--act-batch",
        type=batch_size_arg,
        default=DEFAULT_ACT_BATCH,
        help=(
            "Batch size for forward-only extraction, or 'auto' to use "
            "recommended_act_batch from logs/t4_calibration.json "
            "(see `calibrate --probe-capacity`)."
        ),
    )
    parser.add_argument(
        "--gen-batch",
        type=batch_size_arg,
        default=DEFAULT_GEN_BATCH,
        help=(
            "Batch size for generation, or 'auto' to use "
            "recommended_gen_batch from logs/t4_calibration.json "
            "(see `calibrate --probe-capacity`)."
        ),
    )
    parser.add_argument(
        "--max-new-tokens", type=int, default=MAX_NEW_TOKENS
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Recompute even where a bound artifact already exists.",
    )


def add_stages(parser) -> None:
    parser.add_argument(
        "--stages",
        nargs="+",
        choices=ALL_STAGES,
        default=ALL_STAGES,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)

    for name in ("extract", "behavior", "direction", "probes"):
        sub = subparsers.add_parser(name)
        add_common(sub)
        add_stages(sub)
        if name == "extract":
            sub.add_argument("--limit", type=int)

    causal = subparsers.add_parser("causal")
    add_common(causal)
    causal.add_argument("--stage", required=True, choices=ALL_STAGES)

    steering = subparsers.add_parser("steering")
    add_common(steering)
    steering.add_argument("--stage", required=True, choices=ALL_STAGES)
    steering.add_argument(
        "--layers", nargs="+", type=int, default=DEFAULT_STEER_LAYERS
    )
    steering.add_argument(
        "--alpha-source",
        choices=["direction_estimation_only", "fixed"],
        default="direction_estimation_only",
    )
    steering.add_argument("--alpha-value", type=float)
    steering.add_argument(
        "--alpha-coefficient", type=float, default=1.0
    )
    steering.add_argument(
        "--quadrants",
        nargs="+",
        choices=["A", "B", "C", "D"],
        default=["A", "B", "C", "D"],
    )
    steering.add_argument("--tag")

    norm_diag = subparsers.add_parser("norm-diag")
    add_common(norm_diag)
    norm_diag.add_argument(
        "--stage", required=True, choices=STEERING_STAGES
    )
    norm_diag.add_argument("--n-prompts", type=int, default=8)
    norm_diag.add_argument(
        "--no-fix-condition", action="store_true"
    )

    status = subparsers.add_parser("status")
    add_common(status)

    calibrate = subparsers.add_parser("calibrate")
    add_common(calibrate)
    calibrate.add_argument("--stage", default="M3", choices=ALL_STAGES)
    calibrate.add_argument("--n-prompts", type=int, default=32)
    calibrate.add_argument(
        "--probe-capacity",
        action="store_true",
        help=(
            "Also measure the largest forward/generation batch size this "
            "GPU survives without OOM, and record a recommended batch "
            "size that --act-batch auto / --gen-batch auto can read back. "
            "Off by default since it costs several extra forward/generate "
            "calls at increasing batch sizes."
        ),
    )
    calibrate.add_argument(
        "--capacity-start", type=int, default=DEFAULT_CAPACITY_START
    )
    calibrate.add_argument(
        "--capacity-cap", type=int, default=DEFAULT_CAPACITY_CAP
    )

    run = subparsers.add_parser("run")
    add_common(run)
    run.add_argument(
        "--analysis-stage", action="append", choices=ALL_STAGES
    )
    run.add_argument(
        "--stage",
        action="append",
        choices=INTERVENTION_STAGES,
        help="Causal-ablation stages (default: the four DPO endpoints).",
    )
    run.add_argument(
        "--steering-stage",
        action="append",
        choices=STEERING_STAGES,
        help="Steering stages (default: every trained stage).",
    )
    run.add_argument(
        "--steer-layers", nargs="+", type=int, default=DEFAULT_STEER_LAYERS
    )
    run.add_argument(
        "--quadrants",
        nargs="+",
        choices=["A", "B", "C", "D"],
        default=["A", "B", "C", "D"],
    )
    run.add_argument("--dry-run", action="store_true")
    run.add_argument(
        "--regenerate", "--force-regen", action="store_true"
    )
    run.add_argument("--with-probes", action="store_true")
    run.add_argument("--with-norm-diag", action="store_true")
    run.add_argument("--no-causal", action="store_true")
    run.add_argument("--no-steering", action="store_true")
    run.add_argument("--no-behavior", action="store_true")

    return parser


DISPATCH = {
    "extract": cmd_extract,
    "behavior": cmd_behavior,
    "direction": cmd_direction,
    "probes": cmd_probes,
    "causal": cmd_causal,
    "steering": cmd_steering,
    "norm-diag": cmd_norm_diag,
    "status": cmd_status,
    "calibrate": cmd_calibrate,
    "run": main_run,
}


def main() -> None:
    args = build_parser().parse_args()

    if args.command == "run" and args.deadline_minutes is None:
        args.deadline_minutes = DEFAULT_DEADLINE_MINUTES

    DISPATCH[args.command](args)


if __name__ == "__main__":
    main()
