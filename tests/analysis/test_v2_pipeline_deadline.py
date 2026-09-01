"""Milestone 2B regression tests: deadline enforcement, safe stage startup,
and session-boundary resume/skip behavior in the stage-major v2 runner.

These sit one level above tests/analysis/test_v2_shards.py: that file
protects the shard primitives (ShardStore, run_sharded, Deadline) in
isolation; this file protects how src/analysis/v2_pipeline.py's main_run
loop actually *uses* them to decide whether a whole stage should be
started or skipped. No GPU/model code is exercised - stage_start_blocked
and stage_is_complete are pure decision functions over ctx/filesystem
state.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.analysis.v2_pipeline import (
    ArtifactPaths,
    RunContext,
    activation_paths,
    stage_is_complete,
    stage_start_blocked,
)
from src.analysis.v2_shards import Deadline
from src.v2_io import binding, write_json_lf


BENCH_SHA = "a" * 64
SPLIT_SHA = "b" * 64


def make_ctx(tmp_path, deadline=None, stage_seconds=None, n_rows=3):
    rows = [
        {"record_id": f"r{i:03d}", "prompt": " ".join(["w"] * (i + 1))}
        for i in range(n_rows)
    ]
    ctx = RunContext(
        benchmark_path=tmp_path / "bench.jsonl",
        benchmark_sha=BENCH_SHA,
        split_path=tmp_path / "split.json",
        split_sha=SPLIT_SHA,
        rows=rows,
        paths=ArtifactPaths(root=tmp_path / "results"),
        deadline=deadline if deadline is not None else Deadline(),
    )
    if stage_seconds:
        ctx.stage_seconds.extend(stage_seconds)
    return ctx


# ---- stage_start_blocked: deadline stop ----------------------------------


def test_unlimited_deadline_never_blocks(tmp_path):
    ctx = make_ctx(tmp_path, deadline=Deadline(minutes=None))
    blocked, reason = stage_start_blocked(ctx, "M3")
    assert blocked is False
    assert reason == ""


def test_already_expired_deadline_blocks_before_the_named_stage(tmp_path):
    ctx = make_ctx(tmp_path, deadline=Deadline(minutes=-1))
    blocked, reason = stage_start_blocked(ctx, "M3_direct")
    assert blocked is True
    assert "M3_direct" in reason
    assert "budget spent" in reason


def test_first_stage_is_always_attempted_with_no_prior_measurement(
    tmp_path,
):
    # No ctx.stage_seconds yet -> estimate is 0.0, so a live-but-nearly-out
    # budget still lets the first stage start rather than stalling forever.
    ctx = make_ctx(tmp_path, deadline=Deadline(minutes=0.001))
    blocked, reason = stage_start_blocked(ctx, "M0")
    assert blocked is False
    assert reason == ""


# ---- stage_start_blocked: calibration/estimate must not be decorative ---


def test_slow_prior_stage_blocks_a_stage_that_cannot_fit(tmp_path):
    # Budget has ~2 minutes left; the previous stage's extraction alone
    # took 10 minutes. Starting the next stage's (equally atomic, not
    # resumable) extraction would very likely be killed mid-way, so the
    # runner must refuse to start it rather than only checking .expired().
    deadline = Deadline(minutes=2.0)
    ctx = make_ctx(tmp_path, deadline=deadline, stage_seconds=[10 * 60.0])

    assert deadline.expired() is False  # sanity: not the trivial case
    blocked, reason = stage_start_blocked(ctx, "M2")
    assert blocked is True
    assert "M2" in reason
    assert "10.0 min" in reason


def test_fast_prior_stage_does_not_block_a_stage_that_comfortably_fits(
    tmp_path,
):
    deadline = Deadline(minutes=60.0)
    ctx = make_ctx(tmp_path, deadline=deadline, stage_seconds=[30.0])

    blocked, reason = stage_start_blocked(ctx, "M2")
    assert blocked is False
    assert reason == ""


def test_estimate_uses_the_slowest_stage_seen_not_the_most_recent(
    tmp_path,
):
    # A later, faster stage must not erase the risk flagged by an earlier
    # slow one - the estimate is a safety ceiling, not a rolling average.
    deadline = Deadline(minutes=5.0)
    ctx = make_ctx(
        tmp_path, deadline=deadline, stage_seconds=[20 * 60.0, 5.0]
    )

    blocked, _ = stage_start_blocked(ctx, "M3")
    assert blocked is True


# ---- stage_is_complete: resume / skip (zero recomputation) --------------


def _write_bound_activations(ctx, stage):
    final_path, pooled_path, metadata_path, binding_path = (
        activation_paths(ctx, stage)
    )
    final_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(final_path, np.zeros((len(ctx.rows), 4)))
    np.save(pooled_path, np.zeros((len(ctx.rows), 4)))
    write_json_lf(metadata_path, ctx.snapshot)
    write_json_lf(
        binding_path,
        {**ctx.bind(), "stage": stage},
    )


def test_stage_incomplete_when_activations_missing(tmp_path):
    ctx = make_ctx(tmp_path)
    item = {
        "stage": "M3",
        "extract": True,
        "behavior": False,
        "causal": False,
        "steering": False,
        "norm_diag": False,
    }
    assert stage_is_complete(ctx, item) is False


def test_stage_complete_once_activations_are_bound_and_match(tmp_path):
    ctx = make_ctx(tmp_path)
    _write_bound_activations(ctx, "M3")

    item = {
        "stage": "M3",
        "extract": True,
        "behavior": False,
        "causal": False,
        "steering": False,
        "norm_diag": False,
    }
    assert stage_is_complete(ctx, item) is True


def test_stage_incomplete_when_bound_to_a_different_benchmark(tmp_path):
    ctx = make_ctx(tmp_path)
    _write_bound_activations(ctx, "M3")

    other = make_ctx(tmp_path)
    other.benchmark_sha = "c" * 64  # simulate a resumed run on new data
    item = {
        "stage": "M3",
        "extract": True,
        "behavior": False,
        "causal": False,
        "steering": False,
        "norm_diag": False,
    }
    assert stage_is_complete(other, item) is False


def test_stage_incomplete_while_behavioral_output_is_still_missing(
    tmp_path,
):
    ctx = make_ctx(tmp_path)
    _write_bound_activations(ctx, "M3")

    item = {
        "stage": "M3",
        "extract": True,
        "behavior": True,
        "causal": False,
        "steering": False,
        "norm_diag": False,
    }
    # Extraction is bound but the merged behavioral file was never written
    # (e.g. the session ended mid-shard) - the stage must not be treated
    # as complete, or a resumed session would silently drop those rows.
    assert stage_is_complete(ctx, item) is False

    write_json_lf(
        ctx.paths.behavioral / "v2_raw_M3.json", [{"record_id": "r000"}]
    )
    # The output file alone is still not enough - it also needs a binding
    # sidecar bound to this run's benchmark/split (see the binding tests
    # below); only once both are present is the stage actually complete.
    assert stage_is_complete(ctx, item) is False

    write_json_lf(
        ctx.paths.behavioral / "v2_raw_M3_binding.json",
        {**ctx.bind(), "stage": "M3"},
    )
    assert stage_is_complete(ctx, item) is True


def test_stage_incomplete_when_behavioral_output_binding_is_stale(
    tmp_path,
):
    # A merged behavioral file surviving from a different (stale)
    # benchmark/split keeps the same stage-based filename. If only
    # .exists() were checked, this would be mistaken for fresh output and
    # the whole stage (including stage_behavior's own regeneration path)
    # would be skipped outright.
    ctx = make_ctx(tmp_path)
    _write_bound_activations(ctx, "M3")

    write_json_lf(
        ctx.paths.behavioral / "v2_raw_M3.json", [{"record_id": "r000"}]
    )
    write_json_lf(
        ctx.paths.behavioral / "v2_raw_M3_binding.json",
        {
            "benchmark_sha256": "stale" * 16,
            "split_manifest_sha256": "stale" * 16,
        },
    )

    item = {
        "stage": "M3",
        "extract": True,
        "behavior": True,
        "causal": False,
        "steering": False,
        "norm_diag": False,
    }
    assert stage_is_complete(ctx, item) is False


def test_stage_incomplete_when_causal_output_binding_is_missing(tmp_path):
    ctx = make_ctx(tmp_path)
    _write_bound_activations(ctx, "M3")

    ctx.paths.raw.mkdir(parents=True, exist_ok=True)
    write_json_lf(
        ctx.paths.raw / "causal_ablation_v2_M3_L24-28.json",
        [{"record_id": "r000"}],
    )
    # No binding sidecar at all - must not count as complete.
    item = {
        "stage": "M3",
        "extract": True,
        "behavior": False,
        "causal": True,
        "steering": False,
        "norm_diag": False,
    }
    assert stage_is_complete(ctx, item) is False

    write_json_lf(
        ctx.paths.raw / "causal_ablation_v2_M3_L24-28_binding.json",
        {**ctx.bind(), "stage": "M3"},
    )
    assert stage_is_complete(ctx, item) is True


def test_stage_incomplete_when_norm_diag_output_binding_is_stale(tmp_path):
    ctx = make_ctx(tmp_path)
    _write_bound_activations(ctx, "M3")

    ctx.paths.raw.mkdir(parents=True, exist_ok=True)
    write_json_lf(
        ctx.paths.raw / "residual_norm_v2_M3.json", {"stage": "M3"}
    )
    write_json_lf(
        ctx.paths.raw / "residual_norm_v2_M3_binding.json",
        {
            "benchmark_sha256": "stale" * 16,
            "split_manifest_sha256": "stale" * 16,
        },
    )

    item = {
        "stage": "M3",
        "extract": True,
        "behavior": False,
        "causal": False,
        "steering": False,
        "norm_diag": True,
    }
    assert stage_is_complete(ctx, item) is False


def test_stage_incomplete_while_steering_output_is_still_missing(
    tmp_path,
):
    ctx = make_ctx(tmp_path)
    _write_bound_activations(ctx, "M3")

    item = {
        "stage": "M3",
        "extract": True,
        "behavior": False,
        "causal": False,
        "steering": True,
        "norm_diag": False,
    }
    assert stage_is_complete(ctx, item) is False

    ctx.paths.raw.mkdir(parents=True, exist_ok=True)
    write_json_lf(
        ctx.paths.raw / "steering_v2_M3_L24_tag.json", [{"record_id": "r000"}]
    )
    # As with behavioral output, the result file alone is not enough - it
    # needs a matching binding sidecar too.
    assert stage_is_complete(ctx, item) is False

    write_json_lf(
        ctx.paths.raw / "steering_v2_M3_L24_tag_binding.json",
        {**ctx.bind(), "stage": "M3"},
    )
    assert stage_is_complete(ctx, item) is True


def test_stage_incomplete_when_steering_output_binding_is_stale(tmp_path):
    ctx = make_ctx(tmp_path)
    _write_bound_activations(ctx, "M3")

    ctx.paths.raw.mkdir(parents=True, exist_ok=True)
    write_json_lf(
        ctx.paths.raw / "steering_v2_M3_L24_tag.json", [{"record_id": "r000"}]
    )
    write_json_lf(
        ctx.paths.raw / "steering_v2_M3_L24_tag_binding.json",
        {
            "benchmark_sha256": "stale" * 16,
            "split_manifest_sha256": "stale" * 16,
        },
    )

    item = {
        "stage": "M3",
        "extract": True,
        "behavior": False,
        "causal": False,
        "steering": True,
        "norm_diag": False,
    }
    assert stage_is_complete(ctx, item) is False


def test_steering_binding_sidecar_alone_does_not_count_as_output(
    tmp_path,
):
    # steering_v2_*_binding.json is metadata about a steering file, not a
    # result file; a resumed run must not mistake it for completed output.
    ctx = make_ctx(tmp_path)
    _write_bound_activations(ctx, "M3")

    item = {
        "stage": "M3",
        "extract": True,
        "behavior": False,
        "causal": False,
        "steering": True,
        "norm_diag": False,
    }
    ctx.paths.raw.mkdir(parents=True, exist_ok=True)
    write_json_lf(
        ctx.paths.raw / "steering_v2_M3_L24_binding.json", {"stage": "M3"}
    )
    assert stage_is_complete(ctx, item) is False
