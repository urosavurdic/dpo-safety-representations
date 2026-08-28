"""Milestone 5B regression tests: the v2 direction family (adjacent,
adjacent_alt, direct_branch, cross_branch) and the behavioral merge must
have the actual semantic structure downstream analyses need - not just
exist as files.

Exercises the REAL production functions (stage_direction, stage_behavior,
aggregate_directions, merge_behavioral) against small, exactly-solvable
synthetic activations, so cosine values can be asserted to an exact
closed form (cos of an angle difference) rather than merely "not None".

Two production bugs this file specifically regression-tests (both fixed
in this milestone): aggregate_directions and merge_behavioral used to
discover only whichever stages the CALLING command's own `--stages`
happened to cover, rather than every bound stage actually on disk. Since
a full 9-stage v2 rerun is assembled across several T4 sessions (each
session's `direction`/`behavior` invocation naturally scoped to that
session's own stages), the old behavior would silently drop
cross_branch/adjacent_alt/direct_branch pairs (and, worse, erase
already-merged behavioral stages) the moment a later, narrower-scoped
session ran. `test_..._survives_across_session_scoped_calls` below
reproduces that exact multi-call pattern.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from src.analysis import v2_pipeline as vp
from src.analysis.v2_shards import Deadline
from src.v2_io import load_json

BENCH_SHA = "a" * 64
SPLIT_SHA = "b" * 64

# One fixed 2D unit-vector angle per stage. Layer 0 is always the zero
# vector (mirrors the real "layer 0 is a zero-vector template-token
# artifact" property summarize_cross_branch.py's direction_cross_branch_
# similarity relies on - it drops layer 0 before averaging). Layer 1
# carries the angle so every pairwise cosine has an exact closed form:
# cos(theta_a - theta_b).
STAGE_ANGLES_DEG = {
    "M0": 0.0,
    "M1": 20.0,
    "M2": 35.0,
    "M3": 45.0,
    "M3_direct": 30.0,
    "M1_alt": 25.0,
    "M2_alt": 50.0,
    "M3_alt": 55.0,
    "M3_direct_alt": 60.0,
}


def expected_cosine(stage_a: str, stage_b: str) -> float:
    theta_a = math.radians(STAGE_ANGLES_DEG[stage_a])
    theta_b = math.radians(STAGE_ANGLES_DEG[stage_b])
    return math.cos(theta_a - theta_b)


def pooled_for_angle(theta_deg: float) -> np.ndarray:
    """(4 rows, 2 layers, 2 dims): rows are [A, A, D, D]. Layer 0 is the
    zero vector for every row (so mean(A)-mean(D) at layer 0 is exactly
    the zero vector too, matching the real template-token artifact); layer
    1 is +/-(cos theta, sin theta) so diff_in_means_direction's
    mean(A)-mean(D), once unit-normalized, is EXACTLY (cos theta, sin
    theta) at layer 1 - no averaging noise, no approximation.
    """
    theta = math.radians(theta_deg)
    vec = np.array([math.cos(theta), math.sin(theta)])
    zero = np.zeros(2)
    layer0 = np.stack([zero, zero, zero, zero])
    layer1 = np.stack([vec, vec, -vec, -vec])
    return np.stack([layer0, layer1], axis=1)


def direction_rows() -> list[dict]:
    return [
        {"record_id": "a1", "prompt": "prompt a1", "quadrant": "A",
         "split": "direction_estimation", "source_dataset": "unit_test"},
        {"record_id": "a2", "prompt": "prompt a2", "quadrant": "A",
         "split": "direction_estimation", "source_dataset": "unit_test"},
        {"record_id": "d1", "prompt": "prompt d1", "quadrant": "D",
         "split": "direction_estimation", "source_dataset": "unit_test"},
        {"record_id": "d2", "prompt": "prompt d2", "quadrant": "D",
         "split": "direction_estimation", "source_dataset": "unit_test"},
    ]


def make_ctx(tmp_path, rows=None) -> "vp.RunContext":
    return vp.RunContext(
        benchmark_path=tmp_path / "bench.jsonl",
        benchmark_sha=BENCH_SHA,
        split_path=tmp_path / "split.json",
        split_sha=SPLIT_SHA,
        rows=rows if rows is not None else direction_rows(),
        paths=vp.ArtifactPaths(tmp_path / "results"),
        deadline=Deadline(None),
    )


def write_bound_activation(ctx, stage: str, pooled: np.ndarray) -> None:
    """Writes a real, bound activation (final/pooled/metadata/binding) for
    `stage` directly - the fast path, bypassing stage_extract's shard/
    batching machinery, which tests/analysis/test_v2_pipeline.py already
    covers on its own. `final` is unused by stage_direction/
    aggregate_directions, so it's just a copy of `pooled`.
    """
    final_path, pooled_path, metadata_path, binding_path = (
        vp.activation_paths(ctx, stage)
    )
    vp.save_array(final_path, pooled)
    vp.save_array(pooled_path, pooled)

    from src.v2_io import write_json_lf

    write_json_lf(metadata_path, ctx.snapshot)
    write_json_lf(
        binding_path,
        {
            **ctx.bind(),
            "stage": stage,
            "activation_shape_final": list(pooled.shape),
            "activation_shape_pooled": list(pooled.shape),
        },
    )


def build_stage_direction(ctx, stage: str) -> None:
    write_bound_activation(ctx, stage, pooled_for_angle(STAGE_ANGLES_DEG[stage]))
    vp.stage_direction(ctx, stage)


# ---- aggregate_directions: all four required sections ---------------------


def test_aggregate_directions_produces_all_four_required_sections(tmp_path):
    ctx = make_ctx(tmp_path)
    for stage in vp.ALL_STAGES:
        build_stage_direction(ctx, stage)

    vp.aggregate_directions(ctx)

    cosine = load_json(ctx.paths.refusal_direction / "cosine_similarity_v2.json")

    assert set(cosine["adjacent"]) == {"M0_vs_M1", "M1_vs_M2", "M2_vs_M3"}
    assert set(cosine["adjacent_alt"]) == {
        "M0_vs_M1_alt", "M1_alt_vs_M2_alt", "M2_alt_vs_M3_alt",
    }
    assert set(cosine["direct_branch"]) == {
        "M1_vs_M3_direct", "M3_direct_vs_M3",
        "M1_alt_vs_M3_direct_alt", "M3_direct_alt_vs_M3_alt",
    }
    assert set(cosine["cross_branch"]) == {
        "M1_vs_M1_alt", "M2_vs_M2_alt", "M3_vs_M3_alt",
        "M3_direct_vs_M3_direct_alt",
    }


def test_adjacent_pairs_map_to_the_correct_cosine_values(tmp_path):
    ctx = make_ctx(tmp_path)
    for stage in vp.ALL_STAGES:
        build_stage_direction(ctx, stage)
    vp.aggregate_directions(ctx)

    cosine = load_json(ctx.paths.refusal_direction / "cosine_similarity_v2.json")
    adjacent = cosine["adjacent"]

    assert adjacent["M0_vs_M1"][0] == pytest.approx(0.0)  # layer 0 artifact
    assert adjacent["M0_vs_M1"][1] == pytest.approx(expected_cosine("M0", "M1"))
    assert adjacent["M1_vs_M2"][1] == pytest.approx(expected_cosine("M1", "M2"))
    assert adjacent["M2_vs_M3"][1] == pytest.approx(expected_cosine("M2", "M3"))


def test_adjacent_alt_pairs_map_to_the_correct_cosine_values(tmp_path):
    ctx = make_ctx(tmp_path)
    for stage in vp.ALL_STAGES:
        build_stage_direction(ctx, stage)
    vp.aggregate_directions(ctx)

    cosine = load_json(ctx.paths.refusal_direction / "cosine_similarity_v2.json")
    adjacent_alt = cosine["adjacent_alt"]

    assert adjacent_alt["M0_vs_M1_alt"][1] == pytest.approx(
        expected_cosine("M0", "M1_alt")
    )
    assert adjacent_alt["M1_alt_vs_M2_alt"][1] == pytest.approx(
        expected_cosine("M1_alt", "M2_alt")
    )
    assert adjacent_alt["M2_alt_vs_M3_alt"][1] == pytest.approx(
        expected_cosine("M2_alt", "M3_alt")
    )


def test_direct_branch_pairs_map_to_the_correct_cosine_values(tmp_path):
    ctx = make_ctx(tmp_path)
    for stage in vp.ALL_STAGES:
        build_stage_direction(ctx, stage)
    vp.aggregate_directions(ctx)

    cosine = load_json(ctx.paths.refusal_direction / "cosine_similarity_v2.json")
    direct_branch = cosine["direct_branch"]

    assert direct_branch["M1_vs_M3_direct"][1] == pytest.approx(
        expected_cosine("M1", "M3_direct")
    )
    assert direct_branch["M3_direct_vs_M3"][1] == pytest.approx(
        expected_cosine("M3_direct", "M3")
    )
    assert direct_branch["M1_alt_vs_M3_direct_alt"][1] == pytest.approx(
        expected_cosine("M1_alt", "M3_direct_alt")
    )
    assert direct_branch["M3_direct_alt_vs_M3_alt"][1] == pytest.approx(
        expected_cosine("M3_direct_alt", "M3_alt")
    )


def test_cross_branch_pairs_map_to_the_correct_cosine_values(tmp_path):
    ctx = make_ctx(tmp_path)
    for stage in vp.ALL_STAGES:
        build_stage_direction(ctx, stage)
    vp.aggregate_directions(ctx)

    cosine = load_json(ctx.paths.refusal_direction / "cosine_similarity_v2.json")
    cross_branch = cosine["cross_branch"]

    for orig, alt in vp.CROSS_BRANCH_PAIRS:
        assert cross_branch[f"{orig}_vs_{alt}"][1] == pytest.approx(
            expected_cosine(orig, alt)
        )


def test_diagnostics_binding_records_every_stage_and_section(tmp_path):
    ctx = make_ctx(tmp_path)
    for stage in vp.ALL_STAGES:
        build_stage_direction(ctx, stage)
    vp.aggregate_directions(ctx)

    binding = load_json(ctx.paths.refusal_direction / "v2_diagnostics_binding.json")
    assert binding["stages"] == sorted(vp.ALL_STAGES)
    assert set(binding["sections"]) >= {
        "adjacent", "adjacent_alt", "direct_branch", "cross_branch",
    }
    assert binding["benchmark_sha256"] == BENCH_SHA
    assert binding["split_manifest_sha256"] == SPLIT_SHA


# ---- the actual multi-session scoping bug ---------------------------------


def test_aggregate_directions_survives_across_session_scoped_calls(tmp_path):
    """Reproduces the real T4 multi-session pattern: session 1 builds
    M0..M3 and calls aggregate_directions (as `direction --stages M0 M1 M2
    M3` would); session 2, in a LATER, separate call, builds only the alt/
    direct stages and calls aggregate_directions again (as `direction
    --stages M1_alt M2_alt M3_alt M3_direct M3_direct_alt` would). Session
    2's own call never mentions M0..M3, but their `_v2_direction.npy`
    files are still sitting on disk from session 1 - the final aggregate
    must still include every cross-session pair.
    """
    ctx = make_ctx(tmp_path)

    session_1 = ["M0", "M1", "M2", "M3"]
    for stage in session_1:
        build_stage_direction(ctx, stage)
    vp.aggregate_directions(ctx)  # what `direction --stages M0 M1 M2 M3` would do

    cosine_after_session_1 = load_json(
        ctx.paths.refusal_direction / "cosine_similarity_v2.json"
    )
    # Partial run: cross_branch/adjacent_alt/direct_branch correctly empty,
    # not missing-key or crashed - no alt/direct stages exist yet.
    assert cosine_after_session_1["adjacent_alt"] == {}
    assert cosine_after_session_1["direct_branch"] == {}
    assert cosine_after_session_1["cross_branch"] == {}
    assert set(cosine_after_session_1["adjacent"]) == {
        "M0_vs_M1", "M1_vs_M2", "M2_vs_M3",
    }

    session_2 = ["M1_alt", "M2_alt", "M3_alt", "M3_direct", "M3_direct_alt"]
    for stage in session_2:
        build_stage_direction(ctx, stage)
    vp.aggregate_directions(ctx)  # session 2's own call - never mentions M0..M3

    cosine_after_session_2 = load_json(
        ctx.paths.refusal_direction / "cosine_similarity_v2.json"
    )
    # The regression: these three sections need BOTH a session-1 stage and
    # a session-2 stage together. Before the fix, session 2's call only
    # ever "knew about" session 2's own stages, so these would still be
    # empty even though every underlying .npy file is present on disk.
    assert set(cosine_after_session_2["cross_branch"]) == {
        "M1_vs_M1_alt", "M2_vs_M2_alt", "M3_vs_M3_alt",
        "M3_direct_vs_M3_direct_alt",
    }
    assert set(cosine_after_session_2["direct_branch"]) == {
        "M1_vs_M3_direct", "M3_direct_vs_M3",
        "M1_alt_vs_M3_direct_alt", "M3_direct_alt_vs_M3_alt",
    }
    assert set(cosine_after_session_2["adjacent_alt"]) == {
        "M0_vs_M1_alt", "M1_alt_vs_M2_alt", "M2_alt_vs_M3_alt",
    }
    # session 1's section is still intact too, not clobbered.
    assert set(cosine_after_session_2["adjacent"]) == {
        "M0_vs_M1", "M1_vs_M2", "M2_vs_M3",
    }


# ---- merge_behavioral: the same bug, but data-loss instead of gaps -------


def make_behavior_ctx(tmp_path, n_rows=3):
    rows = [
        {"record_id": f"r{i:03d}", "prompt": f"prompt {i}",
         "quadrant": "A", "source_dataset": "unit_test"}
        for i in range(n_rows)
    ]
    return make_ctx(tmp_path, rows=rows)


def run_stage_behavior(ctx, stage, monkeypatch):
    def fake_generation_batch(model, tokenizer, prompts, device, max_new_tokens):
        return [f"resp_{stage}_{i}" for i in range(len(prompts))]

    monkeypatch.setattr(vp, "generation_batch", fake_generation_batch)
    assert vp.stage_behavior(ctx, stage, model=None, tokenizer=object(), device="cpu")


def test_merge_behavioral_survives_across_session_scoped_calls(tmp_path, monkeypatch):
    """Same multi-session pattern as the direction test above, but for
    merge_behavioral - and a stricter check, since the old code didn't
    just leave sections empty, it OVERWROTE v2_raw.json wholesale with
    only the stages passed to that particular call, erasing prior ones.
    """
    ctx = make_behavior_ctx(tmp_path)

    run_stage_behavior(ctx, "M0", monkeypatch)
    run_stage_behavior(ctx, "M1", monkeypatch)
    vp.merge_behavioral(ctx)  # what `behavior --stages M0 M1` would do

    combined_after_session_1 = load_json(ctx.paths.behavioral / "v2_raw.json")
    assert set(combined_after_session_1) == {"M0", "M1"}

    run_stage_behavior(ctx, "M2", monkeypatch)
    vp.merge_behavioral(ctx)  # session 2's own call - never mentions M0/M1

    combined_after_session_2 = load_json(ctx.paths.behavioral / "v2_raw.json")
    # The regression: M0 and M1 must not have been erased just because
    # this call's own session only touched M2.
    assert set(combined_after_session_2) == {"M0", "M1", "M2"}
    assert combined_after_session_2["M0"] == combined_after_session_1["M0"]
    assert combined_after_session_2["M1"] == combined_after_session_1["M1"]


# ---- downstream consumability: real reader functions, not just schema ----


def test_bridged_direction_family_is_consumable_by_summarize_cross_branch(
    tmp_path, monkeypatch
):
    from src.analysis.v2_compat import sync_diagnostics
    from src.analysis.summarize_cross_branch import direction_cross_branch_similarity

    ctx = make_ctx(tmp_path)
    for stage in vp.ALL_STAGES:
        build_stage_direction(ctx, stage)
    vp.aggregate_directions(ctx)

    root = ctx.paths.root
    assert sync_diagnostics(root=root) is True

    monkeypatch.chdir(tmp_path)
    result = direction_cross_branch_similarity(
        "M1", "M1_alt", cosine_path=str(root / "refusal_direction" / "cosine_similarity.json")
    )
    assert result is not None
    assert result["mean"] == pytest.approx(expected_cosine("M1", "M1_alt"))


def test_bridged_direction_family_is_consumable_by_direction_stability(tmp_path):
    from src.analysis.v2_compat import sync_diagnostics
    from src.interpretability.direction_stability import analyze_direction_stability

    ctx = make_ctx(tmp_path)
    for stage in vp.ALL_STAGES:
        build_stage_direction(ctx, stage)
    vp.aggregate_directions(ctx)

    root = ctx.paths.root
    assert sync_diagnostics(root=root) is True

    report = analyze_direction_stability(
        cosine_sim_path=str(root / "refusal_direction" / "cosine_similarity.json"),
        output_path=str(tmp_path / "stability_report.json"),
    )
    assert "missing_stages" not in report["metadata"]
    assert set(report["drift_dynamics"]["aggregate"]) == {
        "mean_drift_M0_vs_M1", "mean_drift_M1_vs_M2", "mean_drift_M2_vs_M3",
    }


def test_bridged_behavioral_merge_is_consumable_by_summarize_cross_branch(
    tmp_path, monkeypatch
):
    from src.analysis.v2_compat import sync_behavioral
    from src.analysis.summarize_cross_branch import (
        behavioral_rates_for_stage,
        load_raw_behavioral,
    )

    ctx = make_behavior_ctx(tmp_path, n_rows=4)
    run_stage_behavior(ctx, "M1", monkeypatch)
    vp.merge_behavioral(ctx)

    root = ctx.paths.root
    assert sync_behavioral(root=root) is True

    raw = load_raw_behavioral(str(root / "behavioral_eval" / "raw.json"))
    rates = behavioral_rates_for_stage(raw, "M1")
    assert rates is not None
    assert "A" in rates
