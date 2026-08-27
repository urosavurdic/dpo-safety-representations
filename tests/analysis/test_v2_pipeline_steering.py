"""Milestone 6A regression tests: v2 steering calibration, stage-selection
separation, and norm-diagnostic condition registration.

These sit alongside tests/analysis/test_v2_pipeline_deadline.py (which
protects stage_start_blocked/stage_is_complete) and
tests/interpretability/test_residual_norm_tracking.py (which protects the
norm-comparison math in isolation with tiny CPU tensors). This file
protects the pieces in src/analysis/v2_pipeline.py that neither of those
cover: that steering's alpha calibration only ever reads the
quadrant-A/direction_estimation split (never the held-out behavioral
rows), that a missing calibration artifact fails loudly instead of
silently calibrating on the wrong thing, that steering_tag produces
unique/deterministic/stage-specific names, that stage_plan keeps
causal-ablation stage selection and steering stage selection independent,
and that stage_norm_diag registers exactly the four intended conditions
(baseline, collapsing multi-layer, non-collapsing single-layer,
norm-preserving) through the same v2 architecture.

No real model/tokenizer is used anywhere in this file. The norm-diagnostic
test monkeypatches v2_pipeline.decoder_layers and v2_pipeline.generation_batch
with fakes built from tiny CPU nn.Module layers (the same
_FakeDecoderLayer pattern tests/interpretability/test_residual_norm_tracking.py
and tests/analysis/test_eval_causal_ablation.py already use), so the real
hook-registration/removal code path is exercised without touching a GPU
or a real checkpoint.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch
import torch.nn as nn

from src.analysis.v2_pipeline import (
    ArtifactPaths,
    RunContext,
    activation_paths,
    calibration_alpha,
    resolve_alphas,
    stage_norm_diag,
    stage_plan,
    steering_tag,
)
from src.analysis.v2_shards import Deadline
from src.v2_io import write_json_lf

BENCH_SHA = "a" * 64
SPLIT_SHA = "b" * 64


# --------------------------------------------------------------------------
# shared fixtures
# --------------------------------------------------------------------------


def make_ctx(tmp_path, rows, deadline=None):
    return RunContext(
        benchmark_path=tmp_path / "bench.jsonl",
        benchmark_sha=BENCH_SHA,
        split_path=tmp_path / "split.json",
        split_sha=SPLIT_SHA,
        rows=rows,
        paths=ArtifactPaths(root=tmp_path / "results"),
        deadline=deadline if deadline is not None else Deadline(),
    )


def write_bound_activation(ctx, stage, pooled_array, final_array=None):
    final_path, pooled_path, metadata_path, binding_path = (
        activation_paths(ctx, stage)
    )
    final_path.parent.mkdir(parents=True, exist_ok=True)
    if final_array is None:
        final_array = pooled_array
    np.save(final_path, final_array)
    np.save(pooled_path, pooled_array)
    write_json_lf(metadata_path, ctx.snapshot)
    write_json_lf(binding_path, {**ctx.bind(), "stage": stage})


# --------------------------------------------------------------------------
# calibration_alpha: correct split, missing-artifact failure
# --------------------------------------------------------------------------


def _calibration_rows():
    return [
        {
            "record_id": "r0",
            "prompt": "p0",
            "quadrant": "A",
            "split": "direction_estimation",
        },
        {
            # Same quadrant, but the held-out half -- must NOT influence
            # calibration (requirement 5: held-out behavioral rows are
            # not used for alpha selection).
            "record_id": "r1",
            "prompt": "p1",
            "quadrant": "A",
            "split": "held_out_behavioral",
        },
        {
            # Direction-estimation split, but the wrong quadrant -- must
            # NOT influence calibration either.
            "record_id": "r2",
            "prompt": "p2",
            "quadrant": "D",
            "split": "direction_estimation",
        },
        {
            "record_id": "r3",
            "prompt": "p3",
            "quadrant": "A",
            "split": "direction_estimation",
        },
    ]


def test_calibration_alpha_uses_only_quadrant_a_direction_estimation_rows(
    tmp_path,
):
    rows = _calibration_rows()
    ctx = make_ctx(tmp_path, rows)

    # 2 layers x hidden_dim 2. Layer 1 projections onto direction [1, 0]:
    # r0 -> 3.0, r1 -> 100.0 (must be excluded), r2 -> 200.0 (must be
    # excluded), r3 -> 5.0. Mean over the correct rows (r0, r3) is 4.0;
    # including either excluded row would change the result.
    pooled = np.zeros((4, 2, 2), dtype=np.float32)
    pooled[0, 1] = [3.0, 0.0]
    pooled[1, 1] = [100.0, 0.0]
    pooled[2, 1] = [200.0, 0.0]
    pooled[3, 1] = [5.0, 0.0]
    write_bound_activation(ctx, "M3", pooled)

    direction = np.zeros((2, 2), dtype=np.float32)
    direction[1] = [1.0, 0.0]

    alpha = calibration_alpha(ctx, "M3", layer=1, direction=direction)
    assert alpha == pytest.approx(4.0)


def test_calibration_alpha_raises_when_no_direction_estimation_a_rows(
    tmp_path,
):
    # Activations exist and are correctly bound, but no row satisfies
    # quadrant==A and split==direction_estimation -- must fail loudly,
    # not silently calibrate on whatever rows happen to be present.
    rows = [
        {
            "record_id": "r0",
            "prompt": "p0",
            "quadrant": "A",
            "split": "held_out_behavioral",
        },
        {
            "record_id": "r1",
            "prompt": "p1",
            "quadrant": "D",
            "split": "direction_estimation",
        },
    ]
    ctx = make_ctx(tmp_path, rows)
    pooled = np.zeros((2, 2, 2), dtype=np.float32)
    write_bound_activation(ctx, "M3", pooled)
    direction = np.zeros((2, 2), dtype=np.float32)

    with pytest.raises(RuntimeError, match="direction-estimation"):
        calibration_alpha(ctx, "M3", layer=1, direction=direction)


def test_calibration_alpha_raises_when_activation_artifacts_are_missing(
    tmp_path,
):
    # Nothing was ever extracted/bound for this stage -- must fail loudly
    # (requirement 6) rather than falling back to a stale or default
    # calibration.
    rows = _calibration_rows()
    ctx = make_ctx(tmp_path, rows)
    direction = np.zeros((2, 2), dtype=np.float32)

    with pytest.raises(FileNotFoundError):
        calibration_alpha(ctx, "M3", layer=1, direction=direction)


def test_calibration_alpha_raises_when_bound_to_a_different_benchmark(
    tmp_path,
):
    # Artifacts exist on disk but were bound against a different
    # benchmark/split -- a resumed session against new data must not
    # silently calibrate against the old one.
    rows = _calibration_rows()
    ctx = make_ctx(tmp_path, rows)
    pooled = np.zeros((4, 2, 2), dtype=np.float32)
    write_bound_activation(ctx, "M3", pooled)

    other = make_ctx(tmp_path, rows)
    other.benchmark_sha = "c" * 64
    direction = np.zeros((2, 2), dtype=np.float32)

    with pytest.raises(RuntimeError, match="different benchmark"):
        calibration_alpha(other, "M3", layer=1, direction=direction)


# --------------------------------------------------------------------------
# resolve_alphas
# --------------------------------------------------------------------------


def test_resolve_alphas_fixed_source_requires_alpha_value(tmp_path):
    ctx = make_ctx(tmp_path, _calibration_rows())
    direction = np.zeros((2, 2), dtype=np.float32)

    with pytest.raises(ValueError, match="alpha-value"):
        resolve_alphas(ctx, "M3", direction, [1], "fixed", None, 1.0)


def test_resolve_alphas_fixed_source_applies_coefficient(tmp_path):
    ctx = make_ctx(tmp_path, _calibration_rows())
    direction = np.zeros((2, 2), dtype=np.float32)

    alphas = resolve_alphas(
        ctx, "M3", direction, [1], "fixed", 3.5, 2.0
    )
    assert alphas == {1: 7.0}


def test_resolve_alphas_calibration_source_scales_calibration_by_coefficient(
    tmp_path,
):
    rows = _calibration_rows()
    ctx = make_ctx(tmp_path, rows)
    pooled = np.zeros((4, 2, 2), dtype=np.float32)
    pooled[0, 1] = [3.0, 0.0]
    pooled[3, 1] = [5.0, 0.0]
    write_bound_activation(ctx, "M3", pooled)

    direction = np.zeros((2, 2), dtype=np.float32)
    direction[1] = [1.0, 0.0]

    alphas = resolve_alphas(
        ctx,
        "M3",
        direction,
        [1],
        "direction_estimation_only",
        None,
        2.0,
    )
    # calibration_alpha(...) == 4.0 (see the split test above); coefficient
    # doubles it.
    assert alphas == {1: pytest.approx(8.0)}


# --------------------------------------------------------------------------
# steering_tag: unique, deterministic, stage-specific names
# --------------------------------------------------------------------------


def test_steering_tag_is_deterministic():
    tag_a = steering_tag("M3", [24], "fixed", 1.0, ["A", "D"])
    tag_b = steering_tag("M3", [24], "fixed", 1.0, ["A", "D"])
    assert tag_a == tag_b


def test_steering_tag_differs_by_stage():
    tag_m3 = steering_tag("M3", [24], "fixed", 1.0, ["A", "D"])
    tag_m3_alt = steering_tag("M3_alt", [24], "fixed", 1.0, ["A", "D"])
    assert tag_m3 != tag_m3_alt
    assert tag_m3.startswith("M3_")
    assert tag_m3_alt.startswith("M3_alt_")


def test_steering_tag_differs_by_config():
    # Same stage, one field changed at a time -- every variant must be
    # unique so two different configs for the same stage can never
    # collide on one output path (requirement 9).
    base = steering_tag("M3", [24], "fixed", 1.0, ["A", "D"])
    variants = {
        base,
        steering_tag("M3", [25], "fixed", 1.0, ["A", "D"]),
        steering_tag("M3", [24], "direction_estimation_only", 1.0, ["A", "D"]),
        steering_tag("M3", [24], "fixed", 2.0, ["A", "D"]),
        steering_tag("M3", [24], "fixed", 1.0, ["A"]),
    }
    assert len(variants) == 5


# --------------------------------------------------------------------------
# stage_plan: causal-ablation and steering stage selection stay separate
# --------------------------------------------------------------------------


class _FakeRunArgs:
    """Minimal stand-in for the `run` subcommand's argparse.Namespace,
    only the fields stage_plan reads."""

    def __init__(self, **kw):
        defaults = dict(
            analysis_stage=None,
            no_causal=False,
            stage=None,
            no_steering=False,
            steering_stage=None,
            with_norm_diag=False,
            no_behavior=False,
        )
        defaults.update(kw)
        self.__dict__.update(defaults)


def test_stage_plan_default_separates_causal_and_steering_selection():
    plan = stage_plan(_FakeRunArgs())
    by_stage = {item["stage"]: item for item in plan}

    # M0 has no trained refusal behaviour to steer and is not one of the
    # four DPO endpoints: neither causal nor steering should touch it.
    assert by_stage["M0"]["causal"] is False
    assert by_stage["M0"]["steering"] is False

    # M3 is both a DPO endpoint (causal) and a trained stage (steering).
    assert by_stage["M3"]["causal"] is True
    assert by_stage["M3"]["steering"] is True

    # M1 is a trained stage but not one of the four DPO endpoints: it
    # must get steering without getting causal ablation.
    assert by_stage["M1"]["causal"] is False
    assert by_stage["M1"]["steering"] is True


def test_stage_plan_explicit_overrides_keep_the_two_selections_independent():
    # Requesting only M3 for causal and only M1 for steering must not
    # leak into each other -- confirms the two `--stage`/`--steering-stage`
    # CLI knobs are genuinely independent, not aliases of one list.
    plan = stage_plan(
        _FakeRunArgs(stage=["M3"], steering_stage=["M1"])
    )
    by_stage = {item["stage"]: item for item in plan}

    assert by_stage["M3"]["causal"] is True
    assert by_stage["M3"]["steering"] is False
    assert by_stage["M1"]["steering"] is True
    assert by_stage["M1"]["causal"] is False


def test_stage_plan_norm_diag_requires_both_the_flag_and_steering_selection():
    plan_off = stage_plan(_FakeRunArgs(with_norm_diag=False))
    assert all(item["norm_diag"] is False for item in plan_off)

    plan_on = stage_plan(
        _FakeRunArgs(with_norm_diag=True, steering_stage=["M3"])
    )
    by_stage = {item["stage"]: item for item in plan_on}
    assert by_stage["M3"]["norm_diag"] is True
    # M1 was not selected for steering, so norm_diag must not run there
    # even though the flag is on globally.
    assert by_stage["M1"]["norm_diag"] is False


def test_stage_plan_no_causal_and_no_steering_flags_empty_both_lists():
    plan = stage_plan(_FakeRunArgs(no_causal=True, no_steering=True))
    assert all(item["causal"] is False for item in plan)
    assert all(item["steering"] is False for item in plan)


# --------------------------------------------------------------------------
# stage_norm_diag: registers the four intended diagnostic conditions
# --------------------------------------------------------------------------


N_DECODER_LAYERS = 28
HIDDEN_DIM = 2


class _FakeDecoderLayer(nn.Module):
    """Same no-op-passthrough pattern as
    tests/interpretability/test_residual_norm_tracking.py's
    _FakeDecoderLayer: returns (hidden_states,) like a real HF decoder
    block, so both the steering hooks and ResidualNormTracker's hook can
    register on it without a real model."""

    def forward(self, x):
        return (x,)


def _norm_diag_rows():
    # One quadrant-A/direction_estimation row so calibration_alpha has
    # something to calibrate from, plus the held-out quadrant-D rows the
    # diagnostic itself runs on.
    rows = [
        {
            "record_id": "a0",
            "prompt": "calibration prompt",
            "quadrant": "A",
            "split": "direction_estimation",
        }
    ]
    for i in range(2):
        rows.append(
            {
                "record_id": f"d{i}",
                "prompt": f"held-out prompt {i}",
                "quadrant": "D",
                "split": "held_out_behavioral",
            }
        )
    return rows


def _make_norm_diag_ctx(tmp_path):
    rows = _norm_diag_rows()
    ctx = make_ctx(tmp_path, rows)
    # pooled/direction need one row per decoder layer plus the embedding
    # row (hidden_states convention: index 0 = embeddings, index k = the
    # output of decoder block k-1), so N_DECODER_LAYERS + 1 layers.
    pooled = np.zeros((len(rows), N_DECODER_LAYERS + 1, HIDDEN_DIM), dtype=np.float32)
    write_bound_activation(ctx, "M3", pooled)
    direction = np.zeros((N_DECODER_LAYERS + 1, HIDDEN_DIM), dtype=np.float32)
    return ctx, direction


def _patch_fake_model(monkeypatch):
    import src.analysis.v2_pipeline as v2_pipeline

    fake_layers = [_FakeDecoderLayer() for _ in range(N_DECODER_LAYERS)]

    def fake_decoder_layers(_model):
        return fake_layers

    def fake_generation_batch(model, tokenizer, prompts, device, max_new_tokens):
        # Simulate one prefill forward pass through every decoder layer so
        # any active hooks (steering + ResidualNormTracker) actually fire,
        # the same way a real model.generate() call would on token 0.
        x = torch.zeros(len(prompts), 1, HIDDEN_DIM)
        for layer in fake_layers:
            x = layer(x)[0]
        return [f"response-{i}" for i in range(len(prompts))]

    monkeypatch.setattr(v2_pipeline, "decoder_layers", fake_decoder_layers)
    monkeypatch.setattr(v2_pipeline, "generation_batch", fake_generation_batch)
    return fake_layers


def test_stage_norm_diag_registers_all_four_conditions_by_default(
    tmp_path, monkeypatch
):
    ctx, direction = _make_norm_diag_ctx(tmp_path)
    _patch_fake_model(monkeypatch)

    ok = stage_norm_diag(
        ctx,
        "M3",
        model=object(),
        tokenizer=object(),
        device="cpu",
        direction=direction,
        n_prompts=2,
        also_test_fix=True,
    )
    assert ok is True

    from src.v2_io import load_json

    output_path = ctx.paths.raw / "residual_norm_v2_M3.json"
    payload = load_json(output_path)

    assert set(payload["conditions"]) == {
        "baseline",
        "collapsing_L14-28",
        "noncollapsing_L24",
        "norm_preserving_L14-28",
    }
    # Every non-baseline condition must carry a comparison against the
    # baseline range -- that's the actual diagnostic, not just a response
    # dump.
    for name, condition in payload["conditions"].items():
        if name == "baseline":
            continue
        assert "comparison_to_baseline" in condition
        assert "first_step_exceeding_p99" in condition


def test_stage_norm_diag_omits_the_fix_condition_when_disabled(
    tmp_path, monkeypatch
):
    ctx, direction = _make_norm_diag_ctx(tmp_path)
    _patch_fake_model(monkeypatch)

    stage_norm_diag(
        ctx,
        "M3",
        model=object(),
        tokenizer=object(),
        device="cpu",
        direction=direction,
        n_prompts=2,
        also_test_fix=False,
    )

    from src.v2_io import load_json

    payload = load_json(ctx.paths.raw / "residual_norm_v2_M3.json")
    assert set(payload["conditions"]) == {
        "baseline",
        "collapsing_L14-28",
        "noncollapsing_L24",
    }


def test_stage_norm_diag_uses_held_out_quadrant_d_only(tmp_path, monkeypatch):
    ctx, direction = _make_norm_diag_ctx(tmp_path)
    _patch_fake_model(monkeypatch)

    stage_norm_diag(
        ctx,
        "M3",
        model=object(),
        tokenizer=object(),
        device="cpu",
        direction=direction,
        n_prompts=2,
        also_test_fix=False,
    )

    from src.v2_io import load_json

    payload = load_json(ctx.paths.raw / "residual_norm_v2_M3.json")
    assert payload["quadrant"] == "D"
    assert payload["split"] == "held_out_behavioral"
    assert payload["n_prompts"] == 2
