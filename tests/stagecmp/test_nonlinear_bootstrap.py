"""The shared-resample bootstrap for ratio statistics.

A ratio does not collapse to one scalar per prompt, so ``paired_bootstrap_ci``
does not apply and the groups have to be resampled with the statistic
recomputed. The test that earns its keep is
``test_pre_and_post_are_read_on_the_same_resampled_rows``: it fails if pre and
post are ever resampled independently, which is the one mistake that would make
every reported interval wrong in a way no other check would notice.
"""
from __future__ import annotations

import json

import numpy as np
import pytest

from src.common.stats import bootstrap_nonlinear_contrast
from src.stagecmp.repr_metrics import CI_LAYERS, ratio_confidence_intervals


def test_it_is_re_exported_from_contrasts_for_existing_callers():
    from src.stagecmp.contrasts import bootstrap_nonlinear_contrast as reexport

    assert reexport is bootstrap_nonlinear_contrast

N_LAYERS = 29          # real depth, so the fixed CI grid applies
HIDDEN = 6


def _two_stage_act_dir(tmp_path, transform, n_per_quadrant=20, n_est=8,
                       n_layers=N_LAYERS):
    """Activations for M1 and M2 where ``M2 = transform(M1)``, elementwise.

    ``n_est`` rows per quadrant carry the direction-estimation split, so the
    A/D contrast bootstrap has more than one unit to resample and a
    zero-width interval is genuinely informative rather than forced.
    """
    records = []
    for q in "ABCD":
        for i in range(n_per_quadrant):
            records.append({
                "record_id": f"{q}{i:03d}",
                "quadrant": q,
                "split": "direction_estimation" if i < n_est else "held_out_behavioral",
            })
    rng = np.random.default_rng(7)
    pre = rng.normal(size=(len(records), n_layers, HIDDEN)).astype(np.float32)
    for i, row in enumerate(records):          # make the A-D contrast non-zero
        if row["quadrant"] == "A":
            pre[i, :, 0] += 2.0
    post = transform(pre).astype(np.float32)

    d = tmp_path / "activations"
    d.mkdir()
    for stage, arr in (("M1", pre), ("M2", post)):
        for suffix in ("final", "pooled"):
            np.save(d / f"{stage}_{suffix}.npy", arr)
        (d / f"{stage}_metadata.json").write_text(json.dumps(records), encoding="utf-8")
    return d


# ---- the generic driver -------------------------------------------------

def _ratio_setup():
    rng = np.random.default_rng(0)
    values = np.concatenate([rng.normal(10.0, 1.0, 150), rng.normal(5.0, 1.0, 150)])
    groups = [np.arange(150), np.arange(150, 300)]

    def statistic(g):
        return float(values[g[0]].mean() / values[g[1]].mean())

    return groups, statistic


def test_interval_brackets_the_point_and_recovers_the_true_ratio():
    groups, statistic = _ratio_setup()
    out = bootstrap_nonlinear_contrast(groups, statistic, b=2000)
    assert out["ci_low"] <= out["point"] <= out["ci_high"]
    assert out["ci_low"] < 2.0 < out["ci_high"]          # true ratio is 10/5
    assert out["group_sizes"] == [150, 150]


def test_deterministic_under_the_frozen_seed():
    groups, statistic = _ratio_setup()
    a = bootstrap_nonlinear_contrast(groups, statistic, b=500)
    b = bootstrap_nonlinear_contrast(groups, statistic, b=500)
    assert a == b


def test_a_different_seed_gives_a_different_interval():
    groups, statistic = _ratio_setup()
    a = bootstrap_nonlinear_contrast(groups, statistic, b=500, seed=1)
    b = bootstrap_nonlinear_contrast(groups, statistic, b=500, seed=2)
    assert a["ci_low"] != b["ci_low"]


def test_degenerate_replicates_are_dropped_not_propagated():
    """A resample the statistic cannot evaluate must not turn the CI into nan."""
    groups, _ = _ratio_setup()
    calls = {"n": 0}

    def sometimes_nan(g):
        calls["n"] += 1
        return float("nan") if calls["n"] % 2 else 1.5

    out = bootstrap_nonlinear_contrast(groups, sometimes_nan, b=400)
    assert out["n"] < 400                                 # some were dropped
    assert out["ci_low"] == pytest.approx(1.5)
    assert np.isfinite(out["ci_high"])


def test_an_empty_group_returns_nulls_rather_than_raising():
    groups, statistic = _ratio_setup()
    out = bootstrap_nonlinear_contrast([np.array([], int), groups[1]], statistic, b=10)
    assert out["point"] is None and out["ci_low"] is None
    assert out["group_sizes"] == [0, 150]


# ---- the representational ratios ---------------------------------------

def test_pre_and_post_are_read_on_the_same_resampled_rows(tmp_path):
    """If ``post = 2 * pre`` elementwise, the contrast ratio is exactly 2 for
    EVERY resample -- but only because both sides use the same rows. Resample
    pre and post independently and this interval acquires width."""
    act_dir = _two_stage_act_dir(tmp_path, lambda a: 2.0 * a)
    out = ratio_confidence_intervals("M1", "M2", act_dir, "final_token", b=300)
    cell = out["by_layer"][str(out["layers"][0])]["contrast_norm_ratio"]
    assert cell["point"] == pytest.approx(2.0, abs=1e-5)
    assert cell["ci_low"] == pytest.approx(2.0, abs=1e-5)
    assert cell["ci_high"] == pytest.approx(2.0, abs=1e-5)


def test_a_stage_against_itself_gives_unit_ratio_and_zero_drift(tmp_path):
    act_dir = _two_stage_act_dir(tmp_path, lambda a: a.copy())
    out = ratio_confidence_intervals("M1", "M2", act_dir, "final_token", b=200)
    for cells in out["by_layer"].values():
        c = cells["contrast_norm_ratio"]
        assert c["point"] == pytest.approx(1.0, abs=1e-6)
        # no displacement anywhere, so both drift ratios are undefined (0/0)
        assert cells["drift_ratio_A_over_D"]["point"] is None
        assert cells["drift_ratio_C_over_B"]["point"] is None


def test_isotropic_movement_scores_one_whatever_its_magnitude(tmp_path):
    """The property that makes these ratios answer the scale objection: a
    transition that moves every prompt equally is not 'selective', however
    large the move."""
    shift = np.zeros((1, N_LAYERS, HIDDEN), dtype=np.float32)
    shift[0, :, 1] = 50.0
    act_dir = _two_stage_act_dir(tmp_path, lambda a: a + shift)
    out = ratio_confidence_intervals("M1", "M2", act_dir, "final_token", b=200)
    for cells in out["by_layer"].values():
        for key in ("drift_ratio_A_over_D", "drift_ratio_C_over_B"):
            assert cells[key]["point"] == pytest.approx(1.0, abs=1e-4)


def test_layers_outside_the_stack_are_dropped_not_indexed(tmp_path):
    """A 22-layer stack keeps 16 and 20 from the grid and drops 24 and 28,
    rather than indexing past the end."""
    act_dir = _two_stage_act_dir(tmp_path, lambda a: 1.5 * a, n_layers=22)
    out = ratio_confidence_intervals("M1", "M2", act_dir, "final_token", b=50)
    assert out["layers"] == [16, 20]
    assert set(out["by_layer"]) == {"16", "20"}


def test_the_full_grid_is_used_on_a_full_depth_stack(tmp_path):
    act_dir = _two_stage_act_dir(tmp_path, lambda a: 1.5 * a)
    out = ratio_confidence_intervals("M1", "M2", act_dir, "final_token", b=50)
    assert out["layers"] == list(CI_LAYERS)


def test_the_ci_grid_is_the_matrix_grid_so_it_is_not_chosen_from_results():
    from src.stagecmp.run_wp3 import MATRIX_LAYERS

    assert CI_LAYERS == MATRIX_LAYERS
