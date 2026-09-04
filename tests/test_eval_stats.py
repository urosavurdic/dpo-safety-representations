from src.eval_stats import rate_with_ci

def test_rate_with_ci_basic():
    result = rate_with_ci(5, 10)
    assert result["rate"] == 0.5
    assert result["n"] == 10
    assert 0 <= result["ci_low"] < 0.5 < result["ci_high"] <= 1


def test_rate_with_ci_zero_total():
    result = rate_with_ci(0, 0)
    assert result["rate"] is None
    assert result["n"] == 0


def test_rate_with_ci_all_success():
    result = rate_with_ci(10, 10)
    assert result["rate"] == 1.0
    assert result["ci_high"] <= 1.0

# --- WP-Stat: bootstrap helpers ---

import numpy as np

from src.eval_stats import (
    BOOTSTRAP_B,
    BOOTSTRAP_SEED,
    joint_resample_indices,
    paired_bootstrap_ci,
    percentile_ci,
)


def test_frozen_bootstrap_constants():
    assert BOOTSTRAP_SEED == 20260904
    assert BOOTSTRAP_B == 10_000


def test_percentile_ci_brackets_the_mean_and_ignores_nan():
    vals = np.concatenate([np.full(100, 0.2), np.full(100, 0.4), [np.nan]])
    ci = percentile_ci(vals)
    assert ci["n"] == 200
    assert ci["ci_low"] <= ci["mean"] <= ci["ci_high"]


def test_paired_bootstrap_ci_is_deterministic_and_reports_point():
    diffs = np.linspace(-0.1, 0.3, 40)
    a = paired_bootstrap_ci(diffs, b=500, seed=BOOTSTRAP_SEED)
    b = paired_bootstrap_ci(diffs, b=500, seed=BOOTSTRAP_SEED)
    assert a == b
    assert a["point"] == float(diffs.mean())
    assert a["n_effective"] == 40
    assert a["interval"] == "percentile"
    assert a["ci_low"] < a["point"] < a["ci_high"]


def test_paired_bootstrap_ci_empty():
    out = paired_bootstrap_ci([], b=10)
    assert out["n_effective"] == 0 and out["point"] is None


def test_joint_resample_indices_preserves_group_sizes_and_is_seeded():
    a_idx = np.arange(0, 6)
    d_idx = np.arange(6, 10)
    idx1, lab1 = joint_resample_indices([a_idx, d_idx], seed=1)
    idx2, lab2 = joint_resample_indices([a_idx, d_idx], seed=1)
    np.testing.assert_array_equal(idx1, idx2)
    assert (lab1 == 0).sum() == 6 and (lab1 == 1).sum() == 4
    assert set(idx1[lab1 == 0]).issubset(set(a_idx.tolist()))


def test_paired_bootstrap_ci_empty_input_has_mean_key():
    # regression: confirmatory_behavioral_endpoints indexes boot["mean"]
    # unconditionally; the empty-input branch must include it (as None).
    out = paired_bootstrap_ci([], b=10)
    assert out["mean"] is None
    assert "mean" in out
