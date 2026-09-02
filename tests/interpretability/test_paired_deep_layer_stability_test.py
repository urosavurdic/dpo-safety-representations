"""Tests for the frozen prompt-level joint-bootstrap deep-layer stability
difference (WP-Stat). The old replicate-index Wilcoxon path is deprecated and
only smoke-checked for importability.
"""
import json

import numpy as np
import pytest

from src.interpretability import paired_deep_layer_stability_test as pd


def _synth(n_layers=29, hidden=12, seed=0, wobble=0.0):
    """6 A + 6 D + 4 B/C rows. A/D separated on axis 0 at every layer; `wobble`
    adds per-row noise to the deep layers (16-28) to make the resampled
    direction less stable."""
    rng = np.random.default_rng(seed)
    quads = np.array(["A"] * 6 + ["D"] * 6 + ["B"] * 4 + ["C"] * 4)
    n = len(quads)
    pooled = rng.standard_normal((n, n_layers, hidden)) * 0.05
    for l in range(n_layers):
        pooled[quads == "A", l, 0] += 1.0
        pooled[quads == "D", l, 0] -= 1.0
    if wobble:
        deep = list(range(16, 29))
        pooled[:, deep] += rng.standard_normal((n, len(deep), hidden)) * wobble
    return pooled, quads


def test_joint_bootstrap_returns_frozen_shape_and_is_deterministic():
    p_d, q = _synth(seed=1)
    p_m, q2 = _synth(seed=1)
    a = pd.joint_bootstrap_deep_layer_stability_difference(
        p_d, q, p_m, q2, n_bootstrap=100, seed=20260904
    )
    b = pd.joint_bootstrap_deep_layer_stability_difference(
        p_d, q, p_m, q2, n_bootstrap=100, seed=20260904
    )
    assert a["difference_direct_minus_mediated"] == b["difference_direct_minus_mediated"]
    assert a["seed"] == 20260904 and a["interval"] == "percentile"
    for key in ("direct_stability", "mediated_stability", "difference_direct_minus_mediated"):
        assert {"mean", "ci_low", "ci_high"} <= set(a[key])


def test_joint_bootstrap_detects_direct_more_stable():
    # mediated stage has extra deep-layer wobble -> lower stability -> diff > 0
    p_direct, q = _synth(seed=2, wobble=0.0)
    p_mediated, q2 = _synth(seed=2, wobble=0.9)
    res = pd.joint_bootstrap_deep_layer_stability_difference(
        p_direct, q, p_mediated, q2, n_bootstrap=300, seed=20260904
    )
    assert res["difference_direct_minus_mediated"]["mean"] > 0
    assert res["frac_replicates_direct_gt_mediated"] > 0.6


def test_joint_bootstrap_rejects_mismatched_quadrant_order():
    p, q = _synth(seed=3)
    p2, _ = _synth(seed=3)
    q_bad = q.copy()
    q_bad[0] = "D"
    with pytest.raises(ValueError, match="quadrant order must match"):
        pd.joint_bootstrap_deep_layer_stability_difference(p, q, p2, q_bad, n_bootstrap=10)


def test_main_uses_activations_and_writes_output(tmp_path, monkeypatch, capsys):
    stages = {}
    for direct, mediated in pd.DIRECT_VS_MEDIATED_PAIRS:
        stages[direct] = _synth(seed=hash(direct) % 100, wobble=0.0)
        stages[mediated] = _synth(seed=hash(direct) % 100, wobble=0.4)

    def fake_load_stage(stage):
        pooled, quads = stages[stage]
        splits = np.array(["direction_estimation"] * len(quads))
        return pooled, quads, splits

    monkeypatch.setattr(pd, "load_stage", fake_load_stage)
    monkeypatch.setattr(pd, "activations_available", lambda s: s in stages)
    monkeypatch.setattr(pd, "filter_to_direction_estimation_split",
                        lambda pooled, quads, splits: (pooled, quads))
    monkeypatch.setattr(pd, "OUT_PATH", tmp_path / "out.json")
    monkeypatch.setattr("sys.argv", ["x", "--n-bootstrap", "80"])

    pd.main()
    out = json.loads((tmp_path / "out.json").read_text())
    assert out["method"] == "prompt_level_joint_bootstrap"
    assert len(out["per_branch"]) == len(pd.DIRECT_VS_MEDIATED_PAIRS)
    assert "pooled_across_branches" in out


def test_deprecated_wilcoxon_helpers_still_importable_but_warn():
    with pytest.warns(DeprecationWarning):
        pd.paired_stability_test(np.ones(10) * 0.9, np.ones(10) * 0.8)
