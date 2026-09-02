"""Toy tests for src/analysis/control_directions.py (WP-Ctrl).

Synthetic activations only - no model. Checks the gamma math, the RMS-match
property, cos(r, d_AD) recording, the zero-magnitude failure, and that the
calibration set is the A/D direction_estimation split (never held-out
behavioral rows).
"""
import numpy as np
import pytest

from src.analysis import control_directions as cd


def _toy(n_layers=5, hidden=8, seed=0):
    rng = np.random.default_rng(seed)
    # 6 A + 6 D + 4 B rows; A/D split half estimation / half held_out.
    quads, splits = [], []
    for q in ("A", "D"):
        for k in range(6):
            quads.append(q)
            splits.append("direction_estimation" if k < 4 else "held_out_behavioral")
    for _ in range(4):
        quads.append("B")
        splits.append("")
    n = len(quads)
    pooled = rng.standard_normal((n, n_layers, hidden)).astype(np.float64)
    # inject a real A>D mean gap on a couple of layers
    a = np.array([q == "A" for q in quads])
    d = np.array([q == "D" for q in quads])
    pooled[a, 1] += 3.0
    pooled[d, 1] -= 3.0
    return pooled, np.array(quads), np.array(splits)


def test_ad_direction_is_unit_and_estimation_split_only():
    pooled, quads, splits = _toy()
    d_ad = cd.ad_direction(pooled, quads, splits=splits)
    assert d_ad.shape == (5, 8)
    np.testing.assert_allclose(np.linalg.norm(d_ad, axis=-1), 1.0, atol=1e-9)

    # perturbing ONLY held-out rows must not change the estimation-split direction
    pooled2 = pooled.copy()
    held = splits == "held_out_behavioral"
    pooled2[held] += 50.0
    d_ad2 = cd.ad_direction(pooled2, quads, splits=splits)
    np.testing.assert_allclose(d_ad, d_ad2, atol=1e-9)


def test_seeded_random_directions_reproducible_and_unit():
    r1 = cd.seeded_random_directions(5, 8, seed=123)
    r2 = cd.seeded_random_directions(5, 8, seed=123)
    np.testing.assert_array_equal(r1, r2)
    np.testing.assert_allclose(np.linalg.norm(r1, axis=-1), 1.0, atol=1e-9)
    assert not np.allclose(r1, cd.seeded_random_directions(5, 8, seed=124))


def test_gamma_makes_random_ablation_rms_match_the_learned_one():
    pooled, quads, splits = _toy()
    d_ad = cd.ad_direction(pooled, quads, splits=splits)
    r = cd.seeded_random_directions(5, 8, seed=7)
    ctrl = cd.build_ablation_control(
        pooled, quads, splits, d_ad, r, layers=range(5),
        record_ids=[f"row{i}" for i in range(len(quads))],
    )
    calib = pooled[((quads == "A") | (quads == "D")) & (splits == "direction_estimation")]
    for l in range(5):
        a_ad = cd.rms_projected_norm(calib[:, l], d_ad[l])
        # applying gamma * (h.r) r should have the SAME rms projected magnitude
        scaled_r_proj = cd.rms_projected_norm(calib[:, l], r[l]) * ctrl.gamma[l]
        assert scaled_r_proj == pytest.approx(a_ad, rel=1e-9)
    assert ctrl.n_calibration_rows == 8  # 4 A + 4 D estimation rows
    assert len(ctrl.calibration_record_ids) == 8


def test_control_records_cos_r_dAD_and_seed():
    pooled, quads, splits = _toy()
    rec = cd.build_stage_control_record(
        pooled, quads, splits, seed=20260904, layers=range(5)
    )
    assert rec["random_direction_seed"] == 20260904
    per_layer = rec["ablation_control"]["per_layer"]
    assert set(per_layer) == {"0", "1", "2", "3", "4"}
    for entry in per_layer.values():
        assert -1.0 <= entry["realised_cos_r_dAD"] <= 1.0


def test_zero_magnitude_random_direction_raises():
    pooled, quads, splits = _toy()
    d_ad = cd.ad_direction(pooled, quads, splits=splits)
    r = cd.seeded_random_directions(5, 8, seed=1)
    # make the calibration rows exactly orthogonal to r on layer 0 by zeroing
    # that layer's activations
    pooled = pooled.copy()
    pooled[:, 0] = 0.0
    with pytest.raises(cd.ZeroMagnitudeError):
        cd.build_ablation_control(pooled, quads, splits, d_ad, r, layers=[0])
    # non-strict: recorded, not raised
    ctrl = cd.build_ablation_control(
        pooled, quads, splits, d_ad, r, layers=[0], strict_zero=False
    )
    assert ctrl.zero_magnitude_layers == [0]
    assert np.isnan(ctrl.gamma[0])


def test_dAB_cosine_is_descriptive_no_gate():
    pooled, quads, splits = _toy()
    rec = cd.build_stage_control_record(pooled, quads, splits, layers=range(5))
    cos = rec["d_AB_vs_d_AD_cosine_per_layer"]
    assert len(cos) == 5
    assert "NONE" in rec["d_AB_gate"]


def test_apply_random_ablation_removes_the_r_component():
    rng = np.random.default_rng(0)
    r = rng.standard_normal(8)
    r /= np.linalg.norm(r)
    h = rng.standard_normal((3, 8))
    out = cd.apply_random_ablation(h, r, gamma_layer=1.0)
    # with gamma=1 this is a full projection removal -> residual is orthogonal to r
    np.testing.assert_allclose(out @ r, 0.0, atol=1e-9)
