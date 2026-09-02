"""Toy tests for src/analysis/projection_trajectory.py (WP-Geom §4.5)."""
import numpy as np
import pytest

from src.analysis import projection_trajectory as pt


def _toy(n_layers=3, hidden=6, seed=0, c_frac=0.5):
    rng = np.random.default_rng(seed)
    quads = np.array(["A"] * 6 + ["D"] * 6 + ["B"] * 4 + ["C"] * 4)
    splits = np.array(["direction_estimation"] * 12 + [""] * 8)
    n = len(quads)
    pooled = rng.standard_normal((n, n_layers, hidden))
    axis = np.zeros(hidden); axis[0] = 1.0
    # place A at +1, D at -1 on layer 1's axis0; C at c_frac between them
    pooled[quads == "A", 1] = axis * 1.0 + rng.standard_normal((6, hidden)) * 1e-3
    pooled[quads == "D", 1] = -axis * 1.0 + rng.standard_normal((6, hidden)) * 1e-3
    pooled[quads == "C", 1] = axis * (2 * c_frac - 1) + rng.standard_normal((4, hidden)) * 1e-3
    return pooled, quads, splits


def test_z_c_recovers_the_planted_position():
    pooled, quads, splits = _toy(c_frac=0.25)
    traj = pt.stage_trajectory(pooled, quads, splits)
    z_c = traj["z_C_per_layer"][1]
    assert z_c == pytest.approx(0.25, abs=0.05)


def test_z_c_missing_when_ad_gap_negligible():
    pooled, quads, splits = _toy()
    pooled[:, 2] = 0.0  # layer 2: no A-D separation at all
    traj = pt.stage_trajectory(pooled, quads, splits)
    assert traj["z_C_per_layer"][2] is None


def test_ad_gap_equals_contrast_norm():
    pooled, quads, splits = _toy()
    traj = pt.stage_trajectory(pooled, quads, splits)
    est = splits == "direction_estimation"
    m_a = pooled[(quads == "A") & est, 1].mean(0)
    m_d = pooled[(quads == "D") & est, 1].mean(0)
    assert traj["ad_gap_per_layer"][1] == pytest.approx(np.linalg.norm(m_a - m_d), rel=1e-6)


def test_build_trajectories_includes_fixed_references():
    stages = {
        "M1": _toy(seed=1), "M2": _toy(seed=2), "M3": _toy(seed=3),
    }
    out = pt.build_trajectories(stages)
    assert set(out["stage_specific"]) == {"M1", "M2", "M3"}
    assert "M1_reference" in out["fixed_reference"]
    assert "M3_reference" in out["fixed_reference"]
    assert set(out["fixed_reference"]["M1_reference"]) == {"M1", "M2", "M3"}


def test_bootstrap_z_c_deterministic_and_percentile():
    pooled, quads, splits = _toy(c_frac=0.4)
    a = pt.bootstrap_z_c(pooled, quads, splits, layer=1, n_boot=200, seed=20260904)
    b = pt.bootstrap_z_c(pooled, quads, splits, layer=1, n_boot=200, seed=20260904)
    assert a == b
    assert a["interval"] == "percentile"
    assert a["mean"] == pytest.approx(0.4, abs=0.15)
