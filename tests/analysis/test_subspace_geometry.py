"""Toy tests for src/analysis/subspace_geometry.py (WP-Geom).

Known-rotation -> recovered principal angle; isotropic activations -> PR ~ dim;
a purely in-subspace update -> rho_AD_perp ~ 0; an orthogonal update ->
rho_AD_perp ~ 1.
"""
import numpy as np
import pytest

from src.analysis import subspace_geometry as sg


def _masks(n_a=8, n_d=8):
    quads = np.array(["A"] * n_a + ["D"] * n_d + ["B"] * 4 + ["C"] * 4)
    splits = np.array(
        ["direction_estimation"] * n_a + ["direction_estimation"] * n_d + [""] * 8
    )
    return quads, splits


def test_participation_ratio_isotropic_is_near_dim():
    rng = np.random.default_rng(0)
    sv = np.ones(10)  # perfectly isotropic
    assert sg.participation_ratio(sv) == pytest.approx(10.0)
    assert sg.effective_rank(sv) == pytest.approx(10.0)


def test_participation_ratio_rank_one_is_near_one():
    sv = np.array([5.0, 0, 0, 0])
    assert sg.participation_ratio(sv) == pytest.approx(1.0)
    assert sg.effective_rank(sv) == pytest.approx(1.0)


def test_principal_angles_known_rotation_recovered():
    # two subspaces spanned by e0 and by cos t e0 + sin t e1
    t = np.radians(30.0)
    u_a = np.array([[1.0], [0.0], [0.0]])
    u_b = np.array([[np.cos(t)], [np.sin(t)], [0.0]])
    res = sg.principal_angles_deg(u_a, u_b)
    assert res["max_deg"] == pytest.approx(30.0, abs=1e-6)


def test_orthogonal_update_fraction_in_subspace_update_is_zero():
    U = np.eye(6)[:, :3]                     # subspace = first 3 dims
    c_m2 = np.zeros(6)
    c_m3 = np.array([1.0, 2.0, -1.0, 0, 0, 0])  # update lives entirely in U
    res = sg.orthogonal_update_fraction(c_m2, c_m3, U)
    assert res["rho_AD_perp"] == pytest.approx(0.0, abs=1e-12)


def test_orthogonal_update_fraction_orthogonal_update_is_one():
    U = np.eye(6)[:, :3]
    c_m2 = np.zeros(6)
    c_m3 = np.array([0, 0, 0, 1.0, -2.0, 0.5])  # entirely outside U
    res = sg.orthogonal_update_fraction(c_m2, c_m3, U)
    assert res["rho_AD_perp"] == pytest.approx(1.0, abs=1e-12)


def test_orthogonal_update_fraction_zero_update_returns_none():
    U = np.eye(4)[:, :2]
    res = sg.orthogonal_update_fraction(np.ones(4), np.ones(4), U)
    assert res["rho_AD_perp"] is None


def test_layer_report_smoke_and_keys():
    rng = np.random.default_rng(1)
    quads, splits = _masks()
    n = len(quads)
    m2 = rng.standard_normal((n, 3, 12))
    m3 = m2 + rng.standard_normal((n, 3, 12)) * 0.01
    rep = sg.layer_report(m2, m3, quads, splits, layer=1, r=3)
    assert rep["layer"] == 1
    assert set(rep) >= {
        "orthogonal_update", "principal_angles_M2_vs_M3", "participation_ratio",
        "effective_rank", "leading_vs_orthogonal_variance_M3",
        "global_drift_diagnostic", "contrast_norm",
    }
    assert "SECONDARY" in rep["global_drift_diagnostic"]["note"]


def test_bootstrap_rho_is_deterministic_under_seed():
    rng = np.random.default_rng(2)
    quads, splits = _masks()
    n = len(quads)
    m2 = rng.standard_normal((n, 2, 10))
    m3 = m2 + rng.standard_normal((n, 2, 10)) * 0.5
    a = sg.bootstrap_rho_ad_perp(m2, m3, quads, splits, layer=0, r=3, n_boot=200, seed=20260904)
    b = sg.bootstrap_rho_ad_perp(m2, m3, quads, splits, layer=0, r=3, n_boot=200, seed=20260904)
    assert a == b
    assert a["interval"] == "percentile" and a["seed"] == 20260904
