"""Toy-data tests for the 2x2 factorial direction audit."""
import json
import numpy as np
import pytest

from src.analysis import factorial_direction_audit as fda


def _write(tmp_path, stage, harm_vec, cue_vec, n_layers=3, hidden=6, noise=0.001):
    """Build activations where cell mean = harm*harm_vec + cue*cue_vec."""
    rng = np.random.default_rng(0)
    rows, arrs = [], []
    plan = [("A", 8), ("B", 8), ("C", 8), ("D", 8)]
    for cell, k in plan:
        h, c = fda.FACTORS[cell]
        for i in range(k):
            rows.append({"record_id": f"{cell}{i}", "quadrant": cell,
                         "split": "direction_estimation"})
            base = h * harm_vec + c * cue_vec
            arrs.append(np.tile(base, (n_layers, 1))
                        + noise * rng.standard_normal((n_layers, hidden)))
    d = tmp_path / "act"
    d.mkdir(parents=True, exist_ok=True)
    (d / f"{stage}_metadata.json").write_text(json.dumps(rows), encoding="utf-8")
    np.save(d / f"{stage}_final.npy", np.stack(arrs))
    return d


def test_decomposition_is_exact_and_recovers_planted_factors(tmp_path, monkeypatch):
    harm = np.array([1.0, 0, 0, 0, 0, 0])
    cue = np.array([0, 1.0, 0, 0, 0, 0])
    d = _write(tmp_path, "M3", harm, cue)
    monkeypatch.setattr(fda, "ACT_DIR", d)
    r = fda.audit_stage("M3", layer=1)

    # d_AD == d_H + d_S must hold to numerical precision
    assert r["decomposition_residual_should_be_0"] < 1e-4

    # planted factors are orthogonal -> each main effect isolates its own factor
    sep = r["separation_cohens_d"]
    assert abs(sep["d_H__cuestrong(AB)_vs_cuereduced(CD)"]) < 0.5
    assert abs(sep["d_S__harmful(AC)_vs_benign(BD)"]) < 0.5
    assert sep["d_H__harmful(AC)_vs_benign(BD)"] > 3
    assert sep["d_S__cuestrong(AB)_vs_cuereduced(CD)"] > 3

    # with equal-magnitude planted factors d_AD sits ~45 deg from each
    a = r["alignment_of_preregistered_d_AD"]
    assert 0.6 < a["cos_with_d_H_harmfulness"] < 0.8
    assert 0.6 < a["cos_with_d_S_surface_cue"] < 0.8


def test_dominant_harm_factor_shows_up_as_higher_alignment(tmp_path, monkeypatch):
    # harmfulness planted 3x larger than the cue factor
    d = _write(tmp_path, "M3", np.array([3.0, 0, 0, 0, 0, 0]),
               np.array([0, 1.0, 0, 0, 0, 0]))
    monkeypatch.setattr(fda, "ACT_DIR", d)
    a = fda.audit_stage("M3", layer=1)["alignment_of_preregistered_d_AD"]
    assert a["cos_with_d_H_harmfulness"] > a["cos_with_d_S_surface_cue"]
    assert a["cos_with_d_H_harmfulness"] > 0.9


def test_empty_cell_reports_error(tmp_path, monkeypatch):
    d = tmp_path / "act"; d.mkdir(parents=True, exist_ok=True)
    (d / "M3_metadata.json").write_text(json.dumps(
        [{"quadrant": "A", "split": "direction_estimation"}]), encoding="utf-8")
    np.save(d / "M3_final.npy", np.zeros((1, 3, 6)))
    monkeypatch.setattr(fda, "ACT_DIR", d)
    assert "error" in fda.audit_stage("M3", layer=1)
