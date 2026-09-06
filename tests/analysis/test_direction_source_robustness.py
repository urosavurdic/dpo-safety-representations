"""Toy-data test for direction_source_robustness."""
import json
import numpy as np
import pytest

from src.analysis import direction_source_robustness as dsr


def _write_stage(tmp_path, stage, n_layers=3, hidden=8):
    rows, arrs = [], []
    rng = np.random.default_rng(0)
    # 6 A_est (one source), 6 D_est (3 sources x 2), 4 C
    plan = ([("A", "HarmBench", "direction_estimation")] * 6
            + [("D", "Alpaca", "direction_estimation")] * 2
            + [("D", "Dolly-15k", "direction_estimation")] * 2
            + [("D", "OASST1", "direction_estimation")] * 2
            + [("C", "SR", None)] * 4)
    for i, (q, src, sp) in enumerate(plan):
        rows.append({"record_id": f"r{i}", "prompt": f"p{i}", "quadrant": q,
                     "source": src, "source_dataset": src, "split": sp})
        base = {"A": 1.0, "D": -1.0, "C": 0.3}[q]
        arrs.append(base + 0.01 * rng.standard_normal((n_layers, hidden)))
    d = tmp_path / "results" / "activations"
    d.mkdir(parents=True, exist_ok=True)
    (d / f"{stage}_metadata.json").write_text(json.dumps(rows), encoding="utf-8")
    np.save(d / f"{stage}_final.npy", np.stack(arrs))


def test_headline_cosine_and_zc_present(tmp_path, monkeypatch):
    _write_stage(tmp_path, "M3")
    monkeypatch.setattr(dsr, "ACT_DIR", tmp_path / "results" / "activations")
    monkeypatch.setattr(dsr, "LAYER", 1)
    r = dsr.analyse_stage("M3")
    assert set(r["per_source"]) == {"Alpaca", "Dolly-15k", "OASST1"}
    assert -1.0 <= r["headline_cos_full_vs_OASST1_at_L24"] <= 1.0
    assert r["z_C_full_at_L24"] is not None
    # near-identical A/D clusters per source -> direction barely moves
    assert r["headline_cos_full_vs_OASST1_at_L24"] > 0.9


def test_missing_split_rows_returns_error(tmp_path, monkeypatch):
    d = tmp_path / "results" / "activations"
    d.mkdir(parents=True, exist_ok=True)
    (d / "M3_metadata.json").write_text(json.dumps(
        [{"quadrant": "A", "source": "x", "split": None}]), encoding="utf-8")
    np.save(d / "M3_final.npy", np.zeros((1, 3, 8)))
    monkeypatch.setattr(dsr, "ACT_DIR", d)
    r = dsr.analyse_stage("M3")
    assert "error" in r
