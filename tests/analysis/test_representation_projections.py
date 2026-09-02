"""Toy tests for WP-Repr / WP-ReprRobust / WP-Adjunct."""
import json
from pathlib import Path

import numpy as np
import pytest

from src.analysis import matched_pair_representation as mpr
from src.analysis import representation_projections as rp
from src.analysis import representation_robustness as rr
from src.analysis import build_c_source_overt_adjunct as adj
from src.analysis.control_directions import ad_direction

FIX = Path(__file__).resolve().parents[1] / "fixtures"


def _stage(n_layers=4, hidden=6, seed=0):
    rng = np.random.default_rng(seed)
    quads = np.array(["A"] * 6 + ["D"] * 6 + ["B"] * 3 + ["C"] * 3)
    splits = np.array(["direction_estimation"] * 12 + [""] * 6)
    n = len(quads)
    arr = rng.standard_normal((n, n_layers, hidden))
    arr[quads == "A", 1, 0] += 2.0
    arr[quads == "D", 1, 0] -= 2.0
    ids = [f"r{i}" for i in range(n)]
    return arr, quads, splits, ids


# --- WP-Repr ---------------------------------------------------------------
def test_load_stage_final_prefers_final(tmp_path):
    arr, quads, splits, ids = _stage()
    np.save(tmp_path / "M3_final.npy", arr)
    (tmp_path / "M3_metadata.json").write_text(json.dumps(
        [{"quadrant": q, "split": s, "record_id": i}
         for q, s, i in zip(quads, splits, ids)]
    ), encoding="utf-8")
    a, q, s, i, pos = rp.load_stage_final("M3", act_dir=tmp_path)
    assert pos == "final"
    assert a.shape == arr.shape


def test_load_stage_final_falls_back_to_pooled_with_warning(tmp_path, capsys):
    arr, quads, splits, ids = _stage()
    np.save(tmp_path / "M2_pooled.npy", arr)
    (tmp_path / "M2_metadata.json").write_text(json.dumps(
        [{"quadrant": q, "split": s, "record_id": i}
         for q, s, i in zip(quads, splits, ids)]
    ), encoding="utf-8")
    _a, _q, _s, _i, pos = rp.load_stage_final("M2", act_dir=tmp_path)
    assert pos == "pooled_fallback"
    assert "falling back to _pooled" in capsys.readouterr().out


def test_build_stage_projections_has_per_prompt_and_fixed_refs():
    stages = {s: _stage(seed=hash(s) % 50) for s in ("M1", "M2", "M3")}
    payload = rp.build_stage_projections(stages)
    assert payload["canonical_position"] == "final"
    pp = payload["per_prompt_projections"]["M3"]
    assert len(pp) == 18
    assert len(pp[0]["projection_per_layer"]) == 4
    assert set(payload["fixed_reference_quadrant_means"]) == {"M1_reference", "M3_reference"}
    assert "trajectories" in payload


# --- WP-ReprRobust -------------------------------------------------------------
def test_compare_poolings_reports_cosine_and_gaps():
    arr, quads, splits, ids = _stage()
    pooled = arr + np.random.default_rng(1).standard_normal(arr.shape) * 0.01
    cmp = rr.compare_poolings(arr, pooled, quads, splits)
    assert len(cmp["cos_dAD_final_vs_pooled_per_layer"]) == 4
    summ = rr.summarize(cmp)
    assert summ["poolings_agree"] is True  # tiny perturbation -> aligned


def test_compare_poolings_flags_divergent_pooling():
    arr, quads, splits, ids = _stage()
    rng = np.random.default_rng(2)
    pooled = rng.standard_normal(arr.shape)  # totally different
    summ = rr.summarize(rr.compare_poolings(arr, pooled, quads, splits))
    assert summ["poolings_agree"] is False
    assert "limitation" in summ["verdict"]


# --- WP-Adjunct: matched pairs ---------------------------------------------
def test_matched_pair_deltas_and_aggregate():
    rng = np.random.default_rng(0)
    meta, rows = [], []
    n_layers, hidden = 3, 5
    arr = rng.standard_normal((10, n_layers, hidden))
    for p in range(5):
        meta.append({"pair_id": f"P{p}", "judged_prompt_variant": "source_overt"})
        meta.append({"pair_id": f"P{p}", "judged_prompt_variant": "candidate_reduced_cue"})
    direction = np.zeros((n_layers, hidden)); direction[:, 0] = 1.0
    # make reduced-cue variant sit further along axis 0 at layer 1
    for p in range(5):
        arr[2 * p + 1, 1, 0] = arr[2 * p, 1, 0] + 1.0
    deltas = mpr.matched_pair_deltas(arr, meta, direction, layer=1)
    assert len(deltas) == 5
    assert all(d["proj_delta_candidate_minus_overt"] == pytest.approx(1.0) for d in deltas)
    agg = mpr.aggregate_paired(deltas, n_boot=200)
    assert agg["n_pairs"] == 5
    assert agg["mean_proj_delta"] == pytest.approx(1.0)
    assert "not zero" in agg["honest_limit"]


# --- WP-Adjunct: source_overt companion set ------------------------------
def test_build_c_source_overt_adjunct_from_fixture(tmp_path):
    pointer = adj.run(
        latest_path=str(FIX / "benchmark_654.LATEST_BENCHMARK.json"),
        out_path=tmp_path / "adjunct.jsonl",
        pointer_path=tmp_path / "adjunct.LATEST.json",
    )
    assert pointer["n_rows"] == 4  # 4 C rows in the fixture
    rows = [json.loads(l) for l in (tmp_path / "adjunct.jsonl").read_text().splitlines()]
    assert all(r["judged_prompt_variant"] == "source_overt" for r in rows)
    assert all(r["record_id"].endswith("__source_overt") for r in rows)
    # prompt is the OVERT source_prompt, not the candidate
    assert any("bot network" in r["prompt"] for r in rows)


def test_adjunct_is_idempotent(tmp_path):
    kw = dict(latest_path=str(FIX / "benchmark_654.LATEST_BENCHMARK.json"),
              out_path=tmp_path / "a.jsonl", pointer_path=tmp_path / "a.LATEST.json")
    p1 = adj.run(**kw)
    b1 = (tmp_path / "a.jsonl").read_bytes()
    p2 = adj.run(**kw)
    assert (tmp_path / "a.jsonl").read_bytes() == b1
    assert p1["adjunct_sha256"] == p2["adjunct_sha256"]
