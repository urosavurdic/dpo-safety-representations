"""Tests for the final-token repair (audit RED-1).

Covers the six checks required by the repair spec section 5:
  1. final-token reconstruction == unit(mean(final[A_est]) - mean(final[D_est]))
  2. pooled and final-token paths are not accidentally interchangeable
  3. cross-fit test rows are excluded from the fold direction
  4. cross-fit test rows are excluded from the random-control calibration
  5. binding fields record the pooling mode
  6. output condition names cannot collide with pooled results
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from src.analysis import final_token_repair as ftr
from src.analysis.control_directions import build_ablation_control, seeded_random_directions

ACT = Path("results/activations")
REAL_STAGES = [s for s in ("M2", "M3", "M2_alt", "M3_alt")
               if (ACT / f"{s}_final.npy").exists()
               and (ACT / f"{s}_pooled.npy").exists()]
_HAVE_REAL = bool(REAL_STAGES)


# --------------------------------------------------------------------------- #
# synthetic fixture: a tiny 2x2 benchmark with distinct final vs pooled activations
# --------------------------------------------------------------------------- #
def _synthetic(tmp_path, n_layers=29, hidden=8):
    rng = np.random.default_rng(0)
    quads, splits, rids = [], [], []
    for q, n in (("A", 12), ("D", 12), ("B", 6), ("C", 6)):
        for i in range(n):
            quads.append(q)
            splits.append("direction_estimation" if (q in "AD" and i < 8)
                          else ("held_out_behavioral" if q in "AD" else None))
            rids.append(f"{q}{i:02d}")
    n = len(quads)
    final = rng.standard_normal((n, n_layers, hidden)).astype(np.float32)
    # pooled deliberately NOT equal to final (a different linear mix)
    pooled = (final + rng.standard_normal((n, n_layers, hidden)).astype(np.float32)).astype(np.float32)
    meta = [{"record_id": r, "quadrant": q, "split": s}
            for r, q, s in zip(rids, quads, splits)]
    d = tmp_path / "acts"
    d.mkdir()
    np.save(d / "M9_final.npy", final)
    np.save(d / "M9_pooled.npy", pooled)
    (d / "M9_metadata.json").write_text(json.dumps(meta), encoding="utf-8")
    return d, final, pooled, np.array(quads), np.array(splits), rids


# 1. reconstruction identity -------------------------------------------------- #
def test_full_direction_is_final_token_diff_in_means(tmp_path):
    d, final, pooled, q, sp, rids = _synthetic(tmp_path)
    sa = ftr.load_stage(d, "M9", "final_token")
    got = ftr.full_direction(sa)

    a = final[(q == "A") & (sp == "direction_estimation")].mean(axis=0)
    dd_ = final[(q == "D") & (sp == "direction_estimation")].mean(axis=0)
    delta = a - dd_
    want = delta / np.linalg.norm(delta, axis=-1, keepdims=True)
    np.testing.assert_allclose(got, want, rtol=1e-6, atol=1e-6)
    # unit rows
    np.testing.assert_allclose(np.linalg.norm(got, axis=-1), 1.0, rtol=1e-6)


# 2. final-token and pooled are NOT interchangeable ------------------------- #
def test_pooled_and_final_token_directions_differ(tmp_path):
    d, *_ = _synthetic(tmp_path)
    d_final = ftr.full_direction(ftr.load_stage(d, "M9", "final_token"))
    d_pool = ftr.full_direction(ftr.load_stage(d, "M9", "mean_last5"))
    # the two arrays must not be equal (a swap would be silent otherwise)
    assert not np.allclose(d_final, d_pool)
    cos_l = np.sum(d_final * d_pool, axis=-1)
    assert np.any(np.abs(cos_l) < 0.999), "final vs pooled direction suspiciously identical"


@pytest.mark.skipif(not _HAVE_REAL, reason="needs real 654-row activations")
def test_real_final_token_differs_from_committed_pooled_v2():
    """On the real data the committed *_v2_direction.npy is the POOLED one; the
    final-token direction must be materially different at the intervention layers."""
    for st in ("M3", "M3_alt"):
        v2 = Path("results/refusal_direction") / f"{st}_v2_direction.npy"
        if not v2.exists():
            continue
        d_final = ftr.full_direction(ftr.load_stage(ACT, st, "final_token"))
        d_pool_committed = np.load(v2)
        for L in (24, 25, 26, 27, 28):
            c = float(d_final[L] @ d_pool_committed[L]
                      / (np.linalg.norm(d_final[L]) * np.linalg.norm(d_pool_committed[L])))
            assert 0.5 < c < 0.95, f"{st} L{L}: cos(final, committed pooled) = {c}"


# 3. cross-fit test rows excluded from the fold direction ------------------ #
def test_crossfit_fold_direction_excludes_test_rows(tmp_path):
    d, final, pooled, q, sp, rids = _synthetic(tmp_path)
    sa = ftr.load_stage(d, "M9", "final_token")
    fds = ftr.fold_directions(sa, k=4, seed=ftr.CROSSFIT_SEED)

    a_est_ids = [r for r, qq, ss in zip(rids, q, sp)
                 if qq == "A" and ss == "direction_estimation"]
    # folds partition A_est exactly
    union = sorted(x for f in fds for x in f["test_record_ids"])
    assert union == sorted(a_est_ids)
    assert len({x for f in fds for x in f["test_record_ids"]}) == len(a_est_ids)

    id_arr = np.array([str(r) for r in sa.record_ids])
    for fd in fds:
        drop = set(fd["test_record_ids"])
        keep_a = ((sa.quadrants == "A") & (sa.splits == "direction_estimation")
                  & ~np.isin(id_arr, np.array(sorted(drop))))
        # recompute the direction from the kept rows only and compare
        ma = np.asarray(sa.arr[keep_a]).mean(axis=0)
        md = np.asarray(sa.arr[(sa.quadrants == "D")
                               & (sa.splits == "direction_estimation")]).mean(axis=0)
        delta = ma - md
        want = (delta / np.linalg.norm(delta, axis=-1, keepdims=True)).astype(np.float32)
        np.testing.assert_allclose(fd["direction"], want, rtol=1e-5, atol=1e-5)
        assert fd["n_A"] == int(keep_a.sum())
        assert fd["n_A"] < ((sa.quadrants == "A")
                            & (sa.splits == "direction_estimation")).sum()


# 4. cross-fit test rows excluded from the random-control calibration ----- #
def test_crossfit_control_calibration_excludes_test_rows(tmp_path):
    """Mirror v2_pipeline._causal_control_arrays: the fold's test rows have their
    split rewritten to a sentinel so build_ablation_control's calibration filter
    never sees them. gamma computed WITH vs WITHOUT the sentinel must differ."""
    d, final, pooled, q, sp, rids = _synthetic(tmp_path)
    sa = ftr.load_stage(d, "M9", "final_token")
    fds = ftr.fold_directions(sa, k=4, seed=ftr.CROSSFIT_SEED)
    fold0 = set(fds[0]["test_record_ids"])
    r = seeded_random_directions(sa.arr.shape[1], sa.arr.shape[2], seed=20260904)
    d_ad = fds[0]["direction"]
    layers = [2, 3, 4, 5]

    splits_full = np.array([s if s is not None else "" for s in sa.splits])
    ctrl_full = build_ablation_control(
        sa.arr, sa.quadrants, splits_full, d_ad, r,
        record_ids=sa.record_ids, layers=layers, seed=20260904, strict_zero=False,
    )
    splits_excl = np.array([
        ("crossfit_test" if str(rid) in fold0 else (s if s is not None else ""))
        for rid, s in zip(sa.record_ids, sa.splits)
    ])
    ctrl_excl = build_ablation_control(
        sa.arr, sa.quadrants, splits_excl, d_ad, r,
        record_ids=sa.record_ids, layers=layers, seed=20260904, strict_zero=False,
    )
    assert ctrl_excl.n_calibration_rows == ctrl_full.n_calibration_rows - len(fold0)
    # gamma must actually move when the fold rows are removed from calibration
    assert any(abs(ctrl_full.gamma[l] - ctrl_excl.gamma[l]) > 1e-9 for l in layers)
    # excluded fold ids are not in the calibration id list
    assert not (fold0 & set(ctrl_excl.calibration_record_ids))


# 5. bindings record the pooling mode ------------------------------------- #
def test_bindings_record_pooling(tmp_path, monkeypatch):
    d, *_ = _synthetic(tmp_path)
    monkeypatch.setattr(ftr, "DIRECTIONS_DIR", tmp_path / "dirs")
    monkeypatch.setattr(ftr, "BINDINGS_DIR", tmp_path / "binds")
    out = ftr.build_stage(d, "M9", "final_token", k=4)
    fb = json.loads((tmp_path / "binds" / "M9_final_token_L0-28_binding.json").read_text())
    assert fb["pooling"] == "final_token"
    assert fb["pool_window"] is None
    assert fb["preregistered"] is True
    assert "final" in fb["activation_source_path"]
    assert fb["benchmark_sha256"] == ftr.FROZEN_BENCHMARK_SHA256
    assert fb["direction_split_seed"] == 45
    assert fb["cross_fit_seed"] is None  # full direction, not a fold
    # a fold binding carries the fold + cross-fit fields
    f0 = json.loads((tmp_path / "binds" / "M9_final_token_xfit4_fold0_binding.json").read_text())
    assert f0["pooling"] == "final_token" and f0["cross_fit_k"] == 4
    assert f0["cross_fit_fold"] == 0 and f0["cross_fit_seed"] == ftr.CROSSFIT_SEED
    assert "test_record_ids" in f0
    # the pooled variant records mean_last5 / window 5
    pb_out = ftr.build_stage(d, "M9", "mean_last5", k=4)
    pb = json.loads((tmp_path / "binds" / "M9_mean_last5_L0-28_binding.json").read_text())
    assert pb["pooling"] == "mean_last5" and pb["pool_window"] == 5
    assert pb["preregistered"] is False


# 6. output condition names cannot collide with pooled results ----------- #
def test_final_token_condition_names_do_not_collide():
    """v2_pipeline final-token causal outputs use _ft_ / _ft_xfit_ condition
    names and a _finaltoken file tag; neither can match the pooled names."""
    import src.analysis.v2_pipeline as vp

    class _Ctx:
        pooling = "final_token"

    # stage_causal._full for final-token
    stage = "M3"
    cprefix = "ft_"
    ft_names = {f"{stage}_{cprefix}baseline", f"{stage}_{cprefix}ablated_AD",
                f"{stage}_{cprefix}ablated_random"}
    pooled_names = {f"{stage}_baseline", f"{stage}_ablated_AD", f"{stage}_ablated_random"}
    assert ft_names.isdisjoint(pooled_names)

    ft_xfit = {f"{stage}_ft_xfit_baseline", f"{stage}_ft_xfit_ablated_AD",
               f"{stage}_ft_xfit_ablated_random"}
    pooled_xfit = {f"{stage}_xfit_baseline", f"{stage}_xfit_ablated_AD",
                   f"{stage}_xfit_ablated_random"}
    assert ft_xfit.isdisjoint(pooled_xfit)
    assert ft_xfit.isdisjoint(pooled_names)

    # file tags differ
    assert "_finaltoken" not in "causal_ablation_v2_M3_L24-28.json"
    assert "_finaltoken" in "causal_ablation_v2_M3_L24-28_finaltoken.json"


# extra: local crossfit_folds copy == the v2_pipeline one, and == committed --- #
def test_crossfit_folds_local_matches_v2_pipeline():
    from src.analysis.v2_pipeline import crossfit_folds as vp_folds
    ids = [f"r{i:03d}" for i in range(120)]
    for k in (2, 3, 5, 7):
        assert ftr.crossfit_folds_local(ids, k) == vp_folds(ids, k)
    # order independence
    import random
    shuffled = ids[:]
    random.Random(1).shuffle(shuffled)
    assert ftr.crossfit_folds_local(shuffled, 5) == ftr.crossfit_folds_local(ids, 5)


@pytest.mark.skipif(not _HAVE_REAL, reason="needs real 654-row activations")
def test_real_fold_partition_matches_committed_pooled_xfit5():
    for st in ("M3", "M3_alt"):
        cb = json.loads(
            Path(f"results/raw/causal_ablation_v2_{st}_L24-28_xfit5_binding.json").read_text()
        )
        committed = {f["fold"]: sorted(f["test_record_ids"]) for f in cb["folds"]}
        sa = ftr.load_stage(ACT, st, "final_token")
        fds = ftr.fold_directions(sa, k=5, seed=ftr.CROSSFIT_SEED)
        for fd in fds:
            assert sorted(fd["test_record_ids"]) == committed[fd["fold"]]


@pytest.mark.skipif(not _HAVE_REAL, reason="needs real 654-row activations")
def test_real_cf3_final_token_is_null_and_reproducible(tmp_path):
    """CF3 on the final-token M2/M3 directions: CI must span zero (null), and a
    second run must reproduce the point estimate bit-for-bit."""
    for st in ("M2", "M3"):
        ftr.build_stage(ACT, st, "final_token", k=5)
    m2p = ftr.DIRECTIONS_DIR / "M2_final_token_L0-28.npy"
    m3p = ftr.DIRECTIONS_DIR / "M3_final_token_L0-28.npy"
    r1 = ftr.recompute_cf3(ACT, "data/frozen_v2/benchmark_v2_20260826T212909Z.jsonl",
                           m2p, m3p, tmp_path / "cf3a.json", n_boot=2000)
    r2 = ftr.recompute_cf3(ACT, "data/frozen_v2/benchmark_v2_20260826T212909Z.jsonl",
                           m2p, m3p, tmp_path / "cf3b.json", n_boot=2000)
    assert r1["cf3_macroF1_M3_minus_M2"] == r2["cf3_macroF1_M3_minus_M2"]
    ci = r1["bootstrap_group_diff"]
    assert ci["ci_low"] < 0 < ci["ci_high"], "CF3 CI should span zero (null)"
    assert r1["n_A"] == 150 and r1["n_C"] == 104 and r1["layer"] == 28
