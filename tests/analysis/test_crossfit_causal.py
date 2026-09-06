"""Tests for the cross-fitted causal ablation (v2_pipeline --cross-fit K)
and its CF2 analysis block.

Three things are worth protecting here, and each has already been a live bug
class in this project:

1. **The fold partition** must be a clean, deterministic cover. A silent
   overlap double-counts a prompt in the pooled out-of-fold estimate; a
   silent gap shrinks n below the reported figure. Neither raises.
2. **The condition names must not collide** with the ordinary causal
   conditions. Cross-fitted and _fullAD rows share record_ids, and
   `_cf2_block` keys triples by record_id - identical condition names would
   let one silently overwrite the other inside the merged judge output.
3. **The judge's file scan must actually see the tagged causal files.** The
   glob was anchored to `_L24-28.json`, which excluded every `_fullAD` /
   `_xfit` file - producing a judged file that looks complete while the
   sensitivity blocks stay at n=0, with no error anywhere.

No torch/GPU: only pure helpers and the analysis layer are exercised, the
same split between "testable pure logic" and "GPU-only orchestration" that
test_eval_steering_v2.py / test_eval_causal_ablation.py already use.
"""
from __future__ import annotations

import fnmatch

import pytest

from src.analysis import behavioral_judges as bj
from src.analysis import confirmatory_behavioral_endpoints as cbe
from src.analysis import v2_pipeline as vp

from tests.analysis.test_confirmatory_behavioral_endpoints import _rec


# --- fold partition ------------------------------------------------------------
def test_folds_are_disjoint_and_cover_every_row():
    ids = [f"A{i:03d}" for i in range(120)]
    folds = vp.crossfit_folds(ids, 5)

    assert len(folds) == 5
    flat = [r for f in folds for r in f]
    assert len(flat) == len(set(flat)), "folds overlap - a prompt would be double-counted"
    assert set(flat) == set(ids), "folds do not cover every tested row"
    assert [len(f) for f in folds] == [24] * 5


def test_fold_sizes_differ_by_at_most_one_when_k_does_not_divide_n():
    folds = vp.crossfit_folds([f"A{i}" for i in range(121)], 5)
    sizes = sorted(len(f) for f in folds)
    assert sizes[-1] - sizes[0] <= 1
    assert sum(sizes) == 121


def test_partition_depends_on_the_id_set_not_the_input_order():
    """Two machines loading rows in different orders must get the same folds,
    or the pooled out-of-fold estimate is not reproducible."""
    ids = [f"A{i:03d}" for i in range(60)]
    assert vp.crossfit_folds(ids, 4) == vp.crossfit_folds(list(reversed(ids)), 4)
    assert vp.crossfit_folds(ids, 4) == vp.crossfit_folds(sorted(ids, key=hash), 4)


def test_seed_changes_the_partition():
    ids = [f"A{i:03d}" for i in range(60)]
    assert vp.crossfit_folds(ids, 4, seed=1) != vp.crossfit_folds(ids, 4, seed=2)


def test_duplicate_and_none_ids_are_collapsed_not_silently_double_counted():
    folds = vp.crossfit_folds(["a", "a", "b", "c", None], 2)
    flat = sorted(r for f in folds for r in f)
    assert flat == ["a", "b", "c"]


@pytest.mark.parametrize("ids,k", [([f"A{i}" for i in range(10)], 1),
                                   ([f"A{i}" for i in range(3)], 5)])
def test_degenerate_k_raises_rather_than_producing_empty_folds(ids, k):
    with pytest.raises(ValueError):
        vp.crossfit_folds(ids, k)


# --- condition naming ----------------------------------------------------------
def test_crossfit_conditions_never_collide_with_ordinary_ones():
    for stage in ("M3", "M3_direct", "M3_alt", "M3_direct_alt"):
        ordinary = set(cbe._cf2_conditions_for_stage(stage))
        crossfit = set(cbe._cf2_crossfit_conditions_for_stage(stage))
        assert not (ordinary & crossfit), (
            f"{stage}: cross-fitted and ordinary conditions share a name; the "
            f"same record_id would overwrite itself in _cf2_block"
        )


def test_crossfit_conditions_stay_in_the_judge_confirmatory_scope():
    """A cross-fitted quadrant-A row must still be judged, or CF2's
    cross_fitted block is computed from regex-only rows (n=0)."""
    for cond in cbe._cf2_crossfit_conditions_for_stage("M3"):
        rec = {"quadrant": "A", "model_stage": "M3", "stage": cond}
        assert bj._row_in_scope(rec, "confirmatory"), cond


# --- the judge's file scan (regression: the _L24-28 anchor) --------------------
@pytest.mark.parametrize("name", [
    "raw/causal_ablation_v2_M3_L24-28.json",
    "raw/causal_ablation_v2_M3_L24-28_fullAD.json",
    "raw/causal_ablation_v2_M3_L24-28_xfit5.json",
    "raw/causal_ablation_v2_M3_direct_alt_L24-28_fullAD.json",
])
def test_response_globs_match_every_tagged_causal_variant(name):
    assert any(fnmatch.fnmatch(name, pat) for pat in bj.RESPONSE_GLOBS), (
        f"{name} is not picked up by RESPONSE_GLOBS - it would be silently "
        f"omitted from the consolidated judge manifest"
    )


def test_response_globs_still_exclude_binding_sidecars_by_name():
    """The glob is deliberately broad; the loop's explicit endswith check is
    what keeps sidecars out. Both halves have to hold."""
    sidecar = "raw/causal_ablation_v2_M3_L24-28_fullAD_binding.json"
    assert any(fnmatch.fnmatch(sidecar, pat) for pat in bj.RESPONSE_GLOBS)
    assert sidecar.endswith("_binding.json")


def test_consolidated_manifest_picks_up_a_fullAD_file(tmp_path):
    """End to end through the real scanner, not just the pattern."""
    raw = tmp_path / "raw"
    raw.mkdir()
    for stem in ("causal_ablation_v2_M3_L24-28",
                 "causal_ablation_v2_M3_L24-28_fullAD",
                 "causal_ablation_v2_M3_L24-28_xfit5"):
        (raw / f"{stem}.json").write_text("[]", encoding="utf-8")
        (raw / f"{stem}_binding.json").write_text(
            '{"benchmark_sha256": "s", "split_manifest_sha256": "t"}', encoding="utf-8")

    manifest = bj.build_consolidated_from_results(tmp_path, tmp_path / "m.json")
    found = {e["response_file"].rsplit("/", 1)[-1] for e in manifest["entries"]}
    assert found == {"causal_ablation_v2_M3_L24-28.json",
                     "causal_ablation_v2_M3_L24-28_fullAD.json",
                     "causal_ablation_v2_M3_L24-28_xfit5.json"}


# --- the CF2 cross-fitted block ------------------------------------------------
def _xfit_triple(rid, stage, base, ad, rand):
    b, a, r = cbe._cf2_crossfit_conditions_for_stage(stage)
    return [_rec(rid, stage, "A", base, stage=b),
            _rec(rid, stage, "A", ad, stage=a),
            _rec(rid, stage, "A", rand, stage=r)]


def test_cross_fitted_block_uses_estimation_rows_and_xfit_conditions():
    recs = []
    for i, (ad, rand) in enumerate([(0.5, 0.2), (0.6, 0.3), (0.7, 0.4)]):
        recs += _xfit_triple(f"a{i}", "M3", 0.1, ad, rand)
    id_to_split = {f"a{i}": "direction_estimation" for i in range(3)}

    block = cbe._cf2_block(
        recs, id_to_split, stage="M3", population="crossfit",
        conditions=cbe._cf2_crossfit_conditions_for_stage("M3"),
    )
    assert block["n_effective_triples"] == 3
    # contribution = SR_ablated_AD - SR_ablated_random, all three are +0.3
    assert block["cf2"] == pytest.approx(0.3)
    assert "out-of-fold" in block["population"]


def test_cross_fitted_block_ignores_held_out_rows():
    """Cross-fitting only ever generates estimation-split rows; a held-out row
    carrying an _xfit condition would mean the two runs got mixed."""
    recs = _xfit_triple("a1", "M3", 0.1, 0.5, 0.2) + _xfit_triple("h1", "M3", 0.1, 0.9, 0.1)
    id_to_split = {"a1": "direction_estimation", "h1": "held_out_behavioral"}

    block = cbe._cf2_block(
        recs, id_to_split, stage="M3", population="crossfit",
        conditions=cbe._cf2_crossfit_conditions_for_stage("M3"),
    )
    assert block["n_effective_triples"] == 1


def test_ordinary_and_cross_fitted_rows_for_one_prompt_do_not_contaminate():
    """The whole point of the separate condition names: the same record_id can
    appear in both the _fullAD file and the _xfit file with different
    responses, and each block must read only its own."""
    recs = _xfit_triple("a1", "M3", 0.1, 0.9, 0.1)          # cross-fitted: +0.8
    b, a, r = cbe._cf2_conditions_for_stage("M3")
    recs += [_rec("a1", "M3", "A", 0.1, stage=b),           # ordinary: +0.1
             _rec("a1", "M3", "A", 0.3, stage=a),
             _rec("a1", "M3", "A", 0.2, stage=r)]
    id_to_split = {"a1": "direction_estimation"}

    xfit = cbe._cf2_block(recs, id_to_split, stage="M3", population="crossfit",
                          conditions=cbe._cf2_crossfit_conditions_for_stage("M3"))
    ordinary = cbe._cf2_block(recs, id_to_split, stage="M3", population="estimation")

    assert xfit["cf2"] == pytest.approx(0.8)
    assert ordinary["cf2"] == pytest.approx(0.1)


def test_compute_cf2_for_stage_reports_cross_fitted_as_zero_when_not_run():
    """A missing cross-fit run must read as 'not run', not as a null result."""
    recs = []
    b, a, r = cbe._cf2_conditions_for_stage("M3")
    for cond, sr in ((b, 0.1), (a, 0.4), (r, 0.2)):
        recs.append(_rec("h1", "M3", "A", sr, stage=cond))

    out = cbe.compute_cf2_for_stage(recs, {"h1": "held_out_behavioral"}, "M3")
    assert out["cross_fitted"]["n_effective_triples"] == 0
    assert "NEVER 'independent n=120'" in out["cross_fitted_note"]
    assert "not a null result" in out["cross_fitted_note"].lower()
    # and the ordinary held-out block is unaffected by the absent cross-fit
    assert out["primary"]["n_effective_triples"] == 1


def test_population_must_be_a_known_key():
    with pytest.raises(ValueError):
        cbe._cf2_block([], {}, stage="M3", population="nonsense")


# --- the fold actually reaches the random control's calibration ----------------
class _FakeCtx:
    """Minimal stand-in: _causal_control_arrays only needs load_bound_activation
    to hand back (final, pooled, metadata)."""
    force = False


def _fake_activations(n_layers=29, hidden=16, n_a=8, n_d=8):
    import numpy as np
    rng = np.random.default_rng(0)
    meta = ([{"record_id": f"a{i}", "quadrant": "A", "split": "direction_estimation"}
             for i in range(n_a)]
            + [{"record_id": f"d{i}", "quadrant": "D", "split": "direction_estimation"}
               for i in range(n_d)]
            + [{"record_id": "b0", "quadrant": "B", "split": None}])
    final = rng.normal(size=(len(meta), n_layers, hidden))
    return final, final, meta


def test_exclude_ids_removes_the_fold_from_the_random_control_calibration(monkeypatch):
    """The direction and the matched random control must be built from the SAME
    reduced row set. If only the direction folds and the control does not, the
    fold's rows are back in the intervention's own magnitude calibration and
    the cross-fit is silently a no-op for the control arm."""
    import numpy as np

    final, pooled, meta = _fake_activations()
    monkeypatch.setattr(vp, "load_bound_activation",
                        lambda ctx, stage: (final, pooled, meta))
    direction = np.zeros((29, 16))
    direction[24:29] = 1.0 / np.sqrt(16)

    _, _, full = vp._causal_control_arrays(_FakeCtx(), "M3", direction)
    _, _, folded = vp._causal_control_arrays(
        _FakeCtx(), "M3", direction, exclude_ids=["a0", "a1"])

    assert full.n_calibration_rows == 16
    assert folded.n_calibration_rows == 14
    assert {"a0", "a1"} <= set(full.calibration_record_ids)
    assert not ({"a0", "a1"} & set(folded.calibration_record_ids)), (
        "the folded-out rows are still calibrating the random control"
    )


def test_exclude_ids_accepts_non_string_record_ids(monkeypatch):
    """crossfit_folds normalises ids to str; metadata may carry ints. A silent
    type mismatch would leave the fold in the calibration set and fail nothing."""
    import numpy as np

    final, pooled, meta = _fake_activations()
    for i, row in enumerate(meta):
        row["record_id"] = i           # ints, not strings
    monkeypatch.setattr(vp, "load_bound_activation",
                        lambda ctx, stage: (final, pooled, meta))
    direction = np.zeros((29, 16))
    direction[24:29] = 1.0 / np.sqrt(16)

    _, _, folded = vp._causal_control_arrays(
        _FakeCtx(), "M3", direction, exclude_ids=["0", "1"])
    assert folded.n_calibration_rows == 14


def test_no_exclude_ids_leaves_the_control_exactly_as_before(monkeypatch):
    """The ordinary (non-cross-fit) path must be untouched by this change."""
    import numpy as np

    final, pooled, meta = _fake_activations()
    monkeypatch.setattr(vp, "load_bound_activation",
                        lambda ctx, stage: (final, pooled, meta))
    direction = np.zeros((29, 16))
    direction[24:29] = 1.0 / np.sqrt(16)

    a, _, ctrl_a = vp._causal_control_arrays(_FakeCtx(), "M3", direction)
    b, _, ctrl_b = vp._causal_control_arrays(_FakeCtx(), "M3", direction, exclude_ids=None)
    assert np.array_equal(a, b)
    assert ctrl_a.calibration_record_ids == ctrl_b.calibration_record_ids
    assert ctrl_a.n_calibration_rows == 16
