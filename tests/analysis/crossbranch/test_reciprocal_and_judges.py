"""Reciprocal comparison, judge-manifest building, judge analysis, and the
direction-collision guard on delta assembly. Toy data, CPU only.
"""
import json

import numpy as np
import pytest

from src.analysis.crossbranch import analyze_judges as J
from src.analysis.crossbranch import build_judge_manifest as M
from src.analysis.crossbranch import compare_directions as C
from src.analysis.crossbranch import delta as D
from src.analysis.crossbranch.analyze_stage2 import IDENTITY, NORMMATCHED, SHUF_WQ
from src.v2_io import write_json_lf

BENCH, SPLIT = "bench" * 8, "split" * 8


# --------------------------------------------------------------------------
# direction-collision guard
# --------------------------------------------------------------------------


def test_guard_allows_a_fresh_directory(tmp_path):
    D.assert_direction_matches_dir(tmp_path, "A", "B")  # no binding yet


def test_guard_allows_reassembling_the_same_direction(tmp_path):
    write_json_lf(
        tmp_path / "crossbranch_deltas_binding.json",
        {"roles": {"source_branch": "A", "target_branch": "B"}},
    )
    D.assert_direction_matches_dir(tmp_path, "A", "B")


def test_guard_refuses_the_reciprocal_over_an_existing_direction(tmp_path):
    """The real hazard: artifact filenames are direction-neutral, so B->A
    assembly into the A->B directory would overwrite delta_source with Delta_B
    while every condition name and consuming path stays identical."""
    write_json_lf(
        tmp_path / "crossbranch_deltas_binding.json",
        {"roles": {"source_branch": "A", "target_branch": "B"}},
    )
    with pytest.raises(SystemExit, match="direction-neutral"):
        D.assert_direction_matches_dir(tmp_path, "B", "A")


def test_guard_can_be_forced(tmp_path):
    write_json_lf(
        tmp_path / "crossbranch_deltas_binding.json",
        {"roles": {"source_branch": "A", "target_branch": "B"}},
    )
    D.assert_direction_matches_dir(tmp_path, "B", "A", force=True)


def test_guard_ignores_a_binding_without_roles(tmp_path):
    write_json_lf(tmp_path / "crossbranch_deltas_binding.json", {"layer": 24})
    D.assert_direction_matches_dir(tmp_path, "B", "A")


# --------------------------------------------------------------------------
# judge manifest
# --------------------------------------------------------------------------


def test_parse_raw_filename_round_trips_model_and_vector_conditions():
    assert M.parse_raw_filename("crossbranch_AtoB_baseline_target_coefna") == (
        "AtoB", "baseline_target", "na"
    )
    assert M.parse_raw_filename("crossbranch_BtoA_xfer_delta_source_identity_coef0.5") == (
        "BtoA", "xfer_delta_source_identity", "0.5"
    )


def test_arm_key_round_trips():
    key = M.arm_key("AtoB", "xfer_delta_source_identity", "1")
    assert M.parse_arm_key(key) == ("AtoB", "xfer_delta_source_identity", "1")


def _raw_rows(n=6, quads=("A", "B", "C", "D", "A", "C")):
    return [
        {
            "record_id": f"r{i}", "prompt": f"p{i}", "quadrant": quads[i],
            "response": "text", "stage": "own_delta_target",
            "condition": "own_delta_target", "model_stage": "M2_alt",
            "benchmark_sha256": BENCH, "split_manifest_sha256": SPLIT,
        }
        for i in range(n)
    ]


def _write_raw(raw_dir, name, rows):
    raw_dir.mkdir(parents=True, exist_ok=True)
    write_json_lf(raw_dir / f"{name}.json", rows)
    write_json_lf(
        raw_dir / f"{name}_binding.json",
        {"benchmark_sha256": BENCH, "split_manifest_sha256": SPLIT},
    )


def test_filter_rows_keeps_only_requested_quadrants_and_qualifies_the_key():
    rows = _raw_rows()
    kept = M.filter_rows(rows, ("A", "C"), "AtoB", "own_delta_target", "1")
    assert [r["quadrant"] for r in kept] == ["A", "C", "A", "C"]
    assert all(r["stage"] == "AtoB|own_delta_target|coef1" for r in kept)
    assert all(r["condition"] == r["stage"] for r in kept)
    # originals preserved, not destroyed
    assert all(r["original_stage"] == "own_delta_target" for r in kept)


def test_filter_rows_does_not_mutate_the_input():
    rows = _raw_rows()
    M.filter_rows(rows, ("A",), "AtoB", "own_delta_target", "1")
    assert rows[0]["stage"] == "own_delta_target"


def test_build_writes_filtered_copies_bindings_and_a_manifest(tmp_path):
    raw = tmp_path / "raw"
    _write_raw(raw, "crossbranch_AtoB_baseline_target_coefna", _raw_rows())
    _write_raw(raw, "crossbranch_AtoB_xfer_delta_source_identity_coef1", _raw_rows())

    out = tmp_path / "judge_inputs"
    man_path = tmp_path / "m.json"
    manifest = M.build(raw, out, man_path, ("A", "C"), allow_unbound=True)

    assert manifest["kind"] == "consolidated_response_manifest"
    assert len(manifest["entries"]) == 2
    for entry in manifest["entries"]:
        assert Path_exists(entry["response_file"])
        assert Path_exists(entry["binding_file"])
    keys = {a["arm_key"] for a in manifest["crossbranch"]["arms"]}
    assert keys == {
        "AtoB|baseline_target|coefna",
        "AtoB|xfer_delta_source_identity|coef1",
    }
    assert all(a["n_rows"] == 4 for a in manifest["crossbranch"]["arms"])


def Path_exists(p):
    from pathlib import Path
    return Path(p).exists()


def test_build_can_restrict_to_one_direction(tmp_path):
    raw = tmp_path / "raw"
    _write_raw(raw, "crossbranch_AtoB_baseline_target_coefna", _raw_rows())
    _write_raw(raw, "crossbranch_BtoA_baseline_target_coefna", _raw_rows())
    manifest = M.build(
        raw, tmp_path / "o", tmp_path / "m.json", ("A", "C"),
        directions=["BtoA"], allow_unbound=True,
    )
    assert [a["direction"] for a in manifest["crossbranch"]["arms"]] == ["BtoA"]


def test_build_raises_when_nothing_matches(tmp_path):
    raw = tmp_path / "raw"
    _write_raw(raw, "crossbranch_AtoB_baseline_target_coefna", _raw_rows())
    with pytest.raises(SystemExit, match="matched"):
        M.build(raw, tmp_path / "o", tmp_path / "m.json", ("D",),
                directions=["ZtoZ"], allow_unbound=True)


# --------------------------------------------------------------------------
# judge analysis
# --------------------------------------------------------------------------


def _judge_rec(tag, condition, coef, rid, quadrant, sr, wg_refusal, wg_harm):
    return {
        "record_id": rid,
        "stage": M.arm_key(tag, condition, coef),
        "condition": M.arm_key(tag, condition, coef),
        "quadrant": quadrant,
        "strong_reject": {"score": sr, "malformed": False, "judge_status": "scored"},
        "wildguard": {
            "response_refusal": wg_refusal, "response_harm": wg_harm,
            "prompt_harm": "yes", "malformed": False, "judge_status": "scored",
        },
    }


def _judge_output(n=8):
    recs = []
    for i in range(n):
        rid = f"r{i}"
        # baseline = maximally harmful, reference = maximally refusing
        recs.append(_judge_rec("AtoB", "baseline_target", "na", rid, "C", 5.0, "no", "yes"))
        recs.append(_judge_rec("AtoB", "reference_target", "na", rid, "C", 1.0, "yes", "no"))
        recs.append(_judge_rec("AtoB", IDENTITY, "1", rid, "C", 1.0, "yes", "no"))
        recs.append(_judge_rec("AtoB", SHUF_WQ, "1", rid, "C", 3.0, "no", "yes"))
        recs.append(_judge_rec("AtoB", NORMMATCHED, "1", rid, "C", 5.0, "no", "yes"))
    return {"records": recs, "judge_status": {"strong_reject": "scored"}}


def test_index_records_groups_by_direction_coef_condition_quadrant():
    idx = J.index_records(_judge_output(3)["records"])
    assert ("AtoB", "na") in idx and ("AtoB", "1") in idx
    # the coefficient bucket holds its three vector arms...
    assert {IDENTITY, SHUF_WQ, NORMMATCHED} <= set(idx[("AtoB", "1")])
    # ...plus the two coefficient-free model anchors, fanned in so every
    # coefficient's comparison has a baseline to shift against
    assert {"baseline_target", "reference_target"} <= set(idx[("AtoB", "1")])


def test_model_anchors_are_shared_not_copied_per_coefficient():
    recs = _judge_output(4)["records"]
    recs += [
        _judge_rec("AtoB", IDENTITY, "2", f"r{i}", "C", 2.0, "no", "yes")
        for i in range(4)
    ]
    idx = J.index_records(recs)
    assert idx[("AtoB", "1")]["baseline_target"] is idx[("AtoB", "2")]["baseline_target"]


def test_index_records_keeps_directions_apart():
    recs = _judge_output(3)["records"]
    recs += [_judge_rec("BtoA", "baseline_target", "na", "r0", "C", 4.0, "no", "yes")]
    idx = J.index_records(recs)
    assert "baseline_target" in idx[("BtoA", "na")]
    # BtoA anchors must not leak into AtoB's buckets
    assert idx[("AtoB", "na")]["baseline_target"]["C"]["r0"]["strong_reject"]["score"] == 5.0


def test_index_records_skips_rows_without_a_qualified_key():
    recs = _judge_output(2)["records"] + [
        {"record_id": "x", "stage": "own_delta_target", "quadrant": "C"}
    ]
    idx = J.index_records(recs)
    assert all("|" not in c for by in idx.values() for c in by)


def test_judge_analysis_recovers_the_expected_shifts():
    out = J.analyze_judges(_judge_output(10), b=200)
    # model conditions carry coef "na" and are bucketed into every coefficient
    key = "AtoB|coef1"
    assert key in out["by_direction_and_coef"]
    blk = out["by_direction_and_coef"][key]["per_quadrant"]["C"]["strongreject"]
    assert blk["baseline_mean"]["point"] == pytest.approx(5.0)
    assert blk["reference_mean"]["point"] == pytest.approx(1.0)
    # identity matches the reference -> shift of -4 from a baseline of 5
    assert blk["arms"][IDENTITY]["shift"]["point"] == pytest.approx(-4.0)
    assert blk["arms"][SHUF_WQ]["shift"]["point"] == pytest.approx(-2.0)
    assert blk["arms"][NORMMATCHED]["shift"]["point"] == pytest.approx(0.0)


def test_judge_contrast_is_the_paired_difference_of_shifts():
    out = J.analyze_judges(_judge_output(10), b=200)
    blk = out["by_direction_and_coef"]["AtoB|coef1"]["per_quadrant"]["C"]["strongreject"]
    name = f"{IDENTITY}__minus__{SHUF_WQ}"
    expected = (
        blk["arms"][IDENTITY]["shift"]["point"] - blk["arms"][SHUF_WQ]["shift"]["point"]
    )
    assert blk["contrasts"][name]["point"] == pytest.approx(expected)
    assert blk["contrasts"][name]["ci_high"] < 0  # identity reliably lower


def test_wildguard_rates_are_fractions_and_shift_correctly():
    out = J.analyze_judges(_judge_output(10), b=200)
    blk = out["by_direction_and_coef"]["AtoB|coef1"]["per_quadrant"]["C"]
    harm = blk["wildguard_harm"]
    assert harm["baseline_mean"]["point"] == pytest.approx(1.0)
    assert harm["reference_mean"]["point"] == pytest.approx(0.0)
    assert harm["arms"][IDENTITY]["shift"]["point"] == pytest.approx(-1.0)
    refusal = blk["wildguard_refusal"]
    assert refusal["arms"][IDENTITY]["shift"]["point"] == pytest.approx(1.0)


def test_unscored_rows_are_excluded_and_coverage_is_reported():
    data = _judge_output(10)
    dropped = 0
    for rec in data["records"]:
        if rec["stage"].endswith("|coef1") and IDENTITY in rec["stage"] and dropped < 4:
            rec["strong_reject"] = {"score": None, "malformed": True,
                                    "judge_status": "scored"}
            dropped += 1
    out = J.analyze_judges(data, b=100)
    cov = out["by_direction_and_coef"]["AtoB|coef1"]["per_quadrant"]["C"]["strongreject"]["coverage"]
    assert cov["scored"] == 6 and cov["shared_rows"] == 10
    # wildguard is untouched, so its coverage stays complete
    wcov = out["by_direction_and_coef"]["AtoB|coef1"]["per_quadrant"]["C"]["wildguard_harm"]["coverage"]
    assert wcov["scored"] == 10


def test_gap_closed_is_suppressed_when_the_denominator_is_tiny():
    recs = []
    for i in range(8):
        rid = f"r{i}"
        recs.append(_judge_rec("AtoB", "baseline_target", "na", rid, "C", 3.0, "no", "yes"))
        recs.append(_judge_rec("AtoB", "reference_target", "na", rid, "C", 3.0, "no", "yes"))
        recs.append(_judge_rec("AtoB", IDENTITY, "1", rid, "C", 1.0, "yes", "no"))
    out = J.analyze_judges({"records": recs}, b=100)
    blk = out["by_direction_and_coef"]["AtoB|coef1"]["per_quadrant"]["C"]["strongreject"]
    assert blk["gap_closed_reported"] is False
    assert blk["arms"][IDENTITY]["gap_closed"] is None


def test_judge_analysis_is_deterministic():
    a = J.analyze_judges(_judge_output(8), b=150)
    b = J.analyze_judges(_judge_output(8), b=150)
    assert a == b


def test_judge_analysis_raises_without_qualified_keys():
    with pytest.raises(RuntimeError, match="fully-qualified arm key"):
        J.analyze_judges({"records": [{"stage": "baseline_target", "quadrant": "C",
                                       "record_id": "r0"}]})


# --------------------------------------------------------------------------
# reciprocal comparison
# --------------------------------------------------------------------------


def _stat(point, lo, hi):
    return {"point": point, "ci_low": lo, "ci_high": hi}


def _analysis(tag, *, moves, structured, prompt_cond, n=104, width=0.02):
    """Build a minimal Stage-2 analysis block with controllable verdicts."""
    def ci(neg):
        return _stat(-0.1, -0.15, -0.05) if neg else _stat(0.0, -width / 2, width / 2)
    return {
        "direction_tag": tag,
        "coef": 1.0,
        "per_quadrant": {
            "C": {
                "n": n,
                "tv_baseline_to_reference": 0.28,
                "arms": {IDENTITY: {"dtv": ci(moves)}},
                "contrasts": {
                    f"{IDENTITY}__minus__{NORMMATCHED}": {"dtv_diff": ci(structured)},
                    f"{IDENTITY}__minus__{SHUF_WQ}": {"dtv_diff": ci(prompt_cond)},
                },
            }
        },
    }


def test_verdict_yes_no_and_underpowered():
    assert C.verdict(_stat(-0.1, -0.2, -0.05))["verdict"] == "yes"
    assert C.verdict(_stat(0.1, 0.05, 0.2))["verdict"] == "no_opposite"
    assert C.verdict(_stat(0.0, -0.005, 0.005))["verdict"] == "no"
    assert C.verdict(_stat(-0.05, -0.30, 0.20))["verdict"] == "underpowered"
    assert C.verdict(None)["verdict"] == "absent"


def test_underpowered_is_distinct_from_a_genuine_null():
    """A 30-row quadrant must never read as evidence of absence."""
    wide = C.verdict(_stat(-0.2, -0.30, 0.00))
    tight = C.verdict(_stat(0.0, -0.01, 0.01))
    assert wide["verdict"] == "underpowered"
    assert tight["verdict"] == "no"


def test_compare_reports_agreement_when_both_directions_match():
    a = _analysis("AtoB", moves=True, structured=True, prompt_cond=False)
    b = _analysis("BtoA", moves=True, structured=True, prompt_cond=False)
    out = C.compare(a, b)
    ag = out["per_quadrant"]["C"]["agreement"]
    assert ag["moves"]["agree"] and ag["moves"]["both_conclusive"]
    assert ag["structured"]["agree"]
    assert ag["prompt_conditioned"]["agree"]
    assert ag["prompt_conditioned"]["AtoB"] == "no"


def test_compare_flags_disagreement_between_directions():
    a = _analysis("AtoB", moves=True, structured=True, prompt_cond=False)
    b = _analysis("BtoA", moves=False, structured=False, prompt_cond=False)
    out = C.compare(a, b)
    ag = out["per_quadrant"]["C"]["agreement"]
    assert not ag["moves"]["agree"]
    assert ag["moves"]["AtoB"] == "yes" and ag["moves"]["BtoA"] == "no"


def test_compare_only_uses_quadrants_present_in_both():
    a = _analysis("AtoB", moves=True, structured=True, prompt_cond=False)
    b = _analysis("BtoA", moves=True, structured=True, prompt_cond=False)
    b["per_quadrant"]["A"] = a["per_quadrant"]["C"]
    out = C.compare(a, b)
    assert list(out["per_quadrant"]) == ["C"]


def test_compare_records_both_ns_and_is_printable(capsys):
    a = _analysis("AtoB", moves=True, structured=True, prompt_cond=False, n=104)
    b = _analysis("BtoA", moves=True, structured=True, prompt_cond=False, n=99)
    out = C.compare(a, b)
    assert out["per_quadrant"]["C"]["n"] == {"AtoB": 104, "BtoA": 99}
    C.print_comparison(out)
    printed = capsys.readouterr().out
    assert "prompt_conditioned" in printed and "AtoB" in printed


def test_compare_notes_warn_against_a_combined_test():
    out = C.compare(
        _analysis("AtoB", moves=True, structured=True, prompt_cond=False),
        _analysis("BtoA", moves=True, structured=True, prompt_cond=False),
    )
    assert any("not a paired contrast" in n for n in out["notes"])


def test_missing_contrast_reports_absent_rather_than_crashing():
    a = _analysis("AtoB", moves=True, structured=True, prompt_cond=False)
    del a["per_quadrant"]["C"]["contrasts"][f"{IDENTITY}__minus__{SHUF_WQ}"]
    b = _analysis("BtoA", moves=True, structured=True, prompt_cond=False)
    out = C.compare(a, b)
    assert out["per_quadrant"]["C"]["AtoB"]["prompt_conditioned"]["verdict"] == "absent"


# --------------------------------------------------------------------------
# shard unit-key collision between directions
# --------------------------------------------------------------------------


def test_shard_unit_key_includes_the_direction_tag():
    """Regression: condition names are direction-neutral, so keying shards on
    (condition, coefficient) alone lets a B->A unit resume A->B's completed
    shards -- skipping generation and merging the wrong branch's rows out under
    the reciprocal filename. The worker builds the key from the direction tag
    plus the condition; assert the two directions cannot collide."""
    from src.analysis.v2_shards import ShardStore

    a = ShardStore.unit_key("AtoB_own_delta_target", "coef1.0")
    b = ShardStore.unit_key("BtoA_own_delta_target", "coef1.0")
    assert a != b
    assert "AtoB" in a and "BtoA" in b


def test_worker_builds_the_shard_key_from_tag_and_condition():
    """Pin the exact construction worker.run_unit uses, so a refactor that
    drops the tag fails here rather than silently in a reciprocal run."""
    import inspect

    from src.analysis.crossbranch import worker

    src = inspect.getsource(worker.run_unit)
    assert 'f"{tag}_{args.condition}"' in src, (
        "run_unit must key shards on the direction tag + condition"
    )
