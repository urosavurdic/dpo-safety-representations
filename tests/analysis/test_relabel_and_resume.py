"""Tests for the condition-label repair and resumable judging.

Both exist because of one bug: `stage_causal` passed its SHARD-STORE unit name
through as the generated row's condition, relabelling only `ablated_AD`. A
`--all-ad-sensitivity` run therefore wrote `M3_baseline_fullAD` and
`M3_ablated_random_fullAD`, which (a) fell outside the judge's confirmatory
scope so were never scored and (b) never matched CF2's condition tuple, so
every full-A triple was dropped as incomplete. `estimation_split_only` sat at
n=0 and `full_A_sensitivity` silently equalled the held-out block - no error,
plausible-looking output.
"""
from __future__ import annotations

import json

import pytest

from src.analysis import behavioral_judges as bj
from src.analysis import confirmatory_behavioral_endpoints as cbe
from src.analysis import relabel_causal_conditions as rc


# --- label repair --------------------------------------------------------------
@pytest.mark.parametrize("label,stage,expected", [
    ("M3_baseline_fullAD", "M3", "M3_baseline"),
    ("M3_ablated_random_fullAD", "M3", "M3_ablated_random"),
    ("M3_ablated", "M3", "M3_ablated_AD"),                 # legacy shard spelling
    ("M3_ablated_fullAD", "M3", "M3_ablated_AD"),
    ("M3_direct_alt_baseline_fullAD", "M3_direct_alt", "M3_direct_alt_baseline"),
    ("M3_baseline_dirfrom_M2", "M3", "M3_baseline"),
    # already canonical -> untouched (idempotent)
    ("M3_baseline", "M3", "M3_baseline"),
    ("M3_ablated_AD", "M3", "M3_ablated_AD"),
    ("M3_ablated_random", "M3", "M3_ablated_random"),
])
def test_canonical_condition(label, stage, expected):
    assert rc.canonical_condition(label, stage) == expected


@pytest.mark.parametrize("label", ["M3_xfit_baseline", "M3_xfit_ablated_AD",
                                   "M3_xfit_ablated_random"])
def test_crossfit_labels_are_never_collapsed(label):
    """The _xfit infix is a real condition distinction, not a shard tag.
    Collapsing it would let cross-fitted rows overwrite ordinary ones for the
    same record_id."""
    assert rc.canonical_condition(label, "M3") == label


def test_repaired_labels_are_exactly_what_cf2_expects():
    stage = "M3"
    repaired = {rc.canonical_condition(l, stage) for l in
                (f"{stage}_baseline_fullAD", f"{stage}_ablated_AD",
                 f"{stage}_ablated_random_fullAD")}
    assert repaired == set(cbe._cf2_conditions_for_stage(stage))


def _in_scope(label):
    return bj._row_in_scope(
        {"quadrant": "A", "model_stage": "M3", "stage": label}, "confirmatory")


def test_repaired_labels_are_all_in_the_judge_confirmatory_scope():
    """Only the baseline arm actually fell out of scope - "ablated_random" is a
    substring match, so that arm WAS judged. That asymmetry is why the failure
    looked like "incomplete triples" rather than "nothing was judged"."""
    assert not _in_scope("M3_baseline_fullAD"), "the arm that was never scored"
    assert _in_scope("M3_ablated_random_fullAD"), "substring match kept this one in"

    for broken in ("M3_baseline_fullAD", "M3_ablated_random_fullAD"):
        assert _in_scope(rc.canonical_condition(broken, "M3"))


def test_baseline_fullAD_was_genuinely_out_of_scope():
    """The specific silent failure: the baseline arm was never judged."""
    assert not bj._row_in_scope(
        {"quadrant": "A", "model_stage": "M3", "stage": "M3_baseline_fullAD"},
        "confirmatory")


@pytest.mark.parametrize("name,stage", [
    ("causal_ablation_v2_M3_L24-28_fullAD.json", "M3"),
    ("causal_ablation_v2_M3_direct_L24-28_fullAD.json", "M3_direct"),
    ("causal_ablation_v2_M3_direct_alt_L24-28_xfit5.json", "M3_direct_alt"),
])
def test_stage_of(name, stage):
    from pathlib import Path
    assert rc.stage_of(Path(name)) == stage


def _fullad_file(tmp_path, stage="M3"):
    rows = []
    for cond in (f"{stage}_baseline_fullAD", f"{stage}_ablated_AD",
                 f"{stage}_ablated_random_fullAD"):
        for i in range(3):
            rows.append({"record_id": f"a{i}", "quadrant": "A",
                         "stage": cond, "condition": cond, "response": f"r{i}"})
    raw = tmp_path / "raw"
    raw.mkdir(exist_ok=True)
    p = raw / f"causal_ablation_v2_{stage}_L24-28_fullAD.json"
    p.write_text(json.dumps(rows), encoding="utf-8")
    return p


def test_relabel_file_rewrites_both_fields_and_is_idempotent(tmp_path):
    p = _fullad_file(tmp_path)

    first = rc.relabel_file(p)
    assert first["modified"]
    assert set(first["conditions_after"]) == set(cbe._cf2_conditions_for_stage("M3"))
    rows = json.loads(p.read_text(encoding="utf-8"))
    assert all(r["stage"] == r["condition"] for r in rows)

    second = rc.relabel_file(p)
    assert not second["modified"], "second pass should be a no-op"
    assert second["changes"] == {}


def test_dry_run_does_not_write(tmp_path):
    p = _fullad_file(tmp_path)
    before = p.read_text(encoding="utf-8")
    r = rc.relabel_file(p, dry_run=True)
    assert r["changes"] and not r["modified"]
    assert p.read_text(encoding="utf-8") == before


def test_repaired_file_yields_complete_cf2_triples(tmp_path):
    """End to end: broken labels -> 0 usable triples; repaired -> 3."""
    p = _fullad_file(tmp_path)
    rc.relabel_file(p)
    rows = json.loads(p.read_text(encoding="utf-8"))

    recs = [{**r, "model_stage": "M3", "judged_prompt_variant": "candidate",
             "strong_reject": {"score": 0.5, "malformed": False,
                               "judge_status": "scored"},
             "wildguard": {"response_harm": None, "judge_status": "not_scored"}}
            for r in rows]
    id_to_split = {f"a{i}": "direction_estimation" for i in range(3)}
    block = cbe._cf2_block(recs, id_to_split, stage="M3", population="estimation")
    assert block["n_effective_triples"] == 3


# --- resumable judging ---------------------------------------------------------
def _rec(rid, cond, response, sr=None, wg=None):
    return {"record_id": rid, "stage": cond, "condition": cond, "quadrant": "A",
            "model_stage": "M3", "response": response,
            "strong_reject": ({"score": sr, "malformed": False, "judge_status": "scored"}
                              if sr is not None else
                              {"score": None, "judge_status": "not_scored"}),
            "wildguard": ({"response_harm": wg, "malformed": False,
                           "judge_status": "scored"} if wg is not None else
                          {"response_harm": None, "judge_status": "not_scored"})}


def test_carry_forward_copies_matching_scores():
    prev = [_rec("a1", "M3_baseline", "hello", sr=0.4, wg=0)]
    cur = [_rec("a1", "M3_baseline", "hello")]
    carried = bj.carry_forward_scores(cur, prev)

    assert carried == {"strong_reject": 1, "wildguard": 1}
    assert cur[0]["strong_reject"]["score"] == 0.4
    assert cur[0]["strong_reject"]["judge_status"] == "scored"


def test_carry_forward_refuses_when_the_response_text_differs():
    """A regenerated response under the same key must be re-scored, not
    silently given the old response's score."""
    prev = [_rec("a1", "M3_baseline", "OLD TEXT", sr=0.4, wg=0)]
    cur = [_rec("a1", "M3_baseline", "NEW TEXT")]
    carried = bj.carry_forward_scores(cur, prev)

    assert carried == {"strong_reject": 0, "wildguard": 0}
    assert cur[0]["strong_reject"]["judge_status"] == "not_scored"


def test_carry_forward_is_per_judge_not_per_row():
    """StrongREJECT scored, WildGuard failed -> carry SR, still re-score WG."""
    prev = [_rec("a1", "M3_baseline", "hello", sr=0.4)]      # wg not scored
    cur = [_rec("a1", "M3_baseline", "hello")]
    carried = bj.carry_forward_scores(cur, prev)

    assert carried == {"strong_reject": 1, "wildguard": 0}
    assert cur[0]["strong_reject"]["score"] == 0.4
    assert cur[0]["wildguard"]["judge_status"] == "not_scored"


def test_carry_forward_skips_malformed_previous_scores():
    prev = [_rec("a1", "M3_baseline", "hello", sr=0.4)]
    prev[0]["strong_reject"]["malformed"] = True
    cur = [_rec("a1", "M3_baseline", "hello")]
    assert bj.carry_forward_scores(cur, prev)["strong_reject"] == 0


def test_carry_forward_leaves_unmatched_rows_alone():
    """The 1200 rows that were never scored must still be pending."""
    prev = [_rec("a1", "M3_ablated_AD", "x", sr=0.4, wg=1)]
    cur = [_rec("a1", "M3_ablated_AD", "x"), _rec("a2", "M3_baseline", "y")]
    carried = bj.carry_forward_scores(cur, prev)

    assert carried == {"strong_reject": 1, "wildguard": 1}
    assert cur[1]["strong_reject"]["judge_status"] == "not_scored"


def test_carry_forward_keeps_the_first_duplicate():
    prev = [_rec("a1", "M3_baseline", "same", sr=0.10),
            _rec("a1", "M3_baseline", "same", sr=0.90)]
    cur = [_rec("a1", "M3_baseline", "same")]
    bj.carry_forward_scores(cur, prev)
    assert cur[0]["strong_reject"]["score"] == 0.10
