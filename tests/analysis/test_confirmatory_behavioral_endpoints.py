"""Tests for src/analysis/confirmatory_behavioral_endpoints.py (CF1 / CF2)."""
import json

import pytest

from src.analysis import confirmatory_behavioral_endpoints as cbe


def _rec(record_id, model_stage, quadrant, sr, *, condition=None, wg_harm=None,
         variant="candidate", stage=None):
    r = {
        "record_id": record_id,
        "model_stage": model_stage,
        "stage": stage if stage is not None else (condition or model_stage),
        "condition": condition,
        "quadrant": quadrant,
        "judged_prompt_variant": variant,
        "strong_reject": (
            {"score": sr, "malformed": False, "judge_status": "scored"} if sr is not None
            else {"score": None, "malformed": None, "judge_status": "not_scored"}
        ),
        "wildguard": (
            {"response_harm": wg_harm, "malformed": False, "judge_status": "scored"}
            if wg_harm is not None
            else {"response_harm": None, "malformed": None, "judge_status": "not_scored"}
        ),
    }
    return r


# --- CF1 -----------------------------------------------------------------------
def test_cf1_delta_is_mean_of_M3_minus_M2_on_complete_pairs():
    recs = [
        _rec("c1", "M2", "C", 0.2), _rec("c1", "M3", "C", 0.1),   # delta -0.1
        _rec("c2", "M2", "C", 0.4), _rec("c2", "M3", "C", 0.5),   # delta +0.1
        _rec("c3", "M2", "C", 0.6),                                # no M3 -> dropped
        _rec("c4", "M3", "C", 0.3),                                # no M2 -> dropped
    ]
    cf1 = cbe.compute_cf1(recs)
    assert cf1["n_effective_pairs"] == 2
    assert cf1["delta_c"] == pytest.approx(0.0)  # mean(-0.1, +0.1)
    assert cf1["dropped"]["incomplete_pairs"] == 2


def test_cf1_excludes_source_overt_and_intervention_rows():
    recs = [
        _rec("c1", "M2", "C", 0.2), _rec("c1", "M3", "C", 0.1),
        _rec("c1", "M3", "C", 0.9, variant="source_overt"),          # excluded
        _rec("c2", "M2", "C", 0.3, condition="M2_ablated_AD"),        # excluded (intervention)
        _rec("c2", "M3", "C", 0.3, condition="M3_ablated_AD"),
    ]
    cf1 = cbe.compute_cf1(recs)
    assert cf1["n_effective_pairs"] == 1
    assert cf1["delta_c"] == pytest.approx(-0.1)


def test_cf1_drops_unusable_sr_rows():
    recs = [
        _rec("c1", "M2", "C", 0.2), _rec("c1", "M3", "C", None),   # M3 not scored
        _rec("c2", "M2", "C", 0.4), _rec("c2", "M3", "C", 0.4),
    ]
    cf1 = cbe.compute_cf1(recs)
    assert cf1["n_effective_pairs"] == 1
    assert cf1["dropped"]["unusable_sr_rows"] == 1


# --- CF2 -----------------------------------------------------------------------
def _cf2_recs():
    # a1 held-out, a2 held-out, a3 NOT held-out; each with the 3 conditions
    out = []
    for rid, base, ad, rand in [("a1", 0.1, 0.5, 0.2), ("a2", 0.2, 0.4, 0.3), ("a3", 0.0, 0.9, 0.1)]:
        out += [
            _rec(rid, "M3", "A", base, condition="M3_baseline"),
            _rec(rid, "M3", "A", ad, condition="M3_ablated_AD"),
            _rec(rid, "M3", "A", rand, condition="M3_ablated_random"),
        ]
    return out


def test_cf2_primary_uses_held_out_only_and_is_AD_minus_random():
    recs = _cf2_recs()
    id_to_split = {"a1": "held_out_behavioral", "a2": "held_out_behavioral",
                   "a3": "direction_estimation"}
    cf2 = cbe.compute_cf2(recs, id_to_split)
    p = cf2["primary"]
    # per-prompt SR_AD - SR_random: a1 = 0.5-0.2=0.3 ; a2 = 0.4-0.3=0.1 ; mean 0.2
    assert p["n_effective_triples"] == 2
    assert p["cf2"] == pytest.approx(0.2)
    assert p["E_AD"] == pytest.approx(((0.5 - 0.1) + (0.4 - 0.2)) / 2)
    assert p["E_random"] == pytest.approx(((0.2 - 0.1) + (0.3 - 0.2)) / 2)
    # full-A sensitivity includes a3
    assert cf2["full_A_sensitivity"]["n_effective_triples"] == 3


def test_cf2_drops_incomplete_triples():
    recs = [
        _rec("a1", "M3", "A", 0.1, condition="M3_baseline"),
        _rec("a1", "M3", "A", 0.5, condition="M3_ablated_AD"),
        # a1 missing M3_ablated_random -> dropped
        _rec("a2", "M3", "A", 0.2, condition="M3_baseline"),
        _rec("a2", "M3", "A", 0.4, condition="M3_ablated_AD"),
        _rec("a2", "M3", "A", 0.3, condition="M3_ablated_random"),
    ]
    cf2 = cbe.compute_cf2(recs, {"a1": "held_out_behavioral", "a2": "held_out_behavioral"})
    assert cf2["primary"]["n_effective_triples"] == 1
    assert cf2["primary"]["dropped"]["incomplete_triples"] == 1


def test_cf2_wildguard_secondary_only_when_wg_present():
    recs = _cf2_recs()
    id_to_split = {"a1": "held_out_behavioral", "a2": "held_out_behavioral", "a3": "x"}
    cf2_no_wg = cbe.compute_cf2(recs, id_to_split)
    assert "secondary_binary_wildguard" not in cf2_no_wg["primary"]

    recs_wg = []
    for rid, base, ad, rand in [("a1", 0.1, 0.5, 0.2), ("a2", 0.2, 0.4, 0.3)]:
        recs_wg += [
            _rec(rid, "M3", "A", base, condition="M3_baseline", wg_harm=False),
            _rec(rid, "M3", "A", ad, condition="M3_ablated_AD", wg_harm=True),
            _rec(rid, "M3", "A", rand, condition="M3_ablated_random", wg_harm=False),
        ]
    cf2_wg = cbe.compute_cf2(recs_wg, {"a1": "held_out_behavioral", "a2": "held_out_behavioral"})
    assert "secondary_binary_wildguard" in cf2_wg["primary"]
    assert cf2_wg["primary"]["secondary_binary_wildguard"]["n_effective_triples"] == 2


# --- driver ------------------------------------------------------------------
def test_build_report_flags_strongreject_unavailable(tmp_path):
    judged = tmp_path / "j.json"
    judged.write_text(json.dumps({"records": [
        _rec("c1", "M2", "C", None), _rec("c1", "M3", "C", None),
    ]}), encoding="utf-8")
    bench = tmp_path / "b.jsonl"
    bench.write_text(json.dumps({"record_id": "c1", "split": None}) + "\n", encoding="utf-8")
    rep = cbe.build_report(judged, bench)
    assert "STRONGREJECT_UNAVAILABLE" in rep["status"]
    assert "CF1" not in rep


def test_build_report_end_to_end_and_deterministic(tmp_path):
    recs = [
        _rec("c1", "M2", "C", 0.3), _rec("c1", "M3", "C", 0.1),
        _rec("c2", "M2", "C", 0.5), _rec("c2", "M3", "C", 0.2),
    ] + _cf2_recs()
    judged = tmp_path / "j.json"
    judged.write_text(json.dumps({"records": recs, "models": {"strong_reject": "x"}}), encoding="utf-8")
    bench = tmp_path / "b.jsonl"
    bench.write_text("\n".join(json.dumps({"record_id": r, "split": s}) for r, s in [
        ("c1", None), ("c2", None), ("a1", "held_out_behavioral"),
        ("a2", "held_out_behavioral"), ("a3", "direction_estimation"),
    ]), encoding="utf-8")

    r1 = cbe.build_report(judged, bench)
    r2 = cbe.build_report(judged, bench)
    assert r1 == r2
    assert r1["status"] == "ok"
    assert r1["CF1"]["delta_c"] == pytest.approx(-0.25)  # mean(-0.2, -0.3)
    assert r1["CF1"]["ci_low"] <= r1["CF1"]["delta_c"] <= r1["CF1"]["ci_high"]
    assert r1["CF2"]["primary"]["n_effective_triples"] == 2


# --- regression: stage/condition relabel-mismatch bug -----------------------
def test_condition_prefers_stage_over_stale_condition_field():
    # reproduces v2_pipeline.stage_causal's legacy-shard-reuse relabel bug:
    # stage was correctly renamed to "M3_ablated_AD" but condition was left at
    # the old shard-unit name "M3_ablated".
    rec = {"stage": "M3_ablated_AD", "condition": "M3_ablated"}
    assert cbe._condition(rec) == "M3_ablated_AD"


def test_cf2_recognizes_ablated_AD_rows_even_with_stale_condition_field():
    recs = []
    for rid, base, ad, rand in [("a1", 0.1, 0.5, 0.2), ("a2", 0.2, 0.4, 0.3)]:
        recs += [
            _rec(rid, "M3", "A", base, condition="M3_baseline", stage="M3_baseline"),
            # stale condition field, correct stage field (the actual bug shape)
            _rec(rid, "M3", "A", ad, condition="M3_ablated", stage="M3_ablated_AD"),
            _rec(rid, "M3", "A", rand, condition="M3_ablated_random", stage="M3_ablated_random"),
        ]
    id_to_split = {"a1": "held_out_behavioral", "a2": "held_out_behavioral"}
    cf2 = cbe.compute_cf2(recs, id_to_split)
    assert cf2["primary"]["n_effective_triples"] == 2
    assert cf2["primary"]["cf2"] == pytest.approx(0.2)
