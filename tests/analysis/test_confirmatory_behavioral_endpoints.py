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


# --- regression: per-branch CF2 must never pool across stages --------------
def test_cf2_by_branch_keeps_stages_strictly_separate():
    # M3 and M3_direct each get their own complete triple, with DIFFERENT
    # effect sizes. A prior version filtered `stage.startswith("M3")`, which
    # would have pooled both branches' triples into a single CF2 number.
    recs = []
    for rid, base, ad, rand in [("a1", 0.1, 0.5, 0.2), ("a2", 0.2, 0.4, 0.3)]:
        recs += [
            _rec(rid, "M3", "A", base, condition="M3_baseline"),
            _rec(rid, "M3", "A", ad, condition="M3_ablated_AD"),
            _rec(rid, "M3", "A", rand, condition="M3_ablated_random"),
        ]
    for rid, base, ad, rand in [("a1", 0.0, 0.05, 0.9), ("a2", 0.0, 0.05, 0.85)]:
        recs += [
            _rec(rid, "M3_direct", "A", base, condition="M3_direct_baseline"),
            _rec(rid, "M3_direct", "A", ad, condition="M3_direct_ablated_AD"),
            _rec(rid, "M3_direct", "A", rand, condition="M3_direct_ablated_random"),
        ]
    id_to_split = {"a1": "held_out_behavioral", "a2": "held_out_behavioral"}

    by_stage = cbe.compute_causal_by_branch(recs, id_to_split, stages=("M3", "M3_direct"))
    m3 = by_stage["M3"]["primary"]
    m3_direct = by_stage["M3_direct"]["primary"]

    assert m3["n_effective_triples"] == 2
    assert m3_direct["n_effective_triples"] == 2
    # M3: AD-random = (0.5-0.2)+(0.4-0.3) / 2 = 0.2 ; M3_direct: (0.05-0.9)+(0.05-0.85)/2 = -0.825
    assert m3["cf2"] == pytest.approx(0.2)
    assert m3_direct["cf2"] == pytest.approx(-0.825)
    assert by_stage["M3"]["confirmatory"] is True
    assert by_stage["M3_direct"]["confirmatory"] is False


def test_branch_interaction_is_paired_difference_of_differences():
    # M3: AD-random contribution = +0.3 per prompt.  M3_direct_alt: -0.1 per prompt.
    # Interaction (M3 - M3_direct_alt) should be a tight +0.4, CI excluding zero.
    recs = []
    for rid in ("a1", "a2", "a3"):
        recs += [
            _rec(rid, "M3", "A", 0.1, condition="M3_baseline"),
            _rec(rid, "M3", "A", 0.5, condition="M3_ablated_AD"),
            _rec(rid, "M3", "A", 0.2, condition="M3_ablated_random"),
            _rec(rid, "M3_direct_alt", "A", 0.1, condition="M3_direct_alt_baseline"),
            _rec(rid, "M3_direct_alt", "A", 0.2, condition="M3_direct_alt_ablated_AD"),
            _rec(rid, "M3_direct_alt", "A", 0.3, condition="M3_direct_alt_ablated_random"),
        ]
    id_to_split = {r: "held_out_behavioral" for r in ("a1", "a2", "a3")}
    bi = cbe.compute_branch_interactions(recs, id_to_split,
                                         stages=("M3", "M3_direct_alt"))
    p = bi["pairs"]["M3_vs_M3_direct_alt"]
    assert p["n_shared_prompts"] == 3
    assert p["delta_reference"] == pytest.approx(0.3)
    assert p["delta_branch"] == pytest.approx(-0.1)
    assert p["interaction"] == pytest.approx(0.4)
    assert p["ci_excludes_zero"] is True
    assert bi["status"].startswith("EXPLORATORY")


def test_branch_interaction_only_uses_prompts_scored_in_both_branches():
    recs = [
        _rec("a1", "M3", "A", 0.1, condition="M3_baseline"),
        _rec("a1", "M3", "A", 0.5, condition="M3_ablated_AD"),
        _rec("a1", "M3", "A", 0.2, condition="M3_ablated_random"),
        # a2 only has M3 data, not M3_alt -> must be dropped from the pair
        _rec("a2", "M3", "A", 0.0, condition="M3_baseline"),
        _rec("a2", "M3", "A", 0.9, condition="M3_ablated_AD"),
        _rec("a2", "M3", "A", 0.1, condition="M3_ablated_random"),
        _rec("a1", "M3_alt", "A", 0.1, condition="M3_alt_baseline"),
        _rec("a1", "M3_alt", "A", 0.3, condition="M3_alt_ablated_AD"),
        _rec("a1", "M3_alt", "A", 0.2, condition="M3_alt_ablated_random"),
    ]
    id_to_split = {"a1": "held_out_behavioral", "a2": "held_out_behavioral"}
    bi = cbe.compute_branch_interactions(recs, id_to_split, stages=("M3", "M3_alt"))
    assert bi["pairs"]["M3_vs_M3_alt"]["n_shared_prompts"] == 1


def test_build_report_includes_cf2_by_stage_and_m3_alias(tmp_path):
    recs = [
        _rec("c1", "M2", "C", 0.3), _rec("c1", "M3", "C", 0.1),
    ] + _cf2_recs()
    judged = tmp_path / "j.json"
    judged.write_text(json.dumps({"records": recs}), encoding="utf-8")
    bench = tmp_path / "b.jsonl"
    bench.write_text("\n".join(json.dumps({"record_id": r, "split": s}) for r, s in [
        ("c1", None), ("a1", "held_out_behavioral"),
        ("a2", "held_out_behavioral"), ("a3", "direction_estimation"),
    ]), encoding="utf-8")

    rep = cbe.build_report(judged, bench)
    assert set(rep["CF2_by_stage"]) == set(cbe.CAUSAL_STAGES)
    assert rep["CF2"] == rep["CF2_by_stage"]["M3"]
    # no causal data generated yet for the other three branches -> self-explanatory zero
    assert rep["CF2_by_stage"]["M3_direct"]["primary"]["n_effective_triples"] == 0


# --- regression: main() must not crash when CF2 has zero triples -----------
def test_main_prints_unavailable_instead_of_crashing_when_cf2_is_empty(tmp_path, monkeypatch, capsys):
    recs = [
        _rec("c1", "M2", "C", 0.5), _rec("c1", "M3", "C", 0.2),
        # quadrant A present but incomplete (missing ablated_random) -> CF2 empty
        _rec("a1", "M3", "A", 0.1, condition="M3_baseline"),
        _rec("a1", "M3", "A", 0.4, condition="M3_ablated_AD"),
    ]
    judged = tmp_path / "j.json"
    judged.write_text(json.dumps({"records": recs}), encoding="utf-8")
    bench = tmp_path / "b.jsonl"
    bench.write_text("\n".join(json.dumps({"record_id": r, "split": s}) for r, s in [
        ("c1", None), ("a1", "held_out_behavioral"),
    ]), encoding="utf-8")
    out = tmp_path / "out.json"

    monkeypatch.setattr("sys.argv", [
        "x", "--judged", str(judged), "--benchmark", str(bench), "--out", str(out),
    ])
    cbe.main()  # must not raise
    printed = capsys.readouterr().out
    assert "CF2" in printed and "UNAVAILABLE" in printed
    assert "CF1  Delta_C" in printed


# --- circularity check: three-way split population ---------------------------
def test_cf2_reports_held_out_estimation_and_all_separately():
    # a1/a2 held-out (contribution +0.3 each), e1/e2 estimation (+0.9 each,
    # i.e. an inflated "self-influence" effect). The three blocks must not mix.
    recs = []
    for rid, base, ad, rand in [("a1", 0.1, 0.5, 0.2), ("a2", 0.2, 0.6, 0.3)]:
        recs += [
            _rec(rid, "M3", "A", base, condition="M3_baseline"),
            _rec(rid, "M3", "A", ad, condition="M3_ablated_AD"),
            _rec(rid, "M3", "A", rand, condition="M3_ablated_random"),
        ]
    for rid, base, ad, rand in [("e1", 0.0, 0.9, 0.0), ("e2", 0.0, 0.9, 0.0)]:
        recs += [
            _rec(rid, "M3", "A", base, condition="M3_baseline"),
            _rec(rid, "M3", "A", ad, condition="M3_ablated_AD"),
            _rec(rid, "M3", "A", rand, condition="M3_ablated_random"),
        ]
    id_to_split = {"a1": "held_out_behavioral", "a2": "held_out_behavioral",
                   "e1": "direction_estimation", "e2": "direction_estimation"}
    cf2 = cbe.compute_cf2(recs, id_to_split)

    assert cf2["primary"]["n_effective_triples"] == 2
    assert cf2["primary"]["cf2"] == pytest.approx(0.3)
    assert cf2["estimation_split_only"]["n_effective_triples"] == 2
    assert cf2["estimation_split_only"]["cf2"] == pytest.approx(0.9)
    assert cf2["full_A_sensitivity"]["n_effective_triples"] == 4
    assert cf2["full_A_sensitivity"]["cf2"] == pytest.approx(0.6)
    assert cf2["primary"]["population_key"] == "held_out"
    assert cf2["estimation_split_only"]["population_key"] == "estimation"
    assert cf2["full_A_sensitivity"]["population_key"] == "all"


def test_cf2_block_rejects_unknown_population():
    with pytest.raises(ValueError):
        cbe._cf2_block([], {}, stage="M3", population="nonsense")
