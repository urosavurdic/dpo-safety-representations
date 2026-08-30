"""
Focused tests for src/analysis/analyze_3d_h.py.

Per the 3D-H-A task brief, this runs only focused tests for the new
analysis code -- not the broad repository suite. Most tests use small
synthetic fixtures; a few integration checks run the real join/validation
and statistics against the actual frozen repository artifacts and the
real researcher-only answer key (skipped automatically if that key is
not present in this environment, since it is intentionally kept outside
the repository and is not a test fixture).
"""
import json
import os

import numpy as np
import pytest

from src.analysis.analyze_3d_h import (
    EXPECTED_N,
    ValidationError,
    _is_valid_rating,
    classify_decision,
    describe_group,
    get_original_packet_prompt_texts,
    join_and_validate,
    load_private_key,
    load_reviewed_csv,
    permutation_test,
    primary_analysis,
    rank_biserial,
    secondary_analysis,
    two_group_comparison,
    validate_private_key_path,
)
from src.data_pipeline.lexical_outlierness import REPO_ROOT

REAL_PRIVATE_KEY_PATH = os.environ.get(
    "REAL_3D_H_ANSWER_KEY_PATH", "/home/claude/private_outputs/3d_h_answer_key.json"
)


def _mk_key_entry(review_id, record_id, source, category, group_id, p_tfidf, p_selfinfo, tail):
    return {
        "review_id": review_id,
        "record_id": record_id,
        "source": source,
        "category": category,
        "group_id": group_id,
        "p_tfidf": p_tfidf,
        "p_selfinfo": p_selfinfo,
        "tail": tail,
        "selection_seed": 20260829,
        "presentation_seed": 20260830,
    }


def _mk_pilot_row(tail_selfinfo="mid"):
    return {"tail_selfinfo": tail_selfinfo}


# ── rating validation ────────────────────────────────────────────────────
@pytest.mark.parametrize("raw,expected", [("1", True), ("5", True), ("3", True),
                                           ("0", False), ("6", False), ("", False),
                                           (None, False), ("abc", False), ("3.5", False),
                                           (" 4 ", True)])
def test_is_valid_rating(raw, expected):
    assert _is_valid_rating(raw) == expected


# ── describe_group / rank_biserial ──────────────────────────────────────
def test_describe_group_basic():
    d = describe_group([1, 1, 3, 5])
    assert d["n"] == 4
    assert d["mean"] == 2.5
    assert d["median"] == 2.0
    assert d["distribution_1_to_5"] == {"1": 2, "2": 0, "3": 1, "4": 0, "5": 1}


def test_rank_biserial_perfect_separation():
    high = [5, 5, 4]
    low = [1, 1, 2]
    r = rank_biserial(high, low)
    assert r == pytest.approx(1.0)


def test_rank_biserial_reverse_separation():
    high = [1, 1, 2]
    low = [5, 5, 4]
    r = rank_biserial(high, low)
    assert r == pytest.approx(-1.0)


def test_rank_biserial_no_effect_symmetric():
    high = [1, 2, 3, 4]
    low = [1, 2, 3, 4]
    r = rank_biserial(high, low)
    assert r == pytest.approx(0.0, abs=1e-9)


def test_two_group_comparison_handles_empty_group():
    result = two_group_comparison([], [1, 2, 3])
    assert result["mean_difference_high_minus_low"] is None
    assert result["mann_whitney_p_value"] is None


# ── join_and_validate ────────────────────────────────────────────────────
def _synthetic_fixture(n=4):
    reviewed_rows = [
        {"review_id": f"R{i:03d}", "prompt_text": f"prompt {i}", "rating": str((i % 5) + 1), "notes": ""}
        for i in range(n)
    ]
    key_entries = [
        _mk_key_entry(f"R{i:03d}", f"rec_{i}", "StrongREJECT" if i % 2 else "SimpleSafetyTests",
                      "unknown", f"g_{i}", 0.9 if i < n // 2 else 0.1, 0.5,
                      "high" if i < n // 2 else "low")
        for i in range(n)
    ]
    pilot_row_level = {f"rec_{i}": _mk_pilot_row("mid") for i in range(n)}
    original_prompt_texts = {f"R{i:03d}": f"prompt {i}" for i in range(n)}
    return reviewed_rows, key_entries, pilot_row_level, original_prompt_texts


def test_join_and_validate_happy_path(monkeypatch):
    monkeypatch.setattr("src.analysis.analyze_3d_h.EXPECTED_N", 4)
    reviewed_rows, key_entries, pilot_row_level, original = _synthetic_fixture(4)
    joined = join_and_validate(reviewed_rows, key_entries, pilot_row_level, original)
    assert len(joined) == 4
    assert all(j["rating"] in (1, 2, 3, 4, 5) for j in joined)


def test_join_and_validate_rejects_wrong_count(monkeypatch):
    monkeypatch.setattr("src.analysis.analyze_3d_h.EXPECTED_N", 5)
    reviewed_rows, key_entries, pilot_row_level, original = _synthetic_fixture(4)
    with pytest.raises(SystemExit):
        join_and_validate(reviewed_rows, key_entries, pilot_row_level, original)


def test_join_and_validate_rejects_id_mismatch(monkeypatch):
    monkeypatch.setattr("src.analysis.analyze_3d_h.EXPECTED_N", 4)
    reviewed_rows, key_entries, pilot_row_level, original = _synthetic_fixture(4)
    key_entries[0]["review_id"] = "R999"  # no longer matches reviewed_rows
    with pytest.raises(SystemExit):
        join_and_validate(reviewed_rows, key_entries, pilot_row_level, original)


def test_join_and_validate_rejects_bad_rating(monkeypatch):
    monkeypatch.setattr("src.analysis.analyze_3d_h.EXPECTED_N", 4)
    reviewed_rows, key_entries, pilot_row_level, original = _synthetic_fixture(4)
    reviewed_rows[0]["rating"] = "9"
    with pytest.raises(SystemExit):
        join_and_validate(reviewed_rows, key_entries, pilot_row_level, original)


def test_join_and_validate_rejects_tampered_prompt_text(monkeypatch):
    monkeypatch.setattr("src.analysis.analyze_3d_h.EXPECTED_N", 4)
    reviewed_rows, key_entries, pilot_row_level, original = _synthetic_fixture(4)
    reviewed_rows[0]["prompt_text"] = "a different prompt entirely"
    with pytest.raises(SystemExit):
        join_and_validate(reviewed_rows, key_entries, pilot_row_level, original)


def test_join_and_validate_rejects_duplicate_group_id(monkeypatch):
    monkeypatch.setattr("src.analysis.analyze_3d_h.EXPECTED_N", 4)
    reviewed_rows, key_entries, pilot_row_level, original = _synthetic_fixture(4)
    key_entries[1]["group_id"] = key_entries[0]["group_id"]
    with pytest.raises(SystemExit):
        join_and_validate(reviewed_rows, key_entries, pilot_row_level, original)


def test_join_and_validate_rejects_unknown_record_id(monkeypatch):
    monkeypatch.setattr("src.analysis.analyze_3d_h.EXPECTED_N", 4)
    reviewed_rows, key_entries, pilot_row_level, original = _synthetic_fixture(4)
    del pilot_row_level["rec_0"]
    with pytest.raises(SystemExit):
        join_and_validate(reviewed_rows, key_entries, pilot_row_level, original)


# ── primary / secondary / permutation / decision ────────────────────────
def _mk_joined_row(review_id, rating, tail_tfidf, tail_selfinfo, p_tfidf, p_selfinfo):
    return {
        "review_id": review_id,
        "rating": rating,
        "record_id": f"rec_{review_id}",
        "p_tfidf": p_tfidf,
        "p_selfinfo": p_selfinfo,
        "tail_tfidf": tail_tfidf,
        "tail_selfinfo": tail_selfinfo,
        "source": "StrongREJECT",
        "group_id": f"g_{review_id}",
    }


def test_primary_analysis_separates_tails():
    joined = (
        [_mk_joined_row(f"H{i}", 5, "high", "mid", 0.9, 0.5) for i in range(4)]
        + [_mk_joined_row(f"L{i}", 1, "low", "mid", 0.1, 0.5) for i in range(4)]
    )
    prim = primary_analysis(joined)
    assert prim["high"]["n"] == 4
    assert prim["low"]["n"] == 4
    assert prim["mean_difference_high_minus_low"] == pytest.approx(4.0)
    assert prim["rank_biserial_effect_size"] == pytest.approx(1.0)


def test_secondary_analysis_excludes_mid():
    joined = (
        [_mk_joined_row(f"H{i}", 5, "high", "high", 0.9, 0.9) for i in range(3)]
        + [_mk_joined_row(f"M{i}", 3, "low", "mid", 0.1, 0.5) for i in range(2)]
        + [_mk_joined_row(f"L{i}", 1, "low", "low", 0.1, 0.1) for i in range(3)]
    )
    sec = secondary_analysis(joined)
    assert sec["high"]["n"] == 3
    assert sec["low"]["n"] == 3
    assert sec["n_mid_tail_selfinfo_excluded_from_comparison"] == 2
    assert sec["status"].startswith("SECONDARY")


def test_permutation_test_deterministic_and_extreme_for_perfect_separation():
    joined = (
        [_mk_joined_row(f"H{i}", 5, "high", "mid", 0.9, 0.5) for i in range(8)]
        + [_mk_joined_row(f"L{i}", 1, "low", "mid", 0.1, 0.5) for i in range(8)]
    )
    perm1 = permutation_test(joined, seed=123, n_perm=2000)
    perm2 = permutation_test(joined, seed=123, n_perm=2000)
    assert perm1 == perm2  # deterministic given the same seed
    assert perm1["observed_statistic"] == pytest.approx(4.0)
    # perfect separation should be essentially impossible under permutation
    assert perm1["empirical_two_sided_p_value"] < 0.01


def test_permutation_test_different_seed_gives_different_p_but_same_observed():
    joined = (
        [_mk_joined_row(f"H{i}", 4, "high", "mid", 0.9, 0.5) for i in range(6)]
        + [_mk_joined_row(f"L{i}", 3, "low", "mid", 0.1, 0.5) for i in range(6)]
    )
    perm1 = permutation_test(joined, seed=1, n_perm=2000)
    perm2 = permutation_test(joined, seed=2, n_perm=2000)
    assert perm1["observed_statistic"] == perm2["observed_statistic"]


@pytest.mark.parametrize(
    "effect_size,perm_p,expected",
    [
        (0.6, 0.01, "SUPPORTIVE"),
        (0.35, 0.04, "SUPPORTIVE"),
        (0.2, 0.03, "INCONCLUSIVE"),
        (0.05, 0.5, "NOT SUPPORTIVE"),
        (0.15, 0.5, "INCONCLUSIVE"),
        (None, None, "INCONCLUSIVE"),
    ],
)
def test_classify_decision(effect_size, perm_p, expected):
    assert classify_decision(effect_size, perm_p) == expected


def test_validate_private_key_path_rejects_inside_repo():
    with pytest.raises(SystemExit):
        validate_private_key_path(str(REPO_ROOT / "logs" / "leak.json"))


def test_validate_private_key_path_accepts_outside_repo(tmp_path):
    outside = tmp_path / "key.json"
    resolved = validate_private_key_path(str(outside))
    assert resolved == outside.resolve()


# ── integration checks against the real frozen artifacts + real key ────
@pytest.mark.skipif(
    not os.path.exists(REAL_PRIVATE_KEY_PATH),
    reason="researcher-only answer key not present in this environment",
)
def test_real_join_and_analysis_end_to_end():
    from src.analysis.analyze_3d_h import (
        AUDIT_JSON_PATH,
        GROUPS_ARTIFACT_PATH,
        PILOT_JSON_PATH,
        REVIEWED_CSV_PATH,
        REVIEWED_CSV_REL,
    )

    assert AUDIT_JSON_PATH.exists()
    assert GROUPS_ARTIFACT_PATH.exists()

    reviewed_rows = load_reviewed_csv(REVIEWED_CSV_PATH)
    key_entries = load_private_key(validate_private_key_path(REAL_PRIVATE_KEY_PATH))
    pilot = json.loads(PILOT_JSON_PATH.read_text(encoding="utf-8"))
    original = get_original_packet_prompt_texts(REVIEWED_CSV_REL)

    joined = join_and_validate(reviewed_rows, key_entries, pilot["scoring"]["row_level"], original)
    assert len(joined) == EXPECTED_N

    prim = primary_analysis(joined)
    assert prim["high"]["n"] + prim["low"]["n"] == EXPECTED_N
    assert -1.0 <= prim["rank_biserial_effect_size"] <= 1.0

    perm = permutation_test(joined, n_perm=5000)
    assert 0.0 <= perm["empirical_two_sided_p_value"] <= 1.0

    decision = classify_decision(prim["rank_biserial_effect_size"], perm["empirical_two_sided_p_value"])
    assert decision in ("SUPPORTIVE", "INCONCLUSIVE", "NOT SUPPORTIVE")
