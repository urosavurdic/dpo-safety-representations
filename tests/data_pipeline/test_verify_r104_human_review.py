"""
Focused tests for src/data_pipeline/verify_r104_human_review.py.

Mirrors the convention in
tests/data_pipeline/test_build_r104_human_review_packet.py: unit tests
on pure logic (row-count/uniqueness/decision-vocabulary validation,
fail-closed behavior on unexpected text mismatches) plus integration
checks against the real frozen repository artifacts, since this task
specifically requires verifying against current files rather than a
historical or assumed count.
"""
import copy
import csv

import pytest

from src.data_pipeline.build_r104_human_review_packet import (
    InvariantError,
    assign_review_ids,
    load_eligible_population,
)
from src.data_pipeline.verify_r104_human_review import (
    KNOWN_ORTHOGRAPHIC_DISCREPANCIES,
    assert_full_known_discrepancy_set_realized,
    load_blind_review,
    verify_mapping,
)


def _write_csv(path, rows):
    fieldnames = ["review_id", "source_prompt", "candidate_prompt", "decision", "notes"]
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _row(review_id, decision="accept"):
    return {
        "review_id": review_id,
        "source_prompt": f"source {review_id}",
        "candidate_prompt": f"candidate {review_id}",
        "decision": decision,
        "notes": "",
    }


# ── load_blind_review: row count / uniqueness / decision validation ────
def test_load_blind_review_rejects_wrong_row_count(tmp_path):
    path = tmp_path / "blind.csv"
    _write_csv(path, [_row(f"R{i:03d}") for i in range(1, 104)])  # 103, not 104
    with pytest.raises(InvariantError):
        load_blind_review(path)


def test_load_blind_review_rejects_duplicate_review_ids(tmp_path):
    path = tmp_path / "blind.csv"
    rows = [_row(f"R{i:03d}") for i in range(1, 104)]
    rows.append(_row("R001"))  # duplicate, total 104 rows
    _write_csv(path, rows)
    with pytest.raises(InvariantError):
        load_blind_review(path)


def test_load_blind_review_rejects_empty_decision(tmp_path):
    path = tmp_path / "blind.csv"
    rows = [_row(f"R{i:03d}") for i in range(1, 105)]
    rows[0]["decision"] = ""
    _write_csv(path, rows)
    with pytest.raises(InvariantError):
        load_blind_review(path)


def test_load_blind_review_rejects_unrecognized_decision_vocabulary(tmp_path):
    path = tmp_path / "blind.csv"
    rows = [_row(f"R{i:03d}") for i in range(1, 105)]
    rows[0]["decision"] = "KEEP"  # instructions vocabulary, not what's used
    _write_csv(path, rows)
    with pytest.raises(InvariantError):
        load_blind_review(path)


def test_load_blind_review_accepts_well_formed_csv(tmp_path):
    path = tmp_path / "blind.csv"
    rows = [_row(f"R{i:03d}") for i in range(1, 105)]
    _write_csv(path, rows)
    by_id = load_blind_review(path)
    assert len(by_id) == 104
    assert set(by_id) == {f"R{i:03d}" for i in range(1, 105)}


# ── verify_mapping: pure-logic fail-closed behavior ─────────────────────
def _tiny_population():
    return {
        "aaa": {"source_prompt": "src aaa", "candidate_prompt": "cand aaa", "pair_id": "aaa"},
        "bbb": {"source_prompt": "src bbb", "candidate_prompt": "cand bbb", "pair_id": "bbb"},
    }


def test_verify_mapping_passes_on_exact_matches():
    population = _tiny_population()
    # assign_review_ids sorts keys: R001 -> aaa, R002 -> bbb
    blind_by_id = {
        "R001": {"source_prompt": "src aaa", "candidate_prompt": "cand aaa"},
        "R002": {"source_prompt": "src bbb", "candidate_prompt": "cand bbb"},
    }
    pairs, discrepancies = verify_mapping(population, blind_by_id)
    assert pairs == [("R001", "aaa"), ("R002", "bbb")]
    assert discrepancies == {}


def test_verify_mapping_fails_closed_on_unknown_text_mismatch():
    population = _tiny_population()
    blind_by_id = {
        "R001": {"source_prompt": "src aaa", "candidate_prompt": "cand aaa"},
        # Not a pre-verified known discrepancy -> must fail closed.
        "R002": {"source_prompt": "SOMETHING ELSE ENTIRELY", "candidate_prompt": "cand bbb"},
    }
    with pytest.raises(InvariantError):
        verify_mapping(population, blind_by_id)


def test_verify_mapping_fails_closed_on_candidate_prompt_mismatch():
    population = _tiny_population()
    blind_by_id = {
        "R001": {"source_prompt": "src aaa", "candidate_prompt": "cand aaa"},
        "R002": {"source_prompt": "src bbb", "candidate_prompt": "DIFFERENT CANDIDATE"},
    }
    with pytest.raises(InvariantError):
        verify_mapping(population, blind_by_id)


def test_verify_mapping_fails_closed_on_review_id_set_mismatch():
    population = _tiny_population()
    blind_by_id = {
        "R001": {"source_prompt": "src aaa", "candidate_prompt": "cand aaa"},
        "R999": {"source_prompt": "src bbb", "candidate_prompt": "cand bbb"},
    }
    with pytest.raises(InvariantError):
        verify_mapping(population, blind_by_id)


# ── integration: real frozen repository artifacts ───────────────────────
def test_real_inputs_produce_exactly_the_two_known_discrepancies():
    population, _ = load_eligible_population()
    blind_by_id = load_blind_review(
        _real_blind_csv_path()
    )
    pairs, discrepancies = verify_mapping(population, blind_by_id)

    assert len(pairs) == 104
    assert set(discrepancies.keys()) == {"R029", "R065"}
    assert discrepancies["R029"]["record_id"] == "SR_disinfo_12"
    assert discrepancies["R065"]["record_id"] == "SR_harass_09"
    # And the release-specific completeness check accepts it.
    assert_full_known_discrepancy_set_realized(discrepancies)


def test_assert_full_known_discrepancy_set_realized_rejects_partial_set():
    with pytest.raises(InvariantError):
        assert_full_known_discrepancy_set_realized({"R029": {}})


def test_assert_full_known_discrepancy_set_realized_rejects_empty_set():
    with pytest.raises(InvariantError):
        assert_full_known_discrepancy_set_realized({})


def test_assert_full_known_discrepancy_set_realized_accepts_exact_set():
    assert_full_known_discrepancy_set_realized(
        dict(KNOWN_ORTHOGRAPHIC_DISCREPANCIES)
    )


def test_real_inputs_reconstructed_mapping_is_deterministic():
    population, _ = load_eligible_population()
    pairs1 = assign_review_ids(population)
    pairs2 = assign_review_ids(population)
    assert pairs1 == pairs2
    assert len(pairs1) == 104
    assert len(set(rid for rid, _ in pairs1)) == 104
    assert len(set(record_id for _, record_id in pairs1)) == 104


def test_real_inputs_all_104_decisions_are_accept():
    blind_by_id = load_blind_review(_real_blind_csv_path())
    decisions = {row["decision"].strip() for row in blind_by_id.values()}
    assert decisions == {"accept"}


def test_known_discrepancies_are_orthographic_only_not_semantic():
    # Guard against a future edit silently widening this fixed lookup
    # into something that isn't actually orthographic-only.
    assert set(KNOWN_ORTHOGRAPHIC_DISCREPANCIES) == {"R029", "R065"}
    for entry in KNOWN_ORTHOGRAPHIC_DISCREPANCIES.values():
        assert entry["field"] == "source_prompt"
        auth = entry["authoritative_text"]
        blind = entry["blind_csv_text"]
        assert auth != blind
        # Word-level overlap should be near-total: at most one or two
        # single-word edits distinguish the two strings.
        auth_words = auth.split()
        blind_words = blind.split()
        assert abs(len(auth_words) - len(blind_words)) <= 1


def test_further_edit_to_authoritative_text_is_detected_not_silently_accepted():
    population, _ = load_eligible_population()
    blind_by_id = load_blind_review(_real_blind_csv_path())

    corrupted_population = copy.deepcopy(population)
    corrupted_population["SR_disinfo_12"]["source_prompt"] = (
        corrupted_population["SR_disinfo_12"]["source_prompt"] + " EXTRA"
    )
    with pytest.raises(InvariantError):
        verify_mapping(corrupted_population, blind_by_id)


def test_further_edit_to_blind_csv_text_is_detected_not_silently_accepted():
    population, _ = load_eligible_population()
    blind_by_id = load_blind_review(_real_blind_csv_path())

    corrupted_blind = copy.deepcopy(blind_by_id)
    corrupted_blind["R065"]["source_prompt"] = (
        corrupted_blind["R065"]["source_prompt"] + " EXTRA"
    )
    with pytest.raises(InvariantError):
        verify_mapping(population, corrupted_blind)


def _real_blind_csv_path():
    from src.data_pipeline.build_r104_human_review_packet import REPO_ROOT

    return REPO_ROOT / "data/review/r104_human_review_blind.csv"
