"""
Tests for src/data_pipeline/validate_c_source_authored_review_queue.py.

Two layers:
  - unit tests against synthetic rows, proving each check function actually
    catches the violation it claims to (not just that it passes on
    already-good data);
  - one integration test that runs the real script against the actual
    committed queue/inputs end-to-end, mirroring how a researcher or CI
    run would invoke it.
"""
import json
import subprocess
import sys
from pathlib import Path

from src.data_pipeline.validate_c_source_authored_review_queue import (
    check_construction_identity,
    check_contamination_and_overlap,
    check_population_relationship,
    check_provenance_fields,
    check_review_status,
    check_row_identity,
    check_schema,
)
from src.data_pipeline.score_and_queue_c_source_authored import CSV_FIELDNAMES

REPO_ROOT = Path(__file__).resolve().parents[2]


def _good_row(**overrides):
    row = {
        "arm": "c_source_authored",
        "record_id": "CSA_SR_abc123",
        "candidate_id": "CSA_SR_abc123",
        "pair_id": "",
        "c_construction": "c_source_authored",
        "source_dataset": "StrongREJECT",
        "source_repository": "alexandrasouly/strongreject",
        "source_row_index": "5",
        "original_source_row_id": "",
        "source_topic_category": "Illegal goods and services",
        "project_category": "",
        "domain": "",
        "prompt_function": "",
        "source_prompt": "text",
        "candidate_prompt": "text",
        "scored_prompt": "text",
        "word_count": "10",
        "character_count": "50",
        "fightin_words_score_unnormalized": "1.0",
        "fightin_words_score_normalized": "0.5",
        "fw_z_score": "0.1",
        "fw_token_recognition_fraction": "0.9",
        "low_coverage_flag": "False",
        "global_rank": "0.1",
        "source_rank": "0.1",
        "in_Q10": "True",
        "in_Q25": "True",
        "in_Q40": "True",
        "review_stratum": "Q25",
        "contamination_status": "checked_exact_zero_near_unknown",
        "overlap_status": "no_c_paired_or_quadrant_a_overlap",
        "structural_classifier_label": "complete_user_facing_prompt",
        "classifier_status": "confirmed",
        "provenance_class": "source_dataset_direct",
        "provenance_notes": "",
        "source_revision": "deadbeef",
        "source_file_sha256": "filehash",
        "prompt_sha256": "prompthash",
        "review_status": "pending",
        "review_notes": "",
    }
    row.update(overrides)
    return row


def test_check_schema_passes_on_real_fieldnames():
    assert check_schema(CSV_FIELDNAMES) == []


def test_check_schema_catches_drift():
    problems = check_schema(["arm", "record_id"])
    assert problems


def test_check_row_identity_catches_duplicates():
    rows = [_good_row(record_id="A"), _good_row(record_id="A")]
    problems = check_row_identity(rows, expected_count=2)
    assert any("duplicate" in p for p in problems)


def test_check_row_identity_catches_count_mismatch():
    rows = [_good_row(record_id="A")]
    problems = check_row_identity(rows, expected_count=52)
    assert any("52" in p for p in problems)


def test_check_review_status_catches_non_pending():
    rows = [_good_row(review_status="accept")]
    problems = check_review_status(rows)
    assert any("pending" in p for p in problems)


def test_check_review_status_catches_stray_notes():
    rows = [_good_row(review_notes="looks fine")]
    problems = check_review_status(rows)
    assert any("review_notes" in p for p in problems)


def test_check_construction_identity_catches_wrong_construction():
    rows = [_good_row(c_construction="c_paired")]
    problems = check_construction_identity(rows)
    assert any("c_construction" in p for p in problems)


def test_check_construction_identity_catches_nonempty_pair_id():
    rows = [_good_row(pair_id="some-pair")]
    problems = check_construction_identity(rows)
    assert any("pair_id" in p for p in problems)


def test_check_contamination_and_overlap_catches_flagged_row():
    rows = [_good_row(contamination_status="contamination_flagged")]
    problems = check_contamination_and_overlap(rows)
    assert any("contamination_status" in p for p in problems)


def test_check_contamination_and_overlap_catches_overlap():
    rows = [_good_row(overlap_status="overlap_present(c_paired=True,quadrant_a=False)")]
    problems = check_contamination_and_overlap(rows)
    assert any("overlap_status" in p for p in problems)


def test_check_provenance_fields_catches_missing_field():
    rows = [_good_row(source_dataset="")]
    problems = check_provenance_fields(rows, eligible_by_id={})
    assert any("source_dataset" in p for p in problems)


def test_check_population_relationship_catches_unknown_record():
    rows = [_good_row(record_id="NOT_IN_ELIGIBLE_SET")]
    problems = check_population_relationship(rows, eligible_by_id={})
    assert any("not found" in p for p in problems)


def test_check_population_relationship_catches_ineligible_status():
    rows = [_good_row(record_id="X")]
    eligible_by_id = {"X": {"candidate_universe_status": "excluded_low_confidence"}}
    problems = check_population_relationship(rows, eligible_by_id)
    assert any("excluded_low_confidence" in p for p in problems)


def test_end_to_end_validation_passes_on_real_committed_queue(tmp_path):
    """
    Integration test: runs the real script against the real committed
    queue, 3A3-validated input, and 3A4 log, exactly as a researcher would.
    """
    out_json = tmp_path / "milestone_3b.json"
    out_md = tmp_path / "milestone_3b.md"
    result = subprocess.run(
        [
            sys.executable, "-m",
            "src.data_pipeline.validate_c_source_authored_review_queue",
            "--out-log-json", str(out_json),
            "--out-log-md", str(out_md),
        ],
        cwd=REPO_ROOT, capture_output=True, text=True,
    )
    assert result.returncode == 0, f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    assert out_json.exists() and out_md.exists()

    with open(out_json) as f:
        summary = json.load(f)
    assert summary["overall_pass"] is True
    assert summary["queue_csv"]["row_count"] == 52
    assert summary["reproduction_check"]["byte_identical"] is True
    for name, result_ in summary["checks"].items():
        assert result_["pass"] is True, f"{name} failed: {result_['problems']}"
