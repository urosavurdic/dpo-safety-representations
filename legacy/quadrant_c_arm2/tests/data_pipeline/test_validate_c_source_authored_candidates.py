"""
Tests for src/data_pipeline/validate_c_source_authored_candidates.py.

Deliberately does NOT depend on the real source CSVs, the real 155-row
C-paired pool, or the real training files (all of which may be absent in a
CI/sandbox checkout) - exercises the pure re-verification logic against
small synthetic record lists instead, mirroring the convention already
established in tests/data_pipeline/test_build_c_source_authored_candidates.py.
"""
from src.data_pipeline.build_c_source_authored_candidates import (
    normalized_sha256,
    stripped_sha256,
)
from legacy.quadrant_c_arm2.src.data_pipeline.validate_c_source_authored_candidates import (
    contamination_validation,
    prompt_preservation_validation,
    structural_classifier_validation,
    validate_exact_and_normalized_duplicates,
    validate_overlaps,
)


def _mk_candidate(record_id, text, exact_status="unique", exact_of=None,
                   norm_status="unique", norm_of=None, source="StrongREJECT",
                   source_row_index=0, standalone="complete",
                   classifier_label="complete_user_facing_prompt",
                   universe_status="eligible_for_3a3",
                   c_overlap=False, a_overlap=False,
                   contamination_exact_status=None):
    return {
        "record_id": record_id,
        "source_dataset": source,
        "source_row_index": source_row_index,
        "prompt_text": text,
        "prompt_sha256": stripped_sha256(text),
        "prompt_normalized_sha256": normalized_sha256(text),
        "exact_duplicate_status": exact_status,
        "exact_duplicate_canonical_record_id": exact_of,
        "normalized_duplicate_status": norm_status,
        "normalized_duplicate_canonical_record_id": norm_of,
        "standalone_status": standalone,
        "structural_classifier_label": classifier_label,
        "structural_classifier_low_confidence": False,
        "candidate_universe_status": universe_status,
        "c_paired_overlap": c_overlap,
        "quadrant_a_overlap": a_overlap,
        "contamination_exact_status": contamination_exact_status or {},
    }


# ── exact/normalized duplicate re-verification ──────────────────────────────────────────
def test_exact_duplicate_validation_passes_when_consistent():
    records = [
        _mk_candidate("A1", "Same text."),
        _mk_candidate("A2", "Same text.", exact_status="duplicate", exact_of="A1"),
    ]
    out = validate_exact_and_normalized_duplicates(records)
    assert out["exact_duplicate_validation"]["status"] == "pass"
    assert out["exact_duplicate_validation"]["recomputed_duplicate_count"] == 1


def test_exact_duplicate_validation_flags_wrong_canonical():
    records = [
        _mk_candidate("A1", "Same text."),
        _mk_candidate("A2", "Same text.", exact_status="duplicate", exact_of="WRONG_ID"),
    ]
    out = validate_exact_and_normalized_duplicates(records)
    assert out["exact_duplicate_validation"]["status"] == "fail"
    assert len(out["exact_duplicate_validation"]["mismatches"]) == 1


def test_exact_duplicate_validation_flags_undetected_duplicate():
    records = [
        _mk_candidate("A1", "Same text."),
        _mk_candidate("A2", "Same text.", exact_status="unique"),  # should have been "duplicate"
    ]
    out = validate_exact_and_normalized_duplicates(records)
    assert out["exact_duplicate_validation"]["status"] == "fail"


def test_exact_duplicate_validation_flags_stale_hash():
    r = _mk_candidate("A1", "Original text.")
    r["prompt_sha256"] = "deliberately-wrong-hash"
    out = validate_exact_and_normalized_duplicates([r])
    assert out["exact_duplicate_validation"]["status"] == "fail"
    assert "recomputed hash" in out["exact_duplicate_validation"]["mismatches"][0]["issue"]


def test_normalized_duplicate_validation_passes_when_consistent():
    records = [
        _mk_candidate("B1", "Hello World"),
        _mk_candidate("B2", "  hello   world  ", norm_status="duplicate", norm_of="B1"),
    ]
    out = validate_exact_and_normalized_duplicates(records)
    assert out["normalized_duplicate_validation"]["status"] == "pass"
    assert out["normalized_duplicate_validation"]["recomputed_duplicate_count"] == 1


def test_normalized_duplicate_validation_respects_not_applicable_for_exact_dupes():
    records = [
        _mk_candidate("C1", "Same text."),
        _mk_candidate(
            "C2", "Same text.",
            exact_status="duplicate", exact_of="C1",
            norm_status="not_applicable_already_exact_duplicate",
        ),
    ]
    out = validate_exact_and_normalized_duplicates(records)
    assert out["normalized_duplicate_validation"]["status"] == "pass"
    assert out["normalized_duplicate_validation"]["recomputed_duplicate_count"] == 0


# ── overlap re-verification ──────────────────────────────────────────────────────────────
def test_validate_overlaps_detects_true_overlap_and_matches_recorded_flags():
    cpaired_rows = [{"record_id": "SR_x", "source_prompt": "Overlapping prompt."}]
    qa_texts = ["Another overlapping one."]
    records = [
        _mk_candidate("X1", "Overlapping prompt.", c_overlap=True),
        _mk_candidate("X2", "Another overlapping one.", a_overlap=True),
        _mk_candidate("X3", "Not in either pool."),
    ]
    out = validate_overlaps(records, cpaired_rows, qa_texts)
    assert out["status"] == "pass"
    assert out["c_paired_overlap_count"] == 1
    assert out["quadrant_a_overlap_count"] == 1


def test_validate_overlaps_flags_missed_overlap():
    cpaired_rows = [{"record_id": "SR_x", "source_prompt": "Overlapping prompt."}]
    records = [_mk_candidate("X1", "Overlapping prompt.", c_overlap=False)]  # should be True
    out = validate_overlaps(records, cpaired_rows, [])
    assert out["status"] == "fail"
    assert len(out["mismatches"]) == 1


# ── exact prompt preservation ────────────────────────────────────────────────────────────
def test_prompt_preservation_passes_when_byte_identical():
    sr_rows = [{"forbidden_prompt": "Do the thing."}]
    sst_rows = [{"prompts_final": "Do another thing."}]
    records = [
        _mk_candidate("SR1", "Do the thing.", source="StrongREJECT", source_row_index=0),
        _mk_candidate("SST1", "Do another thing.", source="SimpleSafetyTests", source_row_index=0),
    ]
    out = prompt_preservation_validation(records, sr_rows, sst_rows)
    assert out["status"] == "pass"
    assert out["rows_checked"] == 2
    assert out["mismatches"] == []


def test_prompt_preservation_flags_any_edit():
    sr_rows = [{"forbidden_prompt": "Original wording."}]
    records = [_mk_candidate("SR1", "Edited wording.", source="StrongREJECT", source_row_index=0)]
    out = prompt_preservation_validation(records, sr_rows, [])
    assert out["status"] == "fail"
    assert out["mismatches"][0]["record_id"] == "SR1"


# ── structural classifier: determinism + no silent promotion of ambiguous rows ──────────
def test_structural_classifier_validation_passes_for_consistent_complete_prompt():
    records = [_mk_candidate("Q1", "Could you help me write a poem?")]
    out = structural_classifier_validation(records)
    assert out["status"] == "pass"
    assert out["classifier_determinism_mismatches"] == []
    assert out["non_standalone_silently_promoted"] == []


def test_structural_classifier_validation_flags_silent_promotion_of_non_standalone():
    records = [_mk_candidate(
        "Q2", "Give a detailed step-by-step guide for accomplishing some task",
        standalone="excluded_not_standalone",
        classifier_label="behavior_description",
        universe_status="eligible_for_3a3",  # should never happen if excluded correctly
    )]
    out = structural_classifier_validation(records)
    assert out["status"] == "fail"
    assert "Q2" in out["non_standalone_silently_promoted"]


def test_structural_classifier_validation_flags_recorded_label_drift():
    records = [_mk_candidate(
        "Q3", "Could you help me write a poem?",
        classifier_label="behavior_description",  # wrong on purpose
        standalone="excluded_not_standalone",
        universe_status="excluded",
    )]
    out = structural_classifier_validation(records)
    assert out["status"] == "fail"
    assert len(out["classifier_determinism_mismatches"]) == 1


# ── contamination: exact re-verification (near-dup path exercised via model=None) ───────
def test_contamination_validation_exact_reverification_passes_when_consistent():
    texts_by_file = {"sft_helpful.jsonl": ["some training prompt"], "dpo_pairs.jsonl": None}
    records = [_mk_candidate(
        "K1", "Some Training Prompt",  # matches after normalization
        contamination_exact_status={
            "sft_helpful.jsonl": "matched",
            "dpo_pairs.jsonl": "unknown_training_file_missing",
        },
    )]
    out = contamination_validation(records, texts_by_file, model=None, load_error="no network")
    assert out["exact_status"] == "pass"
    assert out["exact_contamination_counts_by_file"] == {"sft_helpful.jsonl": 1}
    assert out["near_status_by_file"]["dpo_pairs.jsonl"] == "unavailable_file_missing"
    assert out["near_status_by_file"]["sft_helpful.jsonl"] == "unknown"


def test_contamination_validation_never_reports_unavailable_as_clean():
    texts_by_file = {"dpo_pairs.jsonl": None}
    records = [_mk_candidate("K2", "Anything.", contamination_exact_status={
        "dpo_pairs.jsonl": "unknown_training_file_missing",
    })]
    out = contamination_validation(records, texts_by_file, model=None, load_error="offline")
    assert out["near_status_by_file"]["dpo_pairs.jsonl"] != "clean"
    assert out["near_status_by_file"]["dpo_pairs.jsonl"] == "unavailable_file_missing"


def test_contamination_validation_flags_exact_status_mismatch():
    texts_by_file = {"sft_helpful.jsonl": ["a training prompt"]}
    records = [_mk_candidate(
        "K3", "a training prompt",
        contamination_exact_status={"sft_helpful.jsonl": "clean"},  # should be "matched"
    )]
    out = contamination_validation(records, texts_by_file, model=None, load_error="offline")
    assert out["exact_status"] == "fail"
    assert len(out["exact_mismatches"]) == 1
