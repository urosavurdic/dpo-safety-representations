"""
Tests for src/data_pipeline/score_and_queue_c_source_authored.py.

Deliberately does NOT depend on the real 3A3 artifact, the real eval set, or
network access - exercises the pure scoring/ranking/queue-building logic
against a small synthetic FightinWords fit and synthetic candidate records,
mirroring the convention already established in
tests/data_pipeline/test_validate_c_source_authored_candidates.py.
"""
from src.corpus_discrimination import FightinWords, assign_strata, empirical_rank
from legacy.quadrant_c_arm2.src.data_pipeline.score_and_queue_c_source_authored import (
    build_queue_row,
    classifier_status_label,
    contamination_status_label,
    overlap_status_label,
    review_stratum_label,
    score_and_rank,
    select_review_queue,
    validate_input_contract,
    validate_queue,
    verify_3a3_artifact_hash,
)


def _mk_record(record_id, text, source="StrongREJECT",
                c_overlap=False, a_overlap=False,
                universe_status="eligible_for_3a3",
                low_confidence=False,
                contamination_exact=None):
    return {
        "record_id": record_id,
        "c_construction": "c_source_authored",
        "source_dataset": source,
        "source_repository": "org/repo",
        "source_revision": "deadbeef",
        "source_file_sha256": "filehash",
        "prompt_text": text,
        "prompt_sha256": f"hash-{record_id}",
        "candidate_universe_status": universe_status,
        "structural_classifier_label": "complete_user_facing_prompt",
        "structural_classifier_low_confidence": low_confidence,
        "c_paired_overlap": c_overlap,
        "quadrant_a_overlap": a_overlap,
        "contamination_exact_status": contamination_exact
        if contamination_exact is not None
        else {"dpo_pairs.jsonl": "clean"},
        "provenance_class": "upstream-derived",
        "provenance_notes": "note",
        "source_row_index": 0,
        "original_source_row_id": None,
        "source_topic_category": "cat",
        "project_category": None,
        "validation_3a3": {
            "contamination_near_status": {"dpo_pairs.jsonl": "unknown"},
        },
    }


def _fitted_fw():
    # H strongly favors words like "hack"/"bomb"; D favors benign words.
    fw = FightinWords(prior_strength=0.01, min_count=1)
    fw.fit(
        h_texts=["hack the bomb system", "build a bomb device", "hack passwords"] * 5,
        d_texts=["bake a cake today", "walk the dog outside", "read a good book"] * 5,
    )
    return fw


# ── input contract ───────────────────────────────────────────────────────────
def test_validate_input_contract_passes_on_well_formed_records():
    records = [_mk_record("R1", "Some prompt."), _mk_record("R2", "Another prompt.")]
    assert validate_input_contract(records) == []


def test_validate_input_contract_flags_missing_field():
    r = _mk_record("R1", "Some prompt.")
    del r["provenance_class"]
    assert any("missing field provenance_class" in p for p in validate_input_contract([r]))


def test_validate_input_contract_flags_duplicate_record_id():
    records = [_mk_record("R1", "A."), _mk_record("R1", "B.")]
    assert any("duplicate record_id" in p for p in validate_input_contract(records))


def test_validate_input_contract_flags_empty_prompt_for_eligible_row():
    r = _mk_record("R1", "   ")
    assert any("empty prompt_text" in p for p in validate_input_contract([r]))


def test_verify_3a3_artifact_hash_passes_when_matching(tmp_path):
    log_path = tmp_path / "3a3_validation.json"
    log_path.write_text(
        '{"output_artifacts": {"validated_candidate_universe_jsonl": {"sha256": "abc123"}}}'
    )
    assert verify_3a3_artifact_hash(tmp_path / "in.jsonl", "abc123", log_path) == []


def test_verify_3a3_artifact_hash_flags_mismatch(tmp_path):
    log_path = tmp_path / "3a3_validation.json"
    log_path.write_text(
        '{"output_artifacts": {"validated_candidate_universe_jsonl": {"sha256": "abc123"}}}'
    )
    problems = verify_3a3_artifact_hash(tmp_path / "in.jsonl", "different", log_path)
    assert any("mismatch" in p for p in problems)


# ── scoring and ranking ──────────────────────────────────────────────────────
def test_score_and_rank_orders_more_d_like_candidates_first():
    fw = _fitted_fw()
    records = [
        _mk_record("HACK", "hack the bomb system now"),   # very H-like
        _mk_record("BAKE", "bake a cake and read a book"),  # very D-like
    ]
    scored = score_and_rank(records, fw, min_token_recognition_fraction=0.0)
    # sorted by (global_rank, record_id) ascending == most D-like first
    assert [r["record_id"] for r in scored] == ["BAKE", "HACK"]
    assert scored[0]["_global_rank"] < scored[1]["_global_rank"]


def test_score_and_rank_is_deterministic_across_runs():
    fw = _fitted_fw()
    records = [
        _mk_record("A", "hack passwords for a bomb"),
        _mk_record("B", "bake a cake outside"),
        _mk_record("C", "walk the dog and read"),
    ]
    run1 = score_and_rank(records, fw, 0.0)
    run2 = score_and_rank(records, fw, 0.0)
    assert [r["record_id"] for r in run1] == [r["record_id"] for r in run2]
    assert [r["_global_rank"] for r in run1] == [r["_global_rank"] for r in run2]


def test_source_rank_is_scoped_to_same_source_dataset():
    fw = _fitted_fw()
    records = [
        _mk_record("SR1", "hack the bomb system", source="StrongREJECT"),
        _mk_record("SR2", "bake a cake today", source="StrongREJECT"),
        _mk_record("SST1", "bake a cake today", source="SimpleSafetyTests"),
    ]
    scored = score_and_rank(records, fw, 0.0)
    by_id = {r["record_id"]: r for r in scored}
    # SST1 is alone in its source, so it must rank 1.0 within its own group
    # regardless of how the StrongREJECT rows score.
    assert by_id["SST1"]["_source_rank"] == 1.0


def test_c_candidates_never_enter_the_reference_fit():
    # The reference FightinWords instance is built entirely from H/D texts
    # passed to fit() - candidate prompt_text is only ever passed to score(),
    # never to fit(). This test documents/guards that contract at the
    # call-signature level used by score_and_rank.
    fw = _fitted_fw()
    original_vocab = set(fw.vocab_)
    records = [_mk_record("X", "some totally novel unseen wordforms zzqx")]
    score_and_rank(records, fw, 0.0)
    assert set(fw.vocab_) == original_vocab


# ── label derivation ─────────────────────────────────────────────────────────
def test_review_stratum_label_picks_tightest_stratum():
    assert review_stratum_label({"in_Q10": True, "in_Q25": True, "in_Q40": True}) == "Q10"
    assert review_stratum_label({"in_Q10": False, "in_Q25": True, "in_Q40": True}) == "Q25"
    assert review_stratum_label({"in_Q10": False, "in_Q25": False, "in_Q40": True}) == "Q40"
    assert review_stratum_label({"in_Q10": False, "in_Q25": False, "in_Q40": False}) == "none"


def test_contamination_status_label_matches_c_paired_queue_convention():
    r = _mk_record("R1", "text")
    assert contamination_status_label(r) == "checked_exact_zero_near_unknown"


def test_contamination_status_label_flags_dirty_exact():
    r = _mk_record("R1", "text", contamination_exact={"dpo_pairs.jsonl": "flagged"})
    assert contamination_status_label(r) == "contamination_flagged"


def test_overlap_status_label_clean():
    r = _mk_record("R1", "text", c_overlap=False, a_overlap=False)
    assert overlap_status_label(r) == "no_c_paired_or_quadrant_a_overlap"


def test_overlap_status_label_flags_overlap():
    r = _mk_record("R1", "text", c_overlap=True, a_overlap=False)
    assert "overlap_present" in overlap_status_label(r)


def test_classifier_status_label():
    assert classifier_status_label(_mk_record("R1", "t", low_confidence=True)) == \
        "provisional_low_confidence"
    assert classifier_status_label(_mk_record("R1", "t", low_confidence=False)) == \
        "confirmed"


# ── queue-row construction and equality invariants ──────────────────────────
def test_build_queue_row_preserves_prompt_equality_and_fixed_fields():
    fw = _fitted_fw()
    records = [_mk_record("R1", "bake a cake today")]
    scored = score_and_rank(records, fw, 0.0)
    row = build_queue_row(scored[0])
    assert row["source_prompt"] == row["candidate_prompt"] == row["scored_prompt"] == \
        "bake a cake today"
    assert row["pair_id"] == ""
    assert row["c_construction"] == "c_source_authored"
    assert row["review_status"] == "pending"


# ── selection (stratum + cap) ────────────────────────────────────────────────
def test_select_review_queue_respects_stratum_and_cap():
    fw = _fitted_fw()
    # 5 tied D-like + 8 tied H-like = 13 total. Each D item's empirical rank
    # is 5/13 ≈ 0.385 (<=0.40, in Q40); each H item's rank is 13/13 = 1.0
    # (not in any stratum). So exactly 5 candidates qualify for Q40 - enough
    # to exercise the limit=3 cap.
    records = [_mk_record(f"D{i}", "bake a cake and read a book") for i in range(5)] + \
        [_mk_record(f"H{i}", "hack the bomb system now") for i in range(8)]
    scored = score_and_rank(records, fw, 0.0)
    qualifying = [r for r in scored if r["_strata"]["in_Q40"]]
    assert len(qualifying) == 5
    selected = select_review_queue(scored, "Q40", limit=3)
    assert len(selected) == 3
    assert all(r["record_id"].startswith("D") for r in selected)
    # Deterministic: same call twice gives the same rows in the same order.
    selected2 = select_review_queue(scored, "Q40", limit=3)
    assert [r["record_id"] for r in selected] == [r["record_id"] for r in selected2]


def test_select_review_queue_keeps_smaller_count_without_topup():
    fw = _fitted_fw()
    # 1 D-like + 9 tied H-like = 10 total. The D item's empirical rank is
    # 1/10 = 0.10 (<=0.10, in Q10); each H item's rank is 10/10 = 1.0.
    # Only one candidate qualifies - the limit (150) must not pull in any
    # weaker H-like candidate to top up the queue.
    records = [_mk_record("ONLY_D", "bake a cake and read a book")] + \
        [_mk_record(f"H{i}", "hack the bomb system now") for i in range(9)]
    scored = score_and_rank(records, fw, 0.0)
    selected = select_review_queue(scored, "Q10", limit=150)
    assert [r["record_id"] for r in selected] == ["ONLY_D"]


# ── queue validation ──────────────────────────────────────────────────────────
def _scored_with_padding(target_record, padding_count=8):
    """R1's own rank depends on what it's compared against; pad the
    reference set with clearly-H-like filler so a D-like target record lands
    safely inside Q40 (see the rank arithmetic in the selection tests above)."""
    fw = _fitted_fw()
    records = [target_record] + \
        [_mk_record(f"PAD{i}", "hack the bomb system now") for i in range(padding_count)]
    return fw, score_and_rank(records, fw, 0.0)


def test_validate_queue_passes_for_well_formed_queue():
    _, scored = _scored_with_padding(_mk_record("R1", "bake a cake today"))
    eligible_by_id = {r["record_id"]: r for r in scored}
    target = eligible_by_id["R1"]
    assert target["_strata"]["in_Q40"]  # sanity: rank 1/9 <= 0.40
    queue_rows = [build_queue_row(target)]
    assert validate_queue(queue_rows, eligible_by_id, "Q40") == []


def test_validate_queue_flags_c_paired_overlap_row():
    _, scored = _scored_with_padding(_mk_record("R1", "bake a cake today", c_overlap=True))
    eligible_by_id = {r["record_id"]: r for r in scored}
    queue_rows = [build_queue_row(eligible_by_id["R1"])]
    problems = validate_queue(queue_rows, eligible_by_id, "Q40")
    assert any("c_paired-overlapping row" in p for p in problems)


def test_validate_queue_flags_prompt_mismatch():
    _, scored = _scored_with_padding(_mk_record("R1", "bake a cake today"))
    eligible_by_id = {r["record_id"]: r for r in scored}
    row = build_queue_row(eligible_by_id["R1"])
    row["candidate_prompt"] = "a different prompt"
    problems = validate_queue([row], eligible_by_id, "Q40")
    assert any("prompt not equal" in p for p in problems)
