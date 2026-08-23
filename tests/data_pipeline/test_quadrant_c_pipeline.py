import pytest

from src.data_pipeline.quadrant_c_pipeline import (
    CANDIDATES,
    classify_and_prescreen,
    verify_source_prompts_are_real,
)


def test_candidates_have_unique_source_ids():
    ids = [c["source_id"] for c in CANDIDATES]
    assert len(ids) == len(set(ids))


def test_candidates_have_all_required_fields():
    required = {"source_id", "source_category", "project_category", "source_prompt",
                "candidate_prompt", "harmful_objective", "requested_assistance_type", "evasion_dominant"}
    for c in CANDIDATES:
        assert set(c.keys()) == required
        assert isinstance(c["source_prompt"], str) and len(c["source_prompt"]) > 0
        assert isinstance(c["candidate_prompt"], str) and len(c["candidate_prompt"]) > 0


def test_verify_source_prompts_are_real_catches_truncated_text():
    # Regression test for a real bug this caught: an earlier session stored
    # truncated previews as if they were the full source_prompt.
    class FakeDF:
        pass
    import pandas as pd
    df = pd.DataFrame({"forbidden_prompt": ["The full, real, untruncated forbidden prompt text goes here."]})
    candidates = [{"source_id": "X", "source_prompt": "The full, real, untruncated..."}]  # truncated
    missing = verify_source_prompts_are_real(candidates, df)
    assert missing == ["X"]


def test_verify_source_prompts_are_real_passes_for_verbatim_match():
    import pandas as pd
    df = pd.DataFrame({"forbidden_prompt": ["Exact text."]})
    candidates = [{"source_id": "X", "source_prompt": "Exact text."}]
    assert verify_source_prompts_are_real(candidates, df) == []


def test_classify_excludes_on_exact_contamination():
    contamination = {"sft_helpful.jsonl": {"exact": ["What's a way to test this?"], "near": []}}
    candidate = {
        "source_id": "TEST_01", "source_category": "x", "project_category": "illegal",
        "source_prompt": "source", "candidate_prompt": "What's a way to test this?",
        "harmful_objective": "obj", "requested_assistance_type": "planning", "evasion_dominant": False,
    }
    record = classify_and_prescreen(candidate, contamination)
    assert record["agent_pre_screen"] == "exclude"
    assert "contamination" in record["agent_reason"]


def test_classify_excludes_on_near_contamination():
    contamination = {"sft_helpful.jsonl": {"exact": [], "near": [
        {"eval_prompt": "What's a way to test this?", "closest_train_prompt": "similar", "similarity": 0.95}
    ]}}
    candidate = {
        "source_id": "TEST_02", "source_category": "x", "project_category": "illegal",
        "source_prompt": "source", "candidate_prompt": "What's a way to test this?",
        "harmful_objective": "obj", "requested_assistance_type": "planning", "evasion_dominant": False,
    }
    record = classify_and_prescreen(candidate, contamination)
    assert record["agent_pre_screen"] == "exclude"


def test_classify_routes_evasion_dominant_to_secondary_not_primary():
    contamination = {"sft_helpful.jsonl": {"exact": [], "near": []}}
    candidate = {
        "source_id": "TEST_03", "source_category": "x", "project_category": "illegal",
        "source_prompt": "source", "candidate_prompt": "A perfectly clean, cue-free wording",
        "harmful_objective": "obj", "requested_assistance_type": "planning", "evasion_dominant": True,
    }
    record = classify_and_prescreen(candidate, contamination)
    assert record["agent_pre_screen"] == "secondary_only"
    assert record["evasion_dominant"] is True


def test_classify_routes_unresolved_cue_level_to_secondary():
    contamination = {"sft_helpful.jsonl": {"exact": [], "near": []}}
    candidate = {
        "source_id": "TEST_04", "source_category": "x", "project_category": "illegal",
        "source_prompt": "source", "candidate_prompt": "unauthorized access without permission",  # 2 lexicon hits: "unauthorized", "without permission"
        "harmful_objective": "obj", "requested_assistance_type": "planning", "evasion_dominant": False,
    }
    record = classify_and_prescreen(candidate, contamination)
    assert record["surface_cue_level"] == "unresolved"
    assert record["agent_pre_screen"] == "secondary_only"


def test_classify_eligible_when_clean():
    contamination = {"sft_helpful.jsonl": {"exact": [], "near": []}}
    candidate = {
        "source_id": "TEST_05", "source_category": "x", "project_category": "illegal",
        "source_prompt": "source", "candidate_prompt": "A perfectly clean, cue-free wording",
        "harmful_objective": "obj", "requested_assistance_type": "planning", "evasion_dominant": False,
    }
    record = classify_and_prescreen(candidate, contamination)
    assert record["agent_pre_screen"] == "eligible_candidate"
    assert record["surface_cue_level"] == "neutral"


def test_classify_never_invents_or_drops_source_fields():
    contamination = {"sft_helpful.jsonl": {"exact": [], "near": []}}
    candidate = {
        "source_id": "TEST_06", "source_category": "cat_x", "project_category": "illegal",
        "source_prompt": "the real source text", "candidate_prompt": "the reworded text",
        "harmful_objective": "obj", "requested_assistance_type": "planning", "evasion_dominant": False,
    }
    record = classify_and_prescreen(candidate, contamination)
    assert record["source_prompt"] == "the real source text"
    assert record["source_category"] == "cat_x"
    assert record["candidate_id"] == "TEST_06"


def test_real_candidates_match_live_strongreject_source():
    # NOTE: network-dependent (fetches the real StrongREJECT CSV) - matches
    # this project's existing convention for check_leakage-adjacent tests.
    # Regression coverage for a real bug this caught once already: an
    # earlier session stored truncated previews as if they were the full
    # source_prompt for 6/20 candidates.
    from src.data_pipeline.quadrant_c_pipeline import fetch_strongreject
    df = fetch_strongreject()
    missing = verify_source_prompts_are_real(CANDIDATES, df)
    assert missing == []


def test_build_matched_pairs_produces_two_rows_per_candidate():
    from src.data_pipeline.quadrant_c_pipeline import build_matched_pairs
    fake_records = [
        {"candidate_id": "X1", "source_prompt": "overt version", "candidate_prompt": "reduced version", "harmful_objective": "obj1"},
        {"candidate_id": "X2", "source_prompt": "overt version 2", "candidate_prompt": "reduced version 2", "harmful_objective": "obj2"},
    ]
    pairs = build_matched_pairs(fake_records)
    assert len(pairs) == 4
    x1_pairs = [p for p in pairs if p["pair_id"] == "X1"]
    assert len(x1_pairs) == 2
    variants = {p["variant"] for p in x1_pairs}
    assert variants == {"source_overt", "candidate_reduced_cue"}
    prompts = {p["prompt"] for p in x1_pairs}
    assert prompts == {"overt version", "reduced version"}


def test_build_matched_pairs_preserves_pairing_via_shared_pair_id():
    # The whole point of this structure: every pair_id appears exactly
    # twice, once per variant, so a downstream analysis can group by
    # pair_id and compute a WITHIN-pair difference rather than treating
    # the two arms as independent groups.
    from src.data_pipeline.quadrant_c_pipeline import build_matched_pairs
    fake_records = [{"candidate_id": f"X{i}", "source_prompt": f"s{i}", "candidate_prompt": f"c{i}", "harmful_objective": f"o{i}"}
                     for i in range(5)]
    pairs = build_matched_pairs(fake_records)
    from collections import Counter
    counts = Counter(p["pair_id"] for p in pairs)
    assert all(v == 2 for v in counts.values())
    assert len(counts) == 5


def test_build_matched_pairs_empty_input():
    from src.data_pipeline.quadrant_c_pipeline import build_matched_pairs
    assert build_matched_pairs([]) == []
