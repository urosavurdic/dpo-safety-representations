from src.data_pipeline.build_eval_set import build_quadrant_c, QUADRANT_C_PROMPTS, assign_direction_split


def test_assign_direction_split_ratio_and_determinism():
    quadrant_a = [{"prompt": f"a{i}", "quadrant": "A"} for i in range(150)]
    quadrant_d = [{"prompt": f"d{i}", "quadrant": "D"} for i in range(150)]
    a1, d1 = assign_direction_split(quadrant_a, quadrant_d)
    a2, _ = assign_direction_split(quadrant_a, quadrant_d)
    assert a1 == a2  # deterministic given the fixed default seed

    from collections import Counter
    a_counts = Counter(r["split"] for r in a1)
    assert a_counts["direction_estimation"] == 120
    assert a_counts["held_out_behavioral"] == 30


def test_assign_direction_split_no_overlap_and_full_coverage():
    quadrant_a = [{"prompt": f"a{i}", "quadrant": "A"} for i in range(50)]
    quadrant_d = [{"prompt": f"d{i}", "quadrant": "D"} for i in range(50)]
    a_split, _ = assign_direction_split(quadrant_a, quadrant_d)
    est = {r["prompt"] for r in a_split if r["split"] == "direction_estimation"}
    held = {r["prompt"] for r in a_split if r["split"] == "held_out_behavioral"}
    assert est.isdisjoint(held)
    assert est | held == {r["prompt"] for r in quadrant_a}


def test_assign_direction_split_respects_train_frac():
    quadrant_a = [{"prompt": f"a{i}", "quadrant": "A"} for i in range(100)]
    quadrant_d = [{"prompt": f"d{i}", "quadrant": "D"} for i in range(100)]
    a_split, _ = assign_direction_split(quadrant_a, quadrant_d, train_frac=0.7)
    from collections import Counter
    counts = Counter(r["split"] for r in a_split)
    assert counts["direction_estimation"] == 70
    assert counts["held_out_behavioral"] == 30


def test_quadrant_c_shape():
    records = build_quadrant_c()
    assert len(records) == len(QUADRANT_C_PROMPTS)
    for r in records:
        assert set(r.keys()) == {"prompt", "quadrant", "source", "category"}
        assert r["quadrant"] == "C"
        assert isinstance(r["prompt"], str) and len(r["prompt"]) > 0


def test_quadrant_c_prompts_are_unique():
    assert len(set(QUADRANT_C_PROMPTS)) == len(QUADRANT_C_PROMPTS)


def test_quadrant_c_records_have_real_category_and_source():
    # Regression guard: the pre-promotion version used a generic
    # "hand-curated"/None placeholder for every item, throwing away the
    # provenance the whole pipeline exists to preserve. Promoted records
    # must carry the real project_category and a real source attribution.
    from src.data_pipeline.build_eval_set import QUADRANT_C_RECORDS
    records = build_quadrant_c()
    assert len(records) == 104
    valid_categories = {"illegal", "cybercrime_intrusion", "harassment_bullying", "misinformation_disinformation"}
    for r in records:
        assert r["category"] in valid_categories
        assert "StrongREJECT" in r["source"]
    for r in QUADRANT_C_RECORDS:
        assert "candidate_id" in r and r["candidate_id"].startswith("SR_")


def test_quadrant_c_scores_clean_on_lexical_risk_lexicon():
    # This is the whole point of the promotion - the OLD 20 scored
    # mean_cue_hits=0.550 (40% had a hit). The promoted 15 should score 0.
    from src.diagnostics.score_lexical_risk_cues import score_prompt
    scores = [score_prompt(p)[0] for p in QUADRANT_C_PROMPTS]
    assert sum(scores) == 0
