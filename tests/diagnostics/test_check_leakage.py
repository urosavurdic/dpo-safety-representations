import numpy as np

from src.diagnostics.check_leakage import (
    normalize,
    find_exact_duplicates,
    flag_near_duplicates_from_similarity,
)


def test_normalize_handles_case_and_whitespace():
    assert normalize("  How Do I Do X?  ") == "how do i do x?"


def test_find_exact_duplicates_detects_case_insensitive_match():
    eval_prompts = ["How do I do X?", "Something unique"]
    train_prompts = ["how do i do x?", "unrelated prompt"]
    assert find_exact_duplicates(eval_prompts, train_prompts) == ["How do I do X?"]


def test_find_exact_duplicates_no_false_positives():
    eval_prompts = ["A completely different question"]
    train_prompts = ["Something else entirely"]
    assert find_exact_duplicates(eval_prompts, train_prompts) == []


def test_flag_near_duplicates_respects_threshold():
    eval_prompts = ["eval1", "eval2"]
    train_prompts = ["train1", "train2"]
    sim_matrix = np.array([[0.95, 0.40], [0.30, 0.20]])
    flagged = flag_near_duplicates_from_similarity(eval_prompts, train_prompts, sim_matrix, threshold=0.9)
    assert len(flagged) == 1
    assert flagged[0]["eval_prompt"] == "eval1"
    assert flagged[0]["closest_train_prompt"] == "train1"
    assert flagged[0]["similarity"] == 0.95


def test_flag_near_duplicates_empty_when_below_threshold():
    sim_matrix = np.array([[0.5]])
    flagged = flag_near_duplicates_from_similarity(["eval1"], ["train1"], sim_matrix, threshold=0.9)
    assert flagged == []


def test_flag_near_duplicates_catches_multiple_matches_per_eval_prompt():
    eval_prompts = ["eval1"]
    train_prompts = ["train1", "train2", "train3"]
    sim_matrix = np.array([[0.95, 0.92, 0.40]])
    flagged = flag_near_duplicates_from_similarity(eval_prompts, train_prompts, sim_matrix, threshold=0.9)
    assert len(flagged) == 2  # both train1 and train2 should be caught, not just the best
    matched = {f["closest_train_prompt"] for f in flagged}
    assert matched == {"train1", "train2"}

    