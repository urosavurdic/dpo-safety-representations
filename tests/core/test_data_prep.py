from src.core.data_prep import build_matched_pairs


def _fake_row(prompt, r0, r1, safe0, safe1, safer_id):
    return {
        "prompt": prompt,
        "response_0": r0,
        "response_1": r1,
        "is_response_0_safe": safe0,
        "is_response_1_safe": safe1,
        "safer_response_id": safer_id,
    }


def test_filters_out_same_safety_rows():
    rows = [
        _fake_row("p1", "resp a", "resp b", True, True, 0),   # both safe -> dropped
        _fake_row("p2", "resp c", "resp d", False, False, 1), # both unsafe -> dropped
    ]
    dpo_pairs, sft_examples = build_matched_pairs(rows, n_target=10)
    assert dpo_pairs == []
    assert sft_examples == []


def test_keeps_safety_contrastive_rows_with_correct_chosen():
    rows = [
        _fake_row("What is X?", "unsafe answer", "safe answer", False, True, safer_id=1),
    ]
    dpo_pairs, sft_examples = build_matched_pairs(rows, n_target=10)
    assert len(dpo_pairs) == 1
    assert dpo_pairs[0]["prompt"] == "What is X?"
    assert dpo_pairs[0]["chosen"] == "safe answer"
    assert dpo_pairs[0]["rejected"] == "unsafe answer"
    assert sft_examples[0]["response"] == "safe answer"
    assert sft_examples[0]["prompt"] == "What is X?"


def test_dpo_and_sft_share_identical_prompts():
    rows = [
        _fake_row("A", "bad", "good", False, True, 1),
        _fake_row("B", "good", "bad", True, False, 0),
    ]
    dpo_pairs, sft_examples = build_matched_pairs(rows, n_target=10)
    dpo_prompts = {r["prompt"] for r in dpo_pairs}
    sft_prompts = {r["prompt"] for r in sft_examples}
    assert dpo_prompts == sft_prompts  # the matched-data guarantee, tested directly


def test_skips_empty_responses():
    rows = [
        _fake_row("p", "", "safe answer", False, True, 1),  # empty rejected -> dropped
    ]
    dpo_pairs, _ = build_matched_pairs(rows, n_target=10)
    assert dpo_pairs == []


def test_respects_n_target():
    rows = [
        _fake_row(f"p{i}", "unsafe", "safe", False, True, 1) for i in range(5)
    ]
    dpo_pairs, _ = build_matched_pairs(rows, n_target=3)
    assert len(dpo_pairs) == 3