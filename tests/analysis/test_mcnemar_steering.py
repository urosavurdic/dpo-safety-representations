from src.analysis.mcnemar_steering import build_contingency


def test_build_contingency_counts_matched_prompts_only():
    rows = [
        {"prompt": "p1", "stage": "M3_baseline", "response": "Sure, here's how."},
        {"prompt": "p1", "stage": "M3_steered", "response": "I can't help with that."},
        {"prompt": "p2", "stage": "M3_baseline", "response": "Sure, here's how."},
        {"prompt": "p2", "stage": "M3_steered", "response": "Sure, here's how."},
        {"prompt": "p3", "stage": "M3_baseline", "response": "Sure, here's how."},  # unmatched, no M3_steered row
    ]
    table, n_paired = build_contingency(rows, category="refusal")
    assert n_paired == 2  # p3 excluded, unmatched
    assert table == [[0, 1], [0, 1]]  # p1: no->yes(steered), p2: no->no