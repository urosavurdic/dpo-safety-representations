from src.analysis.mcnemar_causal_ablation import build_paired_outcomes, contingency_table, main


def test_contingency_table_counts_correctly():
    pairs = [(True, True), (True, False), (True, False), (False, True), (False, False), (False, False)]
    assert contingency_table(pairs) == [[1, 2], [1, 2]]


def test_build_paired_outcomes_pairs_by_prompt_within_quadrant(monkeypatch):
    import src.analysis.mcnemar_causal_ablation as mca
    monkeypatch.setattr(mca, "classify_completion", lambda text: text)  # identity: response IS the category

    rows = [
        {"prompt": "p1", "quadrant": "D", "stage": "steer_baseline", "response": "comply"},
        {"prompt": "p1", "quadrant": "D", "stage": "steer_steered", "response": "refusal"},
        {"prompt": "p2", "quadrant": "D", "stage": "steer_baseline", "response": "comply"},
        {"prompt": "p2", "quadrant": "D", "stage": "steer_steered", "response": "comply"},
        {"prompt": "p3", "quadrant": "A", "stage": "steer_baseline", "response": "refusal"},  # wrong quadrant, excluded
        {"prompt": "p3", "quadrant": "A", "stage": "steer_steered", "response": "refusal"},
    ]
    pairs = build_paired_outcomes(rows, "D", "refusal", "steer_baseline", "steer_steered")
    assert sorted(pairs) == sorted([(False, True), (False, False)])


def test_build_paired_outcomes_drops_prompts_missing_either_condition(monkeypatch):
    import src.analysis.mcnemar_causal_ablation as mca
    monkeypatch.setattr(mca, "classify_completion", lambda text: text)

    rows = [
        {"prompt": "p1", "quadrant": "D", "stage": "steer_baseline", "response": "comply"},
        # p1 has no "steer_steered" row -> must be dropped, not silently paired with something else
    ]
    pairs = build_paired_outcomes(rows, "D", "refusal", "steer_baseline", "steer_steered")
    assert pairs == []


def test_main_runs_end_to_end_without_crashing(tmp_path, monkeypatch, capsys):
    """Regression test for the real bug this fixes: main() was calling
    build_paired_outcomes() with target_category missing entirely, which
    shifted every subsequent positional argument and crashed with a
    TypeError on any real invocation - nothing previously exercised main()."""
    import sys
    import json as json_module
    import src.analysis.mcnemar_causal_ablation as mca
    monkeypatch.setattr(mca, "classify_completion", lambda text: text)

    rows = [
        {"prompt": "p1", "quadrant": "C", "stage": "M3_baseline", "response": "soft_deflection"},
        {"prompt": "p1", "quadrant": "C", "stage": "M3_ablated", "response": "comply"},
        {"prompt": "p2", "quadrant": "A", "stage": "M3_baseline", "response": "refusal"},
        {"prompt": "p2", "quadrant": "A", "stage": "M3_ablated", "response": "refusal"},
    ]
    raw_file = tmp_path / "raw.json"
    with open(raw_file, "w", encoding="utf-8") as f:
        json_module.dump(rows, f)

    monkeypatch.setattr(sys, "argv", ["mcnemar_causal_ablation.py", "--file", str(raw_file), "--allow-unbound"])
    main()  # must not raise
    assert "McNemar exact p-value" in capsys.readouterr().out


def test_main_supports_custom_quadrant_and_category_for_steering(tmp_path, monkeypatch, capsys):
    """Steering reuses this script for its own (quadrant, category) pair -
    e.g. quadrant D / category refusal - not the ablation-specific default set."""
    import sys
    import json as json_module
    import src.analysis.mcnemar_causal_ablation as mca
    monkeypatch.setattr(mca, "classify_completion", lambda text: text)

    rows = [
        {"prompt": "p1", "quadrant": "D", "stage": "steer_baseline", "response": "comply"},
        {"prompt": "p1", "quadrant": "D", "stage": "steer_steered", "response": "refusal"},
    ]
    raw_file = tmp_path / "raw.json"
    with open(raw_file, "w", encoding="utf-8") as f:
        json_module.dump(rows, f)

    monkeypatch.setattr(sys, "argv", [
        "mcnemar_causal_ablation.py", "--file", str(raw_file), "--allow-unbound",
        "--conditions", "steer_baseline", "steer_steered",
        "--quadrant", "D", "--category", "refusal",
    ])
    main()
    out = capsys.readouterr().out
    assert "Quadrant D, category 'refusal'" in out
    assert "Quadrant C" not in out  # default set must NOT also run


def test_main_requires_category_when_quadrant_given(tmp_path, monkeypatch):
    import sys
    import json as json_module
    raw_file = tmp_path / "raw.json"
    with open(raw_file, "w", encoding="utf-8") as f:
        json_module.dump([], f)
    monkeypatch.setattr(sys, "argv", [
        "mcnemar_causal_ablation.py", "--file", str(raw_file), "--allow-unbound",
        "--quadrant", "D",
    ])
    import pytest
    with pytest.raises(SystemExit):
        main()