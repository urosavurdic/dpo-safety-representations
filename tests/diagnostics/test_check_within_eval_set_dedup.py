import json

from src.diagnostics.check_within_eval_set_dedup import group_by_source, load_eval_set


def test_group_by_source_splits_quadrant_d_by_source():
    rows = [
        {"prompt": "a1", "quadrant": "D", "source": "Alpaca"},
        {"prompt": "a2", "quadrant": "D", "source": "Alpaca"},
        {"prompt": "b1", "quadrant": "D", "source": "Dolly-15k"},
        {"prompt": "c1", "quadrant": "D", "source": "OASST1"},
        {"prompt": "irrelevant", "quadrant": "A", "source": "HarmBench"},
    ]
    grouped = group_by_source(rows, "D")
    assert grouped == {
        "Alpaca": ["a1", "a2"],
        "Dolly-15k": ["b1"],
        "OASST1": ["c1"],
    }


def test_group_by_source_ignores_other_quadrants():
    rows = [
        {"prompt": "a1", "quadrant": "A", "source": "HarmBench"},
        {"prompt": "b1", "quadrant": "B", "source": "XSTest"},
    ]
    assert group_by_source(rows, "D") == {}


def test_load_eval_set_reads_jsonl(tmp_path):
    path = tmp_path / "eval.jsonl"
    rows = [{"prompt": "p1", "quadrant": "A"}, {"prompt": "p2", "quadrant": "D"}]
    path.write_text("\n".join(json.dumps(r) for r in rows))
    loaded = load_eval_set(str(path))
    assert loaded == rows


def test_check_all_pairs_skips_when_single_source(capsys):
    from src.diagnostics.check_within_eval_set_dedup import check_all_pairs_within_quadrant
    rows = [{"prompt": "a1", "quadrant": "A", "source": "HarmBench"}]
    result = check_all_pairs_within_quadrant(rows, "A")
    assert result == {}
    assert "only 1 source" in capsys.readouterr().out


def test_check_all_pairs_detects_exact_duplicate_across_sources():
    # NOTE: exercises find_near_duplicates too (via check_all_pairs_within_quadrant),
    # which downloads the sentence-transformers model on first use - needs network.
    # Matches this project's existing convention (test_check_leakage.py doesn't
    # unit-test find_near_duplicates directly for the same reason).
    from src.diagnostics.check_within_eval_set_dedup import check_all_pairs_within_quadrant
    rows = [
        {"prompt": "What is a bond?", "quadrant": "D", "source": "Alpaca"},
        {"prompt": "What is a bond?", "quadrant": "D", "source": "Dolly-15k"},
        {"prompt": "Totally different question", "quadrant": "D", "source": "OASST1"},
    ]
    result = check_all_pairs_within_quadrant(rows, "D")
    assert "Alpaca_vs_Dolly-15k" in result
    assert len(result["Alpaca_vs_Dolly-15k"]["exact_duplicates"]) == 1
