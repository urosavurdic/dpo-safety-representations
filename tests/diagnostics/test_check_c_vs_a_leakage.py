"""Toy tests for src/diagnostics/check_c_vs_a_leakage.py (WP-Leak)."""
import json
from pathlib import Path

from src.diagnostics import check_c_vs_a_leakage as leak

FIX = Path(__file__).resolve().parents[1] / "fixtures"


def test_exact_overlap_normalizes():
    assert leak.exact_overlap(["Hello  World "], ["hello world"]) == ["Hello  World "]
    assert leak.exact_overlap(["unique C prompt"], ["something else"]) == []


def test_jaccard_symmetric_and_bounded():
    assert leak.jaccard("a b c", "a b c") == 1.0
    assert leak.jaccard("a b c d", "a b x y") == leak.jaccard("a b x y", "a b c d")
    assert 0.0 <= leak.jaccard("a b", "c d") <= 1.0


def test_near_duplicates_fallback_flags_high_overlap():
    pairs, method = leak.near_duplicates(
        ["how do I smuggle goods across a border undetected"],
        ["how do I smuggle goods across a border without being detected",
         "totally unrelated cooking question"],
        jac_threshold=0.5,
    )
    assert len(pairs) >= 1
    assert "Jaccard" in method or "sentence-transformers" in method


def test_category_coverage_reports_cf3_match():
    rows = [
        {"quadrant": "A", "project_category": "illegal"},
        {"quadrant": "A", "project_category": "harassment_bullying"},
        {"quadrant": "A", "project_category": "cybercrime_intrusion"},
        {"quadrant": "A", "project_category": "misinformation_disinformation"},
        {"quadrant": "C", "project_category": "illegal"},
        {"quadrant": "C", "project_category": "harassment_bullying"},
        {"quadrant": "C", "project_category": "cybercrime_intrusion"},
        {"quadrant": "C", "project_category": "misinformation_disinformation"},
        {"quadrant": "B", "project_category": "homonyms"},
    ]
    cov = leak.category_coverage(rows)
    assert cov["cf3_label_sets_match"] is True
    assert cov["A_cf3_only"]["illegal"] == 1
    assert cov["A_non_cf3_categories"] == []


def test_category_coverage_detects_mismatch():
    rows = [
        {"quadrant": "A", "project_category": "illegal"},
        {"quadrant": "C", "project_category": "harassment_bullying"},
    ]
    assert leak.category_coverage(rows)["cf3_label_sets_match"] is False


def test_run_against_fixture_benchmark(tmp_path):
    report = leak.run(latest_path=str(FIX / "benchmark_654.LATEST_BENCHMARK.json"))
    assert report["n_C"] == 4 and report["n_A"] == 5
    assert "C_vs_A" in report and "category_coverage" in report
    assert isinstance(report["C_vs_A"]["exact"], list)
