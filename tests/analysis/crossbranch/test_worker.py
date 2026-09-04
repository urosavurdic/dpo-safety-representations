"""Worker logic that does not need a model or GPU.

run_unit() itself is untested by design (matches eval_steering_v2.main /
eval_causal_ablation.main precedent): it needs a real checkpoint. Everything
it composes that is pure Python is tested here.
"""
import pytest

from src.analysis.crossbranch.worker import output_path, rows_for_unit

# Mirrors the real benchmark's actual interleaving, confirmed against
# data/frozen_v2/benchmark_v2_20260826T212909Z.jsonl: quadrant-A rows do NOT
# group direction_estimation before held_out_behavioral, they interleave.
ROWS = [
    {"record_id": "a0", "quadrant": "A", "split": "direction_estimation"},
    {"record_id": "a1", "quadrant": "A", "split": "direction_estimation"},
    {"record_id": "a2", "quadrant": "A", "split": "held_out_behavioral"},
    {"record_id": "a3", "quadrant": "A", "split": "direction_estimation"},
    {"record_id": "a4", "quadrant": "A", "split": "held_out_behavioral"},
    {"record_id": "a5", "quadrant": "A", "split": "held_out_behavioral"},
    {"record_id": "a6", "quadrant": "A", "split": "direction_estimation"},
    {"record_id": "a7", "quadrant": "A", "split": "held_out_behavioral"},
    {"record_id": "b0", "quadrant": "B", "split": None},
    {"record_id": "b1", "quadrant": "B", "split": None},
]


def test_limit_is_applied_after_quadrant_and_split_filtering():
    """The bug this regression-tests: --limit used to slice the raw
    benchmark BEFORE filtering, so `--limit 8 --quadrants A` on this fixture
    would have returned only the 4 held_out_behavioral rows among the first
    8 raw rows, not 4 held-out rows chosen after filtering to quadrant A."""
    out = rows_for_unit(ROWS, ["A"], limit=4)
    assert [r["record_id"] for r in out] == ["a2", "a4", "a5", "a7"]
    assert all(r["quadrant"] == "A" for r in out)
    assert all(r["split"] == "held_out_behavioral" for r in out)


def test_limit_larger_than_available_rows_returns_all_of_them():
    out = rows_for_unit(ROWS, ["A"], limit=100)
    assert len(out) == 4  # only 4 quadrant-A rows are held_out_behavioral


def test_no_limit_returns_every_filtered_row():
    out = rows_for_unit(ROWS, ["A"], limit=None)
    assert len(out) == 4


def test_quadrant_b_has_no_split_restriction():
    """B/C never feed the direction, so quadrant_rows keeps them whole --
    unlike A/D, which are restricted to held_out_behavioral."""
    out = rows_for_unit(ROWS, ["B"], limit=None)
    assert [r["record_id"] for r in out] == ["b0", "b1"]


def test_limit_zero_returns_no_rows():
    assert rows_for_unit(ROWS, ["A"], limit=0) == []


def test_multiple_quadrants_preserve_benchmark_order():
    out = rows_for_unit(ROWS, ["A", "B"], limit=None)
    assert [r["record_id"] for r in out] == ["a2", "a4", "a5", "a7", "b0", "b1"]


# ---- output_path -----------------------------------------------------


def test_output_path_names_vector_and_model_conditions_differently():
    vector = output_path("AtoB", "own_delta_target", 1.0)
    model = output_path("AtoB", "baseline_target", None)
    assert "coef1" in vector.name
    assert "coefna" in model.name
    assert vector != model
