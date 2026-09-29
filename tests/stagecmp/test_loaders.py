"""Loading invariants: row order, and honest scorer coverage."""
from __future__ import annotations

import json

import numpy as np
import pytest

from src.stagecmp.loaders import (
    CoverageError,
    assert_row_order,
    behavior_rows,
    load_stage_arrays,
    quadrants_of,
    require_scorer,
    scorer_coverage,
)
from tests.stagecmp.conftest import judge_payload


def test_row_order_agrees_across_stages(act_dir):
    ids = assert_row_order(["M1", "M2", "M3_direct"], act_dir)
    assert len(ids) == len(set(ids))
    assert ids[0] == "A000"


def test_row_order_mismatch_raises(tmp_path, act_dir):
    """Silent misalignment would make every paired analysis wrong with no
    symptom, so it must fail loudly."""
    meta = json.loads((act_dir / "M2_metadata.json").read_text(encoding="utf-8"))
    meta[3], meta[7] = meta[7], meta[3]
    (act_dir / "M2_metadata.json").write_text(json.dumps(meta), encoding="utf-8")
    with pytest.raises(ValueError, match="row order differs"):
        assert_row_order(["M1", "M2"], act_dir)


def test_load_stage_arrays_checks_shape_against_metadata(act_dir):
    arr, meta = load_stage_arrays("M1", "final_token", act_dir)
    assert arr.shape[0] == len(meta)
    assert arr.ndim == 3


def test_unknown_pooling_is_rejected(act_dir):
    with pytest.raises(ValueError, match="pooling must be"):
        load_stage_arrays("M1", "last_five", act_dir)


def test_missing_stage_raises_rather_than_reaching_for_real_data(tmp_path):
    with pytest.raises(Exception):
        load_stage_arrays("M1", "final_token", tmp_path / "absent")


def test_quadrants_are_read_from_metadata(act_dir):
    q = quadrants_of("M1", act_dir)
    assert set(q) == {"A", "B", "C", "D"}
    assert (q == "A").sum() == 24


def test_coverage_distinguishes_uncovered_from_zero(judge_file):
    """The whole point of this function: an unscored stage must report 0 of 96
    covered, not a rate of zero."""
    cov = scorer_coverage(["M1", "M3_direct"], judge_file)
    assert cov["M1"]["regex"] == cov["M1"]["n_rows"] == 96
    assert cov["M1"]["strong_reject"] == 0
    assert cov["M1"]["wildguard"] == 0


def test_coverage_breaks_down_by_quadrant(judge_file):
    cov = scorer_coverage(["M3_direct"], judge_file)
    assert cov["M3_direct"]["by_quadrant"]["regex"]["A"] == 24


def test_require_scorer_blocks_an_uncovered_analysis(judge_file):
    require_scorer("regex", ["M1", "M3_direct"], judge_file)   # complete, passes
    with pytest.raises(CoverageError, match="not fully populated"):
        require_scorer("strong_reject", ["M1", "M3_direct"], judge_file)


def test_require_scorer_names_the_remedy(judge_file):
    with pytest.raises(CoverageError) as exc:
        require_scorer("wildguard", ["M1"], judge_file)
    assert "--scope all" in str(exc.value)


def test_partial_coverage_is_reported_partially(tmp_path, records):
    payload = judge_payload({"S": {"A": 0.5}}, records, scorers=("regex", "strong_reject"))
    for rec in payload["records"]:
        if rec["quadrant"] != "A":
            rec["strong_reject"] = {"score": None, "judge_status": "out_of_scope"}
    path = tmp_path / "partial.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    cov = scorer_coverage(["S"], path)
    assert cov["S"]["strong_reject"] == 24
    assert cov["S"]["regex"] == 96


def test_behavior_rows_are_keyed_by_record_id(judge_file):
    rows = behavior_rows("M2", judge_file)
    assert len(rows) == 96
    assert all(rid == row["record_id"] for rid, row in rows.items())
