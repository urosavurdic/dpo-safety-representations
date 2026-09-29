"""WP-2: a judge that fails to load must not produce a complete-looking file.

Three byte-identical 15 MB outputs exist on disk from a run where both judges
failed to load with a bitsandbytes error. The run recorded the failure in
``judge_status``, marked every row ``model_unavailable``, wrote the file and
exited 0. These tests pin the corrected behaviour.
"""
from __future__ import annotations

import pytest

from src.analysis.behavioral_judges import (
    CoverageContractError,
    JudgeUnavailableError,
    check_coverage,
)


def rec(condition, quadrant, **statuses):
    row = {"condition": condition, "quadrant": quadrant, "record_id": f"{condition}-{quadrant}"}
    for scorer, status in statuses.items():
        row[scorer] = {"judge_status": status}
    return row


def test_contract_is_satisfied_when_every_declared_cell_is_scored():
    records = [rec("M1_behavior", q, strong_reject="scored") for q in "ABCD"]
    assert check_coverage(records, {"strong_reject": [("M1_behavior", None)]}) == []


def test_contract_reports_partially_scored_cells():
    records = [rec("M1_behavior", "A", strong_reject="scored"),
               rec("M1_behavior", "B", strong_reject="out_of_scope")]
    unmet = check_coverage(records, {"strong_reject": [("M1_behavior", None)]})
    assert len(unmet) == 1
    assert "1/2 scored" in unmet[0]


def test_contract_can_be_scoped_to_one_quadrant():
    records = [rec("M1_behavior", "A", strong_reject="scored"),
               rec("M1_behavior", "B", strong_reject="out_of_scope")]
    assert check_coverage(records, {"strong_reject": [("M1_behavior", "A")]}) == []
    assert check_coverage(records, {"strong_reject": [("M1_behavior", "B")]}) != []


def test_contract_catches_a_condition_with_no_rows_at_all():
    """The failure mode where a glob silently excluded a whole condition."""
    records = [rec("M1_behavior", "A", strong_reject="scored")]
    unmet = check_coverage(records, {"strong_reject": [("M2_behavior", None)]})
    assert "no rows at all" in unmet[0]


def test_contract_checks_each_scorer_independently():
    """The real gap: regex complete everywhere, the LLM judges not."""
    records = [rec("M1_behavior", q, regex="scored", strong_reject="out_of_scope")
               for q in "ABCD"]
    contract = {"regex": [("M1_behavior", None)],
                "strong_reject": [("M1_behavior", None)]}
    unmet = check_coverage(records, contract)
    assert len(unmet) == 1
    assert unmet[0].startswith("strong_reject")


def test_model_unavailable_is_not_counted_as_scored():
    records = [rec("M1_behavior", "A", strong_reject="model_unavailable")]
    assert check_coverage(records, {"strong_reject": [("M1_behavior", None)]}) != []


def test_the_exceptions_are_distinct_and_catchable():
    assert issubclass(JudgeUnavailableError, RuntimeError)
    assert issubclass(CoverageContractError, RuntimeError)
    assert JudgeUnavailableError is not CoverageContractError


def test_run_judges_defaults_to_failing_closed():
    """The default must be the safe one; the escape hatch must be explicit."""
    import inspect

    from src.analysis.behavioral_judges import run_judges

    params = inspect.signature(run_judges).parameters
    assert params["fail_closed"].default is True
    assert params["coverage_contract"].default is None


# --- transformers 5 compatibility ------------------------------------------- #

def test_dtype_kwarg_is_chosen_by_transformers_major_version():
    """transformers renamed `torch_dtype` -> `dtype` at 5.0 and removed the old
    name. Colab ships 5.x. This was the last caller in src/ still on the old
    kwarg, and it is only reached on a real GPU load, so nothing caught it until
    a judge preflight failed in a live session."""
    import re
    from pathlib import Path

    src = Path("src/analysis/behavioral_judges.py").read_text(encoding="utf-8")
    assert '_dtype_kw = "dtype" if _major >= 5 else "torch_dtype"' in src
    # no bare torch_dtype= call sites remain anywhere in src/
    offenders = []
    for path in Path("src").rglob("*.py"):
        for i, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            if re.search(r'"torch_dtype":\s', line) or re.search(r"\btorch_dtype=", line):
                offenders.append(f"{path}:{i}")
    assert not offenders, f"unguarded torch_dtype call sites: {offenders}"
