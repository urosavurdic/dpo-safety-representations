"""Toy fixtures for the stagecmp tests.

Everything is synthetic and tiny. No test in this package may read the real
2 GB activation set or the 75 MB judge file: ``act_dir`` and ``judge_path`` are
always passed explicitly so a missing fixture raises instead of silently
falling back to production data (CONTRIBUTING.md).
"""
from __future__ import annotations

import json

import numpy as np
import pytest

from src.stagecmp.pairs import TRANSITIONS_2X2

STAGES = sorted({s for t in TRANSITIONS_2X2.values() for s in t[:2]})
N_LAYERS = 6
HIDDEN = 8


def _records(n_per_quadrant=24):
    """Deterministic (record_id, quadrant) listing shared by every stage.

    24 per quadrant, so the thirds and two-thirds rates below land on whole
    prompts and the bootstrap has enough distinct units to produce a
    non-degenerate percentile interval.
    """
    out = []
    for q in "ABCD":
        for i in range(n_per_quadrant):
            out.append({"record_id": f"{q}{i:03d}", "quadrant": q,
                        "split": "direction_estimation" if i == 0 else "held_out_behavioral"})
    return out


@pytest.fixture
def records():
    return _records()


@pytest.fixture
def act_dir(tmp_path, records):
    """Activations for every 2x2 stage, identical row order, known geometry."""
    rng = np.random.default_rng(0)
    d = tmp_path / "activations"
    d.mkdir()
    for k, stage in enumerate(STAGES):
        arr = rng.normal(size=(len(records), N_LAYERS, HIDDEN)).astype(np.float32)
        # give quadrant A a stage-specific offset along axis 0 so the A-D
        # contrast is non-trivial and differs between stages
        for i, row in enumerate(records):
            if row["quadrant"] == "A":
                arr[i, :, 0] += 1.0 + 0.5 * k
        for suffix in ("final", "pooled"):
            np.save(d / f"{stage}_{suffix}.npy", arr)
        (d / f"{stage}_metadata.json").write_text(
            json.dumps(records), encoding="utf-8"
        )
    return d


@pytest.fixture
def directions_dir(tmp_path):
    """A unit direction per stage, pointing along axis 0."""
    d = tmp_path / "refusal_direction"
    d.mkdir()
    for stage in STAGES:
        v = np.zeros((N_LAYERS, HIDDEN), dtype=np.float32)
        v[:, 0] = 1.0
        np.save(d / f"{stage}_direction_654.npy", v)
    return d


def judge_payload(rates, records=None, scorers=("regex",)):
    """Build a judge file where ``rates[stage][quadrant]`` is the withhold rate.

    Withholding is assigned to the FIRST ``round(rate * n)`` prompts of each
    quadrant, so the outcome is deterministic and hand-checkable. That also
    makes the arms *nested* -- a prompt refused at a low rate is refused at
    every higher one -- which is the positively-correlated structure real
    prompts have, and the reason the paired bootstrap is narrower than an
    independent one. With independent arms that ordering would not hold, and
    neither would the width comparison.
    """
    records = records or _records()
    out = []
    for stage, per_q in rates.items():
        by_q = {}
        for row in records:
            by_q.setdefault(row["quadrant"], []).append(row)
        for q, rows in by_q.items():
            k = int(round(per_q.get(q, 0.0) * len(rows)))
            for j, row in enumerate(rows):
                refused = j < k
                rec = {
                    "record_id": row["record_id"],
                    "quadrant": q,
                    "condition": f"{stage}_behavior",
                    "stage": f"{stage}_behavior",
                    "response": "refusal" if refused else "sure, here",
                    "regex": {"refused": bool(refused), "soft_deflection": False,
                              "degenerate": False, "malformed": False},
                    "strong_reject": {"score": None, "judge_status": "out_of_scope"},
                    "wildguard": {"response_harm": None, "response_refusal": None,
                                  "judge_status": "out_of_scope"},
                }
                if "strong_reject" in scorers:
                    rec["strong_reject"] = {"score": 1.0 if refused else 0.0}
                if "wildguard" in scorers:
                    rec["wildguard"] = {"response_harm": 0.0 if refused else 1.0,
                                        "response_refusal": 1.0 if refused else 0.0}
                out.append(rec)
    return {"records": out, "judge_versions": {"regex": "toy"}}


@pytest.fixture
def judge_file(tmp_path, records):
    """A 2x2 where DPO moves withholding far more than safety SFT."""
    rates = {
        "M1": {q: 0.0 for q in "ABCD"},
        "M1_alt": {q: 0.0 for q in "ABCD"},
        "M2": {"A": 1 / 3, "B": 0.0, "C": 0.0, "D": 0.0},
        "M2_alt": {"A": 1 / 3, "B": 0.0, "C": 0.0, "D": 0.0},
        "M3_direct": {"A": 1.0, "B": 1 / 3, "C": 2 / 3, "D": 1 / 3},
        "M3_direct_alt": {"A": 2 / 3, "B": 1 / 3, "C": 1 / 3, "D": 1 / 3},
    }
    path = tmp_path / "judges.json"
    path.write_text(json.dumps(judge_payload(rates, records)), encoding="utf-8")
    return path
