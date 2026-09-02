"""Toy tests for src/analysis/build_human_review_packet.py (WP-Sample §5.2).

Determinism; coverage independent of judge info; sealed key rejected inside
the repo; re-annotation floor per quadrant; targeted sample uses judge info
and never overlaps coverage.
"""
import json
import subprocess
import sys
from pathlib import Path

import pytest

from src.analysis import build_human_review_packet as hp

REPO_ROOT = Path(__file__).resolve().parents[2]


def _rows(n_per_cell=4):
    stages = ["M0", "M1", "M2", "M3", "M3_direct", "M1_alt", "M2_alt", "M3_alt"]
    conditions = ["M3_baseline", "M3_ablated_AD", "M3_steered"]
    rows = []
    for q in ("A", "B", "C", "D"):
        for st in stages:
            for cond in conditions:
                for k in range(n_per_cell):
                    rid = f"{q}_{st}_{cond}_{k}"
                    rows.append({
                        "record_id": rid, "stage": st, "condition": cond,
                        "quadrant": q, "prompt": f"prompt {rid}",
                        "response": "I cannot help." if k % 2 else "Sure, here is how.",
                    })
    return rows


def test_coverage_is_deterministic_and_judge_blind():
    rows = _rows()
    cov1, log1 = hp.coverage_sample(rows, seed=20260904)
    cov2, log2 = hp.coverage_sample(rows, seed=20260904)
    assert cov1 == cov2 and log1 == log2
    # adding judge-like fields to the rows must not change the selection
    rows_with_judge = [{**r, "strong_reject": {"score": 0.5}, "regex": {"degenerate": True}}
                       for r in rows]
    cov3, _ = hp.coverage_sample(rows_with_judge, seed=20260904)
    assert cov3 == cov1


def test_coverage_hits_30_per_quadrant_when_supply_is_ample():
    rows = _rows(n_per_cell=6)
    cov, _log = hp.coverage_sample(rows, seed=1)
    by_q = {}
    for rid, stage, cond in cov:
        q = rid.split("_")[0]
        by_q[q] = by_q.get(q, 0) + 1
    assert set(by_q) == {"A", "B", "C", "D"}
    assert all(v == 30 for v in by_q.values()), by_q


def test_build_packet_counts_and_no_overlap():
    rows = _rows()
    judged = [{"record_id": r["record_id"], "stage": r["stage"], "condition": r["condition"],
               "strong_reject": {"score": 0.4}, "regex": {"refused": True, "degenerate": False},
               "wildguard": {"response_refusal": False}}
              for r in rows if r["quadrant"] == "C"]
    packet, key = hp.build_packet(rows, judged, seed=20260904)
    assert packet["counts"]["coverage"] == 120
    assert packet["counts"]["targeted"] == 40
    assert packet["counts"]["unique_responses"] == 160
    assert packet["counts"]["reannotated"] == 40
    assert packet["counts"]["total_items"] == 200
    # blinded items expose ONLY neutral_id / prompt / response
    assert all(set(item) == {"neutral_id", "prompt", "response"} for item in packet["items"])
    # sealed key has 200 neutral ids, 40 flagged as re-annotation copies
    assert len(key) == 200
    assert sum(1 for v in key.values() if "reannotation_of" in v) == 40


def test_reannotation_floor_per_quadrant():
    rows = _rows()
    packet, key = hp.build_packet(rows, None, seed=7)
    reanno = hp.reannotation_ids({nid: v for nid, v in key.items() if "reannotation_of" not in v}, seed=7)
    by_q = {}
    for nid in reanno:
        q = key[nid]["quadrant"]
        by_q[q] = by_q.get(q, 0) + 1
    assert all(by_q.get(q, 0) >= 8 for q in ("A", "B", "C", "D")), by_q


def test_cli_rejects_key_out_inside_repo(tmp_path):
    resp = tmp_path / "resp.json"
    resp.write_text(json.dumps(_rows()), encoding="utf-8")
    r = subprocess.run(
        [sys.executable, "-m", "src.analysis.build_human_review_packet",
         "--responses", str(resp), "--packet-out", str(tmp_path / "p.json"),
         "--key-out", str(REPO_ROOT / "sealed_key.json")],
        cwd=REPO_ROOT, capture_output=True, text=True,
    )
    assert r.returncode != 0
    assert "outside the repo" in (r.stdout + r.stderr)
