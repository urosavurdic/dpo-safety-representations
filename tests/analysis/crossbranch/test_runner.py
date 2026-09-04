"""Runner planning is pure, so the whole gating story is testable on CPU.

Mirrors tests/analysis/test_run_full_steering.py in shape: build a context,
assert what plan_run decides, and monkeypatch subprocess for execution.
"""
import json
from types import SimpleNamespace

import numpy as np
import pytest

from src.analysis.crossbranch import runner as R
from src.analysis.crossbranch.branches import P0_CONDITIONS
from src.analysis.crossbranch.delta import artifact_path, save_delta_npz
from src.v2_io import binding, identity_snapshot, write_json_lf

BENCH_SHA, SPLIT_SHA = "bench" * 8, "split" * 8
STAGES = ("M2", "M3", "M2_alt", "M3_alt")
HIDDEN, N_LAYERS = 4, 26


def make_rows(n=6):
    spec = [("A", "held_out_behavioral"), ("A", "direction_estimation"),
            ("B", None), ("C", None), ("D", "held_out_behavioral"),
            ("D", "direction_estimation")]
    return [
        {"record_id": f"r{i}", "prompt": f"p{i}", "quadrant": q, "split": s,
         "source": "toy"}
        for i, (q, s) in enumerate(spec[:n])
    ]


def make_ctx(tmp_path, rows, bound=True):
    act = tmp_path / "activations"
    act.mkdir(parents=True, exist_ok=True)
    if bound:
        for stage in STAGES:
            arr = np.zeros((len(rows), N_LAYERS, HIDDEN))
            np.save(act / f"{stage}_final.npy", arr)
            np.save(act / f"{stage}_pooled.npy", arr)
            write_json_lf(act / f"{stage}_metadata.json", identity_snapshot(rows))
            write_json_lf(
                act / f"{stage}_metadata_binding.json",
                binding("b.jsonl", BENCH_SHA, "s.json", SPLIT_SHA),
            )
    ctx = SimpleNamespace(
        rows=rows, benchmark_sha=BENCH_SHA, split_sha=SPLIT_SHA,
        paths=SimpleNamespace(activations=act), snapshot=identity_snapshot(rows),
    )
    ctx.bind = lambda: binding("b.jsonl", BENCH_SHA, "s.json", SPLIT_SHA)
    return ctx


def make_deltas(tmp_path, rows):
    d = tmp_path / "deltas"
    ids = [r["record_id"] for r in rows]
    for key in ("delta_target", "normmatched_random_target"):
        save_delta_npz(artifact_path(key, 24, d), np.zeros((len(rows), HIDDEN)), ids)
    return d


def plan(tmp_path, *, bound=True, deltas=True, out_dir=None, **kw):
    rows = make_rows()
    ctx = make_ctx(tmp_path, rows, bound=bound)
    dd = make_deltas(tmp_path, rows) if deltas else tmp_path / "nodeltas"
    return R.plan_run(
        ctx, list(P0_CONDITIONS), (0.5, 1.0, 2.0), "A", "B",
        deltas_dir=dd, out_dir=out_dir or (tmp_path / "raw"), **kw
    )


# ---------------------------------------------------------------------------


def test_stage_one_plans_exactly_eight_units(tmp_path):
    p, _ = plan(tmp_path)
    assert len(p) == 8


def test_model_conditions_are_planned_once_with_no_coefficient(tmp_path):
    p, _ = plan(tmp_path)
    model_units = [i for i in p if i["kind"] == "model"]
    assert len(model_units) == 2
    assert all(i["coef"] is None for i in model_units)


def test_model_conditions_are_not_blocked_on_a_missing_delta_artifact(tmp_path):
    p, _ = plan(tmp_path, deltas=False)
    for item in p:
        if item["kind"] == "model":
            assert not any("missing" in b for b in item["blockers"]), (
                "a model condition injects nothing and must never require an .npz"
            )


def test_everything_runs_when_preconditions_are_met(tmp_path):
    p, msgs = plan(tmp_path)
    assert all(i["status"] == R.RUN for i in p), [i["blockers"] for i in p]
    assert any("OK" in m for m in msgs)


def test_unbound_activations_block_every_unit(tmp_path):
    p, msgs = plan(tmp_path, bound=False)
    assert all(i["status"] == R.BLOCKED for i in p)
    assert all("activations not bound" in i["blockers"] for i in p)
    assert any("never extracts" in m for m in msgs)


def test_missing_delta_artifact_blocks_only_vector_units(tmp_path):
    p, _ = plan(tmp_path, deltas=False)
    vector = [i for i in p if i["kind"] == "vector"]
    assert vector and all(i["status"] == R.BLOCKED for i in vector)


def test_missing_split_labels_block_everything(tmp_path):
    rows = make_rows()
    rows[0]["split"] = None                      # an A row with no split
    ctx = make_ctx(tmp_path, rows)
    p, msgs = R.plan_run(
        ctx, list(P0_CONDITIONS), (1.0,), "A", "B",
        deltas_dir=make_deltas(tmp_path, rows), out_dir=tmp_path / "raw",
    )
    assert all(i["status"] == R.BLOCKED for i in p)
    assert any("no split label" in m for m in msgs)


def test_existing_output_is_skipped_and_force_reruns_it(tmp_path):
    out = tmp_path / "raw"
    out.mkdir()
    (out / "crossbranch_AtoB_baseline_target_coefna.json").write_text("[]", "utf-8")

    p, _ = plan(tmp_path, out_dir=out)
    assert [i["status"] for i in p if i["condition"] == "baseline_target"] == [R.SKIP]

    p2, _ = plan(tmp_path, out_dir=out, force=True)
    assert [i["status"] for i in p2 if i["condition"] == "baseline_target"] == [R.RUN]


def test_stage_two_condition_is_blocked_without_the_explicit_flag(tmp_path):
    rows = make_rows()
    ctx = make_ctx(tmp_path, rows)
    p, _ = R.plan_run(
        ctx, ["xfer_delta_source_identity"], (1.0,), "A", "B",
        deltas_dir=make_deltas(tmp_path, rows), out_dir=tmp_path / "raw",
    )
    assert p[0]["status"] == R.BLOCKED
    assert any("stage2" in b for b in p[0]["blockers"])


def test_build_command_shape_and_conditional_coefficient():
    cmd = R.build_command("own_delta_target", 1.0, "A", "B")
    assert cmd[1:3] == ["-m", "src.analysis.crossbranch.worker"]
    assert "--coef" in cmd and "1" in cmd
    assert "--source-branch" in cmd and "--target-branch" in cmd

    model_cmd = R.build_command("baseline_target", None, "A", "B")
    assert "--coef" not in model_cmd


def test_build_command_carries_the_reciprocal_direction():
    cmd = R.build_command("own_delta_target", 1.0, "B", "A")
    i = cmd.index("--source-branch")
    assert cmd[i + 1] == "B"


def test_run_plan_only_invokes_runnable_units(monkeypatch, tmp_path):
    calls = []

    def fake_run(cmd, *a, **k):
        calls.append(cmd)
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(R.subprocess, "run", fake_run)
    p = [
        {"condition": "own_delta_target", "coef": 1.0, "status": R.RUN,
         "kind": "vector", "stage_gate": "p0", "output": "x", "blockers": []},
        {"condition": "own_delta_target", "coef": 2.0, "status": R.BLOCKED,
         "kind": "vector", "stage_gate": "p0", "output": "y", "blockers": ["b"]},
    ]
    results = R.run_plan(p, "A", "B")
    assert len(calls) == 1
    assert results[1]["returncode"] is None


def test_run_plan_continues_after_a_failure(monkeypatch):
    codes = iter([1, 0])
    monkeypatch.setattr(
        R.subprocess, "run",
        lambda cmd, *a, **k: SimpleNamespace(returncode=next(codes)),
    )
    p = [
        {"condition": "c1", "coef": 1.0, "status": R.RUN, "kind": "vector",
         "stage_gate": "p0", "output": "x", "blockers": []},
        {"condition": "c2", "coef": 1.0, "status": R.RUN, "kind": "vector",
         "stage_gate": "p0", "output": "y", "blockers": []},
    ]
    results = R.run_plan(p, "A", "B")
    assert [r["returncode"] for r in results] == [1, 0]


def test_manifest_embeds_both_hashes(tmp_path, monkeypatch):
    monkeypatch.setattr(R, "MANIFEST_DIR", tmp_path / "man")
    ctx = make_ctx(tmp_path, make_rows())
    args = SimpleNamespace(conditions=list(P0_CONDITIONS), dry_run=False)
    path = R.write_manifest([], args, ctx, "AtoB")
    data = json.loads(path.read_text(encoding="utf-8"))
    assert data["benchmark_sha256"] == BENCH_SHA
    assert data["split_manifest_sha256"] == SPLIT_SHA
    assert data["direction"] == "AtoB"


def test_print_plan_is_safe_on_a_fully_blocked_plan(tmp_path, capsys):
    p, msgs = plan(tmp_path, bound=False)
    R.print_plan(p, msgs)
    out = capsys.readouterr().out
    assert "blocked=8" in out
    assert "Planned units: 8" in out
