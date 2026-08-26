import json

import pytest

from src.analysis.run_full_steering import (
    ALL_STAGES,
    DEFAULT_STAGES,
    activation_metadata_matches,
    build_command,
    check_split_assigned,
    default_tag,
    direction_exists,
    output_path_for,
    plan_run,
    run_plan,
)


def _row(prompt, quadrant, source="src", split=None):
    r = {"prompt": prompt, "quadrant": quadrant, "source": source}
    if split is not None:
        r["split"] = split
    return r


def test_default_tag_matches_eval_steering_v2_format():
    # Byte-identical to eval_steering_v2.default_tag's output for the same
    # inputs -- this is load-bearing (see module docstring), so pin it
    # against a hand-computed expectation independent of the other module.
    tag = default_tag("M3_direct", [14, 15, 28], "quadrant_a_projection", 0.2, ["A", "D"])
    assert tag == "M3_direct_L14-15-28_quadrant_a_projection_coef0p2_QAD"


def test_default_tag_sorts_and_dedupes_layers():
    assert default_tag("M3", [28, 24, 24], "fixed", 1.0, ["D"]) == "M3_L24-28_fixed_coef1_QD"


def test_check_split_assigned_true_when_all_ad_rows_have_split():
    rows = [
        _row("p1", "A", split="direction_estimation"),
        _row("p2", "D", split="held_out_behavioral"),
        _row("p3", "B"),  # B/C never get a split key -- must not be required
        _row("p4", "C"),
    ]
    ok, msg = check_split_assigned(rows)
    assert ok is True
    assert "1 direction_estimation" in msg
    assert "1 held_out_behavioral" in msg


def test_check_split_assigned_false_when_any_ad_row_missing_split():
    rows = [
        _row("p1", "A", split="direction_estimation"),
        _row("p2", "D"),  # missing split
    ]
    ok, msg = check_split_assigned(rows)
    assert ok is False
    assert "1/2" in msg
    assert "build_eval_set" in msg


def test_check_split_assigned_true_on_empty_ad_set():
    # Degenerate but shouldn't crash -- an eval set with no A/D rows at all
    # (e.g. a B/C-only toy fixture) trivially satisfies "every A/D row has
    # a split".
    ok, msg = check_split_assigned([_row("p1", "B")])
    assert ok is True


def test_activation_metadata_matches_true_when_identical(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "results" / "activations").mkdir(parents=True)
    eval_rows = [_row("p1", "A", "src1", "direction_estimation")]
    meta_path = tmp_path / "results" / "activations" / "M3_metadata.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump([{"prompt": "p1", "quadrant": "A", "source": "src1", "split": "direction_estimation"}], f)

    ok, msg = activation_metadata_matches("M3", eval_rows)
    assert ok is True
    assert "matches" in msg


def test_activation_metadata_matches_false_when_missing_file(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "results" / "activations").mkdir(parents=True)
    ok, msg = activation_metadata_matches("M3", [_row("p1", "A", split="direction_estimation")])
    assert ok is False
    assert "missing" in msg


def test_activation_metadata_matches_false_when_stale_content(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "results" / "activations").mkdir(parents=True)
    meta_path = tmp_path / "results" / "activations" / "M3_metadata.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump([{"prompt": "OLD", "quadrant": "A", "source": "src1", "split": "direction_estimation"}], f)

    ok, msg = activation_metadata_matches("M3", [_row("p1", "A", split="direction_estimation")])
    assert ok is False
    assert "stale" in msg


def test_activation_metadata_matches_false_when_split_missing_in_saved(tmp_path, monkeypatch):
    """Regression guard for the exact bug class CLAUDE.md documents (toy
    fixtures missing a 'split' key silently passing): saved metadata from
    BEFORE the split existed (no "split" key in the row at all) must not
    equal current metadata that has split=None for a B/C row -- these are
    represented consistently as split: None on both sides, so this test
    pins that None-vs-None is fine but a real mismatch (A/D with vs.
    without a split value) is caught."""
    monkeypatch.chdir(tmp_path)
    (tmp_path / "results" / "activations").mkdir(parents=True)
    meta_path = tmp_path / "results" / "activations" / "M3_metadata.json"
    # Saved BEFORE split existed: quadrant A row with no split recorded.
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump([{"prompt": "p1", "quadrant": "A", "source": "src1", "split": None}], f)

    ok, _ = activation_metadata_matches("M3", [_row("p1", "A", split="direction_estimation")])
    assert ok is False


def test_direction_exists_true_and_false(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "results" / "refusal_direction").mkdir(parents=True)
    ok, path = direction_exists("M3")
    assert ok is False

    (tmp_path / "results" / "refusal_direction" / "M3_direction.npy").write_bytes(b"\x00")
    ok, path = direction_exists("M3")
    assert ok is True


def test_output_path_for_uses_raw_dir_and_tag():
    # as_posix(), not str(): str() renders with the host OS separator, so
    # this assertion fails on Windows against a forward-slash literal.
    assert (
        output_path_for("M3_L24_x_coef1_QAD").as_posix()
        == "results/raw/steering_v2_M3_L24_x_coef1_QAD.json"
    )


def test_build_command_includes_overwrite_only_when_forced():
    cmd_no_force = build_command("M3", [24], "fixed", 1.0, ["A", "D"], "tag1", force=False)
    assert "--overwrite" not in cmd_no_force
    assert cmd_no_force[:4] == ["python", "-m", "src.analysis.eval_steering_v2", "--stage"]

    cmd_force = build_command("M3", [24], "fixed", 1.0, ["A", "D"], "tag1", force=True)
    assert "--overwrite" in cmd_force


def test_plan_run_all_blocked_when_eval_set_missing_split(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "data" / "processed").mkdir(parents=True)
    (tmp_path / "results" / "activations").mkdir(parents=True)
    (tmp_path / "results" / "refusal_direction").mkdir(parents=True)
    eval_path = tmp_path / "data" / "processed" / "controlled_eval.jsonl"
    with open(eval_path, "w", encoding="utf-8") as f:
        f.write(json.dumps(_row("p1", "A")) + "\n")  # no split -> should block everything

    plan, split_msg = plan_run(["M3"], [24], "fixed", 1.0, ["A", "D"], force=False)
    assert plan[0]["status"] == "blocked"
    assert "split" in plan[0]["blockers"][0]


def test_plan_run_status_run_when_all_preconditions_met_and_no_existing_output(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "data" / "processed").mkdir(parents=True)
    (tmp_path / "results" / "activations").mkdir(parents=True)
    (tmp_path / "results" / "refusal_direction").mkdir(parents=True)
    (tmp_path / "results" / "raw").mkdir(parents=True)

    row = {"prompt": "p1", "quadrant": "A", "source": "src1", "split": "direction_estimation"}
    with open(tmp_path / "data" / "processed" / "controlled_eval.jsonl", "w", encoding="utf-8") as f:
        f.write(json.dumps(row) + "\n")
    with open(tmp_path / "results" / "activations" / "M3_metadata.json", "w", encoding="utf-8") as f:
        json.dump([row], f)
    (tmp_path / "results" / "refusal_direction" / "M3_direction.npy").write_bytes(b"\x00")

    plan, _ = plan_run(["M3"], [24], "fixed", 1.0, ["A", "D"], force=False)
    assert plan[0]["status"] == "run"
    assert plan[0]["blockers"] == []


def test_plan_run_status_skip_already_done_when_output_exists(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "data" / "processed").mkdir(parents=True)
    (tmp_path / "results" / "activations").mkdir(parents=True)
    (tmp_path / "results" / "refusal_direction").mkdir(parents=True)
    (tmp_path / "results" / "raw").mkdir(parents=True)

    row = {"prompt": "p1", "quadrant": "A", "source": "src1", "split": "direction_estimation"}
    with open(tmp_path / "data" / "processed" / "controlled_eval.jsonl", "w", encoding="utf-8") as f:
        f.write(json.dumps(row) + "\n")
    with open(tmp_path / "results" / "activations" / "M3_metadata.json", "w", encoding="utf-8") as f:
        json.dump([row], f)
    (tmp_path / "results" / "refusal_direction" / "M3_direction.npy").write_bytes(b"\x00")

    plan, _ = plan_run(["M3"], [24], "fixed", 1.0, ["A", "D"], force=False)
    out_path = plan[0]["output_path"]
    from pathlib import Path
    Path(out_path).write_text("[]")

    plan2, _ = plan_run(["M3"], [24], "fixed", 1.0, ["A", "D"], force=False)
    assert plan2[0]["status"] == "skip_already_done"

    plan3, _ = plan_run(["M3"], [24], "fixed", 1.0, ["A", "D"], force=True)
    assert plan3[0]["status"] == "run"


def test_run_plan_skips_subprocess_for_non_run_items(monkeypatch):
    plan = [
        {"stage": "M1", "tag": "t1", "output_path": "x1.json", "status": "blocked",
         "blockers": ["nope"], "command": ["python", "-m", "x"]},
        {"stage": "M2", "tag": "t2", "output_path": "x2.json", "status": "skip_already_done",
         "blockers": [], "command": ["python", "-m", "x"]},
    ]
    calls = []
    monkeypatch.setattr("subprocess.run", lambda cmd: calls.append(cmd))
    results = run_plan(plan)
    assert calls == []  # neither blocked nor already-done items should invoke subprocess
    assert all(r["ran"] is False for r in results)


def test_run_plan_invokes_subprocess_only_for_run_items(monkeypatch):
    class _FakeCompletedProcess:
        returncode = 0

    plan = [
        {"stage": "M3", "tag": "t3", "output_path": "x3.json", "status": "run",
         "blockers": [], "command": ["python", "-m", "src.analysis.eval_steering_v2", "--stage", "M3"]},
    ]
    calls = []

    def fake_run(cmd):
        calls.append(cmd)
        return _FakeCompletedProcess()

    monkeypatch.setattr("subprocess.run", fake_run)
    results = run_plan(plan)
    assert len(calls) == 1
    assert results[0]["ran"] is True
    assert results[0]["succeeded"] is True


def test_all_stages_excludes_m0():
    assert "M0" not in ALL_STAGES
    assert len(ALL_STAGES) == 8


def test_default_stages_matches_the_4_dpo_endpoints_notebook_uses():
    # Mirrors notebooks/colab_unified_analysis.ipynb's STAGES_FOR_CAUSAL by
    # hand -- pin it so the two don't silently drift apart again.
    assert DEFAULT_STAGES == ["M3", "M3_direct", "M3_alt", "M3_direct_alt"]
    assert set(DEFAULT_STAGES).issubset(set(ALL_STAGES))
    assert "M1" not in DEFAULT_STAGES and "M2" not in DEFAULT_STAGES  # SFT-only, excluded on purpose
