"""
Regression tests for the causal_stats component of src/reproduce.py.

Background (see logs/release_gap_audit.md section 5A): causal_stats used
to point at results/raw/causal_ablation_raw_narrow.json, a file that uses
the field name `model_stage` and M3_direct_baseline/M3_direct_ablated
condition names -- while the three summarization scripts it calls
(summarize_causal_ablation.py, mcnemar_causal_ablation.py,
bootstrap_causal_effect.py) all read row["stage"] and expect
M3_baseline/M3_ablated. On top of that, one of the invocations passed
--stage M3, a flag none of the three scripts accept. The result was a
documented, single-command reproduction path that could not execute.

tests/analysis/test_summarize_causal_ablation.py only ever exercised
classify_completion() in isolation, so it never caught this. These tests
exercise the actual wiring: the exact commands registered in
COMPONENTS["causal_stats"], against the real committed input file.
"""
import os
import shutil
import subprocess
from pathlib import Path

from src.io_utils import load_json
from src.reproduce import COMPONENTS

REPO_ROOT = Path(__file__).resolve().parents[1]


def _file_arg(cmd):
    parts = cmd.split()
    return parts[parts.index("--file") + 1]


def test_causal_stats_requires_and_commands_reference_one_consistent_file():
    commands = COMPONENTS["causal_stats"]["commands"]
    input_paths = {_file_arg(cmd) for cmd in commands}
    assert len(input_paths) == 1, f"causal_stats commands disagree on --file: {input_paths}"
    assert COMPONENTS["causal_stats"]["requires"] == [next(iter(input_paths))]


def test_causal_stats_input_file_uses_stage_key_and_expected_conditions():
    """
    Guards against causal_stats being repointed at a file (like the old
    *_narrow.json) whose schema the summarization scripts can't read.
    """
    input_path = COMPONENTS["causal_stats"]["requires"][0]
    rows = load_json(REPO_ROOT / input_path)
    assert rows, "expected at least one row in the causal_stats input file"
    assert "stage" in rows[0], (
        f"causal_stats input file must use the 'stage' key (found keys: {sorted(rows[0].keys())})"
    )
    stages = {r["stage"] for r in rows}
    assert stages == {"M3_baseline", "M3_ablated"}, f"unexpected stage values: {stages}"


def test_causal_stats_commands_have_no_unsupported_stage_flag():
    for cmd in COMPONENTS["causal_stats"]["commands"]:
        parts = cmd.split()
        assert "--stage" not in parts, f"'--stage' is not accepted by any causal_stats script: {cmd}"


def test_causal_stats_commands_run_end_to_end(tmp_path):
    """
    Runs the exact registered commands, in an isolated cwd, against a copy
    of the real committed input file. This is the actual reproduction path
    a researcher running `python -m src.reproduce --components causal_stats`
    would hit.
    """
    commands = COMPONENTS["causal_stats"]["commands"]
    input_path = COMPONENTS["causal_stats"]["requires"][0]

    src_input = REPO_ROOT / input_path
    dst_input = tmp_path / input_path
    dst_input.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy(src_input, dst_input)

    env = dict(os.environ)
    env["PYTHONPATH"] = str(REPO_ROOT) + os.pathsep + env.get("PYTHONPATH", "")

    for cmd in commands:
        result = subprocess.run(
            cmd, shell=True, cwd=tmp_path, env=env, capture_output=True, text=True,
        )
        assert result.returncode == 0, (
            f"causal_stats command failed: {cmd}\n"
            f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
        )

    for produced in COMPONENTS["causal_stats"]["produces"]:
        assert (tmp_path / produced).exists(), f"expected produced artifact missing: {produced}"
