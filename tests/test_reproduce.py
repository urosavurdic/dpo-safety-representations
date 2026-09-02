import json
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

from src.reproduce import (
    COMPONENTS,
    artifact_exists,
    missing_requirements,
    already_produced,
    resolve_component_order,
    run_component,
    write_manifest,
)


def test_every_component_has_required_fields():
    for name, spec in COMPONENTS.items():
        assert "description" in spec
        assert isinstance(spec["requires"], list) and spec["requires"]
        assert isinstance(spec["produces"], list) and spec["produces"]
        assert isinstance(spec["commands"], list) and spec["commands"]


def test_artifact_exists_for_file(tmp_path):
    f = tmp_path / "x.json"
    assert artifact_exists(str(f)) is False
    f.write_text("{}")
    assert artifact_exists(str(f)) is True


def test_artifact_exists_for_directory_requires_nonempty(tmp_path):
    d = tmp_path / "activations"
    assert artifact_exists(str(d)) is False  # doesn't exist yet
    d.mkdir()
    assert artifact_exists(str(d)) is False  # exists but empty -> not a real artifact
    (d / "M0_pooled.npy").write_bytes(b"x")
    assert artifact_exists(str(d)) is True


def test_missing_requirements_reports_only_absent_paths(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    # behavioral_stats requires results/behavioral_eval/raw.json
    missing = missing_requirements("behavioral_stats")
    assert missing == ["results/behavioral_eval/raw.json"]

    Path("results/behavioral_eval").mkdir(parents=True)
    Path("results/behavioral_eval/raw.json").write_text("{}")
    assert missing_requirements("behavioral_stats") == []


def test_already_produced_true_only_when_all_outputs_exist(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    assert already_produced("causal_stats") is False
    Path("results/summaries").mkdir(parents=True)
    Path("results/summaries/causal_ablation_v2_M3_L24-28_summary.json").write_text("{}")
    assert already_produced("causal_stats") is True


def test_resolve_component_order_preserves_registry_order_regardless_of_input_order():
    order = resolve_component_order(["direction", "behavioral_stats"])
    assert order == ["behavioral_stats", "direction"]  # registry order, not input order


def test_resolve_component_order_rejects_unknown_component():
    with pytest.raises(ValueError, match="Unknown component"):
        resolve_component_order(["not_a_real_component"])


def test_run_component_stops_and_reports_failure(monkeypatch):
    calls = []

    def fake_run(cmd, shell):
        calls.append(cmd)
        result = MagicMock()
        result.returncode = 1  # first command fails
        return result

    with patch("subprocess.run", side_effect=fake_run):
        ok = run_component("probes")  # has 2 commands
    assert ok is False
    assert len(calls) == 1  # must NOT run the second command after the first fails


def test_run_component_runs_all_commands_on_success(monkeypatch):
    calls = []

    def fake_run(cmd, shell):
        calls.append(cmd)
        result = MagicMock()
        result.returncode = 0
        return result

    with patch("subprocess.run", side_effect=fake_run):
        ok = run_component("probes")
    assert ok is True
    assert len(calls) == 2


def test_write_manifest_records_git_commit_and_results(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    with patch("src.reproduce.get_git_commit", return_value="deadbeef"):
        manifest_path = write_manifest(
            requested=["probes"],
            results={"probes": {"status": "ran"}},
            out_dir="results/manifests",
        )
    assert manifest_path.exists()
    with open(manifest_path) as f:
        manifest = json.load(f)
    assert manifest["git_commit"] == "deadbeef"
    assert manifest["requested_components"] == ["probes"]
    assert "probes" in manifest["produced_artifacts"]


def test_write_manifest_excludes_non_ran_components_from_produced_artifacts(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    with patch("src.reproduce.get_git_commit", return_value="abc"):
        manifest_path = write_manifest(
            requested=["probes", "direction"],
            results={"probes": {"status": "skipped_already_done"}, "direction": {"status": "blocked"}},
        )
    with open(manifest_path) as f:
        manifest = json.load(f)
    assert manifest["produced_artifacts"] == {}
