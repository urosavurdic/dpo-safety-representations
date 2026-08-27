"""Milestone 5A regression tests: v2 output -> legacy CPU statistics
compatibility bridge (src/analysis/v2_compat.py).

Synthetic data only - no GPU/model dependency. Each artifact type is
checked for: (1) the bridge reproduces the source content under the legacy
name, (2) stage/record/behavioral-condition identity survives the bridge,
(3) the bridged file is actually consumable by the real legacy reader
function (not just schema-equal), and (4) a pre-existing legacy artifact
the bridge did not itself produce is never silently overwritten.
"""

from __future__ import annotations

import json

import numpy as np

from src.analysis.v2_compat import (
    sync_all,
    sync_behavioral,
    sync_diagnostics,
    sync_direction,
    sync_probes,
)
from src.v2_io import binding, write_json_lf


BENCH_SHA = "a" * 64
SPLIT_SHA = "b" * 64


def make_binding(bench=BENCH_SHA, split=SPLIT_SHA):
    return binding(
        "data/frozen_v2/bench.jsonl",
        bench,
        "logs/direction_split_manifest.json",
        split,
    )


def write_source_direction(refusal_dir, stage, array, bench=BENCH_SHA, split=SPLIT_SHA):
    refusal_dir.mkdir(parents=True, exist_ok=True)
    np.save(refusal_dir / f"{stage}_v2_direction.npy", array)
    write_json_lf(
        refusal_dir / f"{stage}_v2_direction_binding.json",
        {**make_binding(bench, split), "stage": stage},
    )


def write_source_diagnostics(refusal_dir, cosine, projections, bench=BENCH_SHA, split=SPLIT_SHA):
    refusal_dir.mkdir(parents=True, exist_ok=True)
    write_json_lf(refusal_dir / "cosine_similarity_v2.json", cosine)
    write_json_lf(refusal_dir / "quadrant_projections_v2.json", projections)
    write_json_lf(
        refusal_dir / "v2_diagnostics_binding.json",
        {**make_binding(bench, split), "stages": list(projections)},
    )


def write_source_probes(root, stage, results, bench=BENCH_SHA, split=SPLIT_SHA):
    probes_v2_dir = root / "probes_v2"
    probes_v2_dir.mkdir(parents=True, exist_ok=True)
    write_json_lf(probes_v2_dir / f"{stage}_probe_results.json", results)
    write_json_lf(
        probes_v2_dir / f"{stage}_probe_binding.json",
        {**make_binding(bench, split), "stage": stage},
    )


def write_source_behavioral(root, combined, bench=BENCH_SHA, split=SPLIT_SHA):
    beh_dir = root / "behavioral_eval"
    beh_dir.mkdir(parents=True, exist_ok=True)
    write_json_lf(beh_dir / "v2_raw.json", combined)
    write_json_lf(
        beh_dir / "v2_binding.json",
        {**make_binding(bench, split), "stages": sorted(combined)},
    )


def probe_layer_row(layer):
    return {
        "layer": layer,
        "cv_accuracy_mean": 0.9,
        "cv_accuracy_std": 0.01,
        "cv_fold_scores": [0.9] * 5,
        "holdout_b_flagged_unsafe_frac": 0.1,
        "quadrant_c_flagged_unsafe_frac": 0.2,
        "quadrant_d_flagged_unsafe_frac": 0.0,
    }


# ---- direction --------------------------------------------------------


def test_sync_direction_bridges_array_and_preserves_identity(tmp_path):
    root = tmp_path / "results"
    refusal_dir = root / "refusal_direction"
    array = np.arange(12, dtype=np.float32).reshape(3, 4)
    write_source_direction(refusal_dir, "M0", array)

    assert sync_direction("M0", root=root) is True

    dest = refusal_dir / "M0_direction.npy"
    np.testing.assert_array_equal(np.load(dest), array)

    dest_binding = json.loads(
        (refusal_dir / "M0_direction_v2_compat_binding.json").read_text()
    )
    assert dest_binding["benchmark_sha256"] == BENCH_SHA
    assert dest_binding["split_manifest_sha256"] == SPLIT_SHA


def test_sync_direction_no_source_is_a_noop(tmp_path):
    root = tmp_path / "results"
    assert sync_direction("M0", root=root) is False
    assert not (root / "refusal_direction" / "M0_direction.npy").exists()


def test_sync_direction_never_overwrites_a_non_bridged_legacy_file(tmp_path):
    root = tmp_path / "results"
    refusal_dir = root / "refusal_direction"
    refusal_dir.mkdir(parents=True)
    legacy_array = np.ones((3, 4), dtype=np.float32)
    np.save(refusal_dir / "M0_direction.npy", legacy_array)

    write_source_direction(refusal_dir, "M0", np.zeros((3, 4), dtype=np.float32))

    assert sync_direction("M0", root=root) is False
    np.testing.assert_array_equal(
        np.load(refusal_dir / "M0_direction.npy"), legacy_array
    )


def test_sync_direction_force_overwrites_a_non_bridged_legacy_file(tmp_path):
    root = tmp_path / "results"
    refusal_dir = root / "refusal_direction"
    refusal_dir.mkdir(parents=True)
    np.save(refusal_dir / "M0_direction.npy", np.ones((2, 2), dtype=np.float32))

    new_array = np.zeros((2, 2), dtype=np.float32)
    write_source_direction(refusal_dir, "M0", new_array)

    assert sync_direction("M0", root=root, force=True) is True
    np.testing.assert_array_equal(
        np.load(refusal_dir / "M0_direction.npy"), new_array
    )


def test_sync_direction_rerun_after_bridge_is_a_safe_noop(tmp_path):
    root = tmp_path / "results"
    refusal_dir = root / "refusal_direction"
    array = np.arange(4, dtype=np.float32).reshape(2, 2)
    write_source_direction(refusal_dir, "M0", array)

    assert sync_direction("M0", root=root) is True
    # No new v2 output arrived and --force wasn't passed, but the dest was
    # produced by us last time -> this must not be treated as a foreign
    # file and blocked.
    assert sync_direction("M0", root=root) is True


# ---- diagnostics (cosine similarity / quadrant projections) -----------


def test_sync_diagnostics_bridges_both_files(tmp_path):
    root = tmp_path / "results"
    refusal_dir = root / "refusal_direction"
    cosine = {"adjacent": {"M0_vs_M1": [0.1, 0.2, 0.3]}}
    projections = {"M0": {"A": [1.0, 2.0], "D": [0.1, 0.2]}}
    write_source_diagnostics(refusal_dir, cosine, projections)

    assert sync_diagnostics(root=root) is True

    assert json.loads((refusal_dir / "cosine_similarity.json").read_text()) == cosine
    assert (
        json.loads((refusal_dir / "quadrant_projections.json").read_text())
        == projections
    )


def test_sync_diagnostics_never_overwrites_a_non_bridged_legacy_file(tmp_path):
    root = tmp_path / "results"
    refusal_dir = root / "refusal_direction"
    refusal_dir.mkdir(parents=True)
    legacy_cosine = {"adjacent": {"M0_vs_M1": [0.99]}}
    legacy_projections = {"M0": {"A": [0.99]}}
    write_json_lf(refusal_dir / "cosine_similarity.json", legacy_cosine)
    write_json_lf(refusal_dir / "quadrant_projections.json", legacy_projections)

    write_source_diagnostics(refusal_dir, {"adjacent": {}}, {"M0": {}})

    assert sync_diagnostics(root=root) is False
    assert (
        json.loads((refusal_dir / "cosine_similarity.json").read_text())
        == legacy_cosine
    )
    assert (
        json.loads((refusal_dir / "quadrant_projections.json").read_text())
        == legacy_projections
    )


def test_bridged_diagnostics_are_consumable_by_the_legacy_summary(
    tmp_path, monkeypatch, capsys
):
    root = tmp_path / "results"
    refusal_dir = root / "refusal_direction"
    n_layers = 29
    stages = ["M0", "M1", "M2", "M3"]
    cosine = {
        "vs_M0": {stage: [0.5] * n_layers for stage in stages},
        "adjacent": {
            "M0_vs_M1": [0.4] * n_layers,
            "M1_vs_M2": [0.4] * n_layers,
            "M2_vs_M3": [0.4] * n_layers,
        },
    }
    projections = {
        stage: {
            "A": [1.0] * n_layers,
            "B": [0.5] * n_layers,
            "C": [0.2] * n_layers,
            "D": [0.0] * n_layers,
        }
        for stage in stages
    }
    write_source_diagnostics(refusal_dir, cosine, projections)
    assert sync_diagnostics(root=root) is True

    monkeypatch.chdir(tmp_path)
    from src.analysis import summarize_refusal_direction

    summarize_refusal_direction.main()  # must run against the bridged files
    out = capsys.readouterr().out
    assert "Cosine similarity vs M0" in out


# ---- probes -------------------------------------------------------------


def test_sync_probes_bridges_and_preserves_stage_identity(tmp_path):
    root = tmp_path / "results"
    results = [probe_layer_row(layer) for layer in range(29)]
    write_source_probes(root, "M2", results)

    assert sync_probes("M2", root=root) is True

    dest = root / "probes" / "M2_probe_results.json"
    assert json.loads(dest.read_text()) == results


def test_sync_probes_never_overwrites_a_non_bridged_legacy_file(tmp_path):
    root = tmp_path / "results"
    probes_dir = root / "probes"
    probes_dir.mkdir(parents=True)
    legacy_results = [probe_layer_row(0)]
    write_json_lf(probes_dir / "M2_probe_results.json", legacy_results)

    write_source_probes(root, "M2", [probe_layer_row(layer) for layer in range(29)])

    assert sync_probes("M2", root=root) is False
    assert (
        json.loads((probes_dir / "M2_probe_results.json").read_text())
        == legacy_results
    )


def test_bridged_probes_are_consumable_by_the_legacy_summary(tmp_path, monkeypatch):
    root = tmp_path / "results"
    results = [probe_layer_row(layer) for layer in range(29)]
    write_source_probes(root, "M0", results)
    assert sync_probes("M0", root=root) is True

    monkeypatch.chdir(tmp_path)
    from src.analysis.summarize_probe_findings import load_layer

    row = load_layer("M0")
    assert row["layer"] == 28
    assert row["quadrant_c_flagged_unsafe_frac"] == 0.2


# ---- behavioral -----------------------------------------------------------


def test_sync_behavioral_adds_completion_alias_and_preserves_identity(tmp_path):
    root = tmp_path / "results"
    combined = {
        "M0": [
            {
                "record_id": "r001",
                "prompt": "How do I pick a lock?",
                "quadrant": "A",
                "condition": "M0_behavior",
                "stage": "M0_behavior",
                "model_stage": "M0",
                "response": "I can't help with that.",
                "benchmark_sha256": BENCH_SHA,
                "split_manifest_sha256": SPLIT_SHA,
            }
        ]
    }
    write_source_behavioral(root, combined)

    assert sync_behavioral(root=root) is True

    bridged = json.loads((root / "behavioral_eval" / "raw.json").read_text())
    row = bridged["M0"][0]
    # legacy field added; every original v2 field, including "response",
    # is preserved rather than renamed away.
    assert row["completion"] == "I can't help with that."
    assert row["response"] == "I can't help with that."
    assert row["record_id"] == "r001"
    assert row["condition"] == "M0_behavior"
    assert row["quadrant"] == "A"


def test_sync_behavioral_never_overwrites_a_non_bridged_legacy_file(tmp_path):
    root = tmp_path / "results"
    beh_dir = root / "behavioral_eval"
    beh_dir.mkdir(parents=True)
    legacy_raw = {
        "M0": [
            {
                "prompt": "p",
                "quadrant": "A",
                "source": "HarmBench",
                "completion": "old",
                "refused": True,
            }
        ]
    }
    write_json_lf(beh_dir / "raw.json", legacy_raw)

    write_source_behavioral(
        root, {"M0": [{"record_id": "r1", "quadrant": "A", "response": "new"}]}
    )

    assert sync_behavioral(root=root) is False
    assert json.loads((beh_dir / "raw.json").read_text()) == legacy_raw


def test_bridged_behavioral_is_consumable_by_the_legacy_cross_branch_reader(tmp_path):
    root = tmp_path / "results"
    combined = {
        "M1": [
            {
                "record_id": "r1",
                "quadrant": "A",
                "condition": "M1_behavior",
                "response": "I cannot assist with that request.",
            },
            {
                "record_id": "r2",
                "quadrant": "A",
                "condition": "M1_behavior",
                "response": "Sure, here is how you would do it step by step.",
            },
        ],
    }
    write_source_behavioral(root, combined)
    assert sync_behavioral(root=root) is True

    from src.analysis.summarize_cross_branch import (
        behavioral_rates_for_stage,
        load_raw_behavioral,
    )

    raw = load_raw_behavioral(str(root / "behavioral_eval" / "raw.json"))
    rates = behavioral_rates_for_stage(raw, "M1")

    assert rates is not None
    assert rates["A"]["refusal"]["rate"] == 0.5


# ---- orchestration --------------------------------------------------------


def test_sync_all_bridges_direction_probes_diagnostics_and_behavioral(tmp_path):
    root = tmp_path / "results"
    refusal_dir = root / "refusal_direction"
    write_source_direction(refusal_dir, "M0", np.zeros((2, 2), dtype=np.float32))
    write_source_diagnostics(refusal_dir, {"adjacent": {}}, {"M0": {}})
    write_source_probes(root, "M0", [probe_layer_row(0)])
    write_source_behavioral(
        root, {"M0": [{"record_id": "r1", "quadrant": "A", "response": "x"}]}
    )

    results = sync_all(stages=["M0"], root=root)

    assert results["direction:M0"] is True
    assert results["probes:M0"] is True
    assert results["diagnostics"] is True
    assert results["behavioral"] is True


def test_sync_all_reports_false_for_stages_with_no_v2_source_yet(tmp_path):
    root = tmp_path / "results"
    results = sync_all(stages=["M0", "M1"], root=root)

    assert results["direction:M0"] is False
    assert results["probes:M1"] is False
    assert results["diagnostics"] is False
    assert results["behavioral"] is False
