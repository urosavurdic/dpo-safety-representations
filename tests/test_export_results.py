import json
from pathlib import Path

import pytest

from src.export_results import (
    categorize,
    collect_essential_files,
    collect_direction_vectors,
    collect_raw_activations,
    build_manifest,
    file_checksum,
    EXPORT_CATEGORIES,
)


def test_categorize_behavioral():
    assert categorize(Path("behavioral_eval/summary_v2.json")) == "behavioral"
    assert categorize(Path("behavioral_eval/raw.json")) == "behavioral"


def test_categorize_probes():
    assert categorize(Path("probes/M3_probe_results.json")) == "probes"


def test_categorize_causal_raw_and_summary():
    assert categorize(Path("raw/causal_ablation_raw_narrow.json")) == "causal"
    assert categorize(Path("raw/causal_ablation_raw_wide.json")) == "causal"
    assert categorize(Path("summaries/causal_ablation_narrow_summary.json")) == "causal"
    assert categorize(Path("summaries/bootstrap_ci_causal_ablation_raw_wide_A_refusal.json")) == "causal"


def test_categorize_steering_v1_and_v2():
    assert categorize(Path("raw/steering_raw_D.json")) == "steering"
    assert categorize(Path("raw/steering_raw_D_L21.json")) == "steering"
    assert categorize(Path("raw/steering_v2_M3_L24_quadrant_a_projection_coef1_QAD.json")) == "steering"


def test_categorize_robustness_alt_branch_causal_ablation():
    assert categorize(Path("raw/causal_ablation_raw_m1d_narrow.json")) == "robustness"
    assert categorize(Path("raw/causal_ablation_raw_m3_direct_alt_narrow.json")) == "robustness"


def test_categorize_interpretability():
    assert categorize(Path("interpretability/bootstrap_direction_stability.json")) == "interpretability"
    assert categorize(Path("interpretability/direction_stability/stability_report.json")) == "interpretability"
    assert categorize(Path("interpretability/bottleneck_layer.json")) == "interpretability"


def test_categorize_refusal_direction_json():
    assert categorize(Path("refusal_direction/cosine_similarity.json")) == "refusal_direction"
    assert categorize(Path("refusal_direction/quadrant_projections.json")) == "refusal_direction"


def test_categorize_unmatched_returns_none():
    assert categorize(Path("some_totally_unexpected/file.json")) is None


def test_categorize_bare_top_level_file_falls_back_to_diagnostics():
    # results/qualitative_spot_check.json, results/classifier_validation_sample.json
    assert categorize(Path("qualitative_spot_check.json")) == "diagnostics"
    assert categorize(Path("classifier_validation_sample.json")) == "diagnostics"


def test_categorize_activations_metadata_json_not_swallowed_as_diagnostics():
    assert categorize(Path("activations/M0_metadata.json")) == "activations_metadata"


def test_no_category_pattern_accidentally_shadows_another():
    """steering must never get miscategorized as causal (both live under raw/)."""
    steering_result = categorize(Path("raw/steering_raw_D.json"))
    causal_result = categorize(Path("raw/causal_ablation_raw_wide.json"))
    assert steering_result == "steering"
    assert causal_result == "causal"
    assert steering_result != causal_result


def test_file_checksum_is_deterministic_and_content_sensitive(tmp_path):
    f1 = tmp_path / "a.json"
    f1.write_text('{"x": 1}')
    f2 = tmp_path / "b.json"
    f2.write_text('{"x": 1}')
    f3 = tmp_path / "c.json"
    f3.write_text('{"x": 2}')
    assert file_checksum(f1) == file_checksum(f2)  # same content -> same hash
    assert file_checksum(f1) != file_checksum(f3)   # different content -> different hash


def test_collect_functions_against_a_fake_results_tree(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "results" / "behavioral_eval").mkdir(parents=True)
    (tmp_path / "results" / "behavioral_eval" / "summary_v2.json").write_text("{}")
    (tmp_path / "results" / "refusal_direction").mkdir(parents=True)
    (tmp_path / "results" / "refusal_direction" / "M3_direction.npy").write_bytes(b"fake")
    (tmp_path / "results" / "refusal_direction" / "cosine_similarity.json").write_text("{}")
    (tmp_path / "results" / "activations").mkdir(parents=True)
    (tmp_path / "results" / "activations" / "M0_pooled.npy").write_bytes(b"fake_activations")

    import src.export_results as er
    monkeypatch.setattr(er, "RESULTS_DIR", Path("results"))

    essential = collect_essential_files()
    assert len(essential) == 2  # summary_v2.json + cosine_similarity.json
    categories = {c for _, _, c in essential}
    assert categories == {"behavioral", "refusal_direction"}

    vectors = collect_direction_vectors()
    assert len(vectors) == 1
    assert vectors[0].name == "M3_direction.npy"

    raw_acts = collect_raw_activations()
    assert len(raw_acts) == 1
    assert raw_acts[0].name == "M0_pooled.npy"


def test_build_manifest_totals_and_checksums(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "results" / "behavioral_eval").mkdir(parents=True)
    f = tmp_path / "results" / "behavioral_eval" / "summary_v2.json"
    f.write_text('{"a": 1}')

    import src.export_results as er
    monkeypatch.setattr(er, "RESULTS_DIR", Path("results"))

    essential = [(f, Path("behavioral_eval/summary_v2.json"), "behavioral")]
    manifest = build_manifest(essential, [], [], Path("results_export"))
    assert manifest["total_files"] == 1
    assert manifest["total_size_bytes"] == f.stat().st_size
    assert manifest["includes_raw_activations"] is False
    assert manifest["files"][0]["sha256"] == file_checksum(f)


def test_every_export_category_pattern_is_valid_and_specific():
    # No two patterns should be byte-identical (that would make the second dead code).
    patterns = [p for _, p in EXPORT_CATEGORIES]
    assert len(patterns) == len(set(patterns))
