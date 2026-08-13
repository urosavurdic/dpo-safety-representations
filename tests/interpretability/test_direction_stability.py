import json

from src.interpretability.direction_stability import analyze_direction_stability


def _write_toy_cosine_sim(path):
    data = {
        "vs_M0": {
            "M0": [1.0, 1.0],
            "M1": [0.5, 0.6],
            "M2": [0.5, 0.55],
            "M3": [0.4, 0.5],
        },
        "adjacent": {
            "M0_vs_M1": [0.5, 0.6],
            "M1_vs_M2": [0.95, 0.9],
            "M2_vs_M3": [0.8, 0.85],
        },
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f)


def test_per_layer_stability_reads_stage_keyed_lists_correctly(tmp_path):
    cosine_path = tmp_path / "cosine_similarity.json"
    _write_toy_cosine_sim(cosine_path)
    out_path = tmp_path / "out" / "stability_report.json"

    results = analyze_direction_stability(str(cosine_path), str(out_path))

    assert results["per_layer_stability"][0]["M3"] == 0.4
    assert results["per_layer_stability"][1]["M3"] == 0.5
    assert out_path.exists()


def test_biggest_drift_correctly_identified_as_M0_to_M1(tmp_path):
    cosine_path = tmp_path / "cosine_similarity.json"
    _write_toy_cosine_sim(cosine_path)
    out_path = tmp_path / "out" / "stability_report.json"

    results = analyze_direction_stability(str(cosine_path), str(out_path))

    assert "M0_vs_M1" in results["interpretation"]["summary"]