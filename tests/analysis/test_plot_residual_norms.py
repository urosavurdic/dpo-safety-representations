import json

from src.analysis.plot_residual_norms import (
    load_diagnostic,
    most_anomalous_layer,
    norm_matrix_for_prompt,
    plot_anomalous_layer_comparison,
    plot_condition_heatmaps,
)


def _synthetic_diagnostic(tmp_path):
    """Mirrors eval_residual_norm_diagnostic.py's real output schema:
    {stage, quadrant, baseline_range, conditions: {name: {summary, prompts:
    [{prompt, response, is_degenerate, norm_records, norm_summary}]}}}.
    Simulates the qualitative pattern the deprecated runs showed: baseline
    stays flat, "collapsing" grows unboundedly at a deep layer, "noncollapsing"
    stays close to baseline."""
    layers = ["20", "24", "27"]
    n_steps = 6

    def make_records(kind):
        records = {}
        for layer in layers:
            base = 15.0 + int(layer) * 0.1
            if kind == "baseline":
                records[layer] = [[base] for _ in range(n_steps)]
            elif kind == "collapsing" and layer == "27":
                records[layer] = [[base + 5.0 * step] for step in range(n_steps)]
            elif kind == "noncollapsing" and layer == "27":
                records[layer] = [[base + 0.2 * step] for step in range(n_steps)]
            else:
                records[layer] = [[base] for _ in range(n_steps)]
        return records

    baseline_records = [make_records("baseline") for _ in range(3)]
    baseline_pooled = {layer: [] for layer in layers}
    for r in baseline_records:
        for layer in layers:
            baseline_pooled[layer].extend(r[layer])

    from src.interpretability.residual_norm_tracking import compare_to_baseline, compute_baseline_range
    baseline_range = compute_baseline_range(baseline_pooled)

    def prompt_entries(kind, n=2):
        entries = []
        for i in range(n):
            records = make_records(kind)
            norm_summary = None
            if kind != "baseline":
                comparison = compare_to_baseline(records, baseline_range)
                norm_summary = {
                    layer: {
                        "first_step_exceeding_p99": next(
                            (e["step"] for e in entries_ if e["exceeds_p99"]), None
                        ),
                        "max_z_score": max((e["z_score"] for e in entries_), default=None),
                    }
                    for layer, entries_ in comparison.items()
                }
            entries.append({
                "prompt": f"prompt {i}", "response": "some text",
                "is_degenerate": (kind == "collapsing"),
                "norm_records": records, "norm_summary": norm_summary,
            })
        return entries

    diagnostic = {
        "stage": "M3", "quadrant": "D", "baseline_range": baseline_range,
        "conditions": {
            "baseline": {"summary": {"n_prompts": 3, "n_degenerate": 0, "degenerate_rate": 0.0},
                         "prompts": prompt_entries("baseline", n=3)},
            "collapsing": {"summary": {"n_prompts": 2, "n_degenerate": 2, "degenerate_rate": 1.0},
                           "prompts": prompt_entries("collapsing")},
            "noncollapsing": {"summary": {"n_prompts": 2, "n_degenerate": 0, "degenerate_rate": 0.0},
                              "prompts": prompt_entries("noncollapsing")},
        },
    }

    path = tmp_path / "residual_norm_diagnostic_M3.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(diagnostic, f)
    return path, diagnostic


def test_load_diagnostic_round_trips(tmp_path):
    path, diagnostic = _synthetic_diagnostic(tmp_path)
    loaded = load_diagnostic(path)
    assert loaded["stage"] == "M3"
    assert set(loaded["conditions"].keys()) == {"baseline", "collapsing", "noncollapsing"}


def test_norm_matrix_for_prompt_shape_and_values(tmp_path):
    _, diagnostic = _synthetic_diagnostic(tmp_path)
    layer_indices, matrix = norm_matrix_for_prompt(diagnostic["conditions"]["collapsing"], 0)
    assert layer_indices == [20, 24, 27]
    assert matrix.shape == (6, 3)
    # layer 27's norm should be growing over generation steps for the collapsing condition
    layer27_col = matrix[:, layer_indices.index(27)]
    assert (layer27_col == sorted(layer27_col)).all()
    assert layer27_col[-1] > layer27_col[0]


def test_most_anomalous_layer_identifies_layer_27_for_collapsing():
    layers = ["27"]
    prompts = [
        {"norm_summary": {"27": {"first_step_exceeding_p99": 2, "max_z_score": 50.0}}},
        {"norm_summary": {"27": {"first_step_exceeding_p99": 1, "max_z_score": 5.0}}},
    ]
    cond = {"prompts": prompts}
    layer, z = most_anomalous_layer(cond)
    assert layer == "27"
    assert z == 50.0


def test_most_anomalous_layer_none_when_no_norm_summary():
    cond = {"prompts": [{"norm_summary": None}]}
    layer, z = most_anomalous_layer(cond)
    assert layer is None and z is None


def test_plot_condition_heatmaps_produces_real_png(tmp_path):
    _, diagnostic = _synthetic_diagnostic(tmp_path)
    out_path = tmp_path / "heatmaps.png"
    result_path = plot_condition_heatmaps(diagnostic, prompt_index=0, out_path=out_path)
    assert result_path == out_path
    assert out_path.exists()
    assert out_path.stat().st_size > 1000  # a real rendered image, not an empty/stub file


def test_plot_anomalous_layer_comparison_produces_real_png(tmp_path):
    _, diagnostic = _synthetic_diagnostic(tmp_path)
    out_path = tmp_path / "anomalous_layer.png"
    result_path = plot_anomalous_layer_comparison(diagnostic, out_path)
    assert result_path == out_path
    assert out_path.exists()
    assert out_path.stat().st_size > 1000


def test_plot_condition_heatmaps_handles_missing_prompt_index_gracefully(tmp_path):
    _, diagnostic = _synthetic_diagnostic(tmp_path)
    out_path = tmp_path / "heatmaps_oob.png"
    # Only 2 prompts in "collapsing"/"noncollapsing", 3 in "baseline" -- index 2 is out of range for two of them.
    result_path = plot_condition_heatmaps(diagnostic, prompt_index=2, out_path=out_path)
    assert result_path.exists()
