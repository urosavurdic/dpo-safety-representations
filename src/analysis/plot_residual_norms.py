"""
Reads a results/raw/residual_norm_diagnostic_{stage}.json file (produced by
eval_residual_norm_diagnostic.py) and produces the "norm vs layer vs
generation step" plots Next Steps item 4 asks for -- one heatmap per
condition (layer x generation-step, colored by norm) for a representative
prompt, plus a line plot at the single most-anomalous layer comparing all
conditions against the baseline's trained-range ceiling (p99).

Pure numpy/matplotlib/json -- no torch, no model, fully CPU-runnable and
testable against synthetic data matching the diagnostic script's real
output schema (see tests/analysis/test_plot_residual_norms.py, which
exercises this end-to-end and actually renders PNGs in this sandbox, since
nothing here needs a GPU or the real model).

Usage:
    python -m src.analysis.plot_residual_norms --file results/raw/residual_norm_diagnostic_M3.json
"""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # headless -- this may run on a machine with no display
import matplotlib.pyplot as plt
import numpy as np


def load_diagnostic(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def norm_matrix_for_prompt(condition_data, prompt_index):
    """Returns (layer_indices, steps_x_layers) where steps_x_layers is a
    (n_steps, n_layers) array of norms for ONE prompt -- built from
    norm_records = {decoder_idx_str: [step][batch] of norms}, batch=1 for
    this diagnostic script, so [step][0] is the value used."""
    prompt_entry = condition_data["prompts"][prompt_index]
    norm_records = prompt_entry["norm_records"]
    layer_indices = sorted(int(idx) for idx in norm_records.keys())
    n_steps = max(len(norm_records[str(idx)]) for idx in layer_indices)

    matrix = np.full((n_steps, len(layer_indices)), np.nan)
    for col, idx in enumerate(layer_indices):
        steps = norm_records[str(idx)]
        for row, step_vals in enumerate(steps):
            matrix[row, col] = float(np.mean(step_vals))
    return layer_indices, matrix


def most_anomalous_layer(condition_data):
    """Across all prompts in this condition, which decoder layer had the
    single largest max_z_score against the baseline range? Returns
    (layer_idx_str, max_z) or (None, None) if no norm_summary is present
    (e.g. this IS the baseline condition)."""
    best_layer, best_z = None, -float("inf")
    for prompt_entry in condition_data["prompts"]:
        norm_summary = prompt_entry.get("norm_summary")
        if not norm_summary:
            continue
        for layer_idx, s in norm_summary.items():
            z = s.get("max_z_score")
            if z is not None and z > best_z:
                best_z, best_layer = z, layer_idx
    return best_layer, (best_z if best_layer is not None else None)


def plot_heatmap(ax, layer_indices, matrix, title):
    im = ax.imshow(matrix.T, aspect="auto", origin="lower", cmap="viridis",
                    extent=[0, matrix.shape[0], layer_indices[0], layer_indices[-1]])
    ax.set_xlabel("generation step")
    ax.set_ylabel("decoder layer")
    ax.set_title(title)
    return im


def plot_condition_heatmaps(diagnostic, prompt_index, out_path):
    conditions = diagnostic["conditions"]
    names = list(conditions.keys())
    fig, axes = plt.subplots(1, len(names), figsize=(5 * len(names), 4), squeeze=False)
    axes = axes[0]

    for ax, name in zip(axes, names):
        cond = conditions[name]
        if prompt_index >= len(cond["prompts"]):
            ax.set_title(f"{name} (no prompt {prompt_index})")
            ax.axis("off")
            continue
        layer_indices, matrix = norm_matrix_for_prompt(cond, prompt_index)
        im = plot_heatmap(ax, layer_indices, matrix, name)
        fig.colorbar(im, ax=ax, label="residual norm")

    fig.suptitle(f"{diagnostic['stage']} / quadrant {diagnostic['quadrant']} -- "
                 f"prompt {prompt_index}: {conditions[names[0]]['prompts'][prompt_index]['prompt'][:60]!r}")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def plot_anomalous_layer_comparison(diagnostic, out_path):
    conditions = diagnostic["conditions"]
    baseline_range = diagnostic["baseline_range"]

    # Pick the layer with the largest z-score seen in ANY non-baseline condition.
    global_best_layer, global_best_z = None, -float("inf")
    for name, cond in conditions.items():
        if name == "baseline":
            continue
        layer, z = most_anomalous_layer(cond)
        if z is not None and z > global_best_z:
            global_best_z, global_best_layer = z, layer

    fig, ax = plt.subplots(figsize=(7, 4.5))
    if global_best_layer is None:
        ax.set_title("No non-baseline condition with norm_summary data found")
        fig.savefig(out_path, dpi=150)
        plt.close(fig)
        return out_path

    for name, cond in conditions.items():
        if not cond["prompts"]:
            continue
        norm_records = cond["prompts"][0]["norm_records"]
        if global_best_layer not in norm_records:
            continue
        steps = norm_records[global_best_layer]
        y = [float(np.mean(v)) for v in steps]
        ax.plot(range(len(y)), y, label=name, marker="o", markersize=3)

    p99 = baseline_range.get(global_best_layer, {}).get("p99")
    if p99 is not None:
        ax.axhline(p99, color="red", linestyle="--", label=f"baseline p99 (layer {global_best_layer})")

    ax.set_xlabel("generation step")
    ax.set_ylabel("residual norm (mean over batch)")
    ax.set_title(f"Most-anomalous layer ({global_best_layer}) across conditions, prompt 0")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", required=True)
    parser.add_argument("--prompt-index", type=int, default=0)
    parser.add_argument("--out-dir", default="results/figures")
    args = parser.parse_args()

    diagnostic = load_diagnostic(args.file)
    stage = diagnostic["stage"]
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    heatmap_path = out_dir / f"residual_norm_heatmaps_{stage}_prompt{args.prompt_index}.png"
    line_path = out_dir / f"residual_norm_anomalous_layer_{stage}.png"

    plot_condition_heatmaps(diagnostic, args.prompt_index, heatmap_path)
    plot_anomalous_layer_comparison(diagnostic, line_path)

    print(f"Wrote {heatmap_path}")
    print(f"Wrote {line_path}")

    print("\nDegenerate rates by condition:")
    for name, cond in diagnostic["conditions"].items():
        s = cond["summary"]
        if s["degenerate_rate"] is not None:
            print(f"  {name:28s} {s['n_degenerate']}/{s['n_prompts']} ({s['degenerate_rate']:.0%})")


if __name__ == "__main__":
    main()
