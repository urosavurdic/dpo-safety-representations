"""
Single local entry point for reproducing the CPU-feasible parts of the
analysis pipeline.

Training (all 8 stages) and the GPU-generation components (behavioral eval,
activation extraction, causal ablation, steering) need a GPU and are NOT run
by this script - see colab_unified_training.ipynb / colab_unified_analysis.ipynb
for those. Everything below operates on artifacts those GPU steps already
produced (pulled from Drive, or already committed to the repo for M0-M3),
and is genuinely CPU-only: reclassification/summarization, probes (already-
extracted activations, no model), refusal-direction geometry, direction
stability, bootstrap analyses, bottleneck-layer analysis, and causal-
ablation statistics.

Usage:
    python -m src.reproduce --list                          # show status, run nothing
    python -m src.reproduce --components probes direction   # run just these
    python -m src.reproduce --components all                # run everything with satisfied prerequisites
    python -m src.reproduce --components all --force         # also rerun components whose outputs already exist

Never overwrites a component's outputs unless --force is passed. Writes a
manifest to results/manifests/ every run recording git commit, timestamp,
what was requested/run/skipped/blocked, and the resulting artifact paths.
"""
import argparse
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

COMPONENTS = {
    "behavioral_stats": {
        "description": "Wilson-CI behavioral summary from already-generated raw generations",
        "requires": ["results/behavioral_eval/raw.json"],
        "produces": ["results/behavioral_eval/summary_v2.json"],
        "commands": ["python -m src.analysis.reclassify_behavioral"],
    },
    "probes": {
        "description": "Linear probes + held-out flagging rate (needs activations, not the model)",
        "requires": ["results/activations"],
        "produces": ["results/probes"],
        "commands": [
            "python -m src.analysis.eval_probes",
            "python -m src.analysis.summarize_probe_findings",
        ],
    },
    "direction": {
        "description": "Refusal-direction geometry, stability, bootstrap, bottleneck-layer analysis",
        "requires": ["results/activations"],
        "produces": [
            "results/refusal_direction/cosine_similarity.json",
            "results/interpretability/direction_stability/stability_report.json",
            "results/interpretability/bootstrap_direction_stability.json",
            "results/interpretability/bottleneck_layer.json",
            "results/interpretability/bootstrap_cross_branch_difference.json",
            "results/interpretability/paired_deep_layer_stability_test.json",
        ],
        "commands": [
            "python -m src.analysis.eval_refusal_direction",
            "python -m src.interpretability.direction_stability",
            "python -m src.interpretability.bootstrap_direction_stability",
            "python -m src.interpretability.bottleneck_layer",
            "python -m src.interpretability.bootstrap_cross_branch_difference",
            "python -m src.interpretability.paired_deep_layer_stability_test",
        ],
    },
    "causal_stats": {
        "description": "Wilson-CI summary, McNemar test, bootstrap CI on an already-generated causal-ablation raw file",
        "requires": ["results/raw/causal_ablation_raw_narrow.json"],
        "produces": ["results/summaries/causal_ablation_narrow_summary.json"],
        "commands": [
            "python -m src.analysis.summarize_causal_ablation --file results/raw/causal_ablation_raw_narrow.json --stage M3",
            "python -m src.analysis.mcnemar_causal_ablation --file results/raw/causal_ablation_raw_narrow.json --conditions M3_baseline M3_ablated",
            "python -m src.analysis.bootstrap_causal_effect --file results/raw/causal_ablation_raw_narrow.json --quadrant C --category soft_deflection",
        ],
    },
}

GPU_ONLY_COMPONENTS = {
    "training": "colab_unified_training.ipynb (all 8 stages)",
    "behavioral_generation": "src.analysis.eval_behavioral (colab_unified_analysis.ipynb Component 1)",
    "activation_extraction": "src.analysis.eval_extract_activations (colab_unified_analysis.ipynb Component 2)",
    "causal_ablation_generation": "src.analysis.eval_causal_ablation (colab_unified_analysis.ipynb Component 5)",
    "steering_generation": "src.analysis.eval_steering_v2 (colab_unified_analysis.ipynb Component 5b)",
}


def get_git_commit():
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"]).decode().strip()
    except Exception:
        return "unknown"


def artifact_exists(path):
    p = Path(path)
    if p.is_dir():
        return p.exists() and any(p.iterdir())
    return p.exists()


def missing_requirements(component_name):
    return [r for r in COMPONENTS[component_name]["requires"] if not artifact_exists(r)]


def already_produced(component_name):
    return all(artifact_exists(p) for p in COMPONENTS[component_name]["produces"])


def resolve_component_order(selected):
    """Simple stable order: dict insertion order of COMPONENTS, filtered to
    `selected` -- no cross-component dependency graph needed today since
    each component's prerequisites are GPU-produced artifacts, not other
    local components' outputs. If a future component depends on another
    local component's output, extend this the same way
    stage_registry.resolve_run_order does."""
    unknown = set(selected) - set(COMPONENTS)
    if unknown:
        raise ValueError(f"Unknown component(s): {sorted(unknown)}. Valid: {sorted(COMPONENTS)} or 'all'")
    return [c for c in COMPONENTS if c in selected]


def run_component(name):
    for cmd in COMPONENTS[name]["commands"]:
        print(f"  $ {cmd}")
        result = subprocess.run(cmd, shell=True)
        if result.returncode != 0:
            return False
    return True


def write_manifest(requested, results, out_dir="results/manifests"):
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc)
    manifest = {
        "git_commit": get_git_commit(),
        "timestamp_utc": timestamp.isoformat(),
        "requested_components": requested,
        "results": results,
        "produced_artifacts": {
            name: COMPONENTS[name]["produces"] for name in results if results[name]["status"] == "ran"
        },
    }
    manifest_path = Path(out_dir) / f"reproduce_{timestamp.strftime('%Y%m%dT%H%M%SZ')}.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    return manifest_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--components", nargs="+", default=None,
                         help="Component names to run, or 'all'. See --list for the full set.")
    parser.add_argument("--force", action="store_true",
                         help="Rerun a component even if its outputs already exist (still never "
                              "deletes anything - the underlying scripts overwrite their own derived "
                              "summary files, never raw GPU-generated data)")
    parser.add_argument("--list", action="store_true", help="Show component status and exit; runs nothing")
    args = parser.parse_args()

    if args.list or args.components is None:
        print("CPU-feasible components (this script):\n")
        for name, spec in COMPONENTS.items():
            missing = missing_requirements(name)
            produced = already_produced(name)
            status = "BLOCKED (missing prerequisites)" if missing else ("done" if produced else "ready to run")
            print(f"  {name:16s} [{status}]")
            print(f"    {spec['description']}")
            if missing:
                print(f"    missing: {missing}")
            print()
        print("GPU-only (not run by this script - use the Colab notebooks):\n")
        for name, where in GPU_ONLY_COMPONENTS.items():
            print(f"  {name:28s} -> {where}")
        if args.components is None:
            print("\nPass --components <names> or --components all to actually run something.")
        return

    selected = list(COMPONENTS.keys()) if args.components == ["all"] else args.components
    order = resolve_component_order(selected)

    results = {}
    for name in order:
        missing = missing_requirements(name)
        if missing:
            print(f"\n=== {name}: BLOCKED, missing {missing} ===")
            results[name] = {"status": "blocked", "missing": missing}
            continue
        if already_produced(name) and not args.force:
            print(f"\n=== {name}: already produced, skipping (--force to rerun) ===")
            results[name] = {"status": "skipped_already_done"}
            continue

        print(f"\n=== {name}: running ===")
        ok = run_component(name)
        results[name] = {"status": "ran" if ok else "failed"}
        if not ok:
            print(f"  {name} FAILED - stopping (later components may depend on this one's output)")
            break

    manifest_path = write_manifest(selected, results)
    print(f"\nManifest written to {manifest_path}")
    print("\nSummary:")
    for name, r in results.items():
        print(f"  {name:16s} {r['status']}")


if __name__ == "__main__":
    main()
