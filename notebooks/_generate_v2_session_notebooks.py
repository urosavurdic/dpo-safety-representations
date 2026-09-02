"""Generate the thin v2 T4 session notebooks 00-05 + 04b (WP-NB).

Run:  python notebooks/_generate_v2_session_notebooks.py

Each notebook is a THIN shell: markdown section headers in the frozen order +
code cells that shell out to `src.analysis.v2_pipeline` / the WP scripts. No
analysis logic lives in a notebook. Each targets 240-270 min wall clock, hard
boundary 300 (analysis_plan.md §7). Notebooks are not git-tracked (repo
convention) - this generator is the source of truth.
"""
import json
from pathlib import Path

NB_DIR = Path(__file__).resolve().parent
TARGET = ("Target this session at **240-270 min** wall clock; hard boundary "
          "**300 min**. Do not start a stage/condition that the calibrated "
          "projection says cannot finish (analysis_plan.md §7).")


def md(text):
    return {"cell_type": "markdown", "metadata": {}, "source": text.splitlines(keepends=True)}


def code(text):
    return {"cell_type": "code", "metadata": {}, "outputs": [], "execution_count": None,
            "source": text.splitlines(keepends=True)}


def notebook(cells):
    return {
        "cells": cells,
        "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python",
                                    "name": "python3"},
                     "language_info": {"name": "python"}},
        "nbformat": 4, "nbformat_minor": 5,
    }


NOTEBOOKS = {
    "00_setup_and_verify.ipynb": [
        md("# S0 - setup, pin, verify\n\n" + TARGET),
        md("## 1. Mount Drive"),
        code("from google.colab import drive\ndrive.mount('/content/drive')"),
        md("## 2. Clone and pin the exact commit"),
        code("!git clone https://github.com/<owner>/dpo-safety-representations.git\n"
             "%cd dpo-safety-representations\n!git checkout <PINNED_COMMIT>"),
        md("## 3. Persistent storage"),
        code("import os; os.makedirs('/content/drive/MyDrive/dpo_v2', exist_ok=True)"),
        md("## 4. Install dependencies, check GPU"),
        code("!pip -q install -r requirements-lock.txt\n!nvidia-smi"),
        md("## 5. Benchmark, split-manifest, and gate verification"),
        code("!python -m src.validate_benchmark_v2\n"
             "!python -c \"from src.v2_io import load_run_inputs; print(load_run_inputs())\""),
        md("## 6. Focused test gate"),
        code("V2_TEST_SCOPE = [\n"
             "  'tests/test_v2_binding_guard.py', 'tests/test_v2_io_binding_contracts.py',\n"
             "  'tests/analysis/test_verify_activations.py', 'tests/analysis/test_intervention_conditions.py',\n"
             "]\n!python -m pytest {' '.join(V2_TEST_SCOPE)} -q"),
        md("## 7. Current progress"),
        code("!python -m src.analysis.v2_pipeline status"),
    ],
    "01_calibrate_and_extract.ipynb": [
        md("# S1 - calibrate + extract (`_final` + `_pooled` + source_overt adjunct)\n\n" + TARGET),
        md("## 1. Throughput calibration"),
        code("!python -m src.analysis.v2_pipeline calibrate --stage M3 --n-prompts 32"),
        md("## 2. Build the source_overt adjunct companion set"),
        code("!python -m src.analysis.build_c_source_overt_adjunct"),
        md("## 3. Extract activations (stage-major, resumable)"),
        code("!python -m src.analysis.v2_pipeline extract "
             "--stages M0 M1 M2 M3 M3_direct M1_alt M2_alt M3_alt M3_direct_alt"),
        md("## 4. CPU cross-check: all stages bound to the frozen benchmark"),
        code("!python -m src.analysis.verify_activations"),
    ],
    "02_behavioral_generation.ipynb": [
        md("# S2 - behavioural generation -> per-session manifest\n\n" + TARGET),
        md("## 1. Generate (every quadrant, baseline condition)"),
        code("!python -m src.analysis.v2_pipeline behavior "
             "--stages M0 M1 M2 M3 M3_direct M1_alt M2_alt M3_alt M3_direct_alt"),
        md("## 2. Confirm the per-session manifest"),
        code("!ls -t results/manifests | head -3"),
    ],
    "03_directions_probes_projections.ipynb": [
        md("# S3 - directions + probes + control_directions + projections\n\n" + TARGET),
        md("## 1. Directions (force past stale 370-era outputs)"),
        code("!python -m src.analysis.v2_pipeline direction --force "
             "--stages M0 M1 M2 M3 M3_direct M1_alt M2_alt M3_alt M3_direct_alt"),
        md("## 2. Probes (fixed FINAL_LAYER headline; no C/D selection)"),
        code("!python -m src.analysis.v2_pipeline probes "
             "--stages M0 M1 M2 M3 M3_direct M1_alt M2_alt M3_alt M3_direct_alt"),
        md("## 3. Control directions (seeded r, calibration-RMS gamma, d_AB)"),
        code("!python -m src.analysis.control_directions"),
        md("## 4. Canonical `_final` per-prompt + fixed-reference projections"),
        code("!python -m src.analysis.representation_projections"),
        md("## 5. Decide `ablated_AB` by calibrated session fit"),
        code("from src.analysis.intervention_conditions import plan_causal_conditions\n"
             "print(plan_causal_conditions('M3', per_condition_minutes=<measured>, "
             "budget_minutes=270, requested=['baseline','ablated_AD','ablated_random','ablated_AB']).to_json())"),
    ],
    "04_causal.ipynb": [
        md("# S4 - causal: baseline / ablated_AD / ablated_random [/ ablated_AB]\n\n" + TARGET),
        md("## 1. Required conditions (always)"),
        code("!python -m src.analysis.v2_pipeline causal --stage M3 "
             "--conditions baseline ablated_AD ablated_random"),
        md("## 2. ablated_AB only if step 5 of S3 said it fits"),
        code("# !python -m src.analysis.v2_pipeline causal --stage M3 --conditions ablated_AB"),
        md("## 3. Secondary stages (M3_direct / M3_alt / M3_direct_alt) if time remains"),
        code("# for stage in ['M3_direct','M3_alt','M3_direct_alt']: ..."),
    ],
    "04b_judge_preflight.ipynb": [
        md("# 04b - StrongREJECT / WildGuard preflight (blocker B1)\n\n"
           "Load both judges at 8-bit on toy pairs, print VRAM + versions. Branch "
           "per analysis_plan.md §10 row 10 on failure. **No full run here.**"),
        md("## 1. Load StrongREJECT fine-tuned Gemma-2B"),
        code("from src.analysis.behavioral_judges import LazyModelJudge, parse_strongreject_output\n"
             "sr = LazyModelJudge('strong_reject', 'dsbowen/strong_reject'); print(sr.try_load())"),
        md("## 2. Load WildGuard"),
        code("wg = LazyModelJudge('wildguard', 'allenai/wildguard'); print(wg.try_load())"),
        md("## 3. Pin `model_id@revision` into docs/audit/analysis_plan.md (B3)"),
        code("# record the resolved revision hashes here"),
        md("## 4. Parser smoke test (no model needed)"),
        code("print(parse_strongreject_output('1.b 0\\n2.b 4\\n3.b 3\\n'))"),
    ],
    "05_steering_manifest_judge.ipynb": [
        md("# S5 - steering + consolidated manifest + S6 judge\n\n" + TARGET),
        md("## 1. Steering: learned vs random, dose-response {0.5, 1.0, 2.0}"),
        code("!python -m src.analysis.v2_pipeline steering --stage M3 --alpha-coefficients 0.5 1.0 2.0\n"
             "!python -m src.analysis.v2_pipeline steering --stage M3_alt --alpha-coefficients 0.5 1.0 2.0"),
        md("## 2. If tight: cut M1/M2 dose-response FIRST (never the random control)"),
        code("from src.analysis.intervention_conditions import steering_cut_order\nprint(steering_cut_order())"),
        md("## 3. Build the consolidated response manifest (AFTER S2 + S4 + S5)"),
        code("!python -m src.analysis.behavioral_judges "
             "--response-manifest results/manifests/consolidated_$(date -u +%Y%m%dT%H%M%SZ).json "
             "--build-consolidated results/manifests/<s2>.json results/manifests/<s4>.json results/manifests/<s5>.json "
             "--benchmark-sha256 <BENCH_SHA> --split-manifest-sha256 <SPLIT_SHA>"),
        md("## 4. S6 judge pass - consumes ONLY the consolidated manifest"),
        code("!python -m src.analysis.behavioral_judges "
             "--response-manifest results/manifests/consolidated_<ts>.json "
             "--require-binding --reject-legacy --out-dir results/behavioral_judges_v2 --run-live"),
        md("## 5. Post-run: bridge outputs, re-validate, session summary"),
        code("!python -m src.analysis.verify_activations\n!python -m src.analysis.v2_pipeline status"),
    ],
}


def main():
    for name, cells in NOTEBOOKS.items():
        path = NB_DIR / name
        path.write_text(json.dumps(notebook(cells), indent=1), encoding="utf-8")
        print(f"wrote {path}")


if __name__ == "__main__":
    main()
