# REPRODUCE.md — pre-T4 / T4 / post-T4 command sequence (frozen)

Mirrors `docs/audit/analysis_plan.md` §9. The pre-T4 steps are CPU-only and run
now; the T4 steps need a GPU + local model weights; the post-T4 steps are CPU +
~½ day of human annotation.

## PRE-T4 (CPU / code / tests / docs — done in the WP1–WP20 pass)

```bash
# frozen plan + fixtures
cat docs/audit/analysis_plan.md docs/audit/endpoint_table.md
python -m tests.fixtures._generate          # (re)builds the binding-guard fixtures

# leakage + category compat (real, runs now)
python -m src.diagnostics.check_c_vs_a_leakage        # -> logs/c_vs_a_leakage.json

# targeted test gate (NOT the full suite)
python -m pytest \
  tests/test_v2_binding_guard.py tests/test_reproduce_causal_stats.py \
  tests/analysis/test_verify_activations.py tests/analysis/test_control_directions.py \
  tests/analysis/test_subspace_geometry.py tests/analysis/test_projection_trajectory.py \
  tests/analysis/test_direction_decodability.py tests/analysis/test_intervention_conditions.py \
  tests/analysis/test_behavioral_judges.py tests/analysis/test_build_human_review_packet.py \
  tests/analysis/test_check_behavioral_agreement.py tests/analysis/test_representation_projections.py \
  tests/test_eval_stats.py tests/interpretability/test_paired_deep_layer_stability_test.py \
  -q
python -m compileall -q src tests
```

`python -m src.reproduce --list` shows `causal_stats` as **BLOCKED** pre-T4 —
that is the intended state (it needs `results/raw/causal_ablation_v2_M3_L24-28.json`
from the T4 run).

## T4 (one notebook per session, 240–270 min target, hard 300, resumable)

Do **not** schedule a notebook merely because a point estimate is < 300 min —
use `v2_pipeline calibrate` / the 04b preflight cell. Notebooks 00–05 + 04b are
thin shells over `v2_pipeline` + the scripts below.

```bash
# S0 (nb 00): clone/pin, mount storage, env, benchmark + split verification, focused test gate
# 04b preflight: load StrongREJECT (dsbowen/strong_reject) + WildGuard (allenai/wildguard)
#   at 8-bit on toy pairs; print VRAM + versions; branch per analysis_plan.md §10 row 10.

# S1 (nb 01): calibrate + extract  (_final + _pooled + source_overt adjunct)
python -m src.analysis.build_c_source_overt_adjunct
python -m src.analysis.v2_pipeline extract --stages M0 M1 M2 M3 M3_direct M1_alt M2_alt M3_alt M3_direct_alt
python -m src.analysis.verify_activations            # CPU cross-check, all stages bound

# S2 (nb 02): behavioural generation -> per-session manifest
python -m src.analysis.v2_pipeline behavior --stages <...>

# S3 (nb 03): directions + probes + control_directions + projections -> per-session manifest
python -m src.analysis.v2_pipeline direction --stages <...>
python -m src.analysis.v2_pipeline probes    --stages <...>
python -m src.analysis.control_directions
python -m src.analysis.representation_projections
#   compute cos(d_AB, d_AD); decide ablated_AB by CALIBRATED session fit
#   (src/analysis/intervention_conditions.plan_causal_conditions)

# S4 (nb 04): causal  baseline / ablated_AD / ablated_random [/ ablated_AB]
python -m src.analysis.v2_pipeline causal --stage M3 \
  --conditions baseline ablated_AD ablated_random

# S5 (nb 05): steering  baseline / steered_learned / steered_random  x  alpha_coef {0.5,1,2}
python -m src.analysis.v2_pipeline steering --stage M3 --alpha-coefficients 0.5 1.0 2.0
#   if tight: cut M1/M2 dose-response cells FIRST (never the random control /
#   M3,M3_alt dose-response / the required A-D vs random contrast)

# build the consolidated response manifest (AFTER S2 + S4 + S5)
python -m src.analysis.behavioral_judges --response-manifest results/manifests/consolidated_<ts>.json \
  --build-consolidated results/manifests/<s2>.json results/manifests/<s4>.json results/manifests/<s5>.json \
  --benchmark-sha256 <BENCH_SHA> --split-manifest-sha256 <SPLIT_SHA>

# S6 (nb 05 tail): judge pass — consumes ONLY the consolidated manifest
python -m src.analysis.behavioral_judges \
  --response-manifest results/manifests/consolidated_<ts>.json \
  --require-binding --reject-legacy --out-dir results/behavioral_judges_v2 --run-live
```

`v2_pipeline direction` will **silently no-op** if the six
`results/refusal_direction/*.json` from the old 370-prompt run still exist —
pass `--force` after re-extraction (or delete the stale files first).

## POST-T4 (CPU + ~½ day human)

```bash
# geometry / decodability
python -m src.analysis.subspace_geometry
python -m src.analysis.projection_trajectory
python -m src.analysis.direction_decodability        # CF3 (secondary)

# CF1 + CF2 continuous endpoints from the judge output (frozen paired bootstrap)
python -m src.analysis.confirmatory_behavioral_endpoints \
  --judged results/behavioral_judges_v2/behavioral_judges_v2_<ts>.json \
  --benchmark data/frozen_v2/benchmark_v2_20260826T212909Z.jsonl \
  --out results/summaries/confirmatory_endpoints.json

# descriptive regex-category causal/steering summaries (complement CF2)
python -m src.analysis.summarize_causal_ablation --file results/raw/causal_ablation_v2_M3_L24-28.json
python -m src.analysis.mcnemar_causal_ablation  --file results/raw/causal_ablation_v2_M3_L24-28.json \
  --conditions M3_baseline M3_ablated_AD
python -m src.analysis.bootstrap_causal_effect  --file results/raw/causal_ablation_v2_M3_L24-28.json \
  --quadrant A --category refusal
for f in results/raw/steering_v2_*_QABCD.json; do python -m src.analysis.summarize_steering --file "$f"; done

# _pooled sensitivity (needs {stage}_final.npy AND {stage}_pooled.npy)
python -m src.analysis.representation_robustness

# matched C-pair deltas (SECONDARY - needs the source_overt adjunct extracted):
python -m src.analysis.build_c_source_overt_adjunct
python -m src.analysis.v2_pipeline extract --stage M3 \
  --latest-pointer data/frozen_v2/adjunct_c_source_overt.LATEST_BENCHMARK.json \
  --split-manifest data/frozen_v2/adjunct_c_source_overt.split_manifest.json \
  --namespace c_source_overt          # ~2 min GPU
python -m src.analysis.matched_pair_representation

# human audit
python -m src.analysis.build_human_review_packet --responses results/behavioral_judges_v2/<judge>.json \
  --judged results/behavioral_judges_v2/<judge>.json \
  --packet-out results/human_review/packet.json --key-out ../SEALED_KEY_outside_repo.json
#   ... annotate ...
python -m src.analysis.check_behavioral_agreement --sealed-key ../SEALED_KEY_outside_repo.json \
  --annotations ../annotations.json --judged results/behavioral_judges_v2/<judge>.json
python -m src.analysis.behavioral_robustness --conclusions results/human_review/conclusions.json

# cross-branch on regenerated data; rewrite Findings per §3 claim audit;
# repo restructure (git mv) — ONLY NOW.
```
