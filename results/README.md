# results/ layout — 654 (frozen) vs 370era (pre-freeze)

The T4 rerun writes a **new, benchmark-bound** family of artifacts alongside the
historical 370-prompt ones. Nothing here is moved or deleted (`git mv` happens
only in the post-T4 restructure). This file is the map.

## 654 (frozen benchmark, `benchmark_sha256 e4946b07…`)

Every 654-era file carries `benchmark_sha256` + `split_manifest_sha256` per row and a
`<file>_binding.json` sidecar. `src/pipeline/binding_guard.py` refuses anything else on
the frozen-v2 path.

| path | producer | notes |
|---|---|---|
| `results/activations/{stage}_final.npy` / `_pooled.npy` / `_metadata.json` / `_metadata_binding.json` | `frozen_run_pipeline extract` | `_final` is canonical (analysis_plan.md §4). `src/analysis/verify_activations.py` CPU-verifies binding. |
| `results/behavioral/responses_654_{stage}.json` (+ `_binding.json`) | `frozen_run_pipeline behavior` | one row per (prompt, response); no judge scores yet |
| `results/raw/causal_ablation_654_{stage}_L24-28.json` (+ `_binding.json`) | `frozen_run_pipeline causal` | conditions `baseline` / `ablated_AD` / `ablated_random` [/ `ablated_AB`]; §6.1 seed/γ/RMS/cos in the sidecar |
| `results/raw/steering_654_{tag}.json` (+ `_binding.json`) | `frozen_run_pipeline steering` | `steered_learned` / `steered_random`; α_coef {0.5,1,2}; §6.2 provenance + degeneration rate |
| `results/refusal_direction/{stage}_direction_final.npy`, `per_prompt_projections.json` | `representation_projections.py` | per-prompt + M1-ref + M3-ref projections |
| `results/interpretability/subspace_geometry.json` | `subspace_geometry.py` | H1/H2: ρ_AD⊥, principal angles, PR/erank ([exec:T4]) |
| `results/refusal_direction/projection_trajectory.json` | `projection_trajectory.py` | §4.5 p_{q,s,l}, z_C, z_B ([exec:T4]) |
| `results/interpretability/direction_decodability_cf3.json` | `direction_decodability.py` | CF3 (secondary) ([exec:T4]) |
| `results/manifests/consolidated_<ts>.json` | `behavioral_judges.py --build-consolidated` | the ONLY input S6 consumes |
| `results/behavioral_judges_v2/behavioral_judges_v2_<ts>.json` | `behavioral_judges.py` | unified regex + StrongREJECT + WildGuard, one flat record per response |
| `results/behavioral_judges_v2/agreement_report.json` | `check_behavioral_agreement.py` | human-audit agreement + §5.4 aggregation |
| `logs/c_vs_a_leakage.json` | `check_c_vs_a_leakage.py` | C-vs-A / C-vs-training near-dup + CF3 category compat |

## legacy (370-prompt, pre-freeze) — kept for the paper trail, NOT current

`results/behavioral_eval/raw.json`, `results/behavioral_eval/summary_v2.json`,
`results/raw/causal_ablation_raw_{narrow,wide}.json`,
`results/raw/steering_raw_D*.json`, and the pre-freeze
`results/refusal_direction/*.json` / `results/probes/*.json` /
`results/interpretability/*.json`. These are 370-era. The CPU stats scripts
reject them by default; pass `--allow-unbound` only for deliberate historical
reproduction. `docs/audit/` records why each deprecated file is kept.

## post-T4

`docs/audit/analysis_plan.md` §9 POST-T4: geometry/decodability runs, CF1/CF2
CIs, matched-pair deltas, `_pooled` sensitivity, the human packet + agreement
re-tabulation, cross-branch on regenerated data, then the `git mv` restructure
(370-era material → a labelled appendix directory).

---

## Reading the filenames

Result filenames are **not renamed** even when the code that writes them is.
They are the experimental record, and `*_binding.json` sidecars, published
figures and provenance fields all reference them by name. So the historical
vocabulary persists here; `docs/NAMING.md` decodes it. The short version:

| In a filename | Means |
|---|---|
| `654` | the frozen 654-row benchmark era; the counterpart is `370era` |
| `_final` | activations/direction at the **final prompt token** |
| `_pooled` | **mean over the last five** non-padding tokens |
| `_L24-28` | the layer band an intervention was applied to |
| `_L0-28` | a direction spanning all 29 layers (a different axis from the band) |
| `_narrow` / `_wide` | older spelling for bands 24-28 and 14-28 |
| `_fullAD` | sensitivity run covering quadrants A and D in full |
| `_xfit5` | 5-fold cross-fitted, out-of-fold |
| `_QAD` / `_QABCD` | which quadrants a steering run covered |
| `_coef*` | steering coefficient |

## Reading the keys inside

Two keys are easy to misread, and one is an outright collision:

| Key | Means |
|---|---|
| `stage` | the **condition** name (`"M3_baseline"`), byte-identical to `condition` in every row |
| `condition` | same value as `stage` |
| `model_stage` | the **actual training stage** (`"M3"`) |
| `rate` | a **refusal** rate |
| `dtv` | delta total-variation distance against a reference label distribution |
| `raw_sims` | per-replicate bootstrap cosine similarities |
| `cv_accuracy_mean` | 5-fold probe accuracy; saturates near 1.0 — never select a layer by it |
| **`b`** | **bootstrap replicate count** in the bootstrap-CI files, but the **McNemar discordant-cell count** in `mcnemar_direction_specificity.json`. Same letter, sibling directories, unrelated quantities. |

## Legacy artifacts

`results/_legacy_370era/` holds the only tracked copy of the pre-freeze
artifacts, including two deprecated steering runs kept as the documented record
of why single-layer steering replaced multi-layer steering. Byte-identical
copies still sit at their original paths on disk and are gitignored, so each
file is committed exactly once.

`results/crossbranch/` is a separate activation-transfer study, outside the
frozen preregistration. Its numbers must not be folded into the confirmatory
endpoints. The two `.ipynb` files in there are executed run artifacts, not
source; the source lives in `notebooks/`.
