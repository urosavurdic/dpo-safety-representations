# EXPERIMENT.md — frozen pre-T4 state

Portfolio-facing summary of *what is fixed* going into the T4 run. The
authoritative pre-registration is `docs/audit/analysis_plan.md`; this file is
the readable orientation.

## Frozen benchmark

| | |
|---|---|
| file | `data/frozen_v2/benchmark_v2_20260826T212909Z.jsonl` |
| sha256 | `e4946b070f441c7a0676db830c65257b78a2d1b46abb0a61cce4cc86352f838b` |
| rows | **654** — A=150 (HarmBench), B=250 (XSTest), C=104 (StrongREJECT reduced-cue), D=150 (Alpaca/Dolly/OASST1) |
| split | `direction_estimation` (240) / `held_out_behavioral` (60) assigned once, upstream, on A/D only (`logs/direction_split_manifest.json`, sha `880381606de7…`) |

Every v2 artifact carries `benchmark_sha256` + `split_manifest_sha256` per row
and a `*_binding.json` sidecar. The CPU stats scripts refuse any 370-era /
unbound file (`src/v2_binding_guard.py`); `--allow-unbound` is a break-glass for
historical work only.

## Prompt annotation vs response annotation (frozen clarification — §5.0)

The **104/104 `accept`** in `data/review/r104_human_review_blind.csv` is
**prompt-rewrite eligibility review** — whether each reduced-cue rewrite
preserved the harmful objective. It is **not** a human evaluation of any model
response. The project has **zero response-level human behavioral annotation** on
either the 370- or 654-row benchmark; `results/classifier_validation_sample.json`
does not exist; the regex classifier's patterns are hand-authored constants,
anecdotally calibrated during development. The §5.2 human audit (160 responses =
120 coverage + 40 targeted, 40 re-annotated) is the **first** response-level
human labeling and is deliberately scoped as a blinded audit + instrument
validation, **not a gold-standard truth set**.

## Endpoints (see `docs/audit/endpoint_table.md`)

- **CF1** (confirmatory): C M2→M3 behavioural transition, `Δ_C = mean_i(SR_i^{M3} −
  SR_i^{M2})`, prompt-level paired bootstrap CI. Named the *M2→M3 stage
  transition*, **never** "the causal effect of DPO".
- **CF2** (confirmatory): direction-specific causal effect on M3 held-out A
  (~30), `CF2 = E_AD − E_random` (ablate the learned A–D direction vs a
  calibration-RMS-matched random ablation). The n≈30 limitation is reported
  explicitly; full-A is a *predeclared* sensitivity analysis, never a post-hoc
  replacement.
- **CF3** (predeclared secondary, NOT confirmatory): after residualizing out
  each stage's own A–D direction at layer 28, is the 4-way benchmark-category
  distinction more linearly decodable at M3 than M2? Wording: *"DPO made the
  preregistered benchmark-category distinction more linearly decodable after the
  A–D contrast direction was removed"* — **never** *"DPO created a richer safety
  representation."*

## Terminology guardrails (§3)

The PCA/SVD object is the **top-variance subspace of the centered, equal-weighted
A_est ∪ D_est activations** — call it the *stage-specific centered A/D activation
subspace*. **Never** "the safety subspace", "the full safety representation",
"the model's complete refusal mechanism", "semantic safety understanding",
"invariant across models/datasets". A stable cosine alone never proves H2; report
all four outcomes (one-vector preservation / subspace preservation / added
orthogonal structure / amplification).

## Branch and corpus wording

- **M1 → M2 → M3** vs **M1 → M3_direct** answers *"does inserting a safety-SFT
  stage before DPO change the resulting geometry/behaviour?"* — **not** "DPO the
  method".
- **M1_alt** is *"a second instruction corpus"* (Dolly-15k), **not** "only
  dataset identity". A corpus-stats table (row counts, length distributions,
  sampling rate) accompanies any alt-branch claim. M1_alt draws ~40% of Dolly's
  single-turn pool, so training/eval overlap is structural — the eval set avoids
  reusing exact prompts going forward, but already-trained M*_alt checkpoints
  saw some.

## LoRA claim scope

The "not a LoRA artifact" claim is **dropped**. The LoRA-subspace check bounds
the confound only for the inspected rank-64 `o_proj` / `down_proj` subspaces at
layers 7/14/21/28.

## C-F / construct-validity analyses

`c_b_paired_delta_analysis.py`, `cf_joint_geometry.py`, `c_c_construction_audit.py`
and the lexical-cue tooling are **input-text / construct-validity** analyses, not
internal-representation results. Label them as such.

## Leakage status (WP-Leak, CPU, run 2026-09-02)

`python -m src.diagnostics.check_c_vs_a_leakage` against the frozen benchmark:

- **C vs A**: 0 exact, 0 near-duplicate (sentence-transformers all-MiniLM-L6-v2,
  cos ≥ 0.9).
- **C vs training** (`sft_helpful`, `sft_helpful_alt`, `sft_safety`,
  `dpo_pairs`): 0 exact, 0 near-duplicate in all four.
- **CF3 category compat**: A and C carry the **same** 4-way label set
  (`misinformation_disinformation`, `harassment_bullying`, `illegal`,
  `cybercrime_intrusion`); A has no non-CF3 categories. Distributions differ
  (A illegal-heavy 57 vs C 6; C harassment-heavy 41 vs A 19) — handled by the
  grouped-stratified CV in `direction_decodability.py`; only the label *sets*
  need to match (§4.4).

Full report: `logs/c_vs_a_leakage.json`.

## What favorable results still CANNOT support (§3)

DPO-as-method causing the effect independent of data/hyperparameters/epochs/LR;
a complete or general safety representation; the category probe measuring
semantic safety understanding; generalization across model families / scales /
seeds / datasets; A–D being purely safety-specific; the direction not being a
LoRA artifact; global necessity/sufficiency of the direction; a single annotator
as gold truth; WildGuard or StrongREJECT as ground truth; A–B causal specificity
if its ablation was omitted for wall-time; a non-significant n≈30 CF2 proving
absence of an effect.
