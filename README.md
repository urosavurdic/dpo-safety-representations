# DPO Safety Representations: A Mechanistic Study

## Overview

Does safety training (SFT, then DPO) give a language model genuinely richer
internal representations of harm, or does it mainly reshape how an existing
refusal-associated direction gets used? We train the same small model through
a four-stage chain — base → helpful-SFT → safety-SFT → DPO — and compare
internal representations and causal interventions at each stage.

This is an independent research project on a 1.5B-parameter model, not a
peer-reviewed publication. See **Limitations** before drawing broad conclusions.

**Status:** Phase 4 (interpretability) complete.

---

## The Four Quadrants

|                       | **Worded harmful-sounding**                          | **Worded neutrally**                                          |
|-----------------------|-------------------------------------------------------|-----------------------------------------------------------------|
| **Actually harmful**  | **A** — obviously harmful (HarmBench, n=50). Correct response: refuse. | **C** — harmful, disguised (hand-curated, n=20). Correct response: refuse. Sharpest test of real understanding. |
| **Actually benign**   | **B** — sounds risky, isn't (XSTest, n=250). Correct response: comply. Classic over-refusal trap. | **D** — benign, plainly worded (Alpaca, n=50). Correct response: comply. |

---

## Quick Start

```bash
python -m venv .venv
source .venv/bin/activate  # or .venv\Scripts\activate on Windows
pip install -r requirements.txt
pytest tests/ -v
```

### Reproduce the analysis pipeline

```bash
# GPU-dependent (Colab T4 used originally)
python -m src.analysis.eval_extract_activations
python -m src.analysis.eval_causal_ablation
python -m src.analysis.eval_steering --skip-baseline --layer 21

# CPU-only, local
python -m src.analysis.eval_behavioral
python -m src.analysis.eval_probes
python -m src.analysis.eval_refusal_direction
python -m src.analysis.summarize_causal_ablation --file results/raw/causal_ablation_raw_wide.json
python -m src.analysis.summarize_causal_ablation --file results/raw/causal_ablation_raw_narrow.json
python -m src.analysis.mcnemar_causal_ablation --file results/raw/causal_ablation_raw_wide.json
python -m src.analysis.mcnemar_causal_ablation --file results/raw/causal_ablation_raw_narrow.json
python -m src.analysis.summarize_steering --file results/raw/steering_raw_D_L21.json
python -m src.analysis.mcnemar_steering
python -m src.interpretability.direction_stability
python -m src.interpretability.lora_subspace_check
```

---

## Methodology

**Training chain** (Qwen2.5-1.5B): M0 (base) → M1 (SFT on Alpaca, no safety
content) → M2 (SFT on PKU-SafeRLHF *chosen* responses) → M3 (DPO on the
*same* PKU-SafeRLHF prompts as M2 — matched chosen/rejected pairs). M2/M3
sharing identical prompts and differing only in training objective isolates
"DPO the method" from "DPO the data." LoRA (r=64) throughout — see
Limitations for a quantified check of what this constrains.

**Refusal direction** — diff-in-means (not PCA): mean(activation | quadrant
A) − mean(activation | quadrant D), per layer per stage, unit-normalized.
This is the same core method as Arditi et al. (2024, NeurIPS), applied here
across a controlled training-stage chain rather than across many pretrained
models.

**Causal ablation** — project the direction out of the residual stream at
generation time, at a chosen layer range, on M3. Paired McNemar's exact test
on discordant baseline-vs-ablated pairs.

**Steering** — the reverse intervention: add the direction into the residual
stream, magnitude anchored to that layer's real mean quadrant-A projection.

**LoRA-subspace check** — for each checked layer, what fraction of the
refusal direction's norm lies in the column space of that layer's LoRA `B`
matrix (`o_proj`/`down_proj` only — the two module types that write directly
into the residual stream), compared against 200 random unit vectors as a
null baseline.

---

## Key Results

| Experiment | Quadrant A | Quadrant B | Quadrant C |
|---|---|---|---|
| Baseline (M3) | 14% refusal | 5.6% refusal+soft-defl. | 80% soft-deflection |
| Wide ablation (14–28) | 0% (p=0.0156) | 0% | 0% (p=0.00003) |
| Narrow ablation (24–28) | 0% (p=0.0156) | 1.2% (79% relative drop) | 25% (p=0.0010, 69% relative drop) |

**Steering (quadrant D, benign):** 15-layer addition → 98% degenerate output
(not refusal). Single-layer (21) addition → small, non-significant shift
(McNemar p=0.50, n=2 discordant pairs). Neither is a clean causal complement
to ablation — reported as a genuine null result, not a positive finding.

**LoRA-subspace:** direction's norm captured by the rank-64 subspace ranges
6–10% across checked layers/modules — 90%+ lies outside it everywhere. The
overlap that does exist is real (3–10 standard deviations above a
random-direction baseline, not noise) and concentrated at the deep layers
(21, 28) and `down_proj` specifically — matching where the direction-rotation
analysis independently finds DPO's action concentrates.

---

## Key Findings

**1. The causal effect is real and statistically significant.** Wide
ablation: 16/16 quadrant-C pairs flip away from soft-deflection (p=0.000031),
7/7 quadrant-A pairs flip away from refusal (p=0.015625). 100%-directional,
no reverse flips, confirmed by paired exact tests, not just non-overlapping CIs.

**2. Layer-range dependence is real but not quadrant-selective between B and
C.** Quadrant A's suppression is *fully* explained by layers 24–28 alone
(100% relative reduction under the narrow range). Quadrants B and C are only
*partially* explained by that same narrow range (79% and 69% relative
reduction respectively) — layers 14–23 carry real signal for both, at similar
magnitude. This revises "non-selective" to something more precise: A has a
narrowly-concentrated dependency the other two don't share, but B and C don't
separate cleanly from each other at this resolution.

**3. The direction rotates most during generic instruction-tuning, not
safety training.** Mean drift (1 − cosine similarity) across 28 non-artifact
layers: M0→M1 ≈ 0.335, M1→M2 ≈ 0.040, M2→M3 ≈ 0.070. DPO adds ~1.75× the
rotation SFT-safety did, concentrated in the deepest third of the network —
but M0→M1 remains far larger (4.8× M2→M3, 8.4× M1→M2). Layers 1–5 stay highly
similar to M0 even at M3 (0.73–0.90); the single lowest point is layer 20
(0.380). *(Layer 0 excluded — zero-vector template-token artifact.)*

**4. The direction is not primarily a LoRA artifact, but isn't fully
independent of it either.** 90%+ of the direction's norm lies outside the
rank-64 LoRA subspace everywhere checked — ruling out "this is just what
LoRA happened to be capable of writing" as the main explanation. But the
alignment that does exist is statistically real (not random-chance overlap)
and concentrated exactly where the direction rotates most during DPO (deep
layers, `down_proj`) — a genuine, if secondary, point of convergence between
two independent analysis methods.

**5. Steering is an honest null result, not a confirmed causal complement.**
Multi-layer addition collapses output to near-total degenerate text (98%),
most likely from compounding across 15 residual-stream injections rather than
a targeted behavioral shift. Single-layer addition at the deepest
signal-bearing layer produces a small, statistically non-significant change
(p=0.50). Ablation's causal story stands on its own; steering did not add
independent confirmation, and further layer/magnitude tuning was
deliberately not pursued past this point.

**6. Overall verdict.** We find no evidence that DPO builds a new,
safety-specific representation from scratch. The refusal-associated direction
is already present after generic instruction-tuning alone (M1 representationally
flags 85% of quadrant C despite 0% behavioral soft-deflection at that stage —
representation precedes behavior). DPO primarily strengthens the coupling
between that pre-existing representation and output behavior, with real but
smaller additional rotation concentrated in deeper layers, and only a modest,
not-dominant relationship to the LoRA subspace it was trained through. The
causal effect of ablating this direction is real and significant for both
overtly harmful and disguised-harm prompts, but the layer ranges differ: the
deepest 5 layers fully account for the overt-refusal effect, while a broader
range is needed to fully account for the disguised-harm and over-refusal
effects — which don't separate from each other at this resolution. Closer to
"coupling/amplification" (Hypothesis B) than "genuinely new representation"
(Hypothesis A).

---

## Project Structure

```
dpo-safety-representations/
├── .gitignore
├── HANDOFF.md
├── PROJECT_CONTEXT.md
├── README.md
├── pyproject.toml
├── requirements.txt
│
├── configs/
│   ├── m1_gpu_dryrun.yaml
│   ├── m1_sft_helpful.yaml
│   ├── m1_smoke_test.yaml
│   ├── m2_gpu_dryrun.yaml
│   ├── m2_sft_safety.yaml
│   ├── m3_dpo.yaml
│   └── m3_gpu_dryrun.yaml
│
├── data/
│   ├── dedup_report.json
│   ├── dedup_report_m1.json
│   └── processed/
│       ├── alpaca_reserved_for_eval.json
│       ├── controlled_eval.jsonl
│       ├── dpo_pairs.jsonl
│       ├── m1_near_dup_exclusions.json
│       ├── sft_helpful.jsonl
│       └── sft_safety.jsonl
│
├── outputs/
│   └── smoke_test_m1/
│       ├── checkpoints/
│       │   ├── README.md
│       │   └── checkpoint-2/
│       │       ├── README.md
│       │       ├── adapter_config.json
│       │       ├── chat_template.jinja
│       │       ├── optimizer.pt
│       │       ├── rng_state.pth
│       │       ├── scheduler.pt
│       │       ├── tokenizer.json
│       │       ├── tokenizer_config.json
│       │       ├── trainer_state.json
│       │       └── training_args.bin
│       │
│       └── final/
│           ├── README.md
│           ├── adapter_config.json
│           ├── chat_template.jinja
│           ├── config_used.yaml
│           ├── git_commit.txt
│           ├── requirements.txt
│           ├── tokenizer.json
│           ├── tokenizer_config.json
│           └── training_args.bin
│
├── results/
│   ├── activations/
│   │   ├── M0_metadata.json
│   │   ├── M1_metadata.json
│   │   ├── M2_metadata.json
│   │   └── M3_metadata.json
│   │
│   ├── behavioral_eval_capability.json
│   ├── behavioral_eval_raw.json
│   ├── behavioral_eval_summary_v2.json
│   ├── causal_ablation_raw.json
│   ├── causal_ablation_raw_narrow.json
│   ├── causal_ablation_summary.json
│   ├── classifier_validation_sample.json
│   ├── qualitative_spot_check.json
│   │
│   ├── probes/
│   │   ├── M0_probe_results.json
│   │   ├── M1_probe_results.json
│   │   ├── M2_probe_results.json
│   │   └── M3_probe_results.json
│   │
│   ├── refusal_direction/
│   │   ├── M0_direction.npy
│   │   ├── M1_direction.npy
│   │   ├── M2_direction.npy
│   │   ├── M3_direction.npy
│   │   ├── cosine_similarity.json
│   │   └── quadrant_projections.json
│   │
│   └── summaries/
│       └── causal_ablation_raw_narrow_summary.json
│
├── src/
│   ├── __init__.py
│   ├── io_utils.py
│   │
│   ├── analysis/
│   │   ├── __init__.py
│   │   ├── eval_behavioral.py
│   │   ├── eval_causal_ablation.py
│   │   ├── eval_probes.py
│   │   ├── eval_refusal_direction.py
│   │   ├── mcnemar_causal_ablation.py
│   │   ├── reclassify_behavioral.py
│   │   ├── summarize_causal_ablation.py
│   │   ├── summarize_probe_findings.py
│   │   └── summarize_refusal_direction.py
│   │
│   ├── core/
│   │   ├── __init__.py
│   │   ├── build_eval_set.py
│   │   ├── build_m1_data.py
│   │   ├── data_prep.py
│   │   ├── eval_generation.py
│   │   ├── train_dpo.py
│   │   └── train_sft.py
│   │
│   ├── diagnostics/
│   │   ├── __init__.py
│   │   ├── analyze_data_coverage.py
│   │   ├── check_classifier_agreement.py
│   │   ├── check_leakage.py
│   │   ├── diagnose_probe_layers.py
│   │   ├── eval_extract_activations.py
│   │   ├── eval_qualitative.py
│   │   ├── eval_refusal_classifier.py
│   │   ├── inspect_quadrant_c.py
│   │   ├── search_alpaca_leakage.py
│   │   ├── search_source_data.py
│   │   ├── validate_refusal_classifier.py
│   │   ├── verify_activations.py
│   │   └── verify_cross_stage_diff.py
│   │
│   ├── interpretability/
│   │   ├── __init__.py
│   │   ├── alpha_scaling.py
│   │   ├── direction_stability.py
│   │   ├── integrated_report.py
│   │   ├── per_layer_analysis.py
│   │   └── utils.py
│   │
│   ├── training/
│   │   ├── __init__.py
│   │   ├── callbacks.py
│   │   ├── data.py
│   │   ├── dpo_data.py
│   │   ├── formatting.py
│   │   ├── model.py
│   │   └── utils.py
│   │
│   └── utils/
│       ├── __init__.py
│       └── eval_stats.py
│
└── tests/
    ├── test_environment.py
    ├── test_eval_stats.py
    │
    ├── analysis/
    │   ├── __init__.py
    │   ├── test_eval_causal_ablation.py
    │   ├── test_eval_probes.py
    │   ├── test_eval_refusal_direction.py
    │   ├── test_mcnemar_causal_ablation.py
    │   ├── test_summarize_causal_ablation.py
    │   └── test_summarize_probe_findings.py
    │
    ├── core/
    │   ├── __init__.py
    │   ├── test_build_eval_set.py
    │   ├── test_build_m1_data.py
    │   ├── test_data_prep.py
    │   ├── test_eval_generation.py
    │   ├── test_train_dpo.py
    │   └── test_train_sft.py
    │
    ├── diagnostics/
    │   ├── __init__.py
    │   ├── test_check_leakage.py
    │   ├── test_eval_extract_activations.py
    │   └── test_eval_refusal_classifier.py
    │
    └── training/
        ├── __init__.py
        ├── test_callbacks.py
        ├── test_dpo_data.py
        ├── test_model.py
        ├── test_training_data.py
        ├── test_training_formatting.py
        └── test_training_utils.py
```


---

## Limitations

1. **LoRA, quantified, not fully resolved.** 90%+ of the refusal direction's
   norm lies outside the rank-64 LoRA subspace at every checked layer —
   the direction is not primarily a LoRA artifact. But real, above-chance
   alignment (3–10σ over a random-direction baseline) exists at deep layers,
   so a full-fine-tuning robustness check would still add confidence, not
   just close a theoretical gap.
2. **Single diff-in-means direction.** Other orthogonal safety-relevant
   directions may exist; not searched for.
3. **Ablation shows sufficiency, not necessity.** Steering (the natural test
   of the complementary direction) was inconclusive (Finding 5), so this
   isn't independently confirmed from the addition side.
4. **Small sample for the sharpest test.** Quadrant C, n=20.
5. **1.5B scale.** Not claimed to generalize to frontier-scale models without
   further work.
6. **M1's Alpaca data may itself skew "safe,"** independent of generic
   instruction-following per se — the M0→M1 jump (Finding 3) could be partly
   a data-content effect. Not disentangled here.
7. **Why multi-layer steering collapses to degenerate output isn't fully
   diagnosed** — hypothesized as compounding across the residual stream, not
   independently confirmed (e.g., by tracking activation norm growth
   layer-by-layer during generation).

---

## Future Work

- Full fine-tuning robustness check (removes the LoRA-rank confound).
- Diagnose the steering degenerate-collapse mechanism directly (track
  residual-stream norm growth across layers under multi-layer addition) —
  would resolve Limitation 7 and could unlock a working steering complement.
- Train on a second, independent safety dataset to check generalization
  across data sources, not just training stages.
- DPO applied directly to M1 (skipping M2) — isolates the SFT-safety step's
  independent contribution.
- Retrain M1 on a harm-balanced instruction set to address Limitation 6.

---

## Repository Hygiene

- Smoke-test binaries removed from tracking; `.gitignore` updated.
- `src/`/`tests/` reorganized by function, `tests/` mirrors `src/` exactly.
- `results/` split into `raw/`/`summaries/`, consistently named.
- Run `pytest tests/ -v` — all tests pass.

---

## How to Cite

```bibtex
@misc{dpo_safety_representations,
  title={DPO Safety Representations: A Mechanistic Study},
  author={[Your Name]},
  year={2026},
  howpublished={\url{https://github.com/urosavurdic/dpo-safety-representations}},
  note={Independent research project, not peer-reviewed}
}
```