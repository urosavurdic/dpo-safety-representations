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

Prompts vary along two independent axes: whether the request is actually
harmful, and whether it's *worded* that way. The off-diagonal cases are where
surface pattern-matching and real understanding of harm come apart.

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

Activations, probes, and refusal-direction extraction require the trained
adapters (Hugging Face Hub — see Methodology) and, for activation extraction,
a GPU (Colab T4 used originally). Everything downstream is CPU-only.

```bash
# GPU-dependent (Colab)
python -m src.analysis.eval_extract_activations
python -m src.analysis.eval_causal_ablation

# CPU-only, local
python -m src.analysis.eval_behavioral
python -m src.analysis.eval_probes
python -m src.analysis.eval_refusal_direction
python -m src.analysis.summarize_causal_ablation --file results/raw/causal_ablation_raw_wide.json
python -m src.analysis.summarize_causal_ablation --file results/raw/causal_ablation_raw_narrow.json
python -m src.analysis.mcnemar_causal_ablation --file results/raw/causal_ablation_raw_wide.json
python -m src.analysis.mcnemar_causal_ablation --file results/raw/causal_ablation_raw_narrow.json
python -m src.interpretability.direction_stability
```

---

## Methodology

**Training chain** (Qwen2.5-1.5B):
- **M0** — base checkpoint, no training.
- **M1** — SFT on Alpaca (general instruction-following, deliberately no
  safety-specific content). Isolates "learned to be a chat assistant" from
  "learned about safety."
- **M2** — SFT on PKU-SafeRLHF *chosen* responses.
- **M3** — DPO on the *same* PKU-SafeRLHF prompts as M2 (matched chosen/rejected
  pairs). M2 and M3 seeing identical prompts/content and differing only in
  training objective isolates "DPO the method" from "DPO the data."
- LoRA (r=64 for M2→M3) throughout, not full fine-tuning — see Limitations.

**Activation extraction** — hidden states at every layer, two pooled
positions (final token; mean of last 5 tokens), across all 370 controlled-eval
prompts × 4 stages.

**Behavioral evaluation** — rule-based classifier (degenerate / hard refusal /
soft-deflection / comply), human-validated for agreement, Wilson confidence
intervals throughout.

**Linear probes** — logistic regression per layer. Naive CV accuracy saturates
near 1.0 at nearly every layer, including untrained M0 — a dataset/style
fingerprint confound, not a real safety signal. Retired as the headline metric;
real signal is the *held-out flagging rate* (fraction of a held-out quadrant a
trained A-vs-B boundary calls "unsafe").

**Refusal direction** — diff-in-means (not PCA): mean(activation | quadrant A)
− mean(activation | quadrant D), per layer per stage, unit-normalized. Used for
(1) cross-stage cosine similarity — does the direction stay put or rotate? —
and (2) causal ablation.

**Causal ablation** — project the diff-in-means direction out of the residual
stream at generation time, at a chosen layer range, on M3. Paired McNemar's
exact test on discordant baseline-vs-ablated pairs.

---

## Key Results

| Experiment | Quadrant C (soft-deflection) | Quadrant A (hard refusal) |
|---|---|---|
| **Baseline (M3, no ablation)** | 80% (16/20) | 14% (7/50) |
| **Wide ablation (layers 14–28)** | 0% (0/20), p=0.000031 | 0% (0/50), p=0.015625 |
| **Narrow ablation (layers 24–28)** | 25% (5/20), p=0.000977 | 0% (0/50), p=0.015625 |

---

## Key Findings

**1. The causal effect is real and statistically significant.** Both wide and
narrow ablation produce 100%-directional flips (no reverse flips) on quadrant
C, confirmed by paired McNemar's exact tests, not just non-overlapping CIs.

**2. Not selective between legitimate refusal and disguised-harm detection —
but layer range matters for one of them.** Narrowing from 15 layers to the
deepest 5 leaves quadrant A's suppression complete (7/7 flips either way) but
only partially reduces quadrant C's (16/20 baseline → 5/20 ablated, vs. 0/20
under the wide range). Layers 14–23 carry real C-specific signal that layers
24–28 alone don't capture; A's dependence is concentrated in the deepest 5
layers.
*(TODO before this section is final: quadrant B's soft-deflection rate under
the narrow ablation specifically — it was 4.8%→0% under the wide ablation but
hasn't been checked for the narrow run. Run `summarize_causal_ablation.py
--file results/raw/causal_ablation_raw_narrow.json` and fill in.)*

**3. The refusal-associated direction rotates most during generic
instruction-tuning, not safety training — confirmed across all 28
non-artifact layers, not just a sample.** Mean drift (1 − cosine similarity)
per adjacent-stage transition, layers 1–28: M0→M1 ≈ 0.335, M1→M2 ≈ 0.040,
M2→M3 ≈ 0.070. DPO adds roughly 1.75× the rotation SFT-safety did,
concentrated in the deepest third of the network — but M0→M1 remains far
larger than both (4.8× M2→M3, 8.4× M1→M2). Stability generally decreases
with depth: layers 1–5 stay highly similar to M0 even at M3 (0.73–0.90),
while the single lowest point is layer 20 (0.380). Mean M0-vs-M3 similarity
across layers 1–28: 0.582. *(Layer 0 excluded throughout — its direction is
the zero vector, the known template-token artifact from Component 4, not
real instability.)*

**4. Overall verdict.** Post-training doesn't appear to build a new,
DPO-specific safety module from scratch — sensitivity to disguised harm
(quadrant C) is already representationally present after generic
instruction-tuning (M1 flags 85% of quadrant C despite 0% behavioral
soft-deflection at that stage). DPO measurably reshapes — not just amplifies —
the direction more than safety-SFT does, and changes how strongly that
representation converts into behavior. Closer to "amplification/coupling"
than "genuinely new representation," with real nuance rather than a clean
binary answer.

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

1. **LoRA, not full fine-tuning.** LoRA constrains updates to a low-rank
   subspace by construction, which can mechanically bias findings toward
   "amplification looks low-dimensional." Not fully resolved — stated
   explicitly rather than hidden.
2. **Single diff-in-means direction.** Other orthogonal safety-relevant
   directions may exist; not searched for.
3. **Inference-time ablation only.** Unknown how this interacts with training
   or generalizes beyond the controlled eval set.
4. **Small samples for the sharpest test.** Quadrant C, n=20; CIs are wide but
   non-overlapping across stages (see behavioral eval).
5. **1.5B scale.** Normal and expected for this kind of study, but findings
   are not claimed to generalize to frontier-scale models without further work.
6. **M1's Alpaca data may itself skew toward "safe" content**, independent of
   generic instruction-following per se — meaning the M0→M1 representational
   jump (Finding 3) could be partly a data-content effect, not purely an
   instruction-tuning effect. Not disentangled here; would need an M1 retrained
   on a harm-balanced instruction corpus.
7. **Causal ablation shows sufficiency, not necessity.** The direction is
   causally sufficient to modulate behavior; other pathways may exist that
   the ablation doesn't touch.

---

## Future Work

- Full fine-tuning robustness check (removes the LoRA-rank confound).
- Steering (add the direction, rather than ablate it) — cheap, reuses the
  existing ablation infrastructure with the sign/magnitude flipped.
- Train on a second, independent safety dataset to check whether the direction
  and findings generalize across data sources, not just across training stages.
- DPO applied directly to M1 (skipping M2) — isolates whether the SFT-safety
  step matters independently of DPO.
- Retrain M1 on a harm-balanced instruction set to address Limitation 6.

---

## Repository Hygiene

- Smoke-test binaries (`outputs/smoke_test_m1/`) removed from tracking (`git rm -r --cached`, `.gitignore` updated).
- `src/` and `tests/` reorganized by function; `tests/` mirrors `src/` exactly.
- `results/` split into `raw/` (per-run outputs) and `summaries/` (aggregated stats), consistently.
- Run `pytest tests/ -v` — all tests pass, including new coverage for `direction_stability.py`.

---

## Questions

- **What's the headline finding?** See "Overall verdict" above.
- **How do I reproduce this?** "Quick Start" above; GPU steps need a Colab
  T4 or equivalent, everything else is CPU-only.
- **Where's the raw data?** `results/` — see Project Structure.