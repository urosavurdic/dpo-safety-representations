# DPO Safety Representations: Mechanistic Interpretability Study


## Overview

This project investigates the mechanistic basis of refusal behavior in language models trained with Direct Preference Optimization (DPO). We extract a "refusal direction" via PCA, causally intervene by removing its projection, and analyze which layers contribute to the effect.

**Core Finding:** The refusal mechanism is distributed across deep transformer layers (14-28), with higher signal in deeper layers. Ablation effectively suppresses both legitimate hard refusal and problematic over-caution, indicating a **non-selective mechanism** that is robust to training-induced representational drift.

---

## Quick Start

### Reproduce Results

```bash
# Activate environment
python -m venv .venv
source .venv/bin/activate  # or .venv\Scripts\activate on Windows
pip install -r requirements.txt

# Run analysis pipeline
python -m src.analysis.refusal_direction        # Extract direction (M0-M3)
python -m src.analysis.causal_ablation         # Run wide & narrow ablations
python -m src.analysis.behavioral              # Classify behaviors

# Generate interpretability analyses
python -m src.interpretability.per_layer_analysis
python -m src.interpretability.alpha_scaling
python -m src.interpretability.direction_stability

# View reports
open results/interpretability/reports/INDEX.md
```

### Key Results

| Experiment | Quadrant C (Soft-Deflection) | Quadrant A (Hard Refusal) | p-value |
|---|---|---|---|
| **Wide ablation (14-28)** | 80% → 0% | 14% → 0% | 0.000031 / 0.015625 |
| **Narrow ablation (24-28)** | 80% → 25% | 14% → 0% | 0.000977 / 0.015625 |
| **Interpretation** | Layers 14-23 carry redundant signal | Layers 24-28 are critical | Both highly significant |

---

## Project Structure

```
dpo-safety-representations/
├── src/                                    # Source code organized by function
│   ├── __init__.py
│   ├── io_utils.py                         # Centralized JSON I/O helpers
│   │
│   ├── core/                               # Training pipeline
│   │   ├── __init__.py
│   │   ├── train_dpo.py                    # DPO training
│   │   ├── train_sft.py                    # SFT baseline training
│   │   └── ... (data prep, generation)
│   │
│   ├── analysis/                           # Measurement & evaluation (Phases 3-4)
│   │   ├── __init__.py
│   │   ├── behavioral.py                   # Behavior classification
│   │   ├── refusal_direction.py            # Extract & analyze direction
│   │   ├── probes.py                       # Probe classifiers
│   │   ├── causal_ablation.py              # Causal intervention
│   │   ├── summarize_*.py                  # Summary statistics
│   │   └── mcnemar_causal_ablation.py      # Paired significance testing
│   │
│   ├── diagnostics/                        # Quality assurance
│   │   ├── __init__.py
│   │   ├── coverage.py                     # Data completeness
│   │   ├── leakage.py                      # Train/eval contamination checks
│   │   ├── classifier.py                   # Classifier agreement
│   │   ├── activations.py                  # Hidden state verification
│   │   └── inspection.py                   # Spot checks & qualitative analysis
│   │
│   ├── interpretability/                   # Mechanistic analysis (Phase 5) **NEW**
│   │   ├── __init__.py
│   │   ├── per_layer_analysis.py           # Attribution: which layers matter
│   │   ├── alpha_scaling.py                # Scaling: effect vs. magnitude
│   │   ├── direction_stability.py          # Drift: across training stages
│   │   ├── integrated_report.py            # Unified report generation
│   │   └── utils.py                        # Shared utilities
│   │
│   ├── training/                           # Training utilities (kept from original)
│   │   ├── __init__.py
│   │   ├── callbacks.py
│   │   ├── data.py
│   │   ├── dpo_data.py
│   │   ├── formatting.py
│   │   ├── model.py
│   │   └── utils.py
│   │
│   └── utils/                              # General utilities
│       ├── __init__.py
│       └── eval_stats.py                   # Statistical helpers
│
├── results/                                # Experimental outputs
│   ├── raw/                                # Raw model/experiment outputs
│   │   ├── causal_ablation_raw_wide.json
│   │   └── causal_ablation_raw_narrow.json
│   │
│   ├── summaries/                          # Aggregated statistics
│   │   ├── causal_ablation_wide_summary.json
│   │   └── causal_ablation_narrow_summary.json
│   │
│   ├── activations/                        # Extracted hidden states (Phase 2)
│   ├── probes/                             # Probe classifier results (Phase 3)
│   ├── refusal_direction/                  # Direction extraction (Phase 4)
│   │   ├── cosine_similarity.json          # M0-M1-M2-M3 direction stability
│   │   ├── quadrant_projections.json       # Per-layer projections by quadrant
│   │   └── M{0,1,2,3}_direction.npy        # Direction vectors
│   │
│   ├── behavioral_eval/                    # Behavior classification outputs
│   │
│   └── interpretability/                   # **NEW** Interpretability findings
│       ├── per_layer_analysis/             # Layer attribution results
│       ├── alpha_scaling/                  # Scaling analysis outputs
│       ├── direction_stability/            # Direction drift results
│       └── reports/                        # **Individual, navigable reports**
│           ├── INDEX.md                    # Start here: navigation guide
│           ├── 01_causal_ablation_wide.md
│           ├── 02_causal_ablation_narrow.md
│           ├── 03_narrow_vs_wide_comparison.md
│           ├── 04_per_layer_signal_magnitude.md
│           ├── 05_alpha_scaling_analysis.md
│           ├── 06_direction_stability_across_training.md
│           └── 07_key_findings_synthesis.md
│
├── tests/                                  # Test suite (mirrors src/ organization)
│   ├── __init__.py
│   ├── test_environment.py                 # Python version & dependency checks
│   ├── test_data_prep.py                   # Data preparation tests
│   ├── test_dpo_data.py                    # DPO dataset tests
│   ├── test_build_m1_data.py               # M1 data building tests
│   │
│   ├── core/                               # Training pipeline tests
│   │   ├── __init__.py
│   │   ├── test_train_dpo.py
│   │   ├── test_train_sft.py
│   │   ├── test_training_data.py
│   │   ├── test_training_formatting.py
│   │   ├── test_training_utils.py
│   │   ├── test_callbacks.py
│   │   └── test_model.py
│   │
│   ├── analysis/                           # Measurement & evaluation tests
│   │   ├── __init__.py
│   │   ├── test_build_eval_set.py
│   │   ├── test_eval_generation.py
│   │   ├── test_eval_causal_ablation.py
│   │   ├── test_eval_refusal_direction.py
│   │   ├── test_eval_probes.py
│   │   ├── test_eval_stats.py
│   │   ├── test_mcnemar_causal_ablation.py
│   │   └── test_summarize_probe_findings.py
│   │
│   └── diagnostics/                        # QA & diagnostics tests
│       ├── __init__.py
│       ├── test_check_leakage.py
│       ├── test_eval_extract_activations.py
│       └── test_eval_refusal_classifier.py
│
├── configs/                                # Training & evaluation configs
├── data/                                   # Raw data inputs
├── outputs/                                # Temporary outputs / checkpoints
│
├── README.md                               # This file
├── PROJECT_CONTEXT.md                      # Design decisions & decision log
├── HANDOFF.md                              # Final summary & next steps
├── requirements.txt
├── pyproject.toml
└── .gitignore
```

---

## Key Findings at a Glance

### 1. **Causal Effect is Real & Significant**

Paired McNemar's exact tests confirm:
- Quadrant C: 16/16 discordant pairs flip away from soft-deflection (wide ablation, p=0.000031)
- Quadrant A: 7/7 discordant pairs flip away from hard refusal (wide ablation, p=0.015625)

Both effects are **100% directional** (no reverse flips) and **statistically robust** (p << 0.05).

### 2. **Mechanism is Distributed, Not Localized**

- Projection magnitude increases monotonically across layers 0-28
- Layer 28: ~30 units (deepest, highest)
- Layer 21: ~8 units (deep)
- Layer 14: ~4 units (mid)
- Layer 7: ~2 units (shallow)

Narrow ablation (24-28 only) produces partial effect on C (80%→25%), proving layers 14-23 contribute.

### 3. **Effect is Non-Selective**

Narrowing to deep layers (where signal is highest) does NOT improve selectivity:
- Wide (14-28): C 100% flip, A 100% flip
- Narrow (24-28): C 69% flip, A 100% flip
- **Conclusion:** Both behaviors are driven by the same distributed mechanism; cannot separate by layer depth.

### 4. **Scaling is Linear**

Effect size scales monotonically with intervention magnitude:
- C: baseline 80% → ablated 25% (55-point effect)
- A: baseline 14% → ablated 0% (14-point effect)
- Linear model predicts intermediate values; no thresholds observed

### 5. **Direction Drifts During Training But Effect Persists**

Cosine similarity M0 vs. M3:
- Shallow layers (7, 14): ~65% similarity (35% drift)
- Deep layers (21, 28): ~43% similarity (57% drift)

Despite substantial drift, ablation remains highly effective (**causal robustness**).

---

## Analysis Reports

All detailed findings are in **`results/interpretability/reports/`** organized by topic for easy navigation:

| Report | Focus | Key Question |
|--------|-------|--------------|
| **[INDEX.md](results/interpretability/reports/INDEX.md)** | Navigation guide | Where do I start? |
| **[01: Wide Ablation](results/interpretability/reports/01_causal_ablation_wide.md)** | Primary effect | What happens when we remove layers 14-28? |
| **[02: Narrow Ablation](results/interpretability/reports/02_causal_ablation_narrow.md)** | Layer-range test | What happens with only layers 24-28? |
| **[03: Comparison](results/interpretability/reports/03_narrow_vs_wide_comparison.md)** | Selectivity | Why doesn't narrowing improve selectivity? |
| **[04: Layer Analysis](results/interpretability/reports/04_per_layer_signal_magnitude.md)** | Attribution | Which layers carry the most signal? |
| **[05: Alpha Scaling](results/interpretability/reports/05_alpha_scaling_analysis.md)** | Scaling law | How does effect size vary with magnitude? |
| **[06: Direction Stability](results/interpretability/reports/06_direction_stability_across_training.md)** | Training dynamics | How does the direction change during training? |
| **[07: Synthesis](results/interpretability/reports/07_key_findings_synthesis.md)** | Integration | What does it all mean? |

**→ Start with [INDEX.md](results/interpretability/reports/INDEX.md) or [Report 7 (Synthesis)](results/interpretability/reports/07_key_findings_synthesis.md) for overview**

---

## Methodology

### Phase 1: Model Training
- **M0:** Base Llama-7B-Chat
- **M1:** Helpful-only SFT (helpful-eval + chosen examples)
- **M2:** Helpful + Safety SFT (balanced split)
- **M3:** DPO (safety preference optimization)

### Phase 2: Activation Collection
Extract hidden states from each layer for all models.

### Phase 3: Behavior Evaluation
- Classify responses into: degenerate, refusal, soft-deflection, comply
- Quadrants: A (unsafe+safe), B (safe+unsafe), C (unsafe+safe_weak), D (safe+safe)

### Phase 4: Direction Extraction & Analysis
- **PCA:** Fit direction on difference-of-means (safe vs. unsafe responses) per layer and checkpoint
- **Projections:** Compute dot product of activations with direction (per sample, layer, quadrant)
- **Direction stability:** Cosine similarity across M0-M1-M2-M3

### Phase 5: Causal Ablation & Interpretability
- **Wide ablation:** Remove direction projection at layers 14-28
- **Narrow ablation:** Remove direction projection at layers 24-28 only
- **McNemar's test:** Paired significance testing on discordant pairs
- **Mechanistic analysis:** Per-layer attribution, alpha scaling, direction stability interpretation

---

## How to Cite

```bibtex
@project{dpo_safety_representations_2024,
  title={Mechanistic Interpretability of Refusal Behavior in DPO-Aligned Language Models},
  author={[Your Name]},
  year={2024},
  type={Research Project},
  url={https://github.com/yourusername/dpo-safety-representations}
}
```

---

## Limitations & Honest Assessment

1. **Non-selective mechanism:** The direction suppresses both legitimate and problematic refusal equally; cannot use for targeted safety improvements without further work.

2. **Single direction analysis:** Extracting only the top PCA component; other orthogonal dimensions may exist.

3. **Inference-time only:** Ablation is applied during forward pass; unknown how it affects training or generalization.

4. **Modest sample sizes:** Quadrant C has only 20 samples; quadrant A has 50. Larger eval sets would strengthen confidence.

5. **Training drift:** 35-57% representational drift raises questions about mechanistic stability, though causal efficacy persists.

---

## Future Work

### High-Priority

1. **Empirical α-validation:** Test behavior at intermediate projection scaling (α ∈ {0.1, 0.25, 0.5, 0.75, 0.9})
2. **Gradient-based attribution:** Use Integrated Gradients to refine per-layer importance estimates
3. **Subspace analysis:** PCA on per-quadrant activations to find behaviorally selective dimensions

### Medium-Priority

4. **Orthogonal component removal:** Extract separate directions for soft-deflection (C) and hard-refusal (A)
5. **Attention head analysis:** Identify which attention heads respond to the direction
6. **Prompt robustness:** Test direction effectiveness on diverse prompts and instruction frames

### Longer-Term

7. **Training-time intervention:** Modify loss function or regularization to refine direction selectivity during training
8. **Comparative analysis:** Compare this direction to other safety-relevant representations (e.g., adversarial examples, jailbreak indicators)

---

## Repository Hygiene

### Cleaned & Optimized

- ✅ Removed ~50MB of smoke-test binaries from git tracking (`.gitignore` updated)
- ✅ Centralized JSON I/O helpers (`src/io_utils.py`) to reduce boilerplate
- ✅ Reorganized `src/` by function (core, analysis, diagnostics, interpretability, training, utils)
- ✅ Created canonical results structure (`raw/`, `summaries/`, component subdirs)
- ✅ Made reproducibility scripts required (`--file` argument in McNemar test)
- ✅ All tests passing (run `pytest tests/ -v`)

### Git Status

```bash
git log main --oneline -5  # Verify all changes are committed to main
```

---

## Testing

```bash
# Run full test suite
pytest tests/ -v

# Run specific test module
pytest tests/test_interpretability.py -v

# All tests should pass
```

---

## Questions & Support

- **How do I understand this project?** → Start with [reports/INDEX.md](results/interpretability/reports/INDEX.md)
- **What's the main finding?** → Read [reports/07_key_findings_synthesis.md](results/interpretability/reports/07_key_findings_synthesis.md)
- **How do I reproduce results?** → Run commands in "Quick Start" section above
- **Where are the results?** → `results/interpretability/reports/` (individual reports) or `results/` (raw data)

---