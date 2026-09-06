# 654-benchmark findings synthesis (for the FLLMPT paper)

All numbers verified against `results/` JSONs from the 2026-09-05 Colab run
(judge file `behavioral_judges_v2_20260905T194720Z.json`). Every claim here
has a file behind it — no 370-era numbers.

## Confirmatory endpoints

| Endpoint | Estimate | 95% CI | n | source |
|---|---|---|---|---|
| CF1 (C, M2→M3, continuous SR) | −0.4008 | [−0.4603, −0.3396] | 104 | `summaries/confirmatory_endpoints.json` |
| CF2 (M3, held-out A, SR_AD − SR_random) | +0.1136 | [+0.0279, +0.2064] | 30 | same |
| CF2 secondary M3_direct | +0.0247 | [−0.0072, +0.0642] | 30 | same |
| CF2 secondary M3_alt | +0.0391 | [−0.0268, +0.1154] | 30 | same |
| CF2 secondary M3_direct_alt | +0.0185 | [−0.0009, +0.0419] | 30 | same |

`mean_SR`: M2 = 0.547, M3 = 0.146. WildGuard secondary (M3): +0.20 [+0.033, +0.367], n=30.
**Caveat:** `full_A_sensitivity` == `primary` for every branch — causal
ablation was only generated on the 30 held-out A prompts, so the
predeclared full-A (n=150) sensitivity analysis is **unavailable**.

## Branch-interaction bootstrap (EXPLORATORY — not preregistered)

`Δ_M3 − Δ_branch`, `Δ_s = mean_i(SR_AD − SR_random)` on prompts scored in
both branches, prompt-level paired bootstrap, seed 20260904.

| Comparison | interaction | 95% CI | verdict |
|---|---|---|---|
| M3 vs M3_direct_alt | +0.095 | [+0.007, +0.192] | **excludes 0** |
| M3 vs M3_alt | +0.075 | [+0.010, +0.140] | **excludes 0** |
| M3 vs M3_direct | +0.089 | [−0.015, +0.196] | spans 0 |

n=30, same-30-prompts resampling caveat (as the old paired-stability
p-values). Two of three CIs exclude zero with consistent sign.

## AD-vs-random McNemar, quadrant C soft_deflection (n=104) — the correct direction-specificity test

`--conditions {stage}_ablated_AD {stage}_ablated_random --quadrant C --category soft_deflection`

| Branch | discordant (AD→rand) | McNemar exact p | direction-specific? |
|---|---|---|---|
| M3 | 0 / 29 | **< 0.000001** | yes |
| M3_alt | 3 / 18 | **0.013** | yes |
| M3_direct | 3 / 8 | 0.227 | no |
| M3_direct_alt | 7 / 7 | **1.000** | no (perfectly random-equivalent) |

**IMPORTANT correction:** the earlier-cited `p = 0.000002` for M3_direct was
`baseline` vs `ablated_AD` (a generic-ablation effect), NOT `ablated_AD` vs
`ablated_random`. On the correct test, M3_direct's large quadrant-C drop is
**not** direction-specific.

**Synthesis:** direction-specific causal effects are confirmed only for the
two safety-SFT-mediated branches (M3, M3_alt). Neither direct-DPO branch
shows a confirmed direction-specific effect on either endpoint;
M3_direct_alt is indistinguishable from its random control (McNemar p=1.00,
interaction vs M3 excludes 0). The pivot is **safety-SFT present vs absent**,
not the instruction corpus alone.

## CF3 — orthogonal benchmark-category decodability (H1 test)

macro-F1: M2 = 0.893 → M3 = 0.878. `cf3 = M3 − M2 = −0.016`, bootstrap over
independent groups CI **[−0.038, +0.005]** (spans 0, point estimate
slightly negative). n_A=150, n_C=104, 4 categories. **No evidence DPO adds
linearly-decodable structure orthogonal to the A–D contrast.**
Source: `interpretability/direction_decodability_cf3.json`.

## Subspace geometry (H1/H2) — MIXED, report all four §4 outcomes

r=5 primary, layers 24–28 (`interpretability/subspace_geometry.json`):

- **Amplification (H2):** `contrast_norm` M2→M3 grows — L24: 36.2 → 41.5
  (×1.15); L28: 37.2 → 55.3 (×1.49). Participation ratio ~flat (L24:
  33.6 → 33.8).
- **Orthogonal update (H1):** `ρ_AD_perp` = 0.80–0.97 at L20–27 — most of
  the M2→M3 update to the contrast is orthogonal to M2's top-5 A/D
  subspace. Principal angles M2↔M3 mean 20–26°, max to 63°. Effective rank
  rises slightly (77 → 79 at L24).
- Reconciliation: large geometric orthogonal component, but CF3 says it
  carries no additional decodable category structure. Not pure
  amplification; the "extra" is not recoverable richer structure. Report as
  a **mixture** per analysis_plan.md §4, not a headline.

## Projection-magnitude trajectory (§4.5) — z_C, a clean figure

z_C = C's position between D (=0) and overt-harmful A (=1) along each
stage's own direction, L24 (`refusal_direction/projection_trajectory.json`):

| stage | M0 | M1 | M2 | M3 | M3_direct | M1_alt | M2_alt | M3_alt | M3_direct_alt |
|---|---|---|---|---|---|---|---|---|---|
| z_C@L24 | +0.33 | +0.72 | +0.65 | +0.90 | +1.04 | +0.70 | +0.61 | +0.80 | +0.95 |

C moves toward the overt-harmful cluster in two jumps — instruction-tuning
(M0→M1) and DPO (M2→M3) — with a slight retreat during safety-SFT. Direct-DPO
overshoots (M3_direct z_C > 1: C past A). This is the project's founding
Hypothesis B in one figure.

**A–D gap** `ad_gap@L24`: M0 22.2 → M1 37.0 → M2 36.7 → M3 41.8 (+14% for
DPO; instruction-tuning did the bulk). M3_direct 56.3, M3_direct_alt 58.8
(direct-DPO widens it ~53% over M2 — Finding 3).

**z_B caveat:** z_B@L24 ≈ +0.27–0.42 across all stages (stable). Benign-but-
alarming prompts sit ~⅓ of the way toward the harmful side along d — the
A–D contrast carries topic/style/wording structure, not only safety.
State this.

## Findings that survived the 654 rebuild

| Finding | 370-era | 654-era | verdict |
|---|---|---|---|
| F1: direction forms early (M0→M1 cosine) | ~0.665 | **0.654** (M1→M2 0.958, M2→M3 0.930) | holds |
| F3: cross-branch mediated − direct | +0.044 [.037,.052] | **+0.038 [+0.033, +0.044]**, 100% reps | holds |
| F3: deep-layer stability (direct − mediated) | +0.015, Wilcoxon p≈0 | **+0.009 [+0.005, +0.016]** pooled, 100% reps | holds; method upgraded to joint bootstrap |
| Bottleneck layer (M3, A-vs-D) | 16, 99% mode | **16, mode 99.2%, CI [16,16]**, d=4.55 | holds exactly |
| LoRA subspace | "90%+ outside rank-64" | 90–94% outside; but L21/28 in-subspace fraction 0.10 vs random 0.04 | holds with nuance |
| F3: "7-layer harm-vs-surface gap" | claimed | argmax noise (correctly labelled `_EXPLORATORY` in the file) | **walked back — keep visible** |

## Corrections for the paper (not OpenReview)

1. Abstract cosine "0.875–0.919" — true per-pair layer-means are **0.870
   (M3_direct) – 0.910 (M2)**; bootstrap CI envelope reaches 0.916. Paper
   table uses the real values.
2. `full_A_sensitivity` unavailable (see CF2 caveat above) → Limitations.
3. "reference-free DPO" → π_ref = merged preceding checkpoint (CLAUDE.md
   already fixed).

## Cross-branch cosine (654), per-pair layer-means

M1_vs_M1_alt 0.891 · M2_vs_M2_alt 0.910 · M3_vs_M3_alt 0.906 ·
M3_direct_vs_M3_direct_alt 0.870. Bootstrap per-pair CIs in
`interpretability/bootstrap_cross_branch_difference.json`.
