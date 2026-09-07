# 654-benchmark findings synthesis (for the FLLMPT paper)

Numbers verified against `results/` JSONs. The causal endpoints (CF2 and
below) are from the **2026-09-07 Colab run** — judge file
`behavioral_judges_v2_20260907T043919Z.json`, which added the full-A/D and
5-fold cross-fitted causal generations and re-scored them (see the
"cross-fitted" session in CLAUDE.md / the git log around `c5a16ec`). CF1,
CF3, geometry and the interpretability findings are unchanged from the
2026-09-05 run. Every claim here has a file behind it — no 370-era numbers.

## Confirmatory endpoints

| Endpoint | Estimate | 95% CI | n | source |
|---|---|---|---|---|
| CF1 (C, M2→M3, continuous SR) | −0.4008 | [−0.4603, −0.3396] | 104 | `summaries/confirmatory_endpoints.json` |
| **CF2 (M3, held-out A, SR_AD − SR_random) — PREREGISTERED ANCHOR** | +0.1136 | [+0.0279, +0.2064] | 30 | same |
| CF2 secondary M3_direct (held-out) | +0.0247 | [−0.0072, +0.0642] | 30 | same |
| CF2 secondary M3_alt (held-out) | +0.0391 | [−0.0268, +0.1154] | 30 | same |
| CF2 secondary M3_direct_alt (held-out) | +0.0185 | [−0.0009, +0.0419] | 30 | same |

`mean_SR`: M2 = 0.547, M3 = 0.146. WildGuard secondary (M3): +0.20 [+0.033, +0.367], n=30.
The held-out-30 M3 number is the **only preregistered confirmatory causal
test**. Everything below is post-hoc sensitivity.

### Cross-fitted out-of-fold estimate (POST HOC — 5-fold, n=120/branch)

Each quadrant-A `direction_estimation` prompt generated under a direction
(and matched-random-control magnitude) estimated from the other 4 folds.
Fold directions vs full direction: cos 0.998–0.9998 @ L24. Report as
**"out-of-fold n=120"**, never "independent n=120" (fold-training sets
overlap by 3/4).

| Branch | held-out n=30 | **cross-fitted n=120** | full-A n=150 |
|---|---|---|---|
| M3 | +0.1136 [+0.028, +0.206] | **+0.1540 [+0.1054, +0.2029]** | +0.1577 [+0.1161, +0.1981] |
| M3_alt | +0.0391 [−0.027, +0.115] | **+0.0442 [+0.0054, +0.0846]** | +0.0474 [+0.0118, +0.0846] |
| M3_direct | +0.0247 [−0.007, +0.064] | **+0.0252 [+0.0129, +0.0385]** | +0.0244 [+0.0139, +0.0362] |
| M3_direct_alt | +0.0185 [−0.001, +0.042] | **+0.0120 [+0.0027, +0.0227]** | +0.0173 [+0.0076, +0.0283] |

**All four cross-fitted CIs exclude zero.** This replaces the old
"detected in / not detected in" framing: the effect is positive in every
branch; its **magnitude** is path-dependent (~13× M3 vs M3_direct_alt).

### Circularity bias — estimation_split minus cross_fitted, SAME 120 rows, paired

| Branch | est_split | cross_fit | bias (est − xf) | 95% CI | detectable? |
|---|---|---|---|---|---|
| M3 | +0.1687 | +0.1540 | +0.0147 | [−0.0149, +0.0456] | no |
| M3_direct | +0.0243 | +0.0252 | −0.0009 | [−0.0112, +0.0096] | no |
| M3_alt | +0.0495 | +0.0442 | +0.0053 | [−0.0156, +0.0261] | no |
| M3_direct_alt | +0.0170 | +0.0120 | +0.0050 | [+0.0010, +0.0093] | yes, tiny |

Self-inclusion bias is undetectable for 3/4 branches and ≤ +0.005 where
detectable → `full_A_sensitivity` can be read close to face value. This is
the measurement the "circularity" critique asked for (paired, same rows),
NOT a held-out-n30-vs-estimation-n120 comparison (different populations).

## Cross-fitted branch contrasts (POST HOC — the properly-powered path-dependence test)

Paired bootstrap on the **cross-fitted per-prompt effects**; folds are
deterministic on the shared prompt set so all contrasts are prompt-paired.
n=120, seed 20260904.

| Contrast | estimate | 95% CI | verdict |
|---|---|---|---|
| M3 vs M3_alt | +0.1098 | [+0.0546, +0.1624] | **excludes 0** |
| M3 vs M3_direct | +0.1287 | [+0.0805, +0.1753] | **excludes 0** |
| M3 vs M3_direct_alt | +0.1420 | [+0.0931, +0.1891] | **excludes 0** |
| M3_alt vs M3_direct_alt | +0.0322 | [−0.0088, +0.0737] | spans 0 |
| history main effect (mediated − direct) | +0.0805 | [+0.0447, +0.1154] | **excludes 0** |
| corpus main effect (Alpaca − Dolly) | +0.0615 | [+0.0327, +0.0887] | **excludes 0** |
| **corpus × history interaction** | **+0.0965** | **[+0.0404, +0.1504]** | **excludes 0** |

The interaction is the substantive result: corpus moves the cross-fitted
effect by only +0.013 within the direct-DPO pair but +0.110 within the
mediated pair. Within Dolly, mediated-vs-direct (M3_alt vs M3_direct_alt)
spans zero → the mediation effect is carried almost entirely by the Alpaca
branch. **Post hoc, single seed per cell — see Limitations.**

## Branch-interaction bootstrap (EXPLORATORY — held-out n=30, SUPERSEDED by the above)

`Δ_M3 − Δ_branch` on the 30 held-out A prompts. **Unstable at n=30** — two
of three CIs moved across the zero boundary between the 2026-09-05 and
2026-09-07 judge runs (which barely touched held-out rows). Kept only as
corroboration; lead with the n=120 cross-fitted contrasts.

| Comparison | 2026-09-05 | 2026-09-07 |
|---|---|---|
| M3 vs M3_alt | +0.075 [+0.010, +0.140] | +0.073 [+0.005, +0.142] |
| M3 vs M3_direct | +0.089 [−0.015, +0.196] | +0.100 [+0.004, +0.205] |
| M3 vs M3_direct_alt | +0.095 [+0.007, +0.192] | +0.088 [−0.003, +0.188] |

## AD-vs-random McNemar, quadrant C soft_deflection (n=104) — the correct direction-specificity test

`--conditions {stage}_ablated_AD {stage}_ablated_random --quadrant C --category soft_deflection`

Authoritative counts (re-run 2026-09-07 against the frozen held-out files,
which the re-judge did **not** touch — regex classifier, not judge scores).
`b` = discordant where AD-ablation flagged `soft_deflection` and random did
not; `c` = the reverse. Full JSON: `results/summaries/mcnemar_direction_specificity.json`.

| Branch | b / c | McNemar exact p | reaches p<0.05? |
|---|---|---|---|
| M3 | 0 / 29 | **8.9e−7** | yes |
| M3_alt | 3 / 14 | **0.0127** | yes |
| M3_direct | 3 / 8 | 0.2266 | no (11 discordant, underpowered) |
| M3_direct_alt | 7 / 7 | **1.0000** | no (14 discordant, split evenly → random-equivalent) |

### n=150 quadrant A/D McNemar (SENSITIVITY — category `refusal`, coarse flag)

`_fullAD` files. Noisier than the continuous endpoint; label "sensitivity".

| | M3 | M3_alt | M3_direct | M3_direct_alt |
|---|---|---|---|---|
| **quad A** b/c (p) | 1/13 (0.0018) | 8/20 (0.036) | 6/7 (1.00) | 12/27 (0.024) |
| **quad D** b/c (p) | 1/1 (1.00) | 0/4 (0.125) | 4/8 (0.39) | 12/10 (0.83) |

Quadrant A: M3 clean, M3_alt marginal, M3_direct null; **M3_direct_alt's
p=0.024 rests on 39 discordant pairs of a binary flag flipping both ways
under both ablations** — classifier instability on the least-stable branch,
not a clean directional effect (its continuous cross-fitted effect is the
smallest at +0.012). Quadrant D: **no over-refusal side effect anywhere**
(all p ≥ 0.125). The continuous full-A + cross-fitted estimates carry the
power argument; this regex check neither helps nor hurts it.

**IMPORTANT correction:** the earlier-cited `p = 0.000002` for M3_direct was
`baseline` vs `ablated_AD` (a generic-ablation effect), NOT `ablated_AD` vs
`ablated_random`. On the correct test, M3_direct's large quadrant-C drop is
**not** direction-specific.

### Duplicate-row verification (keep-first merge)

The `_fullAD` run regenerated the 30 held-out A rows already in the frozen
`causal_ablation_v2_{stage}_L24-28.json`, under identical
`(record_id, stage, condition)` keys → 660 duplicate keys in the merged
judged file, **123 with divergent response text** and 48 with divergent SR
score (fp16 non-determinism across batch compositions; greedy decode).
Verified: the sorted-glob manifest puts the frozen file first, keep-first
skips **90 duplicate rows** for M3's held-out triples, and recomputed CF2
primary = **+0.113623**, byte-identical to the pre-run anchor. The
preregistered number is provably reading the frozen generation.

**Synthesis (updated after the cross-fitted run):** the quadrant-C McNemar
reaches significance for the two branches with the *largest* cross-fitted
effects (M3, M3_alt) and not for the two smallest (M3_direct,
M3_direct_alt) — this tracks the magnitude ordering, it is NOT a
categorical presence/absence split. A coarse categorical test on quadrant
C has little power against effects as small as +0.025 (M3_direct) or
+0.012 (M3_direct_alt). M3_direct_alt's p=1.00 rests on 7/0 discordant
pairs — too few to detect an effect that size, not evidence of absence.
The direction-specific effect is present in all four branches
(cross-fitted CIs all exclude 0); what is path-dependent is its
**magnitude**. Safety-SFT-mediated paths show larger effects than
direct-DPO paths, but the design does not isolate safety-SFT as the sole
cause (M3-vs-M3_direct also varies the full downstream trajectory).

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
2. ~~`full_A_sensitivity` unavailable~~ — RESOLVED by the 2026-09-07 run;
   real n=150 numbers in the CF2 table above, plus the n=120 cross-fitted
   estimate and the paired circularity bias. Limitations now says "the
   preregistered anchor is n=30; everything larger is sensitivity", not
   "full-A unavailable".
3. "reference-free DPO" → π_ref = merged preceding checkpoint (CLAUDE.md
   already fixed).

## Cross-branch cosine (654), per-pair layer-means

M1_vs_M1_alt 0.891 · M2_vs_M2_alt 0.910 · M3_vs_M3_alt 0.906 ·
M3_direct_vs_M3_direct_alt 0.870. Bootstrap per-pair CIs in
`interpretability/bootstrap_cross_branch_difference.json`.
