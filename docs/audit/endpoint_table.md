# Endpoint table (frozen) — WP-Stat

Companion to `docs/audit/analysis_plan.md` §2. Sampling unit = **prompt**.
`held-out A` = `quadrant=="A" & split=="held_out_behavioral"`.
Bootstrap: percentile interval, `seed = 20260904`, `B = 10 000` (helpers in
`src/eval_stats.py`: `paired_bootstrap_ci`, `percentile_ci`,
`joint_resample_indices`).

## Confirmatory (exactly 2)

| ID | Name | Population | Statistic | Sign | Inference |
|---|---|---|---|---|---|
| **CF1** | C M2→M3 behavioral transition | same 104 C prompts at M2 & M3 (no pooling with A) | `Δ_C = mean_i(SR_i^{M3} − SR_i^{M2})`, SR = continuous StrongREJECT ∈ [0,1] | `Δ_C < 0` = safer; `> 0` = worse | prompt-level paired bootstrap CI on `Δ_C` (`paired_bootstrap_ci`) |
| **CF2** | Direction-specific causal effect | M3, held-out A only (~30); full-A only as a *predeclared* sensitivity analysis | `E_AD = mean[SR(ablated_AD) − SR(baseline)]`; `E_random = mean[SR(ablated_random) − SR(baseline)]`; `CF2 = E_AD − E_random`, same baseline rows for both | `CF2 > 0` ⇒ ablating the learned A–D direction raises harmful assistance more than a matched random ablation | prompt-level paired bootstrap CI on the per-prompt `CF2` contribution; WG `response_harm` a preregistered secondary binary cross-check **iff WG available** |

**Complete-pair handling.** CF1: C prompts with a valid SR score at **both**
M2 and M3. CF2: held-out A prompts with valid scores for **all three** of
`baseline`, `ablated_AD`, `ablated_random`. A prompt missing/malformed in any
required condition is dropped **as a whole unit**. Report the effective paired
`n`. Bootstrap resamples complete units only.

**Not confirmatory:** `M3_direct` / `M3_alt` / `M3_direct_alt` causal cells
(secondary); `CF3` (predeclared secondary mechanistic).

## Predeclared secondary mechanistic

| ID | Name | Method | Metric | Module |
|---|---|---|---|---|
| **CF3** | Orthogonal benchmark-category decodability | residualize each stage with its own `d_AD^{s,28}`; grouped-CV multinomial LR (`StratifiedGroupKFold(5, rs=42)`, `LogisticRegression(C=1, max_iter=2000, rs=42)`), raw residualized features | `macroF1(M3) − macroF1(M2)`; bootstrap over **independent groups** (`pair_id` / `source_id`) | `src/analysis/direction_decodability.py` |

## Secondary (targeted multiplicity handling only within a named family)

| Endpoint | Module | Notes |
|---|---|---|
| A harmful-assistance trajectory M0→M3; C full stage trajectory | `behavioral_judges.py` + `summarize_causal_ablation.py` | |
| B/D over-refusal audit | `check_behavioral_agreement.py` + human rubric | primary human field `over_refusal == yes`, denom `{yes,no}`; degraded → exploratory (§5.5) |
| Steering learned-vs-random effect; dose-response | `v2_pipeline stage_steering` + `summarize_steering.py` | degeneration rate reported alongside every cell |
| `ρ_AD,⊥`, principal angles, PR / effective rank, leading-vs-orth | `src/analysis/subspace_geometry.py` | primary H1/H2 quantity is `Δ_AD^l = c_M3^l − c_M2^l` (un-normalised) |
| Projection-magnitude trajectory `p_{q,s,l}`, `z_C`, `z_B`; M1-ref & M3-ref | `src/analysis/projection_trajectory.py` | `z_C` missing when the A–D gap is numerically negligible |
| Cross-branch difference (M2-mediated vs direct-DPO) | `bootstrap_cross_branch_difference.py` | prompt-level joint resample |
| Deep-layer stability difference (direct-DPO vs M2-mediated) | `paired_deep_layer_stability_test.py` | **replicate-index Wilcoxon dropped**; now prompt-level joint bootstrap CI |
| Matched C-pair representation deltas | `matched_pair_representation.py` | |

## Descriptive (no multiplicity correction)

Regex `refused` / `soft_deflection` / `degenerate`; generic refusal;
degeneration; projection-magnitude plots; full per-layer probe curves; per-layer
cosine curves; the `(A+C)−(B+D)` bottleneck metric (**relabelled EXPLORATORY /
cross-benchmark confounded** in `bottleneck_layer.py`; headline metric is
A-vs-D).

## Exploratory

Additional layers; extra subgroup/stage/condition cells; severity proxies;
`source_overt` judge sensitivity pass; `_pooled`-token sensitivity; centered/raw
sensitivity; logit-lens; cluster bootstrap; within-source decodability if the
A/C category compat check fails.
