# Stage comparison — findings

*EXPLORATORY - not preregistered (analysis_plan.md frozen: CF1/CF2 confirmatory, CF3 predeclared secondary)*

Produced 2026-09-29T12:28:10Z · pooling `final_token` · behavioural metric `withhold` (refusal or soft deflection)

`*` marks an interval that excludes zero.


## 1. Withhold rate by stage and quadrant

| stage | A | B | C | D |
|---|---:|---:|---:|---:|
| M0 | 0.053 | 0.040 | 0.019 | 0.067 |
| M1 | 0.013 | 0.004 | 0.000 | 0.000 |
| M2 | 0.040 | 0.052 | 0.038 | 0.020 |
| M3 | 0.240 | 0.072 | 0.356 | 0.033 |
| M3_direct | 0.427 | 0.240 | 0.250 | 0.220 |
| M1_alt | 0.013 | 0.000 | 0.000 | 0.007 |
| M2_alt | 0.027 | 0.052 | 0.029 | 0.013 |
| M3_alt | 0.320 | 0.076 | 0.308 | 0.040 |
| M3_direct_alt | 0.233 | 0.216 | 0.067 | 0.187 |

## 2. The 2x2 — objective x corpus

Objective-and-recipe contrast at matched initialisation and matched source corpus. M1->M2 and M1->M3_direct share the start checkpoint and the PKU-SafeRLHF material but differ in training signal (DPO consumes chosen AND rejected responses, SFT only the chosen one), in epochs (2 vs 1) and in learning rate (2e-4 vs 5e-5). Not an isolation of the DPO objective. n=2 corpora, one instantiation each: intervals on the corpus and interaction terms are over PROMPTS, not over corpora.


### A  harmful, overt wording  (n = 150)

| arm | theta | 95% CI |
|---|---:|---|
| safety SFT  (M1 -> M2) | +0.027 | [+0.000, +0.060] |
| DPO         (M1 -> M3_direct) | +0.413 | [+0.333, +0.493] |
| safety SFT  (M1_alt -> M2_alt) | +0.013 | [-0.020, +0.047] |
| DPO         (M1_alt -> M3_direct_alt) | +0.220 | [+0.147, +0.293] |

| effect | estimate |
|---|---|
| OBJ | +0.297 [+0.237, +0.357]  * |
| CORP | +0.103 [+0.043, +0.163]  * |
| INT | +0.180 [+0.073, +0.280]  * |
| DPO − SFT, Alpaca | +0.387 [+0.300, +0.473]  * |
| DPO − SFT, Dolly | +0.207 [+0.133, +0.280]  * |

### B  benign, alarming wording  (n = 250)

| arm | theta | 95% CI |
|---|---:|---|
| safety SFT  (M1 -> M2) | +0.048 | [+0.024, +0.076] |
| DPO         (M1 -> M3_direct) | +0.236 | [+0.184, +0.288] |
| safety SFT  (M1_alt -> M2_alt) | +0.052 | [+0.024, +0.080] |
| DPO         (M1_alt -> M3_direct_alt) | +0.216 | [+0.168, +0.268] |

| effect | estimate |
|---|---|
| OBJ | +0.176 [+0.130, +0.222]  * |
| CORP | +0.008 [-0.024, +0.038] |
| INT | +0.024 [-0.036, +0.084] |
| DPO − SFT, Alpaca | +0.188 [+0.132, +0.244]  * |
| DPO − SFT, Dolly | +0.164 [+0.108, +0.220]  * |

### C  harmful, reduced-cue wording  (n = 104)

| arm | theta | 95% CI |
|---|---:|---|
| safety SFT  (M1 -> M2) | +0.038 | [+0.010, +0.077] |
| DPO         (M1 -> M3_direct) | +0.250 | [+0.173, +0.337] |
| safety SFT  (M1_alt -> M2_alt) | +0.029 | [+0.000, +0.067] |
| DPO         (M1_alt -> M3_direct_alt) | +0.067 | [+0.019, +0.115] |

| effect | estimate |
|---|---|
| OBJ | +0.125 [+0.077, +0.173]  * |
| CORP | +0.096 [+0.053, +0.144]  * |
| INT | +0.173 [+0.067, +0.279]  * |
| DPO − SFT, Alpaca | +0.212 [+0.125, +0.298]  * |
| DPO − SFT, Dolly | +0.038 [-0.010, +0.096] |

### D  benign, plain wording  (n = 150)

| arm | theta | 95% CI |
|---|---:|---|
| safety SFT  (M1 -> M2) | +0.020 | [+0.000, +0.047] |
| DPO         (M1 -> M3_direct) | +0.220 | [+0.153, +0.287] |
| safety SFT  (M1_alt -> M2_alt) | +0.007 | [-0.013, +0.033] |
| DPO         (M1_alt -> M3_direct_alt) | +0.180 | [+0.120, +0.247] |

| effect | estimate |
|---|---|
| OBJ | +0.187 [+0.137, +0.237]  * |
| CORP | +0.027 [-0.013, +0.067] |
| INT | +0.027 [-0.053, +0.107] |
| DPO − SFT, Alpaca | +0.200 [+0.140, +0.267]  * |
| DPO − SFT, Dolly | +0.173 [+0.113, +0.233]  * |

## 3. Selectivity — discrimination, not just more refusing


### A minus D (overt axis)  (n = 150 vs 150)

| arm | selectivity | 95% CI |
|---|---:|---|
| safety SFT  (M1 -> M2) | +0.007 | [-0.033, +0.047] |
| DPO         (M1 -> M3_direct) | +0.193 | [+0.087, +0.300]  * |
| safety SFT  (M1_alt -> M2_alt) | +0.007 | [-0.033, +0.047] |
| DPO         (M1_alt -> M3_direct_alt) | +0.040 | [-0.053, +0.133] |
| **OBJ (DPO − SFT)** | +0.110 | [+0.033, +0.190]  * |

Positive arm selectivity means the transition raised withholding on harmful prompts more than on benign ones. A transition can have a large OBJ on both quadrants and near-zero selectivity, which would mean it raised refusal indiscriminately.


### C minus B (reduced-cue axis)  (n = 104 vs 250)

| arm | selectivity | 95% CI |
|---|---:|---|
| safety SFT  (M1 -> M2) | -0.010 | [-0.054, +0.038] |
| DPO         (M1 -> M3_direct) | +0.014 | [-0.082, +0.113] |
| safety SFT  (M1_alt -> M2_alt) | -0.023 | [-0.064, +0.022] |
| DPO         (M1_alt -> M3_direct_alt) | -0.149 | [-0.218, -0.078]  * |
| **OBJ (DPO − SFT)** | -0.051 | [-0.117, +0.018] |

Positive arm selectivity means the transition raised withholding on harmful prompts more than on benign ones. A transition can have a large OBJ on both quadrants and near-zero selectivity, which would mean it raised refusal indiscriminately.


## 4. Depth attribution — where the contrast is written

Per-layer write of the A−D contrast onto each stage's own direction, `write[l] = (h_l − h_{l−1}) · d_l`. Exact, not an estimate.

| stage | total A−D | fraction written by layer 24 | share falling in layers 24–28 |
|---|---:|---:|---:|
| M0 | 51.2 | 0.652 | 0.416 |
| M1 | 61.9 | 0.779 | 0.288 |
| M2 | 63.0 | 0.779 | 0.287 |
| M3 | 83.2 | 0.721 | 0.334 |
| M3_direct | 110.1 | 0.825 | 0.243 |
| M1_alt | 57.8 | 0.845 | 0.242 |
| M2_alt | 59.3 | 0.794 | 0.283 |
| M3_alt | 81.3 | 0.733 | 0.320 |
| M3_direct_alt | 105.2 | 0.887 | 0.177 |

Peak layer is **not** reported: the per-layer curve is multi-peaked and its argmax is unstable across stages, the same failure mode that retired the earlier bottleneck-gap claim.


## 5. Judge coverage — what these numbers rest on

| stage | rows | regex | StrongREJECT | WildGuard |
|---|---:|---:|---:|---:|
| M0 | 654 | 654 | 0 | 0 |
| M1 | 654 | 654 | 0 | 0 |
| M2 | 654 | 654 | 104 | 104 |
| M3 | 654 | 654 | 104 | 104 |
| M3_direct | 654 | 654 | 0 | 0 |
| M1_alt | 654 | 654 | 0 | 0 |
| M2_alt | 654 | 654 | 0 | 0 |
| M3_alt | 654 | 654 | 0 | 0 |
| M3_direct_alt | 654 | 654 | 0 | 0 |

Everything above uses the frozen regex classifier, the only scorer complete for all nine stages. StrongREJECT and WildGuard currently cover quadrant-C rows of M2 and M3 only.
