# Stage comparison - findings memo

*EXPLORATORY - not preregistered (analysis_plan.md frozen: CF1/CF2 confirmatory, CF3 predeclared secondary)*

Generated from the artifacts in `results/stagecmp/`. Every number here is reproducible from them; none is typed by hand.


## The question

`M1 -> M2` (safety SFT) and `M1 -> M3_direct` (DPO) start from the **same checkpoint** and consume the **same PKU-SafeRLHF material**, differing in objective and recipe. Same for the `_alt` pair with Dolly in place of Alpaca. The question is not which refuses more, but whether the two objectives differ in **building an internal harmful-versus-benign distinction at all**.

Note the recipe asymmetry runs in safety SFT's favour: it gets 2 epochs at lr 2e-4 against DPO's 1 epoch at 5e-5, so its small effect below cannot be blamed on a smaller optimisation budget. What DPO does see, and SFT does not, is the rejected response of each pair.


## 0. Can the instruments carry this?

**Degenerate output.** A refusal label on garbled text means nothing, so a stage above 10% degenerate generations carries no behavioural claim. Excluded: M0.

| stage | degenerate | usable |
|---|---:|---|
| M0 | 0.552 | **NO** |
| M1 | 0.005 | yes |
| M2 | 0.034 | yes |
| M3 | 0.009 | yes |
| M3_direct | 0.000 | yes |
| M1_alt | 0.002 | yes |
| M2_alt | 0.017 | yes |
| M3_alt | 0.015 | yes |
| M3_direct_alt | 0.003 | yes |

**The judge is validated on this benchmark, not only on its own.** WildGuard's prompt-harm label never enters the analysis, and it recovers the benchmark's design labels at kappa 0.953 (accuracy 0.977). That validates the judge and the intent axis together.

**The two judges are not interchangeable.** They agree on refusal (AUROC 0.940 against the continuous StrongREJECT score) and much less on graded harm (AUROC 0.718). Refusal is therefore the behavioural metric to lead with. Chance-corrected agreement needs a threshold on a continuous score, and it peaks at kappa 0.720 - good, not the near-perfect figure an AUROC of 0.94 might suggest.


## 1. The representation - safety SFT does not build the contrast

`contrast_norm_ratio` is ||c_post|| / ||c_pre||, where c is the harmful-minus-benign difference in means. It is a **ratio**, so a transition that moves every prompt equally scores 1.0 however far it moved. `*` marks an interval excluding 1.0.

| arm | L16 | L20 | L24 | L28 |
|---|---|---|---|---|
| safety SFT, Alpaca | 0.985 [0.973, 0.999] * | 0.989 [0.973, 1.006] | 0.992 [0.974, 1.010] | 0.947 [0.927, 0.964] * |
| safety SFT, Dolly | 1.094 [1.077, 1.113] * | 1.040 [1.021, 1.062] * | 1.000 [0.979, 1.024] | 1.066 [1.033, 1.101] * |
| DPO, Alpaca | 1.321 [1.232, 1.398] * | 1.778 [1.622, 1.901] * | 1.540 [1.419, 1.596] * | 1.627 [1.508, 1.689] * |
| DPO, Dolly | 1.643 [1.511, 1.751] * | 1.811 [1.675, 1.918] * | 1.603 [1.483, 1.655] * | 1.673 [1.528, 1.763] * |

**The intervals do not overlap between objectives.** Safety SFT stays within a few percent of 1.0 and not consistently on one side of it; DPO sits far above at every layer, in both corpora. The earlier phrasing - that safety SFT leaves the ratio *indistinguishable* from 1.0 - was wrong: at some layers its interval does exclude 1.0, narrowly and in opposite directions across corpora. Non-overlap between objectives is the claim that survives.

Drift ratios tell the same story on a second scale-free measure: safety SFT displaces the quadrants roughly equally, DPO displaces the harmful ones more.

| arm | drift A/D (L24) | drift C/B (L24) |
|---|---|---|
| safety SFT, Alpaca | 1.069 [1.025, 1.114] * | 1.052 [1.015, 1.092] * |
| safety SFT, Dolly | 1.023 [0.985, 1.063] | 1.087 [1.061, 1.114] * |
| DPO, Alpaca | 1.424 [1.332, 1.530] * | 1.202 [1.168, 1.236] * |
| DPO, Dolly | 1.419 [1.343, 1.505] * | 1.187 [1.158, 1.218] * |

The A/D ratio separates the objectives cleanly. The C/B ratio does **not**: both objectives show some anisotropy there and the intervals come close, so it is reported and not leant on.


**Saturation pre-check passed**: max CKA spread 0.503, max cosine spread 0.505, both far above the 0.02 ceiling threshold. The published warning that CKA saturates near 1.0 on fine-tuning checkpoints does not bind here.


## 2. Where in depth

The DPO/SFT separation in contrast ratio is noise through the first half of the network, becomes sustained from layer **16**, and peaks at layer **19** (gap 1.00). Layer 24 carries 55% of the peak separation, so reporting layer 24 alone - inherited from the preregistered ablation window, which no analysis here uses - understates the effect. The fixed grid 16, 20, 24, 28 is reported instead, and the full profile is the figure.


## 3. Behaviour - the objective dominates the corpus

WildGuard `response_refusal`, paired bootstrap over prompts. `*` marks an interval excluding zero.

| quadrant | OBJ (DPO - safety SFT) | corpus | interaction |
|---|---|---|---|
| A | +0.829 [+0.775, +0.879] * | -0.023 [-0.081, +0.034] | -0.074 [-0.134, -0.013] * |
| B | +0.316 [+0.266, +0.368] * | -0.064 [-0.098, -0.030] * | -0.144 [-0.212, -0.076] * |
| C | +0.894 [+0.846, +0.938] * | -0.058 [-0.111, -0.010] * | -0.019 [-0.087, +0.048] |
| D | +0.147 [+0.093, +0.203] * | -0.027 [-0.067, +0.013] | -0.027 [-0.093, +0.040] |

Every objective term is positive with an interval excluding zero, from +0.147 to +0.894, and the harmful quadrants move several times as far as the benign ones. |corpus| never exceeds 0.064.

Under StrongREJECT, which scores harmful assistance and therefore carries the opposite sign, the corpus term is null on **all four** quadrants - cleaner than under WildGuard.

| quadrant | OBJ | corpus |
|---|---|---|
| A | -0.430 [-0.472, -0.390] * | -0.009 [-0.054, +0.036] |
| B | -0.207 [-0.232, -0.182] * | -0.012 [-0.036, +0.012] |
| C | -0.525 [-0.570, -0.481] * | -0.010 [-0.061, +0.040] |
| D | -0.033 [-0.054, -0.012] * | +0.018 [-0.007, +0.043] |

## 4. What this does NOT show: better discrimination

The objective difference is first of all a difference in magnitude.

| arm | mean absolute movement |
|---|---:|
| safety SFT, Alpaca | 0.039 |
| safety SFT, Dolly | 0.049 |
| DPO, Alpaca | 0.553 |
| DPO, Dolly | 0.629 |

So any additive quadrant gap is larger for DPO partly because DPO moved everything further. Two ratio scales separate the two, and they disagree instructively.

| checkpoint | risk ratio C/B | its ceiling | % of ceiling | log odds ratio C/B |
|---|---|---:|---:|---|
| M1 | *unreliable* (1 prompts) | 250.0 | 5% | +2.22 [+0.88, +3.79] * |
| M2 | 2.40 [0.92, 6.01] | 25.0 | 10% | +0.93 [-0.02, +1.87] |
| M3_direct | 3.45 [2.87, 4.30] * | 3.5 | 98% | +4.63 [+3.76, +6.38] * |
| M1_alt | *unreliable* (0 prompts) | 125.0 | - | -0.74 [-1.54, +0.87] |
| M2_alt | *unreliable* (9 prompts) | 27.8 | 10% | +1.04 [+0.07, +2.05] * |
| M3_direct_alt | 2.36 [2.07, 2.78] * | 2.4 | 100% | +5.65 [+5.41, +5.92] * |

**The risk ratio structurally cannot answer the question here.** A risk ratio cannot exceed 1 / p_benign, and the DPO arms sit at 98% and 100% of that arithmetic ceiling while the safety-SFT arms sit at about 10% of theirs. The DPO numbers are pinned by the benign rate, not measuring discrimination, which is exactly why the two objectives look similar on this scale. Add the small denominators - one arm's benign quadrant withholds on zero prompts - and the scale is unusable for this comparison.

The Haldane-corrected **log odds ratio is unbounded** and stays finite where a rate reaches exactly 0 or 1. On it, DPO's discrimination is clearly the larger, with the pre-DPO baselines shown so the change is visible rather than the level alone. Note the baselines are not equal: `M1` already discriminates somewhat and `M2` sits *below* it.

This is the honest version of a claim withdrawn from an earlier memo. "DPO buys refusal, not discrimination" was a regex artifact. "DPO discriminates better" is supportable, but only on the log-odds scale and only with the baselines beside it.


## 5. What bounds the reading

- **Objective and recipe, not objective alone.** DPO sees chosen and rejected responses; SFT only the chosen one. The accurate name is an objective-and-recipe contrast at matched initialisation and matched source corpus.
- **Safety SFT may be weak at this scale.** 4,000 chosen-only responses is a small intervention, so the finding is that *this* recipe at *this* scale does not build the contrast - not that supervised fine-tuning cannot.
- **One seed per cell.** The interaction term is a single draw.
- **n = 2 corpora.** Intervals on the corpus and interaction terms are over prompts, not over corpora.
- **Quadrant contrasts are between-source.** B is XSTest and D is instruction data; A is HarmBench and C is reworded StrongREJECT. The cue axis is not a validated construct in this project, so no quadrant difference is attributed to wording.
- **The DPO arms saturate** on the harmful quadrants, so any harmful-side difference is compressed against a ceiling.
- **Not preregistered.** Exploratory; intervals are nominal and uncorrected across quadrants.
