# Stage comparison — decision memo

*EXPLORATORY - not preregistered (analysis_plan.md frozen: CF1/CF2 confirmatory, CF3 predeclared secondary)*

Generated from the artifacts in `results/stagecmp/`. Every number here is reproducible from them; none is typed by hand.


## The question

`M1 -> M2` (safety SFT) and `M1 -> M3_direct` (DPO) start from the **same checkpoint** and consume the **same PKU-SafeRLHF material**, differing in objective and recipe. Same for the `_alt` pair with Dolly in place of Alpaca. That is a 2x2 the experiment was built to support and that had never been analysed.


## 1. Behaviour — DPO moves refusal far more than safety SFT

Withhold rate = refusal or soft deflection, frozen regex classifier, paired bootstrap (seed 20260904, B=10000). `*` marks an interval excluding zero.

| quadrant | OBJ (DPO − safety SFT) | corpus | interaction |
|---|---|---|---|
| A | +0.829 [+0.775, +0.879] * | -0.023 [-0.081, +0.034] | -0.074 [-0.134, -0.013] * |
| A *(regex)* | +0.297 [+0.237, +0.357] * | +0.103 [+0.043, +0.163] * | +0.180 [+0.073, +0.280] * |
| B | +0.316 [+0.266, +0.368] * | -0.064 [-0.098, -0.030] * | -0.144 [-0.212, -0.076] * |
| B *(regex)* | +0.176 [+0.130, +0.222] * | +0.008 [-0.024, +0.038] | +0.024 [-0.036, +0.084] |
| C | +0.894 [+0.846, +0.938] * | -0.058 [-0.111, -0.010] * | -0.019 [-0.087, +0.048] |
| C *(regex)* | +0.125 [+0.077, +0.173] * | +0.096 [+0.053, +0.144] * | +0.173 [+0.067, +0.279] * |
| D | +0.147 [+0.093, +0.203] * | -0.027 [-0.067, +0.013] | -0.027 [-0.093, +0.040] |
| D *(regex)* | +0.187 [+0.137, +0.237] * | +0.027 [-0.013, +0.067] | +0.027 [-0.053, +0.107] |

**The asymmetry is the finding.** On the benign quadrants (B, D) the corpus and interaction terms both span zero: the over-refusal cost is a property of the objective, not of the corpus. On the harmful quadrants (A, C) both exclude zero: the benefit is corpus-contingent. What you pay is reliable; what you get is not.


## 2. Selectivity — DPO does discriminate, and the regex said otherwise

| arm | A − D (overt axis) | C − B (reduced-cue axis) |
|---|---|---|
| safety SFT, Alpaca | +0.047 | +0.012 |
| safety SFT, Dolly | +0.020 | +0.068 * |
| DPO, Alpaca | +0.706 * | +0.653 * |
| DPO, Dolly | +0.726 * | +0.584 * |

Both DPO arms gain large, significant selectivity on **both** axes, including C vs B where wording no longer betrays intent. Both safety-SFT arms sit near zero.

**CORRECTION.** An earlier version of this memo, computed on the frozen regex classifier, reported that no arm gained selectivity on the C-vs-B axis and that the Dolly DPO arm was significantly *anti*-selective at −0.149. Under the validated judge that same arm is **+0.584 and strongly selective**. "DPO buys refusal, not discrimination" was an artifact of an instrument that misses untemplated refusals, not a finding. It is withdrawn.


## 3. Representation — safety SFT barely moves it

At layer 24, `_final` pooling:

| arm | CKA | cos(d) | contrast ratio | rel. drift | selectivity A−D |
|---|---:|---:|---:|---:|---:|
| safety SFT, Alpaca | 0.970 | 0.947 | 0.992 | 0.191 | 1.78 |
| safety SFT, Dolly | 0.944 | 0.893 | 1.000 | 0.278 | 0.93 |
| DPO, Alpaca | 0.712 | 0.442 | 1.540 | 0.515 | 27.40 |
| DPO, Dolly | 0.698 | 0.357 | 1.603 | 0.557 | 31.01 |

Safety SFT leaves the contrast norm at ~1.00 and selectivity at ~1. DPO from the same checkpoint on the same data grows the contrast by 54–60%, rotates the axis to cos 0.36–0.44, and produces 15–30x the selectivity. The representational result mirrors the behavioural one.


**Saturation pre-check passed**: max CKA spread 0.503, max cosine spread 0.505, both far above the 0.02 ceiling threshold. The published warning that CKA saturates near 1.0 on fine-tuning checkpoints does not bind here, so the full 9x9 matrix was computed rather than skipped.


## 4. Pooling sensitivity

| quantity (layer 24) | `_final` | `_pooled` | robust? |
|---|---:|---:|---|
| contrast ratio, DPO Alpaca | 1.540 | 2.431 | yes, same sign |
| selectivity A−D, DPO Alpaca | 27.398 | 20.711 | yes, same sign |

**One result did NOT survive** and is therefore not claimed: the fraction of the contrast written by layer 24 is higher for the direct branches under `_final` (+0.104 Alpaca, +0.153 Dolly, both excluding zero) but the Alpaca sign **reverses** under `_pooled` (−0.034). "Direct DPO front-loads the contrast" is reported as a sensitivity, not a finding.


## 5. What is still open

**The behavioural numbers rest on one proxy scorer.** Regex-vs-WildGuard agreement on the 208 rows carrying all three scorers is **kappa = 0.293**, below the 0.60 threshold fixed before running. The regex classifier finds 1 refusal in 104 quadrant-C rows where WildGuard finds 42, because it only matches templated refusals while the models refuse these prompts in plain language.

A judging pass over the ~5,900 unscored behaviour rows is running. Until it lands, every behavioural number above is **unvalidated outside quadrant C**, and the over-refusal headline sits in B and D where no agreement data exists at all.


## 6. Proposed angle, and the honest caveats

Lead with the objective contrast at matched initialisation, and the asymmetry: the over-refusal cost is corpus-invariant while the harm-refusal benefit is not. The representational result gives it a mechanism — safety SFT barely perturbs the contrast, DPO transforms it.

Caveats that must appear in the paper, not the appendix:

- **Not matched on training signal.** DPO sees chosen *and* rejected responses; SFT only the chosen one. Also 2 epochs at 2e-4 vs 1 at 5e-5. The honest name is an objective-and-recipe contrast at matched initialisation and matched source corpus.
- **Safety SFT may simply be undertrained** — 4,000 chosen-only examples, 2 epochs. Cannot be ruled out with these runs. A reviewer will ask.
- **One seed per cell.** The 2x2 interaction is a single draw.
- **n = 2 corpora.** Intervals on the corpus and interaction terms are over prompts, not over corpora.
- **The 24–28 intervention window is undocumented** and the figure shows the DPO/SFT divergence beginning around layer 13–16, so that window catches the tail of where the contrast is built, not its onset.
