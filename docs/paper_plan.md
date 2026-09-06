# FLLMPT paper — execution plan (Sept 6 to Sept 12)

Abstract registered ("Preserved but Uncoupled"). Paper due **Sept 12,
23:00 GMT**. NeurIPS 2026 main-track format, double-blind, non-archival.
Internal-review target: **Monday Sept 8**.

Every number below is verified against a `results/` JSON and recorded in
`docs/findings_654_synthesis.md`. This file is the *execution* plan; that
file is the *evidence ledger*.

---

## Wording rules (from external review — non-negotiable in the manuscript)

| Say | Never say |
|---|---|
| "preregistered final-token A-D contrast, under the specified pooling and intervention procedure" | "the harmful-benign direction", "the safety subspace" |
| "harmfulness-related contrast" (justified by the factorial audit) | "pure harmfulness direction", "82% harmfulness" |
| "d_AD-tilde = d_H-tilde + d_S-tilde, exact for the UNNORMALIZED vectors" | "d_AD = d_H + d_S" for the unit vectors the intervention uses |
| "the harmfulness component contributed approximately 82% of the squared unnormalized norm under the stated decomposition; the cross-term is negative so the remainder is not '18% cue'" | "18% surface cue" |
| "across the safety-SFT to DPO transition the contrast became more aligned with the harmfulness component and less dominated by cue-related variation" | "DPO rotates the contrast toward harmfulness" |
| "direction-specific effects were DETECTED IN both safety-SFT-mediated branches and NOT DETECTED IN either direct-DPO branch" | "present only in", "safety-SFT caused the difference" |
| "the held-out 30 is the primary independent test; full-A is a sensitivity analysis mixing independent and non-independent rows; the estimation vs held-out comparison MEASURES possible in-sample influence rather than confirming the effect" | full-A is "proper power" or "independent confirmation" |
| "out-of-fold n=120 estimate" (if cross-fitting is run) | "independent n=120" |
| "the direction is source-sensitive (cos with OASST1-only approx 0.88-0.90) and pooling-sensitive (cos final vs pooled approx 0.59-0.74)" | anything implying the vector is construction-invariant |

Paper title may differ from the abstract registration. Candidates:
**"Preserved but Uncoupled"** (matches abstract) or **"Preserved but
Path-Dependent"** (more accurate now the positive and null branches form a
structured pattern, not a global uncoupling). Decide at draft time.

---

## Saturday Sept 6 — GPU + your time in parallel

### GPU session (~1.5 h): notebook `07_full_ad_and_robustness.ipynb`, Run all

Re-open from GitHub so it pulls the latest pinned commit.

1. **Step 3 — full-A/D causal, M3 FIRST.** Smoke-test cell asserts 900 rows
   (300 A/D x 3 conditions) and stops if wrong — `--all-ad-sensitivity`
   has never touched a real model. Verify row count, `_fullAD.json`
   written, binding sidecar present, BEFORE the other three branches launch.
2. **Step 3 cont. — M3_direct / M3_alt / M3_direct_alt** full-A/D. Writes
   `causal_ablation_v2_{stage}_L24-28_fullAD.json`; the frozen held-out-30
   files are untouched.
3. **Step 4 — McNemar at n=150** (CPU, ~2 min). Quadrant A (refusal) and
   quadrant D (over-refusal side-effect), AD-vs-random, all 4 branches,
   plus `summarize_causal_ablation` and `bootstrap_causal_effect`. Label
   every n=150 number "sensitivity", never "confirmation".
4. **Step 5 — re-judge (OPTIONAL, ~90-120 min).** Only if the session
   holds. Produces the continuous-StrongREJECT CF2 at n=150 and fills
   `CF2_by_stage[*].estimation_split_only` (currently n=0). Skippable —
   step 4 already gives regex direction-specificity at n=150.
5. **Step 6 — D-source robustness, all 9 stages** (CPU, ~1 min).
   PRECONDITION: every stage must use 654-row metadata. `v2_pipeline
   status` must show 9/9 bound first; do not let stale 370-row metadata
   into the all-stage table.
6. **Step 7 — factorial direction audit, all 9 stages** (CPU, ~1 min).
   Same 654-metadata precondition. Runs `--ad-rows est` and `--ad-rows all`.

**Send back:** step 4 p-values (A and D, all branches); step 6 cosine
table; step 7 alignment table + held-out A/D separation; step 5 printout
if run.

### Your time (starts now, independent of GPU)

- **Human annotation — the long pole.** 200 items in the artifact tool.
  Unblocks the instrument-validation section. Finish today + tomorrow.
- **Citation verification.** Open every arXiv id in the review threads /
  `findings_654_synthesis.md` refs. Confirm each exists and says what was
  attributed to it. Arditi et al. especially — if it already establishes
  the refusal direction pre-exists safety tuning, Finding 1 is a
  replication and the intro shifts from "we show" to "consistent with, and
  we quantify".

---

## Sunday Sept 7

1. Finish annotation -> `check_behavioral_agreement` -> hand-author
   `results/human_review/conclusions.json` -> `behavioral_robustness`
   (CPU, minutes).
2. I fold steps 4-7 into `findings_654_synthesis.md` -> one results table +
   one figure list.
3. **Cross-fitting decision gate.** If annotation is fully done Sunday
   night, I build the leave-fold-out direction flag (out-of-fold n=120
   causal estimate) and you run it Monday AM. Otherwise it is the first
   Future Work item — held-out 30 + full-A sensitivity + factorial audit
   is already a defensible package.

---

## Monday Sept 8 — draft for internal review

Section map (numbers current as of `findings_654_synthesis.md`):

1. **Intro.** H1 (richer representation) vs H2 (amplify existing low-dim
   mechanism); underdetermined by geometry alone. Position vs Arditi et
   al. after citation check.
2. **Design.** 2x2 (harmful x surface-cue) x 2 corpora x {safety-SFT
   chain, direct-DPO} = 9 checkpoints. Matched preference data M2/M3.
   pi_ref = merged preceding checkpoint (not "reference-free"). Frozen
   preregistration; A/D held-out split (240 est / 60 held-out).
3. **The contrast forms early.** Adjacent cosine M0->M1 0.654, M1->M2
   0.958, M2->M3 0.930. Base-model Cohen's d = 4.19. z_C trajectory
   0.33 -> 0.72 -> 0.65 -> 0.90 (Hypothesis B, one figure).
4. **What the contrast is.** Factorial audit: d_AD-tilde = d_H-tilde +
   d_S-tilde (unnormalized, residual ~1e-6). cos(d_AD, d_H) 0.77->0.84
   across M2->M3 (alt 0.77->0.82). Squared-norm share: harmfulness ~82%,
   negative cross-term. CONSTRUCTION DIAGNOSTIC: separates harmful(A+C) vs
   benign(B+D) d=+2.41 vs cue-strong vs cue-reduced d=+0.53. HELD-OUT A-vs-D
   separation: d_AD 3.39, d_H 3.05, d_S 1.97. Source-sensitivity cos with
   OASST1-only ~0.88-0.90; pooling-sensitivity cos(final, pooled)
   ~0.59-0.74. This is a representation diagnostic; causal claims rest on
   the held-out intervention.
5. **Behaviour shifts at the DPO transition.** CF1 = -0.40, 95% CI
   [-0.46, -0.34], n=104 (continuous StrongREJECT, quadrant C).
6. **Direction-specific causal effect.**
   - CF2 (M3, held-out 30, preregistered anchor): +0.114 [+0.028, +0.206].
   - Sensitivity: `estimation_split_only` and `full_A_sensitivity` blocks;
     the comparison measures possible self-inclusion inflation, does not
     confirm.
   - AD-vs-random McNemar, quadrant C, n=104: M3 p < 1e-6, M3_alt p = 0.013,
     M3_direct p = 0.23, M3_direct_alt p = 1.00.
     Correction: the p ~ 2e-6 cited in earlier drafts for M3_direct was
     baseline-vs-AD, NOT AD-vs-random.
   - n=150 McNemar from Saturday — labelled sensitivity.
7. **Path dependence.** Cosine 0.870-0.910 across matched branches
   (bootstrap CIs in `bootstrap_cross_branch_difference.json`). Effects
   detected in M3, M3_alt; not detected in M3_direct, M3_direct_alt.
   Branch-interaction bootstrap (EXPLORATORY, n=30): M3 vs M3_alt
   +0.075 [+0.010, +0.140] (excludes 0); M3 vs M3_direct_alt
   +0.095 [+0.007, +0.192] (excludes 0); M3 vs M3_direct
   +0.089 [-0.015, +0.196] (spans 0). State: establishes path dependence
   in this design; does NOT isolate safety-SFT as the sole causal factor
   (the same-corpus M3 vs M3_direct interval spans zero).
8. **No added orthogonal decodable structure.** CF3 = -0.016,
   [-0.038, +0.005] (macro-F1 M3 minus M2 after residualizing each stage's
   own direction). H1 not supported.
9. **Geometry is a mixture.** Contrast norm grows M2->M3 x1.15 (L24) to
   x1.49 (L28); participation ratio ~ flat; rho_AD-perp 0.80-0.97;
   principal angles mean 20-26 deg. Report all four analysis_plan.md sec 4
   outcomes, no headline.
10. **Instrument validation.** Human audit + agreement report; regex vs
    human vs judges.
11. **Limitations.** One seed (M3_direct_alt null is single-seed — say
    so); one model family / scale; LoRA rank 64 only (direction ~90-94%
    outside the rank-64 subspace, but ~2.5x chance in-subspace at L21/28);
    held-out causal n=30; full-A is sensitivity not power; D contains
    training-adjacent sub-sources (Alpaca/Dolly), OASST1-only direction
    cos ~0.88-0.90; pooling choice moves the direction (cos 0.59-0.74);
    the "7-layer harm-vs-surface" claim from earlier work was walked back
    to argmax noise — keep visible.
12. **Future work.** Multiple seeds per branch (biggest single upgrade);
    cross-fitted causal estimate; cross-branch direction / delta transfer
    (separate paper); full fine-tuning.

**Double-blind hygiene:** no repo URL, no GitHub handle, no
acknowledgements. Scrub PDF metadata (author field, doc properties). Do not
push the manuscript text into the public repo before notification
(October) — a distinctive-phrase search must not resolve to the repo.

---

## Tue-Thu Sept 9-11 — revise from internal reviewers

Watch for: (a) does the path-dependence framing hold, (b) is the factorial
audit clearly a diagnostic not a validation, (c) is the circularity
treatment convincing.

## Fri Sept 12 — final pass, submit before 23:00 GMT

---

## Explicitly out of scope for this paper

- **Transfer matrix** (`--direction-from` is written and tested, not run) —
  separate question, separate paper; will not resolve the direction-purity
  uncertainty.
- **Multiple seeds** — Future Work, stated as a limitation.
- **Procrustes / learned alignment** — not in this project.

## Priority order if forced to cut

1. Full-A McNemar sensitivity (steps 3-4) — keep.
2. Human annotation (feeds sec 10) — keep, highest value.
3. Factorial audit all-stages (step 7) — keep, cheap, answers "what is the
   vector".
4. D-source audit all-stages (step 6) — keep, cheap.
5. Re-judge for continuous n=150 (step 5) — drop first if GPU-tight.
6. Cross-fitting — drop unless annotation finishes early.
