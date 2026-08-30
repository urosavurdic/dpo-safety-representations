# 3F-A — Common Surface-Cue (`C_cue`) Axis: Design Specification

**Repository:** `https://github.com/urosavurdic/dpo-safety-representations`

**Branch:** `agent/c-quadrant-end-to-end-e0e2317a`

**Status:** Design-only. No scoring code implemented, no candidate A/B/C/D
datasets built, no benchmark files, `src/cue_scoring.py`,
`src/corpus_discrimination.py`, or `data/frozen_v2/*` touched. This document
recommends a construction method and a minimal pilot for a future,
separately-authorized task; it does not run anything.

---

## 0. Relationship to existing constructs — read this first

This repository already contains three lexical/human-judgment artifacts
that sound like `C_cue` but are not:

1. **`CUE`** (`src/cue_scoring.py`, TF-IDF+LogReg and Fightin' Words,
   design recorded in `logs/cue_reference_audit.md`) is an **operational
   lexical harmful-*association*** score. Both methods are fit directly on
   a harmful/benign label — harmful reference sources (HarmBench,
   StrongREJECT, SimpleSafetyTests, rotated leave-one-source-out) as the
   positive class, benign sources (XSTest-safe, quadrant-D pool) as the
   negative class. Because it is fit on intent labels, `CUE` cannot be
   `C_cue` — it is intent-entangled by construction, not merely by
   coincidence. §2.1 below adds a concrete empirical reason not to reuse
   it even as a starting point.
2. **`p_tfidf` / `p_selfinfo`** (3D-A/3D-B, `logs/3d_a_lexical_outlierness_design.md`,
   `logs/3d_b_lexical_outlierness_pilot.md`) measure **within-harmful
   lexical outlierness** — atypicality relative to other harmful prompts.
   This is computed entirely inside the harmful population; it has never
   been applied to a benign prompt and is not a candidate common axis by
   itself (see §2.2 for why it also should not be extended to one as-is).
3. **3D-H-A's 32 human ratings** (`logs/3d_h_construct_check_analysis.md`)
   measure `E_harm(x)` — perceived clarity of harmful-operational
   intent — on within-harmful prompts only, and were explicitly scoped as
   development-only, non-final evidence. Per the task brief §5, they
   cannot be reused as final cue labels, and per repository convention
   (3D-A/3D-C/3D-H's own repeated framing) a construct check that has
   already informed a design choice cannot double as that choice's
   independent validation.

None of the three is `C_cue(x)`. This document treats all three as
informative background only, never as ground truth.

---

## 1. Construct definitions

| Symbol | Definition | Existing repo artifact that is closest (and why it is not this) |
|---|---|---|
| `I(x)` | Underlying intent (harmful vs. benign), from trusted dataset curation. | Quadrant/source labels (A/B/C/D). Already well established; not at issue here. |
| `C_cue(x)` | Surface-cue salience: how strongly the *wording* contains safety-relevant surface signals, independent of whether `I(x)` is harmful or benign. | No existing artifact. `CUE` (above) measures harmful-*association*, which is `I(x)`-entangled by construction, not a wording-only signal independent of `I(x)`. |
| `E_harm(x)` | Human-perceived clarity that the wording communicates a harmful *operational* objective. | 3D-H-A's 32 ratings — but within-harmful only, and this rates clarity-of-harm, not cue salience in a way that is meaningful for a benign prompt (a benign prompt cannot have high "harmful-operational" clarity by definition; it can still have high `C_cue`). |
| `Y(x)` | Model behavior (e.g., refusal). | `results/behavioral_eval/*`. Out of scope for this design task. |

A benign prompt can have low `E_harm` (it isn't harmful) but high `C_cue`
(it uses safety-adjacent surface wording). This is the entire reason a
common axis is needed and is preserved as a hard constraint throughout
§§2–8.

---

## 2. Existing resources inventory

### 2.1 Reusing the existing `CUE` score would not just be conceptually wrong — it already disagrees with the pipeline's own within-harmful cue-reduction labels

`data/review/c_review_queue.csv` (104 human-reviewed rows, all
`review_status=accept`, `objective_preserved=yes`, `operational_detail_changed=yes` —
this is quadrant C of the frozen benchmark) already contains
paired Fightin'-Words-style scores for each row's `source_prompt`
(original StrongREJECT prompt) and `candidate_prompt` (the
`reduced_cue_source_rewrite`, i.e. the version the pipeline's own naming
asserts is *lower*-cue). Aggregated across all 104 accepted pairs:

| Field | Source (original) | Candidate (reworded, intended lower-cue) |
|---|---|---|
| `word_count` mean (sd) | 28.5 (14.7) | 17.5 (3.6) |
| `character_count` mean (sd) | 159.7 (77.8) | 111.0 (22.5) |
| `fightin_words` score mean (sd) | 5.47 (17.9) | 14.87 (9.8) |

`fightin_words_paired_difference` (defined in the file as source −
candidate) has mean **−9.40** (sd 18.1) across the 104 pairs — i.e. on
average the existing Fightin'-Words/CUE-style score is *higher* for the
"reduced-cue" reworded version than for the original, the opposite
direction implied by the pipeline's own `reduced_cue_source_rewrite`
label. This does not mean the rewording failed at its actual goal
(preserving the harmful objective while changing surface wording,
confirmed by 100% `objective_preserved=yes`); it means the existing
automated CUE-style score and the construction-family label disagree in
direction, on the one resource in this repository that comes closest to
a same-intent cue contrast. Neither the construction label nor the
existing CUE score can be trusted as `C_cue` ground truth here — this is
direct, already-computed evidence for that conclusion, not a new
speculative concern.

### 2.2 `p_tfidf` is confounded with length in a way 3D-C ruled uncorrectable

`logs/3d_c_length_dependence_audit.md` found `Spearman(length, p_tfidf
percentile) = -0.567`, ruled this a **structural property** of raw-count
TF-IDF + (1,2)-gram + cosine-to-centroid on this dataset — "every other
length adjustment... would change what S2/S3 measure... a redesign, not
a correction" — and closed 3D-C with "length dependence status: remains,
unmitigated, by design." `logs/3d_b_lexical_outlierness_pilot.md` also
shows source is a real, if modest, confound on tail membership
(in-sample source-prediction accuracy 0.684 vs. 0.632 majority-class
baseline). Any candidate that reuses this operationalization inherits
both confounds.

### 2.3 3D-H-A is within-harmful only and points in a direction worth noting, not resolving

3D-H-A found `p_tfidf` and `E_harm` **negatively** correlated (rho =
−0.4755, p = 0.006; high-`p_tfidf`-tail mean rating 3.19 vs.
low-tail 4.63): lexically *typical* harmful prompts were rated as
*more* clearly harmful-operational than lexically *atypical* ones. This
is informative only — it is within-harmful, already spent (cannot be
reused as an independent validation set per task §5), and does not by
itself define `C_cue` for either population. It is one more reason not
to treat "lexical outlierness" and "surface cue salience" as
interchangeable.

### 2.4 The repository's own diagnostic already contradicts "XSTest-safe = high cue"

`data/quadrant_composition_report.json`
(`src/diagnostics/quadrant_composition_check.py`) scored all four
quadrants with an existing simple keyword-hit measure (not opened or
modified here; only its already-computed aggregate output is cited) and
recorded, as a predeclared prediction that explicitly **did not hold**:
quadrant A (HarmBench) mean cue-hit rate 0.307 (29.3% of rows with ≥1
hit) vs. quadrant B (XSTest-safe) mean 0.032 (3.2% of rows). This is
already-existing evidence, on record in the repository, against treating
source identity (XSTest-safe) as a high-cue label — exactly the
assumption the task brief prohibits in §4.

### 2.5 An unused same-intent, cue-varying matched-pair resource already exists — harmful side only

`data/quadrant_c_pipeline/candidate_records_v2.jsonl` (155 rows) and its
human-reviewed subset `data/review/c_review_queue.csv` (104 rows, all
`accept`) each hold, per row: `source_prompt` (original, StrongREJECT),
`candidate_prompt`/`scored_prompt` (AI-reworded), `harmful_objective`
(free text, shared across the pair), `pair_id`, `objective_preserved`
(100% `yes` on the 104 accepted rows), `operational_detail_changed`
(100% `yes`), plus the length and Fightin'-Words fields in §2.1. This is
a **human-reviewed, same-underlying-objective, differently-worded pair
set** — precisely the raw material Candidate C (§3.C) requires — and it
already exists, at zero new acquisition cost, for the harmful side.

### 2.6 No equivalent resource exists on the benign side

`data/processed/controlled_eval.jsonl`'s quadrant-B rows (250, all
XSTest-safe; fields: `prompt`, `quadrant`, `source`, `category`, `split`
only) carry no `pair_id`/`source_prompt`/`candidate_prompt` structure.
XSTest's own category taxonomy (`homonyms`, `figurative_language`,
`safe_targets`, `safe_contexts`, `definitions`, etc.) pairs *safe*
prompts against *unsafe* prompts sharing a surface trigger word — i.e.
it varies `I(x)` while holding surface wording roughly fixed, the
opposite of the same-intent, varying-cue pairs this design needs. No
benign-side same-intent cue-varying pair resource currently exists
anywhere in this repository. Building one is explicitly out of scope for
this design-only task (task brief: "do not create candidate datasets");
it is identified here as the one genuinely new artifact any pilot will
require, sized in §6.

---

## 3. Candidate-method comparison

### 3.A Fixed global lexical reference

- **Estimand:** a single scoring function `C_cue(x) = f(x)` from one fixed
  vocabulary/reference distribution, applied identically to all prompts,
  regardless of `I(x)`.
- **Exact score:** the closest existing analogue is `p_tfidf`
  (LOGO TF-IDF centroid cosine distance) or the existing `CUE`
  TF-IDF+LogReg score, generalized to run on all four quadrants rather
  than within-harmful or harmful-vs-benign only.
- **High vs. low:** higher score = more lexically atypical relative to
  the fixed reference (`p_tfidf`-style) or more harmful-associated
  (`CUE`-style) — two different meanings that must not be conflated.
- **Common scale:** technically trivial (one fitted function scores both
  populations) but substantively unresolved: a reference fit on
  harmful-vs-benign labels (as `CUE` is, §2.1) *is* an intent measure by
  construction; a reference fit only on general corpus frequency (not
  this project's harmful/benign corpora) would avoid that but has no
  existing implementation here and still needs a definition of which
  *words* count as "cue" — which the task brief prohibits sourcing from
  a new lexicon.
- **Required human labels:** none to fit; would need the same held-out
  human check as §5 to interpret the resulting score as cue rather than
  something else (genre, formality, register).
- **Split/leakage/source/template/length/topic controls:** would need to
  reproduce 3D-A/3D-B's full LOGO-fold + length-sensitivity + category-
  balanced-sensitivity apparatus in full, since it inherits the same
  raw-count-TF-IDF machinery.
- **Sample size:** the existing 209-row within-harmful pool, or the full
  654-row `controlled_eval.jsonl` if extended across quadrants.
- **Minimum manual effort:** low to build, but §2.1–2.2 already show its
  two closest existing instances (i) don't track the pipeline's own
  cue-reduction labels in the expected direction and (ii) have an
  uncorrectable length confound.
- **Transfer test:** not meaningfully definable without first resolving
  what "cue" means independent of intent — see §2.1.
- **Failure condition already triggered:** yes, in the aggregate (§2.1,
  §2.2). Retained only as a required baseline (§3.5), not viable as
  primary.

### 3.B Human-anchored representation

- **Estimand:** a direction in a frozen embedding/representation space,
  defined by a small set of human-rated examples, such that projection
  onto it gives `C_cue(x)`.
- **Exact score:** projection of an embedding onto the fitted direction
  (e.g. mean-difference-of-embeddings direction between high- and
  low-rated examples, analogous in spirit to this repo's own diff-in-
  means refusal-direction method in `src/interpretability/`, but on a
  *lexical* rather than *activation* representation, and rated for cue
  rather than refusal).
- **High vs. low:** higher projection = more human-judged surface cue
  salience.
- **Common scale:** yes, in principle — one direction, evaluated on both
  populations — provided the anchoring set spans both `I(x)` conditions.
- **Required human labels:** a fresh set of human cue ratings covering
  *both* harmful and benign prompts. The only existing human rating set
  (3D-H-A, n=32) is within-harmful only, rates `E_harm` not `C_cue`, and
  is already spent (task §5). No usable anchoring set currently exists.
- **Split/leakage/controls:** would need a frozen, audited embedding
  space (none is currently chosen or audited in this repo for this
  purpose — choosing one is itself a design decision requiring
  justification, out of scope for a design-only task) plus the same
  source/length/topic controls as 3.C below.
- **Sample size / minimum manual effort:** the anchoring set alone would
  need to be larger than 3D-H-A's 32 (which covered one population) to
  cover both populations with any power — a materially bigger new-
  annotation cost than 3.C, for a method whose only structural advantage
  over 3.C (a chosen embedding geometry) is not otherwise required by
  the research question.
- **Transfer test:** fit direction on harmful-only-rated examples,
  evaluate on benign-only-rated examples (or vice versa) — well-defined,
  but blocked on the missing anchoring data above.
- **Failure condition:** no defensible anchoring set currently exists
  without new data collection larger than what 3.C needs for the same
  purpose.
- **Disposition:** not recommended as primary. Its one valuable idea —
  calibrating the axis against direct human judgment rather than an
  automated proxy — is retained inside the recommended method (§4) as a
  small ordering check, without committing to a new embedding-space
  choice.

### 3.C Pairwise / matched contrast

- **Estimand:** whether a cue-difference signal, learned from pairs that
  share `I(x)` and objective but differ in surface wording, transfers to
  the other `I(x)` condition.
- **Exact score:** a 1-D pairwise ranking score (e.g. Bradley-Terry-style
  logistic regression over paired feature differences: length, an
  audited lexical-difference feature, and/or a blind human ordering
  judgment — never `p_tfidf` tails per task §3.C instruction), fit so
  that `f(source_prompt) > f(candidate_prompt)` is the target ordering
  where humans confirm it holds.
- **High vs. low:** higher score = judged (by the fitted ranker) as more
  operationally cue-explicit, on a scale calibrated by human pairwise
  judgments, not by construction-family labels alone (§2.1 shows why).
- **Common scale:** yes, structurally — the ranker is fit once, on
  features computable for any prompt, harmful or benign; only the
  *training pairs* are harmful-only, which is exactly what the transfer
  test (§7) is designed to check.
- **Required human labels:** a blind pairwise ordering judgment per pair
  (not a 1–5 scale on a single prompt) — see §5–§6 for the proposed
  instrument and size.
- **Split:** fit on harmful-side pairs only; evaluate on benign-side
  pairs only (held out, never used in fitting) — this *is* the transfer
  test, not a separate step.
- **Leakage prevention:** raters never see which side of a pair is
  `source_prompt` vs. `candidate_prompt`, nor the pair's
  `transformation_family` or `pair_id`; left/right position randomized
  per rater per pair.
- **Source/template controls:** pairs are matched *within* a `pair_id`,
  so source and template are identical on both sides of every
  comparison by construction — this is the structural advantage over
  3.A/3.B, which compare *across* templates/sources.
- **Length controls:** length difference is a required predictor to
  audit and, if it dominates, to explicitly flag (§2.1's own numbers
  show source/candidate differ sharply in mean length and length
  variance) — mirroring the 3D-C length-partial-correlation convention
  already established in this repo.
- **Topic/category controls:** `project_category`/`harmful_objective`
  are already recorded per pair in the existing resource and can be used
  to check the ranking isn't concentrated in one category.
- **Required sample size:** harmful side free (104 existing accepted
  pairs, or up to 155 if the un-reviewed 51 are separately screened);
  benign side needs new construction, sized in §6.
- **Minimum manual effort:** smallest of all four candidates — reuses
  104 already-human-reviewed harmful pairs, needs only (i) a small blind
  ordering check on a sample of them and (ii) a small new benign
  matched-pair set for the transfer test.
- **Transfer test harmful → benign:** primary test, §7.
- **Transfer test benign → harmful:** symmetric check once (ii) above
  exists, useful as a robustness check but not required for a first
  pilot decision.
- **Failure conditions:** (a) blind raters do not, on average, confirm
  the pipeline's own intended `source > candidate` cue ordering on the
  existing harmful pairs (§8); (b) no defensible benign matched pairs can
  be constructed (§2.6, §8); (c) a fitted ranking does not transfer above
  chance, or a length-only baseline reproduces the same result (§7–§8).

### 3.D Supervised contrastive learning

- **Estimand:** a learned embedding space in which cue-positive and
  cue-negative examples are pulled apart, using the same conceptual
  cue-positive/cue-negative relations as 3.C but trained end-to-end
  rather than scored with a 1-D ranker.
- **Required data:** substantially more matched relations than a simple
  ranker to avoid overfitting/circularity — well beyond the 104 harmful
  pairs (and 0 benign pairs) currently available.
- **Disposition:** deferred, per the task brief's own preference ("a
  simple one-dimensional pairwise ranking model before a full neural
  contrastive system") and per data availability (§2.5–§2.6). Revisit
  only if 3.C's pilot (§5) validates the underlying same-intent-pair
  approach *and* a materially larger matched-pair pool is built in a
  later, separately-authorized task. Not scored against the full
  requirement list below since it is not a candidate for the next pilot.

### 3.5 Required baselines

| Baseline | Definition | Status |
|---|---|---|
| Source-only | predict cue tail from source dataset alone | Already run for `p_tfidf` in 3D-B: 0.684 in-sample accuracy vs. 0.632 majority baseline — real but modest. Must be re-run for whatever score 3.C's pilot produces. |
| Length-only | predict cue tail from token/character count alone | Already shown dominant for `p_tfidf` (§2.2); required for 3.C's ranker given §2.1's raw length gap between `source_prompt`/`candidate_prompt`. |
| Formatting/style-only | list markers, numbered steps, code-block delimiters, multi-sentence structure | 3D-B's formatting diagnostics (regexes logged in its JSON output) are directly reusable. |
| Simple TF-IDF | plain TF-IDF cosine or logistic-regression score, no LOGO/leakage controls | `CUE`'s TF-IDF+LogReg component, or a de-novo simple fit; required as the "why not just use the obvious thing" comparison per task §6. |
| Random-direction | a random unit vector / random pair ordering, same evaluation pipeline | Standard null; mirrors the LoRA-subspace check's own random-direction baseline already used elsewhere in this repo (`HANDOFF.md`), so the convention is precedented. |

---

## 4. Recommended primary method

**Candidate 3.C (pairwise / matched contrast), calibrated by a small
blind human ordering check rather than by construction-family labels or
the existing `CUE` score.**

Against the task's own preference order:

1. **Closest to the original scientific question** — holding `I(x)`
   fixed within each pair is the only structural way to isolate
   `C_cue(x)` from `I(x)`; 3.A and 3.B compare across sources/templates
   and inherit exactly the confounds §2.1–§2.4 already document.
2. **Genuinely common across harmful and benign** — the scoring function
   itself is population-agnostic; what's missing is benign-side
   *training* pairs, and the design here does not require them for
   fitting — only for the held-out transfer test (§7), which is the
   correct place for benign data to enter.
3. **Least dependent on source labels** — cue is defined by within-pair
   human ordering, not by which dataset or `transformation_family` a row
   carries (§2.1 shows why source/construction labels alone cannot be
   trusted here).
4. **Minimal manual annotation** — reuses 104 already-human-reviewed
   pairs at zero new cost; §6 sizes the only new annotation required.
5. **Simple enough to audit** — a 1-D pairwise ranking model over a
   small, inspectable feature set, not a new embedding space or a neural
   contrastive system.
6. **Allows held-out cross-intent transfer** — this is the explicit
   design of §5, not an afterthought.

3.A is rejected as primary because its two closest existing instances
already fail in ways this repository has itself already measured
(§2.1 direction mismatch, §2.2 uncorrectable length confound). 3.B is
rejected as primary because its required anchoring data does not exist
and would cost more new annotation than 3.C for no offsetting structural
advantage the research question actually needs. 3.D is deferred per the
task brief's own stated preference and current data volume.

This is not a forced positive recommendation: §5's Step 1 is itself a
falsification test of the one resource this recommendation leans on, and
§8 states exactly what result would produce a NO-GO instead.

---

## 5. Exact minimal pilot (proposed scope for a future 3F-B; **not started by this task**)

1. **Harmful-side blind ordering check (no new prompt data).** Draw a
   stratified blind sample of ~24–30 pairs from the 104 accepted
   `c_review_queue.csv` rows (stratified by `project_category` so no
   single harm area dominates). For each pair, a rater sees the two
   prompts in random left/right position with no `source`/`candidate`
   labeling and answers a forced-choice/scaled comparative question
   ("which prompt states the underlying request more explicitly /
   operationally?"), adapting 3D-H's existing single-item rating
   instructions (`logs/3d_h_rating_instructions.md`) to a paired format.
   This is a *fresh* sample — the c_paired population was never touched
   by 3D-H's construct check (which used the separate, unpaired
   `c_source_authored` population), so this does not violate the
   "don't reuse the same 32" rule.
2. **New benign matched pairs (the one genuinely new artifact; not
   built by this task).** Construct, in a later task, a similarly sized
   set (~24–30) of benign same-objective pairs — one more surface-
   explicit phrasing, one softer/less trigger-worded phrasing of the
   *same* benign request — reviewed by a researcher against the same
   `objective_preserved`/`operational_detail_changed` fields the harmful
   pipeline already uses, so the two populations are schema-compatible.
3. **Blind ordering check, benign side.** Same instrument as Step 1,
   applied to the new benign pairs.
4. **Fit.** A 1-D pairwise ranking model (logistic regression on paired
   feature differences: length, an audited lexical-difference feature,
   category) using only the harmful-side pairs and Step 1's blind labels
   (falling back to the construction label only if Step 1 confirms it at
   the pool level — see §8).
5. **Held-out transfer evaluation.** Apply the fitted ranker to the
   benign pairs; compare its implied ordering against Step 3's blind
   benign labels. This is the decisive test (§7).

Not part of this pilot: any B/D quadrant benchmark construction, any
embedding/contrastive model, any new lexicon search, any change to
`src/cue_scoring.py` or the frozen benchmark.

---

## 6. Human annotation requirement

| Item | New human effort |
|---|---|
| Step 1 (harmful blind ordering) | ~24–30 pairwise ratings, reusing existing pair text |
| Step 2 (benign pair construction + review) | ~24–30 new pairs written + researcher review against `objective_preserved`/`operational_detail_changed` |
| Step 3 (benign blind ordering) | ~24–30 pairwise ratings |

Total: on the order of 50–60 new pairwise ratings plus ~24–30 new pair-
construction/review actions — comparable in scale to 3D-H-A's existing
32-item check, not a larger commitment, and an order of magnitude below
what 3.B or 3.D would require for the same purpose.

---

## 7. Transfer test — decision criteria

**Supports the common-axis hypothesis:** a ranking direction fit on
harmful-side pairs alone (Step 4) predicts the benign-side blind human
ordering (Step 3) at above-chance concordance (e.g. pairwise accuracy or
Kendall's tau against blind labels, evaluated against a permutation-null
distribution, mirroring 3D-H-A's own permutation-test method already
used in this repo), and that result survives with length partialed out
(mirroring 3D-C's length-sensitivity convention) and is not reproduced by
the length-only or random-direction baselines (§3.5) within the same
confidence interval.

**Does not support it — falsification, not merely "inconclusive":** the
fitted direction fails to beat chance on the benign transfer set, or a
length-only baseline reproduces the same apparent effect. Per task §9,
simple separation between two datasets (or two sides of a pair) is
explicitly *not* sufficient evidence on its own — transfer under
controls is the bar.

---

## 8. Explicit failure / stop conditions

- If Step 1 does not confirm, at the pool level, that blind raters judge
  `source_prompt` as more operationally cue-explicit than
  `candidate_prompt` on average, the `c_review_queue.csv` construction-
  family label (`reduced_cue_source_rewrite`) cannot be used as a cue-
  direction proxy as currently labeled. Recommendation in that case: use
  only the fresh blind labels from Step 1 as ground truth going forward,
  not the construction label — a redesign of the *labeling*, not
  necessarily of the whole method.
- If no benign pairs can be constructed that a researcher confirms as
  `objective_preserved=yes` (i.e. no defensible same-intent benign cue
  contrast exists), report that explicitly, per task §4/§7, rather than
  fabricating pairs from source-identity or template membership.
- If the held-out transfer test (§7) does not clear the predeclared
  above-chance bar, or is matched by the length-only baseline, the
  correct output is an explicit NO-GO / redesign recommendation for a
  common-axis construction from this data, not a forced positive.

---

## 9. Summary

- **Recommended method:** pairwise/matched contrast (3.C), calibrated by
  blind human ordering judgments, not by the existing `CUE` score or by
  construction-family labels (§2.1 shows both are unreliable here).
- **Why:** the only candidate that holds `I(x)` fixed by construction,
  reuses an already-existing, already-human-reviewed 104-pair harmful
  resource at zero new cost, and has a well-defined held-out
  cross-intent transfer test.
- **Minimum data/human effort:** ~50–60 new pairwise ratings + ~24–30
  new benign matched pairs (§6) — smaller than any alternative that
  meets the task's own requirements.
- **Exact next step:** a separately-authorized 3F-B task to run §5's
  five steps and report the §7 transfer result; no B/D construction, no
  benchmark change, and no scorer implementation until that result is in.
