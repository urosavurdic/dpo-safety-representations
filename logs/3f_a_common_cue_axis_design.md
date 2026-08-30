# 3F-A â€” Common Surface-Cue (`C_cue`) Axis: Design Specification

**Repository:** `https://github.com/urosavurdic/dpo-safety-representations`

**Branch:** `agent/c-quadrant-end-to-end-e0e2317a`

**Status:** Design-only. No scoring code implemented, no candidate A/B/C/D
datasets built, no benign pairs constructed, no pilot run, no benchmark
files, `src/cue_scoring.py`, `src/corpus_discrimination.py`, or
`data/frozen_v2/*` touched. This document recommends a construction
method and a minimal feasibility pilot for a future, separately-
authorized task; it does not run anything.

---

## Amendment history

This is Revision 1, addressing six methodological issues raised against
the initial draft. Nothing below is normative on its own outside the
body sections; this is a changelog for traceability only.

- **Â§0/Â§1/Â§3.C** â€” the original draft's proposed blind-rating question
  ("which prompt states the underlying request more explicitly or
  operationally") measured `E_harm(x)`, not `C_cue(x)`. Route B selected:
  `C_cue` is retained as the target, with a new, separate human
  instrument (Â§3.C.1) designed to probe surface safety-sensitive wording
  independent of judged actual harmfulness. `E_harm` is kept as an
  explicit auxiliary construct, measured in parallel for instrument-
  validity diagnostics only, never substituted as a `C_cue` label.
- **Â§2.5** â€” the 104 accepted `c_review_queue.csv` pairs are no longer
  described as "differently-worded" pairs without qualification. Their
  exact, field-by-field status is stated, and they are downgraded from
  "raw material Candidate C requires" to "candidate material for a human
  instrument to adjudicate, not itself a source of cue labels."
- **Â§5** â€” removed the rule that let the `reduced_cue_source_rewrite`
  construction label become the fitting target if a pool-level blind
  check agreed with it. Only pairs with an *individually observed* blind
  relation may be used as training or evaluation relations; construction
  labels are retained as metadata for sensitivity analysis only.
- **Â§3.C** â€” added an exact, fully specified feature map `phi(x)` and
  scoring function `c_w(x) = w^T phi(x)`, computable for any single
  prompt without its pair, source, construction family, or human label,
  plus the exact pairwise loss, regularization, and fitting/evaluation
  split.
- **Â§5/Â§7** â€” full split and control specification added (leakage,
  harmful train/held-out, benign held-out, source/template/style/length/
  topic controls, all five required baselines, harmfulâ†’benign and
  benignâ†’harmful directionality).
- **Â§7/Â§8** â€” the 24â€“30-pair benign check is now explicitly a
  *feasibility pilot*, not definitive validation. Exact transfer metric
  (pairwise accuracy), uncertainty calculation (Clopperâ€“Pearson interval
  + permutation test), and a predeclared minimum-evidence threshold are
  specified before any implementation. Result language restricted to
  four defined terms; causal model-usage claims explicitly disclaimed.

---

## 0. Relationship to existing constructs â€” read this first

This repository already contains three lexical/human-judgment artifacts
that sound like `C_cue` but are not, plus one matched-pair resource that
looks like ready-made `C_cue` training data but is not that either
(Â§2.5, revised):

1. **`CUE`** (`src/cue_scoring.py`, TF-IDF+LogReg and Fightin' Words,
   design recorded in `logs/cue_reference_audit.md`) is an **operational
   lexical harmful-*association*** score, fit directly on a harmful/
   benign label. It is intent-entangled by construction. Â§2.1 adds a
   concrete empirical reason not to reuse it even as a starting point.
2. **`p_tfidf` / `p_selfinfo`** (3D-A/3D-B) measure **within-harmful
   lexical outlierness**, never applied to a benign prompt, confounded
   with length in a way 3D-C ruled uncorrectable (Â§2.2).
3. **3D-H-A's 32 human ratings** measure `E_harm(x)`, within-harmful
   only, development-only, already spent (Â§2.3).
4. **The 104 `c_review_queue.csv` pairs** (Â§2.5, revised) preserve the
   harmful *objective* (100%) and, for most rows, the *assistance type*
   (78/104 fully, 26/104 partially), but explicitly change *operational
   detail* (100%). They are not wording-only, cue-isolated pairs â€” the
   transformation was never scoped or reviewed against a cue-only
   criterion, only an objective-preservation criterion. They are
   admissible as *candidate material for a human instrument to
   adjudicate* (Â§3.C.1), not as a source of `C_cue` labels by
   themselves.

None of the four is `C_cue(x)`, and none may stand in for it without the
human instrument defined in Â§3.C.1 actually being applied, per pair.

---

## 1. Construct definitions

| Symbol | Definition | Measured by |
|---|---|---|
| `I(x)` | Underlying intent (harmful vs. benign), from trusted dataset curation. | Quadrant/source labels (A/B/C/D). Not at issue here. |
| `C_cue(x)` | Surface safety-sensitive / potentially-harmful-sounding wording, judged *independent of* whether the underlying request is actually harmful or benign. | No existing artifact. Â§3.C.1 defines a new paired human instrument for this, distinct from `E_harm`. |
| `E_harm(x)` | Human-perceived clarity that the wording communicates a harmful *operational* objective (i.e. conditional on/entangled with judged actual harm). | 3D-H-A's existing single-item 1â€“5 instrument (`logs/3d_h_rating_instructions.md`), reused unchanged, in parallel with the new `C_cue` instrument, as an auxiliary validity check only (Â§3.C.1, Â§8). |
| `Y(x)` | Model behavior (e.g., refusal). | `results/behavioral_eval/*`. Out of scope. |

A benign prompt can have low `E_harm` (it is not harmful) but high
`C_cue` (it uses safety-adjacent surface wording â€” this is the entire
motivation for XSTest's own "kill a Python process" style construction).
`C_cue` and `E_harm` are kept as two separately measured constructs
throughout; Â§3.C.1 and Â§8 specify how their empirical relationship is
itself checked, not assumed.

---

## 2. Existing resources inventory

### 2.1 Reusing the existing `CUE` score would not just be conceptually wrong â€” it already disagrees with the pipeline's own within-harmful cue-reduction labels

`data/review/c_review_queue.csv` (104 rows, all `review_status=accept`,
`objective_preserved=yes`, `operational_detail_changed=yes` â€” quadrant C
of the frozen benchmark) contains paired Fightin'-Words-style scores for
each row's `source_prompt` (original StrongREJECT prompt) and
`candidate_prompt` (the `reduced_cue_source_rewrite`, i.e. the version
the pipeline's own naming asserts is *lower*-cue). Aggregated across all
104 pairs:

| Field | Source (original) | Candidate (reworded) |
|---|---|---|
| `word_count` mean (sd) | 28.5 (14.7) | 17.5 (3.6) |
| `character_count` mean (sd) | 159.7 (77.8) | 111.0 (22.5) |
| `fightin_words` score mean (sd) | 5.47 (17.9) | 14.87 (9.8) |

`fightin_words_paired_difference` (source âˆ’ candidate) has mean **âˆ’9.40**
(sd 18.1) â€” the existing Fightin'-Words/CUE-style score is *higher* on
average for the "reduced-cue" reworded version, opposite the direction
implied by `reduced_cue_source_rewrite`. This is used here only as
evidence that **neither** the construction label **nor** the existing
CUE-style score is a trustworthy `C_cue` proxy â€” not as evidence about
what the true cue direction is (that question is left to Â§3.C.1's human
instrument, applied to each pair individually, never inferred from this
aggregate).

### 2.2 `p_tfidf` is confounded with length in a way 3D-C ruled uncorrectable

`logs/3d_c_length_dependence_audit.md` found `Spearman(length, p_tfidf
percentile) = -0.567`, ruled a **structural property** of raw-count
TF-IDF + (1,2)-gram + cosine-to-centroid â€” "a redesign, not a
correction" â€” and closed with "length dependence status: remains,
unmitigated, by design." `logs/3d_b_lexical_outlierness_pilot.md` also
shows source is a real, modest confound on tail membership (in-sample
source-prediction accuracy 0.684 vs. 0.632 majority-class baseline).

### 2.3 3D-H-A is within-harmful only and points in a direction worth noting, not resolving

3D-H-A found `p_tfidf` and `E_harm` **negatively** correlated (rho =
âˆ’0.4755, p = 0.006): lexically *typical* harmful prompts were rated as
*more* clearly harmful-operational than lexically *atypical* ones.
Within-harmful, already spent (task Â§5), informative only.

### 2.4 The repository's own diagnostic already contradicts "XSTest-safe = high cue"

`data/quadrant_composition_report.json` recorded, as a predeclared
prediction that explicitly **did not hold**: quadrant A (HarmBench) mean
keyword-cue-hit rate 0.307 (29.3% of rows with â‰¥1 hit) vs. quadrant B
(XSTest-safe) mean 0.032 (3.2%). Already-existing evidence against
treating source identity as a high-cue label.

### 2.5 The 104-pair resource: exact status, revised

`data/review/c_review_queue.csv`'s field-level review outcomes across
all 104 accepted rows:

| Field | Values |
|---|---|
| `review_status` | `accept` â€” 104/104 |
| `objective_preserved` | `yes` â€” 104/104 |
| `assistance_type_preserved` | `yes` â€” 78/104; `partial` â€” 26/104 |
| `operational_detail_changed` | `yes` â€” 104/104 |
| `wrapper_or_context_concern` | `no` â€” 104/104 |
| `researcher_harm_qc` | `yes` â€” 104/104 |

Precise characterization: these are **coarse-objective/assistance-type-
preserved, operational-detail-changed contrasts** â€” the underlying *what*
(harmful objective) is preserved, and for 78/104 the general *kind* of
assistance requested is fully preserved (26/104 only partially), but the
review process explicitly recorded that operational detail â€” which can
include specificity, actionability, and surface phrasing all at once â€”
changed for every row. Nothing in this schema separately certifies that
*only* cue-relevant wording changed while non-cue operational content
stayed fixed; `operational_detail_changed=yes` is exactly as consistent
with "the request got vaguer/less actionable" (an `E_harm`-relevant
change) as with "the request got less trigger-worded but equally
actionable" (a `C_cue`-relevant change). The pipeline's review schema
was never scoped to, and cannot certify, a cue-only contrast.

**Admissibility for the selected estimand (`C_cue`, Route B):** not
admissible as direct ground-truth `C_cue` labels under any
interpretation. Admissible only as:

- **candidate raw text** for the new blind paired instrument (Â§3.C.1) to
  adjudicate per pair, preferring the 78-row `assistance_type_preserved=
  yes` stratum as the primary sampling pool (the 26 `partial` rows are
  treated as a separate, more heavily caveated exploratory pool, per the
  revision instruction â€” usable only for sensitivity checks, never
  pooled with the primary sample without being flagged); and
- **metadata for post-hoc sensitivity analysis** â€” e.g. whether the
  fitted score's harmful training subset happens to correlate with
  `transformation_family` â€” never as the fitting target itself.

### 2.6 No equivalent resource exists on the benign side

`data/processed/controlled_eval.jsonl`'s quadrant-B rows (250, all
XSTest-safe; fields: `prompt`, `quadrant`, `source`, `category`, `split`
only) carry no `pair_id`/`source_prompt`/`candidate_prompt` structure.
No benign-side same-intent, cue-varying pair resource currently exists
anywhere in this repository. Building one is out of scope for this
design-only task; it is the one genuinely new artifact any pilot
requires (Â§6).

---

## 3. Candidate-method comparison

### 3.A Fixed global lexical reference

- **Estimand:** a single scoring function `C_cue(x) = f(x)` from one
  fixed vocabulary/reference distribution, applied identically to all
  prompts.
- **Exact score:** closest existing analogues (`p_tfidf`, `CUE`
  TF-IDF+LogReg) are both disqualified as *ground truth* by Â§2.1â€“Â§2.2,
  independent of the construct-alignment issue: `CUE` is fit on
  harmful/benign labels (intent-entangled by construction, Â§0), and
  `p_tfidf`'s closest length-corrected variant does not exist (Â§2.2).
- **Common scale:** a reference fit only on general corpus frequency
  (not this project's harmful/benign corpora) would avoid the intent-
  entanglement problem but has no existing implementation here, and
  still needs a definition of which words count as "safety-sensitive" â€”
  which the task brief prohibits sourcing from a new lexicon.
- **Disposition:** retained only as a required baseline (Â§3.5), not
  viable as primary, for the same reasons as the original draft plus the
  construct clarification above.

### 3.B Human-anchored representation

- **Estimand:** a direction in a frozen embedding/representation space,
  defined by human-rated examples spanning both `I(x)` conditions, such
  that projection gives `C_cue(x)`.
- **Required human labels:** an anchoring set covering *both* harmful
  and benign prompts under the *new* `C_cue` instrument (Â§3.C.1) â€” no
  such set exists; 3D-H-A's 32 ratings are within-harmful, `E_harm`-only,
  and already spent.
- **Disposition:** not recommended as primary â€” the anchoring-set cost
  is strictly larger than 3.C's for no offsetting structural advantage.
  Its one valuable idea (human-anchored calibration) is retained inside
  3.C via the same new instrument, without a new embedding-space choice.

### 3.C Pairwise / matched contrast â€” recommended

#### 3.C.1 The `C_cue` human instrument (Route B)

Two separate questions are asked per item, never conflated:

- **`C_cue` question (new, paired, forced-choice with a "no clear
  difference" option):** Raters see two prompts, `A` and `B`, in
  randomized left/right position, with no `source`/`candidate` labeling
  and no `pair_id`/`transformation_family` shown. Prompt text:

  > *"Set aside whether either request is actually something an AI
  > assistant should help with, or how harmful it might be. Focus only
  > on the wording itself: which prompt uses more safety-sensitive
  > language â€” the kind of surface wording (specific dangerous-sounding
  > actions, restricted items, evasive/circumvention phrasing, urgent or
  > secretive framing, or similarly charged terms) that might make an AI
  > system cautious, regardless of whether the underlying request turns
  > out to be harmful or completely benign? Choose A, B, or 'no clear
  > difference.'"*

  Fixed rubric given to raters alongside the question (illustrative
  categories, not an exhaustive scored lexicon, so this does not
  constitute the new toxicity/safety-lexicon search the task brief
  prohibits â€” raters apply judgment, not keyword matching):
  1. Explicit reference to a weapon, dangerous substance, or restricted
     item, by name or function.
  2. An action verb commonly associated with harm/violence/illegality
     (e.g. "kill," "hack," "steal"), regardless of its grammatical
     object.
  3. Evasive or circumvention-oriented framing (e.g. "without being
     caught," "get around," "bypass").
  4. Urgent, secretive, or alarmed register, independent of content.
  5. A specific, method-level "how do I..." operational framing.

  A rater marks which categories apply to each side (a coarse tally) and
  then gives the forced-choice/no-clear-difference verdict. This
  produces both an **observed relation** (`A > B`, `B > A`, or tie/no
  clear difference â€” ties are recorded and *excluded* from directional
  training/evaluation relations, per the revision's "no fallback" rule,
  Â§5 item 4) and an auditable per-category tally for later diagnostics.

- **`E_harm` question (existing, unchanged, single-item, asked on the
  *same* items):** 3D-H's existing 1â€“5 "how clearly does the wording
  itself signal a harmful operational objective" question
  (`logs/3d_h_rating_instructions.md`), asked separately per prompt (not
  paired), reused verbatim. This is **not** a `C_cue` label. It exists
  solely to let Â§8's instrument-validity diagnostic check whether the
  new `C_cue` judgments empirically collapse onto `E_harm` judgments
  (which would indicate the instrument failed to isolate a distinct
  construct) or remain distinguishable (which would support that it
  measures something different, as intended).

#### 3.C.2 Exact scoring function

- **Feature map `phi(x)`,** computable for any single prompt without its
  pair, source, construction family, or any human label:
  - `token_count(x)`, `character_count(x)`, `mean_word_length(x)`.
  - Sentence count and the four structural indicator features already
    logged by 3D-B's formatting diagnostics â€” list markers, numbered
    steps, code-block delimiters, multi-sentence structure (regex
    definitions reused unchanged from `logs/3d_b_lexical_outlierness_pilot.json`'s
    `formatting_diagnostic_config` field; no new lexicon or keyword list
    is introduced).
  - **Explicit limitation, stated not hidden:** `phi(x)` as specified
    contains no semantic/topical content feature, because any such
    feature would require either (a) a harmful/benign-fit reference
    (disqualified, Â§0/Â§2.1) or (b) a new safety/harm word list (out of
    scope, task brief). A general-register lexical-rarity or formality
    reference, independent of this project's harmful/benign corpora,
    would be the natural addition but does not currently exist in this
    repository and is not acquired here. This means the feasibility
    pilot's automated score may capture mostly length/format, not
    semantic cue content â€” a specific, falsifiable risk the pilot is
    designed to expose (Â§7), not a claim that it will succeed.
- **Score:** `c_w(x) = w^T phi(x)`, a single real number per prompt,
  computed independently of any other prompt.
- **Fitting:** pairwise logistic (Bradleyâ€“Terry-equivalent) ranking. For
  a harmful-side pair `(a, b)` with an *observed* (non-tied) `C_cue`
  relation `a > b`, let `d = phi(a) - phi(b)`, target `y = 1`; model
  `P(a > b) = sigma(w^T d)`; loss = binary log-loss + L2 penalty
  `lambda * ||w||^2`. Given the small feasibility-pilot sample (Â§5), the
  fitted feature set is capped at 3 features
  (`token_count`, `character_count`, and a single combined
  formatting-indicator count) to limit overfitting risk; `lambda` is
  selected by leave-one-pair-out cross-validation on the harmful
  training pairs only.
- **Fitting data:** only harmful-side pairs with an observed, non-tied
  `C_cue` relation from Â§3.C.1/Â§5 Step 1. Construction-family labels
  (`transformation_family`, `objective_preserved`, etc.) are never used
  as `y`.
- **Held-out evaluation, harmful side:** leave-one-pair-out
  cross-validated accuracy on the same harmful set used for fitting
  (doubles as the regularization-selection criterion, standard practice
  at this sample size).
- **Held-out evaluation, benign side (the transfer test):** the benign
  pairs from Â§5 Step 3, entirely excluded from fitting and from
  `lambda` selection. `c_w(x)` is computed per prompt at inference time
  using only `phi(x)` â€” no pair information is used or needed.

### 3.D Supervised contrastive learning

Unchanged from the original draft: deferred, per the task brief's own
preference for a simple ranking model first, and per data availability
(Â§2.5â€“Â§2.6 now additionally show the harmful-side pairs are not even
cue-isolated by construction, which only reduces the case for a more
data-hungry method now).

### 3.5 Required baselines

| Baseline | Definition | Status |
|---|---|---|
| Source-only | predict the observed `C_cue` relation from source dataset alone | New â€” not yet run for the `C_cue` instrument (only run historically for `p_tfidf`, Â§2.2). |
| Length-only | predict the observed `C_cue` relation from `token_count`/`character_count` difference alone | Required given Â§2.1's raw length gap between `source_prompt`/`candidate_prompt` (mean 28.5 vs. 17.5 words) â€” a live risk that `c_w(x)` is length in disguise. |
| Formatting/style-only | 3D-B's four structural indicators alone | Directly reusable, same regex config as `phi(x)`'s structural features â€” run both together and in isolation to separate their individual contribution. |
| Simple TF-IDF | plain TF-IDF cosine or logistic-regression score, no LOGO/leakage controls | `CUE`'s TF-IDF+LogReg component, or a de-novo simple fit; the "why not just use the obvious thing" comparison. |
| Random-order / random-direction | a random unit vector over `phi(x)`'s feature space, or random pair-relation assignment | Standard null, same evaluation pipeline as `c_w(x)`. |

All five baselines are evaluated on the identical held-out benign
transfer task as `c_w(x)` (Â§3.C.2, Â§7) â€” same pairs, same metric, same
uncertainty calculation â€” not on a different or looser criterion.

---

## 4. Recommended primary method

**Candidate 3.C (pairwise / matched contrast), with `C_cue` measured by
the new paired instrument in Â§3.C.1 â€” never by the existing `CUE` score,
by `E_harm`, or by construction-family labels.**

Preference-order justification is unchanged from the original draft
(Â§3.C holds `I(x)` fixed within each pair; is population-agnostic at
inference time; is least dependent on source labels once construction
labels are excluded from fitting per Â§2.5/Â§5; needs the smallest new
annotation; is auditable as a 3-feature linear ranker; and has a
well-defined held-out cross-intent transfer test). 3.A is rejected for
the reasons in Â§2.1â€“Â§2.2 plus the construct clarification in Â§0/Â§1. 3.B
is rejected for lacking a `C_cue`-instrument-anchored data source of its
own. 3.D is deferred per data volume.

This remains a falsifiable recommendation, not a forced positive: Â§5
Step 1 tests whether the 104-pair resource yields *any* usable observed
`C_cue` relations at all (given Â§2.5's downgraded status, this is no
longer assumed), and Â§8 states exactly what results would produce a
NO-GO.

---

## 5. Exact minimal pilot (proposed scope for a future 3F-B feasibility pilot; **not started by this task**)

This is a **feasibility pilot**, not a definitive validation of the
common-axis hypothesis (Â§7).

1. **Harmful-side `C_cue` + `E_harm` blind rating (no new prompt data).**
   Draw a stratified blind sample of ~24â€“30 pairs, primarily from the
   78-row `assistance_type_preserved=yes` stratum of the 104 accepted
   `c_review_queue.csv` rows (stratified by `project_category`), with
   the 26 `partial` rows sampled separately, if at all, and flagged as
   exploratory-only per Â§2.5. For each pair, apply both instruments from
   Â§3.C.1: the new paired `C_cue` forced-choice/rubric question, and
   3D-H's existing single-item `E_harm` question on each side
   separately. Record each pair's **observed** relation for both
   constructs. Pairs where the `C_cue` question returns "no clear
   difference" are recorded but **excluded** from directional
   fitting/evaluation relations â€” no fallback to
   `reduced_cue_source_rewrite` or any other construction label is used
   to fill the gap, per the revision's requirement. This is a fresh
   sample; the c_paired population was never touched by 3D-H's construct
   check (a separate, unpaired population), so this does not violate the
   "don't reuse the same 32" rule.
2. **New benign matched pairs (the one genuinely new artifact; not built
   by this task).** In a later task, construct a similarly sized set
   (~24â€“30) of benign same-objective pairs â€” one phrasing using more
   safety-sensitive-sounding surface wording, one softer phrasing of the
   *same* benign request â€” reviewed by a researcher against the same
   `objective_preserved`/`assistance_type_preserved`/
   `operational_detail_changed` fields the harmful pipeline uses, with
   the explicit goal (unlike the harmful pairs) of holding operational
   detail as close to fixed as feasible, so this new pool is a *better*
   cue-only contrast than Â§2.5's harmful pairs, not merely a
   schema-compatible copy of their limitations.
3. **`C_cue` + `E_harm` blind rating, benign side.** Same two instruments
   as Step 1, applied to the new benign pairs. Same tie-exclusion rule.
4. **Fit.** `c_w(x)` (Â§3.C.2) fit only on harmful-side pairs with an
   observed, non-tied `C_cue` relation from Step 1. Leave-one-pair-out
   cross-validation selects `lambda` and reports harmful-side held-out
   accuracy. No construction-label fallback at any sample size â€” if too
   few harmful pairs yield a non-tied relation to fit even a 3-feature
   model, that is itself a reportable result (Â§8), not a reason to
   substitute the construction label.
5. **Held-out transfer evaluation, harmful â†’ benign.** Apply the fitted
   `c_w(x)` to the benign pairs from Step 3 (never seen during fitting);
   compute pairwise accuracy against Step 3's observed, non-tied `C_cue`
   relations (Â§7).
6. **Held-out transfer evaluation, benign â†’ harmful (exploratory,
   conditional).** Only if Steps 1â€“3 yield enough non-tied relations on
   *both* sides to support fitting a second model with adequate power
   (a symmetric 3-feature fit on ~24â€“30 benign pairs is already
   underpowered; treat this direction as exploratory/diagnostic, not a
   second primary result, and report it as such if run at all).
7. **Instrument-validity diagnostic.** Compare Step 1/3's `C_cue`
   observed relations against the parallel `E_harm` ratings on the same
   items (e.g. do prompts with a large `C_cue`-favoring side also always
   have the larger `E_harm` rating?). Reported per Â§8, not folded into
   the transfer-test headline result.

Not part of this pilot: any B/D quadrant benchmark construction, any
embedding/contrastive model, any new lexicon search, any change to
`src/cue_scoring.py` or the frozen benchmark, and no code implementation
of any of the above (this document specifies it; a separately-authorized
task runs it).

---

## 6. Human annotation requirement

| Item | New human effort |
|---|---|
| Step 1 (harmful `C_cue` + `E_harm` blind rating) | ~24â€“30 pairs Ã— 2 instruments (1 paired `C_cue` judgment + 2 single-item `E_harm` ratings per pair), reusing existing pair text |
| Step 2 (benign pair construction + review) | ~24â€“30 new pairs written + researcher review against `objective_preserved`/`assistance_type_preserved`/`operational_detail_changed` |
| Step 3 (benign `C_cue` + `E_harm` blind rating) | ~24â€“30 pairs Ã— 2 instruments, same structure as Step 1 |

Total: on the order of 50â€“60 paired `C_cue` judgments, ~100â€“120 single-
item `E_harm` ratings (reusing an existing, already-validated
instrument), plus ~24â€“30 new pair-construction/review actions. Larger
than the original draft's estimate because of the added `E_harm`
validity arm, but still comparable in order of magnitude to 3D-H-A's
existing 32-item check, and well below what 3.B or 3.D would require.

---

## 7. Transfer test â€” decision criteria (feasibility pilot, not definitive validation)

**Exact metric:** pairwise accuracy = (number of held-out benign pairs
where `sign(c_w(a) - c_w(b))` matches the pair's observed, non-tied
`C_cue` relation) / (number of held-out benign pairs with a non-tied
relation). Tied benign pairs are excluded from both numerator and
denominator, not scored as failures or successes.

**Uncertainty:** an exact (Clopperâ€“Pearson) binomial confidence interval
on pairwise accuracy against chance = 0.5, **and** a permutation test â€”
shuffle the harmful-side observed `C_cue` relations before fitting,
re-fit `c_w(x)` under the same procedure, re-evaluate on the same
held-out benign pairs, repeat â‰¥10,000 times (mirroring 3D-H-A's own
permutation-test method, already used in this repository), reporting the
empirical two-sided p-value of the observed accuracy against this null.

**Minimum evidence threshold (predeclared here, before any
implementation):** the Clopperâ€“Pearson lower bound on pairwise accuracy
must exceed 0.5 **and** the permutation p-value must be < 0.05, **and**
this result must not be matched (within the same confidence interval) by
the length-only or formatting-only baselines (Â§3.5) evaluated on the
identical held-out benign pairs. All three conditions are required, not
any one alone. Given the pilot's feasibility scale (~24â€“30 benign pairs,
even fewer non-tied), a positive result licenses only the narrow claim
in Â§9 (proceed to a larger-scale 3F-C), never a claim that the common
axis is established.

**Result language â€” restricted to four terms, used precisely and never
substituted for one another:**

- **Separability** â€” the weakest claim: `c_w(x)` or a baseline can
  distinguish two groups (e.g. source vs. candidate, or harmful vs.
  benign) on some axis in aggregate. Simple XSTest/Alpaca or source/
  candidate separability is **not**, by itself, evidence of a common
  cue axis (per task Â§9) and must never be reported as such.
- **Transferability** â€” the actual target claim: a direction fit on
  harmful-side observed relations predicts benign-side observed
  relations above chance, under the Â§7 thresholds, surviving the
  baseline comparisons.
- **Human construct alignment** â€” a property of the *instrument*, not
  of `c_w(x)`: whether the new `C_cue` judgments are empirically
  distinguishable from `E_harm` judgments on the same items (Â§5 Step 7,
  Â§8). A finding that they collapse together is a valid, reportable
  negative result about the instrument, not about the ranking model.
- **Causal model usage** â€” explicitly **not** claimed by this pilot at
  any threshold. Nothing here establishes, or is designed to establish,
  that a trained model's behavior *uses* `C_cue` causally. Any such
  claim would require separate model-side work (e.g. probing/ablation
  linking `C_cue`-scored inputs to internals or behavior), out of scope
  for this design and for the proposed feasibility pilot.

---

## 8. Explicit failure / stop conditions

- If Step 1 yields fewer non-tied `C_cue` relations than needed to fit
  even the capped 3-feature model with meaningful leave-one-pair-out
  variance (e.g. most pairs return "no clear difference"), report that
  the 104-pair resource does not yield a usable `C_cue` contrast under
  blind human judgment, full stop â€” do **not** fall back to
  `reduced_cue_source_rewrite` or any other construction label to
  manufacture relations.
- If no benign pairs can be constructed that a researcher confirms as
  `objective_preserved=yes` (i.e. no defensible same-intent benign cue
  contrast exists), report that explicitly, per task Â§4/Â§7, rather than
  fabricating pairs from source-identity or template membership.
- If the held-out transfer test (Â§7) does not clear all three
  predeclared conditions, the correct output is an explicit NO-GO /
  redesign recommendation for a common-axis construction from this data
  â€” not a forced positive, and not a partial-credit claim using looser
  language than Â§7's four defined terms.
- If the instrument-validity diagnostic (Â§5 Step 7) shows `C_cue` and
  `E_harm` judgments are not empirically distinguishable on the rated
  items, report this as a specific negative finding about the new
  instrument (it may be re-designed in a later task) â€” it does not by
  itself invalidate the ranking-model result, but the two findings must
  be reported separately, never merged into a single verdict.
- Any report of this pilot must not claim, or be phrased in a way that
  implies, "causal model usage" (Â§7) â€” that specific overclaim is called
  out because it is the most likely one for a reader to import from
  adjacent behavioral/interpretability work already in this repository
  (`HANDOFF.md`), which this pilot does not touch.

---

## 9. Summary

- **Recommended method:** pairwise/matched contrast (3.C), with `C_cue`
  measured by a new paired human instrument (Â§3.C.1) kept explicitly
  separate from `E_harm`, never by the existing `CUE` score, by
  `E_harm` itself, or by construction-family labels (Â§0, Â§2.5).
- **Why:** the only candidate that holds `I(x)` fixed by construction,
  reuses existing harmful-side pair text at near-zero new text-writing
  cost, and has a fully specified, auditable scoring function (Â§3.C.2)
  and held-out cross-intent transfer test (Â§7) with predeclared
  thresholds.
- **Minimum data/human effort:** ~50â€“60 new paired `C_cue` judgments +
  ~100â€“120 single-item `E_harm` ratings (reusing an existing instrument)
  + ~24â€“30 new benign matched pairs (Â§6).
- **Exact next step:** a separately-authorized 3F-B task to run Â§5's
  seven steps as a **feasibility pilot** and report the Â§7 result using
  only the four defined result-language terms; no B/D construction, no
  benchmark change, no scorer implementation, and no causal claim until
  a materially larger, separately-authorized follow-on task.