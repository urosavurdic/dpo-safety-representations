STATUS: DRAFT FOR REVIEW

# 3F-B — Benchmark/Resource Comparison and Decision Memo

**Repository:** `https://github.com/urosavurdic/dpo-safety-representations`
**Branch:** `agent/c-quadrant-end-to-end-e0e2317a`
**Scope:** Design-only synthesis of the preceding read-only resource audit.
No new repository inspection was performed to produce this memo — every
claim below traces to evidence already gathered against `HEAD`
`2e15cd2d778851a49fe61c4ea3e2c988d52d3627`. No scoring code was run, no
benchmark file was touched, no candidate was promoted, no benign pair was
constructed, no ranker was fit, and no transfer statistic was computed.
This document defines decision gates for future, separately-authorized
work; it does not pass any of them.

---

## 1. What this memo is and is not

This memo compares four named resources — **R104**, **R-HUMAN**,
**R-AUTHORED**, **R-CONTRAST** — against two constructs — **`C_cue(x)`**
and **`E_harm(x)`** — and states what each resource can and cannot be
used for. It contains no raw prompt text, no private ratings, no answer
keys, no blind-ID mappings, no model completions, no generated prompt
pairs, no fitted ranker weights, and no transfer-test statistics, per the
data-handling boundary already in force for this project.

---

## 2. Construct definitions (restated, not re-derived)

| Symbol | Definition | Notes |
|---|---|---|
| `I(x)` | Underlying intent (harmful vs. benign). | Fixed by dataset curation/quadrant membership, not at issue here. |
| **`C_cue(x)`** | Surface, safety-sensitive or potentially-harmful-sounding wording salience, judged **independent of** whether the underlying request is actually harmful or benign. | No existing artifact in this repository measures this directly. A benign prompt can have low `E_harm` but high `C_cue` (e.g. XSTest-style "kill a process" phrasing) — the two constructs are not interchangeable and must be measured separately. |
| **`E_harm(x)`** | Human-perceived clarity that the wording communicates a harmful **operational** objective — conditional on, and entangled with, judged actual harm. | Measured, within-harmful-only, by the existing 3D-H single-item 1–5 instrument. |
| `Y(x)` | Model behavior (refusal, compliance, output distribution). | Out of scope for this memo. |

**`C_cue` and `E_harm` are distinct constructs and must not be substituted
for one another in any report, threshold, or resource label.** A rating
of "unmistakably apparent harmful objective" (`E_harm`) says nothing, on
its own, about whether the wording is surface-safety-sensitive in a way
that would generalize to a benign prompt (`C_cue`), and vice versa.

### 2.1 What does *not* count as `C_cue` ground truth

The following are all real, documented signals in this repository, and
none of them is `C_cue(x)`:

- **Source identity** (e.g. "drawn from StrongREJECT," "drawn from
  SimpleSafetyTests," "authored vs. AI-reworded"). Source identity is a
  provenance fact, not a surface-wording judgment.
- **`c_source_authored` status** (i.e. being an unchanged external
  record). Being unchanged says nothing about how safety-sensitive the
  wording reads.
- **Lexical zero-hit status** against any single predeclared lexicon
  (e.g. the 104-pair resource's zero hits under one substring lexicon).
  A zero-hit result demonstrates absence of that lexicon's specific
  terms, not absence of safety-sensitive wording in general, and the
  project's own external-lexicon audit found that raw lexical matching
  does not discriminate harmful from benign text at all, and that even
  collision-audited matching does not cleanly separate the harmful pool
  from a benign control pool.
- **`p_tfidf` / `p_selfinfo`** (the 3D-A/3D-B within-harmful lexical
  outlierness scores). These were never applied to a benign prompt, are
  confounded with length in a way the 3D-C audit could not correct, and
  are explicitly described in this project's own design work as
  measuring within-harmful outlierness, not a cross-intent `C_cue` axis.
- **Construction labels** such as `c_paired`, `reduced_cue_source_rewrite`,
  or `c_source_authored`. These describe how a candidate was built, not
  an observed property of its wording, and this project's own design
  work explicitly removed the rule that would have let a construction
  label stand in for an observed blind judgment.

Treating any of the above as `C_cue` ground truth would be a construct
substitution, not a measurement.

---

## 3. Resource identity and construct support

| Resource | Exact artifact(s) | Construct actually supported | Status |
|---|---|---|---|
| **R104** | `data/review/c_review_queue.csv` (104 rows, all `review_status=accept`) → frozen as quadrant C (`c_paired`) in `data/frozen_v2/benchmark_v2_20260826T212909Z.*` | Supports an **objective-preservation, operational-detail-changing rewrite** claim (verified: `objective_preserved=yes` and `operational_detail_changed=yes` for all 104 rows). Does **not** support a `C_cue`-only claim and does not support an `E_harm` claim either — no blind human rating of *this specific pair set* on either construct exists. | Frozen and in active use as the project's quadrant-C harmful stress test; not validated for any cue-isolation claim. |
| **R-HUMAN** | `data/review/3d_h_blind_construct_check.csv` (32 rows) + `logs/3d_h_rating_instructions.md` + `logs/3d_h_construct_check_analysis.md` | Supports an **`E_harm`** measurement only, on a small (n=32), selected-tail (high/low `p_tfidf`), harmful-only, blind sample. Its own analysis explicitly disclaims establishing a universal cue variable, independence from intent, causal validity, or a validated shared axis. | Descriptive/diagnostic pilot; already "spent" for its original purpose per the project's own design note. |
| **R-AUTHORED** | `data/quadrant_c_pipeline/c_source_authored_candidates_raw_v1.jsonl` → `..._validated_v1.jsonl` → `data/review/c_source_authored_review_queue.csv` (52-row Q10/Q25 queue) | Intended to support a **strict unchanged-external-record** ("source-authored") arm. Currently supports **no construct claim at all**: all 52 queued rows show `review_status=pending` — zero human review has occurred. | Pipeline stage `NOT_BUILT`/pending; not a resource yet, only a candidate queue. |
| **R-CONTRAST** | `logs/3f_a_common_cue_axis_design.md` (design specification, no data) | Intended to support the actual **`C_cue`** target construct via a new paired human instrument, kept explicit separate from `E_harm`. | Design-only. No benign pairs exist. No pilot has been run. |

### 3.1 R104 is not a clean cue-only resource

This must be stated plainly, because the resource's naming history
(`reduced_cue_source_rewrite`) invites the opposite reading: **R104 is
not a clean, same-intent, wording-only contrast.** Per-row evidence
verified in the prior audit shows every one of its 104 pairs changed
operational detail between the source and candidate prompt, and only
78/104 fully preserved assistance type (26/104 only partially). No blind
human instrument has ever been applied to these 104 pairs to confirm a
cue-only relationship; the resource's own design documentation (`3f_a`,
§0.4/§2.5) independently reaches the same conclusion and downgrades R104
from "raw material a cue-only benchmark requires" to "candidate material
for a future human instrument to adjudicate." R104 remains usable as a
secondary, confounded stress test — it is not usable as `C_cue` ground
truth, and must not be described as such in any future report.

### 3.2 R-AUTHORED must not be assumed better than R-HUMAN

These two resources measure different things and neither is a strict
upgrade of the other:

- R-HUMAN has an administered instrument, a completed statistical
  analysis, and a defined construct (`E_harm`), but a small, selected,
  harmful-only sample.
- R-AUTHORED has a larger candidate pool and a stricter provenance
  property (unchanged external record) by design, but **zero** rows have
  been human-reviewed, so it currently carries no validated construct
  claim of any kind — not `E_harm`, not `C_cue`, not even a confirmed
  "standalone complete request" property for the un-reviewed remainder.

Ranking R-AUTHORED above R-HUMAN on the strength of "authored/unchanged
status" alone would repeat exactly the error this project's own
constraints warn against — treating provenance or a pending selection
score as if it were a completed, validated measurement. The two
resources are not comparable on a single scale until R-AUTHORED's queue
receives the human review it is currently pending.

---

## 4. Does any current resource support a common `C_cue`-axis claim?

**No.** Stated explicitly, resource by resource:

- R104 is confounded (operational detail changes) and has never been
  blind-rated for cue-only status.
- R-HUMAN measures `E_harm`, not `C_cue`, and only on the harmful side.
- R-AUTHORED has no human-reviewed rows at all.
- R-CONTRAST — the one resource actually designed to test the common-axis
  hypothesis — has no data: no benign pairs, no blind ratings, no fitted
  scorer, no transfer evaluation.

No resource currently in this repository supports a claim that a common,
wording-level `C_cue` axis transfers across intent conditions. This is
consistent with the project's own design document, which frames a
positive pilot result as licensing only a narrow "proceed to a larger
study" recommendation — never a claim that the common axis is
established — and this memo does not relax that standard.

---

## 5. The B–D → A–C transfer hypothesis: preserved as a future route

The specific common-axis hypothesis under study — that a score or
direction learned from a benign, intent-matched **B–D** contrast should
transfer, without refitting, to the harmful-side **A–C** contrast — is
**not tested, not falsified, and not abandoned** by anything in this
memo. It remains the target scientific question for a future, adequately
resourced study, for two independent reasons already established in the
audited evidence:

1. No benign matched-pair resource (the R-CONTRAST "new benign matched
   pairs" artifact) currently exists to run the B-side of the test at
   all.
2. The harmful-side candidate material (R104, and eventually R-AUTHORED
   once reviewed) has not yet been blind-rated for `C_cue` specifically,
   so even the A–C half of the eventual transfer test is not yet built
   from validated inputs.

Nothing in this memo should be read as a recommendation against pursuing
the B–D → A–C direction; it should be read as a statement that the
prerequisite instruments and data do not yet exist, per §6 below.

---

## 6. Purpose-dependent recommendations

Recommendations are tied to a specific estimand, not offered as a single
global ranking:

1. **For an `E_harm` audit today:** R-HUMAN is the best-supported
   resource — it is the only one with an administered instrument and a
   completed analysis — but any use must carry its caveats (n=32,
   selected-tail, harmful-only, single instrument, descriptive not
   causal).
2. **For the current DPO harmful behavioral/interpretability stress
   test:** R104 remains the appropriate resource, because it is the only
   fully built, reviewed, frozen quadrant-C set wired into the existing
   behavioral/probe/ablation results — provided every report using it
   states plainly that it is an operational-detail-reduced rewrite set,
   not a cue-isolated set (§3.1).
3. **For testing a common `C_cue` axis:** no resource is usable yet.
   R-CONTRAST's design (§3.C of `3f_a_common_cue_axis_design.md`) is the
   correct *method* — a paired human instrument feeding a small, fully
   specified linear scorer, tested harmful→benign — but it requires the
   3F-B feasibility pilot (§7 below) to be run before any data exists.
4. **Resources that must remain secondary/exploratory regardless of
   purpose:** R104 (per its own design-document downgrade), R-HUMAN (per
   its own "already spent for its original purpose" status), and the
   `secondary_c2`–`c5` stylistic/contextual/dual-use/evasion families,
   none of which were ever eligible for primary promotion.
5. **R-AUTHORED specifically:** do not use for any construct claim until
   Gate 1 (§7) — human review of the 52-row queue — is cleared.

---

## 7. Next decision gates (defined, not implemented)

Each gate below states an entry condition, the action it authorizes, and
an exit condition. **None of these gates has been opened by this memo.**
No gate authorizes itself — each requires separate, explicit
authorization before the corresponding task begins.

### Gate 1 — R-AUTHORED human review (a "3B" task)
- **Entry condition:** the 52-row `data/review/c_source_authored_review_queue.csv`
  queue exists with `review_status=pending` for all rows (confirmed).
- **Action authorized if opened:** researcher review of each queued row
  against the same standalone-completeness / provenance criteria already
  used elsewhere in this pipeline.
- **Exit condition:** every row has a terminal `review_status`
  (`accept`/`reject`, not `pending`); an updated count of accepted
  `c_source_authored` rows is reported before any further use of this
  resource in an eligibility, scoring, or promotion claim.

### Gate 2 — Harmful-side `C_cue` + `E_harm` blind rating pilot (the "3F-B" pilot proper, per `3f_a` §5 Step 1)
- **Entry condition:** a stratified blind sample (~24–30 pairs) can be
  drawn from the 78-row `assistance_type_preserved=yes` stratum of
  R104's accepted rows (partial-preservation rows sampled separately and
  flagged exploratory only, per `3f_a` §2.5/§5).
- **Action authorized if opened:** administer both the new paired
  `C_cue` instrument (`3f_a` §3.C.1) and the existing 3D-H `E_harm`
  instrument to the same sampled pairs, recording each pair's observed
  relation on both constructs; "no clear difference" pairs are recorded
  but excluded from any downstream fitting.
- **Exit condition:** a report stating how many pairs yielded a
  non-tied `C_cue` relation. If too few pairs yield a usable relation
  to fit even the capped 3-feature model, that is itself the reportable
  outcome (`3f_a` §8) — not a trigger to fall back to a construction
  label.

### Gate 3 — Benign matched-pair construction (`3f_a` §5 Step 2)
- **Entry condition:** Gate 2 has produced enough non-tied harmful-side
  relations to make fitting meaningful, per Gate 2's exit report.
- **Action authorized if opened:** construct ~24–30 new benign
  same-objective pairs (one more safety-sensitive-sounding phrasing, one
  softer phrasing, same underlying benign request), reviewed against the
  same `objective_preserved` / `assistance_type_preserved` /
  `operational_detail_changed` fields used for the harmful pipeline, with
  operational detail held as close to fixed as feasible.
- **Exit condition:** a researcher-confirmed set of benign pairs with
  `objective_preserved=yes`; if no such pairs can be confirmed, that is
  reported explicitly rather than substituted with source-identity or
  template-membership pairs (`3f_a` §8).

### Gate 4 — Harmful→benign transfer evaluation (and conditional benign→harmful), `3f_a` §5 Steps 3–7 / §7
- **Entry condition:** Gates 2 and 3 both cleared with usable non-tied
  relations on both sides.
- **Action authorized if opened:** fit the 3-feature `c_w(x)` scorer on
  harmful-side relations only; evaluate held-out on the benign pairs;
  report pairwise accuracy with a Clopper–Pearson interval and a
  permutation test against length-only and formatting-only baselines.
  The symmetric benign→harmful direction is exploratory/diagnostic only,
  conditional on adequate power on both sides, and is never a second
  primary result.
- **Exit condition — predeclared, unchanged from `3f_a` §7:** the result
  is reported using only the four defined terms (separability,
  transferability, human construct alignment, causal model usage — the
  last never claimed by this pilot at any threshold). A positive result
  (Clopper–Pearson lower bound > 0.5 **and** permutation p < 0.05 **and**
  not matched by either baseline) licenses only a recommendation to
  proceed to a larger-scale study — never a claim that the common axis
  is established. A negative or underpowered result is reported as a
  NO-GO / redesign recommendation, not softened.

**No gate above has been opened, and no step of any gate has been run,
by this memo.**

---

## 8. Explicit non-actions

For the avoidance of doubt, this memo did **not**: run any scoring or
classifier code; modify `data/frozen_v2/*`, `src/cue_scoring.py`, or
`src/corpus_discrimination.py`; promote, accept, or reject any
`c_source_authored` candidate; construct any benign pair; fit any
ranker; compute any transfer statistic; or begin Gate 1, 2, 3, or 4 above.
It is a comparison and decision-gate specification only.
