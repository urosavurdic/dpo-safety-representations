STATUS: DRAFT FOR REVIEW — read-only inventory + locked analysis contract. No analysis code implemented, no benchmark data modified, no R104/R-AUTHORED records touched, no prompts created or rewritten, no model inference run, no web access used, no C-B/C-C/C-D work started, no common-CUE/contrastive construction started, no final resource decision made.

# C-A — Existing C-Construction Inventory and Locked C-B Analysis Contract

**Repository:** `https://github.com/urosavurdic/dpo-safety-representations`
**Branch:** `agent/c-quadrant-end-to-end-e0e2317a`
**Git commit inspected:** `432d19767fef600696658e7413cc638a066fc909`
**Task type:** read-only data inventory + documentation. This document is the only output.

This task inspected only the files listed in the task brief's "minimum
relevant current files" list (plus the frozen benchmark itself, to verify
quadrant identity/counts, and the small number of source modules that
implement fields already appearing in the inventoried CSVs/JSONs). No
broad repository scan or test-suite run was performed. Every count below
was read from the current file at the commit above — no historical count
from any prior log was assumed to still hold; where a current value
matches a historical log, that is stated as a verification, not assumed.

---

## 1. A/B/C/D — current benchmark identity

**Files:**
- `data/frozen_v2/LATEST_BENCHMARK.json` → points to
  `data/frozen_v2/benchmark_v2_20260826T212909Z.jsonl`
  (sha256 `e4946b070f441c7a0676db830c65257b78a2d1b46abb0a61cce4cc86352f838b`,
  matches both the pointer file and the file's own manifest — verified,
  no CRLF/LF hash drift observed in this clone).
- `data/processed/controlled_eval.jsonl` (654 rows, sha256
  `e640c2fba47afe2853c8717ae8492c62bf26cce21f6ec677f68ea88b117c05af`) — a
  reduced-schema (`prompt`, `quadrant`, `source`, `category`, `split`)
  companion file with identical per-quadrant counts to the frozen
  benchmark; used by `src/corpus_discrimination.py::load_quadrant_texts`.

**Current counts (verified against the frozen `.jsonl`, not assumed from
any manifest or prior log):**

| Quadrant | n | source_dataset | c_construction | review_status |
|---|---|---|---|---|
| A | 150 | HarmBench (100%) | null (all rows) | null (all rows) |
| B | 250 | XSTest (100%) | null (all rows) | null (all rows) |
| C | 104 | StrongREJECT (100%) | `c_paired` (100%) | `accept` (100%) |
| D | 150 | Alpaca 50 / Dolly-15k 50 / OASST1 50 | null (all rows) | null (all rows) |

**Stable IDs:** `record_id` is unique per row; for quadrant C it is
identical to `pair_id` and `candidate_id` (e.g. `SR_disinfo_01`).
`source_id` is populated for quadrant C only — **it is `null` for every
quadrant-A row**, which is the reason no stable cross-quadrant join key
exists (§2.6 below).

**Category fields:** `project_category` is populated for A and C (not for
B/D, which use their own `category` field instead). A and C share the
same four-category taxonomy — `harassment_bullying`, `illegal`,
`cybercrime_intrusion`, `misinformation_disinformation` — but with very
different proportions (A: 19/57/40/34; C: 41/6/20/37). This is recorded
as a real, current confound, not inferred from any prior log.

---

## 2. R104 (`c_paired`, `data/review/c_review_queue.csv`)

**File:** 104 rows, sha256 `8f6dfba182e5d3595d9ac6292d13956dd1a027b18770da01f4ef510f236787bb`
(current). Note: `logs/queue_scoring_summary.json` records a *different*
hash (`9e0f8259...`) for this same path — that is the pre-review hash,
captured while `review_status` was still `pending` for all 104 rows
(`"all_pending": true` in that file). The current hash reflects the
completed researcher review (`git log` shows `59f533e chore: researcher
review of c_review_queue.csv` → `c40620b csv accepted` →
`cb31787 Renormalize review CSVs to LF in the index`). This is a
provenance fact, not a discrepancy to resolve.

**Field-level status (all 104 rows, verified directly from the CSV):**

| Field | Values |
|---|---|
| `review_status` | `accept` — 104/104 |
| `objective_preserved` | `yes` — 104/104 |
| `assistance_type_preserved` | `yes` — 78/104; `partial` — 26/104 |
| `operational_detail_changed` | `yes` — 104/104 |
| `wrapper_or_context_concern` | `no` — 104/104 |
| `researcher_harm_qc` | `yes` — 104/104 |
| `source_dataset` | `StrongREJECT` — 104/104 |
| `contamination_status` | `checked_exact_zero_near_unknown` — 104/104 |
| `review_evidence_status` | `missing_or_list_level_only` — 104/104 |
| `pair_id` | 104 unique values, no duplicates |

**Operational-detail fields:** the schema has only the single binary
`operational_detail_changed` flag — there is no finer-grained field
separating "specificity changed" from "actionability changed" from
"surface phrasing changed." This is exactly the gap `3f_a` (§2.5) already
identifies: the flag cannot certify a cue-only contrast, because a
`yes` here is equally consistent with a wording-only change and with a
substantive request change.

**Exact overlap with quadrant A:** computed directly (case-insensitive,
whitespace-normalized exact string match) between all 104 R104
`source_prompt` values and all 150 quadrant-A `prompt` values in the
frozen benchmark: **0/104 exact matches.** This is consistent with, and
independently confirms, the fact that R104 is drawn entirely from
StrongREJECT while quadrant A is drawn entirely from HarmBench — two
disjoint upstream datasets.

**Near-duplicate overlap with quadrant A:** **UNTESTED — requires
embedding-model inference (`sentence-transformers`), which is outside
this task's CPU-only, no-model-inference constraint.** No existing
repository artifact covers this specific check: the repository's
existing `dedup_report_*.json` files (`data/dedup_report*.json`) all
check *training-data* leakage against the eval set, not a quadrant-A-vs-
quadrant-C cross-check, and `src/diagnostics/check_within_eval_set_dedup.py`
is scoped to within-quadrant-D sub-source comparison only. This is a
genuine, reportable evidence gap, not an oversight — recorded here per
the task brief's fail-closed instruction rather than reconstructed
approximately.

**Whether an exact stable join to A is possible:** **No.** Quadrant A's
`source_id` field is `null` for all 150 rows, so no ID-based join exists
between R104's `source_prompt` provenance and quadrant A. Only a
full-text exact-match check is possible, and it has been run (above,
0/104).

**Category agreement:** see §1 — same 4-category taxonomy, different
proportions. This is evidence available, not evidence of a clean match.

**Source identity:** confirmed directly from the CSV and the frozen
benchmark — R104 is 100% StrongREJECT; quadrant A is 100% HarmBench. No
row in either the CSV or the frozen benchmark's C rows is drawn from any
other dataset.

---

## 3. R-AUTHORED (`c_source_authored`)

**Pipeline files (current row counts, verified directly):**

| File | Rows | sha256 |
|---|---|---|
| `data/quadrant_c_pipeline/c_source_authored_candidates_raw_v1.jsonl` | 413 | `921ebe1687f2926115d4b1d846a97aebaaae0d0d10d5943e141bf2be696581c1` |
| `data/quadrant_c_pipeline/c_source_authored_candidates_validated_v1.jsonl` | 413 | `8be074b8f42e87aa1c3eee83abedec9743e48ce1a65758e329118b34a6c37ca7` |
| `data/review/c_source_authored_review_queue.csv` | 52 | `c62725ec37b3d950d7fda164c89d6e71a315cc5dad1a701147100c9f8e8e485a` |

413 raw/validated rows, 209 of which were scored as eligible
(`logs/3a4_scoring.md`, matched here), of which 52 were queued
(Q10=20, Q25=32) — all counts match the historical `3a4_scoring` log
exactly; no drift found.

**Review status:** **`pending` — 52/52.** Zero human review has
occurred on this queue. No terminal `accept`/`reject` value exists on
any row.

**Provenance (52 queued rows):** `provenance_class` = `custom` 37,
`curated` 13, `upstream-derived` 2. Source dataset: `StrongREJECT` 39,
`SimpleSafetyTests` 13. `classifier_status` = `confirmed` 35,
`provisional_low_confidence` 17 (i.e. ~1/3 of the queue's structural
"is this a standalone user-facing prompt" classification is itself
low-confidence, on top of zero human review).

**Wording modification:** not applicable — R-AUTHORED is defined as
unchanged external record; the pipeline records `upstream_provenance_detail`
per row rather than any rewrite step, consistent with §0 item 2 of
`3f_a`.

**Overlap/contamination (already computed by the existing pipeline,
verified current):** `overlap_status` = `no_c_paired_or_quadrant_a_overlap`
— 52/52. `contamination_status` = `checked_exact_zero_near_unknown` —
52/52 (same near-duplicate-unknown caveat as R104, §2 above).

**Construct claim currently justified:** **none.** Not `E_harm`, not
`C_cue`, not even a human-confirmed "standalone complete request"
property for the un-reviewed remainder — `classifier_status` is a
structural heuristic, not a review outcome.

---

## 4. 3D-B / 3D-H

**3D-B** (`logs/3d_b_lexical_outlierness_pilot.json`, sha256
`95b0b7771244f0c162627eb1aaeb92986b4e7ec9de737f4f38edaefec53ebce5`):
within-harmful lexical-outlierness pilot, n=209 (StrongREJECT 132 /
SimpleSafetyTests 77 — the same 209-eligible population that feeds
R-AUTHORED's scoring), producing two per-prompt scores (`p_tfidf`,
`p_selfinfo`) computed only within the harmful population — **never
applied to a benign (B/D) prompt.** `3d_c_length_dependence_audit.md`
(verified, sha256 `935923a44afa7ed049ce7c4a37525a8b02df4a0beb40cf438779715b80b8ef40`)
independently re-derived the `p_tfidf`/length correlation from first
principles and confirmed it is a structural property of the locked
TF-IDF-centroid-distance definition, not an implementation bug — "a
redesign, not a correction." `p_tfidf` and `p_selfinfo` are explicitly
**not** a `C_cue` axis and are not treated as one anywhere in this
document.

**3D-H** (`logs/3d_h_construct_check_analysis.json` +
`3d_h_blind_review_provenance.json`, hashes
`cf7a77eccd11820b2f0447642209e94e734d17851b849f79e44084eb22fe0ea7` /
`3318916dc22b9d841b07c1d140e7602b8431286b8c8ef6a75999c94c0a9619fd`):
n=32 blind human ratings (16 high-`p_tfidf`-tail / 16 low-tail,
StrongREJECT+SimpleSafetyTests, 8 per tail per source), selection seed
`20260829`, presentation seed `20260830`, `numpy` `2.4.4`, PCG64. Measures
**`E_harm`** — "how clearly does the wording itself signal a harmful
operational objective" — a single-item 1–5 rating, within-harmful only.
Result: high-`p_tfidf`-tail mean 3.19 vs. low-tail mean 4.625 (Mann–Whitney
U=50, p=0.0020; permutation test, 100,000 draws, seed `20260831`,
two-sided empirical p≈0.00102). This is a real, disclosed, negative
correlation between lexical typicality and perceived harm-clarity —
**within the harmful population only**, and it is 3D-H's own analysis,
not re-derived here.

**Explicit distinction (population, not just construct):** 3D-H's 32
rated items are a *selected tail sample* of the 209-row 3D-B population —
they are **not** R104, not R-AUTHORED, and not any C candidate
population evaluated for inclusion in a benchmark. `3f_a`/`3f_b`
already state this; this task's direct inspection of
`3d_h_blind_review_provenance.json`'s `exact_sampling_rule` field
confirms selection used only `p_tfidf` tail membership, source, and
group ID — never prompt content, category, or the R104/R-AUTHORED review
schema. No claim in this document conflates the 3D-H sample with a C
candidate population.

**Freshness check on an adjacent diagnostic:** `data/quadrant_composition_report.json`
(sha256 `30bcf932e29ba00672e0be89c3d77868f214d0645f0dd2eb2602a7870c9b9c8f`)
was re-read directly and matches `3f_a`'s cited figures exactly (A
mean-cue-hit-rate 0.307/29.3%, B 0.032/3.2%, current — not stale). Its
same lexicon reports **C=0 and D=0 mean-cue-hits** — i.e. under this
particular narrow lexicon, both quadrant C and quadrant D sit at the
floor with no separation between them; this is a real limitation of
that specific instrument's power, not evidence that C and D are
otherwise similar.

---

## 5. R-CONTRAST (`3f_a`/`3f_b` design documents)

Both re-read in full and verified against the current repository state
(no scoring code, no benign pairs, no `src/cue_scoring.py` or
`src/corpus_discrimination.py` change, no frozen-benchmark change — all
confirmed by direct inspection, not merely by trusting the documents'
own "no new inspection was performed" disclaimer). `3f_a`
(sha256 `6c20de7000389393304e506770a6385c0cc67b98f0d7211445d1bf8b08a5aaf7`)
defines the `C_cue` human instrument and the exact 3-feature scorer
`c_w(x)`; `3f_b` (sha256 `8c0a53c753a5a19a2fc92bbfc6bf4f6263ff0912f7e45a1e1ac4a7ccef6dfdc5`)
compares R104/R-HUMAN/R-AUTHORED/R-CONTRAST against `C_cue`/`E_harm` and
defines four not-yet-opened decision gates. Neither document is data;
both are consistent with everything independently verified in §§1–4
above. No resource currently in the repository supports a common
`C_cue`-axis claim — this document does not relax that conclusion, and
does not attempt to test it.

---

## 6. Inventory table

| Resource | Observed property | Evidence available | Missing evidence | Main confounds | Minimum next analysis |
|---|---|---|---|---|---|
| **A/B/C/D benchmark structure** | 150/250/104/150 rows; A=HarmBench, B=XSTest, C=StrongREJECT (`c_paired`), D=Alpaca/Dolly/OASST1; frozen hash verified against manifest | Full schema (28 fields incl. `record_id`, `source_dataset`, `project_category`, per-row hashes) | `source_id` is null for A/B/D — no cross-quadrant join key except full-text match | A and C share a 4-category taxonomy at very different proportions; A/C category fields absent for B/D | None required — this is confirmed structural ground truth, not a construct claim |
| **R104** (`c_review_queue.csv`, 104 rows) | 100% StrongREJECT; 100% `objective_preserved=yes`; 100% `operational_detail_changed=yes`; 78/104 `assistance_type_preserved=yes`, 26/104 `partial`; 0/104 exact-text overlap with quadrant A | Full paired source/candidate text, word/char counts, Fightin'-Words scores (both directions), review outcome fields | Near-duplicate overlap with A (requires model inference, out of scope); no field separates "cue wording changed" from "request substance changed" within `operational_detail_changed` | `operational_detail_changed=yes` on every row is exactly as consistent with an `E_harm`-relevant change as a `C_cue`-relevant one; single source dataset (no within-R104 source-confound check possible) | Locked C-B contract below: paired delta analysis on structural/lexical features, stratified by `assistance_type_preserved` |
| **R-AUTHORED** (52-row queue) | 413 raw/validated candidates → 209 eligible → 52 queued (Q10=20/Q25=32); provenance_class custom 37/curated 13/upstream-derived 2; 0 overlap with C-paired/quadrant A | Full scoring/rank/stratum fields, structural classifier label + confidence | **100% `review_status=pending`** — zero human review of any kind | 17/52 rows carry a `provisional_low_confidence` structural-classifier flag on top of no human review | Not started by this task (Gate 1 in `3f_b` §7) — no analysis possible until human review produces a terminal status |
| **3D-B** (n=209 within-harmful pilot) | `p_tfidf`/`p_selfinfo` computed only within the harmful population; length-confound re-derived and confirmed structural (not a bug) by `3d_c` | Full per-record scores, source-balanced sensitivity check | Never scored against a benign prompt; not itself a labeled C population | `Spearman(length, p_tfidf) ≈ -0.567` by construction (design property, not correctable in-place) | Auxiliary/diagnostic only per `3f_a` — no promotion to a C-cue score |
| **3D-H** (n=32 blind `E_harm` ratings) | High-tail vs. low-tail `E_harm` mean difference −1.44 (Mann–Whitney p=0.0020; 100k-draw permutation p≈0.00102), within-harmful only | Full rating distributions, permutation-test null, selection/presentation seeds and exact sampling algorithm | Never measures `C_cue`; sample already spent for its original purpose | Selected-tail (not random) sample, n=32, harmful-only | Auxiliary validity check only (§3.C.1 of `3f_a`) — not a C-cue label source |
| **R-CONTRAST** (`3f_a`/`3f_b` design docs) | Fully specified `C_cue` human instrument and 3-feature scorer; four unopened decision gates | Complete method spec, predeclared thresholds, predeclared failure conditions | No data of any kind — no benign pairs, no blind ratings, no fitted scorer | N/A (design-only) | Requires Gate 2 (harmful-side blind rating pilot) before any C-B/C-C work on this axis specifically — out of scope for the current C-B task below |

---

## 7. Locked analysis contract for C-B (predeclared; C-B must not deviate)

This section is the reproducible **specification**. No implementation of
it exists yet, and no numerical result from it exists yet — nothing in
this section should be read as having been run. C-B's own commit must
state, separately, (a) that its code matches this spec, (b) the runtime
software versions actually used, and (c) the resulting numbers — none of
those three is satisfied by this document alone.

### 7.1 Exact input paths and hashes (pin these; abort if any differs)

| Path | sha256 |
|---|---|
| `data/review/c_review_queue.csv` | `8f6dfba182e5d3595d9ac6292d13956dd1a027b18770da01f4ef510f236787bb` |
| `data/frozen_v2/benchmark_v2_20260826T212909Z.jsonl` | `e4946b070f441c7a0676db830c65257b78a2d1b46abb0a61cce4cc86352f838b` |
| `data/frozen_v2/LATEST_BENCHMARK.json` | `817885c1c50dcbb5babddaec05b938f0f47067151ababa3c669e893f38ea937a` |
| `logs/benchmark_gate_config.json` | `1ac73585f08a4d685996c96eecafdbfcc74478ab07f7f762a4a9de2b2568b743` |
| `logs/3d_b_lexical_outlierness_pilot.json` (source of `formatting_diagnostic_config`) | `95b0b7771244f0c162627eb1aaeb92986b4e7ec9de737f4f38edaefec53ebce5` |
| `src/corpus_discrimination.py` (tokenizer, Fightin' Words, `load_quadrant_texts`) | `1ca62c4f7c1f88398c2d22c60bc1f2f6be27be678b68e9675a8800bdb41a9bcc` |
| `src/cue_scoring.py` (existing CUE score, reused as descriptive-only feature) | `ea0aa39faee7f8358121cee460be3c3f45d840c555beaba8ee534da2033b7d1d` |
| `src/diagnostics/score_lexical_risk_cues.py` (existing fixed lexical-risk lexicon — not reproduced here; see §9) | `5fda0d1856814b0582c07cc50fb2a42acb3275601ccb0b17e6a602d32545b89a` |
| `src/v2_io.py` (`resolve_benchmark` strict hash-bound loader) | `34ca0e74ac669061a6e0f9fd1758c87c034cbeebb3f4a5af7783aebdea524e72` |

C-B must load the benchmark only via `src/v2_io.py::resolve_benchmark`
(which itself re-verifies `LATEST_BENCHMARK.json`'s hash at load time) —
never by opening `data/frozen_v2/*.jsonl` directly by filename — so any
future re-freeze is caught rather than silently used.

### 7.2 Primary paired estimand — **exact sign convention, stated because it does not match an existing column**

For each valid R104 pair `i` and feature `f`:

```
delta_i^f = f(candidate_prompt_i) - f(source_prompt_i)
```

**This direction (candidate − source) is the one predeclared by the
task brief and used throughout this contract.** It is the *opposite*
sign of the existing `fightin_words_paired_difference` column already
present in `c_review_queue.csv`, which is computed as
`source − candidate` (verified directly: row 1 has
`fightin_words_source=10.017945`, `fightin_words_candidate=12.801672`,
stored `fightin_words_paired_difference=-2.783727`, i.e.
`source − candidate`; column mean −9.396, matching `3f_a`'s cited
−9.40). **C-B must negate the existing column** (or recompute from the
raw `fightin_words_source`/`fightin_words_candidate` fields) before
using it as `delta_i^{fightin\_words}` — using the stored column
unmodified would silently flip the sign of this one feature relative to
every other feature in the contract.

### 7.3 Primary populations (per task brief, unchanged)

1. All 104 valid accepted pairs.
2. `assistance_type_preserved=yes` subset (n=78).
3. `assistance_type_preserved=partial` subset (n=26) — reported
   separately, exploratory, never pooled with population 2 without an
   explicit flag (per `3f_a` §2.5's revision instruction).

No row is currently excludable on `review_status` (100% `accept`) or on
`researcher_harm_qc`/`wrapper_or_context_concern` (100% clean on both).

### 7.4 Missing-data rule

`low_coverage_flag_source`/`low_coverage_flag_candidate` are `False` for
all 104 rows in the current file (verified) — so no row is currently
dropped by this rule. If a future re-freeze introduces any `True` value,
that row is excluded from the **lexical-audit family only** (Fightin'
Words, CUE, lexical-diversity features below) but retained for the
**structural family** (word/character/sentence counts, formatting
indicators), since low token-recognition coverage degrades a
frequency-based lexical score but not a length/format count. This
asymmetric rule must be applied, not a blanket row-drop.

### 7.5 Feature families, classification, and directional hypothesis

| Feature | Definition (reuse existing column where marked) | Family | Directional hypothesis |
|---|---|---|---|
| `word_count` | existing `word_count_source`/`word_count_candidate` columns | structural | **None predeclared.** Per the task brief's explicit prohibition ("do not assume shorter wording = lower cue"), this feature carries no assumed sign relative to any cue claim. All significance tests below are two-sided. |
| `character_count` | existing `character_count_source`/`character_count_candidate` columns | structural | None (see above) |
| `sentence_count` | count of `[.!?]+` matches on raw (non-normalized) prompt text — identical rule to `3d_b`'s `multi_sentence_rule`, applied as a count rather than a ≥2 boolean | structural | None |
| `mean_word_length` | mean character length of tokens from `src/corpus_discrimination.py::word_tokenize` (`word_tokenize_v1_lower_alphanum_apostrophe`) | structural | None |
| list/numbered-step/code-block/multi-sentence indicators | exact regexes reused verbatim from `3d_b_lexical_outlierness_pilot.json`'s `formatting_diagnostic_config` (`bullet_marker_regex`, `numbered_step_regex`, `code_block_regex`, `multi_sentence_rule`), applied to raw (non-normalized) text | formatting/confound | None — reported as a required baseline family (source-code identical to `3f_a`'s Formatting/style-only baseline), not as a construct-relevant feature |
| `fightin_words` | existing `fightin_words_source`/`fightin_words_candidate` columns; paired delta **recomputed** per §7.2 (do not reuse the stored `fightin_words_paired_difference` column unmodified) | lexical-audit | None — `3f_a` §2.1 already found this score points in the direction opposite the `reduced_cue_source_rewrite` naming, so no direction is assumed |
| `fw_z_score` | existing `fw_z_score_source`/`fw_z_score_candidate` columns | lexical-audit | None |
| `lexical_diversity` (type-token ratio) | **no existing implementation in this repository** — C-B must define it fresh as `len(set(tokens)) / len(tokens)` using the same `word_tokenize` as above; report this as newly authored, not reused | lexical-audit | None |
| existing fixed lexical-risk lexicon (`src/diagnostics/score_lexical_risk_cues.py`) | reuse `score_prompt()` unmodified; report only aggregate hit-rate/hit-count per side, never the matched terms | lexical-audit | None — this lexicon already showed near-floor hit rates project-wide (`data/quadrant_composition_report.json`: C mean-cue-hits = 0 under a related lexicon), so it is retained only as a required baseline, not expected to carry the analysis |
| existing CUE score (`src/cue_scoring.py::TfidfLogRegCueScorer`) | reuse unmodified, scored per prompt | distributional/exploratory | **Explicitly not construct-relevant** — `3f_a` §0/§2.1 already disqualifies this score as `C_cue` ground truth (it is fit directly on the harmful/benign label). Reported for completeness only; a significant paired difference here must not be described as evidence about `C_cue` |

No feature above is assigned a "construct-relevant" direction, because
no existing artifact in this repository measures `C_cue` (§5) — assigning
one would repeat exactly the substitution `3f_a`/`3f_b` already warn
against.

### 7.6 Statistical contract (fixed seeds; predeclared before any implementation)

For each feature × population (3 populations × ~10 features = up to 30
tests):

1. **Descriptive statistics:** mean, median, sd, IQR, n, for source and
   candidate separately, plus the paired delta.
2. **Paired effect size:** Cohen's `d_z` = `mean(delta) / sd(delta)`.
3. **Bootstrap CI:** paired bootstrap — resample pair indices with
   replacement, `n_bootstrap = 10,000`, `seed = 20260901`, percentile
   95% CI on `mean(delta)`. (This seed is newly assigned for this
   contract; it is distinct from every seed already in use elsewhere in
   this repository — `45`, `271828`, `20260829`, `20260830`,
   `20260831` — so no shared-RNG-state assumption is possible.)
4. **Paired permutation/sign test:** sign-flip permutation test on
   `delta_i` (randomly negate each pair's delta with probability 0.5,
   recompute the mean, repeat), `n_permutations = 100,000`,
   `seed = 20260902`, two-sided empirical p-value against the observed
   mean delta — same method already used by `3d_h`'s own permutation
   test (`src/analysis/analyze_3d_h.py`), applied here to a different
   feature set and population.
5. **Multiple-comparison procedure:** Holm–Bonferroni, applied
   **separately within each of the 3 populations** (i.e. 3 independent
   families of ~10 tests each, not one pooled family of ~30) — because
   the 3 populations represent different predeclared evidence tiers
   (§7.3), not interchangeable repeated measurements of the same
   question.
6. **Category sensitivity:** compare `project_category` distribution
   (§1) between the delta's sign/magnitude — descriptive only (no
   formal test predeclared, given category is unbalanced 6–41 rows
   across 4 levels within R104 alone).
7. **Source sensitivity:** **not computable within R104** — R104 is
   100% StrongREJECT (§2), so there is no within-R104 source contrast
   to test. This must be reported as "not applicable — single source,"
   not skipped silently.
8. **Length/format sensitivity:** partial correlation (or, given n=104,
   a stratified comparison at the word-count median) between each
   lexical-audit feature's delta and the `word_count` delta, to check
   whether an apparent lexical effect is fully explained by the known
   length confound (`3f_a` §2.1/§2.2 precedent).

### 7.7 Decision framework (defined here; **not applied by this document**)

`KEEP FOR HUMAN REVIEW` / `KEEP AS SECONDARY` / `INCONCLUSIVE` / `DROP`,
to be assigned per feature × population by C-B based on: data integrity
(no missing-data exclusions triggered, §7.4), paired evidence strength
(bootstrap CI + corrected permutation p-value, §7.6), robustness in the
`assistance_type_preserved=yes` subset specifically (population 2 is the
least-caveated of the three), practical effect size (`d_z` magnitude,
not p-value alone — per the task brief, statistical significance alone
is never a sufficient `KEEP` criterion), and confound severity (§7.6
item 8). **No feature is assigned a label by this document** — that is
C-B's output, not this contract's.

### 7.8 Software/library versions (record actual runtime values; do not assume this sandbox's values transfer)

This inventory was produced with Python `3.12.3`, `numpy` `2.4.4`,
`scipy` `1.17.1`, `pandas` `3.0.2` (all read directly from this sandbox
at inspection time). `requirements.txt` pins only `numpy>=1.26`,
`scipy>=1.13`, `pandas>=2.2` — with no upper bound — so **C-B must record
its own actual runtime versions in its output JSON**; the values above
must not be assumed to match the researcher's local machine or Colab
environment, which is why this is listed as a required output field
(§7.9), not asserted as fixed.

### 7.9 Canonical output paths and exact command

Following the existing per-milestone convention
(`logs/3a4_scoring.{md,json}`, `logs/3d_b_lexical_outlierness_pilot.{md,json}`,
etc.):

- `logs/c_b_paired_delta_analysis.md` (human-readable report)
- `logs/c_b_paired_delta_analysis.json` (machine-readable: all
  descriptive stats, bootstrap CIs, permutation p-values, corrected
  p-values, sensitivity results, actual software versions, and this
  contract's input hashes re-verified at run time)

Exact command C-B must implement and run:

```
python -m src.analysis.c_b_paired_delta_analysis \
    --review-csv data/review/c_review_queue.csv \
    --benchmark-latest data/frozen_v2/LATEST_BENCHMARK.json \
    --gate-config logs/benchmark_gate_config.json \
    --formatting-config-source logs/3d_b_lexical_outlierness_pilot.json \
    --bootstrap-seed 20260901 --n-bootstrap 10000 \
    --permutation-seed 20260902 --n-permutations 100000 \
    --out-md logs/c_b_paired_delta_analysis.md \
    --out-json logs/c_b_paired_delta_analysis.json
```

C-B must fail closed (abort, not warn) if any hash in §7.1 does not
match at load time, mirroring the existing fail-closed pattern already
used by `src/data_pipeline/build_c_source_authored_candidates.py`.

---

## 8. Reproducibility contract — what is and is not established by this document

- **Reproducible specification:** everything in §7 — exact inputs and
  hashes, exact feature definitions, exact sign convention, exact
  statistical procedure, exact seeds and draw counts, exact output
  paths and command. This is complete and locked.
- **Implementation:** **does not exist.** No `src/analysis/c_b_paired_delta_analysis.py`
  module has been written by this task.
- **Numerical results:** **do not exist.** No delta, effect size,
  confidence interval, or p-value for any feature has been computed by
  this task. Every number in §§1–6 above is either a row/field count
  read directly from a current file, or a previously-computed number
  from an existing log verified to still match its recorded hash — none
  of it is a C-B result.

Code existing for a related purpose elsewhere in this repository (e.g.
`analyze_3d_h.py`'s permutation-test machinery) does not make this
contract's numbers reproducible until C-B's own module is written,
matches this spec exactly, and is run — that is future, separately-
authorized work.

---

## 9. Explicit non-actions

This task did **not**: implement any analysis code; modify
`data/frozen_v2/*`, `data/review/c_review_queue.csv`,
`data/review/c_source_authored_review_queue.csv`, `src/cue_scoring.py`,
or `src/corpus_discrimination.py`; create or rewrite any prompt; run any
model inference or GPU code; access the web; run a broad test suite;
begin C-B, C-C, or C-D; begin common-CUE or contrastive construction;
or make a final `KEEP`/`DROP`/etc. resource decision for R104,
R-AUTHORED, 3D-B, or 3D-H. It also did not reproduce the discriminative
lexicon in `src/diagnostics/score_lexical_risk_cues.py`, any raw prompt
text, any classifier weight, or any prompt-level rewrite pattern
anywhere above.

**Stop.**
