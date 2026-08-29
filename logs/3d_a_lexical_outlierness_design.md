# 3D-A — Within-Harmful Lexical Outlierness Pilot: Design Specification

**Repository:** `https://github.com/urosavurdic/dpo-safety-representations`

**Branch:** `agent/c-quadrant-end-to-end-e0e2317a`

**Status:** Design-only. No scoring code implemented. No benchmark files,
`src/cue_scoring.py`, or `data/frozen_v2/*` touched. This document freezes
decisions for a future 3D-B implementation; it does not run anything.

---

## Amendment history

This document has been amended twice since the initial 3D-A draft. Both
rounds are now fully integrated into the body sections below (§§3, 4, 6,
7, 8, 9) — nothing here is normative on its own; it is a changelog for
traceability only.

**Round 1:**
- Defined the weighted-empirical-CDF calibration rule used by balanced
  variants (now the general rule in §6, applied with `w_j = 1` for the
  primary variant and with balancing weights in §8/§9).
- Defined the multiplicity-weighted bootstrap Jaccard/Spearman for
  resampling with duplicate record IDs (now integrated directly into
  §7).

**Round 2 (this revision):**
- Replaced the invalid pooled-raw-score percentile in §6 with a
  fold-calibrated percentile computed against each held-out group's own
  fitted reference statistics. Raw scores from different folds use
  different vocabularies/statistics and were never comparable across
  the pool; §6 is the sole percentile definition now.
- §7 and §9 now operate on calibrated percentiles (`p_tfidf`,
  `p_selfinfo`), never on raw scores, except as within-fold intermediate
  arithmetic.
- Added the OOV-rate diagnostic (§9, item 9).
- §4: added canonical `group_id` assignment, defined Jaccard for
  empty shingle sets, and corrected the artifact-commit wording.
- §9: removed the `harm_area` → `source_topic_category` fallback and
  froze a single category-field policy; made category-balanced
  weighting explicit.
- §8: added halt-and-report behavior when a fold has zero rows from a
  required source.
- §3: replaced the "genuinely independent operationalizations" claim
  with a claim of complementarity, not statistical independence.

---

## 0. Relationship to CUE — read this first

This pilot defines (and, in 3D-B, will measure) **within-harmful lexical
outlierness**:

> How atypical a trusted harmful prompt's lexical composition is relative
> to the other trusted harmful prompts in the same trusted pool.

This is computed *entirely inside* the harmful population. It never
compares harmful vs. benign text, never uses a harmful/benign label, and
is not a classifier. It is a distinct, complementary operationalization
from CUE (`src/cue_scoring.py`: TF-IDF+LogReg and Fightin' Words *between*
quadrants). If a future task checks agreement between this measure's
tails and CUE's tails, agreement would be evidence of shared structure —
not proof the two constructs are identical, and not itself a claim that
this measure *is* CUE. This document does not compute CUE and does not
modify `src/cue_scoring.py`.

---

## 1. Reference population (frozen elsewhere; not re-derived here)

- Artifact: `data/quadrant_c_pipeline/c_source_authored_candidates_validated_v1.jsonl`
  — sha256 `8be074b8f42e87aa1c3eee83abedec9743e48ce1a65758e329118b34a6c37ca7`,
  413 rows.
- Reference pool for this pilot: the **209** rows flagged eligible by 3A3
  (`logs/3a3_validation.md`, `logs/3a3_validation.json`; eligible=209,
  excluded=204 — `overlaps_c_paired_pool`=155,
  `not_standalone_user_facing_request`=71). By source (`logs/3a4_scoring.md`):
  StrongREJECT=132, SimpleSafetyTests=77.
- 3D-B must select this exact subset by reading the per-record eligibility
  already recorded on each row of the validated_v1 artifact (the
  `validation_3a3` / `exclusion_reasons` fields present on every row)
  using the same predicate 3A3 applied — never a redefinition — and must
  assert the resulting count is exactly 209, failing loudly otherwise.
- 3A3 is sole authority for this population. This document does not
  reconstruct, re-audit, or reproduce any prompt text, row, or category
  label from it.

---

## 2. Method 1 — Leave-one-group-out TF-IDF centroid distance

**Score:**

```
s_tfidf(x_i) = 1 - cos(v_i, mu_i)
```

`cos(a,b) = (a·b) / (||a|| ||b||)`. Both vectors are non-negative
(TF-IDF with positive IDF), so `s_tfidf ∈ [0,1]`. **High = more lexically
atypical** relative to the rest of the harmful pool; **low = more typical
/ central**. This raw score is a within-fold quantity only — see §6 for
how it is converted into the calibrated percentile used everywhere
downstream.

**Locked preprocessing (shared with Method 2 unless noted):**

- Unicode NFKC normalization, then lowercasing.
- Collapse all internal whitespace runs to a single space; strip
  leading/trailing whitespace.
- URLs (`https?://\S+`) replaced with a single literal placeholder token
  `<url>`.
- Tokenizer: regex word-boundary tokens `\b\w+\b` on the normalized text.
  Numbers are kept as literal tokens (no `<NUM>` normalization).
  Apostrophes are not word characters under this regex (e.g. `"don't"` →
  `["don","t"]`); this artifact is accepted as-is, not special-cased.
- Stopwords: **not removed**, for both methods. Rationale: harmful-request
  scaffolding ("how do i", "write a", "explain how") is carried largely by
  function words and short formulaic phrases; removing them would
  discard exactly the signal most relevant to outlierness and to
  template detection.
- Features (Method 1 only): word **n-grams, range (1,2)** — unigrams and
  bigrams — over the tokens above. No min/max document-frequency pruning
  (`df_min=1`); IDF handles down-weighting of rare/common terms directly.
- TF definition: **raw within-document count**, no augmentation, no log
  scaling.

**Vocabulary leakage / policy: (B) group-local, LOGO.**

For held-out group `g = g(i)`, the reference set is
`R_{-g} = P \ {j : g(j) = g}` (the entire group, not just row `i`, is
excluded — per §4). Vocabulary `V_{-g}` = the set of distinct (unigram
or bigram) features occurring anywhere in `R_{-g}`. No predeclared
external vocabulary is used (policy A rejected: this pilot runs with no
web access and no external reference corpus is currently frozen in the
repo; group-local avoids inventing one). **Consequence, locked:** because
`V_{-g}`/`idf_{-g}` differ by fold, raw score magnitudes are never
compared across rows in different groups directly for anything — all
cross-row comparison (rank, tails, correlation, overlap) is done via the
fold-calibrated percentile in §6, computed after every row has its own
LOGO score.

**IDF definition (smoothed, sklearn-style):**

```
idf_{-g}(t) = ln( (1 + |R_{-g}|) / (1 + df_{-g}(t)) ) + 1
```

where `df_{-g}(t)` = number of documents in `R_{-g}` containing `t` at
least once. This form is bounded away from 0/∞ for terms present in none
or all reference docs.

**Document vector construction:**

For row `i` (held out, group `g`), and for every reference row `j` in
`R_{-g}`: build the raw TF-IDF vector restricted to `V_{-g}` (features of
the document that are not in `V_{-g}` are dropped — see OOV below), then
**L2-normalize each document vector to unit length**. This
normalize-first step is applied identically to `v_i` and to every
reference vector before centroid construction.

**Centroid construction — averaging happens after normalization (locked):**

```
mu_i = (1 / |R_{-g}|) * sum_{j in R_{-g}} v_j_normalized
```

`mu_i` is **not** re-normalized to unit length afterward. This is
immaterial to the score: cosine similarity is scale-invariant in each
argument, so renormalizing `mu_i` would not change `s_tfidf`. Stated
explicitly so 3D-B does not treat it as an open choice.

**OOV / zero-vector handling (no undefined edge cases):**

- Held-out-only tokens (present in `x_i`, absent from `V_{-g}`): dropped
  from `v_i`; contribute nothing.
- If `v_i` has no surviving features (zero vector): `s_tfidf(x_i) = NaN`,
  the row is **excluded from percentile calibration and ranking**
  (§6) and flagged `insufficient_lexical_overlap = true` in the 3D-B
  output table — never silently dropped, never imputed to an arbitrary
  numeric value.
- If `mu_i` is the zero vector (only possible if `V_{-g}` is empty, i.e.
  the held-out group spans nearly the whole 209-row pool): 3D-B must
  halt with an explicit assertion failure rather than emit a score.
  Expected not to occur given pool size and the grouping procedure in
  §4, but the behavior is specified rather than left implicit.

---

## 3. Method 2 — Leave-one-group-out smoothed token self-information

**Score:**

```
s_i = (1/n_i) * sum_{t in x_i} -ln( (c_{-g(i)}(t) + alpha) / (N_{-g(i)} + alpha*|V_{-g(i)}|) )
```

**Locked choices:**

- Unit: **word unigrams only** (not bigrams) — deliberately different
  granularity from Method 1. The two methods are complementary
  operationalizations of within-harmful lexical outlierness; agreement
  between them is evidence of robustness, not statistical independence,
  and is not proof that the constructs are identical.
- Tokenizer/normalization/stopword handling: identical to §2 (shared
  preprocessing), stopwords included.
- Repeated tokens **count repeatedly**: the sum is over token
  *occurrences* in `x_i`, not over `x_i`'s distinct vocabulary — matches
  the formula as given.
- `n_i` = total token count of `x_i` after tokenization (length in
  tokens), including any tokens that turn out OOV in the reference (see
  below — they still get a defined score, so they still count toward
  `n_i`).
- `alpha = 1.0` (add-one/Laplace smoothing), fixed for 3D-B.
- `V_{-g(i)}` = distinct unigram types observed anywhere in `R_{-g(i)}`
  (group-local, same LOGO exclusion as Method 1, same policy-B
  rationale). This same vocabulary is also the reference vocabulary for
  the shared OOV-rate diagnostic in §9.
- `c_{-g(i)}(t)` = raw **token-occurrence** count of `t` across all of
  `R_{-g(i)}` (a token-count quantity, distinct in kind from Method 1's
  document-frequency-based IDF — intentional, for method independence
  of feature construction, distinct from the complementarity point
  above about the resulting scores).
- `N_{-g(i)}` = total token count across `R_{-g(i)}` = `sum_t c_{-g(i)}(t)`.
- Stopwords contribute to both numerator and denominator counts
  (consistent with "not removed" above).

**Held-out-only / OOV tokens:** need no special case for scoring. A
token unseen in `R_{-g(i)}` simply has `c_{-g(i)}(t) = 0`, giving
`-ln( alpha / (N_{-g(i)} + alpha|V_{-g(i)}|) )` — the maximal
(hapax-equivalent) per-token surprisal under this smoothing. This falls
naturally out of add-alpha smoothing; no separate branch is needed for
the score itself. (The rate at which this occurs is separately reported
via the OOV-rate diagnostic, §9.)

**Zero-scored-token behavior:** impossible by construction — with
`alpha=1.0` and any non-degenerate reference, every term in the sum is
strictly positive and finite, so no `-ln(0)` case can occur.

**Degenerate `n_i = 0`** (prompt tokenizes to nothing after
normalization): `s_i = NaN`, excluded from percentile calibration and
ranking (§6), flagged `empty_after_normalization = true` — same
treatment pattern as Method 1's zero-vector case. Expected not to occur
in this pool.

**Interpretation, aligned with Method 1:** **high `s_i`** = higher
average surprisal = **more lexically atypical**; **low `s_i`** = more
typical/predictable vocabulary. Both methods use the same
high=atypical, low=typical direction — required for §§6–7 to compare
tails meaningfully.

---

## 4. Duplicate / template grouping

**Already-frozen fields (reused, not recomputed):**
`exact_duplicate_status`/`exact_duplicate_canonical_record_id` and
`normalized_duplicate_status`/`normalized_duplicate_canonical_record_id`,
computed and re-verified in 3A3 (`logs/3a3_validation.md`: 0 exact
duplicates, 0 normalized-only duplicates found, 0 mismatches on
re-verification). 3D-B must seed its union-find grouping from these two
fields first (rows sharing a canonical id are same-group by
construction) — this currently contributes no merges given the 0/0
result, but must stay wired for correctness if upstream data changes.

**Not frozen:** near-duplicate/template grouping. 3A3 logged this as
`unknown` (blocked: no network access to load a sentence-transformers
model). It remains unavailable here too (this design stage runs with no
GPU and no web access). This pilot therefore defines one deterministic,
embedding-free procedure for 3D-B to execute and freeze **before**
scoring:

1. Build a "similarity-normalized" text per row: NFKC + lowercase +
   whitespace-collapse (§2), **plus** strip all punctuation
   (`[^\w\s]` removed) — a text variant used only for grouping, never
   for scoring.
2. Represent each row as its set of character 5-gram shingles over that
   text (contiguous substrings of length 5, no padding).
3. Metric: Jaccard similarity of shingle sets,
   `J(x_i,x_j) = |S_i ∩ S_j| / |S_i ∪ S_j|`, with the explicit edge
   case `J(x_i,x_j) = 0.0` whenever either `S_i` or `S_j` is empty
   (including when both are empty) — this avoids an undefined `0/0`
   division and means empty-shingle rows (e.g., punctuation-only or
   sub-5-character normalized text) are never merged into another
   group by this rule alone.
4. Threshold: **J ≥ 0.6** — locked in advance, chosen to catch
   templated/near-duplicate variants (e.g., one clause substituted)
   while not merging merely topically related but differently worded
   prompts. Not to be re-tuned after seeing lexical-outlierness scores.
5. Grouping: compute all `C(209,2) = 21,736` pairwise similarities
   (cheap, no GPU required), union any pair with `J ≥ 0.6` via
   deterministic union-find (pairs traversed in ascending
   `(record_id, record_id)` order for reproducible logging; the
   resulting connected components are order-independent regardless).
   Singleton rows form size-1 groups.
6. `group_id` assignment: for each connected component, `group_id` is
   the lexicographically smallest `record_id` among its members. This
   is deterministic and independent of traversal order, so it does not
   need to be re-derived if the union-find implementation changes.
7. Groups are formed **once**, before any Method 1/2 scoring, using only
   the frozen 209-row pool and the locked threshold, and are never
   altered after seeing scores.
8. Freeze the result as a new artifact,
   `data/quadrant_c_pipeline/lexical_outlierness_groups_v1.json`
   (record_id → group_id, plus the threshold and shingle parameters
   used). Construct and hash this artifact before scoring, treat the
   hash as immutable for the duration of the run, and commit it
   together with the implementation and results in the single 3D-B
   commit — this is the frozen group artifact the task requires, since
   none currently exists for near-duplicates.

The held-out group `g(i)` (per this grouping) is excluded from
vocabulary, counts, IDF, centroid, and every other reference statistic
in §§2–3, for both methods.

---

## 5. Preprocessing summary

All preprocessing is frozen in §§2–4 above (normalization, tokenizer,
n-gram ranges per method, stopword policy, URL handling, similarity-text
variant for grouping). No tokenization, normalization, filtering, or
feature transform may be introduced ad hoc in 3D-B; any change requires
updating this document first.

---

## 6. Tails and calibrated percentiles (primary definition)

Raw scores `s_tfidf`, `s_selfinfo` are fold-local: each held-out group
`g` has its own fitted vocabulary/IDF/centroid or unigram counts (§§2–3),
so raw magnitudes are not comparable across rows in different groups.
**Pooling raw scores across the full 209-row sample and computing a
single empirical CDF over them, as an earlier draft of this document
did, is invalid** and is superseded by this section. The percentile
defined here is the sole percentile definition used anywhere in this
document.

**Fold-calibrated percentile.** For each held-out group `g` and each
method `m ∈ {tfidf, selfinfo}`:

1. Fit `m`'s reference statistics (vocabulary, IDF, centroid, or
   unigram counts, per §§2–3) on `R_{-g}`.
2. Score the held-out row using those fitted statistics: `s_{i,m}`.
3. Score every row `j` in `R_{-g}` using those *same* fitted statistics
   (each reference row is scored directly against fold `g`'s fitted
   reference, not re-fit with itself excluded).
4. Let `F_{g,m}` be the set of finite (non-NaN) scores obtained in step
   3, i.e. the reference distribution against which row `i` is
   calibrated.
5. Compute:

```
p_{i,m} =
  [ |{ r ∈ F_{g,m} : r < s_{i,m} }|
    + 0.5 · |{ r ∈ F_{g,m} : r = s_{i,m} }| ]
  / |F_{g,m}|
```

using `w_j = 1` for every reference row (the unweighted / primary
variant). Balanced variants (§8 source-balanced, §9 item 7
category-balanced) use the weighted version of the same rule:

```
p_{i,m}^bal =
  [ Σ_j w_j · 1(r_j < s_{i,m})
    + 0.5 · Σ_j w_j · 1(r_j = s_{i,m}) ]
  / Σ_j w_j
```

with `r_j` the balanced reference scores and `w_j` the fold's balancing
weights (§8/§9). If `s_{i,m}` is NaN, or `|F_{g,m}| = 0` (unweighted) /
`Σ_j w_j = 0` (weighted), `p_{i,m}` is **undefined** and reported as
such, never imputed. Report, per fold and method, the number of
reference rows and (for weighted variants) the total calibration weight.

**`p_{i,m}` — never raw `s_{i,m}` — is the quantity used for tails,
correlations, overlaps, permutation tests, bootstrap statistics, and
confound diagnostics throughout §§7 and 9.** Raw scores may appear only
as within-fold intermediate arithmetic (e.g., inside the percentile
computation itself) and must never be pooled or compared directly
across folds or reported as a cross-pool ranking.

**Tie handling:** ties within `F_{g,m}` naturally receive identical
`p_{i,m}` under the "<" / "=" split above. Where a strict total order is
needed (e.g. a fixed-size top-k cut), ties break by `(p_{i,m},
record_id)` ascending — same convention as `logs/3a4_scoring.md`.

**High tail:** `p_{i,m} ≥ 0.75`. **Low tail:** `p_{i,m} ≤ 0.25`. Tails
are method-specific (`High_tfidf` from `p_tfidf`, `High_si` from
`p_selfinfo`, etc.), consistent with their use in §7.

Because of ties, realized tail counts may not equal exactly
`0.25 × 209 ≈ 52`; 3D-B must report the **actual** counts, not assume
quartile counts. Rows with undefined `p_{i,m}` (§§2–3 edge cases) are
excluded from tails and reported separately as "unscored."

---

## 7. Agreement / robustness statistics (3D-B must report)

All statistics in this section are computed on the calibrated
percentiles `p_tfidf`, `p_selfinfo` defined in §6 — never on raw
scores.

- `Spearman(p_tfidf, p_selfinfo)` over all jointly-defined (non-NaN in
  both) rows.
- High-tail overlap coefficient:
  `|High_tfidf ∩ High_si| / min(|High_tfidf|, |High_si|)`.
- High-tail Jaccard: `|High_tfidf ∩ High_si| / |High_tfidf ∪ High_si|`.
- Low-tail overlap coefficient and Jaccard (symmetric definitions).
- Random-ranking/permutation baseline: for each of the four
  overlap/Jaccard statistics, permute one percentile vector against
  `record_id` 10,000 times (fixed, logged seed) and report the observed
  value's position in that null distribution plus the null mean/SD.
- **Bootstrap uncertainty:** row-level bootstrap — resample 209 rows
  with replacement, 10,000 resamples, fixed logged seed — for
  `Spearman(p_tfidf, p_selfinfo)` and both Jaccards, with the following
  multiplicity-aware definitions (resampling with replacement produces
  duplicate record IDs, so ordinary set-based Jaccard is undefined
  without this):
  - Retain the multiplicity `k_i` of each resampled record `i`.
  - Spearman is computed on the paired percentile observations
    (`p_tfidf`, `p_selfinfo`) with those multiplicities.
  - For each high- or low-tail Jaccard, use the full-sample tail
    membership defined in §6 and compute the multiplicity-weighted
    Jaccard:
    ```
    J =
      Σ_i k_i · 1(i ∈ T_1 ∩ T_2)
      / Σ_i k_i · 1(i ∈ T_1 ∪ T_2)
    ```
  - If the union has zero weight, that replicate is **undefined** and
    is reported, not imputed. Record the number of undefined bootstrap
    replicates.
  - Report 95% percentile CIs computed from the defined replicates
    only.

These are evidence summaries only. No numeric GO/NO-GO threshold is
predeclared here (§11 governs how they're used).

---

## 8. Source-balanced sensitivity

Sources: StrongREJECT (132/209), SimpleSafetyTests (77/209).

Weight each reference row `j` in `R_{-g}` by
`w_j = 0.5 / (# reference rows in R_{-g} from j's source)`, so each
source contributes total weight 0.5 (sums to 1.0 overall). This is the
single, exact application of source balancing under the frozen §§2–3
design; no ambiguity with the frozen LOGO/vocabulary/group-exclusion
mechanics was identified (weighting only touches the reference
statistics below and the calibration step, not group membership or
vocabulary membership):

- **Method 1:** `df_{-g}(t)` becomes the weighted count
  `sum_{j in R_{-g}, t in doc_j} w_j` (replacing the unweighted count in
  the same smoothed-IDF formula, with `|R_{-g}|` replaced by the total
  weight `1.0`). `mu_i^bal = sum_j w_j * v_j_normalized` (weights
  already sum to 1, no further division).
- **Method 2:** `c_{-g}^bal(t) = sum_j w_j * count(t, doc_j)`;
  `N_{-g}^bal = sum_t c_{-g}^bal(t)`. Same `alpha`, same `|V_{-g}|`,
  same formula otherwise.

**Calibration:** the balanced percentile `p_{i,m}^bal` uses the
weighted rule defined in §6, with these same `w_j`, applied to the
balanced reference scores `r_j`. Report the number of reference rows
and the total calibration weight for each fold; if the total weight is
zero, report the percentile as undefined.

**Absent-source behavior:** if `R_{-g}` contains zero rows from either
required source (StrongREJECT or SimpleSafetyTests), source-balanced
scoring for that fold **halts with an assertion failure** and is
reported as undefined for that fold/row. No renormalization over the
remaining source is permitted — this would silently change what
"balanced" means for that fold.

**Report:** Spearman rank correlation between source-balanced and
primary (unbalanced) percentiles per method, plus a tail-membership
confusion count (does each row keep the same high/low/mid tail under
balancing). Source-balanced results are diagnostic only — the primary
(unweighted) percentile remains the one used for tails and the decision
framework; balancing must never be used to select a "nicer" primary
result after the fact.

---

## 9. Confound diagnostics (3D-B must produce)

1. **Tail-by-source table** (StrongREJECT vs. SimpleSafetyTests counts
   in each tail, both methods).
2. **Tail-by-category table**, using the category field policy below.
3. **Length/token-count summary:** distribution of `n_i` overall and by
   tail, plus Spearman correlation of `n_i` with each method's
   calibrated percentile (`p_tfidf`, `p_selfinfo`) — a direct
   length-confound check.
4. **Formatting summary:** deterministic regex-based counts (list
   markers, numbered steps, code-block delimiters, multi-sentence
   structure) tabulated by tail.
5. **Source-association/source-prediction diagnostic:** logistic
   regression predicting source (StrongREJECT vs. SimpleSafetyTests)
   from tail membership (high/low/mid) alone; report accuracy/AUC
   against a majority-class baseline.
6. **Source-balanced sensitivity** — §8, including the weighted
   calibration defined in §6/§8.
7. **Category sensitivity** — same weighting mechanics as §8
   (including the weighted calibration from §6), substituting category
   for source: each category present in `R_{-g}` receives total weight
   `1/K`, where `K` is the number of distinct categories present in
   `R_{-g}` under the category field policy below. Report the same
   rank-correlation/tail-membership comparison as §8, on percentiles.
8. **Length sensitivity:** regress each method's calibrated percentile
   on `n_i` (linear regression over the scored pool); report the rank
   correlation between the length-residualized percentile and the
   original calibrated percentile, and whether tail membership changes.
9. **OOV-rate diagnostic:** define, for row `i` in held-out group
   `g(i)`, using the Method 2 unigram vocabulary `V_{-g(i)}` (§3) as
   the shared reference vocabulary for this diagnostic across both
   methods:
   ```
   oov_rate_i =
     ( unseen normalized token occurrences in x_i )
     / ( total normalized token occurrences in x_i )
   ```
   This diagnostic does not change how either method treats OOV
   tokens: Method 1 continues to drop OOV features from `v_i` (§2),
   and Method 2 continues to assign OOV tokens the smoothed maximum
   surprisal (§3); `oov_rate_i` is reported alongside, not substituted
   into, either score. Report the Spearman correlation of `oov_rate_i`
   with each method's calibrated percentile (`p_tfidf`, `p_selfinfo`).

**Category field policy (frozen, no fallback):**
- Primary and only field: `harm_area`.
- Missing or blank values are recorded as the literal category
  `unknown` — never dropped and never treated as missing data for
  grouping/weighting purposes.
- Small categories are retained as-is; there is no minimum-count
  threshold and no merging of small categories into an "other" bucket.
- Rows are never dropped for having a missing or unusual category
  value.
- The field is never switched after seeing lexical-outlierness results
  (there is no fallback to `source_topic_category` or any other field).
- If `harm_area` turns out to be uninformative for this pool (e.g., a
  single value across all 209 rows, or values that do not distinguish
  any tail composition), 3D-B must report this explicitly as a finding,
  rather than substituting a different field to manufacture a more
  informative table.

3D-B must keep two conclusions visibly separate: (a) agreement
statistics (§7) are evidence the ranking is a **reproducible
corpus-relative measure**; (b) confound tables (this section) are
evidence about **suitability for downstream mechanistic comparison**. A
method can pass (a) and fail (b) at the same time; the report must not
conflate the two. No arbitrary post-hoc confounding threshold is
predeclared — tables are reported, dominance is a researcher judgment
call.

---

## 10. Human validation — design only, not run now

- Sample: from the 209-row pool, an equal-count high/low sample (exact
  `n` fixed at 3D-B time from realized tail sizes, e.g. up to 20+20),
  drawn **source-balanced** within each tail (equal
  StrongREJECT/SimpleSafetyTests counts, subject to availability).
- Presentation: prompt text only; source, category, and numeric score
  are **hidden**; high- and low-tail rows are interleaved in a single,
  fixed-seed randomized order so raters cannot infer tail from position.
- Rating target: **surface explicitness of harmful operational intent
  only** (e.g. a short ordinal scale, "how explicit/operational vs.
  vague/abstract the request reads on its face") — raters are not asked
  to judge actual harmfulness, severity, or realism.
- This check is a precondition for claiming the ranking tracks the
  intended surface-explicitness construct, and is mandatory before this
  measure is ever called "CUE." **Not implemented or run in 3D-A or
  3D-B.**

---

## 11. Decision framework

- **Methods strongly disagree** (`Spearman(p_tfidf, p_selfinfo)` and
  tail-overlap statistics indistinguishable from, or below, the
  permutation baseline) → evidence against a stable within-harmful
  lexical-outlierness ranking → **STOP/REDESIGN**.
- **Methods agree** (rank correlation and tail overlap clearly above
  the permutation baseline) **but confound diagnostics (§9) show
  source/category/length/formatting clearly dominate the ranking** →
  **STOP/REDESIGN for downstream use**, even though the corpus-relative
  measure may still be "real."
- **Methods agree and diagnostics are acceptable** (no diagnostic
  reduces the tail split to a pure source/category/length classifier)
  → eligible to proceed to the (separately gated, not-yet-implemented)
  human validation in §10.
- **Human validation remains mandatory** regardless of the above
  before any claim that the ranking corresponds to surface
  explicitness of harmful intent.
- **3D-B alone never establishes CUE.** CUE is computed elsewhere
  (`src/cue_scoring.py`, between-quadrant TF-IDF+LogReg and Fightin'
  Words) and is out of scope here.
- No numerical GO/NO-GO threshold is predeclared in this document; the
  statistics in §§7–9 are reported and the proceed/stop call is left to
  the researcher, as instructed.

---

## 12. Explicit non-establishment statement

This document defines a **within-harmful, corpus-relative
lexical-outlierness measure**. It does not compute CUE, does not
compare harmful to benign text, does not modify `src/cue_scoring.py` or
any frozen benchmark/quadrant data, and any eventual agreement (or
disagreement) between its tails and CUE's tails is evidence about
shared structure only — never a claim of identity, and never
sufficient on its own to justify calling this measure "CUE."