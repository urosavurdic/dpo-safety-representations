# 3D-A — Within-Harmful Lexical Outlierness Pilot: Design Specification

**Repository:** `https://github.com/urosavurdic/dpo-safety-representations`
**Branch:** `agent/c-quadrant-end-to-end-e0e2317a`
**Status:** Design-only. No scoring code implemented. No benchmark files,
`src/cue_scoring.py`, or `data/frozen_v2/*` touched. This document freezes
decisions for a future 3D-B implementation; it does not run anything.

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
quadrants). If a future task checks agreement between this measure's tails
and CUE's tails, agreement would be evidence of shared structure — not
proof the two constructs are identical, and not itself a claim that this
measure *is* CUE. This document does not compute CUE and does not modify
`src/cue_scoring.py`.

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
  `validation_3a3` / `exclusion_reasons` fields present on every row) using
  the same predicate 3A3 applied — never a redefinition — and must assert
  the resulting count is exactly 209, failing loudly otherwise.
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
/ central**.

**Locked preprocessing (shared with Method 2 unless noted):**
- Unicode NFKC normalization, then lowercasing.
- Collapse all internal whitespace runs to a single space; strip leading
  /trailing whitespace.
- URLs (`https?://\S+`) replaced with a single literal placeholder token
  `<url>`.
- Tokenizer: regex word-boundary tokens `\b\w+\b` on the normalized text.
  Numbers are kept as literal tokens (no `<NUM>` normalization).
  Apostrophes are not word characters under this regex (e.g. `"don't"` →
  `["don","t"]`); this artifact is accepted as-is, not special-cased.
- Stopwords: **not removed**, for both methods. Rationale: harmful-request
  scaffolding ("how do i", "write a", "explain how") is carried largely by
  function words and short formulaic phrases; removing them would discard
  exactly the signal most relevant to outlierness and to template
  detection.
- Features (Method 1 only): word **n-grams, range (1,2)** — unigrams and
  bigrams — over the tokens above. No min/max document-frequency pruning
  (`df_min=1`); IDF handles down-weighting of rare/common terms directly.
- TF definition: **raw within-document count**, no augmentation, no log
  scaling.

**Vocabulary leakage / policy: (B) group-local, LOGO.**
For held-out group `g = g(i)`, the reference set is
`R_{-g} = P \ {j : g(j) = g}` (the entire group, not just row `i`, is
excluded — per §4). Vocabulary `V_{-g}` = the set of distinct (unigram or
bigram) features occurring anywhere in `R_{-g}`. No predeclared external
vocabulary is used (policy A rejected: this pilot runs with no web access
and no external reference corpus is currently frozen in the repo; group-
local avoids inventing one). **Consequence, locked:** because
`V_{-g}`/`idf_{-g}` differ by fold, raw score magnitudes are not compared
across rows in different groups directly for anything beyond rank —
downstream use is via the within-pool percentile/tail definitions in §6,
which are computed after every row has its own LOGO score, exactly as
required.

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
**L2-normalize each document vector to unit length**. This normalize-first
step is applied identically to `v_i` and to every reference vector before
centroid construction.

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
  the row is **excluded from ranking/percentiles** and flagged
  `insufficient_lexical_overlap = true` in the 3D-B output table — never
  silently dropped, never imputed to an arbitrary numeric value.
- If `mu_i` is the zero vector (only possible if `V_{-g}` is empty, i.e.
  the held-out group spans nearly the whole 209-row pool): 3D-B must halt
  with an explicit assertion failure rather than emit a score. Expected
  not to occur given pool size and the grouping procedure in §4, but the
  behavior is specified rather than left implicit.

---

## 3. Method 2 — Leave-one-group-out smoothed token self-information

**Score:**
```
s_i = (1/n_i) * sum_{t in x_i} -ln( (c_{-g(i)}(t) + alpha) / (N_{-g(i)} + alpha*|V_{-g(i)}|) )
```

**Locked choices:**
- Unit: **word unigrams only** (not bigrams) — deliberately different
  granularity from Method 1, so the two methods are genuinely independent
  operationalizations rather than the same feature space scored two ways.
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
  (group-local, same LOGO exclusion as Method 1, same policy-B rationale).
- `c_{-g(i)}(t)` = raw **token-occurrence** count of `t` across all of
  `R_{-g(i)}` (a token-count quantity, distinct in kind from Method 1's
  document-frequency-based IDF — intentional, for method independence).
- `N_{-g(i)}` = total token count across `R_{-g(i)})` = `sum_t c_{-g(i)}(t)`.
- Stopwords contribute to both numerator and denominator counts (consistent
  with "not removed" above).

**Held-out-only / OOV tokens:** need no special case. A token unseen in
`R_{-g(i)}` simply has `c_{-g(i)}(t) = 0`, giving
`-ln( alpha / (N_{-g(i)} + alpha|V_{-g(i)}|) )` — the maximal
(hapax-equivalent) per-token surprisal under this smoothing. This falls
naturally out of add-alpha smoothing; no separate branch is needed.

**Zero-scored-token behavior:** impossible by construction — with
`alpha=1.0` and any non-degenerate reference, every term in the sum is
strictly positive and finite, so no `-ln(0)` case can occur.

**Degenerate `n_i = 0`** (prompt tokenizes to nothing after normalization):
`s_i = NaN`, excluded from ranking, flagged `empty_after_normalization =
true` — same treatment pattern as Method 1's zero-vector case. Expected
not to occur in this pool.

**Interpretation, aligned with Method 1:** **high `s_i`** = higher average
surprisal = **more lexically atypical**; **low `s_i`** = more typical/
predictable vocabulary. Both methods use the same high=atypical,
low=typical direction — required for §6–7 to compare tails meaningfully.

---

## 4. Duplicate / template grouping

**Already-frozen fields (reused, not recomputed):** `exact_duplicate_status`
/`exact_duplicate_canonical_record_id` and `normalized_duplicate_status`/
`normalized_duplicate_canonical_record_id`, computed and re-verified in
3A3 (`logs/3a3_validation.md`: 0 exact duplicates, 0 normalized-only
duplicates found, 0 mismatches on re-verification). 3D-B must seed its
union-find grouping from these two fields first (rows sharing a canonical
id are same-group by construction) — this currently contributes no merges
given the 0/0 result, but must stay wired for correctness if upstream data
changes.

**Not frozen:** near-duplicate/template grouping. 3A3 logged this as
`unknown` (blocked: no network access to load a sentence-transformers
model). It remains unavailable here too (this design stage runs with no
GPU and no web access). This pilot therefore defines one deterministic,
embedding-free procedure for 3D-B to execute and freeze **before** scoring:

1. Build a "similarity-normalized" text per row: NFKC + lowercase +
   whitespace-collapse (§2), **plus** strip all punctuation
   (`[^\w\s]` removed) — a text variant used only for grouping, never for
   scoring.
2. Represent each row as its set of character 5-gram shingles over that
   text (contiguous substrings of length 5, no padding).
3. Metric: Jaccard similarity of shingle sets,
   `J(x_i,x_j) = |S_i ∩ S_j| / |S_i ∪ S_j|`.
4. Threshold: **J ≥ 0.6** — locked in advance, chosen to catch templated/
   near-duplicate variants (e.g., one clause substituted) while not
   merging merely topically related but differently worded prompts. Not
   to be re-tuned after seeing lexical-outlierness scores.
5. Grouping: compute all `C(209,2) = 21,736` pairwise similarities (cheap,
   no GPU required), union any pair with `J ≥ 0.6` via deterministic
   union-find (pairs traversed in ascending `(record_id, record_id)`
   order for reproducible logging; the resulting connected components are
   order-independent regardless). Singleton rows form size-1 groups.
6. Groups are formed **once**, before any Method 1/2 scoring, using only
   the frozen 209-row pool and the locked threshold, and are never altered
   after seeing scores.
7. Freeze the result as a new artifact,
   `data/quadrant_c_pipeline/lexical_outlierness_groups_v1.json`
   (record_id → group_id, plus the threshold and shingle parameters used),
   committed with its own sha256 before scoring, and cited by hash in the
   3D-B report — this is the frozen group artifact the task requires,
   since none currently exists for near-duplicates.

The held-out group `g(i)` (per this grouping) is excluded from vocabulary,
counts, IDF, centroid, and every other reference statistic in §§2–3, for
both methods.

---

## 5. Preprocessing summary

All preprocessing is frozen in §§2–4 above (normalization, tokenizer,
n-gram ranges per method, stopword policy, URL handling, similarity-text
variant for grouping). No tokenization, normalization, filtering, or
feature transform may be introduced ad hoc in 3D-B; any change requires
updating this document first.

---

## 6. Tails

- **Percentile convention:** empirical CDF over the realized score
  distribution of the full 209-row pool (not the LOGO reference subset —
  the LOGO exclusion governs score *computation* only; percentiles
  describe each row's place in the whole observed pool). Matches the
  existing `empirical_rank` convention in `logs/3a4_scoring.md`:
  `percentile_i = |{ j in pool : s_j <= s_i }| / 209`.
- **Tie handling:** ties naturally receive identical percentile values
  under the "≤" definition above. Where a strict total order is needed
  (e.g. a fixed-size top-k cut), ties break by `(percentile, record_id)`
  ascending — same convention as `3a4_scoring.md`.
- **High tail:** `percentile_i ≥ 0.75`.
- **Low tail:** `percentile_i ≤ 0.25`.
- Because of ties, realized tail counts may not equal exactly
  `0.25 × 209 ≈ 52`; 3D-B must report the **actual** counts, not assume
  quartile counts.
- `NaN`-scored rows (§§2–3 edge cases) are excluded from percentiles and
  from both tails, and reported separately as "unscored."

---

## 7. Agreement / robustness statistics (3D-B must report)

- Spearman rank correlation between `s_tfidf` and `s_selfinfo` over all
  jointly-scored (non-NaN in both) rows.
- High-tail overlap coefficient: `|High_tfidf ∩ High_si| / min(|High_tfidf|, |High_si|)`.
- High-tail Jaccard: `|High_tfidf ∩ High_si| / |High_tfidf ∪ High_si|`.
- Low-tail overlap coefficient and Jaccard (symmetric definitions).
- Random-ranking/permutation baseline: for each of the four overlap/
  Jaccard statistics, permute one score vector against `record_id`
  10,000 times (fixed, logged seed) and report the observed value's
  position in that null distribution plus the null mean/SD.
- Bootstrap uncertainty: row-level bootstrap (resample 209 rows with
  replacement, 10,000 resamples, fixed logged seed) for Spearman rho and
  both Jaccards; report 95% percentile CIs.

These are evidence summaries only. No numeric GO/NO-GO threshold is
predeclared here (§10 governs how they're used).

---

## 8. Source-balanced sensitivity

Sources: StrongREJECT (132/209), SimpleSafetyTests (77/209).

Weight each reference row `j` in `R_{-g}` by
`w_j = 0.5 / (# reference rows in R_{-g} from j's source)`, so each
source contributes total weight 0.5 (sums to 1.0 overall). This is the
single, exact application of source balancing under the frozen §§2–3
design; no ambiguity with the frozen LOGO/vocabulary/group-exclusion
mechanics was identified (weighting only touches the reference
statistics below, not group membership or vocabulary membership):

- **Method 1:** `df_{-g}(t)` becomes the weighted count
  `sum_{j in R_{-g}, t in doc_j} w_j` (replacing the unweighted count in
  the same smoothed-IDF formula, with `|R_{-g}|` replaced by the total
  weight `1.0`). `mu_i^bal = sum_j w_j * v_j_normalized` (weights already
  sum to 1, no further division).
- **Method 2:** `c_{-g}^bal(t) = sum_j w_j * count(t, doc_j)`;
  `N_{-g}^bal = sum_t c_{-g}^bal(t)`. Same `alpha`, same `|V_{-g}|`, same
  formula otherwise.

**Report:** Spearman rank correlation between source-balanced and primary
(unbalanced) scores per method, plus a tail-membership confusion count
(does each row keep the same high/low/mid tail under balancing).
Source-balanced results are diagnostic only — the primary (unweighted)
score remains the one used for tails and the decision framework; balancing
must never be used to select a "nicer" primary result after the fact.

---

## 9. Confound diagnostics (3D-B must produce)

1. **Tail-by-source table** (StrongREJECT vs. SimpleSafetyTests counts in
   each tail, both methods).
2. **Tail-by-category table**, using the existing `harm_area` field
   already present on each row (closest existing field to "harm
   category"; no new label invented). If `harm_area` is too sparsely
   populated to be useful, fall back to `source_topic_category` and
   document which was used.
3. **Length/token-count summary:** distribution of `n_i` overall and by
   tail, plus Spearman correlation of `n_i` with each raw score (a direct
   length-confound check).
4. **Formatting summary:** deterministic regex-based counts (list markers,
   numbered steps, code-block delimiters, multi-sentence structure)
   tabulated by tail.
5. **Source-association/source-prediction diagnostic:** logistic
   regression predicting source (StrongREJECT vs. SimpleSafetyTests) from
   tail membership (high/low/mid) alone; report accuracy/AUC against a
   majority-class baseline.
6. **Source-balanced sensitivity** — §8.
7. **Category sensitivity** — same weighting mechanics as §8, substituting
   category for source; report the same rank-correlation/tail-membership
   comparison.
8. **Length sensitivity:** regress each raw score on `n_i` (linear
   regression over the scored pool); report rank correlation between raw
   and length-residualized scores, and whether tail membership changes.

3D-B must keep two conclusions visibly separate: (a) agreement statistics
(§7) are evidence the ranking is a **reproducible corpus-relative
measure**; (b) confound tables (this section) are evidence about
**suitability for downstream mechanistic comparison**. A method can pass
(a) and fail (b) at the same time; the report must not conflate the two.
No arbitrary post-hoc confounding threshold is predeclared — tables are
reported, dominance is a researcher judgment call.

---

## 10. Human validation — design only, not run now

- Sample: from the 209-row pool, an equal-count high/low sample (exact `n`
  fixed at 3D-B time from realized tail sizes, e.g. up to 20+20), drawn
  **source-balanced** within each tail (equal StrongREJECT/
  SimpleSafetyTests counts, subject to availability).
- Presentation: prompt text only; source, category, and numeric score are
  **hidden**; high- and low-tail rows are interleaved in a single,
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

- **Methods strongly disagree** (Spearman rho and tail-overlap statistics
  indistinguishable from, or below, the permutation baseline) → evidence
  against a stable within-harmful lexical-outlierness ranking →
  **STOP/REDESIGN**.
- **Methods agree** (rank correlation and tail overlap clearly above the
  permutation baseline) **but confound diagnostics (§9) show
  source/category/length/formatting clearly dominate the ranking** →
  **STOP/REDESIGN for downstream use**, even though the corpus-relative
  measure may still be "real."
- **Methods agree and diagnostics are acceptable** (no diagnostic reduces
  the tail split to a pure source/category/length classifier) → eligible
  to proceed to the (separately gated, not-yet-implemented) human
  validation in §10.
- **Human validation remains mandatory** regardless of the above before
  any claim that the ranking corresponds to surface explicitness of
  harmful intent.
- **3D-B alone never establishes CUE.** CUE is computed elsewhere
  (`src/cue_scoring.py`, between-quadrant TF-IDF+LogReg and Fightin'
  Words) and is out of scope here.
- No numerical GO/NO-GO threshold is predeclared in this document; the
  statistics in §§7–9 are reported and the proceed/stop call is left to
  the researcher, as instructed.

---

## 12. Explicit non-establishment statement

This document defines a **within-harmful, corpus-relative lexical-
outlierness measure**. It does not compute CUE, does not compare harmful
to benign text, does not modify `src/cue_scoring.py` or any frozen
benchmark/quadrant data, and any eventual agreement (or disagreement)
between its tails and CUE's tails is evidence about shared structure only
— never a claim of identity, and never sufficient on its own to justify
calling this measure "CUE."
