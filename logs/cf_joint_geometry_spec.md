STATUS: DESIGN-ONLY — locked analysis contract. No analysis code implemented,
no audit run, no embeddings computed, no benchmark data modified, no
R104/R-AUTHORED records touched, no contrastive/representation-learning
training started. This document is the only output.

# C-F-A — Joint-Geometry Analysis Contract for A/B/C/D (+ R104-source, R-AUTHORED)

**Repository:** `https://github.com/urosavurdic/dpo-safety-representations`
**Branch:** `agent/c-quadrant-end-to-end-e0e2317a`
**Git commit inspected:** `b41f924f39f9422798e90e5b6e79f33c0eb3c895` (HEAD at
inspection time — the C-E commit; C-A through C-E are present and, per the
hash re-verification in §1, unchanged since C-A pinned them).
**Task type:** design-only specification. No implementation module exists
for any part of this document.

---

## 0. What this document is and is not

This is the specification C-F-B (a future, separately-authorized task) must
implement without deviation, in the same sense C-A §7 was the specification
C-B implemented. It answers the five C-F questions from the task brief at
the *design* level only:

1. How distinct are A/B/C/D in one fixed feature space? → §5.1–5.3
2. Is C closer to D than A, on relevant surface/structural dimensions? → §5.1, §5.6
3. Do the observed relationships support the intended 2×2 interpretation at all? → §5.2, §5.4
4. Are the relationships dominated by length, source, category, or formatting? → §5.6, §5.3 (three views)
5. Where does R-AUTHORED sit in the same space? → §6

No number in this document is a result. Every threshold, seed, formula, and
file path below is a predeclaration, not an observation.

---

## 1. Input files, hashes, and freshness verification

The following hashes were **re-verified directly against the current
working tree at the commit above** (not copied from C-A without checking —
all match C-A §7.1's pinned values exactly, so no drift has occurred
between C-A and this task):

| Path | sha256 | Verified against C-A |
|---|---|---|
| `data/frozen_v2/benchmark_v2_20260826T212909Z.jsonl` | `e4946b070f441c7a0676db830c65257b78a2d1b46abb0a61cce4cc86352f838b` | match |
| `data/frozen_v2/LATEST_BENCHMARK.json` | `817885c1c50dcbb5babddaec05b938f0f47067151ababa3c669e893f38ea937a` | match |
| `data/processed/controlled_eval.jsonl` | `e640c2fba47afe2853c8717ae8492c62bf26cce21f6ec677f68ea88b117c05af` | match |
| `data/review/c_review_queue.csv` (R104-source, `source_prompt` column) | `8f6dfba182e5d3595d9ac6292d13956dd1a027b18770da01f4ef510f236787bb` | match |
| `data/review/c_source_authored_review_queue.csv` (R-AUTHORED) | `c62725ec37b3d950d7fda164c89d6e71a315cc5dad1a701147100c9f8e8e485a` | match |
| `logs/benchmark_gate_config.json` | `1ac73585f08a4d685996c96eecafdbfcc74478ab07f7f762a4a9de2b2568b743` | match |
| `logs/3d_b_lexical_outlierness_pilot.json` (source of `formatting_diagnostic_config`) | `95b0b7771244f0c162627eb1aaeb92986b4e7ec9de737f4f38edaefec53ebce5` | match |
| `src/corpus_discrimination.py` (tokenizer, `FightinWords`, `load_quadrant_texts`) | `1ca62c4f7c1f88398c2d22c60bc1f2f6be27be678b68e9675a8800bdb41a9bcc` | match |
| `src/cue_scoring.py` (source of `FROZEN_CUE_CONFIG` TF-IDF parameters — parameters reused, fitted model **not** reused, §4.2) | `ea0aa39faee7f8358121cee460be3c3f45d840c555beaba8ee534da2033b7d1d` | match |
| `src/diagnostics/score_lexical_risk_cues.py` (`score_prompt()`, reused unmodified; lexicon not reproduced here) | `5fda0d1856814b0582c07cc50fb2a42acb3275601ccb0b17e6a602d32545b89a` | match |

C-F-B must load the benchmark only via `src/v2_io.py::resolve_benchmark`
(never by opening `data/frozen_v2/*.jsonl` by filename), fail closed if any
hash above differs at load time, and record its own actually-observed
hashes in its output JSON rather than assuming this table stays current.

---

## 2. Population definitions (exact roles, per onboarding table)

| Label | n | Text field(s) | Role in this spec |
|---|---|---|---|
| A | 150 | `prompt` (frozen benchmark) | Quadrant, used in centroid/contrast/PCA/distribution-distance |
| B | 250 | `prompt` (frozen benchmark) | Quadrant, used in centroid/contrast/PCA/distribution-distance |
| C | 104 | `prompt` (frozen benchmark) — **identical population to R104 candidate side**; not a separate fit or a separate text source | Quadrant, used in centroid/contrast/PCA/distribution-distance |
| D | 150 | `prompt` (frozen benchmark) | Quadrant, used in centroid/contrast/PCA/distribution-distance |
| R104-source | 104 | `source_prompt` column, `data/review/c_review_queue.csv` | Auxiliary, projection-only, never in A/B/C/D contrasts |
| R-AUTHORED | 52 | `candidate_prompt` column, `data/review/c_source_authored_review_queue.csv` | Auxiliary, projection-only, never in A/B/C/D contrasts |

**Required pre-implementation check (C-F-B, not this document):** confirm
directly, before scoring, that R-AUTHORED's existing `word_count` /
`character_count` / `fightin_words_score_normalized` columns in the review
CSV were computed on `candidate_prompt` and not `source_prompt` — the
column names alone do not settle this, and C-E did not need to check it
because C-E never joined R-AUTHORED against a text-derived feature computed
elsewhere. If the check fails, C-F-B must recompute `word_count` /
`character_count` fresh from `candidate_prompt` rather than reuse the
existing columns, and must record which branch was taken.

**Fit population vs. projected population — the single most consequential
design choice in this document:** the representation in §4 is fit **only**
on the pooled A∪B∪C∪D rows (n=654, the frozen benchmark). R104-source
(n=104) and R-AUTHORED (n=52) are transformed (not fit) into that space.
This is stricter than the task brief's literal wording — the brief forbids
fitting on R-AUTHORED but does not explicitly forbid fitting on
R104-source — because R104-source is explicitly excluded from the
A/B/C/D-contrast definition in the same onboarding table, and a fit that
silently included it would let an auxiliary population influence the
geometry the quadrant contrasts are read against. This choice is stated
here precisely so C-F-B cannot silently substitute a different fit
population later.

---

## 3. Category / source field availability (verified directly, not assumed from C-A)

Direct inspection of the frozen benchmark's `source_dataset`,
`project_category`, and `source_category` fields per quadrant (re-run at
this task's commit, not copied from any prior log):

| Quadrant | `source_dataset` | `project_category` | `source_category` |
|---|---|---|---|
| A | 100% HarmBench | populated, 4-level (`illegal` 57, `cybercrime_intrusion` 40, `misinformation_disinformation` 34, `harassment_bullying` 19) | identical to `project_category` |
| B | 100% XSTest | populated, **disjoint 10-level taxonomy** (`homonyms`, `figurative_language`, `safe_targets`, `safe_contexts`, `definitions`, `nons_group_real_discr`, `real_group_nons_discr`, `historical_events`, `privacy_public`, `privacy_fictional`; 25 each) | identical to `project_category` |
| C | 100% StrongREJECT | populated, same 4-level taxonomy as A but different proportions (`harassment_bullying` 41, `misinformation_disinformation` 37, `cybercrime_intrusion` 20, `illegal` 6) | different raw label strings (e.g. `"Hate, harassment and discrimination"`) mapped onto the same 4-level `project_category` |
| D | Alpaca 50 / Dolly-15k 50 / OASST1 50 | **null for all 150 rows** | null for all 150 rows |

This is a direct correction of one imprecision in C-A §1, which described
`project_category` as populated "for A and C (not for B/D, which use their
own category field instead)." Direct inspection shows `project_category`
**is** populated for B (a disjoint 10-level taxonomy, not the harm-category
one) and **is not** populated for D at all; there is no separate `category`
field anywhere in the schema. This does not change any C-A/C-B/C-D/C-E
conclusion — none of those documents used B's or D's category values — but
it directly determines what is and is not computable in §5.6 below:

- **A vs. C category-stratified sensitivity:** computable — same label set.
- **B, D category sensitivity:** **not applicable.** B's taxonomy shares no
  labels with A/C's; D has no category values at all. Report as "not
  applicable," per the same fail-closed convention C-A used for R104's
  single-source non-applicability (§7.6 item 7 there) — never silently
  omit the row.
- **Within-quadrant source sensitivity:** computable only for D (3 sources).
  A, B, C are each 100% one upstream dataset, so `source_dataset` is
  perfectly confounded with quadrant membership for three of the four
  quadrants — this is a structural property of the benchmark, not an
  artifact of this analysis, and it is the single largest reason PCA/
  centroid separation between quadrants must not be read as evidence for
  the intended intent/cue construct (a dataset-of-origin effect would
  produce the same separation).

---

## 4. Primary common representation

Fit **once** on A∪B∪C∪D (n=654). R104-source and R-AUTHORED are transformed
using the fitted parameters from this step, never re-fit (§2).

### 4.1 Structural feature block

| Feature | Definition | Source |
|---|---|---|
| `word_count` | existing frozen-benchmark column | reused as-is |
| `character_count` | existing frozen-benchmark column | reused as-is |
| `sentence_count` | count of `[.!?]+` matches on raw `prompt` text | `3d_b`'s `multi_sentence_rule`, applied as a count (identical rule to C-A §7.5) |
| `mean_word_length` | mean character length of tokens from `word_tokenize` (`word_tokenize_v1_lower_alphanum_apostrophe`) | `src/corpus_discrimination.py::word_tokenize`, reused unmodified |
| `lexical_diversity` | `len(set(tokens)) / len(tokens))`, same tokenizer | C-B's definition (C-A §7.5), reused, not redefined |
| `has_bullet_marker`, `has_numbered_step`, `has_code_block`, `multi_sentence_flag` | exact regexes from `3d_b_lexical_outlierness_pilot.json`'s `formatting_diagnostic_config` (`bullet_marker_regex`, `numbered_step_regex`, `code_block_regex`, `multi_sentence_rule≥2`), applied to raw `prompt` text | reused verbatim, byte-identical to C-A/C-B |
| `lexical_risk_hit_count` | `score_prompt(text)` hit count | `src/diagnostics/score_lexical_risk_cues.py`, reused unmodified; matched terms never reported (§9) |

**Zero-variance exclusion rule (predeclared, not post-hoc):** any feature
above that is zero-variance across the A∪B∪C∪D fit population is dropped
from the structural block before standardization, and the drop is logged
by name — this is expected for `has_numbered_step` and `has_code_block`
given C-D's finding that these are floor-effect/zero-variance within R104
already; the same check must be re-run on the full A∪B∪C∪D pool rather
than assumed to transfer.

**Standardization:** z-score each surviving feature using the mean/sd
computed on the A∪B∪C∪D fit population only; apply the same mean/sd
(never refit) when transforming R104-source and R-AUTHORED.

### 4.2 Lexical feature block

Two TF-IDF views, using the **parameter values** already locked in
`src/cue_scoring.py::FROZEN_CUE_CONFIG["tfidf_logreg"]` — reusing the
predeclared n-gram ranges, `min_df`, and weighting choices already
justified and frozen for this repository, **not** reusing the fitted
LogisticRegression/LOSO-scored model itself, which was fit for a different
task (harmful/benign classification under leave-one-source-out folds) on a
different, non-A∪B∪C∪D corpus:

| Parameter | Value (reused from `FROZEN_CUE_CONFIG`) |
|---|---|
| Word analyzer | `word`, n-gram range `(1, 2)`, `min_df=2` |
| Char analyzer | `char_wb`, n-gram range `(3, 5)`, `min_df=2` |
| `lowercase` | `True` |
| `sublinear_tf` | `True` |
| `max_features` per view | `20000` |
| Row normalization | L2 (`TfidfVectorizer` default), applied per view |

Fit both vectorizers on A∪B∪C∪D `prompt` text only (n=654); `transform()`
(never `fit_transform()`) on R104-source and R-AUTHORED. Concatenate the
word-ngram and char-ngram sparse matrices to form the raw lexical block.

**Dimensionality reduction (required before any dense PCA step, §5.3):**
the raw lexical block is high-dimensional relative to n=654 (up to 40,000
columns). Reduce via `TruncatedSVD` (not randomized full PCA) directly on
the sparse TF-IDF block: `n_components = min(50, n_fit_rows − 1) = 50`,
`algorithm='arpack'`, `random_state=20260905` (new seed — not previously
used anywhere in this repository, §8). This is a standard, deterministic,
sparse-safe dimensionality-reduction step (equivalent to LSA on this
corpus), reported as such — it is not itself the "PCA is a visualization
only" step required in §5.3, which operates on the already-reduced,
concatenated representation below.

### 4.3 Fightin' Words — a fresh, single common fit (required; do not reuse existing columns)

C-E §3 already established that R104's `fightin_words_score_normalized`
(fit LOSO with StrongREJECT held out of H, per C-B) and R-AUTHORED's
`fightin_words_score_normalized` (fit with H=A∪B, D=quadrant D, per
`3a4_scoring.md`) use **different fitted references and are not
comparable**. Neither existing column is usable in a common joint space.
This spec instead requires a **third, single, common fit**, reusing only
the `FightinWords` class from `src/corpus_discrimination.py` unmodified:

- `H = A ∪ B`, `D = quadrant D` — identical convention to
  `build_fw_from_eval`'s existing default (the same H/D split already used
  to originally score R-AUTHORED), chosen because it is already the
  repository's established baseline pairing, not a new invention.
- `prior_strength=0.01`, `min_count=1` (existing defaults, unmodified).
- Score **all six populations** (A, B, C, D, R104-source, R-AUTHORED) with
  this one fitted instance — including A, B, and D themselves, which have
  no existing Fightin'-Words score at all in the frozen benchmark (word_count/
  character_count are the only pre-computed lexical-adjacent columns there).
- Output field name: `fw_score_common_v1` (a new name, precisely so it is
  never confused with either existing, differently-fitted
  `fightin_words_score_normalized` column — reusing that name here would
  silently misrepresent which reference it was fit against).

This is the one place in this spec where a "reused, existing" component
(the `FightinWords` class and its H/D convention) is applied to a genuinely
new fit — flagged explicitly because §7.5 of C-A's contract required the
opposite (reuse existing *scored columns* wherever possible) for a
different, paired-only analysis; the constraint here (one common space for
six populations) makes reuse of any of the three currently-existing,
mutually-incomparable fits impossible by construction.

### 4.4 Three representation views (combining §4.1–§4.3)

| View | Contents |
|---|---|
| **Structural-only** | §4.1 z-scored block only (plus `fw_score_common_v1`, which is lexical in content but is included here only if a variant of this view is run *without* it — primary structural-only view **excludes** `fw_score_common_v1`; see below) |
| **Lexical-only** | §4.2 50-dim SVD block, concatenated with `fw_score_common_v1` (z-scored using the A∪B∪C∪D fit population's mean/sd) and `lexical_risk_hit_count` (already in §4.1 — duplicated here deliberately, since it is a lexical-content feature; C-A itself classifies it "lexical-audit," §7.5) |
| **Combined** | Row-wise unit-L2-normalize the structural-only block and the lexical-only block **separately**, then concatenate as `sqrt(0.5) · structural_normalized ⊕ sqrt(0.5) · lexical_normalized` — an explicit, predeclared, equal-weighting choice, not a claim that structural and lexical evidence deserve equal weight substantively. This weighting is arbitrary-but-fixed; it exists only so the combined view is well-defined, and §5.6/§5.3's separate structural-only/lexical-only views (not the combined view) are the primary tools for answering question 4 (dominance by length/source/category/formatting). |

`lexical_risk_hit_count` appearing in both §4.1's structural block and the
lexical-only view is intentional, not a duplication bug — it is retained in
the structural z-scored block for the combined view's structural half (it
is, after all, a per-row count like the other structural features) and
also surfaced alone in the lexical-only view, consistent with its
C-A-assigned "lexical-audit" family label. C-F-B's output must state this
choice explicitly rather than let a reviewer assume it is an oversight.

### 4.5 Empty / OOV row handling

Any row whose `word_tokenize(prompt)` returns zero tokens is retained in
the structural-only view (word_count/character_count/sentence_count remain
well-defined at 0) but **excluded** from the lexical-only and combined
views (undefined TF-IDF row, undefined `fw_score_common_v1` token
recognition), mirroring C-A §7.4's asymmetric low-coverage rule. C-F-B must
report the exact count of excluded rows per population (expected: 0, given
none of A/B/C/D/R104-source/R-AUTHORED's existing `word_count` columns are
0, but this must be verified, not assumed).

---

## 5. Required analysis blocks

### 5.1 Centroid geometry

Compute the A, B, C, D centroids (mean vector) in each of the three views
(§4.4) and all six pairwise Euclidean distances among them. **Equal
quadrant weighting**: each quadrant contributes one centroid regardless of
n (150/250/104/150) — do not pool rows before centroiding, and do not
weight the six pairwise distances by `n_i · n_j` or any other
sample-size-derived quantity. R104-source and R-AUTHORED centroids are
computed and reported in the same table, clearly separated from the six
A/B/C/D pairs, never mixed into an implied seventh/eighth quadrant.

### 5.2 Factorial contrasts

```
intent_contrast  = (mu_A + mu_C)/2 − (mu_B + mu_D)/2
surface_contrast = (mu_A + mu_B)/2 − (mu_C + mu_D)/2
```

computed in each of the three views, using the equal-weighted centroids
from §5.1 (never row-pooled means). **These vectors are meaningful only as
descriptive contrasts between the four equal-weighted centroids computed in
a fixed, already-specified feature space** — they are not meaningful as, and
must not be reported as, latent axes recovered by an unsupervised method,
since both are constructed directly from the quadrant labels, not
discovered. Report the cosine angle between `intent_contrast` and
`surface_contrast` as one descriptive scalar per view (3 values total). Per
the task brief, this angle must be reported as a **geometric fact about
this specific fitted representation** (e.g. "in the structural-only view at
this fit, the two contrast vectors are approximately orthogonal / at
constant X°") and explicitly must not be characterized as evidence for or
against latent psychological independence of intent and surface cue — that
inferential step is outside what a cosine angle between two
label-constructed contrast vectors can support, regardless of its value.

### 5.3 PCA (visualization only)

- **Feature matrix:** the combined view (§4.4), all six populations
  projected (A/B/C/D fit rows + R104-source + R-AUTHORED transform rows).
- **Scaling:** none beyond what §4.4 already applies (the combined view is
  already unit-normalized per block) — do not re-standardize a second time.
- **Method:** deterministic PCA (`sklearn.decomposition.PCA`,
  `svd_solver='full'`, no random seed required — full SVD has no
  stochastic component) fit on the A∪B∪C∪D rows only; R104-source and
  R-AUTHORED projected via the fitted components, never included in the
  PCA fit itself (mirrors §2's fit/transform split).
- **Components displayed:** PC1–PC2 as the primary 2-D plot (matching the
  researcher's stated 2-D intuition, §"Why C-F exists" in onboarding), plus
  a PC1–PC3 secondary plot; report cumulative explained-variance ratio for
  both so the plot's limitations are visible alongside it.
- **Populations displayed:** all six, with R104-source and R-AUTHORED
  drawn in a visually distinct marker style (e.g. open markers vs. filled)
  from the start — never presented as if they were a fifth/sixth quadrant.
- **Explicit non-criterion:** a visually clean four-cluster PC1–PC2 plot is
  not, by itself, evidence for the intended construct, and its absence is
  not, by itself, evidence against it — PCA here is reporting, not testing.

### 5.4 Distribution distance (one primary method, predeclared)

**Statistic:** energy distance (Székely & Rizzo), computed once per view
(structural-only, lexical-only, combined) for each of the six unordered
A/B/C/D pairs (AB, AC, AD, BC, BD, CD) — 18 statistic values total.

**Null/permutation procedure:** label-permutation test. For a given pair
(e.g. A vs. C), pool the two populations' rows in the given view, then
repeatedly (i) randomly reassign each pooled row to "A" or "C" preserving
the original group sizes (150/104), (ii) recompute the energy-distance
statistic, (iii) repeat `n_permutations = 10,000`, `seed = 20260903` (new,
unused seed — §8) per pair, drawn independently per pair (not one shared
permutation stream reused across all six pairs). Two-sided empirical
p-value: fraction of permuted statistics ≥ the observed statistic.

**Resampling unit:** one row (one prompt) — there is no pairing structure
between quadrants (unlike R104's own source→candidate pairing), so this is
an unpaired two-sample permutation test at the row level, run separately
per view.

**Multiple-comparison handling:** Holm–Bonferroni applied **separately
within each of the three views** (structural-only, lexical-only, combined)
— three independent families of 6 tests each, mirroring C-A §7.6's
per-tier-family precedent (not one pooled family of 18).

No other distribution-distance statistic (MMD, Wasserstein, etc.) is
computed by this spec — this is a single predeclared choice, not a
metric-shopping menu; if energy distance and a differently-shaped
inference are both wanted later, that is new, separately-authorized work,
not a silent addition to this contract.

### 5.5 Token / n-gram distributions (separate from §4.2's TF-IDF block)

- **Scope:** unigrams and bigrams, computed separately (two divergence
  matrices, not one combined score) — both over the same tokenizer as
  everywhere else in this document (`word_tokenize`).
- **Common vocabulary:** built from the A∪B∪C∪D pooled corpus only (never
  including R104-source/R-AUTHORED in vocabulary construction, mirroring
  §2/§4.2's fit/transform split), retaining a unigram or bigram only if it
  appears in **at least 2 documents** — reusing `FROZEN_CUE_CONFIG`'s
  `min_df=2` convention rather than inventing a new threshold.
- **Rare-token handling:** any token/bigram in a given population's text
  that falls outside the common vocabulary is mapped to a single shared
  `<RARE>` bucket per n-gram order, rather than dropped — this keeps each
  population's distribution summing to 1 over the same fixed support.
- **Smoothing:** additive (Laplace-style) smoothing with
  `alpha = 0.01` — reusing `FightinWords`'s own `prior_strength` value
  (§4.3) for consistency of convention across the document, not because a
  new value was derived — applied to every population's normalized count
  distribution (including the `<RARE>` bucket) before divergence
  computation.
- **Divergence:** Jensen–Shannon divergence (base-2, bounded in [0, 1]),
  computed pairwise for the same six A/B/C/D pairs as §5.4, separately for
  unigram and bigram scope (12 values total), plus R104-source-vs-C and
  R-AUTHORED-vs-{A,B,C,D} as descriptive, non-blocking auxiliary numbers
  (not folded into the six primary pairs' correction family in §5.4).
- **Explicit prohibition (per task brief):** report only the scalar
  divergence value per pair per scope. **Do not** report, rank, or
  otherwise surface which individual tokens/bigrams contribute most to a
  given divergence — that is a discriminative-lexicon output, exactly the
  kind of artifact C-A §9 already declined to reproduce, and is out of
  scope here regardless of how naturally JS divergence's per-token
  contribution decomposition would make it available.

### 5.6 Confound sensitivity

For each of length (word/character count), source, category, and
formatting, predeclare an *association/attenuation* check — never a causal
claim — run on the combined-view centroid distances and the §5.4
energy-distance statistics:

1. **Length:** residualize every §4.1/§4.2-derived feature on `word_count`
   (simple linear regression, residuals only) within the A∪B∪C∪D fit
   population; recompute §5.1 centroids and §5.4 statistics on the
   residualized structural-only and combined views; report the resulting
   attenuation (or lack of it) in each pairwise distance, alongside a
   median-split stratified check (below/above the pooled A∪B∪C∪D
   `word_count` median) as a second, non-parametric view of the same
   question.
2. **Source:** per §3, only D supports a within-quadrant source contrast
   (Alpaca/Dolly-15k/OASST1); A, B, C are reported as **not applicable**
   for within-quadrant source sensitivity, with an explicit note that
   `source_dataset` is perfectly confounded with quadrant membership for
   those three quadrants — this confound cannot be "checked away" by any
   sensitivity analysis available in this repository, and must be reported
   as a standing limitation on any A/B/C/D geometry finding, not resolved.
3. **Category:** per §3, only an A-vs-C within-category (4-level,
   `project_category`) stratified comparison is computable; B and D are
   reported as **not applicable** (disjoint taxonomy / entirely null,
   respectively) rather than silently omitted.
4. **Formatting:** given C-D's own finding that
   `has_bullet_marker`/`has_numbered_step`/`has_code_block` are
   near-zero-variance at the R104 scale, re-check variance on the full
   A∪B∪C∪D pool specifically (§4.1's drop rule) before relying on these
   features for any sensitivity claim; if they survive the drop rule at
   the full-pool scale, report their individual association with the §5.1
   pairwise distances the same way as the other three confounds; if they
   are dropped, report that explicitly rather than silently absent them
   from this section too.

---

## 6. R-AUTHORED

Projected into all three views (§4.4) using the A∪B∪C∪D-fitted
transformations only (§2, §4.2) — never contributes to fitting any
vectorizer, PCA component, standardization mean/sd, or common-vocabulary
construction (§5.5). Not used to tune any threshold (SVD component count,
`min_df`, permutation count, or view-weighting in §4.4) anywhere in this
document. Reported in §5.1's centroid table and §5.3's PCA plot as a
clearly distinct, non-quadrant population, consistent with its `pending`
review status (C-E) and its Q25 rank-selection bias (C-D Gate 4) — this
spec does not relax either caveat, and C-F-B's report must repeat both
alongside any R-AUTHORED-derived number.

---

## 7. Embeddings (optional, secondary, non-blocking)

A frozen sentence-embedding view is worth proposing as a **secondary
robustness check only**, not a required part of the primary audit (per
task brief and CPU-only constraint):

- **Model:** `sentence-transformers/all-MiniLM-L6-v2` — the same model
  already used by this repository's existing near-duplicate-check tooling
  (`src/diagnostics/check_leakage.py`, `src/diagnostics/complete_neardup_check.py`,
  `src/data_pipeline/validate_c_source_authored_candidates.py`), reused for
  consistency of convention rather than introducing a second embedding
  model into the repository.
- **Version pin:** C-F-B must record the exact
  `sentence-transformers` package version and the model's resolved
  revision/commit hash from the Hugging Face Hub at run time (the same
  fail-closed provenance convention as §7.8 of C-A) — this document does
  not assume a version, since the model has not been downloaded in this
  sandbox (same network-unavailable constraint already logged in C-A §2 and
  `agent_state.json`).
- **Preprocessing:** raw `prompt` (and R104-source's `source_prompt`,
  R-AUTHORED's `candidate_prompt`) text, no normalization, mean-pooled
  sentence embeddings (the library's default), cosine similarity/distance
  as the resulting geometry's metric — no fine-tuning, no contrastive
  objective, no additional training of any kind.
- **Role if run:** repeat §5.1 (centroid distances) and §5.3 (PCA/plot)
  in embedding space as a robustness cross-check against the structural/
  lexical representation in §4 — **not** a replacement for it, and not a
  fourth "representation view" folded into §4.4's three predeclared views.
- **Status:** optional and non-blocking. If GPU/network access remains
  unavailable, C-F-B must report this section as **not run** (same
  fail-closed disclosure convention as R104's untested near-duplicate
  check in C-A §2) rather than omit it silently.

---

## 8. Reproducibility contract

**New seeds introduced by this document** (verified against every seed
already in use anywhere in this repository — `42`, `43`, `45`, `1337`,
`271828`, `20260829`, `20260830`, `20260831`, `20260901`, `20260902` — none
of the three below collide with any of those):

| Seed | Value | Used for |
|---|---|---|
| SVD seed | `20260905` | `TruncatedSVD(algorithm='arpack', random_state=...)`, §4.2 |
| Permutation seed | `20260903` | Energy-distance permutation test, §5.4 (one stream per pair, independently) |
| *(reserved, not required unless C-F-B adds a bootstrap CI on centroid distances)* | `20260904` | Not used by any procedure predeclared in this document; reserved here so a future addition does not silently collide with `20260903`/`20260905` |

**Locked:** exact input files and hashes (§1); population and text-field
definitions (§2); category/source availability facts (§3); feature
definitions for structural, lexical, and Fightin'-Words blocks (§4);
standardization and view-combination rules (§4.4); missing-data rule
(§4.5); all five required analysis-block specifications (§5); R-AUTHORED's
projection-only role (§6); the optional embeddings check's exact model and
non-blocking status (§7); the three seeds above.

**Software versions:** not fixed by this document — C-F-B must record its
own actual runtime `python`/`numpy`/`scipy`/`pandas`/`scikit-learn`
versions in its output JSON, per the same rationale as C-A §7.8
(`requirements.txt` pins lower bounds only).

**Canonical output paths (for C-F-B, not produced by this document):**

- `logs/cf_joint_geometry_analysis.md` (human-readable report)
- `logs/cf_joint_geometry_analysis.json` (machine-readable: all centroid
  distances, contrast vectors and cosine angles, PCA loadings/explained
  variance, energy-distance statistics and corrected p-values, JS
  divergence matrices, confound-sensitivity results, dropped-feature list,
  excluded-row counts, actual software versions, and this contract's input
  hashes re-verified at run time)

**Exact command C-F-B must implement and run** (module does not exist yet):

```
python -m src.analysis.cf_joint_geometry \
    --benchmark-latest data/frozen_v2/LATEST_BENCHMARK.json \
    --review-csv data/review/c_review_queue.csv \
    --r-authored-csv data/review/c_source_authored_review_queue.csv \
    --gate-config logs/benchmark_gate_config.json \
    --formatting-config-source logs/3d_b_lexical_outlierness_pilot.json \
    --svd-seed 20260905 \
    --permutation-seed 20260903 --n-permutations 10000 \
    --out-md logs/cf_joint_geometry_analysis.md \
    --out-json logs/cf_joint_geometry_analysis.json
```

C-F-B must fail closed (abort, not warn) if any hash in §1 does not match
at load time, mirroring the same convention already required of C-B
(C-A §7.9) and `src/data_pipeline/build_c_source_authored_candidates.py`.

---

## 9. Explicit non-actions

This task did **not**: implement any analysis code (`src/analysis/cf_joint_geometry.py`
does not exist); run PCA, TF-IDF fitting, energy-distance computation, JS
divergence, or any embedding model; compute a single centroid, contrast
vector, or distance; modify `data/frozen_v2/*`, `data/review/c_review_queue.csv`,
`data/review/c_source_authored_review_queue.csv`, or any source module; view
or reproduce any raw prompt text beyond the field-name/schema inspection
needed to write §2–§3; reproduce the `lexical_risk_hit_count` lexicon's
matched terms; start contrastive or representation-learning training of any
kind; decide whether C is valid, whether the intended 2×2 construct holds,
or whether R-AUTHORED should be promoted; or begin C-F-B.

**Stop.**
