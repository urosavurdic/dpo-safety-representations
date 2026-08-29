# 3D-C — Implementation-Mismatch Audit for the S2/S3 Length-Dependence Warning

Status: **audit only — no scoring code changed, no pilot re-run against a
modified implementation.** This document does not compute CUE, does not
touch the frozen benchmark, and does not modify
`data/quadrant_c_pipeline/lexical_outlierness_groups_v1.json` or either
`logs/3d_b_lexical_outlierness_pilot.*` file.

Git commit at audit start: `e6a77742e5b6f63fa7b77619101e2c31fa23f36b`
(tip of `agent/c-quadrant-end-to-end-e0e2317a` at the time this task began).

---

## 1. Question this task answers

3D-B reported `Spearman(p_tfidf, token_count) ≈ -0.567` alongside a
`Spearman(p_tfidf, p_selfinfo) ≈ 0.291` MIXED result. The question is not
whether that correlation can be made smaller — it's whether it reflects a
bug in how S2/S3 were implemented, or is an inherent property of the
locked operationalization itself.

## 2. Immutable-baseline provenance (verified before any audit work)

| Artifact | Path | SHA-256 |
|---|---|---|
| 3D-B results (JSON) | `logs/3d_b_lexical_outlierness_pilot.json` | `95b0b7771244f0c162627eb1aaeb92986b4e7ec9de737f4f38edaefec53ebce5` |
| 3D-B results (report) | `logs/3d_b_lexical_outlierness_pilot.md` | `20704ac61861eef26630ae1989b88b428eeb3669677378f9f6646f7eba31142a` |
| Duplicate/template grouping | `data/quadrant_c_pipeline/lexical_outlierness_groups_v1.json` | `9d5f14ae3597d5cd1c20d58aa194739be5c956beeaeda0b638b7f5d4a8ff0f39` |
| Input population (209-eligible, 413-row file) | `data/quadrant_c_pipeline/c_source_authored_candidates_validated_v1.jsonl` | `8be074b8f42e87aa1c3eee83abedec9743e48ce1a65758e329118b34a6c37ca7` |

All four match the values already on record in `logs/3d_a_lexical_outlierness_design.md`
(population) and `logs/3d_b_lexical_outlierness_pilot.md` (results). None
were regenerated or modified by this task.

## 3. Mandatory mathematical decision gate — audit result

### S2 (LOGO TF-IDF centroid distance)

| Requirement | Result |
|---|---|
| Raw within-document TF counts | **MATCH** |
| Group-local IDF (LOGO) | **MATCH** |
| Per-document L2 normalization before averaging | **MATCH** |
| Cosine distance to the LOGO centroid | **MATCH** |

The IDF formula `idf_{-g}(t) = ln((1+total_weight)/(1+df)) + 1` used in
`fold_tfidf()` is algebraically identical to the design doc's
`ln((1+|R_{-g}|)/(1+df_{-g}(t)))+1` in the unweighted case. The centroid
is built as the mean of already-L2-normalized reference vectors
(`mu_i = (1/|R_-g|) * Σ v_j_normalized`), never re-normalized afterward —
exactly as locked.

### S3 (LOGO smoothed token self-information)

| Requirement | Result |
|---|---|
| Repeated token occurrences counted, not deduplicated | **MATCH** |
| `alpha = 1.0`, fixed | **MATCH** |
| Division by `n_i` | **MATCH** |

### Shared

| Requirement | Result |
|---|---|
| Held-out-*group* (not just held-out row) excluded from all reference statistics | **MATCH** |
| Locked preprocessing (NFKC, lowercasing, URL placeholder, `\b\w+\b` tokenizer, stopwords retained) | **MATCH** |
| Exact group assignments | **MATCH** (byte-identical re-derivation, see §4) |
| Fold-calibrated percentile, including "reference rows scored against the fold's own fitted statistics, not further leave-one-out'd" | **MATCH** |

**No implementation mismatch was found.**

## 4. Independent verification performed

1. **Manual re-derivation.** The S2 IDF/vector/centroid construction and
   the S3 count/`N`/`|V|` construction were re-derived from scratch in a
   standalone script (not calling into the module's internals except for
   the final comparison) on a synthetic 3-document fold, then diffed
   against `fold_tfidf()` / `fold_selfinfo()`. Result: 0.0 max absolute
   difference on every IDF and centroid component; exact dict equality
   on every S3 count; exact agreement on a held-out S3 score.
2. **Full pilot re-run.** `run_pilot()` was executed unmodified from a
   scratch copy of this exact commit, so the frozen `logs/3d_b_*` files
   and the frozen grouping artifact in this repository were never
   opened for writing. The re-run reproduced the frozen results
   **bit-for-bit**:
   - `Spearman(p_tfidf, p_selfinfo)`: `0.29099051803952836` (frozen) vs.
     `0.29099051803952836` (re-run).
   - `Spearman(length, p_tfidf)`: `-0.5674511106345929` (frozen) vs.
     `-0.5674511106345929` (re-run).
   - Grouping artifact SHA-256: identical (`9d5f14ae…`) in both.
3. **Unit tests.** All 19 pre-existing tests in
   `tests/data_pipeline/test_lexical_outlierness.py` pass, unmodified.

This confirms the code currently in the repository is exactly the code
that produced the frozen 3D-B artifacts — no drift, no stray uncommitted
edits, no environment-dependent nondeterminism.

## 5. Decision logic applied

Per the task brief, since **no mismatch exists (Case 2)**:

- The scoring implementation was **not** modified.
- No corrected score was invented.
- The S9 length-residualization diagnostic was **not** promoted to a
  primary score.
- The 209-row pilot was **not** re-run against a modified implementation
  (there was nothing to change).
- `data/quadrant_c_pipeline/c_source_authored_candidates_validated_v1.jsonl`,
  `lexical_outlierness_groups_v1.json`, and both frozen `3d_b_*` files
  are untouched.

## 6. Why no construct-preserving length correction is available

The design doc already forecloses the one length-normalization move
that looks obvious: dividing a document's raw TF-IDF vector by its
token count *before* the prescribed L2 normalization is pure scalar
rescaling, and cancels exactly under cosine similarity — it isn't a
distinct construct, just a no-op wrapped in extra arithmetic. Every
other length adjustment listed as out-of-bounds in the task brief
(sublinear TF, BM25, L1 normalization, document-length penalties,
residualized percentiles, regression-adjusted primary scores) would
change what S2/S3 measure, not merely re-express it — i.e., it would be
a redesign, not a correction.

The audit above rules out a coding-error explanation. The most
plausible mechanistic account of the length association is structural
to raw-count TF-IDF with (1,2)-gram features and cosine-to-centroid
distance: longer prompts contribute proportionally more distinct,
low-document-frequency bigrams (bigram type-count grows faster than
unigram type-count with length, and individual bigrams are rarer across
the pool), which pushes a longer prompt's L2-normalized vector further
from a centroid dominated by shorter, more template-like prompts. This
is a property of the exact locked S2 operationalization on this
dataset, not an implementation defect.

## 7. Conclusion

> **No construct-preserving correction is justified by the locked
> S2/S3 definitions; the observed length dependence remains a property
> of this exact operationalization and dataset.**

Length dependence status: **remains**, unmitigated, by design.

This is a complete, valid 3D-C outcome per the task brief. No further
milestone (3D-H, 3F, or otherwise) is started.
