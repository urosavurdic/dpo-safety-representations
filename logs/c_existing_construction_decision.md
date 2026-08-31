# C-D — Independent Decision on Existing C Constructions

Status: independent decision only. Reads `results/c_construction_audit/audit.json`,
`results/c_construction_audit/audit_summary.md`, and the small set of upstream
artifacts needed to interpret them (`logs/c_b_paired_delta_analysis.json`
`pair_integrity`/`provenance_integrity`, `results/c_construction_audit/input_manifest.json`,
`logs/3d_b_lexical_outlierness_pilot.md`, `logs/3d_h_construct_check_analysis.md`).
No code, benchmark data, or prompt wording was modified, created, or viewed
verbatim in producing this memo. No common-CUE/contrastive construction is
started here.

## Gate 0 — Integrity

**R104 / `c_paired`:** clean. `provenance_integrity` shows the frozen
quadrant-C row count (104) matches the review-queue row count (104) with
matching record-ID sets, 100% `c_construction=c_paired`, 100%
`source_dataset=StrongREJECT`. `pair_integrity` shows 104/104 valid pairs
with zero exclusions, and the two subsets (`assistance_type_preserved=yes`,
n=78; `partial`, n=26) sum exactly to 104. All C-A-pinned inputs are
fail-closed hash-verified, and the C-C re-execution is byte-identical to
the committed C-B output. Gap: near-duplicate (semantic, as opposed to
exact/normalized-string) overlap between R104 and quadrant A was **not
run** (embedding inference is out of scope for this CPU-only stream) —
this is an open integrity gap, not a pass.

**R-AUTHORED:** provenance bookkeeping is clean (hashes recorded, source/
classifier/provenance-class counts all present), but `review_status` is
`pending` for all 52/52 rows — zero human review has occurred. Its own
descriptive stats are additionally a product of Q25 rank-selection
(lowest-`fightin_words`-score quartile), not a random sample of the
candidate pool.

**3D-B/3D-H:** 3D-B (n=209) and 3D-H (n=32) are self-contained, reproducible
artifacts (seeded permutation/bootstrap, documented ambiguity resolutions).
Not integrity-relevant to R104/R-AUTHORED directly; auxiliary only.

## Gate 1 — Primary paired signal (R104)

Across all three populations there is a **large, highly significant, and
consistently replicated structural shift**: candidates are shorter
(word/char/sentence count all down, Holm-adj p≈0.0001–0.005 in every
population) with longer average word length and fewer multi-sentence
responses (`multi_sentence_flag`, d_z ≈ −1.13 to −1.15 in all three
populations — the single largest-magnitude effect in the entire feature
set).

The **lexical/cue-content** signal is not consistent in direction:
`fightin_words` moves *up* in candidates (more harmful-associated wording
by that instrument; d_z≈0.46–0.70, significant in populations 1–2, only
borderline in population 3), while `lexical_risk_hit_count` (fixed lexicon)
moves *down* (fewer flagged terms; d_z≈−0.51 to −0.60, significant in
populations 1–2, borderline in population 3) — the opposite direction. The
audit itself flags this as unreconciled: two real, current, non-comparable
lexical instruments disagree, and neither is treated as authoritative here.
`cue_tfidf_logreg_margin` (distributional/exploratory, fit on this data) is
positive and significant in populations 1 and 3 but only marginal in
population 2 (Holm-adj p=0.045) — the population with the best evidence
quality.

**Population 2 (`assistance_type_preserved=yes`, n=78)** reproduces
population 1's direction and significance pattern on every feature, which
is the strongest single piece of support for treating R104 as more than a
population-1 artifact.

## Gate 2 — Confounds

Length is a real, consistent confound, not a one-off: the paired delta in
`lexical_diversity` correlates with the `word_count` paired delta at
ρ≈−0.50 (p=6.6e-8 pop.1; ρ≈−0.50 pop.2; ρ≈−0.51 pop.3), and
`cue_tfidf_logreg_margin` similarly (ρ≈−0.35 to −0.37, significant in
pops 1–2). `fightin_words` also correlates with the length delta in
populations 1–2 (ρ≈−0.34 to −0.43) but not significantly in population 3
(likely underpowered, n=26). `lexical_risk_hit_count` is the one lexical
feature *not* significantly associated with the length delta (ρ≈−0.08 to
0.09, all p>0.4) — notably, it is also the feature pointing in the
direction R104's own naming implies.

Formatting: bullet/numbered-step/code-block indicators are zero-variance
(floor effect, uninformative here). `multi_sentence_flag` — a structural,
not lexical-content, feature — is simultaneously the single largest effect
in the whole feature set. Source and category confounds are not testable
(R104 is 100% StrongREJECT, so no within-source contrast exists; category
robustness has no predeclared formal test and is unbalanced 6–41 rows
across 4 levels). Repeated-rewrite-template concentration is an
acknowledged **evidence gap** — no metric for it exists anywhere in the
repository, so template-driven inflation of the observed effect cannot be
ruled in or out.

Net read: the most robust, most significant, most population-stable signal
in R104 is structural/compression-style (shorter, fewer sentences, longer
average words), not a clean lexical-cue signal — the lexical-cue evidence
is real but split between two disagreeing instruments and partially
length-confounded.

## Gate 3 — Benchmark-role plausibility

R104 retains harmful intent/provenance by construction (source→candidate
pairs drawn from StrongREJECT, source-authored to preserve the harmful
objective). The candidate/source shift is real and reproducible in the
preserved-assistance subset, but the evidence does not cleanly rule out
"C is a shorter, more compressed restatement of A" as a competing
description of what changed — the dominant, most significant effect
(`multi_sentence_flag`) and two of the three length-correlated lexical
features are exactly the pattern that reading would predict.
`lexical_risk_hit_count` is the one feature that is both in the intended
direction and not length-confounded, which keeps a "cue reduction"
reading alive, but it is contradicted by `fightin_words` moving the other
way. Per the task's own bar, R104 does not need a clean independent cue
axis to be worth review — it needs integrity, a meaningful paired signal,
preserved-assistance support, and no single confound that makes the
intended reading implausible. The first three hold; the fourth is a real
tension (structural compression dominates) but does not, on its own, make
"harmful-intent-preserving rewrite with an ambiguous cue-content shift"
an implausible thing for a researcher to look at directly — it makes it a
thing that specifically needs a researcher's eyes on actual pairs rather
than an automatic promotion.

## Gate 4 — R-AUTHORED

Treated strictly as unlabeled external evidence, consistent with its own
`treated_as` field (evidence-hierarchy tier 7, below the A/B/C/D secondary
tier). Clean provenance bookkeeping is not construct validity: 0/52 rows
have been reviewed, and the population's own lexical distribution is a
byproduct of Q25 rank-selection rather than an independent property of the
candidate pool, so it cannot presently be compared to R104 or read as
distributional support for anything. No criterion above was tuned against
these numbers.

## Decision table

| Construction | Evidence strength | Preserved-assistance robustness | Main confound | Decision | Human work required |
| --- | --- | --- | --- | --- | --- |
| R104 / `c_paired` | Strong, consistent structural shift; real but internally contradictory lexical-cue shift (two instruments disagree) | Strong — population 2 (n=78) reproduces population 1's direction/significance on every feature | `multi_sentence_flag` (structural, d_z≈−1.13 to −1.15, largest effect overall); `lexical_diversity`/`cue_tfidf_logreg_margin` both ρ≈−0.35 to −0.50 with length delta | **KEEP FOR HUMAN REVIEW** | Read a sample of the 104 pairs (prioritizing the 78 preserved-assistance pairs) to judge whether the rewrite plausibly reduces surface harm-signaling vs. is a length/format compression; adjudicate the `fightin_words` vs. `lexical_risk_hit_count` contradiction against actual text; assess rewrite-template concentration by inspection (no metric exists yet) |
| R-AUTHORED | Clean pipeline/provenance; zero analytic evidence — no pairing, no comparison test, distribution is a selection-rule artifact | N/A — no paired or assistance-preservation structure defined | Q25 rank-selection bias (queue is by construction the most D-like quartile, not a representative sample) | **INCONCLUSIVE** | Complete the 52 pending human reviews via the existing review protocol; a construct-relevant comparison to R104 is not possible before that |
| 3D-B / 3D-H (auxiliary only) | 3D-B: moderate/mixed, source- and length-confounded (tail membership changes for ~35% of rows under source- or category-balanced sensitivity checks); 3D-H: small (n=32) but statistically significant blind-human corroboration (permutation p=0.001, ρ≈−0.48) of the same proxy | N/A — validates a lexical-outlierness proxy, not a prompt-pair construction | Source-tail imbalance in 3D-B (StrongREJECT vs. SimpleSafetyTests skew across tails) and length correlation with the tfidf percentile (ρ≈−0.57) | **KEEP AS SECONDARY** | None required to retain current auxiliary status; a larger, source-balanced blind-rating replication would be needed before this could support more than an auxiliary role |

## Mandatory limitations

What remains unestablished after this audit:

- **Cue-only manipulation:** not shown. R104's most robust effects are
  structural/length-related; the lexical-content evidence is genuinely
  split between two disagreeing instruments.
- **Common cross-intent `C_cue`:** not shown. No resource in this
  repository defines or measures a validated shared cue axis across
  A/B/C/D; 3D-B/3D-H explicitly disclaim this themselves.
- **Causal model usage:** not shown. Every association reported here
  (length↔lexical-feature correlations, tail-membership sensitivity) is
  descriptive/correlational; none of the source artifacts claim causal
  identification, and this memo does not either.
- **Semantic equivalence beyond measured fields:** not independently
  verified. "Preserves the broader harmful objective" is a construction
  property and an `assistance_type_preserved` label, not a measured
  semantic-similarity result — no embedding/NLI check was run (out of
  scope, CPU-only).
- **Benign-side matched validation:** not shown. There is no paired B↔D
  construction analogous to R104 in this audit; the A/B/D quadrant numbers
  used here are unpaired, descriptive, and no formal cross-quadrant test
  is predeclared or computed anywhere in the repository.
- **Near-duplicate overlap with quadrant A:** not checked (embedding
  inference out of scope for this CPU-only stream) — an open integrity
  gap for R104, not resolved by this memo.
- **Repeated-rewrite-template concentration:** not computed for either
  R104 or R-AUTHORED — no such metric exists in the repository yet.

**Stop.**
