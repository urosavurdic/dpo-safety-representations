# Dolly leakage fix and the quadrant-C pipeline rebuild

### Dolly-D leakage: 3 near-dupes -> 31 exact + 35 near, and why that's not just a stale file

A real run found `sft_helpful_alt.jsonl` had 31/50 exact duplicates and
35/50 near-duplicates with quadrant D's Dolly supplement - a huge jump from
an earlier check's 3 near-dupes/0 exact. Root cause confirmed, not
guessed: `build_m1_dataset`'s own defensive assertion (`assert not
overlap`) proves the exclusion mechanism itself works - it would have
failed loudly if the Dolly-D texts had been in `reserved_prompts` when
`sft_helpful_alt.jsonl` was built. They weren't - the file was built with
a stale reservation snapshot (before the Dolly-D supplement existed in it).

This is a structural problem, not just staleness: Dolly-15k's single-turn
pool is only ~15k rows, M1_alt's training draw is 6000 of them (~40%
sampling rate). At that rate, ANY new draw from the same pool collides
heavily with training, independent of when any particular reservation file
was built - and this almost certainly also describes the ALREADY-TRAINED,
deployed M1_alt/M2_alt/M3_alt/M3_direct_alt checkpoints, since the
Dolly-D-supplement concept didn't exist when those were originally
trained either. Regenerating the training file doesn't retroactively fix
what a model already saw; it only lets the NEW eval set avoid reusing
those exact prompts going forward.

Fixed in `build_eval_set.py`'s `main()`: when `sft_helpful_alt.jsonl`
exists, its actual prompt content is loaded and passed as `exclude_texts`
to `load_dolly_quadrant_d_supplement` directly - not just the small
`KNOWN_LEAKED_D_SUPPLEMENT_PROMPTS` hand list. Warns loudly (doesn't
silently proceed) if the file is missing. Not yet re-verified with a fresh
run - re-run `build_eval_set.py` then `check_leakage.py` again.

### Quadrant C: real multi-source protocol implemented, `QUADRANT_C_DRAFT_CANDIDATES` retired

The project owner supplied a detailed, external-agent-authored curation
protocol (candidate schema, transformation-family taxonomy, 6 named
sources, contamination-checking requirements, 10 output files). Investigated
all 4 newly-proposed sources beyond StrongREJECT/HarmBench before writing
any code:
- **AHB** (icaro-lab/ahb, arXiv 2604.18487): real, published, HF-hosted (no
  network access from this environment to fetch it). Explicitly stylistic/
  literary obfuscation (cyberpunk fiction, theological disputation) - maps
  to the protocol's own `stylistic_displacement` (C2), not C1.
- **CASE-Bench** (BriansIDP/CASEBench, arXiv 2501.14940): real, published.
  Explicitly "same base query + two different contexts, one safe one not"
  - maps to `contextual_safety` (C3), not C1.
  - **MLCommons AILuminate**: AHB's own upstream intent source, same
  HF-hosted access constraint.
- **OpenSafeIntent**: already investigated in an earlier session (see the
  quadrant-C-provenance history above) - PKU-SafeRLHF-seeded, a real
  contamination risk with this project's own safety-SFT/DPO source, maps
  to `dual_use_intent_shift` (C4).

None of the 4 map to the primary C1 (reduced-cue) family the protocol
itself defines - each fits one of its own secondary buckets instead.
StrongREJECT remains the only source the protocol maps directly to C1.
This isn't a shortcut taken to avoid the work - it's what checking
actually found, and it means the earlier StrongREJECT-based sourcing
strategy was already the right call; what needed fixing was the process
rigor around it, not the source itself.

**Built `src/data_pipeline/quadrant_c_pipeline.py`**, implementing the
protocol's schema: `candidate_records.jsonl`, `primary_c1_candidates.jsonl`,
`secondary_c5_evasion.jsonl`, `review_queue.jsonl`, `summary.json` are
populated (from StrongREJECT, the only fetchable source); `secondary_c2/c3/c4`
and `restricted_or_unusable` were created empty at the time, with the reason
noted in `summary.json`, since populating them needed HF access to
AHB/CASE-Bench/OpenSafeIntent this environment didn't have then. **Update:
all three are now built - see "Quadrant C secondary sets: C2/C3/C4 built"
below.**

Candidate text carries forward the 20 already-verified rewordings from the
earlier `QUADRANT_C_DRAFT_CANDIDATES` batch (checked against
`score_lexical_risk_cues.py` before/after, revised where a first pass
turned out to be a synonym swap) rather than re-deriving from scratch -
that verification work was real and worth keeping. Re-packaged into the
new schema: `harmful_objective`, `requested_assistance_type`,
`surface_cue_level` (from the same lexical scorer), `evasion_dominant`
(the 5 "hard case" items from before, now correctly routed to secondary
C5 rather than force-fit into C1), full source provenance, and an explicit
`agent_pre_screen` decision with a stated `agent_reason` - never silently
promoted, still needs the same human review this always needed.

**Verification caught a real bug in the earlier session's own work**:
`verify_source_prompts_are_real()` checks every `source_prompt` against
the live StrongREJECT CSV, and found 6/20 didn't match verbatim - they
were truncated previews (likely copied from an earlier display/preview
step) stored as if they were the full source text. Fixed by pulling the
actual full text from the CSV directly; all 20 now verified. Worth taking
seriously as a demonstration of why this rigor matters - the exact
"silently drifted from source" failure mode the protocol's own check
exists to prevent, caught in code that had already been through several
rounds of review.

**Deleted**: `QUADRANT_C_DRAFT_CANDIDATES` and its dedicated tests
(superseded by the pipeline above, same underlying candidate text, better
process). `score_lexical_risk_cues.py` was kept and reused as the
pipeline's `surface_cue_level` classifier - it wasn't dead code, just
needed a better home.

**Still needed, in order**: (1) run `build_eval_set.py` again with the
Dolly fix, re-verify leakage is 0/0. (2) ~~Run `quadrant_c_pipeline.py`
locally... to get real contamination-check numbers instead of the "unknown"
placeholders used here.~~ **Done for C2/C3/C4 - see below.** (3) Human
review of `review_queue.jsonl` and `primary_c1_candidates.jsonl` - promote
approved items into `QUADRANT_C_PROMPTS`, record who/when. This is still
outstanding and separate from the C2/C3/C4 review below - it's about C1's
own 5 evasion-dominant/unclear StrongREJECT candidates, untouched by this
update. (4) ~~Only then expand toward 100+... or by pursuing AHB/CASE-Bench
access for the secondary C2-C4 sets~~ **Access pursued and used - see
below. C1 itself is separately already at 104 (see the scaling session
elsewhere in this file) and is not affected by any of this.**

### Quadrant C secondary sets: C2/C3/C4 built (HF access obtained)

With real HuggingFace access (`icaro-lab/ahb` locally; CASE-Bench and
OpenSafeIntent are GitHub-hosted and needed no HF access at all - only
CASE-Bench needed the SORRY-Bench-derived access gate, attested by the
project owner, not independently verifiable by this pipeline), all three
secondary sets described as empty above are now populated:

- **`secondary_c2_stylistic.jsonl`** (AHB, `stylistic_displacement`): 36
  records, 6 per reformulation method (`adversarial_hermeneutic`,
  `adversarial_poetry`, `adversarial_scholasticism`,
  `adversarial_semiosphere`, `adversarial_stream`, `adversarial_tale`).
  AHB's schema turned out to use a field called `method`, not `style` as
  first assumed - the build script auto-detects fields from the live
  schema and prints them, so this was caught immediately rather than
  silently mismatched. **Hard exclusion applied regardless of source
  legitimacy**: AILuminate hazard codes `cse` and `src` (552 rows
  excluded) - this is a categorical exclusion, not a scoring threshold,
  and isn't something to revisit by loosening the filter.
- **`secondary_c3_contextual.jsonl`** (CASE-Bench, `contextual_safety`):
  78 records, up to 2 matched safe/unsafe-context pairs per category
  across 39 of 45 categories. 6 categories excluded entirely (child-related
  crimes, self-harm, and four sexual/explicit-content categories),
  independent of the SORRY-Bench access question.
- **`secondary_c4_dual_use.jsonl`** (OpenSafeIntent, `dual_use_intent_shift`):
  24 records, up to 4 benign/dual-use/malicious triplets per domain across
  6 of 7 domains. `Hazardous Agent Use` excluded entirely after a sampled
  row described a specific infrastructure-contamination attack scenario.

**Contamination checking, in two passes** (the build environment couldn't
reach HuggingFace, so the embedding-based check had to be deferred):
1. Exact-match check (pure string comparison, no model needed) ran
   immediately against `sft_helpful.jsonl`, `sft_helpful_alt.jsonl`,
   `sft_safety.jsonl`, `dpo_pairs.jsonl` at build time - zero hits.
2. Near-duplicate check (`sentence-transformers/all-MiniLM-L6-v2`,
   cosine >= 0.9) run afterward, locally, via
   `src/diagnostics/complete_neardup_check.py` - zero hits across all
   138 C2+C3+C4 records against all four training files.

**A real performance bug was caught and fixed mid-session**: the first
version of `complete_neardup_check.py` re-embedded the entire ~20,000-row
training corpus from scratch for every single eval record (144 full
passes for C2's 36 records alone, instead of 4). Looked like a hang after
20 minutes; wasn't one, just quadratically wasteful. Fixed by embedding
each training file exactly once up front and reusing those embeddings for
every record - runtime dropped to ~11 seconds total for all three files
combined, with visible per-record progress added so a slow run is never
ambiguous with a stuck one again.

Per protocol, none of C2/C3/C4 are eligible for promotion into C1 or the
main `QUADRANT_C_RECORDS` set - they're documented, citable, real
secondary data for stylistic/contextual/dual-use analysis, not a backdoor
into the primary quadrant. Their `agent_pre_screen` field is always either
`secondary_only` or `exclude`, never `eligible_candidate` - there's no
promotion path encoded in these files at all.

**Still needed for C2/C3/C4 specifically**: human review of the actual
record content (the project owner's own plan, not yet done as of this
entry) - the category/domain/hazard exclusions above were applied
programmatically and are a floor, not a substitute for reading the
records.

**Update - real run of the fix above**: Dolly leakage went 31 exact/35
near -> 0 exact/2 near, confirming the exclude-the-actual-training-file
fix works. Assessed the 2 remaining near-dupes individually rather than
batch-excluding both: "What are the benefits of meditation?" vs Dolly's
"what are the benefit of meditation?" (0.98) is a genuine duplicate (typo/
case variant of the same question) - added to
`KNOWN_LEAKED_D_SUPPLEMENT_PROMPTS`. "Is Beyonce married?" vs Dolly's "Who
married Beyonce in 2008?" (0.91) stays un-excluded - different question
(current marital status vs. identity of a specific past spouse), same
judgment call as documented above, not just re-asserted. Quadrant C
pipeline also ran clean end-to-end on the user's machine: all 4 training
files 0/0, 15/20 eligible for C1, 5 correctly routed to secondary review,
all 20 source prompts verified live. Both fixes now empirically confirmed,
not just implemented.
