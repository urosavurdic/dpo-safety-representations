# CUE Reference-Data Audit — Revised 2×2 (Intent × CUE) Exploratory Benchmark

**Repository:** `https://github.com/urosavurdic/dpo-safety-representations`
**Branch:** `agent/c-quadrant-end-to-end-e0e2317a`
**Scope:** Audit only. No scorer implemented, no candidates scored, no
Q10/Q25/Q40 assignment, no benchmark freeze, no changes to
`data/frozen_v2/*` or `logs/benchmark_gate_config.json`. This is a design
note for a **different, broader** effort than the existing 3A
(`c_source_authored`) track: that track scores new harmful/low-cue
*candidates* against an already-fixed reference. This note is about what
should be allowed to define the reference in the first place, for a
revised 2×2 (intent × CUE) framing where CUE — via TF-IDF+LogReg and
smoothed lexical log-odds (Fightin' Words) — may eventually be asked to
characterize *all four* quadrants, not just score new C material.

CUE, per project definition, is an **operational lexical harmful-association
score** (two independent lexical methods, agreement = robust tail), not a
claim of a pure latent construct. Intent labels come from trusted existing
dataset curation, per source, as audited below — never from response-level
safety scores, chosen/rejected preference status, or source name alone.

---

## Headline finding: the current reference is definitionally identical to two of the four quadrants

`src/corpus_discrimination.py`'s Fightin' Words reference is fixed as
`H = quadrant A ∪ quadrant B` (150 HarmBench + 250 XSTest) and `D =
quadrant D` (150: Alpaca/Dolly/OASST1). Quadrant A **is** HarmBench in its
entirety and quadrant B **is** XSTest-safe in its entirety
(`logs/3a0_source_plan.md` §2, verified again directly against
`data/processed/controlled_eval.jsonl`). So today's reference doesn't
independently confirm A and B are "high-CUE" — they're high-CUE **by
definition** of the reference.

This was already handled correctly in one direction: `logs/3a4_scoring.md`
records explicitly that "C-source-authored candidates were NOT used to
construct H or D" — StrongREJECT/SimpleSafetyTests candidates never leak
into the reference they're scored against. It was never handled in the
**reverse** direction, because under the old design HarmBench/XSTest were
only ever reference, never candidates. If the revised 2×2 wants CUE to be
a general-purpose measure (e.g. to check whether A/B genuinely score
high-CUE by an independent measure, not by construction), that reverse
case now matters and needs its own held-out design — see "Source-held-out
scoring plan" below.

---

## Per-source audit

| Source | Prompt-level label defensible? | Basis | Existing role in this repo |
|---|---|---|---|
| **HarmBench** | Yes | Entire corpus is red-team-authored harmful behaviors (`Behavior` field = the prompt itself); this is corpus-level curation intent, verified against actual row content via `classify_source_provenance.py`'s already-run pass, not "source name alone." | **Is** quadrant A (150/150, exact hash match, `logs/3a0_source_plan.md` §2). Also the fixed `H` half of the existing Fightin' Words reference. |
| **StrongREJECT** | Yes | Per-row `source` column documents real upstream provenance (custom/DAN/AdvBench/HarmfulQ/MaliciousInstruct/MasterKey/OpenAI System Card); per-row `category` field gives harm-area diversity (already surfaces in `c_counts_by_construction`'s category distribution). Structural-completeness classified: 342/413 (pooled with SimpleSafetyTests) `complete_user_facing_prompt`. Access verified, sha256-pinned in `logs/3a2_candidate_universe.md`. | Existing candidate pool for `c_source_authored` (not reference) — 313 rows, 132 eligible after validation, 0% exact-hash overlap with HarmBench/SimpleSafetyTests (`logs/3a1_source_inventory.md`). Also the source of the *frozen* benchmark's live `c_paired` (104 rows, AI-reworded, different construction). |
| **SimpleSafetyTests** | Yes | Entire 100-row corpus is hand-crafted unsafe requests across 5 harm areas; no per-row author field, but source-level curation intent is unambiguous and already vetted (`logs/3a1_source_inventory.md`). | Existing candidate pool for `c_source_authored` (not reference) — 100 rows, 77 eligible after validation, 0% exact-hash overlap with the other two harmful sources. |
| **XSTest (safe subset)** | Yes | Per-row `type` field (loaded as `category` in `build_eval_set.py:360`) and the dataset's own `label == "safe"` split — a real per-item field, not an assumption from source name. | **Is** quadrant B (250/250). Also the fixed second half of `H` in the existing Fightin' Words reference. |
| **JailbreakBench benign** | **Unresolved — inaccessible this session** | Verified directly this session: the installable `jailbreakbench==0.1.0` package (PyPI, allowed domain) bundles exactly one file, `behaviors.csv`, 100 rows, columns `Goal, Target, Behavior, Category` — all harmful, no benign split, no `Source` column (confirms `logs/3a1b_source_inventory.md`'s prior finding for the harmful side, and additionally establishes the benign split isn't bundled either). The full JBB-Behaviors dataset, including any benign contrast set, is HuggingFace-hosted only (`dedeswim/JBB-Behaviors`); `huggingface.co` is not in this sandbox's allowed egress list, consistent with the repeated `host_not_allowed` findings already logged in `logs/3a1a_source_inventory.json`/`3a1b_source_inventory.json`. | Not used anywhere in this repo. Cannot be acquired or schema-checked further under current network policy. |
| **OR-Bench** *(optional)* | **Excluded — inaccessible, and weaker provenance even if it were reachable** | Confirmed via its own paper/repo (`justincui03/or-bench`, searched this session): 80K "safe" + 600 "toxic" prompts are LLM-**generated** (rewritten from toxic seeds) and LLM-**moderator**-labeled, not human-curated — a materially different provenance class than the hand-curated/verified sources above. Dataset itself is hosted only at `huggingface.co/datasets/bench-llm/or-bench` — same network block as JailbreakBench. | Not used anywhere in this repo. |
| **PKU-SafeRLHF** *(optional)* | **No — fails the prompt-level-label requirement by its own schema** | `src/data_pipeline/data_prep.py` uses exactly `prompt`, `response_0`, `response_1`, `is_response_0_safe`, `is_response_1_safe`, `safer_response_id` — safety is a **per-response** field, and `build_matched_pairs()` explicitly *keeps only* rows where the two responses **disagree** on safety, i.e. the same prompt routinely pairs with both a safe and an unsafe response. There is no field that assigns one harmful/benign label to the prompt itself. This is precisely the "response-level safety scores" / "chosen-rejected preference status" the task rules out. | Used for M2/M3 training data (`dpo_pairs.jsonl`, `sft_safety.jsonl`) — a different purpose (training content, not intent labeling) that this audit does not touch. |
| **Existing DPO/safety-training sets** (`dpo_pairs.jsonl`, `sft_safety.jsonl`) *(optional)* | **No — same root cause as PKU-SafeRLHF** | Derived directly from PKU-SafeRLHF's `chosen`/`rejected` split (see `data_prep.py` above); inherits the same lack of a prompt-level intent field. | Training data only, per `PROJECT_CONTEXT.md`/`CLAUDE.md` design decision #2. |

**On the 20-row PKU-SafeRLHF sanity check:** not run. The task makes it
conditional on "if considering inclusion" — this repo's own already-working
`data_prep.py` shows the schema itself has no prompt-level field to sample
for, so no row sample would change the exclusion. It is also moot twice
over: `huggingface.co` (where PKU-SafeRLHF is hosted) is not reachable
under this sandbox's network policy, so the 20 rows couldn't be pulled
fresh even if the schema question were open.

---

## Selected sources (for CUE reference fitting)

- **Harmful:** HarmBench, StrongREJECT, SimpleSafetyTests — all three,
  rotated under leave-one-source-out (below), not a fixed pooled corpus.
- **Benign, high-register:** XSTest (safe subset) — the only accessible
  source with a defensible per-row benign-intent label at this register.
  Single-source; no rotation is possible (see limitation below).
- **Benign, low-register (Fightin' Words `D` side only):** the existing
  quadrant D pool (Alpaca/Dolly/OASST1) — unchanged, already fixed, not
  re-litigated here.

No new dataset is proposed. This is the smallest set that gives clear
harmful supervision (3 already-vetted sources, near-zero measured
cross-source exact-hash overlap per `logs/3a1_source_inventory.md`), clear
benign supervision (1 already-vetted source), real surface-language
diversity on the harmful side (HarmBench's terse red-team-behavior
register vs. StrongREJECT's more conversational, upstream-mixed register
vs. SimpleSafetyTests' blunt hand-crafted register), and zero new
acquisition risk. OR-Bench, PKU-SafeRLHF, and the derived DPO/safety-
training sets are excluded per the audit above, not for size reasons.

## Label mapping

| Source | Label | Field/basis |
|---|---|---|
| HarmBench | harmful | corpus membership (100% red-team behaviors) |
| StrongREJECT | harmful | corpus membership + per-row `source`/`category` |
| SimpleSafetyTests | harmful | corpus membership (100% unsafe requests) |
| XSTest, `type`/label `safe` | benign | per-row field |
| Quadrant D pool (Alpaca/Dolly/OASST1) | benign (low-register, `D` role only) | existing quadrant assignment, unchanged |

Explicitly not used as a label: XSTest's own `unsafe` contrast rows
(different intent, not part of this audit's benign reference); any
response-level field from PKU-SafeRLHF; any `chosen`/`rejected` tag;
dataset name alone for any source not shown above.

## Excluded sources

- **JailbreakBench benign** — unresolved/inaccessible (network policy;
  verified fresh this session, see table).
- **OR-Bench** — excluded: inaccessible (same network policy) and, even if
  reachable, LLM-generated/LLM-labeled provenance is a weaker basis than
  the four hand-curated sources above; also fails "don't add for size"
  (80K rows for a benchmark whose candidate pools run in the hundreds).
- **PKU-SafeRLHF** — excluded: no prompt-level intent field exists in the
  schema this repo already uses; response-level only.
- **Existing DPO/safety-training sets** — excluded: same root cause,
  inherited from PKU-SafeRLHF.

## Deduplication rule

Reuse verbatim, do not recreate: `stripped_sha256` (sha256 of
`.strip()`'d text, exact-duplicate key) and `normalized_sha256` (sha256 of
lowercased/whitespace-collapsed text) from
`src/data_pipeline/build_c_source_authored_candidates.py` (itself a
verbatim carry-over from `check_leakage.py` / `build_secondary_c3_c4.py` —
one hashing convention project-wide). Apply both:
1. within each reference source (should already be ~0 per prior audits);
2. across the three harmful reference sources against each other and
   against quadrant A (StrongREJECT/SimpleSafetyTests already measured at
   0% exact-hash overlap with HarmBench in `logs/3a1_source_inventory.md`
   — reuse that result, don't recompute);
3. XSTest against quadrant B (identity by construction, not a real check);
4. reference sources against whatever candidate pool they are *not*
   currently playing reference for in a given LOSO fold (see below).

Near-duplicate (embedding-cosine) checking remains the same open
environment gap already logged repeatedly (`3a0_source_plan.md` §1.5,
`3a2_candidate_universe.md`): no local path to an embedding model in this
sandbox. Not attempted here; flagged, not silently skipped.

## Reference/candidate separation

**Reference** = whatever subset of instances is used to *fit* a CUE method
(TF-IDF+LR's training labels; Fightin' Words' `H`/`D` word counts).
**Candidate** = the intent-labeled prompt pool being *scored* by an
already-fitted CUE method to determine its high/low-CUE tail membership.

A source may serve as reference for one scoring pass and as candidate for
another, but never both roles for the same pass over the same instances —
this must be enforced by source tag when a scorer is eventually built, not
just by sha256 (sha256 dedup alone would not catch a same-source,
different-row leak of house style/register, which is the actual risk
here, not textual duplication).

## Source-held-out scoring plan

- **Harmful side (3-fold leave-one-source-out):** to score any candidate
  pool drawn from HarmBench, fit both CUE methods on
  {StrongREJECT, SimpleSafetyTests} only (plus the benign side, below); to
  score StrongREJECT-drawn candidates, fit on {HarmBench,
  SimpleSafetyTests}; to score SimpleSafetyTests-drawn candidates, fit on
  {HarmBench, StrongREJECT}. This generalizes what `3a4_scoring.md`
  already did in one direction (excluding StrongREJECT/SimpleSafetyTests
  from `H` while scoring them) into a symmetric rule that also covers
  scoring HarmBench/XSTest themselves — the case the old design never
  needed and never built.
- **Benign side — disclosed limitation, not solved:** XSTest is the only
  accessible benign-high-register source, so there is no held-out fold
  for it; any CUE score for XSTest-drawn material is necessarily fit on a
  reference that either includes XSTest itself (circular) or excludes the
  benign side entirely (asymmetric, weakens the harmful/benign contrast
  Fightin' Words needs). Recommended stance: state this plainly in any
  downstream report exactly as `corpus_discrimination.py` already states
  the "not_identified" caveat for source/label alignment — do not paper
  over it by quietly reusing XSTest as its own reference. Resolving it
  for real requires either a network-policy change (allow
  `huggingface.co`, unblocking JailbreakBench-benign and/or OR-Bench) or a
  new hand-curated benign-high-register source, which is out of this
  project's own established "no generated prompts, no shortcuts" policy
  (`logs/3a1_source_inventory.md`) to produce quickly.
- **Agreement/robust-tail integration:** reuse `empirical_rank()` and
  `assign_strata()` from `corpus_discrimination.py` for both methods
  independently (TF-IDF+LR via its predicted probability or decision
  score, Fightin' Words via `fightin_words_score_unnormalized`), then use
  the already-predeclared, never-yet-wired
  `max_metric_rank_disagreement: 0.25` field in
  `logs/benchmark_gate_config.json` (frozen, `"Do not modify after
  creation"` — read-only here) as the disagreement threshold: an item
  counts as a robust high/low-CUE tail member only if both methods place
  it in the same stratum within that tolerance; larger disagreement routes
  to human review, mirroring the existing `c_review_queue.csv` /
  low-coverage-flag triage pattern already used elsewhere in this project.
- **Class balance, flagged for whoever implements the scorer (not done
  here):** the harmful reference pool across all three sources is much
  larger than the single benign reference pool (150+313+100=563 vs. 250) —
  worth stratified sampling or class weighting in TF-IDF+LR, not naive
  pooling.

## Major overlap/reuse already identified (not recomputed here)

- Quadrant A = HarmBench, 150/150 exact hash (`3a0_source_plan.md` §2).
- Quadrant B = XSTest safe, 250/250 (`3a0_source_plan.md` §2;
  `build_eval_set.py:356-360`).
- StrongREJECT's own `source` column shows ~29% of its 313 rows are
  themselves upstream-derived (DAN 35, AdvBench 25, HarmfulQ 11,
  MaliciousInstruct 12, MasterKey 3, OpenAI System Card 3, Jailbreaking-
  via-Prompt-Engineering 3) — internal redundancy already flagged,
  sub-source disjointness checks still outstanding per
  `logs/3a1_source_inventory.md`.
- JBB's own partial file independently verbatim-matches 11 AdvBench rows,
  9 HarmBench rows, 3 StrongREJECT rows (`logs/3a1b_source_inventory.json`)
  — irrelevant to this note's selected sources (JBB unused) but relevant
  if network policy ever changes and JBB becomes usable: it would need the
  same overlap screening against HarmBench/StrongREJECT before being
  treated as an independent benign *or* harmful source.
- Zero exact-hash overlap measured between HarmBench, StrongREJECT, and
  SimpleSafetyTests (`logs/3a1_source_inventory.md`,
  `3a2_candidate_universe.md`) — the three proposed harmful reference
  sources are already known-independent at the exact-text level.

## Not performed this session

Fitting either CUE method; scoring any candidate; Q10/Q25/Q40
stratification; near-duplicate embedding checks (environment gap,
unchanged from prior sessions); any modification to
`corpus_discrimination.py`, `logs/benchmark_gate_config.json`, or
`data/frozen_v2/*`. The only new acquisition this session was a schema-only
inspection: `pip download jailbreakbench` (PyPI, allowed domain) to
confirm the bundled file has no benign split — the wheel itself was not
committed to this repo.

**Next milestone (not started):** implement the TF-IDF+LR scorer under
the LOSO plan above, alongside the existing `FightinWords` class, and wire
`max_metric_rank_disagreement` into an actual agreement check.
