# Milestone 3A1 — Consolidated C-Source-Authored Source Policy

**Repository:** `https://github.com/urosavurdic/dpo-safety-representations`
**Branch:** `agent/c-quadrant-end-to-end-e0e2317a`
**Consolidates:** 3A0, 3A1A, 3A1B, 3A1C-0
**Scope:** Policy consolidation only. No acquisition, no candidate creation,
no scoring, no Q10/Q25/Q40 assignment, no review queue, no benchmark
finalization, no gate/classifier/C-paired modification, no GPU/model code.

**Note on 3A1C-0:** `logs/3a1b_reconciliation.json`/`.md` were drafted in
an earlier session that lacked push credentials. Since then, the branch
was fast-forwarded upstream by four commits — one of which is this exact
errata pair, landed as `ff55fa0` (parent `f473639`, the actual 3A1B
commit), byte-for-byte identical to the earlier draft (confirmed by
diff). It is genuinely present in the repository; this milestone does not
recreate it. The other three new commits (`2274921` Milestone 5B,
`d0a3ba9` Milestone 6B, `6e77658` Step 7 follow-up) are confirmed
unrelated GPU/steering-track work, touching only the notebook,
`v2_pipeline.py`, and a new test file — verified to have no bearing on
candidate selection, the gate config, or the classifier.

---

## Strict primary (with record-level eligibility checks still required)

### StrongREJECT
- 158-row disjoint remainder (of 313 total) vs. all 155 existing C-paired
  rows — not only the 104 currently live rows.
- Per-row upstream provenance (`custom`, `DAN`, `AdvBench`,
  `MaliciousInstruct`, `MasterKey`, `OpenAI System Card`) must be
  preserved; no row is described as blanket "human-authored."
- Identity for dedup/disjointness uses sha256 of stripped text — no
  persistent upstream ID exists.
- **Cache status:** `data/raw/3a1a_source_cache/strongreject/strongreject_dataset.csv`
  is **not present** in this fresh clone (gitignored). Reacquisition URL,
  revision, and expected SHA-256 are recorded in the JSON twin — 3A2 must
  reacquire and verify before use.
- Outstanding: sub-source disjointness/wrapper checks on the 29
  non-`custom` rows; human review of the 20 imperative-fallback rows.

### SimpleSafetyTests
- All 100 rows, subject to record-level checks; zero exact-hash overlap
  against StrongREJECT/HarmBench/AdvBench/C-paired (measured in 3A1B).
- No per-row author field — described as an unchanged external dataset
  record with source-level provenance, not individually-verified
  human-authored text.
- 20/100 rows are in the "Child safety" harm area; no example text from
  any harm area has been reproduced in any inventory document, and that
  restraint must continue.
- **Cache status:** not present in this fresh clone (gitignored).
  Reacquisition URL/revision/expected hash recorded in the JSON twin.
- Outstanding: review of the 22 `behavior_description` + 1
  `category_label` rows; near-dup/contamination checks beyond the
  exact-hash pass already done.

---

## Supplementary only

| Source | Why supplementary |
|---|---|
| AdvBench | Near-dup/templating concern (literature-documented) not cleared by exact-dedup alone; only 37/520 rows structurally read as complete requests |
| Do-Not-Answer | Source-generated (GPT-4) then human-filtered — not human-authored regardless of text completeness |
| JailbreakBench partial PyPI file | Missing the official `Source` column; provenance reconstructed by hash; 80-row "unmatched" bucket is an upper bound on "Original," not confirmed |
| JailbreakHub in-the-wild | Genuinely natural provenance, but the classifier's 463/1405 complete-prompt count is a demonstrated overestimate (27/72 spot-checked rows had missed wrapper cues) — per-item review required |
| JailbreakHub `forbidden_question_set.csv` | Researcher-constructed per the dataset's own README, not in-the-wild; 34/390 rows collide with StrongREJECT, 24 already in C-paired |

## Excluded from strict primary

**HarmBench** — live Quadrant A is 100% HarmBench (150/150 by hash), and
HarmBench is part of the fixed `H = A ∪ B` Fightin' Words reference corpus
(confirmed directly in `src/corpus_discrimination.py`). This is a
predeclared protocol decision, not a claim that HarmBench's score
distribution is provably incapable of scoring low. May be retained
supplementary for sensitivity analysis only, never merged into the
strict-primary/Q25 pool.

## Unresolved (not used by 3A2 unless later verified)

- **SORRY-Bench base set** — gated HuggingFace dataset, confirmed from the
  code repo's own README; also blocked by network policy.
- **Authoritative Source-tagged JBB-Behaviors** — HuggingFace-hosted,
  blocked; only a partial, Source-column-less PyPI substitute exists.

---

## Policy correction applied in this milestone

Per explicit instruction: an issue blocks a **source's** inclusion in the
strict-primary pool; it does not block **3A2 as a whole** when the
strict-primary input list simply excludes that source. 3A2 may proceed
with StrongREJECT and SimpleSafetyTests as its only strict-primary inputs;
none of the supplementary or unresolved sources above is a reason to halt
3A2.

## Scientific definition (unchanged, restated for consolidation)

C-source-authored is not a generic harmful-prompt collection and is not
defined by absence of obvious harmful words. The required sequence is:

```
source & provenance audit → complete standalone user-facing request check
→ exact dedup → normalized dedup → near-dup check → contamination check
→ fixed A/B/D Fightin' Words scoring → global & source-stratified ranks
→ Q10/Q25/Q40 → researcher semantic review
```

A low Fightin' Words or TF-IDF score never by itself establishes
harmfulness, completeness, human authorship, or semantic neutrality.

## Classifier limitations (restated, code unchanged)

`src/diagnostics/classify_source_provenance.py`'s `ClassificationResult`
has exactly two fields (`label`, `reason`), no separate confidence field.
Only the imperative-fallback `complete_user_facing_prompt` branch embeds a
low-confidence note in its `reason` string; the opening-verb
`behavior_description` branch does not, despite the module's own docstring
treating both as equally ambiguous. Both must be treated as equally
provisional pending human review. The tool remains triage only — never a
harmfulness or eligibility adjudicator — and was not modified in 3A1A,
3A1B, 3A1C-0, or this milestone.

## Fixed gate configuration (verified unchanged)

`logs/benchmark_gate_config.json`: `Q10=0.10`, `Q25=0.25`, `Q40=0.40`,
default source-authored review stratum `Q25`, review limit `150`. Not
modified this session. No candidate counts or Q25 counts are calculated in
this milestone.

## No-shortcuts policy

No source quotas. No artificial rebalancing. No Q40 top-up to reach 150.
No generated prompts. No rewrites or paraphrases. No silent prompt edits.
Fewer than 150 candidates is an acceptable 3A2 outcome. Rows with pending
review status cannot enter the frozen benchmark.

See `logs/3a1_source_inventory.json` for the full machine-readable
version, including the exact 3A2 input contract and required-checks list.
