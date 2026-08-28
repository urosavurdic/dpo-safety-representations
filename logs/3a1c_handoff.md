# Milestone 3A1C — Handoff to 3A2

**Branch:** `agent/c-quadrant-end-to-end-e0e2317a`
**HEAD before this commit:** `6e7765815dc6dec579c7f0ca4cade358bbd3ee4d`
**3A1B commit/parent:** `f473639` / `9778cfc` (verified reachable)
**3A1C-0 commit/parent:** `ff55fa0` / `f473639` (present and preserved —
confirmed genuinely landed on the branch, byte-identical to its original
draft)
**Historical unreachable SHAs not used as anchors:** `a8aa91a9…`, `17d82da…`

Three unrelated commits landed on the branch since 3A1B (Milestone 5B,
Milestone 6B, and a follow-up) — all confirmed out of scope for the
C-source-authored track (GPU/steering/notebook work only; the follow-up
commit's own message independently corroborates this).

---

## Strict-primary sources (record-level checks still required in 3A2)

1. **StrongREJECT** — 158-row disjoint remainder vs. all 155 existing
   C-paired rows.
2. **SimpleSafetyTests** — 100 rows; the heuristic classifier count is
   not the final eligible count.

## Supplementary sources (excluded from strict-primary input list)

AdvBench, Do-Not-Answer, the partial JBB PyPI file, JailbreakHub
in-the-wild, JailbreakHub `forbidden_question_set.csv`.

## Excluded

HarmBench (predeclared protocol decision — reference-corpus dependence).

## Unresolved

SORRY-Bench base dataset; the authoritative Source-tagged JBB-Behaviors
dataset.

## Policy correction applied

An issue blocks a **source's** inclusion in the strict-primary pool; it
does not block **3A2 as a whole**. Since the strict-primary input list
below excludes every supplementary/unresolved source, none of their open
issues blocks 3A2.

---

## Required 3A2 input files

| Source | Path | Status |
|---|---|---|
| StrongREJECT | `data/raw/3a1a_source_cache/strongreject/strongreject_dataset.csv` | **Not present** in working tree (gitignored) — reacquire from `alexandrasouly/strongreject` @ `f7cad6c1…`, verify SHA-256 `4dd70357…` before use |
| SimpleSafetyTests | `data/raw/3a1b_source_cache/simplesafetytests/SimpleSafetyTests - test cases.csv` | **Not present** in working tree (gitignored) — reacquire from `bertiev/SimpleSafetyTests` @ `d7aee9a9…`, verify SHA-256 `6d95a130…` before use |

Any fresh clone/sandbox will lack `data/raw/` entirely — 3A2 must check
for and hash-verify both files in every session, not assume a prior
report's word for it.

**Explicitly disallowed as strict-primary inputs:** HarmBench, AdvBench,
Do-Not-Answer, the partial JBB PyPI file, SORRY-Bench, JailbreakHub
in-the-wild, `forbidden_question_set.csv`.

---

## Required 3A2 checks (mandatory, in this order of concern — not all sequential)

1. Exact prompt-text preservation
2. Deterministic IDs independent of row position
3. Exact duplicate detection
4. Normalized duplicate detection
5. Near-duplicate detection using existing repository machinery where available
6. Overlap against all Quadrant-A rows
7. Overlap against all 155 existing C-paired records
8. Contamination checks against every available training file
9. Explicit unknown status when a required check cannot run
10. Preservation of source, revision, source-file hash, prompt hash, and provenance
11. Source-category and project-category mapping without invented labels
12. No use of model behavior, activations, probes, steering, or causal results

## Distinctions 3A2 must maintain

- **Source eligibility** — decided here, not re-litigated per candidate.
- **Provenance class** — per-row/per-source, travels with every record.
- **Structural classifier triage** — advisory only; both the opening-verb
  and imperative-fallback buckets remain provisional; never an
  auto-include/exclude signal.
- **Later semantic researcher review** — human judgment, downstream of triage.
- **Fixed score-based selection** — A/B/D Fightin' Words + Q10/Q25/Q40,
  applied only after all of the above, never as a substitute for any of it.

## Fixed gate (verified unchanged, not modified this milestone)

`Q10=0.10`, `Q25=0.25`, `Q40=0.40`, default stratum `Q25`, review limit
`150` — from `logs/benchmark_gate_config.json`.

## No-shortcuts policy

No quotas. No rebalancing. No Q40 top-up to reach 150. No generated
prompts. No rewrites/paraphrases. No silent edits. Fewer than 150
candidates is acceptable. Pending-review rows cannot enter the frozen
benchmark.

---

**Next milestone: 3A2.** Not begun in this session.
