# Reconciliation Report — v2 Benchmark Audit

**Base commit:** `e0e2317a52e89b0d614b99152a9ca71758baf489`
**Branch:** `agent/c-quadrant-end-to-end-e0e2317a`

---

## Evaluation Set

| Field | Value |
|---|---|
| Path | `data/processed/controlled_eval.jsonl` |
| SHA-256 | `e640c2fb...` |
| Rows | 654 |
| Schema | `{prompt, quadrant, source, category, split}` |
| C rows | 104 |
| c_construction field | **MISSING** |
| record_id field | **MISSING** |
| pair_id field | **MISSING** |
| source_prompt field | **MISSING** |

**Quadrant counts:** A=150, B=250, C=104, D=150
**A/D split:** 240 direction_estimation + 60 held_out_behavioral

---

## C-Paired Provenance — CRITICAL FINDING

Only **15 of 104** live C records had provenance in `candidate_records.jsonl`
(the original 20-item file from the 15-candidate era). The remaining **89 records**
were added to `QUADRANT_C_RECORDS` in `build_eval_set.py` during the 15→104 scaling
session without updating the output artifact.

**Resolution this session:** `candidate_records_v2.jsonl` has been written by
extracting all 155 entries from the `CANDIDATES` list in `quadrant_c_pipeline.py`.
All 104 live C prompts are verified (exact_live_candidate_match=True).
Source prompts are present for all 155 entries. The StrongREJECT source CSV
SHA-256 is not available (network-fetched at build time, not stored locally).

**Review evidence status for all 104 rows:** `missing_or_list_level_only`
A list-level note ("Reviewer: project owner via conversation. Date: 2026-08-23")
exists in `build_eval_set.py` but per-record review fields are absent.
All rows are marked `pending` in the review queue.

---

## Activation Artifacts — STALE

| File | Rows | Expected | Status |
|---|---|---|---|
| All 9 `*_metadata.json` | 370 | 654 | **STALE** |
| `*.npy` arrays | 0 | present | **MISSING** |
| Eval hash in metadata | No | Yes | **MISSING** |

No `.npy` activation arrays are present in `results/activations/`. All
behavioral, probe, direction, and ablation results are from the 370-row era
and cannot be used for claims about the current 654-row evaluation set.
**GPU rerun is required.**

---

## A/B/D Audit — CLEAN

| Quadrant | Exact dupes | Norm dupes | Training contamination |
|---|---|---|---|
| A | 0 | 0 | 0 (all 4 training files) |
| B | 0 | 0 | 0 |
| D | 0 | 0 | 0 |

Near-duplicate check: not run (model unavailable in sandbox). Result: unknown.

---

## Fightin' Words Diagnostics

H = A∪B (400 prompts); D = quadrant D (150 prompts). Vocab: 2199 tokens.

| Metric | Value |
|---|---|
| Live C zero hits (risk lexicon) | 104/104 (100%) |
| Q10 (≤ 10th pct of 155-candidate dist.) | 11 |
| Q25 | 28 |
| Q40 | 43 |
| Low coverage flags | 0 |
| Positive paired_score_difference | 34/104 |

Positive paired_score_difference means the candidate prompt has a lower
Fightin' Words score than its source prompt — operationally more D-like.
This is a screening metric only. It is not proof of semantic neutrality and
does not establish a wording-only causal effect.

---

## Schema Issues Confirmed

- `model_stage` vs `stage` mismatch: `causal_ablation_raw_narrow.json` uses
  `model_stage`; `causal_ablation_raw_wide.json` uses `stage`.
- No eval-set hash embedded in any result artifact.
- C construction arm field absent from live eval file.

---

## Blockers Before Benchmark Can Be Frozen

1. **Researcher must review `data/review/c_review_queue.csv`** (104 rows, all `pending`).
   Set `review_status` to `accept` or `reject`, fill `review_notes` and preservation fields.
   Do not edit any prompt text — reject and note if a prompt needs changes.
2. **GPU rerun required** for all mechanistic results (activations, directions, behavioral, ablation).
3. **C-source-authored arm not built** — HarmBench CSV unavailable. Conclusions limited to C-paired.

---

## Artifacts Created This Session

| File | Purpose |
|---|---|
| `logs/benchmark_gate_config.json` | Fixed gate parameters (written before validation) |
| `logs/direction_split_manifest.json` | A/D split record (seed=45) |
| `data/quadrant_c_pipeline/candidate_records_v2.jsonl` | 155-item provenance file |
| `data/review/c_review_queue.csv` | **The thing you need to review** (104 rows) |
| `data/review/d_spot_check.csv` | 50-row D manual spot-check |
| `src/corpus_discrimination.py` | Fightin' Words implementation |
| `src/audit_existing_quadrants.py` | A/B/D audit script |
| `src/finalize_benchmark.py` | Post-review benchmark freezer |
| `src/validate_benchmark_v2.py` | Validation and status report |
| `rerun_mechanistic_v2.sh` | GPU gate + orchestrator |
