# 3A2-1 — C-Source-Authored Candidate Universe

Total candidates constructed: **413** (StrongREJECT: 313, SimpleSafetyTests: 100)

- Eligible for 3A3: **209**
- Excluded (with reason, provenance preserved): **204**

## Input source verification

| Source | Rows | SHA-256 verified |
|---|---|---|
| StrongREJECT | 313 | True |
| SimpleSafetyTests | 100 | True |

## Structural completeness (Step 3, classify_source_provenance.py)

Breakdown across both sources: `{"complete_user_facing_prompt": 342, "behavior_description": 70, "category_label": 1}`

Low-confidence (ambiguous imperative fallback, kept provisional per Step 9): 46

## Duplicate / overlap checks (Steps 4-7)

- Exact duplicates (within/across the two sources): 0
- Normalized-only duplicates: 0
- C-paired pool overlap (checked against the full 155-row pool, not just the live 104): 155 (`{"StrongREJECT": 155, "SimpleSafetyTests": 0}`)
- Quadrant-A overlap (checked against 150 rows): 0

## Provenance (Step 8)

Provenance class breakdown: `{"upstream-derived": 92, "custom": 221, "curated": 100}`

StrongREJECT upstream 'source' column breakdown: `{"DAN": 35, "custom": 221, "Jailbreaking via Prompt Engineering": 3, "AdvBench": 25, "HarmfulQ": 11, "MaliciousInstruct": 12, "OpenAI System Card": 3, "MasterKey": 3}`

Rows whose upstream provenance is 'AdvBench' (AdvBench is itself an excluded strict-primary input - flagged per-record, not excluded, sub-source disjointness unresolved): 25

## Contamination (Step 10 - informational only this milestone)

Exact-text matches against training files (dpo_pairs.jsonl, sft_helpful.jsonl, sft_helpful_alt.jsonl, sft_safety.jsonl): 0

Near-duplicate contamination status: unknown_for_all_records - embedding model unavailable (no network path to huggingface.co in this sandbox)

## Content-sensitivity note

SimpleSafetyTests includes a 'Child safety' harm_area (20/100 rows) and a 'Suicide, Self-Harm, and Eating Disorders' harm_area (20/100 rows). Consistent with the convention already established in logs/3a1b_source_inventory.md, no example prompt text from any harm_area is reproduced in this report - counts and category labels only. Full exact text is preserved only in the structured JSONL data artifact, as required for reproducibility.

## Not performed this milestone

- near_duplicate_embedding_check
- fightin_words_scoring
- Q10_Q25_Q40_stratification
- review_queue_generation
- benchmark_freeze
- project_category_taxonomy_mapping

## Output artifact

`data/quadrant_c_pipeline/c_source_authored_candidates_raw_v1.jsonl` (sha256 `921ebe1687f2926115d4b1d846a97aebaaae0d0d10d5943e141bf2be696581c1`, 413 rows)

**Next milestone:** 3A3 - candidate validation, near-duplicate, overlap, and contamination checks.
