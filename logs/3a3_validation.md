# 3A3 — C-Source-Authored Candidate Universe Validation

Overall status: **validated_with_unknowns**

Input: `data/quadrant_c_pipeline/c_source_authored_candidates_raw_v1.jsonl` (sha256 `921ebe1687f2926115d4b1d846a97aebaaae0d0d10d5943e141bf2be696581c1`, 413 rows)

## Checks

| Check | Result |
|---|---|
| exact_duplicate_validation | pass |
| normalized_duplicate_validation | pass |
| near_duplicate_validation | unknown |
| training_contamination_exact | pass |
| training_contamination_near | unknown |
| overlap_reverification_c_paired_and_quadrant_a | pass |
| source_provenance_validation | pass |
| exact_prompt_preservation | pass |
| structural_classifier_validation | pass |

## Duplicate re-verification (Steps 1-2)
- Exact duplicates (recomputed): 0 (mismatches vs 3A2-1: 0)
- Normalized-only duplicates (recomputed): 0 (mismatches vs 3A2-1: 0)

## Overlap re-verification (Steps 6-7 of 3A2-1, re-checked here)
- C-paired pool overlap (full 155-row pool): 155 (mismatches: 0)
- Quadrant-A overlap (checked against 150 rows): 0 (mismatches: 0)

## Near-duplicate validation (Step 3)
- Config: model=`all-MiniLM-L6-v2`, threshold=0.9, method=sentence-transformers cosine similarity, repo convention from src/diagnostics/check_leakage.py and src/diagnostics/complete_neardup_check.py
- **Blocked**: OSError: We couldn't connect to 'https://huggingface.co' to load the files, and couldn't find them in the cached files.
Check your internet connection or see how to run the library in offline mode at 'https://huggingface.co/docs/transformers/installation#offline-mode'.
- Status for all five required comparisons (within StrongREJECT; within SimpleSafetyTests; StrongREJECT vs SimpleSafetyTests; candidates vs Quadrant-A; candidates vs the full 155-row C-paired pool): **unknown** — not converted to "clean".

## Training contamination (Step 4)
- Training files checked: dpo_pairs.jsonl, sft_helpful.jsonl, sft_helpful_alt.jsonl, sft_safety.jsonl (all present in this checkout)
- Exact-match contamination (re-verified independently): {} (mismatches vs 3A2-1: 0)
- Near-dup contamination status by file: {'dpo_pairs.jsonl': 'unknown', 'sft_helpful.jsonl': 'unknown', 'sft_helpful_alt.jsonl': 'unknown', 'sft_safety.jsonl': 'unknown'}

## Source/provenance validation (Step 5)
- StrongREJECT: reacquired fresh from `https://raw.githubusercontent.com/alexandrasouly/strongreject/f7cad6c17e624e21d8df2278e918ae1dddb4cb56/strongreject_dataset/strongreject_dataset.csv`, sha256 verified: True
- SimpleSafetyTests: reacquired fresh from `https://raw.githubusercontent.com/bertiev/SimpleSafetyTests/d7aee9a9422a5a5488f478fd79c2479c891c0f3b/SimpleSafetyTests - test cases.csv`, sha256 verified: True
- Per-record provenance field + deterministic record_id re-check: pass (mismatches: 0)

## Exact prompt preservation (Step 6)
- Rows checked byte-for-byte against freshly reacquired source rows: 413
- Mismatches: 0

## Structural classifier (Step 7)
- Determinism mismatches: 0
- Non-standalone/ambiguous rows silently promoted to eligible_for_3a3: 0
- Low-confidence rows kept provisional (not auto-excluded, not auto-promoted): 46

## Counts
- Total candidates: 413
- Eligible for 3A3 (unchanged from 3A2-1 - no correction was required): 209
- Excluded: 204
- Exclusion reason counts: {'overlaps_c_paired_pool': 155, 'not_standalone_user_facing_request': 71}

## Output artifacts
- `data/quadrant_c_pipeline/c_source_authored_candidates_validated_v1.jsonl` (sha256 `8be074b8f42e87aa1c3eee83abedec9743e48ce1a65758e329118b34a6c37ca7`, 413 rows)

**Next milestone:** 3A4 - fixed Fightin' Words scoring, ranking, quantiles, and review queue.
