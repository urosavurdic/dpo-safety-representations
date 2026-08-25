# Benchmark v2 Validation Report
Generated: 2026-08-25T22:50:59.694421+00:00

**Technical benchmark status:** `FAIL`
**Reduced-cue evidence status:** `INCONCLUSIVE`
**Wording-only claim status:** `INCONCLUSIVE` (fixed — this project does not identify a causal effect of wording alone)

## Gate fields
| Field | Value |
|---|---|
| schema_integrity_pass | True |
| prompt_integrity_pass | True |
| c_review_pass | True |
| benchmark_hash_pass | True |
| artifact_freshness_pass | False |
| source_confound_pass | False |
| category_confound_pass | None |
| prompt_function_confound_pass | None |
| length_confound_pass | False |
| surface_separation_pass | False |
| wording_only_claim_pass | INCONCLUSIVE |

## Counts
| Quadrant | Count |
|---|---|
| A | 150 |
| B | 250 |
| C | 104 |
| D | 150 |

## C arm counts
| Arm | Count |
|---|---|
| c_paired | 104 |

## Paired Fightin' Words diagnostics
- n_pairs: 104
- n_positive_diff: 34
- mean_diff: -9.3965
- 95% CI: [-12.8489, -5.842]
- SMD (not Cohen's d): -0.518
- Note: Positive diff = candidate has lower Fightin' Words score than source. Not a causal effect of wording alone (source-confounded).

## Stale artifacts
- M0_metadata.json: 370 rows (benchmark has 654)
- M1_alt_metadata.json: 370 rows (benchmark has 654)
- M1_metadata.json: 370 rows (benchmark has 654)
- M2_alt_metadata.json: 370 rows (benchmark has 654)
- M2_metadata.json: 370 rows (benchmark has 654)
- M3_alt_metadata.json: 370 rows (benchmark has 654)
- M3_direct_alt_metadata.json: 370 rows (benchmark has 654)
- M3_direct_metadata.json: 370 rows (benchmark has 654)
- M3_metadata.json: 370 rows (benchmark has 654)

## Warnings
- Stale activation metadata (9 files). GPU rerun required before mechanistic analysis.
- Length confound: |d|=0.704 between A and C-paired word counts. Source-confounded comparison — label accordingly.
- source_cue_effect_status=not_identified: A and C-paired come from entirely different datasets (HarmBench vs StrongREJECT). Source and label are nearly perfectly aligned. A-versus-C differences cannot be attributed to wording alone.
