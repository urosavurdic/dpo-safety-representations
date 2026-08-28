# 3A4 — C-Source-Authored Scoring, Ranking, and Q25 Review Queue

Input: `data/quadrant_c_pipeline/c_source_authored_candidates_validated_v1.jsonl` (sha256 `8be074b8f42e87aa1c3eee83abedec9743e48ce1a65758e329118b34a6c37ca7`, 413 rows; matches 3A3 log: True)

Eligible candidates scored: 209

## Scoring reference (H = A ∪ B, D = quadrant D)
- Eval set: `data/processed/controlled_eval.jsonl` (sha256 `e640c2fba47afe2853c8717ae8492c62bf26cce21f6ec677f68ea88b117c05af`)
- |A| = 150, |B| = 250, |D| = 150
- corpus_h_sha256: `42510627856418066fb0a069e917bd27c869c177e2cc659c92ad15beaf5c5d89`
- corpus_d_sha256: `6623cdcab46fb89914b8017bed911ca176e9417f7b17bbd5fead07814d59a2af`
- prior config: {'prior_strength_per_token': 0.01, 'min_count': 1, 'vocab_size': 2199}
- min_token_recognition_fraction: 0.5
- C-source-authored candidates were NOT used to construct H or D.

## Ranking
- Score field: `fightin_words_score_unnormalized`
- empirical_rank(score, reference_scores): fraction of reference scores <= this score; lower rank = more D-like = more desirable for c_source_authored
- Tie handling: empirical_rank gives exact-score ties identical rank values by construction; any total order needed downstream (queue row order, the review-limit cutoff) breaks ties with (global_rank, record_id) ascending

## Quantiles
- Q10=0.1, Q25=0.25, Q40=0.4
- Default review stratum: Q25
- Review queue limit: 150

## Counts
- Eligible by source: {'StrongREJECT': 132, 'SimpleSafetyTests': 77}
- Qualifying for Q25 by source: {'StrongREJECT': 39, 'SimpleSafetyTests': 13} (total 52)
- Queued by source: {'StrongREJECT': 39, 'SimpleSafetyTests': 13}
- Final queue count: 52
- Capped by limit: False

## Output artifacts
- `data/review/c_source_authored_review_queue.csv` (sha256 `c62725ec37b3d950d7fda164c89d6e71a315cc5dad1a701147100c9f8e8e485a`, 52 rows)

**Next milestone:** 3B - human review of the C-source-authored queue.
