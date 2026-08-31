# C-E -- R-AUTHORED Distributional Characterization

Status: descriptive characterization only. No promotion/rejection, no KEEP/DROP decision, no new prompts, no modification of R-AUTHORED records or the frozen benchmark.

Input: `data/review/c_source_authored_review_queue.csv`
Input SHA-256: `c62725ec37b3d950d7fda164c89d6e71a315cc5dad1a701147100c9f8e8e485a`
Row count: 52
review_status counts: {'pending': 52}

> R-AUTHORED is a Q25-selected subset (lowest-fightin_words-score quartile of the eligible source-authored candidate pool, i.e. most D-like by that instrument, per src/data_pipeline/score_and_queue_c_source_authored.py and logs/3a4_scoring.md), not a random or representative sample of all source-authored candidates. This report's distribution must not be used to define a new threshold, must not be used to tune any existing metric, and must not be described as an independent external validation set for the current audit. review_status is pending for 100% of rows -- these are UNVALIDATED, not accepted, C labels.

## 1. R-AUTHORED descriptive statistics (n=52)

| Feature | n | mean | median | sd | IQR |
|---|---|---|---|---|---|
| word_count | 52 | 29.7692 | 25.0000 | 24.2977 | 19.7500 |
| character_count | 52 | 161.4808 | 151.0000 | 131.4371 | 111.0000 |
| sentence_count | 52 | 2.0000 | 2.0000 | 1.4951 | 2.0000 |
| mean_word_length | 52 | 4.3129 | 4.3258 | 0.6537 | 0.8335 |
| lexical_diversity | 52 | 0.9050 | 0.9298 | 0.0932 | 0.1542 |
| lexical_risk_hit_count | 52 | 0.1154 | 0.0000 | 0.4272 | 0.0000 |
| has_bullet_marker | 52 | 0.0192 | 0.0000 | 0.1387 | 0.0000 |
| has_numbered_step | 52 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| has_code_block | 52 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| multi_sentence_flag | 52 | 0.6538 | 1.0000 | 0.4804 | 1.0000 |
| fightin_words_score_normalized | 52 | -0.5494 | -0.4163 | 0.4422 | 0.5646 |
| fw_z_score | 52 | -3.8298 | -1.4290 | 12.4759 | 10.7283 |

`fightin_words_score_normalized`/`fw_z_score` are reused as-is from the queue CSV. See section 3 -- these are NOT comparable to R104's fightin_words numbers.

## 2. Unpaired comparison vs. R104 (source / candidate), compatible features only

Descriptive only. `NOT COMPARABLE` never appears here -- these six features share an identical definition and implementation on both sides. Effect size is unpaired (independent-samples) Cohen's d, not R104's own paired d_z. No CI/p-value: not predeclared for an unpaired comparison by the locked contract.

| Feature | R-AUTHORED mean (n=52) | R104 source mean (n=104) | R104 candidate mean (n=104) | d vs. source | d vs. candidate |
|---|---|---|---|---|---|
| word_count | 29.7692 | 28.5385 | 17.5385 | 0.0666 | 0.8561 |
| character_count | 161.4808 | 159.6827 | 111.0000 | 0.0182 | 0.6484 |
| sentence_count | 2.0000 | 1.8269 | 1.0000 | 0.1320 | 1.1623 |
| mean_word_length | 4.3129 | 4.5905 | 5.2855 | -0.4912 | -1.6692 |
| lexical_diversity | 0.9050 | 0.9054 | 0.9548 | -0.0042 | -0.7180 |
| lexical_risk_hit_count | 0.1154 | 0.2692 | 0.0000 | -0.3192 | 0.4693 |

## 3. Fightin' Words / fw_z_score -- restricted comparison

- R-AUTHORED `fightin_words_score_normalized`: NOT COMPARABLE -- different fitted reference
- R-AUTHORED `fw_z_score`: NOT COMPARABLE -- different fitted reference
- Reason: R-AUTHORED's fightin_words_score_normalized/fw_z_score were fit against H=quadrant A union quadrant B, D=quadrant D (logs/3a4_scoring.md). R104's fightin_words feature in the locked C-B contract was fit LOSO with StrongREJECT held out of H (C-B IMPLEMENTATION DECISION 1). Different fitted references, not a common scale -- reported separately, never differenced.

## 4. Comparison vs. frozen A/B/C/D populations (compatible aggregate features)

Word count (mean words):
- R-AUTHORED: n=52, mean=29.77
- Quadrant A: n=150, mean=14.53
- Quadrant B: n=250, mean=8.37
- Quadrant D: n=150, mean=16.06
- quadrant C word-length stats equal R104's candidate-side stats already reported above (quadrant C IS R104's 104 accepted candidates) -- see comparisons_vs_r104.word_count.

Lexical-risk-lexicon hit rate:
- R-AUTHORED: n=52, mean_cue_hits=0.1154, pct_with_cue_hit=7.7
- quadrant_A: n=150, mean_cue_hits=0.307, pct_with_cue_hit=29.3
- quadrant_B: n=250, mean_cue_hits=0.032, pct_with_cue_hit=3.2
- quadrant_C: n=104, mean_cue_hits=0, pct_with_cue_hit=0.0
- quadrant_D: n=150, mean_cue_hits=0, pct_with_cue_hit=0.0

## 5. Not computed (evidence gaps)

- **punctuation_question_density**: NOT COMPUTED -- no implementation of a punctuation/question density feature exists anywhere in this repository, and none is defined in the locked C-A section 7.5 contract. Defining one here would be a new feature definition, out of scope for this task. Evidence gap, not a silent omission.
- **cue_tfidf_logreg_margin**: NOT COMPUTED -- not in this task's required feature-family list; no existing scored value for R-AUTHORED to reuse; scoring it fresh for an unscored population is out of scope for a reuse-only task. Evidence gap, not a silent omission.
- **near_duplicate_check**: NOT RUN -- embedding-model inference is out of scope for this CPU-only task, consistent with the same gap already noted for R104/quadrant A/B/D in the prior C-C audit.

## 6. Decision status

No KEEP/DROP or promote/reject decision is made in this document. review_status remains pending for all 52 rows.

**Stop.**
