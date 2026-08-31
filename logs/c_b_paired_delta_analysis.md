# C-B -- Paired Delta Analysis (R104 `c_paired` Quadrant-C Construction)

Status: implementation of the locked contract in `logs/c_existing_construction_audit_spec.md` section 7. Descriptive and inferential statistics only -- no KEEP/DROP resource decision is made by this document (C-A section 7.7).

Generation commit: `ddba4c1b7fc2f34dc6477e110daf0e4c3bfff505` (working tree dirty: True)

## Provenance integrity

- Frozen quadrant-C rows: 104
- Review-queue rows: 104
- record_id sets match: True

## Pair integrity

| Population | expected_pairs | valid_pairs | excluded_pairs |
|---|---|---|---|
| population_1_all_valid_accepted_pairs | 104 | 104 | 0 |
| population_2_assistance_type_preserved_yes | 78 | 78 | 0 |
| population_3_assistance_type_preserved_partial | 26 | 26 | 0 |

## Per-population, per-feature summary

### population_1_all_valid_accepted_pairs (n=104)

| Feature | Family | valid_n | mean(delta) | d_z | bootstrap 95% CI | perm p | Holm-adj p |
|---|---|---|---|---|---|---|---|
| word_count | structural | 104 | -11.0000 | -0.8434 | [-13.5481, -8.5096] | 1e-05 | 0.00013 |
| character_count | structural | 104 | -48.6827 | -0.7363 | [-61.5099, -35.9325] | 1e-05 | 0.00013 |
| sentence_count | structural | 104 | -0.8269 | -0.6832 | [-1.0673, -0.6058] | 1e-05 | 0.00013 |
| mean_word_length | structural | 104 | 0.6949 | 1.1886 | [0.5810, 0.8049] | 1e-05 | 0.00013 |
| has_bullet_marker | formatting_confound | 104 | 0.0000 | n/a | [0.0000, 0.0000] | 1 | 1 |
| has_numbered_step | formatting_confound | 104 | 0.0000 | n/a | [0.0000, 0.0000] | 1 | 1 |
| has_code_block | formatting_confound | 104 | 0.0000 | n/a | [0.0000, 0.0000] | 1 | 1 |
| multi_sentence_flag | formatting_confound | 104 | -0.5673 | -1.1395 | [-0.6635, -0.4712] | 1e-05 | 0.00013 |
| fightin_words | lexical_audit | 104 | 9.3965 | 0.5180 | [5.9829, 13.0041] | 1e-05 | 0.00013 |
| fw_z_score | lexical_audit | 104 | 1.7679 | 0.1872 | [-0.0311, 3.6177] | 0.0598 | 0.2392 |
| lexical_diversity | lexical_audit | 104 | 0.0494 | 0.6073 | [0.0335, 0.0652] | 1e-05 | 0.00013 |
| lexical_risk_hit_count | lexical_audit | 104 | -0.2692 | -0.5312 | [-0.3750, -0.1731] | 1e-05 | 0.00013 |
| cue_tfidf_logreg_margin | distributional_exploratory | 104 | 0.3786 | 0.4286 | [0.2053, 0.5473] | 4e-05 | 0.0002 |

Source sensitivity: R104 is 100% StrongREJECT (C-A section 2) -- no within-R104 source contrast is computable.

KEEP FOR HUMAN REVIEW / KEEP AS SECONDARY / INCONCLUSIVE / DROP labels (C-A section 7.7) are not assigned by this implementation -- final interpretation is out of scope for the C-B implementation task.

### population_2_assistance_type_preserved_yes (n=78)

| Feature | Family | valid_n | mean(delta) | d_z | bootstrap 95% CI | perm p | Holm-adj p |
|---|---|---|---|---|---|---|---|
| word_count | structural | 78 | -11.3846 | -0.8169 | [-14.5256, -8.3333] | 1e-05 | 0.00013 |
| character_count | structural | 78 | -49.7564 | -0.7109 | [-65.5388, -34.3327] | 1e-05 | 0.00013 |
| sentence_count | structural | 78 | -0.8718 | -0.6591 | [-1.1795, -0.6026] | 1e-05 | 0.00013 |
| mean_word_length | structural | 78 | 0.7316 | 1.2124 | [0.5977, 0.8658] | 1e-05 | 0.00013 |
| has_bullet_marker | formatting_confound | 78 | 0.0000 | n/a | [0.0000, 0.0000] | 1 | 1 |
| has_numbered_step | formatting_confound | 78 | 0.0000 | n/a | [0.0000, 0.0000] | 1 | 1 |
| has_code_block | formatting_confound | 78 | 0.0000 | n/a | [0.0000, 0.0000] | 1 | 1 |
| multi_sentence_flag | formatting_confound | 78 | -0.5641 | -1.1303 | [-0.6667, -0.4487] | 1e-05 | 0.00013 |
| fightin_words | lexical_audit | 78 | 8.6252 | 0.4627 | [4.6338, 12.7350] | 6e-05 | 0.00036 |
| fw_z_score | lexical_audit | 78 | 0.8736 | 0.0915 | [-1.1566, 2.9882] | 0.4235 | 1 |
| lexical_diversity | lexical_audit | 78 | 0.0507 | 0.6300 | [0.0322, 0.0688] | 1e-05 | 0.00013 |
| lexical_risk_hit_count | lexical_audit | 78 | -0.2692 | -0.5113 | [-0.3846, -0.1538] | 1e-05 | 0.00013 |
| cue_tfidf_logreg_margin | distributional_exploratory | 78 | 0.2778 | 0.3056 | [0.0786, 0.4742] | 0.00896 | 0.0448 |

Source sensitivity: R104 is 100% StrongREJECT (C-A section 2) -- no within-R104 source contrast is computable.

KEEP FOR HUMAN REVIEW / KEEP AS SECONDARY / INCONCLUSIVE / DROP labels (C-A section 7.7) are not assigned by this implementation -- final interpretation is out of scope for the C-B implementation task.

### population_3_assistance_type_preserved_partial (n=26)

| Feature | Family | valid_n | mean(delta) | d_z | bootstrap 95% CI | perm p | Holm-adj p |
|---|---|---|---|---|---|---|---|
| word_count | structural | 26 | -9.8462 | -0.9812 | [-13.7308, -6.2308] | 4e-05 | 0.00048 |
| character_count | structural | 26 | -45.4615 | -0.8431 | [-66.3846, -26.2683] | 0.00011 | 0.0011 |
| sentence_count | structural | 26 | -0.6923 | -0.8781 | [-1.0000, -0.4231] | 0.00043 | 0.00344 |
| mean_word_length | structural | 26 | 0.5850 | 1.1254 | [0.3941, 0.7788] | 3e-05 | 0.00039 |
| has_bullet_marker | formatting_confound | 26 | 0.0000 | n/a | [0.0000, 0.0000] | 1 | 1 |
| has_numbered_step | formatting_confound | 26 | 0.0000 | n/a | [0.0000, 0.0000] | 1 | 1 |
| has_code_block | formatting_confound | 26 | 0.0000 | n/a | [0.0000, 0.0000] | 1 | 1 |
| multi_sentence_flag | formatting_confound | 26 | -0.5769 | -1.1451 | [-0.7692, -0.3846] | 6e-05 | 0.00066 |
| fightin_words | lexical_audit | 26 | 11.7102 | 0.7020 | [5.5546, 18.1243] | 0.00114 | 0.00798 |
| fw_z_score | lexical_audit | 26 | 4.4510 | 0.5080 | [1.3539, 7.9355] | 0.01252 | 0.0626 |
| lexical_diversity | lexical_audit | 26 | 0.0453 | 0.5326 | [0.0144, 0.0783] | 0.01031 | 0.06186 |
| lexical_risk_hit_count | lexical_audit | 26 | -0.2692 | -0.5952 | [-0.4615, -0.1154] | 0.0164 | 0.0656 |
| cue_tfidf_logreg_margin | distributional_exploratory | 26 | 0.6809 | 0.9242 | [0.4078, 0.9559] | 0.00011 | 0.0011 |

Source sensitivity: R104 is 100% StrongREJECT (C-A section 2) -- no within-R104 source contrast is computable.

KEEP FOR HUMAN REVIEW / KEEP AS SECONDARY / INCONCLUSIVE / DROP labels (C-A section 7.7) are not assigned by this implementation -- final interpretation is out of scope for the C-B implementation task.

## CUE score construct-relevance caveat

Not construct-relevant: 3f_a already disqualifies this score as C_cue ground truth, because it is fit directly on the harmful/benign label. Reported for completeness only -- a significant paired difference on this feature must not be described as evidence about C_cue.

## Software versions (actual runtime; C-A section 7.8)

- Python: 3.12.3 (main, Mar  3 2026, 12:15:18) [GCC 13.3.0]
- numpy: 2.4.4
- scipy: 1.17.1
- pandas: 3.0.2
- scikit-learn: 1.8.0

## Explicit non-actions (mirrors C-A section 9 / task brief scope)

- Did not create, rewrite, or modify any prompt.
- Did not modify any frozen input listed in section 7.1.
- Did not run model inference or GPU code, and did not access the web.
- Did not begin B/D construction or common-CUE/contrastive construction.
- Did not assign a KEEP/KEEP-AS-SECONDARY/INCONCLUSIVE/DROP label.
- Did not analyze R-AUTHORED: C-A section 3/6 states R-AUTHORED analysis has not started (100% review_status=pending) and section 7 does not include it in the locked contract.

**Stop.**
