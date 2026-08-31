# C-C -- Execute and Freeze the Existing-C Audit

Status: execution/aggregation only. Reruns the locked C-B contract (`logs/c_existing_construction_audit_spec.md` section 7) unmodified and compiles already-existing secondary evidence. No analysis definition, feature, or inclusion rule is changed or added here. No KEEP/DROP resource decision is made.

C-B implementation commit: `4b700f1f8c41c828d068a9a3b3d723595320ae06`
C-C parent commit: `4b700f1f8c41c828d068a9a3b3d723595320ae06`
Reproducibility check (byte-identical vs. committed C-B output, excluding generation-commit fields): `True`

## 1. Primary: R104 paired analysis (locked C-B contract, re-executed)

### population_1_all_valid_accepted_pairs (n=104)

| Feature | Family | mean(delta) | d_z | Holm-adj p |
|---|---|---|---|---|
| word_count | structural | -11.0000 | -0.8434 | 0.00013 |
| character_count | structural | -48.6827 | -0.7363 | 0.00013 |
| sentence_count | structural | -0.8269 | -0.6832 | 0.00013 |
| mean_word_length | structural | 0.6949 | 1.1886 | 0.00013 |
| has_bullet_marker | formatting_confound | 0.0000 | n/a | 1 |
| has_numbered_step | formatting_confound | 0.0000 | n/a | 1 |
| has_code_block | formatting_confound | 0.0000 | n/a | 1 |
| multi_sentence_flag | formatting_confound | -0.5673 | -1.1395 | 0.00013 |
| fightin_words | lexical_audit | 9.3965 | 0.5180 | 0.00013 |
| fw_z_score | lexical_audit | 1.7679 | 0.1872 | 0.2392 |
| lexical_diversity | lexical_audit | 0.0494 | 0.6073 | 0.00013 |
| lexical_risk_hit_count | lexical_audit | -0.2692 | -0.5312 | 0.00013 |
| cue_tfidf_logreg_margin | distributional_exploratory | 0.3786 | 0.4286 | 0.0002 |

### population_2_assistance_type_preserved_yes (n=78)

| Feature | Family | mean(delta) | d_z | Holm-adj p |
|---|---|---|---|---|
| word_count | structural | -11.3846 | -0.8169 | 0.00013 |
| character_count | structural | -49.7564 | -0.7109 | 0.00013 |
| sentence_count | structural | -0.8718 | -0.6591 | 0.00013 |
| mean_word_length | structural | 0.7316 | 1.2124 | 0.00013 |
| has_bullet_marker | formatting_confound | 0.0000 | n/a | 1 |
| has_numbered_step | formatting_confound | 0.0000 | n/a | 1 |
| has_code_block | formatting_confound | 0.0000 | n/a | 1 |
| multi_sentence_flag | formatting_confound | -0.5641 | -1.1303 | 0.00013 |
| fightin_words | lexical_audit | 8.6252 | 0.4627 | 0.00036 |
| fw_z_score | lexical_audit | 0.8736 | 0.0915 | 1 |
| lexical_diversity | lexical_audit | 0.0507 | 0.6300 | 0.00013 |
| lexical_risk_hit_count | lexical_audit | -0.2692 | -0.5113 | 0.00013 |
| cue_tfidf_logreg_margin | distributional_exploratory | 0.2778 | 0.3056 | 0.0448 |

### population_3_assistance_type_preserved_partial (n=26)

| Feature | Family | mean(delta) | d_z | Holm-adj p |
|---|---|---|---|---|
| word_count | structural | -9.8462 | -0.9812 | 0.00048 |
| character_count | structural | -45.4615 | -0.8431 | 0.0011 |
| sentence_count | structural | -0.6923 | -0.8781 | 0.00344 |
| mean_word_length | structural | 0.5850 | 1.1254 | 0.00039 |
| has_bullet_marker | formatting_confound | 0.0000 | n/a | 1 |
| has_numbered_step | formatting_confound | 0.0000 | n/a | 1 |
| has_code_block | formatting_confound | 0.0000 | n/a | 1 |
| multi_sentence_flag | formatting_confound | -0.5769 | -1.1451 | 0.00066 |
| fightin_words | lexical_audit | 11.7102 | 0.7020 | 0.00798 |
| fw_z_score | lexical_audit | 4.4510 | 0.5080 | 0.0626 |
| lexical_diversity | lexical_audit | 0.0453 | 0.5326 | 0.06186 |
| lexical_risk_hit_count | lexical_audit | -0.2692 | -0.5952 | 0.0656 |
| cue_tfidf_logreg_margin | distributional_exploratory | 0.6809 | 0.9242 | 0.0011 |

## 2. Secondary A/B/C/D distributions (descriptive only; existing evidence)

No cross-quadrant (A-vs-B-vs-C-vs-D) effect size, significance test, or multiple-comparison correction is defined anywhere in this repository. C-C's brief explicitly states C need not differ from every quadrant, and instructs execution only, so no such test is newly defined here. What follows is descriptive-only, reusing numbers already computed and committed by `src/audit_existing_quadrants.py` (A/B/D) and `src/diagnostics/quadrant_composition_check.py` (all four quadrants' lexical-risk-lexicon hit rate), plus a direct read of quadrant C's own category/source fields from the frozen benchmark.

Quadrant C category composition (n=104): {'misinformation_disinformation': 37, 'harassment_bullying': 41, 'illegal': 6, 'cybercrime_intrusion': 20}

Lexical-risk-lexicon hit rate by quadrant: {'A': {'n': 150, 'mean_words': 14.5, 'median_words': 14.0, 'mean_cue_hits': 0.307, 'pct_with_cue_hit': 29.3}, 'B': {'n': 250, 'mean_words': 8.4, 'median_words': 8.0, 'mean_cue_hits': 0.032, 'pct_with_cue_hit': 3.2}, 'C': {'n': 104, 'mean_words': 17.5, 'median_words': 17.0, 'mean_cue_hits': 0, 'pct_with_cue_hit': 0.0}, 'D': {'n': 150, 'mean_words': 16.1, 'median_words': 10.0, 'mean_cue_hits': 0, 'pct_with_cue_hit': 0.0}}

## 3. Confounds

Repeated-template concentration: {'computed': False, 'reason': 'No existing repository artifact defines or computes a repeated-rewrite-template concentration metric for R104 or R-AUTHORED. Defining one now would be a new metric, which is out of scope for this execution-only task. This is reported as an evidence gap, not silently skipped.'}

## 4. R-AUTHORED

Present, n=52, review_status={'pending': 52} -- treated as unlabeled distributional evidence only, per the caveats below.

- word_count here is len(text.split()) (src/data_pipeline/score_and_queue_c_source_authored.py); R104's word_count_source/word_count_candidate columns in data/review/c_review_queue.csv were not verified to use the identical tokenization rule -- treat any numeric gap as approximate, not a controlled contrast.
- fightin_words_score_normalized/fw_z_score here were fit against H=quadrant A union quadrant B vs D=quadrant D (logs/3a4_scoring.md); R104's fightin_words feature in the C-B contract above was fit LOSO with StrongREJECT held out of H (C-B IMPLEMENTATION DECISION 1). These are two different fitted references -- the two fightin_words-family numbers are not on a directly comparable scale and must not be differenced against each other.
- the R-AUTHORED queue was explicitly rank-selected (Q25 = lowest-fightin_words-score quartile, i.e. most D-like) by the 3A4 scoring pipeline, not randomly sampled -- so this population's own fightin_words/fw_z_score distribution is a product of the selection rule, not an independent observation about the source-authored candidate pool as a whole. Comparing it to R104 without accounting for this selection would conflate a sampling artifact with a construction property.

## 5. Decision status (no KEEP/DROP assigned)

Strongest paired signal (population 1): {'feature': 'mean_word_length', 'family': 'structural', 'd_z': 1.1885924839758126, 'holm_adjusted_p': 0.0001299987000129999, 'mean_delta': 0.694947828814967}

Strongest preserved-assistance result (population 2): {'feature': 'mean_word_length', 'family': 'structural', 'd_z': 1.2124327279789686, 'holm_adjusted_p': 0.0001299987000129999, 'mean_delta': 0.7316022323942055}

Strongest confound: {'type': 'length_sensitivity', 'feature': 'lexical_diversity', 'spearman_corr_with_word_count_delta': -0.499799378276997, 'spearman_p_value': 6.607982724922425e-08, 'note': 'Largest-magnitude length association among lexical-audit/distributional-exploratory features in population 1 (all 104 pairs); see also the multi_sentence_flag formatting confound, which is the single largest-|d_z| effect in the entire feature set and is a structural (not lexical-content) change.'}

Contradictory findings:

- fightin_words shows candidates scoring HIGHER (more harmful-associated wording) than sources (population 1 mean delta = 9.396, d_z = 0.518), which is the opposite direction implied by R104's own historical 'reduced_cue_source_rewrite' naming -- already flagged in 3f_a section 2.1 and independently reproduced here, not a new finding.
- In the same population, lexical_risk_hit_count (the fixed lexicon) moves in the opposite direction (mean delta = -0.269, d_z = -0.531) -- i.e. one lexical instrument reports candidates as 'more distinctively harmful-registered' while another reports them as 'triggering fewer fixed-lexicon risk terms.' Both are real, current numbers; they are not reconcilable into a single 'candidates are more/less cue-salient' statement without picking one instrument as authoritative, which this document does not do.
- R-AUTHORED's own fightin_words/fw_z_score distribution sits toward the D-like (benign-associated) end of its reference scale, which could misleadingly read as 'R-AUTHORED is lower-cue than R104' -- but this is a direct artifact of the Q25 rank-selection rule used to build the 52-row queue (see load_r_authored_summary comparability_caveats), not an independent distributional finding, and the two fightin_words scores are fit against different H/D references besides.

What this audit cannot establish:

- Whether R104's paired changes isolate surface-cue wording independent of the harmful/benign construct (C_cue) -- no resource in this repository measures C_cue directly (C-A section 5); this audit does not change that.
- Near-duplicate (as opposed to exact-string) overlap between R104 and quadrant A -- requires embedding-model inference, which is out of scope for this CPU-only task (C-A section 2, restated, not re-tested here).
- Any KEEP / KEEP AS SECONDARY / INCONCLUSIVE / DROP resource decision for R104, R-AUTHORED, 3D-B, or 3D-H (C-A section 7.7; explicitly deferred by both C-B and C-C).
- A formal statistical comparison between R104 and R-AUTHORED, or across A/B/C/D quadrants generally -- no such test is predeclared anywhere in this repository, and defining one is out of scope for this execution-only task.
- Repeated-rewrite-template concentration within R104 or R-AUTHORED -- no existing repository artifact defines this metric (see build_confounds).
- A construct claim of any kind about R-AUTHORED -- 100% of its 52-row queue remains review_status=pending; zero human review has occurred.

**Stop.**
