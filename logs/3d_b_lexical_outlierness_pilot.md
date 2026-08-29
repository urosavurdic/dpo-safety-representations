# 3D-B — Within-Harmful Lexical Outlierness Pilot: Results

Generated: 2026-08-29T04:05:19.083225+00:00 | commit: 8bdb6a4149c1ac6118eaa06c18d96e27463fece8

Implements logs/3d_a_lexical_outlierness_design.md exactly. Does not compute CUE.

## 1. Observed evidence

- Spearman(p_tfidf, p_selfinfo) over 209 jointly-defined rows: **0.29099051803952836**
- High-tail overlap coefficient: 0.8421052631578947 | Jaccard: 0.15384615384615385
- Low-tail overlap coefficient: 0.7931034482758621 | Jaccard: 0.1597222222222222
- Permutation baseline (seed=1337, reps=10000):
  - high_tail_overlap: {'observed': 0.8421052631578947, 'null_mean': 0.4814631578947368, 'null_sd': 0.11020951037585332, 'observed_percentile_in_null': 1.0, 'n_null_defined': 10000}
  - high_tail_jaccard: {'observed': 0.15384615384615385, 'null_mean': 0.08290950915134974, 'null_sd': 0.020494828563770703, 'observed_percentile_in_null': 1.0, 'n_null_defined': 10000}
  - low_tail_overlap: {'observed': 0.7931034482758621, 'null_mean': 0.6597103448275863, 'null_sd': 0.08305322162251574, 'observed_percentile_in_null': 0.9697, 'n_null_defined': 10000}
  - low_tail_jaccard: {'observed': 0.1597222222222222, 'null_mean': 0.12968194232508592, 'null_sd': 0.018381345657014644, 'observed_percentile_in_null': 0.9697, 'n_null_defined': 10000}
- Bootstrap (seed=20260829, reps=10000):
  - Spearman 95% CI: {'lo': 0.15966890938074518, 'hi': 0.41494737502569184, 'n_defined': 10000} (undefined reps: 0)
  - High-tail Jaccard 95% CI: {'lo': 0.08823529411764706, 'hi': 0.22549019607843138, 'n_defined': 10000} (undefined reps: 0)
  - Low-tail Jaccard 95% CI: {'lo': 0.10204081632653061, 'hi': 0.22142857142857142, 'n_defined': 10000} (undefined reps: 0)

### Confound diagnostics (do not hide)

- Tail-by-source (tfidf): {'high': {'StrongREJECT': 8, 'SimpleSafetyTests': 11}, 'low': {'StrongREJECT': 102, 'SimpleSafetyTests': 36}, 'mid': {'StrongREJECT': 22, 'SimpleSafetyTests': 30}}
- Tail-by-source (selfinfo): {'high': {'StrongREJECT': 75, 'SimpleSafetyTests': 26}, 'low': {'StrongREJECT': 9, 'SimpleSafetyTests': 20}, 'mid': {'StrongREJECT': 48, 'SimpleSafetyTests': 31}}
- Category field uninformative-or-fully-source-confounded: **False**
- Spearman(length, percentile): tfidf=-0.5674511106345929, selfinfo=0.25472851840669986
- Source-prediction diagnostic (tail membership alone): {'tfidf': {'n': 209, 'in_sample_accuracy': 0.6842105263157895, 'in_sample_auc': 0.6526957890594255, 'majority_class_baseline_accuracy': 0.631578947368421, 'note': 'in-sample fit/evaluate diagnostic - see AMBIGUITY_NOTES #2'}, 'selfinfo': {'n': 209, 'in_sample_accuracy': 0.6842105263157895, 'in_sample_auc': 0.6487603305785123, 'majority_class_baseline_accuracy': 0.631578947368421, 'note': 'in-sample fit/evaluate diagnostic - see AMBIGUITY_NOTES #2'}}
- Source-balanced sensitivity: {'halted_folds': [], 'n_halted_folds': 0, 'tfidf': {'n_paired': 209, 'spearman_primary_vs_balanced': 0.8768810295495849, 'tail_membership_changed_count': 45}, 'selfinfo': {'n_paired': 209, 'spearman_primary_vs_balanced': 0.8039254493703117, 'tail_membership_changed_count': 77}}
- Category-balanced sensitivity: {'tfidf': {'n_paired': 209, 'spearman_primary_vs_balanced': 0.7718890212756991, 'tail_membership_changed_count': 73}, 'selfinfo': {'n_paired': 209, 'spearman_primary_vs_balanced': 0.7743085660776954, 'tail_membership_changed_count': 68}}
- Length sensitivity: {'tfidf': {'n': 209, 'regression_coef': -0.0069869179485065065, 'regression_intercept': 0.3899509147053789, 'spearman_residual_percentile_vs_original_percentile': 0.7689253596516622, 'tail_membership_changed_count': 117}, 'selfinfo': {'n': 209, 'regression_coef': 0.004871677377618403, 'regression_intercept': 0.5705318682650488, 'spearman_residual_percentile_vs_original_percentile': 0.9416405982209598, 'tail_membership_changed_count': 71}}
- OOV-rate vs percentile (Spearman): {'tfidf': -0.1578063486425422, 'selfinfo': 0.6794053492939433}

## 2. Predeclared criteria

None. logs/3d_a_lexical_outlierness_design.md S11 predeclares no numerical GO/NO-GO threshold. The decision framework there is qualitative (clearly-disagree / agree-but-confounded / agree-and-acceptable), and even that qualitative call is explicitly left to the researcher.

## 3. Researcher decision

**Not made by this task.** Descriptive evidence characterization only: **mixed**. This script does not decide STOP/REDESIGN vs. proceed-to-S10-human-validation; per the 3D-B task brief and S11 of the design doc, that call belongs to the researcher, informed by both the agreement statistics above and the confound diagnostics (a method can pass agreement and still fail suitability for downstream use if diagnostics show source/category/length/formatting dominate the ranking - both conclusions must stay visibly separate, per S9).

## 4. Flagged implementation ambiguities

- **#1 (S2 normalize_text)**: S2 lists whitespace-collapse before the URL-placeholder bullet but does not state execution order between the two. Implemented order: NFKC -> lowercase -> URL substitution -> whitespace collapse/strip. Effect on this pool is expected to be nil (URLs, if any, contain no internal whitespace requiring reordering) but is flagged rather than silently assumed.
- **#2 (S9 item 5, source-prediction diagnostic)**: S9 specifies 'logistic regression predicting source from tail membership alone; report accuracy/AUC against majority-class baseline' but does not specify a train/test split. Implemented as an in-sample fit-and-evaluate on the full scored pool (this is a diagnostic characterizing association in the realized sample, not a held-out predictive-generalization claim).
- **#3 (S9 item 8, length sensitivity 'tail membership changes')**: S9 does not specify how tail membership is redefined from the length-residualized percentile. Implemented: residuals are converted to an empirical percentile within the scored pool using the same '<' + 0.5*'=' / N rule as S6, then the same 0.75/0.25 cutoffs are applied; rows whose tail label differs from the original are counted as 'changed'.
- **#4 (S9 item 4, formatting diagnostics)**: S9 names the required categories (list markers, numbered steps, code-block delimiters, multi-sentence structure) but not exact regexes. Implemented regexes are recorded verbatim in the pilot JSON output's formatting_diagnostic_config field for auditability.
