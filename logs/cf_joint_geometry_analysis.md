# C-F -- Joint-Geometry Analysis (A/B/C/D + R104-source, R-AUTHORED)

Status: implementation of the locked contract in `logs/cf_joint_geometry_spec.md` (C-F-A). Descriptive/geometric reporting only -- no construct-validity decision is made here (C-F-A section 0 / section 5.2's explicit non-inference rule).

Generation commit: `854744300749f664ee9c42c9295d3a8d4a754295`

## Population counts

| Population | n |
|---|---|
| A | 150 |
| B | 250 |
| C | 104 |
| D | 150 |
| R104_source | 104 |
| R_AUTHORED | 52 |

## Dropped / surviving structural features

Dropped (zero-variance on A u B u C u D): ['has_code_block']
Surviving: ['word_count', 'character_count', 'sentence_count', 'mean_word_length', 'lexical_diversity', 'has_bullet_marker', 'has_numbered_step', 'multi_sentence_flag', 'lexical_risk_hit_count']

## Centroid pairwise distances by view

### structural_only

- AB: 2.2628
- AC: 1.8099
- AD: 2.1990
- BC: 1.9815
- BD: 1.6876
- CD: 1.2515
- intent/surface contrast angle: 84.03 deg (geometric fact about this fit only; not evidence for/against latent independence)

### lexical_only

- AB: 1.0638
- AC: 1.3147
- AD: 2.6375
- BC: 0.6039
- BD: 2.1949
- CD: 1.8008
- intent/surface contrast angle: 19.47 deg (geometric fact about this fit only; not evidence for/against latent independence)

### combined

- AB: 0.8884
- AC: 0.8281
- AD: 1.1377
- BC: 0.9684
- BD: 1.0204
- CD: 0.9608
- intent/surface contrast angle: 80.81 deg (geometric fact about this fit only; not evidence for/against latent independence)

## PCA (section 5.3 -- reporting/visualization only)

Cumulative explained variance, PC1-PC2: 0.4361
Cumulative explained variance, PC1-PC3: 0.5569
PC1-PC2 plot: `logs/cf_joint_geometry_pca_pc1_pc2.png`
PC1-PC3 plot: `logs/cf_joint_geometry_pca_pc1_pc3.png`
Populations projected: 810 rows across 6 populations (R104-source/R-AUTHORED transformed only, never fit).

A visually clean four-cluster PC1-PC2 plot is not, by itself, evidence for the intended construct, and its absence is not, by itself, evidence against it (section 5.3's explicit non-criterion) -- PCA here is reporting, not testing.

## R-AUTHORED caveats (repeated per C-F-A section 6)

- review_status: pending (C-E)
- Q25 rank-selection bias (C-D Gate 4)

## Embeddings (section 7)

Status: not_run -- optional and non-blocking (section 7); not requested (--run-embeddings not set)

## Explicit non-actions

- did not modify any C-F-A section 1 pinned input, candidate wording, or frozen benchmark file
- did not redefine any feature or add a metric beyond section 5's five required analysis blocks
- did not add alternative preprocessing after inspecting results
- did not train a supervised classifier to define an axis, and did not run contrastive/representation-learning training
- did not run GPU inference; section-7 embeddings reported as not_run unless --run-embeddings is set, and even then no network access is available in this sandbox
- did not characterize any PCA axis, distance, or divergence as evidence for or against latent psychological independence of intent and surface cue (section 5.2)
- did not decide whether C is valid, whether the intended 2x2 construct holds, or whether R-AUTHORED should be promoted
- did not print raw prompt text, matched lexicon terms, or per-token/per-bigram JSD contributions
