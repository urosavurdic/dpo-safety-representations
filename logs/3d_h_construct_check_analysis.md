# 3D-H-A — Construct Check Analysis (Blind Human Ratings)

Status: **descriptive analysis of already-collected human ratings only.** Does not modify S2/S3, the frozen benchmark, or any 3D-B/3D-C/3D-H artifact. Not a benchmark GO/NO-GO.

Reviewer question: *"How clearly does the wording itself signal a harmful operational objective?"* (1 = not apparent ... 5 = unmistakably apparent).

n = 32 blind-reviewed prompts (StrongREJECT/SimpleSafetyTests-sourced; join validated against the researcher-only answer key and the original frozen packet).

## Primary analysis (predeclared sampling variable: p_tfidf)

| Group | n | mean | median | sd | 1/2/3/4/5 |
|---|---|---|---|---|---|
| high p_tfidf tail | 16 | 3.188 | 3.0 | 1.424 | 1:3 / 2:1 / 3:6 / 4:2 / 5:4 |
| low p_tfidf tail | 16 | 4.625 | 5.0 | 0.500 | 1:0 / 2:0 / 3:0 / 4:6 / 5:10 |

- Mean difference (high - low): **-1.4375**
- Mann-Whitney U = 50.0, p = 0.002008
- Rank-biserial effect size: **-0.6094**
- Spearman(p_tfidf, rating): rho = -0.4755, p = 0.00595 (n=32)

## Random-label permutation test

- Test statistic: mean difference (high tail mean - low tail mean)
- Observed statistic: -1.4375
- 100000 permutations, seed=20260831
- Empirical two-sided p-value: **0.00102**
- Null distribution: mean=-0.0003, sd=0.4515

Not used as the sole evidence for the decision output below.

## SECONDARY diagnostic (original p_selfinfo tail -- not the primary check)

| Group | n | mean | median | sd | 1/2/3/4/5 |
|---|---|---|---|---|---|
| high p_selfinfo tail | 19 | 3.684 | 4.0 | 1.455 | 1:3 / 2:0 / 3:5 / 4:3 / 5:8 |
| low p_selfinfo tail | 6 | 4.667 | 5.0 | 0.516 | 1:0 / 2:0 / 3:0 / 4:2 / 5:4 |

- mid-tail rows excluded from this comparison: 7
- Mean difference (high - low): -0.9825
- Rank-biserial effect size: -0.3860
- Spearman(p_selfinfo, rating): rho = -0.1150, p = 0.5307

**SECONDARY -- diagnostic only, not the primary construct check**

## Decision output

### SUPPORTIVE

This is a descriptive classification of the construct check, not a formal benchmark GO/NO-GO.

## Interpretation boundaries

- Does NOT establish a universal CUE variable.
- Does NOT establish independence from intent.
- Does NOT establish causal validity.
- Does NOT establish a validated shared A/B/C/D axis.
- At most establishes whether the provisional lexical-outlierness ranking is associated with human judgments of surface harmful-operational explicitness in this small blind sample.
