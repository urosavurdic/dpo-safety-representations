# Naming conventions

This file exists because several names in this repo do not mean what a new reader
would reasonably guess. Read it before renaming anything, and before assuming a
suffix tells you what it seems to.

## "v2" does not imply a "v1"

**There is no `v1` and there never was.** No `v1_pipeline.py`, no `data/frozen_v1/`,
no `results/probes_v1/`. Searching for one is a dead end.

`v2` marks **the frozen 654-row benchmark era**. Its counterpart — the earlier,
mutable 370-row evaluation set — is called `370era` or `legacy` everywhere it
appears (`results/_legacy_370era/`, `LEGACY_370_BASENAMES`,
`tests/fixtures/benchmark_370.jsonl`). The two eras are `370era` and `v2`, not
`v1` and `v2`.

Confusingly, `v2` additionally marks two unrelated things:

| Usage | Meaning | Does a "v1" exist? |
|---|---|---|
| `v2_pipeline.py`, `data/frozen_v2/`, `results/probes_v2/`, `causal_ablation_v2_*` | the frozen 654-row benchmark era | No — the counterpart is `370era` |
| `eval_steering_v2.py`, `results/raw/steering_v2_*` | the second *steering implementation* (single-layer, calibrated coefficient) | Yes — `eval_steering.py` |
| `results/behavioral_eval/summary_v2.json` | the second *refusal classifier* — and this file is **370-era data**, not 654-era | Yes, implicitly |

So `v2` is three different axes wearing one label. When reading a filename, check
which axis it is on before drawing a conclusion.

## `_final` and `_pooled` are pooling modes, not versions

This is the most misleading pair in the repo. Neither means "the final version".

| Suffix | Meaning |
|---|---|
| `_final` | activations at the **final prompt token** |
| `_pooled` | **mean over the last 5 non-padding tokens** |

Ground truth is one line, `src/analysis/final_token_repair.py`:

```python
POOLING_TO_SUFFIX = {"final_token": "final", "mean_last5": "pooled"}
```

Two consequences worth knowing:

- Some geometry outputs are **unsuffixed for final-token** and suffixed `_pooled`
  only for the mean-pooled sensitivity variant — that is, the default filename is
  the non-obvious one.
- The causal/intervention direction used for the headline results is the
  **mean-pooled (last-5)** one, not final-token, despite the preregistration
  fixing final-token. This is a recorded deviation, not an accident.

The same concept is spelled four ways across the codebase: `_final_token`,
`_finaltoken`, `ft_` (as a condition-name prefix), and `_final`. They all mean
final-token.

## Two names that must not be renamed

`data/frozen_v2/` and `src/v2_io.py` are **SHA-pinned by path string** in
`PINNED_INPUT_HASHES` (`quadrant_c_paired_delta_analysis.py`,
`quadrant_population_geometry.py`). Renaming either would require re-pinning the
very hashes that exist to prove nothing drifted. They keep their names
deliberately.

## Other filename suffixes

| Suffix | Meaning |
|---|---|
| `_L24-28` | the layer band an intervention was applied to |
| `_L0-28` | a direction spanning all 29 layer indices — a different axis from the band above |
| `_narrow` / `_wide` | older spelling for layer bands 24-28 and 14-28 respectively |
| `_fullAD` | sensitivity run generating quadrants A and D in full, including the direction-estimation half |
| `_xfit5` | 5-fold cross-fitted (out-of-fold) causal ablation |
| `_QAD` / `_QABCD` | which quadrants a steering run covered |
| `_coef0.5` / `_coef1` / `_coef2` | the steering coefficient |

## Codes used in filenames and docs

| Code | Meaning |
|---|---|
| `M0`–`M3` | training stages: base, SFT-helpful, SFT-safety, DPO |
| `M3_direct` | DPO applied straight to M1, skipping the safety-SFT stage |
| `_alt` | the parallel branch whose M1 trained on a different source corpus |
| `A` / `B` / `C` / `D` | eval quadrants: harmful-overt, benign-harmful-sounding, harmful-reduced-cue, benign-plain |
| `C1`–`C5` | quadrant-C sub-families: reduced-cue, stylistic, contextual, dual-use, evasion-dominant |
| `CF1` / `CF2` / `CF3` | the confirmatory endpoints |
| `WP-*` | work packages in the frozen analysis plan |
| `RED-*`, `P*` | findings from the repository integrity audit |
| `3A*`, `3D-*`, `C-A`…`C-F`, `R104` | internal task identifiers for benchmark-construction work |

## Result row keys

Two keys in the committed causal-ablation rows are easy to misread:

- `stage` holds the **condition** name (e.g. `"M3_baseline"`), and is byte-identical
  to `condition` in every row.
- `model_stage` holds the **actual training stage** (e.g. `"M3"`).

`stage` is a duplicate of `condition`, not of `model_stage`. The committed files
keep both for compatibility.

Also note a genuine collision in statistics output: the key `"b"` means *number of
bootstrap replicates* in the bootstrap CI files, and *McNemar discordant-cell
count* in the direction-specificity files. Same letter, sibling directories,
unrelated quantities.
