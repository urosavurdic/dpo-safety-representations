# Naming conventions

This file exists because several names in this repo do not mean what a new reader
would reasonably guess. Read it before renaming anything, and before assuming a
suffix tells you what it seems to.

## "v2" does not imply a "v1"

**There is no `v1` and there never was.** No `v1_pipeline.py`, no
`data/frozen_v1/`, no `probes_v1/`. Searching for one is a dead end.

`v2` meant **the frozen 654-row benchmark era**. Its counterpart — the earlier,
mutable 370-row evaluation set — is called `370era` (`results/_legacy_370era/`,
`PRE_FREEZE_ARTIFACT_BASENAMES`, `tests/fixtures/benchmark_370.jsonl`). The two
eras are `370era` and `654`, never `v1` and `v2`.

The label has been retired. Code, constants and result files now say `654` for
the frozen era and `370era` for what came before:

| Was | Is |
|---|---|
| `v2_pipeline.py` | `src/pipeline/frozen_run_pipeline.py` |
| `v2_binding_guard.py` | `src/pipeline/binding_guard.py` |
| `validate_benchmark_v2.py` | `src/pipeline/validate_benchmark.py` |
| `causal_ablation_v2_*` | `causal_ablation_654_*` |
| `steering_v2_*` | `steering_654_*` |
| `{stage}_v2_direction*.npy` | `{stage}_direction_654*.npy` |
| `v2_raw*.json` | `responses_654*.json` |
| `results/probes_v2/` | `results/probes_654/` |
| `results/behavioral_judges_v2/` | `results/behavioral_judges/` |
| `cosine_similarity_v2.json` | `cosine_similarity_654.json` |
| **`summary_v2.json`** | **`refusal_rates_370era.json`** |

That last row is the trap. Its `v2` meant the second *refusal classifier*, not
the era, and the file holds **370-era** data (A=50, B=250, C=20, D=50 = 370
rows). Renaming it to `_654` would have relabelled pre-freeze numbers as
post-freeze ones. It was checked by row count before being renamed.

### Two names that keep `v2` deliberately

`data/frozen_v2/` and `src/v2_io.py` are **SHA-pinned by path string** in
`PINNED_INPUT_HASHES`. Renaming either means re-pinning the very hashes that
exist to prove nothing drifted, so both keep their names.

`eval_steering_v2.py` also keeps its name: there its `v2` genuinely means the
second steering implementation, and `eval_steering.py` (the first) is archived.

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

## Files that must not be edited, renamed, or moved

`PINNED_INPUT_HASHES` in `c_b_paired_delta_analysis.py` and `cf_joint_geometry.py`
pins these by **path string and content hash**. Editing one changes its hash;
moving one changes its key. Either breaks the benchmark gate, which exists to
prove that nothing drifted between the frozen benchmark and the analysis that
reads it.

Source files (byte-pinned -- do not edit, not even a docstring):

- `src/v2_io.py`
- `src/cue_scoring.py`
- `src/corpus_discrimination.py`
- `src/diagnostics/score_lexical_risk_cues.py`

Data and config (pinned by path and content):

- `data/frozen_v2/benchmark_v2_20260826T212909Z.jsonl`
- `data/frozen_v2/LATEST_BENCHMARK.json`
- `data/processed/controlled_eval.jsonl`
- `data/review/c_review_queue.csv`
- `data/review/c_source_authored_review_queue.csv`
- `logs/benchmark_gate_config.json`
- `logs/3d_b_lexical_outlierness_pilot.json`

Two of these are easy to trip over: `logs/benchmark_gate_config.json` and
`logs/3d_b_lexical_outlierness_pilot.json` sit in `logs/` among 65 dated audit
dumps, but they are **live pipeline inputs**, not logs. Do not tidy `logs/`
without accounting for them.

To check the pins at any time:

```bash
python -m src.analysis.c_b_paired_delta_analysis --help   # gate runs on execute
```

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
