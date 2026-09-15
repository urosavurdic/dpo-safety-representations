# What is computed, what needs recomputing, and the exact commands

Written 2026-09-10, after reconciling the repo against
`Downloads/dpo_v2-20260908T091342Z-1-{001,002,003}.zip`. Companion to
`results/DATA_STATE.md` (which describes the data state itself).

---

## 0. Correction to the 2026-09-10 audit

My earlier audit (`paper_draft/AUDIT.md` §3) listed seven analysis outputs as
"mixed-benchmark contaminated — invalid for cross-stage claims". **That verdict
was wrong. All three files I tested reproduce from the fresh activations.**

`results/refusal_direction/projection_trajectory.json` **reproduces
byte-identically** when re-run against the freshly imported 654-row `_final`
activations across all nine stages. It was computed on the Colab machine, where
all nine stages were already fresh. I inferred staleness from my *local*
inability to reproduce it — the local checkout had 370-row copies — and stated
that inference as a property of the file. It isn't.

Same for `results/interpretability/factorial_direction_audit.json`: re-running it
on nine stages left the four pre-existing stages **numerically identical** and
only added the five that were missing. And
`results/interpretability/bootstrap_cross_branch_difference.json` re-runs to
7 significant figures — the sole diff is a float last-digit
(`0.0443504304` → `0.0443504266`), i.e. BLAS summation order, not data.

What survives from that section is a narrower and still-real point: several
scripts load `_pooled.npy` rather than the preregistered `_final.npy`
(§3 below). That is a **pooling** question, not a staleness one, and the paper
already handles it with an explicit deviations table.

Verification command (run it yourself, it is the ground truth):

```bash
python -m src.analysis.verify_activations
```

Current output: all nine stages `ok`, `all present stages bound & consistent: True`.

---

## 1. Already computed and valid — no rerun needed

| Artifact | Coverage | Status |
|---|---|---|
| `results/activations/*` | **9 stages, 654 rows, both poolings** | verified bound to the frozen benchmark; row order identical across stages |
| `refusal_direction/projection_trajectory.json` | 9 stages, `_final` | reproduces byte-identically |
| `interpretability/factorial_direction_audit.json` | **now 9 stages**, `_final` | 4 prior stages unchanged; 5 added 2026-09-10 |
| `final_token_repair/**` (directions, folds, controls, bindings, summaries) | 6 stages | all 6 direction bindings verify: activation sha **and** direction sha |
| `summaries/confirmatory_endpoints.json` | pooled CF1/CF2 | valid; used as the pooling-sensitivity comparison |
| `final_token_repair/summaries/final_token_endpoints*.json` | final-token CF2 | the paper's primary intervention numbers |
| `summaries/mcnemar_direction_specificity.json` | 4 branches | repo version is newer than the bundle's |
| `probes_v2/` | 9 stages, 654-row, with bindings | imported 2026-09-10 (`probes/` is 370-era — do not use) |
| `behavioral_eval/v2_raw_M*.json` | 9 stages | imported — **CF1 is now reproducible from raw generations** |
| `raw/steering_v2_*.json` | 12 files | imported — steering no longer judge-file-only |
| `human_review/packet.json` | 160 blinded items | already built; only the annotation is outstanding |

## 2. Needs recomputing — infrastructure is ready, CPU only

| What | Why | Command |
|---|---|---|
| Direction-source robustness, 9 stages | currently 9 stages but worth re-confirming on the imported set | `python -m src.analysis.direction_source_robustness --pooling final` |
| CF3 decodability | re-pin on the imported activations | `python -m src.analysis.direction_decodability` |
| Cross-branch bootstrap / stability / bottleneck | these load `_pooled`; re-running confirms them on the imported set | `python -m src.interpretability.bootstrap_direction_stability`<br>`python -m src.interpretability.bottleneck_layer`<br>`python -m src.interpretability.bootstrap_cross_branch_difference`<br>`python -m src.interpretability.paired_deep_layer_stability_test` |

None of the four in the last row feed the current paper — they support README
Findings 2 and 3, which the paper does not use. Run them only if you want those
findings back.

## 3. Needs a small code change before it can be recomputed

| Script | Gap | Change needed |
|---|---|---|
| `src/analysis/subspace_geometry.py` | **hardcodes the M2→M3 pair**; has `--pooling` and `--layers` but no stage arguments | add `--pre`/`--post` (≈10 lines) to get the direct path (M1→M3_direct, M1_alt→M3_direct_alt) |
| `src/analysis/direction_decodability.py` | hardcodes M2/M3 | add stage arguments if CF3 is wanted on other pairs |
| `src/analysis/eval_refusal_direction.py::load_stage` | hardcodes `_pooled.npy`; four `src/interpretability/*` scripts route through it | add a `pooling` parameter and thread it through — the same edit already applied to `factorial_direction_audit.py`, `subspace_geometry.py` and `direction_source_robustness.py` in the 2026-09-08 session |

## 4. What the paper actually needs

The paper's geometry does **not** come from the `src/interpretability/*` scripts.
It comes from `paper_draft/make_evidence.py`, which reads the committed direction
arrays and the activations directly. That script is currently restricted to the
four stages that used to be fresh:

```python
FRESH = ["M2", "M3", "M2_alt", "M3_alt"]
```

With all nine now available this can cover the whole design, which turns two
stated limitations into results:

- **§4.3 becomes two-sided** — the contrast-norm growth and `z_C` shift can be
  computed for the direct path (M1→M3_direct, M1_alt→M3_direct_alt), not just the
  mediated one;
- **Figure 1b gains real bootstrap CIs** on every pairwise cosine, including the
  direct branches, replacing point estimates plus a fold-based noise floor.

That edit is to my own code and is not something you need to run blind — see the
handover note at the end.

## 5. New result already visible from the 9-stage factorial

Re-running the factorial audit on all nine stages surfaced something the paper
does not yet say. At layer 24:

| Stage | cos(d, d_H) | ‖d_AD‖ |
|---|---|---|
| M2 / M3 (mediated, Alpaca) | 0.770 / 0.842 | 36.2 / 41.5 |
| M2_alt / M3_alt (mediated, Dolly) | 0.768 / 0.822 | 36.5 / 42.5 |
| **M3_direct** (direct, Alpaca) | **0.935** | **56.2** |
| **M3_direct_alt** (direct, Dolly) | **0.941** | **58.6** |

The direct-DPO branches have a **larger and more harmfulness-aligned** A–D
contrast than the mediated ones — and their behaviour depends on it six to eleven
times *less*. This independently corroborates the `a_AD_rms` control already in
the paper (49.9 / 39.7 vs 36.3 / 32.2 at L24) and strengthens the argument: the
direct-path null is not "a weak or noisy direction", it is a bigger, cleaner
direction that is not behaviourally load-bearing.

---

## 6. Commands, in order

```bash
# 0. ground truth: is the activation set sound?  (expect: 9x ok, True)
python -m src.analysis.verify_activations

# 1. already run for you on 2026-09-10 -- listed so you can reproduce
python -m src.analysis.factorial_direction_audit --pooling final
python -m src.analysis.projection_trajectory

# 2. cheap re-confirmations on the imported set
python -m src.analysis.direction_source_robustness --pooling final
python -m src.analysis.subspace_geometry --pooling final        # M2->M3 only

# 3. only if you want README Findings 2/3 back (all _pooled; minutes each)
python -m src.interpretability.bootstrap_direction_stability
python -m src.interpretability.bottleneck_layer
python -m src.interpretability.bootstrap_cross_branch_difference
python -m src.interpretability.paired_deep_layer_stability_test

# 4. regression suite -- expect 1215 passed, 9 skipped (plus 227 crossbranch)
python -m pytest tests/ -q
```

No Colab is required for any of the above. Everything is CPU and torch-free
except the test suite.
