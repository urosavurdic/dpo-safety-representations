# results/stagecmp/ — stage-vs-stage comparison

**EXPLORATORY.** Outside the frozen preregistration in
`docs/audit/analysis_plan.md`, in exactly the way `results/crossbranch/` is. Do
not fold these numbers into `results/summaries/confirmatory_endpoints.json` or
cite them as CF1/CF2/CF3.

Produced by `src/stagecmp/`. Rebuild with:

```bash
python -m src.stagecmp.cli all --pooling final_token
python -m src.stagecmp.cli all --pooling mean_last5
python -m src.stagecmp.report --pooling final_token
```

## Files

| file | what it holds |
|---|---|
| `stage_comparison_final_token.json` | the whole comparison under the **preregistered** final-prompt-token pooling |
| `stage_comparison_mean_last5.json` | the same under mean-last-5 pooling |
| `stage_comparison_*_binding.json` | sha256 of every input read — the nine activation files and the judge file |
| `FINDINGS_stagecmp_*.md` | generated prose summary; never hand-edited, regenerate instead |

## What is inside

- **`behavioural`** — four-way label distribution and withhold / refusal /
  degenerate rates with Wilson CIs, per stage and quadrant.
- **`pairwise`** — all 36 unordered stage pairs: four-way total variation with
  its distributions and per-prompt agreement, and exact-binomial McNemar.
  Bootstrap CIs on paired differences are computed for the **eight pre-named
  headline pairs only**; 36 pairs × 4 quadrants of intervals per metric would
  invite misreading.
- **`contrasts_2x2`** — the objective × corpus factorial (OBJ / CORP / INT) and
  the selectivity contrasts, with the paired bootstrap.
- **`depth_attribution`** — per-layer write of the A−D contrast onto each
  stage's direction, `write[l] = (h_l − h_{l−1}) · d_l`, and its cumulative
  curve.

## Three things to read before quoting anything

1. **The behavioural half is regex-only.** `diagnostics.scorer_coverage`
   records it: the frozen regex classifier covers all nine stages, while
   StrongREJECT and WildGuard currently cover quadrant-C rows of M2 and M3 and
   nothing else. A judging pass with `--scope all` would fill them in.

2. **The behavioural half is pooling-independent.** It reads response text, so
   `stage_comparison_final_token.json` and `stage_comparison_mean_last5.json`
   differ only in `depth_attribution`. That is expected, not a bug.

3. **Peak layer is not reported.** The per-layer write curve is multi-peaked
   and its argmax is unstable across stages — the same failure mode that retired
   the earlier bottleneck-gap claim (`docs/FINDINGS.md`). Read the cumulative
   curve and the A−D gap instead.

## Known pooling sensitivity

The fraction of the A−D contrast written by layer 24 is higher in the
direct-DPO branches than the mediated ones under `final_token`
(+0.104 Alpaca, +0.153 Dolly, both excluding zero) but the Alpaca sign
**reverses** under `mean_last5` (−0.034). "Direct DPO front-loads the contrast"
is therefore **not established** and is reported as a sensitivity, not a finding.
