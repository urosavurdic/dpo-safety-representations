# Selection-leakage scan — linear probes (WP-Probe)

**Question.** Does any headline probe number depend on a choice made *after*
looking at the held-out quadrants (C, D) the probe is evaluated on?

## What the probe does

`eval_probes.py` / `v2_pipeline.compute_probes` fit a logistic-regression probe
per layer on **quadrant A vs 50 quadrant-B rows** (5-fold stratified CV), then
report the fraction of each held-out set the fitted probe flags "unsafe":
held-out B (the other ~200 B rows), quadrant C (104), quadrant D (150).

## Leakage points found and closed

| # | Leakage point | Status |
|---|---|---|
| 1 | `pick_most_informative_layer()` selected the layer that **maximises quadrant-C flagging**, then that layer's C/D numbers were reported as the headline. Selecting the layer by the very quantity being reported, on the held-out quadrant, is circular. | **Closed.** Headline path now uses a fixed preregistered layer (`FINAL_LAYER = 28`). `pick_most_informative_layer` is retained but gated behind `--exploratory-layer-scan` and every use is labelled "EXPLORATORY / data-dependent selection, not a headline result". |
| 2 | `summarize_cross_branch.py` imported `pick_most_informative_layer` and used its output for the cross-branch probe comparison. | **Closed.** Now `probe_final_layer_for_stage()` → fixed layer 28. |
| 3 | `summarize_probe_findings.py` denominators were hard-coded 370-era values (`C=20, D=50, B_holdout=200`). Not leakage, but a stale-n bug that mis-sizes every Wilson CI. | **Closed.** `n` now derived from `LATEST_BENCHMARK.json` (`C=104, D=150, B_holdout = |B| − 50`). |
| 4 | Could a C or D row enter probe *training* (not just evaluation)? | **No, and now asserted.** `v2_pipeline.compute_probes` raises `RuntimeError` if the training index set intersects the C∪D index set; the probe binding sidecar records `no_cd_selection_asserted: true` and `layer_selection: "none; ... neither C nor D is used for layer selection or probe training"`. |

## Residual, disclosed (not leakage, but stated)

- The **B train/holdout split** uses `np.random.RandomState(42)` — fixed seed,
  no dependence on any outcome.
- CV accuracy saturates near 1.0 at almost every layer for every stage,
  including untrained M0; it is **not** used to pick a layer (that was an
  earlier, separately-fixed bug — see `CLAUDE.md` "Bugs already found and
  fixed"). It is reported descriptively only.
- The probe measures linear separability of an A-vs-B *surface* contrast, not
  "semantic safety understanding" — see `docs/audit/analysis_plan.md` §3.

## Verdict

No headline probe number is selected on the held-out quadrants after the
change. The only selection on the headline path is the preregistered
`FINAL_LAYER = 28`.
