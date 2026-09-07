# Final-token repair — reproduction package

Repairs audit **RED-1**: the committed causal pipeline built the A–D
difference-in-means direction from `*_pooled.npy` (mean of the last 5 non-padding
tokens); `docs/audit/analysis_plan.md` §4 fixes the canonical direction on the
**final non-padding prompt token** (`*_final.npy`). Also repairs **RED-2**: CF3
(`direction_decodability_cf3.json`) was not reproducible from the repo.

Nothing here overwrites a pooled artifact. The committed
`results/refusal_direction/*_v2_direction.npy`, `results/raw/causal_ablation_v2_*`
(files with **no** `_finaltoken` tag), and `results/summaries/confirmatory_endpoints.json`
are untouched.

## What is in this directory

```
directions/    {stage}_final_token_L0-28.npy            explicit final-token A-D direction (29,1536)
               {stage}_final_token_xfit5_fold{0..4}.npy 5-fold cross-fit directions (same partition
                                                        as the committed pooled xfit5 run)
bindings/      {stage}_final_token_*_binding.json        pooling=final_token, pool_window=null, activation
                                                        source + sha, split seed 45, model hint, code commit
               {stage}_final_token_control.json          matched RMS random-ablation control (final
                                                        activations + final direction, self-consistent)
summaries/     final_token_directions.json               per-stage build report + cos(final, committed pooled)
               final_token_cf3.json                      CF3 on final-token M2/M3 directions  (COMPLETE, CPU)
               final_token_geometry.json                 factorial / trajectory / source-robustness on
                                                        BOTH poolings + adjacent-stage cosines
               pooled_vs_final_token_comparison.json      committed vs final_token vs mean_last5 table
               final_token_endpoints.json                CF2 / cross-fit / 2x2 on final-token   (PENDING GPU)
manifests/     consolidated_judge_final_token.json        final-token-only judge manifest        (PENDING GPU)
```

## Reproduction order

### A. CPU-only (done on the audit machine; re-runnable anywhere with the 654-row activations)

```bash
# 1. final-token directions + fold directions + control + CF3
python -m src.analysis.final_token_repair \
    --stages M2 M3 M2_alt M3_alt --recompute-cf3 \
    --out-dir results/final_token_repair/summaries

# 2. geometry / factorial / trajectory / source-robustness on both poolings + comparison
python -m src.analysis.final_token_summaries --stages M2 M3 M2_alt M3_alt

# 3. tests
python -m pytest tests/analysis/test_final_token_repair.py -q
```

`final_token_repair` and `final_token_summaries` are **CPU-only, torch-free**.
CF3 is a `LogisticRegression` probe — no GPU, no judge model.

**Requires** `results/activations/{M2,M3,M2_alt,M3_alt}_{final,pooled}.npy` +
`_metadata.json` (654-row). M3_direct / M3_direct_alt need their 654-row
`_final.npy` from the Drive `activations_bundle` (see
`private/final_token_repair/missing_artifacts.json`).

### B. One consolidated Colab GPU session

Run `notebooks/08_final_token_repair.ipynb` end to end (Runtime → Run all).
Pinned to commit — see the first cell. Produces the final-token held-out CF2,
5-fold cross-fit, optional full-A/D, and StrongREJECT + WildGuard scores, then
`final_token_endpoints.json` (`confirmatory_behavioral_endpoints.py
--condition-infix ft_`). Archives everything new to Drive with a SHA256 manifest.

### C. Post-GPU CPU (local, after downloading the archive)

```bash
python -m src.analysis.confirmatory_behavioral_endpoints \
    --judged <final-token judged file> --condition-infix ft_ \
    --out results/final_token_repair/summaries/final_token_endpoints.json
python -m src.analysis.final_token_summaries --stages M2 M3 M2_alt M3_alt
```

## Frozen conventions (unchanged)

- direction: `unit(mean(final[A_est]) − mean(final[D_est]))`, per layer, RAW (no
  centering), normalised after the mean difference; `direction_estimation` split
  only (seed 45; 120 A + 120 D).
- cross-fit: K=5, seed 20260907, deterministic partition **identical to the
  committed pooled xfit5** (a test asserts this); D centroid never folded;
  evaluated fold excluded from the direction AND the RMS-γ calibration.
- bootstrap: percentile, B=10,000, seed 20260904 (confirmatory); McNemar exact.
- layers 24–28; `hidden_states[l]` = output of decoder block `l`, hook on
  `blocks[l-1]`.

## Environment

Python 3.11.9; numpy 2.4.6, scipy 1.17.1, scikit-learn 1.9.0, pandas 3.0.5.
`requirements-lock.txt` pins numpy 2.4.4 / scipy 1.17.1 / pandas 3.0.2 /
scikit-learn 1.8.0 for byte-identical CPU reproduction; the small patch drift
above does not affect any qualitative result.

Gated models for job D: `allenai/wildguard` (and `google/gemma-2b` for the
StrongREJECT lineage). Set the Colab secret `HF_TOKEN`.
