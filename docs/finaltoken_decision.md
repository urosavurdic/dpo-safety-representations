# Final-token vs mean-pooled direction — the decision

**Short version.** The methodologically preferred, preregistered, field-standard
choice is the **final prompt token**. The committed causal pipeline ablates the
**mean of the last 5 tokens** instead (audit RED-1). The two directions have
cosine ≈ 0.76–0.87 at the ablation layers — close, not identical. Everything
*descriptive* is already final-token or pooling-invariant; only the
*ablation-based* endpoints (CF2, cross-fit, 2×2, circularity, quad-C McNemar)
were computed on the pooled direction. `notebooks/10_finaltoken_causal.ipynb`
regenerates those on final-token.

## Why final-token is better here

- **Field convention.** The refusal-direction / activation-steering literature
  (Arditi et al. 2024 and everything downstream) extracts the direction at the
  last prompt-token position — the residual state the model produces its first
  response token from, i.e. the actual comply-vs-refuse decision point.
- **The frozen plan chose it.** `docs/audit/analysis_plan.md` §4: *"All on
  `_final` activations. Canonical direction (RAW `_final`, no centering)."*
- **Cleaner interpretation.** Mean-pooling the last 5 tokens mixes the decision
  state with 4 preceding token states. Lower variance (averaging), but you are
  now ablating a direction defined partly by positions that are not the
  decision point.

Mean-pooled is not *wrong* — it is a valid direction and the causal analysis
of it is internally valid (the audit verified every number reproduces
byte-for-byte). It is just not the direction the plan specified or the one a
reviewer expects.

## What is already final-token (no rerun needed)

| result | script | pooling |
|---|---|---|
| z_C trajectory (Fig 1, Finding 1 headline) | `projection_trajectory.py` | **final-token** (`_final.npy`, line 156) — committed M3 z_C@L24 = 0.897 reproduces |
| factorial audit (§4) | `factorial_direction_audit.py` | **final-token** (recompute identical); `--pooling pooled` also run → `*_pooled.json` |
| subspace geometry (§9) | `subspace_geometry.py` | **final-token** (recompute identical); pooled also run |
| source-robustness (§4) | `direction_source_robustness.py` | **final-token**; pooled also run |
| CF1 (behavioral shift, §5) | `confirmatory_behavioral_endpoints.py` | **pooling-invariant** — CF1 is `mean_i(SR^M3_i − SR^M2_i)` on quadrant C, no direction, no ablation. −0.4008 stands regardless. |
| CF3 (§8) | re-pinned this session | reported on **both**; null under both |

## What is mean-pooled and matters (the notebook fills this)

- **CF2 held-out** (the preregistered confirmatory anchor)
- **cross-fitted n=120**, **2×2 interaction**, **branch contrasts**, **circularity bias**
- **quadrant-C McNemar** (regex, but on the ablation outputs)

## What is mean-pooled and is minor (disclosed, not worth a rerun)

- Finding 1's **adjacent-stage cosines** — `eval_refusal_direction.py` uses
  `_pooled`. Committed 0.930 (M2→M3, layer-mean). Final-token = **0.947**
  (M2→M3) / **0.928** (M2_alt→M3_alt), computed on CPU from `_final.npy` for the
  4 fresh stages. Same qualitative claim (largest jump is M0→M1). The paper
  already labels these as "layer-mean" and flags them as pooled-derived.
- Base-model **Cohen's d = 4.19** (`bottleneck_layer.py`, `_pooled`). Not
  recomputed on final-token; it is a single descriptive number and the "the
  contrast forms early" claim does not hinge on its exact value.

## The run: `notebooks/10_finaltoken_causal.ipynb`

`SCOPE = "m3_anchor"` — **M3 held-out CF2 only, ~1 h.** The one question that
matters: does the preregistered confirmatory anchor replicate under the
preregistered pooling? If final-token CF2 primary has a CI that excludes zero
and sits within ~1 CI-width of the pooled +0.114, **RED-1 flips from a
disclosed weakness to "the result is robust to the pooling choice"** and the
paper's causal section moves to final-token primary (mean-pooled becomes the
sensitivity).

`SCOPE = "full"` — all 4 branches, held-out + 5-fold cross-fit, both judges
(StrongREJECT + WildGuard). ~4–5 h. Generation (§4) and judging (§6) are
separate resumable sections, so it splits across two Colab sessions if a
single session will not hold. After this the whole causal section is
final-token.

**Recommended:** run `m3_anchor` first. If it replicates cleanly, decide
whether `full` is worth the extra session or whether "the anchor replicates;
the exploratory cross-fit/2×2 are reported on the mean-pooled direction with
the anchor confirming the pooling is not driving them" is enough.
