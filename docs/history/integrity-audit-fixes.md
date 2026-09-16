# Independent integrity audit and the fixes it drove

An independent auditor produced `private/audit/repo_integrity_audit_20260907.{md,json}`
+ per-issue evidence under `private/audit/evidence/`. **Verdict: overall
RED, rerun decision CPU_ONLY, GPU rerun: NO.** A previous session (and this
one, briefly) chased a full-A/D + cross-fit regeneration with
`--pooling final_token` — that CONTRADICTS the audit and is abandoned. The
three `notebooks/08_*` / `09_finaltoken_finish.ipynb` are deleted. **No
scientific number is invalidated; every committed causal result reproduces
byte-for-byte.**

**RED-1 — the causal direction is mean-pooled (last-5 tokens), not
final-token.** `v2_pipeline.stage_direction` / `stage_direction_crossfit`
build `{stage}_direction_654.npy` from `_pooled.npy` (`POOL_WINDOW=5`), not
the `_final` the frozen `analysis_plan.md` §4 fixes. `cos(pooled, final)
≈ 0.76–0.87` at L24–28. Fix = **paper relabel** ("final-token" →
"mean-pooled (last-5)" wherever it names the causal/intervention direction;
abstract, §design, §what, §limitations), a deviations-table row, and the
pooling-sensitivity limitation re-framed (the write-up USES pooled; final-token
is the alternative). `eval_refusal_direction.py`'s adjacent cosines
(Finding 1) are also `_pooled`; `factorial_direction_audit.py` /
`subspace_geometry.py` / `direction_source_robustness.py` load `_final`.

**RED-1 optional CPU recompute — DONE** (user wanted full internal
consistency). Added `--pooling {final,pooled}` to those three scripts
(`--pooling final` reproduces the committed JSON byte-for-byte). Re-ran on
`_pooled` → `results/interpretability/*_pooled.json`. Write-up §4/§9 now
report BOTH. Story holds under both poolings; on the mean-pooled (causal)
direction the geometry leans harder toward amplification (norm ×1.41 vs
×1.15 at L24; participation ratio and effective rank *contract* M2→M3
instead of staying flat/rising); d_H squared-norm share ~69% pooled vs
~82% final (and the pooled cross-term is +44, not −204).

**RED-2 — CF3 not reproducible** (`M2_direction_654.npy` never committed).
Re-pinned on CPU from the fresh 654-row `_final` activations, both poolings:
final-token cf3 = **+0.004 [−0.018, +0.027]**, mean-pooled +0.008
[−0.013, +0.030] — **null under both**. Write-up §8 + abstract updated (was
−0.016, "slightly negative", off the broken artifact).
`results/interpretability/direction_decodability_cf3.json` rewritten with
provenance + both variants.

**P14 — McNemar had no producer.** New `src/analysis/mcnemar_direction_specificity.py`
regenerates every cell from the committed raw causal files, in-repo regex
classifier, two-sided exact binomial. All b/c match the write-up. M3 quad-C p
= **3.7e-9** (was an unattributable 8.9e-7); still `<1e-6`, no paper
change. `*_binding.json` pins raw sha256 + classifier commit.

**9 YELLOW wording fixes applied to the write-up:** P1 (adjacent cosine
"(L24)" → layer-mean + L24 values 0.53/0.95/0.85), P2 (self-inclusion "at
most +0.005" → point +0.005, CI [+0.001,+0.009]), P3 ("significant
interaction" ×2 → "CI excludes zero; post hoc, not multiplicity-corrected"),
P8 (z_B "every stage incl. base" → "every trained checkpoint; M0 is
−0.27"), P9 (blanket B=10,000 → 10k confirmatory/trajectory, 1k secondary),
P10 ("isolates DPO the method from the data" → "removes the training-data
distribution as a confound; does not isolate the objective"), P11 (abstract
"targeted amplification" → amplification + orthogonal-update framing), P12
(LoRA in-subspace → L21/L28 `down_proj`), P13 (fold cos "0.998–0.9998" →
"0.998–0.999"). Plus a CI-conditionality sentence in §limitations.

**Commits:** `622ed0c` (RED-1/RED-2/YELLOW), `e499862` (optional pooled
recompute). Write-up is `private/` (gitignored), delivered by file. 4
`\pending` markers remain, all author tasks (Arditi citation, §10
annotation, acknowledgements).
