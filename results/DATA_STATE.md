# results/ — data state and provenance

**Read this before auditing anything under `results/`.** It exists because a
previous audit concluded "only 4 of 9 checkpoints have fresh activations" and
built a paper limitation around it — when the other five were sitting unpacked
in a Google-Drive bundle. The local checkout was stale, not the experiment.

Last reconciled: **2026-09-10**, against
`Downloads/dpo_v2-20260908T091342Z-1-{001,002,003}.zip` (3.8 GB, produced
2026-09-08 at code commit `983caac`).

---

## 1. Activations — ALL NINE STAGES ARE FRESH

`results/activations/{stage}_{final,pooled}.npy` — **all 9 stages, 654 rows,
shape (654, 29, 1536)**, both poolings. Verified 2026-09-10:

- every `{stage}_metadata.json` has n=654, A=150 / B=250 / C=104 / D=150,
  240 `direction_estimation` + 60 `held_out_behavioral`;
- **row order is identical across all nine stages** (checked by `record_id`),
  which is what makes joint prompt-level resampling across stages valid;
- all six committed direction bindings in
  `final_token_repair/bindings/*_final_token_L0-28_binding.json` verify:
  `activation_source_sha256` matches the local `.npy`, and
  `direction_sha256` matches the committed direction array.

The `.npy` are gitignored (~2 GB). If a fresh clone lacks them, restore from the
bundle above — do **not** conclude the experiment wasn't run.

Until 2026-09-10 the local copies of M0 / M1 / M1_alt / M3_direct /
M3_direct_alt were the **370-row legacy** files (65.9 MB instead of 116.5 MB),
with the retired 20-item quadrant C and no `split` key. Any analysis output
dated **before 2026-09-07** that spans more than the four stages
M2 / M3 / M2_alt / M3_alt mixed the two benchmarks and is invalid — see §3.

## 2. Which judge file is authoritative

Whole-run judged files live in `results/behavioral_judges_v2/` (gitignored).
Thirteen exist across all bundles; **only these are current**:

| File | Covers | Used for |
|---|---|---|
| `behavioral_judges_v2_20260907T043919Z.json` (30,798 rec) | all 9 behavioural stages + pooled causal + steering | CF1, steering, behavioural trajectory |
| `final_token_repair/judges/behavioral_judges_v2_20260908T060458Z.json` (10,008 rec) | final-token causal, SR + WG | **CF2 and everything in the paper's intervention section** |
| `final_token_repair/judges/behavioral_judges_v2_20260908T001504Z.json` | final-token causal, SR only | the `_strongreject_only` summaries |

The eleven older runs (2026-09-03 → 2026-09-07T034708Z, ~503 MB) were
**deliberately not imported**. They are superseded; keeping them locally only
creates ambiguity about which file a number came from.

## 3. Known-stale outputs — recompute before use

These were computed **before** the 654-row activations existed for all nine
stages, so any cross-stage claim in them mixes the 370- and 654-row benchmarks:

```
results/refusal_direction/projection_trajectory.json
results/refusal_direction/quadrant_projections_v2.json
results/refusal_direction/cosine_similarity_v2.json
results/interpretability/bootstrap_direction_stability.json
results/interpretability/bottleneck_layer.json
results/interpretability/bootstrap_cross_branch_difference.json
results/interpretability/paired_deep_layer_stability_test.json
results/interpretability/subspace_geometry.json           (M2-vs-M3 only)
results/interpretability/factorial_direction_audit.json   (4 stages only)
```

They are all now **recomputable on CPU** from the fresh activations. Two further
issues, independent of staleness: `projection_trajectory.json` and
`quadrant_projections_v2.json` were built on `_pooled`, not the preregistered
`_final` (verified by exact numerical reproduction); and `results/probes/` is
370-era — use `results/probes_v2/` (9 stages, 654-row, with bindings).

## 4. What was imported on 2026-09-10

129 new files + 15 replaced, 1.25 GB. Replaced = the ten stale `.npy` and the
five stale `_metadata.json`.

**Taken:** fresh activations + metadata + bindings (9 stages);
`behavioral_eval/v2_raw_M*.json` (all 9, un-intervened generations — CF1 is now
reproducible from raw); `raw/steering_v2_*.json` (all 12); `probes_v2/`;
`human_review/packet.json`; `refusal_direction/` extras; new `summaries/`.

**Skipped:** the 11 superseded judge runs (§2) and 5,130 intermediate
generation shards under `*_shards/parts/` — resume checkpoints already
consolidated into the final per-condition JSONs, never the source of a number.
Both are now gitignored.

**Kept at the repo version, not the bundle's:** 14 tracked JSONs under
`final_token_repair/bindings/`, `interpretability/` and `summaries/`. The repo
is at commit `c784139` (after the 2026-09-08 audit fixes); the bundle predates
them. Differences were checked individually and are: a `code_commit` field,
float last-digit repr, key ordering, and one genuinely thinner file
(`mcnemar_direction_specificity.json` in the bundle predates its producer
script). **The repo version is newer in every case.**

## 5. Duplicate-download artifacts

21 files named `...(1).json` / `...(1).npy` were removed after verifying each
was byte-identical to its original. If Drive re-downloads create more, delete
them the same way — never assume a `(1)` copy is a distinct result.

## 6. Not part of the main experiment

`results/crossbranch/` is a **separate cross-branch activation-transfer study**
(a MATS project), deliberately outside the frozen preregistration in
`docs/audit/analysis_plan.md`. It shares the benchmark and the checkpoints but
answers a different question. Do not fold its numbers into the main endpoints.
