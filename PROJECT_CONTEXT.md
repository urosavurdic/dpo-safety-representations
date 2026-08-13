# Project Context: Safety Representations Under Post-Training
 
Living document. Update at the end of each work session — research question and
decisions rarely change; status and experiment log should grow every session.
 
---
 
## Research Question
 
**Short:** Does DPO give a language model genuinely richer internal representations
of safety, or does it mainly amplify an existing low-dimensional refusal mechanism,
causing over-refusal on ambiguous prompts?
 
**Long:** Across a Base → SFT-Helpful → SFT-Safety → DPO training chain, do internal
representations of safe/unsafe/ambiguous prompts become more linearly separable and
semantically richer (Hypothesis A), or does the model mainly strengthen sensitivity
along a pre-existing refusal direction, pulling ambiguous prompts toward the unsafe
cluster (Hypothesis B)?
 
---
 
## Dissemination & impact

Target audiences and impact: the work is scoped to be maximally useful and
reproducible for the ML safety and interpretability research communities and
for practitioner teams integrating safety checks into deployed models. The
artifacts (activations, raw results JSON, summaries, and scripts) are organized
so reviewers and downstream researchers can validate and re-run the core
analyses with minimal friction.

Low-cost followups and framing additions (do not block finalization):
1. Add a brief information-theoretic framing paragraph in the write-up linking
   refusal-direction structure to compressibility/representational rank (MDL-lite
   intuition). This is writing-only and reuses extracted activations.
2. Document optional probe variants (e.g., MDL-style probe scoring) as future
   appendices rather than running them now — keep Phase 5 focused on clarity.

---
 
## Core Design Decisions (locked in — do not relitigate without reason)
 
1. **Four-model chain:** M0 (Qwen2.5-1.5B-Base) → M1 (SFT-Helpful, Alpaca) →
   M2 (SFT-Safety) → M3 (DPO). Preserve unless a compelling reason emerges.
2. **Matched-data ablation between M2 and M3 (critical fix):** Build DPO
   chosen/rejected pairs from PKU-SafeRLHF first. M2's SFT-safety data =
   the *chosen* responses on the *same prompts*, formatted as plain SFT
   examples. This isolates "DPO objective" from "DPO data content."
3. **LoRA rank confound (critical, flagged, not fully solved in Phase 1):**
   LoRA constrains updates to a low-rank subspace by construction, which can
   mechanically bias findings toward "amplification looks low-dimensional."
   Mitigation for Phase 1: use r=64 (not default r=8) for the M2→M3 step.
   Full fix (full-fine-tune robustness check) deferred to Phase 2. State this
   limitation explicitly in the write-up regardless.
4. **Template-matching protocol:** M0 has no chat template; M1–M3 do. Wrap M0's
   prompts in the same literal template tokens used for M1–M3 before extracting
   activations, so template surface form isn't a confound. Do not compare
   raw-prompt M0 activations to chat-templated M3 activations directly.
5. **Controlled 4-quadrant eval set:** harmful-intent×harmful-wording,
   benign-intent×harmful-wording, harmful-intent×neutral-wording,
   benign-intent×neutral-wording. Pull quadrant B heavily from XSTest.
6. **Datasets:** Alpaca subsample (M1, ~5–8k), PKU-SafeRLHF matched pairs
   (M2/M3, ~3–5k), HarmBench subset (held-out unsafe eval), XSTest full
   450 (over-refusal / ambiguous eval), OR-Bench optional supplement.
7. **Stack:** Transformers + PEFT (QLoRA) + TRL (SFTTrainer, DPOTrainer with
   `ref_model=None` for PEFT reference-free DPO) + Accelerate. Activation
   extraction via raw forward hooks (not TransformerLens, for compatibility
   robustness on Colab). scikit-learn for probes, scipy/statsmodels for
   effect sizes and CIs.
8. **Phase 1 causal check:** minimal refusal-direction ablation on M3 to get
   causal (not just correlational) evidence for H4 — cheap, no training needed.
9. **Checkpointing/tracking:** Git for code, Google Drive for large training
   artifacts, Hugging Face Hub for versioned finished adapters, W&B for metrics.
10. **Smoke-test convention (`tests/`):** fast, CPU-only, tiny-data tests that check
    plumbing ("does this run and produce the right shape"), not scientific
    correctness. Run `pytest tests/` locally before every Colab push and after
    any change to `src/`. Deliberately minimal — do not let this grow into a
    full test suite.
---
 
## Roadmap (phases — see chat for full phase-by-phase detail)
 
- **Phase 0** — Repo scaffolding, environment, dependencies, tracking setup, data folders,
  `tests/` bootstrap (prove pytest + imports work before any real code exists)
- **Phase 1** — Data acquisition/matching, dedup/leakage check, controlled eval set,
  template protocol validated, eval harness established on M0
- **Phase 2** — Train M1 (SFT-Helpful), train M2 (SFT-Safety, matched data), behavioral eval M0–M2
- **Phase 3** — Train M3 (DPO, `ref_model=None`, LoRA r=64), full behavioral eval (H1)
- **Phase 4** — Activation extraction, linear probes (H2/H3), refusal direction analysis
  (H4), causal ablation check, statistical rigor (CIs, effect sizes), optional MDL probing
- **Phase 5** — Write-up, limitations section, repo cleanup, portfolio polish
---
 
## Current Status

*(update this section every session)*

- [x] Roadmap agreed
- [x] Phase 0 complete
- [x] Phase 1 complete
- [x] Phase 2 complete
- [x] Phase 3 complete
- [x] Phase 4 complete
- [ ] Phase 5 complete
Last updated: 2026-08-12
 
---
 
## Experiment Log
 
| Date | Phase | Change | Result | Notes |
|------|-------|--------|--------|-------|
| 2026-08-05 | Phase 1 | Built matched DPO/SFT-safety pairs from PKU-SafeRLHF (`src/data_prep.py`) | 73,907 raw rows → 4,000 matched pairs after safety-disagreement filter | Comfortably above target; no scarcity concern. Smoke tests (5) passing. Fixed bare-`pytest` import issue with `pyproject.toml` pythonpath config. |
| 2026-08-05 | Phase 1 | Built controlled 4-quadrant eval set (`src/build_eval_set.py`) | A=50 (HarmBench), B=250 (XSTest safe), C=20 (hand-curated), D=50 (Alpaca reserved) = 370 total | Reserved Alpaca prompts saved to `alpaca_reserved_for_eval.json` for M1 exclusion. |
| 2026-08-05 | Phase 1 | Dedup check vs. M1 Alpaca data found near-duplicates (photosynthesis, haiku prompts — same task, reworded, not template noise); iterated to a persistent exclusion list after finding `build_m1_data.py` had no memory across runs | **Phase 1 complete.** Converged: reran with no new report → 0 new merges, 0 near-duplicates, stable at 5 total exclusions ever found | Final counts: dpo_pairs=4000, sft_helpful=6000, controlled_eval=370 (A=50,B=250,C=20,D=50). Both training sources verified clean vs. eval set. |
 
---

 
