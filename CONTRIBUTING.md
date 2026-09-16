# Working on this repo

Conventions, commands, and the things that will bite you. For *why* the project
looks the way it does, see [docs/history/](docs/history/). For what the results
are, see [README.md](README.md) and [docs/FINDINGS.md](docs/FINDINGS.md).

## Layout

```
src/
  common/        shared helpers: io, stats, provenance, stages, quadrants,
                 activations, generation, refusal_classifier, normalize
  training/      SFT and DPO trainers, stage registry, adapter chains
  data_pipeline/ benchmark construction, quadrant-C curation
  analysis/      endpoints, geometry, judges, summaries, the bound pipeline
  analysis/crossbranch/   the cross-branch transfer study (see below)
  interpretability/       direction stability, bottleneck layers
  diagnostics/   leakage and composition checks
  reproduce.py   CPU-only orchestrator; start here
configs/         one YAML per training stage, plus dry-run siblings
data/            benchmark, processed corpora, curation artifacts
docs/            this guide's companions; docs/audit/ is frozen
notebooks/       Colab runners, GPU work
results/         every committed number, with *_binding.json provenance
tests/           mirrors src/
archive/         dead code, kept for provenance. Not maintained.
```

## Setup

```bash
python -m venv .venv && . .venv/Scripts/activate   # or bin/activate
pip install -r requirements.txt
pip install -e .
python -m pytest -q
```

For bit-level reproduction use `requirements-lock.txt` rather than the ranges in
`requirements.txt`.

## Commands

CPU-only, once activations exist:

```bash
python -m src.reproduce --list                   # status, runs nothing
python -m src.reproduce --components all
python -m src.analysis.summarize_cross_branch
python -m src.export_results
```

GPU work runs from `notebooks/`. The training notebook is config-driven: set
`STAGES_TO_RUN` and prerequisites resolve automatically.

## Conventions

**Tests before every push.** `python -m pytest -q`. Smoke tests check plumbing,
not scientific correctness — they are fast and CPU-only by design.

**Frozen things stay frozen.** `docs/audit/analysis_plan.md` and
`endpoint_table.md` are the preregistration. Changing an endpoint after the fact
needs a recorded deviation, not an edit.

**Never change a published number to make code tidier.** Category strings,
condition-name prefixes, bootstrap seeds and `N_BOOTSTRAP` values are baked into
committed results.

**Docstrings describe current behaviour.** One line of history at the end if it
is load-bearing, otherwise link to `docs/history/`. Not a paragraph, not at the
top. A docstring that opens by explaining a bug that was fixed is a docstring
the reader has to wade through before learning what the module does.

**The long docstrings that remain are deliberate.** A handful of modules carry
docstrings well over any sensible length ceiling — `cf_joint_geometry.py`,
`c_b_paired_delta_analysis.py`, `analyze_3d_h.py`, the R104 packet builders.
Those are not narration: they record a locked analysis contract, naming every
input path, population role, seed and sign convention, and stating explicitly
where the module made a decision the contract left open. Trimming them would
delete the record that makes those analyses auditable. `src/cue_scoring.py` is
additionally byte-pinned and cannot be edited at all.

**Paths named in prose are checked.** `tests/test_referenced_paths_exist.py`
asserts that every `src/`, `tests/`, `docs/` or `archive/` path mentioned in a
source docstring or comment actually exists, and
`tests/test_documented_commands_resolve.py` does the same for every
`python -m src.X`. Both exist because these references are invisible to the
import graph, so renames silently rot them — and several had, including into
published result JSON.

**Naming.** Read [docs/NAMING.md](docs/NAMING.md) before renaming anything. It
explains why half the repo says `v2` when there is no `v1`, and why `_final` and
`_pooled` are pooling modes rather than versions.

## Things that will bite you

**Four source files are byte-hash-pinned** and cannot be edited or moved — not
even a docstring. `src/v2_io.py`, `src/cue_scoring.py`,
`src/corpus_discrimination.py`, `src/diagnostics/score_lexical_risk_cues.py`.
The full list, including pinned data paths, is in `docs/NAMING.md`.

**Two live pipeline inputs hide in `logs/`** among 65 dated audit dumps:
`logs/direction_split_manifest.json` and `logs/benchmark_gate_config.json`. Do
not tidy `logs/` without accounting for them.

**`ACT_DIR` is monkeypatched by module in several tests.** Loaders take
`act_dir` explicitly for that reason. A loader that reads a default instead will
silently read the real 2 GB of activations and the test will pass for the wrong
reason.

**Commands written in prose are not checked by the import graph.** Every
`python -m src.X` in docs, notebooks and scripts is verified by
`tests/test_documented_commands_resolve.py`. Add commands there or they rot.

**Probe layer selection.** Pick by `quadrant_c_flagged_unsafe_frac`, never by
`cv_accuracy_mean` — that saturates near 1.0 everywhere including the untrained
base model, so ties break to the shallowest, least informative layer.

**The cross-branch study is outside the preregistration.**
`src/crossbranch/` and `results/crossbranch/` are a separate
activation-transfer study. Its numbers must not be folded into the confirmatory
endpoints.

## Design decisions worth not relitigating

**The chain.** M0 (Qwen2.5-1.5B base) → M1 (SFT-helpful) → M2 (SFT-safety) →
M3 (DPO). `M3_direct` applies DPO straight to M1, skipping M2. The `_alt` branch
mirrors the whole chain with only M1's source corpus changed — a single-variable
design.

**Matched data between M2 and M3.** Both draw from PKU-SafeRLHF on the same
prompts. This removes the training-data distribution as a confound between them.
It does not isolate "the DPO objective" from everything else that differs.

**The DPO reference policy is not reference-free.** `train_dpo.py` merges the
prior stage's adapter into the base weights, then hands the trainer that dense
model with a fresh LoRA config and `ref_model=None`. The trainer computes
reference log-probs by disabling the trainable adapter, which yields exactly the
merged prior-stage checkpoint. So the reference is the immediately preceding
checkpoint; `ref_model=None` is a memory-saving trick, not a reference-free
objective. Use this phrasing in any write-up.

**LoRA rank is a quantified confound, not an eliminated one.** r=64 throughout.
The subspace check bounds it: over 90% of the direction's norm lies outside the
rank-64 subspace. It does not remove it.

**Template matching.** M0 has no chat template; M1 onward do. M0's prompts are
wrapped in the same literal template tokens before extraction, so surface form
is not the difference being measured.

**The benchmark is fixed.** 654 rows, A=150 B=250 C=104 D=150, SHA-pinned. Do
not regenerate it without updating every downstream result.
