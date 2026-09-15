# Data: provenance, licensing, and content warnings

## Content warning

This repository contains, in committed result files, **real language-model
completions to harmful prompts**. Some are refusals; some are not. They are
committed deliberately, because the central claims are about how often and under
what interventions a model refuses, and those claims are not checkable without
the underlying generations.

Affected paths:

- `results/behavioral_eval/` — per-stage responses to all four quadrants
- `results/raw/causal_ablation_*` — responses under ablation conditions
- `results/raw/steering_*` — responses under steering conditions
- `results/crossbranch/judge_inputs/` — responses submitted to the judges
- `results/_legacy_370era/` — the same, from the pre-freeze evaluation set

If you are cloning this to reuse the code rather than audit the results, none of
those paths are needed.

## Where the evaluation prompts come from

The 654-row benchmark crosses intent against surface form:

| Quadrant | n | Intent | Surface form | Source |
|---|---:|---|---|---|
| A | 150 | harmful | overtly harmful | HarmBench |
| B | 250 | benign | harmful-sounding | XSTest |
| C | 104 | harmful | reduced-cue | StrongREJECT, reworded |
| D | 150 | benign | plain | Alpaca, Dolly, OASST1 |

Quadrant C is the only quadrant whose prompt text was authored rather than taken
verbatim. Each item starts from a published StrongREJECT prompt and is reworded
to remove wrongdoing-signalling vocabulary while preserving the underlying
request. Every item retains its `source_prompt`, so the rewording is auditable,
and the paired comparison against the unmodified source is the evidence that the
rewording did what it claims. Drafting was AI-assisted and human-reviewed; that
provenance is recorded per item rather than obscured, because the validity
argument depends on it.

## Training corpora

| Stage | Corpus | Upstream |
|---|---|---|
| M1 | Alpaca | `tatsu-lab/alpaca` |
| M1_alt | Databricks Dolly 15k | `databricks/databricks-dolly-15k` |
| M2, M3 | PKU-SafeRLHF | `PKU-Alignment/PKU-SafeRLHF` |

`data/processed/*.jsonl` are **derivatives** of those corpora, produced by the
build scripts in `src/data_pipeline/`. They are committed so a Colab run can
clone and train without re-deriving them, and because the exact rows used are
part of the experimental record.

Each upstream dataset carries its own license and terms. The MIT license in this
repository covers the code, not the redistributed data.

### Inherited personal data

The upstream instruction corpora contain third-party personal information —
email addresses, phone numbers, postal addresses — some real, some fabricated by
the original annotators. These flow through into `data/processed/` unchanged. We
did not introduce them and have not attempted to scrub them, since scrubbing
would change the exact training inputs and break reproducibility of the
checkpoints. If you need a PII-free derivative, rebuild from the upstream
datasets with your own filter using `src/data_pipeline/build_m1_data.py`.

## Leakage control

Quadrant D is drawn from the same instruction corpora used for training, so
overlap is a live risk rather than a theoretical one. It is controlled two ways:
a reserved slice held out of training before M1's data is built, and a
near-duplicate scan of the final evaluation set against every training file.

One near-duplicate is knowingly retained: *"Is Beyonce married?"* against the
training set's *"Who married Beyonce in 2008?"* (cosine 0.91). These ask
different questions — current marital status versus the identity of a specific
past spouse — and it is kept rather than dropped so the decision is visible.
Dedup reports are under `data/`.

## Models

Trained adapters live on HuggingFace under `urosavurdic/qwen2.5-1.5b-*`. No
model weights are committed here. The base model is `Qwen/Qwen2.5-1.5B`.
