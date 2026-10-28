# Safety SFT preserves, DPO redirects

**From the same instruction-tuned start and the same 4,000 PKU-SafeRLHF safety
pairs, supervised fine-tuning on the safer responses keeps a small model's
internal harmful–benign contrast almost unchanged. DPO on the full pairs
redirects it: from layer 20 of 28 on, most of the contrast points in a direction
the start's contrast did not.** Qwen2.5-1.5B, two instruction-tuned starts,
three training runs per arm.

The *harmful–benign contrast* is the difference between the model's mean
activation on harmful prompts and on benign prompts at one layer. Its length
says how strongly the two groups differ inside the model; its direction says in
what way. It describes a representation, not safety itself.

---

## Results

### 1. SFT keeps the contrast, DPO turns it

Each fine-tuned model's contrast is split into a part along the start's contrast
and a part orthogonal to it, both in units of the start contrast's length.
Layer 24, final prompt token, range over three training runs per arm:

| start | recipe | length ratio | cosine to start | orthogonal part |
|---|---|---|---|---|
| Alpaca | safety SFT | 0.98–0.99 | 0.97 | 0.23–0.24 |
| Alpaca | DPO | 1.38–1.54 | 0.53–0.59 | 1.16–1.30 |
| Dolly | safety SFT | 0.99–1.01 | 0.92 | 0.38–0.40 |
| Dolly | DPO | 1.50–1.60 | 0.38–0.41 | 1.39–1.46 |

- The two recipes' ranges of cosine and orthogonal part never overlap, at any of
  the four analysed layers (16, 20, 24, 28) and under both poolings (final token,
  mean of the last five tokens).
- From layer 20 on, every DPO run's orthogonal part is larger than its along
  part. In the original runs the direction measured before DPO carries only
  11–48% of the squared post-DPO contrast.
- DPO's change in the contrast is 4.5–5.8× the 95th percentile of a
  label-permutation null, against 2.7–3.6× for SFT: DPO moves harmful and benign
  prompts differently, not just further.
- SFT with the loss on the response only matches ordinary SFT on every measure
  (orthogonal part 0.20 at layer 24).

### 2. The larger update did less

| start | recipe | weight change ‖ΔW‖ | first-token KL | activation drift (L24) | CKA to start (L24) |
|---|---|---|---|---|---|
| Alpaca | safety SFT | 18.75 | 0.32 | 0.19 | 0.97 |
| Alpaca | DPO | 2.82 | 3.99 | 0.52 | 0.71 |
| Dolly | safety SFT | 18.94 | 0.73 | 0.28 | 0.94 |
| Dolly | DPO | 2.85 | 2.85 | 0.56 | 0.70 |

SFT's merged weight change is 6.6–6.7× DPO's, about what its four-times learning
rate and two epochs imply. In activations and outputs the order reverses.

**Matched dose.** Checkpoints saved during the reruns compare the recipes at
equal movement. In the Dolly branch, DPO after 20 of its 125 steps had moved the
activations (drift 0.26 against 0.27–0.28) and the first-token output
distribution (KL 0.62–0.64 against 0.67–0.73) about as far as the complete SFT
runs, and its orthogonal part was already more than twice SFT's (0.85–0.88
against 0.38–0.40). SFT set its orthogonal part by its first checkpoint and kept
it while its weight change grew more than fivefold.

### 3. Behaviour: SFT changes little, DPO overshoots

WildGuard rates (%) on the 654-prompt benchmark, original runs:

| start | model | refuses A | refuses B | refuses C | refuses D | harmful answer A | harmful answer C |
|---|---|---|---|---|---|---|---|
| Alpaca | start | 9 | 0 | 5 | 1 | 83 | 75 |
| | safety SFT | 15 | 4 | 10 | 2 | 73 | 70 |
| | DPO | 95 | 28 | 98 | 15 | 1 | 0 |
| Dolly | start | 7 | 1 | 0 | 1 | 81 | 86 |
| | safety SFT | 12 | 4 | 10 | 3 | 77 | 67 |
| | DPO | 99 | 42 | 100 | 19 | 1 | 0 |

DPO almost removes harmful answers but refuses many benign prompts that merely
sound alarming (B): 24–53% across the six DPO runs. The recipe difference (DPO
minus SFT) excludes zero in every quadrant, the benign ones included.

SFT's small effect concerns these data, not SFT in general: WildGuard reads 43%
of the SFT targets (the "safer" responses) as refusals and 27% as harmful, so
imitation saw a mixed signal.

**Human check.** One annotator, blind to model and label, judged 40
hash-sampled benign answers by XSTest's scheme. Of 30 that WildGuard read as DPO
refusals, 29 were full or partial refusals (95% CI 83–99%); of 10 it read as
compliant, 2 were refusals. WildGuard's refusal labels hold here but may miss
some.

### 4. Knowing is not refusing

The starts already separate the groups (held-out d′ 3.6–3.7; AUROC 0.90–0.91
for reworded harmful against alarming benign prompts), yet give harmful answers
to 75–86% of harmful prompts. A larger contrast is not better separation
either: d′ is 3.6–3.7 after SFT and 4.2–4.9 after DPO, not a reliable rise.

### 5. Does the redirected part act?

An exploratory test added a fixed vector to each start's layer-24 activation at
the final prompt token, on the reworded harmful prompts (C). A push along DPO's
orthogonal direction, as large as DPO's own average push on harmful prompts, cut
harmful answers by 14–17 points; the same push along the start direction cut
them by 2–4, a random one by 1.

Removing each direct-DPO model's own direction (layers 24–28, cross-fitted)
raised StrongREJECT harm scores by only 0.009–0.020 over a random control,
against 0.105–0.119 for DPO after safety SFT. So the direction can move
behaviour, but is not shown to carry DPO's: refusal may also live elsewhere, or
the models may sit far from the boundary.

### 6. What drives the contrast

- **No training overlap.** No benchmark prompt has an exact or near-duplicate
  (character n-gram TF-IDF cosine ≥ 0.9) in any training file.
- **Source.** With only OpenAssistant prompts as benign, a source neither start
  saw, the orthogonal part moves by at most 0.17.
- **Style.** For reworded harmful minus alarming benign prompts, whose surface
  cues point the other way, and for XSTest's unsafe minus safe minimal pairs
  (one source, one format), DPO's orthogonal part is 0.89–1.77 against
  0.31–0.65 for SFT. The recipe difference excludes zero at every layer after
  Bonferroni adjustment.

### 7. The full chain: DPO after safety SFT

In the four-stage chain (base → instruction tuning → safety SFT → DPO), DPO
after safety SFT changes the weights as much as direct DPO but moves the
contrast mostly along the old direction (cosine 0.85–0.89 at layer 24), so
redirection depends on where DPO starts. The chain was run on a preregistered
analysis plan ([docs/audit/analysis_plan.md](docs/audit/analysis_plan.md)). Its
confirmatory endpoints: DPO after safety SFT lowers StrongREJECT harm on the
reworded harmful prompts by 0.40 [0.34, 0.46], and ablating that model's
direction raises it by +0.11 [+0.03, +0.21] over a magnitude-matched random
control (held-out, n=30). Those and the branch comparisons are in
[docs/FINDINGS.md](docs/FINDINGS.md).

---

## How it was measured

| Checkpoint | What it is |
|---|---|
| M0 | Qwen2.5-1.5B base |
| M1 / M1_alt | the instruction-tuned starts: SFT on 6,000 Alpaca / Dolly prompts |
| M2 / M2_alt | safety SFT from the start on the chosen responses of 4,000 PKU-SafeRLHF pairs (2 epochs, lr 2e-4) |
| M3_direct / M3_direct_alt | DPO from the start on the same 4,000 pairs (1 epoch, lr 5e-5, β 0.1) |
| M3 / M3_alt | DPO after safety SFT |

LoRA r=64 throughout. The safety-SFT and direct-DPO arms were trained twice more
under new seeds, saving intermediate checkpoints for the dose curves, plus one
safety-SFT run with the loss on the response only.

**Contrast.** Mean activation on 120 overt harmful prompts (A) minus 120 plain
benign prompts (D), at the final prompt token (and the mean of the last five as a
second pooling), layers 16, 20, 24 and 28. Length ratio, cosine and the
along/orthogonal split are measured against the start's contrast; intervals come
from a joint prompt bootstrap, Bonferroni-adjusted over the four layers.

**Behaviour.** A fixed 654-prompt benchmark crossing intent against surface
form, labelled by WildGuard and StrongREJECT. WildGuard recovers the
benchmark's intent labels at κ 0.95.

| | overt wording | plain / reduced-cue wording |
|---|---|---|
| **harmful intent** | A (150, HarmBench) | C (104, StrongREJECT, reworded) |
| **benign intent** | B (250, XSTest) | D (150, Alpaca/Dolly/OASST1) |

## Limitations

- One model at one scale (1.5B), one safety dataset, three runs per arm, LoRA
  throughout.
- Objective and recipe differ together: DPO also sees the rejected response, and
  the two schedules differ. The comparison is matched in start and data, not in
  everything.
- The contrast is a difference between prompt groups, not a pure safety feature.
  The controls reduce but cannot exclude source and style effects, and the
  contrast mixes harmfulness and refusal signals.
- Behaviour labels come from automated judges, with 40 answers checked by hand.
  DPO's over-refusal varies from run to run.
- Quadrant C's prompts were authored: each is a reworded published prompt with
  its source kept, so the rewording is auditable, but it bundles cue removal with
  changes in length and register.
- The SFT-against-DPO comparison is exploratory, outside the preregistered plan.

## Reproduce

```bash
python -m src.stagecmp.run_geometry                  # along/orthogonal split, update size, separability
python -m src.stagecmp.contrast_controls --xstest-csv PATH
python -m src.seedrep.analyze --bank hf://USER/seedrep-bank   # reruns, dose curves, injection
```

Results land in [results/stagecmp/](results/stagecmp/), each with a sidecar
recording the sha256 of every input. The chain's command sequence is in
[docs/REPRODUCE.md](docs/REPRODUCE.md).
