# Where does safety live?

Tracing a refusal direction from a base model through SFT to DPO, and asking
whether preference optimization builds richer internal structure for safety or
mostly leans harder on structure that was already there.

**Short answer: mostly the latter — but the size of the effect depends on how
the model got there.**

---

## The question

Direct Preference Optimization makes models refuse harmful requests more
reliably. That much is easy to measure. The harder question is what changes
*inside*:

- **Hypothesis A** — DPO builds genuinely richer representations of safety:
  more linearly separable, carrying structure that was not previously there.
- **Hypothesis B** — DPO mainly sharpens sensitivity along a refusal direction
  that already exists, which would also pull ambiguous prompts toward the
  harmful cluster and produce over-refusal.

The evidence here favours B, with one real complication: the causal effect of
that direction is present in every training path tested, but its magnitude
varies roughly thirteen-fold depending on the path.

## Setup

A four-stage chain on Qwen2.5-1.5B, plus controls:

| Stage | What it is |
|---|---|
| M0 | base model, no fine-tuning |
| M1 | SFT on a helpful corpus (Alpaca) |
| M2 | SFT on safety data (PKU-SafeRLHF) |
| M3 | DPO, initialised from M2, on matched pairs from the same data |
| M3_direct | DPO applied straight to M1, skipping safety-SFT |
| `*_alt` | the whole chain again, with M1 trained on Dolly instead |

The `_alt` branch changes exactly one thing — M1's source corpus — and keeps
every downstream stage on identical data. M2 and M3 draw from the same
PKU-SafeRLHF prompts, which removes the training-data distribution as a
confound between them. It does not isolate "the DPO objective" from everything
else that differs between an SFT and a preference-optimization run.

Evaluation is a fixed 654-prompt benchmark crossing intent against surface form:

| | overt wording | plain / reduced-cue wording |
|---|---|---|
| **harmful intent** | A (150, HarmBench) | C (104, StrongREJECT, reworded) |
| **benign intent** | B (250, XSTest) | D (150, Alpaca/Dolly/OASST1) |

Quadrant C is the interesting cell: genuinely harmful requests with the
wrongdoing vocabulary stripped out. Quadrant B is its mirror — benign requests
that *sound* alarming.

## What the interventions show

**The direction is causally load-bearing, in every branch.** Ablating the A–D
contrast direction and comparing against a magnitude-matched random direction,
on cross-fitted out-of-fold estimates (n=120 per branch):

| Branch | effect | 95% CI |
|---|---|---|
| M3 | +0.154 | [+0.105, +0.203] |
| M3_alt | +0.044 | [+0.005, +0.085] |
| M3_direct | +0.025 | [+0.013, +0.039] |
| M3_direct_alt | +0.012 | [+0.003, +0.023] |

All four CIs exclude zero. An earlier reading of this experiment — based on
n=30 held-out prompts — appeared to show the effect present in some branches
and absent in others. That was a significance-pattern artifact of low power,
not a real qualitative difference, and it has been retracted. The preregistered
anchor (M3, held-out n=30, +0.114 [+0.028, +0.206]) is unchanged.

**The magnitude is path-dependent.** On paired bootstrap over the cross-fitted
per-prompt effects, the corpus × training-history interaction is +0.097
[+0.040, +0.150]. Changing the instruction-tuning corpus moves the effect by
+0.013 within the direct-DPO pair but +0.110 within the safety-SFT-mediated
pair. Within the Dolly branch, mediated-versus-direct spans zero — so the
mediation effect is carried almost entirely by the Alpaca branch. *Post hoc,
single seed per cell; the CI excludes zero but is not multiplicity-corrected.*

**Ambiguous prompts move toward the harmful cluster.** Measuring where
quadrant C sits between benign D (0) and overt-harmful A (1) along each stage's
own direction, at layer 24:

| M0 | M1 | M2 | M3 | M3_direct |
|---|---|---|---|---|
| +0.33 | +0.72 | +0.65 | +0.90 | +1.04 |

C climbs in two jumps — instruction-tuning, then DPO — with a slight retreat
during safety-SFT. Direct-DPO overshoots: C ends up *past* A. This is
Hypothesis B in one row of numbers.

## The honest null, and the caveats that matter

**DPO adds no decodable structure orthogonal to the direction.** Residualizing
out the A–D contrast and testing whether harm *category* remains linearly
decodable: +0.004 [−0.018, +0.027] using the preregistered final-token
direction, +0.008 [−0.013, +0.030] using the mean-pooled one. Null under both.
Whatever DPO adds geometrically, it is not recoverable richer category
structure.

**The geometry is genuinely mixed, and is reported that way.** The contrast
norm grows M2→M3 (×1.15 at layer 24, ×1.49 at layer 28), which looks like
amplification. But most of the update to the contrast is *orthogonal* to M2's
top-5 A/D subspace, which does not. Both are true; the null above is what stops
the orthogonal component from being read as "richer safety representation".

**The direction is not purely about safety.** Benign-but-alarming quadrant-B
prompts sit about a third of the way toward the harmful side along the same
direction, at every stage. The A–D contrast carries topic, register and wording
structure too. Any claim that this is "the refusal direction" should be read
with that in mind.

**The intervention direction is mean-pooled, not final-token.** The
preregistration fixes the direction on the final prompt token; the pipeline
actually built it from the mean of the last five tokens. Cosine between the two
is 0.76–0.87 at the intervention layers. No number changes — every causal result
is a valid analysis of the mean-pooled contrast — but the label was wrong and is
corrected throughout. Recorded as a deviation.

**One earlier claim was walked back.** A reported "7-layer, dataset-sensitive
bottleneck gap" did not survive bootstrapping and was mostly argmax noise. It is
kept visible rather than quietly dropped.

## Limitations

- One model, one scale (1.5B). Nothing here establishes that the picture holds
  at larger scale.
- LoRA r=64 throughout. The subspace check bounds the confound — 90–94% of the
  direction's norm lies outside the rank-64 subspace — but does not remove it.
- Refusal labels come from a regex classifier plus two LLM judges. The regex
  lexicon was tuned against observed output and is frozen, but it is still a
  proxy.
- Quadrant C's prompt text was authored, not sampled. Each item is a reworded
  published prompt with its source retained, so the rewording is auditable — but
  the rewording bundles cue-removal with incidental changes to length and
  register. It is not a single orthogonal factor.
- The cross-fitted contrasts are post hoc, one seed per cell.

## A separate study: cross-branch transfer

`src/crossbranch/` and `results/crossbranch/` hold a distinct
experiment — taking the activation delta a DPO step induces in one branch and
injecting it into another, to ask whether the change is branch-specific or
transferable.

**It sits outside the preregistration** in `docs/audit/analysis_plan.md`, and
its numbers are exploratory. They must not be folded into the confirmatory
endpoints above.

## Reproducing

Trained adapters are on HuggingFace under `urosavurdic/qwen2.5-1.5b-*`; no
weights are committed here. Committed results carry `*_binding.json` sidecars
recording the benchmark hash each number was computed against.

```bash
git clone https://github.com/urosavurdic/dpo-safety-representations
cd dpo-safety-representations
pip install -r requirements.txt && pip install -e .
python -m pytest -q

python -m src.reproduce --list          # what exists, what is missing
python -m src.reproduce --components all
```

GPU work (training, generation, interventions) runs from `notebooks/`.
Layout, conventions and the traps are in [CONTRIBUTING.md](CONTRIBUTING.md).
Naming — including why `v2` does not imply a `v1` — is in
[docs/NAMING.md](docs/NAMING.md). The full evidence ledger, with every number
and its source file, is in [docs/FINDINGS.md](docs/FINDINGS.md).

## Data and content warning

Committed result files contain **real model completions to harmful prompts**,
including non-refusals. They are committed because the claims are about refusal
rates under intervention, and those are not checkable without the generations.

Evaluation and training data derive from HarmBench, XSTest, StrongREJECT,
PKU-SafeRLHF, Alpaca and Dolly, each under its own terms. The processed training
files inherit third-party personal data from the upstream instruction corpora.
Details, and what is and is not scrubbed, are in [docs/DATA.md](docs/DATA.md).

## Repo map

```
src/common/           shared helpers        src/training/     SFT and DPO
src/data_pipeline/    benchmark building    src/analysis/     endpoints, geometry
src/analysis/ direction analysis    src/diagnostics/  leakage checks
docs/                 FINDINGS, DATA, NAMING, history/, audit/ (frozen)
results/              every committed number + binding sidecars
archive/              dead code, kept for provenance
```

## How to cite

```bibtex
@misc{dpo_safety_representations,
  title  = {Where Does Safety Live? Tracing a Refusal Direction from Base Model to DPO},
  author = {Uros Savurdic},
  year   = {2026},
  howpublished = {\url{https://github.com/urosavurdic/dpo-safety-representations}},
  note   = {Independent research project, not peer-reviewed}
}
```
