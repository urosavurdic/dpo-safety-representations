# Preserved but path-dependent

**Preference optimization does not build a model's safety representation. It
inherits one, leans on it harder, and how hard depends on the road taken to get
there.**

That is the claim this repository tests, and the evidence below is organised
around it.

---

## The claim, in three parts

**1. The mechanism is inherited, not created.** A single linear direction
separating harmful from benign prompts is already present after plain
instruction tuning. Across the chain its orientation barely moves — adjacent
cosine similarity is 0.65 for the first step, then 0.96 and 0.93. DPO arrives to
find the direction already there.

**2. What DPO adds is magnitude, not structure.** The harmful–benign contrast
grows substantially through preference optimization (norm ×1.15 at layer 24,
×1.49 at layer 28). But residualize that contrast away and test the remaining
space for linearly decodable harm *category*, and the result is null:
**+0.004 [−0.018, +0.027]**. Geometrically the update has a large component
orthogonal to the prior subspace; informationally, that component carries
nothing recoverable. Bigger, not richer.

**3. The mechanism's causal weight is path-dependent.** Ablating the direction
and comparing against a magnitude-matched random control, on cross-fitted
out-of-fold estimates (n=120 per branch):

| Branch | effect | 95% CI |
|---|---|---|
| M3 — safety-SFT then DPO, Alpaca | +0.154 | [+0.105, +0.203] |
| M3_alt — safety-SFT then DPO, Dolly | +0.044 | [+0.005, +0.085] |
| M3_direct — DPO straight from M1, Alpaca | +0.025 | [+0.013, +0.039] |
| M3_direct_alt — DPO straight from M1, Dolly | +0.012 | [+0.003, +0.023] |

Every CI excludes zero — the direction is load-bearing in all four — but the
magnitude spans roughly thirteen-fold. The corpus × training-history
interaction is **+0.097 [+0.040, +0.150]**: changing the instruction-tuning
corpus moves the effect by +0.013 within the direct-DPO pair but +0.110 within
the safety-SFT-mediated pair. *Post hoc, one seed per cell, not
multiplicity-corrected.*

## The consequence: ambiguous prompts drift

If preference optimization sharpens an existing axis rather than learning a new
distinction, prompts sitting between the poles should be dragged toward the
harmful end. They are. Measuring where reduced-cue harmful prompts sit between
benign (0) and overtly harmful (1) along each stage's own direction, layer 24:

| M0 | M1 | M2 | M3 | M3_direct |
|---|---|---|---|---|
| +0.33 | +0.72 | +0.65 | +0.90 | **+1.04** |

Two jumps — instruction tuning, then DPO — with a slight retreat during
safety-SFT. Direct DPO overshoots: those prompts end up *past* the overtly
harmful cluster. This is over-refusal pressure visible as geometry, rather than
inferred from behaviour.

## How it was measured

A four-stage chain on Qwen2.5-1.5B, with two controls that separate the two
factors in the claim:

| Stage | What it is |
|---|---|
| M0 | base model |
| M1 | SFT on a helpful corpus (Alpaca) |
| M2 | SFT on safety data (PKU-SafeRLHF) |
| M3 | DPO from M2, matched pairs from the same data |
| M3_direct | DPO straight from M1 — isolates *training history* |
| `*_alt` | the chain again with Dolly as M1's corpus — isolates *corpus* |

M2 and M3 draw on the same prompts, which removes the training-data
distribution as a confound between them. It does not isolate "the DPO
objective" from everything else that differs between an SFT and a preference run.

Evaluation is a fixed 654-prompt benchmark crossing intent against surface form:

| | overt wording | plain / reduced-cue wording |
|---|---|---|
| **harmful intent** | A (150, HarmBench) | C (104, StrongREJECT, reworded) |
| **benign intent** | B (250, XSTest) | D (150, Alpaca/Dolly/OASST1) |

Quadrant C is the load-bearing cell: genuinely harmful requests with the
wrongdoing vocabulary removed. Quadrant B is its mirror — benign requests that
merely sound alarming.

## Where the claim is weakest

- **The direction is not purely about safety.** Benign-but-alarming quadrant-B
  prompts sit about a third of the way toward the harmful pole at every stage.
  The contrast carries topic, register and wording structure too. Read "refusal
  direction" with that qualification.
- **The geometry is mixed, and is reported that way.** Most of the M2→M3 update
  is orthogonal to the prior top-5 subspace. On its own that would suggest new
  structure; the null in part 2 is what rules that reading out. Both results are
  stated, not only the convenient one.
- **The intervention direction is mean-pooled over the last five tokens**, not
  the final token the preregistration fixes. Cosine between the two is 0.76–0.87
  at the intervention layers. No number changes — every causal result is a valid
  analysis of the mean-pooled contrast — but the label was wrong and is
  corrected throughout. Carried as a recorded deviation.
- **One earlier claim was withdrawn.** A reported dataset-sensitive bottleneck
  gap did not survive bootstrapping and was mostly argmax noise. It is kept
  visible rather than quietly dropped.
- One model, one scale (1.5B). LoRA r=64 throughout — the subspace check bounds
  that confound (90–94% of the direction's norm lies outside the rank-64
  subspace) without removing it. Refusal labels come from a frozen regex
  classifier plus two LLM judges; still a proxy.
- Quadrant C's prompts were authored, not sampled. Each is a reworded published
  prompt with its source retained, so the rewording is auditable — but it bundles
  cue removal with incidental changes in length and register, so it is not a
  single clean factor.

Per-number provenance, including what each result does and does not support, is
in [docs/FINDINGS.md](docs/FINDINGS.md).

## A separate, exploratory study

`src/crossbranch/` and `results/crossbranch/` hold a distinct experiment: taking
the activation delta a DPO step induces in one branch and injecting it into
another, to ask whether the change is branch-specific or transferable.

**It sits outside the preregistration** in `docs/audit/analysis_plan.md` and its
numbers are exploratory. They are not folded into anything above.

## Reproducing

Trained adapters live on HuggingFace under `urosavurdic/qwen2.5-1.5b-*`; no
weights are committed here. Every committed result carries a `*_binding.json`
sidecar recording the benchmark hash it was computed against.

```bash
git clone https://github.com/urosavurdic/dpo-safety-representations
cd dpo-safety-representations
pip install -r requirements.txt && pip install -e .
python -m pytest -q

python -m src.reproduce --list          # what exists, what is missing
python -m src.reproduce --components all
```

GPU work — training, generation, interventions — runs from `notebooks/`.
Conventions and the traps that actually bite are in
[CONTRIBUTING.md](CONTRIBUTING.md). Naming, including why `654` and `370era`
mark the two benchmark eras, is in [docs/NAMING.md](docs/NAMING.md).

## Data and content warning

Committed result files contain **real model completions to harmful prompts**,
including non-refusals. They are committed because the claims are about refusal
rates under intervention, and those are not checkable without the generations.

Evaluation and training data derive from HarmBench, XSTest, StrongREJECT,
PKU-SafeRLHF, Alpaca and Dolly, each under its own terms. The processed training
files inherit third-party personal data from the upstream instruction corpora.
What is and is not scrubbed is documented in [docs/DATA.md](docs/DATA.md).

## Layout

```
src/common/        shared helpers        src/training/    SFT and DPO
src/data_pipeline/ benchmark building    src/analysis/    endpoints, geometry
src/pipeline/      benchmark-bound runs  src/crossbranch/ the exploratory study
src/diagnostics/   leakage checks        docs/audit/      frozen analysis plan
results/           every committed number + binding sidecars
archive/           retired code, kept for provenance
```

## How to cite

```bibtex
@misc{dpo_safety_representations,
  title  = {Preserved but Path-Dependent: the Refusal Direction Across a DPO Training Chain},
  author = {Uros Savurdic},
  year   = {2026},
  howpublished = {\url{https://github.com/urosavurdic/dpo-safety-representations}},
  note   = {Independent research project, not peer-reviewed}
}
```
