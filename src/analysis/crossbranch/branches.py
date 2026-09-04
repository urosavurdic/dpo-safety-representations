"""Branch map and the canonical condition table.

Single source of truth. The worker imports CONDITIONS directly; the runner
mirrors the P0 subset and a test pins the two together, so the two
vocabularies cannot drift apart silently.

Two rules this module exists to enforce:

1. Condition names are DIRECTION-NEUTRAL. "target" and "source" are roles,
   not branches. The concrete pair lives in row metadata (source_branch,
   target_branch, resolved_stage) and in output filenames, never in a
   condition name -- so the reciprocal direction (Dolly -> Alpaca) is a
   flag, not a rename of every artifact, test and analysis key.

2. Conditions declare a KIND and a STAGE GATE.
   - kind "model"  : a different checkpoint is loaded and NO vector is
     injected. These have no .npz artifact; a runner precondition that
     demands one for them is a bug.
   - kind "vector" : exactly one .npz artifact, injected via
     PerRowDeltaInjector.
   - stage_gate decides what the first implementation pass may execute.
     Anything not "p0" is refused by the worker unless explicitly allowed.
"""
from __future__ import annotations

from dataclasses import dataclass, field


# Branch -> (pre-DPO, post-DPO) repo stage names.
BRANCHES: dict[str, dict[str, str]] = {
    "A": {"pre": "M2", "post": "M3", "corpus": "Alpaca"},
    "B": {"pre": "M2_alt", "post": "M3_alt", "corpus": "Dolly-15k"},
}

# Frozen elsewhere (analysis_plan.md section 9) -- mirrored here with a
# pinned test rather than imported, so this module stays dependency-light.
INJECT_LAYER = 24
COEFFICIENTS = (0.5, 1.0, 2.0)
PRIMARY_COEFFICIENT = 1.0
POSITION = "final"
INJECT_MODE = "last_prompt_only"

# Gate values.
P0 = "p0"
STAGE2 = "stage2"
OPTIONAL = "optional"
APPENDIX = "appendix"
DEFERRED = "deferred"

MODEL = "model"
VECTOR = "vector"


@dataclass(frozen=True)
class Condition:
    name: str
    kind: str
    stage_gate: str
    # Which checkpoint to load: "target_pre" (B2) for every injected arm,
    # "target_post" (B3) only for the post-DPO reference.
    checkpoint: str
    # Artifact key into the delta store; None for model conditions.
    artifact: str | None = None
    # True when the condition varies with the coefficient sweep.
    takes_coefficient: bool = True
    note: str = ""


CONDITIONS: tuple[Condition, ...] = (
    # ---- Stage 1: the gate. The only conditions P0 may execute. --------
    Condition(
        "baseline_target", MODEL, P0, "target_pre",
        takes_coefficient=False,
        note="B2 as-is; the pre-DPO reference the gate measures movement from.",
    ),
    Condition(
        "reference_target", MODEL, P0, "target_post",
        takes_coefficient=False,
        note="B3 as-is; the post-DPO behavioural profile the gate measures movement toward.",
    ),
    Condition(
        "own_delta_target", VECTOR, P0, "target_pre",
        artifact="delta_target",
        note="B2 + coef * Delta_B. Within-branch sufficiency: does the branch's own "
             "measured delta reproduce its own post-DPO behaviour at this site?",
    ),
    Condition(
        "own_normmatched_random", VECTOR, P0, "target_pre",
        artifact="normmatched_random_target",
        note="B2 + coef * r, per-row ||r(x)|| = ||Delta_B(x)||. Magnitude control: "
             "separates 'the delta did it' from 'a perturbation this big did it'.",
    ),

    # ---- Stage 2: cross-branch transfer. Built later, gated off now. ---
    Condition(
        "xfer_delta_source_identity", VECTOR, STAGE2, "target_pre",
        artifact="delta_source",
        note="B2 + coef * Delta_A, identity transfer. The primary scientific arm.",
    ),
    Condition(
        "xfer_delta_source_shuf_wq", VECTOR, STAGE2, "target_pre",
        artifact="delta_source_shuf_wq",
        note="Within-quadrant shuffled Delta_A: destroys prompt<->delta correspondence "
             "while preserving the per-quadrant delta distribution.",
    ),
    Condition(
        "xfer_delta_source_normmatched", VECTOR, STAGE2, "target_pre",
        artifact="normmatched_random_source",
        note="Per-row random matched to ||Delta_A(x)||.",
    ),
    Condition(
        "xfer_delta_source_dosematched", VECTOR, STAGE2, "target_pre",
        artifact="delta_source_dosematched",
        note="Delta_A rescaled per row to ||Delta_B(x)||. NOTE: consumes the target "
             "branch's post-DPO activation norm, so it is a STRONGER oracle than the "
             "identity arm and must be disclosed as such.",
    ),
    Condition(
        "dir_source_matched", VECTOR, STAGE2, "target_pre",
        artifact="dir_source_const",
        note="Constant vector s_A * d_A, same injector/site/timing as the delta arms. "
             "s_A = median ||Delta_A|| over direction_estimation only.",
    ),
    Condition(
        "dir_target_matched", VECTOR, STAGE2, "target_pre",
        artifact="dir_target_const",
        note="Constant vector s_B * d_B; the within-branch concept reference, dosed to "
             "its OWN branch so it is comparable to own_delta_target.",
    ),

    # ---- Optional / appendix / deferred --------------------------------
    Condition(
        "xfer_delta_source_shuf_global", VECTOR, OPTIONAL, "target_pre",
        artifact="delta_source_shuf_global",
        note="Global shuffle; secondary diagnostic only.",
    ),
    Condition(
        "xfer_delta_source_procrustes", VECTOR, APPENDIX, "target_pre",
        artifact="delta_source_procrustes",
        note="Exploratory appendix. Never used to rescue a null identity result.",
    ),
)

# Deferred and deliberately absent from CONDITIONS: a direction-decomposition
# arm. Injecting only the perpendicular component reduces the injected norm
# (||Delta_perp|| < ||Delta||), so a null would conflate removing the d_A
# component with injecting less. A defensible version needs BOTH the parallel
# and perpendicular components, each dose-matched back to ||Delta_A(x)||, run
# as a pair. Not implemented until explicitly approved.
DEFERRED_CONDITIONS: tuple[str, ...] = ("xfer_delta_source_decomposition",)


BY_NAME: dict[str, Condition] = {c.name: c for c in CONDITIONS}

P0_CONDITIONS: tuple[str, ...] = tuple(
    c.name for c in CONDITIONS if c.stage_gate == P0
)


def get(name: str) -> Condition:
    try:
        return BY_NAME[name]
    except KeyError:
        if name in DEFERRED_CONDITIONS:
            raise KeyError(
                f"{name!r} is DEFERRED and deliberately unimplemented. See "
                "the note in branches.py; it needs a dose-matched "
                "parallel/perpendicular pair and explicit approval."
            ) from None
        raise KeyError(
            f"Unknown condition {name!r}. Known: {sorted(BY_NAME)}"
        ) from None


def resolve(source_branch: str, target_branch: str) -> dict[str, str]:
    """Map branch roles to concrete repo stage names.

    Raises when the branches are equal: a "cross-branch" run against itself
    is never what the caller meant, and silently producing a within-branch
    result under a cross-branch filename would be a data-integrity problem,
    not a convenience.
    """
    for role, branch in (("source", source_branch), ("target", target_branch)):
        if branch not in BRANCHES:
            raise ValueError(
                f"{role}_branch={branch!r} is not a known branch "
                f"{sorted(BRANCHES)}"
            )
    if source_branch == target_branch:
        raise ValueError(
            "source_branch and target_branch must differ; got "
            f"{source_branch!r} for both."
        )

    src, tgt = BRANCHES[source_branch], BRANCHES[target_branch]
    return {
        "source_branch": source_branch,
        "target_branch": target_branch,
        "source_pre": src["pre"],
        "source_post": src["post"],
        "target_pre": tgt["pre"],
        "target_post": tgt["post"],
        "source_corpus": src["corpus"],
        "target_corpus": tgt["corpus"],
    }


def stages_needed(source_branch: str, target_branch: str) -> list[str]:
    """The four checkpoints whose activations must be bound before any run."""
    r = resolve(source_branch, target_branch)
    return [r["source_pre"], r["source_post"], r["target_pre"], r["target_post"]]


def checkpoint_for(condition: str, source_branch: str, target_branch: str) -> str:
    r = resolve(source_branch, target_branch)
    key = get(condition).checkpoint
    if key not in r:
        raise ValueError(f"Condition {condition!r} names unknown checkpoint {key!r}")
    return r[key]


def direction_tag(source_branch: str, target_branch: str) -> str:
    """Filename/analysis tag for one direction, e.g. 'AtoB'."""
    return f"{source_branch}to{target_branch}"


def planned_units(
    conditions: list[str],
    coefficients: tuple[float, ...] = COEFFICIENTS,
) -> list[tuple[str, float | None]]:
    """(condition, coefficient) pairs. Model conditions appear once, with None.

    This is what makes "Stage 1 is 8 units" arithmetic rather than a claim:
    2 model conditions + 2 vector conditions x 3 coefficients.
    """
    units: list[tuple[str, float | None]] = []
    for name in conditions:
        cond = get(name)
        if cond.takes_coefficient:
            units.extend((name, float(c)) for c in coefficients)
        else:
            units.append((name, None))
    return units
