"""Frozen intervention-condition vocabulary + runtime-capacity planner
(WP-Causal / WP-Steer), analysis_plan.md §6.

CPU-only, no torch. ``v2_pipeline`` imports these to (a) parse the new
``--conditions`` / ``--alpha-coefficients`` CLI, (b) decide - by *calibrated
wall-time*, never by results - which optional causal conditions fit a session
(§6.3), and (c) build the provenance blocks (seed / gamma / RMS / cos(r,d_AD)
for ablation; stage / layer / A_est rows / alpha_0 / coef / realised additive
norm / degeneration rate for steering) that go into each artifact's
``*_binding.json``.

The GPU generation of the ``ablated_random`` / ``ablated_AB`` / ``steered_random``
cells and the ``alpha_coef ∈ {0.5, 1.0, 2.0}`` dose-response sweep is wired into
``v2_pipeline.stage_causal`` / ``stage_steering`` (driven by ``--conditions`` /
``--alpha-coefficients``); this module supplies the frozen vocabulary, the
runtime-capacity planner, and the provenance blocks they record.
"""
from __future__ import annotations

from dataclasses import dataclass, field

# --- frozen vocabulary ---------------------------------------------------- #
# Causal (analysis_plan.md §2 CF2, §6.1, §6.3). "AD" = learned diff-in-means
# A-D direction; "random" = calibration-RMS-matched seeded random direction;
# "AB" = A-B direction (high-priority secondary, safety-specificity).
CAUSAL_REQUIRED = ("baseline", "ablated_AD", "ablated_random")
CAUSAL_SECONDARY = ("ablated_AB",)
CAUSAL_ALL = CAUSAL_REQUIRED + CAUSAL_SECONDARY

# Steering (analysis_plan.md §6.2). alpha_coef in {0.5, 1.0, 2.0}; learned vs
# seeded-random direction, additive norm controlled by alpha directly (do NOT
# reuse the ablation gamma).
STEERING_CONDITIONS = ("baseline", "steered_learned", "steered_random")
STEERING_ALPHA_COEFFICIENTS = (0.5, 1.0, 2.0)

# §6.3 execution priority (runtime-capacity, NOT result-dependent).
CAUSAL_PRIORITY = [
    ("baseline", 1), ("ablated_AD", 1), ("ablated_random", 1),   # required
    ("ablated_AB", 2),                                            # high-priority secondary
]


class ConditionError(ValueError):
    pass


def parse_conditions_arg(value) -> list[str]:
    """``--conditions baseline ablated_AD ablated_random`` -> validated list.
    ``None`` / empty -> the frozen required set."""
    if not value:
        return list(CAUSAL_REQUIRED)
    requested = [value] if isinstance(value, str) else list(value)
    unknown = [c for c in requested if c not in CAUSAL_ALL]
    if unknown:
        raise ConditionError(
            f"unknown causal condition(s) {unknown}; valid: {list(CAUSAL_ALL)}"
        )
    # always include the required set, preserve caller order for extras
    ordered = [c for c in CAUSAL_REQUIRED if c in requested or True]
    for c in requested:
        if c not in ordered:
            ordered.append(c)
    return ordered


def parse_alpha_coefficients_arg(value) -> list[float]:
    """``--alpha-coefficients 0.5 1.0 2.0`` -> validated floats.
    ``None`` -> the frozen {0.5, 1.0, 2.0}."""
    if not value:
        return list(STEERING_ALPHA_COEFFICIENTS)
    coeffs = [float(v) for v in ([value] if isinstance(value, (str, float, int)) else value)]
    if any(c <= 0 for c in coeffs):
        raise ConditionError(f"alpha coefficients must be > 0; got {coeffs}")
    return coeffs


# --- runtime-capacity planner (§6.3) ------------------------------------- #
@dataclass
class CausalPlan:
    stage: str
    scheduled: list[str]
    omitted: list[str]
    projected_minutes: float
    budget_minutes: float
    per_condition_minutes: float
    notes: list[str] = field(default_factory=list)

    def to_json(self) -> dict:
        d = {
            "stage": self.stage,
            "scheduled_conditions": self.scheduled,
            "omitted_conditions": self.omitted,
            "projected_minutes": round(self.projected_minutes, 1),
            "budget_minutes": self.budget_minutes,
            "per_condition_minutes": self.per_condition_minutes,
            "priority_rule": "analysis_plan.md §6.3 (runtime-capacity, not result-dependent)",
            "notes": self.notes,
        }
        if "ablated_AB" in self.omitted:
            d["ablated_AB_omission_effect"] = (
                "NO causal safety-specificity claim; do NOT say cosine geometry "
                "establishes causal equivalence."
            )
        return d


def plan_causal_conditions(
    stage: str,
    per_condition_minutes: float,
    budget_minutes: float,
    *,
    requested: list[str] | None = None,
) -> CausalPlan:
    """Schedule conditions by §6.3 priority: required first; ``ablated_AB``
    ONLY if the calibrated wall-time projection shows it fits after the
    required conditions. Never drops a required condition to fit a secondary
    one - if even the required set does not fit, that is reported (the caller
    then splits the stage across sessions, per blocker B4)."""
    requested = requested or list(CAUSAL_REQUIRED)
    want = set(requested) | set(CAUSAL_REQUIRED)  # required always in

    scheduled, omitted, notes = [], [], []
    minutes = 0.0
    for cond, tier in CAUSAL_PRIORITY:
        if cond not in want:
            continue
        if minutes + per_condition_minutes <= budget_minutes:
            scheduled.append(cond)
            minutes += per_condition_minutes
        else:
            omitted.append(cond)
            if tier == 1:
                notes.append(
                    f"REQUIRED condition {cond!r} does not fit the budget - split "
                    "this stage's causal run across sessions (ShardStore resume, "
                    "blocker B4) or stop that cell."
                )
            else:
                notes.append(f"{cond!r} omitted for wall-time; recorded in the manifest.")

    return CausalPlan(
        stage=stage, scheduled=scheduled, omitted=omitted,
        projected_minutes=minutes, budget_minutes=budget_minutes,
        per_condition_minutes=per_condition_minutes, notes=notes,
    )


def steering_cut_order():
    """If the steering session is tight, cut in this order (§6.3): M1/M2
    dose-response cells FIRST; never the random control, the M3/M3_alt
    dose-response, or the required A-D / random contrast."""
    return [
        "M1 dose-response cells",
        "M2 dose-response cells",
        "(stop here) - never cut: steered_random, M3/M3_alt dose-response, "
        "the learned-vs-random contrast",
    ]


# --- provenance blocks -------------------------------------------------- #
def ablation_provenance_block(stage_control_record: dict) -> dict:
    """Wrap ``control_directions.build_stage_control_record`` output for the
    causal binding sidecar."""
    return {
        "reference": "analysis_plan.md §6.1",
        "d_AB_vs_d_AD_cosine_per_layer": stage_control_record["d_AB_vs_d_AD_cosine_per_layer"],
        "d_AB_gate": stage_control_record["d_AB_gate"],
        "random_direction_seed": stage_control_record["random_direction_seed"],
        "ablation_control": stage_control_record["ablation_control"],
    }


def steering_provenance_block(
    stage: str, layer: int, n_a_est_rows: int, alpha_0: float,
    alpha_coefficient: float, realised_additive_norm: float | None,
    degeneration_rate: float | None, random_direction_seed: int | None,
) -> dict:
    return {
        "reference": "analysis_plan.md §6.2",
        "stage": stage,
        "layer": layer,
        "n_A_est_rows": n_a_est_rows,
        "alpha_0": alpha_0,
        "alpha_coefficient": alpha_coefficient,
        "alpha": alpha_0 * alpha_coefficient,
        "realised_additive_perturbation_norm": realised_additive_norm,
        "degeneration_rate": degeneration_rate,
        "random_direction_seed": random_direction_seed,
        "notes": (
            "alpha_0 = mean_{i in A_est}(h_i^{s,24} . d_AD^{s,24}); random "
            "steering reuses the SAME seeded r at the SAME coefficients and does "
            "NOT reuse the ablation gamma. Degeneration rate reported alongside "
            "every steering cell."
        ),
    }
