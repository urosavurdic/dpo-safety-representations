"""Control directions for the causal / steering interventions (WP-Ctrl).

CPU-only. Produces, per intervention stage:

  * ``d_AD`` - the learned diff-in-means A-D direction (RAW ``_final``, no
    centering), estimated on the ``direction_estimation`` split only;
  * ``d_AB`` - the A-B diff-in-means direction, and ``cos(d_AB, d_AD)`` per
    layer (descriptive, NO result-dependent gate - see analysis_plan.md §4);
  * ``r`` - a fixed seeded random unit direction per (stage, layer);
  * ``gamma`` - the calibration-RMS ratio that makes the random ablation match
    ``d_AD``'s RMS projected-perturbation magnitude on the calibration rows
    (analysis_plan.md §6.1):
        a_AD  = RMS_{i in calib} |h_i . d_AD|
        a_rnd = RMS_{i in calib} |h_i . r|
        gamma = a_AD / a_rnd
        h' = h - gamma * (h . r) * r

The GPU intervention code (``v2_pipeline.stage_causal``) consumes the recorded
``r`` and ``gamma`` verbatim; this module never touches a model. Everything it
records - seed, ``r``, calibration row ids/count, per-layer ``gamma``, both RMS
values, realised ``cos(r, d_AD)``, any zero-magnitude failure - goes into the
ablation provenance block.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

import numpy as np

DIRECTION_SPLIT = "direction_estimation"
ABLATION_LAYERS = tuple(range(24, 29))
CONTROL_SEED = 20260904


# --------------------------------------------------------------------------- #
# direction estimation
# --------------------------------------------------------------------------- #
def _split_mask(splits: np.ndarray, wanted: str) -> np.ndarray:
    return np.array([str(s) == wanted for s in splits])


def group_mean(
    pooled: np.ndarray,
    quadrants: np.ndarray,
    quadrant: str,
    *,
    splits: np.ndarray | None = None,
    restrict_split: str | None = None,
) -> np.ndarray:
    """Mean activation (n_layers, hidden) over rows in ``quadrant``. If
    ``restrict_split`` is given, only rows with that split are used (A/D
    directions are estimated on ``direction_estimation`` only)."""
    mask = quadrants == quadrant
    if restrict_split is not None:
        if splits is None:
            raise ValueError("restrict_split given but splits is None")
        mask = mask & _split_mask(splits, restrict_split)
    if not mask.any():
        raise ValueError(f"no rows for quadrant={quadrant!r} split={restrict_split!r}")
    return pooled[mask].mean(axis=0)


def unit_rows(vec: np.ndarray) -> np.ndarray:
    """Unit-normalise the last axis; a zero row stays zero (flagged elsewhere)."""
    norms = np.linalg.norm(vec, axis=-1, keepdims=True)
    return np.divide(vec, np.where(norms == 0.0, 1.0, norms))


def diff_in_means_direction(
    pooled: np.ndarray,
    quadrants: np.ndarray,
    pos: str,
    neg: str,
    *,
    splits: np.ndarray | None = None,
    restrict_split: str | None = DIRECTION_SPLIT,
) -> np.ndarray:
    """RAW diff-in-means, unit-normalised per layer. No centering."""
    m_pos = group_mean(pooled, quadrants, pos, splits=splits, restrict_split=restrict_split)
    m_neg = group_mean(pooled, quadrants, neg, splits=splits, restrict_split=restrict_split)
    return unit_rows(m_pos - m_neg)


def ad_direction(pooled, quadrants, *, splits=None):
    return diff_in_means_direction(pooled, quadrants, "A", "D", splits=splits)


def ab_direction(pooled, quadrants, *, splits=None):
    """d_AB uses ALL of B (B carries no split) and the estimation split of A."""
    m_a = group_mean(pooled, quadrants, "A", splits=splits, restrict_split=DIRECTION_SPLIT)
    m_b = group_mean(pooled, quadrants, "B", restrict_split=None)
    return unit_rows(m_a - m_b)


def cosine_per_layer(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.sum(unit_rows(a) * unit_rows(b), axis=-1)


# --------------------------------------------------------------------------- #
# seeded random direction
# --------------------------------------------------------------------------- #
def seeded_random_directions(
    n_layers: int, hidden: int, *, seed: int = CONTROL_SEED
) -> np.ndarray:
    """A fixed unit random direction per layer, ``(n_layers, hidden)``.
    Reproducible from ``seed`` alone."""
    rng = np.random.default_rng(seed)
    raw = rng.standard_normal((n_layers, hidden))
    return unit_rows(raw)


# --------------------------------------------------------------------------- #
# calibration-RMS gamma (analysis_plan.md §6.1)
# --------------------------------------------------------------------------- #
def rms_projected_norm(pooled_layer: np.ndarray, direction_layer: np.ndarray) -> float:
    """RMS over rows of ||(h.d) d||. With ``d`` a unit vector this is
    RMS_i |h_i . d| = sqrt(mean_i (h_i . d)^2)."""
    proj = pooled_layer @ direction_layer
    return float(np.sqrt(np.mean(proj ** 2)))


class ZeroMagnitudeError(RuntimeError):
    """Raised when a random direction has ~zero projected magnitude on the
    calibration rows, so gamma would be undefined (0/0 or x/0)."""


@dataclass
class AblationControl:
    seed: int
    layers: tuple[int, ...]
    calibration_split: str
    n_calibration_rows: int
    calibration_record_ids: list[str]
    gamma: dict[int, float]
    a_ad: dict[int, float]
    a_rand: dict[int, float]
    realised_cos_r_dAD: dict[int, float]
    zero_magnitude_layers: list[int] = field(default_factory=list)

    def to_json(self) -> dict:
        return {
            "control": "calibration_rms_matched_random_ablation",
            "reference": "analysis_plan.md §6.1",
            "seed": self.seed,
            "layers": list(self.layers),
            "calibration_split": self.calibration_split,
            "n_calibration_rows": self.n_calibration_rows,
            "calibration_record_ids": self.calibration_record_ids,
            "per_layer": {
                str(l): {
                    "gamma": self.gamma[l],
                    "a_AD_rms": self.a_ad[l],
                    "a_rand_rms": self.a_rand[l],
                    "realised_cos_r_dAD": self.realised_cos_r_dAD[l],
                }
                for l in self.layers
            },
            "zero_magnitude_layers": self.zero_magnitude_layers,
            "note": (
                "gamma matches RMS perturbation magnitude on the calibration "
                "rows, NOT the per-example intervention norm; realised norms "
                "during the eval run are recorded separately as diagnostics."
            ),
        }


def build_ablation_control(
    pooled: np.ndarray,
    quadrants: np.ndarray,
    splits: np.ndarray,
    d_ad: np.ndarray,
    r: np.ndarray,
    *,
    record_ids: Sequence[str] | None = None,
    layers: Sequence[int] = ABLATION_LAYERS,
    seed: int = CONTROL_SEED,
    calibration_split: str = DIRECTION_SPLIT,
    strict_zero: bool = True,
) -> AblationControl:
    """Compute per-layer gamma on the calibration rows (A/D
    ``direction_estimation`` split). Raises ``ZeroMagnitudeError`` on a layer
    whose random projected magnitude is ~0 unless ``strict_zero=False`` (then
    the layer is recorded in ``zero_magnitude_layers`` with gamma=nan)."""
    calib_mask = (
        ((quadrants == "A") | (quadrants == "D")) & _split_mask(splits, calibration_split)
    )
    if not calib_mask.any():
        raise ValueError("no calibration (A/D direction_estimation) rows found")
    calib = pooled[calib_mask]
    ids = (
        [str(record_ids[i]) for i in np.flatnonzero(calib_mask)]
        if record_ids is not None else []
    )

    gamma, a_ad, a_rand, cos_map, zero_layers = {}, {}, {}, {}, []
    for l in layers:
        aad = rms_projected_norm(calib[:, l], d_ad[l])
        arn = rms_projected_norm(calib[:, l], r[l])
        a_ad[l], a_rand[l] = aad, arn
        cos_map[l] = float(np.dot(unit_rows(d_ad[l]), unit_rows(r[l])))
        if arn <= 1e-12:
            zero_layers.append(int(l))
            if strict_zero:
                raise ZeroMagnitudeError(
                    f"layer {l}: random direction has ~zero projected magnitude "
                    f"on the calibration rows (a_rand={arn:.3e}); gamma undefined."
                )
            gamma[l] = float("nan")
        else:
            gamma[l] = aad / arn

    return AblationControl(
        seed=seed,
        layers=tuple(int(l) for l in layers),
        calibration_split=calibration_split,
        n_calibration_rows=int(calib_mask.sum()),
        calibration_record_ids=ids,
        gamma=gamma,
        a_ad=a_ad,
        a_rand=a_rand,
        realised_cos_r_dAD=cos_map,
        zero_magnitude_layers=zero_layers,
    )


def apply_random_ablation(h: np.ndarray, r_layer: np.ndarray, gamma_layer: float) -> np.ndarray:
    """h' = h - gamma * (h . r) r. Vectorised over leading axes of ``h``."""
    proj = h @ r_layer
    return h - gamma_layer * np.multiply.outer(proj, r_layer)


# --------------------------------------------------------------------------- #
# convenience: full record for one stage
# --------------------------------------------------------------------------- #
def build_stage_control_record(
    pooled: np.ndarray,
    quadrants: np.ndarray,
    splits: np.ndarray,
    *,
    record_ids: Sequence[str] | None = None,
    layers: Sequence[int] = ABLATION_LAYERS,
    seed: int = CONTROL_SEED,
) -> dict:
    d_ad = ad_direction(pooled, quadrants, splits=splits)
    d_ab = ab_direction(pooled, quadrants, splits=splits)
    n_layers, hidden = d_ad.shape
    r = seeded_random_directions(n_layers, hidden, seed=seed)
    usable_layers = [int(l) for l in layers if 0 <= int(l) < n_layers]
    if not usable_layers:
        raise ValueError(
            f"none of layers={list(layers)} are in range for {n_layers}-layer activations"
        )
    control = build_ablation_control(
        pooled, quadrants, splits, d_ad, r,
        record_ids=record_ids, layers=usable_layers, seed=seed,
    )
    return {
        "d_AB_vs_d_AD_cosine_per_layer": cosine_per_layer(d_ab, d_ad).tolist(),
        "d_AB_gate": "NONE - descriptive only (analysis_plan.md §4, correction #8)",
        "random_direction_seed": seed,
        "ablation_control": control.to_json(),
    }


def main() -> None:  # pragma: no cover - thin CLI, exercised at T4
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--activations-dir", default="results/activations")
    parser.add_argument("--stages", nargs="+",
                        default=["M3", "M3_direct", "M3_alt", "M3_direct_alt"])
    parser.add_argument("--out", default="results/refusal_direction/control_directions.json")
    parser.add_argument("--seed", type=int, default=CONTROL_SEED)
    args = parser.parse_args()

    act = Path(args.activations_dir)
    out: dict = {}
    for stage in args.stages:
        pooled = np.load(act / f"{stage}_final.npy")
        meta = json.loads((act / f"{stage}_metadata.json").read_text(encoding="utf-8"))
        quadrants = np.array([r["quadrant"] for r in meta])
        splits = np.array([r.get("split") or "" for r in meta])
        ids = [r.get("record_id") for r in meta]
        out[stage] = build_stage_control_record(
            pooled, quadrants, splits, record_ids=ids, seed=args.seed
        )

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"wrote {out_path} for stages {args.stages}")


if __name__ == "__main__":  # pragma: no cover
    main()
