"""CF3 - orthogonal benchmark-category decodability (WP-Decode).

Predeclared SECONDARY mechanistic endpoint (NOT confirmatory). Frozen
procedure, docs/audit/analysis_plan.md §4.4:

  * rows = A u C with a valid 4-way ``project_category`` label
    (misinformation_disinformation / harassment_bullying / illegal /
    cybercrime_intrusion). A/C category-set compat is checked FIRST; on
    failure the combined analysis is downgraded to within-source exploratory.
  * residualize each stage with THAT STAGE's own A-D direction at layer 28:
        h_tilde = h - (h . d^s) d^s      (never residualize M2 with M3's d)
  * features: raw residualized activations, NO standardization.
  * common group key: C -> pair_id ; A -> source_id (else a deterministic
    prompt-family hash). Same schema applied to both.
  * folds: StratifiedGroupKFold(n_splits=5, random_state=42); SAME fold
    assignment for M2 and M3; no group crosses train/test. A fold missing a
    category -> report + downgrade.
  * classifier: multinomial LogisticRegression(C=1.0, max_iter=2000,
    random_state=42). Hyperparameters fixed before results.
  * metric: macro-F1 (primary) + balanced accuracy.
  * CF3 = macroF1(M3) - macroF1(M2); CI by bootstrapping INDEPENDENT GROUPS.

Wording: *"DPO made the preregistered benchmark-category distinction more
linearly decodable after the A-D contrast direction was removed."* Never
*"DPO created a richer safety representation."*
"""
from __future__ import annotations

import hashlib

import numpy as np

CF3_LAYER = 28
CF3_CATEGORIES = (
    "misinformation_disinformation",
    "harassment_bullying",
    "illegal",
    "cybercrime_intrusion",
)
N_SPLITS = 5
RANDOM_STATE = 42
BOOTSTRAP_SEED = 20260904


class ACCompatError(RuntimeError):
    """A and C do not carry the same 4-way category label set."""


def _prompt_family_hash(prompt: str) -> str:
    return "fam_" + hashlib.sha256(prompt.strip().lower().encode()).hexdigest()[:12]


def group_key(record: dict) -> str:
    q = record.get("quadrant")
    if q == "C":
        return str(record.get("pair_id") or record.get("record_id"))
    # A rows
    sid = record.get("source_id")
    if sid:
        return str(sid)
    return _prompt_family_hash(record.get("prompt", record.get("record_id", "")))


def select_ac_rows(metadata):
    """Indices, labels, groups for A u C rows with a valid CF3 category."""
    idx, labels, groups, quad = [], [], [], []
    for i, row in enumerate(metadata):
        if row.get("quadrant") not in ("A", "C"):
            continue
        cat = row.get("project_category") or row.get("source_category")
        if cat not in CF3_CATEGORIES:
            continue
        idx.append(i)
        labels.append(cat)
        groups.append(group_key(row))
        quad.append(row["quadrant"])
    return (np.array(idx), np.array(labels), np.array(groups), np.array(quad))


def check_ac_category_compat(labels, quad):
    a_set = set(labels[quad == "A"].tolist())
    c_set = set(labels[quad == "C"].tolist())
    if a_set != c_set:
        raise ACCompatError(
            f"A category set {sorted(a_set)} != C category set {sorted(c_set)} - "
            "combined A u C CF3 analysis unavailable; downgrade to within-source."
        )
    return sorted(a_set)


def residualize(pooled_layer: np.ndarray, direction_layer: np.ndarray) -> np.ndarray:
    """h - (h . d) d, d assumed unit."""
    proj = pooled_layer @ direction_layer
    return pooled_layer - np.multiply.outer(proj, direction_layer)


def _make_folds(labels, groups):
    from sklearn.model_selection import StratifiedGroupKFold

    skf = StratifiedGroupKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    folds = list(skf.split(np.zeros(len(labels)), labels, groups))
    fold_missing_category = []
    for k, (tr, te) in enumerate(folds):
        if set(labels[te].tolist()) != set(labels.tolist()):
            fold_missing_category.append(k)
    return folds, fold_missing_category


def decodability(X, labels, folds):
    """Grouped-CV macro-F1 + balanced accuracy from PRECOMPUTED folds
    (shared between M2 and M3). Raw features, no standardization."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import balanced_accuracy_score, f1_score

    y_true_all, y_pred_all = [], []
    for tr, te in folds:
        clf = LogisticRegression(
            C=1.0, max_iter=2000, random_state=RANDOM_STATE,
        )
        clf.fit(X[tr], labels[tr])
        y_true_all.append(labels[te])
        y_pred_all.append(clf.predict(X[te]))
    y_true = np.concatenate(y_true_all)
    y_pred = np.concatenate(y_pred_all)
    return {
        "macro_f1": float(f1_score(y_true, y_pred, average="macro")),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "n": int(len(y_true)),
    }


def cf3(
    pooled_m2, pooled_m3, metadata, directions_by_stage, *, layer=CF3_LAYER,
    n_boot=10000, seed=BOOTSTRAP_SEED,
):
    """directions_by_stage: {"M2": (n_layers,hidden) unit A-D dir, "M3": ...}.
    Returns the frozen CF3 report (or a downgrade marker)."""
    idx, labels, groups, quad = select_ac_rows(metadata)
    report = {
        "endpoint": "CF3", "status": "confirmatory_secondary_NOT_confirmatory",
        "layer": layer, "n_rows": int(len(idx)),
        "n_A": int((quad == "A").sum()), "n_C": int((quad == "C").sum()),
    }
    try:
        report["categories"] = check_ac_category_compat(labels, quad)
    except ACCompatError as exc:
        report["status"] = "downgraded_within_source_exploratory"
        report["downgrade_reason"] = str(exc)
        return report

    folds, fold_missing = _make_folds(labels, groups)
    if fold_missing:
        report["status"] = "downgraded_fold_missing_category"
        report["folds_missing_a_category"] = fold_missing

    x_m2 = residualize(pooled_m2[idx, layer], directions_by_stage["M2"][layer])
    x_m3 = residualize(pooled_m3[idx, layer], directions_by_stage["M3"][layer])

    d_m2 = decodability(x_m2, labels, folds)
    d_m3 = decodability(x_m3, labels, folds)
    report["M2"] = d_m2
    report["M3"] = d_m3
    report["cf3_macroF1_M3_minus_M2"] = d_m3["macro_f1"] - d_m2["macro_f1"]
    report["bootstrap_group_diff"] = _bootstrap_group_diff(
        x_m2, x_m3, labels, groups, folds, n_boot=n_boot, seed=seed
    )
    return report


def _bootstrap_group_diff(x_m2, x_m3, labels, groups, folds, *, n_boot, seed):
    """Bootstrap the M3-M2 macro-F1 difference by resampling INDEPENDENT
    GROUPS (not rows). Predictions are made once per stage with the frozen
    folds; each replicate resamples groups and recomputes macro-F1 on the
    pooled out-of-fold predictions."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import f1_score

    def oof_predictions(X):
        pred = np.empty(len(labels), dtype=labels.dtype)
        for tr, te in folds:
            clf = LogisticRegression(C=1.0, max_iter=2000, random_state=RANDOM_STATE)
            clf.fit(X[tr], labels[tr])
            pred[te] = clf.predict(X[te])
        return pred

    pred_m2 = oof_predictions(x_m2)
    pred_m3 = oof_predictions(x_m3)

    uniq_groups = np.unique(groups)
    group_to_rows = {g: np.flatnonzero(groups == g) for g in uniq_groups}
    rng = np.random.default_rng(seed)
    diffs = []
    for _ in range(n_boot):
        sampled = rng.choice(uniq_groups, size=len(uniq_groups), replace=True)
        rows = np.concatenate([group_to_rows[g] for g in sampled])
        f_m2 = f1_score(labels[rows], pred_m2[rows], average="macro")
        f_m3 = f1_score(labels[rows], pred_m3[rows], average="macro")
        diffs.append(f_m3 - f_m2)
    diffs = np.asarray(diffs)
    return {
        "n_boot": int(n_boot), "seed": seed, "interval": "percentile",
        "unit": "independent groups (pair_id / source_id), not rows",
        "mean": float(diffs.mean()),
        "ci_low": float(np.percentile(diffs, 2.5)),
        "ci_high": float(np.percentile(diffs, 97.5)),
    }


def main():  # pragma: no cover - [exec:T4], needs regenerated 654-row activations
    import argparse
    import json
    from pathlib import Path

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--act-dir", default="results/activations")
    parser.add_argument("--direction-dir", default="results/refusal_direction")
    parser.add_argument("--out", default="results/interpretability/direction_decodability_cf3.json")
    args = parser.parse_args()

    act = Path(args.act_dir)
    ddir = Path(args.direction_dir)
    meta = json.loads((act / "M2_metadata.json").read_text(encoding="utf-8"))
    m2 = np.load(act / "M2_final.npy")
    m3 = np.load(act / "M3_final.npy")
    directions = {
        "M2": np.load(ddir / "M2_direction_final.npy"),
        "M3": np.load(ddir / "M3_direction_final.npy"),
    }
    report = cf3(m2, m3, meta, directions)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"CF3 status={report['status']} -> {args.out}")


if __name__ == "__main__":  # pragma: no cover
    main()
