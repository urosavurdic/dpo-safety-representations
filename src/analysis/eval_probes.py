"""
Component 3: linear probes (H2/H3).

Revised design: originally trained on quadrant A (unsafe) vs quadrant D
(safe) - found M0 (zero safety training) hit perfect 100% CV accuracy,
implausible as genuine safety understanding. Real cause: A and D differ in
topic/genre as well as safety (HarmBench's direct harm requests vs
Alpaca's generic writing/translation tasks) - a probe on this contrast
can't distinguish "learned safety" from "learned topic classification."

Fix: train on quadrant A vs a subset of quadrant B (XSTest - designed to
be topically/lexically adjacent to unsafe content while being safe). This
holds surface style roughly constant, isolating safety specifically. The
untrained remainder of B is a genuine held-out over-triggering check.
"""
import json
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, StratifiedKFold

from src.analysis.eval_refusal_direction import activations_available

STAGES = [
    "M0", "M1", "M2", "M3", "M3_direct",
    "M1_alt", "M2_alt", "M3_alt", "M3_direct_alt",
]
ACT_DIR = Path("results/activations")
B_TRAIN_SIZE = 50
SEED = 42


def load_stage_activations(stage, kind="final"):
    arr = np.load(ACT_DIR / f"{stage}_{kind}.npy")
    with open(ACT_DIR / f"{stage}_metadata.json", encoding="utf-8") as f:
        meta = json.load(f)
    return arr, meta


def split_by_quadrant(arr, meta):
    idx = {"A": [], "B": [], "C": [], "D": []}
    for i, row in enumerate(meta):
        idx[row["quadrant"]].append(i)
    return {q: arr[indices] for q, indices in idx.items()}


def split_b_train_holdout(b_array, train_size=B_TRAIN_SIZE, seed=SEED):
    rng = np.random.RandomState(seed)
    indices = rng.permutation(len(b_array))
    return b_array[indices[:train_size]], b_array[indices[train_size:]]


def probe_layer(train_unsafe, train_safe, holdout_b, test_c, test_d, layer_idx, cv_folds=5, seed=SEED):
    X_train = np.concatenate([train_unsafe[:, layer_idx], train_safe[:, layer_idx]], axis=0)
    y_train = np.concatenate([np.ones(len(train_unsafe)), np.zeros(len(train_safe))])

    clf = LogisticRegression(max_iter=2000, random_state=seed)
    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=seed)
    fold_scores = cross_val_score(clf, X_train, y_train, cv=cv)
    clf.fit(X_train, y_train)

    def flagged_frac(test_set):
        return float(clf.predict(test_set[:, layer_idx]).mean()) if len(test_set) else None

    return {
        "layer": layer_idx,
        "cv_accuracy_mean": float(fold_scores.mean()),
        "cv_accuracy_std": float(fold_scores.std()),
        "cv_fold_scores": [float(s) for s in fold_scores],
        "holdout_b_flagged_unsafe_frac": flagged_frac(holdout_b),
        "quadrant_c_flagged_unsafe_frac": flagged_frac(test_c),
        "quadrant_d_flagged_unsafe_frac": flagged_frac(test_d),
    }


def run_for_stage(stage, kind="final"):
    arr, meta = load_stage_activations(stage, kind)
    by_quadrant = split_by_quadrant(arr, meta)
    b_train, b_holdout = split_b_train_holdout(by_quadrant["B"])
    return [probe_layer(by_quadrant["A"], b_train, b_holdout, by_quadrant["C"], by_quadrant["D"], i)
            for i in range(arr.shape[1])]


def pick_most_informative_layer(layer_results):
    """NOT max cv_accuracy_mean - that metric saturates near 1.0 at almost
    every layer for every stage, including untrained M0 (see HANDOFF.md:
    "Naive CV accuracy retired (saturates even at untrained M0)"). Using it
    to pick a "best" layer just returns whichever layer happens to be first
    among the tied maximum - layer 0 or 1, the shallowest, LEAST
    informative layer, silently making every stage's "best layer" row look
    identical and its quadrant-C/D flagging rates look near-zero regardless
    of real per-layer variation (visible instead in summarize_probe_
    findings.py's FINAL_LAYER-based report). Pick by the metric that's
    actually meaningful instead: quadrant C flagging rate.
    """
    return max(layer_results, key=lambda r: r["quadrant_c_flagged_unsafe_frac"])


def main():
    out_dir = Path("results/probes")
    out_dir.mkdir(parents=True, exist_ok=True)

    all_results = {}
    for stage in STAGES:
        result_path = out_dir / f"{stage}_probe_results.json"
        if result_path.exists():
            # Adding a new stage to STAGES must not re-run/overwrite prior
            # stages' already-finalized probe results (results/probes/*.json
            # - see HANDOFF.md "Status: Phase 4 complete"). Reload instead.
            print(f"\n=== {stage}: already computed, loading from disk ===")
            with open(result_path, encoding="utf-8") as f:
                layer_results = json.load(f)
            all_results[stage] = layer_results
            best = pick_most_informative_layer(layer_results)
            print(f"  Best layer {best['layer']}: CV acc {best['cv_accuracy_mean']:.3f} ± {best['cv_accuracy_std']:.3f}")
            continue
        if not activations_available(stage):
            # Alt branch trains/pushes independently across sessions - not
            # every stage in STAGES is necessarily ready yet.
            print(f"\n=== {stage}: SKIPPED, activations not yet extracted ===")
            continue
        print(f"\n=== {stage} ===")
        layer_results = run_for_stage(stage)
        all_results[stage] = layer_results
        with open(result_path, "w", encoding="utf-8") as f:
            json.dump(layer_results, f, ensure_ascii=False, indent=2)
        best = pick_most_informative_layer(layer_results)
        print(f"  Best layer {best['layer']}: CV acc {best['cv_accuracy_mean']:.3f} ± {best['cv_accuracy_std']:.3f}")
        print(f"    Held-out B -> unsafe: {best['holdout_b_flagged_unsafe_frac']:.3f}")
        print(f"    Quadrant C -> unsafe: {best['quadrant_c_flagged_unsafe_frac']:.3f}")
        print(f"    Quadrant D -> unsafe: {best['quadrant_d_flagged_unsafe_frac']:.3f}")

    print(f"\n{'Model':<6} {'Layer':<7} {'CV acc':<10} {'B(holdout)':<12} {'C':<8} {'D':<8}")
    for stage in STAGES:
        if stage not in all_results:
            continue
        best = pick_most_informative_layer(all_results[stage])
        print(f"{stage:<6} {best['layer']:<7} {best['cv_accuracy_mean']:.3f}     "
              f"{best['holdout_b_flagged_unsafe_frac']:.3f}        "
              f"{best['quadrant_c_flagged_unsafe_frac']:.3f}   {best['quadrant_d_flagged_unsafe_frac']:.3f}")


if __name__ == "__main__":
    main()