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
import argparse
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
# The frozen headline layer for every probe comparison (WP-Probe). The full
# per-layer curve is still saved for descriptive plotting, but no layer is
# ever *selected* from it for a headline number - selecting the layer that
# maximises quadrant-C (or D) flagging is a data-dependent choice on the
# held-out quadrants themselves. See docs/audit/selection_leakage_scan.md.
FINAL_LAYER = 28


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


def layer_row(layer_results, layer=FINAL_LAYER):
    """The fixed-layer row from a saved per-layer probe curve. This is the
    ONLY selection used on the headline path - `layer` is preregistered
    (FINAL_LAYER), never chosen from the results."""
    for row in layer_results:
        if row["layer"] == layer:
            return row
    raise ValueError(f"Layer {layer} not present in probe results "
                     f"(have {[r['layer'] for r in layer_results]})")


def pick_most_informative_layer(layer_results):
    """EXPLORATORY ONLY (gated behind --exploratory-layer-scan). Selecting the
    layer that maximises quadrant-C flagging is a data-dependent choice on a
    held-out quadrant; its output is labelled exploratory and never used as a
    headline number. See docs/audit/selection_leakage_scan.md.

    NOT max cv_accuracy_mean - that metric saturates near 1.0 at almost
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


def probe_metadata_is_fresh(stage, out_dir):
    """True only if results/probes/{stage}_metadata.json (a snapshot of the
    activation metadata this probe was actually trained on, saved alongside
    the probe results) matches the CURRENT results/activations/{stage}_
    metadata.json exactly -- same bug class as eval_behavioral.py and
    eval_extract_activations.py: `if result_path.exists(): skip` alone
    treats a probe trained on an old, smaller eval set as "done" forever,
    even after controlled_eval.jsonl grows. The comment this replaced
    ("adding a new stage must not re-run prior stages") was solving a real,
    different problem (don't waste GPU-adjacent CV-training time just
    because STAGES grew) but didn't account for the eval SET itself
    changing size/content, which invalidates every stage's probe, not just
    new ones. eval_extract_activations.py's own metadata is already
    freshness-guaranteed by its own fix, so it's used here as the ground
    truth to snapshot/compare against, rather than re-deriving anything
    from controlled_eval.jsonl directly."""
    meta_snapshot_path = out_dir / f"{stage}_metadata.json"
    live_metadata_path = ACT_DIR / f"{stage}_metadata.json"
    if not meta_snapshot_path.exists() or not live_metadata_path.exists():
        return False
    with open(meta_snapshot_path, encoding="utf-8") as f:
        saved = json.load(f)
    with open(live_metadata_path, encoding="utf-8") as f:
        live = json.load(f)
    return saved == live


def _report_row(stage, row, label):
    print(f"  {label} layer {row['layer']}: CV acc {row['cv_accuracy_mean']:.3f} "
          f"± {row['cv_accuracy_std']:.3f}")
    for name, key in (("Held-out B", "holdout_b_flagged_unsafe_frac"),
                      ("Quadrant C", "quadrant_c_flagged_unsafe_frac"),
                      ("Quadrant D", "quadrant_d_flagged_unsafe_frac")):
        val = row.get(key)
        print(f"    {name} -> unsafe: {val:.3f}" if val is not None else f"    {name} -> unsafe: n/a")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--exploratory-layer-scan", action="store_true",
        help="Also print the layer that maximises quadrant-C flagging. This is "
             "a data-dependent selection on a held-out quadrant - EXPLORATORY "
             "ONLY, never a headline number (see docs/audit/selection_leakage_scan.md).",
    )
    args = parser.parse_args()

    out_dir = Path("results/probes")
    out_dir.mkdir(parents=True, exist_ok=True)

    all_results = {}
    for stage in STAGES:
        result_path = out_dir / f"{stage}_probe_results.json"
        if result_path.exists() and probe_metadata_is_fresh(stage, out_dir):
            print(f"\n=== {stage}: already computed, fresh, loading from disk ===")
            with open(result_path, encoding="utf-8") as f:
                layer_results = json.load(f)
            all_results[stage] = layer_results
            _report_row(stage, layer_row(layer_results), "FINAL")
            continue
        if result_path.exists():
            print(f"\n=== {stage}: existing probe result is STALE (predates the current eval set) -- re-running ===")
        if not activations_available(stage):
            print(f"\n=== {stage}: SKIPPED, activations not yet extracted ===")
            continue
        print(f"\n=== {stage} ===")
        layer_results = run_for_stage(stage)
        all_results[stage] = layer_results
        with open(result_path, "w", encoding="utf-8") as f:
            json.dump(layer_results, f, ensure_ascii=False, indent=2)
        with open(ACT_DIR / f"{stage}_metadata.json", encoding="utf-8") as f:
            live_metadata = json.load(f)
        with open(out_dir / f"{stage}_metadata.json", "w", encoding="utf-8") as f:
            json.dump(live_metadata, f, ensure_ascii=False, indent=2)
        _report_row(stage, layer_row(layer_results), "FINAL")

    print(f"\nHeadline: fixed layer {FINAL_LAYER} (preregistered, not selected)\n")
    print(f"{'Model':<6} {'Layer':<7} {'CV acc':<10} {'B(holdout)':<12} {'C':<8} {'D':<8}")
    for stage in STAGES:
        if stage not in all_results:
            continue
        row = layer_row(all_results[stage])
        c = row.get("quadrant_c_flagged_unsafe_frac")
        d = row.get("quadrant_d_flagged_unsafe_frac")
        b = row.get("holdout_b_flagged_unsafe_frac")
        print(f"{stage:<6} {row['layer']:<7} {row['cv_accuracy_mean']:.3f}     "
              f"{(b if b is not None else float('nan')):.3f}        "
              f"{(c if c is not None else float('nan')):.3f}   "
              f"{(d if d is not None else float('nan')):.3f}")

    if args.exploratory_layer_scan:
        print("\n[EXPLORATORY] layer maximising quadrant-C flagging "
              "(data-dependent selection - NOT a headline result):")
        for stage in STAGES:
            if stage not in all_results:
                continue
            best = pick_most_informative_layer(all_results[stage])
            print(f"  {stage:<6} layer {best['layer']:<3} "
                  f"C={best['quadrant_c_flagged_unsafe_frac']:.3f}")


if __name__ == "__main__":
    main()