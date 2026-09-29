"""WP-3 / WP-3b driver: saturation check, headline pairs, full matrix.

    python -m src.stagecmp.run_wp3 --pooling final_token
    python -m src.stagecmp.run_wp3 --pooling mean_last5

Writes into ``results/stagecmp/``. The full 9x9 matrix runs only when the
saturation check says the bounded metrics have range, and it skips the
subspace SVDs, which are the expensive part and are only interpretable for the
pre-named headline pairs anyway (``rho_AD_perp`` is asymmetric in the pre stage).
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

from src.common.activations import ACT_DIR, DIRECTIONS_DIR
from src.common.stages import ALL_STAGES
from src.stagecmp import STATUS_EXPLORATORY
from src.stagecmp.pairs import pair_key, unordered_pairs
from src.stagecmp.repr_metrics import headline_pairs, pair_report, saturation_check
from src.v2_io import write_json_lf

OUT_DIR = Path("results/stagecmp")
MATRIX_LAYERS = (16, 20, 24, 28)


def full_matrix(act_dir, directions_dir, pooling, layers=MATRIX_LAYERS) -> dict:
    """All 36 unordered pairs, bounded metrics only, at a few named layers."""
    out = {}
    for pre, post in unordered_pairs(ALL_STAGES):
        rep = pair_report(pre, post, act_dir, directions_dir, pooling,
                          layers=layers, with_subspace=False)
        out[pair_key(pre, post)] = {
            str(r["layer"]): {
                "cka": r["cka"],
                "direction_cosine": r["direction_cosine"],
                "contrast_norm_ratio": r["contrast_norm"]["ratio"],
                "relative_drift_mean": r["relative_drift_mean"],
                "selectivity_A_minus_D": r["selectivity"]["A_minus_D"],
                "selectivity_C_minus_B": r["selectivity"]["C_minus_B"],
            }
            for r in rep["per_layer"]
        }
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--pooling", default="final_token",
                    choices=["final_token", "mean_last5"])
    ap.add_argument("--act-dir", default=str(ACT_DIR))
    ap.add_argument("--directions-dir", default=str(DIRECTIONS_DIR))
    ap.add_argument("--skip-matrix", action="store_true")
    args = ap.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    print(f"[{args.pooling}] saturation check ...")
    sat = saturation_check(args.act_dir, args.directions_dir, args.pooling)
    v = sat["verdict"]
    print(f"   max CKA spread {v['max_cka_spread']:.4f} -> has range: {v['cka_has_range']}")
    print(f"   max cos spread {v['max_cosine_spread']:.4f} -> has range: {v['cosine_has_range']}")

    print(f"[{args.pooling}] headline pairs, all layers ...")
    head = headline_pairs(args.act_dir, args.directions_dir, args.pooling)

    payload = {
        "status": STATUS_EXPLORATORY,
        "pooling": args.pooling,
        "saturation_check": sat,
        "headline_pairs": head["pairs"],
    }

    if v["cka_has_range"] and not args.skip_matrix:
        print(f"[{args.pooling}] full 9x9 matrix at layers {MATRIX_LAYERS} ...")
        payload["full_matrix"] = full_matrix(
            args.act_dir, args.directions_dir, args.pooling)
        payload["full_matrix_layers"] = list(MATRIX_LAYERS)
    else:
        payload["full_matrix"] = None
        payload["full_matrix_skipped_because"] = (
            "saturation check says the bounded metrics are at ceiling"
            if not v["cka_has_range"] else "--skip-matrix")

    target = OUT_DIR / f"representational_{args.pooling}.json"
    write_json_lf(target, payload)
    print(f"\nwrote {target}   ({time.time() - t0:.0f}s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
