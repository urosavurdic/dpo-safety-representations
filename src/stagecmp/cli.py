"""Command-line entry point for the stage-vs-stage comparison.

    python -m src.stagecmp.cli all      --pooling final_token
    python -m src.stagecmp.cli behav
    python -m src.stagecmp.cli attrib   --pooling final_token
    python -m src.stagecmp.cli contrast --quadrant A
    python -m src.stagecmp.cli coverage

``weights`` is built but not wired to the real adapters: it needs ~1.1 GB of
downloads, so it refuses to run without ``--allow-download``.
"""
from __future__ import annotations

import argparse
import json

from src.common.activations import ACT_DIR, DIRECTIONS_DIR
from src.common.stages import ALL_STAGES
from src.stagecmp.attribution import profile as depth_profile
from src.stagecmp.behav_metrics import stage_profile
from src.stagecmp.contrasts import factorial_2x2, selectivity_2x2
from src.stagecmp.loaders import JUDGE_DEFAULT, scorer_coverage
from src.stagecmp.matrix import build, write

POOLINGS = ("final_token", "mean_last5")


def _print(payload) -> None:
    print(json.dumps(payload, indent=2, ensure_ascii=False))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("command",
                    choices=["all", "behav", "repr", "attrib", "contrast",
                             "coverage", "weights"])
    ap.add_argument("--pooling", default="final_token", choices=POOLINGS)
    ap.add_argument("--metric", default="withhold")
    ap.add_argument("--quadrant", default=None, choices=[None, "A", "B", "C", "D"])
    ap.add_argument("--stage", default=None)
    ap.add_argument("--fixed-reference", default=None,
                    help="project every stage onto this stage's axis instead of its own")
    ap.add_argument("--saturation-check", action="store_true",
                    help="report only the headline pairs, to test dynamic range first")
    ap.add_argument("--act-dir", default=str(ACT_DIR))
    ap.add_argument("--directions-dir", default=str(DIRECTIONS_DIR))
    ap.add_argument("--judge", default=str(JUDGE_DEFAULT))
    ap.add_argument("--allow-download", action="store_true",
                    help="weights only: permit fetching adapters (~1.1 GB)")
    args = ap.parse_args(argv)

    if args.command == "coverage":
        _print(scorer_coverage(ALL_STAGES, args.judge))
        return 0

    if args.command == "behav":
        _print({s: stage_profile(s, args.judge) for s in ALL_STAGES})
        return 0

    if args.command == "contrast":
        _print({
            "factorial": factorial_2x2(args.metric, args.quadrant, args.judge),
            "selectivity_A_minus_D": selectivity_2x2(args.metric, "A", "D", args.judge),
            "selectivity_C_minus_B": selectivity_2x2(args.metric, "C", "B", args.judge),
        })
        return 0

    if args.command in ("attrib", "repr"):
        stages = [args.stage] if args.stage else list(ALL_STAGES)
        _print({
            s: depth_profile(s, args.act_dir, args.directions_dir,
                             args.pooling, args.fixed_reference)
            for s in stages
        })
        return 0

    if args.command == "weights":
        if not args.allow_download:
            print(
                "weights: refusing to run.\n"
                "  The module is built and unit-tested, but comparing the real\n"
                "  checkpoints needs the eight LoRA adapters (~1.1 GB). Pass\n"
                "  --allow-download to fetch them, or run\n"
                "  `python -m pytest tests/stagecmp/test_weights.py` to exercise\n"
                "  the algebra on toy factors without any download."
            )
            return 1
        raise SystemExit(
            "weights: the real-adapter driver is Phase 5 and is deliberately not "
            "wired up yet. src/stagecmp/weights.py has the algebra; the loop over "
            "stages, layers and modules is the next step."
        )

    payload = build(pooling=args.pooling, metric=args.metric,
                    act_dir=args.act_dir, directions_dir=args.directions_dir,
                    judge_path=args.judge, full_matrix=not args.saturation_check)
    target = write(payload, act_dir=args.act_dir, judge_path=args.judge)
    print(f"wrote {target}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
