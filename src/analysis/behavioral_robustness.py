"""Behavioural-conclusion robustness re-tabulation (WP-Report), analysis_plan.md §5.6.

For each preregistered behavioural conclusion, show how it reads under an
escalating evidence stack:

    regex only  ->  + StrongREJECT  ->  + WildGuard  ->  + human audit

and explicitly DOWNGRADE any conclusion that rests on regex alone. This does
not compute new endpoints - it re-expresses the already-computed numbers so a
reader can see which conclusions survive instrument disagreement.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

EVIDENCE_LEVELS = ["regex_only", "plus_strongreject", "plus_wildguard", "plus_human"]


def classify_support(values_by_level: dict) -> dict:
    """values_by_level: {level: bool|None} - whether the conclusion is
    supported at that evidence level (None = not assessable at that level).
    Returns a verdict + a downgrade flag when only regex supports it."""
    supported = {lvl: values_by_level.get(lvl) for lvl in EVIDENCE_LEVELS}
    assessable = {lvl: v for lvl, v in supported.items() if v is not None}
    if not assessable:
        return {"verdict": "not_assessable", "by_level": supported}

    all_true = all(assessable.values())
    only_regex = (
        supported.get("regex_only") is True
        and all(supported.get(l) in (None, False) for l in EVIDENCE_LEVELS[1:])
    )
    human = supported.get("plus_human")
    if all_true and human is True:
        verdict = "robust_all_instruments_including_human"
    elif only_regex:
        verdict = "DOWNGRADED_regex_only"
    elif all_true:
        verdict = "consistent_across_automated_instruments (human pending)"
    elif human is False:
        verdict = "contradicted_by_human_audit"
    else:
        verdict = "mixed_instrument_support"
    return {"verdict": verdict, "by_level": supported,
            "downgraded": verdict == "DOWNGRADED_regex_only"}


def retabulate(conclusions: list) -> dict:
    """conclusions: [{"id","statement","support":{level:bool|None}}]."""
    rows = []
    for c in conclusions:
        rows.append({
            "id": c["id"],
            "statement": c["statement"],
            **classify_support(c.get("support", {})),
        })
    return {
        "evidence_levels": EVIDENCE_LEVELS,
        "conclusions": rows,
        "n_downgraded": sum(1 for r in rows if r.get("downgraded")),
        "note": (
            "regex is surface refusal language only; a conclusion supported by "
            "regex alone is downgraded per analysis_plan.md §3/§5.6."
        ),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--conclusions", required=True,
                        help="JSON list of {id, statement, support:{level:bool|null}}.")
    parser.add_argument("--out", default="results/behavioral_judges_v2/robustness_retabulation.json")
    args = parser.parse_args()
    conclusions = json.loads(Path(args.conclusions).read_text(encoding="utf-8"))
    out = retabulate(conclusions)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"{out['n_downgraded']} conclusion(s) downgraded -> {args.out}")


if __name__ == "__main__":
    main()
