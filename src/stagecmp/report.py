"""Render the stage-comparison artifact as a readable findings memo.

Generated from the JSON rather than typed, so the prose and the artifact cannot
drift apart. Writes Markdown to ``results/stagecmp/FINDINGS_stagecmp.md``.
"""
from __future__ import annotations

import json
from pathlib import Path

from src.common.stages import ALL_STAGES
from src.stagecmp.matrix import OUT_DIR

QUADRANT_LABEL = {
    "A": "A  harmful, overt wording",
    "B": "B  benign, alarming wording",
    "C": "C  harmful, reduced-cue wording",
    "D": "D  benign, plain wording",
}
ARM_LABEL = {
    "sft_alpaca": "safety SFT  (M1 -> M2)",
    "dpo_alpaca": "DPO         (M1 -> M3_direct)",
    "sft_dolly": "safety SFT  (M1_alt -> M2_alt)",
    "dpo_dolly": "DPO         (M1_alt -> M3_direct_alt)",
}


def _ci(d, key="point"):
    if d is None or d.get("ci_low") is None:
        return "n/a"
    star = "" if d["ci_low"] <= 0 <= d["ci_high"] else "  *"
    return f"{d.get(key, 0):+.3f} [{d['ci_low']:+.3f}, {d['ci_high']:+.3f}]{star}"


def render(payload: dict) -> str:
    L = []
    add = L.append
    add(f"# Stage comparison — findings\n")
    add(f"*{payload['status']}*\n")
    add(f"Produced {payload['produced_utc']} · pooling `{payload['pooling']}` · "
        f"behavioural metric `{payload['behavioural_metric']}` "
        f"(refusal or soft deflection)\n")
    add("`*` marks an interval that excludes zero.\n")

    add("\n## 1. Withhold rate by stage and quadrant\n")
    add("| stage | A | B | C | D |")
    add("|---|---:|---:|---:|---:|")
    for s in payload["stages"]:
        pq = payload["behavioural"][s]["per_quadrant"]
        cells = " | ".join(f"{pq[q]['withhold']['rate']:.3f}" for q in "ABCD")
        add(f"| {s} | {cells} |")

    add("\n## 2. The 2x2 — objective x corpus\n")
    add(payload["caveat"] + "\n")
    for q in "ABCD":
        c = payload["contrasts_2x2"]["by_quadrant"][q]
        add(f"\n### {QUADRANT_LABEL[q]}  (n = {c['n_effective']})\n")
        add("| arm | theta | 95% CI |")
        add("|---|---:|---|")
        for arm, a in c["arms"].items():
            lo, hi = a.get("ci_low"), a.get("ci_high")
            ci = f"[{lo:+.3f}, {hi:+.3f}]" if lo is not None else "n/a"
            add(f"| {ARM_LABEL[arm]} | {a['theta']:+.3f} | {ci} |")
        add("")
        add("| effect | estimate |")
        add("|---|---|")
        for k in ("OBJ", "CORP", "INT"):
            add(f"| {k} | {_ci(c[k])} |")
        g = c["within_corpus_gaps"]
        add(f"| DPO − SFT, Alpaca | {_ci(g['alpaca_dpo_minus_sft'])} |")
        add(f"| DPO − SFT, Dolly | {_ci(g['dolly_dpo_minus_sft'])} |")

    add("\n## 3. Selectivity — discrimination, not just more refusing\n")
    for key, title in (("A_minus_D", "A minus D (overt axis)"),
                       ("C_minus_B", "C minus B (reduced-cue axis)")):
        s = payload["contrasts_2x2"]["selectivity"][key]
        if s.get("n_harm", 0) == 0:
            continue
        add(f"\n### {title}  (n = {s['n_harm']} vs {s['n_benign']})\n")
        add("| arm | selectivity | 95% CI |")
        add("|---|---:|---|")
        for arm, a in s["arms"].items():
            add(f"| {ARM_LABEL[arm]} | {a['point']:+.3f} | "
                f"[{a['ci_low']:+.3f}, {a['ci_high']:+.3f}]"
                f"{'' if a['ci_low'] <= 0 <= a['ci_high'] else '  *'} |")
        o = s["OBJ_selectivity"]
        add(f"| **OBJ (DPO − SFT)** | {o['point']:+.3f} | "
            f"[{o['ci_low']:+.3f}, {o['ci_high']:+.3f}]"
            f"{'' if o['ci_low'] <= 0 <= o['ci_high'] else '  *'} |")
        add(f"\n{s['reading']}\n")

    add("\n## 4. Depth attribution — where the contrast is written\n")
    add("Per-layer write of the A−D contrast onto each stage's own direction, "
        "`write[l] = (h_l − h_{l−1}) · d_l`. Exact, not an estimate.\n")
    add("| stage | total A−D | fraction written by layer 24 | share falling in layers 24–28 |")
    add("|---|---:|---:|---:|")
    for s in payload["stages"]:
        g = payload["depth_attribution"][s].get("ad_gap")
        if not g:
            continue
        cum = g["cumulative"]
        total = cum[-1]
        add(f"| {s} | {total:.1f} | {cum[23] / total:.3f} | {g['share_in_24_28']:.3f} |")
    add("\nPeak layer is **not** reported: the per-layer curve is multi-peaked and "
        "its argmax is unstable across stages, the same failure mode that retired "
        "the earlier bottleneck-gap claim.\n")

    add("\n## 5. Judge coverage — what these numbers rest on\n")
    add("| stage | rows | regex | StrongREJECT | WildGuard |")
    add("|---|---:|---:|---:|---:|")
    for s in payload["stages"]:
        c = payload["diagnostics"]["scorer_coverage"][s]
        add(f"| {s} | {c['n_rows']} | {c['regex']} | {c['strong_reject']} | {c['wildguard']} |")
    add("\nEverything above uses the frozen regex classifier, the only scorer "
        "complete for all nine stages. StrongREJECT and WildGuard currently cover "
        "quadrant-C rows of M2 and M3 only.\n")
    return "\n".join(L)


def main() -> int:
    import argparse

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--pooling", default="final_token",
                    choices=["final_token", "mean_last5"])
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    src = Path(OUT_DIR) / f"stage_comparison_{args.pooling}.json"
    payload = json.loads(src.read_text(encoding="utf-8"))
    text = render(payload)
    out = Path(args.out or Path(OUT_DIR) / f"FINDINGS_stagecmp_{args.pooling}.md")
    out.write_text(text, encoding="utf-8", newline="\n")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
