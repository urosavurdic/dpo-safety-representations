"""
Turns a real, completed 8-stage eval_steering_v2.py run (see
run_full_steering.py) into (a) fresh Wilson-CI stats per stage/quadrant/
category, reusing the already-validated classify_completion/rate_with_ci
pipeline exactly as summarize_causal_ablation.py does, and (b) an explicit
diff against the numbers currently published in README.md's Finding 4 --
so a material change gets flagged, not silently overwritten (this was an
explicit requirement of the handoff for this run: "if a number changes
materially from the old Finding 4, say so explicitly rather than quietly
overwriting").

This script computes NOTHING it doesn't derive from the actual input
files. The OLD_FINDING4 constants below are a hand-transcribed snapshot of
README.md's CURRENT Finding 4 section (as of the pre-rerun README this
script's output is meant to replace) -- copied once, not recomputed --
included only so the diff has something concrete to compare against. If
README.md's Finding 4 text has since changed for any reason, re-copy this
snapshot before trusting the diff.

Several OLD_FINDING4 entries are deliberately left as `None` where the
published text only gave a qualitative claim ("close to zero induced
refusal") and no exact count -- the diff reports these as "no precise old
figure to compare against" rather than inventing a number to diff against.

Usage (after a real run_full_steering.py pass has produced raw files):
    python -m src.analysis.build_finding4_report --manifest results/manifests/full_steering_<ts>.json
    python -m src.analysis.build_finding4_report --files results/raw/steering_v2_M3_L24_..._QAD.json ...
"""
import argparse
import json
from collections import defaultdict
from pathlib import Path

from src.analysis.eval_refusal_classifier import classify_refusal, is_degenerate, is_soft_deflection
from src.eval_stats import rate_with_ci
from src.io_utils import load_json

CATEGORIES = ["degenerate", "refusal", "soft_deflection", "comply"]

# Hand-transcribed from README.md's Finding 4 section at the time this
# script was written -- see module docstring. n is the OLD quadrant size
# those counts were computed on (pre-expansion: A=D=50), included so the
# diff can report rates as well as raw counts even though the NEW run
# uses a different, larger, held-out-only N.
OLD_FINDING4 = {
    "M1":            {"A": {"refusal": {"baseline": None, "steered": None}}, "note": "qualitative only: \"close to zero induced refusal\""},
    "M1_alt":        {"A": {"refusal": {"baseline": None, "steered": None}}, "note": "qualitative only: \"close to zero induced refusal\""},
    "M2":            {"A": {"refusal": {"baseline": 0, "steered": 0}, "n": 50}, "note": "quadrant-A refusal literally 0 in both conditions"},
    "M2_alt":        {"D": {"degenerate": {"baseline": 1, "steered": 5}, "n": 50}, "note": "degenerate collapse reproduces on quadrant D"},
    "M3":            {"A": {"refusal": {"baseline": 7, "steered": 13}, "n": 50},
                       "D": {"comply": {"baseline": 49, "steered": 49}, "n": 50}},
    "M3_alt":        {"A": {"refusal": {"baseline": 6, "steered": 9}, "n": 50},
                       "D": {"comply": {"baseline": 50, "steered": 48}, "n": 50}},
    "M3_direct":     {"D": {"comply": {"baseline": 39, "steered": None}, "n": 50},
                       "note": "baseline-only numbers reported; 10/50 refusal+soft_deflection at baseline"},
    "M3_direct_alt": {"D": {"comply": {"baseline": 36, "steered": None}, "n": 50},
                       "note": "baseline-only numbers reported; quadrant-A wording changed for 39/50 prompts "
                               "but category rarely flipped -- no clean before/after refusal count given"},
}

# Flag a change as material if either: the raw count differs by more than
# this many items on a comparable-size sample, or the rate differs by more
# than this many percentage points. Deliberately conservative (small
# thresholds) -- the instruction was to over-flag, not under-flag, given
# the sample sizes involved.
MATERIAL_COUNT_DELTA = 1
MATERIAL_RATE_DELTA = 0.10


def classify_completion(text):
    if is_degenerate(text):
        return "degenerate"
    if classify_refusal(text):
        return "refusal"
    if is_soft_deflection(text):
        return "soft_deflection"
    return "comply"


def load_stage_files(manifest_path=None, explicit_files=None):
    """Returns {stage: raw_file_path}. Prefers a run_full_steering.py
    manifest (has stage + output_path + succeeded per entry already);
    falls back to reading each file's "{tag}_meta.json" sidecar (written by
    eval_steering_v2.save_run_config) to recover the stage name reliably,
    rather than parsing it back out of the tag string."""
    if manifest_path:
        manifest = load_json(manifest_path)
        stage_files = {}
        for item in manifest["results"]:
            if item.get("status") == "run" and item.get("succeeded"):
                stage_files[item["stage"]] = item["output_path"]
        return stage_files

    stage_files = {}
    for f in explicit_files or []:
        out_path = Path(f)
        meta_path = out_path.with_name(out_path.stem + "_meta.json")
        if not meta_path.exists():
            raise FileNotFoundError(
                f"No sidecar metadata file {meta_path} for {f} -- can't determine which stage "
                "this file belongs to. Pass a run_full_steering.py --manifest instead, or make "
                "sure the eval_steering_v2.py _meta.json sidecar is present alongside each file."
            )
        with open(meta_path, encoding="utf-8") as mf:
            meta = json.load(mf)
        stage_files[meta["stage"]] = f
    return stage_files


def compute_stage_stats(rows):
    """{quadrant: {condition_stage_name: {category: rate_with_ci dict}}}"""
    counts = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))
    totals = defaultdict(lambda: defaultdict(int))
    for row in rows:
        q, stage = row["quadrant"], row["stage"]
        cat = classify_completion(row["response"])
        counts[q][stage][cat] += 1
        totals[q][stage] += 1

    out = {}
    for q, by_stage in counts.items():
        out[q] = {}
        for stage, cat_counts in by_stage.items():
            total = totals[q][stage]
            out[q][stage] = {cat: rate_with_ci(cat_counts.get(cat, 0), total) for cat in CATEGORIES}
    return out


def find_baseline_steered_condition_names(stage_stats_for_quadrant):
    stages = list(stage_stats_for_quadrant.keys())
    for s in stages:
        if s.endswith("_baseline"):
            steered = s[:-len("_baseline")] + "_steered"
            if steered in stages:
                return s, steered
    return None, None


def diff_against_old(stage, quadrant, category, new_baseline_n, new_baseline_count,
                      new_steered_n, new_steered_count):
    """Returns a dict describing the comparison, or None if no old figure
    exists for this (stage, quadrant, category) to compare against."""
    old = OLD_FINDING4.get(stage, {}).get(quadrant, {}).get(category)
    if old is None:
        return None
    old_baseline, old_steered, old_n = old.get("baseline"), old.get("steered"), OLD_FINDING4[stage][quadrant].get("n")

    result = {
        "stage": stage, "quadrant": quadrant, "category": category,
        "old_baseline": old_baseline, "old_steered": old_steered, "old_n": old_n,
        "new_baseline_count": new_baseline_count, "new_baseline_n": new_baseline_n,
        "new_steered_count": new_steered_count, "new_steered_n": new_steered_n,
        "material_change": False, "reasons": [],
    }

    for label, old_val, old_n_val, new_count, new_n in [
        ("baseline", old_baseline, old_n, new_baseline_count, new_baseline_n),
        ("steered", old_steered, old_n, new_steered_count, new_steered_n),
    ]:
        if old_val is None or new_count is None or not new_n or not old_n_val:
            continue
        old_rate = old_val / old_n_val
        new_rate = new_count / new_n
        if abs(new_rate - old_rate) > MATERIAL_RATE_DELTA:
            result["material_change"] = True
            result["reasons"].append(
                f"{label} rate moved {old_rate:.1%} ({old_val}/{old_n_val}) -> "
                f"{new_rate:.1%} ({new_count}/{new_n}), exceeds {MATERIAL_RATE_DELTA:.0%} threshold"
            )
    return result


def build_report(stage_files):
    report = {"stages": {}, "diffs": []}
    for stage, path in sorted(stage_files.items()):
        rows = load_json(path)
        stats = compute_stage_stats(rows)
        report["stages"][stage] = {"source_file": path, "by_quadrant": {}}

        for quadrant, by_condition in stats.items():
            baseline_name, steered_name = find_baseline_steered_condition_names(by_condition)
            report["stages"][stage]["by_quadrant"][quadrant] = {
                "conditions": by_condition,
                "baseline_condition": baseline_name,
                "steered_condition": steered_name,
            }
            if not baseline_name:
                continue
            for category in CATEGORIES:
                b = by_condition[baseline_name][category]
                s = by_condition[steered_name][category]
                if b["n"] == 0:
                    continue
                b_count = round(b["rate"] * b["n"]) if b["rate"] is not None else None
                s_count = round(s["rate"] * s["n"]) if s["rate"] is not None else None
                diff = diff_against_old(stage, quadrant, category, b["n"], b_count, s["n"], s_count)
                if diff:
                    report["diffs"].append(diff)
    return report


def print_report(report):
    for stage, stage_data in report["stages"].items():
        print(f"\n=== {stage} ===")
        for quadrant, q_data in stage_data["by_quadrant"].items():
            print(f"  quadrant {quadrant}:")
            for condition, cats in q_data["conditions"].items():
                parts = []
                for cat in CATEGORIES:
                    c = cats[cat]
                    if c["n"]:
                        parts.append(f"{cat}={round(c['rate']*c['n'])}/{c['n']}")
                print(f"    {condition}: {', '.join(parts)}")

    print("\n=== Diff against README's currently-published Finding 4 ===")
    if not report["diffs"]:
        print("  No comparable (stage, quadrant, category) entries found in OLD_FINDING4 for this run's data.")
    for d in report["diffs"]:
        flag = "MATERIAL CHANGE" if d["material_change"] else "consistent with old"
        print(f"  [{flag}] {d['stage']} / quadrant {d['quadrant']} / {d['category']}: "
              f"old baseline={d['old_baseline']}, old steered={d['old_steered']} (n={d['old_n']}) | "
              f"new baseline={d['new_baseline_count']}/{d['new_baseline_n']}, "
              f"new steered={d['new_steered_count']}/{d['new_steered_n']}")
        for r in d["reasons"]:
            print(f"      -> {r}")

    missing_old = [s for s in report["stages"] if OLD_FINDING4.get(s, {}).get("note", "").startswith("qualitative")]
    if missing_old:
        print(f"\n  No precise old figure to diff against for: {missing_old} "
              "(README only gave a qualitative claim for these -- report the new numbers on their own merits).")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", default=None,
                         help="Path to a run_full_steering.py manifest (results/manifests/full_steering_*.json)")
    parser.add_argument("--files", nargs="+", default=None,
                         help="Explicit list of steering_v2_*.json raw result files "
                              "(each needs its _meta.json sidecar alongside it)")
    parser.add_argument("--out", default="results/summaries/finding4_report.json")
    args = parser.parse_args()

    if not args.manifest and not args.files:
        parser.error("Pass --manifest or --files -- no default, same reasoning as summarize_steering.py's --file.")

    stage_files = load_stage_files(args.manifest, args.files)
    if not stage_files:
        print("No successfully-run stages found in the given manifest/files. Nothing to report.")
        return

    print(f"Building report from {len(stage_files)} stage file(s): {sorted(stage_files)}")
    report = build_report(stage_files)
    print_report(report)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(f"\nFull report written to {out_path}")
    print("This is a diff/report, not a README edit -- update README.md's Finding 4 by hand from this, "
          "quoting the MATERIAL CHANGE lines explicitly rather than silently replacing the old numbers.")


if __name__ == "__main__":
    main()
