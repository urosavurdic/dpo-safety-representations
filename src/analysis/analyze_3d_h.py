"""
Task 3D-H-A — Analyze the completed 3D-H blind human construct check.

This script ONLY analyzes the already-completed human ratings in
data/review/3d_h_blind_construct_check.csv. It does not modify S2/S3,
does not recompute any score, does not modify 3D-B/3D-C artifacts or the
frozen benchmark, does not touch the existing 3D-H blind packet or
instructions, and does not make a final benchmark GO/NO-GO decision.

Frozen inputs (read-only, hashed, never rewritten):
  - data/review/3d_h_blind_construct_check.csv  (reviewed ratings)
  - logs/3d_b_lexical_outlierness_pilot.json     (3D-B; sole source of
    p_tfidf / p_selfinfo / tail_selfinfo per record)
  - logs/3d_c_length_dependence_audit.json       (3D-C; hashed only)
  - data/quadrant_c_pipeline/lexical_outlierness_groups_v1.json
    (grouping artifact; hashed only, used for a group-uniqueness check)
  - the researcher-only 3D-H answer key, at a path supplied via the
    required --private-key-path CLI argument (never guessed or searched
    for; must resolve outside the repository)

Validation (fails closed, no output written, on any violation):
  - exactly 32 review IDs in the reviewed CSV, all unique;
  - exactly 32 review IDs in the answer key, all unique;
  - the two ID sets are identical;
  - every rating is an integer 1-5;
  - every reviewed prompt_text matches the ORIGINAL frozen packet (the
    version of the CSV as first committed, before any rating was filled
    in -- recovered from git history, not assumed);
  - the answer key path does not resolve inside the repository;
  - group IDs recovered for the 32 selected rows are pairwise distinct.

Analysis (all descriptive statistics reported in the committed JSON/MD
are group-level aggregates only -- no row-level record_id/group_id/
source/tail mapping and no answer-key contents are ever written to a
committed artifact):
  - PRIMARY: human rating vs. the predeclared p_tfidf tail (high/low),
    using n/mean/median/sd/rating-distribution per group, the mean
    difference, a Mann-Whitney-U-based rank-biserial effect size, and
    Spearman(p_tfidf, rating) across all 32 rows.
  - Fixed-seed permutation test (label permutation on the p_tfidf tail,
    test statistic = mean difference), reporting the null distribution
    summary and an empirical two-sided p-value. Not used as the sole
    evidence for the decision output.
  - SECONDARY diagnostic: the same group comparison for the original
    p_selfinfo tail (tail_selfinfo), explicitly marked secondary; the
    method that "looks better" is never substituted for the primary.
  - A descriptive SUPPORTIVE / INCONCLUSIVE / NOT SUPPORTIVE
    classification of the construct check (not a benchmark decision).

Run:
    python -m src.analysis.analyze_3d_h \
        --private-key-path /absolute/path/outside/repo/3d_h_answer_key.json
"""
from __future__ import annotations

import argparse
import csv
import io
import json
import subprocess
from collections import Counter
from pathlib import Path
from typing import Dict, List

import numpy as np
from scipy import stats

from src.data_pipeline.build_c_source_authored_candidates import file_sha256
from src.data_pipeline.lexical_outlierness import REPO_ROOT

REVIEWED_CSV_PATH = REPO_ROOT / "data/review/3d_h_blind_construct_check.csv"
PILOT_JSON_PATH = REPO_ROOT / "logs/3d_b_lexical_outlierness_pilot.json"
AUDIT_JSON_PATH = REPO_ROOT / "logs/3d_c_length_dependence_audit.json"
GROUPS_ARTIFACT_PATH = REPO_ROOT / "data/quadrant_c_pipeline/lexical_outlierness_groups_v1.json"

OUT_JSON_PATH = REPO_ROOT / "logs/3d_h_construct_check_analysis.json"
OUT_MD_PATH = REPO_ROOT / "logs/3d_h_construct_check_analysis.md"
OUT_PROVENANCE_PATH = REPO_ROOT / "logs/3d_h_construct_check_analysis_provenance.json"

REVIEWED_CSV_REL = "data/review/3d_h_blind_construct_check.csv"

EXPECTED_N = 32

PERMUTATION_SEED = 20260831
N_PERMUTATIONS = 100_000

RATING_QUESTION = (
    "How clearly does the wording itself signal a harmful operational "
    "objective?"
)

# Decision thresholds for the descriptive SUPPORTIVE / INCONCLUSIVE /
# NOT SUPPORTIVE classification. These follow conventional small/medium/
# large guidelines for an ordinal (rank-biserial) effect size together
# with the permutation-test p-value, applied uniformly rather than tuned
# to any one dataset.
SUPPORTIVE_P_MAX = 0.05
SUPPORTIVE_EFFECT_MIN = 0.3
INCONCLUSIVE_P_MAX = 0.10
INCONCLUSIVE_EFFECT_MIN = 0.1

INTERPRETATION_BOUNDARIES = [
    "Does NOT establish a universal CUE variable.",
    "Does NOT establish independence from intent.",
    "Does NOT establish causal validity.",
    "Does NOT establish a validated shared A/B/C/D axis.",
    "At most establishes whether the provisional lexical-outlierness "
    "ranking is associated with human judgments of surface "
    "harmful-operational explicitness in this small blind sample.",
]


class ValidationError(SystemExit):
    """Raised (as SystemExit) when join/validation fails. Fail closed:
    no analysis output is written."""


# ── input loading ────────────────────────────────────────────────────────
def load_reviewed_csv(path: Path) -> List[dict]:
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def load_private_key(path: Path) -> List[dict]:
    return json.loads(path.read_text(encoding="utf-8"))


def validate_private_key_path(raw_path: str) -> Path:
    resolved = Path(raw_path).expanduser().resolve()
    repo_resolved = REPO_ROOT.resolve()
    try:
        resolved.relative_to(repo_resolved)
    except ValueError:
        return resolved
    raise ValidationError(
        f"FAIL CLOSED: --private-key-path ({resolved}) resolves inside "
        f"the repository ({repo_resolved}). The answer key must never be "
        "a repository artifact or appear in the git patch."
    )


def get_original_packet_commit(path_rel: str) -> str:
    """Finds the commit that first added path_rel, so the ORIGINAL
    (pre-rating) prompt_text values can be recovered from git history
    rather than assumed."""
    out = subprocess.check_output(
        ["git", "log", "--follow", "--diff-filter=A", "--format=%H", "--", path_rel],
        cwd=REPO_ROOT,
        text=True,
    ).strip().splitlines()
    if not out:
        raise ValidationError(
            f"FAIL CLOSED: could not find the commit that added {path_rel}"
        )
    return out[-1]


def get_original_packet_prompt_texts(path_rel: str) -> Dict[str, str]:
    commit = get_original_packet_commit(path_rel)
    content = subprocess.check_output(
        ["git", "show", f"{commit}:{path_rel}"], cwd=REPO_ROOT, text=True
    )
    reader = csv.DictReader(io.StringIO(content))
    return {row["review_id"]: row["prompt_text"] for row in reader}


def get_generation_commit() -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True
    ).strip()


def _is_valid_rating(raw) -> bool:
    if raw is None:
        return False
    try:
        val = int(str(raw).strip())
    except (TypeError, ValueError):
        return False
    return 1 <= val <= 5


# ── join and validation ─────────────────────────────────────────────────
def join_and_validate(
    reviewed_rows: List[dict],
    key_entries: List[dict],
    pilot_row_level: dict,
    original_prompt_texts: Dict[str, str],
) -> List[dict]:
    if len(reviewed_rows) != EXPECTED_N:
        raise ValidationError(
            f"FAIL CLOSED: expected exactly {EXPECTED_N} reviewed rows, "
            f"got {len(reviewed_rows)}"
        )
    reviewed_ids = [r["review_id"] for r in reviewed_rows]
    if len(set(reviewed_ids)) != EXPECTED_N:
        raise ValidationError("FAIL CLOSED: reviewed review_id values are not all distinct")

    if len(key_entries) != EXPECTED_N:
        raise ValidationError(
            f"FAIL CLOSED: expected exactly {EXPECTED_N} answer-key entries, "
            f"got {len(key_entries)}"
        )
    key_ids = [e["review_id"] for e in key_entries]
    if len(set(key_ids)) != EXPECTED_N:
        raise ValidationError("FAIL CLOSED: answer-key review_id values are not all distinct")

    if set(reviewed_ids) != set(key_ids):
        raise ValidationError(
            "FAIL CLOSED: reviewed CSV and answer key do not contain the "
            "same set of review IDs"
        )

    if set(reviewed_ids) != set(original_prompt_texts.keys()):
        raise ValidationError(
            "FAIL CLOSED: reviewed CSV review IDs do not match the "
            "original frozen packet's review IDs"
        )

    reviewed_by_id = {r["review_id"]: r for r in reviewed_rows}
    key_by_id = {e["review_id"]: e for e in key_entries}

    joined = []
    for rid in reviewed_ids:
        row = reviewed_by_id[rid]
        rating_raw = row["rating"]
        if not _is_valid_rating(rating_raw):
            raise ValidationError(
                f"FAIL CLOSED: rating for {rid} is not an integer 1-5: {rating_raw!r}"
            )
        if row["prompt_text"] != original_prompt_texts[rid]:
            raise ValidationError(
                f"FAIL CLOSED: prompt text for {rid} does not match the "
                "frozen reviewed packet"
            )
        entry = key_by_id[rid]
        record_id = entry["record_id"]
        pilot_row = pilot_row_level.get(record_id)
        if pilot_row is None:
            raise ValidationError(
                f"FAIL CLOSED: record_id {record_id!r} for {rid} not found "
                "in the 3D-B pilot row_level data"
            )
        joined.append(
            {
                "review_id": rid,
                "rating": int(str(rating_raw).strip()),
                "record_id": record_id,
                "p_tfidf": entry["p_tfidf"],
                "p_selfinfo": entry["p_selfinfo"],
                "tail_tfidf": entry["tail"],
                "tail_selfinfo": pilot_row["tail_selfinfo"],
                "source": entry["source"],
                "group_id": entry["group_id"],
            }
        )

    group_ids = [j["group_id"] for j in joined]
    if len(set(group_ids)) != len(group_ids):
        raise ValidationError("FAIL CLOSED: duplicate group_id across joined rows")

    return joined


# ── statistics helpers ───────────────────────────────────────────────────
def describe_group(ratings: List[int]) -> dict:
    arr = np.array(ratings, dtype=float)
    dist = Counter(ratings)
    return {
        "n": len(ratings),
        "mean": float(arr.mean()) if len(arr) else None,
        "median": float(np.median(arr)) if len(arr) else None,
        "sd": float(arr.std(ddof=1)) if len(arr) > 1 else None,
        "distribution_1_to_5": {str(v): dist.get(v, 0) for v in range(1, 6)},
    }


def rank_biserial(group_a: List[int], group_b: List[int]) -> float:
    """Wendt's rank-biserial correlation from the Mann-Whitney U statistic
    for group_a vs. group_b: r = 2U/(n_a*n_b) - 1, so +1 means every
    group_a value exceeds every group_b value."""
    u_stat, _ = stats.mannwhitneyu(group_a, group_b, alternative="two-sided")
    return float(2 * u_stat / (len(group_a) * len(group_b)) - 1)


def two_group_comparison(high: List[int], low: List[int]) -> dict:
    result = {
        "high": describe_group(high),
        "low": describe_group(low),
        "mean_difference_high_minus_low": None,
        "mann_whitney_u": None,
        "mann_whitney_p_value": None,
        "rank_biserial_effect_size": None,
    }
    if high and low:
        result["mean_difference_high_minus_low"] = float(np.mean(high) - np.mean(low))
        u_stat, p_val = stats.mannwhitneyu(high, low, alternative="two-sided")
        result["mann_whitney_u"] = float(u_stat)
        result["mann_whitney_p_value"] = float(p_val)
        result["rank_biserial_effect_size"] = rank_biserial(high, low)
    return result


# ── primary / secondary / permutation analyses ──────────────────────────
def primary_analysis(joined: List[dict]) -> dict:
    high = [j["rating"] for j in joined if j["tail_tfidf"] == "high"]
    low = [j["rating"] for j in joined if j["tail_tfidf"] == "low"]
    comparison = two_group_comparison(high, low)

    p_tfidf_vals = [j["p_tfidf"] for j in joined]
    ratings_vals = [j["rating"] for j in joined]
    rho, sp_p = stats.spearmanr(p_tfidf_vals, ratings_vals)

    comparison["spearman_p_tfidf_vs_rating"] = {
        "rho": float(rho),
        "p_value": float(sp_p),
        "n": len(joined),
    }
    comparison["sampling_variable"] = "p_tfidf"
    return comparison


def secondary_analysis(joined: List[dict]) -> dict:
    high = [j["rating"] for j in joined if j["tail_selfinfo"] == "high"]
    low = [j["rating"] for j in joined if j["tail_selfinfo"] == "low"]
    n_mid_excluded = sum(1 for j in joined if j["tail_selfinfo"] == "mid")
    comparison = two_group_comparison(high, low)
    comparison["n_mid_tail_selfinfo_excluded_from_comparison"] = n_mid_excluded

    p_selfinfo_vals = [j["p_selfinfo"] for j in joined]
    ratings_vals = [j["rating"] for j in joined]
    rho, sp_p = stats.spearmanr(p_selfinfo_vals, ratings_vals)
    comparison["spearman_p_selfinfo_vs_rating"] = {
        "rho": float(rho),
        "p_value": float(sp_p),
        "n": len(joined),
    }
    comparison["sampling_variable"] = "p_selfinfo"
    comparison["status"] = "SECONDARY -- diagnostic only, not the primary construct check"
    return comparison


def permutation_test(
    joined: List[dict], seed: int = PERMUTATION_SEED, n_perm: int = N_PERMUTATIONS
) -> dict:
    ratings = np.array([j["rating"] for j in joined], dtype=float)
    is_high = np.array([j["tail_tfidf"] == "high" for j in joined])
    n_high = int(is_high.sum())
    n_total = len(ratings)
    observed = float(ratings[is_high].mean() - ratings[~is_high].mean())

    rng = np.random.default_rng(seed)
    # Vectorized permutation via argsort of iid random keys (deterministic
    # given the seeded generator; equivalent in distribution to repeated
    # calls to rng.permutation).
    perms = np.argsort(rng.random((n_perm, n_total)), axis=1)
    permuted_ratings = ratings[perms]
    diffs = permuted_ratings[:, :n_high].mean(axis=1) - permuted_ratings[:, n_high:].mean(axis=1)

    p_value = float((np.sum(np.abs(diffs) >= abs(observed)) + 1) / (n_perm + 1))

    percentiles = [1, 5, 25, 50, 75, 95, 99]
    return {
        "test_statistic": "mean difference (high tail mean - low tail mean)",
        "observed_statistic": observed,
        "n_permutations": n_perm,
        "seed": seed,
        "empirical_two_sided_p_value": p_value,
        "null_distribution_summary": {
            "mean": float(diffs.mean()),
            "sd": float(diffs.std(ddof=1)),
            "min": float(diffs.min()),
            "max": float(diffs.max()),
            "percentiles": {str(p): float(np.percentile(diffs, p)) for p in percentiles},
        },
        "note": "Not used as the sole evidence for the decision output.",
    }


def classify_decision(effect_size: float, permutation_p: float) -> str:
    if effect_size is None or permutation_p is None:
        return "INCONCLUSIVE"
    abs_effect = abs(effect_size)
    if permutation_p <= SUPPORTIVE_P_MAX and abs_effect >= SUPPORTIVE_EFFECT_MIN:
        return "SUPPORTIVE"
    if permutation_p <= INCONCLUSIVE_P_MAX or abs_effect >= INCONCLUSIVE_EFFECT_MIN:
        return "INCONCLUSIVE"
    return "NOT SUPPORTIVE"


# ── output writers ───────────────────────────────────────────────────────
def build_markdown(analysis: dict) -> str:
    prim = analysis["primary_analysis"]
    sec = analysis["secondary_analysis"]
    perm = analysis["permutation_test"]
    decision = analysis["decision"]

    def dist_row(desc):
        d = desc["distribution_1_to_5"]
        return " / ".join(f"{k}:{d[k]}" for k in ["1", "2", "3", "4", "5"])

    def fmt(value, spec=".3f"):
        return "n/a" if value is None else format(value, spec)

    lines = [
        "# 3D-H-A — Construct Check Analysis (Blind Human Ratings)",
        "",
        "Status: **descriptive analysis of already-collected human ratings "
        "only.** Does not modify S2/S3, the frozen benchmark, or any "
        "3D-B/3D-C/3D-H artifact. Not a benchmark GO/NO-GO.",
        "",
        f"Reviewer question: *\"{RATING_QUESTION}\"* (1 = not apparent ... "
        "5 = unmistakably apparent).",
        "",
        f"n = {analysis['n_total']} blind-reviewed prompts "
        "(StrongREJECT/SimpleSafetyTests-sourced; join validated against "
        "the researcher-only answer key and the original frozen packet).",
        "",
        "## Primary analysis (predeclared sampling variable: p_tfidf)",
        "",
        "| Group | n | mean | median | sd | 1/2/3/4/5 |",
        "|---|---|---|---|---|---|",
        f"| high p_tfidf tail | {prim['high']['n']} | {prim['high']['mean']:.3f} "
        f"| {prim['high']['median']:.1f} | {prim['high']['sd']:.3f} | {dist_row(prim['high'])} |",
        f"| low p_tfidf tail | {prim['low']['n']} | {prim['low']['mean']:.3f} "
        f"| {prim['low']['median']:.1f} | {prim['low']['sd']:.3f} | {dist_row(prim['low'])} |",
        "",
        f"- Mean difference (high - low): **{prim['mean_difference_high_minus_low']:.4f}**",
        f"- Mann-Whitney U = {prim['mann_whitney_u']:.1f}, "
        f"p = {prim['mann_whitney_p_value']:.4g}",
        f"- Rank-biserial effect size: **{prim['rank_biserial_effect_size']:.4f}**",
        f"- Spearman(p_tfidf, rating): rho = "
        f"{prim['spearman_p_tfidf_vs_rating']['rho']:.4f}, "
        f"p = {prim['spearman_p_tfidf_vs_rating']['p_value']:.4g} "
        f"(n={prim['spearman_p_tfidf_vs_rating']['n']})",
        "",
        "## Random-label permutation test",
        "",
        f"- Test statistic: {perm['test_statistic']}",
        f"- Observed statistic: {perm['observed_statistic']:.4f}",
        f"- {perm['n_permutations']} permutations, seed={perm['seed']}",
        f"- Empirical two-sided p-value: **{perm['empirical_two_sided_p_value']:.5g}**",
        f"- Null distribution: mean={perm['null_distribution_summary']['mean']:.4f}, "
        f"sd={perm['null_distribution_summary']['sd']:.4f}",
        "",
        "Not used as the sole evidence for the decision output below.",
        "",
        "## SECONDARY diagnostic (original p_selfinfo tail -- not the primary check)",
        "",
        "| Group | n | mean | median | sd | 1/2/3/4/5 |",
        "|---|---|---|---|---|---|",
        f"| high p_selfinfo tail | {sec['high']['n']} | {fmt(sec['high']['mean'])} "
        f"| {fmt(sec['high']['median'], '.1f')} | {fmt(sec['high']['sd'])} | {dist_row(sec['high'])} |",
        f"| low p_selfinfo tail | {sec['low']['n']} | {fmt(sec['low']['mean'])} "
        f"| {fmt(sec['low']['median'], '.1f')} | {fmt(sec['low']['sd'])} | {dist_row(sec['low'])} |",
        "",
        f"- mid-tail rows excluded from this comparison: "
        f"{sec['n_mid_tail_selfinfo_excluded_from_comparison']}",
        f"- Mean difference (high - low): {fmt(sec['mean_difference_high_minus_low'], '.4f')}",
        f"- Rank-biserial effect size: {fmt(sec['rank_biserial_effect_size'], '.4f')}",
        f"- Spearman(p_selfinfo, rating): rho = "
        f"{sec['spearman_p_selfinfo_vs_rating']['rho']:.4f}, "
        f"p = {sec['spearman_p_selfinfo_vs_rating']['p_value']:.4g}",
        "",
        f"**{sec['status']}**",
        "",
        "## Decision output",
        "",
        f"### {decision}",
        "",
        "This is a descriptive classification of the construct check, not "
        "a formal benchmark GO/NO-GO.",
        "",
        "## Interpretation boundaries",
        "",
    ]
    for boundary in INTERPRETATION_BOUNDARIES:
        lines.append(f"- {boundary}")
    lines.append("")
    return "\n".join(lines)


def main(argv: List[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--private-key-path",
        required=True,
        help="Exact path to the researcher-only 3D-H answer key (outside "
        "the repository). Never guessed or searched for.",
    )
    args = parser.parse_args(argv)

    private_key_path = validate_private_key_path(args.private_key_path)
    if not private_key_path.exists():
        raise ValidationError(
            f"FAIL CLOSED: --private-key-path does not exist: {private_key_path}"
        )

    reviewed_rows = load_reviewed_csv(REVIEWED_CSV_PATH)
    key_entries = load_private_key(private_key_path)
    pilot = json.loads(PILOT_JSON_PATH.read_text(encoding="utf-8"))
    pilot_row_level = pilot["scoring"]["row_level"]
    original_prompt_texts = get_original_packet_prompt_texts(REVIEWED_CSV_REL)

    joined = join_and_validate(reviewed_rows, key_entries, pilot_row_level, original_prompt_texts)

    prim = primary_analysis(joined)
    sec = secondary_analysis(joined)
    perm = permutation_test(joined)
    decision = classify_decision(prim["rank_biserial_effect_size"], perm["empirical_two_sided_p_value"])

    analysis = {
        "task": "3D-H-A construct check analysis",
        "reviewer_question": RATING_QUESTION,
        "n_total": len(joined),
        "primary_analysis": prim,
        "permutation_test": perm,
        "secondary_analysis": sec,
        "decision": decision,
        "interpretation_boundaries": INTERPRETATION_BOUNDARIES,
        "is_benchmark_go_no_go": False,
    }

    OUT_JSON_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON_PATH.write_text(json.dumps(analysis, indent=2), encoding="utf-8")
    OUT_MD_PATH.write_text(build_markdown(analysis), encoding="utf-8")

    reviewed_csv_sha256 = file_sha256(REVIEWED_CSV_PATH)
    pilot_sha256 = file_sha256(PILOT_JSON_PATH)
    audit_sha256 = file_sha256(AUDIT_JSON_PATH)
    groups_sha256 = file_sha256(GROUPS_ARTIFACT_PATH)
    private_key_sha256 = file_sha256(private_key_path)
    generation_commit = get_generation_commit()

    provenance = {
        "task": "3D-H-A construct check analysis",
        "inputs": {
            "reviewed_csv": {"path": REVIEWED_CSV_REL, "sha256": reviewed_csv_sha256},
            "pilot_json_3d_b": {
                "path": "logs/3d_b_lexical_outlierness_pilot.json",
                "sha256": pilot_sha256,
            },
            "audit_json_3d_c": {
                "path": "logs/3d_c_length_dependence_audit.json",
                "sha256": audit_sha256,
            },
            "grouping_artifact": {
                "path": "data/quadrant_c_pipeline/lexical_outlierness_groups_v1.json",
                "sha256": groups_sha256,
            },
        },
        "private_key_sha256": private_key_sha256,
        "permutation_random_seed": PERMUTATION_SEED,
        "n_permutations": N_PERMUTATIONS,
        "analysis_code_commit": generation_commit,
        "exact_statistical_procedures": (
            "Primary: two-group (high/low p_tfidf tail) comparison of "
            "human ratings via n/mean/median/sample-sd/rating distribution "
            "per group, mean difference, Mann-Whitney U with Wendt's "
            "rank-biserial effect size (r = 2U/(n1*n2) - 1), and "
            "Spearman correlation between continuous p_tfidf and rating "
            "across all reviewed rows. Fixed-seed permutation test "
            "(label permutation on the p_tfidf tail; test statistic = "
            "mean difference; two-sided empirical p-value with +1/+1 "
            "continuity correction). Secondary: identical group "
            "comparison and Spearman correlation using the original "
            "p_selfinfo tail/score, explicitly marked secondary, with "
            "rows whose p_selfinfo tail is 'mid' excluded from the "
            "two-group comparison only."
        ),
        "sample_sizes": {
            "n_total": len(joined),
            "n_high_p_tfidf": prim["high"]["n"],
            "n_low_p_tfidf": prim["low"]["n"],
            "n_high_p_selfinfo": sec["high"]["n"],
            "n_low_p_selfinfo": sec["low"]["n"],
            "n_mid_p_selfinfo_excluded": sec["n_mid_tail_selfinfo_excluded_from_comparison"],
        },
    }
    OUT_PROVENANCE_PATH.write_text(json.dumps(provenance, indent=2), encoding="utf-8")

    print(f"decision={decision}")
    print(
        f"primary: n_high={prim['high']['n']} n_low={prim['low']['n']} "
        f"mean_diff={prim['mean_difference_high_minus_low']:.4f} "
        f"rank_biserial={prim['rank_biserial_effect_size']:.4f} "
        f"mw_p={prim['mann_whitney_p_value']:.4g}"
    )
    print(f"permutation_p={perm['empirical_two_sided_p_value']:.5g}")
    print(f"analysis_json={OUT_JSON_PATH.relative_to(REPO_ROOT)}")
    print(f"analysis_md={OUT_MD_PATH.relative_to(REPO_ROOT)}")
    print(f"provenance={OUT_PROVENANCE_PATH.relative_to(REPO_ROOT)}")
    print(f"generation_commit={generation_commit}")


if __name__ == "__main__":
    main()
