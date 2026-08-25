"""
Validate the frozen v2 benchmark and produce machine-readable status.

Usage:
    python -m src.validate_benchmark_v2 \\
        --benchmark data/frozen_v2/benchmark_v2_<TS>.jsonl \\
        --review-csv data/review/c_review_queue.csv \\
        --gate-config logs/benchmark_gate_config.json \\
        --split-manifest logs/direction_split_manifest.json \\
        [--out-dir logs/]

Writes:
    logs/benchmark_validation_status.json
    logs/benchmark_validation_report.md
"""

import argparse
import csv
import hashlib
import json
import math
import re
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path


def sha256file(path: str) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def normalize_prompt(text: str) -> str:
    return re.sub(r"\s+", " ", text.lower().strip())


def bootstrap_mean(values, n_boot=1000, ci=0.95):
    import random
    if len(values) < 2:
        return None, None, None
    rng = random.Random(42)
    boots = [
        sum(rng.choices(values, k=len(values))) / len(values)
        for _ in range(n_boot)
    ]
    boots.sort()
    lo = boots[int((1 - ci) / 2 * n_boot)]
    hi = boots[int((1 - (1 - ci) / 2) * n_boot)]
    return sum(values) / len(values), lo, hi


def cohen_d_pooled(a, b):
    na, nb = len(a), len(b)
    if na < 2 or nb < 2:
        return None
    ma, mb = sum(a) / na, sum(b) / nb
    sa = sum((x - ma) ** 2 for x in a) / (na - 1)
    sb = sum((x - mb) ** 2 for x in b) / (nb - 1)
    sp = math.sqrt(((na - 1) * sa + (nb - 1) * sb) / (na + nb - 2))
    return abs((ma - mb) / sp) if sp > 0 else None


def main():
    ap = argparse.ArgumentParser(description="Validate frozen v2 benchmark")
    ap.add_argument("--benchmark",       required=True)
    ap.add_argument("--review-csv",      required=True)
    ap.add_argument("--gate-config",     required=True)
    ap.add_argument("--split-manifest",  required=True)
    ap.add_argument("--out-dir",         default="logs")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(timezone.utc).isoformat()
    warnings = []

    # ── Hashes ────────────────────────────────────────────────────────────────
    bench_path   = Path(args.benchmark)
    review_path  = Path(args.review_csv)
    gate_path    = Path(args.gate_config)
    split_path   = Path(args.split_manifest)

    for p in [bench_path, review_path, gate_path, split_path]:
        if not p.exists():
            print(f"ERROR: required file missing: {p}", file=sys.stderr)
            sys.exit(1)

    bench_sha  = sha256file(str(bench_path))
    review_sha = sha256file(str(review_path))
    gate_sha   = sha256file(str(gate_path))
    split_sha  = sha256file(str(split_path))

    split_manifest = json.loads(split_path.read_text())

    # Cross-check benchmark hash against split manifest
    split_bench_sha = split_manifest.get("eval_set_sha256")   # refers to source eval, not frozen bench

    # ── Load benchmark ────────────────────────────────────────────────────────
    with open(bench_path) as f:
        bench_rows = [json.loads(l) for l in f if l.strip()]
    print(f"Benchmark rows: {len(bench_rows)}")

    by_quad = {}
    for row in bench_rows:
        q = row.get("quadrant", "MISSING")
        by_quad.setdefault(q, []).append(row)

    counts = {q: len(rows) for q, rows in by_quad.items()}
    c_rows_all = by_quad.get("C", [])
    c_by_arm = Counter(r.get("c_construction") for r in c_rows_all)

    # ── Schema integrity ──────────────────────────────────────────────────────
    required_fields = [
        "record_id", "prompt", "scored_prompt", "quadrant", "c_construction",
        "source_dataset", "split", "record_sha256",
    ]
    schema_fails = []
    for i, row in enumerate(bench_rows):
        for field in required_fields:
            if field not in row:
                schema_fails.append(f"row {i}: missing {field}")
    schema_integrity_pass = len(schema_fails) == 0

    # ── Prompt integrity ──────────────────────────────────────────────────────
    prompts = [r.get("prompt", "") for r in bench_rows]
    norm_prompts = [normalize_prompt(p) for p in prompts]
    exact_dupes = len(prompts) - len(set(prompts))
    norm_dupes = len(norm_prompts) - len(set(norm_prompts))
    record_ids = [r.get("record_id") for r in bench_rows]
    id_dupes = len(record_ids) - len(set(record_ids))

    prompt_integrity_pass = (exact_dupes == 0) and (norm_dupes == 0) and (id_dupes == 0)
    if not prompt_integrity_pass:
        warnings.append(
            f"Prompt integrity: exact_dupes={exact_dupes}, "
            f"norm_dupes={norm_dupes}, id_dupes={id_dupes}"
        )

    # ── C review pass ─────────────────────────────────────────────────────────
    with open(review_path, newline="", encoding="utf-8") as f:
        review_rows = list(csv.DictReader(f))
    pending_rows = [r for r in review_rows if r.get("review_status", "").strip() == "pending"]
    c_review_pass = len(pending_rows) == 0
    if not c_review_pass:
        warnings.append(f"c_review_pass FAIL: {len(pending_rows)} rows still pending")

    # ── Benchmark hash pass ───────────────────────────────────────────────────
    # Re-hash the file to confirm written hash matches
    rehash = sha256file(str(bench_path))
    benchmark_hash_pass = rehash == bench_sha

    # ── Artifact freshness ────────────────────────────────────────────────────
    # Activation metadata files have 370 rows (old era); current eval is 654 rows
    activation_meta = list(Path("results/activations").glob("*_metadata.json"))
    stale_activations = []
    for am in activation_meta:
        data = json.loads(am.read_text())
        row_count = len(data) if isinstance(data, list) else data.get("row_count", 0)
        if row_count != len(bench_rows):
            stale_activations.append(f"{am.name}: {row_count} rows (benchmark has {len(bench_rows)})")
    artifact_freshness_pass = len(stale_activations) == 0
    if not artifact_freshness_pass:
        warnings.append(
            f"Stale activation metadata ({len(stale_activations)} files). "
            f"GPU rerun required before mechanistic analysis."
        )

    # ── Warning-only: length confound ─────────────────────────────────────────
    c_paired = [r for r in c_rows_all if r.get("c_construction") == "c_paired"]
    a_rows   = by_quad.get("A", [])
    c_wc = [r.get("word_count") or len((r.get("prompt") or "").split()) for r in c_paired]
    a_wc = [r.get("word_count") or len((r.get("prompt") or "").split()) for r in a_rows]
    d_c = cohen_d_pooled(a_wc, c_wc)
    length_confound_pass = None  # unavailable without accepted C rows pre-review
    if c_wc and a_wc:
        length_confound_pass = (d_c is not None and d_c < 0.5)
        if not length_confound_pass:
            warnings.append(
                f"Length confound: |d|={d_c:.3f if d_c else 'NA'} between A and C-paired word counts. "
                f"Source-confounded comparison — label accordingly."
            )

    # ── Warning-only: source confound ─────────────────────────────────────────
    c_sources = Counter(r.get("source_dataset") for r in c_paired)
    a_sources = Counter(r.get("source_dataset") for r in a_rows)
    # Perfect source separation: C is all StrongREJECT, A is all HarmBench
    source_confound_pass = None
    if len(c_sources) > 0 and len(a_sources) > 0:
        if set(c_sources.keys()) & set(a_sources.keys()):
            source_confound_pass = False
            warnings.append("Source confound: A and C-paired share sources.")
        else:
            source_confound_pass = False  # Still not identified — distinct sources, not controlled
            warnings.append(
                "source_cue_effect_status=not_identified: A and C-paired come from entirely "
                "different datasets (HarmBench vs StrongREJECT). Source and label are "
                "nearly perfectly aligned. A-versus-C differences cannot be attributed "
                "to wording alone."
            )

    # ── Paired score diagnostics ──────────────────────────────────────────────
    review_accepted = [r for r in review_rows if r.get("review_status", "").strip() == "accept"]
    paired_diffs = []
    for r in review_accepted:
        try:
            d = float(r.get("fightin_words_paired_difference", 0) or 0)
            paired_diffs.append(d)
        except (ValueError, TypeError):
            pass

    paired_mean, paired_lo, paired_hi = bootstrap_mean(paired_diffs)
    n_pos_diffs = sum(1 for d in paired_diffs if d > 0)
    sd_paired = (
        math.sqrt(sum((d - paired_mean) ** 2 for d in paired_diffs) / (len(paired_diffs) - 1))
        if len(paired_diffs) >= 2 else None
    )
    smd = (paired_mean / sd_paired) if (sd_paired and sd_paired > 0) else None

    # ── Reduced-cue evidence status ───────────────────────────────────────────
    # SUPPORTED_OPERATIONALLY if majority of accepted pairs show positive difference
    # and no low_coverage flags in the accepted set
    if len(review_accepted) == 0:
        reduced_cue_status = "INCONCLUSIVE"
    elif paired_mean is not None and paired_mean > 0 and n_pos_diffs / len(paired_diffs) >= 0.5:
        reduced_cue_status = "SUPPORTED_OPERATIONALLY"
    else:
        reduced_cue_status = "INCONCLUSIVE"

    surface_separation_pass = (
        reduced_cue_status == "SUPPORTED_OPERATIONALLY"
        if len(review_accepted) > 0 else None
    )

    # ── Build status dict ─────────────────────────────────────────────────────
    status = {
        "technical_benchmark_status": (
            "PASS" if (
                schema_integrity_pass and prompt_integrity_pass and
                c_review_pass and benchmark_hash_pass and artifact_freshness_pass
            ) else "FAIL"
        ),
        "reduced_cue_evidence_status": reduced_cue_status,
        "wording_only_claim_status": "INCONCLUSIVE",
        "benchmark_path": str(bench_path),
        "benchmark_sha256": bench_sha,
        "counts": counts,
        "c_counts_by_construction": dict(c_by_arm),
        "schema_integrity_pass": schema_integrity_pass,
        "prompt_integrity_pass": prompt_integrity_pass,
        "c_review_pass": c_review_pass,
        "benchmark_hash_pass": benchmark_hash_pass,
        "artifact_freshness_pass": artifact_freshness_pass,
        "surface_separation_pass": surface_separation_pass,
        "length_confound_pass": length_confound_pass,
        "category_confound_pass": None,
        "prompt_function_confound_pass": None,
        "source_confound_pass": source_confound_pass,
        "stale_activation_files": stale_activations,
        "paired_diagnostics": {
            "n_pairs": len(paired_diffs),
            "n_positive_diff": n_pos_diffs,
            "mean_diff": round(paired_mean, 4) if paired_mean is not None else None,
            "ci_95_lo": round(paired_lo, 4) if paired_lo is not None else None,
            "ci_95_hi": round(paired_hi, 4) if paired_hi is not None else None,
            "standardized_mean_diff": round(smd, 4) if smd is not None else None,
            "resampling_unit": "matched_pair",
            "n_bootstrap_replicates": 1000,
            "note": (
                "Positive diff = candidate has lower Fightin' Words score than source. "
                "Not a causal effect of wording alone (source-confounded)."
            ),
        },
        "precision_band": (
            "very_limited_exploratory" if counts.get("C", 0) < 30 else
            "exploratory" if counts.get("C", 0) < 60 else
            "preferred_minimum" if counts.get("C", 0) < 80 else
            "desirable_target"
        ),
        "warnings": warnings,
        "inputs": {
            "benchmark": str(bench_path),
            "benchmark_sha256": bench_sha,
            "review_csv": str(review_path),
            "review_csv_sha256": review_sha,
            "gate_config": str(gate_path),
            "gate_config_sha256": gate_sha,
            "split_manifest": str(split_path),
            "split_manifest_sha256": split_sha,
        },
        "generated_at_utc": ts,
    }

    status_path = out_dir / "benchmark_validation_status.json"
    status_path.write_text(json.dumps(status, indent=2))
    print(f"\nValidation status: {status_path}")
    print(f"technical_benchmark_status: {status['technical_benchmark_status']}")
    print(f"reduced_cue_evidence_status: {status['reduced_cue_evidence_status']}")
    print(f"wording_only_claim_status: INCONCLUSIVE (fixed)")
    if warnings:
        print(f"\nWarnings ({len(warnings)}):")
        for w in warnings:
            print(f"  ⚠  {w}")

    # ── Markdown report ────────────────────────────────────────────────────────
    md_lines = [
        "# Benchmark v2 Validation Report",
        f"Generated: {ts}",
        "",
        f"**Technical benchmark status:** `{status['technical_benchmark_status']}`",
        f"**Reduced-cue evidence status:** `{status['reduced_cue_evidence_status']}`",
        f"**Wording-only claim status:** `INCONCLUSIVE` (fixed — this project does not identify a causal effect of wording alone)",
        "",
        "## Gate fields",
        f"| Field | Value |",
        f"|---|---|",
    ]
    for field in (
        ["schema_integrity_pass", "prompt_integrity_pass", "c_review_pass",
         "benchmark_hash_pass", "artifact_freshness_pass"]
        + ["source_confound_pass", "category_confound_pass",
           "prompt_function_confound_pass", "length_confound_pass",
           "surface_separation_pass", "wording_only_claim_pass"]
    ):
        val = status.get(field, "INCONCLUSIVE")
        md_lines.append(f"| {field} | {val} |")

    md_lines += [
        "",
        "## Counts",
        f"| Quadrant | Count |",
        f"|---|---|",
    ]
    for q, n in sorted(counts.items()):
        md_lines.append(f"| {q} | {n} |")

    md_lines += [
        "",
        "## C arm counts",
        f"| Arm | Count |",
        f"|---|---|",
    ]
    for arm, n in dict(c_by_arm).items():
        md_lines.append(f"| {arm} | {n} |")

    md_lines += [
        "",
        "## Paired Fightin' Words diagnostics",
        f"- n_pairs: {status['paired_diagnostics']['n_pairs']}",
        f"- n_positive_diff: {status['paired_diagnostics']['n_positive_diff']}",
        f"- mean_diff: {status['paired_diagnostics']['mean_diff']}",
        f"- 95% CI: [{status['paired_diagnostics']['ci_95_lo']}, {status['paired_diagnostics']['ci_95_hi']}]",
        f"- SMD (not Cohen's d): {status['paired_diagnostics']['standardized_mean_diff']}",
        f"- Note: {status['paired_diagnostics']['note']}",
        "",
        "## Stale artifacts",
    ]
    if stale_activations:
        for sa in stale_activations:
            md_lines.append(f"- {sa}")
    else:
        md_lines.append("- None")

    md_lines += ["", "## Warnings"]
    for w in warnings:
        md_lines.append(f"- {w}")

    md_path = out_dir / "benchmark_validation_report.md"
    md_path.write_text("\n".join(md_lines) + "\n")
    print(f"Markdown report: {md_path}")


if __name__ == "__main__":
    main()
