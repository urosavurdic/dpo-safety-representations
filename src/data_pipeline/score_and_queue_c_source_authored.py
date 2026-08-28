"""
Milestone 3A4: scores the C-source-authored candidate universe validated by
3A3 with the repository's existing Fightin' Words diagnostic, computes
deterministic global and source-stratified empirical ranks and Q10/Q25/Q40
stratum membership, and builds the human review queue.

Continues from
data/quadrant_c_pipeline/c_source_authored_candidates_validated_v1.jsonl
(3A3 output). Does NOT redo any 3A2/3A3 check - no duplicate, overlap,
contamination, or provenance re-verification here beyond the minimum
input-contract checks the 3A4 brief requires (artifact exists, its hash
matches the 3A3 log, required fields are present, exact prompt text is
present for every eligible row).

Reuses the existing FightinWords implementation (src/corpus_discrimination.py)
unchanged - no second scoring implementation. Reference corpus is
H = quadrant A union quadrant B, D = quadrant D, both read from the eval set
already established for this purpose (EVAL_SET_PATH, see
build_c_source_authored_candidates.py) via
corpus_discrimination.build_fw_from_eval. The C-source-authored candidates
are never used to construct H or D.

Fixed configuration (read from logs/benchmark_gate_config.json, never
modified here):
  - screening_strata: Q10=0.10, Q25=0.25, Q40=0.40
  - default_source_authored_review_stratum: Q25
  - source_authored_review_limit: 150
  - min_token_recognition_fraction: 0.5

Ranking: empirical_rank(score, reference_scores) from corpus_discrimination.py
- "fraction of reference scores <= this score" on the unnormalized Fightin'
Words score (the same score field the existing c_paired queue's
`fightin_words_source` / `fightin_words_candidate` columns use, via
FightinWords.score_pair). Lower rank = more D-like = more desirable for this
arm. global_rank uses every scored eligible candidate as the reference set;
source_rank restricts the reference set to the same source_dataset. Both are
computed once, over the full eligible set, before any stratum filtering, so
rank values do not shift depending on who else makes the review queue.

Tie handling: exact score ties get identical empirical_rank values by
construction. Any place that needs a total order (queue row order, the
150-row cap at a stratum boundary) breaks ties with (global_rank, record_id)
ascending - record_id is a stable, content-derived id (see
build_c_source_authored_candidates.stable_record_id), so this is
reproducible across runs.

Review queue: only candidates in the configured stratum (Q25 by default) are
queued, capped at source_authored_review_limit (150). No top-up, no
rebalancing, no source quotas - if fewer than the limit qualify, the queue
is smaller.
"""
import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

from src.corpus_discrimination import (
    assign_strata,
    build_fw_from_eval,
    empirical_rank,
    load_quadrant_texts,
)
from src.data_pipeline.build_c_source_authored_candidates import (
    EVAL_SET_PATH,
    file_sha256,
)
from src.data_pipeline.validate_c_source_authored_candidates import (
    OUT_JSONL as VALIDATED_JSONL_PATH,
)

REPO_ROOT = Path(__file__).resolve().parents[2]

GATE_CONFIG_PATH = REPO_ROOT / "logs/benchmark_gate_config.json"
INPUT_LOG_JSON = REPO_ROOT / "logs/3a3_validation.json"
OUT_QUEUE_CSV = REPO_ROOT / "data/review/c_source_authored_review_queue.csv"
OUT_LOG_JSON = REPO_ROOT / "logs/3a4_scoring.json"
OUT_LOG_MD = REPO_ROOT / "logs/3a4_scoring.md"

REQUIRED_RECORD_FIELDS = [
    "record_id", "c_construction", "source_dataset", "source_repository",
    "source_revision", "source_file_sha256", "prompt_text", "prompt_sha256",
    "candidate_universe_status", "structural_classifier_label",
    "structural_classifier_low_confidence", "c_paired_overlap",
    "quadrant_a_overlap", "contamination_exact_status", "provenance_class",
    "provenance_notes",
]

CSV_FIELDNAMES = [
    "arm", "record_id", "candidate_id", "pair_id", "c_construction",
    "source_dataset", "source_repository", "source_row_index",
    "original_source_row_id", "source_topic_category", "project_category",
    "domain", "prompt_function", "source_prompt", "candidate_prompt",
    "scored_prompt", "word_count", "character_count",
    "fightin_words_score_unnormalized", "fightin_words_score_normalized",
    "fw_z_score", "fw_token_recognition_fraction", "low_coverage_flag",
    "global_rank", "source_rank", "in_Q10", "in_Q25", "in_Q40",
    "review_stratum", "contamination_status", "overlap_status",
    "structural_classifier_label", "classifier_status",
    "provenance_class", "provenance_notes", "source_revision",
    "source_file_sha256", "prompt_sha256", "review_status", "review_notes",
]


# ── input loading ────────────────────────────────────────────────────────────
def load_gate_config(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_validated_records(path):
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def verify_3a3_artifact_hash(in_path, in_sha256, input_log_path):
    """Confirms the freshly-computed hash of the 3A3 output artifact matches
    what the committed 3A3 log recorded for it - the minimum re-check this
    milestone owes the 3A3 artifact, without redoing 3A3's own checks."""
    if not input_log_path.exists():
        return [f"3A3 log not found at {input_log_path}"]
    with open(input_log_path, "r", encoding="utf-8") as f:
        log = json.load(f)
    recorded = (
        log.get("output_artifacts", {})
        .get("validated_candidate_universe_jsonl", {})
        .get("sha256")
    )
    if recorded is None:
        return ["3A3 log has no recorded sha256 for the validated artifact"]
    if recorded != in_sha256:
        return [
            f"3A3 artifact hash mismatch: log recorded {recorded}, "
            f"actual file is {in_sha256}"
        ]
    return []


def validate_input_contract(records):
    """Minimum required invariants per the 3A4 brief section 1 - not a
    re-run of 3A2/3A3's own validation."""
    problems = []
    if not records:
        problems.append("no records loaded from validated 3A3 artifact")
    seen_ids = set()
    for r in records:
        rid = r.get("record_id", "?")
        for field in REQUIRED_RECORD_FIELDS:
            if field not in r:
                problems.append(f"{rid}: missing field {field}")
        if rid in seen_ids:
            problems.append(f"duplicate record_id in input: {rid}")
        seen_ids.add(rid)
        if r.get("candidate_universe_status") == "eligible_for_3a3":
            if not (r.get("prompt_text") or "").strip():
                problems.append(f"{rid}: eligible row has empty prompt_text")
    return problems


# ── scoring and ranking ──────────────────────────────────────────────────────
def score_and_rank(eligible, fw, min_token_recognition_fraction):
    """Scores every eligible record, then computes deterministic global and
    source-stratified empirical ranks over that same scored set (before any
    stratum filtering)."""
    scored = []
    for r in eligible:
        s = fw.score(r["prompt_text"], min_token_recognition_fraction)
        row = dict(r)
        row["_fw"] = s
        scored.append(row)

    all_scores = [row["_fw"]["fightin_words_score_unnormalized"] for row in scored]
    by_source = defaultdict(list)
    for row in scored:
        by_source[row["source_dataset"]].append(
            row["_fw"]["fightin_words_score_unnormalized"]
        )

    for row in scored:
        score = row["_fw"]["fightin_words_score_unnormalized"]
        row["_global_rank"] = empirical_rank(score, all_scores)
        row["_source_rank"] = empirical_rank(score, by_source[row["source_dataset"]])
        row["_strata"] = assign_strata(row["_global_rank"])

    # Deterministic total order for any downstream truncation/output.
    scored.sort(key=lambda row: (row["_global_rank"], row["record_id"]))
    return scored


# ── field derivation for the review-queue row ───────────────────────────────
def review_stratum_label(strata):
    if strata["in_Q10"]:
        return "Q10"
    if strata["in_Q25"]:
        return "Q25"
    if strata["in_Q40"]:
        return "Q40"
    return "none"


def contamination_status_label(record):
    exact = record.get("contamination_exact_status") or {}
    exact_clean = all(v == "clean" for v in exact.values()) if exact else False
    near = (record.get("validation_3a3") or {}).get("contamination_near_status") or {}
    near_unknown = bool(near) and all(v == "unknown" for v in near.values())
    if exact_clean and near_unknown:
        return "checked_exact_zero_near_unknown"
    if exact_clean:
        return "checked_exact_zero_near_status_mixed"
    return "contamination_flagged"


def overlap_status_label(record):
    c = record.get("c_paired_overlap")
    a = record.get("quadrant_a_overlap")
    if c is False and a is False:
        return "no_c_paired_or_quadrant_a_overlap"
    return f"overlap_present(c_paired={c},quadrant_a={a})"


def classifier_status_label(record):
    return (
        "provisional_low_confidence"
        if record.get("structural_classifier_low_confidence")
        else "confirmed"
    )


def build_queue_row(record):
    fw = record["_fw"]
    text = record["prompt_text"]
    strata = record["_strata"]
    return {
        "arm": "c_source_authored",
        "record_id": record["record_id"],
        "candidate_id": record["record_id"],
        "pair_id": "",
        "c_construction": "c_source_authored",
        "source_dataset": record["source_dataset"],
        "source_repository": record.get("source_repository", ""),
        "source_row_index": record.get("source_row_index", ""),
        "original_source_row_id": record.get("original_source_row_id") or "",
        "source_topic_category": record.get("source_topic_category") or "",
        "project_category": record.get("project_category") or "",
        "domain": "",
        "prompt_function": "",
        "source_prompt": text,
        "candidate_prompt": text,
        "scored_prompt": text,
        "word_count": len(text.split()),
        "character_count": len(text),
        "fightin_words_score_unnormalized": fw["fightin_words_score_unnormalized"],
        "fightin_words_score_normalized": fw["fightin_words_score_normalized"],
        "fw_z_score": fw["fightin_words_z_score_aggregate"],
        "fw_token_recognition_fraction": fw["token_recognition_fraction"],
        "low_coverage_flag": fw["low_coverage_flag"],
        "global_rank": record["_global_rank"],
        "source_rank": record["_source_rank"],
        "in_Q10": strata["in_Q10"],
        "in_Q25": strata["in_Q25"],
        "in_Q40": strata["in_Q40"],
        "review_stratum": review_stratum_label(strata),
        "contamination_status": contamination_status_label(record),
        "overlap_status": overlap_status_label(record),
        "structural_classifier_label": record.get("structural_classifier_label", ""),
        "classifier_status": classifier_status_label(record),
        "provenance_class": record.get("provenance_class") or "",
        "provenance_notes": record.get("provenance_notes") or "",
        "source_revision": record.get("source_revision", ""),
        "source_file_sha256": record.get("source_file_sha256", ""),
        "prompt_sha256": record.get("prompt_sha256", ""),
        "review_status": "pending",
        "review_notes": "",
    }


def select_review_queue(scored, stratum_key, limit):
    stratum_flag = f"in_{stratum_key}"
    qualifying = [row for row in scored if row["_strata"][stratum_flag]]
    # `scored` is already sorted by (global_rank, record_id); qualifying
    # preserves that order, so this slice is the deterministic top-`limit`.
    return qualifying[:limit]


# ── focused output validation (brief section 6) ─────────────────────────────
def validate_queue(queue_rows, eligible_by_id, stratum_key):
    problems = []
    seen = set()
    stratum_flag = f"in_{stratum_key}"
    for row in queue_rows:
        rid = row["record_id"]
        if rid in seen:
            problems.append(f"duplicate record_id in queue: {rid}")
        seen.add(rid)
        src = eligible_by_id.get(rid)
        if src is None:
            problems.append(f"{rid}: not found in 3A3 eligible set")
            continue
        if not src["_strata"][stratum_flag]:
            problems.append(f"{rid}: not in {stratum_key} stratum")
        if row["prompt_sha256"] != src.get("prompt_sha256"):
            problems.append(f"{rid}: prompt_sha256 mismatch")
        if row["source_file_sha256"] != src.get("source_file_sha256"):
            problems.append(f"{rid}: source_file_sha256 mismatch")
        if not (row["source_prompt"] == row["candidate_prompt"] == row["scored_prompt"]):
            problems.append(f"{rid}: source/candidate/scored prompt not equal")
        if row["review_status"] != "pending":
            problems.append(f"{rid}: review_status not pending")
        if row["pair_id"] != "":
            problems.append(f"{rid}: pair_id not null")
        if row["c_construction"] != "c_source_authored":
            problems.append(f"{rid}: c_construction not c_source_authored")
        if src.get("c_paired_overlap") is not False:
            problems.append(f"{rid}: c_paired-overlapping row entered the queue")
    if len(seen) != len({row["record_id"] for row in queue_rows}):
        problems.append("record_id uniqueness check inconsistent")
    return problems


# ── CSV writer (LF line endings, matching repo convention) ──────────────────
def write_queue_csv(queue_rows, path):
    import csv

    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDNAMES, lineterminator="\n")
        writer.writeheader()
        for row in queue_rows:
            writer.writerow(row)


# ── main ──────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--in-jsonl", default=str(VALIDATED_JSONL_PATH))
    parser.add_argument("--input-log-json", default=str(INPUT_LOG_JSON))
    parser.add_argument("--eval-set", default=str(EVAL_SET_PATH))
    parser.add_argument("--gate-config", default=str(GATE_CONFIG_PATH))
    parser.add_argument("--out-queue-csv", default=str(OUT_QUEUE_CSV))
    parser.add_argument("--out-log-json", default=str(OUT_LOG_JSON))
    parser.add_argument("--out-log-md", default=str(OUT_LOG_MD))
    args = parser.parse_args()

    in_path = Path(args.in_jsonl)
    in_sha256 = file_sha256(in_path)
    records = load_validated_records(in_path)

    problems = verify_3a3_artifact_hash(in_path, in_sha256, Path(args.input_log_json))
    problems += validate_input_contract(records)
    if problems:
        raise SystemExit("3A4 input contract validation failed:\n" + "\n".join(problems))

    config = load_gate_config(Path(args.gate_config))
    stratum_key = config["default_source_authored_review_stratum"]
    limit = config["source_authored_review_limit"]
    min_tok_frac = config["min_token_recognition_fraction"]

    eligible = [r for r in records if r["candidate_universe_status"] == "eligible_for_3a3"]

    eval_path = Path(args.eval_set)
    eval_sha256 = file_sha256(eval_path)
    a_texts = load_quadrant_texts(str(eval_path), "A")
    b_texts = load_quadrant_texts(str(eval_path), "B")
    d_texts = load_quadrant_texts(str(eval_path), "D")
    fw = build_fw_from_eval(str(eval_path))

    scored = score_and_rank(eligible, fw, min_tok_frac)
    eligible_by_id = {row["record_id"]: row for row in scored}

    queue_records = select_review_queue(scored, stratum_key, limit)
    queue_rows = [build_queue_row(r) for r in queue_records]

    queue_problems = validate_queue(queue_rows, eligible_by_id, stratum_key)
    if queue_problems:
        raise SystemExit("3A4 review queue validation failed:\n" + "\n".join(queue_problems))

    out_csv = Path(args.out_queue_csv)
    write_queue_csv(queue_rows, out_csv)
    out_csv_sha256 = file_sha256(out_csv)

    by_source_eligible = Counter(r["source_dataset"] for r in eligible)
    stratum_flag = f"in_{stratum_key}"
    by_source_qualifying = Counter(
        r["source_dataset"] for r in scored if r["_strata"][stratum_flag]
    )
    by_source_queued = Counter(r["source_dataset"] for r in queue_records)

    summary = {
        "milestone": "3A4",
        "objective": "Score the 3A3-validated C-source-authored candidate universe with "
                     "the existing Fightin' Words diagnostic, rank it, and build the "
                     "Q25 human review queue.",
        "input_artifact": {
            "path": str(in_path.relative_to(REPO_ROOT)),
            "sha256": in_sha256,
            "row_count": len(records),
            "matches_3a3_log": True,
        },
        "eligible_count": len(eligible),
        "scoring_reference": {
            "eval_set_path": str(eval_path.relative_to(REPO_ROOT)),
            "eval_set_sha256": eval_sha256,
            "h_quadrant_a_count": len(a_texts),
            "h_quadrant_b_count": len(b_texts),
            "d_quadrant_count": len(d_texts),
            "corpus_h_sha256": fw.corpus_h_sha256_,
            "corpus_d_sha256": fw.corpus_d_sha256_,
            "prior_config": fw.prior_config(),
            "tokenizer_version": scored[0]["_fw"]["tokenizer_version"] if scored else None,
            "min_token_recognition_fraction": min_tok_frac,
            "c_used_to_construct_reference": False,
        },
        "score_field_used_for_ranking": "fightin_words_score_unnormalized",
        "ranking_rules": {
            "method": "empirical_rank(score, reference_scores): fraction of reference "
                      "scores <= this score; lower rank = more D-like = more desirable "
                      "for c_source_authored",
            "global_rank_reference": "all scored eligible candidates (n={})".format(len(eligible)),
            "source_rank_reference": "scored eligible candidates within the same source_dataset",
            "computed_before_stratum_filtering": True,
            "tie_handling": "empirical_rank gives exact-score ties identical rank values by "
                            "construction; any total order needed downstream (queue row "
                            "order, the review-limit cutoff) breaks ties with "
                            "(global_rank, record_id) ascending",
        },
        "quantile_definitions": config["screening_strata"],
        "default_review_stratum": stratum_key,
        "review_queue_limit": limit,
        "counts": {
            "eligible_by_source": dict(by_source_eligible),
            "qualifying_for_stratum_by_source": dict(by_source_qualifying),
            "qualifying_for_stratum_total": sum(by_source_qualifying.values()),
            "queued_by_source": dict(by_source_queued),
            "final_queue_count": len(queue_rows),
            "capped_by_limit": sum(by_source_qualifying.values()) > limit,
        },
        "output_artifacts": {
            "review_queue_csv": {
                "path": str(out_csv.relative_to(REPO_ROOT)),
                "sha256": out_csv_sha256,
                "row_count": len(queue_rows),
            },
        },
        "not_performed_this_milestone": [
            "3A2/3A3 re-verification (duplicate/overlap/contamination/provenance)",
            "human review (milestone 3B)",
            "benchmark freeze",
        ],
        "next_milestone": "3B - human review of the C-source-authored queue.",
    }

    out_log_json = Path(args.out_log_json)
    out_log_json.parent.mkdir(parents=True, exist_ok=True)
    with open(out_log_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
        f.write("\n")

    md_lines = [
        "# 3A4 — C-Source-Authored Scoring, Ranking, and Q25 Review Queue",
        "",
        f"Input: `{summary['input_artifact']['path']}` "
        f"(sha256 `{in_sha256}`, {len(records)} rows; matches 3A3 log: "
        f"{summary['input_artifact']['matches_3a3_log']})",
        "",
        f"Eligible candidates scored: {len(eligible)}",
        "",
        "## Scoring reference (H = A ∪ B, D = quadrant D)",
        f"- Eval set: `{summary['scoring_reference']['eval_set_path']}` "
        f"(sha256 `{eval_sha256}`)",
        f"- |A| = {len(a_texts)}, |B| = {len(b_texts)}, |D| = {len(d_texts)}",
        f"- corpus_h_sha256: `{fw.corpus_h_sha256_}`",
        f"- corpus_d_sha256: `{fw.corpus_d_sha256_}`",
        f"- prior config: {fw.prior_config()}",
        f"- min_token_recognition_fraction: {min_tok_frac}",
        "- C-source-authored candidates were NOT used to construct H or D.",
        "",
        "## Ranking",
        f"- Score field: `{summary['score_field_used_for_ranking']}`",
        f"- {summary['ranking_rules']['method']}",
        f"- Tie handling: {summary['ranking_rules']['tie_handling']}",
        "",
        "## Quantiles",
        f"- Q10={config['screening_strata']['Q10']}, "
        f"Q25={config['screening_strata']['Q25']}, "
        f"Q40={config['screening_strata']['Q40']}",
        f"- Default review stratum: {stratum_key}",
        f"- Review queue limit: {limit}",
        "",
        "## Counts",
        f"- Eligible by source: {dict(by_source_eligible)}",
        f"- Qualifying for {stratum_key} by source: {dict(by_source_qualifying)} "
        f"(total {sum(by_source_qualifying.values())})",
        f"- Queued by source: {dict(by_source_queued)}",
        f"- Final queue count: {len(queue_rows)}",
        f"- Capped by limit: {summary['counts']['capped_by_limit']}",
        "",
        "## Output artifacts",
        f"- `{summary['output_artifacts']['review_queue_csv']['path']}` "
        f"(sha256 `{out_csv_sha256}`, {len(queue_rows)} rows)",
        "",
        "**Next milestone:** 3B - human review of the C-source-authored queue.",
        "",
    ]
    out_log_md = Path(args.out_log_md)
    out_log_md.parent.mkdir(parents=True, exist_ok=True)
    out_log_md.write_text("\n".join(md_lines), encoding="utf-8")

    print(json.dumps({
        "eligible_count": len(eligible),
        "final_queue_count": len(queue_rows),
        "counts": summary["counts"],
    }, indent=2))


if __name__ == "__main__":
    main()
