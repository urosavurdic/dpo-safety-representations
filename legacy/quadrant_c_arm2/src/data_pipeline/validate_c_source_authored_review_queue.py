"""
Milestone 3B: validation of the C-source-authored ("Arm-2") human-review
queue built by 3A4 (src/data_pipeline/score_and_queue_c_source_authored.py).

Context: logs/release_gap_audit.md item 2 flagged that 3A4 already runs its
own inline `validate_queue()` at construction time, but no *separate*,
recorded validation step exists that independently re-checks the queue as
currently committed in the repository against its current inputs. This
script is that missing, explicit Milestone 3B step. It is read-only:

  - it does not perform the human review itself (that is Milestone 3C,
    HUMAN ONLY, tracked by the `review_status` column staying "pending");
  - it does not change any of the 52 candidate rows, decisions, or the
    review CSV in any way;
  - it does not invent a new construction method, scoring metric, or
    stratification rule - the one "recompute and compare" check below
    calls the existing, already-reviewed 3A4 scoring/queue-selection
    functions unchanged, it does not reimplement them.

Checks performed (per the Milestone 3B brief):
  1. Schema           - queue CSV header matches the 3A4 script's own
                         CSV_FIELDNAMES contract exactly (no drift).
  2. Row identity      - exactly one row per record_id, no duplicates,
                         row count matches the 3A4 log's recorded count.
  3. Review status     - every row is still "pending" (i.e. Milestone 3C
                         human review has not silently been skipped or
                         partially applied) with no review_notes tampering.
  4. Provenance fields  - source_dataset/source_repository/source_revision/
                         source_file_sha256/prompt_sha256/provenance_class
                         are present and non-empty for every row, and
                         prompt_sha256/source_file_sha256 match what the
                         3A3-validated input records for the same
                         record_id
  5. Contamination/     - contamination_status and overlap_status take
     overlap fields        only the "clean" values the 3A4 pipeline is
                         supposed to guarantee for anything it queues
                         (exact-contamination-clean, no c_paired/quadrant-A
                         overlap) - a flagged row reaching the queue would
                         be a pipeline defect.
  6. Construction        - c_construction == "c_source_authored" and
     identity              arm == "c_source_authored" for every row (this
                         queue must never silently mix in c_paired rows),
                         and pair_id is empty (Arm-2 is unpaired).
  7. Relationship to      - (a) every queued record_id is present in the
     eligible population    current 3A3-validated candidate universe with
                         candidate_universe_status == "eligible_for_3a3";
                         (b) re-running the existing, unmodified 3A4
                         scoring + Q25-selection functions against the
                         *current* committed 3A3 input reproduces a
                         record_id set and CSV byte-for-byte identical to
                         the committed queue - i.e. the queue is exactly
                         what the current repository state says it should
                         be, not a stale artifact from an earlier input.

Does NOT perform: near-duplicate semantic dedup (still an open,
documented limitation per release_gap_audit.md item 10, HUMAN/scientific
call), Milestone 3C review, or any benchmark integration.
"""
import argparse
import csv
import json
import tempfile
from pathlib import Path

from legacy.quadrant_c_arm2.src.data_pipeline.score_and_queue_c_source_authored import (
    CSV_FIELDNAMES,
    GATE_CONFIG_PATH,
    OUT_QUEUE_CSV,
    VALIDATED_JSONL_PATH,
    load_gate_config,
    load_validated_records,
    score_and_rank,
)
from src.data_pipeline.build_c_source_authored_candidates import (
    EVAL_SET_PATH,
    file_sha256,
)
from src.corpus_discrimination import build_fw_from_eval

REPO_ROOT = Path(__file__).resolve().parents[4]

QUEUE_CSV_PATH = OUT_QUEUE_CSV
SCORING_LOG_JSON = REPO_ROOT / "logs/3a4_scoring.json"

OUT_LOG_JSON = REPO_ROOT / "logs/milestone_3b_arm2_queue_validation.json"
OUT_LOG_MD = REPO_ROOT / "logs/milestone_3b_arm2_queue_validation.md"

REQUIRED_PROVENANCE_FIELDS = [
    "source_dataset", "source_repository", "source_revision",
    "source_file_sha256", "prompt_sha256", "provenance_class",
]
EXPECTED_CONTAMINATION_STATUSES = {
    "checked_exact_zero_near_unknown",
    "checked_exact_zero_near_status_mixed",
}
EXPECTED_OVERLAP_STATUS = "no_c_paired_or_quadrant_a_overlap"


def load_queue_rows(path):
    with open(path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        header = reader.fieldnames
        rows = list(reader)
    return header, rows


def check_schema(header):
    problems = []
    if header != CSV_FIELDNAMES:
        problems.append(
            "queue CSV header does not match score_and_queue_c_source_authored.CSV_FIELDNAMES:\n"
            f"  csv header: {header}\n  expected:   {CSV_FIELDNAMES}"
        )
    return problems


def check_row_identity(rows, expected_count):
    problems = []
    ids = [r["record_id"] for r in rows]
    if len(ids) != len(set(ids)):
        seen = set()
        dupes = set()
        for i in ids:
            if i in seen:
                dupes.add(i)
            seen.add(i)
        problems.append(f"duplicate record_id values in queue: {sorted(dupes)}")
    if len(rows) != expected_count:
        problems.append(
            f"queue row count {len(rows)} does not match 3A4 log's recorded "
            f"final_queue_count {expected_count}"
        )
    return problems


def check_review_status(rows):
    problems = []
    for r in rows:
        if r["review_status"] != "pending":
            problems.append(
                f"{r['record_id']}: review_status is {r['review_status']!r}, "
                "expected 'pending' (Milestone 3C human review has not run yet)"
            )
        if r["review_notes"] != "":
            problems.append(
                f"{r['record_id']}: review_notes is non-empty ({r['review_notes']!r}) "
                "before Milestone 3C human review has occurred"
            )
    return problems


def check_provenance_fields(rows, eligible_by_id):
    problems = []
    for r in rows:
        rid = r["record_id"]
        for field in REQUIRED_PROVENANCE_FIELDS:
            if not (r.get(field) or "").strip():
                problems.append(f"{rid}: provenance field '{field}' is missing or empty")
        src = eligible_by_id.get(rid)
        if src is None:
            continue  # reported separately by check_population_relationship
        if r["prompt_sha256"] != src.get("prompt_sha256"):
            problems.append(f"{rid}: queue prompt_sha256 does not match 3A3-validated input record")
        if r["source_file_sha256"] != src.get("source_file_sha256"):
            problems.append(f"{rid}: queue source_file_sha256 does not match 3A3-validated input record")
    return problems


def check_contamination_and_overlap(rows):
    problems = []
    for r in rows:
        rid = r["record_id"]
        if r["contamination_status"] not in EXPECTED_CONTAMINATION_STATUSES:
            problems.append(
                f"{rid}: contamination_status {r['contamination_status']!r} is not one of "
                f"the expected clean-for-queueing values {sorted(EXPECTED_CONTAMINATION_STATUSES)}"
            )
        if r["overlap_status"] != EXPECTED_OVERLAP_STATUS:
            problems.append(
                f"{rid}: overlap_status {r['overlap_status']!r} != expected {EXPECTED_OVERLAP_STATUS!r} "
                "(a c_paired/quadrant-A overlapping row must never reach this queue)"
            )
    return problems


def check_construction_identity(rows):
    problems = []
    for r in rows:
        rid = r["record_id"]
        if r["c_construction"] != "c_source_authored":
            problems.append(f"{rid}: c_construction is {r['c_construction']!r}, expected 'c_source_authored'")
        if r["arm"] != "c_source_authored":
            problems.append(f"{rid}: arm is {r['arm']!r}, expected 'c_source_authored'")
        if r["pair_id"] != "":
            problems.append(f"{rid}: pair_id is {r['pair_id']!r}, expected empty (Arm-2 is unpaired)")
    return problems


def check_population_relationship(rows, eligible_by_id):
    """(a) membership: every queued row traces to a currently-eligible 3A3 record."""
    problems = []
    for r in rows:
        rid = r["record_id"]
        src = eligible_by_id.get(rid)
        if src is None:
            problems.append(f"{rid}: not found among current 3A3-eligible records")
        elif src.get("candidate_universe_status") != "eligible_for_3a3":
            problems.append(
                f"{rid}: source record's candidate_universe_status is "
                f"{src.get('candidate_universe_status')!r}, expected 'eligible_for_3a3'"
            )
    return problems


def reproduce_queue_from_current_inputs(tmp_dir):
    """(b) full-pipeline reproduction: re-run the existing, unmodified 3A4
    scoring + selection functions against the current committed 3A3 input
    and gate config, and write a CSV using the same writer the 3A4 script
    uses, for a byte-for-byte comparison against the committed queue.
    Reuses score_and_rank/select_review_queue/build_queue_row/write_queue_csv
    unchanged - no new scoring or stratification logic is introduced here.
    """
    from legacy.quadrant_c_arm2.src.data_pipeline.score_and_queue_c_source_authored import (
        build_queue_row,
        select_review_queue,
        write_queue_csv,
    )

    records = load_validated_records(VALIDATED_JSONL_PATH)
    config = load_gate_config(GATE_CONFIG_PATH)
    stratum_key = config["default_source_authored_review_stratum"]
    limit = config["source_authored_review_limit"]
    min_tok_frac = config["min_token_recognition_fraction"]

    eligible = [r for r in records if r["candidate_universe_status"] == "eligible_for_3a3"]
    fw = build_fw_from_eval(str(EVAL_SET_PATH))
    scored = score_and_rank(eligible, fw, min_tok_frac)
    eligible_by_id = {row["record_id"]: row for row in scored}

    queue_records = select_review_queue(scored, stratum_key, limit)
    queue_rows = [build_queue_row(r) for r in queue_records]

    reproduced_path = tmp_dir / "reproduced_queue.csv"
    write_queue_csv(queue_rows, reproduced_path)
    return reproduced_path, eligible_by_id


def check_reproduction_matches_committed_queue(reproduced_path, committed_path):
    problems = []
    reproduced_sha = file_sha256(reproduced_path)
    committed_sha = file_sha256(committed_path)
    matches = reproduced_sha == committed_sha
    if not matches:
        problems.append(
            "queue reproduced from current committed inputs does NOT match the "
            f"committed queue byte-for-byte (reproduced sha256={reproduced_sha}, "
            f"committed sha256={committed_sha}) -- the queue is stale relative to "
            "its current inputs"
        )
    return problems, reproduced_sha, committed_sha


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--queue-csv", default=str(QUEUE_CSV_PATH))
    parser.add_argument("--validated-jsonl", default=str(VALIDATED_JSONL_PATH))
    parser.add_argument("--scoring-log-json", default=str(SCORING_LOG_JSON))
    parser.add_argument("--out-log-json", default=str(OUT_LOG_JSON))
    parser.add_argument("--out-log-md", default=str(OUT_LOG_MD))
    args = parser.parse_args()

    queue_path = Path(args.queue_csv)
    header, rows = load_queue_rows(queue_path)
    queue_sha256 = file_sha256(queue_path)

    with open(args.scoring_log_json, "r", encoding="utf-8") as f:
        scoring_log = json.load(f)
    expected_count = scoring_log["counts"]["final_queue_count"]
    expected_queue_sha = scoring_log["output_artifacts"]["review_queue_csv"]["sha256"]

    records = load_validated_records(Path(args.validated_jsonl))
    validated_sha256 = file_sha256(Path(args.validated_jsonl))
    eligible_records = [r for r in records if r["candidate_universe_status"] == "eligible_for_3a3"]
    eligible_by_id = {r["record_id"]: r for r in eligible_records}

    checks = {}
    checks["schema"] = check_schema(header)
    checks["row_identity"] = check_row_identity(rows, expected_count)
    checks["review_status"] = check_review_status(rows)
    checks["provenance_fields"] = check_provenance_fields(rows, eligible_by_id)
    checks["contamination_and_overlap"] = check_contamination_and_overlap(rows)
    checks["construction_identity"] = check_construction_identity(rows)
    checks["population_relationship_membership"] = check_population_relationship(rows, eligible_by_id)

    checks["queue_hash_matches_3a4_log"] = (
        [] if queue_sha256 == expected_queue_sha else
        [f"committed queue sha256 {queue_sha256} != 3A4 log's recorded {expected_queue_sha}"]
    )

    with tempfile.TemporaryDirectory() as tmp:
        tmp_dir = Path(tmp)
        reproduced_path, _ = reproduce_queue_from_current_inputs(tmp_dir)
        repro_problems, reproduced_sha, committed_sha = check_reproduction_matches_committed_queue(
            reproduced_path, queue_path
        )
    checks["population_relationship_full_reproduction"] = repro_problems

    all_problems = [p for plist in checks.values() for p in plist]
    overall_pass = not all_problems

    summary = {
        "milestone": "3B",
        "objective": "Independently validate the committed Arm-2 (c_source_authored) "
                     "review queue against the current repository state, without "
                     "performing the human review (Milestone 3C) or changing any "
                     "candidate decision.",
        "queue_csv": {
            "path": str(queue_path.relative_to(REPO_ROOT)) if queue_path.is_absolute() else str(queue_path),
            "sha256": queue_sha256,
            "row_count": len(rows),
        },
        "validated_input": {
            "path": str(Path(args.validated_jsonl).relative_to(REPO_ROOT))
            if Path(args.validated_jsonl).is_absolute() else args.validated_jsonl,
            "sha256": validated_sha256,
            "eligible_count": len(eligible_records),
        },
        "reproduction_check": {
            "reproduced_sha256": reproduced_sha,
            "committed_sha256": committed_sha,
            "byte_identical": reproduced_sha == committed_sha,
        },
        "checks": {name: {"pass": not problems, "problems": problems} for name, problems in checks.items()},
        "overall_pass": overall_pass,
        "not_performed_this_milestone": [
            "Milestone 3C human review of the 52 queued candidates (HUMAN ONLY)",
            "semantic near-duplicate check (still open per release_gap_audit.md item 10)",
            "any modification of the 52 candidate rows or decisions",
            "benchmark integration",
        ],
        "next_milestone": "3C - human review of the C-source-authored queue (HUMAN ONLY), "
                           "then Milestone 4A benchmark integration once both C-arm human "
                           "reviews (R104 and Arm-2) are complete.",
    }

    out_log_json = Path(args.out_log_json)
    out_log_json.parent.mkdir(parents=True, exist_ok=True)
    with open(out_log_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
        f.write("\n")

    md_lines = [
        "# Milestone 3B — Arm-2 (c_source_authored) Review-Queue Validation",
        "",
        f"Queue: `{summary['queue_csv']['path']}` (sha256 `{queue_sha256}`, "
        f"{len(rows)} rows)",
        "",
        f"Validated input: `{summary['validated_input']['path']}` "
        f"(sha256 `{validated_sha256}`, {len(eligible_records)} eligible rows)",
        "",
        f"Reproduction check: re-running the existing, unmodified 3A4 scoring + "
        f"Q25-selection pipeline against the current committed input reproduces "
        f"a queue that is byte-identical to the committed queue: "
        f"**{summary['reproduction_check']['byte_identical']}** "
        f"(reproduced sha256 `{reproduced_sha}`, committed sha256 `{committed_sha}`).",
        "",
        "## Checks",
        "",
        "| Check | Pass | Problems |",
        "|---|---|---|",
    ]
    for name, result in summary["checks"].items():
        problems = result["problems"]
        cell = "-" if not problems else "; ".join(problems[:3]) + (" ..." if len(problems) > 3 else "")
        md_lines.append(f"| {name} | {result['pass']} | {cell} |")

    md_lines += [
        "",
        f"## Overall: {'PASS' if overall_pass else 'FAIL'}",
        "",
        "Not performed this milestone:",
    ]
    for item in summary["not_performed_this_milestone"]:
        md_lines.append(f"- {item}")
    md_lines += ["", f"**Next milestone:** {summary['next_milestone']}"]

    out_log_md = Path(args.out_log_md)
    out_log_md.parent.mkdir(parents=True, exist_ok=True)
    out_log_md.write_text("\n".join(md_lines) + "\n", encoding="utf-8")

    print(f"Milestone 3B validation: {'PASS' if overall_pass else 'FAIL'}")
    if not overall_pass:
        for name, result in summary["checks"].items():
            for p in result["problems"]:
                print(f"  [{name}] {p}")
        raise SystemExit(1)

    print(f"Wrote {out_log_json}")
    print(f"Wrote {out_log_md}")


if __name__ == "__main__":
    main()
