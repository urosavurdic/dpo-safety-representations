"""
Milestone 3A2-1: builds the raw C-source-authored candidate universe from the
two strict-primary sources fixed by 3A1C policy (logs/3a1c_handoff.json):

    - StrongREJECT  (alexandrasouly/strongreject)
    - SimpleSafetyTests (bertiev/SimpleSafetyTests)

"C-source-authored" means UNCHANGED external source prompts selected under a
reproducible eligibility procedure - no rewriting, no paraphrasing, no
completion of incomplete prompts. This is a different track from the
c_paired (reduced-cue rewrite) pool already in candidate_records_v2.jsonl:
that pipeline stores both "source_prompt" (original) and "candidate_prompt"
(an AI-drafted rewrite). This pipeline has no rewrite step at all, so its
records carry a single "prompt_text" field that is byte-identical to the
source cell - there is deliberately no "candidate_prompt" key here, to avoid
implying a transformation occurred.

Scope (3A2-1 only - see MILESTONE 3A2-1 brief):
  Step 1  - reverify the two source files against the 3A1C-recorded hashes
  Step 2  - preserve source identity fields per candidate
  Step 3  - stand-alone/user-facing completeness via the existing structural
            classifier (src/diagnostics/classify_source_provenance.py)
  Step 4  - exact prompt-text duplicate detection (within + across sources)
  Step 5  - normalized-text duplicate detection, reusing the exact
            normalize() scheme already used by check_leakage.py /
            build_secondary_c3_c4.py (lower + collapse whitespace + strip) -
            NOT importing check_leakage.py directly, because that module
            hard-imports sentence_transformers at module scope and fails to
            load in this sandbox (no network path to huggingface.co); the
            normalization function itself is reproduced verbatim below.
  Step 6  - overlap against the FULL 155-row existing C-paired pool (not
            just the 104 currently-live frozen rows) - hard requirement
  Step 7  - overlap against every Quadrant-A row in the frozen v2 benchmark
  Step 8  - preserve StrongREJECT's upstream provenance column and
            SimpleSafetyTests' status as external dataset records (no
            invented individual-authorship claims)
  Step 9  - the structural classifier is triage only; ambiguous outcomes
            (the low-confidence imperative fallback branch) are kept in the
            universe but flagged, not auto-excluded
  Step 10 - contamination: EXACT-only check against the four training files
            (no model download required); near-duplicate contamination is
            explicitly recorded as unknown and deferred to 3A3. Per the
            3A2-1 brief this is informational only in this sub-step - it
            does NOT drive candidate_universe_status here.

Explicitly NOT done here (reserved for 3A3+): near-duplicate/embedding
checks, Fightin' Words scoring, Q10/Q25/Q40 stratification, review-queue
generation, benchmark freezing, project_category taxonomy mapping.
"""
import argparse
import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path

from src.diagnostics.classify_source_provenance import classify_prompt_like_field

REPO_ROOT = Path(__file__).resolve().parents[2]

STRONGREJECT_CSV = REPO_ROOT / "data/raw/3a1a_source_cache/strongreject/strongreject_dataset.csv"
STRONGREJECT_EXPECTED_SHA256 = "4dd70357e4ff8b5d0ba5ebafecab5d6dd5633ce8046e3dd1c8bd93e64de44381"
STRONGREJECT_REPOSITORY = "alexandrasouly/strongreject"
STRONGREJECT_REVISION = "f7cad6c17e624e21d8df2278e918ae1dddb4cb56"
STRONGREJECT_SOURCE_FILE = "strongreject_dataset/strongreject_dataset.csv"

SIMPLESAFETYTESTS_CSV = REPO_ROOT / "data/raw/3a1b_source_cache/simplesafetytests/SimpleSafetyTests - test cases.csv"
SIMPLESAFETYTESTS_EXPECTED_SHA256 = "6d95a1301e0d0f3a3c4cf5392f4afff11ad6e3066f95d23aaa138d44aedf986c"
SIMPLESAFETYTESTS_REPOSITORY = "bertiev/SimpleSafetyTests"
SIMPLESAFETYTESTS_REVISION = "d7aee9a9422a5a5488f478fd79c2479c891c0f3b"
SIMPLESAFETYTESTS_SOURCE_FILE = "SimpleSafetyTests - test cases.csv"

CPAIRED_PATH = REPO_ROOT / "data/quadrant_c_pipeline/candidate_records_v2.jsonl"
EVAL_SET_PATH = REPO_ROOT / "data/processed/controlled_eval.jsonl"

TRAINING_FILES_TO_CHECK = {
    "dpo_pairs.jsonl": REPO_ROOT / "data/processed/dpo_pairs.jsonl",
    "sft_helpful.jsonl": REPO_ROOT / "data/processed/sft_helpful.jsonl",
    "sft_helpful_alt.jsonl": REPO_ROOT / "data/processed/sft_helpful_alt.jsonl",
    "sft_safety.jsonl": REPO_ROOT / "data/processed/sft_safety.jsonl",
}

OUT_JSONL = REPO_ROOT / "data/quadrant_c_pipeline/c_source_authored_candidates_raw_v1.jsonl"
OUT_LOG_JSON = REPO_ROOT / "logs/3a2_candidate_universe.json"
OUT_LOG_MD = REPO_ROOT / "logs/3a2_candidate_universe.md"


# ── hashing / normalization (reproduced verbatim from src/diagnostics/check_leakage.py
#    and src/data_pipeline/build_secondary_c3_c4.py - NOT a new scheme) ─────────────────
def normalize(text: str) -> str:
    return " ".join(text.strip().lower().split())


def stripped_sha256(text: str) -> str:
    """Matches the exact convention already used for source_prompt_sha256 in
    data/quadrant_c_pipeline/candidate_records_v2.jsonl (sha256 of .strip()'d text,
    verified byte-for-byte against that file's recorded hash before writing this)."""
    return hashlib.sha256(text.strip().encode("utf-8")).hexdigest()


def normalized_sha256(text: str) -> str:
    return hashlib.sha256(normalize(text).encode("utf-8")).hexdigest()


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def stable_record_id(prefix: str, text: str) -> str:
    """Deterministic, content-derived, independent of row position - two
    runs against the same pinned source content always produce the same id."""
    h = hashlib.sha256(text.strip().encode("utf-8")).hexdigest()[:16]
    return f"CSA_{prefix}_{h}"


# ── Step 1: verify inputs ───────────────────────────────────────────────────────────────
def verify_source_file(path: Path, expected_sha256: str, label: str) -> str:
    if not path.exists():
        raise SystemExit(
            f"FAIL CLOSED: {label} source file missing at {path}. "
            f"Reacquire from the pinned revision recorded in logs/3a1c_handoff.json "
            f"before running this build - do not invent or substitute content."
        )
    actual = file_sha256(path)
    if actual != expected_sha256:
        raise SystemExit(
            f"FAIL CLOSED: {label} source file hash mismatch.\n"
            f"  path:     {path}\n"
            f"  expected: {expected_sha256}\n"
            f"  actual:   {actual}\n"
            f"Refusing to build candidates from unverified content."
        )
    return actual


# ── Step 2/3/8/9: per-record construction ──────────────────────────────────────────────
def classify_and_flag(text: str):
    result = classify_prompt_like_field(text)
    standalone = "complete" if result.label == "complete_user_facing_prompt" else "excluded_not_standalone"
    low_confidence = "low-confidence" in result.reason
    return result.label, result.reason, standalone, low_confidence


def build_strongreject_records(sha256_of_file: str):
    records = []
    with open(STRONGREJECT_CSV, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    for idx, row in enumerate(rows):
        text = row["forbidden_prompt"]
        upstream_source = row["source"].strip()
        label, reason, standalone, low_conf = classify_and_flag(text)

        if upstream_source == "custom":
            provenance_class = "custom"
            provenance_notes = "StrongREJECT curator-authored ('source' column value: 'custom')."
        else:
            provenance_class = "upstream-derived"
            provenance_notes = (
                f"StrongREJECT 'source' column records this row as drawn from '{upstream_source}'. "
                "StrongREJECT (Souly et al. 2024) is the strict-primary source approved by 3A1C policy; "
                f"'{upstream_source}' itself is not independently verified against its own upstream file in "
                "this milestone. Sub-source disjointness verification for non-'custom' StrongREJECT rows "
                "remains open per logs/3a1c_handoff.json (non_blocking_issues) and is not resolved here."
            )
            if upstream_source == "AdvBench":
                provenance_notes += (
                    " NOTE: AdvBench is itself an explicitly disallowed strict-primary input for this "
                    "milestone (see logs/3a1c_handoff.json explicitly_disallowed_3a2_strict_primary_inputs). "
                    "This row is retained as a StrongREJECT-sourced record (StrongREJECT is the directly "
                    "approved source and did the curation/selection), but the upstream lineage is flagged "
                    "explicitly rather than silently treated as StrongREJECT-custom text."
                )

        records.append({
            "record_id": stable_record_id("SR", text),
            "c_construction": "c_source_authored",
            "source_dataset": "StrongREJECT",
            "source_repository": STRONGREJECT_REPOSITORY,
            "source_file": STRONGREJECT_SOURCE_FILE,
            "source_revision": STRONGREJECT_REVISION,
            "source_file_sha256": sha256_of_file,
            "source_row_index": idx,
            "original_source_row_id": None,
            "prompt_text": text,
            "prompt_sha256": stripped_sha256(text),
            "prompt_normalized_sha256": normalized_sha256(text),
            "source_topic_category": row["category"].strip(),
            "source_request_type_category": None,
            "harm_area": None,
            "project_category": None,
            "upstream_provenance_detail": upstream_source,
            "provenance_class": provenance_class,
            "provenance_notes": provenance_notes,
            "structural_classifier_label": label,
            "structural_classifier_reason": reason,
            "structural_classifier_version": "classify_source_provenance.py (3A1A)",
            "structural_classifier_low_confidence": low_conf,
            "standalone_status": standalone,
        })
    return records


def build_simplesafetytests_records(sha256_of_file: str):
    records = []
    with open(SIMPLESAFETYTESTS_CSV, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    for idx, row in enumerate(rows):
        text = row["prompts_final"]
        label, reason, standalone, low_conf = classify_and_flag(text)

        records.append({
            "record_id": stable_record_id("SST", text),
            "c_construction": "c_source_authored",
            "source_dataset": "SimpleSafetyTests",
            "source_repository": SIMPLESAFETYTESTS_REPOSITORY,
            "source_file": SIMPLESAFETYTESTS_SOURCE_FILE,
            "source_revision": SIMPLESAFETYTESTS_REVISION,
            "source_file_sha256": sha256_of_file,
            "source_row_index": idx,
            "original_source_row_id": row["id"].strip(),
            "prompt_text": text,
            "prompt_sha256": stripped_sha256(text),
            "prompt_normalized_sha256": normalized_sha256(text),
            "source_topic_category": row["harm_area"].strip(),
            "source_request_type_category": row["category"].strip(),
            "harm_area": row["harm_area"].strip(),
            "project_category": None,
            "upstream_provenance_detail": None,
            "provenance_class": "curated",
            "provenance_notes": (
                "SimpleSafetyTests is described here as an external curated dataset record. Per 3A2-1 "
                "Step 8, individual human authorship of this specific row is NOT independently verified "
                "and is not claimed."
            ),
            "structural_classifier_label": label,
            "structural_classifier_reason": reason,
            "structural_classifier_version": "classify_source_provenance.py (3A1A)",
            "structural_classifier_low_confidence": low_conf,
            "standalone_status": standalone,
        })
    return records


# ── Step 4/5: exact + normalized duplicate detection ────────────────────────────────────
def annotate_duplicates(records):
    """First occurrence (source order: StrongREJECT then SimpleSafetyTests, original row
    order within each) becomes canonical. Later occurrences are marked duplicate but never
    removed from the file - provenance is preserved for every row per Step 4/5."""
    exact_seen = {}
    norm_seen = {}
    for r in records:
        r["exact_duplicate_status"] = "unique"
        r["exact_duplicate_canonical_record_id"] = None
        r["normalized_duplicate_status"] = "unique"
        r["normalized_duplicate_canonical_record_id"] = None

    for r in records:
        sh = r["prompt_sha256"]
        if sh in exact_seen:
            r["exact_duplicate_status"] = "duplicate"
            r["exact_duplicate_canonical_record_id"] = exact_seen[sh]
        else:
            exact_seen[sh] = r["record_id"]

    # Normalized-duplicate pass only meaningful for rows not already an exact duplicate
    # (an exact duplicate is trivially also a normalized duplicate - don't double-report).
    for r in records:
        if r["exact_duplicate_status"] == "duplicate":
            r["normalized_duplicate_status"] = "not_applicable_already_exact_duplicate"
            continue
        nh = r["prompt_normalized_sha256"]
        if nh in norm_seen:
            r["normalized_duplicate_status"] = "duplicate"
            r["normalized_duplicate_canonical_record_id"] = norm_seen[nh]
        else:
            norm_seen[nh] = r["record_id"]

    return records


# ── Step 6/7: overlap against C-paired (all 155) and Quadrant A ────────────────────────
def load_cpaired_hashes():
    if not CPAIRED_PATH.exists():
        raise SystemExit(f"FAIL CLOSED: required C-paired pool missing at {CPAIRED_PATH}")
    rows = [json.loads(line) for line in open(CPAIRED_PATH, encoding="utf-8")]
    if len(rows) != 155:
        raise SystemExit(
            f"FAIL CLOSED: expected the full 155-row C-paired pool per Step 6's hard "
            f"requirement, found {len(rows)} rows in {CPAIRED_PATH}."
        )
    exact = {}
    norm = {}
    for r in rows:
        text = r["source_prompt"]
        exact[stripped_sha256(text)] = r["record_id"]
        norm[normalized_sha256(text)] = r["record_id"]
    return exact, norm, len(rows)


def load_quadrant_a_hashes():
    if not EVAL_SET_PATH.exists():
        raise SystemExit(f"FAIL CLOSED: frozen eval set missing at {EVAL_SET_PATH}")
    rows = [json.loads(line) for line in open(EVAL_SET_PATH, encoding="utf-8")]
    qa_prompts = [r["prompt"] for r in rows if r.get("quadrant") == "A"]
    exact = {stripped_sha256(t) for t in qa_prompts}
    norm = {normalized_sha256(t) for t in qa_prompts}
    return exact, norm, len(qa_prompts)


def annotate_overlaps(records):
    cpaired_exact, cpaired_norm, cpaired_n = load_cpaired_hashes()
    qa_exact, qa_norm, qa_n = load_quadrant_a_hashes()

    for r in records:
        sh, nh = r["prompt_sha256"], r["prompt_normalized_sha256"]

        if sh in cpaired_exact:
            r["c_paired_overlap"] = True
            r["c_paired_overlap_match_type"] = "exact"
            r["c_paired_overlap_match_record_ids"] = [cpaired_exact[sh]]
        elif nh in cpaired_norm:
            r["c_paired_overlap"] = True
            r["c_paired_overlap_match_type"] = "normalized"
            r["c_paired_overlap_match_record_ids"] = [cpaired_norm[nh]]
        else:
            r["c_paired_overlap"] = False
            r["c_paired_overlap_match_type"] = None
            r["c_paired_overlap_match_record_ids"] = []

        if sh in qa_exact:
            r["quadrant_a_overlap"] = True
            r["quadrant_a_overlap_match_type"] = "exact"
        elif nh in qa_norm:
            r["quadrant_a_overlap"] = True
            r["quadrant_a_overlap_match_type"] = "normalized"
        else:
            r["quadrant_a_overlap"] = False
            r["quadrant_a_overlap_match_type"] = None

    return records, cpaired_n, qa_n


# ── Step 10: contamination (exact-only; near-dup explicitly unknown) ───────────────────
def annotate_contamination(records):
    train_normed_sets = {}
    for label, path in TRAINING_FILES_TO_CHECK.items():
        if not path.exists():
            train_normed_sets[label] = None  # unknown - file itself missing
            continue
        normed = set()
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                if "prompt" in row:
                    normed.add(normalize(row["prompt"]))
        train_normed_sets[label] = normed

    for r in records:
        nh_text = normalize(r["prompt_text"])
        exact_status = {}
        for label, normed in train_normed_sets.items():
            if normed is None:
                exact_status[label] = "unknown_training_file_missing"
            elif nh_text in normed:
                exact_status[label] = "matched"
            else:
                exact_status[label] = "clean"
        r["contamination_exact_status"] = exact_status
        r["contamination_near_status"] = (
            "unknown_embedding_model_unavailable_no_network_path_to_huggingface_co"
        )
    return records


# ── final eligibility roll-up (Steps 3/4/5/6/7 only - contamination is informational
#    per Step 10's explicit "prepare for the next step" scoping) ───────────────────────
def compute_eligibility(records):
    for r in records:
        reasons = []
        if r["standalone_status"] != "complete":
            reasons.append(
                f"not_standalone_user_facing_request:{r['structural_classifier_label']}"
            )
        if r["exact_duplicate_status"] == "duplicate":
            reasons.append(
                f"exact_duplicate_of:{r['exact_duplicate_canonical_record_id']}"
            )
        if r["normalized_duplicate_status"] == "duplicate":
            reasons.append(
                f"normalized_duplicate_of:{r['normalized_duplicate_canonical_record_id']}"
            )
        if r["c_paired_overlap"]:
            reasons.append(
                "overlaps_c_paired_pool:" + ",".join(r["c_paired_overlap_match_record_ids"])
            )
        if r["quadrant_a_overlap"]:
            reasons.append("overlaps_quadrant_a")

        r["exclusion_reasons"] = reasons
        r["candidate_universe_status"] = "excluded" if reasons else "eligible_for_3a3"
    return records


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-jsonl", default=str(OUT_JSONL))
    parser.add_argument("--out-log-json", default=str(OUT_LOG_JSON))
    parser.add_argument("--out-log-md", default=str(OUT_LOG_MD))
    args = parser.parse_args()

    sr_hash = verify_source_file(STRONGREJECT_CSV, STRONGREJECT_EXPECTED_SHA256, "StrongREJECT")
    sst_hash = verify_source_file(SIMPLESAFETYTESTS_CSV, SIMPLESAFETYTESTS_EXPECTED_SHA256, "SimpleSafetyTests")

    records = build_strongreject_records(sr_hash) + build_simplesafetytests_records(sst_hash)
    records = annotate_duplicates(records)
    records, cpaired_n, qa_n = annotate_overlaps(records)
    records = annotate_contamination(records)
    records = compute_eligibility(records)

    out_jsonl = Path(args.out_jsonl)
    out_jsonl.parent.mkdir(parents=True, exist_ok=True)
    with open(out_jsonl, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    artifact_sha256 = file_sha256(out_jsonl)

    by_source = defaultdict(list)
    for r in records:
        by_source[r["source_dataset"]].append(r)

    def counter(field_records, field):
        c = defaultdict(int)
        for r in field_records:
            c[r[field]] += 1
        return dict(c)

    summary = {
        "milestone": "3A2-1",
        "objective": "Construct the reproducible C-source-authored candidate universe "
                     "from StrongREJECT and SimpleSafetyTests only.",
        "input_source_hashes": {
            "StrongREJECT": {
                "path": str(STRONGREJECT_CSV.relative_to(REPO_ROOT)),
                "sha256": sr_hash,
                "expected_sha256": STRONGREJECT_EXPECTED_SHA256,
                "verified": sr_hash == STRONGREJECT_EXPECTED_SHA256,
                "row_count": len(by_source["StrongREJECT"]),
            },
            "SimpleSafetyTests": {
                "path": str(SIMPLESAFETYTESTS_CSV.relative_to(REPO_ROOT)),
                "sha256": sst_hash,
                "expected_sha256": SIMPLESAFETYTESTS_EXPECTED_SHA256,
                "verified": sst_hash == SIMPLESAFETYTESTS_EXPECTED_SHA256,
                "row_count": len(by_source["SimpleSafetyTests"]),
            },
        },
        "reference_pools_checked": {
            "c_paired_pool_size_checked": cpaired_n,
            "c_paired_pool_note": "Full 155-row pool, not restricted to the 104 currently-live frozen rows (Step 6 hard requirement).",
            "quadrant_a_size_checked": qa_n,
        },
        "candidate_count_total": len(records),
        "candidate_count_by_source": {k: len(v) for k, v in by_source.items()},
        "eligible_for_3a3_count": sum(1 for r in records if r["candidate_universe_status"] == "eligible_for_3a3"),
        "excluded_count": sum(1 for r in records if r["candidate_universe_status"] == "excluded"),
        "standalone_status_breakdown": counter(records, "standalone_status"),
        "structural_classifier_label_breakdown": counter(records, "structural_classifier_label"),
        "structural_classifier_label_breakdown_by_source": {
            src: counter(rows, "structural_classifier_label") for src, rows in by_source.items()
        },
        "low_confidence_structural_flag_count": sum(1 for r in records if r["structural_classifier_low_confidence"]),
        "exact_duplicate_count": sum(1 for r in records if r["exact_duplicate_status"] == "duplicate"),
        "normalized_duplicate_count": sum(1 for r in records if r["normalized_duplicate_status"] == "duplicate"),
        "c_paired_overlap_count": sum(1 for r in records if r["c_paired_overlap"]),
        "c_paired_overlap_count_by_source": {
            src: sum(1 for r in rows if r["c_paired_overlap"]) for src, rows in by_source.items()
        },
        "quadrant_a_overlap_count": sum(1 for r in records if r["quadrant_a_overlap"]),
        "provenance_class_breakdown": counter(records, "provenance_class"),
        "strongreject_upstream_provenance_breakdown": counter(by_source["StrongREJECT"], "upstream_provenance_detail"),
        "advbench_flagged_upstream_rows": sum(
            1 for r in by_source["StrongREJECT"] if r["upstream_provenance_detail"] == "AdvBench"
        ),
        "contamination_exact_status_note": "Exact-text-only check against the four training files listed below; "
                                            "informational in 3A2-1, not used to drive candidate_universe_status "
                                            "(deferred to 3A3 per Step 10 scoping).",
        "contamination_training_files_checked": list(TRAINING_FILES_TO_CHECK.keys()),
        "contamination_exact_matches_found": sum(
            1 for r in records if any(v == "matched" for v in r["contamination_exact_status"].values())
        ),
        "contamination_near_status": "unknown_for_all_records - embedding model unavailable (no network path to huggingface.co in this sandbox)",
        "unresolved_from_prior_milestones_not_addressed_here": [
            "StrongREJECT sub-source disjointness verification for non-'custom' rows (92 rows total, "
            "25 of which are labeled 'AdvBench' upstream) - flagged per-record via provenance_notes, "
            "not independently re-verified against AdvBench's own file this milestone.",
            "SORRY-Bench acquisition and authoritative JBB-Behaviors dataset remain unresolved sources; "
            "neither is a strict-primary input so neither blocks 3A2.",
        ],
        "output_artifacts": {
            "candidate_universe_jsonl": {
                "path": str(out_jsonl.relative_to(REPO_ROOT)),
                "sha256": artifact_sha256,
                "row_count": len(records),
            }
        },
        "not_performed_this_milestone": [
            "near_duplicate_embedding_check",
            "fightin_words_scoring",
            "Q10_Q25_Q40_stratification",
            "review_queue_generation",
            "benchmark_freeze",
            "project_category_taxonomy_mapping",
        ],
        "next_milestone": "3A3 - candidate validation, near-duplicate, overlap, and contamination checks.",
    }

    out_log_json = Path(args.out_log_json)
    out_log_json.parent.mkdir(parents=True, exist_ok=True)
    with open(out_log_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    md_lines = [
        "# 3A2-1 — C-Source-Authored Candidate Universe",
        "",
        f"Total candidates constructed: **{summary['candidate_count_total']}** "
        f"(StrongREJECT: {summary['candidate_count_by_source'].get('StrongREJECT', 0)}, "
        f"SimpleSafetyTests: {summary['candidate_count_by_source'].get('SimpleSafetyTests', 0)})",
        "",
        f"- Eligible for 3A3: **{summary['eligible_for_3a3_count']}**",
        f"- Excluded (with reason, provenance preserved): **{summary['excluded_count']}**",
        "",
        "## Input source verification",
        "",
        "| Source | Rows | SHA-256 verified |",
        "|---|---|---|",
        f"| StrongREJECT | {summary['input_source_hashes']['StrongREJECT']['row_count']} | "
        f"{summary['input_source_hashes']['StrongREJECT']['verified']} |",
        f"| SimpleSafetyTests | {summary['input_source_hashes']['SimpleSafetyTests']['row_count']} | "
        f"{summary['input_source_hashes']['SimpleSafetyTests']['verified']} |",
        "",
        "## Structural completeness (Step 3, classify_source_provenance.py)",
        "",
        f"Breakdown across both sources: `{json.dumps(summary['structural_classifier_label_breakdown'])}`",
        "",
        f"Low-confidence (ambiguous imperative fallback, kept provisional per Step 9): "
        f"{summary['low_confidence_structural_flag_count']}",
        "",
        "## Duplicate / overlap checks (Steps 4-7)",
        "",
        f"- Exact duplicates (within/across the two sources): {summary['exact_duplicate_count']}",
        f"- Normalized-only duplicates: {summary['normalized_duplicate_count']}",
        f"- C-paired pool overlap (checked against the full "
        f"{summary['reference_pools_checked']['c_paired_pool_size_checked']}-row pool, not just the "
        f"live 104): {summary['c_paired_overlap_count']} "
        f"(`{json.dumps(summary['c_paired_overlap_count_by_source'])}`)",
        f"- Quadrant-A overlap (checked against "
        f"{summary['reference_pools_checked']['quadrant_a_size_checked']} rows): "
        f"{summary['quadrant_a_overlap_count']}",
        "",
        "## Provenance (Step 8)",
        "",
        f"Provenance class breakdown: `{json.dumps(summary['provenance_class_breakdown'])}`",
        "",
        f"StrongREJECT upstream 'source' column breakdown: "
        f"`{json.dumps(summary['strongreject_upstream_provenance_breakdown'])}`",
        "",
        f"Rows whose upstream provenance is 'AdvBench' (AdvBench is itself an excluded strict-primary "
        f"input - flagged per-record, not excluded, sub-source disjointness unresolved): "
        f"{summary['advbench_flagged_upstream_rows']}",
        "",
        "## Contamination (Step 10 - informational only this milestone)",
        "",
        f"Exact-text matches against training files "
        f"({', '.join(summary['contamination_training_files_checked'])}): "
        f"{summary['contamination_exact_matches_found']}",
        "",
        f"Near-duplicate contamination status: {summary['contamination_near_status']}",
        "",
        "## Content-sensitivity note",
        "",
        "SimpleSafetyTests includes a 'Child safety' harm_area (20/100 rows) and a "
        "'Suicide, Self-Harm, and Eating Disorders' harm_area (20/100 rows). Consistent with the "
        "convention already established in logs/3a1b_source_inventory.md, no example prompt text from "
        "any harm_area is reproduced in this report - counts and category labels only. Full exact text "
        "is preserved only in the structured JSONL data artifact, as required for reproducibility.",
        "",
        "## Not performed this milestone",
        "",
        *[f"- {item}" for item in summary["not_performed_this_milestone"]],
        "",
        f"## Output artifact",
        "",
        f"`{summary['output_artifacts']['candidate_universe_jsonl']['path']}` "
        f"(sha256 `{summary['output_artifacts']['candidate_universe_jsonl']['sha256']}`, "
        f"{summary['output_artifacts']['candidate_universe_jsonl']['row_count']} rows)",
        "",
        f"**Next milestone:** {summary['next_milestone']}",
        "",
    ]
    out_log_md = Path(args.out_log_md)
    out_log_md.parent.mkdir(parents=True, exist_ok=True)
    with open(out_log_md, "w", encoding="utf-8") as f:
        f.write("\n".join(md_lines))

    print(f"Wrote {len(records)} candidate records to {out_jsonl}")
    print(f"  eligible_for_3a3: {summary['eligible_for_3a3_count']}")
    print(f"  excluded: {summary['excluded_count']}")
    print(f"Wrote summary to {out_log_json} and {out_log_md}")


if __name__ == "__main__":
    main()
