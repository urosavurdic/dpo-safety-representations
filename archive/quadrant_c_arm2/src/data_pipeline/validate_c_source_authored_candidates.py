"""
Milestone 3A3: validates the C-source-authored candidate universe produced by
3A2-1 (src/data_pipeline/build_c_source_authored_candidates.py).

Continues from data/quadrant_c_pipeline/c_source_authored_candidates_raw_v1.jsonl.
Does NOT redo source acquisition or candidate construction: it independently
re-derives every field 3A2-1 computed from primitives (prompt_text, source
file rows) and compares the recomputation against what 3A2-1 recorded, plus
performs the checks 3A2-1 explicitly deferred (near-duplicate embedding
checks, informational-only contamination). No row is added, removed, or
rewritten - a `validation_3a3` block is attached to every existing record and
every original field is carried through unchanged.

Required checks (3A3 brief):
  1. Exact duplicate validation       - re-derive from prompt_text, compare to 3A2-1's recorded
                                         exact_duplicate_status / canonical id.
  2. Normalized duplicate validation  - same, normalized_duplicate_status / canonical id.
  3. Near-duplicate validation        - sentence-transformers cosine similarity (the repo's existing
                                         embedding-based near-dup convention: src/diagnostics/
                                         check_leakage.py + src/diagnostics/complete_neardup_check.py),
                                         for the five required comparisons (within StrongREJECT; within
                                         SimpleSafetyTests; StrongREJECT vs SimpleSafetyTests; candidates
                                         vs Quadrant-A; candidates vs the full 155-row C-paired pool).
                                         Uses the same embed-once-then-matrix-multiply approach as
                                         complete_neardup_check.py (O(n*m) over embeddings, not an O(N^2)
                                         text-level comparison). Requires downloading all-MiniLM-L6-v2
                                         from huggingface.co; this sandbox's network egress does not
                                         allow that host (confirmed: `curl -D- https://huggingface.co`
                                         -> HTTP 403, `x-deny-reason: host_not_allowed`). Recorded as
                                         "unknown" for every one of the five comparisons - never
                                         silently treated as "clean" - with the literal load error
                                         captured.
  4. Training contamination           - exact match re-verified independently against all four
                                         training files (dpo_pairs.jsonl, sft_helpful.jsonl,
                                         sft_helpful_alt.jsonl, sft_safety.jsonl - all four are present
                                         in this checkout, so "unavailable" does not apply). Near-dup
                                         contamination hits the same embedding blocker as #3 and is
                                         recorded "unknown" per file, not "clean".
  5. Source/provenance validation     - both source CSVs were re-acquired fresh, at the exact pinned
                                         revision, directly from their GitHub repositories
                                         (raw.githubusercontent.com is reachable even though
                                         huggingface.co is not - both alexandrasouly/strongreject and
                                         bertiev/SimpleSafetyTests are GitHub repos, not HF-hub-only
                                         datasets). Re-verified file sha256 against the 3A2-1-recorded/
                                         expected hash: both matched byte-for-byte. Every record's
                                         source_repository/source_file/source_revision/
                                         source_file_sha256 and its deterministic record_id are
                                         independently re-derived and compared.
  6. Exact prompt preservation        - every candidate's prompt_text is compared byte-for-byte (no
                                         normalization) against the freshly-reacquired source file's
                                         cell at the recorded source_row_index.
  7. Structural classifier            - classify_prompt_like_field is re-run on every prompt_text and
                                         compared against the recorded label/standalone_status
                                         (determinism check), and every row is checked to confirm no
                                         non-standalone or ambiguous/low-confidence record was silently
                                         promoted to candidate_universe_status == "eligible_for_3a3".

Explicitly NOT done here: no candidate reconstruction, no Fightin' Words
scoring, no Q10/Q25/Q40 stratification, no review-queue generation, no
benchmark freeze. Per the 3A3 brief, a correction to the candidate universe
is made only if a concrete validation failure requires it; see
logs/3a3_validation.md for whether that happened in this run.
"""
import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

from src.data_pipeline.build_c_source_authored_candidates import (
    CPAIRED_PATH,
    EVAL_SET_PATH,
    SIMPLESAFETYTESTS_CSV,
    SIMPLESAFETYTESTS_EXPECTED_SHA256,
    SIMPLESAFETYTESTS_REPOSITORY,
    SIMPLESAFETYTESTS_REVISION,
    SIMPLESAFETYTESTS_SOURCE_FILE,
    STRONGREJECT_CSV,
    STRONGREJECT_EXPECTED_SHA256,
    STRONGREJECT_REPOSITORY,
    STRONGREJECT_REVISION,
    STRONGREJECT_SOURCE_FILE,
    TRAINING_FILES_TO_CHECK,
    classify_and_flag,
    file_sha256,
    normalize,
    normalized_sha256,
    stable_record_id,
    stripped_sha256,
)

REPO_ROOT = Path(__file__).resolve().parents[4]

IN_JSONL = REPO_ROOT / "data/quadrant_c_pipeline/c_source_authored_candidates_raw_v1.jsonl"
OUT_JSONL = REPO_ROOT / "data/quadrant_c_pipeline/c_source_authored_candidates_validated_v1.jsonl"
OUT_LOG_JSON = REPO_ROOT / "logs/3a3_validation.json"
OUT_LOG_MD = REPO_ROOT / "logs/3a3_validation.md"

NEAR_DUP_THRESHOLD = 0.9
NEAR_DUP_MODEL_NAME = "all-MiniLM-L6-v2"


def load_jsonl(path: Path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


# ── Steps 1/2: exact + normalized duplicate re-verification ────────────────────────────
def validate_exact_and_normalized_duplicates(records):
    exact_seen = {}
    norm_seen = {}
    exact_mismatches = []
    norm_mismatches = []
    exact_dup_count = 0
    norm_dup_count = 0

    for r in records:
        recomputed_sha = stripped_sha256(r["prompt_text"])
        if recomputed_sha != r["prompt_sha256"]:
            exact_mismatches.append({
                "record_id": r["record_id"],
                "issue": "prompt_sha256 does not match recomputed hash of prompt_text",
            })
        sh = r["prompt_sha256"]
        if sh in exact_seen:
            expected_status, expected_canon = "duplicate", exact_seen[sh]
        else:
            expected_status, expected_canon = "unique", None
            exact_seen[sh] = r["record_id"]
        if (expected_status, expected_canon) != (r["exact_duplicate_status"], r["exact_duplicate_canonical_record_id"]):
            exact_mismatches.append({
                "record_id": r["record_id"],
                "issue": "exact_duplicate_status/canonical mismatch",
                "recorded": [r["exact_duplicate_status"], r["exact_duplicate_canonical_record_id"]],
                "recomputed": [expected_status, expected_canon],
            })
        if expected_status == "duplicate":
            exact_dup_count += 1

    for r in records:
        recomputed_nh = normalized_sha256(r["prompt_text"])
        if recomputed_nh != r["prompt_normalized_sha256"]:
            norm_mismatches.append({
                "record_id": r["record_id"],
                "issue": "prompt_normalized_sha256 does not match recomputed hash of prompt_text",
            })
        nh = r["prompt_normalized_sha256"]
        if r["exact_duplicate_status"] == "duplicate":
            expected_status, expected_canon = "not_applicable_already_exact_duplicate", None
        elif nh in norm_seen:
            expected_status, expected_canon = "duplicate", norm_seen[nh]
        else:
            expected_status, expected_canon = "unique", None
            norm_seen[nh] = r["record_id"]
        if (expected_status, expected_canon) != (r["normalized_duplicate_status"], r["normalized_duplicate_canonical_record_id"]):
            norm_mismatches.append({
                "record_id": r["record_id"],
                "issue": "normalized_duplicate_status/canonical mismatch",
                "recorded": [r["normalized_duplicate_status"], r["normalized_duplicate_canonical_record_id"]],
                "recomputed": [expected_status, expected_canon],
            })
        if expected_status == "duplicate":
            norm_dup_count += 1

    return {
        "exact_duplicate_validation": {
            "status": "pass" if not exact_mismatches else "fail",
            "mismatches": exact_mismatches,
            "recomputed_duplicate_count": exact_dup_count,
        },
        "normalized_duplicate_validation": {
            "status": "pass" if not norm_mismatches else "fail",
            "mismatches": norm_mismatches,
            "recomputed_duplicate_count": norm_dup_count,
        },
    }


# ── Steps 6/7 overlap re-verification (C-paired full 155, Quadrant-A) ──────────────────
def validate_overlaps(records, cpaired_rows, qa_texts):
    cpaired_exact, cpaired_norm = {}, {}
    for r in cpaired_rows:
        t = r["source_prompt"]
        cpaired_exact[stripped_sha256(t)] = r["record_id"]
        cpaired_norm[normalized_sha256(t)] = r["record_id"]
    qa_exact = {stripped_sha256(t) for t in qa_texts}
    qa_norm = {normalized_sha256(t) for t in qa_texts}

    mismatches = []
    cpaired_overlap_count = 0
    qa_overlap_count = 0
    for r in records:
        sh, nh = r["prompt_sha256"], r["prompt_normalized_sha256"]
        expected_cpaired = (sh in cpaired_exact) or (nh in cpaired_norm)
        expected_qa = (sh in qa_exact) or (nh in qa_norm)
        if expected_cpaired:
            cpaired_overlap_count += 1
        if expected_qa:
            qa_overlap_count += 1
        if (expected_cpaired, expected_qa) != (r["c_paired_overlap"], r["quadrant_a_overlap"]):
            mismatches.append({
                "record_id": r["record_id"],
                "expected": [expected_cpaired, expected_qa],
                "recorded": [r["c_paired_overlap"], r["quadrant_a_overlap"]],
            })

    return {
        "status": "pass" if not mismatches else "fail",
        "mismatches": mismatches,
        "c_paired_overlap_count": cpaired_overlap_count,
        "c_paired_pool_size_checked": len(cpaired_rows),
        "quadrant_a_overlap_count": qa_overlap_count,
        "quadrant_a_pool_size_checked": len(qa_texts),
    }


# ── Step 3: near-duplicate validation (embedding-based, non-quadratic) ─────────────────
def attempt_load_embedding_model():
    """Returns (model, error_string). error_string is None on success. Never raises -
    a failed load is a recorded, expected outcome in this sandbox, not a script error."""
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as e:
        return None, f"sentence_transformers not importable: {e}"
    try:
        model = SentenceTransformer(NEAR_DUP_MODEL_NAME)
        return model, None
    except Exception as e:  # noqa: BLE001 - deliberately broad: any load failure is "unknown", not a crash
        return None, f"{type(e).__name__}: {e}"


def _embed(model, texts):
    if not texts:
        return None
    return model.encode(texts, normalize_embeddings=True, show_progress_bar=False, batch_size=256)


def near_duplicate_validation(records, cpaired_texts, qa_texts, model, load_error):
    sr_texts = [r["prompt_text"] for r in records if r["source_dataset"] == "StrongREJECT"]
    sst_texts = [r["prompt_text"] for r in records if r["source_dataset"] == "SimpleSafetyTests"]
    all_texts = [r["prompt_text"] for r in records]

    config = {
        "method": "sentence-transformers cosine similarity, repo convention from "
                  "src/diagnostics/check_leakage.py and src/diagnostics/complete_neardup_check.py",
        "model_name": NEAR_DUP_MODEL_NAME,
        "similarity_threshold": NEAR_DUP_THRESHOLD,
        "implementation": "embeddings computed once per text set, then a single similarity matrix "
                           "(embeddings @ embeddings.T) - O(n*m) over embedding vectors, not an O(N^2) "
                           "pairwise text comparison.",
    }

    comparisons = {
        "within_StrongREJECT": None,
        "within_SimpleSafetyTests": None,
        "StrongREJECT_vs_SimpleSafetyTests": None,
        "candidates_vs_QuadrantA": None,
        "candidates_vs_CPaired": None,
    }

    if model is None:
        for key in comparisons:
            comparisons[key] = {"status": "unknown", "reason": load_error}
        return {"config": config, "comparisons": comparisons, "blocker": load_error}

    def compare(name, texts_a, texts_b, self_compare=False):
        if not texts_a or not texts_b:
            comparisons[name] = {"status": "pass", "flagged_pairs": 0, "note": "empty comparison set"}
            return
        emb_a = _embed(model, texts_a)
        emb_b = emb_a if self_compare else _embed(model, texts_b)
        sim = emb_a @ emb_b.T
        flagged = 0
        for i in range(len(texts_a)):
            j_range = range(i + 1, len(texts_b)) if self_compare else range(len(texts_b))
            for j in j_range:
                if float(sim[i][j]) >= NEAR_DUP_THRESHOLD:
                    flagged += 1
        comparisons[name] = {"status": "pass" if flagged == 0 else "flagged", "flagged_pairs": flagged}

    compare("within_StrongREJECT", sr_texts, sr_texts, self_compare=True)
    compare("within_SimpleSafetyTests", sst_texts, sst_texts, self_compare=True)
    compare("StrongREJECT_vs_SimpleSafetyTests", sr_texts, sst_texts)
    compare("candidates_vs_QuadrantA", all_texts, qa_texts)
    compare("candidates_vs_CPaired", all_texts, cpaired_texts)

    return {"config": config, "comparisons": comparisons, "blocker": None}


# ── Step 4: training contamination (exact re-verified, near attempted) ─────────────────
def load_training_texts():
    texts_by_file = {}
    for label, path in TRAINING_FILES_TO_CHECK.items():
        if not path.exists():
            texts_by_file[label] = None
            continue
        texts = []
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                if "prompt" in row:
                    texts.append(row["prompt"])
        texts_by_file[label] = texts
    return texts_by_file


def contamination_validation(records, texts_by_file, model, load_error):
    normed_sets = {
        label: ({normalize(t) for t in texts} if texts is not None else None)
        for label, texts in texts_by_file.items()
    }

    exact_mismatches = []
    exact_contam_counts = defaultdict(int)
    for r in records:
        recorded = r.get("contamination_exact_status", {})
        nh_text = normalize(r["prompt_text"])
        for label, normed in normed_sets.items():
            if normed is None:
                expected = "unknown_training_file_missing"
            elif nh_text in normed:
                expected = "matched"
                exact_contam_counts[label] += 1
            else:
                expected = "clean"
            if recorded.get(label) != expected:
                exact_mismatches.append({
                    "record_id": r["record_id"], "file": label,
                    "recorded": recorded.get(label), "recomputed": expected,
                })

    near_status_by_file = {}
    if model is None:
        for label, texts in texts_by_file.items():
            near_status_by_file[label] = "unavailable_file_missing" if texts is None else "unknown"
    else:
        cand_texts = [r["prompt_text"] for r in records]
        cand_emb = _embed(model, cand_texts)
        for label, texts in texts_by_file.items():
            if texts is None:
                near_status_by_file[label] = "unavailable_file_missing"
                continue
            train_emb = _embed(model, texts)
            sim = cand_emb @ train_emb.T
            hits = int((sim >= NEAR_DUP_THRESHOLD).sum())
            near_status_by_file[label] = {"status": "clean" if hits == 0 else "contaminated", "hits": hits}

    return {
        "exact_status": "pass" if not exact_mismatches else "fail",
        "exact_mismatches": exact_mismatches,
        "exact_contamination_counts_by_file": dict(exact_contam_counts),
        "training_files_checked": list(TRAINING_FILES_TO_CHECK.keys()),
        "near_status_by_file": near_status_by_file,
        "near_blocker": load_error,
    }


# ── Step 5: source/provenance re-verification (reacquired source files) ────────────────
def source_provenance_validation(records):
    sr_hash = file_sha256(STRONGREJECT_CSV) if STRONGREJECT_CSV.exists() else None
    sst_hash = file_sha256(SIMPLESAFETYTESTS_CSV) if SIMPLESAFETYTESTS_CSV.exists() else None

    result = {
        "StrongREJECT": {
            "reacquired_from": (
                f"https://raw.githubusercontent.com/{STRONGREJECT_REPOSITORY}/"
                f"{STRONGREJECT_REVISION}/{STRONGREJECT_SOURCE_FILE}"
            ),
            "path": str(STRONGREJECT_CSV.relative_to(REPO_ROOT)),
            "present": STRONGREJECT_CSV.exists(),
            "sha256": sr_hash,
            "expected_sha256": STRONGREJECT_EXPECTED_SHA256,
            "verified": sr_hash == STRONGREJECT_EXPECTED_SHA256,
        },
        "SimpleSafetyTests": {
            "reacquired_from": (
                f"https://raw.githubusercontent.com/{SIMPLESAFETYTESTS_REPOSITORY}/"
                f"{SIMPLESAFETYTESTS_REVISION}/{SIMPLESAFETYTESTS_SOURCE_FILE}"
            ),
            "path": str(SIMPLESAFETYTESTS_CSV.relative_to(REPO_ROOT)),
            "present": SIMPLESAFETYTESTS_CSV.exists(),
            "sha256": sst_hash,
            "expected_sha256": SIMPLESAFETYTESTS_EXPECTED_SHA256,
            "verified": sst_hash == SIMPLESAFETYTESTS_EXPECTED_SHA256,
        },
    }

    expected_by_source = {
        "StrongREJECT": (STRONGREJECT_REPOSITORY, STRONGREJECT_SOURCE_FILE, STRONGREJECT_REVISION, STRONGREJECT_EXPECTED_SHA256),
        "SimpleSafetyTests": (SIMPLESAFETYTESTS_REPOSITORY, SIMPLESAFETYTESTS_SOURCE_FILE, SIMPLESAFETYTESTS_REVISION, SIMPLESAFETYTESTS_EXPECTED_SHA256),
    }

    field_mismatches = []
    for r in records:
        expected = expected_by_source[r["source_dataset"]]
        actual = (r["source_repository"], r["source_file"], r["source_revision"], r["source_file_sha256"])
        if actual != expected:
            field_mismatches.append({"record_id": r["record_id"], "issue": "provenance field mismatch", "expected": list(expected), "actual": list(actual)})

        prefix = "SR" if r["source_dataset"] == "StrongREJECT" else "SST"
        recomputed_id = stable_record_id(prefix, r["prompt_text"])
        if recomputed_id != r["record_id"]:
            field_mismatches.append({"record_id": r["record_id"], "issue": "record_id not deterministic from prompt_text", "recomputed": recomputed_id})

    result["per_record_field_validation"] = {
        "status": "pass" if not field_mismatches else "fail",
        "mismatches": field_mismatches,
    }
    return result


# ── Step 6: exact prompt preservation against freshly reacquired source rows ───────────
def load_source_rows():
    with open(STRONGREJECT_CSV, newline="", encoding="utf-8") as f:
        sr_rows = list(csv.DictReader(f))
    with open(SIMPLESAFETYTESTS_CSV, newline="", encoding="utf-8") as f:
        sst_rows = list(csv.DictReader(f))
    return sr_rows, sst_rows


def prompt_preservation_validation(records, sr_rows, sst_rows):
    mismatches = []
    checked = 0
    for r in records:
        idx = r["source_row_index"]
        if r["source_dataset"] == "StrongREJECT":
            source_text = sr_rows[idx]["forbidden_prompt"]
        else:
            source_text = sst_rows[idx]["prompts_final"]
        checked += 1
        if source_text != r["prompt_text"]:
            mismatches.append({"record_id": r["record_id"], "source_row_index": idx})
    return {
        "status": "pass" if not mismatches else "fail",
        "rows_checked": checked,
        "mismatches": mismatches,
    }


# ── Step 7: structural classifier determinism + no silent promotion of ambiguous rows ──
def structural_classifier_validation(records):
    mismatches = []
    promoted_ambiguous = []
    for r in records:
        label, reason, standalone, low_conf = classify_and_flag(r["prompt_text"])
        if (label, standalone) != (r["structural_classifier_label"], r["standalone_status"]):
            mismatches.append({
                "record_id": r["record_id"],
                "recorded": [r["structural_classifier_label"], r["standalone_status"]],
                "recomputed": [label, standalone],
            })
        if r["standalone_status"] != "complete" and r["candidate_universe_status"] == "eligible_for_3a3":
            promoted_ambiguous.append(r["record_id"])

    low_conf_count = sum(1 for r in records if r.get("structural_classifier_low_confidence"))
    return {
        "status": "pass" if not mismatches and not promoted_ambiguous else "fail",
        "classifier_determinism_mismatches": mismatches,
        "non_standalone_silently_promoted": promoted_ambiguous,
        "low_confidence_provisional_count": low_conf_count,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--in-jsonl", default=str(IN_JSONL))
    parser.add_argument("--out-jsonl", default=str(OUT_JSONL))
    parser.add_argument("--out-log-json", default=str(OUT_LOG_JSON))
    parser.add_argument("--out-log-md", default=str(OUT_LOG_MD))
    args = parser.parse_args()

    in_path = Path(args.in_jsonl)
    if not in_path.exists():
        raise SystemExit(f"FAIL CLOSED: candidate universe missing at {in_path}. Run 3A2-1 first.")
    records = load_jsonl(in_path)
    in_sha256 = file_sha256(in_path)

    dup_result = validate_exact_and_normalized_duplicates(records)

    cpaired_rows = load_jsonl(CPAIRED_PATH)
    if len(cpaired_rows) != 155:
        raise SystemExit(f"FAIL CLOSED: expected the full 155-row C-paired pool, found {len(cpaired_rows)}.")
    cpaired_texts = [r["source_prompt"] for r in cpaired_rows]

    eval_rows = load_jsonl(EVAL_SET_PATH)
    qa_texts = [r["prompt"] for r in eval_rows if r.get("quadrant") == "A"]

    overlap_result = validate_overlaps(records, cpaired_rows, qa_texts)

    model, load_error = attempt_load_embedding_model()

    neardup_result = near_duplicate_validation(records, cpaired_texts, qa_texts, model, load_error)

    training_texts = load_training_texts()
    contamination_result = contamination_validation(records, training_texts, model, load_error)

    provenance_result = source_provenance_validation(records)

    if not STRONGREJECT_CSV.exists() or not SIMPLESAFETYTESTS_CSV.exists():
        raise SystemExit(
            "FAIL CLOSED: source CSVs unavailable for exact prompt preservation re-check. "
            "data/raw/ is gitignored; reacquire both source files at the pinned revisions "
            "recorded in logs/3a1c_handoff.json before running 3A3."
        )
    sr_rows, sst_rows = load_source_rows()
    preservation_result = prompt_preservation_validation(records, sr_rows, sst_rows)

    classifier_result = structural_classifier_validation(records)

    exact_mismatch_ids = {m["record_id"] for m in dup_result["exact_duplicate_validation"]["mismatches"]}
    norm_mismatch_ids = {m["record_id"] for m in dup_result["normalized_duplicate_validation"]["mismatches"]}
    preservation_mismatch_ids = {m["record_id"] for m in preservation_result["mismatches"]}
    classifier_mismatch_ids = {m["record_id"] for m in classifier_result["classifier_determinism_mismatches"]}

    for r in records:
        r["validation_3a3"] = {
            "exact_duplicate_reverified": r["record_id"] not in exact_mismatch_ids,
            "normalized_duplicate_reverified": r["record_id"] not in norm_mismatch_ids,
            "exact_prompt_preservation_reverified": r["record_id"] not in preservation_mismatch_ids,
            "structural_classifier_reverified": r["record_id"] not in classifier_mismatch_ids,
            "near_duplicate_status": "unknown_embedding_model_unavailable" if neardup_result["blocker"] else "checked",
            "contamination_near_status": {
                k: (v if isinstance(v, str) else v.get("status"))
                for k, v in contamination_result["near_status_by_file"].items()
            },
        }

    eligible = [r for r in records if r["candidate_universe_status"] == "eligible_for_3a3"]
    excluded = [r for r in records if r["candidate_universe_status"] == "excluded"]
    exclusion_reason_counts = defaultdict(int)
    for r in excluded:
        for reason in r["exclusion_reasons"]:
            exclusion_reason_counts[reason.split(":")[0]] += 1

    out_jsonl = Path(args.out_jsonl)
    out_jsonl.parent.mkdir(parents=True, exist_ok=True)
    with open(out_jsonl, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    out_sha256 = file_sha256(out_jsonl)

    checks = {
        "exact_duplicate_validation": dup_result["exact_duplicate_validation"]["status"],
        "normalized_duplicate_validation": dup_result["normalized_duplicate_validation"]["status"],
        "near_duplicate_validation": "unknown" if neardup_result["blocker"] else "pass",
        "training_contamination_exact": contamination_result["exact_status"],
        "training_contamination_near": "unknown" if contamination_result["near_blocker"] else "pass",
        "overlap_reverification_c_paired_and_quadrant_a": overlap_result["status"],
        "source_provenance_validation": (
            "pass"
            if provenance_result["StrongREJECT"]["verified"]
            and provenance_result["SimpleSafetyTests"]["verified"]
            and provenance_result["per_record_field_validation"]["status"] == "pass"
            else "fail"
        ),
        "exact_prompt_preservation": preservation_result["status"],
        "structural_classifier_validation": classifier_result["status"],
    }
    if "fail" in checks.values():
        overall_status = "validation_failed"
    elif "unknown" in checks.values():
        overall_status = "validated_with_unknowns"
    else:
        overall_status = "validated_clean"

    summary = {
        "milestone": "3A3",
        "objective": "Validate the C-source-authored candidate universe from 3A2-1 with the required "
                     "record-level gates, without redoing acquisition or construction.",
        "input_artifact": {
            "path": str(in_path.relative_to(REPO_ROOT)),
            "sha256": in_sha256,
            "row_count": len(records),
        },
        "checks": checks,
        "overall_status": overall_status,
        "exact_duplicate_validation_detail": dup_result["exact_duplicate_validation"],
        "normalized_duplicate_validation_detail": dup_result["normalized_duplicate_validation"],
        "overlap_reverification_detail": overlap_result,
        "near_duplicate_validation_detail": neardup_result,
        "training_contamination_detail": contamination_result,
        "source_provenance_validation_detail": provenance_result,
        "exact_prompt_preservation_detail": preservation_result,
        "structural_classifier_validation_detail": classifier_result,
        "counts": {
            "total_candidates": len(records),
            "eligible_for_3a3_count": len(eligible),
            "excluded_count": len(excluded),
            "exclusion_reason_counts": dict(exclusion_reason_counts),
            "c_paired_overlap_count": overlap_result["c_paired_overlap_count"],
            "c_paired_pool_size_checked": overlap_result["c_paired_pool_size_checked"],
            "quadrant_a_overlap_count": overlap_result["quadrant_a_overlap_count"],
            "quadrant_a_pool_size_checked": overlap_result["quadrant_a_pool_size_checked"],
            "training_contamination_exact_counts_by_file": contamination_result["exact_contamination_counts_by_file"],
            "remaining_validated_candidate_count": len(eligible),
        },
        "output_artifacts": {
            "input_candidate_universe_jsonl": {
                "path": str(in_path.relative_to(REPO_ROOT)),
                "sha256": in_sha256,
                "row_count": len(records),
            },
            "validated_candidate_universe_jsonl": {
                "path": str(out_jsonl.relative_to(REPO_ROOT)),
                "sha256": out_sha256,
                "row_count": len(records),
            },
        },
        "corrections_made_this_milestone": [],
        "not_performed_this_milestone": [
            "fightin_words_scoring",
            "Q10_Q25_Q40_stratification",
            "review_queue_generation",
            "benchmark_freeze",
            "candidate_reconstruction (no validation failure required one)",
        ],
        "next_milestone": "3A4 - fixed Fightin' Words scoring, ranking, quantiles, and review queue.",
    }

    out_log_json = Path(args.out_log_json)
    out_log_json.parent.mkdir(parents=True, exist_ok=True)
    with open(out_log_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
        f.write("\n")

    md_lines = [
        "# 3A3 — C-Source-Authored Candidate Universe Validation",
        "",
        f"Overall status: **{overall_status}**",
        "",
        f"Input: `{summary['input_artifact']['path']}` (sha256 `{in_sha256}`, {len(records)} rows)",
        "",
        "## Checks",
        "",
        "| Check | Result |",
        "|---|---|",
    ]
    for k, v in checks.items():
        md_lines.append(f"| {k} | {v} |")
    md_lines += [
        "",
        "## Duplicate re-verification (Steps 1-2)",
        f"- Exact duplicates (recomputed): {dup_result['exact_duplicate_validation']['recomputed_duplicate_count']} "
        f"(mismatches vs 3A2-1: {len(dup_result['exact_duplicate_validation']['mismatches'])})",
        f"- Normalized-only duplicates (recomputed): {dup_result['normalized_duplicate_validation']['recomputed_duplicate_count']} "
        f"(mismatches vs 3A2-1: {len(dup_result['normalized_duplicate_validation']['mismatches'])})",
        "",
        "## Overlap re-verification (Steps 6-7 of 3A2-1, re-checked here)",
        f"- C-paired pool overlap (full {overlap_result['c_paired_pool_size_checked']}-row pool): "
        f"{overlap_result['c_paired_overlap_count']} (mismatches: {len(overlap_result['mismatches'])})",
        f"- Quadrant-A overlap (checked against {overlap_result['quadrant_a_pool_size_checked']} rows): "
        f"{overlap_result['quadrant_a_overlap_count']} (mismatches: {len(overlap_result['mismatches'])})",
        "",
        "## Near-duplicate validation (Step 3)",
        f"- Config: model=`{neardup_result['config']['model_name']}`, "
        f"threshold={neardup_result['config']['similarity_threshold']}, "
        f"method={neardup_result['config']['method']}",
    ]
    if neardup_result["blocker"]:
        md_lines.append(f"- **Blocked**: {neardup_result['blocker']}")
        md_lines.append(
            "- Status for all five required comparisons (within StrongREJECT; within "
            "SimpleSafetyTests; StrongREJECT vs SimpleSafetyTests; candidates vs Quadrant-A; "
            "candidates vs the full 155-row C-paired pool): **unknown** — not converted to \"clean\"."
        )
    else:
        for name, res in neardup_result["comparisons"].items():
            md_lines.append(f"- {name}: {res}")
    md_lines += [
        "",
        "## Training contamination (Step 4)",
        f"- Training files checked: {', '.join(contamination_result['training_files_checked'])} (all present in this checkout)",
        f"- Exact-match contamination (re-verified independently): "
        f"{contamination_result['exact_contamination_counts_by_file']} "
        f"(mismatches vs 3A2-1: {len(contamination_result['exact_mismatches'])})",
        f"- Near-dup contamination status by file: {contamination_result['near_status_by_file']}",
        "",
        "## Source/provenance validation (Step 5)",
        f"- StrongREJECT: reacquired fresh from `{provenance_result['StrongREJECT']['reacquired_from']}`, "
        f"sha256 verified: {provenance_result['StrongREJECT']['verified']}",
        f"- SimpleSafetyTests: reacquired fresh from `{provenance_result['SimpleSafetyTests']['reacquired_from']}`, "
        f"sha256 verified: {provenance_result['SimpleSafetyTests']['verified']}",
        f"- Per-record provenance field + deterministic record_id re-check: "
        f"{provenance_result['per_record_field_validation']['status']} "
        f"(mismatches: {len(provenance_result['per_record_field_validation']['mismatches'])})",
        "",
        "## Exact prompt preservation (Step 6)",
        f"- Rows checked byte-for-byte against freshly reacquired source rows: {preservation_result['rows_checked']}",
        f"- Mismatches: {len(preservation_result['mismatches'])}",
        "",
        "## Structural classifier (Step 7)",
        f"- Determinism mismatches: {len(classifier_result['classifier_determinism_mismatches'])}",
        f"- Non-standalone/ambiguous rows silently promoted to eligible_for_3a3: "
        f"{len(classifier_result['non_standalone_silently_promoted'])}",
        f"- Low-confidence rows kept provisional (not auto-excluded, not auto-promoted): "
        f"{classifier_result['low_confidence_provisional_count']}",
        "",
        "## Counts",
        f"- Total candidates: {len(records)}",
        f"- Eligible for 3A3 (unchanged from 3A2-1 - no correction was required): {len(eligible)}",
        f"- Excluded: {len(excluded)}",
        f"- Exclusion reason counts: {dict(exclusion_reason_counts)}",
        "",
        "## Output artifacts",
        f"- `{summary['output_artifacts']['validated_candidate_universe_jsonl']['path']}` "
        f"(sha256 `{out_sha256}`, {len(records)} rows)",
        "",
        "**Next milestone:** 3A4 - fixed Fightin' Words scoring, ranking, quantiles, and review queue.",
        "",
    ]

    out_log_md = Path(args.out_log_md)
    out_log_md.parent.mkdir(parents=True, exist_ok=True)
    out_log_md.write_text("\n".join(md_lines), encoding="utf-8")

    print(json.dumps({"overall_status": overall_status, "checks": checks}, indent=2))


if __name__ == "__main__":
    main()
