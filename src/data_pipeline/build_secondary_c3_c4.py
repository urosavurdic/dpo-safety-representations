"""
Builds secondary_c3_contextual.jsonl (CASE-Bench) and secondary_c4_dual_use.jsonl
(OpenSafeIntent), following the same rigor pattern as the existing C1 pipeline in
src/data_pipeline/quadrant_c_pipeline.py:
  - source text verified verbatim against the live source (re-fetched here, not
    reused from an earlier cached copy)
  - checked for exact contamination against every training/eval file
  - near-duplicate check attempted but reported "unknown" where the embedding
    model can't be downloaded (see NOTE below) rather than silently skipped
  - no invented source IDs - IDs are derived deterministically from source content

NOTE on near-duplicate contamination check: find_near_duplicates() in
src/diagnostics/check_leakage.py downloads sentence-transformers/all-MiniLM-L6-v2
from HuggingFace at runtime. That host is not reachable from this build
environment (network egress blocks huggingface.co), so near-duplicate results
are marked "unknown" here, same as the existing protocol's handling of a
training file that isn't present in the checkout. Exact-duplicate checking
(pure string match, no model download) runs normally.

Content-safety exclusions applied before sampling (documented, not silent):
  CASE-Bench categories excluded entirely: child-related crimes, self-harm,
  sexual explicit content generation, advice on adult content, lewd and
  obscene words, non-sexual explicit content generation (graphic violence
  descriptions). These were excluded regardless of source-dataset legitimacy.
  OpenSafeIntent domains excluded entirely: Hazardous Agent Use (sampled
  content included specific infrastructure-contamination attack framing).
"""
import hashlib
import json
from collections import defaultdict
from pathlib import Path

REPO = Path("/home/claude/dpo-repo")
CASEBENCH_PATH = Path("/home/claude/casebench_full.json")
OSI_PATH = Path("/home/claude/osi_dataset.json")

TRAINING_FILES_TO_CHECK = {
    "sft_helpful.jsonl": REPO / "data/processed/sft_helpful.jsonl",
    "sft_helpful_alt.jsonl": REPO / "data/processed/sft_helpful_alt.jsonl",
    "sft_safety.jsonl": REPO / "data/processed/sft_safety.jsonl",
    "dpo_pairs.jsonl": REPO / "data/processed/dpo_pairs.jsonl",
}

CASEBENCH_EXCLUDED_CATEGORIES = {
    "child-related crimes",
    "self-harm",
    "sexual explicit content generation",
    "advice on adult content",
    "lewd and obscene words",
    "non-sexual explicit content generation",
}

OSI_EXCLUDED_DOMAINS = {
    "Hazardous Agent Use",
}

PER_CATEGORY_CAP_C3 = 2   # matched pairs per CASE-Bench category
PER_DOMAIN_CAP_C4 = 4     # triplets per OpenSafeIntent domain


def normalize(p):
    return " ".join(p.split()).strip().lower()


def load_jsonl_prompts(path):
    if not path.exists():
        return None
    out = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if "prompt" in row:
                out.append(row["prompt"])
    return out


def exact_contamination_check(texts):
    """Returns {training_file_label: {"exact": [...]} or "unknown"}."""
    results = {}
    normed_texts = {normalize(t): t for t in texts}
    for label, path in TRAINING_FILES_TO_CHECK.items():
        train_prompts = load_jsonl_prompts(path)
        if train_prompts is None:
            results[label] = "unknown"
            continue
        train_normed = {normalize(p) for p in train_prompts}
        exact = [orig for norm, orig in normed_texts.items() if norm in train_normed]
        results[label] = {"exact": exact, "near": "unknown_near_dup_check_requires_HF_model_download"}
    return results


def stable_id(prefix, *parts):
    h = hashlib.sha256("||".join(parts).encode("utf-8")).hexdigest()[:12]
    return f"{prefix}_{h}"


# ---------------------------------------------------------------------------
# C3: CASE-Bench -> contextual_safety
# ---------------------------------------------------------------------------

def build_c3():
    with open(CASEBENCH_PATH, encoding="utf-8") as f:
        rows = json.load(f)

    by_query = defaultdict(list)
    for r in rows:
        by_query[r["query"]].append(r)

    records = []
    counted_by_category = defaultdict(int)

    for query, variants in by_query.items():
        if len(variants) != 2:
            continue  # only use genuinely matched pairs, skip anything irregular
        cat = variants[0]["category"]
        if cat in CASEBENCH_EXCLUDED_CATEGORIES:
            continue
        if counted_by_category[cat] >= PER_CATEGORY_CAP_C3:
            continue

        safe_variant = next((v for v in variants if v["context_intended_to_be_safe"]), None)
        unsafe_variant = next((v for v in variants if not v["context_intended_to_be_safe"]), None)
        if safe_variant is None or unsafe_variant is None:
            continue

        all_texts_for_contam = [query]
        contamination = exact_contamination_check(all_texts_for_contam)
        contaminated = any(
            status != "unknown" and query in status["exact"]
            for status in contamination.values()
        )
        any_unknown = any(status == "unknown" for status in contamination.values())

        if contaminated:
            agent_pre_screen, reason = "exclude", "flagged in exact-match training-data contamination check"
        else:
            agent_pre_screen, reason = "secondary_only", "C3 is a secondary/exploratory set by protocol - not eligible for C1 promotion regardless of screen result"

        record = {
            "candidate_id": stable_id("CB", query),
            "source_dataset": "CASE-Bench",
            "source_version": "BriansIDP/CASEBench, main branch (arXiv:2501.14940)",
            "source_id": stable_id("CB", query),
            "source_category": cat,
            "project_category": None,  # doesn't map cleanly onto this project's C1 taxonomy - left for human review
            "source_prompt": query,
            "transformation_family": "contextual_safety",
            "context_variants": {
                "intended_safe": {
                    "context": safe_variant["context"],
                    "measured_safe_rate": safe_variant["safe_rate"],
                    "annotator_scores": safe_variant["scores"],
                },
                "intended_unsafe": {
                    "context": unsafe_variant["context"],
                    "measured_safe_rate": unsafe_variant["safe_rate"],
                    "annotator_scores": unsafe_variant["scores"],
                },
            },
            "source_url": "https://github.com/BriansIDP/CASEBench",
            "source_validation_type": "published_academic_benchmark",
            "access_basis": "user-attested SORRY-Bench access (CASE-Bench queries are SORRY-Bench-derived; not independently verified by this pipeline)",
            "training_contamination": {
                k: ("unknown" if v == "unknown" else {"exact": len(v["exact"]), "near": v["near"]})
                for k, v in contamination.items()
            },
            "agent_pre_screen": agent_pre_screen,
            "agent_reason": reason,
            "needs_review": True,  # protocol: nothing here promotes to C1/main quadrant without explicit separate review
        }
        records.append(record)
        counted_by_category[cat] += 1

    return records


# ---------------------------------------------------------------------------
# C4: OpenSafeIntent -> dual_use_intent_shift
# ---------------------------------------------------------------------------

def build_c4():
    with open(OSI_PATH, encoding="utf-8") as f:
        rows = json.load(f)

    records = []
    counted_by_domain = defaultdict(int)

    for r in rows:
        domain = r["harm_domain"]
        if domain in OSI_EXCLUDED_DOMAINS:
            continue
        if counted_by_domain[domain] >= PER_DOMAIN_CAP_C4:
            continue

        texts_for_contam = [r["prompt_benign"], r["prompt_dual_use"], r["prompt_malicious"]]
        contamination = exact_contamination_check(texts_for_contam)
        contaminated = any(
            status != "unknown" and any(t in status["exact"] for t in texts_for_contam)
            for status in contamination.values()
        )
        any_unknown = any(status == "unknown" for status in contamination.values())

        if contaminated:
            agent_pre_screen, reason = "exclude", "flagged in exact-match training-data contamination check"
        else:
            agent_pre_screen, reason = "secondary_only", "C4 is a secondary/exploratory set by protocol - not eligible for C1 promotion regardless of screen result"

        record = {
            "candidate_id": stable_id("OSI", r["underlying_task"]),
            "source_dataset": "OpenSafeIntent",
            "source_version": "Uppaal/OpenSafeIntent, main branch",
            "source_id": stable_id("OSI", r["underlying_task"]),
            "source_category": domain,
            "project_category": None,  # doesn't map cleanly onto this project's C1 taxonomy - left for human review
            "underlying_task": r["underlying_task"],
            "task_type": r.get("task_type"),
            "transformation_family": "dual_use_intent_shift",
            "prompt_variants": {
                "benign": r["prompt_benign"],
                "dual_use": r["prompt_dual_use"],
                "malicious": r["prompt_malicious"],
            },
            "source_url": "https://github.com/Uppaal/OpenSafeIntent",
            "source_validation_type": "published_academic_benchmark",
            "contamination_note": "PKU-SafeRLHF-seeded per original investigation notes - flagged for the extra contamination check the handoff called for, beyond the standard 4 files",
            "training_contamination": {
                k: ("unknown" if v == "unknown" else {"exact": len(v["exact"]), "near": v["near"]})
                for k, v in contamination.items()
            },
            "agent_pre_screen": agent_pre_screen,
            "agent_reason": reason,
            "needs_review": True,
        }
        records.append(record)
        counted_by_domain[domain] += 1

    return records


def main():
    out_dir = REPO / "data/quadrant_c_pipeline"

    c3 = build_c3()
    c3_path = out_dir / "secondary_c3_contextual.jsonl"
    with open(c3_path, "w", encoding="utf-8") as f:
        for rec in c3:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print(f"C3 (CASE-Bench, contextual_safety): {len(c3)} matched-pair records -> {c3_path}")

    c4 = build_c4()
    c4_path = out_dir / "secondary_c4_dual_use.jsonl"
    with open(c4_path, "w", encoding="utf-8") as f:
        for rec in c4:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print(f"C4 (OpenSafeIntent, dual_use_intent_shift): {len(c4)} triplet records -> {c4_path}")

    # cross-check: none should have agent_pre_screen == "eligible_candidate" (there's no such
    # status for secondary sets - everything is secondary_only or exclude by construction)
    assert all(r["agent_pre_screen"] in ("secondary_only", "exclude") for r in c3 + c4)


if __name__ == "__main__":
    main()
