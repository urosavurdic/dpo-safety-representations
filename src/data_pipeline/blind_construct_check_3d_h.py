"""
Task 3D-H — Blind Human Construct Check.

Prepares a small blind human check of the provisional within-harmful
lexical outlierness construct (3D-A design, 3D-B pilot, 3D-C audit).

This script does NOT:
  - modify the frozen benchmark;
  - modify S2/S3 scoring or recompute any score/percentile;
  - modify 3D-B artifacts or the grouping artifact;
  - construct the B/D quadrants or run 3F;
  - search for datasets/resources or redesign the construct.

Frozen inputs (read-only, hashed, never rewritten):
  - logs/3d_b_lexical_outlierness_pilot.json   (SOLE source of row-level
    p_tfidf / p_selfinfo / tail labels)
  - logs/3d_c_length_dependence_audit.json     (hashed for provenance only)
  - data/quadrant_c_pipeline/c_source_authored_candidates_validated_v1.jsonl
    (209-row input population, via the existing eligible_for_3a3 predicate
    in lexical_outlierness.load_population - never re-derived here)
  - data/quadrant_c_pipeline/lexical_outlierness_groups_v1.json
    (immutable group mapping)

Sampling uses p_tfidf ONLY (>=0.75 high, <=0.25 low, else ineligible) to
stratify by tail x source (StrongREJECT, SimpleSafetyTests), then a
deterministic, source-balanced, group-unique selection driven by two
predeclared seeds (selection_seed=20260829, presentation_seed=20260830)
via numpy.random.default_rng (PCG64). See select_blind_sample() and
compute_presentation_order() for the exact algorithm, which mirrors the
task brief section 3 line for line.

Outputs:
  - data/review/3d_h_blind_construct_check.csv   (reviewer-visible, blind)
  - logs/3d_h_rating_instructions.md             (reviewer-visible)
  - logs/3d_h_blind_review_provenance.json       (non-sensitive; committed)
  - <--private-key-out> (REQUIRED, must resolve outside the repository)
    researcher-only answer key - NEVER a repository artifact, NEVER part
    of the git patch. Generation FAILS CLOSED if this path resolves
    inside the repository.

Run:
    python -m src.data_pipeline.blind_construct_check_3d_h \
        --private-key-out /absolute/path/outside/repo/3d_h_answer_key.json
"""
from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

from src.data_pipeline.build_c_source_authored_candidates import file_sha256
from src.data_pipeline.lexical_outlierness import (
    GROUPS_ARTIFACT_PATH,
    HIGH_TAIL_CUTOFF,
    LOW_TAIL_CUTOFF,
    POPULATION_ARTIFACT_PATH,
    REPO_ROOT,
    REQUIRED_SOURCES,
    load_population,
)

PILOT_JSON_PATH = REPO_ROOT / "logs/3d_b_lexical_outlierness_pilot.json"
AUDIT_JSON_PATH = REPO_ROOT / "logs/3d_c_length_dependence_audit.json"

OUT_CSV_PATH = REPO_ROOT / "data/review/3d_h_blind_construct_check.csv"
OUT_INSTRUCTIONS_PATH = REPO_ROOT / "logs/3d_h_rating_instructions.md"
OUT_PROVENANCE_PATH = REPO_ROOT / "logs/3d_h_blind_review_provenance.json"

EXPECTED_POPULATION_SIZE = 209

SELECTION_SEED = 20260829
PRESENTATION_SEED = 20260830

MAX_K = 20
MIN_K = 2

# Fixed processing order for the four strata (tail-major, source-minor),
# exactly as required by the task brief section 3. Do not reorder.
STRATUM_ORDER: List[Tuple[str, str]] = [
    ("high", REQUIRED_SOURCES[0]),
    ("high", REQUIRED_SOURCES[1]),
    ("low", REQUIRED_SOURCES[0]),
    ("low", REQUIRED_SOURCES[1]),
]

# The committed provenance artifact must never contain these strings.
_FORBIDDEN_PROVENANCE_STRINGS = tuple(REQUIRED_SOURCES)

RATING_QUESTION = (
    "How clearly does the wording itself signal a harmful operational "
    "objective?"
)

CSV_FIELDNAMES = ["review_id", "prompt_text", "rating", "notes"]


class InvariantError(SystemExit):
    """Raised (as SystemExit) when a required frozen-input invariant fails."""


# ── Section 1: frozen-input invariant checks ────────────────────────────
def verify_frozen_invariants(
    pilot: dict, groups: dict, population_rows: List[dict]
) -> Dict[str, str]:
    """Asserts every invariant required before any output is generated.
    Returns a dict of the four required SHA-256 hashes. Raises SystemExit
    (fail closed, no output written) on any violation."""
    population_ids = [r["record_id"] for r in population_rows]
    if len(population_rows) != EXPECTED_POPULATION_SIZE:
        raise InvariantError(
            f"FAIL CLOSED: expected exactly {EXPECTED_POPULATION_SIZE} "
            f"population rows, got {len(population_rows)}"
        )
    if len(set(population_ids)) != EXPECTED_POPULATION_SIZE:
        raise InvariantError(
            "FAIL CLOSED: population record IDs are not all distinct"
        )

    group_map = groups["record_id_to_group_id"]
    if len(group_map) != EXPECTED_POPULATION_SIZE:
        raise InvariantError(
            f"FAIL CLOSED: expected exactly {EXPECTED_POPULATION_SIZE} "
            f"record_id -> group_id mappings, got {len(group_map)}"
        )
    if len(set(group_map.keys())) != EXPECTED_POPULATION_SIZE:
        raise InvariantError(
            "FAIL CLOSED: grouping artifact record IDs are not all distinct"
        )
    if len(set(group_map.values())) != EXPECTED_POPULATION_SIZE:
        raise InvariantError(
            "FAIL CLOSED: grouping artifact group IDs are not all distinct"
        )

    pop_id_set = set(population_ids)
    group_id_keys = set(group_map.keys())
    if pop_id_set != group_id_keys:
        missing = pop_id_set - group_id_keys
        extra = group_id_keys - pop_id_set
        raise InvariantError(
            "FAIL CLOSED: grouping artifact record IDs do not exactly match "
            f"the 209-row population (missing={len(missing)}, extra={len(extra)})"
        )

    pilot_row_ids = set(pilot["scoring"]["row_level"].keys())
    if pilot_row_ids != pop_id_set:
        raise InvariantError(
            "FAIL CLOSED: 3D-B pilot row_level record IDs do not exactly "
            "match the 209-row population"
        )

    return {
        "pilot_json_sha256": file_sha256(PILOT_JSON_PATH),
        "audit_json_sha256": file_sha256(AUDIT_JSON_PATH),
        "population_sha256": file_sha256(POPULATION_ARTIFACT_PATH),
        "groups_sha256": file_sha256(GROUPS_ARTIFACT_PATH),
    }


# ── Section 2: p_tfidf-only stratum assignment ──────────────────────────
def assign_strata_from_p_tfidf(pilot_row_level: dict) -> Dict[Tuple[str, str], List[str]]:
    """Buckets every row into (tail, source) using p_tfidf alone, per the
    predeclared cutoffs. Rows strictly between the cutoffs are ineligible
    for this blind sample and are dropped here. Never touches p_selfinfo
    or any other score for stratum assignment."""
    buckets: Dict[Tuple[str, str], List[str]] = {key: [] for key in STRATUM_ORDER}
    for record_id, row in pilot_row_level.items():
        p_tfidf = row["p_tfidf"]
        source = row["source"]
        if p_tfidf >= HIGH_TAIL_CUTOFF:
            tail = "high"
        elif p_tfidf <= LOW_TAIL_CUTOFF:
            tail = "low"
        else:
            continue
        key = (tail, source)
        if key in buckets:
            buckets[key].append(record_id)
    return buckets


# ── Section 3: deterministic source-balanced sampling ───────────────────
def select_blind_sample(
    buckets: Dict[Tuple[str, str], List[str]]
) -> Tuple[int, Dict[Tuple[str, str], List[str]]]:
    """Implements the exact selection procedure in task brief section 3:
    one selection RNG initialized once, the four strata permuted in the
    fixed STRATUM_ORDER (never reinitializing the RNG between strata),
    then the largest feasible even k (<=20) tested in descending order
    against the already-fixed permutations. Returns (k, permuted_buckets)
    where permuted_buckets holds each stratum's full permutation (not yet
    truncated to k/2) so the caller can take the deterministic head.
    """
    rng = np.random.default_rng(SELECTION_SEED)
    permuted: Dict[Tuple[str, str], List[str]] = {}
    for key in STRATUM_ORDER:
        eligible_sorted = sorted(buckets[key])
        if eligible_sorted:
            perm_idx = rng.permutation(len(eligible_sorted))
        else:
            # Still consume the RNG deterministically even for an empty
            # stratum, so downstream strata are unaffected by whether an
            # earlier stratum happened to be empty.
            perm_idx = rng.permutation(0)
        permuted[key] = [eligible_sorted[i] for i in perm_idx]

    chosen_k = None
    for k in range(MAX_K, MIN_K - 1, -2):
        half = k // 2
        candidate: Dict[Tuple[str, str], List[str]] = {}
        feasible = True
        for key in STRATUM_ORDER:
            if len(permuted[key]) < half:
                feasible = False
                break
            candidate[key] = permuted[key][:half]
        if not feasible:
            continue

        # Assertions required before accepting this k.
        all_ids = [rid for key in STRATUM_ORDER for rid in candidate[key]]
        if len(all_ids) != 2 * k:
            continue
        high_ids = candidate[("high", REQUIRED_SOURCES[0])] + candidate[("high", REQUIRED_SOURCES[1])]
        low_ids = candidate[("low", REQUIRED_SOURCES[0])] + candidate[("low", REQUIRED_SOURCES[1])]
        if len(high_ids) != k or len(low_ids) != k:
            continue
        if any(len(candidate[key]) != half for key in STRATUM_ORDER):
            continue
        if len(set(all_ids)) != len(all_ids):
            continue  # duplicate record id somehow -- reject

        chosen_k = k
        selected = candidate
        break

    if chosen_k is None:
        raise SystemExit(
            "FAIL CLOSED: no positive even k <= 20 satisfies source/tail "
            "balance -- refusing to generate the review packet."
        )
    return chosen_k, selected


def assign_review_ids(k: int, selected: Dict[Tuple[str, str], List[str]]) -> List[Tuple[str, str]]:
    """Assigns R001, R002, ... in the deterministic concatenated selection
    order (STRATUM_ORDER, each stratum's already-permuted head). Returns a
    list of (review_id, record_id) pairs in that assignment order."""
    ordered_record_ids = [rid for key in STRATUM_ORDER for rid in selected[key]]
    assert len(ordered_record_ids) == 2 * k
    return [(f"R{i+1:03d}", rid) for i, rid in enumerate(ordered_record_ids)]


def compute_presentation_order(
    review_id_pairs: List[Tuple[str, str]]
) -> List[Tuple[str, str]]:
    """Initializes the presentation RNG exactly once (separately from the
    selection RNG) and permutes the review IDs for presentation."""
    rng2 = np.random.default_rng(PRESENTATION_SEED)
    perm_idx = rng2.permutation(len(review_id_pairs))
    return [review_id_pairs[i] for i in perm_idx]


# ── Sections 4/6/7/8: output writers ────────────────────────────────────
def write_blind_csv(
    presented: List[Tuple[str, str]], prompt_text_by_id: Dict[str, str], path: Path
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDNAMES, lineterminator="\n")
        writer.writeheader()
        for review_id, record_id in presented:
            writer.writerow(
                {
                    "review_id": review_id,
                    "prompt_text": prompt_text_by_id[record_id],
                    "rating": "",
                    "notes": "",
                }
            )


def write_rating_instructions(path: Path, n_total: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    content = f"""# 3D-H Blind Construct Check — Rating Instructions

You will rate {n_total} prompts, identified only by a neutral review ID
(`R001`, `R002`, ...). The prompts are presented in a random order that
carries no meaning.

For each prompt, answer only this question:

> **{RATING_QUESTION}**

Use this scale:

| Rating | Meaning |
|---|---|
| 1 | not apparent |
| 2 | weakly apparent |
| 3 | ambiguous / moderate |
| 4 | clearly apparent |
| 5 | unmistakably apparent |

This is **not** an intent-classification task. Do not ask yourself whether
the prompt is actually harmful, whether it should be refused, or whether it
is legal — rate only how clearly the *wording itself* signals a harmful
operational objective. Please do not rewrite or edit any prompt text.

Enter your rating (1-5) in the `rating` column. The `notes` column is
optional free text for anything you want to flag.
"""
    path.write_text(content, encoding="utf-8")


def build_provenance(
    *,
    k: int,
    unique_group_count: int,
    input_hashes: Dict[str, str],
    blind_csv_sha256: str,
    answer_key_sha256: str,
    generation_commit: str,
) -> dict:
    provenance = {
        "task": "3D-H blind human construct check",
        "selected_method": "p_tfidf",
        "high_tail_cutoff": HIGH_TAIL_CUTOFF,
        "low_tail_cutoff": LOW_TAIL_CUTOFF,
        "realized_k": k,
        "realized_high_tail_sample_size": k,
        "realized_low_tail_sample_size": k,
        "number_of_sources": len(REQUIRED_SOURCES),
        "high_tail_count_per_source": k // 2,
        "low_tail_count_per_source": k // 2,
        "unique_group_count": unique_group_count,
        "selection_seed": SELECTION_SEED,
        "presentation_seed": PRESENTATION_SEED,
        "numpy_version": np.__version__,
        "exact_selection_algorithm": (
            "One selection RNG (numpy.random.default_rng, PCG64) "
            "initialized once with selection_seed. Four (tail x source) "
            "strata processed in a single fixed predeclared order "
            "(tail-major, source-minor); within each stratum, eligible "
            "record IDs are sorted lexicographically then permuted with "
            "one rng.permutation call, consuming RNG state sequentially "
            "without reinitialization. After all four strata are "
            "permuted, even candidate k values from 20 down to 2 are "
            "tested in descending order; for each k, the first k/2 "
            "already-permuted IDs from every stratum are taken and all "
            "balance/uniqueness assertions are checked; the first k for "
            "which every assertion passes is realized. No row is skipped, "
            "substituted, or re-permuted to force a larger k."
        ),
        "exact_sampling_rule": (
            "Stratum assignment uses p_tfidf alone (>= high cutoff, <= low "
            "cutoff, else ineligible). Within an eligible stratum, "
            "selection uses only record ID, frozen tail label, source "
            "category, and group ID -- never prompt content, category "
            "label, token count, formatting, or any numeric score. The "
            "realized sample contains k record IDs from the high tail and "
            "k from the low tail, split k/2 per source category within "
            "each tail, with every group ID in the complete 2k-row sample "
            "required to be pairwise distinct."
        ),
        "pilot_json_sha256": input_hashes["pilot_json_sha256"],
        "audit_json_sha256": input_hashes["audit_json_sha256"],
        "input_population_sha256": input_hashes["population_sha256"],
        "grouping_artifact_sha256": input_hashes["groups_sha256"],
        "blind_review_csv_sha256": blind_csv_sha256,
        "answer_key_sha256": answer_key_sha256,
        "source_generation_commit": generation_commit,
    }
    serialized = json.dumps(provenance)
    for forbidden in _FORBIDDEN_PROVENANCE_STRINGS:
        if forbidden in serialized:
            raise SystemExit(
                f"FAIL CLOSED: committed provenance artifact would contain "
                f"the forbidden string {forbidden!r}"
            )
    return provenance


def build_answer_key(
    review_id_pairs: List[Tuple[str, str]],
    pilot_row_level: dict,
    group_map: dict,
) -> List[dict]:
    entries = []
    for review_id, record_id in review_id_pairs:
        row = pilot_row_level[record_id]
        entries.append(
            {
                "review_id": review_id,
                "record_id": record_id,
                "source": row["source"],
                "category": row["category"],
                "group_id": group_map[record_id],
                "p_tfidf": row["p_tfidf"],
                "p_selfinfo": row["p_selfinfo"],
                "tail": row["tail_tfidf"],
                "selection_seed": SELECTION_SEED,
                "presentation_seed": PRESENTATION_SEED,
            }
        )
    return entries


def validate_private_key_path(raw_path: str) -> Path:
    """Fails closed if the private key output path is inside the
    repository (and therefore could ever land in a git patch)."""
    resolved = Path(raw_path).expanduser().resolve()
    repo_resolved = REPO_ROOT.resolve()
    try:
        resolved.relative_to(repo_resolved)
    except ValueError:
        return resolved
    raise SystemExit(
        f"FAIL CLOSED: --private-key-out ({resolved}) resolves inside the "
        f"repository ({repo_resolved}). The answer key must never be a "
        "repository artifact or appear in the git patch."
    )


def get_generation_commit() -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True
    ).strip()


def main(argv: List[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--private-key-out",
        required=True,
        help="Absolute path OUTSIDE the repository for the researcher-only "
        "answer key. Generation fails closed if this resolves inside the "
        "repository.",
    )
    args = parser.parse_args(argv)

    private_key_path = validate_private_key_path(args.private_key_out)

    pilot = json.loads(PILOT_JSON_PATH.read_text(encoding="utf-8"))
    groups = json.loads(GROUPS_ARTIFACT_PATH.read_text(encoding="utf-8"))
    population_rows = load_population()  # asserts exactly 209 internally

    input_hashes = verify_frozen_invariants(pilot, groups, population_rows)

    pilot_row_level = pilot["scoring"]["row_level"]
    group_map = groups["record_id_to_group_id"]
    prompt_text_by_id = {r["record_id"]: r["prompt_text"] for r in population_rows}

    buckets = assign_strata_from_p_tfidf(pilot_row_level)
    k, selected = select_blind_sample(buckets)

    review_id_pairs = assign_review_ids(k, selected)

    # Group-uniqueness assertion across the complete sample, using the
    # frozen (immutable) grouping artifact.
    group_ids = [group_map[rid] for _, rid in review_id_pairs]
    assert len(set(group_ids)) == len(group_ids), (
        "FAIL CLOSED: duplicate group_id in the selected sample"
    )
    assert len(review_id_pairs) == 2 * k

    presented = compute_presentation_order(review_id_pairs)

    generation_commit = get_generation_commit()

    write_blind_csv(presented, prompt_text_by_id, OUT_CSV_PATH)
    blind_csv_sha256 = file_sha256(OUT_CSV_PATH)

    write_rating_instructions(OUT_INSTRUCTIONS_PATH, n_total=2 * k)

    answer_key_entries = build_answer_key(review_id_pairs, pilot_row_level, group_map)
    private_key_path.parent.mkdir(parents=True, exist_ok=True)
    private_key_path.write_text(
        json.dumps(answer_key_entries, indent=2), encoding="utf-8"
    )
    answer_key_sha256 = file_sha256(private_key_path)

    provenance = build_provenance(
        k=k,
        unique_group_count=len(set(group_ids)),
        input_hashes=input_hashes,
        blind_csv_sha256=blind_csv_sha256,
        answer_key_sha256=answer_key_sha256,
        generation_commit=generation_commit,
    )
    OUT_PROVENANCE_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PROVENANCE_PATH.write_text(json.dumps(provenance, indent=2), encoding="utf-8")

    print(f"realized_k={k} total_sample={2 * k}")
    print(
        f"high_per_source={k // 2} low_per_source={k // 2} "
        f"unique_groups={len(set(group_ids))}"
    )
    print(f"blind_csv={OUT_CSV_PATH.relative_to(REPO_ROOT)} sha256={blind_csv_sha256}")
    print(f"instructions={OUT_INSTRUCTIONS_PATH.relative_to(REPO_ROOT)}")
    print(f"provenance={OUT_PROVENANCE_PATH.relative_to(REPO_ROOT)}")
    print(f"private_answer_key={private_key_path} (OUTSIDE repository, sha256={answer_key_sha256})")
    print(f"generation_commit={generation_commit}")


if __name__ == "__main__":
    main()
