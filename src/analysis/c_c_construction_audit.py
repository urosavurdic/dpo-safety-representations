"""
Task C-C -- Execute and freeze the existing-C audit.

This module is an EXECUTION/AGGREGATION wrapper, not a new analysis. It
does not define any statistic, feature, inclusion rule, or hypothesis
test that is not already implemented elsewhere in this repository. Its
only two jobs are:

  1. Run the locked, already-implemented C-B contract
     (`src.analysis.c_b_paired_delta_analysis`, matching
     `logs/c_existing_construction_audit_spec.md` section 7) exactly as
     written, with its own predeclared seeds and inputs, and verify the
     result is reproducible.
  2. Read a small set of OTHER already-existing, already-committed
     repository artifacts verbatim (never recomputed, never redefined)
     -- the A/B/D quadrant audit, the quadrant composition report, the
     frozen benchmark's quadrant-C category counts, and the R-AUTHORED
     review-queue's own pre-existing per-row columns -- and assemble
     everything into one frozen audit document per C-C's task brief.

What this module explicitly does NOT do:
  - does not change any C-B feature definition, sign convention,
    seed, or inclusion rule;
  - does not add a new statistical test, effect size, or corrected
    comparison anywhere A/B/C/D quadrants are discussed (C-C's brief:
    "Do NOT require C to differ from every quadrant" / "Report only
    predeclared features"); where no predeclared cross-quadrant test
    exists in this repository, this module reports that as an
    evidence gap rather than defining one;
  - does not compute a "repeated-template concentration" metric --
    no existing repository artifact defines one, so C-C reports this
    as an explicit gap (see `_confounds`);
  - does not assign KEEP / KEEP AS SECONDARY / INCONCLUSIVE / DROP to
    any resource (C-A section 7.7; still out of scope for C-C);
  - does not create, rewrite, or score any prompt, run model
    inference/GPU code, or access the web;
  - does not print raw prompt text, matched lexicon terms, classifier
    weights, or prompt-level rankings -- only aggregate statistics and
    opaque record_ids/paths/hashes, matching C-B's own committed-output
    convention.

Canonical outputs (created fresh by this module; C-B's own
`logs/c_b_paired_delta_analysis.{md,json}` are read-only precedent and
are never overwritten by this module):
  - results/c_construction_audit/audit.json
  - results/c_construction_audit/audit_summary.md
  - results/c_construction_audit/input_manifest.json

Exact command:
    python -m src.analysis.c_c_construction_audit \
        --out-dir results/c_construction_audit
"""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics as stats
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Dict, List, Optional

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.analysis import c_b_paired_delta_analysis as c_b  # noqa: E402
from src import v2_io  # noqa: E402

DEFAULT_OUT_DIR = "results/c_construction_audit"

# Secondary (non-C-A-section-7.1-pinned) inputs this module reads
# verbatim. Hashes are recorded, not enforced fail-closed, mirroring
# C-B's own "additional_recorded_input_hashes_not_pinned_by_c_a"
# convention -- these are read-only supporting evidence, not the
# locked primary contract.
SECONDARY_INPUT_PATHS = {
    "audit_quadrants_report": "logs/audit_quadrants_report.json",
    "quadrant_composition_report": "data/quadrant_composition_report.json",
    "c_source_authored_review_queue": "data/review/c_source_authored_review_queue.csv",
    "c_source_authored_candidates_raw": "data/quadrant_c_pipeline/c_source_authored_candidates_raw_v1.jsonl",
    "c_source_authored_candidates_validated": "data/quadrant_c_pipeline/c_source_authored_candidates_validated_v1.jsonl",
    "milestone_3a4_scoring_md": "logs/3a4_scoring.md",
    "c_existing_construction_audit_spec": "logs/c_existing_construction_audit_spec.md",
}


def sha256_file(path: Path) -> Optional[str]:
    if not path.exists():
        return None
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _resolve(path_str: str) -> Path:
    p = Path(path_str)
    return p if p.is_absolute() else (REPO_ROOT / p)


def _display_path(p: Path) -> str:
    try:
        return str(p.relative_to(REPO_ROOT))
    except ValueError:
        return str(p)


# ── 1. Primary: execute the locked C-B contract exactly as implemented ────
def run_locked_c_b_contract() -> Dict:
    """Runs `src.analysis.c_b_paired_delta_analysis.main()` with exactly
    the arguments in C-A section 7.9, writing its md/json outputs to a
    scratch directory (never to `logs/c_b_paired_delta_analysis.*`,
    which C-C must not overwrite) and returning the resulting analysis
    dict. C-B's own module performs the section-7.1 fail-closed hash
    verification; this wrapper does not re-verify or relax it."""
    exact_command = (
        "python -m src.analysis.c_b_paired_delta_analysis "
        "--review-csv data/review/c_review_queue.csv "
        "--benchmark-latest data/frozen_v2/LATEST_BENCHMARK.json "
        "--gate-config logs/benchmark_gate_config.json "
        "--formatting-config-source logs/3d_b_lexical_outlierness_pilot.json "
        "--bootstrap-seed 20260901 --n-bootstrap 10000 "
        "--permutation-seed 20260902 --n-permutations 100000 "
        "--out-md logs/c_b_paired_delta_analysis.md "
        "--out-json logs/c_b_paired_delta_analysis.json"
    )
    with tempfile.TemporaryDirectory() as tmp:
        tmp_md = str(Path(tmp) / "c_b_rerun.md")
        tmp_json = str(Path(tmp) / "c_b_rerun.json")
        argv = [
            "--review-csv", "data/review/c_review_queue.csv",
            "--benchmark-latest", "data/frozen_v2/LATEST_BENCHMARK.json",
            "--gate-config", "logs/benchmark_gate_config.json",
            "--formatting-config-source", "logs/3d_b_lexical_outlierness_pilot.json",
            "--bootstrap-seed", "20260901", "--n-bootstrap", "10000",
            "--permutation-seed", "20260902", "--n-permutations", "100000",
            "--out-md", tmp_md,
            "--out-json", tmp_json,
        ]
        analysis = c_b.main(argv)

    # Byte-for-byte cross-check against the already-committed C-B output
    # (logs/c_b_paired_delta_analysis.json), ignoring provenance fields
    # that record *what produced this run* rather than *what the run
    # computed*, and that therefore legitimately differ between any two
    # executions of this same deterministic analysis:
    #   - code_version (generation commit / working-tree-dirty flag):
    #     this rerun is by definition on a different commit than the one
    #     that authored the committed output.
    #   - software_versions (python/numpy/scipy/pandas/scikit-learn):
    #     requirements.txt pins only lower bounds ("numpy>=1.26" etc.),
    #     not exact versions, and no lock file exists anywhere in this
    #     repository, so a later `pip install -r requirements.txt`
    #     legitimately resolves to newer compatible releases. Verified
    #     by inspection (see logs/task1_cpu_hardening_report.md): with
    #     numpy 2.4.4->2.5.2, pandas 3.0.2->3.0.5, scikit-learn
    #     1.8.0->1.9.0, scipy 1.17.1->1.18.1, every field of `analysis`
    #     other than software_versions itself -- including every
    #     LogisticRegression-derived statistic in `results` -- is
    #     exactly byte-identical, so this is not silently ignoring a
    #     real numeric drift; it is excluded from the pass/fail verdict
    #     for the same reason code_version is, but the diff is still
    #     recorded below rather than discarded.
    #
    #     Note this only guarantees *structural* reproducibility (same
    #     shape, same non-float values). Byte-identical *floating-point*
    #     leaf values (e.g. results/*/features/*/length_sensitivity/
    #     spearman_p_value) are additionally guaranteed only when running
    #     against the exact versions pinned in requirements-lock.txt --
    #     see software_versions_match below and
    #     tests/analysis/test_c_c_construction_audit.py::
    #     test_run_locked_c_b_contract_is_reproducible for how callers
    #     should scope a byte-identical check on `results` to that
    #     pinned environment rather than asserting it unconditionally.
    committed_path = REPO_ROOT / c_b.DEFAULT_OUT_JSON
    reproducibility_check = {"committed_output_path": c_b.DEFAULT_OUT_JSON}
    if committed_path.exists():
        committed = json.loads(committed_path.read_text(encoding="utf-8"))
        rerun_for_diff = json.loads(json.dumps(analysis))
        committed_software_versions = committed.get("software_versions", {})
        rerun_software_versions = rerun_for_diff.get("software_versions", {})
        for d in (committed, rerun_for_diff):
            d.get("code_version", {}).pop("generation_commit", None)
            d.get("code_version", {}).pop("working_tree_dirty", None)
            d.pop("software_versions", None)
        reproducibility_check["byte_identical_excluding_code_version"] = (
            json.dumps(committed, sort_keys=True) == json.dumps(rerun_for_diff, sort_keys=True)
        )
        reproducibility_check["software_versions_match"] = (
            committed_software_versions == rerun_software_versions
        )
        if not reproducibility_check["software_versions_match"]:
            reproducibility_check["software_versions_diff"] = {
                "committed": committed_software_versions,
                "rerun": rerun_software_versions,
            }
        # pinned_input_hashes_verified (C-A section 7.1) is, unlike the two
        # provenance fields above, part of the actual byte-identical
        # verdict on purpose: it is the fail-closed record of which exact
        # data/code files fed this run, so a genuine change to one of
        # those files (e.g. a tracked bug fix landing in a pinned
        # dependency such as src/corpus_discrimination.py) SHOULD flip
        # this to non-reproducible rather than being silently absorbed --
        # that is the whole point of pinning it. Recorded here, by key,
        # so a False verdict is self-explanatory instead of requiring a
        # manual diff of the two JSON files.
        committed_hashes = committed.get("pinned_input_hashes_verified", {})
        rerun_hashes = rerun_for_diff.get("pinned_input_hashes_verified", {})
        hash_diff = {
            path: {"committed": committed_hashes.get(path), "rerun": rerun_hashes.get(path)}
            for path in sorted(set(committed_hashes) | set(rerun_hashes))
            if committed_hashes.get(path) != rerun_hashes.get(path)
        }
        if hash_diff:
            reproducibility_check["pinned_input_hash_diff"] = hash_diff
    else:
        reproducibility_check["byte_identical_excluding_code_version"] = None
        reproducibility_check["note"] = "no prior committed C-B output found to diff against"

    return {
        "exact_command": exact_command,
        "c_b_module_path": "src/analysis/c_b_paired_delta_analysis.py",
        "c_b_module_sha256": sha256_file(REPO_ROOT / "src/analysis/c_b_paired_delta_analysis.py"),
        "reproducibility_check": reproducibility_check,
        "analysis": analysis,
    }


# ── 2. Secondary A/B/C/D unpaired distributions (existing evidence only) ──
def load_secondary_abcd_distributions() -> Dict:
    hashes = {}
    for key, rel in SECONDARY_INPUT_PATHS.items():
        if key in ("audit_quadrants_report", "quadrant_composition_report"):
            hashes[rel] = sha256_file(_resolve(rel))

    audit_quadrants = json.loads((REPO_ROOT / "logs/audit_quadrants_report.json").read_text(encoding="utf-8"))
    quadrant_composition = json.loads((REPO_ROOT / "data/quadrant_composition_report.json").read_text(encoding="utf-8"))

    # Quadrant-C category/source composition read directly from the
    # frozen benchmark via the existing strict-hash loader (never by
    # opening the .jsonl by filename), mirroring C-B's own convention.
    benchmark_path, benchmark_sha = v2_io.resolve_benchmark()
    rows = []
    with open(benchmark_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    c_rows = [r for r in rows if r.get("quadrant") == "C"]
    c_category_counts: Dict[str, int] = {}
    c_source_counts: Dict[str, int] = {}
    for r in c_rows:
        cat = r.get("project_category")
        c_category_counts[cat] = c_category_counts.get(cat, 0) + 1
        src = r.get("source_dataset")
        c_source_counts[src] = c_source_counts.get(src, 0) + 1

    return {
        "evidence_tier": "C-A section 8.6 evidence-hierarchy tier 5 -- A/B/C/D unpaired distributions, secondary evidence only",
        "not_predeclared": (
            "No cross-quadrant (A-vs-B-vs-C-vs-D) effect size, significance "
            "test, or multiple-comparison correction is defined anywhere in "
            "this repository. C-C's brief explicitly states C need not "
            "differ from every quadrant, and instructs execution only, so "
            "no such test is newly defined here. What follows is "
            "descriptive-only, reusing numbers already computed and "
            "committed by `src/audit_existing_quadrants.py` (A/B/D) and "
            "`src/diagnostics/quadrant_composition_check.py` (all four "
            "quadrants' lexical-risk-lexicon hit rate), plus a direct "
            "read of quadrant C's own category/source fields from the "
            "frozen benchmark."
        ),
        "quadrant_c_category_composition": c_category_counts,
        "quadrant_c_source_composition": c_source_counts,
        "quadrant_c_n": len(c_rows),
        "benchmark_sha256_verified": benchmark_sha,
        "per_quadrant_word_char_length_and_source_category": {
            "A": audit_quadrants["quadrant_audits"]["A"],
            "B": audit_quadrants["quadrant_audits"]["B"],
            "D": audit_quadrants["quadrant_audits"]["D"],
            "C_note": "quadrant C is out of scope for src/audit_existing_quadrants.py (A/B/D only); C's word/char length distribution is instead available from C-B's own descriptive stats (candidate side) and from quadrant_composition_report.json below.",
        },
        "per_quadrant_lexical_risk_lexicon_hit_rate": quadrant_composition["stats"],
        "quadrant_composition_report_predictions": quadrant_composition["predictions"],
    }


# ── 3. R-AUTHORED (unlabeled distributional evidence only) ────────────────
def load_r_authored_summary() -> Dict:
    queue_path = REPO_ROOT / "data/review/c_source_authored_review_queue.csv"
    raw_path = REPO_ROOT / "data/quadrant_c_pipeline/c_source_authored_candidates_raw_v1.jsonl"
    validated_path = REPO_ROOT / "data/quadrant_c_pipeline/c_source_authored_candidates_validated_v1.jsonl"

    if not queue_path.exists():
        return {
            "present": False,
            "missing_path": _display_path(queue_path),
            "note": "R-AUTHORED review-queue path not found; no comparison possible.",
        }

    import csv

    with open(queue_path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    def desc(field: str) -> Dict:
        vals = [float(r[field]) for r in rows if r.get(field) not in (None, "")]
        if not vals:
            return {"n": 0}
        return {
            "n": len(vals),
            "mean": stats.fmean(vals),
            "median": stats.median(vals),
            "sd": stats.pstdev(vals) if len(vals) > 1 else 0.0,
        }

    def counts(field: str) -> Dict[str, int]:
        out: Dict[str, int] = {}
        for r in rows:
            v = r.get(field)
            out[v] = out.get(v, 0) + 1
        return out

    return {
        "present": True,
        "queue_path": _display_path(queue_path),
        "queue_sha256": sha256_file(queue_path),
        "raw_candidates_path": _display_path(raw_path),
        "raw_candidates_sha256": sha256_file(raw_path),
        "validated_candidates_path": _display_path(validated_path),
        "validated_candidates_sha256": sha256_file(validated_path),
        "n_queued": len(rows),
        "review_status_counts": counts("review_status"),
        "source_dataset_counts": counts("source_dataset"),
        "classifier_status_counts": counts("classifier_status"),
        "provenance_class_counts": counts("provenance_class"),
        "aggregate_distributions_unpaired_unlabeled": {
            "word_count": desc("word_count"),
            "character_count": desc("character_count"),
            "fightin_words_score_normalized": desc("fightin_words_score_normalized"),
            "fw_z_score": desc("fw_z_score"),
        },
        "treated_as": "unlabeled external distributional evidence only (C-A section 8.6 tier 7); review_status is pending 52/52, per C-A section 3 -- no construct claim of any kind is made about this population",
        "not_tuned_on_this": "no C-B or C-C feature definition, threshold, or inclusion rule was adjusted based on these numbers",
        "comparability_caveats": [
            "word_count here is len(text.split()) (src/data_pipeline/score_and_queue_c_source_authored.py); R104's word_count_source/word_count_candidate columns in data/review/c_review_queue.csv were not verified to use the identical tokenization rule -- treat any numeric gap as approximate, not a controlled contrast.",
            "fightin_words_score_normalized/fw_z_score here were fit against H=quadrant A union quadrant B vs D=quadrant D (logs/3a4_scoring.md); R104's fightin_words feature in the C-B contract above was fit LOSO with StrongREJECT held out of H (C-B IMPLEMENTATION DECISION 1). These are two different fitted references -- the two fightin_words-family numbers are not on a directly comparable scale and must not be differenced against each other.",
            "the R-AUTHORED queue was explicitly rank-selected (Q25 = lowest-fightin_words-score quartile, i.e. most D-like) by the 3A4 scoring pipeline, not randomly sampled -- so this population's own fightin_words/fw_z_score distribution is a product of the selection rule, not an independent observation about the source-authored candidate pool as a whole. Comparing it to R104 without accounting for this selection would conflate a sampling artifact with a construction property.",
        ],
    }


# ── 4. Confounds (drawn only from numbers C-B already computed) ───────────
def build_confounds(r104_analysis: Dict) -> Dict:
    pops = r104_analysis["results"]
    formatting_zero_variance = ["has_bullet_marker", "has_numbered_step", "has_code_block"]

    length_sensitivity_by_population = {}
    for pop_name, pop in pops.items():
        length_sensitivity_by_population[pop_name] = {
            feat_name: {
                "spearman_corr_with_word_count_delta": feat["length_sensitivity"]["spearman_corr_with_word_count_delta"],
                "spearman_p_value": feat["length_sensitivity"]["spearman_p_value"],
            }
            for feat_name, feat in pop["features"].items()
            if feat.get("length_sensitivity") is not None
        }

    formatting_confound_summary = {}
    for pop_name, pop in pops.items():
        multi_sentence = pop["features"]["multi_sentence_flag"]
        formatting_confound_summary[pop_name] = {
            "zero_variance_indicators": formatting_zero_variance,
            "zero_variance_note": "0/n on both source and candidate sides in every population -- floor effect, uninformative for this resource, not evidence of absence of formatting differences in general",
            "multi_sentence_flag_delta_mean": multi_sentence["delta"]["mean"],
            "multi_sentence_flag_d_z": multi_sentence["paired_effect_size_dz"],
            "multi_sentence_flag_holm_adjusted_p": pop["multiple_comparison_correction"]["per_feature"]["multi_sentence_flag"]["adjusted_p_holm"],
        }

    return {
        "length_sensitivity": {
            "method": "Spearman correlation of each lexical-audit/distributional-exploratory feature's paired delta against the word_count paired delta, per C-A section 7.6 item 8 (already computed by C-B; not recomputed differently here)",
            "raw_association_by_population": length_sensitivity_by_population,
            "interpretation_note": "Descriptive association only, per C-B's own field-level note -- not a causal claim that length change caused any lexical feature's delta. Reported as raw/attenuated association, not adjusted causal estimate.",
        },
        "formatting_sensitivity": formatting_confound_summary,
        "source_association": {
            "applicable": False,
            "reason": pops["population_1_all_valid_accepted_pairs"]["source_sensitivity"]["reason"],
        },
        "category_robustness": {
            "applicable": "descriptive only, no formal test predeclared (C-A section 7.6 item 6)",
            "reason": "project_category is unbalanced across R104 (6-41 rows across 4 levels) and across quadrant A vs quadrant C at very different proportions (C-A section 1); per-feature category_sensitivity breakdowns exist in the underlying C-B JSON (results.<population>.features.<feature>.category_sensitivity) but are not aggregated into a formal robustness statistic here, consistent with C-A's own predeclaration that no such test was specified.",
        },
        "repeated_template_concentration": {
            "computed": False,
            "reason": "No existing repository artifact defines or computes a repeated-rewrite-template concentration metric for R104 or R-AUTHORED. Defining one now would be a new metric, which is out of scope for this execution-only task. This is reported as an evidence gap, not silently skipped.",
        },
    }


# ── 5. Decision status (reporting only; no KEEP/DROP label assigned) ──────
def build_decision_status(r104_analysis: Dict, confounds: Dict) -> Dict:
    pops = r104_analysis["results"]

    def strongest_significant(pop_name: str) -> Dict:
        pop = pops[pop_name]
        holm = pop["multiple_comparison_correction"]["per_feature"]
        best = None
        for feat_name, feat in pop["features"].items():
            dz = feat.get("paired_effect_size_dz")
            if dz is None:
                continue
            reject = holm.get(feat_name, {}).get("reject_at_alpha")
            if not reject:
                continue
            if best is None or abs(dz) > abs(best["d_z"]):
                best = {
                    "feature": feat_name,
                    "family": feat["family"],
                    "d_z": dz,
                    "holm_adjusted_p": holm[feat_name]["adjusted_p_holm"],
                    "mean_delta": feat["delta"]["mean"],
                }
        return best

    strongest_paired_signal = strongest_significant("population_1_all_valid_accepted_pairs")
    strongest_preserved_assistance_result = strongest_significant("population_2_assistance_type_preserved_yes")

    # Strongest confound: largest-magnitude significant length-sensitivity
    # correlation among lexical-audit/distributional features in
    # population 1, per C-B's own already-computed numbers.
    ls = confounds["length_sensitivity"]["raw_association_by_population"]["population_1_all_valid_accepted_pairs"]
    strongest_confound_feature = max(
        ls.items(), key=lambda kv: abs(kv[1]["spearman_corr_with_word_count_delta"])
    )
    strongest_confound = {
        "type": "length_sensitivity",
        "feature": strongest_confound_feature[0],
        "spearman_corr_with_word_count_delta": strongest_confound_feature[1]["spearman_corr_with_word_count_delta"],
        "spearman_p_value": strongest_confound_feature[1]["spearman_p_value"],
        "note": "Largest-magnitude length association among lexical-audit/distributional-exploratory features in population 1 (all 104 pairs); see also the multi_sentence_flag formatting confound, which is the single largest-|d_z| effect in the entire feature set and is a structural (not lexical-content) change.",
    }

    contradictory_findings = [
        (
            "fightin_words shows candidates scoring HIGHER (more harmful-associated wording) than sources "
            f"(population 1 mean delta = {pops['population_1_all_valid_accepted_pairs']['features']['fightin_words']['delta']['mean']:.3f}, "
            f"d_z = {pops['population_1_all_valid_accepted_pairs']['features']['fightin_words']['paired_effect_size_dz']:.3f}), "
            "which is the opposite direction implied by R104's own historical 'reduced_cue_source_rewrite' naming -- already flagged in 3f_a section 2.1 and independently reproduced here, not a new finding."
        ),
        (
            "In the same population, lexical_risk_hit_count (the fixed lexicon) moves in the opposite direction "
            f"(mean delta = {pops['population_1_all_valid_accepted_pairs']['features']['lexical_risk_hit_count']['delta']['mean']:.3f}, "
            f"d_z = {pops['population_1_all_valid_accepted_pairs']['features']['lexical_risk_hit_count']['paired_effect_size_dz']:.3f}) -- i.e. one lexical instrument reports candidates as 'more distinctively harmful-registered' while another reports them as 'triggering fewer fixed-lexicon risk terms.' Both are real, current numbers; they are not reconcilable into a single 'candidates are more/less cue-salient' statement without picking one instrument as authoritative, which this document does not do."
        ),
        (
            "R-AUTHORED's own fightin_words/fw_z_score distribution sits toward the D-like (benign-associated) end "
            "of its reference scale, which could misleadingly read as 'R-AUTHORED is lower-cue than R104' -- but "
            "this is a direct artifact of the Q25 rank-selection rule used to build the 52-row queue (see "
            "load_r_authored_summary comparability_caveats), not an independent distributional finding, and the "
            "two fightin_words scores are fit against different H/D references besides."
        ),
    ]

    cannot_establish = [
        "Whether R104's paired changes isolate surface-cue wording independent of the harmful/benign construct (C_cue) -- no resource in this repository measures C_cue directly (C-A section 5); this audit does not change that.",
        "Near-duplicate (as opposed to exact-string) overlap between R104 and quadrant A -- requires embedding-model inference, which is out of scope for this CPU-only task (C-A section 2, restated, not re-tested here).",
        "Any KEEP / KEEP AS SECONDARY / INCONCLUSIVE / DROP resource decision for R104, R-AUTHORED, 3D-B, or 3D-H (C-A section 7.7; explicitly deferred by both C-B and C-C).",
        "A formal statistical comparison between R104 and R-AUTHORED, or across A/B/C/D quadrants generally -- no such test is predeclared anywhere in this repository, and defining one is out of scope for this execution-only task.",
        "Repeated-rewrite-template concentration within R104 or R-AUTHORED -- no existing repository artifact defines this metric (see build_confounds).",
        "A construct claim of any kind about R-AUTHORED -- 100% of its 52-row queue remains review_status=pending; zero human review has occurred.",
    ]

    return {
        "keep_drop_label_assigned": False,
        "keep_drop_label_note": "C-C does not choose KEEP/DROP, per its own task brief and per C-A section 7.7.",
        "strongest_paired_signal_population_1_all_pairs": strongest_paired_signal,
        "strongest_preserved_assistance_result_population_2_yes": strongest_preserved_assistance_result,
        "strongest_confound": strongest_confound,
        "contradictory_findings": contradictory_findings,
        "what_this_audit_cannot_establish": cannot_establish,
    }


# ── assembly ────────────────────────────────────────────────────────────
def get_code_version() -> Dict:
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True).strip()
        dirty = bool(subprocess.check_output(["git", "status", "--porcelain"], cwd=REPO_ROOT, text=True).strip())
    except Exception as exc:  # pragma: no cover
        return {"c_c_generation_commit": None, "working_tree_dirty": None, "error": str(exc)}
    return {"c_c_generation_commit": commit, "working_tree_dirty": dirty}


def build_input_manifest(c_b_result: Dict, r_authored: Dict) -> Dict:
    pinned = dict(c_b_result["analysis"]["pinned_input_hashes_verified"])
    additional_c_b = dict(c_b_result["analysis"]["additional_recorded_input_hashes_not_pinned_by_c_a"])
    secondary = {}
    for key, rel in SECONDARY_INPUT_PATHS.items():
        secondary[rel] = sha256_file(_resolve(rel))
    if r_authored.get("present"):
        secondary[r_authored["queue_path"]] = r_authored["queue_sha256"]
        secondary[r_authored["raw_candidates_path"]] = r_authored["raw_candidates_sha256"]
        secondary[r_authored["validated_candidates_path"]] = r_authored["validated_candidates_sha256"]
    return {
        "pinned_by_c_a_section_7_1_fail_closed": pinned,
        "additional_recorded_by_c_b_not_pinned": additional_c_b,
        "additional_secondary_inputs_read_by_c_c_not_fail_closed": secondary,
        "note": (
            "The first group is fail-closed hash-verified by C-B's own module "
            "(abort on mismatch). The second and third groups are recorded for "
            "this audit's own reproducibility but are not fail-closed checks -- "
            "a mismatch there would mean C-C's secondary/R-AUTHORED evidence is "
            "stale relative to this manifest, not that the primary R104 result "
            "is invalid."
        ),
    }


def build_audit_summary_md(audit: Dict) -> str:
    lines: List[str] = [
        "# C-C -- Execute and Freeze the Existing-C Audit",
        "",
        "Status: execution/aggregation only. Reruns the locked C-B contract "
        "(`logs/c_existing_construction_audit_spec.md` section 7) unmodified "
        "and compiles already-existing secondary evidence. No analysis "
        "definition, feature, or inclusion rule is changed or added here. "
        "No KEEP/DROP resource decision is made.",
        "",
        f"C-B implementation commit: `{audit['provenance']['c_b_implementation_commit']}`",
        f"C-C parent commit: `{audit['provenance']['c_c_parent_commit']}`",
        f"Reproducibility check (byte-identical vs. committed C-B output, "
        f"excluding generation-commit fields): "
        f"`{audit['provenance']['reproducibility_check'].get('byte_identical_excluding_code_version')}`",
        "",
        "## 1. Primary: R104 paired analysis (locked C-B contract, re-executed)",
        "",
    ]
    for pop_name, pop in audit["primary_r104_paired_analysis"]["results"].items():
        lines.append(f"### {pop_name} (n={pop['n_valid_pairs']})")
        lines.append("")
        lines.append("| Feature | Family | mean(delta) | d_z | Holm-adj p |")
        lines.append("|---|---|---|---|---|")
        holm = pop["multiple_comparison_correction"]["per_feature"]
        for feat_name, feat in pop["features"].items():
            dz = feat["paired_effect_size_dz"]
            dz_str = f"{dz:.4f}" if dz is not None else "n/a"
            hp = holm.get(feat_name, {}).get("adjusted_p_holm")
            hp_str = f"{hp:.4g}" if hp is not None else "n/a"
            lines.append(
                f"| {feat_name} | {feat['family']} | {feat['delta']['mean']:.4f} | {dz_str} | {hp_str} |"
            )
        lines.append("")

    lines += [
        "## 2. Secondary A/B/C/D distributions (descriptive only; existing evidence)",
        "",
        audit["secondary_abcd_distributions"]["not_predeclared"],
        "",
        f"Quadrant C category composition (n={audit['secondary_abcd_distributions']['quadrant_c_n']}): "
        f"{audit['secondary_abcd_distributions']['quadrant_c_category_composition']}",
        "",
        f"Lexical-risk-lexicon hit rate by quadrant: "
        f"{audit['secondary_abcd_distributions']['per_quadrant_lexical_risk_lexicon_hit_rate']}",
        "",
        "## 3. Confounds",
        "",
        f"Repeated-template concentration: {audit['confounds']['repeated_template_concentration']}",
        "",
        "## 4. R-AUTHORED",
        "",
    ]
    ra = audit["r_authored"]
    if ra.get("present"):
        lines.append(
            f"Present, n={ra['n_queued']}, review_status={ra['review_status_counts']} -- "
            "treated as unlabeled distributional evidence only, per the caveats below."
        )
        lines.append("")
        for c in ra["comparability_caveats"]:
            lines.append(f"- {c}")
    else:
        lines.append(f"Missing: {ra.get('missing_path')}")
    lines.append("")

    lines += [
        "## 5. Decision status (no KEEP/DROP assigned)",
        "",
        f"Strongest paired signal (population 1): {audit['decision_status']['strongest_paired_signal_population_1_all_pairs']}",
        "",
        f"Strongest preserved-assistance result (population 2): {audit['decision_status']['strongest_preserved_assistance_result_population_2_yes']}",
        "",
        f"Strongest confound: {audit['decision_status']['strongest_confound']}",
        "",
        "Contradictory findings:",
        "",
    ]
    for c in audit["decision_status"]["contradictory_findings"]:
        lines.append(f"- {c}")
    lines.append("")
    lines.append("What this audit cannot establish:")
    lines.append("")
    for c in audit["decision_status"]["what_this_audit_cannot_establish"]:
        lines.append(f"- {c}")
    lines.append("")
    lines.append("**Stop.**")
    lines.append("")
    return "\n".join(lines)


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", default=DEFAULT_OUT_DIR)
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> Dict:
    args = parse_args(argv)
    out_dir = _resolve(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    c_b_result = run_locked_c_b_contract()
    secondary = load_secondary_abcd_distributions()
    r_authored = load_r_authored_summary()
    confounds = build_confounds(c_b_result["analysis"])
    decision_status = build_decision_status(c_b_result["analysis"], confounds)
    input_manifest = build_input_manifest(c_b_result, r_authored)

    audit = {
        "task": "C-C -- execute and freeze the existing-C audit",
        "spec_reference": "logs/c_existing_construction_audit_spec.md section 7 (locked by C-A; implemented by C-B; re-executed unmodified by C-C)",
        "provenance": {
            "c_b_implementation_commit": "4b700f1f8c41c828d068a9a3b3d723595320ae06",
            "c_c_parent_commit": get_code_version()["c_c_generation_commit"],
            "c_c_working_tree_dirty_at_generation": get_code_version()["working_tree_dirty"],
            "exact_command": c_b_result["exact_command"],
            "exact_configuration": {
                "bootstrap_seed": c_b.BOOTSTRAP_SEED,
                "n_bootstrap": c_b.N_BOOTSTRAP,
                "permutation_seed": c_b.PERMUTATION_SEED,
                "n_permutations": c_b.N_PERMUTATIONS,
                "holm_alpha": c_b.HOLM_ALPHA,
            },
            "software_versions": c_b_result["analysis"]["software_versions"],
            "reproducibility_check": c_b_result["reproducibility_check"],
        },
        "primary_r104_paired_analysis": c_b_result["analysis"],
        "secondary_abcd_distributions": secondary,
        "confounds": confounds,
        "r_authored": r_authored,
        "decision_status": decision_status,
        "explicit_non_actions": [
            "did not change any C-B feature definition, sign convention, seed, or inclusion rule",
            "did not add any new statistical test, effect size, or corrected comparison across A/B/C/D quadrants",
            "did not modify data/review/c_review_queue.csv, data/frozen_v2/*, data/review/c_source_authored_review_queue.csv, or any C-A section 7.1 pinned input",
            "did not create, rewrite, or score any prompt",
            "did not run model inference or GPU code, and did not access the web",
            "did not begin common-CUE/contrastive construction",
            "did not assign a KEEP/KEEP-AS-SECONDARY/INCONCLUSIVE/DROP label to any resource",
            "did not overwrite logs/c_b_paired_delta_analysis.md or logs/c_b_paired_delta_analysis.json",
        ],
    }

    (out_dir / "audit.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")
    (out_dir / "audit_summary.md").write_text(build_audit_summary_md(audit), encoding="utf-8")
    (out_dir / "input_manifest.json").write_text(json.dumps(input_manifest, indent=2), encoding="utf-8")

    print(f"audit_json={_display_path(out_dir / 'audit.json')}")
    print(f"audit_summary_md={_display_path(out_dir / 'audit_summary.md')}")
    print(f"input_manifest_json={_display_path(out_dir / 'input_manifest.json')}")
    print(f"c_b_implementation_commit={audit['provenance']['c_b_implementation_commit']}")
    print(f"reproducible={c_b_result['reproducibility_check']['byte_identical_excluding_code_version']}")

    return audit


if __name__ == "__main__":
    main()
