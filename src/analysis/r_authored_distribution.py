"""
C-E: R-AUTHORED distributional characterization.

Narrow, CPU-only, descriptive-only task. Computes, for the 52-row
R-AUTHORED review queue (data/review/c_source_authored_review_queue.csv,
100% review_status=pending), the same compatible feature families already
locked for the R104 paired-delta contract
(logs/c_existing_construction_audit_spec.md section 7.5), reusing existing
implementations unmodified:

  - word_count, character_count: reused directly from the queue CSV's own
    columns (same len(text.split()) / len(text) convention used
    project-wide, e.g. src/data_pipeline/score_and_queue_c_source_authored.py,
    src/audit_existing_quadrants.py, src/finalize_benchmark.py).
  - sentence_count: count of [.!?]+ matches on raw prompt text (identical
    rule to 3d_b's multi_sentence_rule, per C-A section 7.5).
  - mean_word_length, lexical_diversity: src/corpus_discrimination.py
    ::word_tokenize (word_tokenize_v1_lower_alphanum_apostrophe);
    lexical_diversity = len(set(tokens)) / len(tokens), the same formula
    C-A section 7.5 specifies (freshly authored there, reused verbatim
    here -- not redefined).
  - has_bullet_marker / has_numbered_step / has_code_block /
    multi_sentence_flag: exact regexes reused verbatim from
    logs/3d_b_lexical_outlierness_pilot.json's formatting_diagnostic_config.
  - lexical_risk_hit_count: src/diagnostics/score_lexical_risk_cues.py
    ::score_prompt, reused unmodified. Only the aggregate hit count is
    reported -- matched terms are never printed.
  - fightin_words_score_normalized / fw_z_score: reused as-is from the
    queue CSV's own pre-computed columns (src/data_pipeline/
    score_and_queue_c_source_authored.py, fit against H=quadrant A union
    quadrant B, D=quadrant D -- see logs/3a4_scoring.md). Reported as
    R-AUTHORED's own descriptive stats only. NEVER differenced against
    R104's fightin_words (fit LOSO with StrongREJECT held out of H, per
    C-B IMPLEMENTATION DECISION 1) -- those are two different fitted
    references and are explicitly not on a common scale. Every such
    cross-comparison in this report's output is the literal string
    "NOT COMPARABLE -- different fitted reference", never a computed
    number.
  - punctuation/question density: NOT COMPUTED. No implementation of this
    feature exists anywhere in this repository (checked before writing
    this module) and none is defined in the locked C-A section 7.5
    contract. Adding one here would be a new feature definition, which
    this task explicitly prohibits. Reported as an evidence gap.
  - cue_tfidf_logreg_margin: NOT COMPUTED. Not in this task's required
    feature-family list; no existing scored value for R-AUTHORED exists
    to reuse, and scoring it fresh for a population it has never been
    run on is out of scope for a "reuse existing implementations, add
    nothing new" task. Reported as an evidence gap, not silently omitted.

R104 comparison numbers (source side and candidate side) are NOT
recomputed here -- they are read directly from the already-committed
logs/c_b_paired_delta_analysis.json (population_1_all_valid_accepted_pairs
descriptive stats), which is the single existing source of truth for
those numbers. A/B/C/D comparison numbers are read directly from the
already-committed results/c_construction_audit/audit.json
secondary_abcd_distributions. Nothing here overwrites either file.

All comparisons are UNPAIRED and descriptive. Where a Cohen's d is
reported, it is the standard independent-samples ("pooled-SD") form, not
a paired d_z. No CI is reported for these unpaired comparisons -- the
locked C-A section 7.6 statistical contract only predeclares a CI method
(paired bootstrap) for R104's own within-pair deltas; extending that to
an unpaired comparison here would be a new statistical procedure, which
this task also prohibits. No p-value / significance test is computed for
the same reason.

R-AUTHORED is a Q25-selected subset (lowest-fightin_words-score quartile
of the eligible candidate pool, i.e. most D-like by that instrument, per
src/data_pipeline/score_and_queue_c_source_authored.py and
logs/3a4_scoring.md) -- NOT a representative sample of all source-authored
candidates. Nothing in this module uses R-AUTHORED's distribution to set
a threshold or tune any existing metric. review_status is pending for
100% of rows; nothing here is a construct claim, a promotion, or a
KEEP/DROP decision.

Deterministic. No stochastic procedure is used anywhere in this module,
so seed is recorded as null throughout.
"""
import csv
import hashlib
import json
import re
import statistics
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from src.corpus_discrimination import word_tokenize, TOKENIZER_VERSION  # noqa: E402
from src.diagnostics.score_lexical_risk_cues import score_prompt  # noqa: E402

INPUT_CSV = REPO_ROOT / "data/review/c_source_authored_review_queue.csv"
C_B_JSON = REPO_ROOT / "logs/c_b_paired_delta_analysis.json"
AUDIT_JSON = REPO_ROOT / "results/c_construction_audit/audit.json"

OUT_JSON = REPO_ROOT / "results/c_construction_audit/r_authored_distribution.json"
OUT_MD = REPO_ROOT / "results/c_construction_audit/r_authored_distribution.md"
OUT_PROVENANCE = REPO_ROOT / "results/c_construction_audit/r_authored_distribution_provenance.json"

# Exact regexes reused verbatim from
# logs/3d_b_lexical_outlierness_pilot.json -> confound_diagnostics ->
# formatting_diagnostic_config. Not redefined here.
BULLET_MARKER_REGEX = re.compile(r"(?m)^\s*[-*\u2022]\s+")
NUMBERED_STEP_REGEX = re.compile(r"(?m)^\s*\d+[\.\)]\s+")
CODE_BLOCK_MARKER = "```"
SENTENCE_SPLIT_REGEX = re.compile(r"[.!?]+")

NOT_COMPARABLE = "NOT COMPARABLE -- different fitted reference"


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def sentence_count(text: str) -> int:
    """Count of [.!?]+ matches on raw (non-normalized) text -- identical
    rule to 3d_b's multi_sentence_rule, applied as a count rather than a
    >=2 boolean, per C-A section 7.5."""
    return len(SENTENCE_SPLIT_REGEX.findall(text))


def mean_word_length(tokens):
    if not tokens:
        return 0.0
    return sum(len(t) for t in tokens) / len(tokens)


def lexical_diversity(tokens):
    if not tokens:
        return 0.0
    return len(set(tokens)) / len(tokens)


def describe(values):
    values = [float(v) for v in values]
    n = len(values)
    if n == 0:
        return {"n": 0, "mean": None, "median": None, "sd": None, "iqr": None}
    mean = statistics.fmean(values)
    median = statistics.median(values)
    sd = statistics.stdev(values) if n > 1 else 0.0
    sorted_v = sorted(values)

    def pct(p):
        if n == 1:
            return sorted_v[0]
        k = (n - 1) * p
        f = int(k)
        c = min(f + 1, n - 1)
        if f == c:
            return sorted_v[f]
        return sorted_v[f] + (sorted_v[c] - sorted_v[f]) * (k - f)

    iqr = pct(0.75) - pct(0.25)
    return {"n": n, "mean": mean, "median": median, "sd": sd, "iqr": iqr}


def cohens_d_unpaired(a_stats, b_stats):
    """Independent-samples ('pooled-SD') Cohen's d: (mean_a - mean_b) /
    pooled_sd. Descriptive only -- no CI, no significance test (see
    module docstring for why)."""
    n1, n2 = a_stats["n"], b_stats["n"]
    if n1 < 2 or n2 < 2:
        return None
    s1, s2 = a_stats["sd"], b_stats["sd"]
    pooled_sd = (((n1 - 1) * s1 ** 2 + (n2 - 1) * s2 ** 2) / (n1 + n2 - 2)) ** 0.5
    if pooled_sd == 0:
        return None
    return (a_stats["mean"] - b_stats["mean"]) / pooled_sd


def load_r_authored_rows():
    with open(INPUT_CSV, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    return rows


def compute_r_authored_features(rows):
    """Per-row feature computation. Returns dict of feature_name ->
    list of values, plus row-level review_status counts. No prompt text
    or matched lexicon terms are retained past this function."""
    feature_values = {
        "word_count": [],
        "character_count": [],
        "sentence_count": [],
        "mean_word_length": [],
        "lexical_diversity": [],
        "lexical_risk_hit_count": [],
        "has_bullet_marker": [],
        "has_numbered_step": [],
        "has_code_block": [],
        "multi_sentence_flag": [],
        # reused as-is from the CSV, not recomputed
        "fightin_words_score_normalized": [],
        "fw_z_score": [],
    }
    review_status_counts = {}
    for r in rows:
        text = r["scored_prompt"]
        assert r["source_prompt"] == r["candidate_prompt"] == r["scored_prompt"], (
            "R-AUTHORED rows are expected to have identical source/candidate/"
            "scored prompt fields (single authored text, no rewrite step); "
            f"record_id={r.get('record_id')} violates this."
        )
        tokens = word_tokenize(text)
        n_hits, _matched_terms_never_stored = score_prompt(text)

        feature_values["word_count"].append(int(r["word_count"]))
        feature_values["character_count"].append(int(r["character_count"]))
        feature_values["sentence_count"].append(sentence_count(text))
        feature_values["mean_word_length"].append(mean_word_length(tokens))
        feature_values["lexical_diversity"].append(lexical_diversity(tokens))
        feature_values["lexical_risk_hit_count"].append(n_hits)
        feature_values["has_bullet_marker"].append(
            1 if BULLET_MARKER_REGEX.search(text) else 0
        )
        feature_values["has_numbered_step"].append(
            1 if NUMBERED_STEP_REGEX.search(text) else 0
        )
        feature_values["has_code_block"].append(1 if CODE_BLOCK_MARKER in text else 0)
        feature_values["multi_sentence_flag"].append(
            1 if sentence_count(text) >= 2 else 0
        )
        feature_values["fightin_words_score_normalized"].append(
            float(r["fightin_words_score_normalized"])
        )
        feature_values["fw_z_score"].append(float(r["fw_z_score"]))

        review_status_counts[r["review_status"]] = (
            review_status_counts.get(r["review_status"], 0) + 1
        )

    return feature_values, review_status_counts


def load_r104_reference():
    """Reuse -- not recompute -- R104's already-committed descriptive
    stats (population_1_all_valid_accepted_pairs, source and candidate
    sides) from logs/c_b_paired_delta_analysis.json."""
    with open(C_B_JSON, "r", encoding="utf-8") as f:
        d = json.load(f)
    feats = d["results"]["population_1_all_valid_accepted_pairs"]["features"]
    out = {}
    for feat_name, feat in feats.items():
        out[feat_name] = {"source": feat.get("source"), "candidate": feat.get("candidate")}
    return out, d["code_version"]["generation_commit"]


def load_abcd_reference():
    """Reuse -- not recompute -- the already-committed A/B/C/D aggregate
    numbers from results/c_construction_audit/audit.json."""
    with open(AUDIT_JSON, "r", encoding="utf-8") as f:
        d = json.load(f)
    sd = d["secondary_abcd_distributions"]
    return (
        sd["per_quadrant_word_char_length_and_source_category"],
        sd["per_quadrant_lexical_risk_lexicon_hit_rate"],
    )


# Features compatible between R-AUTHORED and R104 source/candidate (same
# definitions, reused implementations on both sides).
COMPATIBLE_WITH_R104 = [
    "word_count",
    "character_count",
    "sentence_count",
    "mean_word_length",
    "lexical_diversity",
    "lexical_risk_hit_count",
]

# fightin_words / fw_z_score exist on both sides but use different fitted
# references -- always NOT COMPARABLE, never differenced.
NOT_COMPARABLE_FEATURES = ["fightin_words_score_normalized_vs_fightin_words", "fw_z_score"]


def build_report():
    if not INPUT_CSV.exists():
        raise FileNotFoundError(f"required input not found: {INPUT_CSV}")

    input_sha256 = file_sha256(INPUT_CSV)
    rows = load_r_authored_rows()
    row_count = len(rows)

    feature_values, review_status_counts = compute_r_authored_features(rows)
    r_authored_stats = {name: describe(vals) for name, vals in feature_values.items()}

    r104_ref, c_b_commit = load_r104_reference()
    abcd_length, abcd_lexrisk = load_abcd_reference()

    comparisons_vs_r104 = {}
    for feat in COMPATIBLE_WITH_R104:
        ra = r_authored_stats[feat]
        r104_feat = r104_ref.get(feat)
        entry = {"r_authored": ra}
        if r104_feat is None:
            entry["r104_source"] = None
            entry["r104_candidate"] = None
            entry["note"] = "feature not present in c_b_paired_delta_analysis.json"
        else:
            entry["r104_source"] = r104_feat["source"]
            entry["r104_candidate"] = r104_feat["candidate"]
            entry["cohens_d_vs_r104_source"] = (
                cohens_d_unpaired(ra, r104_feat["source"])
                if r104_feat["source"] else None
            )
            entry["cohens_d_vs_r104_candidate"] = (
                cohens_d_unpaired(ra, r104_feat["candidate"])
                if r104_feat["candidate"] else None
            )
            entry["ci_note"] = (
                "no CI reported -- locked C-A section 7.6 contract only "
                "predeclares a CI method for R104's own paired deltas, "
                "not for this unpaired comparison"
            )
        comparisons_vs_r104[feat] = entry

    fightin_words_comparison = {
        "r_authored_fightin_words_score_normalized": r_authored_stats[
            "fightin_words_score_normalized"
        ],
        "r_authored_fw_z_score": r_authored_stats["fw_z_score"],
        "r104_fightin_words_source": r104_ref.get("fightin_words", {}).get("source"),
        "r104_fightin_words_candidate": r104_ref.get("fightin_words", {}).get("candidate"),
        "r104_fw_z_score_source": r104_ref.get("fw_z_score", {}).get("source"),
        "r104_fw_z_score_candidate": r104_ref.get("fw_z_score", {}).get("candidate"),
        "comparison_vs_r104_fightin_words": NOT_COMPARABLE,
        "comparison_vs_r104_fw_z_score": NOT_COMPARABLE,
        "reason": (
            "R-AUTHORED's fightin_words_score_normalized/fw_z_score were fit "
            "against H=quadrant A union quadrant B, D=quadrant D "
            "(logs/3a4_scoring.md). R104's fightin_words feature in the "
            "locked C-B contract was fit LOSO with StrongREJECT held out of "
            "H (C-B IMPLEMENTATION DECISION 1). Different fitted references, "
            "not a common scale -- reported separately, never differenced."
        ),
    }

    comparisons_vs_abcd = {
        "word_count": {
            "r_authored": r_authored_stats["word_count"],
            "quadrant_A": {"n": abcd_length["A"]["total_rows"],
                            "mean_words": abcd_length["A"]["word_length"]["mean"]},
            "quadrant_B": {"n": abcd_length["B"]["total_rows"],
                            "mean_words": abcd_length["B"]["word_length"]["mean"]},
            "quadrant_D": {"n": abcd_length["D"]["total_rows"],
                            "mean_words": abcd_length["D"]["word_length"]["mean"]},
            "quadrant_C_note": (
                "quadrant C word-length stats equal R104's candidate-side "
                "stats already reported above (quadrant C IS R104's 104 "
                "accepted candidates) -- see comparisons_vs_r104.word_count."
            ),
        },
        "lexical_risk_hit_rate": {
            "r_authored": {
                "n": r_authored_stats["lexical_risk_hit_count"]["n"],
                "mean_cue_hits": r_authored_stats["lexical_risk_hit_count"]["mean"],
                "pct_with_cue_hit": (
                    100.0
                    * sum(1 for v in feature_values["lexical_risk_hit_count"] if v > 0)
                    / len(feature_values["lexical_risk_hit_count"])
                ),
            },
            "quadrant_A": abcd_lexrisk["A"],
            "quadrant_B": abcd_lexrisk["B"],
            "quadrant_C": abcd_lexrisk["C"],
            "quadrant_D": abcd_lexrisk["D"],
        },
    }

    not_computed = {
        "punctuation_question_density": (
            "NOT COMPUTED -- no implementation of a punctuation/question "
            "density feature exists anywhere in this repository, and none "
            "is defined in the locked C-A section 7.5 contract. Defining "
            "one here would be a new feature definition, out of scope for "
            "this task. Evidence gap, not a silent omission."
        ),
        "cue_tfidf_logreg_margin": (
            "NOT COMPUTED -- not in this task's required feature-family "
            "list; no existing scored value for R-AUTHORED to reuse; "
            "scoring it fresh for an unscored population is out of scope "
            "for a reuse-only task. Evidence gap, not a silent omission."
        ),
        "near_duplicate_check": (
            "NOT RUN -- embedding-model inference is out of scope for this "
            "CPU-only task, consistent with the same gap already noted for "
            "R104/quadrant A/B/D in the prior C-C audit."
        ),
    }

    selection_bias_warning = (
        "R-AUTHORED is a Q25-selected subset (lowest-fightin_words-score "
        "quartile of the eligible source-authored candidate pool, i.e. "
        "most D-like by that instrument, per "
        "src/data_pipeline/score_and_queue_c_source_authored.py and "
        "logs/3a4_scoring.md), not a random or representative sample of "
        "all source-authored candidates. This report's distribution must "
        "not be used to define a new threshold, must not be used to tune "
        "any existing metric, and must not be described as an independent "
        "external validation set for the current audit. review_status is "
        "pending for 100% of rows -- these are UNVALIDATED, not accepted, "
        "C labels."
    )

    report = {
        "task": "C-E -- R-AUTHORED distributional characterization",
        "spec_reference": (
            "logs/c_existing_construction_audit_spec.md section 7.5 "
            "(feature definitions reused unmodified); "
            "results/c_construction_audit/audit.json and "
            "logs/c_b_paired_delta_analysis.json (R104 reference numbers, "
            "reused not recomputed)"
        ),
        "input": {
            "path": "data/review/c_source_authored_review_queue.csv",
            "sha256": input_sha256,
            "row_count": row_count,
            "review_status_counts": review_status_counts,
        },
        "selection_bias_warning": selection_bias_warning,
        "r_authored_descriptive_stats": r_authored_stats,
        "comparisons_vs_r104_unpaired": comparisons_vs_r104,
        "fightin_words_comparison": fightin_words_comparison,
        "comparisons_vs_abcd": comparisons_vs_abcd,
        "not_computed": not_computed,
        "decision_status": {
            "keep_drop_label_assigned": False,
            "note": "C-E is descriptive characterization only; no KEEP/DROP "
                    "or promote/reject decision is made here.",
        },
        "no_modification_confirmation": {
            "r_authored_records_modified": False,
            "frozen_benchmark_modified": False,
            "c_b_contract_modified": False,
            "previous_c_c_outputs_modified": False,
        },
    }
    return report, c_b_commit


def render_markdown(report):
    ra = report["r_authored_descriptive_stats"]
    lines = []
    lines.append("# C-E -- R-AUTHORED Distributional Characterization")
    lines.append("")
    lines.append(
        "Status: descriptive characterization only. No promotion/rejection, "
        "no KEEP/DROP decision, no new prompts, no modification of "
        "R-AUTHORED records or the frozen benchmark."
    )
    lines.append("")
    lines.append(f"Input: `{report['input']['path']}`")
    lines.append(f"Input SHA-256: `{report['input']['sha256']}`")
    lines.append(f"Row count: {report['input']['row_count']}")
    lines.append(
        f"review_status counts: {report['input']['review_status_counts']}"
    )
    lines.append("")
    lines.append("> " + report["selection_bias_warning"])
    lines.append("")
    lines.append("## 1. R-AUTHORED descriptive statistics (n=%d)" % report["input"]["row_count"])
    lines.append("")
    lines.append("| Feature | n | mean | median | sd | IQR |")
    lines.append("|---|---|---|---|---|---|")
    for feat in [
        "word_count", "character_count", "sentence_count", "mean_word_length",
        "lexical_diversity", "lexical_risk_hit_count", "has_bullet_marker",
        "has_numbered_step", "has_code_block", "multi_sentence_flag",
        "fightin_words_score_normalized", "fw_z_score",
    ]:
        s = ra[feat]
        lines.append(
            f"| {feat} | {s['n']} | {s['mean']:.4f} | {s['median']:.4f} | "
            f"{s['sd']:.4f} | {s['iqr']:.4f} |"
        )
    lines.append("")
    lines.append(
        "`fightin_words_score_normalized`/`fw_z_score` are reused as-is "
        "from the queue CSV. See section 3 -- these are NOT comparable to "
        "R104's fightin_words numbers."
    )
    lines.append("")
    lines.append("## 2. Unpaired comparison vs. R104 (source / candidate), compatible features only")
    lines.append("")
    lines.append(
        "Descriptive only. `NOT COMPARABLE` never appears here -- these "
        "six features share an identical definition and implementation on "
        "both sides. Effect size is unpaired (independent-samples) Cohen's "
        "d, not R104's own paired d_z. No CI/p-value: not predeclared for "
        "an unpaired comparison by the locked contract."
    )
    lines.append("")
    lines.append(
        "| Feature | R-AUTHORED mean (n=%d) | R104 source mean (n=104) | "
        "R104 candidate mean (n=104) | d vs. source | d vs. candidate |"
        % report["input"]["row_count"]
    )
    lines.append("|---|---|---|---|---|---|")
    for feat, entry in report["comparisons_vs_r104_unpaired"].items():
        ra_mean = entry["r_authored"]["mean"]
        src = entry.get("r104_source")
        cand = entry.get("r104_candidate")
        src_mean = f"{src['mean']:.4f}" if src else "n/a"
        cand_mean = f"{cand['mean']:.4f}" if cand else "n/a"
        d_src = entry.get("cohens_d_vs_r104_source")
        d_cand = entry.get("cohens_d_vs_r104_candidate")
        d_src_s = f"{d_src:.4f}" if d_src is not None else "n/a"
        d_cand_s = f"{d_cand:.4f}" if d_cand is not None else "n/a"
        lines.append(
            f"| {feat} | {ra_mean:.4f} | {src_mean} | {cand_mean} | "
            f"{d_src_s} | {d_cand_s} |"
        )
    lines.append("")
    lines.append("## 3. Fightin' Words / fw_z_score -- restricted comparison")
    lines.append("")
    fw = report["fightin_words_comparison"]
    lines.append(f"- R-AUTHORED `fightin_words_score_normalized`: {NOT_COMPARABLE}")
    lines.append(f"- R-AUTHORED `fw_z_score`: {NOT_COMPARABLE}")
    lines.append(f"- Reason: {fw['reason']}")
    lines.append("")
    lines.append("## 4. Comparison vs. frozen A/B/C/D populations (compatible aggregate features)")
    lines.append("")
    wc = report["comparisons_vs_abcd"]["word_count"]
    lines.append("Word count (mean words):")
    lines.append(
        f"- R-AUTHORED: n={wc['r_authored']['n']}, mean={wc['r_authored']['mean']:.2f}"
    )
    lines.append(f"- Quadrant A: n={wc['quadrant_A']['n']}, mean={wc['quadrant_A']['mean_words']}")
    lines.append(f"- Quadrant B: n={wc['quadrant_B']['n']}, mean={wc['quadrant_B']['mean_words']}")
    lines.append(f"- Quadrant D: n={wc['quadrant_D']['n']}, mean={wc['quadrant_D']['mean_words']}")
    lines.append(f"- {wc['quadrant_C_note']}")
    lines.append("")
    lr = report["comparisons_vs_abcd"]["lexical_risk_hit_rate"]
    lines.append("Lexical-risk-lexicon hit rate:")
    lines.append(
        f"- R-AUTHORED: n={lr['r_authored']['n']}, mean_cue_hits="
        f"{lr['r_authored']['mean_cue_hits']:.4f}, pct_with_cue_hit="
        f"{lr['r_authored']['pct_with_cue_hit']:.1f}"
    )
    for q in ["quadrant_A", "quadrant_B", "quadrant_C", "quadrant_D"]:
        v = lr[q]
        lines.append(
            f"- {q}: n={v['n']}, mean_cue_hits={v['mean_cue_hits']}, "
            f"pct_with_cue_hit={v['pct_with_cue_hit']}"
        )
    lines.append("")
    lines.append("## 5. Not computed (evidence gaps)")
    lines.append("")
    for k, v in report["not_computed"].items():
        lines.append(f"- **{k}**: {v}")
    lines.append("")
    lines.append("## 6. Decision status")
    lines.append("")
    lines.append(
        "No KEEP/DROP or promote/reject decision is made in this document. "
        "review_status remains pending for all 52 rows."
    )
    lines.append("")
    lines.append("**Stop.**")
    return "\n".join(lines) + "\n"


def main():
    report, c_b_commit = build_report()
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)

    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, sort_keys=False)
        f.write("\n")

    md = render_markdown(report)
    with open(OUT_MD, "w", encoding="utf-8") as f:
        f.write(md)

    try:
        generation_commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT
        ).decode().strip()
    except Exception:
        generation_commit = None

    provenance = {
        "task": "C-E -- R-AUTHORED distributional characterization",
        "input": {
            "path": "data/review/c_source_authored_review_queue.csv",
            "sha256": report["input"]["sha256"],
            "row_count": report["input"]["row_count"],
        },
        "feature_definitions_reused_from": {
            "word_count": "existing CSV column (len(text.split()) convention, "
                           "src/data_pipeline/score_and_queue_c_source_authored.py)",
            "character_count": "existing CSV column (len(text) convention, same source)",
            "sentence_count": "logs/c_existing_construction_audit_spec.md section 7.5 "
                               "([.!?]+ match count on raw text, same rule as 3d_b's "
                               "multi_sentence_rule)",
            "mean_word_length": "src/corpus_discrimination.py::word_tokenize "
                                 f"({TOKENIZER_VERSION})",
            "lexical_diversity": "logs/c_existing_construction_audit_spec.md section 7.5 "
                                  "(len(set(tokens))/len(tokens), same word_tokenize)",
            "has_bullet_marker/has_numbered_step/has_code_block/multi_sentence_flag": (
                "logs/3d_b_lexical_outlierness_pilot.json -> confound_diagnostics -> "
                "formatting_diagnostic_config, regexes reused verbatim"
            ),
            "lexical_risk_hit_count": "src/diagnostics/score_lexical_risk_cues.py"
                                       "::score_prompt, reused unmodified",
            "fightin_words_score_normalized/fw_z_score": "existing CSV columns, reused "
                                                           "as-is, never recomputed",
        },
        "r104_reference_source": {
            "path": "logs/c_b_paired_delta_analysis.json",
            "c_b_implementation_commit": c_b_commit,
            "note": "R104 numbers are read from this committed file, not recomputed",
        },
        "abcd_reference_source": {
            "path": "results/c_construction_audit/audit.json",
            "note": "A/B/C/D numbers are read from this committed file, not recomputed",
        },
        "software_versions": {
            "python": sys.version,
        },
        "seed": None,
        "seed_note": "no stochastic procedure is used anywhere in this module",
        "deterministic": True,
        "source_commit_at_generation": generation_commit,
        "no_prompt_text_in_outputs": True,
        "outputs": {
            "json": str(OUT_JSON.relative_to(REPO_ROOT)),
            "md": str(OUT_MD.relative_to(REPO_ROOT)),
            "provenance": str(OUT_PROVENANCE.relative_to(REPO_ROOT)),
        },
    }
    try:
        provenance["software_versions"]["numpy"] = __import__("numpy").__version__
        provenance["software_versions"]["scipy"] = __import__("scipy").__version__
        provenance["software_versions"]["pandas"] = __import__("pandas").__version__
    except Exception:
        pass

    with open(OUT_PROVENANCE, "w", encoding="utf-8") as f:
        json.dump(provenance, f, indent=2, sort_keys=False)
        f.write("\n")

    print(f"wrote {OUT_JSON}")
    print(f"wrote {OUT_MD}")
    print(f"wrote {OUT_PROVENANCE}")


if __name__ == "__main__":
    main()
