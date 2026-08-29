# CUE Scorers — Implementation (src/cue_scoring.py)

Implements the two CUE scoring methods using the reference-data design
recorded in `logs/cue_reference_audit.md`. Implementation and focused
unit tests only — no final candidate batch is scored or frozen by this
milestone (see "Not performed this session" below).

## What was built

- `src/cue_scoring.py`
  - **Method 1 — `TfidfLogRegCueScorer`**: TF-IDF (word + character
    n-grams, via a `FeatureUnion` of two `TfidfVectorizer`s) →
    `LogisticRegression`. `score()` returns the raw `decision_function`
    margin (`tfidf_logreg_score_margin`) as the monotonic
    harmful-association score, plus an uncalibrated sigmoid reported
    separately and explicitly flagged
    `tfidf_logreg_score_is_calibrated_probability: False` — no
    calibration step is performed, per the task's "preferably the
    classifier logit/margin rather than treating an uncalibrated
    probability as literal probability" instruction.
  - **Method 2 — Fightin' Words**: reused unchanged from
    `src/corpus_discrimination.FightinWords` — not reimplemented. Its
    existing predeclared parameters (`prior_strength=0.01`,
    `min_count=1`, `min_token_recognition_fraction=0.5`) are carried
    into `FROZEN_CUE_CONFIG["fightin_words"]` for a single point of
    reference alongside Method 1's parameters.
  - `leave_one_source_out_folds()` — general LOSO fold builder over
    named text pools.
  - `grouped_kfold_indices()` — fallback dedup-group-aware fold builder
    (via `sklearn.GroupKFold`, keyed on `normalized_sha256` reused
    verbatim from `build_c_source_authored_candidates.py`) for any
    future pool without cleanly distinct sources, so duplicate/template
    rows never split across a train/held-out boundary.
  - `score_harmful_reference_sources_loso()` — orchestrates the 3-fold
    LOSO plan across `{HarmBench, StrongREJECT, SimpleSafetyTests}`
    (fixed `XSTest` + quadrant D on the benign side, per
    `cue_reference_audit.md`), scoring every harmful reference item
    out-of-fold with both methods and computing per-item empirical rank
    + agreement.
  - `compute_agreement()` — wires the predeclared-but-previously-unwired
    `max_metric_rank_disagreement` gate field (`logs/benchmark_gate_config.json`,
    read-only here) into an actual per-item agreement check between the
    two methods' empirical ranks.
  - `load_reference_texts_from_repo()` — loads the reference pools from
    already-tracked, already-validated repository artifacts only
    (`data/processed/controlled_eval.jsonl`,
    `data/quadrant_c_pipeline/c_source_authored_candidates_validated_v1.jsonl`,
    filtered to `candidate_universe_status == "eligible_for_3a3"`) — no
    new acquisition, no `data/raw/` access, no re-running 3A2/3A3.
  - `XSTEST_LIMITATION_NOTE` — states plainly, per `cue_reference_audit.md`'s
    explicit instruction not to paper over this: XSTest has no LOSO fold
    (only accessible benign-high-register source), so this module does
    not produce an XSTest CUE score.

- `tests/test_cue_scoring.py` — 20 focused unit tests, synthetic data
  only (no dependency on real repo files or network), covering exactly
  the four categories the task calls for: deterministic scoring,
  empty/short prompts, obvious synthetic ordering, and leakage-safe fold
  construction (both `leave_one_source_out_folds` and
  `grouped_kfold_indices`, plus the LOSO orchestrator wiring).

## Frozen configuration

Predeclared in `FROZEN_CUE_CONFIG` (`config_version: "cue_scorer_v1"`)
before any real candidate batch is scored:

- **TF-IDF+LR**: word 1–2-grams (`min_df=2`) + char 3–5-grams
  (`char_wb`, `min_df=2`), `sublinear_tf=True`, `max_features=20000` per
  view, `LogisticRegression(C=1.0, penalty="l2", class_weight="balanced",
  max_iter=2000, random_state=20260829)`.
- **Fightin' Words**: `prior_strength=0.01`, `min_count=1`,
  `min_token_recognition_fraction=0.5` (unchanged existing defaults).
- **Leakage control**: 3-fold LOSO over the harmful sources; XSTest fixed
  (not held out, disclosed limitation); quadrant D fixed;
  `max_metric_rank_disagreement` read from
  `logs/benchmark_gate_config.json` (default `0.25` if absent).

Only prompt text is used as a model feature in either method; source and
category fields are used exclusively to build folds, never as features
(verified by the LOSO orchestration test asserting `reference_n_h`/
`reference_n_d` match the expected fold sizes with the held-out source's
own count excluded).

## Verification this session

- `pytest tests/test_cue_scoring.py` — 20/20 passed.
- `pytest tests/test_cue_scoring.py tests/data_pipeline/test_score_and_queue_c_source_authored.py tests/data_pipeline/test_build_c_source_authored_candidates.py` —
  58/58 passed (focused regression check on the infrastructure this
  module reuses; the full historical suite was intentionally not run,
  per task scope).
- One informal smoke check against the real repository reference data
  (`load_reference_texts_from_repo()` → `score_harmful_reference_sources_loso()`):
  150 HarmBench + 132 StrongREJECT-eligible + 77 SimpleSafetyTests-eligible
  = 359 items scored out-of-fold; ~69.4% of items had the two methods'
  empirical ranks agree within the 0.25 gate threshold. This confirms
  the implementation runs end-to-end on real data — it is **not** a
  scoring artifact and produced no output file; no Q10/Q25/Q40
  stratification or review queue was generated.

## Not performed this session

Scoring or freezing a final candidate batch; any Q10/Q25/Q40
stratification of harmful-reference or C-source-authored items; an
XSTest CUE score (see `XSTEST_LIMITATION_NOTE` — not attempted, not
silently skipped); any modification to `data/frozen_v2/*`,
`logs/benchmark_gate_config.json`, or any other frozen artifact; the
full historical test suite (`pytest tests/ -v`) — only the focused
checks above were run, per task instruction.

**Next milestone (not started):** run the frozen scorers over the full
candidate pools intended for actual quadrant assignment, apply
Q10/Q25/Q40 stratification via the existing `assign_strata()`, and build
the resulting human review queue (mirroring the existing
`c_source_authored_review_queue.csv` pattern).
