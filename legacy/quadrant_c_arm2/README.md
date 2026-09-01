# Quadrant C — Arm-2 (`c_source_authored`) Archive

## Status

Arm-2 / `c_source_authored` was a secondary, source-authored construction
explored for quadrant C. It is **deferred** from the current release and is
**not part of the active benchmark**. Arm-2 was **not proven inferior** to
R104 / `c_paired`; it was set aside for scope reasons, not on the basis of a
demonstrated quality or validity gap. It may be revisited in a future
release.

The active release uses **R104 / `c_paired`** as the sole active C
construction.

## What lives here

This directory holds the three Arm-2 implementation scripts that are
clearly specific to the deferred `c_source_authored` path, along with their
tests and the milestone 3B queue-validation logs:

```
legacy/quadrant_c_arm2/
├── README.md
├── src/data_pipeline/
│   ├── validate_c_source_authored_candidates.py
│   ├── score_and_queue_c_source_authored.py
│   └── validate_c_source_authored_review_queue.py
├── tests/data_pipeline/
│   ├── test_validate_c_source_authored_candidates.py
│   ├── test_score_and_queue_c_source_authored.py
│   └── test_validate_c_source_authored_review_queue.py
└── logs/
    ├── milestone_3b_arm2_queue_validation.json
    └── milestone_3b_arm2_queue_validation.md
```

Moving these files required fixing the cross-references between them (they
import each other) and the `REPO_ROOT` path depth in each, since both
changed as a result of the relocation. No other behavior was changed.

## What was intentionally left in place

The following remain in their original locations because they are shared,
load-bearing for active tooling, or out of scope for this archival pass —
this directory does not represent a complete excision of every Arm-2 trace,
only the retirement of Arm-2 from the active release execution path:

- `src/data_pipeline/build_c_source_authored_candidates.py`
- `data/quadrant_c_pipeline/c_source_authored_candidates_raw_v1.jsonl`
- `data/quadrant_c_pipeline/c_source_authored_candidates_validated_v1.jsonl`
- `data/review/c_source_authored_review_queue.csv`
- shared CUE / comparative-analysis modules
- the lexical-outlierness / 3D-H machinery imported by active R104 tooling
- historical comparative audit reports

## Provenance

Nothing here was deleted. This is a relocation for release-path clarity,
not a claim about the relative merit of the two arms.
