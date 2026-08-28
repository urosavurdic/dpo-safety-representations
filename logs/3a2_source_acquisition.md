# Milestone 3A2-0 — Reacquire and Verify Strict-Primary Source Inputs

**Branch:** `agent/c-quadrant-end-to-end-e0e2317a`
**HEAD:** `1e6e6d3ae749f05d04dae59f38de4fc095f4817e` (the committed 3A1C
consolidation)
**Contract source:** `logs/3a1c_handoff.json` → `required_3a2_input_files`
(read directly, not guessed or reconstructed from memory).

This sub-step only acquires and cryptographically verifies the two
strict-primary source files. No candidate construction, filtering,
scoring, or review-queue creation happened here.

---

## StrongREJECT

| | |
|---|---|
| Repository | `alexandrasouly/strongreject` |
| Requested revision | `f7cad6c17e624e21d8df2278e918ae1dddb4cb56` |
| Actual revision fetched | `f7cad6c17e624e21d8df2278e918ae1dddb4cb56` — **match** |
| URL | `https://raw.githubusercontent.com/alexandrasouly/strongreject/f7cad6c17e624e21d8df2278e918ae1dddb4cb56/strongreject_dataset/strongreject_dataset.csv` |
| Path | `data/raw/3a1a_source_cache/strongreject/strongreject_dataset.csv` |
| Acquisition method | `curl` against the commit-pinned raw URL (HTTP 200) |
| Byte size | 56,359 |
| Expected SHA-256 | `4dd70357e4ff8b5d0ba5ebafecab5d6dd5633ce8046e3dd1c8bd93e64de44381` |
| Actual SHA-256 | `4dd70357e4ff8b5d0ba5ebafecab5d6dd5633ce8046e3dd1c8bd93e64de44381` — **match** |
| Structural check | header `category, source, forbidden_prompt`; 313 rows (excl. header) — matches 3A1A's reported total |
| **Verification status** | **VERIFIED** |
| Discrepancy | None |

## SimpleSafetyTests

| | |
|---|---|
| Repository | `bertiev/SimpleSafetyTests` |
| Requested revision | `d7aee9a9422a5a5488f478fd79c2479c891c0f3b` |
| Actual revision fetched | `d7aee9a9422a5a5488f478fd79c2479c891c0f3b` — **match** |
| URL | `https://raw.githubusercontent.com/bertiev/SimpleSafetyTests/d7aee9a9422a5a5488f478fd79c2479c891c0f3b/SimpleSafetyTests%20-%20test%20cases.csv` |
| Path | `data/raw/3a1b_source_cache/simplesafetytests/SimpleSafetyTests - test cases.csv` |
| Acquisition method | `curl` against the commit-pinned raw URL, space percent-encoded (HTTP 200) |
| Byte size | 12,610 |
| Expected SHA-256 | `6d95a1301e0d0f3a3c4cf5392f4afff11ad6e3066f95d23aaa138d44aedf986c` |
| Actual SHA-256 | `6d95a1301e0d0f3a3c4cf5392f4afff11ad6e3066f95d23aaa138d44aedf986c` — **match** |
| Structural check | header `id, harm_area, counter, category, prompts_final`; 100 rows (excl. header) — matches 3A1B's reported total |
| **Verification status** | **VERIFIED** |
| Discrepancy | None |

---

## Acceptance result

**MET.** Both strict-primary source files are present and hash-verified
against the exact contract recorded in `logs/3a1c_handoff.json`. No
blocker to report for this sub-step.

## Notes

- Both files were fetched from **commit-pinned** `raw.githubusercontent.com`
  URLs (not branch refs), so a future re-fetch of the same URL should
  reproduce byte-identical content unless the upstream repository rewrites
  that specific commit.
- Row counts and headers match what `logs/3a1a_source_inventory.json` and
  `logs/3a1b_source_inventory.json` already reported. No individual prompt
  text was inspected or reproduced in this sub-step — out of scope for
  acquisition/verification, and unnecessary for it.
- The fetched CSVs live under `data/raw/`, which remains **gitignored**
  (`.gitignore` line 6) and is not part of this commit — only this
  acquisition record is committed, consistent with the project's existing
  convention that source caches are working files, not tracked artifacts.
- No candidate, benchmark, gate, classifier, C-paired, or model file was
  touched.

---

**Next milestone: 3A2-1** — construct the raw C-source-authored candidate
universe. Not begun in this sub-step.
