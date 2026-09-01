# TASK 1 — CPU/test hardening report

Parent commit: `32d109dc3dd4685eb6bf86340a3bb6dc3b6f3ef7` (R104 blind human
review commit, branch `agent/c-quadrant-end-to-end-e0e2317a`).

## 1. Windows path-separator failures — `src/validate_benchmark_v2.py`

**Root cause:** the per-stage missing-artifact list was built with
`str(path)` on `pathlib.Path` objects. `str(Path(...))` renders with the
host OS separator, so on Windows it silently produced
`results\activations\M0_metadata.json` instead of the repository's
stable-path contract `results/activations/M0_metadata.json`. Invisible
on POSIX CI/dev machines because `str()` and `.as_posix()` coincide
there — confirmed: all 11 pre-existing tests in
`tests/test_validate_benchmark_v2.py` already passed on this Linux
sandbox before any fix.

**Fix:** extracted the four-artifact existence check into
`_stage_artifact_missing(activation_dir, stage)`, which now returns
`path.as_posix()` (guaranteed forward-slash on every platform),
matching the `.as_posix()` convention already used elsewhere in the
same file (`benchmark_path.as_posix()` etc.).

**Regression coverage added:** two tests in
`tests/test_validate_benchmark_v2.py`.
`test_missing_stage_artifacts_reports_posix_paths_even_on_windows_like_path`
uses a `_FakeWindowsDir`/`_FakeWindowsChildPath` stand-in whose `str()`
mimics real `WindowsPath` backslash behavior (delegating `.exists()` to
a real path), so a POSIX host can still fail this test if someone
reverts to `str(path)` — verified by temporarily reverting the fix and
confirming the new test (and only that test) failed, then restoring it.
`test_missing_stage_artifacts_omits_present_files` covers the
present/missing filtering itself.

**Count on the original Windows run:** 8 distinct pytest IDs (see
section 4).

## 2. Windows UTF-8 decoding failure — `src/corpus_discrimination.py`

**Root cause:** `load_quadrant_texts` opened the eval-set file with
`open(eval_path)` — no explicit encoding — so it decoded using the
platform default (locale-dependent on Windows, e.g. a `cp1252` code
page), which raises `UnicodeDecodeError` on the repository's non-ASCII
prompt text. Repository JSONL/text inputs are UTF-8 (see
`src/v2_io.py`'s `load_json`/`load_jsonl`, which already use
`encoding="utf-8"`).

**Fix:** `open(eval_path, encoding="utf-8")`.

**Regression coverage added:** new file
`tests/test_corpus_discrimination.py`. This sandbox runs Python in
UTF-8 mode (`sys.flags.utf8_mode == 1`), which overrides
`locale.getpreferredencoding()` entirely, so a locale-monkeypatch
approach cannot reproduce the real Windows bug here (verified: patching
`locale.getpreferredencoding` had no effect on `open()`'s resolved
encoding on this host). The regression guard instead mocks
`builtins.open` and asserts the call site passes `encoding="utf-8"`
explicitly — the fix that is correct on every platform regardless of
locale. A second test reads real non-ASCII content (curly quotes, an
em dash, CJK text) filtered by quadrant. Verified by temporarily
reverting the fix and confirming the encoding-assertion test failed,
then restoring it.

**Ripple effect (see section 3 for full detail):** `src/corpus_
discrimination.py` is one of the C-A section 7.1 pinned inputs read by
`src/analysis/c_b_paired_delta_analysis.py` and
`src/analysis/cf_joint_geometry.py` (`PINNED_INPUT_HASHES["src/
corpus_discrimination.py"]`). Fixing the file necessarily changes its
SHA-256 (old: `1ca62c4f7c1f88398c2d22c60bc1f2f6be27be678b68e9675a880
0bdb41a9bcc`, new: `225c003f7c590132f427e7eab604a3865b3bcbd0dba47ee40
b177b6ee44c86db`). Both `PINNED_INPUT_HASHES` dicts were updated to the
new hash — required for `c_b_paired_delta_analysis.main()` to run at
all post-fix, otherwise its own fail-closed check
(`AuditFailClosed`/`SystemExit`) refuses to proceed. This is a
mechanical, non-scientific update (same category as `code_version`);
no statistic, feature definition, or population changed.

**Count on the original Windows run:** 1 (see section 4).

## 3. Two C-construction reproducibility failures —
`tests/analysis/test_c_c_construction_audit.py`

Failing assertion:
`reproducibility_check["byte_identical_excluding_code_version"] is True`
in both `test_run_locked_c_b_contract_is_reproducible` and
`test_end_to_end_creates_exactly_three_files`.

**Investigation.** `run_locked_c_b_contract()` re-executes the locked
C-B contract (`src.analysis.c_b_paired_delta_analysis.main()`, exact
CLI args from C-A section 7.9) and diffs the result against the
already-committed `logs/c_b_paired_delta_analysis.json`, excluding only
`code_version.{generation_commit,working_tree_dirty}`. A full recursive
field-by-field diff (committed vs. a fresh in-sandbox rerun) was run
twice:

- **Before the corpus_discrimination.py fix:** the *only* differences
  were in `software_versions` — `numpy 2.4.4→2.5.2`, `pandas
  3.0.2→3.0.5`, `scikit-learn 1.8.0→1.9.0`, `scipy 1.17.1→1.18.1`.
  Every other field — every population count, every
  `LogisticRegression`-derived statistic in `results`, every hash in
  `provenance_integrity`/`pair_integrity` — was byte-identical.
  `requirements.txt` pins only lower bounds (`numpy>=1.26` etc.); no
  lock file exists anywhere in the repository (checked `git log -p
  requirements.txt`: it has never pinned exact versions except
  `PyYAML==6.0.3`), so a later `pip install -r requirements.txt`
  legitimately resolves to newer compatible releases. This is not
  cross-platform floating-point variation (values are *exactly* equal,
  not merely close) and not `LogisticRegression`/BLAS/solver
  nondeterminism (a `FutureWarning` about the deprecated `penalty=`
  argument appears with the newer scikit-learn, but produces byte-for-
  byte identical output on this repository's inputs). It is
  environment-fingerprint metadata drift with zero effect on the
  computed result.
- **After the corpus_discrimination.py fix (and pinned-hash update):**
  the only remaining difference was
  `pinned_input_hashes_verified["src/corpus_discrimination.py"]` — the
  old committed value vs. the new (correctly fixed) file's hash. This
  is a *real*, expected difference: the pinned file genuinely changed.

**Conclusion on intent:** the repository's own code already treats
`code_version` as "how this run was produced, not what it computed"
and excludes it from the byte-identical verdict for that reason.
`software_versions` sits in the same structural position in the output
(both immediately follow `task`/`spec_reference`; see `analysis =
{...}` construction in `c_b_paired_delta_analysis.py`) and is the same
kind of field — the module's own comment ("Recorded for reproducibility
only") on the analogous `additional_input_hashes` field confirms the
repository's established pattern of recording environment/input
provenance without always gating pass/fail on it.
`pinned_input_hashes_verified`, however, *is* the C-A section 7.1
fail-closed mechanism itself — its entire purpose is to catch exactly
this kind of drift, so a genuine change to a pinned file correctly
flipping the verdict to `False` is the check doing its job, not a bug.

**Fix (not a tolerance, not a redesign):**
1. `run_locked_c_b_contract()` now also excludes `software_versions`
   from the byte-identical computation (same treatment as
   `code_version`, for the same reason), while still recording
   `software_versions_match` and, when they differ,
   `software_versions_diff` — so the information is surfaced, not
   silently dropped.
2. `pinned_input_hashes_verified` remains fully part of the
   byte-identical comparison (this was **not** excluded — doing so
   would blind the fail-closed check to real input drift, which is
   exactly what it exists to catch). A new `pinned_input_hash_diff`
   field lists, by path, exactly which pinned inputs differ and their
   old/new hash, so a `False` verdict is self-explanatory instead of
   requiring a manual JSON diff.
3. The two tests were updated to assert the new, correct, and fully
   specific expected state: `byte_identical_excluding_code_version is
   False`, `pinned_input_hash_diff` contains *exactly*
   `{"src/corpus_discrimination.py": ...}` with the expected old/new
   hash values, and — the scientifically important assertion —
   `analysis["results"]`, `["provenance_integrity"]`, and
   `["pair_integrity"]` are asserted equal to the committed values,
   proving the actual computed science is untouched.

**Scope note:** regenerating/recommitting the frozen
`logs/c_b_paired_delta_analysis.json` (or `logs/
cf_joint_geometry_analysis.json`, or `results/c_construction_audit/*`)
so that their own embedded `pinned_input_hashes_verified` matches the
fixed file was deliberately **not** done — those are committed,
human-review-adjacent C-construction-audit artifacts, and refreshing
them reads as "C construction finalization," which Task 1's scope
boundary and the onboarding's human-review-dependency rule reserve for
Task 2. The two spec docs (`logs/c_existing_construction_audit_spec.md`,
`logs/cf_joint_geometry_spec.md`) that also cite the pre-fix hash in a
documentation table were likewise left untouched for the same reason,
and because `c_existing_construction_audit_spec.md`'s own hash is
itself recorded (non-fail-closed) in `c_c_construction_audit.py`'s
`input_manifest.json` output — editing it would ripple into that
secondary-provenance surface too. **Flagged for Task 2 / a human
decision:** these frozen artifacts and spec docs still reference the
pre-fix `src/corpus_discrimination.py` hash; refreshing them is a
one-line, purely mechanical update whenever C construction is next
finalized.

**Count on the original Windows run:** 2 (matches the task brief
exactly).

## 4. Reconciling the 12-vs-11 discrepancy

Original report: 12 failures total. One (`test_build_comparison_
omits_sections_with_missing_data`) was already fixed by the prior
Task-4 audit, leaving 11 expected remaining.

Enumerated by collecting the actual pytest IDs that would fail on
Windows for each root cause (`pytest --collect-only -v`), before any
fix:

| Category | Failing test IDs (pre-fix) | Count |
|---|---|---|
| Path separator (`test_validate_benchmark_v2.py`) | `test_missing_activation_metadata_is_stale_with_explicit_reason`, `test_missing_activation_binding_metadata_is_stale_with_explicit_reason`, `test_completely_absent_stage_reports_all_four_missing_paths`, `test_multiple_stages_each_report_their_own_missing_reason`, `test_removing_any_single_required_artifact_flips_pass_to_fail[final]`, `[pooled]`, `[metadata]`, `[binding]` | 8 |
| UTF-8 decode (`corpus_discrimination.py`) | `load_quadrant_texts` call site (single failure as reported) | 1 |
| C reproducibility (`test_c_c_construction_audit.py`) | `test_run_locked_c_b_contract_is_reproducible`, `test_end_to_end_creates_exactly_three_files` | 2 |
| **Total** | | **11** |

`8 + 1 + 2 = 11 = 12 − 1` (the one already fixed by Task-4). The
discrepancy was not a real inconsistency in the underlying failures —
it was that the task brief's category bullets (3 named categories)
don't individually state their per-category test-ID counts; the
path-separator category alone accounts for 8 of the 11 because one of
its tests is `@pytest.mark.parametrize`d over the four artifact kinds
(`final`/`pooled`/`metadata`/`binding`), each a distinct pytest ID.

## 5. Full CPU verification

`python -m compileall src tests` — clean, exit 0.

**Focused groups** (all affected files together):
`tests/test_validate_benchmark_v2.py`,
`tests/test_corpus_discrimination.py`,
`tests/analysis/test_c_c_construction_audit.py`,
`tests/analysis/test_summarize_cross_branch.py`,
`tests/analysis/test_cf_joint_geometry.py`,
`tests/analysis/test_c_b_paired_delta_analysis.py` —
**122 passed**, 0 failed.

**Full feasible suite** (`pytest -q --continue-on-collection-errors`,
run once): **611 passed, 1 skipped, 7 failed, 22 errors** (29 items
total).

Every one of the 29 failed/error items traces to exactly one cause:
`torch`, `transformers`, `trl`, `peft`, `datasets`, or
`sentence_transformers` not being importable in this sandbox —
verified by grepping every failure/error traceback in the full run for
its `ModuleNotFoundError` and confirming no other exception type or
message appears anywhere in the 29. This sandbox could not install the
full stack: the default PyPI `torch` wheel for this platform requires
several GB of `nvidia-cu13-*` CUDA packages as hard dependencies
(confirmed via wheel metadata inspection) even for CPU-only use, and
the true CPU-only wheel index (`download.pytorch.org`) is not in this
sandbox's allowed network domains. This is a sandbox/environment
constraint, not a repository defect — none of these 29 items exercise
any of the three problems this task was scoped to fix, and the 1
skip (`test_analyze_3d_h.py`) is an intentional, pre-existing skip
("researcher-only answer key not present in this environment").

`numpy`, `pandas`, `scipy`, `scikit-learn`, `statsmodels`, `PyYAML`,
`matplotlib`, `huggingface_hub`, and `pytest` were installed and used
for everything else; every test that does not import the six missing
packages above passed.

## Files changed

- `src/validate_benchmark_v2.py` — POSIX path fix (§1)
- `tests/test_validate_benchmark_v2.py` — regression coverage (§1)
- `src/corpus_discrimination.py` — UTF-8 encoding fix (§2)
- `tests/test_corpus_discrimination.py` (new) — regression coverage (§2)
- `src/analysis/c_b_paired_delta_analysis.py` — pinned-hash update (§2)
- `src/analysis/cf_joint_geometry.py` — pinned-hash update (§2)
- `src/analysis/c_c_construction_audit.py` — reproducibility-check
  transparency fix (§3)
- `tests/analysis/test_c_c_construction_audit.py` — updated
  expectations + stronger assertions (§3)
- `logs/task1_cpu_hardening_report.md` (this file, new)

## Pre-existing / environment issues (not fixed here, out of scope)

- Full ML stack (`torch`/`transformers`/`trl`/`peft`/`datasets`/
  `sentence_transformers`) not installable in this sandbox (§5).
- Frozen C-construction artifacts and spec docs still cite the
  pre-fix `src/corpus_discrimination.py` hash (§3, flagged for Task 2).

## Not touched (per scope boundary)

R104 human-review data/decisions, Arm-2, final C benchmark integration,
GPU execution, archive cleanup, semantic-near-duplicate redesign.
