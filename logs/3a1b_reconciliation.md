# Milestone 3A1C-0 — Current-State Reconciliation and 3A1B Errata

**Repository:** `https://github.com/urosavurdic/dpo-safety-representations`
**Reviewed branch:** `agent/c-quadrant-end-to-end-e0e2317a`
**Reviewed HEAD:** `f47363968088cdf9cd77b3245658169d49d7b737`
**Scope:** Documentation and provenance reconciliation only. No datasets
acquired or redownloaded, no candidates created, no prompts scored, no
Fightin' Words computed, no review queue created, no benchmark/gate/
C-paired/classifier files modified, no GPU/model/steering code touched.

This file and its JSON twin (`logs/3a1b_reconciliation.json`) are new
errata artifacts. The historical `3a1a_source_inventory.*` and
`3a1b_source_inventory.*` files are left byte-for-byte unchanged; nothing
in them is rewritten to hide the discrepancies below.

---

## 1. Actual Git identity of the 3A1B commit

```
git log --format='%H %P %s' --all --grep='Milestone 3A1B' -n 5
```

returned exactly one match:

- **Actual 3A1B commit:** `f47363968088cdf9cd77b3245658169d49d7b737`
  ("Milestone 3A1B: remaining source acquisition and schema audit",
  2026-08-28 00:48:32+02:00)
- **Actual parent:** `9778cfcf3a1a3a40c80fdb91eb56a1e6fd39a88d`
  ("Step 7: prepare colab_unified_analysis.ipynb for the final v2 GPU
  rerun", 2026-08-27 23:29:28+02:00)
- Both are reachable on this branch in a full, non-shallow clone.

**The SHA values recorded inside 3A1B's own JSON are not reachable.**
`logs/3a1b_source_inventory.json` records `parent_commit_3a1a` and the
`3a1a_patch_recovery.child_commit` as `17d82dade0f8b6070245084f39d59b4ce07aaa5c`,
and `3a1a_patch_recovery.parent_commit` (matching 3A1A's own self-reported
HEAD) as `a8aa91a93bddb75b77bd70ecd4aa46a4ddfe66b5`. Neither hash exists in
this repository under any of: `git cat-file -t`, `git log --all --oneline`,
or `git ls-remote` against every branch (`agent/c-quadrant-end-to-end-e0e2317a`,
`main`, `phase4-wip`). Consequently the old command
`git diff --binary a8aa91a9 17d82dad` (and its recorded 41,462-byte,
`a430d513...` patch) **cannot be reproduced** against real repository
state. The cause is not determined from available evidence — it may be a
recording error at the time, or later history rewriting — but either way
these SHAs must not be treated as valid anchors going forward.

The real git parent of the 3A1B commit is five commits after the actual
3A1A commit (`b3b27df`), separated by four unrelated GPU/steering-track
commits. The old report's "parent_commit_3a1a" field appears to have meant
"prior milestone" conceptually rather than the literal git parent — but
the SHA it used doesn't exist regardless of which meaning was intended.

### Archival patch

A corrected patch was generated using the **actual** parent/commit pair:

```
git diff --binary 9778cfcf3a1a3a40c80fdb91eb56a1e6fd39a88d f47363968088cdf9cd77b3245658169d49d7b737 \
  -- . ':(exclude)artifacts/patches/*.patch'
```

- **Bytes:** 39,452
- **SHA-256:** `b43aa4554ce250052c28f1cd04f1ee643d081774045cf8cc3f4b513f61cd05cc`
- **Verified:** applies cleanly with `git apply --check` in an isolated
  `git worktree` at the actual parent commit.

**This patch could not be written to its canonical path,
`artifacts/patches/c_quadrant_9778cfcf_f4736396.patch`.** The repository
already has a **tracked, empty regular file** named `artifacts` at the
root (added in commit `b94d9b7 "Lst chngs"`), which occupies that path
and blocks creating `artifacts/` as a directory. Deleting or renaming an
existing tracked file is outside this milestone's authorized scope
(documentation/errata only), so it was not done. The patch instead exists
only as an unattached scratch artifact, pending a future milestone's
decision on how to resolve the `artifacts` file/directory collision.

---

## 2. Classifier claim

`src/diagnostics/classify_source_provenance.py` (unmodified this session):

```python
@dataclass
class ClassificationResult:
    label: str
    reason: str
```

- **No separate confidence field** — confidence, where represented at all,
  is a substring inside `reason`.
- The **opening-verb `behavior_description` branch**
  (`_BEHAVIOR_VERB_START` match, e.g. write/give/create/describe/provide/…)
  returns `reason = "imperative task-directive opening verb, no
  first-person/question framing"` — **no low-confidence marker**.
- The **imperative-fallback branch** (same grammatical shape, no
  wrapper/template match, no opening-verb match) returns
  `label = "complete_user_facing_prompt"` with `reason = "...
  (low-confidence: see module docstring on imperative ambiguity)"` — **has**
  a low-confidence marker.

Both branches classify the same underlying ambiguity — the module's own
docstring says a red-teaming "behavior" label and a real chatbot request
can be grammatically indistinguishable — but only one of the two exposes
that in its output. **This errata records, without changing the code,
that both the opening-verb and imperative-fallback outputs remain
provisional and require human review**, and reaffirms that this tool is
triage only: not a harm classifier, not a semantic adjudicator, and not a
final eligibility gate under any circumstance.

---

## 3. JailbreakBench package claim

3A1B stated `jailbreakbench==0.1.0` was "the only version on PyPI,
confirmed via `pip index versions`." Checked against PyPI's own package
index metadata this session (metadata query only — the wheel/data file
was not re-downloaded, per this milestone's no-redownload constraint):

| | |
|---|---|
| Versions existing on PyPI | 0.1.0, 0.1.1, 0.1.2, 0.1.3, **1.0.0** (latest) |
| `requires_python` | 0.1.0: `>=3.10`; 0.1.1–1.0.0: `<3.12,>=3.10` |
| Sandbox Python | 3.12.3 |
| Versions installable in this sandbox | 0.1.0 only |
| Version actually used by 3A1B | 0.1.0 |
| Embedded CSV SHA-256 (as reported by 3A1B, not re-verified this session) | `221c868b3beef1e0c0cc8ad0250d57f21a409094aa57330ad4a73689e0414f73` |
| Embedded file has the official `Source` column | No (per 3A1B's own schema listing and `critical_limitation` note — not re-derived this session) |

**Correction:** "only version on PyPI" is factually wrong — PyPI hosts
five releases. The real, much more mundane explanation is almost
certainly that every release after 0.1.0 declares `requires_python
<3.12`, and this sandbox runs Python 3.12, so `pip` silently excludes
them. **This does not by itself fix the missing-`Source`-column problem**
— whether 0.1.1–1.0.0 bundle a `Source`-tagged CSV is genuinely unknown
and is recorded here as **unresolved**, not corrected, since verifying it
would require installing under Python ≤3.11 or downloading a newer
wheel, both out of this milestone's scope.

---

## 4. Stale state files

`logs/agent_state.json` still describes the **pre-3A1A** state:

- `base_commit`/`code_commit` still point at `e0e2317a52e89b0d614b99152a9ca71758baf489`, the original task base commit.
- `status: "needs_human_review"` refers only to the original 104-row `c_paired` review queue.
- `unresolved_issues` still lists "C-source-authored arm: not built (HarmBench/JailbreakBench CSV unavailable in sandbox)" — both have since been acquired and analyzed.
- `sources_omitted` still lists HarmBench and JailbreakBench as unattempted/unavailable — outdated.
- `next_action` still points back to reviewing the original 104-row CSV and resuming from `RESUME_PROMPT.md`, with no reference to 3A1A/3A1B/3A1C at all.

This file was **not** treated as current truth this session. It requires
an explicit handoff update once 3A1C exists.

---

## 5. 3A0 filename issue

`logs/3a0_souece_plan.json` exists and contains valid JSON matching 3A0's
expected structure; `logs/3a0_source_plan.json` (the canonical name) does
not exist anywhere in the repository. This is recorded as a documentation
issue only. Per instruction, the file was **not** reconstructed from
memory and **not** renamed this session (a content-preserving rename is
possible without ambiguity, but is left to a milestone explicitly scoped
to touch it).

---

## 6. Source-policy implications (unchanged inventory, recorded only)

See `source_policy_implications` in the JSON twin for the full list. In
summary: StrongREJECT's already-computed 155-row-disjoint overlap result
stands; its provenance must stay per-row, never collapsed to
"human-authored"; SimpleSafetyTests may be considered strict-primary
pending its recorded hash/provenance holding up; AdvBench, Do-Not-Answer,
SORRY-Bench, the partial JBB file, and JailbreakHub all remain
supplementary/unresolved for the specific, previously-recorded reasons;
HarmBench remains excluded from strict-primary because live Quadrant A is
100% HarmBench and HarmBench sits inside the `H = A ∪ B` reference corpus.
None of the unresolved supplementary sources by itself blocks 3A2, as long
as the strict-primary pool used for 3A2 is explicitly limited to sources
with verified strict-primary status.

---

## 7. Validation performed

- JSON syntax: `logs/3a1b_reconciliation.json` parses cleanly (`json.load`).
- Every hash quoted above was computed directly against files/diffs
  present in this session (patch SHA-256, and the JBB CSV hash carried
  forward unmodified from 3A1B's own report where re-verification was out
  of scope).
- `git diff --check` run against the two new files: clean, no whitespace
  errors.
- Confirmed no candidate, scored, review, benchmark, gate, or model files
  changed this session, and no Python files changed
  (`git diff --stat` limited to the two new `logs/` files below).
- No source data was redownloaded; the JBB PyPI check queried package
  index metadata only.

## Files changed this session

- `logs/3a1b_reconciliation.json` (new)
- `logs/3a1b_reconciliation.md` (new)

No other tracked file was modified. The regenerated archival patch is
**not** included in this commit — see §1 for why, and what a future
milestone needs to do to unblock it.

---

## 8. Next milestone

**3A1C.** This milestone does not begin 3A1C or 3A2 work.
