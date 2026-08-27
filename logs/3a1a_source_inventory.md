# Milestone 3A1A — Verify 3A0 and Audit Accessible Sources

**Repository:** `https://github.com/urosavurdic/dpo-safety-representations`
**Branch:** `agent/c-quadrant-end-to-end-e0e2317a`
**HEAD (verified via `git rev-parse HEAD`):** `a8aa91a93bddb75b77bd70ecd4aa46a4ddfe66b5`
**Working tree:** clean (`git status --short` empty on a fresh clone)
**Scope:** CPU/data-provenance work only. No candidates created, no prompts
scored, no review queue, no benchmark finalization, no gate-config changes,
no GPU work.

---

## 0. Verifying the 3A0 report before trusting it

`git log --oneline -15` on this branch matches the 3A0 report's log
verbatim, and `logs/benchmark_gate_config.json`, the frozen manifest's
`c_counts_by_construction: {"c_paired": 104}`, and
`data/quadrant_c_pipeline/candidate_records_v2.jsonl` all match what 3A0
claimed.

**But:** `logs/3a0_source_plan.json` and `logs/3a0_source_plan.md`
**do not exist anywhere in this repository** — not on this branch, not on
`main`, not as an uncommitted file in the working tree. They were supplied
to this session as uploads, not read from the repo, and a clean `git clone`
confirms they were never committed. Per instruction, this was treated as
grounds not to trust the report at face value. Every specific factual claim
this milestone depended on was independently re-verified against the real
repository this session (see below) and all of it checked out — the 3A0
content appears accurate, its absence from version control is a process
gap, not a correctness problem. A future milestone should decide whether to
commit those files retroactively.

---

## 1. Sources acquired

All three GitHub-hosted sources were fetched fresh this session, hashed,
and cached byte-exact under `data/raw/3a1a_source_cache/` (gitignored via
the existing `data/raw/` rule — confirmed with `git check-ignore -v`).

| Source | Repo | Revision | Rows | Bytes | SHA-256 |
|---|---|---|---|---|---|
| StrongREJECT | `alexandrasouly/strongreject` | `f7cad6c1` (2024-11-03) | 313 | 56,359 | `4dd70357...4381` |
| HarmBench | `centerforaisafety/HarmBench` | `8e1604d1` (2024-08-05) | 400 | 198,850 | `8d81acce...4afc` |
| AdvBench | `llm-attacks/llm-attacks` | `098262ed` (2024-08-02) | 520 | 82,125 | `6cd1a5c6...9e1` |

All three repos' own `LICENSE` files were fetched and read directly this
session: **MIT** for all three. This resolves the corresponding 3A0
unresolved item for these sources specifically.

**JailbreakBench / JBB-Behaviors: not acquired.** Its source code repo
(`JailbreakBench/jailbreakbench`) shows the actual behaviors data is loaded
via the HuggingFace `datasets` library from `dedeswim/JBB-Behaviors`
(`src/jailbreakbench/dataset.py`), not from GitHub. `huggingface.co` is not
in this sandbox's allowed network egress — a direct request returned
`HTTP 403`, header `x-deny-reason: host_not_allowed`. Both
`JailbreakBench/jailbreakbench` and `JailbreakBench/artifacts` were cloned
in full and searched for any mirrored CSV/JSON copy of the behaviors data;
neither contains one. A web search for a GitHub-hosted mirror also came up
empty. This is a genuine access limitation for this session, not a design
choice — see `unresolved_issues`.

---

## 2. StrongREJECT disjointness (frozen decision #2)

Per protocol, compared against **all 155 rows** of
`data/quadrant_c_pipeline/candidate_records_v2.jsonl` — live and non-live —
not just the 104 currently-live `c_paired` rows.

The raw CSV has no persistent per-row ID column, so identity was
established by **SHA-256 of the stripped `forbidden_prompt` text** rather
than a literal ID-string match — a stronger check, since two rows with
different IDs but identical text would still be a duplicate that pure
ID-matching could miss.

- All 155 `candidate_records_v2.jsonl` rows matched an exact raw-CSV row
  by hash. All 155 rows' own recorded `source_prompt_sha256` field matched
  the recomputed hash too (0 mismatches) — the project's stored hashes are
  trustworthy.
- **158 of 313 raw rows have no corresponding entry anywhere in
  `candidate_records_v2.jsonl`** — the genuinely untouched, disjoint pool.
- By category: Illegal goods and services 38, Sexual content 50, Violence
  54, Non-violent crimes 7, Disinformation and deception 6, Hate/harassment
  3. **Sexual content and Violence together are 66% of the disjoint pool**
  — worth flagging for any future sampling design, though no sampling was
  done this session.
- By upstream `source` column (StrongREJECT's own attestation of where
  each row came from): `custom` 129, `AdvBench` 13, `DAN` 10,
  `MaliciousInstruct` 2, `MasterKey` 2, `OpenAI System Card` 2.
- Structural classification (heuristic, see §4): 132 read as
  `complete_user_facing_prompt`, 26 as `behavior_description`.

No candidates were created or selected from this pool — this is strictly a
count and composition report.

---

## 3. HarmBench exact overlap with live Quadrant A (frozen decision #1)

Same method: SHA-256 of stripped text, `Behavior` (HarmBench) vs. `prompt`
(Quadrant A rows of `data/processed/controlled_eval.jsonl`).

- Quadrant A: 150/150 rows, source `HarmBench` (single value) — confirmed.
- **150/150 exact hash matches** against the full 400-row HarmBench CSV.
- All 150 are `FunctionalCategory == standard`, spanning
  `harassment_bullying` (19), `illegal` (57), `cybercrime_intrusion` (40),
  `misinformation_disinformation` (34).
- **250 HarmBench rows are unused**: the remaining 50 `standard` rows
  (28 `chemical_biological`, 21 `harmful`, 1 `illegal`), plus all 100
  `copyright` and all 100 `contextual` rows.

**Frozen decision #1 reaffirmed**: HarmBench stays excluded from the
strict primary candidate pool. No HarmBench rows were placed in any
primary candidate file this session — no candidate file of any kind was
created this session, per the milestone's constraints. The gap noted in
3A0 (no reusable candidate-vs-Quadrant-A check exists in the pipeline
tooling, only vs.-training-data checks) is still open; this session's
comparison was a one-off analysis for this report, not a new pipeline
module.

---

## 4. Structural classification (heuristic screening tool)

Added `src/diagnostics/classify_source_provenance.py` (with
`tests/diagnostics/test_classify_source_provenance.py`, 9/9 passing) — a
small, reusable text-structure classifier against the fixed taxonomy. It
is explicitly **not** a harm/severity scorer and not part of the
c_source_authored candidate pipeline; it's the same category of tool as
the existing `score_lexical_risk_cues.py`, with the same caveat: a
screening aid for a human reviewer, not an adjudication.

Per instruction, it does **not** trust a source's own field name (e.g.
HarmBench's "Behavior" or AdvBench's "goal") — it inspects the actual text
of every `prompt`-role field. The one exception is AdvBench's/JBB's
`target` field, excluded by construction (not by label-trust): its content
is definitionally the desired GCG affirmative-completion prefix
("Sure, here is..."), never a request.

| Source (field) | n | complete_user_facing_prompt | behavior_description | other |
|---|---|---|---|---|
| StrongREJECT (`forbidden_prompt`) | 313 | 265 (245 high-conf + 20 low-conf) | 48 | — |
| HarmBench-standard (`Behavior`) | 200 | 11 (all low-conf) | 186 | 2 category_label, 1 incomplete |
| AdvBench (`goal`) | 520 | 37 (all low-conf) | 481 | 2 category_label |
| AdvBench (`target`) | 520 | — | — | 520 target_goal (100%, by construction) |

This is a quantitative, text-driven confirmation of what 3A0 argued
qualitatively: StrongREJECT's `forbidden_prompt` field reads overwhelmingly
as complete, addressable requests, while HarmBench's `Behavior` and
AdvBench's `goal` fields read overwhelmingly as third-party-style
imperative task directives. The **low-confidence "imperative fallback"**
buckets (20 / 11 / 37 respectively) are genuinely ambiguous sentences the
heuristic could not confidently place either way and need human
spot-review before any future use.

AdvBench exact-dedup: 520/520 unique `goal` strings — no exact
duplicates. This does **not** clear the near-duplicate/templating concern
3A0 flagged from the literature; that needs the embedding-based near-dup
tooling (as in `check_leakage.py`), which was not run this session.

---

## 5. `secondary_c5_evasion.jsonl` inspection

- 5 rows, 6,518 bytes, `sha256:1b84d891...582d4`.
- Schema: 20 fields including `source_prompt`, `candidate_prompt`,
  `transformation_family`, `evasion_dominant`, `agent_pre_screen`.
- **Confirmed: not documented in `data/quadrant_c_pipeline/summary.json`**
  (that file has no `secondary_c5` key at all).
- **Finding: this file is not a source-authored candidate pool.** All 5
  rows have `source_prompt != candidate_prompt` and
  `transformation_family: reduced_cue_source_rewrite` — the same
  AI-rewording transformation used for `c_paired`. Cross-checking their
  `candidate_id`s (`SR_illegal_01`, `SR_cyber_01`, `SR_cyber_02`,
  `SR_illegal_06`, `SR_illegal_07`) against `candidate_records_v2.jsonl`
  shows all 5 already exist there with `c_construction: c_paired` and
  `inclusion_decision: secondary_only`. This file looks like an
  evasion-flagged export/view of 5 already-existing `c_paired` candidates,
  not a distinct source or construction type — and should not be confused
  with, or counted as progress toward, the `c_source_authored` arm.

---

## 6. Omissions

1. **JailbreakBench/JBB-Behaviors** — not acquired (HF-hosted, blocked by
   sandbox network policy; see §1). In scope for this milestone, not
   achieved.
2. **AdvBench intra-source near-dup/templating audit** — only exact-text
   dedup run (520/520 unique); embedding-based near-dup pass not run this
   session. Out of scope for this milestone (no near-dup tooling was
   invoked), carried to 3A1B.
3. **Dedicated candidate-vs-Quadrant-A contamination tool** — this
   session's HarmBench↔Quadrant-A check was ad hoc, not built into the
   pipeline's reusable contamination tooling. Out of scope for this
   milestone.

---

## 7. Unresolved issues (carried forward + new)

1. `logs/3a0_source_plan.json`/`.md` are absent from version control
   despite being referenced as if committed (§0) — needs a decision in a
   future milestone.
2. JBB-Behaviors still unacquired — needs a network-policy change or an
   alternate access path.
3. SimpleSafetyTests / SORRY-Bench access verification (3A0) remains
   unresolved — out of scope for this milestone's acquire list, carried
   forward.
4. AdvBench's literature-documented near-duplicate/templating pattern is
   not cleared by this session's exact-only dedup.
5. StrongREJECT has no persistent per-row ID; future overlap checks must
   keep using content-hash identity, not upstream ID strings (none exist).
6. The 158-row StrongREJECT disjoint pool skews heavily toward Sexual
   content (32%) and Violence (34%) — a future sampling design should be
   aware of this; no sampling decision was made this session.
7. `classify_source_provenance.py`'s low-confidence "imperative fallback"
   classifications (20 StrongREJECT / 11 HarmBench-standard / 37 AdvBench)
   need human spot-review before any downstream use.
8. No reusable candidate-vs-Quadrant-A contamination check exists in the
   pipeline tooling — gap from 3A0, still open.
9. StrongREJECT/HarmBench/AdvBench licensing (MIT, all three) was
   re-verified directly against each repo's own LICENSE file this session
   — resolves that 3A0 item for these three sources. JBB-Behaviors'
   dataset-specific license/card was not independently reachable.

---

## 8. Next milestone

**3A1B** — pending a decision on the missing 3A0 commit, a resolution path
for JBB-Behaviors access, and the still-open near-dup/contamination
tooling gaps above. No acquisition, scoring, candidate creation, or
promotion was performed this session — this milestone is audit and
inventory only.
