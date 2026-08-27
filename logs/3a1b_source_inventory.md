# Milestone 3A1B — Remaining Source Acquisition and Schema Audit

**Repository:** `https://github.com/urosavurdic/dpo-safety-representations`
**Branch:** `agent/c-quadrant-end-to-end-e0e2317a`
**Parent (3A1A) commit:** `17d82dade0f8b6070245084f39d59b4ce07aaa5c`
**Scope:** CPU/data-provenance work only. No candidates created, no prompts
scored, no review queue, no benchmark finalization, no gate-config changes,
no C-paired records modified.

---

## 0. 3A1A patch recovery

The 3A1A binary patch was a required deliverable but was never committed
or hash-recorded in the repo — it only existed as a chat attachment.
Regenerated this session:

```
git diff --binary a8aa91a93bddb75b77bd70ecd4aa46a4ddfe66b5 17d82dade0f8b6070245084f39d59b4ce07aaa5c
```

- **Bytes:** 41,462
- **SHA-256:** `a430d513b989af2c3a1df6b67a0cea5d7273d7f57fda774ba17261114cb6bb7f`
- **Verified** by applying it with `git apply --check` in an isolated
  `git worktree` checked out at the 3A1A parent (`a8aa91a9`) — applies
  cleanly.

---

## 1. Sources acquired

| Source | Repo | Revision | Rows | License | Overlap w/ SR/HB/AB/C-paired |
|---|---|---|---|---|---|
| SimpleSafetyTests | `bertiev/SimpleSafetyTests` | `d7aee9a9` (2024-03-25) | 100 | CC BY 4.0 | 0/0/0/0 |
| Do-Not-Answer | `Libr-AI/do-not-answer` | `46070348` (2024-06-07) | 939 | Apache-2.0 | 0/0/0/0 |
| JailbreakBench (partial) | PyPI `jailbreakbench==0.1.0` | package v0.1.0 | 100 | MIT | 3/9/11/1 |
| JailbreakHub (in-the-wild) | `verazuo/jailbreak_llms` | `4f4031bf` (2024-12-24) | 1,405 | MIT | 0/0/0/0 |
| JailbreakHub forbidden_question_set | `verazuo/jailbreak_llms` | `4f4031bf` (same) | 390 | MIT | 34/0/0/24 |
| SORRY-Bench base | `SORRY-Bench/SORRY-Bench` | — | — | not acquired | — |

All CSV/wheel bytes cached exactly under `data/raw/3a1b_source_cache/`
(gitignored via the existing `data/raw/` rule, confirmed with
`git check-ignore -v`).

### SimpleSafetyTests
100 hand-crafted items, 5 harm areas × 20 rows (`Suicide, Self-Harm, and
Eating Disorders`, `Physical harm and Violence`, `Illegal and Highly
Regulated items`, `Scams and fraud`, `Child safety`). **77/100 classify as
`complete_user_facing_prompt`**, zero overlap with any existing source.
No example text is reproduced anywhere in this report or its JSON twin —
counts and category labels only, including for the `Child safety`
sub-area.

### Do-Not-Answer
939 rows. Structurally, 927/939 (98.7%) read as
`complete_user_facing_prompt` — but **this is provenance-misleading in
isolation**. Per this milestone's special rule, the `question` field is
GPT-4-generated against a risk taxonomy, then human-filtered (confirmed
from the paper's own description and this project's own prior
characterization). All 939 rows are recorded with `source_generated_prompt`
provenance, not treated as human-authored, and are **not eligible for the
strict primary pool** regardless of how well-formed the text reads.

### SORRY-Bench base — not acquired
Confirmed directly from the code repo's own README: the base
`question.jsonl` and all 20 mutation files are distributed exclusively via
a **gated** HuggingFace dataset (`sorry-bench/sorry-bench-202503`) —
*"you need to first request access."* Doubly blocked this session:
`huggingface.co` isn't in the sandbox network allowlist, and even with
network access, gated approval isn't something an unattended session can
complete. The cloned code repo has mutation *code*
(`data/sorry_bench/mutate/*.py`) but no mutation *data*. Confirmed and
recorded well under the 45-minute cap.

### JailbreakBench / JBB-Behaviors — partial acquisition, important caveat
`huggingface.co` remains blocked (same `403 host_not_allowed` as 3A1A).
Instead of stopping at "inaccessible" again, I checked whether the
`jailbreakbench` PyPI package bundles a copy — **it does**:
`jailbreakbench==0.1.0` (the only version on PyPI) ships
`jailbreakbench/data/behaviors.csv`, 100 rows, `Goal`/`Target`/`Behavior`/
`Category` schema. Cached both the extracted CSV and the full wheel.

**But** this is not confirmed identical to the authoritative dataset. The
package's own loader (`src/jailbreakbench/dataset.py`) and README both
reference a `Source` column (`Original` / `Trojan Detection Challenge
2023`+`HarmBench` / `AdvBench`) that **this bundled file does not have** —
the real, Source-tagged dataset is pulled from HuggingFace at runtime,
which is exactly what's blocked. So per this milestone's special rule
("preserve the per-row distinction between Original, AdvBench-derived,
and HarmBench-derived records"), I reconstructed that distinction
independently, by content hash, rather than trusting an absent label:

| Match | Count |
|---|---|
| Verbatim match to AdvBench `goal` | 11 |
| Verbatim match to HarmBench `Behavior` | 9 |
| Verbatim match to StrongREJECT `forbidden_prompt` | 3 |
| Verbatim match to an existing C-paired candidate's `source_prompt` | 1 |
| Matches none of the four | 80 (upper bound on "Original" — see caveat below) |

The single C-paired match is worth flagging on its own: this bundled JBB
row (Category: *Economic harm*) is byte-identical to
`candidate_records_v2.jsonl`'s `SR_illegal_08` — a genuine cross-dataset
text collision. The 80-row "matches nothing" bucket is an **upper bound**,
not a confirmed "Original" count: it also absorbs any Trojan Detection
Challenge-sourced rows (a fourth documented source this project has no
reference corpus for) and any AdvBench/HarmBench-derived rows that were
paraphrased rather than copied verbatim, which exact-hash matching can't
catch.

Structurally, the `Goal` field itself is mostly directive
(`behavior_description`: 93/100, `complete_user_facing_prompt`: 5/100) —
consistent with AdvBench/HarmBench's house style, which tracks with the
partial provenance reconstruction above.

### JailbreakHub — two files, two very different provenances
`verazuo/jailbreak_llms` contains two datasets that are easy to conflate
since they share one repo and one paper:

**`jailbreak_prompts_2023_12_25.csv` (1,405 rows) — genuinely in-the-wild.**
Real Reddit/Discord/website posts, Dec 2022–Dec 2023, with real
`platform`/`community`/`created_at` metadata. All 1,405 rows are tagged
`jailbreak == True` by the dataset's own authors (this file is the
pre-filtered jailbreak subset of their full scrape). Zero overlap with
StrongREJECT/HarmBench/AdvBench/C-paired.

**`forbidden_question_set.csv` (390 rows) — NOT in-the-wild**, despite
living in the same repo. The README says it plainly: *"we construct a
question set comprising 390 questions across 13 forbidden scenarios
adopted from OpenAI Usage Policy"* (the 14th scenario, Child Sexual Abuse,
is explicitly excluded by the dataset's own authors). This is a
researcher-constructed benchmark-target set, not scraped user content.
It has real overlap risk though: **34 rows are byte-identical to a
StrongREJECT `forbidden_prompt`, and 24 of those are already present in
the C-paired candidate pool** — worth flagging for any future sampling.

---

## 2. A documented reliability problem with the automated classifier

Applied `classify_source_provenance.py` to all 1,405
`jailbreak_prompts_2023_12_25.csv` rows:

| Label | Count |
|---|---|
| `jailbreak_wrapper` | 581 |
| `evaluation_template` | 351 |
| `complete_user_facing_prompt` | 463 |
| `behavior_description` | 6 |
| `category_label` | 3 |
| `incomplete_or_context_dependent` | 1 |

That 463-row `complete_user_facing_prompt` bucket looked too high given
every single row in this file is independently tagged `jailbreak == True`
by the source's own authors. So I checked: of the 72 rows under 300
characters the classifier called `complete_user_facing_prompt`, a broader
manual keyword pass (a superset of the classifier's own wrapper regex —
*pretend, roleplay, act as, persona, character, from now on, developer
mode, jailbroken*, etc.) found wrapper/persona cues in **27/72 (37.5%)**
that the narrow regex missed. **The classifier under-detects wrapper
structure on this source's long, creative, multi-part templates.** Per
instruction, its opening-verb rule was never used as an automatic
eligibility gate anywhere this session, and this specific finding is now
recorded as a known limitation rather than folded quietly into the count:
the 463 figure is a loose upper bound requiring per-item human review, not
a usable number — reaffirming 3A0's original conclusion that this source
needs manual per-item identification, not a bulk pull.

By contrast, SimpleSafetyTests (short, direct, single-sentence) and
JailbreakHub's `forbidden_question_set` (also short, direct) classify with
high confidence and no such discrepancy — the classifier is reliable on
short, single-register text and unreliable on long, multi-register,
creative templates. That's a useful, source-dependent signal for future
milestones, not a reason to distrust the tool universally.

---

## 3. Omissions

1. **SORRY-Bench base set** — gated HF dataset, confirmed from the code
   repo's own README; also blocked by network policy. In scope, not
   achieved, recorded within the 45-minute budget.
2. **Authoritative Source-tagged JBB-Behaviors** — HF-hosted, blocked. A
   partial substitute (PyPI-bundled `behaviors.csv`, no `Source` column)
   was found and used instead — a partial, not total, omission.

---

## 4. Unresolved issues (see JSON for the full list)

- 3A1A's patch had no durable home in version control; regenerated and
  verified this session, but neither 3A1A's nor 3A1B's patch is tracked
  in-repo — a future milestone should decide on a permanent location.
- SORRY-Bench base set and the authoritative JBB-Behaviors dataset both
  need either a network-policy change or an alternate access path.
- JailbreakHub's 1,405-row in-the-wild file needs real per-item manual
  review before any candidate work — the automated pass is demonstrated
  unreliable on this source.
- `forbidden_question_set.csv`'s 34-row StrongREJECT overlap (24 of which
  already sit in the C-paired pool) needs content-hash dedup in any future
  sampling from that file.
- No near-duplicate (as opposed to exact-hash) checking was done between
  this session's new sources and each other or against 3A1A's sources —
  only exact-text hashing, consistent with this milestone's no-scoring
  constraint, but still an open gap before any pool is built.
- Do-Not-Answer and `forbidden_question_set` are both provenance-excluded
  from the strict primary pool (source-generated / researcher-constructed)
  regardless of how complete their text reads — should become a hard rule,
  not just a note, once selection logic exists.

---

## 5. Next milestone

**3A1C.** No acquisition beyond what's listed above was performed, no
candidates were created, no prompts were scored, and no C-paired records
were touched.
