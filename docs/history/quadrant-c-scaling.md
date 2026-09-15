# Scaling quadrant C from 15 to 104

Project owner feedback, direct and correct: 15 was far too small given the
40-60 target this project had already committed to, let alone the 100+
originally requested. The conservative pace up to this point was a
mistake to hold onto once the pipeline itself was already proven working
end-to-end - there was no remaining reason not to scale it up immediately
once the process was validated on the first 20.

**Found a real bug while scaling up**: `CATEGORY_MAPPING`'s key for the
harassment category had an extra comma ("Hate, harassment, and
discrimination") that didn't match StrongREJECT's actual category string
("Hate, harassment and discrimination", no comma before "and") - meaning
every category-filtered query silently excluded that whole category. Also
found and fixed the same typo baked into the 5 already-promoted harassment
candidates' `source_category` field. Neither affected correctness of what
had already been promoted (project_category was hardcoded per-item, not
derived from the broken mapping), but it would have silently excluded 45
real candidates from consideration going forward.

**189 unused StrongREJECT prompts available** across the 4 target
categories once the mapping was fixed. Explicitly declined to author new
phrasings for the weapons/explosives/CBRN-adjacent and drug-synthesis
subset of "Illegal goods and services" (~35 of 45 items - pipe bombs,
napalm, LSD synthesis, firearm modification, etc.) - not something I'll
draft reduced-cue rewordings of regardless of the research purpose behind
it. Drafted from the remaining 154: 135 new candidates across
disinformation (39), harassment (42), non-violent crimes/cybercrime (47),
and a small, selective subset of illegal-goods-and-services (8, chosen
specifically to exclude weapons/synthesis content - counterfeiting,
smuggling logistics, prescription fraud, wildlife trafficking, etc.).

**Process, not just volume**: every one of the 135 new source prompts was
pulled programmatically by unique-substring lookup against the live
StrongREJECT CSV, not hand-transcribed - specifically to avoid the
truncated-preview bug from the original 20. One candidate
(illegal_13, "unlicensed guns... Answer: You can find some at
https://www") turned out to match a jailbreak-formatted source entry
(roleplay-forum framing with a fake partial answer) that the protocol's
own criteria exclude - dropped entirely rather than reworded, since no
amount of rewording fixes a source that fails on format grounds. Checked
all 135 for internal duplication (source prompts and candidate prompts,
both fully unique) and for near-duplication against the existing 20 (none
found). Ran a Jaccard word-overlap check across all 104 eligible
candidate_prompts as a cheap proxy for redundancy (no sentence-transformers
available in this environment for a proper semantic check) - found 2
pairs above 0.5 overlap, both sharing a sentence template
("What kind of X..."/"What are some ways to Y...") while testing
genuinely different underlying harms (joking-insults vs. slur-listing;
neighborhood exclusion vs. voter suppression) - kept, not true duplicates.

**13 candidates initially classified eligible despite a 1-word lexicon
hit** (mostly "fake", plus "demean"/"steal"/"fraud"/"exploit"/"illegal") -
this exposed a real gap in `classify_and_prescreen`'s logic: only 2+ hits
("unresolved") routed to secondary, so single-hit ("attenuated") items
were slipping through as "eligible" despite not being genuinely
zero-cue - inconsistent with the bar already being enforced by the
regression test on the original 15. Revised all 13 to genuinely score
zero (same "fake" -> "invented"/"imitation" pattern already established,
"demeaning"/"belittle" for the ones hitting on "demean") rather than
loosen the bar or silently let them through.

**Final: 155 total candidates (20 original + 135 new), 104 eligible for
C1** (up from 15), 51 secondary (evasion_dominant). All 104 promoted into
`QUADRANT_C_RECORDS`, verified to score exactly zero on the lexical-cue
check as a single aggregate check, not just individually. Clears the
100+ target. The 40-60-then-expand staged plan from earlier in this
project is superseded - went straight to the larger batch once the
process was proven, rather than re-litigating the staging question again.

**Still needed**: rebuild the eval set with the new 104-item quadrant C,
rerun leakage and the composition check to get real numbers at this
scale (expect quadrant C's own internal size to change downstream numbers
that assumed n=15 or n=20). The remaining Dolly near-duplicate flagged in
the last real run (2 near-dupes reported, contents not yet reviewed) is
still open - needs the actual `near_duplicates` section pasted to assess,
same as the last few rounds.

**Update from the next session (steering handoff), for anyone reading this
top-to-bottom**: the above "still needed" paragraph is stale by the time
you're reading it - the eval-set rebuild it describes as pending had
actually already landed in this same commit, just without this paragraph
being updated to say so. Confirmed directly against the file, not just
asserted: `data/processed/controlled_eval.jsonl` is 654 rows (A=150,
B=250, C=104, D=150), the `split` key is present on every A/D row (240
direction_estimation + 60 held_out_behavioral), and
`data/dedup_report_m1_alt_v9.json` (the latest dedup report, n=654) shows
0 exact duplicates and exactly one intentionally-kept near-duplicate (the
Beyonce pair from the leakage-findings section above). So: eval set is
genuinely current, only the downstream GPU artifacts (activations,
direction, steering) still need to be regenerated against it. See the new
session section below for what's been built to make that regeneration a
single clean run instead of two.
