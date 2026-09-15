# Quadrant C promotion and the lexicon coverage gap

### Composition check ran clean, but flagged a real gap - the pipeline's output was never wired into the live eval set

Real run of `quadrant_composition_check.py` showed quadrant C scoring
WORSE than A (mean_cue_hits 0.55 vs 0.22) - the opposite of the intended
design. Diagnosed by direct computation, not guessed: quadrant C's live
prompts came from `QUADRANT_C_PROMPTS`, the ORIGINAL 20 hand-curated items
- confirmed by recomputing their score in isolation and getting an EXACT
match (0.550, 40.0%) to the real run's number. The pipeline's reviewed,
reduced-cue candidates were sitting in `data/quadrant_c_pipeline/*.jsonl`
the whole time, correctly gated behind human review, but nothing had
promoted them into the live set yet - so the composition check was
(correctly) measuring the old, already-known-bad prompts. Confirmed the
fix works before applying it: computed what the 15 eligible candidates
WOULD score in isolation - 0.000, 0%, vs. the old 0.550/40%.

### Quadrant A's low score: a real, separate finding about the tool's scope, not a data problem

Investigated directly against the live HarmBench CSV rather than
theorizing: 341/400 behaviors (85%) score ZERO on the lexicon, despite
being maximally overt to any human reader (bomb-smuggling plans, fentanyl
lab setup, credit card skimmers). HarmBench's overtness comes from NAMING
the harmful act/object directly, not from wrongdoing-signaling vocabulary
("unauthorized," "illegal," "without permission") - which is the only
thing this lexicon looks for. StrongREJECT (quadrant C's source) tends
toward permission-violation framing instead ("without getting caught,"
"without my employer knowing"), which the lexicon DOES catch. This means
a raw A-vs-C comparison via this lexicon inherits the same
cross-source-confound problem flagged earlier in this project (see the
matched-pairs section above) - A and C are overt through structurally
different mechanisms, so the lexicon systematically undersells A
regardless of how overt A actually is. Worth remembering: even a tool
built specifically to check for design validity can reintroduce the exact
confound it was meant to catch, if it's not checked against the actual
data it's being applied to.

Fixed one real, narrow piece of this: added "without getting caught" and
"without getting traced" to `RISK_LEXICON` (score_lexical_risk_cues.py) -
extremely common HarmBench/StrongREJECT evasion phrasing that was
missing. Verified the actual effect against the live HarmBench CSV before
and after (same 400 items, not mixed with the eval set's 150-item sample):
59/400 -> 73/400 items with a hit (14.8% -> 18.3%), mean 0.152 -> 0.193. A
real but modest improvement - most of HarmBench's overtness is still
about naming dangerous acts/objects directly, which no wrongdoing-cue
lexicon will ever catch by design. Don't oversell this fix as closing the
gap; it closes part of it.

**Practical conclusion**: the reliable evidence for "did the rewording
reduce cue density" isn't the raw A-vs-C composition-check number - it's
the within-pair comparison (matched_pairs.jsonl, same StrongREJECT source
style on both sides) and the direct verification that the 15 promoted
candidates score exactly 0 in isolation. Both of those hold up under this
scrutiny; the composition tool's A-vs-C prediction, as originally framed,
doesn't have a comparably solid foundation and shouldn't be read as if it
does.

### Quadrant C promoted - 15 candidates live, replacing the original 20

Reviewed in conversation with the project owner (not rubber-stamped - two
real issues were caught and handled before promotion): the keylogger
candidate (SR_cyber_01) was flagged as a case where stripping the
sympathetic-parent framing may have made the request read as MORE clearly
non-consensual surveillance, not more neutral - correctly NOT among the
15 promoted (it's `evasion_dominant`, already routed to the pipeline's
secondary set). The science-consensus candidate (SR_disinfo_04) was
rewritten before promotion - flagged as genuinely dual-use on its own
(see the earlier "SR_disinfo_04 rewritten" section above for the full
reasoning) - the corrected version is one of the 15 promoted.

`QUADRANT_C_PROMPTS` (flat list, kept for backward compat) is now derived
from a new `QUADRANT_C_RECORDS` structure that preserves real category
and source attribution per item, instead of the old generic
"hand-curated"/`None` placeholder - `build_quadrant_c()` updated to match.
Reviewer/date recorded directly on `QUADRANT_C_RECORDS`, per this
project's own documentation standard. Verified post-promotion: all 15
score `cue_level: neutral` / zero lexicon hits, confirmed by a dedicated
regression test (`test_quadrant_c_scores_clean_on_lexical_risk_lexicon`)
that would catch any future edit accidentally reintroducing a flagged
term.

**Still needed**: rebuild the eval set and rerun the full leakage +
composition-check suite with the new quadrant C live, to get real,
current numbers rather than the pre-promotion snapshot above. Also: only
15 candidates are live now (down from 20), quadrant C is smaller than
before - worth deciding whether to draft more candidates through the same
pipeline to get back toward the 40-60 target, or treat 15 as sufficient
for now given it's fully verified end-to-end.
