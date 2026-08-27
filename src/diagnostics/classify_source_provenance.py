"""
Heuristic structural classifier for external source records, used only for
provenance auditing (milestone 3A1A). This assigns each record ONE label
from the fixed taxonomy below, based on surface text structure and known
field semantics - NOT a harm/severity score, NOT the Fightin' Words
screening machinery in corpus_discrimination.py, and NOT part of the
c_source_authored candidate pipeline. It does not select, rank, or
promote anything.

Explicit instruction this module exists to satisfy (3A1A brief): "Do not
exclude a record merely because its source calls it a 'behavior' or
'goal'. Inspect the actual text." So classification is driven by the text
itself, not by trusting a dataset's own column name - with ONE narrow
exception: GCG-style "target" columns (AdvBench `target`, JBB `Target`)
are excluded by construction, not by label-trust, because their content is
definitionally the desired affirmative assistant-completion prefix
("Sure, here is..."), not a request at all.

LIMITS (read before trusting the output): this is a cheap regex/keyword
pass, exactly the same category of tool as score_lexical_risk_cues.py, and
inherits the same caveat - it is a screening aid for a human reviewer, not
an adjudication. It will misclassify borderline cases (e.g. terse
imperative sentences that are grammatically indistinguishable between a
red-teaming "behavior" label and a real chatbot request; see module
docstring discussion in logs/3a1a_source_inventory.md). Every count this
tool produces should be read as "candidates for human review", not ground
truth. Do not silently promote its output into a candidate or review file.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

TAXONOMY = (
    "complete_user_facing_prompt",
    "behavior_description",
    "target_goal",
    "category_label",
    "jailbreak_wrapper",
    "role_play_template",
    "evaluation_template",
    "attack_specification",
    "incomplete_or_context_dependent",
    "source_generated_prompt",
    "source_paraphrase_or_mutation",
    "unknown",
)

_FIRST_PERSON_OR_QUESTION = re.compile(
    r"\b(i want|i need|i'm trying|im trying|can you|could you|please|"
    r"help me|how do i|how can i|what is the best way|would you)\b",
    re.IGNORECASE,
)

_BEHAVIOR_VERB_START = re.compile(
    r"^\s*(give|write|create|develop|design|generate|provide|explain how|"
    r"describe|draft|compose|produce|come up with|outline|suggest|"
    r"demonstrate|show|devise|list|formulate|detail|instruct|offer)\b",
    re.IGNORECASE,
)

_WRAPPER_MARKERS = re.compile(
    r"\b(DAN|do anything now|ignore (all )?(previous|prior) instructions|"
    r"you are now|pretend (that )?you are|act as|jailbreak|AIM:|"
    r"stay in character|no restrictions|hypothetical response|"
    r"as an ai (with no|without) restrictions)\b",
    re.IGNORECASE,
)

_TEMPLATE_MARKERS = re.compile(r"(\{[^}]{0,40}\}|\[INSERT[^\]]*\]|_{3,})")

_AFFIRMATIVE_TARGET = re.compile(r"^\s*sure,?\s+here", re.IGNORECASE)


@dataclass
class ClassificationResult:
    label: str
    reason: str


def classify_target_field(text: str) -> ClassificationResult:
    """For fields that are structurally GCG-style optimization targets
    (AdvBench `target`, JBB `Target`), not candidate prompts at all."""
    if _AFFIRMATIVE_TARGET.match(text or ""):
        return ClassificationResult("target_goal", "affirmative-completion target field")
    return ClassificationResult("target_goal", "target-role field (by column semantics)")


def classify_prompt_like_field(text: str) -> ClassificationResult:
    """For fields that may or may not actually be complete user-facing
    prompts regardless of what the source dataset calls them (e.g.
    HarmBench `Behavior`, AdvBench `goal`, StrongREJECT `forbidden_prompt`).
    """
    t = (text or "").strip()
    if len(t) == 0:
        return ClassificationResult("unknown", "empty")

    word_count = len(t.split())
    if word_count <= 3:
        return ClassificationResult("incomplete_or_context_dependent", "too short to stand alone")

    if _TEMPLATE_MARKERS.search(t):
        return ClassificationResult("evaluation_template", "contains placeholder/blank markers")

    if _WRAPPER_MARKERS.search(t):
        return ClassificationResult("jailbreak_wrapper", "contains persona/override markers")

    if word_count <= 6 and not re.search(r"[.?!]", t):
        return ClassificationResult("category_label", "short, no verb-object sentence structure")

    if _FIRST_PERSON_OR_QUESTION.search(t) or t.rstrip().endswith("?"):
        return ClassificationResult("complete_user_facing_prompt", "first-person or question framing")

    if _BEHAVIOR_VERB_START.match(t):
        return ClassificationResult(
            "behavior_description",
            "imperative task-directive opening verb, no first-person/question framing",
        )

    # Imperative-but-ambiguous: grammatically could be typed by a user
    # ("Write me a poem") or could be a redteaming behavior label
    # ("Write a script that exploits X"). Flag rather than guess.
    return ClassificationResult(
        "complete_user_facing_prompt",
        "imperative sentence, addressable to an assistant, no template/wrapper markers "
        "(low-confidence: see module docstring on imperative ambiguity)",
    )


def classify_record(text: str, field_role: str) -> ClassificationResult:
    """field_role: 'prompt' | 'target'"""
    if field_role == "target":
        return classify_target_field(text)
    if field_role == "prompt":
        return classify_prompt_like_field(text)
    raise ValueError(f"unknown field_role: {field_role!r}")
