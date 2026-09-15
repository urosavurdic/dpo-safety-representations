"""The four evaluation quadrants and the response categories.

The benchmark crosses intent against surface form. Quadrant letters appear in
filenames and result rows throughout the repo but are defined nowhere else, so
they are spelled out here.
"""

__all__ = ["QUADRANTS", "QUADRANT_MEANINGS", "CATEGORIES"]

QUADRANTS = ["A", "B", "C", "D"]

QUADRANT_MEANINGS = {
    "A": "harmful intent, overt harmful wording",
    "B": "benign intent, harmful-sounding wording",
    "C": "harmful intent, reduced-cue wording",
    "D": "benign intent, plain wording",
}

# Response categories, in the classifier's precedence order.
CATEGORIES = ["degenerate", "refusal", "soft_deflection", "comply"]
