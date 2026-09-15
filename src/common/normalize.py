"""Prompt text normalisation used for deduplication and identity checks.

FROZEN. This function feeds the frozen benchmark's dedup pass. Changing its
behaviour changes which rows are considered duplicates, which changes the
benchmark, which invalidates data/frozen_v2/LATEST_BENCHMARK.json and every
SHA pinned against it.
"""

import re

__all__ = ["normalize_prompt"]


def normalize_prompt(text: str) -> str:
    """Lowercase, strip, and collapse all whitespace runs to single spaces."""
    return re.sub(r"\s+", " ", text.lower().strip())
