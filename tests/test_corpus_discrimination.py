"""Regression tests for src.corpus_discrimination's eval-set file reading.

Repository JSONL/text inputs are UTF-8 (see RELEASE-HARDENING ONBOARDING).
load_quadrant_texts previously opened the eval-set file with
open(eval_path) and no explicit encoding, so the file was decoded using
whatever locale.getpreferredencoding() the interpreter falls back to when
no encoding is given. That default is UTF-8 on this sandbox host (Python
here runs in UTF-8 mode, so locale is never consulted), but on the
researcher's Windows host the platform default is a locale-specific code
page (e.g. cp1252), which raises UnicodeDecodeError on the non-ASCII
prompt text the real eval set contains. That platform difference can't be
reproduced by flipping locale settings on this host (UTF-8 mode overrides
locale-based encoding resolution entirely), so the regression guard below
instead asserts, directly, that the file is always opened with an
explicit encoding="utf-8" -- the fix that is correct on every platform
regardless of locale or default-encoding resolution.
"""
from __future__ import annotations

import builtins
import json
from pathlib import Path
from unittest import mock

from src.corpus_discrimination import load_quadrant_texts


def _write_eval_set(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")


def test_load_quadrant_texts_opens_eval_file_with_explicit_utf8(tmp_path):
    eval_path = tmp_path / "eval_set.jsonl"
    _write_eval_set(
        eval_path,
        [{"quadrant": "A", "prompt": "plain ascii prompt"}],
    )

    real_open = builtins.open
    with mock.patch("builtins.open", wraps=real_open) as mocked_open:
        load_quadrant_texts(str(eval_path), "A")

    matching_calls = [
        call
        for call in mocked_open.call_args_list
        if call.args and str(call.args[0]) == str(eval_path)
    ]
    assert matching_calls, "load_quadrant_texts never opened the eval file"
    call = matching_calls[0]
    encoding = call.kwargs.get("encoding") or (
        call.args[1] if len(call.args) > 1 else None
    )
    assert encoding == "utf-8", (
        "eval-set file must be opened with an explicit encoding='utf-8' "
        f"(got encoding={encoding!r}); relying on the platform default "
        "raises UnicodeDecodeError on non-UTF-8-default hosts (e.g. "
        "Windows with a cp1252 code page) for the repository's non-ASCII "
        "prompt text"
    )


def test_load_quadrant_texts_reads_non_ascii_prompts_and_filters_by_quadrant(
    tmp_path,
):
    eval_path = tmp_path / "eval_set.jsonl"
    _write_eval_set(
        eval_path,
        [
            {"quadrant": "A", "prompt": "curly quote \u2019 and em dash \u2014"},
            {"quadrant": "A", "prompt": "non-latin \u4f60\u597d text"},
            {"quadrant": "D", "prompt": "should not be included"},
        ],
    )

    texts = load_quadrant_texts(str(eval_path), "A")

    assert texts == [
        "curly quote \u2019 and em dash \u2014",
        "non-latin \u4f60\u597d text",
    ]
