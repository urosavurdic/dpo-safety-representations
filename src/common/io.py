"""JSON and JSONL I/O with LF-stable writes.

Re-exports the canonical implementations from ``src.v2_io`` so there is exactly
one of each. ``src/v2_io.py`` is byte-hash-pinned by the benchmark gate and must
not be edited or moved, so this module wraps it rather than absorbing it.

``write_json`` keeps the ``(data, path)`` argument order used across the analysis
scripts, while delegating to the LF-stable writer. The two orders are opposite:
``write_json(data, path)`` but ``write_json_lf(path, data)``.
"""

from pathlib import Path
from typing import Any, Union

from src.v2_io import (
    canonical_json,
    load_json,
    load_jsonl,
    normalize_json_path,
    sha256_bytes,
    sha256_file,
    write_json_lf,
)

__all__ = [
    "canonical_json",
    "load_json",
    "load_jsonl",
    "normalize_json_path",
    "sha256_bytes",
    "sha256_file",
    "write_json",
    "write_json_lf",
]


def write_json(data: Any, path: Union[str, Path], indent: int = 2) -> None:
    """Write ``data`` to ``path`` as JSON with LF newlines, creating parents.

    Argument order is ``(data, path)``, matching the analysis scripts. Writes
    through ``write_json_lf``, which opens with ``newline=""`` so the bytes are
    identical on Windows and Linux -- the previous implementation here did not,
    producing CRLF locally and LF in Colab for the same content.
    """
    write_json_lf(Path(path), data, indent=indent)
