"""Loading extracted activations and refusal directions.

Every consumer passes ``act_dir`` / ``directions_dir`` explicitly rather than
relying on a default held in this module. That is deliberate: several tests
monkeypatch ``ACT_DIR`` as an attribute of the *calling* module, and
``results/activations/`` exists in a normal checkout, so a loader that silently
read a default from here would quietly read the real 2 GB of activations instead
of the test's fixture -- passing or failing for reasons unrelated to the test.

Pooling suffixes are not version markers. ``_final`` is the final prompt token;
``_pooled`` is the mean over the last five non-padding tokens. See docs/NAMING.md.
"""

from pathlib import Path
from typing import Any, Optional, Union

import numpy as np

__all__ = [
    "ACT_DIR",
    "DIRECTIONS_DIR",
    "POOLING_SUFFIX",
    "l2_normalize",
    "activation_path",
    "metadata_path",
    "load_metadata",
    "load_activations",
    "activations_available",
    "resolve_direction_path",
    "load_direction",
]

ACT_DIR = Path("results/activations")
DIRECTIONS_DIR = Path("results/refusal_direction")

# Pooling mode -> the suffix it is stored under.
POOLING_SUFFIX = {"final_token": "final", "mean_last5": "pooled"}

# Direction filenames, most specific first. A stage may have been written by any
# of three generations of the pipeline; the first that exists wins.
_DIRECTION_NAMES = (
    "{stage}_direction_final.npy",
    "{stage}_v2_direction.npy",
    "{stage}_direction.npy",
)


def l2_normalize(v: np.ndarray) -> np.ndarray:
    """Unit-normalise ``v``, returning it unchanged if its norm is zero."""
    n = np.linalg.norm(v)
    return v / n if n > 0 else v


def _suffix(pooling: str) -> str:
    if pooling in POOLING_SUFFIX:
        return POOLING_SUFFIX[pooling]
    if pooling in POOLING_SUFFIX.values():
        return pooling  # already a suffix ("final" / "pooled")
    raise ValueError(
        f"unknown pooling {pooling!r}; expected one of "
        f"{sorted(POOLING_SUFFIX)} or {sorted(set(POOLING_SUFFIX.values()))}"
    )


def activation_path(stage: str, pooling: str, act_dir: Union[str, Path]) -> Path:
    return Path(act_dir) / f"{stage}_{_suffix(pooling)}.npy"


def metadata_path(stage: str, act_dir: Union[str, Path]) -> Path:
    return Path(act_dir) / f"{stage}_metadata.json"


def load_metadata(stage: str, act_dir: Union[str, Path]) -> Any:
    """Return the per-row metadata list for ``stage``."""
    import json

    return json.loads(
        metadata_path(stage, act_dir).read_text(encoding="utf-8", errors="replace")
    )


def load_activations(stage: str, pooling: str, act_dir: Union[str, Path]) -> np.ndarray:
    """Return the ``(n_prompts, n_layers, hidden)`` array for ``stage``."""
    return np.load(activation_path(stage, pooling, act_dir))


def activations_available(stage: str, pooling: str, act_dir: Union[str, Path]) -> bool:
    """Whether ``stage`` has both an activation array and its metadata on disk.

    Stages are extracted across separate sessions, so at any moment some may be
    missing. Checked by file existence, not by contacting the model hub.
    """
    return (
        activation_path(stage, pooling, act_dir).exists()
        and metadata_path(stage, act_dir).exists()
    )


def resolve_direction_path(
    stage: str, directions_dir: Union[str, Path] = DIRECTIONS_DIR
) -> Path:
    """Return the direction array path for ``stage``, trying each naming generation."""
    directions_dir = Path(directions_dir)
    tried = []
    for template in _DIRECTION_NAMES:
        p = directions_dir / template.format(stage=stage)
        tried.append(p.name)
        if p.exists():
            return p
    raise FileNotFoundError(
        f"no direction array for {stage} in {directions_dir} (tried: {', '.join(tried)})"
    )


def load_direction(
    stage: str,
    layer: Optional[int] = None,
    directions_dir: Union[str, Path] = DIRECTIONS_DIR,
) -> np.ndarray:
    """Load ``stage``'s direction, optionally one layer of it.

    When ``layer`` is given the result is checked to be unit norm. A direction
    that is not unit norm means the array is not what the caller thinks it is,
    and silently scaling an intervention by its magnitude would corrupt every
    downstream number.
    """
    path = resolve_direction_path(stage, directions_dir)
    arr = np.load(path)
    if layer is None:
        return arr
    if arr.ndim != 2 or not 0 <= layer < arr.shape[0]:
        raise RuntimeError(f"{path}: expected (layers, hidden); got {arr.shape}")
    d = np.asarray(arr[layer], dtype=np.float64)
    norm = float(np.linalg.norm(d))
    if not np.isfinite(norm) or abs(norm - 1.0) > 1e-3:
        raise RuntimeError(f"{path} layer {layer}: direction norm {norm:.6f}, expected ~1.0")
    return d
