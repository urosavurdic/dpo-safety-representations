"""Depth attribution: where in the network each training stage writes the contrast.

The residual stream accumulates, so for any direction ``d`` the endpoint
projection decomposes exactly across depth::

    h_L . d  =  h_0 . d  +  SUM_l ( h_l - h_{l-1} ) . d

``write[l] = (h_l - h_{l-1}) . d_l`` is therefore an accounting identity, not an
estimate, and it needs no ablation and no generation -- only the block outputs
already cached in ``results/activations/``.

In Qwen2 the per-layer update itself splits into two sublayer terms, since the
block is sequential pre-norm (``Qwen2DecoderLayer.forward``)::

    a_l = h_{l-1} + Attn_l( input_layernorm(h_{l-1}) )
    h_l = a_l     + MLP_l(  post_attention_layernorm(a_l) )
    =>  h_l - h_{l-1} = attn_out_l + mlp_out_l

Splitting ``write`` into those two parts is the natural next step, but the
sublayer outputs are NOT in the cache -- ``activation_batch`` stores only
``output_hidden_states``. It needs a hooked forward pass (no generation,
~1.5 h GPU for nine stages). ``split_available`` says whether that has been
done; until then this module reports the block-level attribution, which is the
free half and answers the depth question on its own.

Two cautions, enforced by what this module reports:

* residual-stream norms grow with depth, so raw ``write`` magnitudes are not
  comparable across layers. The A-minus-D gap is, because the growth is common
  to both quadrants and cancels to first order;
* a large write at layer ``l`` that a later layer cancels contributes nothing to
  the endpoint. ``cumulative`` is always reported beside ``write``.
"""
from __future__ import annotations

import numpy as np

from src.common.activations import l2_normalize
from src.common.quadrants import QUADRANTS
from src.stagecmp import STATUS_EXPLORATORY
from src.stagecmp.loaders import load_stage_arrays, stage_direction

SUBLAYERS = ("attn", "mlp")


def split_available(stage: str, act_dir) -> bool:
    """True once a hooked forward pass has written per-sublayer outputs."""
    from pathlib import Path

    base = Path(act_dir)
    return all((base / f"{stage}_{s}_final.npy").exists() for s in SUBLAYERS)


def layer_writes(activations: np.ndarray, direction: np.ndarray) -> np.ndarray:
    """``(n_prompts, n_layers-1)`` per-layer write onto the direction.

    ``activations`` is ``(n, n_layers, hidden)``; ``direction`` is
    ``(n_layers, hidden)`` for a per-layer axis, or ``(hidden,)`` for one fixed
    axis used at every depth.
    """
    diffs = np.diff(activations, axis=1)                     # (n, L-1, hidden)
    if direction.ndim == 1:
        unit = l2_normalize(direction)
        return np.einsum("nlh,h->nl", diffs, unit)
    # direction[l] belongs to the state AFTER block l, so the write INTO layer l
    # is read with layer l's own axis: drop index 0.
    unit = np.stack([l2_normalize(direction[l]) for l in range(1, direction.shape[0])])
    return np.einsum("nlh,lh->nl", diffs, unit)


def profile(
    stage: str,
    act_dir,
    directions_dir,
    pooling: str = "final_token",
    reference_stage: str | None = None,
    reference_layer: int | None = None,
) -> dict:
    """Per-layer write and cumulative build-up, by quadrant, for one stage.

    ``reference_stage`` projects onto ANOTHER stage's axis -- the fixed-ruler
    variant. Without it a change in the curve could be the axis moving rather
    than the writes changing, so both are reported wherever the comparison
    matters (same reasoning as the fixed-reference projections in the 2x2).
    """
    arr, meta = load_stage_arrays(stage, pooling, act_dir)
    quads = np.array([row["quadrant"] for row in meta])

    axis_stage = reference_stage or stage
    direction = stage_direction(axis_stage, directions_dir)
    if reference_layer is not None:
        direction = direction[reference_layer]

    writes = layer_writes(arr, direction)                    # (n, L-1)
    cumulative = np.cumsum(writes, axis=1)
    n_layers = writes.shape[1]

    per_quadrant = {}
    for q in QUADRANTS:
        mask = quads == q
        if not mask.any():
            continue
        per_quadrant[q] = {
            "n": int(mask.sum()),
            "write": [float(v) for v in writes[mask].mean(axis=0)],
            "cumulative": [float(v) for v in cumulative[mask].mean(axis=0)],
        }

    out = {
        "status": STATUS_EXPLORATORY,
        "stage": stage,
        "pooling": pooling,
        "axis_stage": axis_stage,
        "axis_layer": reference_layer,
        "layers": list(range(1, n_layers + 1)),
        "per_quadrant": per_quadrant,
        "sublayer_split_available": split_available(stage, act_dir),
        "reading": (
            "write[l] = (h_l - h_{l-1}) . d is exact, not an estimate. Raw "
            "magnitudes grow with depth and are not comparable across layers; "
            "read the A-minus-D gap for the contrast-specific part, and read "
            "cumulative beside write because a write a later layer cancels "
            "contributes nothing to the endpoint."
        ),
    }
    if "A" in per_quadrant and "D" in per_quadrant:
        gap = np.array(per_quadrant["A"]["write"]) - np.array(per_quadrant["D"]["write"])
        out["ad_gap"] = {
            "write": [float(v) for v in gap],
            "cumulative": [float(v) for v in np.cumsum(gap)],
            "peak_layer": int(np.argmax(gap) + 1),
            "peak_value": float(gap.max()),
            "share_in_24_28": float(
                gap[23:28].sum() / gap.sum() if abs(gap.sum()) > 1e-12 else np.nan
            ),
        }
    return out


def across_stages(
    stages,
    act_dir,
    directions_dir,
    pooling: str = "final_token",
    reference_stage: str | None = None,
) -> dict:
    """``profile`` for several stages under one convention."""
    return {
        "status": STATUS_EXPLORATORY,
        "pooling": pooling,
        "reference_stage": reference_stage,
        "per_stage": {
            s: profile(s, act_dir, directions_dir, pooling, reference_stage)
            for s in stages
        },
    }
