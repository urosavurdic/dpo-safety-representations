"""Per-row activation-delta injection hook.

The one genuinely new mechanism this extension needs: every existing hook in
the repo adds a single global vector to every position; this adds a
DIFFERENT vector per row, at one position.

Only ``last_prompt_only`` is implemented. The delta is measured at the final
prompt token, so that is where it is injected; the other modes named in the
plan are Stage-2/robustness and raise rather than silently doing something
plausible.

Causal semantics under KV-cached generation, stated precisely because it is
easy to get wrong:

* The hook fires on the OUTPUT of decoder block ``layer - 1``, i.e. the
  residual stream as it leaves ``hidden_states[layer]``. Layers above it,
  the final norm and lm_head all consume the modified state, so the first
  generated token's distribution changes.
* K/V cached at the final prompt position are perturbed for layers ABOVE the
  injection point only; layers at or below it cached their K/V from the
  unmodified stream. The perturbation therefore carries into generation
  through two channels -- the changed first token, and the perturbed upper
  cache that later tokens attend back to. It is neither a one-token blip nor
  a full-depth edit.
"""
from __future__ import annotations

import torch

from src.analysis.v2_pipeline import decoder_layers

LAST_PROMPT_ONLY = "last_prompt_only"
IMPLEMENTED_MODES = (LAST_PROMPT_ONLY,)
DEFERRED_MODES = ("hold", "prompt_positions_only")


def assert_greedy_single_beam(**generate_kwargs) -> None:
    """Beam search reorders and expands the batch, which would silently
    invalidate the batch->record_id mapping. Refuse rather than produce rows
    attributed to the wrong prompt."""
    num_beams = generate_kwargs.get("num_beams", 1)
    if num_beams not in (None, 1):
        raise ValueError(
            f"num_beams={num_beams}: PerRowDeltaInjector requires num_beams == 1. "
            "Beam search reorders the batch and breaks the row->delta mapping."
        )
    if generate_kwargs.get("do_sample", False):
        raise ValueError("do_sample must be False; this protocol is greedy.")


class PerRowDeltaInjector:
    """Adds ``coef * delta[record_id]`` at the final prompt position.

    Usage, once per batch::

        inj = PerRowDeltaInjector(delta_map, coef=1.0, layer=24).register(model)
        inj.set_batch([r["record_id"] for r in rows], inputs["attention_mask"])
        model.generate(**inputs, ...)
        inj.remove()
    """

    def __init__(
        self,
        delta_map: dict,
        coef: float,
        *,
        layer: int = 24,
        mode: str = LAST_PROMPT_ONLY,
        strict: bool = True,
    ) -> None:
        if mode in DEFERRED_MODES:
            raise NotImplementedError(
                f"mode={mode!r} is deliberately not implemented in this pass. "
                f"Implemented: {IMPLEMENTED_MODES}."
            )
        if mode not in IMPLEMENTED_MODES:
            raise ValueError(f"unknown mode {mode!r}; implemented {IMPLEMENTED_MODES}")
        if not delta_map:
            raise ValueError("delta_map is empty")

        self.delta_map = delta_map
        self.coef = float(coef)
        self.layer = int(layer)
        self.mode = mode
        self.strict = strict

        dims = {int(v.shape[-1]) for v in delta_map.values()}
        if len(dims) != 1:
            raise ValueError(f"delta_map has mixed hidden dims: {sorted(dims)}")
        self.hidden_dim = dims.pop()

        self._handle = None
        self._record_ids: list[str] | None = None
        self._prefill_done = False
        self.calls = 0
        self.injections = 0

    # -- lifecycle ---------------------------------------------------------

    def register(self, model, layers=None) -> "PerRowDeltaInjector":
        blocks = decoder_layers(model) if layers is None else layers
        index = self.layer - 1  # hidden_states[i] == output of block i-1
        if not 0 <= index < len(blocks):
            raise IndexError(
                f"layer {self.layer} -> block index {index} outside "
                f"0..{len(blocks) - 1}"
            )
        self._handle = blocks[index].register_forward_hook(self._hook)
        return self

    def remove(self) -> None:
        if self._handle is not None:
            self._handle.remove()
            self._handle = None

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.remove()
        return False

    @staticmethod
    def assert_left_padding(tokenizer) -> None:
        if getattr(tokenizer, "padding_side", None) != "left":
            raise ValueError(
                "tokenizer.padding_side must be 'left': the injector targets "
                "column -1 as the final prompt token for every row."
            )

    # -- per-batch state ---------------------------------------------------

    def set_batch(self, record_ids, attention_mask=None) -> None:
        ids = [str(r) for r in record_ids]
        missing = [r for r in ids if r not in self.delta_map]
        if missing:
            raise KeyError(
                f"{len(missing)} record_id(s) absent from delta_map, first few: "
                f"{missing[:5]}"
            )
        self._record_ids = ids
        self._attention_mask = attention_mask
        self._prefill_done = False

    # -- the hook ----------------------------------------------------------

    def _hook(self, _module, _inputs, output):
        if isinstance(output, tuple):
            hidden = output[0]
            rest = output[1:]
            was_tuple = True
        else:
            hidden = output
            rest = ()
            was_tuple = False

        if self._record_ids is None:
            raise RuntimeError(
                "PerRowDeltaInjector fired before set_batch(); the "
                "batch->record_id mapping must be supplied by the caller, "
                "never inferred from tensor contents."
            )

        if hidden.dim() != 3:
            raise RuntimeError(
                f"expected [batch, sequence, hidden]; got {tuple(hidden.shape)}"
            )
        batch, seq, hid = hidden.shape
        if batch != len(self._record_ids):
            raise RuntimeError(
                f"batch {batch} != {len(self._record_ids)} record_ids; the "
                "mapping would be misaligned."
            )
        if hid != self.hidden_dim:
            raise RuntimeError(
                f"hidden dim {hid} != delta dim {self.hidden_dim}"
            )

        self.calls += 1
        is_prefill = not self._prefill_done

        # Explicit state, corroborated by -- never replaced by -- seq_len.
        if not is_prefill and seq != 1 and self.strict:
            raise RuntimeError(
                f"decode step received seq_len={seq}; expected 1. This looks "
                "like a re-prefill, which would mean the delta was already "
                "consumed at a different position. Refusing to continue."
            )

        if is_prefill:
            self._prefill_done = True
            addend = torch.zeros_like(hidden)
            rows = torch.stack(
                [
                    torch.as_tensor(
                        self.delta_map[rid],
                        dtype=hidden.dtype,
                        device=hidden.device,
                    )
                    for rid in self._record_ids
                ]
            )
            # Left padding puts the true final prompt token at column -1 for
            # every row regardless of prompt length.
            addend[:, -1, :] = self.coef * rows
            self.injections += 1
            new_hidden = hidden + addend  # never in-place
        else:
            new_hidden = hidden  # last_prompt_only: decode is a strict no-op

        if was_tuple:
            return (new_hidden,) + rest
        return new_hidden
