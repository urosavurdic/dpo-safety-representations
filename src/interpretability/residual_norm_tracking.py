"""
Diagnostic instrumentation for Next Steps item 4: track residual-stream
norm growth layer-by-layer, generation-step-by-generation-step, to
actually test (not just plausibly narrate) why multi-layer steering
degenerately collapses almost totally (49/50 on quadrant D, layers 14-28
simultaneously, see results/raw/steering_raw_D_MULTILAYER_14to28_DEPRECATED.json)
while single-layer steering mostly doesn't (3/50 at layer 21, see
results/raw/steering_raw_D_L21_exploratory_DEPRECATED.json).

eval_steering_v2.py's own docstring already offers a plausible mechanism
("each addition persists forward AND gets added to again at every
subsequent steered layer, so total injected magnitude compounds with
layer count") -- that's a hypothesis worth taking seriously (it's the
starting point this module is built to test), not a measured fact. This
module makes it measurable: does compounding actually push the residual
norm outside the range later layers/LayerNorm were trained on, and if so,
does correcting for it (without removing the steering direction itself)
actually prevent the collapse?

Pulled the ACTUAL deprecated-run outputs (not summarized secondhand) to
ground the hypothesis before instrumenting anything: the multi-layer
collapse isn't token soup, it's the model getting stuck in a tight loop
of refusal-flavored tokens ("unfortunately... unfortunately... WARNING
WARNING"). That's consistent with (but doesn't by itself prove) a
magnitude-driven mechanism: a large enough constant perturbation could
plausibly dominate the residual stream's later-layer computation, but a
distribution-collapse-under-greedy-decoding explanation (the perturbation
biases the output distribution toward a small token set, and greedy
decoding then loops on whichever of those tokens gets emitted first) is
also consistent with the same observation and isn't mutually exclusive
with the norm story. This module's baseline-vs-steered norm comparison is
the actual test of the norm-specific part of that hypothesis; it doesn't
by itself distinguish "norm left the trained range" from "greedy decoding
looped for output-distribution reasons unrelated to norm" -- both should
be checked against the collected data (see
eval_residual_norm_diagnostic.py's docstring for what a clean result would
look like either way).

No torch import at THIS module's top level is avoided here (unlike
run_full_steering.py/build_finding4_report.py) because tensor operations
are the whole point -- this module DOES require torch, same tier as
eval_steering_v2.py/eval_causal_ablation.py. Tests use tiny CPU tensors
and fake nn.Module decoder stacks (same pattern
tests/analysis/test_eval_causal_ablation.py already uses for
get_decoder_layers/register_ablation_hooks), not the real model.
"""
import torch


def steer_direction(hidden_states, direction, alpha):
    """Same operation as eval_steering_v2.steer_direction -- duplicated
    (not imported) to keep this module's dependency surface self-contained
    for anyone reading it in isolation; keep in sync by hand if that
    function's math ever changes."""
    direction = direction.to(dtype=hidden_states.dtype, device=hidden_states.device)
    return hidden_states + alpha * direction


class ResidualNormTracker:
    """Registers a forward hook on some/all decoder layers that records
    the L2 norm of the LAST token position's hidden state at every forward
    call. Under model.generate()'s KV-cache path, the first forward call
    processes the whole prompt (last token = end of prompt, "step 0" here);
    every call after that processes exactly one new token (one generation
    step each). So records[decoder_idx] ends up as a list of length
    (1 + num_new_tokens_generated_so_far), each entry a (batch,) numpy-free
    plain-Python-list of per-sequence norms -- deliberately NOT numpy at
    collection time, since this needs to work with whatever dtype/device
    the hook tensors are in without an extra .cpu().numpy() footgun inside
    the hot generation loop; callers convert to numpy once, after
    generation finishes (see collect()).

    Usage:
        tracker = ResidualNormTracker()
        tracker.register(model)                      # all decoder layers
        model.generate(...)
        records = tracker.collect()                  # {decoder_idx: (n_steps, batch) float array}
        tracker.remove()
    """

    def __init__(self):
        self._raw = {}  # {decoder_idx: [[norm_per_batch_item, ...], ...]}
        self._handles = []

    def _make_hook(self, decoder_idx):
        self._raw.setdefault(decoder_idx, [])

        def hook(module, inputs, output):
            h = output[0] if isinstance(output, tuple) else output
            last_token = h[:, -1, :]  # (batch, hidden_dim)
            norms = last_token.detach().float().norm(dim=-1).cpu().tolist()  # (batch,)
            self._raw[decoder_idx].append(norms)
            return output

        return hook

    def register(self, model, decoder_layers=None, layer_indices=None):
        """decoder_layers: pass explicitly to avoid re-importing
        get_decoder_layers here (keeps this module's import surface
        smaller, and lets tests pass a fake layer list directly without
        needing get_decoder_layers' HF-shaped model attribute path at
        all). layer_indices: which decoder indices to hook (default: all).
        """
        if decoder_layers is None:
            decoder_layers = model.model.layers
        indices = layer_indices if layer_indices is not None else range(len(decoder_layers))
        for idx in indices:
            handle = decoder_layers[idx].register_forward_hook(self._make_hook(idx))
            self._handles.append(handle)
        return self

    def remove(self):
        for h in self._handles:
            h.remove()
        self._handles = []

    def collect(self):
        """{decoder_idx: list of length n_steps, each a list of length
        batch_size} -- plain nested Python lists (JSON-serializable
        directly), not numpy, so eval_residual_norm_diagnostic.py can
        json.dump this without a conversion step. Use
        compute_baseline_range/compare_to_baseline (below) for the numpy
        side of the analysis."""
        return {str(idx): steps for idx, steps in self._raw.items()}

    def reset(self):
        for idx in self._raw:
            self._raw[idx] = []


def make_norm_preserving_steering_hook(direction, alpha):
    """Adds alpha*direction exactly like the normal steering hook, then
    rescales the RESULT back to the PRE-steering per-token norm -- the
    direction is still injected (changes the vector's angle/composition),
    but its magnitude contribution is removed, directly testing whether
    magnitude growth specifically (as opposed to the direction's presence
    at all) is what drives collapse. Falls back to the unscaled steered
    vector wherever the pre-steering norm is ~0 (nothing meaningful to
    preserve)."""
    def hook(module, inputs, output):
        h = output[0] if isinstance(output, tuple) else output
        original_norm = h.norm(dim=-1, keepdim=True)
        steered = steer_direction(h, direction, alpha)
        steered_norm = steered.norm(dim=-1, keepdim=True)
        safe_steered_norm = torch.where(steered_norm > 1e-8, steered_norm, torch.ones_like(steered_norm))
        rescaled = steered * (original_norm / safe_steered_norm)
        # Where original_norm itself is ~0, there's nothing to preserve -- leave the raw steered vector.
        result = torch.where(original_norm > 1e-8, rescaled, steered)
        if isinstance(output, tuple):
            return (result,) + output[1:]
        return result
    return hook


def make_norm_clipped_steering_hook(direction, alpha, max_norm):
    """Adds alpha*direction, then clips only vectors that EXCEED max_norm
    back down to exactly max_norm (unlike the norm-preserving hook, which
    always rescales to the pre-steering norm exactly) -- a gentler
    intervention: normal-magnitude tokens are left untouched, only ones
    the steering pushed past a ceiling (e.g. the baseline distribution's
    p99 at this layer, from compute_baseline_range) get pulled back."""
    def hook(module, inputs, output):
        h = output[0] if isinstance(output, tuple) else output
        steered = steer_direction(h, direction, alpha)
        steered_norm = steered.norm(dim=-1, keepdim=True)
        safe_norm = torch.where(steered_norm > 1e-8, steered_norm, torch.ones_like(steered_norm))
        scale = torch.clamp(max_norm / safe_norm, max=1.0)
        clipped = steered * scale
        if isinstance(output, tuple):
            return (clipped,) + output[1:]
        return clipped
    return hook


def compute_baseline_range(baseline_records):
    """baseline_records: {decoder_idx (str or int): list[step][batch] of
    norms}, as produced by ResidualNormTracker.collect() on an UNSTEERED
    run. Pools across steps AND batch items/prompts (the "typical range a
    token at this layer has, regardless of when in generation it occurs")
    -- returns {decoder_idx: {"mean", "std", "p50", "p95", "p99", "n"}}.
    Pure numpy, no torch -- CPU-testable and importable without the real
    model or torch installed."""
    import numpy as np

    out = {}
    for idx, steps in baseline_records.items():
        flat = [v for step in steps for v in step]
        arr = np.array(flat, dtype=float)
        out[str(idx)] = {
            "mean": float(arr.mean()), "std": float(arr.std()),
            "p50": float(np.percentile(arr, 50)), "p95": float(np.percentile(arr, 95)),
            "p99": float(np.percentile(arr, 99)), "n": int(arr.size),
        }
    return out


def compare_to_baseline(steered_records, baseline_range):
    """For each (decoder_idx, step), returns the mean norm across the
    batch at that step, plus how many standard deviations above the
    baseline's mean it is (z-score) and whether it exceeds the baseline's
    p99. {decoder_idx: [{"step", "mean_norm", "z_score", "exceeds_p99"}]}.
    Layers present in steered_records but absent from baseline_range are
    skipped (nothing to compare against) rather than raising -- a
    steering run may hook a different/wider set of layers than the
    baseline pass did."""
    import numpy as np

    out = {}
    for idx, steps in steered_records.items():
        if idx not in baseline_range:
            continue
        b = baseline_range[idx]
        entries = []
        for step_idx, step_norms in enumerate(steps):
            mean_norm = float(np.mean(step_norms))
            z = (mean_norm - b["mean"]) / b["std"] if b["std"] > 0 else float("inf") if mean_norm != b["mean"] else 0.0
            entries.append({
                "step": step_idx, "mean_norm": mean_norm,
                "z_score": z, "exceeds_p99": mean_norm > b["p99"],
            })
        out[idx] = entries
    return out


def first_step_exceeding_p99(comparison_for_layer):
    """Convenience: returns the first step index (or None) at which a
    layer's mean norm first exceeds the baseline's p99 -- useful for
    answering "does the blowup happen immediately, or build up over the
    course of generation" directly, since that distinguishes "steering
    itself is already out of range at token 1" from "compounding across
    generation steps is what pushes it out of range", which are different
    mechanisms with different fixes."""
    for entry in comparison_for_layer:
        if entry["exceeds_p99"]:
            return entry["step"]
    return None
