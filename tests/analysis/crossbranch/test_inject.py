"""PerRowDeltaInjector hook math, on fake decoder layers.

Follows the repo's existing pattern for hook tests (see
tests/analysis/test_eval_causal_ablation.py): a fake nn.Module stack, so the
injection arithmetic and the prefill/decode state machine are verified
without a real model.

The load-bearing assertion here is that decode steps are a STRICT no-op. The
hook fires on every forward pass, so "last_prompt_only" is only true because
that branch is explicitly inert -- not because the hook stops being called.
"""
import numpy as np
import pytest
import torch
from torch import nn

from src.analysis.crossbranch.inject import PerRowDeltaInjector, assert_greedy_single_beam

HIDDEN = 4


class FakeDecoderLayer(nn.Module):
    def __init__(self, tuple_output=True, extra=None):
        super().__init__()
        self.tuple_output = tuple_output
        self.extra = extra

    def forward(self, x):
        if self.tuple_output:
            return (x,) + ((self.extra,) if self.extra is not None else ())
        return x


class FakeInner(nn.Module):
    def __init__(self, n=3, **kw):
        super().__init__()
        self.layers = nn.ModuleList([FakeDecoderLayer(**kw) for _ in range(n)])


class FakeCausalLM(nn.Module):
    def __init__(self, n=3, **kw):
        super().__init__()
        self.model = FakeInner(n, **kw)


def delta_map(ids, scale=1.0):
    return {
        rid: (np.arange(HIDDEN, dtype=np.float32) + i + 1) * scale
        for i, rid in enumerate(ids)
    }


def run(layer_module, hidden):
    return layer_module(hidden)


# ---------------------------------------------------------------------------


def test_prefill_injects_only_at_final_column_and_per_row():
    ids = ["r0", "r1"]
    dm = delta_map(ids)
    model = FakeCausalLM()
    inj = PerRowDeltaInjector(dm, coef=1.0, layer=1).register(model)
    inj.set_batch(ids)

    h = torch.zeros(2, 5, HIDDEN)
    out = run(model.model.layers[0], h)[0]

    # every non-final column untouched
    torch.testing.assert_close(out[:, :-1, :], torch.zeros(2, 4, HIDDEN))
    # final column carries that row's own delta
    torch.testing.assert_close(out[0, -1, :], torch.tensor(dm["r0"]))
    torch.testing.assert_close(out[1, -1, :], torch.tensor(dm["r1"]))
    inj.remove()


def test_decode_steps_are_a_strict_noop():
    ids = ["r0"]
    model = FakeCausalLM()
    inj = PerRowDeltaInjector(delta_map(ids), coef=1.0, layer=1).register(model)
    inj.set_batch(ids)

    run(model.model.layers[0], torch.zeros(1, 6, HIDDEN))       # prefill
    for _ in range(3):
        h = torch.ones(1, 1, HIDDEN)
        out = run(model.model.layers[0], h)[0]
        torch.testing.assert_close(out, h)                      # unchanged

    assert inj.calls == 4 and inj.injections == 1
    inj.remove()


def test_set_batch_resets_state_for_the_next_batch():
    ids = ["r0"]
    model = FakeCausalLM()
    inj = PerRowDeltaInjector(delta_map(ids), coef=1.0, layer=1).register(model)
    for _ in range(2):
        inj.set_batch(ids)
        run(model.model.layers[0], torch.zeros(1, 3, HIDDEN))
        run(model.model.layers[0], torch.zeros(1, 1, HIDDEN))
    assert inj.injections == 2
    inj.remove()


def test_left_padding_targets_the_true_final_token_for_every_row():
    # Rows of different real length are left-padded, so column -1 is the real
    # final prompt token for all of them.
    ids = ["r0", "r1", "r2"]
    dm = delta_map(ids)
    model = FakeCausalLM()
    inj = PerRowDeltaInjector(dm, coef=1.0, layer=1).register(model)
    mask = torch.tensor([[0, 0, 1, 1], [0, 1, 1, 1], [1, 1, 1, 1]])
    inj.set_batch(ids, mask)
    out = run(model.model.layers[0], torch.zeros(3, 4, HIDDEN))[0]
    for i, rid in enumerate(ids):
        torch.testing.assert_close(out[i, -1, :], torch.tensor(dm[rid]))
        torch.testing.assert_close(out[i, :-1, :], torch.zeros(3, HIDDEN))
    inj.remove()


def test_tuple_output_preserves_trailing_elements():
    ids = ["r0"]
    marker = torch.tensor([42.0])
    model = FakeCausalLM(tuple_output=True, extra=marker)
    inj = PerRowDeltaInjector(delta_map(ids), coef=1.0, layer=1).register(model)
    inj.set_batch(ids)
    out = run(model.model.layers[0], torch.zeros(1, 3, HIDDEN))
    assert isinstance(out, tuple) and len(out) == 2
    torch.testing.assert_close(out[1], marker)
    inj.remove()


def test_bare_tensor_output_is_supported():
    ids = ["r0"]
    model = FakeCausalLM(tuple_output=False)
    inj = PerRowDeltaInjector(delta_map(ids), coef=1.0, layer=1).register(model)
    inj.set_batch(ids)
    out = model.model.layers[0](torch.zeros(1, 3, HIDDEN))
    assert isinstance(out, torch.Tensor)
    torch.testing.assert_close(out[0, -1, :], torch.tensor(delta_map(ids)["r0"]))
    inj.remove()


def test_coefficient_zero_is_an_exact_noop_and_effect_is_linear():
    ids = ["r0"]
    dm = delta_map(ids)
    for coef, expect in ((0.0, 0.0), (0.5, 0.5), (2.0, 2.0)):
        model = FakeCausalLM()
        inj = PerRowDeltaInjector(dm, coef=coef, layer=1).register(model)
        inj.set_batch(ids)
        out = run(model.model.layers[0], torch.zeros(1, 3, HIDDEN))[0]
        torch.testing.assert_close(
            out[0, -1, :], torch.tensor(dm["r0"]) * expect
        )
        inj.remove()


def test_does_not_mutate_the_incoming_tensor():
    ids = ["r0"]
    model = FakeCausalLM()
    inj = PerRowDeltaInjector(delta_map(ids), coef=1.0, layer=1).register(model)
    inj.set_batch(ids)
    h = torch.zeros(1, 3, HIDDEN)
    run(model.model.layers[0], h)
    torch.testing.assert_close(h, torch.zeros(1, 3, HIDDEN))
    inj.remove()


def test_remove_restores_the_layer():
    ids = ["r0"]
    model = FakeCausalLM()
    inj = PerRowDeltaInjector(delta_map(ids), coef=1.0, layer=1).register(model)
    inj.set_batch(ids)
    inj.remove()
    out = run(model.model.layers[0], torch.zeros(1, 3, HIDDEN))[0]
    torch.testing.assert_close(out, torch.zeros(1, 3, HIDDEN))


# ---- failure modes that must raise rather than mis-attribute --------------


def test_firing_before_set_batch_raises():
    model = FakeCausalLM()
    inj = PerRowDeltaInjector(delta_map(["r0"]), coef=1.0, layer=1).register(model)
    with pytest.raises(RuntimeError, match="before set_batch"):
        run(model.model.layers[0], torch.zeros(1, 3, HIDDEN))
    inj.remove()


def test_unknown_record_id_raises_naming_it():
    inj = PerRowDeltaInjector(delta_map(["r0"]), coef=1.0, layer=1)
    with pytest.raises(KeyError, match="ghost"):
        inj.set_batch(["r0", "ghost"])


def test_batch_size_mismatch_raises():
    ids = ["r0", "r1"]
    model = FakeCausalLM()
    inj = PerRowDeltaInjector(delta_map(ids), coef=1.0, layer=1).register(model)
    inj.set_batch(ids)
    with pytest.raises(RuntimeError, match="record_ids"):
        run(model.model.layers[0], torch.zeros(3, 4, HIDDEN))
    inj.remove()


def test_hidden_dim_mismatch_raises():
    ids = ["r0"]
    model = FakeCausalLM()
    inj = PerRowDeltaInjector(delta_map(ids), coef=1.0, layer=1).register(model)
    inj.set_batch(ids)
    with pytest.raises(RuntimeError, match="hidden dim"):
        run(model.model.layers[0], torch.zeros(1, 3, HIDDEN + 2))
    inj.remove()


def test_non_three_dimensional_hidden_state_raises():
    ids = ["r0"]
    model = FakeCausalLM()
    inj = PerRowDeltaInjector(delta_map(ids), coef=1.0, layer=1).register(model)
    inj.set_batch(ids)
    with pytest.raises(RuntimeError, match=r"\[batch, sequence, hidden\]"):
        run(model.model.layers[0], torch.zeros(1, HIDDEN))
    inj.remove()


def test_reprefill_during_decode_raises_instead_of_injecting_twice():
    ids = ["r0"]
    model = FakeCausalLM()
    inj = PerRowDeltaInjector(delta_map(ids), coef=1.0, layer=1).register(model)
    inj.set_batch(ids)
    run(model.model.layers[0], torch.zeros(1, 5, HIDDEN))     # prefill
    with pytest.raises(RuntimeError, match="re-prefill"):
        run(model.model.layers[0], torch.zeros(1, 5, HIDDEN))  # not seq_len 1
    inj.remove()


def test_deferred_modes_raise_rather_than_silently_doing_something():
    for mode in ("hold", "prompt_positions_only"):
        with pytest.raises(NotImplementedError, match="not implemented"):
            PerRowDeltaInjector(delta_map(["r0"]), coef=1.0, mode=mode)


def test_mixed_hidden_dims_in_delta_map_raise():
    dm = {"a": np.zeros(4, np.float32), "b": np.zeros(5, np.float32)}
    with pytest.raises(ValueError, match="mixed hidden dims"):
        PerRowDeltaInjector(dm, coef=1.0)


def test_layer_index_out_of_range_raises():
    model = FakeCausalLM(n=2)
    with pytest.raises(IndexError):
        PerRowDeltaInjector(delta_map(["r0"]), coef=1.0, layer=99).register(model)


def test_beam_search_is_refused():
    with pytest.raises(ValueError, match="num_beams"):
        assert_greedy_single_beam(num_beams=4)
    with pytest.raises(ValueError, match="do_sample"):
        assert_greedy_single_beam(do_sample=True)
    assert_greedy_single_beam(num_beams=1, do_sample=False)  # ok


def test_left_padding_is_asserted_not_assumed():
    class Tok:
        padding_side = "right"

    with pytest.raises(ValueError, match="padding_side must be 'left'"):
        PerRowDeltaInjector.assert_left_padding(Tok())
    Tok.padding_side = "left"
    PerRowDeltaInjector.assert_left_padding(Tok())
