"""Stage-2 analyzer: no gate, just per-quadrant arm comparisons with a paired
bootstrap. Toy data, CPU only. The frozen classifier is used as-is.
"""
import pytest

from src.crossbranch import analyze_stage2 as S

TEXT = {
    "comply": "Sure! Here is how: first do this, then do that, then finish up.",
    "refusal": "I'm sorry, but I can't help with that.",
    "soft_deflection": "It is not possible to do this safely.",
    "degenerate": "blah blah blah " * 20,
}


def _rows(labels, quads):
    return [
        {"record_id": f"r{i}", "prompt": f"p{i}", "quadrant": quads[i],
         "response": TEXT[labels[i]]}
        for i in range(len(labels))
    ]


def _sanity_labels():
    """Confirm the toy strings classify the way the tests below assume."""
    from src.common.refusal_classifier import classify_completion
    for name, text in TEXT.items():
        assert classify_completion(text) == name, (name, classify_completion(text))


def build(scenario, quads, *, coef=1.0, with_own=False):
    """scenario: dict of condition-key -> per-row label list.

    Keys use the plain model names and bare arm names; this helper appends
    the @coef suffix to the arm keys exactly as analyze_stage2 expects.
    """
    raw = {}
    for key, labels in scenario.items():
        if key in (S.BASELINE, S.REFERENCE):
            raw[key] = _rows(labels, quads)
        else:
            raw[f"{key}@{coef:g}"] = _rows(labels, quads)
    return raw


# All-one-quadrant helpers keep the arithmetic checkable by hand.
def one_quadrant(q, n):
    return [q] * n


def full_scenario(quads, *, base, ref, identity=None, shuf=None, norm=None,
                  dose=None, dir_s=None, dir_t=None):
    n = len(quads)
    def fill(x):
        return x if x is not None else ["comply"] * n
    return {
        S.BASELINE: base,
        S.REFERENCE: ref,
        S.IDENTITY: fill(identity),
        S.SHUF_WQ: fill(shuf),
        S.NORMMATCHED: fill(norm),
        S.DOSEMATCHED: fill(dose),
        S.DIR_SOURCE: fill(dir_s),
        S.DIR_TARGET: fill(dir_t),
    }


# ---------------------------------------------------------------------------


def test_toy_strings_classify_as_expected():
    _sanity_labels()


def test_dtv_is_minus_tv_when_arm_equals_reference():
    quads = one_quadrant("C", 12)
    base = ["comply"] * 12
    ref = ["refusal"] * 12
    raw = build(full_scenario(quads, base=base, ref=ref, identity=["refusal"] * 12),
                quads)
    out = S.analyze_stage2(raw, b=200)
    c = out["per_quadrant"]["C"]
    assert c["tv_baseline_to_reference"] == pytest.approx(1.0)
    # arm == reference -> TV(arm,ref)=0 -> dTV = 0 - 1 = -1
    assert c["arms"][S.IDENTITY]["dtv"]["point"] == pytest.approx(-1.0)


def test_dtv_is_zero_when_arm_equals_baseline():
    quads = one_quadrant("C", 10)
    base = ["comply"] * 10
    ref = ["refusal"] * 10
    raw = build(full_scenario(quads, base=base, ref=ref, identity=["comply"] * 10),
                quads)
    out = S.analyze_stage2(raw, b=200)
    assert out["per_quadrant"]["C"]["arms"][S.IDENTITY]["dtv"]["point"] == pytest.approx(0.0)


def test_contrast_point_is_arm_a_dtv_minus_arm_b_dtv():
    quads = one_quadrant("C", 16)
    base = ["comply"] * 16
    ref = ["refusal"] * 16
    raw = build(
        full_scenario(
            quads, base=base, ref=ref,
            identity=["refusal"] * 16,               # dTV = -1
            shuf=["refusal"] * 8 + ["comply"] * 8,   # partial move
        ),
        quads,
    )
    out = S.analyze_stage2(raw, b=200)
    c = out["per_quadrant"]["C"]
    a = c["arms"][S.IDENTITY]["dtv"]["point"]
    b = c["arms"][S.SHUF_WQ]["dtv"]["point"]
    got = c["contrasts"]["xfer_delta_source_identity__minus__xfer_delta_source_shuf_wq"]
    assert got["dtv_diff"]["point"] == pytest.approx(a - b)
    assert a < b  # identity moved further toward B3


def test_target_label_match_perfect_and_floor():
    quads = one_quadrant("C", 12)
    base = ["comply"] * 12
    ref = ["refusal"] * 8 + ["comply"] * 4
    raw = build(
        full_scenario(
            quads, base=base, ref=ref,
            identity=list(ref),          # exact match with B3
        ),
        quads,
    )
    out = S.analyze_stage2(raw, b=200)
    c = out["per_quadrant"]["C"]
    assert c["arms"][S.IDENTITY]["target_label_match"]["point"] == pytest.approx(1.0)
    # dir_source defaulted to all-comply == baseline, so its per-prompt
    # agreement with B3 equals the baseline-vs-B3 floor
    assert (
        c["arms"][S.DIR_SOURCE]["target_label_match"]["point"]
        == pytest.approx(c["baseline_target_label_match"]["point"])
    )


def test_refusal_shift_sign_in_quadrant_a():
    quads = one_quadrant("A", 20)
    base = ["comply"] * 20
    ref = ["refusal"] * 20
    raw = build(full_scenario(quads, base=base, ref=ref, identity=["refusal"] * 20),
                quads)
    out = S.analyze_stage2(raw, b=200)
    a = out["per_quadrant"]["A"]
    assert a["primary_refusal_metric"] == "refusal"
    assert a["arms"][S.IDENTITY]["refusal_shift"]["point"] == pytest.approx(1.0)


def test_overrefusal_is_the_primary_metric_for_b_and_d():
    quads = ["B"] * 10 + ["D"] * 10
    base = ["comply"] * 20
    ref = ["comply"] * 20
    raw = build(full_scenario(quads, base=base, ref=ref), quads)
    out = S.analyze_stage2(raw, b=100)
    assert out["per_quadrant"]["B"]["primary_refusal_metric"] == "refusal_or_soft_deflection"
    assert out["per_quadrant"]["D"]["primary_refusal_metric"] == "refusal_or_soft_deflection"


def test_quadrants_are_scored_independently():
    quads = ["A"] * 10 + ["C"] * 10
    base = ["comply"] * 20
    # B3 only shifts in C; A stays put
    ref = ["comply"] * 10 + ["refusal"] * 10
    raw = build(
        full_scenario(quads, base=base, ref=ref,
                      identity=["comply"] * 10 + ["refusal"] * 10),
        quads,
    )
    out = S.analyze_stage2(raw, b=200)
    assert out["per_quadrant"]["A"]["n"] == 10
    assert out["per_quadrant"]["C"]["n"] == 10
    assert out["per_quadrant"]["A"]["tv_baseline_to_reference"] == pytest.approx(0.0)
    assert out["per_quadrant"]["C"]["tv_baseline_to_reference"] == pytest.approx(1.0)
    # identity reproduces B3 in C, does nothing in A
    assert out["per_quadrant"]["C"]["arms"][S.IDENTITY]["dtv"]["point"] == pytest.approx(-1.0)
    assert out["per_quadrant"]["A"]["arms"][S.IDENTITY]["dtv"]["point"] == pytest.approx(0.0)


def test_bootstrap_is_deterministic_at_a_fixed_seed():
    quads = one_quadrant("C", 14)
    base = ["comply"] * 14
    ref = ["refusal"] * 7 + ["comply"] * 7
    raw = build(
        full_scenario(quads, base=base, ref=ref,
                      identity=["refusal"] * 4 + ["comply"] * 10),
        quads,
    )
    a = S.analyze_stage2(raw, b=300, seed=20260904)
    b = S.analyze_stage2(raw, b=300, seed=20260904)
    assert a == b
    c = S.analyze_stage2(raw, b=300, seed=1)
    assert (
        c["per_quadrant"]["C"]["arms"][S.IDENTITY]["dtv"]["ci_low"]
        != a["per_quadrant"]["C"]["arms"][S.IDENTITY]["dtv"]["ci_low"]
    )


def test_ci_brackets_the_point_estimate_everywhere():
    quads = ["C"] * 20 + ["B"] * 20
    base = ["comply"] * 40
    ref = ["refusal"] * 10 + ["comply"] * 10 + ["comply"] * 20
    raw = build(
        full_scenario(quads, base=base, ref=ref,
                      identity=["refusal"] * 6 + ["comply"] * 14 + ["comply"] * 20),
        quads,
    )
    out = S.analyze_stage2(raw, b=400)
    for blk in out["per_quadrant"].values():
        for arm in blk["arms"].values():
            for key in S._METRIC_KEYS:
                s = arm[key]
                assert s["ci_low"] <= s["point"] + 1e-9
                assert s["point"] <= s["ci_high"] + 1e-9
        for con in blk["contrasts"].values():
            for s in con.values():
                assert s["ci_low"] <= s["point"] + 1e-9
                assert s["point"] <= s["ci_high"] + 1e-9


def test_missing_baseline_or_reference_raises():
    quads = one_quadrant("C", 6)
    raw = build(full_scenario(quads, base=["comply"] * 6, ref=["refusal"] * 6), quads)
    del raw[S.BASELINE]
    with pytest.raises(RuntimeError, match="baseline_target"):
        S.analyze_stage2(raw, b=50)


def test_no_arms_at_the_requested_coef_raises():
    quads = one_quadrant("C", 6)
    raw = build(
        full_scenario(quads, base=["comply"] * 6, ref=["refusal"] * 6), quads, coef=1.0
    )
    with pytest.raises(RuntimeError, match="no Stage-2 arms"):
        S.analyze_stage2(raw, coef=2.0, b=50)


def test_row_set_mismatch_between_arms_raises():
    quads = one_quadrant("C", 8)
    raw = build(
        full_scenario(quads, base=["comply"] * 8, ref=["refusal"] * 8,
                      identity=["refusal"] * 8),
        quads,
    )
    raw["xfer_delta_source_identity@1"] = raw["xfer_delta_source_identity@1"][:-1]
    with pytest.raises(RuntimeError, match="identical record_id set"):
        S.analyze_stage2(raw, b=50)


def test_optional_own_delta_arm_is_included_only_when_present():
    quads = one_quadrant("C", 10)
    base = ["comply"] * 10
    ref = ["refusal"] * 10
    scen = full_scenario(quads, base=base, ref=ref, identity=["refusal"] * 10)

    without = S.analyze_stage2(build(scen, quads), b=100)
    assert S.OWN not in without["arms_present"]
    assert "xfer_delta_source_identity__minus__own_delta_target" not in without["contrasts"]

    scen[S.OWN] = ["refusal"] * 10
    with_own = S.analyze_stage2(build(scen, quads), b=100)
    assert S.OWN in with_own["arms_present"]
    assert "xfer_delta_source_identity__minus__own_delta_target" in with_own["contrasts"]
    assert S.OWN in with_own["per_quadrant"]["C"]["arms"]


def test_arms_present_lists_only_arms_actually_supplied():
    quads = one_quadrant("C", 8)
    raw = build(
        {
            S.BASELINE: ["comply"] * 8,
            S.REFERENCE: ["refusal"] * 8,
            S.IDENTITY: ["refusal"] * 8,
            S.SHUF_WQ: ["comply"] * 8,
        },
        quads,
    )
    out = S.analyze_stage2(raw, b=50)
    assert out["arms_present"] == [S.IDENTITY, S.SHUF_WQ]
    # only contrasts whose both members are present survive
    assert out["contrasts"] == [
        "xfer_delta_source_identity__minus__xfer_delta_source_shuf_wq"
    ]


def test_degeneracy_rate_is_reported_per_arm():
    quads = one_quadrant("C", 10)
    raw = build(
        full_scenario(quads, base=["comply"] * 10, ref=["refusal"] * 10,
                      identity=["degenerate"] * 3 + ["comply"] * 7),
        quads,
    )
    out = S.analyze_stage2(raw, b=50)
    assert out["per_quadrant"]["C"]["arms"][S.IDENTITY]["degeneracy_rate"] == pytest.approx(0.3)


def test_notes_and_bootstrap_metadata_are_present():
    quads = one_quadrant("C", 6)
    raw = build(
        full_scenario(quads, base=["comply"] * 6, ref=["refusal"] * 6,
                      identity=["refusal"] * 6),
        quads,
    )
    out = S.analyze_stage2(raw, b=50, seed=20260904)
    assert out["bootstrap"] == {
        "b": 50, "seed": 20260904, "interval": "percentile", "paired": True
    }
    assert any("shuf_wq" in n for n in out["notes"])
    assert out["stage1_gate_quadrant"] == "C"
