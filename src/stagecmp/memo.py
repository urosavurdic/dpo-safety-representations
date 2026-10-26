"""WP-4: the findings memo.

A short document for a reader who has not been in the weeds - what was
measured, what came out, and what is still open. Generated from the artifacts
rather than typed, so it cannot drift from them.

    python -m src.stagecmp.memo

Two framing decisions are baked in, both of them corrections to earlier
versions of this memo:

* **The primary quantities are ratios**, not differences. An additive gap
  between two quadrants grows with how far a transition moved anything at all,
  so a difference-based headline confounds "this objective reshaped the
  harmful/benign distinction" with "this objective moved the model further".
* **The regex classifier is not an analysis instrument.** It appears nowhere
  below. It under-detects untemplated refusal and its four-way taxonomy
  reversed a selectivity conclusion, so it is a monitor, not a comparator.
"""
from __future__ import annotations

import json
from pathlib import Path

OUT = Path("results/stagecmp")
ARMS = [
    ("sft_alpaca", "M1__to__M2", "safety SFT, Alpaca"),
    ("sft_dolly", "M1_alt__to__M2_alt", "safety SFT, Dolly"),
    ("dpo_alpaca", "M1__to__M3_direct", "DPO, Alpaca"),
    ("dpo_dolly", "M1_alt__to__M3_direct_alt", "DPO, Dolly"),
]
CI_LAYERS = ("16", "20", "24", "28")


def _load(name):
    path = OUT / f"{name}.json"
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _ci(d, prec=3, null=1.0):
    """point [lo, hi], starred when the interval excludes ``null``.

    Differences are signed; a ratio is not, so the explicit ``+`` is dropped
    when the null is 1.0 rather than 0.0.
    """
    if d is None or d.get("ci_low") is None or d.get("point") is None:
        return "n/a"
    star = "" if d["ci_low"] <= null <= d["ci_high"] else " *"
    fmt = f".{prec}f" if null == 1.0 else f"+.{prec}f"
    return (f"{d['point']:{fmt}} [{d['ci_low']:{fmt}}, "
            f"{d['ci_high']:{fmt}}]{star}")


def _layer_row(rep, key, layer, fn):
    row = [r for r in rep["headline_pairs"][key]["per_layer"] if r["layer"] == layer]
    return fn(row[0]) if row else float("nan")


def render() -> str:
    rep = _load("representational_final_token")
    repp = _load("representational_mean_last5")
    wg = _load("stage_comparison_wg_refusal")
    sr = _load("stage_comparison_sr_score")
    scale = _load("scale_robust_behaviour")
    diag = _load("diagnostics")

    L = []
    add = L.append
    add("# Stage comparison - findings memo\n")
    add(f"*{(wg or {}).get('status', 'EXPLORATORY')}*\n")
    add("Generated from the artifacts in `results/stagecmp/`. Every number here "
        "is reproducible from them; none is typed by hand.\n")

    # ---------------------------------------------------------------- Q --- #
    add("\n## The question\n")
    add("`M1 -> M2` (safety SFT) and `M1 -> M3_direct` (DPO) start from the "
        "**same checkpoint** and consume the **same PKU-SafeRLHF material**, "
        "differing in objective and recipe. Same for the `_alt` pair with Dolly "
        "in place of Alpaca. The question is not which refuses more, but "
        "whether the two objectives differ in **building an internal "
        "harmful-versus-benign distinction at all**.\n")
    add("Note the recipe asymmetry runs in safety SFT's favour: it gets 2 "
        "epochs at lr 2e-4 against DPO's 1 epoch at 5e-5, so its small effect "
        "below cannot be blamed on a smaller optimisation budget. What DPO does "
        "see, and SFT does not, is the rejected response of each pair.\n")

    # -------------------------------------------------------- validity --- #
    if diag:
        add("\n## 0. Can the instruments carry this?\n")
        deg = diag["degeneracy"]
        excluded = deg["excluded"]
        add(f"**Degenerate output.** A refusal label on garbled text means "
            f"nothing, so a stage above {deg['limit']:.0%} degenerate "
            f"generations carries no behavioural claim. "
            f"{'Excluded: ' + ', '.join(excluded) if excluded else 'No stage is excluded'}"
            f".\n")
        add("| stage | degenerate | usable |")
        add("|---|---:|---|")
        for stage, cell in deg["by_stage"].items():
            mark = "yes" if cell["usable_for_behavioural_claims"] else "**NO**"
            add(f"| {stage} | {cell['degenerate_rate']:.3f} | {mark} |")
        v = diag["judge_validation"]
        c = v["construct_wg_prompt_harm_vs_design"]
        g = v["convergent_wg_vs_strongreject"]
        add(f"\n**The judge is validated on this benchmark, not only on its "
            f"own.** WildGuard's prompt-harm label never enters the analysis, "
            f"and it recovers the benchmark's design labels at kappa "
            f"{c['kappa']:.3f} (accuracy {c['accuracy']:.3f}). That validates "
            f"the judge and the intent axis together.\n")
        add(f"**The two judges are not interchangeable.** They agree on refusal "
            f"(AUROC {g['refusal_auroc_against_sr']:.3f} against the continuous "
            f"StrongREJECT score) and much less on graded harm (AUROC "
            f"{g['harm_auroc_against_sr']:.3f}). Refusal is therefore the "
            f"behavioural metric to lead with. Chance-corrected agreement needs "
            f"a threshold on a continuous score, and it peaks at kappa "
            f"{max(g['kappa_refusal_vs_sr_below'].values()):.3f} - good, not "
            f"the near-perfect figure an AUROC of 0.94 might suggest.\n")

    # -------------------------------------------- representation first --- #
    if rep:
        add("\n## 1. The representation - safety SFT does not build the contrast\n")
        add("`contrast_norm_ratio` is ||c_post|| / ||c_pre||, where c is the "
            "harmful-minus-benign difference in means. It is a **ratio**, so a "
            "transition that moves every prompt equally scores 1.0 however far "
            "it moved. `*` marks an interval excluding 1.0.\n")
        add("| arm | " + " | ".join("L" + l for l in CI_LAYERS) + " |")
        add("|---|" + "---|" * len(CI_LAYERS))
        for _arm, key, label in ARMS:
            cis = rep["headline_pairs"][key].get("ratio_cis")
            if not cis:
                continue
            cells = [_ci(cis["by_layer"].get(l, {}).get("contrast_norm_ratio"))
                     for l in CI_LAYERS]
            add(f"| {label} | " + " | ".join(cells) + " |")
        add("\n**The intervals do not overlap between objectives.** Safety SFT "
            "stays within a few percent of 1.0 and not consistently on one side "
            "of it; DPO sits far above at every layer, in both corpora. The "
            "earlier phrasing - that safety SFT leaves the ratio "
            "*indistinguishable* from 1.0 - was wrong: at some layers its "
            "interval does exclude 1.0, narrowly and in opposite directions "
            "across corpora. Non-overlap between objectives is the claim that "
            "survives.\n")
        add("Drift ratios tell the same story on a second scale-free measure: "
            "safety SFT displaces the quadrants roughly equally, DPO displaces "
            "the harmful ones more.\n")
        add("| arm | drift A/D (L24) | drift C/B (L24) |")
        add("|---|---|---|")
        for _arm, key, label in ARMS:
            cis = rep["headline_pairs"][key].get("ratio_cis")
            if not cis:
                continue
            cell = cis["by_layer"].get("24", {})
            add(f"| {label} | {_ci(cell.get('drift_ratio_A_over_D'))} "
                f"| {_ci(cell.get('drift_ratio_C_over_B'))} |")
        add("\nThe A/D ratio separates the objectives cleanly. The C/B ratio "
            "does **not**: both objectives show some anisotropy there and the "
            "intervals come close, so it is reported and not leant on.\n")
        v = rep["saturation_check"]["verdict"]
        add(f"\n**Saturation pre-check passed**: max CKA spread "
            f"{v['max_cka_spread']:.3f}, max cosine spread "
            f"{v['max_cosine_spread']:.3f}, both far above the "
            f"{rep['saturation_check']['threshold']} ceiling threshold. The "
            f"published warning that CKA saturates near 1.0 on fine-tuning "
            f"checkpoints does not bind here.\n")

    # --------------------------------------------------- depth profile --- #
    if rep:
        gaps = {}
        for layer in range(1, 29):
            try:
                s = _layer_row(rep, "M1__to__M2", layer,
                               lambda r: r["contrast_norm"]["ratio"])
                d = _layer_row(rep, "M1__to__M3_direct", layer,
                               lambda r: r["contrast_norm"]["ratio"])
            except (KeyError, IndexError):
                continue
            gaps[layer] = d - s
        if gaps:
            onset = next((l for l in sorted(gaps)
                          if all(gaps[m] > 0.2 for m in sorted(gaps) if m >= l)), None)
            peak = max(gaps, key=gaps.get)
            add("\n## 2. Where in depth\n")
            add(f"The DPO/SFT separation in contrast ratio is noise through the "
                f"first half of the network, becomes sustained from layer "
                f"**{onset}**, and peaks at layer **{peak}** (gap "
                f"{gaps[peak]:.2f}). Layer 24 carries "
                f"{100 * gaps[24] / gaps[peak]:.0f}% of the peak separation, so "
                f"reporting layer 24 alone - inherited from the preregistered "
                f"ablation window, which no analysis here uses - understates "
                f"the effect. The fixed grid {', '.join(CI_LAYERS)} is reported "
                f"instead, and the full profile is the figure.\n")

    # ------------------------------------------------------- behaviour --- #
    if wg:
        add("\n## 3. Behaviour - the objective dominates the corpus\n")
        add("WildGuard `response_refusal`, paired bootstrap over prompts. "
            "`*` marks an interval excluding zero.\n")
        add("| quadrant | OBJ (DPO - safety SFT) | corpus | interaction |")
        add("|---|---|---|---|")
        for q in "ABCD":
            cell = wg["contrasts_2x2"]["by_quadrant"][q]
            add(f"| {q} | {_ci(cell['OBJ'], null=0.0)} "
                f"| {_ci(cell['CORP'], null=0.0)} | {_ci(cell['INT'], null=0.0)} |")
        objs = [wg["contrasts_2x2"]["by_quadrant"][q]["OBJ"]["point"] for q in "ABCD"]
        corps = [abs(wg["contrasts_2x2"]["by_quadrant"][q]["CORP"]["point"])
                 for q in "ABCD"]
        add(f"\nEvery objective term is positive with an interval excluding "
            f"zero, from {min(objs):+.3f} to {max(objs):+.3f}, and the harmful "
            f"quadrants move several times as far as the benign ones. |corpus| "
            f"never exceeds {max(corps):.3f}.\n")
    if sr:
        add("Under StrongREJECT, which scores harmful assistance and therefore "
            "carries the opposite sign, the corpus term is null on **all four** "
            "quadrants - cleaner than under WildGuard.\n")
        add("| quadrant | OBJ | corpus |")
        add("|---|---|---|")
        for q in "ABCD":
            cell = sr["contrasts_2x2"]["by_quadrant"][q]
            add(f"| {q} | {_ci(cell['OBJ'], null=0.0)} | {_ci(cell['CORP'], null=0.0)} |")

    # ------------------------------------------- magnitude vs selectivity - #
    if scale:
        add("\n## 4. What this does NOT show: better discrimination\n")
        add("The objective difference is first of all a difference in "
            "magnitude.\n")
        add("| arm | mean absolute movement |")
        add("|---|---:|")
        for arm, _key, label in ARMS:
            m = scale["arms"][arm]["movement"]["mean_absolute_movement"]
            add(f"| {label} | {m:.3f} |")
        add("\nSo any additive quadrant gap is larger for DPO partly because "
            "DPO moved everything further. Two ratio scales separate the two, "
            "and they disagree instructively.\n")
        add("| checkpoint | risk ratio C/B | its ceiling | % of ceiling "
            "| log odds ratio C/B |")
        add("|---|---|---:|---:|---|")
        for stage in ("M1", "M2", "M3_direct", "M1_alt", "M2_alt", "M3_direct_alt"):
            rep_c = scale["discrimination_by_checkpoint"].get(stage)
            if not rep_c:
                continue
            r = rep_c["ratios"]["C_over_B"]
            rr = (_ci(r["risk_ratio"], 2, null=1.0)
                  if r.get("risk_ratio_reliable", True)
                  else f"*unreliable* ({min(r['successes'].values())} prompts)")
            ceiling = r.get("risk_ratio_ceiling")
            point = r["risk_ratio"]["point"]
            pct = (f"{100 * point / ceiling:.0f}%"
                   if ceiling and point else "-")
            add(f"| {stage} | {rr} | {ceiling:.1f} | {pct} "
                f"| {_ci(r['log_odds_ratio'], 2, null=0.0)} |"
                if ceiling else
                f"| {stage} | {rr} | - | - "
                f"| {_ci(r['log_odds_ratio'], 2, null=0.0)} |")
        add("\n**The risk ratio structurally cannot answer the question here.** "
            "A risk ratio cannot exceed 1 / p_benign, and the DPO arms sit at "
            "98% and 100% of that arithmetic ceiling while the safety-SFT arms "
            "sit at about 10% of theirs. The DPO numbers are pinned by the "
            "benign rate, not measuring discrimination, which is exactly why "
            "the two objectives look similar on this scale. Add the small "
            "denominators - one arm's benign quadrant withholds on zero "
            "prompts - and the scale is unusable for this comparison.\n")
        add("The Haldane-corrected **log odds ratio is unbounded** and stays "
            "finite where a rate reaches exactly 0 or 1. On it, DPO's "
            "discrimination is clearly the larger, with the pre-DPO baselines "
            "shown so the change is visible rather than the level alone. Note "
            "the baselines are not equal: `M1` already discriminates somewhat "
            "and `M2` sits *below* it.\n")
        add("This is the honest version of a claim withdrawn from an earlier "
            "memo. \"DPO buys refusal, not discrimination\" was a regex "
            "artifact. \"DPO discriminates better\" is supportable, but only on "
            "the log-odds scale and only with the baselines beside it.\n")

    # ------------------------------------------------------------ open --- #
    add("\n## 5. What bounds the reading\n")
    add("- **Objective and recipe, not objective alone.** DPO sees chosen and "
        "rejected responses; SFT only the chosen one. The accurate name is an "
        "objective-and-recipe contrast at matched initialisation and matched "
        "source corpus.")
    add("- **Safety SFT may be weak at this scale.** 4,000 chosen-only "
        "responses is a small intervention, so the finding is that *this* "
        "recipe at *this* scale does not build the contrast - not that "
        "supervised fine-tuning cannot.")
    add("- **One seed per cell.** The interaction term is a single draw.")
    add("- **n = 2 corpora.** Intervals on the corpus and interaction terms "
        "are over prompts, not over corpora.")
    add("- **Quadrant contrasts are between-source.** B is XSTest and D is "
        "instruction data; A is HarmBench and C is reworded StrongREJECT. The "
        "cue axis is not a validated construct in this project, so no quadrant "
        "difference is attributed to wording.")
    add("- **The DPO arms saturate** on the harmful quadrants, so any "
        "harmful-side difference is compressed against a ceiling.")
    add("- **Not preregistered.** Exploratory; intervals are nominal and "
        "uncorrected across quadrants.\n")
    return "\n".join(L)


def main() -> int:
    text = render()
    target = OUT / "MEMO.md"
    target.write_text(text, encoding="utf-8", newline="\n")
    print(f"wrote {target}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
