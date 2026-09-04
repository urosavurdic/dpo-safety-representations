"""Cross-branch transfer of a DPO-induced activation delta.

A self-contained research extension, deliberately OUTSIDE the frozen
pre-registration in docs/audit/analysis_plan.md. Nothing in this package
edits analysis_plan.md, v2_pipeline.py, v2_shards.py, the CF/H modules, or
any existing test; it only imports their free functions.

Scientific question: is the activation change DPO induces in one branch
(Alpaca) causally reusable on the pre-DPO checkpoint of the other branch
(Dolly), at the delta's own measured location?

Notation (see plan section 2):
    A2 = M2      Alpaca pre-DPO      A3 = M3      Alpaca post-DPO
    B2 = M2_alt  Dolly  pre-DPO      B3 = M3_alt  Dolly  post-DPO
    Delta_A(x) = h_A3(x, L24, t_final) - h_A2(x, L24, t_final)
    Delta_B(x) = h_B3(x, L24, t_final) - h_B2(x, L24, t_final)

"Pre-DPO checkpoint" is the deliberate wording. The repo trains with
trl.DPOTrainer(ref_model=None, peft_config=...), whose docstring claims an
adapter-disable trick, but requirements.txt pins only trl>=0.9 and the
lockfile excludes trl, so the training-time reference behaviour is not
confirmed. Do not write "reference-free DPO" anywhere.
"""
