"""Stage-vs-stage comparison across the nine trained checkpoints.

A self-contained study that sits OUTSIDE the frozen preregistration in
``docs/audit/analysis_plan.md``, in the same way ``src/crossbranch/`` does. Its
numbers are exploratory and must not be folded into the confirmatory endpoints
in ``results/summaries/confirmatory_endpoints.json``.

What it answers, which nothing else in the repo does:

* how far apart any two of the nine checkpoints are, at the behavioural and the
  representational level, on the identical 654-prompt benchmark;
* the safety-SFT versus DPO contrast at matched initialisation and matched
  source corpus (``M1->M2`` against ``M1->M3_direct``, and the ``_alt`` pair),
  crossed with the instruction corpus;
* where in depth each training stage writes the A-D contrast.

Every artifact this package writes carries ``STATUS_EXPLORATORY``.
"""
from __future__ import annotations

STATUS_EXPLORATORY = (
    "EXPLORATORY - not preregistered (analysis_plan.md frozen: "
    "CF1/CF2 confirmatory, CF3 predeclared secondary)"
)

__all__ = ["STATUS_EXPLORATORY"]
