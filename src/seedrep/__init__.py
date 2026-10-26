"""Seed replication and dose curves for the matched-start SFT-vs-DPO comparison.

Retrains the two arms that start from M1 (safety SFT and direct DPO) under new
seeds, saves checkpoints along the way, and measures each one with the same
extraction, generation and judging conventions as the original nine stages.

Everything is organised as resumable jobs whose outputs go to a *bank* (a
private Hugging Face dataset repo, or a local folder), so a session can stop
at any point and continue on another machine. See ``jobs.py`` for the job
table and ``worker.py`` for the runner.
"""

STATUS_EXPLORATORY = (
    "EXPLORATORY - not preregistered (analysis_plan.md frozen: CF1/CF2 "
    "confirmatory, CF3 predeclared secondary)"
)
