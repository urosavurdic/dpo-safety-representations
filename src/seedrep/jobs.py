"""The job table: every unit of GPU work, its dependencies and its GPU.

Run ids are ``{recipe}_{branch}_s{seed}``: recipe in ``sft`` / ``dpo`` /
``sft_aol``, branch in ``alpaca`` (start = M1) / ``dolly`` (start = M1_alt).
Seeds 1 and 2 join the original seed-42 runs, for three per arm.

Priorities, in launch order:

* P1 -- branch-1 seeds: train, measure dose checkpoints and endpoint,
  held-out diagnostics, generate, judge; plus WildGuard on 500 SFT targets.
* P2 -- injection test on the start models (C prompts, layer 24): DPO's push
  along the new orthogonal direction, the same size of push along the old
  direction, DPO's whole mean shift, a norm-matched random vector, none.
* P3 -- branch-2 seeds, same pipeline.
* P4 -- SFT with the loss on response tokens only (seed 1, branch 1).

Each GPU gets a fixed partition, in this order. With one GPU the two
partitions are interleaved into one queue.
"""
from __future__ import annotations

from dataclasses import dataclass, field

SEEDS = (1, 2)
SFT_DOSE = (10, 25, 50, 75, 100, 150, 200)
DPO_DOSE = (5, 10, 20, 30, 45, 60, 80, 100)
RESUME_EVERY = {"sft": 50, "sft_aol": 50, "dpo": 25}
INJECT_CONDITIONS = ("baseline", "orth", "old_dir", "mean_shift", "random")
START = {"alpaca": "M1", "dolly": "M1_alt"}


@dataclass
class Job:
    id: str
    kind: str
    gpu: int
    priority: str
    deps: tuple = ()
    params: dict = field(default_factory=dict)
    minutes: float = 5.0


def run_id(recipe: str, branch: str, seed: int) -> str:
    return f"{recipe}_{branch}_s{seed}"


def _run_jobs(recipe, branch, seed, gpu, priority, dose, ref_job) -> list[Job]:
    rid = run_id(recipe, branch, seed)
    after = tuple(d for d in (f"train__{rid}", ref_job) if d)
    train = Job(f"train__{rid}", "train", gpu, priority, (),
                {"run_id": rid, "recipe": recipe, "branch": branch, "seed": seed,
                 "dose_steps": list(dose)}, 45)
    jobs = [train]
    for step in dose:
        jobs.append(Job(f"measure__{rid}__step{step:04d}", "measure", gpu, priority,
                        after, {"run_id": rid, "branch": branch, "step": step}, 2))
    jobs.append(Job(f"measure__{rid}__final", "measure", gpu, priority, after,
                    {"run_id": rid, "branch": branch, "step": "final"}, 3))
    jobs.append(Job(f"diag__{rid}", "diag", gpu, priority, after,
                    {"run_id": rid, "branch": branch}, 4))
    gen = Job(f"gen__{rid}", "gen", gpu, priority, (train.id,),
              {"run_id": rid, "branch": branch}, 25)
    jobs.append(gen)
    jobs.append(Job(f"judge__{rid}", "judge", gpu, priority, (gen.id,),
                    {"unit": rid, "source": f"gen/{rid}/responses.json"}, 25))
    return jobs


ORIGINAL_ARMS = {"alpaca": ("M2", "M3_direct"), "dolly": ("M2_alt", "M3_direct_alt")}


def _orig_jobs(branch, gpu, priority, ref_job) -> list[Job]:
    """The seed-42 arms measured in the same environment as the new seeds."""
    return [Job(f"orig__{stage}", "orig", gpu, priority, (ref_job,),
                {"stage": stage, "branch": branch}, 6) for stage in ORIGINAL_ARMS[branch]]


def build_jobs(priorities=("P1", "P2", "P3", "P4"), sft_dose=SFT_DOSE, dpo_dose=DPO_DOSE,
               seeds=SEEDS, inject_conditions=INJECT_CONDITIONS) -> list[Job]:
    gpu0: list[Job] = []
    gpu1: list[Job] = []
    if "P1" in priorities:
        ref = Job("ref__alpaca", "ref", 0, "P1", (), {"branch": "alpaca"}, 6)
        gpu0.append(ref)
        gpu0 += _orig_jobs("alpaca", 0, "P1", ref.id)
        for seed in seeds:
            gpu0 += _run_jobs("sft", "alpaca", seed, 0, "P1", sft_dose, ref.id)
            gpu1 += _run_jobs("dpo", "alpaca", seed, 1, "P1", dpo_dose, ref.id)
        gpu1.append(Job("gen__pku_chosen", "chosen_rows", 1, "P1", (), {}, 1))
        gpu1.append(Job("judge__pku_chosen", "judge", 1, "P1", ("gen__pku_chosen",),
                        {"unit": "pku_chosen", "source": "gen/pku_chosen/responses.json"}, 15))
    if "P2" in priorities:
        for branch, queue, gpu in (("alpaca", gpu0, 0), ("dolly", gpu1, 1)):
            for cond in inject_conditions:
                unit = f"inject_{branch}_{cond}"
                gen = Job(f"gen__{unit}", "inject", gpu, "P2", (),
                          {"branch": branch, "condition": cond, "unit": unit}, 5)
                queue.append(gen)
                queue.append(Job(f"judge__{unit}", "judge", gpu, "P2", (gen.id,),
                                 {"unit": unit, "source": f"gen/{unit}/responses.json"}, 4))
    if "P3" in priorities:
        ref = Job("ref__dolly", "ref", 0, "P3", (), {"branch": "dolly"}, 6)
        gpu0.append(ref)
        gpu0 += _orig_jobs("dolly", 0, "P3", ref.id)
        for seed in seeds:
            gpu0 += _run_jobs("sft", "dolly", seed, 0, "P3", sft_dose, ref.id)
            gpu1 += _run_jobs("dpo", "dolly", seed, 1, "P3", dpo_dose, ref.id)
    if "P4" in priorities:
        gpu1 += _run_jobs("sft_aol", "alpaca", 1, 1, "P4", sft_dose,
                          "ref__alpaca" if "P1" in priorities else None)
    return gpu0 + gpu1


def partition(jobs: list[Job], gpu: int | None, n_gpus: int) -> list[Job]:
    """This worker's queue. With one GPU, interleave both partitions by
    priority so that neither recipe waits for the other to finish."""
    if n_gpus >= 2 and gpu is not None:
        return [j for j in jobs if j.gpu == gpu]
    by_gpu = [[j for j in jobs if j.gpu == g] for g in (0, 1)]
    merged = []
    for prio in ("P1", "P2", "P3", "P4"):
        a = [j for j in by_gpu[0] if j.priority == prio]
        b = [j for j in by_gpu[1] if j.priority == prio]
        while a or b:
            if a:
                merged.append(a.pop(0))
            if b:
                merged.append(b.pop(0))
    return merged
