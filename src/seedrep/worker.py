"""Run one GPU's queue of jobs against the bank, resuming wherever it stopped.

    python -m src.seedrep.worker --bank hf://USER/seedrep-bank --gpu 0 --n-gpus 2 \\
        --priorities P1 P2 --deadline-minutes 690

The loop scans the queue in order and runs the first job that is not done and
whose dependencies are done. A job's dependencies may belong to the other
GPU's worker, in which case this worker waits for them. A job is done only
when its marker is in the bank (``Bank.mark_done``), written after its
outputs. A job that fails is logged and skipped for the rest of the session,
not retried in a loop. The worker stops cleanly before ``--deadline-minutes``.

Model handling: dose checkpoints and endpoints are measured with the adapter
attached, *unmerged*, to one cached merged start model. That is exact and
leaves the cached weights untouched, and it saves about 40 s per checkpoint
over reloading. Generation loads a fresh merged model, like the originals.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform as _platform
import time
import traceback
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from src.pipeline.shards import Deadline, DeadlineReached
from src.seedrep import STATUS_EXPLORATORY, diagnostics, extract, models, rows
from src.seedrep import generate as gen_mod
from src.seedrep import judge as judge_mod
from src.seedrep import platform as plat
from src.seedrep.jobs import START, Job, build_jobs, partition
from src.seedrep.train import InjectedFault, recipe_config, train

INPUTS = rows.ROOT / "data" / "seedrep"
PARITY_FIXTURE = INPUTS / "parity_fixture.npz"
INJECT_VECTORS = INPUTS / "inject_vectors.npz"
PARITY_MIN_COSINE = 0.999
CHOSEN_SAMPLE = 500
INJECT_LAYER = 24


@dataclass
class Settings:
    base_name: str = models.BASE_MODEL
    start_chains: dict = field(default_factory=lambda: {
        b: models.chain_for(stage=s) for b, s in START.items()})
    #: adapters whose unmerged-path parity is checked by the ref job
    parity_adapters: dict = field(default_factory=lambda: {
        "alpaca": ("M2", "urosavurdic/qwen2.5-1.5b-m2-safety"),
        "dolly": ("M2_alt", "urosavurdic/qwen2.5-1.5b-m2-alt-safety")})
    layers: tuple = extract.ANALYSIS_LAYERS
    n_pairs: int = 400
    bench_limit: int = 0
    train_overrides: dict = field(default_factory=dict)
    judge_scorer: object = None
    load_4bit: bool = True
    batch: int = 8
    inject_vectors: Path = INJECT_VECTORS
    parity_fixture: Path = PARITY_FIXTURE
    inject_layer: int = INJECT_LAYER
    #: smoke tests: an explicit row list instead of the 654 benchmark rows
    rows_override: list | None = None
    #: original arms' own adapter (the last link of each stage's chain)
    orig_adapters: dict = field(default_factory=lambda: {
        s: models.chain_for(stage=s)[-1] for s in ("M2", "M3_direct", "M2_alt", "M3_direct_alt")})


def env_info() -> dict:
    info = {"python": _platform.python_version(), "platform": plat.detect()}
    for name in ("torch", "transformers", "trl", "peft", "bitsandbytes"):
        try:
            info[name] = __import__(name).__version__
        except Exception:
            info[name] = None
    try:
        import torch

        info["gpu"] = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"
    except Exception:
        info["gpu"] = None
    try:
        info["git_commit"] = (rows.ROOT / ".git" / "HEAD").read_text().strip()
        ref = info["git_commit"].split(" ")[-1]
        if ref.startswith("refs/"):
            info["git_commit"] = (rows.ROOT / ".git" / ref).read_text().strip()
    except Exception:
        pass
    return info


class Context:
    def __init__(self, bank, settings: Settings, deadline: Deadline, log=print):
        self.bank, self.s, self.deadline, self.log = bank, settings, deadline, log
        self.paths = plat.paths()
        bench = rows.benchmark_rows()
        if settings.rows_override is not None:
            bench = settings.rows_override
        self.rows = bench[:settings.bench_limit] if settings.bench_limit else bench
        self.shas = rows.benchmark_shas()
        self.device = models.pick_device()
        self._tok = None
        self._start = {}
        self._judges = None

    @property
    def tok(self):
        if self._tok is None:
            self._tok = models.load_tokenizer(self.s.base_name)
        return self._tok

    def start_model(self, branch: str):
        if branch not in self._start:
            for other in list(self._start):
                models.free(self._start.pop(other))
            self._start[branch] = models.load_merged(self.s.start_chains[branch],
                                                     base_name=self.s.base_name, device=self.device)
        return self._start[branch]

    def drop_start(self):
        for b in list(self._start):
            models.free(self._start.pop(b))

    def release_gpu(self):
        """Unload everything this worker keeps on the GPU between jobs."""
        import gc

        self.drop_start()
        if self._judges is not None:
            self._judges.unload()
            self._judges = None
        gc.collect()
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass

    @property
    def judges(self):
        if self._judges is None:
            self._judges = judge_mod.Judges(load_4bit=self.s.load_4bit, scorer=self.s.judge_scorer)
        return self._judges

    def local(self, *parts) -> Path:
        path = self.paths.work.joinpath(*parts)
        path.parent.mkdir(parents=True, exist_ok=True)
        return path

    def fetch(self, remote: str) -> Path:
        local = self.local(*remote.split("/"))
        if not local.exists() and not self.bank.get(remote, local):
            raise FileNotFoundError(f"not in bank: {remote}")
        return local

    def _adapter_copies(self, remote: str) -> list[Path]:
        """Where an ``adapters/<run>/<name>`` directory may sit on this disk: the
        trainer's own output (same machine) first, then a fetched copy."""
        _, run_id, name = remote.split("/")
        own = self.paths.scratch / run_id / ("final" if name == "final" else f"dose/{name}")
        return [own, self.paths.scratch / "fetched" / remote]

    def fetch_dir(self, remote: str) -> Path:
        for path in self._adapter_copies(remote):
            if (path / "adapter_model.safetensors").exists():
                return path
        local = self._adapter_copies(remote)[-1]
        if not self.bank.get_dir(remote, local):
            raise FileNotFoundError(f"not in bank: {remote}/")
        return local

    def discard_dir(self, remote: str) -> None:
        """Free local disk once an adapter is banked and no later job needs it."""
        import shutil

        for path in self._adapter_copies(remote):
            shutil.rmtree(path, ignore_errors=True)


def _save_npz(path: Path, **arrays) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp.npz")
    np.savez(tmp, **arrays)
    tmp.replace(path)
    return path


def _sha(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


# --------------------------------------------------------------- job kinds --

def do_ref(job: Job, ctx: Context) -> dict:
    """Start checkpoint: activations, first-token log-probs, held-out scores,
    and parity of both measurement paths against the stored arrays."""
    branch = job.params["branch"]
    stage = START[branch]
    model = ctx.start_model(branch)
    res = extract.extract(model, ctx.tok, ctx.rows, ctx.device, batch_size=ctx.s.batch,
                          layers=None, keep_logprobs=True)
    acts = _save_npz(ctx.local("ref", f"{branch}_acts.npz"),
                     final=res["final"].astype(np.float16), pooled=res["pooled"].astype(np.float16))
    lp = ctx.local("ref", f"{branch}_first_token_logprobs.npy")
    np.save(lp, res["logprobs"])
    scores = diagnostics.pair_scores(model, ctx.tok, rows.heldout_pairs(n=ctx.s.n_pairs), ctx.device)
    diag_path = ctx.local("diag", f"{stage}.json")
    diag_path.write_text(json.dumps({"pair_scores": scores,
                                     "heldout": diagnostics.summarise(scores)}), encoding="utf-8")

    parity = {}
    if ctx.s.parity_fixture.exists():
        fixture = np.load(ctx.s.parity_fixture)
        n = fixture[f"{stage}_final"].shape[0]
        mine = res["final"][:n][:, list(ctx.s.layers)]
        parity[stage] = _parity(mine, fixture[f"{stage}_final"])
        sft_stage, adapter = ctx.s.parity_adapters[branch]
        if f"{sft_stage}_final" in fixture:
            from peft import PeftModel

            peft = PeftModel.from_pretrained(model, adapter)
            f, _, _ = extract.forward_batch(peft, ctx.tok, [r["prompt"] for r in ctx.rows[:n]],
                                            ctx.device, layers=list(ctx.s.layers), want_logprobs=False)
            ctx._start[branch] = peft.unload()
            parity[f"{sft_stage}_unmerged"] = _parity(f, fixture[f"{sft_stage}_final"])
        bad = {k: v for k, v in parity.items() if v["min_cosine"] < PARITY_MIN_COSINE}
        if bad:
            raise RuntimeError(f"parity failed against the stored activations: {bad}")
    ctx.bank.put({f"ref/{branch}_acts.npz": acts, f"ref/{branch}_first_token_logprobs.npy": lp,
                  f"diag/{stage}.json": diag_path}, f"ref {branch}")
    return {"parity": parity, "heldout": json.loads(diag_path.read_text())["heldout"]}


def _parity(mine: np.ndarray, stored: np.ndarray) -> dict:
    mine, stored = mine.astype(np.float64), stored.astype(np.float64)
    cos = [(a.ravel() @ b.ravel()) / (np.linalg.norm(a) * np.linalg.norm(b))
           for a, b in zip(mine.reshape(len(mine), -1), stored.reshape(len(stored), -1))]
    return {"n": int(len(mine)), "min_cosine": float(min(cos)),
            "max_rel_abs_diff": float(np.abs(mine - stored).max() / np.abs(stored).max())}


def do_train(job: Job, ctx: Context) -> dict:
    p = job.params
    cfg = recipe_config(p["recipe"], p["branch"], p["seed"], ctx.s.train_overrides)
    # training needs the whole GPU: the judges (~7 GB in 4-bit) and the cached
    # start model are unloaded first; both reload on demand afterwards
    ctx.release_gpu()
    out = train(p["run_id"], cfg, ctx.bank, ctx.paths.scratch,
                dose_steps=p["dose_steps"], resume_every=_resume_every(p["recipe"], ctx),
                deadline=ctx.deadline, log=ctx.log)
    if out["status"] != "done":
        raise DeadlineReached(f"{p['run_id']}: training paused at step {out['step']}")
    return out


def _resume_every(recipe: str, ctx: Context) -> int:
    from src.seedrep.jobs import RESUME_EVERY

    return int(ctx.s.train_overrides.get("_resume_every", RESUME_EVERY[recipe]))


def do_measure(job: Job, ctx: Context) -> dict:
    from peft import PeftModel

    p = job.params
    final = p["step"] == "final"
    name = "final" if final else f"step_{int(p['step']):04d}"
    adapter = ctx.fetch_dir(f"adapters/{p['run_id']}/{name}")
    ref = np.load(ctx.fetch(f"ref/{p['branch']}_first_token_logprobs.npy"))
    start = ctx.start_model(p["branch"])
    peft = PeftModel.from_pretrained(start, str(adapter))
    try:
        res = extract.extract(peft, ctx.tok, ctx.rows, ctx.device, batch_size=ctx.s.batch,
                              layers=None if final else list(ctx.s.layers), ref_logprobs=ref)
    finally:
        ctx._start[p["branch"]] = peft.unload()
    norms = extract.update_norms(extract.adapter_update(adapter))
    npz = _save_npz(ctx.local("measure", p["run_id"], f"{name}.npz"),
                    final=res["final"].astype(np.float16), pooled=res["pooled"].astype(np.float16),
                    kl=res["kl"], entropy=res["entropy"], layers=np.array(res["layers"]))
    meta = {"run_id": p["run_id"], "step": p["step"], "weight_change": norms,
            "first_token_kl_mean": float(np.nanmean(res["kl"])), "layers": res["layers"]}
    meta_path = ctx.local("measure", p["run_id"], f"{name}.json")
    meta_path.write_text(json.dumps(meta), encoding="utf-8")
    ctx.bank.put({f"measure/{p['run_id']}/{name}.npz": npz,
                  f"measure/{p['run_id']}/{name}.json": meta_path},
                 f"measure {p['run_id']} {name}")
    if not final:                      # banked; nothing else reads a dose adapter
        ctx.discard_dir(f"adapters/{p['run_id']}/{name}")
    return {"sha256": {"npz": _sha(npz)}, **meta}


def do_diag(job: Job, ctx: Context) -> dict:
    from peft import PeftModel

    p = job.params
    stage = START[p["branch"]]
    ref = json.loads(ctx.fetch(f"diag/{stage}.json").read_text())["pair_scores"]
    adapter = ctx.fetch_dir(f"adapters/{p['run_id']}/final")
    start = ctx.start_model(p["branch"])
    peft = PeftModel.from_pretrained(start, str(adapter))
    try:
        scores = diagnostics.pair_scores(peft, ctx.tok, rows.heldout_pairs(n=ctx.s.n_pairs), ctx.device)
    finally:
        ctx._start[p["branch"]] = peft.unload()
    out = {"pair_scores": scores, "heldout": diagnostics.summarise(scores, ref)}
    path = ctx.local("diag", f"{p['run_id']}.json")
    path.write_text(json.dumps(out), encoding="utf-8")
    ctx.bank.put({f"diag/{p['run_id']}.json": path}, f"diag {p['run_id']}")
    return out["heldout"]


def do_orig(job: Job, ctx: Context) -> dict:
    """An original (seed-42) arm, measured like a seed endpoint: activations at
    every layer, first-token KL and held-out diagnostics, in this environment."""
    from peft import PeftModel

    p = job.params
    adapter = ctx.s.orig_adapters[p["stage"]]
    start_stage = START[p["branch"]]
    ref = np.load(ctx.fetch(f"ref/{p['branch']}_first_token_logprobs.npy"))
    ref_pairs = json.loads(ctx.fetch(f"diag/{start_stage}.json").read_text())["pair_scores"]
    start = ctx.start_model(p["branch"])
    peft = PeftModel.from_pretrained(start, adapter)
    try:
        res = extract.extract(peft, ctx.tok, ctx.rows, ctx.device, batch_size=ctx.s.batch,
                              layers=None, ref_logprobs=ref)
        scores = diagnostics.pair_scores(peft, ctx.tok, rows.heldout_pairs(n=ctx.s.n_pairs), ctx.device)
    finally:
        ctx._start[p["branch"]] = peft.unload()
    npz = _save_npz(ctx.local("measure", f"orig_{p['stage']}", "final.npz"),
                    final=res["final"].astype(np.float16), pooled=res["pooled"].astype(np.float16),
                    kl=res["kl"], entropy=res["entropy"], layers=np.array(res["layers"]))
    out = {"stage": p["stage"], "first_token_kl_mean": float(np.nanmean(res["kl"])),
           "heldout": diagnostics.summarise(scores, ref_pairs)}
    meta = ctx.local("diag", f"orig_{p['stage']}.json")
    meta.write_text(json.dumps({**out, "pair_scores": scores}), encoding="utf-8")
    ctx.bank.put({f"measure/orig_{p['stage']}/final.npz": npz, f"diag/orig_{p['stage']}.json": meta},
                 f"orig {p['stage']}")
    return out


def do_gen(job: Job, ctx: Context) -> dict:
    p = job.params
    adapter = ctx.fetch_dir(f"adapters/{p['run_id']}/final")
    ctx.drop_start()
    model = models.load_merged(ctx.s.start_chains[p["branch"]] + [str(adapter)],
                               base_name=ctx.s.base_name, device=ctx.device)
    try:
        out = gen_mod.generate(model, ctx.tok, ctx.rows, ctx.device, unit=p["run_id"],
                               label=p["run_id"], condition=f"{p['run_id']}_behavior",
                               shas=ctx.shas, bank=ctx.bank, work=ctx.paths.work,
                               scratch=ctx.paths.scratch, batch=ctx.s.batch,
                               deadline=ctx.deadline, log=ctx.log)
    finally:
        models.free(model)
    # the endpoint was measured and diagnosed before generation in the queue;
    # if a later session needs it again, fetch_dir pulls it from the bank
    ctx.discard_dir(f"adapters/{p['run_id']}/final")
    return {"n_rows": len(out)}


def do_chosen_rows(job: Job, ctx: Context) -> dict:
    """The SFT targets as response rows, a deterministic sample, for WildGuard."""
    data = rows.read_jsonl(rows.SFT_DATA)
    keyed = sorted(data, key=lambda r: hashlib.sha256(
        (r["prompt"] + "\x1f" + r["response"]).encode("utf-8")).hexdigest())
    picked = keyed[:CHOSEN_SAMPLE]
    out = [{"record_id": hashlib.sha256((r["prompt"] + r["response"]).encode()).hexdigest()[:16],
            "prompt": r["prompt"], "response": r["response"], "quadrant": "PKU",
            "stage": "pku_chosen", "condition": "pku_chosen", "model_stage": "pku_chosen",
            "benchmark_sha256": ctx.shas[0], "split_manifest_sha256": ctx.shas[1]}
           for r in picked]
    path = ctx.local("gen", "pku_chosen", "responses.json")
    path.write_text(json.dumps(out, ensure_ascii=False), encoding="utf-8")
    ctx.bank.put({"gen/pku_chosen/responses.json": path}, "pku chosen sample")
    return {"n_rows": len(out)}


def do_judge(job: Job, ctx: Context) -> dict:
    p = job.params
    rows_ = json.loads(ctx.fetch(p["source"]).read_text(encoding="utf-8"))
    ctx.drop_start()
    out = judge_mod.judge(rows_, unit=p["unit"], judges=ctx.judges, bank=ctx.bank,
                          work=ctx.paths.work, scratch=ctx.paths.scratch,
                          deadline=ctx.deadline, log=ctx.log)
    refused = [r["wildguard"]["response_refusal"] for r in out
               if r["wildguard"].get("response_refusal") is not None]
    return {"n_records": len(out), "wildguard_refusal_rate": float(np.mean(refused)) if refused else None}


def do_inject(job: Job, ctx: Context) -> dict:
    from src.crossbranch.inject import PerRowDeltaInjector

    p = job.params
    c_rows = [r for r in ctx.rows if r["quadrant"] == "C"]
    model = ctx.start_model(p["branch"])
    injector = None
    meta = {"condition": p["condition"], "layer": ctx.s.inject_layer}
    if p["condition"] != "baseline":
        vectors = np.load(ctx.s.inject_vectors)
        vec = vectors[f"{p['branch']}_{p['condition']}"].astype(np.float32)
        meta["norm"] = float(np.linalg.norm(vec))
        injector = PerRowDeltaInjector({r["record_id"]: vec for r in c_rows}, coef=1.0,
                                       layer=ctx.s.inject_layer).register(model)
    try:
        out = gen_mod.generate(model, ctx.tok, c_rows, ctx.device, unit=p["unit"],
                               label=START[p["branch"]], condition=p["unit"], shas=ctx.shas,
                               bank=ctx.bank, work=ctx.paths.work, scratch=ctx.paths.scratch,
                               batch=ctx.s.batch, injector=injector, deadline=ctx.deadline,
                               log=ctx.log)
    finally:
        if injector is not None:
            injector.remove()
    return {"n_rows": len(out), **meta}


HANDLERS = {"ref": do_ref, "orig": do_orig, "train": do_train, "measure": do_measure, "diag": do_diag,
            "gen": do_gen, "chosen_rows": do_chosen_rows, "judge": do_judge, "inject": do_inject}


# ------------------------------------------------------------------- loop --

#: short jobs whose outputs and done marker go up as one commit (bank.deferred)
ONE_COMMIT_KINDS = {"ref", "orig", "measure", "diag", "chosen_rows"}


def _blocked(todo: list[Job], failed: dict) -> set[str]:
    """Jobs that cannot run this session because something they need failed."""
    dead = set(failed)
    grew = True
    while grew:
        grew = False
        for j in todo:
            if j.id not in dead and any(d in dead for d in j.deps):
                dead.add(j.id)
                grew = True
    return dead


def run(bank, queue: list[Job], settings: Settings, *, deadline_minutes: float | None,
        wait_seconds: float = 60.0, max_wait_minutes: float = 40.0, log=print) -> dict:
    import contextlib

    deadline = Deadline(deadline_minutes)
    ctx = Context(bank, settings, deadline, log)
    all_ids = {j.id for j in queue}
    failed, ran = {}, []
    waited = 0.0
    while True:
        bank.refresh()                 # see the other worker's markers
        done = {j.id for j in queue if bank.is_done(j.id)}
        todo = [j for j in queue if j.id not in done]
        dead = _blocked(todo, failed)
        todo = [j for j in todo if j.id not in dead]
        if not todo:
            if dead:
                log(f"worker: stopping; {len(dead)} job(s) failed or blocked by a failure "
                    "(they rerun next session)")
            break
        runnable = [j for j in todo if all(bank.is_done(d) for d in j.deps)]
        if not runnable:
            pending = sorted({d for j in todo for d in j.deps
                              if d not in all_ids and not bank.is_done(d)})
            if not pending:
                log(f"worker: nothing runnable; {len(todo)} left")
                break
            if waited >= max_wait_minutes * 60 or deadline.would_exceed(wait_seconds):
                log(f"worker: stopped waiting for {pending[:3]} after {waited / 60:.0f} min")
                break
            log(f"worker: waiting for {pending[:3]} from the other worker")
            time.sleep(wait_seconds)
            waited += wait_seconds
            continue
        waited = 0.0
        job = runnable[0]
        if deadline.would_exceed(job.minutes * 60):
            log(f"worker: {job.id} (~{job.minutes:.0f} min) would pass the deadline; stopping")
            break
        log(f"worker: start {job.id}")
        started = time.monotonic()
        batch = bank.deferred() if job.kind in ONE_COMMIT_KINDS else contextlib.nullcontext()
        try:
            with batch:
                out = HANDLERS[job.kind](job, ctx)
                record = {"status": STATUS_EXPLORATORY, "job": job.id, "kind": job.kind,
                          "params": job.params, "elapsed_s": round(time.monotonic() - started, 1),
                          "env": env_info(), "result": out}
                bank.mark_done(job.id, record, ctx.paths.work / "markers")
        except InjectedFault:
            raise
        except DeadlineReached as exc:
            log(f"worker: {job.id} paused: {exc}")
            break
        except Exception as exc:
            failed[job.id] = f"{type(exc).__name__}: {exc}"
            log(f"worker: {job.id} FAILED -- {failed[job.id]}\n{traceback.format_exc()}")
            continue
        ran.append(job.id)
        log(f"worker: done {job.id} in {record['elapsed_s']:.0f}s")
    if ctx._judges is not None:
        ctx._judges.unload()
    ctx.drop_start()
    return {"ran": ran, "failed": failed}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bank", required=True, help="hf://user/repo or a local folder")
    ap.add_argument("--gpu", type=int, default=None)
    ap.add_argument("--n-gpus", type=int, default=None)
    ap.add_argument("--priorities", nargs="+", default=["P1"])
    ap.add_argument("--deadline-minutes", type=float, default=None)
    ap.add_argument("--only", nargs="*", default=None, help="run only these job ids")
    args = ap.parse_args()
    if args.gpu is not None:
        if args.n_gpus is None:
            ap.error("--gpu needs --n-gpus (how many workers share the queue)")
        # see only this GPU, before anything touches CUDA (launch.py sets it too)
        os.environ.setdefault("CUDA_VISIBLE_DEVICES", str(args.gpu))

    from src.seedrep.bank import open_bank

    bank = open_bank(args.bank, token=plat.secret("HF_TOKEN"))
    n_gpus = args.n_gpus if args.n_gpus is not None else plat.gpu_count()
    queue = partition(build_jobs(tuple(args.priorities)), args.gpu, n_gpus)
    if args.only:
        queue = [j for j in queue if j.id in set(args.only)]
    print(f"worker gpu={args.gpu} n_gpus={n_gpus}: {len(queue)} jobs in queue")
    result = run(bank, queue, Settings(), deadline_minutes=args.deadline_minutes)
    print(json.dumps(result, indent=1))
    return 0 if not result["failed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
