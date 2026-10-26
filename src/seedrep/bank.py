"""The bank: the one place every job's outputs live, reachable from anywhere.

Two backends with the same interface:

* ``HubBank`` -- a private Hugging Face *dataset* repo. Kaggle, Colab and a
  laptop all reach it with the same ``HF_TOKEN``, unlike Drive (Colab only) or
  Kaggle outputs (per notebook version).
* ``LocalBank`` -- a folder. Used by the tests and as an emergency fallback
  (zip the folder, upload it by hand, point ``SEEDREP_BANK`` at it).

A job is *done* only when ``done/<job_id>.json`` exists. The marker is written
together with, or after, the job's outputs, and carries their SHA-256s, so a
half-uploaded job is never mistaken for a finished one.

**Commit budget.** The Hub allows 128 commits per hour per repository. Every
``put`` is one commit, so the bank keeps well under that:

* ``HubBank`` spaces its own commits at least ``SEEDREP_COMMIT_INTERVAL``
  seconds apart (default 65, i.e. at most ~55 an hour per worker, ~110 for two);
* ``deferred()`` collects every put of a short job and commits them, with the
  job's done marker, as one commit when the job succeeds;
* a delete rides in the same commit as the upload that replaces it;
* a 429 "too many requests" is waited out (the Hub says how long), not failed.

    spec "hf://user/repo"   -> HubBank
    spec "file://path" | path -> LocalBank
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import time
from contextlib import contextmanager
from pathlib import Path

COMMIT_INTERVAL = float(os.environ.get("SEEDREP_COMMIT_INTERVAL", "65"))
MAX_RATE_LIMIT_WAIT = 75 * 60


def sha256_path(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


class Bank:
    _pending: dict | None = None

    # ---- backend interface ------------------------------------------------
    def _commit(self, files: dict[str, Path], delete: list[str], message: str) -> None:
        raise NotImplementedError

    def get(self, remote: str, local: Path) -> bool:
        raise NotImplementedError

    def list(self, prefix: str = "") -> list[str]:
        raise NotImplementedError

    def refresh(self) -> None:
        """Forget any cached listing (another process may have written)."""

    # ---- writes -----------------------------------------------------------
    def put(self, files: dict[str, Path], message: str, delete=()) -> None:
        """Upload ``files`` and remove ``delete`` in one commit (or, inside
        ``deferred()``, add them to the job's single commit)."""
        if self._pending is not None:
            self._pending["files"].update(files)
            self._pending["delete"].update(delete)
            self._pending["messages"].append(message)
            return
        if files or delete:
            self._commit(dict(files), [d for d in delete if d not in files], message)

    def delete(self, remotes: list[str], message: str) -> None:
        self.put({}, message, delete=remotes)

    @contextmanager
    def deferred(self):
        """Collect every put inside the block into one commit, made only if the
        block finishes without an exception (a failed job leaves no trace)."""
        if self._pending is not None:           # nested: join the outer batch
            yield
            return
        self._pending = {"files": {}, "delete": set(), "messages": []}
        try:
            yield
            batch = self._pending
            self._pending = None
            if batch["files"] or batch["delete"]:
                message = batch["messages"][-1] if batch["messages"] else "batch"
                self._commit(batch["files"], [d for d in batch["delete"]
                                              if d not in batch["files"]], message)
        finally:
            self._pending = None

    # ---- shared helpers ---------------------------------------------------
    def exists(self, remote: str) -> bool:
        return remote in set(self.list(os.path.dirname(remote) + "/"))

    def put_json(self, remote: str, obj, message: str, tmp_dir: Path) -> None:
        tmp_dir.mkdir(parents=True, exist_ok=True)
        local = tmp_dir / remote.replace("/", "__")
        local.write_text(json.dumps(obj, indent=1), encoding="utf-8", newline="\n")
        self.put({remote: local}, message)

    def get_json(self, remote: str, tmp_dir: Path):
        tmp_dir.mkdir(parents=True, exist_ok=True)
        local = tmp_dir / remote.replace("/", "__")
        if not self.get(remote, local):
            return None
        return json.loads(local.read_text(encoding="utf-8"))

    def put_dir(self, local_dir: Path, remote_prefix: str, message: str, delete=()) -> dict:
        files = {f"{remote_prefix}/{p.relative_to(local_dir).as_posix()}": p
                 for p in sorted(Path(local_dir).rglob("*")) if p.is_file()}
        self.put(files, message, delete=delete)
        return {k: sha256_path(v) for k, v in files.items()}

    def get_dir(self, remote_prefix: str, local_dir: Path) -> bool:
        names = [n for n in self.list(remote_prefix + "/")]
        for name in names:
            self.get(name, Path(local_dir) / name[len(remote_prefix) + 1:])
        return bool(names)

    def is_done(self, job_id: str) -> bool:
        return self.exists(f"done/{job_id}.json")

    def mark_done(self, job_id: str, record: dict, tmp_dir: Path) -> None:
        self.put_json(f"done/{job_id}.json", record, f"done {job_id}", tmp_dir)


class LocalBank(Bank):
    def __init__(self, root: Path):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.commits = 0                  # tests count commits

    def _commit(self, files, delete, message):
        for remote, local in files.items():
            target = self.root / remote
            target.parent.mkdir(parents=True, exist_ok=True)
            tmp = target.with_name(target.name + ".tmp")
            shutil.copyfile(local, tmp)
            os.replace(tmp, target)
        for remote in delete:
            (self.root / remote).unlink(missing_ok=True)
        self.commits += 1

    def get(self, remote, local):
        source = self.root / remote
        if not source.exists():
            return False
        Path(local).parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, local)
        return True

    def list(self, prefix=""):
        base = self.root / prefix if prefix else self.root
        if not base.exists():
            return []
        return sorted(p.relative_to(self.root).as_posix() for p in base.rglob("*")
                      if p.is_file() and not p.name.endswith(".tmp"))


def rate_limit_wait(exc: Exception) -> float | None:
    """Seconds to wait if ``exc`` is the Hub's commit rate limit, else None."""
    text = str(exc)
    status = getattr(getattr(exc, "response", None), "status_code", None)
    if status != 429 and "429" not in text:
        return None
    minutes = re.search(r"in (?:about )?(\d+) minute", text)
    if minutes:
        return int(minutes.group(1)) * 60 + 60
    if re.search(r"in (?:about )?(?:an|1) hour", text):
        return 3600 + 60
    return 15 * 60


class HubBank(Bank):
    """Private HF dataset repo. Listing is cached and refreshed after writes."""

    def __init__(self, repo_id: str, token: str | None = None, create: bool = True,
                 interval: float = COMMIT_INTERVAL, log=print):
        from huggingface_hub import HfApi

        self.repo_id = repo_id
        self.api = HfApi(token=token)
        self.interval = interval
        self.log = log
        self._last_commit = 0.0
        if create:
            self.api.create_repo(repo_id, repo_type="dataset", private=True, exist_ok=True)
        self._files: list[str] | None = None

    def refresh(self) -> None:
        self._files = None

    def _refresh(self) -> list[str]:
        self._files = sorted(self.api.list_repo_files(self.repo_id, repo_type="dataset"))
        return self._files

    def _commit(self, files, delete, message):
        from huggingface_hub import CommitOperationAdd, CommitOperationDelete

        present = set(self.list()) if delete else set()
        ops = [CommitOperationAdd(path_in_repo=remote, path_or_fileobj=str(local))
               for remote, local in files.items()]
        ops += [CommitOperationDelete(path_in_repo=r) for r in delete if r in present]
        if not ops:
            return
        other_failures = 0
        while True:
            wait = self._last_commit + self.interval - time.monotonic()
            if wait > 0:
                time.sleep(wait)
            try:
                self.api.create_commit(self.repo_id, operations=ops, commit_message=message,
                                       repo_type="dataset")
                self._last_commit = time.monotonic()
                break
            except Exception as exc:
                self._last_commit = time.monotonic()
                limit = rate_limit_wait(exc)
                if limit is not None:
                    limit = min(limit, MAX_RATE_LIMIT_WAIT)
                    self.log(f"  [bank] commit rate limit reached; waiting {limit / 60:.0f} min "
                             f"before retrying '{message}'")
                    time.sleep(limit)
                    continue
                other_failures += 1
                if other_failures >= 5:
                    raise
                time.sleep(15 * other_failures)
        self._files = None

    def get(self, remote, local):
        from huggingface_hub import hf_hub_download

        if remote not in set(self.list(os.path.dirname(remote) + "/")):
            return False
        staging = Path(local).parent / ".hf_get"
        path = hf_hub_download(self.repo_id, remote, repo_type="dataset", local_dir=str(staging))
        Path(local).parent.mkdir(parents=True, exist_ok=True)
        shutil.move(path, local)          # move, not copy: adapters and arrays are large
        return True

    def list(self, prefix=""):
        files = self._files if self._files is not None else self._refresh()
        return [f for f in files if f.startswith(prefix)] if prefix else list(files)


def open_bank(spec: str | None = None, token: str | None = None) -> Bank:
    spec = spec or os.environ.get("SEEDREP_BANK")
    if not spec:
        raise ValueError("no bank: pass a spec or set SEEDREP_BANK (hf://user/repo or a path)")
    if spec.startswith("hf://"):
        return HubBank(spec[len("hf://"):], token=token)
    if spec.startswith("file://"):
        spec = spec[len("file://"):]
    return LocalBank(Path(spec))
