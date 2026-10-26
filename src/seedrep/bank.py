"""The bank: the one place every job's outputs live, reachable from anywhere.

Two backends with the same interface:

* ``HubBank`` -- a private Hugging Face *dataset* repo. Kaggle, Colab and a
  laptop all reach it with the same ``HF_TOKEN``, unlike Drive (Colab only) or
  Kaggle outputs (per notebook version). Each ``put`` is one commit.
* ``LocalBank`` -- a folder. Used by the tests and as an emergency fallback
  (zip the folder, upload it by hand, point ``SEEDREP_BANK`` at it).

A job is *done* only when ``done/<job_id>.json`` exists. The marker is written
last, after the job's outputs are in the bank, and carries their SHA-256s, so
a half-uploaded job is never mistaken for a finished one.

    spec "hf://user/repo"   -> HubBank
    spec "file://path" | path -> LocalBank
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil
from pathlib import Path


def sha256_path(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


class Bank:
    def put(self, files: dict[str, Path], message: str) -> None:
        raise NotImplementedError

    def get(self, remote: str, local: Path) -> bool:
        raise NotImplementedError

    def list(self, prefix: str = "") -> list[str]:
        raise NotImplementedError

    def delete(self, remotes: list[str], message: str) -> None:
        raise NotImplementedError

    def refresh(self) -> None:
        """Forget any cached listing (another process may have written)."""

    # ---- shared helpers --------------------------------------------------
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

    def put_dir(self, local_dir: Path, remote_prefix: str, message: str) -> dict:
        files = {f"{remote_prefix}/{p.relative_to(local_dir).as_posix()}": p
                 for p in sorted(Path(local_dir).rglob("*")) if p.is_file()}
        self.put(files, message)
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

    def put(self, files, message):
        for remote, local in files.items():
            target = self.root / remote
            target.parent.mkdir(parents=True, exist_ok=True)
            tmp = target.with_name(target.name + ".tmp")
            shutil.copyfile(local, tmp)
            os.replace(tmp, target)

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

    def delete(self, remotes, message):
        for remote in remotes:
            (self.root / remote).unlink(missing_ok=True)


class HubBank(Bank):
    """Private HF dataset repo. Listing is cached and refreshed after writes."""

    def __init__(self, repo_id: str, token: str | None = None, create: bool = True):
        from huggingface_hub import HfApi

        self.repo_id = repo_id
        self.api = HfApi(token=token)
        if create:
            self.api.create_repo(repo_id, repo_type="dataset", private=True, exist_ok=True)
        self._files: list[str] | None = None

    def refresh(self) -> None:
        self._files = None

    def _refresh(self) -> list[str]:
        self._files = sorted(self.api.list_repo_files(self.repo_id, repo_type="dataset"))
        return self._files

    def put(self, files, message):
        from huggingface_hub import CommitOperationAdd

        if not files:
            return
        ops = [CommitOperationAdd(path_in_repo=remote, path_or_fileobj=str(local))
               for remote, local in files.items()]
        for attempt in range(5):
            try:
                self.api.create_commit(self.repo_id, operations=ops, commit_message=message,
                                       repo_type="dataset")
                break
            except Exception:
                if attempt == 4:
                    raise
                import time

                time.sleep(10 * (attempt + 1))
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

    def delete(self, remotes, message):
        from huggingface_hub import CommitOperationDelete

        present = set(self.list())
        ops = [CommitOperationDelete(path_in_repo=r) for r in remotes if r in present]
        if ops:
            self.api.create_commit(self.repo_id, operations=ops, commit_message=message,
                                   repo_type="dataset")
        self._files = None


def open_bank(spec: str | None = None, token: str | None = None) -> Bank:
    spec = spec or os.environ.get("SEEDREP_BANK")
    if not spec:
        raise ValueError("no bank: pass a spec or set SEEDREP_BANK (hf://user/repo or a path)")
    if spec.startswith("hf://"):
        return HubBank(spec[len("hf://"):], token=token)
    if spec.startswith("file://"):
        spec = spec[len("file://"):]
    return LocalBank(Path(spec))
