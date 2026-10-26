"""Chunked, banked judging with the project's two judges, loaded once.

``behavioral_judges.run_judges`` writes its output only at the very end, so a
lost session loses the whole pass. Here the same scorers
(``regex_only_record`` + ``score_strongreject`` + ``score_wildguard``, 4-bit,
documented prompt formats) run over chunks of 100 rows. Each chunk is banked
as it finishes, and a rerun skips banked chunks. Records keep the judge-file
schema, so ``stagecmp.loaders.behavior_rows`` reads the merged file.

Fail-closed: a chunk where either judge reports ``model_unavailable`` is never
banked, because a full-looking file with no scores is how the project once
lost a judging pass (DATA_STATE.md section 2).
"""
from __future__ import annotations

import json
import os
from pathlib import Path

from src.seedrep.train import fault_point

CHUNK = 100


class Judges:
    """Both judges, loaded on first use and kept for the session."""

    def __init__(self, load_4bit: bool = True, allow_download: bool = True, scorer=None):
        self.load_4bit = load_4bit
        self.allow_download = allow_download
        self._scorer = scorer          # tests: a function rec -> None that fills scores
        self.sr = self.wg = None

    def ensure(self):
        if self._scorer is not None or self.sr is not None:
            return
        from src.analysis.behavioral_judges import (
            DEFAULT_STRONGREJECT_MODEL, DEFAULT_WILDGUARD_MODEL, LazyModelJudge)

        self.sr = LazyModelJudge("strong_reject", DEFAULT_STRONGREJECT_MODEL,
                                 load_4bit=self.load_4bit, allow_download=self.allow_download,
                                 mode="score_1_to_5")
        self.wg = LazyModelJudge("wildguard", DEFAULT_WILDGUARD_MODEL,
                                 load_4bit=self.load_4bit, allow_download=self.allow_download,
                                 mode="generate")
        for judge in (self.sr, self.wg):
            if not judge.try_load():
                raise RuntimeError(f"{judge.name} failed to load: {judge.load_error}")

    def score(self, row: dict) -> dict:
        from src.analysis.behavioral_judges import (
            regex_only_record, score_strongreject, score_wildguard)

        rec = regex_only_record(row)
        rec["model_judge_scope"] = "all"
        if self._scorer is not None:
            self._scorer(rec)
        else:
            score_strongreject(rec, self.sr)
            score_wildguard(rec, self.wg)
        return rec

    def unload(self):
        for judge in (self.sr, self.wg):
            if judge is not None:
                judge.unload()
        self.sr = self.wg = None


def _complete(rec: dict) -> bool:
    return all((rec.get(k) or {}).get("judge_status") == "scored"
               for k in ("strong_reject", "wildguard"))


def judge(rows: list[dict], *, unit: str, judges: Judges, bank, work: Path, scratch: Path,
          deadline=None, chunk_seconds: float = 300.0, log=print) -> list[dict]:
    local = Path(work) / "judge" / unit
    remote = f"judge/{unit}"
    for name in bank.list(remote + "/"):
        if Path(name).name.startswith("chunk_") and not (local / Path(name).name).exists():
            bank.get(name, local / Path(name).name)
    chunks = [rows[i:i + CHUNK] for i in range(0, len(rows), CHUNK)]
    for index, chunk in enumerate(chunks):
        path = local / f"chunk_{index:04d}.json"
        if path.exists():
            continue
        if deadline is not None and deadline.would_exceed(chunk_seconds):
            from src.pipeline.shards import DeadlineReached

            raise DeadlineReached(f"{unit}: stopped before chunk {index}")
        judges.ensure()
        fault_point("judge", index, scratch)
        records = [judges.score(r) for r in chunk]
        bad = [r["record_id"] for r in records if not _complete(r)]
        if bad:
            raise RuntimeError(f"{unit} chunk {index}: {len(bad)} rows not scored by both "
                               f"judges (e.g. {bad[:3]}); refusing to bank")
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(path.name + ".tmp")
        tmp.write_text(json.dumps(records, ensure_ascii=False), encoding="utf-8", newline="\n")
        os.replace(tmp, path)
        bank.put({f"{remote}/{path.name}": path}, f"{unit}: judged chunk {index}")
        log(f"  [{unit}] judged chunk {index + 1}/{len(chunks)}")

    merged, seen = [], set()
    for index in range(len(chunks)):
        for rec in json.loads((local / f"chunk_{index:04d}.json").read_text(encoding="utf-8")):
            key = (rec["record_id"], rec["condition"])
            if key not in seen:           # first wins, as everywhere in the project
                seen.add(key)
                merged.append(rec)
    out = local / "judged.json"
    out.write_text(json.dumps({"records": merged}, ensure_ascii=False), encoding="utf-8")
    bank.put({f"{remote}/judged.json": out}, f"{unit}: merged judged records")
    return merged
