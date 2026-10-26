"""Fast checks for the seed-replication plumbing (no models)."""
import json

import pytest

from src.seedrep import jobs
from src.seedrep.bank import LocalBank, open_bank
from src.seedrep.train import InjectedFault, fault_point


def test_local_bank_round_trip(tmp_path):
    bank = LocalBank(tmp_path / "bank")
    src = tmp_path / "a.txt"
    src.write_text("hello", encoding="utf-8")
    bank.put({"x/y/a.txt": src}, "add")
    assert bank.exists("x/y/a.txt")
    assert not bank.exists("x/y/b.txt")
    out = tmp_path / "back" / "a.txt"
    assert bank.get("x/y/a.txt", out) and out.read_text(encoding="utf-8") == "hello"
    assert bank.get("x/y/missing.txt", tmp_path / "m.txt") is False
    bank.delete(["x/y/a.txt"], "rm")
    assert not bank.exists("x/y/a.txt")


def test_prefixes_do_not_bleed_between_runs(tmp_path):
    """resume/sft_alpaca_s1 must not see resume/sft_alpaca_s10."""
    bank = LocalBank(tmp_path)
    f = tmp_path / "f"
    f.write_text("1", encoding="utf-8")
    bank.put({"resume/sft_alpaca_s10/checkpoint-2/x": f}, "add")
    assert bank.list("resume/sft_alpaca_s1/") == []
    assert bank.get_dir("resume/sft_alpaca_s1", tmp_path / "out") is False


def test_done_markers(tmp_path):
    bank = open_bank(str(tmp_path / "bank"))
    assert not bank.is_done("job_a")
    bank.mark_done("job_a", {"ok": True}, tmp_path / "tmp")
    assert bank.is_done("job_a")
    assert json.loads((tmp_path / "bank" / "done" / "job_a.json").read_text())["ok"]


def test_fault_point_fires_once(tmp_path, monkeypatch):
    monkeypatch.setenv("SEEDREP_FAULT_AT", "train:3")
    fault_point("train", 2, tmp_path)
    with pytest.raises(InjectedFault):
        fault_point("train", 3, tmp_path)
    fault_point("train", 3, tmp_path)          # the rerun passes


def test_job_table_shape_and_dependencies():
    table = jobs.build_jobs()
    ids = {j.id for j in table}
    assert len(ids) == len(table), "job ids must be unique"
    for j in table:
        for d in j.deps:
            assert d in ids, f"{j.id} depends on unknown {d}"
    # three runs per arm in all: the original seed 42 plus seeds 1 and 2
    assert {j.params["seed"] for j in table if j.kind == "train" and j.params["recipe"] == "dpo"} == {1, 2}
    # the two partitions together are the whole table, on one GPU interleaved
    two = jobs.partition(table, 0, 2) + jobs.partition(table, 1, 2)
    one = jobs.partition(table, None, 1)
    assert sorted(j.id for j in two) == sorted(ids) == sorted(j.id for j in one)


def test_single_gpu_queue_respects_dependency_order():
    order = [j.id for j in jobs.partition(jobs.build_jobs(), None, 1)]
    pos = {jid: i for i, jid in enumerate(order)}
    for j in jobs.build_jobs():
        for d in j.deps:
            assert pos[d] < pos[j.id], f"{d} must come before {j.id}"


def test_dose_schedules_exclude_the_final_step():
    # the endpoint is measured from the fp32 final adapter, not a dose copy
    assert max(jobs.SFT_DOSE) < 250 and max(jobs.DPO_DOSE) < 125


def test_deferred_puts_become_one_commit_with_the_marker(tmp_path):
    bank = LocalBank(tmp_path / "bank")
    a, b = tmp_path / "a", tmp_path / "b"
    a.write_text("1", encoding="utf-8")
    b.write_text("2", encoding="utf-8")
    with bank.deferred():
        bank.put({"m/a": a}, "a")
        bank.put({"m/b": b}, "b")
        bank.mark_done("job", {"ok": 1}, tmp_path / "tmp")
        assert not bank.exists("m/a"), "nothing lands before the job succeeds"
    assert bank.commits == 1
    assert bank.exists("m/a") and bank.exists("m/b") and bank.is_done("job")


def test_failed_deferred_job_leaves_no_trace(tmp_path):
    bank = LocalBank(tmp_path / "bank")
    a = tmp_path / "a"
    a.write_text("1", encoding="utf-8")
    with pytest.raises(RuntimeError):
        with bank.deferred():
            bank.put({"m/a": a}, "a")
            raise RuntimeError("job failed")
    assert bank.commits == 0 and not bank.exists("m/a")


def test_upload_and_delete_share_one_commit(tmp_path):
    bank = LocalBank(tmp_path / "bank")
    a = tmp_path / "a"
    a.write_text("1", encoding="utf-8")
    bank.put({"resume/r/checkpoint-2/x": a}, "old")
    bank.put({"resume/r/checkpoint-4/x": a}, "new", delete=["resume/r/checkpoint-2/x"])
    assert bank.commits == 2
    assert bank.list("resume/r/") == ["resume/r/checkpoint-4/x"]


def test_rate_limit_wait_reads_the_hub_message():
    from src.seedrep.bank import rate_limit_wait

    hour = Exception("429 Too Many Requests ... You can retry this action in about 1 hour.")
    mins = Exception("429 Too Many Requests ... You can retry this action in 44 minutes.")
    assert rate_limit_wait(hour) == 3660
    assert rate_limit_wait(mins) == 44 * 60 + 60
    assert rate_limit_wait(Exception("500 server error")) is None


def test_failures_block_their_dependents_transitively():
    from src.seedrep.jobs import Job
    from src.seedrep.worker import _blocked

    train = Job("train", "train", 0, "P1")
    gen = Job("gen", "gen", 0, "P1", ("train",))
    judge = Job("judge", "judge", 0, "P1", ("gen",))
    other = Job("other", "train", 0, "P1")
    assert _blocked([train, gen, judge, other], {"train": "x"}) == {"train", "gen", "judge"}
