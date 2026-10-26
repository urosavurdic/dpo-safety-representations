"""End-to-end worker runs on a tiny model, including interrupted runs.

Slow (minutes on a laptop), so opt-in:  SEEDREP_SMOKE=1 python -m pytest tests/seedrep
"""
import json
import os

import pytest

pytestmark = pytest.mark.skipif(os.environ.get("SEEDREP_SMOKE") != "1",
                                reason="set SEEDREP_SMOKE=1 to run the CPU smoke tests")


def _env(monkeypatch, tmp_path, name):
    monkeypatch.setenv("SEEDREP_WORK", str(tmp_path / f"{name}_work"))
    monkeypatch.setenv("SEEDREP_SCRATCH", str(tmp_path / f"{name}_scratch"))


def _all_done(bank_dir):
    from src.seedrep.smoke import smoke_jobs

    return all((bank_dir / "done" / f"{j.id}.json").exists() for j in smoke_jobs())


@pytest.mark.parametrize("fault", ["train:3", "gen:1", "judge:0"])
def test_interrupted_run_resumes_on_same_platform(tmp_path, monkeypatch, fault):
    from src.seedrep.smoke import run_smoke
    from src.seedrep.train import InjectedFault

    _env(monkeypatch, tmp_path, "a")
    bank = tmp_path / "bank"
    with pytest.raises(InjectedFault):
        run_smoke(bank, fault=fault)
    result = run_smoke(bank, fault=fault)       # marker present: the rerun passes
    assert result["failed"] == {}
    assert _all_done(bank)


def test_training_resumes_on_a_fresh_platform_from_the_bank(tmp_path, monkeypatch):
    from src.seedrep.smoke import run_smoke
    from src.seedrep.train import InjectedFault

    bank = tmp_path / "bank"
    _env(monkeypatch, tmp_path, "kaggle")
    with pytest.raises(InjectedFault):
        run_smoke(bank, fault="train:3")
    assert any((bank / "resume").rglob("trainer_state.json")), "resume checkpoint was not banked"
    _env(monkeypatch, tmp_path, "colab")        # nothing local: everything comes from the bank
    logs = []
    result = run_smoke(bank, fault=None, log=logs.append)
    assert result["failed"] == {}
    assert any("pulled resume checkpoint" in line for line in logs)
    assert _all_done(bank)
    marker = json.loads((bank / "done" / "train__dpo_alpaca_s1.json").read_text())
    assert marker["result"]["resumed_from"] == "checkpoint-2"
    assert not list((bank / "resume").rglob("*.safetensors")), "resume files left behind"
