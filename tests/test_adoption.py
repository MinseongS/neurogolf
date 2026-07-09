import json
from pathlib import Path
from neurogolf import adoption, gate

def _setup(tmp_path, monkeypatch):
    nets = tmp_path / "submission" / "overfit_nets"; nets.mkdir(parents=True)
    (nets / "task001.onnx").write_bytes(b"OLD")
    state = tmp_path / "state"; (state / "tasks").mkdir(parents=True)
    (state / "tasks" / "task001.md").write_text("# task001\n")
    json.dump({}, open(state / "manifest.json", "w"))
    for mod in (adoption, adoption.manifest):
        monkeypatch.setattr(mod, "STATE", state, raising=False)
    monkeypatch.setattr(adoption, "OVERFIT_NETS", nets)
    monkeypatch.setattr(adoption, "BACKUPS", tmp_path / "submission" / ".backups")
    return nets, state

def test_adopt_replaces_backs_up_and_stamps(tmp_path, monkeypatch):
    nets, state = _setup(tmp_path, monkeypatch)
    cand = tmp_path / "cand.onnx"; cand.write_bytes(b"NEW")
    ok = gate.GateResult(ok=True, candidate={"cost": 100, "points": 20.0, "ok": True, "fail": 0}, incumbent_cost=500)
    monkeypatch.setattr(adoption, "gate_candidate", lambda c, t: ok)
    row = adoption.adopt(cand, 1, note="test win")
    assert (nets / "task001.onnx").read_bytes() == b"NEW"
    assert list((tmp_path / "submission" / ".backups").glob("task001_*.onnx"))
    assert json.load(open(state / "manifest.json"))["001"]["cost"] == 100
    assert "test win" in (state / "tasks" / "task001.md").read_text()

def test_adopt_refuses_on_gate_failure(tmp_path, monkeypatch):
    nets, _ = _setup(tmp_path, monkeypatch)
    cand = tmp_path / "cand.onnx"; cand.write_bytes(b"NEW")
    bad = gate.GateResult(ok=False, reasons=["bundled fail != 0"])
    monkeypatch.setattr(adoption, "gate_candidate", lambda c, t: bad)
    import pytest
    with pytest.raises(SystemExit):
        adoption.adopt(cand, 1)
    assert (nets / "task001.onnx").read_bytes() == b"OLD"
