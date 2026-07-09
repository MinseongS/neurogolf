from pathlib import Path
from neurogolf import gate

def _fake_eval(fail, cost):
    return {"ok": True, "pass": 260, "fail": fail, "memory": cost, "params": 0,
            "points": 10.0, "error": None, "cost": cost}

def test_gate_rejects_bundled_failure(monkeypatch, tmp_path):
    monkeypatch.setattr(gate, "eval_isolated", lambda p, t: _fake_eval(fail=2, cost=100))
    monkeypatch.setattr(gate, "deployed_cost", lambda t: 500)
    monkeypatch.setattr(gate, "find_unsigned_topk", lambda p: [])
    r = gate.gate(tmp_path / "c.onnx", 1)
    assert not r.ok and any("fail" in s for s in r.reasons)

def test_gate_rejects_not_cheaper(monkeypatch, tmp_path):
    monkeypatch.setattr(gate, "eval_isolated", lambda p, t: _fake_eval(fail=0, cost=500))
    monkeypatch.setattr(gate, "deployed_cost", lambda t: 500)   # 같아도 거부(strictly cheaper)
    monkeypatch.setattr(gate, "find_unsigned_topk", lambda p: [])
    assert not gate.gate(tmp_path / "c.onnx", 1).ok

def test_gate_rejects_unsigned_topk(monkeypatch, tmp_path):
    monkeypatch.setattr(gate, "eval_isolated", lambda p, t: _fake_eval(fail=0, cost=100))
    monkeypatch.setattr(gate, "deployed_cost", lambda t: 500)
    monkeypatch.setattr(gate, "find_unsigned_topk", lambda p: ["TopK uint8"])
    assert not gate.gate(tmp_path / "c.onnx", 1).ok

def test_gate_passes_clean_cheaper(monkeypatch, tmp_path):
    monkeypatch.setattr(gate, "eval_isolated", lambda p, t: _fake_eval(fail=0, cost=100))
    monkeypatch.setattr(gate, "deployed_cost", lambda t: 500)
    monkeypatch.setattr(gate, "find_unsigned_topk", lambda p: [])
    r = gate.gate(tmp_path / "c.onnx", 1)
    assert r.ok and r.candidate["cost"] == 100 and r.incumbent_cost == 500
