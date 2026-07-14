import json
import subprocess

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


def test_eval_isolated_honors_configurable_timeout(monkeypatch, tmp_path):
    seen = {}

    def fake_run(*args, **kwargs):
        seen["timeout"] = kwargs["timeout"]
        return subprocess.CompletedProcess(
            args=args[0],
            returncode=0,
            stdout=json.dumps(_fake_eval(fail=0, cost=100)),
            stderr="",
        )

    monkeypatch.setenv("NG_EVAL_TIMEOUT_SECONDS", "900")
    monkeypatch.setattr(gate.subprocess, "run", fake_run)

    result = gate.eval_isolated(tmp_path / "candidate.onnx", 54)

    assert result["cost"] == 100
    assert seen["timeout"] == 900
