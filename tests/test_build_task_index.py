import importlib.util, math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

def _load():
    spec = importlib.util.spec_from_file_location(
        "build_task_index", ROOT / "reports/scripts/build_task_index.py")
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    return mod

def test_economic_row_computes_cost_and_bloat():
    m = _load()
    row = m.economic_row({"memory": 21526, "params": 100, "points": 15.0, "method": "urad:x"})
    assert row["cost"] == 21626
    assert row["bloat"] == 21626  # class_floor_est None -> floor 0
    assert row["status"] == "unexamined"

def test_structural_row_flags_on_task150():
    # task150 = single-axis Gather col-mirror (memory ~136); must parse without error
    import onnx
    p = ROOT / "networks/task150.onnx"
    if not p.exists():
        import pytest; pytest.skip("networks/task150.onnx absent")
    m = _load()
    row = m.structural_row(onnx.load(str(p)))
    assert "Gather" in row["ops"]
    assert row["node_count"] >= 1
    assert isinstance(row["flags"]["has_topk"], bool)
