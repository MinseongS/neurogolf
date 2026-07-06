import importlib.util
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]

def _load():
    spec = importlib.util.spec_from_file_location(
        "task_index_probes", ROOT / "reports/scripts/task_index_probes.py")
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    return mod

def _identity_samples():
    g = np.array([[1, 2, 0], [0, 3, 4]])
    return [(g.copy(), g.copy()) for _ in range(20)]

def _fill_samples():
    # output = input with all-0 cells set to colour 5 (fixed-colour delta)
    out = []
    for _ in range(20):
        g = np.random.randint(0, 4, size=(5, 5))
        o = g.copy(); o[o == 0] = 5
        out.append((g, o))
    return out

def test_shape_relation_equal_on_identity():
    m = _load()
    val, conf = m.probe_shape_relation(_identity_samples())
    assert val == "equal" and conf == 1.0

def test_delta_copy_frac_high_on_fill():
    m = _load()
    val, conf = m.probe_delta(_fill_samples())
    assert val["copy_frac"] < 1.0 and val["changed_cells"] > 0

def test_color_source_fixed_delta_on_fill():
    m = _load()
    val, conf = m.probe_color_source(_fill_samples())
    assert val == "FIXED_DELTA"

def test_run_probes_returns_all_keys_with_confidence():
    m = _load()
    res = m.run_probes(_identity_samples())
    assert "shape_relation" in res and "confidence" in res["shape_relation"]
    assert res["shape_relation"]["value"] == "equal"
