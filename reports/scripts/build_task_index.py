#!/usr/bin/env python3
"""Build reports/task_index.json — per-task structural + economic + semantic features."""
from __future__ import annotations
import importlib
import json
import sys
from pathlib import Path
import numpy as np
import onnx

ROOT = Path(__file__).resolve().parents[2]
REPORTS = ROOT / "reports"
NETWORKS = ROOT / "networks"

sys.path.append(str(Path(__file__).resolve().parent))
import task_index_probes as _P  # noqa: E402

sys.path.append(str(ROOT / "arc-gen"))

_DTYPE_NAME = {1: "fp32", 2: "uint8", 3: "int8", 6: "int32", 7: "int64",
               9: "bool", 10: "fp16", 11: "fp64", 12: "uint32"}
_DTYPE_BYTES = {1: 4, 2: 1, 3: 1, 6: 4, 7: 8, 9: 1, 10: 2, 11: 8, 12: 4}


def load_manifest() -> dict:
    data = json.loads((REPORTS / "manifest.json").read_text())
    return data.get("tasks", data)


def load_mapping() -> dict:
    return json.loads((REPORTS / "arc_mapping.json").read_text())


def _tensor_bytes(vi) -> int:
    t = vi.type.tensor_type
    n = 1
    for d in t.shape.dim:
        n *= (d.dim_value or 1)
    return n * _DTYPE_BYTES.get(t.elem_type, 4)


def structural_row(model: onnx.ModelProto) -> dict:
    try:
        model = onnx.shape_inference.infer_shapes(model)
    except Exception:
        pass
    nodes = list(model.graph.node)
    ops = sorted({n.op_type for n in nodes})
    dtypes = set()
    for init in model.graph.initializer:
        dtypes.add(_DTYPE_NAME.get(init.data_type, str(init.data_type)))
    for vi in list(model.graph.value_info) + list(model.graph.input) + list(model.graph.output):
        dtypes.add(_DTYPE_NAME.get(vi.type.tensor_type.elem_type, "?"))
    tensors_by_name = {}
    for vi in list(model.graph.value_info) + list(model.graph.input) + list(model.graph.output):
        entry = {"name": vi.name, "bytes": _tensor_bytes(vi),
                 "shape": [d.dim_value or 0 for d in vi.type.tensor_type.shape.dim]}
        prev = tensors_by_name.get(entry["name"])
        if prev is None or entry["bytes"] > prev["bytes"]:
            tensors_by_name[entry["name"]] = entry
    for init in model.graph.initializer:
        n = 1
        for d in init.dims:
            n *= d
        entry = {"name": init.name, "bytes": n * _DTYPE_BYTES.get(init.data_type, 4),
                 "shape": list(init.dims)}
        prev = tensors_by_name.get(entry["name"])
        if prev is None or entry["bytes"] > prev["bytes"]:
            tensors_by_name[entry["name"]] = entry
    tensors = sorted(tensors_by_name.values(), key=lambda x: x["bytes"], reverse=True)
    topk_K = []
    for n in nodes:
        if n.op_type == "TopK":
            for a in n.attribute:
                if a.name == "k":
                    topk_K.append(int(a.i))
    last_op = nodes[-1].op_type if nodes else ""
    flags = {
        "has_topk": any(n.op_type == "TopK" for n in nodes),
        "topk_K": topk_K,
        "has_gridsample": any(n.op_type == "GridSample" for n in nodes),
        "has_qlinearconv": any(n.op_type in ("QLinearConv", "ConvInteger") for n in nodes),
        "has_scatter": any(n.op_type.startswith("Scatter") for n in nodes),
        "has_einsum": any(n.op_type == "Einsum" for n in nodes),
        "output_routed": last_op in ("Where", "Equal", "Einsum", "Greater", "Cast"),
        "output_concat_rebuild": last_op in ("Concat", "Pad"),
    }
    return {"ops": ops, "dtypes": sorted(d for d in dtypes if d),
            "node_count": len(nodes), "top_tensors": tensors[:3], "flags": flags}


def economic_row(manifest_task: dict) -> dict:
    mem = int(manifest_task.get("memory", 0)); params = int(manifest_task.get("params", 0))
    cost = mem + params
    floor = None
    return {"cost": cost, "mem": mem, "params": params,
            "points": float(manifest_task.get("points", 0.0)),
            "method": manifest_task.get("method", ""),
            "class_floor_est": floor, "bloat": cost - (floor or 0), "status": "unexamined"}


def build_structural_economic() -> dict:
    manifest = load_manifest(); mapping = load_mapping(); out = {}
    for num, minfo in manifest.items():
        arc = mapping.get(num, {}).get("arc_id", "")
        row = {"arc_id": arc, "economic": economic_row(minfo)}
        p = NETWORKS / f"task{int(num):03d}.onnx"
        structural = {}
        if p.exists():
            try:
                structural = structural_row(onnx.load(str(p)))
            except Exception:
                structural = {}
        row["structural"] = structural
        out[num] = row
    return out


def _PV():
    return _P.PROBE_VERSION


def _to_grid(x):
    return np.asarray(x, dtype=np.int64)


def sample_pairs(arc_id: str, n: int):
    try:
        gen = importlib.import_module(f"tasks.task_{arc_id}")
    except Exception:
        return []
    pairs = []
    for _ in range(n):
        try:
            ex = gen.generate()
            pairs.append((_to_grid(ex["input"]), _to_grid(ex["output"])))
        except Exception:
            continue
    return pairs


def semantic_row(arc_id: str, n: int) -> dict:
    pairs = sample_pairs(arc_id, n)
    row = _P.run_probes(pairs) if pairs else {}
    row["probe_version"] = _P.PROBE_VERSION
    row["n_samples"] = len(pairs)
    return row


def build_index(n_samples: int = 40, only=None) -> dict:
    base = build_structural_economic()
    cache = {}
    cache_path = REPORTS / "task_index.json"
    if cache_path.exists():
        cache = json.loads(cache_path.read_text())
    for num, row in base.items():
        if only is not None and num not in only:
            cached = cache.get(num, {}).get("semantic")
            if cached and cached.get("probe_version") == _P.PROBE_VERSION:
                row["semantic"] = cached
                continue
        row["semantic"] = semantic_row(row["arc_id"], n_samples)
    return base


def main() -> None:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=40)
    ap.add_argument("--only", nargs="*", default=None)
    a = ap.parse_args()
    idx = build_index(a.n, set(a.only) if a.only else None)
    (REPORTS / "task_index.json").write_text(json.dumps(idx, indent=1))
    print(f"wrote {len(idx)} rows (probe_version {_P.PROBE_VERSION}) -> reports/task_index.json")


if __name__ == "__main__":
    main()
