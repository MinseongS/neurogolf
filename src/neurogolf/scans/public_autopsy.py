#!/usr/bin/env python3
"""Autopsy public ONNX wins to discover reusable compiler mechanisms.

Min-merge answers "can we adopt this artifact?".  This answers "what changed
structurally, and where else should we try that mechanism?".

Ported (Task 13): scan_all-compatible. For every deployed net that has a
pre-replacement backup (submission/overfit_nets/.minmerge_backup/ or the latest
submission/.backups/taskNNN_*.onnx), it treats the backup as the base and the deployed
(won) net as the public winner, extracts the op-delta / lost-tensor fingerprint, and
emits one item per win with the deployed nets that share the same fingerprint as
rescan_candidates (the deep-lane ④ apply list).
"""
from __future__ import annotations

import glob
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import onnx
from onnx import shape_inference

from neurogolf.scoring import evaluate, load_task
from neurogolf.paths import ROOT, OVERFIT_NETS

MINMERGE_BACKUP = OVERFIT_NETS / ".minmerge_backup"
ADOPT_BACKUPS = ROOT / "submission" / ".backups"

DTYPE_BYTES = {1: 4, 2: 1, 3: 1, 4: 2, 5: 2, 6: 4, 7: 8, 9: 1, 10: 2, 11: 8, 12: 4, 13: 8, 16: 2}
SPATIAL_OPS = {"Conv", "ConvTranspose", "ConvInteger", "QLinearConv", "GridSample",
               "MaxPool", "AveragePool", "TopK", "GatherND", "GatherElements"}
INDEX_OPS = {"TopK", "ArgMax", "Gather", "GatherND", "GatherElements", "ScatterND", "ScatterElements"}


def rel(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT))
    except Exception:
        return str(path)


def elem_count(dims: tuple[int, ...]) -> int:
    out = 1
    for dim in dims:
        out *= int(dim)
    return out


def tensor_bytes(elem_type: int, dims: tuple[int, ...]) -> int:
    if not dims or any(dim <= 0 for dim in dims):
        return 0
    return elem_count(dims) * DTYPE_BYTES.get(elem_type, 4)


def infer_model(path: Path):
    try:
        return shape_inference.infer_shapes(onnx.load(str(path)), strict_mode=False)
    except Exception:
        return None


def value_info(model) -> dict:
    out: dict[str, tuple[int, tuple[int, ...]]] = {}
    for vi in list(model.graph.input) + list(model.graph.value_info) + list(model.graph.output):
        if not vi.type.HasField("tensor_type"):
            continue
        tt = vi.type.tensor_type
        dims = tuple(dim.dim_value if dim.HasField("dim_value") else 0 for dim in tt.shape.dim)
        out[vi.name] = (int(tt.elem_type), dims)
    return out


def initializer_stats(model) -> tuple:
    rows = []
    total = 0
    for init in model.graph.initializer:
        dims = tuple(int(dim) for dim in init.dims)
        elems = int(math.prod(dims)) if dims else 1
        total += elems
        rows.append({"name": init.name, "elems": elems, "dtype": int(init.data_type), "shape": list(dims)})
    rows.sort(key=lambda row: row["elems"], reverse=True)
    return total, rows[:20]


def profile(path: Path):
    model = infer_model(path)
    if model is None:
        return None
    vi = value_info(model)
    ops = Counter(node.op_type for node in model.graph.node)
    node_outputs = []
    mem = 0
    spatial_bytes = 0
    index_bytes = 0
    wide_bytes = 0
    dtype_bytes = Counter()
    for idx, node in enumerate(model.graph.node):
        for out in node.output:
            if out in ("input", "output"):
                continue
            elem_type, dims = vi.get(out, (0, ()))
            b = tensor_bytes(elem_type, dims)
            mem += b
            if b:
                dtype_bytes[str(elem_type)] += b
            if node.op_type in SPATIAL_OPS:
                spatial_bytes += b
            if node.op_type in INDEX_OPS:
                index_bytes += b
            if b >= 900:
                wide_bytes += b
            node_outputs.append({"node_index": idx, "op": node.op_type, "name": out, "bytes": b,
                                 "dtype": elem_type, "shape": list(dims), "inputs": list(node.input)})
    init_elems, largest_inits = initializer_stats(model)
    node_outputs.sort(key=lambda row: row["bytes"], reverse=True)
    return {
        "path": rel(path), "node_count": len(model.graph.node),
        "initializer_count": len(model.graph.initializer), "initializer_elems": init_elems,
        "ops": dict(sorted(ops.items())), "memory_static": mem, "spatial_bytes": spatial_bytes,
        "index_bytes": index_bytes, "wide_bytes": wide_bytes, "dtype_bytes": dict(dtype_bytes),
        "largest_initializers": largest_inits, "largest_outputs": node_outputs[:25],
    }


def static_cost(path: Path):
    prof = profile(path)
    if prof is None:
        return None
    return int(prof["memory_static"]) + int(prof["initializer_elems"])


def safe_eval(path: Path, task: int) -> dict:
    try:
        return evaluate(str(path), load_task(task), keep_failures=False)
    except Exception as exc:
        return {"ok": False, "pass": 0, "fail": None, "memory": None, "params": None,
                "points": 0.0, "error": str(exc)}


def diff_counter(public: dict, base: dict) -> dict:
    keys = sorted(set(public) | set(base))
    return {key: public.get(key, 0) - base.get(key, 0) for key in keys if public.get(key, 0) != base.get(key, 0)}


def shape_key(row: dict) -> str:
    return f"{row['op']}:{row['dtype']}:{'x'.join(map(str, row['shape']))}"


def fingerprint_label(row: dict) -> str:
    return f"{shape_key(row)}:{row['bytes']}B"


def lost_big_tensors(base_prof: dict, public_prof: dict) -> list:
    pub_shapes = Counter(shape_key(row) for row in public_prof["largest_outputs"] if row["bytes"] > 0)
    lost = []
    for row in base_prof["largest_outputs"]:
        if row["bytes"] < 100:
            continue
        key = shape_key(row)
        if pub_shapes[key] > 0:
            pub_shapes[key] -= 1
            continue
        lost.append(row)
    return lost[:20]


def gained_big_tensors(base_prof: dict, public_prof: dict) -> list:
    base_shapes = Counter(shape_key(row) for row in base_prof["largest_outputs"] if row["bytes"] > 0)
    gained = []
    for row in public_prof["largest_outputs"]:
        if row["bytes"] < 100:
            continue
        key = shape_key(row)
        if base_shapes[key] > 0:
            base_shapes[key] -= 1
            continue
        gained.append(row)
    return gained[:20]


def output_fingerprints(prof: dict, min_bytes: int = 100) -> Counter:
    return Counter(fingerprint_label(row) for row in prof["largest_outputs"]
                   if int(row.get("bytes") or 0) >= min_bytes)


def learned_lost_fingerprints(rows: list) -> dict:
    fingerprints: dict[str, dict] = {}
    for row in rows:
        for lost in row["lost_big_outputs"]:
            if int(lost.get("bytes") or 0) < 500:
                continue
            key = fingerprint_label(lost)
            item = fingerprints.setdefault(key, {
                "fingerprint": key, "examples": [], "count": 0, "total_delta_points": 0.0,
                "total_saved_cost": 0, "op": lost["op"], "dtype": lost["dtype"],
                "shape": lost["shape"], "bytes": lost["bytes"]})
            item["count"] += 1
            item["total_delta_points"] += float(row["delta_points"])
            item["total_saved_cost"] += -int(row["delta_cost"])
            if len(item["examples"]) < 8:
                item["examples"].append({"task": row["task"], "dump": row["dump"],
                                         "delta_points": row["delta_points"],
                                         "delta_cost": row["delta_cost"], "tags": row["tags"]})
    return fingerprints


def classify(base_prof, public_prof, base_ev, public_ev) -> list:
    tags: list[str] = []
    op_delta = diff_counter(public_prof["ops"], base_prof["ops"])
    lost = lost_big_tensors(base_prof, public_prof)
    gained = gained_big_tensors(base_prof, public_prof)
    mem_delta = int(public_ev["memory"]) - int(base_ev["memory"])
    par_delta = int(public_ev["params"]) - int(base_ev["params"])
    spatial_delta = int(public_prof["spatial_bytes"]) - int(base_prof["spatial_bytes"])
    index_delta = int(public_prof["index_bytes"]) - int(base_prof["index_bytes"])
    wide_delta = int(public_prof["wide_bytes"]) - int(base_prof["wide_bytes"])

    if par_delta < 0 and abs(par_delta) >= max(1, abs(mem_delta)):
        tags.append("params_tail_or_initializer_dedupe")
    if spatial_delta < -500 or any(row["op"] in SPATIAL_OPS and row["bytes"] >= 500 for row in lost):
        tags.append("spatial_plane_removed")
    if index_delta < -300 or op_delta.get("TopK", 0) < 0 or any(row["op"] == "TopK" for row in lost):
        tags.append("index_or_topk_plane_removed")
    if op_delta.get("Einsum", 0) > 0 and (spatial_delta < 0 or wide_delta < 0):
        tags.append("free_input_einsum_substitution")
    if op_delta.get("BitShift", 0) > 0 or op_delta.get("BitwiseAnd", 0) > 0:
        tags.append("bitpack_or_arithmetic_decode")
    if op_delta.get("QLinearConv", 0) > 0 or op_delta.get("QLinearMatMul", 0) > 0:
        tags.append("quantized_integer_route")
    if op_delta.get("Equal", 0) > 0 and int(public_ev["memory"]) <= int(base_ev["memory"]):
        tags.append("final_equal_or_output_only")
    if wide_delta < -900 and not tags:
        tags.append("wide_tensor_removed_unknown")
    if len(gained) and any(row["op"] == "Einsum" for row in gained):
        tags.append("einsum_added")
    return sorted(set(tags)) or ["unclassified_byte_tail"]


def autopsy_task(task: int, base_path: Path, public_path: Path, dump_name: str):
    base_prof = profile(base_path)
    public_prof = profile(public_path)
    if base_prof is None or public_prof is None:
        return None
    base_ev = safe_eval(base_path, task)
    public_ev = safe_eval(public_path, task)
    if not (base_ev.get("ok") and public_ev.get("ok") and public_ev.get("fail") == 0):
        return None
    base_cost = int(base_ev["memory"]) + int(base_ev["params"])
    public_cost = int(public_ev["memory"]) + int(public_ev["params"])
    if public_cost >= base_cost:
        return None
    op_delta = diff_counter(public_prof["ops"], base_prof["ops"])
    tags = classify(base_prof, public_prof, base_ev, public_ev)
    return {
        "task": task, "dump": dump_name, "public_path": rel(public_path), "base_path": rel(base_path),
        "delta_points": float(public_ev["points"]) - float(base_ev["points"]),
        "delta_cost": public_cost - base_cost, "tags": tags,
        "op_delta_public_minus_base": op_delta,
        "lost_big_outputs": lost_big_tensors(base_prof, public_prof),
        "gained_big_outputs": gained_big_tensors(base_prof, public_prof),
    }


def scan_fingerprint_candidates(base_dir: Path, rows: list, exclude_win_tasks: bool = True) -> list:
    learned = learned_lost_fingerprints(rows)
    if not learned:
        return []
    win_tasks = {int(row["task"]) for row in rows} if exclude_win_tasks else set()
    candidates = []
    for path in sorted(base_dir.glob("task*.onnx")):
        task = int(path.stem[4:])
        if task in win_tasks:
            continue
        prof = profile(path)
        if prof is None:
            continue
        counts = output_fingerprints(prof, min_bytes=100)
        matched = []
        score = 0.0
        for key, meta in learned.items():
            count = counts.get(key, 0)
            if not count:
                continue
            saved_per_example = meta["total_saved_cost"] / max(1, meta["count"])
            contribution = count * (float(meta["bytes"]) + saved_per_example)
            score += contribution
            matched.append({"fingerprint": key, "count": count, "estimated_score": contribution})
        if matched:
            matched.sort(key=lambda item: item["estimated_score"], reverse=True)
            candidates.append({"task": task, "score": round(score, 3),
                               "cost": prof["memory_static"] + prof["initializer_elems"],
                               "matches": matched[:8]})
    candidates.sort(key=lambda row: row["score"], reverse=True)
    return candidates[:80]


def _resolve_backup(task: int) -> Path | None:
    mm = MINMERGE_BACKUP / f"task{task:03d}.onnx"
    if mm.exists():
        return mm
    hits = sorted(glob.glob(str(ADOPT_BACKUPS / f"task{task:03d}_*.onnx")))
    if hits:
        return Path(hits[-1])  # latest timestamp
    return None


def scan_all(tasks: list[int] | None = None) -> dict:
    task_range = tasks if tasks else range(1, 401)
    rows = []
    for t in task_range:
        deployed = OVERFIT_NETS / f"task{t:03d}.onnx"
        if not deployed.exists():
            continue
        backup = _resolve_backup(t)
        if backup is None:
            continue
        row = autopsy_task(t, backup, deployed, backup.parent.name)
        if row:
            rows.append(row)

    # rescan candidates: our other deployed nets carrying the same lost-tensor fingerprint
    fp_cands = scan_fingerprint_candidates(OVERFIT_NETS, rows)
    # map fingerprint -> candidate tasks
    fp_to_tasks: dict[str, list[int]] = defaultdict(list)
    for c in fp_cands:
        for m in c["matches"]:
            fp_to_tasks[m["fingerprint"]].append(c["task"])

    items = []
    for row in rows:
        lost_fps = [fingerprint_label(x) for x in row["lost_big_outputs"] if int(x.get("bytes") or 0) >= 500]
        rescan = sorted({tt for fp in lost_fps for tt in fp_to_tasks.get(fp, [])})
        saved = -int(row["delta_cost"])  # delta_cost = public - base (negative for a win)
        items.append({
            "task": row["task"],
            "signature": ",".join(row["tags"]),
            "lost_tensors": lost_fps,
            "rescan_candidates": rescan,
            "delta_points": round(row["delta_points"], 6),
            "saved_bytes": saved,
            "expected_gain": round(row["delta_points"], 6),
        })
    items.sort(key=lambda i: -i["expected_gain"])
    return {"items": items}
