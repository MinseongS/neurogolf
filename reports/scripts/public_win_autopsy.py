#!/usr/bin/env python3
"""Autopsy public ONNX wins to discover reusable compiler mechanisms.

This is intentionally different from `mine_overfit_minmerge.py`.
Min-merge answers "can we adopt this artifact?".  This script answers
"what changed structurally, and where else should we try that mechanism?".

It compares a baseline net directory against one or more teacher dump dirs,
evaluates cheaper bundled-fail=0 teachers, profiles counted tensors, assigns
coarse mechanism tags, and writes JSON/Markdown reports.
"""

from __future__ import annotations

import argparse
import glob
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import onnx
from onnx import shape_inference

from src.harness import evaluate, load_task


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_BASE = ROOT / "submission" / "overfit_nets"
DEFAULT_OUT = ROOT / "reports" / "candidates" / "public_autopsy"

DTYPE_BYTES = {1: 4, 2: 1, 3: 1, 4: 2, 5: 2, 6: 4, 7: 8, 9: 1, 10: 2, 11: 8, 12: 4, 13: 8, 16: 2}
SPATIAL_OPS = {
    "Conv",
    "ConvTranspose",
    "ConvInteger",
    "QLinearConv",
    "GridSample",
    "MaxPool",
    "AveragePool",
    "TopK",
    "GatherND",
    "GatherElements",
}
INDEX_OPS = {"TopK", "ArgMax", "Gather", "GatherND", "GatherElements", "ScatterND", "ScatterElements"}


def rel(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT))
    except Exception:
        return str(path)


def find_net(dump_dir: Path, task: int) -> Path | None:
    patterns = [dump_dir / f"task{task:03d}.onnx", dump_dir / "**" / f"task{task:03d}.onnx"]
    for pattern in patterns:
        hits = glob.glob(str(pattern), recursive=True)
        if hits:
            return Path(hits[0])
    return None


def elem_count(dims: tuple[int, ...]) -> int:
    out = 1
    for dim in dims:
        out *= int(dim)
    return out


def tensor_bytes(elem_type: int, dims: tuple[int, ...]) -> int:
    if not dims or any(dim <= 0 for dim in dims):
        return 0
    return elem_count(dims) * DTYPE_BYTES.get(elem_type, 4)


def infer_model(path: Path) -> onnx.ModelProto | None:
    try:
        return shape_inference.infer_shapes(onnx.load(path), strict_mode=False)
    except Exception:
        return None


def value_info(model: onnx.ModelProto) -> dict[str, tuple[int, tuple[int, ...]]]:
    out: dict[str, tuple[int, tuple[int, ...]]] = {}
    for vi in list(model.graph.input) + list(model.graph.value_info) + list(model.graph.output):
        if not vi.type.HasField("tensor_type"):
            continue
        tt = vi.type.tensor_type
        dims = tuple(dim.dim_value if dim.HasField("dim_value") else 0 for dim in tt.shape.dim)
        out[vi.name] = (int(tt.elem_type), dims)
    return out


def initializer_stats(model: onnx.ModelProto) -> tuple[int, list[dict[str, Any]]]:
    rows = []
    total = 0
    for init in model.graph.initializer:
        dims = tuple(int(dim) for dim in init.dims)
        elems = int(math.prod(dims)) if dims else 1
        total += elems
        rows.append({"name": init.name, "elems": elems, "dtype": int(init.data_type), "shape": list(dims)})
    rows.sort(key=lambda row: row["elems"], reverse=True)
    return total, rows[:20]


def profile(path: Path) -> dict[str, Any] | None:
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
            node_outputs.append(
                {
                    "node_index": idx,
                    "op": node.op_type,
                    "name": out,
                    "bytes": b,
                    "dtype": elem_type,
                    "shape": list(dims),
                    "inputs": list(node.input),
                }
            )
    init_elems, largest_inits = initializer_stats(model)
    node_outputs.sort(key=lambda row: row["bytes"], reverse=True)
    return {
        "path": rel(path),
        "node_count": len(model.graph.node),
        "initializer_count": len(model.graph.initializer),
        "initializer_elems": init_elems,
        "ops": dict(sorted(ops.items())),
        "memory_static": mem,
        "spatial_bytes": spatial_bytes,
        "index_bytes": index_bytes,
        "wide_bytes": wide_bytes,
        "dtype_bytes": dict(dtype_bytes),
        "largest_initializers": largest_inits,
        "largest_outputs": node_outputs[:25],
    }


def static_cost(path: Path) -> int | None:
    prof = profile(path)
    if prof is None:
        return None
    return int(prof["memory_static"]) + int(prof["initializer_elems"])


def safe_eval(path: Path, task: int) -> dict[str, Any]:
    try:
        return evaluate(str(path), load_task(task), keep_failures=False)
    except Exception as exc:
        return {
            "ok": False,
            "pass": 0,
            "fail": None,
            "memory": None,
            "params": None,
            "points": 0.0,
            "error": str(exc),
        }


def diff_counter(public: dict[str, int], base: dict[str, int]) -> dict[str, int]:
    keys = sorted(set(public) | set(base))
    return {key: public.get(key, 0) - base.get(key, 0) for key in keys if public.get(key, 0) != base.get(key, 0)}


def shape_key(row: dict[str, Any]) -> str:
    return f"{row['op']}:{row['dtype']}:{'x'.join(map(str, row['shape']))}"


def fingerprint_label(row: dict[str, Any]) -> str:
    return f"{shape_key(row)}:{row['bytes']}B"


def lost_big_tensors(base_prof: dict[str, Any], public_prof: dict[str, Any]) -> list[dict[str, Any]]:
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


def gained_big_tensors(base_prof: dict[str, Any], public_prof: dict[str, Any]) -> list[dict[str, Any]]:
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


def output_fingerprints(prof: dict[str, Any], min_bytes: int = 100) -> Counter[str]:
    return Counter(
        fingerprint_label(row)
        for row in prof["largest_outputs"]
        if int(row.get("bytes") or 0) >= min_bytes
    )


def learned_lost_fingerprints(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    fingerprints: dict[str, dict[str, Any]] = {}
    for row in rows:
        for lost in row["lost_big_outputs"]:
            if int(lost.get("bytes") or 0) < 500:
                continue
            key = fingerprint_label(lost)
            item = fingerprints.setdefault(
                key,
                {
                    "fingerprint": key,
                    "examples": [],
                    "count": 0,
                    "total_delta_points": 0.0,
                    "total_saved_cost": 0,
                    "op": lost["op"],
                    "dtype": lost["dtype"],
                    "shape": lost["shape"],
                    "bytes": lost["bytes"],
                },
            )
            item["count"] += 1
            item["total_delta_points"] += float(row["delta_points"])
            item["total_saved_cost"] += -int(row["delta_cost"])
            if len(item["examples"]) < 8:
                item["examples"].append(
                    {
                        "task": row["task"],
                        "dump": row["dump"],
                        "delta_points": row["delta_points"],
                        "delta_cost": row["delta_cost"],
                        "tags": row["tags"],
                    }
                )
    return fingerprints


def classify(base_prof: dict[str, Any], public_prof: dict[str, Any], base_ev: dict[str, Any], public_ev: dict[str, Any]) -> list[str]:
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


def autopsy_task(task: int, base_path: Path, public_path: Path, dump_name: str) -> dict[str, Any] | None:
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
        "task": task,
        "dump": dump_name,
        "public_path": rel(public_path),
        "base_path": rel(base_path),
        "delta_points": float(public_ev["points"]) - float(base_ev["points"]),
        "delta_cost": public_cost - base_cost,
        "base_eval": {key: base_ev.get(key) for key in ("points", "memory", "params", "pass", "fail")},
        "public_eval": {key: public_ev.get(key) for key in ("points", "memory", "params", "pass", "fail")},
        "tags": tags,
        "op_delta_public_minus_base": op_delta,
        "metric_delta_public_minus_base": {
            "nodes": public_prof["node_count"] - base_prof["node_count"],
            "initializer_elems": public_prof["initializer_elems"] - base_prof["initializer_elems"],
            "spatial_bytes": public_prof["spatial_bytes"] - base_prof["spatial_bytes"],
            "index_bytes": public_prof["index_bytes"] - base_prof["index_bytes"],
            "wide_bytes": public_prof["wide_bytes"] - base_prof["wide_bytes"],
            "memory_static": public_prof["memory_static"] - base_prof["memory_static"],
        },
        "lost_big_outputs": lost_big_tensors(base_prof, public_prof),
        "gained_big_outputs": gained_big_tensors(base_prof, public_prof),
        "base_largest_outputs": base_prof["largest_outputs"][:10],
        "public_largest_outputs": public_prof["largest_outputs"][:10],
        "base_largest_initializers": base_prof["largest_initializers"][:10],
        "public_largest_initializers": public_prof["largest_initializers"][:10],
    }


def scan_remaining_active_candidates(base_dir: Path, rows: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    tag_to_tasks: dict[str, list[dict[str, Any]]] = defaultdict(list)
    profiles: dict[int, dict[str, Any]] = {}
    for path in sorted(base_dir.glob("task*.onnx")):
        task = int(path.stem[4:])
        prof = profile(path)
        if prof:
            profiles[task] = prof

    learned_tags = sorted({tag for row in rows for tag in row["tags"]})
    for task, prof in profiles.items():
        ops = Counter(prof["ops"])
        for tag in learned_tags:
            score = 0
            reasons = []
            if tag == "spatial_plane_removed":
                score = prof["spatial_bytes"]
                if score:
                    reasons.append(f"spatial_bytes={score}")
            elif tag == "index_or_topk_plane_removed":
                score = prof["index_bytes"] + 200 * ops.get("TopK", 0)
                if score:
                    reasons.append(f"index_bytes={prof['index_bytes']} TopK={ops.get('TopK', 0)}")
            elif tag == "free_input_einsum_substitution":
                score = prof["wide_bytes"] + prof["spatial_bytes"]
                if score:
                    reasons.append(f"wide+spatial={score}")
            elif tag == "bitpack_or_arithmetic_decode":
                score = prof["wide_bytes"] + 300 * ops.get("ArgMax", 0) + 300 * ops.get("TopK", 0)
                if score:
                    reasons.append(f"wide={prof['wide_bytes']} ArgMax={ops.get('ArgMax', 0)} TopK={ops.get('TopK', 0)}")
            elif tag == "quantized_integer_route":
                score = prof["spatial_bytes"] + 500 * (ops.get("Conv", 0) + ops.get("MatMul", 0))
                if score:
                    reasons.append(f"spatial={prof['spatial_bytes']} Conv={ops.get('Conv', 0)} MatMul={ops.get('MatMul', 0)}")
            elif tag == "final_equal_or_output_only":
                score = prof["wide_bytes"] + 300 * ops.get("Where", 0)
                if score:
                    reasons.append(f"wide={prof['wide_bytes']} Where={ops.get('Where', 0)}")
            elif tag == "params_tail_or_initializer_dedupe":
                score = prof["initializer_elems"]
                if score:
                    reasons.append(f"params={score}")
            else:
                score = prof["wide_bytes"]
                if score:
                    reasons.append(f"wide={score}")
            if score > 0:
                tag_to_tasks[tag].append({"task": task, "score": score, "reasons": reasons, "cost": prof["memory_static"] + prof["initializer_elems"]})
    for tag, candidates in tag_to_tasks.items():
        candidates.sort(key=lambda row: row["score"], reverse=True)
        tag_to_tasks[tag] = candidates[:30]
    return dict(tag_to_tasks)


def scan_fingerprint_candidates(base_dir: Path, rows: list[dict[str, Any]], exclude_win_tasks: bool = True) -> list[dict[str, Any]]:
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
            dp_per_example = meta["total_delta_points"] / max(1, meta["count"])
            contribution = count * (float(meta["bytes"]) + saved_per_example)
            score += contribution
            matched.append(
                {
                    "fingerprint": key,
                    "count": count,
                    "examples": meta["examples"],
                    "learned_count": meta["count"],
                    "learned_total_delta_points": meta["total_delta_points"],
                    "learned_total_saved_cost": meta["total_saved_cost"],
                    "estimated_score": contribution,
                    "estimated_points_hint": count * dp_per_example,
                }
            )
        if matched:
            matched.sort(key=lambda item: item["estimated_score"], reverse=True)
            candidates.append(
                {
                    "task": task,
                    "score": round(score, 3),
                    "cost": prof["memory_static"] + prof["initializer_elems"],
                    "matches": matched[:8],
                }
            )
    candidates.sort(key=lambda row: row["score"], reverse=True)
    return candidates[:80]


def write_markdown(
    rows: list[dict[str, Any]],
    candidates: dict[str, list[dict[str, Any]]],
    fingerprint_candidates: list[dict[str, Any]],
    out: Path,
    base_dir: Path,
) -> None:
    lines = [
        "# Public Win Autopsy",
        "",
        f"- baseline: `{rel(base_dir)}`",
        f"- wins analyzed: `{len(rows)}`",
        "",
    ]
    if not rows:
        lines.extend(["No cheaper bundled-fail=0 public wins found.", ""])
    else:
        by_tag = Counter(tag for row in rows for tag in row["tags"])
        lines.extend(["## Mechanism Clusters", ""])
        for tag, count in by_tag.most_common():
            total_dp = sum(row["delta_points"] for row in rows if tag in row["tags"])
            total_cost = sum(-row["delta_cost"] for row in rows if tag in row["tags"])
            lines.append(f"- `{tag}`: {count} wins, local +{total_dp:.6f}, cost -{total_cost}")
        lines.append("")
        lines.extend(
            [
                "## Wins",
                "",
                "| task | dP | dCost | dump | tags | key op delta | lost big outputs |",
                "|---:|---:|---:|---|---|---|---|",
            ]
        )
        for row in sorted(rows, key=lambda r: r["delta_points"], reverse=True):
            op_delta = ", ".join(f"{k}{v:+d}" for k, v in row["op_delta_public_minus_base"].items() if abs(v) > 0) or "none"
            lost = ", ".join(f"{x['op']}:{x['bytes']}:{x['shape']}" for x in row["lost_big_outputs"][:4]) or "none"
            lines.append(
                f"| {row['task']:03d} | {row['delta_points']:+.6f} | {row['delta_cost']} | "
                f"{row['dump']} | `{','.join(row['tags'])}` | {op_delta} | {lost} |"
            )
        lines.append("")
        learned = learned_lost_fingerprints(rows)
        if learned:
            lines.extend(["## Learned Lost-Tensor Fingerprints", ""])
            lines.append("| fingerprint | wins | saved cost | local dP | examples |")
            lines.append("|---|---:|---:|---:|---|")
            for key, meta in sorted(learned.items(), key=lambda item: (item[1]["total_saved_cost"], item[1]["bytes"]), reverse=True)[:40]:
                examples = ", ".join(f"{ex['task']:03d}" for ex in meta["examples"])
                lines.append(
                    f"| `{key}` | {meta['count']} | {meta['total_saved_cost']} | "
                    f"{meta['total_delta_points']:+.6f} | {examples} |"
                )
            lines.append("")
        if fingerprint_candidates:
            lines.extend(["## Active Candidates By Exact Lost-Tensor Fingerprint", ""])
            lines.append("| rank | task | score | cost | matched fingerprints | learned examples |")
            lines.append("|---:|---:|---:|---:|---|---|")
            for rank, row in enumerate(fingerprint_candidates[:30], 1):
                fps = "<br>".join(f"`{m['fingerprint']}` x{m['count']}" for m in row["matches"][:4])
                examples = "<br>".join(
                    ", ".join(f"{ex['task']:03d}" for ex in m["examples"][:5])
                    for m in row["matches"][:4]
                )
                lines.append(f"| {rank} | {row['task']:03d} | {row['score']} | {row['cost']} | {fps} | {examples} |")
            lines.append("")
        lines.extend(["## Rescan Candidates By Learned Tag", ""])
        for tag, task_rows in sorted(candidates.items()):
            if not task_rows:
                continue
            lines.append(f"### `{tag}`")
            lines.append("")
            lines.append("| rank | task | score | cost | reasons |")
            lines.append("|---:|---:|---:|---:|---|")
            for rank, row in enumerate(task_rows[:20], 1):
                lines.append(f"| {rank} | {row['task']:03d} | {row['score']} | {row['cost']} | {'; '.join(row['reasons'])} |")
            lines.append("")
    out.write_text("\n".join(lines))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("dumps", nargs="+", help="Public dump dirs containing taskNNN.onnx")
    parser.add_argument("--base-dir", default=str(DEFAULT_BASE), help="Baseline net dir")
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT), help="Report directory")
    parser.add_argument("--margin", type=int, default=0, help="Static cost margin for candidate prefilter")
    parser.add_argument("--tasks", nargs="*", type=int, help="Optional task subset")
    args = parser.parse_args()

    base_dir = Path(args.base_dir).resolve()
    dump_dirs = [Path(d).resolve() for d in args.dumps]
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    tasks = args.tasks if args.tasks else list(range(1, 401))
    for task in tasks:
        base_path = base_dir / f"task{task:03d}.onnx"
        if not base_path.exists():
            continue
        base_static = static_cost(base_path)
        if base_static is None:
            continue
        best: dict[str, Any] | None = None
        for dump_dir in dump_dirs:
            public_path = find_net(dump_dir, task)
            if public_path is None:
                continue
            public_static = static_cost(public_path)
            if public_static is None or public_static >= base_static - args.margin:
                continue
            row = autopsy_task(task, base_path, public_path, dump_dir.name)
            if row is None:
                continue
            if best is None or row["delta_points"] > best["delta_points"]:
                best = row
        if best:
            rows.append(best)
            print(
                f"WIN task{task:03d} {best['delta_points']:+.6f} cost {best['delta_cost']} "
                f"tags={','.join(best['tags'])} dump={best['dump']}",
                flush=True,
            )

    rows.sort(key=lambda row: row["delta_points"], reverse=True)
    candidates = scan_remaining_active_candidates(base_dir, rows)
    fingerprint_candidates = scan_fingerprint_candidates(base_dir, rows)
    json_out = out_dir / "public_win_autopsy.json"
    md_out = out_dir / "public_win_autopsy.md"
    json_out.write_text(
        json.dumps(
            {
                "base_dir": rel(base_dir),
                "wins": rows,
                "learned_lost_fingerprints": learned_lost_fingerprints(rows),
                "fingerprint_rescan_candidates": fingerprint_candidates,
                "rescan_candidates": candidates,
            },
            indent=2,
            sort_keys=True,
        )
    )
    write_markdown(rows, candidates, fingerprint_candidates, md_out, base_dir)
    print(f"wrote {rel(json_out)}")
    print(f"wrote {rel(md_out)}")


if __name__ == "__main__":
    main()
