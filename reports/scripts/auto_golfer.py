#!/usr/bin/env python3
"""Auto-golfer orchestration for active 8000-mode NeuroGolf overlays.

This script is intentionally conservative:
  1. run active-wide graph-surgery probes that already verify bundled fail=0;
  2. parse their win lines and write a machine-readable candidate report;
  3. rank remaining tasks by counted-tensor archetypes for semantic/compiler work.

It does not adopt candidates by itself. Adoption still means copying the chosen
candidate to submission/overfit_nets/taskNNN.onnx, then rebuilding the overfit
manifest, unsigned-TopK scan, and submission.zip.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import subprocess
import sys
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import onnx


ROOT = Path(__file__).resolve().parents[2]
ACTIVE = ROOT / "submission" / "overfit_nets"
REPORT_JSON = ROOT / "reports" / "auto_golfer_report.json"
REPORT_MD = ROOT / "reports" / "auto_golfer_report.md"
TASKLOG = ROOT / "reports" / "tasklog"


PROBES = [
    "reports/candidates/dedupe_initializers_active_probe.py",
    "reports/candidates/prune_dead_constants_active_probe.py",
    "reports/candidates/noop_reshape_active_probe.py",
    "reports/candidates/dynamic_cse_active_probe.py",
    "reports/candidates/cast_dtype_batch_probe.py",
    "reports/candidates/zero_concat_to_pad_active_probe.py",
    "reports/candidates/contiguous_gather_to_slice_active_probe.py",
    "reports/candidates/negative_pad_normalize_probe.py",
    "reports/candidates/sign_argmax_uint8_active_probe.py",
    "reports/candidates/defer_widening_cast_shape_probe.py",
    "reports/candidates/remove_gather_index_cast_probe.py",
]

WIN_RE = re.compile(
    r"^\+(?P<delta>[0-9.]+)\s+task(?P<task>[0-9]{3})\s+"
    r"cost\s+(?P<old>[0-9]+)->(?P<new>[0-9]+).*?(?P<path>reports/\S+\.onnx)"
)


def run_probe(path: str, timeout: int) -> dict[str, object]:
    cmd = [sys.executable, path]
    started = time.time()
    proc = subprocess.run(
        cmd,
        cwd=ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        timeout=timeout,
        env={**os.environ, "PYTHONPATH": "."},
    )
    output = proc.stdout
    wins = []
    for line in output.splitlines():
        m = WIN_RE.match(line.strip())
        if not m:
            continue
        wins.append(
            {
                "task": int(m.group("task")),
                "delta_points": float(m.group("delta")),
                "old_cost": int(m.group("old")),
                "new_cost": int(m.group("new")),
                "path": m.group("path"),
                "line": line.strip(),
            }
        )
    return {
        "probe": path,
        "returncode": proc.returncode,
        "seconds": round(time.time() - started, 3),
        "wins": wins,
        "stdout_tail": "\n".join(output.splitlines()[-30:]),
    }


def dtype_size(elem_type: int) -> int:
    try:
        return int(np.dtype(onnx.helper.tensor_dtype_to_np_dtype(elem_type)).itemsize)
    except Exception:
        return 4


def dims_of(vi: onnx.ValueInfoProto) -> list[int] | None:
    tt = vi.type.tensor_type
    if not tt.HasField("shape"):
        return None
    dims = []
    for dim in tt.shape.dim:
        if not dim.HasField("dim_value") or dim.dim_value <= 0:
            return None
        dims.append(int(dim.dim_value))
    return dims


def tensor_bytes(vi: onnx.ValueInfoProto) -> int:
    dims = dims_of(vi)
    if not dims:
        return 0
    return int(math.prod(dims)) * dtype_size(int(vi.type.tensor_type.elem_type))


def spatial_dim(dims: list[int]) -> int:
    best = 0
    for dim in dims:
        if 11 <= dim <= 30:
            best = max(best, dim)
        elif dim > 30:
            side = math.isqrt(dim)
            if side * side == dim and 11 <= side <= 30:
                best = max(best, side)
    return best


def classify_tensor(
    name: str,
    bytes_: int,
    dims: list[int],
    producer: str,
    consumers: list[str],
    output_inputs: set[str],
) -> list[str]:
    tags = []
    if name in output_inputs or consumers == ["Where"] or consumers == ["Equal"]:
        tags.append("final_output_welded")
    if spatial_dim(dims) >= 28 and bytes_ >= 900:
        tags.append("full_canvas")
    if producer in {"Conv", "ReduceSum", "ReduceMax", "Einsum"} and bytes_ >= 900:
        tags.append("producer_bound_candidate")
    if producer == "Cast" and bytes_ >= 900:
        tags.append("dtype_candidate")
    if producer in {"Pad", "Concat"} and bytes_ >= 900:
        tags.append("pad_concat_carrier")
    if any(c in {"TopK"} for c in consumers):
        tags.append("topk_feed")
    if any(c in {"ScatterND", "ScatterElements"} for c in consumers):
        tags.append("scatter_feed")
    if any(c in {"ArgMax", "ArgMin", "Gather", "GatherElements"} for c in consumers):
        tags.append("index_or_profile")
    return tags or ["other"]


def active_manifest_costs() -> dict[int, dict[str, object]]:
    path = ROOT / "reports" / "overfit_manifest.json"
    if not path.exists():
        return {}
    data = json.loads(path.read_text())
    out = {}
    for row in data.get("tasks", []):
        task = int(row["task"])
        out[task] = row
    return out


def tasklog_penalty(task: int) -> tuple[float, list[str]]:
    path = TASKLOG / f"task{task:03d}.md"
    if not path.exists():
        return 0.0, []
    text = path.read_text(errors="ignore").lower()
    reasons = []
    penalty = 0.0
    strong_markers = [
        "do not re-probe",
        "do not retry",
        "do not reattempt",
        "재탐사 금지",
        " kill",
        " kill\n",
        "at-floor",
        "floor re-confirmed",
        "verdict: floor",
        "confirmed near-floor",
        "irreducible-floor",
        "floor (",
    ]
    for marker in strong_markers:
        if marker in text:
            penalty += 3500.0
            reasons.append(marker)
    if "info-bottleneck" in text or "ambiguity wall" in text:
        penalty += 5000.0
        reasons.append("ambiguity/info wall")
    if "open angles" in text or "open_angle" in text:
        penalty -= 800.0
        reasons.append("open")
    if "not floor" in text or "floor broken" in text:
        penalty -= 2000.0
        reasons.append("not-floor-signal")
    return max(penalty, 0.0), reasons[:5]


def archetype_scan(limit: int) -> list[dict[str, object]]:
    manifest = active_manifest_costs()
    rows = []
    for task in range(1, 401):
        path = ACTIVE / f"task{task:03d}.onnx"
        if not path.exists():
            continue
        try:
            model = onnx.shape_inference.infer_shapes(onnx.load(path), strict_mode=False)
        except Exception as exc:
            rows.append({"task": task, "error": repr(exc)})
            continue
        graph = model.graph
        producer = {}
        consumers: dict[str, list[str]] = defaultdict(list)
        output_inputs = set()
        for node in graph.node:
            for out in node.output:
                if out:
                    producer[out] = node.op_type
            for inp in node.input:
                if inp:
                    consumers[inp].append(node.op_type)
            if "output" in node.output:
                output_inputs.update(inp for inp in node.input if inp)
        init_names = {i.name for i in graph.initializer}
        tensors = []
        for vi in list(graph.value_info) + list(graph.output):
            name = vi.name
            if name in {"input", "output"} or name in init_names:
                continue
            if not vi.type.HasField("tensor_type"):
                continue
            dims = dims_of(vi)
            if not dims:
                continue
            bytes_ = tensor_bytes(vi)
            if bytes_ <= 0:
                continue
            prod = producer.get(name, "value")
            cons = consumers.get(name, [])
            tags = classify_tensor(name, bytes_, dims, prod, cons, output_inputs)
            tensors.append(
                {
                    "name": name,
                    "bytes": bytes_,
                    "shape": dims,
                    "producer": prod,
                    "consumers": cons,
                    "tags": tags,
                }
            )
        tensors.sort(key=lambda r: int(r["bytes"]), reverse=True)
        tag_bytes: dict[str, int] = defaultdict(int)
        for t in tensors:
            for tag in t["tags"]:
                tag_bytes[tag] += int(t["bytes"])
        cost_row = manifest.get(task, {})
        cost = int(cost_row.get("memory") or 0) + int(cost_row.get("params") or 0)
        raw_score = (
            tag_bytes.get("producer_bound_candidate", 0) * 1.0
            + tag_bytes.get("pad_concat_carrier", 0) * 0.8
            + tag_bytes.get("dtype_candidate", 0) * 0.6
            + tag_bytes.get("index_or_profile", 0) * 0.5
            - tag_bytes.get("final_output_welded", 0) * 0.25
        )
        penalty, penalty_reasons = tasklog_penalty(task)
        score = raw_score - penalty
        rows.append(
            {
                "task": task,
                "cost": cost,
                "points": cost_row.get("points"),
                "score": round(score, 1),
                "raw_score": round(raw_score, 1),
                "tasklog_penalty": round(penalty, 1),
                "tasklog_penalty_reasons": penalty_reasons,
                "tag_bytes": dict(sorted(tag_bytes.items(), key=lambda kv: kv[1], reverse=True)),
                "top_tensors": tensors[:8],
            }
        )
    rows.sort(key=lambda r: (float(r.get("score") or 0), int(r.get("cost") or 0)), reverse=True)
    return rows[:limit]


def write_report(results: list[dict[str, object]], archetypes: list[dict[str, object]]) -> None:
    all_wins = []
    for result in results:
        all_wins.extend(result["wins"])
    all_wins.sort(key=lambda r: float(r["delta_points"]), reverse=True)
    payload = {
        "generated_at_unix": int(time.time()),
        "probe_results": results,
        "wins": all_wins,
        "archetype_rank": archetypes,
    }
    REPORT_JSON.write_text(json.dumps(payload, indent=2))

    lines = ["# Auto Golfer Report", ""]
    lines.append("## Probe Wins")
    if all_wins:
        lines.append("| delta | task | cost | candidate |")
        lines.append("|---:|---:|---:|---|")
        for win in all_wins:
            lines.append(
                f"| +{win['delta_points']:.6f} | {win['task']:03d} | "
                f"{win['old_cost']}->{win['new_cost']} | `{win['path']}` |"
            )
    else:
        lines.append("No bundled-fail=0 lower-cost graph-surgery wins found.")
    lines.append("")
    lines.append("## Probe Summary")
    lines.append("| probe | seconds | wins | returncode |")
    lines.append("|---|---:|---:|---:|")
    for result in results:
        lines.append(
            f"| `{result['probe']}` | {result['seconds']} | "
            f"{len(result['wins'])} | {result['returncode']} |"
        )
    lines.append("")
    lines.append("## Semantic Compiler Targets")
    lines.append("| rank | task | score | cost | penalty | dominant tags | top tensors |")
    lines.append("|---:|---:|---:|---:|---:|---|---|")
    for i, row in enumerate(archetypes[:30], 1):
        tag_text = ", ".join(f"{k}:{v}" for k, v in list(row["tag_bytes"].items())[:4])
        tensor_text = "; ".join(
            f"{t['name']}:{t['producer']}:{t['bytes']}:{'/'.join(t['tags'][:2])}"
            for t in row["top_tensors"][:4]
        )
        lines.append(
            f"| {i} | {row['task']:03d} | {row['score']} | {row['cost']} | "
            f"{row.get('tasklog_penalty', 0)} | {tag_text} | `{tensor_text}` |"
        )
    REPORT_MD.write_text("\n".join(lines) + "\n")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-probes", action="store_true")
    ap.add_argument("--timeout", type=int, default=240)
    ap.add_argument("--archetype-limit", type=int, default=80)
    args = ap.parse_args()

    results = []
    if not args.skip_probes:
        for probe in PROBES:
            print(f"== probe {probe}")
            try:
                result = run_probe(probe, args.timeout)
            except subprocess.TimeoutExpired as exc:
                result = {
                    "probe": probe,
                    "returncode": None,
                    "seconds": args.timeout,
                    "wins": [],
                    "stdout_tail": f"TIMEOUT: {exc}",
                }
            print(result["stdout_tail"])
            results.append(result)
    archetypes = archetype_scan(args.archetype_limit)
    write_report(results, archetypes)
    print(f"wrote {REPORT_JSON}")
    print(f"wrote {REPORT_MD}")


if __name__ == "__main__":
    main()
