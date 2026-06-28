#!/usr/bin/env python3
"""Scan public ONNX teacher candidates without adopting them.

Public candidates are teacher artifacts.  This script compares them against the
current live network and local source builder, then writes a report for deciding
which mechanisms to extract.
"""

from __future__ import annotations

import argparse
import importlib
import json
import math
import re
from collections import Counter
from pathlib import Path
from typing import Any

import onnx

from src.harness import evaluate, load_task


ROOT = Path(__file__).resolve().parents[2]
REPORTS = ROOT / "reports"
NETWORKS = ROOT / "networks"
PUBLIC = ROOT / "public_candidates"


def infer_task(path: Path) -> int | None:
    text = str(path)
    patterns = [
        r"task[_-]?(\d{1,3})",
        r"(?:^|[^\d])(\d{3})(?:[^\d]|$)",
        r"(?:^|[^\d])(\d{1,2})(?:[^\d]|$)",
    ]
    for pattern in patterns:
        match = re.search(pattern, text, flags=re.I)
        if match:
            task = int(match.group(1))
            if 1 <= task <= 400:
                return task
    return None


def op_counts(path: Path) -> dict[str, int]:
    try:
        model = onnx.load(path)
    except Exception:
        return {}
    return dict(sorted(Counter(node.op_type for node in model.graph.node).items()))


def initializer_elems(path: Path) -> int | None:
    try:
        model = onnx.load(path)
    except Exception:
        return None
    total = 0
    for init in model.graph.initializer:
        total += math.prod(init.dims) if init.dims else 1
    return total


def safe_eval(model_or_path: Any, task_num: int) -> dict[str, Any]:
    try:
        return evaluate(model_or_path, load_task(task_num), keep_failures=False)
    except Exception as exc:
        return {
            "ok": False,
            "filesize": None,
            "memory": None,
            "params": None,
            "points": 0.0,
            "pass": 0,
            "fail": None,
            "error": str(exc),
        }


def source_eval(task_num: int) -> dict[str, Any]:
    try:
        mod = importlib.import_module(f"src.custom.task{task_num:03d}")
        importlib.reload(mod)
        return safe_eval(mod.build(load_task(task_num)), task_num)
    except Exception as exc:
        return {
            "ok": False,
            "filesize": None,
            "memory": None,
            "params": None,
            "points": 0.0,
            "pass": 0,
            "fail": None,
            "error": str(exc),
        }


def delta(a: dict[str, Any], b: dict[str, Any], key: str) -> float | None:
    av = a.get(key)
    bv = b.get(key)
    if av is None or bv is None:
        return None
    return float(av) - float(bv)


def op_delta(public_ops: dict[str, int], live_ops: dict[str, int]) -> dict[str, int]:
    keys = sorted(set(public_ops) | set(live_ops))
    return {key: public_ops.get(key, 0) - live_ops.get(key, 0) for key in keys if public_ops.get(key, 0) != live_ops.get(key, 0)}


def flags_for(public_ev: dict[str, Any], live_ev: dict[str, Any], src_ev: dict[str, Any], public_ops: dict[str, int], live_ops: dict[str, int]) -> list[str]:
    flags: list[str] = []
    if not public_ev.get("ok"):
        flags.append("public_fails_stored")
        return flags
    if public_ev.get("points", 0.0) > live_ev.get("points", 0.0) + 1e-9:
        flags.append("score_up")
    if (public_ev.get("memory") is not None and live_ev.get("memory") is not None and public_ev["memory"] < live_ev["memory"]):
        flags.append("memory_down")
    if (public_ev.get("params") is not None and live_ev.get("params") is not None and public_ev["params"] < live_ev["params"]):
        flags.append("params_down")
    if src_ev.get("points", 0.0) + 1e-9 < live_ev.get("points", 0.0):
        flags.append("source_lags_live")
    if src_ev.get("points", 0.0) > live_ev.get("points", 0.0) + 1e-9:
        flags.append("source_ahead_live")
    public_has_qlinear = public_ops.get("QLinearConv", 0) + public_ops.get("QLinearMatMul", 0)
    live_has_qlinear = live_ops.get("QLinearConv", 0) + live_ops.get("QLinearMatMul", 0)
    if public_has_qlinear > live_has_qlinear:
        flags.append("new_qlinear_mechanism")
    if public_ops.get("Equal", 0) > live_ops.get("Equal", 0) and public_ev.get("memory", 10**9) <= live_ev.get("memory", 0):
        flags.append("possible_final_equal_route")
    if {"memory_down", "params_down"}.intersection(flags) and "score_up" not in flags:
        flags.append("mechanism_teacher")
    return sorted(set(flags))


def recommendation(flags: list[str]) -> str:
    if "public_fails_stored" in flags:
        return "reject_or_debug_public_candidate"
    if "score_up" in flags:
        return "extract_source_then_consider_adopt"
    if "mechanism_teacher" in flags or "new_qlinear_mechanism" in flags or "possible_final_equal_route" in flags:
        return "extract_mechanism_before_any_adopt"
    if "source_ahead_live" in flags:
        return "verify_source_candidate_before_public"
    if "source_lags_live" in flags:
        return "repair_source_ownership_from_live_or_teacher"
    return "archive_low_priority"


def scan(public_dir: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in sorted(public_dir.rglob("*.onnx")):
        task_num = infer_task(path)
        if task_num is None:
            rows.append({"candidate_path": str(path), "task": None, "error": "could_not_infer_task"})
            continue
        live_path = NETWORKS / f"task{task_num:03d}.onnx"
        public_ev = safe_eval(str(path), task_num)
        live_ev = safe_eval(str(live_path), task_num) if live_path.exists() else {"ok": False, "points": 0.0}
        src_ev = source_eval(task_num)
        public_ops = op_counts(path)
        live_ops = op_counts(live_path) if live_path.exists() else {}
        flags = flags_for(public_ev, live_ev, src_ev, public_ops, live_ops)
        rows.append(
            {
                "task": task_num,
                "candidate_path": str(path),
                "public_eval": public_ev,
                "live_eval": live_ev,
                "source_eval": src_ev,
                "delta_vs_live": {
                    "points": delta(public_ev, live_ev, "points"),
                    "memory": delta(public_ev, live_ev, "memory"),
                    "params": delta(public_ev, live_ev, "params"),
                },
                "delta_source_vs_live": {
                    "points": delta(src_ev, live_ev, "points"),
                    "memory": delta(src_ev, live_ev, "memory"),
                    "params": delta(src_ev, live_ev, "params"),
                },
                "public_ops": public_ops,
                "live_ops": live_ops,
                "op_delta_public_minus_live": op_delta(public_ops, live_ops),
                "public_initializer_elems": initializer_elems(path),
                "flags": flags,
                "recommendation": recommendation(flags),
            }
        )
    rows.sort(key=lambda row: (row.get("task") is None, row.get("task") or 999, row.get("candidate_path", "")))
    return rows


def write_markdown(rows: list[dict[str, Any]], path: Path) -> None:
    lines = [
        "# Public teacher report",
        "",
        "Public ONNX files are teacher artifacts.  Do not adopt blindly; extract source and mechanisms first.",
        "",
        f"- candidates scanned: {len(rows)}",
        "",
    ]
    if not rows:
        lines.extend(["No public candidates found.", ""])
    else:
        lines.extend(
            [
                "| task | public pts | live pts | src pts | Δpts | Δmem | Δparams | flags | recommendation | path |",
                "|---:|---:|---:|---:|---:|---:|---:|---|---|---|",
            ]
        )
        for row in rows:
            if row.get("task") is None:
                lines.append(f"| — | | | | | | | error | {row.get('error')} | `{row.get('candidate_path')}` |")
                continue
            pe = row["public_eval"]
            le = row["live_eval"]
            se = row["source_eval"]
            dv = row["delta_vs_live"]
            flags = ",".join(row.get("flags", []))
            lines.append(
                f"| {row['task']:03d} | {float(pe.get('points') or 0):.3f} | {float(le.get('points') or 0):.3f} | "
                f"{float(se.get('points') or 0):.3f} | {float(dv.get('points') or 0):+.3f} | "
                f"{int(dv['memory']) if dv.get('memory') is not None else ''} | "
                f"{int(dv['params']) if dv.get('params') is not None else ''} | "
                f"{flags} | {row.get('recommendation')} | `{row.get('candidate_path')}` |"
            )
        lines.append("")
    path.write_text("\n".join(lines))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--public-dir", default=str(PUBLIC))
    parser.add_argument("--out-json", default=str(REPORTS / "public_teacher_report.json"))
    parser.add_argument("--out-md", default=str(REPORTS / "public_teacher_report.md"))
    args = parser.parse_args()
    public_dir = Path(args.public_dir)
    public_dir.mkdir(parents=True, exist_ok=True)
    rows = scan(public_dir)
    Path(args.out_json).write_text(json.dumps(rows, indent=2, sort_keys=True, default=str))
    write_markdown(rows, Path(args.out_md))
    print(f"wrote {args.out_json}")
    print(f"wrote {args.out_md}")
    print(f"candidates: {len(rows)}")


if __name__ == "__main__":
    main()
