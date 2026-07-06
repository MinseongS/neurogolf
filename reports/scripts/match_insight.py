#!/usr/bin/env python3
"""Query the task index: given an insight's applicability predicate (or a saved --mechanism
query), return floored-excluded, bloat-ranked candidate tasks. Emits leads only."""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path

REPORTS = Path(__file__).resolve().parents[1]
INDEX_PATH = REPORTS / "task_index.json"
sys.path.append(str(Path(__file__).resolve().parent))
import coverage_lib  # noqa: E402


def flatten(row: dict) -> dict:
    e = row.get("economic", {}); s = row.get("structural", {}); sem = row.get("semantic", {})
    flat = {"cost": e.get("cost", 0), "bloat": e.get("bloat", 0), "mem": e.get("mem", 0),
            "params": e.get("params", 0), "status": e.get("status", ""), "method": e.get("method", ""),
            "ops": s.get("ops", []), "node_count": s.get("node_count", 0)}
    flat.update({k: v for k, v in s.get("flags", {}).items()})
    for k, cell in sem.items():
        if k in ("probe_version", "n_samples"):
            continue
        val = cell.get("value") if isinstance(cell, dict) else cell
        if k == "separable_rect_output" and isinstance(val, dict):
            flat["separable_rect_is"] = val.get("is"); flat["separable_rect_n"] = val.get("n_rects")
        elif k == "delta" and isinstance(val, dict):
            flat["copy_frac"] = val.get("copy_frac"); flat["changed_cells"] = val.get("changed_cells")
        else:
            flat[k] = val
    return flat


def match(where: str, mechanism=None, include_resolved: bool = False):
    idx = json.loads(INDEX_PATH.read_text())
    drop = set() if include_resolved or not mechanism else coverage_lib.resolved_tasks(mechanism)
    hits = []
    for num, row in idx.items():
        if num in drop:
            continue
        flat = flatten(row)
        try:
            if eval(where, {"__builtins__": {}}, flat):  # predicate over flat features
                hits.append((num, flat))
        except Exception:
            continue
    hits.sort(key=lambda x: x[1].get("bloat", 0), reverse=True)
    return hits


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--where"); ap.add_argument("--mechanism"); ap.add_argument("--all", action="store_true")
    ap.add_argument("--top", type=int, default=30); ap.add_argument("--emit-queue")
    a = ap.parse_args()
    where = a.where
    if a.mechanism and not where:
        data = coverage_lib.load()
        where = data.get(a.mechanism, {}).get("query", "")
        if not where:
            print(f"no saved query for mechanism {a.mechanism}; pass --where"); return
    res = match(where, a.mechanism, a.all)[: a.top]
    for num, flat in res:
        print(f"task{num:>3}  bloat={flat.get('bloat'):>7}  cost={flat.get('cost'):>7}  "
              f"{flat.get('shape_relation','?')}/{flat.get('color_source','?')}  ops={','.join(flat.get('ops', [])[:6])}")
    if a.emit_queue:
        Path(a.emit_queue).write_text("\n".join(n for n, _ in res))
        print(f"\nqueue -> {a.emit_queue} ({len(res)} tasks)")


if __name__ == "__main__":
    main()
