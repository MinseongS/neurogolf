#!/usr/bin/env python3
"""Create non-mutating source and insight drafts from a public teacher ONNX."""

from __future__ import annotations

import argparse
import base64
import gzip
import json
import math
import textwrap
from collections import Counter
from pathlib import Path
from typing import Any

import onnx

from src.harness import evaluate, load_task


ROOT = Path(__file__).resolve().parents[2]
REPORTS = ROOT / "reports"
NETWORKS = ROOT / "networks"
SOURCE_DRAFTS = REPORTS / "public_teacher_sources"
INSIGHT_DRAFTS = REPORTS / "public_teacher_insights"


def safe_eval(path: Path, task_num: int) -> dict[str, Any]:
    try:
        return evaluate(str(path), load_task(task_num), keep_failures=False)
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


def model_stats(path: Path) -> dict[str, Any]:
    model = onnx.load(path)
    init_elems = 0
    largest = []
    for init in model.graph.initializer:
        elems = math.prod(init.dims) if init.dims else 1
        init_elems += elems
        largest.append((elems, init.name, init.data_type, list(init.dims)))
    largest.sort(reverse=True)
    return {
        "node_count": len(model.graph.node),
        "ops": dict(sorted(Counter(node.op_type for node in model.graph.node).items())),
        "initializer_elems": init_elems,
        "largest_initializers": [
            {"name": name, "elems": elems, "dtype": dtype, "shape": shape}
            for elems, name, dtype, shape in largest[:12]
        ],
        "opset": [{"domain": op.domain, "version": op.version} for op in model.opset_import],
    }


def op_delta(public_stats: dict[str, Any], live_stats: dict[str, Any]) -> dict[str, int]:
    pub = public_stats.get("ops", {})
    live = live_stats.get("ops", {})
    keys = sorted(set(pub) | set(live))
    return {key: pub.get(key, 0) - live.get(key, 0) for key in keys if pub.get(key, 0) != live.get(key, 0)}


def source_payload(path: Path) -> str:
    raw = path.read_bytes()
    payload = base64.b64encode(gzip.compress(raw, compresslevel=9)).decode()
    chunks = "\n".join(f"    {chunk!r}" for chunk in textwrap.wrap(payload, 96))
    return chunks


def write_source_draft(task_num: int, public_path: Path, out_path: Path) -> None:
    chunks = source_payload(public_path)
    text = f'''"""Task {task_num:03d} public-teacher exact source draft.

Generated from `{public_path}` by `reports/scripts/public_teacher_extract.py`.
This draft is intentionally non-installed.  Review the mechanism, convert to a
semantic builder when possible, or install explicitly only after fresh/source
gates pass.
"""

from __future__ import annotations

import base64
import gzip

import onnx


_PAYLOAD = (
{chunks}
)


def build(task):
    return onnx.load_from_string(gzip.decompress(base64.b64decode(_PAYLOAD)))
'''
    out_path.write_text(text)


def write_insight_note(task_num: int, public_path: Path, out_path: Path) -> None:
    live_path = NETWORKS / f"task{task_num:03d}.onnx"
    public_ev = safe_eval(public_path, task_num)
    live_ev = safe_eval(live_path, task_num) if live_path.exists() else {"ok": False, "points": 0.0}
    public_stats = model_stats(public_path)
    live_stats = model_stats(live_path) if live_path.exists() else {"ops": {}, "node_count": 0, "initializer_elems": 0}
    delta = {
        "points": float(public_ev.get("points") or 0) - float(live_ev.get("points") or 0),
        "memory": None if public_ev.get("memory") is None or live_ev.get("memory") is None else int(public_ev["memory"]) - int(live_ev["memory"]),
        "params": None if public_ev.get("params") is None or live_ev.get("params") is None else int(public_ev["params"]) - int(live_ev["params"]),
    }
    lines = [
        f"# Public teacher extraction — task{task_num:03d}",
        "",
        f"- public path: `{public_path}`",
        f"- live path: `{live_path}`",
        "",
        "## Stored evaluation",
        "",
        f"- public: ok={public_ev.get('ok')} pts={float(public_ev.get('points') or 0):.6f} mem={public_ev.get('memory')} params={public_ev.get('params')}",
        f"- live: ok={live_ev.get('ok')} pts={float(live_ev.get('points') or 0):.6f} mem={live_ev.get('memory')} params={live_ev.get('params')}",
        f"- delta public-live: pts={delta['points']:+.6f} mem={delta['memory']} params={delta['params']}",
        "",
        "## Structural comparison",
        "",
        f"- public nodes: {public_stats['node_count']}",
        f"- live nodes: {live_stats.get('node_count')}",
        f"- public initializer elems: {public_stats['initializer_elems']}",
        f"- live initializer elems: {live_stats.get('initializer_elems')}",
        "",
        "### Op delta public-live",
        "",
    ]
    od = op_delta(public_stats, live_stats)
    if od:
        for op, count in od.items():
            lines.append(f"- `{op}`: {count:+d}")
    else:
        lines.append("- no op-count delta")
    lines.extend(
        [
            "",
            "## Mechanism extraction checklist",
            "",
            "- [ ] Identify which full-grid intermediate disappeared or changed dtype.",
            "- [ ] Identify whether QLinear/uint8/bool/opset routing changed.",
            "- [ ] Identify whether final one-hot expansion moved to final `Equal`.",
            "- [ ] Decide whether this is a semantic mechanism or exact-preserve only.",
            "- [ ] If reusable, add/update `reports/insight_registry.yaml` and rerun the recursive queue.",
            "",
            "## Adoption status",
            "",
            "Not adopted. This is a teacher artifact until source and fresh gates pass.",
            "",
        ]
    )
    out_path.write_text("\n".join(lines))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("task", type=int)
    parser.add_argument("public_onnx")
    parser.add_argument("--source-out-dir", default=str(SOURCE_DRAFTS))
    parser.add_argument("--insight-out-dir", default=str(INSIGHT_DRAFTS))
    args = parser.parse_args()
    task_num = args.task
    public_path = Path(args.public_onnx)
    if not public_path.exists():
        raise SystemExit(f"public ONNX not found: {public_path}")
    if not (1 <= task_num <= 400):
        raise SystemExit(f"task must be 1..400: {task_num}")
    SOURCE_DRAFTS.mkdir(parents=True, exist_ok=True)
    INSIGHT_DRAFTS.mkdir(parents=True, exist_ok=True)
    source_out = Path(args.source_out_dir) / f"task{task_num:03d}_public_exact.py"
    insight_out = Path(args.insight_out_dir) / f"task{task_num:03d}_public_teacher.md"
    source_out.parent.mkdir(parents=True, exist_ok=True)
    insight_out.parent.mkdir(parents=True, exist_ok=True)
    write_source_draft(task_num, public_path, source_out)
    write_insight_note(task_num, public_path, insight_out)
    print(f"wrote {source_out}")
    print(f"wrote {insight_out}")


if __name__ == "__main__":
    main()
