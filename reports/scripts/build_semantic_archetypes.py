#!/usr/bin/env python3
"""Build a semantic-archetype map for NeuroGolf strategy work.

This report is deliberately different from the layer inventory:

- layer inventory answers "what ONNX ops exist?"
- this report answers "which task families can be rewritten by a reusable
  semantic compiler?"

The classification is heuristic by design.  It ranks candidates for manual
deep-dives; it does not adopt or mutate networks.
"""

from __future__ import annotations

import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path

import onnx


ROOT = Path(__file__).resolve().parents[2]
REPORTS = ROOT / "reports"
CUSTOM = ROOT / "src" / "custom"
NETWORKS = ROOT / "networks"
ARC_MAP = REPORTS / "arc_mapping.json"
MANIFEST = REPORTS / "manifest.json"
INVENTORY = REPORTS / "global_layer_inventory.json"


def read_text(path: Path) -> str:
    try:
        return path.read_text(errors="ignore")
    except FileNotFoundError:
        return ""


def load_json(path: Path, default):
    try:
        return json.loads(path.read_text())
    except FileNotFoundError:
        return default


def generator_text(task: int, arc_map: dict) -> str:
    rec = arc_map.get(str(task)) or {}
    p = rec.get("generator")
    if not p:
        arc_id = rec.get("arc_id")
        if arc_id:
            p = f"/tmp/arc-gen/tasks/task_{arc_id}.py"
    return read_text(Path(p)) if p else ""


def source_class(source_text: str) -> str:
    lower = source_text.lower()
    if not source_text:
        return "no_source"
    if "exact source reconstruction" in lower or "_payload" in lower and "gzip" in lower:
        return "exact_preserve_payload"
    if "heuristic" in lower:
        return "heuristic_source"
    if "rule" in lower or "generator" in lower or "verified" in lower:
        return "semantic_source"
    return "unknown_source"


def op_counts(task: int) -> Counter:
    try:
        m = onnx.load(NETWORKS / f"task{task:03d}.onnx")
    except Exception:
        return Counter()
    return Counter(n.op_type for n in m.graph.node)


def example_shape_stats(task_data: dict) -> dict:
    rows = []
    cols = []
    outs = []
    colors = Counter()
    for split in ("train", "test"):
        for ex in task_data.get(split, []):
            grid = ex.get("input") or []
            if grid:
                rows.append(len(grid))
                cols.append(len(grid[0]))
                for row in grid:
                    colors.update(row)
            out = ex.get("output") or []
            if out:
                outs.append((len(out), len(out[0])))
    return {
        "input_min": [min(rows), min(cols)] if rows and cols else None,
        "input_max": [max(rows), max(cols)] if rows and cols else None,
        "output_shapes": sorted(set(outs)),
        "colors": sorted(k for k, v in colors.items() if v),
    }


PATTERNS = {
    "op_direct_or_color_lut": [
        "random_color",
        "random_colors",
        "color",
        "colors",
        "exclude",
    ],
    "crop_or_bbox": [
        "output = common.grid(wide, tall",
        "output = common.grid(width, height",
        "brow",
        "bcol",
        "crop",
        "top left",
        "bbox",
    ],
    "scale_or_kron": [
        "factor",
        "upscale",
        "scaled",
        "magnified",
        "* factor",
        "2 * row",
        "2 * col",
    ],
    "gather_move_copy": [
        "draw(output",
        "output[",
        "grid[",
        "row +",
        "col +",
        "translate",
        "shift",
    ],
    "local_stencil": [
        "neighbors",
        "neighbours",
        "for dr in [-1, 0, 1]",
        "for dc in [-1, 0, 1]",
        "conway",
        "adjacent",
    ],
    "count_or_global_property": [
        "count",
        "len(",
        "sum(",
        "sample(range",
        "num_",
    ],
    "flood_connectivity": [
        "queue",
        "while queue",
        "connected",
        "diagonally_connected",
        "overlaps",
        "path",
        "flood",
    ],
    "template_correspondence": [
        "sprite",
        "sprites",
        "template",
        "rotate",
        "rotates",
        "matching",
        "idxs",
    ],
    "symmetry_or_transform": [
        "flip",
        "rot",
        "transpose",
        "diagonal",
        "mirror",
    ],
}


WALL_MARKERS = [
    "true wall",
    "genuine",
    "infeasible",
    "irreducible",
    "information",
    "ambiguous",
    "non-deterministic",
    "unreachable",
    "not exact",
]


def classify(text: str, tasklog: str, ops: Counter) -> list[str]:
    blob = (text + "\n" + tasklog).lower()
    archetypes = []
    for name, pats in PATTERNS.items():
        if any(p.lower() in blob for p in pats):
            archetypes.append(name)
    if ops.get("Conv", 0) or ops.get("QLinearConv", 0):
        if "local_stencil" not in archetypes:
            archetypes.append("local_stencil")
    if ops.get("Gather", 0) + ops.get("GatherND", 0) + ops.get("GatherElements", 0) >= 8:
        if "gather_move_copy" not in archetypes:
            archetypes.append("gather_move_copy")
    if ops.get("MaxPool", 0) or ops.get("CumSum", 0) or ops.get("BitShift", 0):
        if "flood_connectivity" not in archetypes:
            archetypes.append("flood_connectivity")
    return sorted(set(archetypes)) or ["unclassified"]


def tiny_builder_score(row: dict) -> float:
    """Higher means more attractive for semantic rewrite."""

    pts = float(row["points"] or 0.0)
    mem = int(row["memory"] or 0)
    params = int(row["params"] or 0)
    method = row.get("method") or ""
    source = row.get("source_class") or ""
    archetypes = set(row.get("archetypes") or [])
    tasklog = row.get("tasklog_lower") or ""

    # Headroom to a 20-point tiny builder.  Cap at zero to avoid wasting top
    # slots on already-good tasks.
    headroom20 = max(0.0, 20.0 - pts)
    resource_pressure = math.log(max(1, mem + params))

    score = headroom20 * 1000 + resource_pressure * 100

    if method.startswith("ext:") or method.startswith("teacher:"):
        score += 900
    if source in {"exact_preserve_payload", "no_source", "unknown_source"}:
        score += 450
    if "semantic_source" in source:
        score -= 250

    # Prefer families where a reusable compiler plausibly exists.
    if archetypes & {"crop_or_bbox", "scale_or_kron", "gather_move_copy", "op_direct_or_color_lut"}:
        score += 900
    if archetypes & {"local_stencil", "count_or_global_property"}:
        score += 350

    # Penalize known hard correspondence/flood walls, but not to zero: some are
    # worth explicit strategic review.
    if "template_correspondence" in archetypes:
        score -= 900
    if "flood_connectivity" in archetypes:
        score -= 700
    if any(marker in tasklog for marker in WALL_MARKERS):
        score -= 1800
    if "true wall" in tasklog or "confirmed true wall" in tasklog:
        score -= 3500
    return score


def compile_target(row: dict) -> str:
    arch = set(row["archetypes"])
    if "scale_or_kron" in arch and "gather_move_copy" in arch:
        return "variable_scale_gather_builder"
    if "crop_or_bbox" in arch and "gather_move_copy" in arch:
        return "bbox_crop_copy_builder"
    if "op_direct_or_color_lut" in arch and row["memory"] <= 1000:
        return "op_direct_tiny_lut_builder"
    if "local_stencil" in arch and "flood_connectivity" not in arch:
        return "single_conv_or_qlinear_stencil_builder"
    if "count_or_global_property" in arch and "template_correspondence" not in arch:
        return "global_count_threshold_builder"
    if "flood_connectivity" in arch:
        return "bounded_unroll_or_pointer_jump_builder"
    if "template_correspondence" in arch:
        return "template_matching_research"
    return "manual_semantic_rewrite"


def main() -> None:
    manifest = load_json(MANIFEST, {}).get("tasks", {})
    arc_map = load_json(ARC_MAP, {})
    inventory = {r["task"]: r for r in load_json(INVENTORY, {}).get("tasks", [])}

    rows = []
    for task in range(1, 401):
        rec = manifest.get(str(task)) or {}
        source_text = read_text(CUSTOM / f"task{task:03d}.py")
        tasklog = read_text(REPORTS / "tasklog" / f"task{task:03d}.md")
        gen = generator_text(task, arc_map)
        ops = op_counts(task)
        data = load_json(ROOT / "data" / f"task{task:03d}.json", {})
        inv = inventory.get(task, {})
        row = {
            "task": task,
            "points": rec.get("points"),
            "memory": rec.get("memory"),
            "params": rec.get("params"),
            "method": rec.get("method"),
            "source_class": source_class(source_text),
            "inventory_source_class": inv.get("source_class"),
            "archetypes": classify(gen + "\n" + source_text, tasklog, ops),
            "compile_target": None,
            "ops": dict(ops),
            "node_count": sum(ops.values()),
            "arc_id": (arc_map.get(str(task)) or {}).get("arc_id"),
            "shape_stats": example_shape_stats(data),
            "has_tasklog": bool(tasklog),
            "tasklog_lower": tasklog.lower(),
        }
        row["compile_target"] = compile_target(row)
        row["tiny_builder_score"] = round(tiny_builder_score(row), 3)
        # Do not leak the full tasklog into JSON.
        del row["tasklog_lower"]
        rows.append(row)

    by_arch: dict[str, list[dict]] = defaultdict(list)
    by_target: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        for arch in row["archetypes"]:
            by_arch[arch].append(row)
        by_target[row["compile_target"]].append(row)

    out = {
        "version": 1,
        "total": sum(float(r["points"] or 0) for r in rows),
        "rows": rows,
        "by_archetype": {k: [r["task"] for r in v] for k, v in sorted(by_arch.items())},
        "by_compile_target": {k: [r["task"] for r in v] for k, v in sorted(by_target.items())},
    }
    (REPORTS / "semantic_archetypes.json").write_text(json.dumps(out, indent=2, sort_keys=True))

    ranked = sorted(rows, key=lambda r: (-r["tiny_builder_score"], r["task"]))
    lines = [
        "# Semantic archetype map",
        "",
        f"- total: {out['total']:.3f}",
        "- purpose: find reusable semantic compiler targets, not one-off public overlays",
        "",
        "## Compile target counts",
        "",
    ]
    for target, vals in sorted(by_target.items(), key=lambda kv: (-len(kv[1]), kv[0])):
        lines.append(f"- `{target}`: {len(vals)}")
    lines += [
        "",
        "## Highest ROI semantic rewrite candidates",
        "",
        "| rank | task | score | pts | mem | params | method | source | compile target | archetypes |",
        "|---:|---:|---:|---:|---:|---:|---|---|---|---|",
    ]
    for i, r in enumerate(ranked[:80], 1):
        lines.append(
            f"| {i} | {r['task']:03d} | {r['tiny_builder_score']:.1f} | "
            f"{float(r['points'] or 0):.3f} | {r['memory']} | {r['params']} | "
            f"{r['method']} | {r['source_class']} | {r['compile_target']} | "
            f"{','.join(r['archetypes'])} |"
        )
    lines += ["", "## Archetype task lists", ""]
    for arch, vals in sorted(by_arch.items(), key=lambda kv: (-len(kv[1]), kv[0])):
        sample = " ".join(f"{r['task']:03d}" for r in vals[:80])
        lines.append(f"- `{arch}` ({len(vals)}): {sample}")
    (REPORTS / "semantic_archetypes.md").write_text("\n".join(lines) + "\n")
    print(f"wrote {REPORTS / 'semantic_archetypes.json'}")
    print(f"wrote {REPORTS / 'semantic_archetypes.md'}")
    print("top candidates:")
    for r in ranked[:20]:
        print(
            f"task{r['task']:03d} score={r['tiny_builder_score']:.1f} "
            f"pts={float(r['points'] or 0):.3f} target={r['compile_target']} "
            f"arch={','.join(r['archetypes'])}"
        )


if __name__ == "__main__":
    main()
