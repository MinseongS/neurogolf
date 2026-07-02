#!/usr/bin/env python3
"""Build human-review semantic class maps for all NeuroGolf tasks.

This is intentionally conservative.  It does not claim final semantics; it
creates broad candidate classes with evidence so the human review loop can
challenge and refine them.
"""

from __future__ import annotations

import json
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from statistics import mean


ROOT = Path(__file__).resolve().parents[2]
REPORTS = ROOT / "reports"
DATA = ROOT / "data"
TASKLOG = REPORTS / "tasklog"
CLASSLOG = REPORTS / "classlog"


@dataclass(frozen=True)
class ClassDef:
    group: str
    id: str
    title: str
    description: str


CLASSES: list[ClassDef] = [
    ClassDef("color", "color_fixed", "Fixed output colour", "Output colour can be hardcoded or selected from a fixed small set."),
    ClassDef("color", "color_preserve_input", "Preserve input colour", "Output keeps arbitrary input colours on copied/moved cells."),
    ClassDef("color", "color_marker_copy", "Copy marker/object colour", "Colour is selected from a marker, hint, or source object."),
    ClassDef("color", "color_recolor_rule", "Recolour by rule", "Input colours are mapped to different output colours."),
    ClassDef("color", "color_mode_count", "Colour by count/mode/rank", "Colour comes from count, mode, rank, argmax, or palette statistics."),
    ClassDef("color", "color_palette_lut", "Palette/LUT remap", "Colour choice is naturally a small lookup table or channel remap."),
    ClassDef("shape", "shape_fixed_template", "Fixed template", "Shape is fixed or from a tiny constant template."),
    ClassDef("shape", "shape_copy_object", "Copy object shape", "Shape is copied from an input object/component."),
    ClassDef("shape", "shape_clone_duplicate", "Clone/duplicate shape", "A shape is repeated one or more times."),
    ClassDef("shape", "shape_extend_line_ray", "Extend line/ray", "Line, ray, or span is extended from observed marks."),
    ClassDef("shape", "shape_bbox_rect", "Bounding box/rectangle", "Shape is a bbox, solid rectangle, frame, or rectangular interior."),
    ClassDef("shape", "shape_enclosed_fill", "Enclosed fill", "Output fills holes/enclosed background regions."),
    ClassDef("shape", "shape_local_stencil", "Local stencil", "Cell state depends mainly on a local neighbourhood."),
    ClassDef("shape", "shape_component", "Connected component", "Rule depends on connected components or object grouping."),
    ClassDef("shape", "shape_row_col_profile", "Row/column profile", "Rule can be read from row/column counts, bands, or separators."),
    ClassDef("shape", "shape_template_match", "Template match", "Small template/sprite is matched, rotated, or stamped."),
    ClassDef("direction", "direction_none_or_fixed", "No/fixed direction", "No dynamic direction, or direction is a fixed constant."),
    ClassDef("direction", "direction_axis_aligned", "Axis-aligned direction", "Uses horizontal/vertical rows, columns, bars, or spans."),
    ClassDef("direction", "direction_diagonal", "Diagonal direction", "Uses diagonal or slanted relationships."),
    ClassDef("direction", "direction_marker_relative", "Marker-relative direction", "Direction or target is inferred from marker/hint placement."),
    ClassDef("direction", "direction_rotation_reflection", "Rotation/reflection candidate", "Needs orientation, flip, rotation, or dihedral candidates."),
    ClassDef("direction", "direction_gravity", "Gravity/drop direction", "Objects move/drop/fall toward an edge or obstacle."),
    ClassDef("action", "action_copy", "Copy", "Output copies cells/objects from input."),
    ClassDef("action", "action_move_translate", "Move/translate", "Object is shifted to a different location."),
    ClassDef("action", "action_clone_repeat", "Clone/repeat", "Object/template is repeated or tiled."),
    ClassDef("action", "action_extend", "Extend", "Existing marks are extended into lines/spans/rays."),
    ClassDef("action", "action_fill", "Fill", "Region is filled or completed."),
    ClassDef("action", "action_crop_resize", "Crop/resize", "Output crops, pads, resizes, upscales, or downscales."),
    ClassDef("action", "action_erase_filter", "Erase/filter/select", "Some input content is removed or a subset is selected."),
    ClassDef("action", "action_reorder_pack", "Reorder/pack/sort", "Objects are sorted, packed, ranked, or rearranged."),
    ClassDef("placement", "placement_same", "Same position", "Output changes colour/value at same positions."),
    ClassDef("placement", "placement_fixed_offset", "Fixed offset", "Placement is a constant translation/offset."),
    ClassDef("placement", "placement_marker_target", "Marker target", "Placement is controlled by markers or target slots."),
    ClassDef("placement", "placement_grid_repeat", "Grid repeat/tile", "Placement follows a repeated grid/lattice/tile."),
    ClassDef("placement", "placement_canonical", "Canonical crop/top-left", "Output is canonicalized to a crop, bbox, or top-left origin."),
    ClassDef("compiler", "compiler_direct_output_algebra", "Direct output algebra", "Emit thresholded final output without full intermediate carriers."),
    ClassDef("compiler", "compiler_direct_onehot_gather", "Direct one-hot gather", "Route input one-hot channels directly to output."),
    ClassDef("compiler", "compiler_final_equal_overlay", "Final Equal/overlay", "Carry scalar labels or masks until final Equal/Where output."),
    ClassDef("compiler", "compiler_single_conv_qlinear", "Single Conv/QLinearConv", "Collapse local predicates/counts into one Conv/QLinearConv family."),
    ClassDef("compiler", "compiler_tiny_lut_gather", "Tiny LUT/Gather", "Use small lookup tables, Gather, or channel remap."),
    ClassDef("compiler", "compiler_einsum_symbolic", "Einsum symbolic", "Use algebraic contraction or selector factorization."),
    ClassDef("compiler", "compiler_roi_pool_crop", "Roi/crop/pool primitive", "Use RoiAlign, MaxRoiPool, Resize, GridSample, or crop primitives."),
    ClassDef("compiler", "compiler_sparse_scatter", "Sparse scatter/edit stream", "Use sparse coordinate updates instead of full masks where safe."),
    ClassDef("compiler", "compiler_bounded_scan", "Bounded scan/flood-fill", "Use MaxPool/CumSum/scan, ideally cropped or compressed."),
    ClassDef("compiler", "compiler_qlinear_uint8", "QLinear/uint8 compression", "Replace fp32/fp16 routing with uint8/QLinear exact forms."),
    ClassDef("cost", "cost_mem0_param_game", "Mem0 param game", "Memory is zero/tiny; improvements must reduce params."),
    ClassDef("cost", "cost_full_label_plane_floor", "Full label plane floor", "A 30x30 scalar label/mask carrier likely dominates."),
    ClassDef("cost", "cost_full_onehot_floor", "Full one-hot floor", "A 10-channel full-canvas carrier likely dominates."),
    ClassDef("cost", "cost_connectivity_wall", "Connectivity wall", "Flood-fill/component connectivity is the hard cost driver."),
    ClassDef("cost", "cost_assignment_wall", "Assignment wall", "Matching/correspondence/ambiguous assignment is the hard cost driver."),
    ClassDef("cost", "cost_exact_preserve_rewrite", "Exact-preserve rewrite target", "Current source is exact-preserve or low-semantics and should be challenged."),
]


CLASS_BY_ID = {c.id: c for c in CLASSES}


LOG_KEYWORDS: list[tuple[str, str]] = [
    ("color_fixed", r"\b(fixed colou?r|hardcod(?:e|ed) colou?r|constant colou?r)"),
    ("color_marker_copy", r"\b(marker|hint).{0,40}\b(colou?r|palette)|preserv(?:e|ing).{0,40}(hint|marker) colou?r"),
    ("color_preserve_input", r"\b(colou?rs? (?:simply )?copy|preserv(?:e|ing).{0,30}colou?r|arbitrary input colou?r|random per-instance colou?r)"),
    ("color_recolor_rule", r"\b(recolou?r|recolor|mapped? to|colour rank|color rank)"),
    ("color_mode_count", r"\b(mode|rank|argmax|argmin|majority|minority|tallest|shortest|colour-count|color-count)"),
    ("shape_enclosed_fill", r"\b(enclosed|surrounded|hole|interior|flood|fill enclosed|pot)"),
    ("shape_bbox_rect", r"\b(bbox|bounding|rectangle|rectangular|box|frame)"),
    ("shape_copy_object", r"\b(copy object|copy the object|object shape|same shape|identical shape)"),
    ("shape_extend_line_ray", r"\b(extend|line|ray|span|connect|between)"),
    ("shape_template_match", r"\b(template|sprite|stamp|glyph|dihedral|rotation|rotated)"),
    ("shape_component", r"\b(component|connected component|object grouping)"),
    ("shape_row_col_profile", r"\b(row|column|profile|separator|band)"),
    ("direction_marker_relative", r"\b(marker|hint).{0,50}\b(direction|target|slot|route|lattice)"),
    ("direction_rotation_reflection", r"\b(rotation|rotated|reflection|flip|dihedral|orientation)"),
    ("direction_diagonal", r"\b(diagonal|slanted|parallelogram)"),
    ("direction_axis_aligned", r"\b(row|column|horizontal|vertical|axis|bar|span)"),
    ("direction_gravity", r"\b(gravity|fall|falling|settle|settling)"),
    ("action_clone_repeat", r"\b(repeat|tile|clone|duplicate|copies of itself|every \d+ cells)"),
    ("action_move_translate", r"\b(translate|shift|move|offset)"),
    ("action_extend", r"\b(extend|span-fill|fill between|ray)"),
    ("action_fill", r"\b(fill|complete|interior|hole)"),
    ("action_crop_resize", r"\b(crop|resize|upscale|downscale|canonical)"),
    ("action_erase_filter", r"\b(remove|erase|filter|select one|subset)"),
    ("action_reorder_pack", r"\b(sort|reorder|rank order|arrange objects|packed objects)"),
    ("placement_grid_repeat", r"\b(tile|lattice|every \d+|periodic|linegrid)"),
    ("placement_marker_target", r"\b(marker|hint).{0,50}\b(target|slot|place|route|lattice)|\b(target|slot).{0,50}\b(marker|hint)"),
    ("placement_canonical", r"\b(top-left|canonical|crop|bbox)"),
]


def grid_size(grid: list[list[int]]) -> tuple[int, int]:
    return len(grid), len(grid[0]) if grid else 0


def colours(grid: list[list[int]]) -> set[int]:
    return {v for row in grid for v in row}


def nonzero_cells(grid: list[list[int]]) -> set[tuple[int, int, int]]:
    return {(r, c, v) for r, row in enumerate(grid) for c, v in enumerate(row) if v != 0}


def mask_cells(grid: list[list[int]]) -> set[tuple[int, int]]:
    return {(r, c) for r, row in enumerate(grid) for c, v in enumerate(row) if v != 0}


def bbox(mask: set[tuple[int, int]]) -> tuple[int, int, int, int] | None:
    if not mask:
        return None
    rs = [r for r, _ in mask]
    cs = [c for _, c in mask]
    return min(rs), min(cs), max(rs), max(cs)


def add(classes: set[str], evidence: dict[str, list[str]], cid: str, reason: str) -> None:
    classes.add(cid)
    evidence[cid].append(reason)


def tasklog_text(n: int) -> str:
    p = TASKLOG / f"task{n:03d}.md"
    return p.read_text(errors="replace") if p.exists() else ""


def one_example_features(n: int, classes: set[str], evidence: dict[str, list[str]]) -> dict[str, object]:
    d = json.loads((DATA / f"task{n:03d}.json").read_text())
    examples = d.get("train", [])
    sizes_in, sizes_out = [], []
    same_size = 0
    output_colours_subset_input = 0
    input_colours_subset_output = 0
    same_nonzero_mask = 0
    same_exact = 0
    out_larger = 0
    out_smaller = 0
    bbox_like = 0
    repeated_sizes = set()
    fixed_new_colours: list[set[int]] = []
    same_mask_count = 0
    translated = 0

    for ex in examples:
        inp, out = ex["input"], ex["output"]
        hi, wi = grid_size(inp)
        ho, wo = grid_size(out)
        sizes_in.append((hi, wi))
        sizes_out.append((ho, wo))
        repeated_sizes.add((hi, wi, ho, wo))
        ci, co = colours(inp), colours(out)
        if (hi, wi) == (ho, wo):
            same_size += 1
            if inp == out:
                same_exact += 1
            mi = mask_cells(inp)
            mo = mask_cells(out)
            if mi == mo:
                same_nonzero_mask += 1
            if len(mi) == len(mo) and len(mi) > 0:
                same_mask_count += 1
                bi, bo = bbox(mi), bbox(mo)
                if bi and bo:
                    dr, dc = bo[0] - bi[0], bo[1] - bi[1]
                    if (dr or dc) and {(r + dr, c + dc) for r, c in mi} == mo:
                        translated += 1
        if co <= ci:
            output_colours_subset_input += 1
        if ci <= co:
            input_colours_subset_output += 1
        fixed_new_colours.append(co - ci)
        if ho * wo > hi * wi:
            out_larger += 1
        if ho * wo < hi * wi:
            out_smaller += 1
        bout = bbox(mask_cells(out))
        if bout:
            r0, c0, r1, c1 = bout
            area = (r1 - r0 + 1) * (c1 - c0 + 1)
            if area and len(mask_cells(out)) / area > 0.75:
                bbox_like += 1

    m = len(examples) or 1
    if same_size == len(examples):
        add(classes, evidence, "placement_same", "all stored train examples keep input/output size")
    if out_larger:
        add(classes, evidence, "action_crop_resize", f"{out_larger}/{m} examples enlarge canvas")
        add(classes, evidence, "shape_clone_duplicate", "larger output often indicates expansion/repetition candidate")
    if out_smaller:
        add(classes, evidence, "action_crop_resize", f"{out_smaller}/{m} examples shrink/crop canvas")
        add(classes, evidence, "placement_canonical", "smaller output suggests canonical crop candidate")
    if output_colours_subset_input == len(examples):
        add(classes, evidence, "color_preserve_input", "all output colours are subset of input colours")
        add(classes, evidence, "action_copy", "colour subset makes copy/preserve route plausible")
        if same_mask_count == len(examples):
            add(classes, evidence, "shape_copy_object", "same nonzero cell count with preserved colours suggests object-copy candidate")
    if input_colours_subset_output == len(examples) and output_colours_subset_input != len(examples):
        add(classes, evidence, "color_recolor_rule", "output introduces colours not present in input")
        if fixed_new_colours and any(fixed_new_colours) and len({tuple(sorted(x)) for x in fixed_new_colours}) == 1:
            add(classes, evidence, "color_fixed", f"same new output colours across stored examples: {sorted(fixed_new_colours[0])}")
    if same_nonzero_mask == len(examples) and same_exact != len(examples):
        add(classes, evidence, "color_recolor_rule", "same nonzero mask but different colours")
        add(classes, evidence, "placement_same", "same occupied positions in stored examples")
    if bbox_like >= max(1, len(examples) // 2):
        add(classes, evidence, "shape_bbox_rect", f"{bbox_like}/{m} output masks are bbox-dense")
    if len(repeated_sizes) == 1:
        hi, wi, ho, wo = next(iter(repeated_sizes))
        if hi and wi and ho % hi == 0 and wo % wi == 0 and (ho > hi or wo > wi):
            add(classes, evidence, "placement_grid_repeat", f"output size is integer multiple of input size: {(hi, wi)} -> {(ho, wo)}")
            add(classes, evidence, "action_clone_repeat", "integer scale suggests tiling/repetition candidate")
    if translated == len(examples) and translated:
        add(classes, evidence, "placement_fixed_offset", "stored examples have bbox-aligned translation of the nonzero mask")
        add(classes, evidence, "action_move_translate", "nonzero mask translates as a unit in stored examples")

    return {
        "train_count": len(d.get("train", [])),
        "test_count": len(d.get("test", [])),
        "arc_gen_count": len(d.get("arc-gen", [])),
        "sizes_in": sorted(set(sizes_in)),
        "sizes_out": sorted(set(sizes_out)),
    }


def inventory_features(
    item: dict,
    manifest: dict[str, dict],
    classes: set[str],
    evidence: dict[str, list[str]],
) -> None:
    n = item["task"]
    ops = set(item.get("ops", {}))
    tags = set(item.get("tags", []))
    mem = item.get("memory", 0)
    params = item.get("params", 0)
    pts = item.get("points", manifest.get(str(n), {}).get("points", 0))
    source = item.get("source_class", "")

    if mem == 0 or mem <= 150:
        add(classes, evidence, "cost_mem0_param_game", f"memory is {mem}; params dominate")
    if mem >= 800 and ("Equal" in ops or "Where" in ops or "Pad" in ops):
        add(classes, evidence, "cost_full_label_plane_floor", "Equal/Where/Pad with nontrivial memory suggests scalar label/mask carrier")
    if mem >= 3000 and ("Slice" in ops or "Cast" in ops) and ("ArgMax" in ops or "ReduceSum" in ops):
        add(classes, evidence, "color_mode_count", "ArgMax/ReduceSum over colour channels suggests colour selection/count")
    if "onehot_final_equal" in tags or "Equal" in ops:
        add(classes, evidence, "compiler_final_equal_overlay", "inventory has Equal/onehot-final pattern")
    if "lut_selection" in tags or "Gather" in ops or "QLinearMatMul" in ops:
        add(classes, evidence, "compiler_tiny_lut_gather", "Gather/LUT operator evidence")
        add(classes, evidence, "color_palette_lut", "LUT/Gather evidence may represent palette/remap")
    if "Einsum" in ops:
        add(classes, evidence, "compiler_einsum_symbolic", "Einsum present")
        add(classes, evidence, "compiler_direct_output_algebra", "Einsum can be direct threshold algebra candidate")
    if "Conv" in ops or "QLinearConv" in ops or "ConvInteger" in ops:
        add(classes, evidence, "shape_local_stencil", "Conv family present")
        add(classes, evidence, "compiler_single_conv_qlinear", "Conv/QLinearConv family present")
    if "QLinearConv" in ops or "QLinearMatMul" in ops or "ConvInteger" in ops or "MatMulInteger" in ops:
        add(classes, evidence, "compiler_qlinear_uint8", "quantized integer op present")
    if "MaxPool" in ops or "CumSum" in ops or "ReduceMax" in ops or "ReduceMin" in ops:
        add(classes, evidence, "compiler_bounded_scan", "scan/pool/reduce operators present")
    if "ScatterND" in ops or "ScatterElements" in ops:
        add(classes, evidence, "compiler_sparse_scatter", "Scatter operator present")
        add(classes, evidence, "action_move_translate", "scatter often represents move/place/edit candidate")
    if "RoiAlign" in ops or "MaxRoiPool" in ops or "Resize" in ops or "GridSample" in ops:
        add(classes, evidence, "compiler_roi_pool_crop", "ROI/crop/resize primitive present")
        add(classes, evidence, "action_crop_resize", "ROI/crop/resize operator evidence")
    if "connectivity_wall" in tags:
        add(classes, evidence, "shape_component", "connectivity_wall tag")
        add(classes, evidence, "cost_connectivity_wall", "connectivity_wall tag")
    if "assignment_wall" in tags:
        add(classes, evidence, "cost_assignment_wall", "assignment_wall tag")
    if source == "exact_preserve" or "exact_preserve" in tags:
        add(classes, evidence, "cost_exact_preserve_rewrite", "exact_preserve source/tag")
    if "local_stencil" in tags:
        add(classes, evidence, "shape_local_stencil", "local_stencil tag")
    if "scatter" in tags:
        add(classes, evidence, "compiler_sparse_scatter", "scatter tag")
    if "matmul" in tags or "MatMul" in ops:
        add(classes, evidence, "compiler_direct_onehot_gather", "MatMul/direct routing candidate")
    if pts >= 20:
        add(classes, evidence, "compiler_direct_output_algebra", f"high-score frontier task ({pts:.3f} pts)")


def log_features(text: str, classes: set[str], evidence: dict[str, list[str]]) -> None:
    low = text.lower()
    for cid, pat in LOG_KEYWORDS:
        if re.search(pat, low):
            add(classes, evidence, cid, f"tasklog keyword match: {pat}")


def review_status(item: dict, text: str) -> str:
    if "verified" in text.lower() or "fresh" in text.lower() or "stored eval passed" in text.lower():
        return "seeded_from_verified_log"
    if item.get("tasklog_exists"):
        return "seeded_from_tasklog_and_inventory"
    return "operator_evidence_only_needs_human_review"


def top_routes(classes: set[str]) -> list[str]:
    priority = [
        "compiler_direct_output_algebra",
        "compiler_direct_onehot_gather",
        "compiler_tiny_lut_gather",
        "compiler_single_conv_qlinear",
        "compiler_final_equal_overlay",
        "compiler_qlinear_uint8",
        "compiler_roi_pool_crop",
        "compiler_sparse_scatter",
        "compiler_bounded_scan",
    ]
    return [c for c in priority if c in classes][:4]


def write_taxonomy() -> None:
    lines = [
        "# Semantic Mechanism Classes",
        "",
        "This taxonomy is for human review.  Tasks may belong to many classes at",
        "once; classes are hypotheses, not exclusive labels.  A class becomes an",
        "optimization target only after at least one task verifies the mechanism.",
        "",
    ]
    for group in ["color", "shape", "direction", "action", "placement", "compiler", "cost"]:
        lines += [f"## {group.title()} Classes", ""]
        for c in CLASSES:
            if c.group == group:
                lines.append(f"- `{c.id}` — **{c.title}**: {c.description}")
        lines.append("")
    (REPORTS / "SEMANTIC_MECHANISM_CLASSES.md").write_text("\n".join(lines) + "\n")


def write_mapping(records: list[dict]) -> None:
    lines = [
        "# 400-Task Semantic Class Map",
        "",
        "Generated by `reports/scripts/build_semantic_class_map.py`.",
        "",
        "This file is a broad, review-first mapping.  `seeded_from_verified_log`",
        "means existing notes contain verification language; it does not mean every",
        "class candidate below is proven.  `operator_evidence_only_needs_human_review`",
        "is especially tentative.",
        "",
    ]
    for r in records:
        n = r["task"]
        m = r["manifest"]
        lines += [
            f"## task{n:03d}",
            "",
            f"- score: {m['points']:.6f}; mem: {m['memory']}; params: {m['params']}; method: `{m['method']}`",
            f"- review status: `{r['review_status']}`",
            f"- data: train {r['example_summary']['train_count']}, test {r['example_summary']['test_count']}, arc-gen {r['example_summary']['arc_gen_count']}",
            f"- sizes: input {r['example_summary']['sizes_in']} -> output {r['example_summary']['sizes_out']}",
            f"- recommended first routes: {', '.join(f'`{x}`' for x in r['top_routes']) or '`manual_semantic_review`'}",
            "",
            "| group | class candidates |",
            "|---|---|",
        ]
        by_group: dict[str, list[str]] = defaultdict(list)
        for cid in r["classes"]:
            by_group[CLASS_BY_ID[cid].group].append(cid)
        for group in ["color", "shape", "direction", "action", "placement", "compiler", "cost"]:
            vals = ", ".join(f"`{v}`" for v in by_group.get(group, []))
            lines.append(f"| {group} | {vals or '-'} |")
        lines += ["", "<details><summary>Evidence</summary>", ""]
        for cid in r["classes"]:
            ev = "; ".join(r["evidence"][cid][:4])
            lines.append(f"- `{cid}`: {ev}")
        lines += ["", "</details>", ""]
    (REPORTS / "semantic_task_map.md").write_text("\n".join(lines))


def write_classlogs(records: list[dict]) -> None:
    index = [
        "# Classlog Index",
        "",
        "Each classlog lists tasks currently mapped to that class.  These are broad",
        "candidate memberships for human review; promote only verified mechanisms to",
        "`reports/insight_registry.yaml`.",
        "",
        "| group | class | tasks |",
        "|---|---|---:|",
    ]
    by_class: dict[str, list[dict]] = defaultdict(list)
    for r in records:
        for cid in r["classes"]:
            by_class[cid].append(r)

    for c in CLASSES:
        rows = sorted(by_class.get(c.id, []), key=lambda r: (r["manifest"]["points"], -r["manifest"]["memory"]))
        index.append(f"| {c.group} | [`{c.id}`]({c.id}.md) | {len(rows)} |")
        lines = [
            f"# {c.id} — {c.title}",
            "",
            c.description,
            "",
            "## Optimization Question",
            "",
            "What is the cheapest verified ONNX family for tasks in this class, and",
            "which semantic preconditions let us avoid full-canvas intermediates?",
            "",
            "## Candidate Tasks",
            "",
            "| task | pts | mem | params | status | first routes | evidence |",
            "|---:|---:|---:|---:|---|---|---|",
        ]
        for r in rows:
            n = r["task"]
            m = r["manifest"]
            ev = "; ".join(r["evidence"].get(c.id, [])[:2]).replace("|", "\\|")
            routes = ", ".join(f"`{x}`" for x in r["top_routes"])
            lines.append(
                f"| {n:03d} | {m['points']:.3f} | {m['memory']} | {m['params']} | "
                f"{r['review_status']} | {routes} | {ev} |"
            )
        lines += [
            "",
            "## Known Best Routes",
            "",
            "- Pending human review.",
            "",
            "## Kill Criteria",
            "",
            "- Pending human review.",
            "",
            "## Successful Applications",
            "",
            "- Pending verification.",
            "",
            "## Failed Applications / Walls",
            "",
            "- Pending verification.",
            "",
        ]
        (CLASSLOG / f"{c.id}.md").write_text("\n".join(lines))
    (CLASSLOG / "_INDEX.md").write_text("\n".join(index) + "\n")


def main() -> None:
    CLASSLOG.mkdir(parents=True, exist_ok=True)
    inv = json.loads((REPORTS / "global_layer_inventory.json").read_text())
    manifest = json.loads((REPORTS / "manifest.json").read_text())["tasks"]
    records = []
    for item in sorted(inv["tasks"], key=lambda x: x["task"]):
        n = item["task"]
        classes: set[str] = set()
        evidence: dict[str, list[str]] = defaultdict(list)
        ex_summary = one_example_features(n, classes, evidence)
        inventory_features(item, manifest, classes, evidence)
        text = tasklog_text(n)
        log_features(text, classes, evidence)
        if not any(CLASS_BY_ID[c].group == "direction" for c in classes):
            add(classes, evidence, "direction_none_or_fixed", "no dynamic direction evidence found")
        records.append(
            {
                "task": n,
                "manifest": manifest[str(n)],
                "example_summary": ex_summary,
                "classes": sorted(classes, key=lambda cid: (CLASS_BY_ID[cid].group, cid)),
                "evidence": {k: v for k, v in sorted(evidence.items())},
                "review_status": review_status(item, text),
                "top_routes": top_routes(classes),
            }
        )

    out_json = {
        "note": "Broad candidate class mapping for human review; task membership is multi-label and non-exclusive.",
        "class_count": len(CLASSES),
        "task_count": len(records),
        "classes": [c.__dict__ for c in CLASSES],
        "tasks": records,
    }
    (REPORTS / "semantic_task_map.json").write_text(json.dumps(out_json, indent=2))
    write_taxonomy()
    write_mapping(records)
    write_classlogs(records)
    print(f"wrote {REPORTS / 'SEMANTIC_MECHANISM_CLASSES.md'}")
    print(f"wrote {REPORTS / 'semantic_task_map.md'}")
    print(f"wrote {REPORTS / 'semantic_task_map.json'}")
    print(f"wrote {CLASSLOG / '_INDEX.md'} and {len(CLASSES)} class logs")


if __name__ == "__main__":
    main()
