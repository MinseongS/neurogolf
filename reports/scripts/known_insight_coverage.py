#!/usr/bin/env python3
"""Audit whether known NeuroGolf insights are represented in active overfit nets.

This is intentionally different from `find_insight_candidates.py`: that script
uses the source-owned `networks/` inventory.  This audit inspects the scoring
artifact set, `submission/overfit_nets/`, then builds a task x insight matrix
with rough states:

- source: the task is one of the insight's source tasks.
- logged: the tasklog mentions the insight id, title keywords, or transformer.
- graph_evidence: the active graph already contains the mechanism's core op cue.
- candidate: the active graph matches the registry predicate.
- candidate_unlogged: high-value gap: predicate matches but no source/log evidence.
- actionable: predicate matches and no local evidence says this family is already blocked.
- known_blocked / byte_negative / kaggle_falsified / unlowered_semantic_compiler:
  predicate matches, but local logs indicate the insight-family has already been
  tried, cannot lower cheaply, or is unsafe for submission.

The report is a triage artifact, not a proof that a rewrite will work.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import onnx


ROOT = Path(__file__).resolve().parents[2]
REPORTS = ROOT / "reports"
ACTIVE = ROOT / "submission" / "overfit_nets"


def load_json(path: Path) -> Any:
    return json.loads(path.read_text())


def load_json_optional(path: Path, default: Any) -> Any:
    try:
        return json.loads(path.read_text())
    except FileNotFoundError:
        return default


def read_text(path: Path) -> str:
    try:
        return path.read_text(errors="ignore")
    except FileNotFoundError:
        return ""


def op_counts(model: onnx.ModelProto) -> Counter[str]:
    return Counter(node.op_type for node in model.graph.node)


def attr_i(node: onnx.NodeProto, name: str, default: int | None = None) -> int | None:
    for attr in node.attribute:
        if attr.name == name:
            return int(attr.i)
    return default


def attr_s(node: onnx.NodeProto, name: str) -> str | None:
    for attr in node.attribute:
        if attr.name == name:
            try:
                return attr.s.decode()
            except Exception:
                return str(attr.s)
    return None


def shape_map(model: onnx.ModelProto) -> dict[str, list[int]]:
    try:
        model = onnx.shape_inference.infer_shapes(model, strict_mode=False)
    except Exception:
        pass
    shapes: dict[str, list[int]] = {}
    for vi in list(model.graph.input) + list(model.graph.value_info) + list(model.graph.output):
        tt = vi.type.tensor_type
        if not tt.HasField("shape"):
            continue
        dims: list[int] = []
        ok = True
        for dim in tt.shape.dim:
            if not dim.HasField("dim_value"):
                ok = False
                break
            dims.append(int(dim.dim_value))
        if ok:
            shapes[vi.name] = dims
    return shapes


def active_tags(ops: Counter[str], row: dict[str, Any], tasklog: str) -> set[str]:
    tags: set[str] = set()
    lower = tasklog.lower()
    memory = int(row.get("memory") or 0)
    points = float(row.get("points") or 0.0)
    node_count = sum(ops.values())
    if ops.get("Conv", 0) + ops.get("QLinearConv", 0) >= 2:
        tags.add("conv_heavy")
    if ops.get("Conv", 0) + ops.get("QLinearConv", 0) >= 1 and node_count <= 60:
        tags.add("local_stencil")
    if ops.get("QLinearConv") or ops.get("QLinearMatMul") or "qlinear" in lower:
        tags.add("qlinear")
    if ops.get("MatMul") or ops.get("MatMulInteger"):
        tags.add("matmul")
    if "lut" in lower or "lookup" in lower:
        tags.add("lut_selection")
    if ops.get("Equal"):
        tags.add("onehot_final_equal")
    if ops.get("CumSum") or ops.get("ReduceMax", 0) + ops.get("ReduceMin", 0) >= 10:
        tags.add("scan")
    if ops.get("MaxPool", 0) >= 5:
        tags.add("maxpool_scan")
    if ops.get("Gather", 0) + ops.get("GatherND", 0) + ops.get("GatherElements", 0) >= 10:
        tags.add("gather_heavy")
    if ops.get("ScatterND") or ops.get("ScatterElements"):
        tags.add("scatter")
    if ops.get("BitShift") or ops.get("BitwiseAnd") or ops.get("BitwiseOr"):
        tags.add("bitwise_program")
    if "assignment" in lower or "correspondence" in lower:
        tags.add("assignment_wall")
    if "connect" in lower or "flood" in lower:
        tags.add("connectivity_wall")
    if "wall" in lower or "infeasible" in lower or "underdetermined" in lower:
        tags.add("documented_wall")
    if memory >= 10000:
        tags.add("high_memory")
    if points < 15.0:
        tags.add("low_score")
    return tags


def any_intersects(values: list[str], target: set[str]) -> bool:
    return bool(set(values or []).intersection(target))


def all_in(values: list[str], target: set[str]) -> bool:
    return set(values or []).issubset(target)


def registry_matches(task: dict[str, Any], insight: dict[str, Any]) -> bool:
    if insight.get("status") != "active":
        return False
    pred = insight.get("applies_when") or {}
    reject = insight.get("reject_when") or {}
    ops = set(task["ops"])
    tags = set(task["tags"])
    memory = int(task.get("memory") or 0)
    points = float(task.get("points") or 0.0)

    if pred.get("any_ops") and not any_intersects(pred["any_ops"], ops):
        return False
    if pred.get("all_ops") and not all_in(pred["all_ops"], ops):
        return False
    if pred.get("any_tags") and not any_intersects(pred["any_tags"], tags):
        return False
    if pred.get("all_tags") and not all_in(pred["all_tags"], tags):
        return False
    if pred.get("min_memory") is not None and memory < int(pred["min_memory"]):
        return False
    if pred.get("max_points") is not None and points > float(pred["max_points"]):
        return False

    if reject.get("any_ops") and any_intersects(reject["any_ops"], ops):
        return False
    if reject.get("any_tags") and any_intersects(reject["any_tags"], tags):
        return False
    return True


def title_tokens(insight: dict[str, Any]) -> list[str]:
    raw = " ".join(
        str(v)
        for v in [insight.get("id", ""), insight.get("title", ""), insight.get("transformer", "")]
    ).lower()
    return [tok for tok in re.split(r"[^a-z0-9_]+", raw) if len(tok) >= 6]


def logged_evidence(tasklog: str, insight: dict[str, Any]) -> bool:
    lower = tasklog.lower()
    if insight["id"].lower() in lower:
        return True
    transformer = str(insight.get("transformer") or "").lower()
    if transformer and Path(transformer).name.lower() in lower:
        return True
    tokens = title_tokens(insight)
    return bool(tokens) and sum(1 for tok in tokens if tok in lower) >= min(2, len(tokens))


def insight_family_terms(insight: dict[str, Any]) -> set[str]:
    iid = str(insight.get("id", "")).lower()
    title = str(insight.get("title", "")).lower()
    text = f"{iid} {title}"
    terms = set(title_tokens(insight))
    if "topk" in text:
        terms.update({"topk", "uint8 topk", "signed int8 topk", "unsigned topk"})
    if "argmax" in text:
        terms.add("argmax")
    if "qlinear" in text:
        terms.update({"qlinear", "qlinearconv", "qlinearmatmul"})
    if "crop" in text or "pad" in text:
        terms.update({"crop", "pad", "padding"})
    if "einsum" in text:
        terms.add("einsum")
    if "exact" in text or "semantic" in text or "marker" in text:
        terms.update({"exact-cover", "semantic compiler", "lowering", "compiler"})
    return terms


def tasklog_mentions_family(tasklog: str, insight: dict[str, Any]) -> bool:
    lower = tasklog.lower()
    if insight["id"].lower() in lower:
        return True
    terms = insight_family_terms(insight)
    return any(term in lower for term in terms)


def blocked_state(tasklog: str, insight: dict[str, Any], state: str) -> str | None:
    """Classify candidate states that are already known non-actionable.

    This is conservative: exact insight-id evidence is strongest; broad tasklog
    walls only demote a candidate to known_blocked when they mention the same
    mechanism family.  Otherwise the candidate remains actionable.
    """

    if not state.startswith("candidate") and state != "source_candidate":
        return None
    lower = tasklog.lower()
    iid = insight["id"].lower()
    family = tasklog_mentions_family(tasklog, insight)

    exact_negative = iid in lower and any(
        phrase in lower
        for phrase in [
            "blocked",
            "rejected",
            "falsified",
            "negative",
            "worse",
            "cost worsened",
            "do not",
            "not adopt",
            "0 wins",
        ]
    )
    if exact_negative and "kaggle" in lower and any(x in lower for x in ["rejected", "falsified", "error"]):
        return "kaggle_falsified"
    if exact_negative and any(x in lower for x in ["worse", "negative", "cost worsened", "byte-negative"]):
        return "byte_negative"
    if exact_negative:
        return "known_blocked"

    if family and "kaggle" in lower and any(x in lower for x in ["rejected", "falsified", "error"]):
        if any(term in iid for term in ["topk", "argmax", "signed", "dynamic"]):
            return "kaggle_falsified"
    if family and any(x in lower for x in ["worse", "negative", "cost worsened", "byte-negative"]):
        return "byte_negative"
    if family and any(x in lower for x in ["exact-cover", "semantic compiler", "lowering"]):
        if any(term in iid for term in ["solid_marker", "bounded_exact", "semantic", "threshold", "public_teacher"]):
            return "unlowered_semantic_compiler"
    if family and any(x in lower for x in ["floor", "do not re", "do not re-attempt", "do not re-probe", "do not patch"]):
        return "known_blocked"
    return None


def graph_evidence(model: onnx.ModelProto, insight_id: str) -> bool:
    ops = op_counts(model)
    nodes = list(model.graph.node)
    shapes = shape_map(model)
    if insight_id in {
        "residual_spatialop_to_free_einsum_collapse",
        "spatial_reducesum_to_einsum_profile_tail",
    }:
        return any(
            n.op_type == "Einsum"
            and n.input
            and n.input[0] == "input"
            and (attr_s(n, "equation") or "").startswith("bchw")
            for n in nodes
        )
    if insight_id == "free_final_onehot_equal":
        return bool(nodes and nodes[-1].op_type == "Equal" and "output" in nodes[-1].output)
    if insight_id in {"qlinear_uint8_lut_or_matmul", "qlinearconv_signed_renderer"}:
        return bool(ops.get("QLinearConv") or ops.get("QLinearMatMul"))
    if insight_id == "sparse_edit_stream_without_mask_planes":
        return bool(ops.get("ScatterND") or ops.get("ScatterElements"))
    if insight_id == "dynamic_bundled_cse_rewire":
        return False
    if insight_id == "dedupe_byte_identical_initializers":
        return False
    if insight_id == "topk_k2_to_argmax_uint8_with_exception_patch":
        return bool(ops.get("ArgMax") and not ops.get("TopK"))
    if insight_id == "uint8_topk_compact_label_grid":
        return bool(ops.get("TopK") and ops.get("Cast"))
    if insight_id == "pad_compensated_spatial_crop":
        return bool(ops.get("Pad") and ops.get("Slice"))
    if insight_id == "branch_einsum_copy_edit_epilogue":
        return bool(nodes and nodes[-1].op_type == "Einsum" and "output" in nodes[-1].output)
    if insight_id == "threshold_linearize_pairwise_onehot_and":
        return bool(ops.get("Einsum") or ops.get("MatMul") or ops.get("QLinearMatMul"))
    if insight_id == "scan_dtype_and_shift_compression":
        return bool(ops.get("CumSum") or ops.get("ReduceMax") or ops.get("ReduceMin"))
    if insight_id == "sparse_conv_single_op_floor":
        return bool((ops.get("Conv") or ops.get("QLinearConv")) and sum(ops.values()) <= 20)
    if insight_id == "direct_onehot_gather_output":
        return bool(nodes and nodes[-1].op_type in {"Gather", "GatherElements", "GatherND"})
    if insight_id == "spatial_reducesum_to_einsum_profile_tail":
        return any(
            n.op_type == "Einsum"
            and n.input == ["input"]
            and attr_s(n, "equation") == "bchw->bc"
            and shapes.get(n.output[0]) == [1, 10]
            for n in nodes
        )
    return False


def priority(task: dict[str, Any], insight: dict[str, Any], state: str) -> float:
    cost = int(task.get("cost") or 0)
    points = float(task.get("points") or 0.0)
    low_score_bonus = max(0.0, 16.0 - points) * 750.0
    state_bonus = {
        "actionable": 4500.0,
        "candidate_unlogged": 3500.0,
        "candidate_logged": 1500.0,
        "source_candidate": 500.0,
        "graph_evidence": -1500.0,
        "known_blocked": -25000.0,
        "byte_negative": -30000.0,
        "kaggle_falsified": -50000.0,
        "unlowered_semantic_compiler": -8000.0,
        "probe_no_win": -12000.0,
    }.get(state, 0.0)
    risk = str((insight.get("expected") or {}).get("risk") or "").lower()
    risk_penalty = 1500.0 if "high" in risk else 500.0 if "medium" in risk else 0.0
    return cost + low_score_bonus + state_bonus - risk_penalty


def build_tasks(manifest: dict[str, Any]) -> list[dict[str, Any]]:
    rows = {int(row["task"]): row for row in manifest["tasks"]}
    tasks = []
    for task_num in range(1, 401):
        path = ACTIVE / f"task{task_num:03d}.onnx"
        model = onnx.load(path)
        row = rows.get(task_num, {})
        tasklog = read_text(REPORTS / "tasklog" / f"task{task_num:03d}.md")
        ops = op_counts(model)
        tags = active_tags(ops, row, tasklog)
        if any(node.op_type == "Conv" and node.input and node.input[0] == "input" for node in model.graph.node):
            tags.add("free_fp32_input_quantize_required")
        tasks.append(
            {
                "task": task_num,
                "path": str(path.relative_to(ROOT)),
                "model": model,
                "ops": dict(ops),
                "tags": sorted(tags),
                "tasklog": tasklog,
                "cost": row.get("cost"),
                "memory": row.get("memory"),
                "params": row.get("params"),
                "points": row.get("points"),
            }
        )
    return tasks


def probe_no_win_tasks() -> dict[str, set[int]]:
    """Return full-active probe misses that should demote syntax-only gaps.

    These are not permanent walls.  They mean "the current active artifact set
    was probed with the named transformer and no lower-cost candidate survived
    for this task."  A new overlay or graph surgery pass should reopen them.
    """

    out: dict[str, set[int]] = defaultdict(set)

    # The 2026-07-08 active dedupe rerun printed wins=0 / TOTAL +0.000000.
    # Dedupe is pure syntax hygiene; until another overlay creates duplicates,
    # all syntax matches are probe misses rather than actionable gaps.
    out["dedupe_byte_identical_initializers"] = set(range(1, 401))

    zero_wins = load_json_optional(
        ROOT / "reports" / "candidates" / "zero_compare_to_bool_cast" / "wins.json",
        [],
    )
    if zero_wins:
        win_tasks = {int(row["task"]) for row in zero_wins}
        out["zero_compare_to_bool_cast"] = set(range(1, 401)) - win_tasks

    # 2026-07-08: reports/candidates/label_pad_order_active_probe.py scanned
    # the active set for safe sentinel Pad(label)->Equal(output) reorderings and
    # found "safe label-pad candidates seen: 0".  This demotes the registry's
    # broad Pad/Equal syntax hits until a new safe-pattern detector is added or
    # a new overlay creates the exact sentinel form.
    out["label_pad_vs_onehot_pad_ordering"] = set(range(1, 401))
    return out


def build_coverage(tasks: list[dict[str, Any]], registry: list[dict[str, Any]]) -> dict[str, Any]:
    task_rows = []
    insight_rows = []
    gaps = []
    matrix: dict[str, dict[str, str]] = defaultdict(dict)
    probe_misses = probe_no_win_tasks()

    for insight in registry:
        if insight.get("status") != "active":
            continue
        iid = insight["id"]
        counts = Counter()
        top = []
        source_tasks = set(int(t) for t in insight.get("source_tasks", []))
        for task in tasks:
            task_num = int(task["task"])
            is_source = task_num in source_tasks
            is_logged = logged_evidence(task["tasklog"], insight)
            has_graph = graph_evidence(task["model"], iid)
            is_candidate = registry_matches(task, insight)
            if is_source and is_candidate:
                state = "source_candidate"
            elif is_source:
                state = "source"
            elif is_candidate and not is_logged and not has_graph:
                state = "candidate_unlogged"
            elif is_candidate:
                state = "candidate_logged" if is_logged else "candidate_graphevidence"
            elif has_graph:
                state = "graph_evidence"
            elif is_logged:
                state = "logged"
            else:
                state = "none"
            blocked = blocked_state(task["tasklog"], insight, state)
            if blocked:
                state = blocked
            elif state.startswith("candidate") and task_num in probe_misses.get(iid, set()):
                state = "probe_no_win"
            elif state == "candidate_unlogged":
                state = "actionable"
            counts[state] += 1
            matrix[f"task{task_num:03d}"][iid] = state
            if state in {
                "actionable",
                "candidate_unlogged",
                "candidate_logged",
                "candidate_graphevidence",
                "source_candidate",
                "known_blocked",
                "byte_negative",
                "kaggle_falsified",
                "unlowered_semantic_compiler",
                "probe_no_win",
            }:
                score = priority(task, insight, state)
                item = {
                    "task": task_num,
                    "insight_id": iid,
                    "state": state,
                    "priority": round(score, 3),
                    "cost": task.get("cost"),
                    "memory": task.get("memory"),
                    "params": task.get("params"),
                    "points": task.get("points"),
                    "tags": task.get("tags", []),
                    "expected": insight.get("expected", {}),
                }
                top.append(item)
                if state == "actionable":
                    gaps.append(item)
        top.sort(key=lambda x: (-x["priority"], x["task"]))
        insight_rows.append(
            {
                "insight_id": iid,
                "title": insight.get("title"),
                "source_tasks": sorted(source_tasks),
                "counts": dict(counts),
                "top_candidates": top[:20],
            }
        )

    for task in tasks:
        states = Counter(matrix[f"task{int(task['task']):03d}"].values())
        candidate_ids = [
            iid
            for iid, state in matrix[f"task{int(task['task']):03d}"].items()
            if state.startswith("candidate") or state == "actionable"
        ]
        task_rows.append(
            {
                "task": task["task"],
                "cost": task.get("cost"),
                "memory": task.get("memory"),
                "params": task.get("params"),
                "points": task.get("points"),
                "tags": task.get("tags"),
                "state_counts": dict(states),
                "candidate_insights": candidate_ids,
            }
        )
    task_rows.sort(key=lambda x: (-(x.get("cost") or 0), x["task"]))
    gaps.sort(key=lambda x: (-x["priority"], x["task"], x["insight_id"]))
    return {
        "summary": {
            "tasks": len(tasks),
            "active_nets": str(ACTIVE.relative_to(ROOT)),
            "insights": len(insight_rows),
        },
        "insights": insight_rows,
        "tasks": task_rows,
        "top_actionable_gaps": gaps[:120],
        "top_candidate_gaps": gaps[:120],
        "matrix": matrix,
    }


def write_markdown(report: dict[str, Any], path: Path) -> None:
    lines = [
        "# Known Insight Coverage",
        "",
        "Active-overfit audit: `submission/overfit_nets/` x `reports/insight_registry.yaml`.",
        "",
        "States are heuristic triage labels, not proof of applicability.",
        "",
        "## Top Actionable Gaps",
        "",
        "| rank | task | insight | state | priority | cost | pts | tags |",
        "|---:|---:|---|---|---:|---:|---:|---|",
    ]
    for rank, row in enumerate(report["top_actionable_gaps"][:60], 1):
        lines.append(
            f"| {rank} | {row['task']:03d} | `{row['insight_id']}` | {row['state']} | "
            f"{row['priority']:.1f} | {row.get('cost')} | {float(row.get('points') or 0):.3f} | "
            f"{','.join(row.get('tags', [])[:8])} |"
        )
    lines.extend(["", "## Insight Coverage Summary", "", "| insight | source | logged | graph | actionable | blocked | unlowered | probe_no_win | candidate_logged | top tasks |", "|---|---:|---:|---:|---:|---:|---:|---:|---:|---|"])
    for insight in report["insights"]:
        counts = insight["counts"]
        top_tasks = ", ".join(
            f"{row['task']:03d}:{row['state']}:{int(row['priority'])}"
            for row in insight["top_candidates"][:5]
        )
        lines.append(
            f"| `{insight['insight_id']}` | {counts.get('source', 0) + counts.get('source_candidate', 0)} | "
            f"{counts.get('logged', 0)} | {counts.get('graph_evidence', 0) + counts.get('candidate_graphevidence', 0)} | "
            f"{counts.get('actionable', 0)} | "
            f"{counts.get('known_blocked', 0) + counts.get('byte_negative', 0) + counts.get('kaggle_falsified', 0)} | "
            f"{counts.get('unlowered_semantic_compiler', 0)} | {counts.get('probe_no_win', 0)} | "
            f"{counts.get('candidate_logged', 0)} | {top_tasks} |"
        )
    lines.extend(["", "## Highest-Cost Task Coverage", "", "| task | cost | pts | candidate insights | state counts |", "|---:|---:|---:|---|---|"])
    for task in report["tasks"][:40]:
        lines.append(
            f"| {task['task']:03d} | {task.get('cost')} | {float(task.get('points') or 0):.3f} | "
            f"{', '.join('`'+x+'`' for x in task.get('candidate_insights', [])[:8])} | "
            f"{task.get('state_counts')} |"
        )
    lines.append("")
    path.write_text("\n".join(lines))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", default=str(REPORTS / "overfit_manifest.json"))
    parser.add_argument("--registry", default=str(REPORTS / "insight_registry.yaml"))
    parser.add_argument("--out-json", default=str(REPORTS / "known_insight_coverage.json"))
    parser.add_argument("--out-md", default=str(REPORTS / "known_insight_coverage.md"))
    args = parser.parse_args()

    manifest = load_json(Path(args.manifest))
    registry = load_json(Path(args.registry))
    tasks = build_tasks(manifest)
    report = build_coverage(tasks, registry)
    Path(args.out_json).write_text(json.dumps(report, indent=2, sort_keys=True))
    write_markdown(report, Path(args.out_md))
    print(f"wrote {args.out_json}")
    print(f"wrote {args.out_md}")
    print(f"top actionable gaps: {len(report['top_actionable_gaps'])}")
    for row in report["top_actionable_gaps"][:20]:
        print(
            f"{row['task']:03d} {row['insight_id']} {row['state']} "
            f"priority={row['priority']:.1f} cost={row['cost']} pts={float(row.get('points') or 0):.3f}"
        )


if __name__ == "__main__":
    main()
