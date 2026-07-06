#!/usr/bin/env python
"""Search exact-25 one-node candidates with one scalar initializer.

The score formula still returns 25.0 when mem + params == 1, so these candidates
are valid exact-25 probes even though they are not zero-cost.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys
from dataclasses import dataclass

import numpy as np
import onnx
import onnxruntime as ort
from onnx import TensorProto, helper, numpy_helper

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.harness import GRID_SHAPE, IR_VERSION, convert_to_numpy, evaluate, load_task, sanitize_model  # noqa: E402

OUT_JSON = ROOT / "reports" / "exact25_scalar_ops_hits.json"
OUT_MD = ROOT / "reports" / "exact25_scalar_ops_hits.md"


@dataclass(frozen=True)
class Candidate:
    name: str
    op_type: str
    value: float
    output_type: int = TensorProto.FLOAT


def make_model(cand: Candidate) -> onnx.ModelProto:
    init = numpy_helper.from_array(np.array(cand.value, dtype=np.float32), "s")
    node = helper.make_node(cand.op_type, ["input", "s"], ["output"])
    graph = helper.make_graph(
        [node],
        cand.name,
        [helper.make_tensor_value_info("input", TensorProto.FLOAT, GRID_SHAPE)],
        [helper.make_tensor_value_info("output", cand.output_type, GRID_SHAPE)],
        [init],
    )
    return helper.make_model(graph, ir_version=IR_VERSION, opset_imports=[helper.make_opsetid("", 13)])


def candidate_space() -> list[Candidate]:
    vals = [-3.0, -2.0, -1.0, -0.5, -0.01, 0.0, 0.01, 0.5, 1.0, 2.0, 3.0, 10.0]
    c: list[Candidate] = []
    for op in ["Add", "Sub", "Mul", "Div", "Pow", "Max", "Min"]:
        for v in vals:
            c.append(Candidate(f"{op.lower()}_{v:g}", op, v))
    for op in ["Equal", "Greater", "Less", "GreaterOrEqual", "LessOrEqual"]:
        for v in vals:
            c.append(Candidate(f"{op.lower()}_{v:g}", op, v, TensorProto.BOOL))
    for v in vals:
        c.append(Candidate(f"prelu_{v:g}", "PRelu", v))
    return c


def make_session(cand: Candidate) -> ort.InferenceSession | None:
    try:
        sanitized = sanitize_model(make_model(cand))
        if sanitized is None:
            return None
        opts = ort.SessionOptions()
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
        opts.log_severity_level = 3
        sess = ort.InferenceSession(
            sanitized.SerializeToString(),
            opts,
            providers=["CPUExecutionProvider"],
        )
        dummy = np.zeros(GRID_SHAPE, dtype=np.float32)
        out = sess.run(["output"], {"input": dummy})[0]
        if tuple(out.shape) != tuple(GRID_SHAPE):
            return None
        return sess
    except Exception:
        return None


def examples_for(task_num: int):
    task = load_task(task_num)
    return task.get("train", []) + task.get("test", []) + task.get("arc-gen", [])


def verify_candidate(session: ort.InferenceSession, examples) -> tuple[bool, int, int]:
    passed = 0
    failed = 0
    for ex in examples:
        bench = convert_to_numpy(ex)
        if bench is None:
            continue
        try:
            out = session.run(["output"], {"input": bench["input"]})[0]
            pred = (out > 0.0).astype(np.float32)
        except Exception:
            failed += 1
            return False, passed, failed
        if np.array_equal(pred, bench["output"]):
            passed += 1
        else:
            failed += 1
            return False, passed, failed
    return failed == 0 and passed > 0, passed, failed


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tasks", nargs="*", type=int, default=list(range(1, 401)))
    parser.add_argument("--progress-every", type=int, default=50)
    args = parser.parse_args()

    sessions = []
    for cand in candidate_space():
        sess = make_session(cand)
        if sess is not None:
            sessions.append((cand, sess))
    print(f"candidate_models={len(candidate_space())} loadable={len(sessions)} tasks={len(args.tasks)}")

    hits = []
    for task_idx, task_num in enumerate(args.tasks, start=1):
        if args.progress_every and (task_idx == 1 or task_idx % args.progress_every == 0):
            print(f"progress task_index={task_idx}/{len(args.tasks)} task={task_num:03d}", flush=True)
        exs = examples_for(task_num)
        for cand, sess in sessions:
            ok, passed, failed = verify_candidate(sess, exs)
            if not ok:
                continue
            model = make_model(cand)
            scored = evaluate(model, load_task(task_num))
            if not scored["ok"] or scored["points"] != 25.0:
                continue
            rec = {
                "task": task_num,
                "candidate": cand.name,
                "op_type": cand.op_type,
                "value": cand.value,
                "memory": scored["memory"],
                "params": scored["params"],
                "pass": passed,
            }
            hits.append(rec)
            print(f"HIT task{task_num:03d}: {cand.name} mem={scored['memory']} params={scored['params']}")
            break

    OUT_JSON.write_text(json.dumps({"hits": hits}, indent=2) + "\n")
    lines = [
        "# Exact 25 scalar-op search hits",
        "",
        f"- candidates loaded: `{len(sessions)}`",
        "",
        "| task | op | candidate | scalar | mem | params | pass |",
        "|---:|---|---|---:|---:|---:|---:|",
    ]
    for h in hits:
        lines.append(
            f"| {h['task']:03d} | {h['op_type']} | `{h['candidate']}` | {h['value']} | "
            f"{h['memory']} | {h['params']} | {h['pass']} |"
        )
    OUT_MD.write_text("\n".join(lines) + "\n")
    print(f"wrote {OUT_JSON}")
    print(f"unique_hit_tasks={len({h['task'] for h in hits})}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
