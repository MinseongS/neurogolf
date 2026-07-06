#!/usr/bin/env python
"""Search exact-25 one-node candidates that use only free ONNX attributes.

This complements search_exact25_zero_cost.py.  It is deliberately conservative:
every candidate has no initializers and writes the benchmark output directly, so
any verified hit should score 25 under the local harness.
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
from onnx import TensorProto, helper

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.harness import GRID_SHAPE, IR_VERSION, convert_to_numpy, load_task, sanitize_model  # noqa: E402

OUT_JSON = ROOT / "reports" / "exact25_attr_ops_hits.json"
OUT_MD = ROOT / "reports" / "exact25_attr_ops_hits.md"


@dataclass(frozen=True)
class Candidate:
    name: str
    op_type: str
    opset: int
    attrs: tuple[tuple[str, object], ...] = ()
    inputs: tuple[str, ...] = ("input",)


def make_model(cand: Candidate) -> onnx.ModelProto:
    node = helper.make_node(cand.op_type, list(cand.inputs), ["output"], **dict(cand.attrs))
    graph = helper.make_graph(
        [node],
        cand.name,
        [helper.make_tensor_value_info("input", TensorProto.FLOAT, GRID_SHAPE)],
        [helper.make_tensor_value_info("output", TensorProto.FLOAT, GRID_SHAPE)],
        [],
    )
    return helper.make_model(
        graph,
        ir_version=IR_VERSION,
        opset_imports=[helper.make_opsetid("", cand.opset)],
    )


def candidate_space() -> list[Candidate]:
    c: list[Candidate] = []
    for op in [
        "Identity",
        "Abs",
        "Relu",
        "Sign",
        "Floor",
        "Ceil",
        "Tanh",
        "Sigmoid",
        "Exp",
        "Neg",
        "Reciprocal",
        "Sqrt",
        "Log",
    ]:
        c.append(Candidate(op.lower(), op, 10))

    for axis in [0, 1, 2, 3, -1, -2, -3, -4]:
        c.append(Candidate(f"hardmax_axis_{axis}", "Hardmax", 13, (("axis", axis),)))

    for axes in ([0], [1], [2], [3], [2, 3], [1, 2, 3], [0, 2, 3]):
        c.append(Candidate(f"mvn_axes_{'_'.join(map(str, axes))}", "MeanVarianceNormalization", 13, (("axes", axes),)))

    for block in [1, 2, 3, 5, 6, 10, 15, 30]:
        c.append(Candidate(f"depth_to_space_{block}", "DepthToSpace", 13, (("blocksize", block),)))
        c.append(Candidate(f"space_to_depth_{block}", "SpaceToDepth", 13, (("blocksize", block),)))

    # Legacy attr Pad is free under calculate_params.  Shape-preserving pads
    # include identity and crop/extend probes if the runtime accepts negatives.
    pad_vectors = [
        [0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 1, 0, 0, 0, -1, 0],
        [0, 0, -1, 0, 0, 0, 1, 0],
        [0, 0, 0, 1, 0, 0, 0, -1],
        [0, 0, 0, -1, 0, 0, 0, 1],
        [0, 0, 1, 1, 0, 0, -1, -1],
        [0, 0, -1, -1, 0, 0, 1, 1],
    ]
    for pads in pad_vectors:
        c.append(Candidate(f"pad2_{'_'.join(map(str, pads))}", "Pad", 10, (("pads", pads), ("mode", "constant"), ("value", 0.0))))

    # Pool ops can be direct shape-preserving morphological transforms when
    # pads/strides/kernel are attributes.
    for op, opset in [("MaxPool", 10), ("AveragePool", 10), ("LpPool", 11)]:
        for k in range(1, 8):
            for dilation in [1, 2, 3]:
                eff = (k - 1) * dilation + 1
                if eff % 2 == 0:
                    continue
                p = eff // 2
                attrs: list[tuple[str, object]] = [
                    ("kernel_shape", [k, k]),
                    ("pads", [p, p, p, p]),
                    ("strides", [1, 1]),
                ]
                if op != "AveragePool":
                    attrs.append(("dilations", [dilation, dilation]))
                c.append(Candidate(f"{op.lower()}_k{k}_d{dilation}", op, opset, tuple(attrs)))

    # Input-independent full-canvas constants/noise.  These are mostly sanity
    # probes for scorer blind spots; verified exact tasks would be unusual.
    for op in ["RandomUniform", "RandomNormal"]:
        c.append(Candidate(f"{op.lower()}_shape", op, 10, (("shape", GRID_SHAPE), ("dtype", TensorProto.FLOAT)), ()))

    return c


def make_session(cand: Candidate) -> ort.InferenceSession | None:
    try:
        model = make_model(cand)
        sanitized = sanitize_model(model)
        if sanitized is None:
            return None
        opts = ort.SessionOptions()
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
        opts.log_severity_level = 3
        return ort.InferenceSession(
            sanitized.SerializeToString(),
            opts,
            providers=["CPUExecutionProvider"],
        )
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
    load_fail = []
    bad_shape = []
    for cand in candidate_space():
        sess = make_session(cand)
        if sess is None:
            load_fail.append(cand.name)
        else:
            try:
                dummy = np.zeros(GRID_SHAPE, dtype=np.float32)
                out = sess.run(["output"], {"input": dummy} if cand.inputs else {})[0]
            except Exception:
                load_fail.append(cand.name)
                continue
            if tuple(out.shape) != tuple(GRID_SHAPE):
                bad_shape.append(cand.name)
                continue
            sessions.append((cand, sess))
    print(f"candidate_models={len(sessions) + len(load_fail)} loadable={len(sessions)} tasks={len(args.tasks)}")
    if load_fail:
        print("load_failed=" + ",".join(load_fail[:40]) + ("..." if len(load_fail) > 40 else ""))
    if bad_shape:
        print("bad_shape=" + ",".join(bad_shape[:40]) + ("..." if len(bad_shape) > 40 else ""))

    hits = []
    for task_idx, task_num in enumerate(args.tasks, start=1):
        if args.progress_every and (task_idx == 1 or task_idx % args.progress_every == 0):
            print(f"progress task_index={task_idx}/{len(args.tasks)} task={task_num:03d}", flush=True)
        exs = examples_for(task_num)
        for cand, sess in sessions:
            ok, passed, failed = verify_candidate(sess, exs)
            if ok:
                rec = {
                    "task": task_num,
                    "candidate": cand.name,
                    "op_type": cand.op_type,
                    "opset": cand.opset,
                    "attrs": dict(cand.attrs),
                    "pass": passed,
                    "fail": failed,
                }
                hits.append(rec)
                print(f"HIT task{task_num:03d}: {cand.name} {dict(cand.attrs)}")
                break

    OUT_JSON.write_text(json.dumps({"hits": hits}, indent=2) + "\n")
    lines = [
        "# Exact 25 attr-op search hits",
        "",
        f"- candidates loaded: `{len(sessions)}`",
        "",
        "| task | op | opset | candidate | attrs | pass |",
        "|---:|---|---:|---|---|---:|",
    ]
    for h in hits:
        lines.append(
            f"| {h['task']:03d} | {h['op_type']} | {h['opset']} | `{h['candidate']}` | "
            f"`{json.dumps(h['attrs'], sort_keys=True)}` | {h['pass']} |"
        )
    OUT_MD.write_text("\n".join(lines) + "\n")
    print(f"wrote {OUT_JSON}")
    print(f"unique_hit_tasks={len({h['task'] for h in hits})}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
