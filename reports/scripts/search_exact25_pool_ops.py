#!/usr/bin/env python
"""Search one-node exact-25 Pool candidates with asymmetric free attributes."""

from __future__ import annotations

import argparse
import json
import pathlib
import sys
from dataclasses import dataclass

import numpy as np
import onnxruntime as ort
from onnx import TensorProto, helper

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.harness import GRID_SHAPE, IR_VERSION, convert_to_numpy, load_task, sanitize_model  # noqa: E402

OUT_JSON = ROOT / "reports" / "exact25_pool_ops_hits.json"
OUT_MD = ROOT / "reports" / "exact25_pool_ops_hits.md"


@dataclass(frozen=True)
class Candidate:
    name: str
    op_type: str
    opset: int
    attrs: tuple[tuple[str, object], ...]


def make_model(cand: Candidate):
    node = helper.make_node(cand.op_type, ["input"], ["output"], **dict(cand.attrs))
    graph = helper.make_graph(
        [node],
        cand.name,
        [helper.make_tensor_value_info("input", TensorProto.FLOAT, GRID_SHAPE)],
        [helper.make_tensor_value_info("output", TensorProto.FLOAT, GRID_SHAPE)],
        [],
    )
    return helper.make_model(graph, ir_version=IR_VERSION, opset_imports=[helper.make_opsetid("", cand.opset)])


def candidate_space(max_kernel: int, max_dilation: int) -> list[Candidate]:
    c: list[Candidate] = []
    for op, opset in [("MaxPool", 10), ("AveragePool", 10), ("LpPool", 11)]:
        for kh in range(1, max_kernel + 1):
            for kw in range(1, max_kernel + 1):
                for dh in range(1, max_dilation + 1):
                    for dw in range(1, max_dilation + 1):
                        eff_h = (kh - 1) * dh + 1
                        eff_w = (kw - 1) * dw + 1
                        total_h = eff_h - 1
                        total_w = eff_w - 1
                        for top in range(total_h + 1):
                            bottom = total_h - top
                            if top >= kh or bottom >= kh:
                                continue
                            for left in range(total_w + 1):
                                right = total_w - left
                                if left >= kw or right >= kw:
                                    continue
                                attrs: list[tuple[str, object]] = [
                                    ("kernel_shape", [kh, kw]),
                                    ("pads", [top, left, bottom, right]),
                                    ("strides", [1, 1]),
                                ]
                                if op in {"MaxPool", "LpPool"}:
                                    attrs.append(("dilations", [dh, dw]))
                                c.append(
                                    Candidate(
                                        f"{op.lower()}_k{kh}x{kw}_d{dh}x{dw}_p{top}_{left}_{bottom}_{right}",
                                        op,
                                        opset,
                                        tuple(attrs),
                                    )
                                )
    return c


def make_session(cand: Candidate) -> ort.InferenceSession | None:
    try:
        sanitized = sanitize_model(make_model(cand))
        if sanitized is None:
            return None
        opts = ort.SessionOptions()
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
        opts.log_severity_level = 3
        sess = ort.InferenceSession(sanitized.SerializeToString(), opts, providers=["CPUExecutionProvider"])
        out = sess.run(["output"], {"input": np.zeros(GRID_SHAPE, dtype=np.float32)})[0]
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
    parser.add_argument("--max-kernel", type=int, default=9)
    parser.add_argument("--max-dilation", type=int, default=2)
    parser.add_argument("--progress-every", type=int, default=25)
    args = parser.parse_args()

    cands = candidate_space(args.max_kernel, args.max_dilation)
    sessions = []
    for i, cand in enumerate(cands, start=1):
        sess = make_session(cand)
        if sess is not None:
            sessions.append((cand, sess))
        if i % 1000 == 0:
            print(f"loaded_probe {i}/{len(cands)} kept={len(sessions)}", flush=True)
    print(f"candidate_models={len(cands)} loadable={len(sessions)} tasks={len(args.tasks)}")

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
        "# Exact 25 pool-op search hits",
        "",
        f"- candidates loaded: `{len(sessions)}`",
        "",
        "| task | op | candidate | attrs | pass |",
        "|---:|---|---|---|---:|",
    ]
    for h in hits:
        lines.append(
            f"| {h['task']:03d} | {h['op_type']} | `{h['candidate']}` | "
            f"`{json.dumps(h['attrs'], sort_keys=True)}` | {h['pass']} |"
        )
    OUT_MD.write_text("\n".join(lines) + "\n")
    print(f"wrote {OUT_JSON}")
    print(f"unique_hit_tasks={len({h['task'] for h in hits})}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
