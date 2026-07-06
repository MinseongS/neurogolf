#!/usr/bin/env python
"""Search all tasks for exact 25-point zero-cost one-node replacements.

This is intentionally brute-force and source-safe: it does not edit
src/custom/ or networks/.  It builds tiny ONNX candidates with no initializers
and exactly one output node, then verifies them against bundled examples.

The first grammar targets the task067 class:

  Einsum(input, input) -> output

where one operand carries the output one-hot value and the other operand(s)
act as axis-activity gates by summing over unused labels.  It also includes
Identity / spatial Transpose baselines, which should rediscover 179/241.
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

OUT_JSON = ROOT / "reports" / "exact25_zero_cost_hits.json"
OUT_MD = ROOT / "reports" / "exact25_zero_cost_hits.md"


@dataclass(frozen=True)
class Candidate:
    name: str
    op_type: str
    attrs: tuple[tuple[str, object], ...] = ()


def make_model(cand: Candidate) -> onnx.ModelProto:
    attrs = dict(cand.attrs)
    if cand.op_type == "Identity":
        node = helper.make_node("Identity", ["input"], ["output"])
    elif cand.op_type == "Transpose":
        node = helper.make_node("Transpose", ["input"], ["output"], **attrs)
    elif cand.op_type == "Einsum":
        n_inputs = int(attrs.pop("n_inputs"))
        node = helper.make_node("Einsum", ["input"] * n_inputs, ["output"], **attrs)
    else:
        raise ValueError(cand.op_type)

    graph = helper.make_graph(
        [node],
        cand.name,
        [helper.make_tensor_value_info("input", TensorProto.FLOAT, GRID_SHAPE)],
        [helper.make_tensor_value_info("output", TensorProto.FLOAT, GRID_SHAPE)],
        [],
    )
    return helper.make_model(graph, ir_version=IR_VERSION, opset_imports=[helper.make_opsetid("", 12)])


def candidate_space(max_inputs: int, allow_diagonal: bool) -> list[Candidate]:
    cands: list[Candidate] = [
        Candidate("identity", "Identity"),
        Candidate("transpose_hw", "Transpose", (("perm", [0, 1, 3, 2]),)),
        Candidate("einsum_identity", "Einsum", (("equation", "bkrc->bkrc"), ("n_inputs", 1))),
        Candidate("einsum_transpose_hw", "Einsum", (("equation", "bkcr->bkrc"), ("n_inputs", 1))),
    ]

    # Input shape labels are [batch, channel, row, col].
    # Batch must stay b.  Channel may carry output k or be reduced (l/m).
    # Spatial axes may carry output r/c or be reduced (p/q/s/t).  Row/col are
    # both 30, so using output c on the row axis is legal and is exactly the
    # task067 trick.
    input_terms = []
    for ch in ("k", "l"):
        for row in ("r", "c", "p"):
            for col in ("r", "c", "q"):
                term = "b" + ch + row + col
                # Repeated spatial labels inside one operand create diagonals.
                # They are valid because row/col are both length 30, but are
                # kept optional because they enlarge the search.
                if allow_diagonal or len(set(term)) == len(term):
                    input_terms.append(term)

    seen_eq = set()
    def add_einsum(terms: tuple[str, ...]) -> None:
        union = set("".join(terms))
        if not set("bkrc") <= union:
            return
        eq = f"{','.join(terms)}->bkrc"
        if eq in seen_eq:
            return
        seen_eq.add(eq)
        cands.append(
            Candidate(
                f"einsum{len(terms)}_{eq}",
                "Einsum",
                (("equation", eq), ("n_inputs", len(terms))),
            )
        )

    if max_inputs >= 2:
        for a in input_terms:
            for b in input_terms:
                add_einsum((a, b))

    if max_inputs >= 3:
        for a in input_terms:
            for b in input_terms:
                for c in input_terms:
                    add_einsum((a, b, c))

    if max_inputs >= 4:
        for a in input_terms:
            for b in input_terms:
                for c in input_terms:
                    for d in input_terms:
                        add_einsum((a, b, c, d))

    return cands


def make_session(model: onnx.ModelProto) -> ort.InferenceSession | None:
    try:
        sanitized = sanitize_model(model)
        if sanitized is None:
            return None
        opts = ort.SessionOptions()
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
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


def run_candidate_numpy(cand: Candidate, x: np.ndarray, optimize: bool | str = False) -> np.ndarray:
    attrs = dict(cand.attrs)
    if cand.op_type == "Identity":
        return x
    if cand.op_type == "Transpose":
        return np.transpose(x, attrs["perm"])
    if cand.op_type == "Einsum":
        eq = attrs["equation"]
        n_inputs = int(attrs["n_inputs"])
        return np.einsum(eq, *([x] * n_inputs), optimize=optimize)
    raise ValueError(cand.op_type)


def verify_candidate_numpy(cand: Candidate, examples, optimize: bool | str = False) -> tuple[bool, int, int]:
    passed = 0
    failed = 0
    for ex in examples:
        bench = convert_to_numpy(ex)
        if bench is None:
            continue
        try:
            out = run_candidate_numpy(cand, bench["input"], optimize=optimize)
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


def cached_examples_for(task_num: int) -> list[tuple[np.ndarray, np.ndarray]]:
    cached = []
    for ex in examples_for(task_num):
        bench = convert_to_numpy(ex)
        if bench is not None:
            cached.append((bench["input"], bench["output"]))
    return cached


def verify_candidate_numpy_cached(cand: Candidate, cached, optimize: bool | str = False) -> tuple[bool, int, int]:
    passed = 0
    failed = 0
    for x, y in cached:
        try:
            out = run_candidate_numpy(cand, x, optimize=optimize)
            pred = (out > 0.0).astype(np.float32)
        except Exception:
            failed += 1
            return False, passed, failed
        if np.array_equal(pred, y):
            passed += 1
        else:
            failed += 1
            return False, passed, failed
    return failed == 0 and passed > 0, passed, failed


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--include-triples", action="store_true", help="compat alias for --max-inputs 3")
    parser.add_argument("--max-inputs", type=int, default=2)
    parser.add_argument("--allow-diagonal", action="store_true")
    parser.add_argument("--numpy-fast", action="store_true")
    parser.add_argument("--einsum-optimize", default="false", choices=["false", "true", "greedy", "optimal"])
    parser.add_argument("--candidate-start", type=int, default=0)
    parser.add_argument("--candidate-limit", type=int, default=0)
    parser.add_argument("--progress-every", type=int, default=25)
    parser.add_argument("--tasks", nargs="*", type=int, default=list(range(1, 401)))
    parser.add_argument("--stop-after", type=int, default=0)
    args = parser.parse_args()
    optimize: bool | str
    if args.einsum_optimize == "false":
        optimize = False
    elif args.einsum_optimize == "true":
        optimize = True
    else:
        optimize = args.einsum_optimize

    max_inputs = max(args.max_inputs, 3 if args.include_triples else 2)
    cands = candidate_space(max_inputs=max_inputs, allow_diagonal=args.allow_diagonal)
    if args.candidate_start or args.candidate_limit:
        end = None if not args.candidate_limit else args.candidate_start + args.candidate_limit
        cands = cands[args.candidate_start:end]
    sessions = []
    if not args.numpy_fast:
        for cand in cands:
            sess = make_session(make_model(cand))
            if sess is not None:
                sessions.append((cand, sess))
        print(
            f"candidate_models={len(cands)} loadable={len(sessions)} tasks={len(args.tasks)} "
            f"max_inputs={max_inputs} diagonal={int(args.allow_diagonal)}"
        )
    else:
        print(
            f"candidate_models={len(cands)} numpy_fast=1 tasks={len(args.tasks)} "
            f"max_inputs={max_inputs} diagonal={int(args.allow_diagonal)}"
        )

    hits = []
    for task_idx, task_num in enumerate(args.tasks, start=1):
        if args.progress_every and (task_idx == 1 or task_idx % args.progress_every == 0):
            print(f"progress task_index={task_idx}/{len(args.tasks)} task={task_num:03d}", flush=True)
        exs = examples_for(task_num)
        cached = cached_examples_for(task_num) if args.numpy_fast else None
        task_hits = []
        iterable = [(cand, None) for cand in cands] if args.numpy_fast else sessions
        for cand, sess in iterable:
            if args.numpy_fast:
                ok, passed, failed = verify_candidate_numpy_cached(cand, cached, optimize=optimize)
            else:
                ok, passed, failed = verify_candidate(sess, exs)
            if ok:
                rec = {
                    "task": task_num,
                    "candidate": cand.name,
                    "op_type": cand.op_type,
                    "attrs": dict(cand.attrs),
                    "pass": passed,
                    "fail": failed,
                }
                task_hits.append(rec)
                hits.append(rec)
                # The first hit is enough to identify the task; keep scanning
                # only if different equations are useful for later analysis.
                break
        if task_hits:
            best = task_hits[0]
            print(f"HIT task{task_num:03d}: {best['candidate']} {best['attrs']}")
        if args.stop_after and len({h["task"] for h in hits}) >= args.stop_after:
            break

    OUT_JSON.write_text(json.dumps({"hits": hits}, indent=2) + "\n")
    lines = [
        "# Exact 25 zero-cost search hits",
        "",
        f"- candidates tested: `{len(cands) if args.numpy_fast else len(sessions)}`",
        f"- numpy fast path: `{int(args.numpy_fast)}`",
        "",
    ]
    lines.append("| task | op | candidate | attrs | pass |")
    lines.append("|---:|---|---|---|---:|")
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
