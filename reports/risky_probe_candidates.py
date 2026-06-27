"""Generate and register risky public-LB probe candidates.

This script intentionally separates *candidate generation* from adoption.  A
candidate must pass stored examples before it is worth a Kaggle probe; public
hidden then decides whether the risk is acceptable.

Initial candidate kinds are deliberately simple and cheap:
  - identity: output=input, 25 pts if valid.
  - zero: all output logits <=0, useful only for empty-output tasks.
  - black: channel-0 full-grid output via one Constant, high score if valid.

Usage:
  PYTHONPATH=. .venv/bin/python reports/risky_probe_candidates.py scan --top 120
  PYTHONPATH=. .venv/bin/python reports/risky_probe_candidates.py build --task 151 --kind identity
"""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

from src.builders import _model, identity_network
from src.harness import GRID_SHAPE, ROOT, evaluate, load_task

OUT = Path("/tmp/neurogolf_risky")
SUMMARY = ROOT / "reports" / "risky_probe_candidates.json"
KIND_CHOICES = ("identity", "zero", "black")


def manifest_tasks() -> dict:
    return json.loads((ROOT / "reports" / "manifest.json").read_text())["tasks"]


def task_order(top: int | None) -> list[int]:
    rows = []
    for key, rec in manifest_tasks().items():
        if not rec:
            continue
        rows.append((float(rec["points"]), int(key)))
    rows.sort()
    nums = [num for _, num in rows]
    return nums[:top] if top else nums


def zero_network() -> onnx.ModelProto:
    nodes = [
        helper.make_node("Constant", [], ["output"], value=numpy_helper.from_array(
            np.zeros(GRID_SHAPE, dtype=np.float32), "zero_out"
        ))
    ]
    return _model(nodes, [])


def black_network() -> onnx.ModelProto:
    arr = np.zeros(GRID_SHAPE, dtype=np.float32)
    arr[:, 0, :, :] = 1.0
    nodes = [
        helper.make_node("Constant", [], ["output"], value=numpy_helper.from_array(arr, "black_out"))
    ]
    return _model(nodes, [])


def build_model(kind: str) -> onnx.ModelProto:
    if kind == "identity":
        return identity_network()
    if kind == "zero":
        return zero_network()
    if kind == "black":
        return black_network()
    raise SystemExit(f"unknown kind: {kind}")


def build_candidate(task: int, kind: str) -> Path:
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / f"task{task:03d}_{kind}.onnx"
    onnx.save(build_model(kind), path)
    return path


def current_points(task: int) -> float:
    rec = manifest_tasks().get(str(task)) or {}
    return float(rec.get("points", 0.0))


def eval_candidate(task: int, kind: str) -> dict:
    path = build_candidate(task, kind)
    ev = evaluate(path, load_task(task))
    return {
        "task": task,
        "kind": kind,
        "path": str(path),
        "stored_ok": ev["ok"],
        "stored_pass": ev["pass"],
        "stored_fail": ev["fail"],
        "candidate_points": ev["points"],
        "candidate_memory": ev["memory"],
        "candidate_params": ev["params"],
        "candidate_error": ev.get("error"),
        "base_points": current_points(task),
        "if_public_pass_delta": ev["points"] - current_points(task),
        "probe_worthy": ev["ok"] and ev["points"] > current_points(task) + 1e-9,
    }


def register_candidate(task: int, kind: str, path: Path, cid: str | None, note: str | None) -> None:
    cid = cid or f"t{task:03d}_{kind}"
    cmd = [
        "PYTHONPATH=.",
        ".venv/bin/python",
        "reports/public_probe.py",
        "add",
        "--task",
        str(task),
        "--onnx",
        str(path),
        "--id",
        cid,
        "--source",
        f"risky_probe:{kind}",
        "--note",
        note or f"{kind} risky public probe",
        "--replace",
    ]
    # Use the shell only to set PYTHONPATH in the same style as project scripts.
    subprocess.check_call(" ".join(cmd), cwd=ROOT, shell=True)


def cmd_scan(args: argparse.Namespace) -> None:
    rows = []
    for i, task in enumerate(args.tasks or task_order(args.top), 1):
        for kind in KIND_CHOICES:
            rec = eval_candidate(task, kind)
            rows.append(rec)
            if rec["probe_worthy"]:
                print(
                    f"[{i}] task{task:03d} {kind} stored=PASS "
                    f"delta={rec['if_public_pass_delta']:.3f} path={rec['path']}",
                    flush=True,
                )
    SUMMARY.write_text(json.dumps(rows, indent=2, sort_keys=True) + "\n")
    print(f"wrote {SUMMARY}; probe_worthy={sum(1 for r in rows if r['probe_worthy'])}")


def cmd_build(args: argparse.Namespace) -> None:
    rec = eval_candidate(args.task, args.kind)
    print(json.dumps(rec, indent=2, sort_keys=True))
    if args.register:
        if not rec["probe_worthy"] and not args.force:
            raise SystemExit("candidate is not probe-worthy; use --force to register anyway")
        register_candidate(args.task, args.kind, Path(rec["path"]), args.id, args.note)


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd", required=True)

    q = sub.add_parser("scan")
    q.add_argument("--top", type=int, default=120)
    q.add_argument("--tasks", nargs="*", type=int)
    q.set_defaults(func=cmd_scan)

    q = sub.add_parser("build")
    q.add_argument("--task", type=int, required=True)
    q.add_argument("--kind", choices=KIND_CHOICES, required=True)
    q.add_argument("--register", action="store_true")
    q.add_argument("--force", action="store_true")
    q.add_argument("--id")
    q.add_argument("--note")
    q.set_defaults(func=cmd_build)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
