"""Lossless/equivalent compression sweep for NeuroGolf ONNX networks.

The script tries conservative graph-level compression strategies and adopts only
when the candidate:
  1. passes official stored evaluation,
  2. scores higher than the current network, and
  3. has thresholded outputs identical to the current network on stored examples
     plus a configurable fresh sample.

Usage:
  PYTHONPATH=. .venv/bin/python reports/scripts/compression_sweep.py --top 80 --fresh 200
  PYTHONPATH=. .venv/bin/python reports/scripts/compression_sweep.py --top 120 --fresh 500 --adopt
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
from onnxsim import simplify

from src.harness import ROOT, convert_to_numpy, evaluate, load_task
from src.genverify import load_gen

NETWORKS = ROOT / "networks"
MANIFEST = ROOT / "reports" / "manifest.json"
OUT = Path("/tmp/neurogolf_compress")
SUMMARY = ROOT / "reports" / "compression_sweep.json"


def load_manifest() -> dict:
    return json.loads(MANIFEST.read_text())["tasks"]


def save_manifest(tasks: dict) -> None:
    MANIFEST.write_text(json.dumps({"tasks": tasks}, indent=1) + "\n")


def task_order(top: int | None) -> list[int]:
    tasks = load_manifest()
    rows = []
    for key, rec in tasks.items():
        if not rec:
            continue
        mem = int(rec.get("memory", 0))
        params = int(rec.get("params", 0))
        points = float(rec.get("points", 0.0))
        rows.append((points, -(mem + params), int(key)))
    rows.sort()
    nums = [num for _, _, num in rows]
    return nums[:top] if top else nums


def make_session(path: Path) -> ort.InferenceSession:
    so = ort.SessionOptions()
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    return ort.InferenceSession(str(path), so)


def threshold_output(session: ort.InferenceSession, example: dict) -> np.ndarray | None:
    bm = convert_to_numpy(example)
    if bm is None:
        return None
    out = session.run(["output"], {"input": bm["input"].astype(np.float32)})[0]
    return (out > 0).astype(np.uint8)


def examples_for_task(task_num: int, fresh_n: int) -> list[dict]:
    task = load_task(task_num)
    examples = list(task.get("train", [])) + list(task.get("test", []))
    if fresh_n <= 0:
        return examples
    try:
        gen = load_gen(task_num)
    except Exception:
        return examples
    run = tries = 0
    while run < fresh_n and tries < fresh_n * 8:
        tries += 1
        try:
            ex = gen.generate()
        except Exception:
            continue
        if max(len(ex["input"]), len(ex["input"][0]), len(ex["output"]), len(ex["output"][0])) > 30:
            continue
        examples.append(ex)
        run += 1
    return examples


def equivalent_on_samples(task_num: int, base_path: Path, cand_path: Path, fresh_n: int) -> tuple[bool, int, int]:
    base_sess = make_session(base_path)
    cand_sess = make_session(cand_path)
    equal = total = 0
    for ex in examples_for_task(task_num, fresh_n):
        base = threshold_output(base_sess, ex)
        cand = threshold_output(cand_sess, ex)
        if base is None or cand is None:
            continue
        total += 1
        if np.array_equal(base, cand):
            equal += 1
        else:
            return False, equal, total
    return total > 0 and equal == total, equal, total


def simplify_candidate(path: Path) -> tuple[str, onnx.ModelProto] | None:
    try:
        model = onnx.load(path)
        sim, ok = simplify(model)
    except Exception:
        return None
    if not ok:
        return None
    return "onnxsim", sim


def candidate_generators(path: Path) -> list[tuple[str, onnx.ModelProto]]:
    out: list[tuple[str, onnx.ModelProto]] = []
    sim = simplify_candidate(path)
    if sim is not None:
        out.append(sim)
    return out


def evaluate_path(path: Path, task_num: int) -> dict:
    return evaluate(path, load_task(task_num))


def sweep_one(task_num: int, fresh_n: int, adopt: bool) -> list[dict]:
    base_path = NETWORKS / f"task{task_num:03d}.onnx"
    if not base_path.exists():
        return []
    OUT.mkdir(parents=True, exist_ok=True)
    base_ev = evaluate_path(base_path, task_num)
    if not base_ev["ok"]:
        return [{
            "task": task_num,
            "strategy": "base_eval",
            "error": base_ev.get("error") or "base eval failed",
        }]

    rows = []
    for strategy, model in candidate_generators(base_path):
        cand_path = OUT / f"task{task_num:03d}_{strategy}.onnx"
        onnx.save(model, cand_path)
        cand_ev = evaluate_path(cand_path, task_num)
        rec = {
            "task": task_num,
            "strategy": strategy,
            "candidate_path": str(cand_path),
            "base_points": base_ev["points"],
            "base_memory": base_ev["memory"],
            "base_params": base_ev["params"],
            "candidate_ok": cand_ev["ok"],
            "candidate_points": cand_ev["points"],
            "candidate_memory": cand_ev["memory"],
            "candidate_params": cand_ev["params"],
            "candidate_error": cand_ev.get("error"),
            "equivalent": False,
            "equivalent_samples": "0/0",
            "adoptable": False,
            "adopted": False,
        }
        if cand_ev["ok"] and cand_ev["points"] > base_ev["points"] + 1e-9:
            eq, good, total = equivalent_on_samples(task_num, base_path, cand_path, fresh_n)
            rec["equivalent"] = eq
            rec["equivalent_samples"] = f"{good}/{total}"
            rec["adoptable"] = eq
            if adopt and eq:
                shutil.copyfile(cand_path, base_path)
                rec["adopted"] = True
        rows.append(rec)
    return rows


def update_manifest_for_adoptions(rows: list[dict]) -> None:
    tasks = load_manifest()
    changed = False
    for rec in rows:
        if not rec.get("adopted"):
            continue
        task = int(rec["task"])
        tasks[str(task)] = {
            "points": rec["candidate_points"],
            "memory": rec["candidate_memory"],
            "params": rec["candidate_params"],
            "method": f"{tasks[str(task)].get('method', 'unknown')}+{rec['strategy']}",
        }
        changed = True
    if changed:
        save_manifest(tasks)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--top", type=int, default=80)
    parser.add_argument("--fresh", type=int, default=200)
    parser.add_argument("--tasks", nargs="*", type=int)
    parser.add_argument("--adopt", action="store_true")
    args = parser.parse_args()

    nums = args.tasks or task_order(args.top)
    all_rows: list[dict] = []
    for i, task_num in enumerate(nums, 1):
        rows = sweep_one(task_num, args.fresh, args.adopt)
        all_rows.extend(rows)
        for rec in rows:
            gain = rec.get("candidate_points", 0.0) - rec.get("base_points", 0.0)
            status = "ADOPT" if rec.get("adopted") else ("KEEP" if rec.get("adoptable") else "skip")
            print(
                f"[{i}/{len(nums)}] task{task_num:03d} {rec.get('strategy')} "
                f"gain={gain:.4f} eq={rec.get('equivalent_samples')} {status}",
                flush=True,
            )
    if args.adopt:
        update_manifest_for_adoptions(all_rows)
    SUMMARY.write_text(json.dumps(all_rows, indent=2, sort_keys=True) + "\n")
    adopted = [r for r in all_rows if r.get("adopted")]
    print(f"wrote {SUMMARY}; adopted={len(adopted)}")


if __name__ == "__main__":
    main()
