"""Merge external network collections into networks/ via keep-best.

Every candidate is re-verified with the official-mirror harness before it can
replace anything, so a bad or mislabeled external network can never make the
submission worse.

Usage: python -m src.merge_external DIR[:label] [DIR:label ...]
Each DIR holds taskXXX.onnx files.
"""

import json
import multiprocessing
import pathlib
import shutil
import sys

from .harness import ROOT, evaluate, load_task
from .pipeline import MANIFEST, NETWORKS, load_manifest, write_scoreboard


def eval_one(job):
    task_num, path, label = job
    try:
        ev = evaluate(path, load_task(task_num))
    except Exception as e:
        return task_num, path, label, None, f"crash: {e}"
    if not ev["ok"]:
        return task_num, path, label, None, ev["error"] or f"{ev['fail']} fail"
    return task_num, path, label, ev, None


def main():
    sources = []
    for arg in sys.argv[1:]:
        d, _, label = arg.partition(":")
        sources.append((pathlib.Path(d), label or pathlib.Path(d).name))

    manifest = load_manifest()
    jobs = []
    for d, label in sources:
        for f in sorted(d.glob("task*.onnx")):
            try:
                num = int(f.stem[4:])
            except ValueError:
                continue
            if 1 <= num <= 400:
                jobs.append((num, f, label))
    print(f"{len(jobs)} candidate networks from {len(sources)} sources")

    improved, rejected = 0, 0
    per_source = {}
    with multiprocessing.Pool(10) as pool:
        for task_num, path, label, ev, err in pool.imap_unordered(eval_one, jobs, chunksize=4):
            if ev is None:
                rejected += 1
                continue
            cur = manifest.get(task_num)
            cur_pts = cur["points"] if cur else 0.0
            if ev["points"] > cur_pts + 1e-9:
                shutil.copy(path, NETWORKS / f"task{task_num:03d}.onnx")
                manifest[task_num] = {
                    "points": ev["points"], "memory": ev["memory"],
                    "params": ev["params"], "method": f"ext:{label}",
                }
                improved += 1
                per_source[label] = per_source.get(label, 0) + 1
                print(f"task{task_num:03d}: {cur_pts:.2f} -> {ev['points']:.2f} ({label})",
                      flush=True)

    with open(MANIFEST, "w") as f:
        json.dump({"tasks": {str(k): v for k, v in sorted(manifest.items())}}, f, indent=1)
    total, solved = write_scoreboard(manifest)
    print(f"\nimproved {improved}, rejected {rejected} (failed verification)")
    for label, n in sorted(per_source.items(), key=lambda kv: -kv[1]):
        print(f"  {label}: {n} wins")
    print(f"TOTAL: {total:.2f} pts, {solved}/400 solved")


if __name__ == "__main__":
    main()
