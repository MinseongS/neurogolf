"""Post-processing polish passes over networks/.

prune: drop initializers not referenced by any node (they still count as
params in the official scorer). Re-verifies and keeps only improvements.
"""

import json
import multiprocessing

import onnx

from .harness import evaluate, load_task
from .pipeline import MANIFEST, NETWORKS, load_manifest, write_scoreboard


def prune_unused(model):
    used = set()
    for node in model.graph.node:
        used.update(node.input)
    used.update(o.name for o in model.graph.output)
    removed = 0
    for field in (model.graph.initializer, model.graph.sparse_initializer):
        keep = [t for t in field if (t.values.name if hasattr(t, "values") else t.name) in used]
        if len(keep) != len(field):
            removed += len(field) - len(keep)
            del field[:]
            field.extend(keep)
    # unused graph inputs (other than 'input') also block nothing; leave them.
    return removed


def polish_one(task_num):
    path = NETWORKS / f"task{task_num:03d}.onnx"
    if not path.exists():
        return task_num, None
    model = onnx.load(path)
    if prune_unused(model) == 0:
        return task_num, None
    task = load_task(task_num)
    ev = evaluate(model, task)
    if not ev["ok"]:
        return task_num, None
    return task_num, (model, ev)


def main():
    manifest = load_manifest()
    improved = 0
    with multiprocessing.Pool(10) as pool:
        for task_num, res in pool.imap_unordered(polish_one, range(1, 401)):
            if res is None:
                continue
            model, ev = res
            cur = manifest.get(task_num)
            if cur and ev["points"] <= cur["points"] + 1e-9:
                continue
            onnx.save(model, NETWORKS / f"task{task_num:03d}.onnx")
            method = (cur.get("method") or "?") if cur else "?"
            if not method.endswith("+pruned"):
                method += "+pruned"
            manifest[task_num] = {"points": ev["points"], "memory": ev["memory"],
                                  "params": ev["params"], "method": method}
            improved += 1
            print(f"task{task_num:03d}: -> {ev['points']:.2f} (pruned)", flush=True)
    with open(MANIFEST, "w") as f:
        json.dump({"tasks": {str(k): v for k, v in sorted(manifest.items())}}, f, indent=1)
    total, solved = write_scoreboard(manifest)
    print(f"pruned-improved {improved}; TOTAL: {total:.2f} pts, {solved}/400")


if __name__ == "__main__":
    main()
