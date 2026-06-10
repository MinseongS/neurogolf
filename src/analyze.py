"""Dataset analysis: duplicates, shape classes, identity/colormap candidates."""

import collections
import hashlib
import json

from .harness import DATA_DIR, load_task

N_TASKS = 400


def all_examples(task):
    return task.get("train", []) + task.get("test", []) + task.get("arc-gen", [])


def usable_examples(task):
    """Examples that the scorer actually checks (grids > 30 are ignored)."""
    out = []
    for ex in all_examples(task):
        dims = [len(ex["input"]), len(ex["input"][0]), len(ex["output"]), len(ex["output"][0])]
        if max(dims) <= 30:
            out.append(ex)
    return out


def main():
    file_hash = {}
    stats = collections.Counter()
    rows = []
    for num in range(1, N_TASKS + 1):
        raw = (DATA_DIR / f"task{num:03d}.json").read_bytes()
        h = hashlib.md5(raw).hexdigest()
        task = json.loads(raw)
        exs = usable_examples(task)
        n_total = len(all_examples(task))
        same_shape = all(
            len(e["input"]) == len(e["output"]) and len(e["input"][0]) == len(e["output"][0])
            for e in exs)
        identity = same_shape and all(e["input"] == e["output"] for e in exs)
        colormap_ok, cmap = True, {}
        if same_shape:
            for e in exs:
                for ri, row in enumerate(e["input"]):
                    for ci, cin in enumerate(row):
                        cout = e["output"][ri][ci]
                        if cmap.setdefault(cin, cout) != cout:
                            colormap_ok = False
                            break
                    if not colormap_ok:
                        break
                if not colormap_ok:
                    break
        else:
            colormap_ok = False
        rows.append({
            "task": num, "hash": h, "n_examples": len(exs), "n_skipped": n_total - len(exs),
            "same_shape": same_shape, "identity": identity, "colormap": colormap_ok,
        })
        file_hash.setdefault(h, []).append(num)
        stats["identity"] += identity
        stats["colormap (incl. identity)"] += colormap_ok
        stats["same_shape"] += same_shape
        stats["has_skipped(>30)"] += (n_total != len(exs))

    dupes = {h: nums for h, nums in file_hash.items() if len(nums) > 1}
    print(f"unique files: {len(file_hash)} / {N_TASKS}")
    print(f"duplicate groups: {len(dupes)} covering {sum(len(v) for v in dupes.values())} tasks")
    for k, v in stats.items():
        print(f"{k}: {v}")
    with open(DATA_DIR.parent / "reports" / "analysis.json", "w") as f:
        json.dump({"rows": rows, "dupes": list(dupes.values())}, f, indent=1)


if __name__ == "__main__":
    main()
