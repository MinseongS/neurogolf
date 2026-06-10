"""Print a task's examples as digit grids (for humans/agents studying a task).

Usage: python -m src.show TASKNUM [--arcgen N]
"""

import argparse

from .analyze import usable_examples
from .harness import load_task


def render(grid):
    return "\n".join("".join(str(c) for c in row) for row in grid)


def side_by_side(a, b, gap="   ->   "):
    la, lb = a.split("\n"), b.split("\n")
    wa = max(len(r) for r in la)
    h = max(len(la), len(lb))
    la += [""] * (h - len(la))
    lb += [""] * (h - len(lb))
    mid = h // 2
    return "\n".join(
        f"{ra:<{wa}}{gap if i == mid else ' ' * len(gap)}{rb}"
        for i, (ra, rb) in enumerate(zip(la, lb)))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("task_num", type=int)
    parser.add_argument("--arcgen", type=int, default=3,
                        help="how many arc-gen examples to print")
    args = parser.parse_args()
    task = load_task(args.task_num)
    for section in ("train", "test"):
        for i, ex in enumerate(task.get(section, [])):
            print(f"=== {section}[{i}] "
                  f"({len(ex['input'])}x{len(ex['input'][0])} -> "
                  f"{len(ex['output'])}x{len(ex['output'][0])}) ===")
            print(side_by_side(render(ex["input"]), render(ex["output"])))
            print()
    gen = [e for e in task.get("arc-gen", []) if e in usable_examples(task)]
    for i, ex in enumerate(gen[:args.arcgen]):
        print(f"=== arc-gen[{i}] of {len(gen)} "
              f"({len(ex['input'])}x{len(ex['input'][0])} -> "
              f"{len(ex['output'])}x{len(ex['output'][0])}) ===")
        print(side_by_side(render(ex["input"]), render(ex["output"])))
        print()


if __name__ == "__main__":
    main()
