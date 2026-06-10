# NeuroGolf 2026

[Kaggle: The 2026 NeuroGolf Championship](https://www.kaggle.com/competitions/neurogolf-2026) —
build the *smallest* ONNX network per ARC-AGI task. Score per task:
`max(1, 25 - ln(memory_bytes + params))`, only if the network passes **all**
train/test/arc-gen examples. 400 tasks, max 10,000 points.
Deadline: **2026-07-15**.

## Layout

```
data/        competition data (gitignored; `kaggle competitions download -c neurogolf-2026 -p data/ && unzip`)
src/
  harness.py   local verify+score, exact mirror of official neurogolf_utils.py
  builders.py  ONNX graph builders (opset 10 / IR 10)
  analyze.py   dataset stats (dupes, shape classes, colormap candidates)
  solvers.py   tiered automatic solvers
  pipeline.py  run solvers across all tasks -> networks/ + manifest + scoreboard
networks/    one taskXXX.onnx per solved task (committed = the deliverable)
reports/     manifest.json, SCOREBOARD.md, analysis.json
submission/  zip artifacts (gitignored)
```

## Workflow

```bash
.venv/bin/python -m src.analyze              # dataset stats
.venv/bin/python -m src.pipeline             # solve everything, write networks/ + scoreboard
.venv/bin/python -m src.pipeline --pack      # also build submission/submission.zip
.venv/bin/python -m src.harness networks/task016.onnx 16   # score one network
```

Every pipeline run rewrites `reports/manifest.json` + `reports/SCOREBOARD.md`;
a network in `networks/` is only ever replaced by one that scores **higher**,
so committing after each run gives a monotonically improving, reviewable history.

## Solver tiers

1. **identity** — `Identity` node, 0 cost, 25 pts (no such tasks in this dataset)
2. **conv** — single no-bias `Conv` (k×k up to 7), integer perceptron fit per
   output channel over the one-hot canvas; exact verify in numpy then ORT
3. **memorizer** — exact-match lookup over all given examples (guaranteed
   correct on the official examples; ~11-13 pts depending on size)
