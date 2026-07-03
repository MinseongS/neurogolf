"""Dirty-process gate. In ONE process:
  1. pollute the ORT arena with `pollutants` (via evaluate, exactly like grader),
  2. evaluate INCUMBENT (expect it to FLIP -> fail>0)  [positive control]
  3. evaluate CANDIDATE (expect fail==0)              [the fix]
Uses harness.evaluate for faithful grader-identical session construction.
"""
import sys, json
sys.path.insert(0, ".")
from src.harness import load_task, evaluate

TASK=int(sys.argv[1]); CAND=sys.argv[2]
POLL=[int(x) for x in sys.argv[3].split(",")]

for p in POLL:
    evaluate(f"networks/task{p:03d}.onnx", load_task(p))

inc=evaluate(f"networks/task{TASK}.onnx", load_task(TASK))
cand=evaluate(CAND, load_task(TASK))
print(json.dumps({
    "task":TASK, "pollutants":POLL,
    "incumbent_dirty":{"pass":inc["pass"],"fail":inc["fail"]},
    "candidate_dirty":{"pass":cand["pass"],"fail":cand["fail"],
                       "mem":cand["memory"],"params":cand["params"]},
}))
