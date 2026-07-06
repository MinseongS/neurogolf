"""Unified gate for task366 candidates (onnx path).
Usage: .venv/bin/python reports/scripts/gate366.py CAND.onnx [N_fresh]
Prints: mem, params, points, bundled fail, fresh fail (vs generator ground truth),
and fresh divergence vs the live incumbent networks/task366.onnx.
"""
import sys, os, importlib, math
import numpy as np, onnx, onnxruntime as ort
_ROOT=os.path.abspath(os.path.join(os.path.dirname(__file__),'..','..'))
sys.path.insert(0,_ROOT)
_ARC=os.path.join(_ROOT,'arc-gen'); sys.path.append(_ARC)
from src.harness import load_task, sanitize_model, evaluate
import json
CAND=sys.argv[1]
N=int(sys.argv[2]) if len(sys.argv)>2 else 2000
TASK=366
mapping=json.load(open(os.path.join(_ROOT,'reports','arc_mapping.json')))
arc=mapping[str(TASK)]['arc_id']; gen=importlib.import_module(f'tasks.task_{arc}')
task=load_task(TASK)

def onehot(grid):
    g=np.asarray(grid); H,W=g.shape
    if max(H,W)>30: return None
    x=np.zeros((1,10,30,30),dtype=np.float32)
    rr,cc=np.meshgrid(np.arange(H),np.arange(W),indexing='ij')
    x[0,g.astype(np.int64),rr,cc]=1.0
    return x
def mksess(path):
    s=sanitize_model(onnx.load(path))
    so=ort.SessionOptions(); so.graph_optimization_level=ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    return ort.InferenceSession(s.SerializeToString(),so,providers=['CPUExecutionProvider'])
def run(s,x): return (s.run(['output'],{'input':x})[0]>0.0).astype(np.float32)

# 1) grader measurement + bundled
r=evaluate(CAND, task)
print(f"MEM={r['memory']} PARAMS={r['params']} POINTS={round(r['points'],4) if r['points'] else r['points']} bundled_fail={r['fail']} ok={r['ok']} err={r['error']}")

# 2) fresh gate
cand=mksess(CAND); inc=mksess(os.path.join(_ROOT,'networks','task366.onnx'))
cf=inc_f=div=n=0
for _ in range(N):
    try: ex=gen.generate()
    except Exception: continue
    x=onehot(ex['input']); y=onehot(ex['output'])
    if x is None or y is None: continue
    yb=(y>0.0).astype(np.float32); n+=1
    try: oc=run(cand,x)
    except Exception: oc=None; cf+=1
    if oc is not None and not np.array_equal(oc,yb): cf+=1
    try: oi=run(inc,x)
    except Exception: oi=None; inc_f+=1
    if oi is not None and not np.array_equal(oi,yb): inc_f+=1
    if oc is not None and oi is not None and not np.array_equal(oc,oi): div+=1
print(f"fresh n={n}  candidate_fail={cf}  incumbent_fail={inc_f}  cand_vs_inc_div={div}")
