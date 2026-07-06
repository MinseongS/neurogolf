"""Fresh-generalization gate for a NeuroGolf candidate.
Usage: freshverify.py TASKNUM [candidate_module_path]
Builds incumbent (src.custom.taskNNN) and optional candidate, generates N fresh
instances via the arc-gen generator, requires candidate fail=0 vs ground-truth.
"""
import sys, os, json, importlib, importlib.util, math
import numpy as np, onnx, onnxruntime as ort
# repo root at path[0] so `import src.*` works when run as a script (python reports/scripts/fresh_verify.py).
# Set NEUROGOLF_ROOT to point at a worktree's root to test that worktree's edited src/custom.
_ROOT = os.environ.get("NEUROGOLF_ROOT") or os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, _ROOT)
# appended (NOT insert-0): arc-gen has its own src/ that would shadow ours.
_ARCGEN = os.path.join(_ROOT, "arc-gen")
if not os.path.isdir(_ARCGEN):
    raise FileNotFoundError(f"repo-local arc-gen not found: {_ARCGEN}")
sys.path.append(_ARCGEN)
from src.harness import load_task, sanitize_model

_argv=[a for a in sys.argv[1:] if a!="--cache"]
TASK=int(_argv[0])
CAND=_argv[1] if len(_argv)>1 else None
N=int(_argv[2]) if len(_argv)>2 else 1500
USE_CACHE="--cache" in sys.argv  # reuse reports/fresh_cache/taskNNN.npz (see fresh_cache.py; final gates should still do one uncached run)

mapping=json.load(open('reports/arc_mapping.json'))
arc=mapping[str(TASK)]['arc_id']
gen=importlib.import_module(f'tasks.task_{arc}')

def to_onehot(grid):
    g=np.asarray(grid)
    H,W=g.shape
    if max(H,W)>30: return None
    x=np.zeros((1,10,30,30),dtype=np.float32)
    rr,cc=np.meshgrid(np.arange(H),np.arange(W),indexing="ij")
    x[0,g.astype(np.int64),rr,cc]=1.0
    return x

def _cached_examples(n):
    import os
    path=os.path.join(_ROOT,"reports","fresh_cache",f"task{TASK:03d}.npz")
    z=np.load(path)
    ins,outs,hs,ws=z["inputs"],z["outputs"],z["heights"],z["widths"]
    ohs=z["oheights"] if "oheights" in z else hs
    ows=z["owidths"] if "owidths" in z else ws
    if ("oheights" not in z) and any(int(a)!=int(b) for a,b in zip(hs,ohs)):
        pass
    if len(ins)<n:
        print(f"  (cache has only {len(ins)} < {n})")
    for i in range(min(n,len(ins))):
        h,w=int(hs[i]),int(ws[i])
        oh,ow=int(ohs[i]),int(ows[i])
        yield ins[i][:h,:w], outs[i][:oh,:ow]

def sess(mod_build, task):
    m=mod_build(task)
    s=sanitize_model(m)
    so=ort.SessionOptions(); so.graph_optimization_level=ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    return ort.InferenceSession(s.SerializeToString(), so, providers=['CPUExecutionProvider'])

task=load_task(TASK)
inc_mod=importlib.import_module(f'src.custom.task{TASK:03d}')
importlib.reload(inc_mod)
inc=sess(inc_mod.build, task)

cand=None
if CAND:
    import importlib.util as _iu
    if CAND.endswith('.py'):
        # exec as if inside src.custom package (for relative ._exact import)
        import types
        spec=_iu.spec_from_file_location('src.custom._candmod', CAND)
        cm=_iu.module_from_spec(spec)
        cm.__package__='src.custom'
        sys.modules['src.custom._candmod']=cm
        spec.loader.exec_module(cm)
    else:
        cm=importlib.import_module(CAND); importlib.reload(cm)
    cand=sess(cm.build, task)

def run(s,x):
    return (s.run(['output'],{'input':x})[0]>0.0).astype(np.float32)

inc_fail=cand_fail=cand_vs_inc=0; n_ok=0
def _examples():
    if USE_CACHE:
        yield from _cached_examples(N); return
    for _ in range(N):
        try:
            ex=gen.generate(); yield ex["input"], ex["output"]
        except Exception:
            continue
for gi,go in _examples():
    x=to_onehot(gi)
    if x is None: continue
    y=to_onehot(go)
    if y is None: continue
    ybool=(y>0.0).astype(np.float32)
    n_ok+=1
    # inference errors count as fails (some incumbents hard-error ORT on rare inputs, task101)
    try:
        oi=run(inc,x)
    except Exception:
        oi=None; inc_fail+=1
    if oi is not None and not np.array_equal(oi,ybool): inc_fail+=1
    if cand is not None:
        try:
            oc=run(cand,x)
        except Exception:
            oc=None; cand_fail+=1
        if oc is not None and not np.array_equal(oc,ybool): cand_fail+=1
        if (oc is None)!=(oi is None) or (oc is not None and oi is not None and not np.array_equal(oc,oi)): cand_vs_inc+=1

print(f"task{TASK} arc={arc} fresh_instances={n_ok}/{N}")
print(f"  incumbent fail = {inc_fail}")
if cand is not None:
    print(f"  candidate fail = {cand_fail}")
    print(f"  candidate != incumbent = {cand_vs_inc}")
