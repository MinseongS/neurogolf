import os, glob, shutil, math, json
from pathlib import Path
import onnx
from src.harness import load_task, evaluate

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / 'submission' / 'overfit_nets'
UD = Path(os.environ.get('NEUROGOLF_UDIT_NETS', ROOT / 'public_candidates' / 'udit_723717'))
PB = Path(os.environ.get('NEUROGOLF_POBY_NETS', ROOT / 'public_candidates' / 'poby_723583'))
OUT = Path(os.environ.get('NEUROGOLF_OVERFIT_MERGE_OUT', ROOT / 'reports' / 'candidates' / 'overfit_minmerge'))
OUT.mkdir(parents=True, exist_ok=True)
improved=[]; base_tot=0.0; new_tot=0.0
for t in range(1,401):
    task=load_task(t); name=f'task{t:03d}.onnx'
    cands=[('base',BASE/name),('udit',UD/name),('poby',PB/name)]
    best=None
    for src,p in cands:
        if not p.exists(): continue
        try: r=evaluate(onnx.load(p),task)
        except Exception: continue
        if r['fail']!=0: continue
        c=r['memory']+r['params']; pts=r['points']
        if best is None or c<best[0]: best=(c,src,p,pts)
    # base points for delta
    rb=evaluate(onnx.load(BASE/name),task); base_c=rb['memory']+rb['params']; base_pts=rb['points']
    base_tot+=base_pts
    if best is None:  # shouldn't happen, base always valid
        best=(base_c,'base',BASE/name,base_pts)
    shutil.copy(best[2], OUT/name)
    new_tot+=best[3]
    if best[1]!='base' and best[0]<base_c:
        improved.append((t,base_c,best[0],best[1],round(best[3]-base_pts,3)))
improved.sort(key=lambda x:-x[4])
print(f'base overfit total: {base_tot:.2f}  new: {new_tot:.2f}  delta +{new_tot-base_tot:.3f}')
print(f'improved {len(improved)} tasks:')
for t,bc,nc,s,d in improved[:20]:
    print(f'  task{t:03d}: {bc}->{nc} ({s}) +{d}')
