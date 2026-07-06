import os, glob, shutil, math, json
import onnx
from src.harness import load_task, evaluate
BASE='submission/overfit_nets'
UD='/private/tmp/claude-501/-Users-minseong-project-neurogolf/94ec265b-0e54-4a1a-bf64-54b97fb5d00f/scratchpad/mine07/extracted/udit_723717'
PB='/private/tmp/claude-501/-Users-minseong-project-neurogolf/94ec265b-0e54-4a1a-bf64-54b97fb5d00f/scratchpad/mine07/extracted/poby_723583'
OUT='/private/tmp/claude-501/-Users-minseong-project-neurogolf/94ec265b-0e54-4a1a-bf64-54b97fb5d00f/scratchpad/mine07/overfit2'
os.makedirs(OUT, exist_ok=True)
improved=[]; base_tot=0.0; new_tot=0.0
for t in range(1,401):
    task=load_task(t); name=f'task{t:03d}.onnx'
    cands=[('base',f'{BASE}/{name}'),('udit',f'{UD}/{name}'),('poby',f'{PB}/{name}')]
    best=None
    for src,p in cands:
        if not os.path.exists(p): continue
        try: r=evaluate(onnx.load(p),task)
        except Exception: continue
        if r['fail']!=0: continue
        c=r['memory']+r['params']; pts=r['points']
        if best is None or c<best[0]: best=(c,src,p,pts)
    # base points for delta
    rb=evaluate(onnx.load(f'{BASE}/{name}'),task); base_c=rb['memory']+rb['params']; base_pts=rb['points']
    base_tot+=base_pts
    if best is None:  # shouldn't happen, base always valid
        best=(base_c,'base',f'{BASE}/{name}',base_pts)
    shutil.copy(best[2], f'{OUT}/{name}')
    new_tot+=best[3]
    if best[1]!='base' and best[0]<base_c:
        improved.append((t,base_c,best[0],best[1],round(best[3]-base_pts,3)))
improved.sort(key=lambda x:-x[4])
print(f'base overfit total: {base_tot:.2f}  new: {new_tot:.2f}  delta +{new_tot-base_tot:.3f}')
print(f'improved {len(improved)} tasks:')
for t,bc,nc,s,d in improved[:20]:
    print(f'  task{t:03d}: {bc}->{nc} ({s}) +{d}')
