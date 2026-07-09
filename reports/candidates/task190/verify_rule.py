import numpy as np, sys
sys.path.insert(0, '/Users/minseong/project/neurogolf')
from src.harness import load_task

t = load_task(190)
exs = t['train'] + t['test'] + t['arc-gen']
eps = 1.0/128
bad = 0
for idx, ex in enumerate(exs):
    g = np.array(ex['input']); o = np.array(ex['output'])
    nz = (g != 0)
    v = g[nz].max()
    assert (g[nz] == v).all(), f'{idx}: multicolor'
    cnt = nz.sum()
    if cnt != 5: print(idx, 'count', cnt); bad+=1; continue
    rowcnt = nz.sum(1).astype(float); colcnt = nz.sum(0).astype(float)
    ids = np.arange(10.)
    m1r = (ids*rowcnt).sum(); m2r = (ids*rowcnt**2).sum()
    m1c = (ids*colcnt).sum(); m2c = (ids*colcnt**2).sum()
    a = 2*m1r - m2r; sr = 4.5*m1r - 2.5*m2r
    b = 2*m1c - m2c; sc = 4.5*m1c - 2.5*m2c
    if abs(sr) != 3 or abs(sc) != 3: print(idx,'s',sr,sc); bad+=1; continue
    if rowcnt[int(a)] != 1 or colcnt[int(b)] != 1: print(idx,'sat',a,b); bad+=1; continue
    dr = np.sign(sr); dc = np.sign(sc)
    V = np.maximum(dr*(ids-a), 0); X = np.maximum(dc*(ids-b), 0)
    f = (2+eps)*np.outer(V,X) - np.outer(V**2, np.ones(10)) - np.outer(np.ones(10), X**2)
    ray = f > 0
    if (ray & nz).any(): print(idx,'ray hits colored'); bad+=1; continue
    pred = g.copy(); pred[ray] = v
    if not (pred == o).all(): print(idx,'MISMATCH'); bad+=1
print('bad:', bad, '/', len(exs))
