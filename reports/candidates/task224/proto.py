import numpy as np
from src.harness import load_task

t = load_task(224)
allex = t['train'] + t['test'] + t['arc-gen']

e5 = np.zeros(10, np.float32); e5[5] = 1
mask = np.array([0,1,1,1,1,0,1,1,1,1], np.float32)
grid = np.arange(16, dtype=np.uint8)
grid2 = grid + 1

# F' [5,5,2], G [2,10,10]
F = np.zeros((5,5,2), np.float32)
F[0,0,0] = 1
F[1,3,0] = -1; F[2,4,0] = 1
F[1,3,1] = 1;  F[2,4,1] = -1
G = np.zeros((2,10,10), np.float32)
G[0] = np.eye(10)
G[1] = mask[:,None] * np.ones(10)

bad = 0
for idx, ex in enumerate(allex):
    gi = np.array(ex['input']); go = np.array(ex['output'])
    H, W = gi.shape
    inp = np.zeros((1,10,30,30), np.float32)
    for c in range(10):
        inp[0,c,:H,:W] = (gi == c)
    # extraction
    row_sum = np.einsum('bchw,c->bh', inp, e5)   # [1,30]
    col_sum = np.einsum('bchw,c->bw', inp, e5)
    rh = (row_sum != 0).astype(np.uint8)[0]
    ch = (col_sum != 0).astype(np.uint8)[0]
    rmin = np.uint8(np.argmax(rh)); rmax = np.uint8(len(rh)-1-np.argmax(rh[::-1]))
    cmin = np.uint8(np.argmax(ch)); cmax = np.uint8(len(ch)-1-np.argmax(ch[::-1]))
    top1 = rmin + np.uint8(1); left1 = cmin + np.uint8(1)
    rs1 = (grid > rmin) & (grid < rmax)
    rs2 = (grid > top1) & (grid2 < rmax)
    cs1 = (grid > cmin) & (grid < cmax)
    cs2 = (grid > left1) & (grid2 < cmax)
    ones16 = np.ones(16, bool)
    Sb = np.stack([ones16, rs1, rs2, cs1, cs2])   # [5,16]
    Sp = np.zeros((5,30), np.float32); Sp[:, :16] = Sb
    S = Sp[None,:,:,None]  # [1,5,30,1]
    out = np.einsum('bshw,bkxy,bphi,bqwj,pqv,vks->bkhw', inp, inp, S, S, F, G, optimize=True)
    # decode >0
    dec = np.zeros((30,30), np.int64)
    pos = out[0] > 0
    dec_ch = np.argmax(out[0], axis=0)
    dec = np.where(pos.any(axis=0), dec_ch, 0)
    pred = dec[:H,:W]
    if not (pred == go).all():
        bad += 1
        if bad <= 2:
            print('MISMATCH ex', idx)
            print(gi); print(go); print(pred)
print('bad:', bad, '/', len(allex))
