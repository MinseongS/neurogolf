"""Kernel-collapse lever (PURE params golf, orthogonal to memory/free-I/O sweeps).

Any Conv whose weight [O,I,KH,KW] has all nonzeros confined to a SINGLE (kh,kw)
spatial position is a 1x1 conv stored oversized. Replace with a [O,I,1,1] kernel
= W[:,:,a,b] and adjust pads (pt-a*d0, pl-b*d1, ...; ORT accepts negative pads) to
keep the output bit-identical. Saves (KH*KW-1)*O*I params. Bit-identical => safe.

Usage: from neurogolf.scans.kernel_collapse import collapse_single_pos_convs
Sweep: found 18 wins / +0.553 LB on 2026-07-08 (task322/187/005/138/286/173/218...).
"""
import math

import onnx, numpy as np
from onnx import numpy_helper, helper

from neurogolf.scoring import load_task, evaluate
from neurogolf.paths import OVERFIT_NETS, CANDIDATES


def collapse_single_pos_convs(inpath, outpath):
    m=onnx.load(inpath); g=m.graph
    inits={i.name:i for i in g.initializer}
    changed=0; saved=0
    for n in g.node:
        if n.op_type!="Conv": continue
        wname=n.input[1]
        if wname not in inits: continue
        W=numpy_helper.to_array(inits[wname])
        if W.ndim!=4: continue
        O,I,KH,KW=W.shape
        if KH*KW<=1: continue
        # nonzero spatial positions (across O,I)
        spatial=np.abs(W).sum(axis=(0,1))  # [KH,KW]
        nzpos=np.argwhere(spatial!=0)
        if len(nzpos)!=1: continue   # only single-position kernels
        a,b=int(nzpos[0][0]),int(nzpos[0][1])
        # current pads/dilations/strides
        pads=[0,0,0,0]; dil=[1,1]; strd=[1,1]
        for at in n.attribute:
            if at.name=="pads": pads=list(at.ints)
            if at.name=="dilations": dil=list(at.ints)
            if at.name=="strides": strd=list(at.ints)
        if dil!=[1,1] or strd!=[1,1]:
            # dilation d: nonzero at (a,b) maps to offset a*d. handle general
            pass
        pt,pl,pb,pr=pads
        # new 1x1 kernel = W[:,:,a,b]; adjusted pads (account for dilation)
        d0,d1=dil
        # effective offset of tap (a,b) from top-left = a*d0, b*d1; kernel eff size (KH-1)*d0+1
        effH=(KH-1)*d0+1; effW=(KW-1)*d1+1
        # out_H(orig)= in + pt+pb-effH+1 ; 1x1: out= in+pt'+pb' ; keep same => pt'+pb'=pt+pb-effH+1
        # alignment: orig out[y]=sum_c in[c, y*st + a*d0 - pt]; 1x1 out[y]=in[c, y - pt']; match => pt'=pt - a*d0
        pt2=pt - a*d0; pb2=(pt+pb-effH+1)-pt2
        pl2=pl - b*d1; pr2=(pl+pr-effW+1)-pl2
        W1=np.ascontiguousarray(W[:,:,a:a+1,b:b+1])
        new_init=numpy_helper.from_array(W1, wname)
        inits[wname].CopyFrom(new_init)
        # update attrs
        newattrs=[]
        for at in n.attribute:
            if at.name in ("pads","dilations","kernel_shape"): continue
            newattrs.append(at)
        del n.attribute[:]
        n.attribute.extend(newattrs)
        n.attribute.append(helper.make_attribute("kernel_shape",[1,1]))
        n.attribute.append(helper.make_attribute("pads",[pt2,pl2,pb2,pr2]))
        n.attribute.append(helper.make_attribute("dilations",[1,1]))
        changed+=1; saved += W.size - W1.size
    if changed:
        onnx.save(m,outpath)
    return changed, saved


def scan_all(tasks: list[int] | None = None) -> dict:
    task_range = tasks if tasks else range(1, 401)
    items = []
    for t in task_range:
        p = OVERFIT_NETS / f"task{t:03d}.onnx"
        if not p.exists():
            continue
        # collapse is a cheap onnx/numpy pass — do it FIRST, only pay evaluate() on a hit
        outdir = CANDIDATES / f"task{t:03d}"
        outdir.mkdir(parents=True, exist_ok=True)
        out = outdir / "kcollapse.onnx"
        ch, sv = collapse_single_pos_convs(str(p), str(out))
        if ch == 0:
            continue
        base = evaluate(str(p), load_task(t)); bc = base["memory"] + base["params"]
        r = evaluate(str(out), load_task(t))
        if r.get("memory") is None:
            continue
        nc = r["memory"] + r["params"]
        if r["fail"] != 0 or nc >= bc:
            continue
        eg = (bc - nc) / bc if bc else 0.0
        items.append({"task": t, "convs": ch, "saved_bytes": bc - nc,
                      "base_cost": bc, "new_cost": nc,
                      "points_delta": round(math.log(bc / nc), 4),
                      "candidate": str(out), "expected_gain": round(eg, 4)})
    return {"items": items}
