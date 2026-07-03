"""Scan all 400 nets for a Cast that only widens a bool/uint8 tensor to satisfy
an old-opset shape-only op (Pad/Concat/Reshape/Transpose/Slice/Tile/etc).
If deleting the Cast (with opset bump to 13 where needed) removes a counted plane,
report the plane bytes as est savings."""
import os, json, glob
import onnx
from onnx import TensorProto, shape_inference

DT = TensorProto
BYTES = {DT.FLOAT:4, DT.FLOAT16:2, DT.DOUBLE:8, DT.INT64:8, DT.INT32:4,
         DT.INT16:2, DT.INT8:1, DT.UINT8:1, DT.BOOL:1, DT.UINT16:2, DT.UINT32:4}
NAME = {v:k for k,v in DT.__dict__.items() if isinstance(v,int)}

# shape-only ops: output is a rearrangement/copy of the DATA input, dtype-preserving.
SHAPE_OPS = {"Pad","Concat","Reshape","Transpose","Slice","Tile","Squeeze",
             "Unsqueeze","Flatten","Expand","Split","SpaceToDepth","DepthToSpace",
             "Identity","Gather","GatherElements","GatherND","ScatterElements",
             "ScatterND","Scatter","Compress","ReverseSequence","Resize"}
# which input indices are the DTYPE-FLEXIBLE data slots (dtype flows through / can stay narrow).
# All other input slots are index/shape/param slots that legitimately require int64/int32
# (casting bool->int for those is semantically REQUIRED, not harvestable).
DATA_SLOTS = {
    "Gather":{0}, "GatherElements":{0}, "GatherND":{0},
    "ScatterElements":{0,2}, "ScatterND":{0,2}, "Scatter":{0,2},
    "Pad":{0,2}, "Concat":None, "Reshape":{0}, "Transpose":{0},
    "Slice":{0}, "Tile":{0}, "Squeeze":{0}, "Unsqueeze":{0}, "Flatten":{0},
    "Expand":{0}, "Split":{0}, "SpaceToDepth":{0}, "DepthToSpace":{0},
    "Identity":{0}, "Compress":{0}, "ReverseSequence":{0}, "Resize":{0},
}
def feeds_data_slot(node, name):
    slots=DATA_SLOTS.get(node.op_type)
    idxs=[i for i,inp in enumerate(node.input) if inp==name]
    if slots is None: return True  # Concat: every input is a data operand
    return any(i in slots for i in idxs)
# ops that (at opset 11) reject bool/uint8 and force a float/int cast (widening).
# Pad in particular is bool-illegal at 11, legal at 13.
OPSET13_FIXABLE = {"Pad"}  # ops whose bool support arrives at opset 13

def dtype_of(name, vimap, initmap):
    if name in initmap: return initmap[name]
    vi = vimap.get(name)
    if vi is None: return None
    return vi.type.tensor_type.elem_type or None

def shape_of(name, vimap):
    vi = vimap.get(name)
    if vi is None: return None
    dims=[]
    for d in vi.type.tensor_type.shape.dim:
        dims.append(d.dim_value if d.HasField("dim_value") else None)
    return dims

def numel(shape):
    if not shape: return None
    n=1
    for d in shape:
        if d is None or d==0: return None
        n*=d
    return n

results=[]
SKIP={120,220,230,294}
for path in sorted(glob.glob("networks/task*.onnx")):
    task=int(os.path.basename(path)[4:7])
    if task in SKIP: continue
    m=onnx.load(path)
    try:
        m2=shape_inference.infer_shapes(m)
        g=m2.graph
    except Exception:
        g=m.graph
    opset=max([o.version for o in m.opset_import if o.domain in ("","ai.onnx")] or [11])
    vimap={vi.name:vi for vi in list(g.value_info)+list(g.input)+list(g.output)}
    initmap={i.name:i.data_type for i in g.initializer}
    # map producer node by output name; consumers by input name
    producers={}; consumers={}
    for n in g.node:
        for o in n.output: producers[o]=n
        for i in n.input: consumers.setdefault(i,[]).append(n)
    outset={o.name for o in g.output}
    for n in g.node:
        if n.op_type!="Cast": continue
        src=n.input[0]; dst=n.output[0]
        sdt=dtype_of(src,vimap,initmap)
        ddt=dtype_of(dst,vimap,initmap)
        # attribute 'to'
        to_dt=None
        for a in n.attribute:
            if a.name=="to": to_dt=a.i
        if ddt is None: ddt=to_dt
        if sdt is None or ddt is None: continue
        # widening from narrow (bool/uint8) to wider
        if sdt not in (DT.BOOL, DT.UINT8): continue
        if BYTES.get(ddt,99) <= BYTES.get(sdt,0): continue  # not a widening
        cons=consumers.get(dst,[])
        if not cons: continue
        # every consumer must be a shape-only op AND take the cast on its DATA slot
        # (index/shape/param slots require int -> cast there is semantically required)
        consops={c.op_type for c in cons}
        if not consops.issubset(SHAPE_OPS): continue
        if not all(feeds_data_slot(c,dst) for c in cons): continue
        # does deleting the cast remove a counted plane? the dst plane is counted.
        shp=shape_of(dst,vimap)
        ne=numel(shp)
        plane_bytes = ne*BYTES.get(ddt,4) if ne else None
        # opset migration needed? if any consumer is bool-illegal at current opset
        needs_bump = (opset < 13) and bool(consops & OPSET13_FIXABLE)
        # only meaningful if consumers can accept the NARROW dtype after fix.
        # Pad: bool ok @13. Concat/Reshape/Transpose/Slice/Tile/Gather*/Squeeze: bool ok already.
        # if a bool-illegal-op exists but opset already >=13, no bump but cast likely spurious.
        est_delta=None
        if plane_bytes:
            import math
            # savings = removing dst plane; add ~+3 params for opset init (axes-input for pad etc) if bump
            new_params_penalty = 3 if needs_bump else 0
            # delta pts ~ ln(mem+params)_before - after; we approximate as saved bytes contribution.
            # rough: pts = 25 - ln(mem+params); delta ≈ ln((M)/(M-saved)) — need M. skip; report bytes.
            est_delta=plane_bytes
        results.append(dict(task=task, cast=n.name or dst, src=src, dst=dst,
                            src_dt=NAME.get(sdt,sdt), dst_dt=NAME.get(ddt,ddt),
                            consumers=sorted(consops), shape=shp, plane_bytes=plane_bytes,
                            opset=opset, needs_bump=needs_bump,
                            to_output=(dst in outset)))

# compute pts delta using measured mem+params per task from manifest
man=json.load(open("reports/manifest.json"))["tasks"]
mem_by={int(k):v for k,v in man.items()}
import math
for r in results:
    e=mem_by.get(r["task"])
    if e and r["plane_bytes"]:
        M=e.get("memory",0)+e.get("params",0)
        pen=3 if r["needs_bump"] else 0
        newM=M - r["plane_bytes"] + pen
        if newM>0 and M>0:
            r["cur_M"]=M
            r["est_delta_pts"]=round(math.log(M)-math.log(newM),4)
        else:
            r["est_delta_pts"]=None
    else:
        r["est_delta_pts"]=None

# --- mechanism verdicts (populated from targeted graph inspection; see boolpad_scan.md) ---
VERDICT={
 126:"FALSE+ : ScatterND updates onto FLOAT graph-input canvas; updates dtype locked to input (float). Narrowing would force casting the 9000B canvas to bool = net loss.",
 106:"FALSE+ : ScatterElements updates onto FLOAT graph-input canvas; updates dtype locked to input.",
 293:"FALSE+ : ScatterND updates onto FLOAT graph-input canvas; updates dtype locked to input.",
 56 :"REFUTED (built+measured): true bool->Pad, but opset9->13 bump converts free Slice/Pad node-attrs into COUNTED int64 inits (+23 params) >> 12B fp16 plane saved. 21.474 -> 21.193 (-0.28 LOSS).",
 255:"FALSE+ : bool->float feeds Reshape whose output goes into MatMul (float required); cast pays for MatMul, not the shape-op. Also matured net.",
 158:"FALSE+ : bool->float feeds Slice whose output goes into Mul (float required).",
 206:"FALSE+ : bool->float16 feeds AveragePool (float required), or int8 feeds ReduceMax->ArgMax (already narrow).",
 92 :"FALSE+ : bool->float feeds ReduceSum (float required); recently matured (S11).",
 54 :"NEGLIGIBLE : already uint8/int8 narrow chains; flagged plane 40B, dpts 0.002.",
 188:"NEGLIGIBLE : 4B float<->bool micro-casts on [1,1,1,1] scalars.",
}
for r in results:
    r["verdict"]=VERDICT.get(r["task"],"low-value (<0.05 dpts) or shape-op cast paying for a downstream float op; not harvestable")
results.sort(key=lambda r:(r.get("est_delta_pts") or 0), reverse=True)
os.makedirs("reports",exist_ok=True)
json.dump(results, open("reports/boolpad_scan.json","w"), indent=1)
print(f"pattern hits: {len(results)}")
for r in results[:20]:
    print(f"  task{r['task']:03d} {r['src_dt']}->{r['dst_dt']} {r['consumers']} shape={r['shape']} bytes={r['plane_bytes']} bump={r['needs_bump']} dpts={r.get('est_delta_pts')}")
