"""Task117 fp16 recast — bit-identical (0/2000 fresh divergence), mem 3922->3908 (+14B, +0.0034pts).
Recasts the small integer count-chain (object_color etc.) to fp16. Self-contained transform inlined.
"""
import os, numpy as np, onnx
from onnx import TensorProto, helper, numpy_helper, shape_inference
from collections import defaultdict

FLOAT_PRESERVING = {
    'Add','Sub','Mul','Div','Slice','Concat','Where','ReduceMax','ReduceMin',
    'ReduceSum','ReduceL1','ReduceMean','ReduceProd','Max','Min','Sum','Unsqueeze',
    'Squeeze','Reshape','Transpose','Pad','CumSum','Neg','Abs','Sign','Floor','Ceil',
    'Round','Gather','GatherND','GatherElements','Expand','Flatten','Split','Sqrt',
    'Log','Relu','Mod','Clip','Identity','Tile','Einsum','OneHot','Exp','Pow','Softmax',
}
# ops that consume float but emit non-float (int/bool) — clean sinks, accept fp16, no fwd-propagate
SINK_OPS = {'Equal','Greater','Less','GreaterOrEqual','LessOrEqual','ArgMax','ArgMin',
            'Cast','TopK','Shape','Sign'}  # note Sign preserves float; handled in preserving too
FLOAT_TYPES = {1,10,11,16}
FP16 = 10
FP32 = 1
ROOT = os.environ.get('NEUROGOLF_ROOT') or os.path.abspath(os.path.join(os.path.dirname(__file__),'..','..'))

def _root_for(task):
    return ROOT

def recast_fp16(task, seeds, root=None):
    root = root or ROOT
    m = onnx.load(os.path.join(root, 'networks', f'task{task:03d}.onnx'))
    try: mi = shape_inference.infer_shapes(m)
    except Exception: mi = m
    g = m.graph  # mutate original graph (has real inits); use mi only for dtype
    dtype = {}
    for vi in list(mi.graph.value_info)+list(mi.graph.input)+list(mi.graph.output):
        dtype[vi.name] = vi.type.tensor_type.elem_type
    inits = {i.name: i for i in g.initializer}
    for name, ini in inits.items(): dtype[name] = ini.data_type
    inputs = {i.name for i in g.input}
    producer = {}
    consumers = defaultdict(list)
    for n in g.node:
        for o in n.output: producer[o] = n
        for i in n.input:
            if i: consumers[i].append(n)
    def is_float(t): return dtype.get(t, FP32) in FLOAT_TYPES

    # ---- backward cone ----
    conv_tensors = set()      # tensors that will become fp16
    init_leaves = set()       # float inits to convert/dup
    cast_leaves = set()       # tensors whose producer Cast -> retarget to=10
    boundaries = set()        # float tensors needing an fp16 in-cast (producer consumes free input, or raw input)
    work = list(seeds); seen = set()
    while work:
        t = work.pop()
        if t in seen: continue
        seen.add(t)
        if t in inputs:
            boundaries.add(t); continue
        if t in inits:
            init_leaves.add(t); continue
        p = producer.get(t)
        if p is None:
            boundaries.add(t); continue
        if p.op_type == 'Cast':
            cast_leaves.add(t); continue
        # producer directly consumes a free graph input? -> boundary (producer_bound)
        if any(pi in inputs for pi in p.input):
            boundaries.add(t); continue
        if p.op_type in FLOAT_PRESERVING or p.op_type in ('Conv','MatMul'):
            conv_tensors.add(t)
            for pi in p.input:
                if pi and is_float(pi): work.append(pi)
        else:
            boundaries.add(t)
    fp16_avail = conv_tensors | cast_leaves | boundaries | set(s for s in seeds)
    # init_leaves become fp16 too
    def new_dtype(t):
        if t in conv_tensors or t in cast_leaves or t in boundaries or t in init_leaves: return FP16
        return dtype.get(t, FP32)

    new_nodes = []
    used_names = set(n.output[0] for n in g.node for _ in [0]) | set(inits) | inputs
    def fresh(base):
        nm = base
        k = 0
        while nm in used_names:
            k += 1; nm = f'{base}_{k}'
        used_names.add(nm); return nm

    # ---- in-cast boundaries: create fp16 copy, rewire cone consumers ----
    boundary_map = {}
    for b in boundaries:
        nm = fresh(b + '__rc16')
        new_nodes.append(helper.make_node('Cast', [b], [nm], to=FP16, name=fresh('cast_'+b)))
        boundary_map[b] = nm

    # ---- init leaves: dup fp16 if also used by non-cone fp32 consumers, else convert in place ----
    init_map = {}
    for iname in init_leaves:
        ini = inits[iname]
        arr = numpy_helper.to_array(ini).astype(np.float16)
        # does any consumer stay fp32? (consumer not converting this operand)
        cone_consumers = [c for c in consumers[iname] if any(o in conv_tensors or o in seeds for o in c.output)]
        noncone = [c for c in consumers[iname] if c not in cone_consumers]
        if noncone:
            nm = fresh(iname + '__f16')
            g.initializer.append(numpy_helper.from_array(arr, nm))
            init_map[iname] = nm
        else:
            ini.CopyFrom(numpy_helper.from_array(arr, iname))

    # ---- cast leaves: retarget or dup ----
    cast_map = {}
    for t in cast_leaves:
        cnode = producer[t]
        cone_cons = [c for c in consumers[t] if any(o in conv_tensors or o in seeds for o in c.output)]
        noncone = [c for c in consumers[t] if c not in cone_cons]
        if noncone:
            nm = fresh(t + '__f16')
            new_nodes.append(helper.make_node('Cast', list(cnode.input), [nm], to=FP16, name=fresh('castdup_'+t)))
            cast_map[t] = nm
        else:
            for a in cnode.attribute:
                if a.name == 'to': a.i = FP16

    # ---- rewire converted nodes' inputs; add back-casts for mixed sinks ----
    backcast_map = {}
    def as_fp16(inp, host_is_cone):
        # return the name to use for `inp` inside a cone node
        if inp in boundary_map: return boundary_map[inp]
        if inp in init_map: return init_map[inp]
        if inp in cast_map: return cast_map[inp]
        return inp
    def as_fp32(inp):
        # ensure fp32 view of a (possibly fp16) tensor for a non-cone op
        if new_dtype(inp) == FP16 and dtype.get(inp) != FP16:
            pass
        if inp not in backcast_map:
            nm = fresh(inp + '__f32')
            new_nodes.append(helper.make_node('Cast', [inp], [nm], to=FP32, name=fresh('bcast_'+inp)))
            backcast_map[inp] = nm
        return backcast_map[inp]

    conv_out = conv_tensors | set(seeds)  # tensors that are now fp16 (node outputs)
    for n in g.node:
        node_out_cone = any(o in conv_out for o in n.output)
        newn = helper.make_node(n.op_type, list(n.input), list(n.output), name=n.name)
        newn.attribute.extend(n.attribute)
        if node_out_cone or n.op_type == 'Cast' and any(o in conv_out for o in n.output):
            # this node produces fp16: route its float inputs to fp16 versions
            for k, inp in enumerate(n.input):
                if inp and is_float(inp):
                    newn.input[k] = as_fp16(inp, True)
        else:
            # non-cone node: if it reads a now-fp16 tensor and also a fp32 float sibling -> mix
            float_ins = [inp for inp in n.input if inp and is_float(inp)]
            reads_fp16 = [inp for inp in float_ins if new_dtype(inp) == FP16]
            reads_fp32 = [inp for inp in float_ins if new_dtype(inp) == FP32]
            if reads_fp16:
                if n.op_type in ('ArgMax','ArgMin','TopK','Cast','Shape','ReduceSum','ReduceMax',
                                 'ReduceMin','ReduceL1','Sign','Sqrt','Floor','Neg','Abs','Slice',
                                 'Unsqueeze','Squeeze','Reshape','Transpose','Gather','Split','Pad','CumSum'):
                    # unary-ish on float: fp16 input fine, output would be fp16 -> but node non-cone
                    # means its output consumed as fp32 downstream. To stay safe cast inputs back to fp32.
                    if reads_fp32 or True:
                        for k, inp in enumerate(n.input):
                            if inp in reads_fp16:
                                newn.input[k] = as_fp32(inp)
                else:
                    # binary/compare: cast fp16 operands back to fp32 to match siblings
                    for k, inp in enumerate(n.input):
                        if inp in reads_fp16:
                            newn.input[k] = as_fp32(inp)
        new_nodes.append(newn)

    # ---- topological sort ----
    avail = set(inits) | inputs | {i.name for i in g.initializer}
    ordered = []
    remaining = list(new_nodes)
    progress = True
    while remaining and progress:
        progress = False
        still = []
        for nd in remaining:
            if all((not i) or (i in avail) for i in nd.input):
                ordered.append(nd)
                for o in nd.output: avail.add(o)
                progress = True
            else:
                still.append(nd)
        remaining = still
    if remaining:
        # fallback: append leftovers (should not happen); keep to surface error
        ordered.extend(remaining)
    new_nodes = ordered

    del g.node[:]
    g.node.extend(new_nodes)
    # fix declared value_info dtypes for converted tensors (avoid stale fp32)
    keep_vi = [vi for vi in g.value_info if vi.name not in conv_out]
    del g.value_info[:]; g.value_info.extend(keep_vi)
    from src.harness import IR_VERSION
    m.ir_version = IR_VERSION
    return m


_SEEDS_117=['object_color']  # placeholder, overwritten below
import json as _json, os as _os
def _seeds117():
    ROOTd=_os.environ.get('NEUROGOLF_ROOT','/Users/minseong/project/neurogolf')
    SC=_json.load(open(_os.path.join(ROOTd,'reports/dtype_overpay_scan.json')))
    T=[x for x in SC['tasks'] if x['task']==117][0]
    return [te['name'] for te in T['tensors'] if te['class'] in ('FP16_SAFE','U8_CANDIDATE')]
def build(task):
    import os as _o
    r=_o.environ.get('NEUROGOLF_ROOT','/Users/minseong/project/neurogolf')
    globals()['ROOT']=r
    return recast_fp16(117, _seeds117(), root=r)
