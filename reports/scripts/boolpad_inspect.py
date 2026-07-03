import sys, onnx
from onnx import TensorProto, shape_inference
NAME={v:k for k,v in TensorProto.__dict__.items() if isinstance(v,int)}
task=int(sys.argv[1]); focus=sys.argv[2] if len(sys.argv)>2 else None
m=onnx.load(f"networks/task{task:03d}.onnx")
try: g=shape_inference.infer_shapes(m).graph
except Exception: g=m.graph
op=max([o.version for o in m.opset_import if o.domain in ("","ai.onnx")])
print("opset",op)
vimap={vi.name:vi for vi in list(g.value_info)+list(g.input)+list(g.output)}
def sh(n):
    vi=vimap.get(n)
    if not vi: return "?"
    d=vi.type.tensor_type.elem_type
    dims=[x.dim_value if x.HasField('dim_value') else '?' for x in vi.type.tensor_type.shape.dim]
    return f"{NAME.get(d,d)}{dims}"
producers={o:n for n in g.node for o in n.output}
consumers={}
for n in g.node:
    for i in n.input: consumers.setdefault(i,[]).append(n)
# find the cast(s)
for n in g.node:
    if n.op_type=="Cast" and (focus is None or focus in (n.name,)+tuple(n.output)):
        src=n.input[0]; dst=n.output[0]
        print(f"\n=== Cast {n.name}: {src}({sh(src)}) -> {dst}({sh(dst)})")
        p=producers.get(src)
        print(f"  producer of src: {p.op_type if p else 'INPUT/init'} {p.name if p else ''}")
        for c in consumers.get(dst,[]):
            slot=[i for i,x in enumerate(c.input) if x==dst]
            print(f"  consumer: {c.op_type} {c.name} slot{slot} out={c.output[0]}({sh(c.output[0])})")
            for cc in consumers.get(c.output[0],[]):
                print(f"     -> then: {cc.op_type} {cc.name} out={sh(cc.output[0])}")
