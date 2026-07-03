import sys, onnx
from onnx import TensorProto, shape_inference
NAME={v:k for k,v in TensorProto.__dict__.items() if isinstance(v,int)}
task=int(sys.argv[1])
m=onnx.load(f"networks/task{task:03d}.onnx")
try: g=shape_inference.infer_shapes(m).graph
except Exception: g=m.graph
vimap={vi.name:vi for vi in list(g.value_info)+list(g.input)+list(g.output)}
initset={i.name:i for i in g.initializer}
producers={o:n for n in g.node for o in n.output}
consumers={}
for n in g.node:
    for i in n.input: consumers.setdefault(i,[]).append(n)
def sh(n):
    if n in initset:
        i=initset[n]; return f"INIT {NAME.get(i.data_type,i.data_type)}{list(i.dims)}"
    vi=vimap.get(n)
    if not vi: return "?"
    d=vi.type.tensor_type.elem_type
    dims=[x.dim_value if x.HasField('dim_value') else '?' for x in vi.type.tensor_type.shape.dim]
    return f"{NAME.get(d,d)}{dims}"
for n in g.node:
    if n.op_type in ("ScatterND","ScatterElements"):
        print(f"{n.op_type} {n.name}:")
        for i,inp in enumerate(n.input):
            p=producers.get(inp)
            print(f"  in{i} {inp} = {sh(inp)}  <- {p.op_type if p else ('INIT' if inp in initset else 'graph')}")
        print(f"  out {n.output[0]} = {sh(n.output[0])}")
        # downstream of output
        for c in consumers.get(n.output[0],[]):
            print(f"  out-consumer: {c.op_type} {c.name}")
        print("  IS GRAPH OUTPUT:", n.output[0] in {o.name for o in g.output})
