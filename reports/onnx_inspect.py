"""Dump an ONNX net's structure for re-golf analysis: op list, params, and the
per-intermediate memory cost (declared dtype x shape) so the dominant plane is
obvious. Usage: python -m reports.onnx_inspect <path-or-tasknum> [--theirs]"""
import sys, pathlib, collections
import onnx
from onnx import shape_inference

# onnx elem_type -> bytes: f32=4,u8=1,i8=1,u16=2,i16=2,i32=4,i64=8,bool=1,f16=2,f64=8,u32=4,u64=8,bf16=2
ITEMSIZE = {1:4, 2:1, 3:1, 4:2, 5:2, 6:4, 7:8, 9:1, 10:2, 11:8, 12:4, 13:8, 16:2}
DT = {1:"f32",2:"u8",3:"i8",4:"u16",5:"i16",6:"i32",7:"i64",9:"bool",10:"f16",11:"f64",12:"u32",16:"bf16"}

def main():
    arg = sys.argv[1]
    if arg.endswith(".onnx"):
        path = arg
    else:
        n = int(arg)
        base = "/tmp/ng7k/extracted" if "--theirs" in sys.argv else "networks"
        path = f"{base}/task{n:03d}.onnx"
    m = onnx.load(path)
    try:
        m = shape_inference.infer_shapes(m)
    except Exception as e:
        print(f"(shape inference failed: {e})")
    g = m.graph
    ops = collections.Counter(n.op_type for n in g.node)
    print(f"== {path} ==")
    print(f"nodes={len(g.node)}  ops: {dict(ops)}")
    # params
    par = 0
    for init in g.initializer:
        c = 1
        for d in init.dims: c *= d
        par += c
    print(f"params(elem count)={par}")
    # intermediates: value_info + outputs (inputs are free)
    rows = []
    for vi in list(g.value_info) + list(g.output):
        tt = vi.type.tensor_type
        dt = tt.elem_type
        shape = []
        known = True
        for d in tt.shape.dim:
            if d.HasField("dim_value"): shape.append(d.dim_value)
            else: shape.append(d.dim_param or "?"); known = False
        if known and shape:
            n_elem = 1
            for s in shape: n_elem *= s
            cost = n_elem * ITEMSIZE.get(dt, 4)
        else:
            cost = -1
        rows.append((cost, vi.name, DT.get(dt, dt), tuple(shape)))
    rows.sort(reverse=True)
    print("top intermediates (declared dtype x shape -> bytes; -1=symbolic/unknown):")
    for cost, name, dt, shape in rows[:18]:
        print(f"  {cost:>8} {dt:>4} {str(shape):<22} {name}")
    print(f"(NOTE: scorer uses TRACE shapes via max(static,runtime); this is the declared-static view)")

if __name__ == "__main__":
    main()
