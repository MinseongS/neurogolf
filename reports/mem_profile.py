"""Per-tensor memory profiler for floor-break work.

Usage: PYTHONPATH=. .venv/bin/python reports/mem_profile.py N [topK]

Builds src/custom/taskNNN.py, runs a profiled ORT session on the stored
examples, and prints each intermediate tensor's byte cost (dtype x runtime
shape), sorted descending, so you can see exactly what to eliminate. Mirrors
src.harness.calculate_memory's accounting (declared value_info dtype x max
runtime shape from the ORT trace; `input`/`output` are free)."""
import importlib, json, math, os, sys, tempfile
import numpy as np
import onnx
import onnxruntime as ort

sys.path.insert(0, "/tmp/arc-gen")
from src.harness import load_task, sanitize_model

N = int(sys.argv[1])
TOPK = int(sys.argv[2]) if len(sys.argv) > 2 else 40

task = load_task(N)
mod = importlib.import_module(f"src.custom.task{N:03d}")
importlib.reload(mod)
model = mod.build(task)

sanitized = sanitize_model(model)
graph = onnx.shape_inference.infer_shapes(sanitized, strict_mode=True).graph

# declared dtype per intermediate tensor name
tensor_map = {t.name: t for t in list(graph.input) + list(graph.value_info) + list(graph.output)}
node_outputs = {node.name: list(node.output) for node in graph.node}
tensor_dtypes, tensor_static = {}, {}
for name, item in tensor_map.items():
    if name in ("input", "output"):
        continue
    if not item.type.HasField("tensor_type"):
        continue
    tt = item.type.tensor_type
    if not tt.HasField("shape"):
        continue
    ne = 1
    ok = True
    for dim in tt.shape.dim:
        if not dim.HasField("dim_value") or dim.dim_value <= 0:
            ok = False; break
        ne *= dim.dim_value
    dt = onnx.helper.tensor_dtype_to_np_dtype(tt.elem_type)
    tensor_dtypes[name] = dt
    if ok:
        tensor_static[name] = ne * np.dtype(dt).itemsize

# run profiled session to get runtime shapes
with tempfile.TemporaryDirectory() as tmp:
    so = ort.SessionOptions()
    so.enable_profiling = True
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    so.profile_file_prefix = os.path.join(tmp, "prof")
    sess = ort.InferenceSession(sanitized.SerializeToString(), so)
    in_name = sess.get_inputs()[0].name
    for ex in (task.get("train", []) + task.get("test", [])):
        from src.harness import convert_to_numpy
        bm = convert_to_numpy(ex)
        sess.run(None, {in_name: bm["input"].astype(np.float32)})
    trace_path = sess.end_profiling()
    trace = json.load(open(trace_path))

mem = dict(tensor_static)
for event in trace:
    if event.get("cat") != "Node" or "args" not in event:
        continue
    if "output_type_shape" not in event["args"]:
        continue
    nm = event.get("name").replace("_kernel_time", "")
    if nm not in node_outputs:
        continue
    for i, shape_dict in enumerate(event["args"]["output_type_shape"]):
        if i >= len(node_outputs[nm]):
            continue
        out = node_outputs[nm][i]
        if out not in tensor_dtypes:
            continue
        itemsize = np.dtype(tensor_dtypes[out]).itemsize
        m = itemsize * sum(math.prod(dims) for dims in shape_dict.values())
        mem[out] = max(mem.get(out, 0), m)

# map tensor -> producing op for readability
producer = {}
for node in graph.node:
    for o in node.output:
        producer[o] = node.op_type

total = sum(mem.values())
items = sorted(mem.items(), key=lambda kv: -kv[1])
print(f"task{N:03d}: total intermediate memory = {total} bytes  ({len(mem)} tensors)")
print(f"{'bytes':>8}  {'dtype':<8} {'op':<12} name")
for name, b in items[:TOPK]:
    dt = np.dtype(tensor_dtypes.get(name, np.float32)).name
    print(f"{b:>8}  {dt:<8} {producer.get(name,'?'):<12} {name}")
cum = sum(b for _, b in items[:TOPK])
print(f"... top{TOPK} = {cum}/{total} ({100*cum//max(1,total)}%)")
