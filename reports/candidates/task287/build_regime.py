import onnx, numpy as np
from onnx import helper, numpy_helper, TensorProto

# S[h,x]: fold one-hot, h in 0..15 -> x=min(h,15-h); rows 16..29 zero (off-grid auto-zero)
S = np.zeros((30, 8), np.float32)
for h in range(16):
    S[h, min(h, 15 - h)] = 1.0
# gate: noise colors 0 and 4 excluded
gate = np.ones(10, np.float32)
gate[0] = 0.0
gate[4] = 0.0

# output[n,c,h,w] = sum_{a,b,x,y} input[n,c,a,b] * S[a,x] * S[b,y] * S[h,x] * S[w,y] * gate[c]
#                = (# clean copies of color c in the mirror orbit of (h,w)), gated
node = helper.make_node(
    "Einsum",
    ["input", "S", "S", "S", "S", "gate"],
    ["output"],
    equation="ncab,ax,by,hx,wy,c->nchw",
)

graph = helper.make_graph(
    [node],
    "task287_regime",
    [helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 10, 30, 30])],
    [helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 10, 30, 30])],
    [numpy_helper.from_array(S, "S"), numpy_helper.from_array(gate, "gate")],
)
model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
model.ir_version = 8
onnx.checker.check_model(model)
onnx.save(model, "/Users/minseong/project/neurogolf/reports/candidates/task287/regime.onnx")
print("built")
