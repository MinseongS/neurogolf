"""task229 (ARC-AGI 9565186b) — keep the majority colour, gray out the rest.

Rule (from the generator, verified fresh):
  A FIXED 3x3 grid is filled with colours drawn from a 2-4 colour palette
  (gray=5 excluded; `square_with_unique_max_color` guarantees a UNIQUE mode).
  mode = the most frequent colour. Output cell = its colour if it equals the
  mode, else gray(5). Off-grid stays background 0.

Closed-form (NO argmax/gather chains, everything on the tiny 3x3 active grid):
  cnt   = ReduceSum(input[:,:,:3,:3], axes=[2,3])  -> [1,10,1,1] per-colour count
  mode1 = Equal(cnt, ReduceMax(cnt, axis=1))       -> [1,10,1,1] one-hot of mode
          (unique mode => exactly one channel true; ch0 counts too but a non-mode
           background cell still gets gray, so no bg masking needed)
  colf  = Conv(input, w=[0,1,..,9])  -> [1,1,3,3] colour index of each cell
  modec = Conv(mode1, w=[0,..,9])    -> [1,1,1,1] the mode colour value
  out_idx = Where(colf == modec, colf, 5)          -> [1,1,3,3] index plane
  Pad to 30x30 with 0 (off-grid background) then Equal(arange) -> FREE bool output.

Memory: only tiny [1,10,1,1] and [1,1,3,3] working tensors; the [1,10,30,30]
expansion lands in the FREE output of the final Equal.
"""

import numpy as np
from onnx import helper, numpy_helper, TensorProto

from ..harness import IR_VERSION

F32 = TensorProto.FLOAT
F16 = TensorProto.FLOAT16
B = TensorProto.BOOL
S = 30
G = 3  # active grid is always 3x3


def build(task):
    inits, nodes = [], []

    def init(name, arr, dt=None):
        inits.append(numpy_helper.from_array(np.ascontiguousarray(arr), name))
        return name

    def n(op, ins, out, **attrs):
        nodes.append(helper.make_node(op, ins, [out], **attrs))
        return out

    # --- crop the 3x3 active grid ---
    init("s", np.array([0, 0], np.int64), np.int64)
    init("e", np.array([G, G], np.int64), np.int64)
    init("ax", np.array([2, 3], np.int64), np.int64)
    n("Slice", ["input", "s", "e", "ax"], "grid")  # [1,10,3,3] fp32

    # --- per-colour pixel count over the 3x3 grid ---
    n("ReduceSum", ["grid"], "cnt", axes=[2, 3], keepdims=1)  # [1,10,1,1]

    # --- mode one-hot (unique mode guaranteed) ---
    n("ReduceMax", ["cnt"], "cmax", axes=[1], keepdims=1)  # [1,1,1,1]
    n("Equal", ["cnt", "cmax"], "mode_b")  # [1,10,1,1] bool
    n("Cast", ["mode_b"], "mode_f", to=F32)  # [1,10,1,1] fp32

    # --- colour-index of each cell: 1x1 Conv with weight [0,1,..,9] ---
    w = np.arange(10, dtype=np.float32).reshape(1, 10, 1, 1)
    init("wcol", w, F32)
    n("Conv", ["grid", "wcol"], "colf")  # [1,1,3,3] colour index

    # --- mode colour value (scalar plane) ---
    n("Conv", ["mode_f", "wcol"], "modec")  # [1,1,1,1] mode colour

    # --- out index = colf if colf==mode else gray(5) ---
    n("Equal", ["colf", "modec"], "ismode")  # [1,1,3,3] bool (broadcast)
    init("grayp", np.full((1, 1, G, G), 5.0, np.float32), F32)
    n("Where", ["ismode", "colf", "grayp"], "outidx")  # [1,1,3,3]

    # --- expand to one-hot on the 3x3 grid (off-grid stays all-zero) ---
    init("arange", np.arange(10, dtype=np.float32).reshape(1, 10, 1, 1), F32)
    n("Equal", ["outidx", "arange"], "oneh_b")  # [1,10,3,3] bool
    n("Cast", ["oneh_b"], "oneh_f", to=F16)     # [1,10,3,3] fp16 (Pad rejects bool)

    # --- pad to 30x30 with 0 -> off-grid is all-zero (matches encoding) ---
    init("pads", np.array([0, 0, 0, 0, 0, 0, S - G, S - G], np.int64), np.int64)
    init("z16", np.array(0.0, np.float16), np.float16)
    n("Pad", ["oneh_f", "pads", "z16"], "output", mode="constant")  # [1,10,30,30] fp16

    x = helper.make_tensor_value_info("input", F32, [1, 10, S, S])
    y = helper.make_tensor_value_info("output", F16, [1, 10, S, S])
    g = helper.make_graph(nodes, "task229", [x], [y], inits)
    return helper.make_model(g, ir_version=IR_VERSION,
                             opset_imports=[helper.make_opsetid("", 11)])
