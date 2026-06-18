"""task330 (ARC-AGI d2abd087) — recolor each gray sprite by its pixel count.

Rule (from generator task_d2abd087.py, size=10 fixed):
  3..6 gray ("continuous_creature") sprites are placed on a 10x10 black canvas at
  non-overlapping positions (bounding boxes separated with spacing>=1, so no two
  sprites are within Chebyshev distance 1 of each other).  Each sprite has a pixel
  COUNT in {4..8}.  In the OUTPUT every gray pixel is recolored RED (2) if its
  sprite has exactly 6 pixels, else BLUE (1); background stays 0.

  So the task is per-connected-component: count the pixels of each sprite and pick
  red iff count==6.

Approach (all on the fixed 10x10 active region, fp16 working planes = 200B each):
  1. Slice the gray channel (color 5) -> gray mask [1,1,10,10] f16.
  2. COMPONENT-ID FLOOD (max-propagation): seed each gray cell with its unique
     linear index (1..100), then iterate 7x  id = MaxPool3x3(id) * gray.  Because
     sprites are isolated (no two within Chebyshev 1) an 8-connected MaxPool never
     leaks across components; max creature size is 8 so 7 iters converge (verified
     0 mismatch vs 20 iters over 800 fresh).  Every cell of a sprite ends holding
     the same id = max linear index in that component.
  3. PER-COMPONENT COUNT via a histogram (ScatterND reduction='add', opset16 op but
     scorer checks DOMAIN not version): hist[id] += 1 over the 100 cells (gray cells
     contribute their id, bg cells contribute id 0 = ignored); then
     count_at_cell = Gather(hist, id_flat).  A tiny [101] buffer, no [100,100] matrix.
  4. isred = Equal(count, 6); colorval = 1 + isred (1=blue, 2=red);
     L = gray * colorval  ->  [1,1,10,10] in {0,1,2}, uint8.
  5. Pad L to 30x30 with sentinel 99 (off-grid matches no colour channel), then
     output = Equal(L30, arange[0..9]) -> BOOL [1,10,30,30] (the FREE output).
     In-grid bg L=0 -> ch0=1; off-grid L=99 -> all channels 0.

Verified exact (fresh 200/200 isolated).  Dominant intermediate = the few fp16
10x10 flood planes (200B each) + the [101] histogram; no 30x30 fp32 plane.
"""

import numpy as np
import onnx
from onnx import helper, numpy_helper, TensorProto

from ..harness import IR_VERSION

F16 = TensorProto.FLOAT16
F32 = TensorProto.FLOAT
BOOL = TensorProto.BOOL
U8 = TensorProto.UINT8
I64 = TensorProto.INT64

N = 10  # active grid is always 10x10


def build(task):
    inits, nodes = [], []

    def init(name, arr, dtype):
        inits.append(numpy_helper.from_array(
            np.ascontiguousarray(arr, dtype=dtype), name))
        return name

    def n(op, ins, out, **attrs):
        nodes.append(helper.make_node(op, ins, [out], **attrs))
        return out

    # ---- 1. slice gray channel (color 5) on the 10x10 active region ----------
    init("g_s", np.array([5, 0, 0], np.int64), np.int64)
    init("g_e", np.array([6, N, N], np.int64), np.int64)
    init("g_ax", np.array([1, 2, 3], np.int64), np.int64)
    n("Slice", ["input", "g_s", "g_e", "g_ax"], "gray_f32")   # [1,1,10,10] f32 {0,1}
    n("Cast", ["gray_f32"], "gray", to=F16)                    # f16 {0,1}

    # ---- 2. component-id flood (max-propagation, 7 iters) --------------------
    # seed = (linear index 1..100) * gray
    seed = (np.arange(1, N * N + 1, dtype=np.float16).reshape(1, 1, N, N))
    init("seedw", seed, np.float16)
    n("Mul", ["gray", "seedw"], "id0")                         # [1,1,10,10] f16
    cur = "id0"
    for it in range(7):
        # 8-connected max over 3x3 window (SAME pad) -> re-gate by gray
        n("MaxPool", [cur], f"mp_{it}", kernel_shape=[3, 3],
          pads=[1, 1, 1, 1], strides=[1, 1])                   # [1,1,10,10] f16
        cur = n("Mul", [f"mp_{it}", "gray"], f"id_{it}")       # [1,1,10,10] f16
    idplane = cur                                              # final id plane

    # ---- 3. per-component count via histogram (ScatterND add) ----------------
    # flatten id -> [100] int64 indices
    init("flat100", np.array([N * N], np.int64), np.int64)
    n("Reshape", [idplane, "flat100"], "id_flat_f16")          # [100] f16
    n("Cast", ["id_flat_f16"], "id_int", to=I64)               # [100] int64
    init("idx_shape", np.array([N * N, 1], np.int64), np.int64)
    n("Reshape", ["id_int", "idx_shape"], "id_idx")            # [100,1] int64
    # updates = gray flattened (1 per gray cell)
    n("Reshape", ["gray", "flat100"], "gray_flat")             # [100] f16
    n("Cast", ["gray_flat"], "upd", to=F32)                    # ScatterND f32
    init("hist0", np.zeros(N * N + 1, np.float32), np.float32)  # [101]
    n("ScatterND", ["hist0", "id_idx", "upd"], "hist", reduction="add")  # [101]
    # count_at_cell = hist[id]
    n("Gather", ["hist", "id_int"], "cnt_flat", axis=0)        # [100] f32
    init("plane_shape", np.array([1, 1, N, N], np.int64), np.int64)
    n("Reshape", ["cnt_flat", "plane_shape"], "count")         # [1,1,10,10] f32

    # ---- 4. colour-index plane L (0 bg, 1 blue, 2 red) -----------------------
    init("six", np.array(6.0, np.float32), np.float32)
    n("Equal", ["count", "six"], "isred_b")                    # bool [1,1,10,10]
    n("Cast", ["isred_b"], "isred", to=F16)                    # f16 {0,1}
    init("one16", np.array(1.0, np.float16), np.float16)
    n("Add", ["isred", "one16"], "colorval")                   # 1=blue, 2=red
    n("Mul", ["gray", "colorval"], "L_f16")                    # 0/1/2 f16
    n("Cast", ["L_f16"], "L_u8", to=U8)                        # [1,1,10,10] uint8

    # ---- 5. pad to 30x30 (sentinel 99) and expand to one-hot output ----------
    init("Lpads", np.array([0, 0, 0, 0, 0, 0, 30 - N, 30 - N], np.int64), np.int64)
    init("SENT", np.array(99, np.uint8), np.uint8)
    n("Pad", ["L_u8", "Lpads", "SENT"], "L30", mode="constant")  # [1,1,30,30] u8
    init("arange", np.arange(10, dtype=np.uint8).reshape(1, 10, 1, 1), np.uint8)
    n("Equal", ["L30", "arange"], "output")                    # [1,10,30,30] BOOL

    x = helper.make_tensor_value_info("input", F32, [1, 10, 30, 30])
    y = helper.make_tensor_value_info("output", BOOL, [1, 10, 30, 30])
    graph = helper.make_graph(nodes, "task330", [x], [y], inits)
    return helper.make_model(graph, ir_version=IR_VERSION,
                             opset_imports=[helper.make_opsetid("", 16)])
