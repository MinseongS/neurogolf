"""task081 (ARC-AGI 3aa6fb7a) — fill the missing corner of each L-tromino with blue.

Rule (from generator, size=7 fixed):
  Several L-trominoes are placed on a 7x7 grid: each occupies a 2x2 block with
  exactly 3 cells cyan (color 8) and one corner empty (background 0).  Shapes are
  isolated (generator's has_neighbor guard keeps the 2x2 blocks from touching).
  OUTPUT: the 3 cyan cells stay cyan; the empty 4th corner becomes blue (color 1).

  Local rule (exact): a background cell (r,c) becomes blue iff it is the missing
  corner of some 2x2 block whose other 3 cells are all cyan.  There are 4 such
  blocks (the cell as TL/TR/BL/BR corner); since shapes are isolated, at most one
  matches.  Cyan cells are copied unchanged.

Encoding:
  - C = cyan plane = channel 8 of input, sliced to the 7x7 active region (fp16).
  - For each of the 4 corner roles, a 3x3 SAME-pad Conv sums the 3 cyan cells of
    that block; ==3 (Equal) means the block is a complete L around (r,c).  OR the
    4 -> any_corner; AND with bg=(C==0) -> blue mask.
  - Output one-hot routed into the FREE bool output:
      ch8 (cyan) = C>0 ; ch1 (blue) = blue mask ; ch0 (bg) = NOT(cyan OR blue) in
      the 7x7 region, 0 off-grid; all other channels 0.
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

N = 7  # active grid is always 7x7


def build(task):
    inits, nodes = [], []

    def init(name, arr, dtype):
        inits.append(numpy_helper.from_array(
            np.ascontiguousarray(arr, dtype=dtype), name))
        return name

    def n(op, ins, out, **attrs):
        nodes.append(helper.make_node(op, ins, [out], **attrs))
        return out

    # ---- cyan plane: channel 8, active 7x7 region -------------------------
    init("sl_s", np.array([8, 0, 0], np.int64), np.int64)
    init("sl_e", np.array([9, N, N], np.int64), np.int64)
    init("sl_ax", np.array([1, 2, 3], np.int64), np.int64)
    n("Slice", ["input", "sl_s", "sl_e", "sl_ax"], "C_f32")   # [1,1,7,7] f32
    n("Cast", ["C_f32"], "C", to=F16)                          # [1,1,7,7] f16 {0,1}

    # ---- four corner-role convs (3x3 SAME pad) ----------------------------
    # cell (r,c) as TL of block -> needs (r,c+1),(r+1,c),(r+1,c+1).
    #   In a 3x3 kernel centered at (r,c) (index 1,1), these are positions
    #   (1,2),(2,1),(2,2).
    # TR -> needs (r,c-1),(r+1,c-1),(r+1,c): positions (1,0),(2,0),(2,1).
    # BL -> needs (r-1,c),(r-1,c+1),(r,c+1): positions (0,1),(0,2),(1,2).
    # BR -> needs (r-1,c-1),(r-1,c),(r,c-1): positions (0,0),(0,1),(1,0).
    ZH = init("ZH", np.array(0.0, np.float16), np.float16)
    THREE = init("THREE", np.array(3.0, np.float16), np.float16)

    roles = {
        "TL": [(1, 2), (2, 1), (2, 2)],
        "TR": [(1, 0), (2, 0), (2, 1)],
        "BL": [(0, 1), (0, 2), (1, 2)],
        "BR": [(0, 0), (0, 1), (1, 0)],
    }
    corner_masks = []
    for tag, coords in roles.items():
        K = np.zeros((1, 1, 3, 3), np.float16)
        for (i, j) in coords:
            K[0, 0, i, j] = 1.0
        init(f"K_{tag}", K, np.float16)
        n("Conv", ["C", f"K_{tag}"], f"cv_{tag}", pads=[1, 1, 1, 1])  # [1,1,7,7]
        n("Equal", [f"cv_{tag}", "THREE"], f"m_{tag}")                # bool
        corner_masks.append(f"m_{tag}")

    # OR the four corner roles
    n("Or", [corner_masks[0], corner_masks[1]], "or01")
    n("Or", [corner_masks[2], corner_masks[3]], "or23")
    n("Or", ["or01", "or23"], "any_corner")                # [1,1,7,7] bool

    # blue cell = any_corner AND bg(not cyan)
    n("Greater", ["C", "ZH"], "cyan_b")                    # bool, cyan present
    n("Not", ["cyan_b"], "bg_b")                           # bool, empty cell
    n("And", ["any_corner", "bg_b"], "blue_b")             # [1,1,7,7] bool

    # ---- build colour-index plane L (0=bg, 1=blue, 8=cyan) ----------------
    # L = 1*blue + 8*cyan  (blue and cyan are disjoint).
    n("Cast", ["blue_b"], "blue16", to=F16)
    n("Mul", ["C", "C8w"], "cyan8")                        # 8*cyan
    init("C8w", np.array(8.0, np.float16), np.float16)
    n("Add", ["blue16", "cyan8"], "L_f16")                 # [1,1,7,7] f16
    n("Cast", ["L_f16"], "L_u8", to=U8)                    # uint8 0/1/8

    # ---- pad L to 30x30 with sentinel 255 (off-grid matches no colour) ----
    init("Lpads", np.array([0, 0, 0, 0, 0, 0, 30 - N, 30 - N], np.int64), np.int64)
    init("SENT", np.array(255, np.uint8), np.uint8)
    n("Pad", ["L_u8", "Lpads", "SENT"], "L30", mode="constant")  # [1,1,30,30] u8

    # ---- output = Equal(L, arange[0..9]) -> BOOL [1,10,30,30] (FREE) ------
    init("arange", np.arange(10, dtype=np.uint8).reshape(1, 10, 1, 1), np.uint8)
    n("Equal", ["L30", "arange"], "output")            # [1,10,30,30] BOOL

    x = helper.make_tensor_value_info("input", F32, [1, 10, 30, 30])
    y = helper.make_tensor_value_info("output", BOOL, [1, 10, 30, 30])
    graph = helper.make_graph(nodes, "task081", [x], [y], inits)
    return helper.make_model(graph, ir_version=IR_VERSION,
                             opset_imports=[helper.make_opsetid("", 11)])
