"""task057 (ARC-AGI 28bf18c6) — "crop the 3x3 sprite and tile it horizontally x2".

Rule (from the generator):
  An 8x8 grid holds a single-colour 3x3 sprite (a diagonally-connected conway
  sprite) placed at a variable top-left (row, col); background is 0.  The output
  is a 3x6 grid = the 3x3 sprite tiled twice horizontally:
      output[r][c] = output[r][c+3] = sprite[r][c]     (r,c in 0..2)
  Verified over all 265 stored instances: grid is always exactly 8x8, exactly one
  non-zero colour, the occupied bbox is exactly 3x3, and
  output == sprite[r0:r0+3, c0:c0+3] tiled x2  with r0/c0 = min occupied row/col.

Encoding (Tier B — data-dependent crop realized as two boolean shift-MatMuls):
  Work on the 8x8 active canvas.  presence P = (cell occupied).
  r0 = min occupied row, c0 = min occupied col (recovered as scalars).
  colour = the single non-zero colour (recovered as a scalar).
      Rmat[i,r]  = (r == i + r0)        i in 0..2   -> [1,1,3,8]
      CmatT[c,j] = (c == (j mod 3) + c0) j in 0..5  -> [1,1,8,6]
      sprite_tiled(3x6 presence) = Rmat @ P @ CmatT
  Build a 3x6 uint8 label L = colour where sprite_tiled>0 else 0, pad to 30x30 with
  a sentinel (10) so off-grid cells match no channel, then
      output = Equal(L, arange[0..9])  -> BOOL [1,10,30,30].
"""

import numpy as np
import onnx
from onnx import helper, numpy_helper, TensorProto

from ..harness import IR_VERSION

F32 = TensorProto.FLOAT
F16 = TensorProto.FLOAT16
U8 = TensorProto.UINT8
BOOL = TensorProto.BOOL
I64 = TensorProto.INT64

G = 8   # input active canvas (grid is always 8x8)
OH = 3  # output rows
OW = 6  # output cols (sprite width 3 tiled x2)
MS = 3  # sprite size


def build(task):
    inits, nodes = [], []

    def init(name, arr, dtype):
        inits.append(numpy_helper.from_array(
            np.ascontiguousarray(arr, dtype=dtype), name))
        return name

    def n(op, ins, out, **attrs):
        nodes.append(helper.make_node(op, ins, [out], **attrs))
        return out

    # ---- slice input to the 8x8 active canvas (all channels) ----------------
    # bg = channel 0 on the 8x8 canvas; full 10-channel slice for colour recovery.
    init("bg_s", np.array([0, 0, 0], np.int64), np.int64)
    init("bg_e", np.array([1, G, G], np.int64), np.int64)
    init("bg_ax", np.array([1, 2, 3], np.int64), np.int64)
    n("Slice", ["input", "bg_s", "bg_e", "bg_ax"], "bg")  # [1,1,G,G] f32 (1=empty)

    # ---- presence P = 1 - bg  (on the 8x8 canvas every cell is one-hot) ------
    # cast bg to fp16 once; P is fp16 {0,1} (exact) used by both ReduceMax and MatMul.
    n("Cast", ["bg"], "bg16", to=F16)
    init("ONEH", np.array(1.0, np.float16), np.float16)
    n("Sub", ["ONEH", "bg16"], "P16")      # [1,1,G,G] fp16 (1=occupied)

    init("ZEROH", np.array(0.0, np.float16), np.float16)

    # ---- colour scalar: which channels appear anywhere, weighted by index ----
    # collapse spatial first (input is free) -> per-channel presence [1,10,1,1].
    n("ReduceMax", ["input"], "chanpres", axes=[2, 3], keepdims=1)  # [1,10,1,1]
    init("chanw", np.arange(10, dtype=np.float32).reshape(1, 10, 1, 1), np.float32)
    n("Mul", ["chanpres", "chanw"], "chanwp")          # [1,10,1,1]
    n("ReduceMax", ["chanwp"], "color_f", axes=[1], keepdims=1)  # [1,1,1,1]

    # ---- min occupied row r0 / min occupied col c0 --------------------------
    init("BIG", np.array(1e6, np.float32), np.float32)

    def min_index(axis_keep, tag):
        red_axes = [3] if axis_keep == 2 else [2]
        n("ReduceMax", ["P16"], f"pres_{tag}", axes=red_axes, keepdims=1)
        n("Greater", [f"pres_{tag}", "ZEROH"], f"presb_{tag}")
        ramp = (np.arange(G, dtype=np.float32).reshape(1, 1, G, 1) if axis_keep == 2
                else np.arange(G, dtype=np.float32).reshape(1, 1, 1, G))
        init(f"ramp_{tag}", ramp, np.float32)
        n("Where", [f"presb_{tag}", f"ramp_{tag}", "BIG"], f"idx_{tag}")
        n("ReduceMin", [f"idx_{tag}"], f"min_{tag}", axes=[2, 3], keepdims=1)
        return f"min_{tag}"  # [1,1,1,1] f32 scalar

    r0 = min_index(2, "r0")
    c0 = min_index(3, "c0")

    # ---- Rmat[i,r] = (r == i + r0)  -> [1,1,OH,G] ----------------------------
    # i on axis2 (0..OH-1), r on axis3 (0..G-1).
    ivec = init("ivec", np.arange(OH, dtype=np.float32).reshape(1, 1, OH, 1), np.float32)
    rvec = init("rvec", np.arange(G, dtype=np.float32).reshape(1, 1, 1, G), np.float32)
    n("Add", [ivec, r0], "ipr0")                       # [1,1,OH,1] = i + r0
    n("Equal", ["rvec", "ipr0"], "Rmat_b")             # [1,1,OH,G] bool
    n("Cast", ["Rmat_b"], "Rmat", to=F16)

    # ---- CmatT[c,j] = (c == (j mod 3) + c0)  -> [1,1,G,OW] -------------------
    # c on axis2 (0..G-1), j on axis3 (0..OW-1). (j mod 3) is a fixed pattern.
    cvec = init("cvec", np.arange(G, dtype=np.float32).reshape(1, 1, G, 1), np.float32)
    jmod = init("jmod", (np.arange(OW) % MS).astype(np.float32).reshape(1, 1, 1, OW), np.float32)
    n("Add", ["jmod", c0], "jpc0")                     # [1,1,1,OW] = (j%3) + c0
    n("Equal", ["cvec", "jpc0"], "CmatT_b")            # [1,1,G,OW] bool
    n("Cast", ["CmatT_b"], "CmatT", to=F16)

    # ---- sprite_tiled = Rmat @ P16 @ CmatT  -> [1,1,OH,OW] presence ---------
    n("MatMul", ["Rmat", "P16"], "rm")                 # [1,1,OH,G] fp16
    n("MatMul", ["rm", "CmatT"], "tiled")              # [1,1,OH,OW] fp16
    n("Greater", ["tiled", "ZEROH"], "tiled_b")        # [1,1,OH,OW] bool

    # ---- label L = colour where sprite present else 0 -----------------------
    n("Cast", ["color_f"], "color_u8", to=U8)          # [1,1,1,1] uint8 scalar
    init("V0", np.array(0, np.uint8), np.uint8)
    n("Where", ["tiled_b", "color_u8", "V0"], "L3x6")  # [1,1,OH,OW] uint8

    # ---- pad to 30x30 with sentinel 10 (off-grid matches no channel) --------
    init("V10", np.array(10, np.uint8), np.uint8)
    init("pads", np.array([0, 0, 0, 0, 0, 0, 30 - OH, 30 - OW], np.int64), np.int64)
    n("Pad", ["L3x6", "pads", "V10"], "L", mode="constant")  # [1,1,30,30] uint8

    init("chan", np.arange(10, dtype=np.uint8).reshape(1, 10, 1, 1), np.uint8)
    n("Equal", ["L", "chan"], "output")                # [1,10,30,30] BOOL

    x = helper.make_tensor_value_info("input", F32, [1, 10, 30, 30])
    y = helper.make_tensor_value_info("output", BOOL, [1, 10, 30, 30])
    g = helper.make_graph(nodes, "task057", [x], [y], inits)
    return helper.make_model(g, ir_version=IR_VERSION,
                             opset_imports=[helper.make_opsetid("", 11)])
