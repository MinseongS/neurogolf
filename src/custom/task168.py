"""task168 (ARC-AGI 6e19193c) — arrow rays from 2x2 blocks with one missing corner.

Rule (generator task_6e19193c.py, size=10 fixed):
  2..3 "arrows", single colour `color` per instance.  Each arrow = a 2x2 block of
  `color` with ONE corner left black (the "tip"); the missing corner indicates a
  diagonal direction d=(dr,dc) pointing OUTWARD (away from the block).  The output
  keeps the 3 coloured corners (missing corner stays empty) AND draws a diagonal ray
  of `color` starting ONE step outside the missing corner, i.e. cells
  (i + t*dr, j + t*dc) for t=1,2,...  until the grid edge.  Blocks are >=4 apart so
  2x2 windows are clean (verified 0/500 fresh against the numpy reference).

Encoding:
  occ = (input non-background) collapsed to a single [1,1,10,10] plane (fp16 {0,1}).
  For each of the 4 diagonal directions d, a "missing-corner tip" is detected with a
  small Conv: response = (+1 on the 3 ON corners) + (-3 on the missing corner); ==3
  iff exactly the L-shape with that corner missing.  The ray for direction d is a
  bounded directional prefix-OR (offsets t=1..9 along -d) = one 10x10 diagonal Conv +
  >0.  OR the 4 ray planes with the original occ to get the output occupancy.
  Colour is a single scalar c = sum_k k*(per-channel pixel-count>0) (40B, no full
  plane); L = Where(outocc, c, 0) uint8, padded with sentinel 255 off-grid,
  output = Equal(L, arange[0..9]) -> BOOL FREE output.

  pts 16.32, mem 5393, params 479, isolated fresh 500/500.  Dominant intermediate =
  the 12 fp16 [1,1,10,10] Conv/cast planes (200B each) across the 4 directions; the
  detection (==3 L-match) is genuinely nonlinear so it must precede the ray
  propagation, blocking a single fused conv.
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
            np.ascontiguousarray(arr, dtype=arr.dtype), name))
        return name

    def n(op, ins, out, **attrs):
        nodes.append(helper.make_node(op, ins, [out], **attrs))
        return out

    # ---- colour scalar from per-channel pixel COUNTS (40B, no full plane) --
    # cnt = ReduceSum(input, [2,3]) -> [1,10,1,1]; exactly one colour channel k>=1
    # is nonzero, so colour = sum_k k*(cnt_k>0).
    n("ReduceSum", ["input"], "cnt", axes=[2, 3], keepdims=1)  # [1,10,1,1] f32
    init("ZF", np.array(0.0, np.float32), F32)
    init("ZF_half", np.array(0.5, np.float32), F32)
    n("Greater", ["cnt", "ZF"], "cnt_pos_b")                   # [1,10,1,1] bool
    n("Cast", ["cnt_pos_b"], "cnt_pos", to=F16)               # [1,10,1,1] f16
    init("kvec", np.arange(10, dtype=np.float16).reshape(1, 10, 1, 1), F16)
    n("Mul", ["cnt_pos", "kvec"], "kpos")                     # [1,10,1,1] f16
    n("ReduceSum", ["kpos"], "cmax", axes=[1], keepdims=1)    # [1,1,1,1] f16 colour

    # ---- occupancy plane on the 10x10 active region ----------------------
    # occ = non-background on the 10x10 grid.  Background channel 0 is 1 at bg cells
    # and 0 at coloured cells, so occ = (ch0 == 0); slice ONLY ch0 (400B) not all 9.
    init("sl_s", np.array([0, 0, 0], np.int64), np.int64)
    init("sl_e", np.array([1, N, N], np.int64), np.int64)
    init("sl_ax", np.array([1, 2, 3], np.int64), np.int64)
    n("Slice", ["input", "sl_s", "sl_e", "sl_ax"], "bg10")    # [1,1,10,10] f32
    n("Less", ["bg10", "ZF_half"], "occ_b")                   # bg<0.5 -> coloured
    n("Cast", ["occ_b"], "occ", to=F16)                       # [1,1,10,10] f16 {0,1}

    # ---- per-direction missing-corner tip detection ----------------------
    # direction d=(dr,dc) outward; the 3 ON corners relative to missing (i,j):
    #   (0,-dc), (-dr,0), (-dr,-dc) ; missing corner (0,0).
    # 2x2 conv window covers rows {0,-dr} cols {0,-dc}.  Build a 3x3 kernel anchored
    # at centre = missing corner, weights +1 on the 3 ON corner offsets and -3 at
    # centre; SAME pad=1.  Response==3 iff exactly that L-shape (corner off, 3 on).
    ZH = init("ZH", np.array(0.0, np.float16), F16)
    THREE = init("THREE", np.array(3.0, np.float16), F16)

    dirs = {
        "TR": (-1, 1),   # up-right
        "TL": (-1, -1),  # up-left
        "BR": (1, 1),    # down-right
        "BL": (1, -1),   # down-left
    }

    ray_planes = []
    for tag, (dr, dc) in dirs.items():
        # 3x3 detection kernel, centre at (1,1): +1 on 3 ON corners, -3 on missing.
        K = np.zeros((1, 1, 3, 3), np.float16)
        K[0, 0, 1, 1] = -3.0
        for (a, b) in [(0, -dc), (-dr, 0), (-dr, -dc)]:
            K[0, 0, 1 + a, 1 + b] = 1.0
        init(f"Kdet_{tag}", K, F16)
        n("Conv", ["occ", f"Kdet_{tag}"], f"resp_{tag}", pads=[1, 1, 1, 1])
        n("Equal", [f"resp_{tag}", "THREE"], f"tip_b_{tag}")
        n("Cast", [f"tip_b_{tag}"], f"tip_{tag}", to=F16)

        # ---- ray: prefix-OR of tip along +d, offsets t=1..9, via a 10x10 kernel +
        # asymmetric SAME pad: output(p,q)=sum K[ki,kj]*tip(p+ki-pt, q+kj-pl).
        Kr = np.zeros((1, 1, N, N), np.float16)
        pt = 0 if -dr > 0 else (N - 1)
        pl = 0 if -dc > 0 else (N - 1)
        for t in range(1, N):
            Kr[0, 0, (-dr * t) + pt, (-dc * t) + pl] = 1.0
        init(f"Kray_{tag}", Kr, F16)
        n("Conv", [f"tip_{tag}", f"Kray_{tag}"], f"rayresp_{tag}",
          pads=[pt, pl, (N - 1) - pt, (N - 1) - pl])
        n("Greater", [f"rayresp_{tag}", "ZH"], f"ray_b_{tag}")
        ray_planes.append(f"ray_b_{tag}")

    # ---- combine: outocc = occ OR all rays -------------------------------
    n("Or", [ray_planes[0], ray_planes[1]], "or01")
    n("Or", [ray_planes[2], ray_planes[3]], "or23")
    n("Or", ["or01", "or23"], "rays_b")
    n("Or", ["occ_b", "rays_b"], "outocc_b")          # [1,1,10,10] bool

    # ---- colour-index L = outocc ? colour : 0 ----------------------------
    n("Cast", ["cmax"], "cmax_u8", to=U8)             # scalar colour uint8
    init("Z_u8", np.array(0, np.uint8), U8)
    n("Where", ["outocc_b", "cmax_u8", "Z_u8"], "L_u8")  # [1,1,10,10] uint8 0..9

    # ---- pad L to 30x30 with sentinel 255 (off-grid matches no colour) ----
    init("Lpads", np.array([0, 0, 0, 0, 0, 0, 30 - N, 30 - N], np.int64), I64)
    init("SENT", np.array(255, np.uint8), U8)
    n("Pad", ["L_u8", "Lpads", "SENT"], "L30", mode="constant")  # [1,1,30,30] u8

    # ---- output = Equal(L, arange[0..9]) -> BOOL [1,10,30,30] FREE --------
    init("arange", np.arange(10, dtype=np.uint8).reshape(1, 10, 1, 1), U8)
    n("Equal", ["L30", "arange"], "output")           # [1,10,30,30] BOOL

    x = helper.make_tensor_value_info("input", F32, [1, 10, 30, 30])
    y = helper.make_tensor_value_info("output", BOOL, [1, 10, 30, 30])
    graph = helper.make_graph(nodes, "task168", [x], [y], inits)
    return helper.make_model(graph, ir_version=IR_VERSION,
                             opset_imports=[helper.make_opsetid("", 11)])
