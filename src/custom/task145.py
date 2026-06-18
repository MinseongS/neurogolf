"""task145 (ARC-AGI 6455b5f5) — "fill the smallest region cyan(8), the largest region blue(1)".

Rule (from the generator):
  A grid is recursively bisected (guillotine cuts) into axis-aligned rectangular leaf
  regions; every cut line is painted red(2).  The INPUT shows ONLY the red lines (every
  other in-grid cell is background 0; off-grid cells are entirely empty).  In the OUTPUT
  every red line is kept, the leaf region(s) of MINIMUM area are filled cyan(8), the
  region(s) of MAXIMUM area are filled blue(1), all other cells stay 0.

  Every region is a solid rectangle bounded by red lines / the grid border, so NO
  flood-fill is needed:  a free cell's component area = (horizontal free-run length) *
  (vertical free-run length).  Run length = (nearest wall to the right) - (nearest wall to
  the left) - 1, where a "wall" is a red cell, an off-grid cell, or the grid border.
  Verified exactly (0/300) against the generator.

Encoding (active WORK=20 sub-canvas; grid is at most 20x20, top-left anchored):
  colf  = Conv(input, arange[1,10,1,1]) -> [1,1,30,30], Slice -> [1,1,20,20]
  ingrid= ReduceSum(input,axis=1) > 0   ;  wall = (~ingrid)|(colf==2) ; free = ~wall
  Nearest-wall positions via log-doubling cumulative max/min done in UINT8 (positions+1,
  fit in a byte; ORT has no uint8 Max/Min so use Where(Greater/Less,...)) -> 400B/plane.
  area = (RM-LM-1)*(DM-UM-1) in fp16 ; amin/amax over free cells.
  Lfinal = colf + 8*cyan + 1*blue, off-grid -> 99 ; Pad to 30x30 ; uint8 ;
  Equal(arange[1,10,1,1]) -> BOOL output (FREE).
"""

import numpy as np
import onnx
from onnx import helper, numpy_helper, TensorProto

from ..harness import IR_VERSION

F32 = TensorProto.FLOAT
F16 = TensorProto.FLOAT16
BOOL = TensorProto.BOOL
U8 = TensorProto.UINT8
I64 = TensorProto.INT64

N = 30
W = 20   # active sub-canvas


def build(task):
    inits, nodes = [], []

    def init(name, arr, dtype):
        inits.append(numpy_helper.from_array(
            np.ascontiguousarray(arr, dtype=dtype), name))
        return name

    def n(op, ins, out, **attrs):
        nodes.append(helper.make_node(op, ins, [out], **attrs))
        return out

    # The INPUT only ever contains colours 0 (bg) and 2 (red), so slice those two
    # channels directly on the WxW active region instead of a full-30 Conv/ReduceSum.
    init("c0_s", np.array([0, 0, 0], np.int64), np.int64)
    init("c0_e", np.array([1, W, W], np.int64), np.int64)
    init("c2_s", np.array([2, 0, 0], np.int64), np.int64)
    init("c2_e", np.array([3, W, W], np.int64), np.int64)
    init("ch_ax", np.array([1, 2, 3], np.int64), np.int64)
    n("Slice", ["input", "c0_s", "c0_e", "ch_ax"], "ch0")  # [1,1,W,W] f32 (bg one-hot)
    n("Slice", ["input", "c2_s", "c2_e", "ch_ax"], "ch2")  # [1,1,W,W] f32 (red one-hot)
    init("Z0", np.array(0.0, np.float32), np.float32)
    n("Greater", ["ch0", "Z0"], "isbg")
    n("Greater", ["ch2", "Z0"], "isred")

    # ---- wall / free ------------------------------------------------------------
    n("Or", ["isbg", "isred"], "ingrid")          # in-grid = bg or red
    n("Not", ["ingrid"], "offgrid")               # off-grid = neither channel set
    n("Or", ["offgrid", "isred"], "wall")         # wall = off-grid or red
    n("Not", ["wall"], "free")

    # ---- uint8 wall-position planes (position+1; 0 / 255 sentinels) --------------
    colp = (np.arange(W) + 1).astype(np.uint8).reshape(1, 1, 1, W)
    rowp = (np.arange(W) + 1).astype(np.uint8).reshape(1, 1, W, 1)
    init("colp", colp, np.uint8)
    init("rowp", rowp, np.uint8)
    init("U0", np.array(0, np.uint8), np.uint8)
    init("U255", np.array(255, np.uint8), np.uint8)
    init("UWB", np.array(W + 1, np.uint8), np.uint8)   # right-border wall position+1
    # cummax inputs (left wall): wall ? pos+1 : 0
    n("Where", ["wall", "colp", "U0"], "wcolmax")   # uint8 [1,1,W,W]
    n("Where", ["wall", "rowp", "U0"], "wrowmax")
    # cummin inputs (right wall): wall ? pos+1 : 255
    n("Where", ["wall", "colp", "U255"], "wcolmin")
    n("Where", ["wall", "rowp", "U255"], "wrowmin")

    # ---- uint8 log-doubling cumulative max / min --------------------------------
    def cum(name_in, kind, axis):
        cur = name_in
        s = 1
        step = 0
        # cummax border-left = 0 (=> Lwall=-1); cummin border-right = W+1 (=> Rwall=W)
        fill = "U0" if kind == "max" else "UWB"
        while s < W:
            sl_a = init(f"{name_in}_sa{step}", np.array([axis], np.int64), np.int64)
            pads = [0] * 8
            if kind == "max":
                sl_s = init(f"{name_in}_ss{step}", np.array([0], np.int64), np.int64)
                sl_e = init(f"{name_in}_se{step}", np.array([W - s], np.int64), np.int64)
                pads[axis] = s
            else:
                sl_s = init(f"{name_in}_ss{step}", np.array([s], np.int64), np.int64)
                sl_e = init(f"{name_in}_se{step}", np.array([W], np.int64), np.int64)
                pads[4 + axis] = s
            sliced = n("Slice", [cur, sl_s, sl_e, sl_a], f"{name_in}_sl{step}")
            pd = init(f"{name_in}_pd{step}", np.array(pads, np.int64), np.int64)
            padded = n("Pad", [sliced, pd, fill], f"{name_in}_pad{step}", mode="constant")
            if kind == "max":
                cmp = n("Greater", [cur, padded], f"{name_in}_gt{step}")
            else:
                cmp = n("Less", [cur, padded], f"{name_in}_lt{step}")
            comb = n("Where", [cmp, cur, padded], f"{name_in}_cm{step}")
            cur = comb
            s *= 2
            step += 1
        return cur

    LM = cum("wcolmax", "max", 3)   # = Lwall+1
    RM = cum("wcolmin", "min", 3)   # = Rwall+1
    UM = cum("wrowmax", "max", 2)
    DM = cum("wrowmin", "min", 2)

    # ---- run lengths and area (fp16) --------------------------------------------
    n("Cast", [LM], "LMf", to=F16)
    n("Cast", [RM], "RMf", to=F16)
    n("Cast", [UM], "UMf", to=F16)
    n("Cast", [DM], "DMf", to=F16)
    init("ONE", np.array(1.0, np.float16), np.float16)
    n("Sub", ["RMf", "LMf"], "rwgap")        # RM-LM
    n("Sub", ["rwgap", "ONE"], "rw")         # -1
    n("Sub", ["DMf", "UMf"], "rhgap")
    n("Sub", ["rhgap", "ONE"], "rh")
    n("Mul", ["rw", "rh"], "area")           # [1,1,W,W] f16

    # ---- global min / max area over FREE cells ----------------------------------
    # At wall cells rw=rh=-1 so area=1 (never exceeds a free region's max, since the
    # generator guarantees a region of area > 1 exists) => amax = ReduceMax(area) needs
    # no mask.  amin DOES need masking (a 1x1 free region also has area 1).
    init("BIGA", np.array(2000.0, np.float16), np.float16)
    n("Where", ["free", "area", "BIGA"], "area_for_min")
    n("ReduceMin", ["area_for_min"], "amin", axes=[2, 3], keepdims=1)
    n("ReduceMax", ["area"], "amax", axes=[2, 3], keepdims=1)

    # ---- selection masks --------------------------------------------------------
    n("Equal", ["area", "amin"], "is_min_area")
    n("And", ["free", "is_min_area"], "cyan_mask")
    # max area is >= 2 (generator guarantees min<max) while wall cells have area 1, so
    # area==amax already implies free -> no extra And needed.
    n("Equal", ["area", "amax"], "blue_mask")

    # ---- final colour-index plane (uint8, priority Where chain) -----------------
    # offgrid -> 99 sentinel (Equal->all-zero); red -> 2 ; min-area -> 8 ; max-area -> 1.
    init("V99", np.array(99, np.uint8), np.uint8)
    init("V2", np.array(2, np.uint8), np.uint8)
    init("V8", np.array(8, np.uint8), np.uint8)
    init("V1", np.array(1, np.uint8), np.uint8)
    n("Where", ["blue_mask", "V1", "U0"], "Lb")          # uint8 [1,1,W,W]
    n("Where", ["cyan_mask", "V8", "Lb"], "Lc")
    n("Where", ["isred", "V2", "Lc"], "Lr")
    n("Where", ["offgrid", "V99", "Lr"], "Lfinal")

    # ---- pad to 30x30 (uint8), Equal(arange) -> BOOL output ---------------------
    init("padsW", np.array([0, 0, 0, 0, 0, 0, N - W, N - W], np.int64), np.int64)
    n("Pad", ["Lfinal", "padsW", "V99"], "Lpad", mode="constant")  # [1,1,30,30] u8
    init("ar10", np.arange(10, dtype=np.uint8).reshape(1, 10, 1, 1), np.uint8)
    n("Equal", ["Lpad", "ar10"], "output")    # [1,10,30,30] BOOL (FREE)

    x = helper.make_tensor_value_info("input", F32, [1, 10, 30, 30])
    y = helper.make_tensor_value_info("output", BOOL, [1, 10, 30, 30])
    graph = helper.make_graph(nodes, "task145", [x], [y], inits)
    return helper.make_model(graph, ir_version=IR_VERSION,
                             opset_imports=[helper.make_opsetid("", 11)])
