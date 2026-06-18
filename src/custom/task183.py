"""Task 183 (ARC-AGI 77fdfe62) — quadrant-recolour the cyan pixels.

Rule (from the generator): the input is an (size+4)x(size+4) grid, size in {2,4,6}.
A blue (1) frame sits at rows/cols {1, size+2}.  The 4 OUTER corner cells carry
four colours:
    in[0][0]            = TL  (colors[0])
    in[0][size+3]       = TR  (colors[1])
    in[size+3][0]       = BL  (colors[2])
    in[size+3][size+3]  = BR  (colors[3])
Inside the frame (rows/cols 2..size+1) sit cyan (8) pixels at in[r+2][c+2].
The OUTPUT is a size x size grid: every cyan pixel (r,c) is recoloured by the
QUADRANT it falls in (relative to size//2):
    r< h, c< h -> TL    r< h, c>=h -> TR
    r>=h, c< h -> BL    r>=h, c>=h -> BR     (h = size//2)
non-pixel cells are background (0); everything outside size x size is unset.

Encoding (no full 30x30 working plane):
  * size = ReduceMax(ReduceSum(input,[1,3])) - 4   (per-row cell count peaks at
    size+4 on grid rows, 0 on padding rows) -> a scalar, [1,1,30,1] is 120B.
  * The 4 corner colours come from tiny [1,1,1,30] colour-index rows (row 0 is a
    fixed Slice; row size+3 is a data-dependent Gather), each reduced over
    channels with a 0..9 ramp Conv; cols 0 and size+3 picked by Gather.
  * quadrant colour plane qcol[6,6] is built SEPARABLY from rowhi=(2r>=size) and
    colhi=(2c>=size) selecting among {TL,TR,BL,BR}.
  * cyan presence is Slice in[8, 2:8, 2:8] -> 6x6 (pads with grid 0s for size<6).
  * in-region mask = (r<size)&(c<size).
  * lab6 = Where(in-region, Where(cyan, qcol, 0), 99)  -- a 6x6 uint8 label.
  * Pad lab6 to 30x30 with sentinel 99, then Equal(lab30_u8, arange[1,10,1,1])
    routes the whole 10-channel one-hot into the FREE bool output.  The only
    full-canvas tensor is the 900B uint8 padded label.
"""

import numpy as np
import onnx
from onnx import helper, numpy_helper, TensorProto

from ..harness import IR_VERSION

F32 = TensorProto.FLOAT
F16 = TensorProto.FLOAT16
I64 = TensorProto.INT64
I32 = TensorProto.INT32
U8 = TensorProto.UINT8
B = TensorProto.BOOL


def build(task):
    inits, nodes = [], []

    def init(name, arr, dtype):
        inits.append(numpy_helper.from_array(
            np.ascontiguousarray(arr, dtype=dtype), name))
        return name

    def n(op, ins, out, **attrs):
        nodes.append(helper.make_node(op, ins, [out], **attrs))
        return out

    # ---------- size scalar (no full-canvas plane) ----------
    # every grid cell sets exactly one channel, so total one-hot sum = (size+4)^2.
    n("ReduceSum", ["input"], "total", axes=[1, 2, 3], keepdims=1)  # [1,1,1,1] f32
    init("half", np.array([[[[0.5]]]], np.float32), np.float32)
    n("Pow", ["total", "half"], "side")                            # [1,1,1,1] = size+4
    init("four", np.array([[[[4.0]]]], np.float32), np.float32)
    n("Sub", ["side", "four"], "sizef")                            # [1,1,1,1] = size

    # ---------- corner colours ----------
    # TL=(0,0) is FIXED -> cheap Slice + channel-ramp reduce (40B plane).
    init("ax23s", np.array([2, 3], np.int64), np.int64)
    init("z2", np.array([0, 0], np.int64), np.int64)
    init("o2", np.array([1, 1], np.int64), np.int64)
    n("Slice", ["input", "z2", "o2", "ax23s"], "tlcell")       # [1,10,1,1]
    ramp4 = np.arange(10, dtype=np.float32).reshape(1, 10, 1, 1)
    init("ramp4", ramp4, np.float32)
    n("Conv", ["tlcell", "ramp4"], "TLf")                      # [1,1,1,1] = TL
    n("Cast", ["TLf"], "TL", to=F16)

    # The 3 data-dependent corners (0,s3),(s3,0),(s3,s3) via GatherND.
    init("three_i", np.array([[[3.0]]], np.float32), np.float32)  # [1,1,1]
    n("Reshape", ["sizef", "shp111"], "size111")                 # [1,1,1]
    init("shp111", np.array([1, 1, 1], np.int64), np.int64)
    n("Add", ["size111", "three_i"], "s3")                       # [1,1,1] = size+3
    # rows3 = s3*[0,1,1] ; cols3 = s3*[1,0,1]  -> points (0,s3),(s3,0),(s3,s3)
    init("rmask", np.array([0., 1., 1.], np.float32).reshape(1, 1, 3), np.float32)
    init("cmask", np.array([1., 0., 1.], np.float32).reshape(1, 1, 3), np.float32)
    n("Mul", ["s3", "rmask"], "rows3")                          # [1,1,3]
    n("Mul", ["s3", "cmask"], "cols3")                          # [1,1,3]
    n("Unsqueeze", ["rows3"], "rows3u", axes=[3])               # [1,1,3,1]
    n("Unsqueeze", ["cols3"], "cols3u", axes=[3])               # [1,1,3,1]
    n("Concat", ["rows3u", "cols3u"], "pts", axis=3)            # [1,1,3,2]
    n("Cast", ["pts"], "ptsi", to=I64)                          # [1,1,3,2] int64
    init("idxshape", np.array([1, 10, 3, 2], np.int64), np.int64)
    n("Expand", ["ptsi", "idxshape"], "idx")                    # [1,10,3,2] (480B)
    n("GatherND", ["input", "idx"], "cornsg", batch_dims=2)     # [1,10,3] f32
    ramp = np.arange(10, dtype=np.float32).reshape(1, 10, 1)
    init("ramp", ramp, np.float32)
    n("Mul", ["cornsg", "ramp"], "cornm")                      # [1,10,3]
    n("ReduceSum", ["cornm"], "corn3", axes=[1], keepdims=1)    # [1,1,3] = [TR,BL,BR]
    init("shp1111", np.array([1, 1, 1, 1], np.int64), np.int64)
    init("i0", np.array([0], np.int64), np.int64)
    init("i1", np.array([1], np.int64), np.int64)
    init("i2", np.array([2], np.int64), np.int64)
    init("i3", np.array([3], np.int64), np.int64)
    init("ax2c", np.array([2], np.int64), np.int64)
    n("Slice", ["corn3", "i0", "i1", "ax2c"], "TRs")           # [1,1,1]
    n("Slice", ["corn3", "i1", "i2", "ax2c"], "BLs")           # [1,1,1]
    n("Slice", ["corn3", "i2", "i3", "ax2c"], "BRs")           # [1,1,1]
    n("Reshape", ["TRs", "shp1111"], "TRf")
    n("Reshape", ["BLs", "shp1111"], "BLf")
    n("Reshape", ["BRs", "shp1111"], "BRf")
    n("Cast", ["TRf"], "TR", to=F16)
    n("Cast", ["BLf"], "BL", to=F16)
    n("Cast", ["BRf"], "BR", to=F16)

    # ---------- quadrant colour plane qcol[1,1,6,6] fp16 (separable, nested Where) --
    # rlo[r] = (2r < size) (top half), clo[c] = (2c < size) (left half)
    init("r2", (2 * np.arange(6)).reshape(1, 1, 6, 1).astype(np.float32), np.float32)
    init("c2", (2 * np.arange(6)).reshape(1, 1, 1, 6).astype(np.float32), np.float32)
    n("Less", ["r2", "sizef"], "rlo")        # [1,1,6,1] top half
    n("Less", ["c2", "sizef"], "clo")        # [1,1,1,6] left half
    # qcol = Where(rlo, Where(clo,TL,TR), Where(clo,BL,BR))  -> ONE 6x6 plane
    n("Where", ["clo", "TL", "TR"], "topcol")   # [1,1,1,6] fp16
    n("Where", ["clo", "BL", "BR"], "botcol")   # [1,1,1,6] fp16
    n("Where", ["rlo", "topcol", "botcol"], "qcol")  # [1,1,6,6] fp16

    # ---------- cyan presence 6x6 ----------
    # ONE Slice on channel 8 AND rows/cols 2:8 -> [1,1,6,6] (no 3600B plane).
    init("cy_start", np.array([8, 2, 2], np.int64), np.int64)
    init("cy_end", np.array([9, 8, 8], np.int64), np.int64)
    init("cy_axes", np.array([1, 2, 3], np.int64), np.int64)
    n("Slice", ["input", "cy_start", "cy_end", "cy_axes"], "cyan6")  # [1,1,6,6]
    init("zerof", np.array([[[[0.0]]]], np.float32), np.float32)
    n("Greater", ["cyan6", "zerof"], "cyanb")                    # [1,1,6,6] bool

    # ---------- in-region mask (r<size)&(c<size) on 6x6 ----------
    init("ridx", np.arange(6).reshape(1, 1, 6, 1).astype(np.float32), np.float32)
    init("cidx", np.arange(6).reshape(1, 1, 1, 6).astype(np.float32), np.float32)
    n("Less", ["ridx", "sizef"], "rin")      # [1,1,6,1]
    n("Less", ["cidx", "sizef"], "cin")      # [1,1,1,6]
    n("And", ["rin", "cin"], "inreg")        # [1,1,6,6]

    # ---------- lab6 = Where(inreg, Where(cyan, qcol, 0), 99) (fp16) ----------
    init("zerolab", np.array([[[[0.0]]]], np.float16), np.float16)
    n("Where", ["cyanb", "qcol", "zerolab"], "lab_in")          # [1,1,6,6] fp16
    init("sent99", np.array([[[[99.0]]]], np.float16), np.float16)
    n("Where", ["inreg", "lab_in", "sent99"], "lab6f")          # [1,1,6,6] fp16
    n("Cast", ["lab6f"], "lab6u", to=U8)                        # [1,1,6,6] u8

    # ---------- Equal on the 6x6 label -> [1,10,6,6] one-hot, then Pad INTO
    #            the free 30x30 output (no full-canvas carrier plane) ----------
    arangeu = np.arange(10, dtype=np.uint8).reshape(1, 10, 1, 1)
    init("arangeu", arangeu, np.uint8)
    n("Equal", ["lab6u", "arangeu"], "oneh6b")                  # [1,10,6,6] bool
    n("Cast", ["oneh6b"], "oneh6", to=U8)                       # [1,10,6,6] u8 (360B)
    pads = np.array([0, 0, 0, 0, 0, 0, 24, 24], np.int64)
    init("pads", pads, np.int64)
    init("zerou", np.array(0, np.uint8), np.uint8)
    n("Pad", ["oneh6", "pads", "zerou"], "output", mode="constant")  # [1,10,30,30] u8

    out_vi = helper.make_tensor_value_info("output", U8, [1, 10, 30, 30])
    in_vi = helper.make_tensor_value_info("input", F32, [1, 10, 30, 30])
    graph = helper.make_graph(nodes, "task183", [in_vi], [out_vi], inits)
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 12)])
    model.ir_version = IR_VERSION
    return model
