"""task036 (ARC-AGI 1f85a75f) — crop the "celestial object" (the clustered colour).

Rule (from the generator, verified 0/20000 fresh on the colour-cluster idea):
  The grid holds ONE small connected blob drawn entirely in the special colour
  `colors[0]` (a 3..5 x 3..5 region, 0.75 density, connected).  All other pixels
  are scattered single-colour NOISE drawn in colours `colors[1..]`, sprinkled over
  the whole grid but kept OUT of the blob's bbox+1 border.  The output is the
  bounding box of the blob, cropped to the top-left corner of a fresh grid
  (everything outside that HxW box is background / all-channels-off).

  Key invariant: the special (blob) colour is the colour whose pixels are
  SPATIALLY CLUSTERED, i.e. the colour with the SMALLEST bounding-box span
  max(rowspan, colspan).  Noise colours span the whole grid; the blob colour
  spans <=4.  Picking the min-span colour identifies the blob exactly
  (verified 0/20000), and the noise never touches the blob, so within the blob's
  bbox the only non-zero colour is the blob colour (channel 0 fills the holes).

Pipeline (ONNX, opset 11):
  1. Per channel c=1..9 derive its bbox span from 1-D occupancy profiles
     (ReduceMax over each axis), forcing span=+BIG for absent channels and ch0.
  2. blobcolor = ArgMin span ; gather that channel's (min_row, min_col, H, W).
  3. Vbig = per-cell colour index (1x1 Conv sum_k k*input_k).  Shift it to the
     top-left by Gather(axis=2, arange+min_row) then Gather(axis=3, arange+min_col).
  4. Label map L[1,1,30,30] uint8: blobcolor where the shifted cell == blobcolor
     (inside the HxW box), 0 elsewhere inside the box, sentinel 10 outside.
  5. output = Equal(L, arange[0..9]) -> free BOOL one-hot output.
"""

import numpy as np
import onnx
from onnx import helper, numpy_helper, TensorProto

from ..harness import IR_VERSION

F32 = TensorProto.FLOAT
U8 = TensorProto.UINT8
I64 = TensorProto.INT64
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

    BIG = 1000.0

    # ---- per-channel occupancy profiles --------------------------------------
    # rowocc[1,10,30,1] = does channel c occupy row r ; colocc[1,10,1,30].
    n("ReduceMax", ["input"], "rowocc", axes=[3], keepdims=1)  # [1,10,30,1] f32
    n("ReduceMax", ["input"], "colocc", axes=[2], keepdims=1)  # [1,10,1,30] f32
    # per-channel any-pixel flag [1,10,1,1]
    n("ReduceMax", ["rowocc"], "anyc", axes=[2], keepdims=1)   # [1,10,1,1] f32

    init("half", np.array(0.5, np.float32), np.float32)
    n("Greater", ["rowocc", "half"], "rowb")                  # bool [1,10,30,1]
    n("Greater", ["colocc", "half"], "colb")                  # bool [1,10,1,30]

    # row / col index ramps (fp16: values <30 and +/-BIG(=1000) are fp16-exact)
    ramp_r = np.arange(30, dtype=np.float16).reshape(1, 1, 30, 1)
    ramp_c = np.arange(30, dtype=np.float16).reshape(1, 1, 1, 30)
    init("rr", ramp_r, np.float16)                            # [1,1,30,1] fp16
    init("rc", ramp_c, np.float16)                            # [1,1,1,30] fp16
    init("PBIG", np.array(BIG, np.float16), np.float16)
    init("NBIG", np.array(-BIG, np.float16), np.float16)

    # min/max occupied row & col per channel (broadcast ramp over channels, fp16)
    n("Where", ["rowb", "rr", "PBIG"], "rmin_src")            # [1,10,30,1] fp16
    n("ReduceMin", ["rmin_src"], "rmin", axes=[2], keepdims=1)  # [1,10,1,1]
    n("Where", ["rowb", "rr", "NBIG"], "rmax_src")
    n("ReduceMax", ["rmax_src"], "rmax", axes=[2], keepdims=1)  # [1,10,1,1]
    n("Where", ["colb", "rc", "PBIG"], "cmin_src")           # [1,10,1,30]
    n("ReduceMin", ["cmin_src"], "cmin", axes=[3], keepdims=1)  # [1,10,1,1]
    n("Where", ["colb", "rc", "NBIG"], "cmax_src")
    n("ReduceMax", ["cmax_src"], "cmax", axes=[3], keepdims=1)  # [1,10,1,1]

    # spans
    n("Sub", ["rmax", "rmin"], "rspan")                       # [1,10,1,1]
    n("Sub", ["cmax", "cmin"], "cspan")
    n("Max", ["rspan", "cspan"], "span")                     # [1,10,1,1]

    # mask out absent channels and channel 0 -> span := +BIG
    n("Greater", ["anyc", "half"], "present")                # [1,10,1,1] bool
    # channel-0 kill mask: constant [1,10,1,1] with True only at ch0
    ch0kill = np.zeros((1, 10, 1, 1), np.bool_)
    ch0kill[0, 0, 0, 0] = True
    init("ch0kill", ch0kill, np.bool_)
    n("Not", ["ch0kill"], "notch0")
    n("And", ["present", "notch0"], "valid")                 # present AND c!=0
    n("Where", ["valid", "span", "PBIG"], "span2")           # [1,10,1,1]

    # ---- blobcolor = argmin span over channel axis ---------------------------
    n("ArgMin", ["span2"], "bc_i", axis=1, keepdims=1)       # [1,1,1,1] int64
    n("Cast", ["bc_i"], "bcf", to=F32)                       # blobcolor as float

    # gather min_row / min_col of the blob channel.
    # reshape span tensors to [1,10] then Gather along axis=1 with bc index.
    n("Reshape", ["rmin", "shp10"], "rmin10")                # [1,10]
    n("Reshape", ["cmin", "shp10"], "cmin10")
    init("shp10", np.array([1, 10], np.int64), np.int64)
    n("Reshape", ["bc_i", "shp11"], "bc11")                  # [1,1]
    init("shp11", np.array([1, 1], np.int64), np.int64)
    n("GatherElements", ["rmin10", "bc11"], "minr16", axis=1)  # [1,1] fp16
    n("GatherElements", ["cmin10", "bc11"], "minc16", axis=1)
    n("Cast", ["minr16"], "minr", to=F32)                    # [1,1] f32
    n("Cast", ["minc16"], "minc", to=F32)

    # ---- blob plane = the blob colour channel, shift to top-left (5x5 window) -
    # The blob bbox is <=5x5 (width,height in 3..5), so we only need a WORK=5
    # window starting at (min_row, min_col).  Selecting the blob channel directly
    # (Gather over the channel axis) gives the blob's one-hot mask plane; no
    # colour Conv / full-plane uint8 cast needed.
    WORK = 5
    init("shp1d", np.array([1], np.int64), np.int64)         # reshape bc -> [1]
    n("Reshape", ["bc_i", "shp1d"], "bc1")                   # [1] int64
    n("Gather", ["input", "bc1"], "bplane", axis=1)          # [1,1,30,30] f32 mask

    # gather a WORK-row x WORK-col window starting at (min_row, min_col).
    # row indices = clip(arange(WORK) + min_row, 0, 29) as int64.
    baseW = np.arange(WORK, dtype=np.float32)
    init("baseW", baseW, np.float32)                         # [WORK]
    init("shp1", np.array([1], np.int64), np.int64)          # reshape -> [1]
    n("Reshape", ["minr", "shp1"], "minr_s")                 # [1]
    n("Add", ["baseW", "minr_s"], "ridx_f")                  # [WORK]
    init("c0", np.array(0.0, np.float32), np.float32)
    init("c29", np.array(29.0, np.float32), np.float32)
    n("Clip", ["ridx_f", "c0", "c29"], "ridx_cl")
    n("Cast", ["ridx_cl"], "ridx", to=I64)                   # [WORK] int64
    n("Reshape", ["minc", "shp1"], "minc_s")                 # [1]
    n("Add", ["baseW", "minc_s"], "cidx_f")
    n("Clip", ["cidx_f", "c0", "c29"], "cidx_cl")
    n("Cast", ["cidx_cl"], "cidx", to=I64)                   # [WORK] int64

    n("Gather", ["bplane", "ridx"], "Vr", axis=2)           # [1,1,WORK,30] f32
    n("Gather", ["Vr", "cidx"], "Vs", axis=3)               # [1,1,WORK,WORK] f32

    # ---- bbox mask (r < H) and (c < W) on the WORK x WORK canvas --------------
    n("Reshape", ["rspan", "shp10"], "rspan10")
    n("Reshape", ["cspan", "shp10"], "cspan10")
    n("GatherElements", ["rspan10", "bc11"], "Hm116", axis=1)  # H-1 fp16 [1,1]
    n("GatherElements", ["cspan10", "bc11"], "Wm116", axis=1)
    n("Cast", ["Hm116"], "Hm1", to=F32)
    n("Cast", ["Wm116"], "Wm1", to=F32)
    init("one", np.array(1.0, np.float32), np.float32)
    n("Add", ["Hm1", "one"], "H")                            # [1,1] f32
    n("Add", ["Wm1", "one"], "W")
    rampw_r = np.arange(WORK, dtype=np.float32).reshape(1, 1, WORK, 1)
    rampw_c = np.arange(WORK, dtype=np.float32).reshape(1, 1, 1, WORK)
    init("wr", rampw_r, np.float32)                          # [1,1,WORK,1]
    init("wc", rampw_c, np.float32)                          # [1,1,1,WORK]
    n("Less", ["wr", "H"], "rmask")                          # [1,1,WORK,1] bool
    n("Less", ["wc", "W"], "cmask")                          # [1,1,1,WORK] bool
    n("And", ["rmask", "cmask"], "boxmask")                  # [1,1,WORK,WORK] bool

    # ---- blob mask M = (Vs > 0.5) AND boxmask --------------------------------
    n("Cast", ["bcf"], "bc_u8", to=U8)                       # blobcolor uint8 [1,1,1,1]
    init("halfw", np.array(0.5, np.float32), np.float32)
    n("Greater", ["Vs", "halfw"], "iseq")                    # [1,1,WORK,WORK] bool
    n("And", ["iseq", "boxmask"], "M")                       # blob cells

    # ---- label map L (WORK x WORK): blobcolor on M, 0 on box&!M --------------
    init("u0", np.array(0, np.uint8), np.uint8)
    init("u10", np.array(10, np.uint8), np.uint8)
    n("Where", ["M", "bc_u8", "u0"], "Lin")                  # [1,1,WORK,WORK] u8
    n("Where", ["boxmask", "Lin", "u10"], "Lw")             # outside box -> 10

    # ---- pad WORK x WORK label map to 30x30 (sentinel 10), final Equal -------
    init("padpads",
         np.array([0, 0, 0, 0, 0, 0, 30 - WORK, 30 - WORK], np.int64), np.int64)
    n("Pad", ["Lw", "padpads", "u10"], "L", mode="constant")  # [1,1,30,30] u8
    init("chan", np.arange(10, dtype=np.uint8).reshape(1, 10, 1, 1), np.uint8)
    n("Equal", ["L", "chan"], "output")                      # [1,10,30,30] BOOL

    x = helper.make_tensor_value_info("input", F32, [1, 10, 30, 30])
    y = helper.make_tensor_value_info("output", B, [1, 10, 30, 30])
    graph = helper.make_graph(nodes, "task036", [x], [y], inits)
    return helper.make_model(graph, ir_version=IR_VERSION,
                             opset_imports=[helper.make_opsetid("", 11)])
