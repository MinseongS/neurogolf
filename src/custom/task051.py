"""Task 051 (25d487eb): extend a beam from a triangle's apex to the grid edge.

Rule (from ARC-GEN generator, verified 8000/0 fresh on the ray-fill model):
  The input holds a solid triangle of colour[0] with a single apex pixel of
  colour[1] at its wide-end centre.  After `apply_gravity` the whole figure is
  rotated/flipped into one of 4 axis orientations.  The transform fills a beam of
  colour[1] starting at the apex, going THROUGH the triangle (toward its tip) and
  on to the grid edge.  Only background (0) cells change; every nonzero input
  cell is copied unchanged.

  Closed-form characterisation (all verified fresh):
    * apex colour  = the unique nonzero colour whose pixel COUNT == 1.
    * apex position (ar,ac) = ArgMax of that channel's row / col profile.
    * triangle colour = the OTHER nonzero colour (count > 1, channel != 0).
    * beam direction (dr,dc) = sign(triangle_centroid - apex).  Because the
      triangle base is symmetric about the apex, the perpendicular component is
      EXACTLY 0, so (dr,dc) is a pure axis unit vector.
    * beam_region = cells with perpendicular offset q == 0 AND parallel offset
      s >= 1  (s = dr*(r-ar)+dc*(c-ac), q = dc*(r-ar)-dr*(c-ac)).
    * output = where(input>0, input, where(beam_region, apex_onehot, 0)).

Encoding: no per-cell colour-index plane is ever built; the 10-channel expansion
lands directly in the FREE bool Where output.  Scalars (apex row/col, direction
signs) come from per-axis profiles of the free input.  Two fp16 [1,1,30,30]
working planes (s and q) and a couple of bool planes are the only full-canvas
intermediates.
"""

import numpy as np
import onnx
from onnx import helper, numpy_helper, TensorProto

from ..harness import IR_VERSION

N = 30


def build(task):
    inits, nodes = [], []

    def init(name, arr, dtype):
        inits.append(numpy_helper.from_array(
            np.ascontiguousarray(arr, dtype=dtype), name))
        return name

    def n(op, ins, out, **attrs):
        nodes.append(helper.make_node(op, ins, [out], **attrs))
        return out

    # ---- constants ----
    init("notch016", np.array([0] + [1] * 9, np.float16).reshape(1, 10, 1, 1),
         np.float16)
    init("notch0f", np.array([0] + [1] * 9, np.float32).reshape(1, 10, 1, 1),
         np.float32)
    init("one_f", np.array(1.0, np.float32), np.float32)
    init("half_f", np.array(0.5, np.float32), np.float32)
    init("zero_f", np.array(0.0, np.float32), np.float32)
    init("rowramp", np.arange(N, dtype=np.float32).reshape(1, 1, N, 1),
         np.float32)
    init("colramp", np.arange(N, dtype=np.float32).reshape(1, 1, 1, N),
         np.float32)
    init("rowramp16", np.arange(N, dtype=np.float16).reshape(1, 1, N, 1),
         np.float16)
    init("colramp16", np.arange(N, dtype=np.float16).reshape(1, 1, 1, N),
         np.float16)
    init("zero16", np.array(0.0, np.float16), np.float16)

    # ---- per-channel pixel counts ----
    n("ReduceSum", ["input"], "cnt", axes=[2, 3], keepdims=1)    # [1,10,1,1]

    # apexsel: channels with count == 1 (the apex colour)
    n("Equal", ["cnt", "one_f"], "apexb")                       # bool [1,10,1,1]
    n("Cast", ["apexb"], "apexf", to=TensorProto.FLOAT)         # fp32 selector
    n("Cast", ["apexb"], "apex16", to=TensorProto.FLOAT16)      # fp16 selector

    # ---- per-axis profiles (fp16 working planes) ----
    # rprof[1,10,30,1] / cprof[1,10,1,30] are the only 10-channel planes; halve
    # them by casting to fp16 before any masking.
    n("ReduceSum", ["input"], "rprof32", axes=[3], keepdims=1)  # [1,10,30,1] fp32
    n("Cast", ["rprof32"], "rprof", to=TensorProto.FLOAT16)
    n("ReduceSum", ["input"], "cprof32", axes=[2], keepdims=1)  # [1,10,1,30] fp32
    n("Cast", ["cprof32"], "cprof", to=TensorProto.FLOAT16)

    # apex position: mask the apex channel, reduce over channels, ArgMax.
    n("Mul", ["rprof", "apex16"], "a_rprof")
    n("ReduceSum", ["a_rprof"], "a_rprof1", axes=[1], keepdims=1)  # [1,1,30,1]
    n("ArgMax", ["a_rprof1"], "ar_i", axis=2, keepdims=1)
    n("Cast", ["ar_i"], "ar", to=TensorProto.FLOAT)
    n("Mul", ["cprof", "apex16"], "a_cprof")
    n("ReduceSum", ["a_cprof"], "a_cprof1", axes=[1], keepdims=1)  # [1,1,1,30]
    n("ArgMax", ["a_cprof1"], "ac_i", axis=3, keepdims=1)
    n("Cast", ["ac_i"], "ac", to=TensorProto.FLOAT)

    # direction = sign(whole-figure centroid - apex); the foreground figure
    # (triangle + apex) is symmetric about the apex base axis, so the centroid
    # offset is purely along the beam direction (verified 8000/0 fresh).
    # foreground row/col mass = sum over channels 1..9 (exclude background ch0).
    n("Mul", ["rprof", "notch016"], "fg_rprof")
    n("ReduceSum", ["fg_rprof"], "fg_r", axes=[1], keepdims=1)  # [1,1,30,1] fp16
    n("Cast", ["fg_r"], "fg_r32", to=TensorProto.FLOAT)
    n("Mul", ["fg_r32", "rowramp"], "wr")
    n("ReduceSum", ["wr"], "swr", axes=[2], keepdims=1)        # Σ r*mass
    n("ReduceSum", ["fg_r32"], "sr", axes=[2], keepdims=1)     # Σ mass
    n("Mul", ["ar", "sr"], "ar_sr")
    n("Sub", ["swr", "ar_sr"], "dr_raw")                      # Σ (r-ar)*mass
    n("Mul", ["cprof", "notch016"], "fg_cprof")
    n("ReduceSum", ["fg_cprof"], "fg_c", axes=[1], keepdims=1)  # [1,1,1,30] fp16
    n("Cast", ["fg_c"], "fg_c32", to=TensorProto.FLOAT)
    n("Mul", ["fg_c32", "colramp"], "wc")
    n("ReduceSum", ["wc"], "swc", axes=[3], keepdims=1)
    n("ReduceSum", ["fg_c32"], "sc", axes=[3], keepdims=1)
    n("Mul", ["ac", "sc"], "ac_sc")
    n("Sub", ["swc", "ac_sc"], "dc_raw")

    n("Sign", ["dr_raw"], "drs")                              # {-1,0,1}
    n("Sign", ["dc_raw"], "dcs")

    # depth of the triangle: along its central axis the colour-0 triangle covers
    # s = 1 .. depth-1 (the apex overwrote one cell), so beam fill starts at
    # s >= depth.  triangle pixel count = depth^2 - 1  =>  depth = sqrt(count+1).
    # max per-channel count over channels 1..9 is the triangle (apex count == 1).
    n("Mul", ["cnt", "notch0f"], "fgcnt")                     # zero out ch0 count
    n("ReduceMax", ["fgcnt"], "tricnt", axes=[1], keepdims=1)  # [1,1,1,1]
    n("Add", ["tricnt", "one_f"], "tri1")
    n("Sqrt", ["tri1"], "depth")                              # scalar
    n("Cast", ["depth"], "depth16", to=TensorProto.FLOAT16)

    # in-grid extent as separable row/col masks (grid is anchored at (0,0)):
    # every in-grid row/col carries at least the background, so its total > 0.
    n("ReduceSum", ["rprof32"], "rowtot", axes=[1], keepdims=1)  # [1,1,30,1] fp32
    n("Greater", ["rowtot", "zero_f"], "rowany")               # [1,1,30,1] bool
    n("ReduceSum", ["cprof32"], "coltot", axes=[1], keepdims=1)  # [1,1,1,30] fp32
    n("Greater", ["coltot", "zero_f"], "colany")               # [1,1,1,30] bool

    # ---- beam region as a SEPARABLE row x col AND (no 30x30 fp32 plane) ----
    # Axis-aligned half-line factorises; fold in s>=depth (skip the triangle) and
    # the in-grid extent so the only full-canvas tensor is the final And output.
    #   vertical (dr!=0):  dr*rdev>=depth  x  col == ac
    #   horizontal:        row == ar       x  dc*cdev>=depth
    n("Cast", ["ar"], "ar16", to=TensorProto.FLOAT16)
    n("Cast", ["ac"], "ac16", to=TensorProto.FLOAT16)
    n("Cast", ["drs"], "dr16", to=TensorProto.FLOAT16)
    n("Cast", ["dcs"], "dc16", to=TensorProto.FLOAT16)
    n("Sub", ["rowramp16", "ar16"], "rdev")                    # [1,1,30,1]
    n("Sub", ["colramp16", "ac16"], "cdev")                    # [1,1,1,30]
    n("Mul", ["dr16", "rdev"], "along_r")                      # [1,1,30,1]
    n("Mul", ["dc16", "cdev"], "along_c")                      # [1,1,1,30]

    # vert scalar = (dr != 0); horiz = not vert.  (ORT has no bool Where, so the
    # branch selection is done with And/Or on the scalar masks.)
    n("Abs", ["drs"], "drabs")
    n("Greater", ["drabs", "half_f"], "vert")                  # bool [1,1,1,1]
    n("Not", ["vert"], "horiz")

    # row condition A = (vert?(dr*rdev>=depth):(r==ar)) AND rowany
    n("Not", [n("Less", ["along_r", "depth16"], "ar_ltd")], "ar_ged")
    n("Equal", ["rdev", "zero16"], "r_eq")                     # r == ar
    n("Or", [n("And", ["vert", "ar_ged"], "A1"),
             n("And", ["horiz", "r_eq"], "A2")], "A0")         # [1,1,30,1] bool
    n("And", ["A0", "rowany"], "A")
    # col condition B = (vert?(c==ac):(dc*cdev>=depth)) AND colany
    n("Equal", ["cdev", "zero16"], "c_eq")                     # c == ac
    n("Not", [n("Less", ["along_c", "depth16"], "ac_ltd")], "ac_ged")
    n("Or", [n("And", ["vert", "c_eq"], "B1"),
             n("And", ["horiz", "ac_ged"], "B2")], "B0")       # [1,1,1,30] bool
    n("And", ["B0", "colany"], "B")
    n("And", ["A", "B"], "beamcell")                          # [1,1,30,30] bool

    # ---- assemble straight into the FREE fp32 output ----
    # output = input one-hot, EXCEPT at beam cells where it becomes the apex
    # colour one-hot.  Passing `input` as the Where false-branch keeps the whole
    # 10-channel expansion in the FREE graph output -- no per-cell copy is built.
    n("Where", ["beamcell", "apexf", "input"], "output")       # fp32 [1,10,30,30]

    x = helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 10, N, N])
    y = helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 10, N, N])
    graph = helper.make_graph(nodes, "graph", [x], [y], inits)
    return helper.make_model(
        graph, ir_version=IR_VERSION,
        opset_imports=[helper.make_opsetid("", 11)])
