"""Task 263 (ARC-AGI a87f7484) — pick the odd-shaped stamp, emit it at the corner.

Rule (from the generator): a 3x3 grid is replicated into K (3..5) side-by-side
stamps, laid out HORIZONTALLY as a 3x(3K) grid (or VERTICALLY as a (3K)x3 grid
when xpose=1).  K-1 stamps share one "basic" Conway-sprite shape; exactly ONE
stamp ("weird", at index `weird`) has a DIFFERENT shape.  The generator
guarantees the weird shape has a DIFFERENT pixel COUNT than the basic shape, so
the weird stamp is the unique-count stamp (counts observed 3..7).  Each stamp is
painted in its own colour `colors[idx]`.  The OUTPUT is a 3x3 grid holding the
weird stamp's shape in colour `colors[weird]` (the weird stamp's own 3x3 block,
moved to the top-left corner).  The whole thing is transposed iff xpose=1.

Recovery (no flood-fill, no argmax op):
  * colf30 = colour-index plane = sum_k k*input_k  ([1,1,30,30] fp32 — the lone
    full-canvas tensor, 3600B).  Slice directly into the two candidate STRIPS
    (horizontal rows0-2 cols0-14 [3x15], vertical rows0-14 cols0-2 [15x3]) and
    cast to fp16; EVERYTHING downstream is fp16 / tiny.
  * normalise BOTH orientations to a common [5,9] (stamp, flat-3x3): the vertical
    strip reshapes to [5,9] for free (already stamp-grouped); the horizontal strip
    is column-interleaved so it needs reshape [3,5,3] + Transpose -> [5,3,3] -> [5,9].
  * orientation flag h = (#nonzero horizontal stamps >= 2); vertical layout leaves
    only stamp0 in the horizontal strip.  Where(h, sh9, sv9) -> s9 (single path).
  * per-stamp pixel count without an occupancy plane: each stamp is mono-colour so
    count = value_sum / colour (ReduceSum/ReduceMax over the flat axis, colour
    clamped >=1 so empty stamps give 0).
  * weird stamp i: matchcount[i] = #{j: cnt[j]==cnt[i]} == 1 AND cnt[i]>0 (basic
    stamps match K-1 others; weird matches only itself; empty gated out).
  * weird 3x3 block = sel[1,5] @ s9[5,9] -> [1,9] (contracts the stamp axis, no
    per-cell Mul plane).
  * pad the 3x3 colour-index block to 30x30 (off-grid sentinel 99), free BOOL
    output = Equal(L, arange[0..9]).

Memory floor (5758B, +0.22 over 16.11 -> MARGINAL): colf30 3600B (10->1 colour
plane) + L 900B (30x30 uint8 output carrier) are at documented floors; the rest is
the dual-orientation working set (the horizontal Transpose + two fp32 strip slices).
"""

import numpy as np
import onnx
from onnx import helper, numpy_helper, TensorProto

from ..harness import IR_VERSION

F32 = TensorProto.FLOAT
F16 = TensorProto.FLOAT16
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

    # ---- colour-index plane via a 1x1 conv on the FREE input (lone big tensor) ----
    kw = np.arange(10, dtype=np.float32).reshape(1, 10, 1, 1)
    init("kw", kw, np.float32)
    n("Conv", ["input", "kw"], "colf30", kernel_shape=[1, 1])  # [1,1,30,30] fp32

    # Slice colf30 directly into the two candidate STRIPS (never the 15x15
    # region): horizontal strip rows0-2 cols0-14 (3x15) and vertical strip
    # rows0-14 cols0-2 (15x3).  Each is 45 elems fp32 (180B) -> fp16 (90B).
    init("ax23", np.array([2, 3], np.int64), np.int64)
    # horizontal strip: rows 0-2, cols 0-14 in ONE multi-axis Slice (3x15 fp32 180B)
    init("hs", np.array([0, 0], np.int64), np.int64)
    init("he", np.array([3, 15], np.int64), np.int64)
    n("Slice", ["colf30", "hs", "he", "ax23"], "cf_h_f32")    # [1,1,3,15] fp32
    n("Cast", ["cf_h_f32"], "cf_h", to=F16)                   # [1,1,3,15] fp16 (90B)
    # vertical strip: rows 0-14, cols 0-2 in ONE multi-axis Slice (15x3 fp32 180B)
    init("vs", np.array([0, 0], np.int64), np.int64)
    init("ve", np.array([15, 3], np.int64), np.int64)
    n("Slice", ["colf30", "vs", "ve", "ax23"], "cf_v_f32")    # [1,1,15,3] fp32
    n("Cast", ["cf_v_f32"], "cf_v", to=F16)                   # [1,1,15,3] fp16 (90B)

    zero = init("zero", np.array(0.0, np.float16), np.float16)

    # ---- normalise BOTH orientations to a common [5,9] (stamp, flat 3x3) tensor ----
    # vertical strip [1,1,15,3] flattens as (3s+r)*3+c = 9s+3r+c -> reshape [5,9]
    # ALREADY groups by stamp (no transpose).  Horizontal strip [1,1,3,15] flat is
    # r*15+3s+c (interleaved) -> reshape [3,5,3] then transpose to (stamp,r,c).
    init("rsh_h", np.array([3, 5, 3], np.int64), np.int64)
    n("Reshape", ["cf_h", "rsh_h"], "cf_h5")                  # [3,5,3] (r,stamp,mc)
    n("Transpose", ["cf_h5"], "stamps_h", perm=[1, 0, 2])     # [5,3,3] (stamp,r,c)
    init("rsh59", np.array([5, 9], np.int64), np.int64)
    n("Reshape", ["stamps_h", "rsh59"], "sh9")               # [5,9]
    n("Reshape", ["cf_v", "rsh59"], "sv9")                   # [5,9] (stamp,flat)

    # ---- orientation flag: #nonzero horizontal stamps >= 2 -> horizontal ----
    # (vertical layout puts all content in cols 0-2 -> only 1 horizontal stamp).
    # colours are 1..9 (positive, bg=0) so a stamp's VALUE-sum>0 iff it has a pixel.
    n("ReduceSum", ["sh9"], "vsum_h", axes=[1], keepdims=0)  # [5]
    n("Greater", ["vsum_h", "zero"], "nzh")                  # [5] bool
    n("Cast", ["nzh"], "nzhf", to=F16)                       # [5]
    n("ReduceSum", ["nzhf"], "nnzh", axes=[0], keepdims=0)   # scalar
    init("onef5", np.array(1.0, np.float16), np.float16)
    n("Greater", ["nnzh", "onef5"], "hflag")                 # scalar bool

    # ---- pick the active orientation ONCE ([5,9]); single downstream path ----
    n("Where", ["hflag", "sh9", "sv9"], "s9")                # [5,9] fp16

    # per-stamp pixel count WITHOUT an occupancy plane: each stamp is mono-colour
    # so value-sum = colour*count; count = value-sum / colour.  colour = ReduceMax
    # (clamped >=1 so empty stamps give 0/1=0).
    n("ReduceSum", ["s9"], "vsum", axes=[1], keepdims=0)     # [5]
    n("ReduceMax", ["s9"], "colr", axes=[1], keepdims=0)     # [5] (0 if empty)
    init("onef", np.array(1.0, np.float16), np.float16)
    n("Max", ["colr", "onef"], "colr1")                      # [5] clamp >=1
    n("Div", ["vsum", "colr1"], "cnt")                       # [5] pixel count

    # ---- weird selector: matchcount==1 AND count>0 ----
    init("c5x1", np.array([5, 1], np.int64), np.int64)
    init("c1x5", np.array([1, 5], np.int64), np.int64)
    n("Reshape", ["cnt", "c5x1"], "cA")                      # [5,1]
    n("Reshape", ["cnt", "c1x5"], "cB")                      # [1,5]
    n("Equal", ["cA", "cB"], "eqm")                          # [5,5] bool
    n("Cast", ["eqm"], "eqf", to=F16)                        # [5,5]
    n("ReduceSum", ["eqf"], "mc", axes=[1], keepdims=0)      # [5]
    n("Greater", ["cnt", "zero"], "pos")                     # [5] bool
    n("Cast", ["pos"], "posf", to=F16)                       # [5]
    n("Equal", ["mc", "onef"], "isw")                        # [5] bool
    n("Cast", ["isw"], "iswf0", to=F16)                      # [5]
    n("Mul", ["iswf0", "posf"], "sel")                       # [5]

    # ---- weird block (3x3) = sel[1,5] @ s9[5,9] -> [1,9] (contracts stamp axis) ----
    init("sel1x5", np.array([1, 5], np.int64), np.int64)
    n("Reshape", ["sel", "sel1x5"], "sel15")                 # [1,5]
    n("MatMul", ["sel15", "s9"], "blk19")                    # [1,9] fp16

    # ---- pad to 30x30, build colour label, free BOOL output ----
    init("blk_shape", np.array([1, 1, 3, 3], np.int64), np.int64)
    n("Reshape", ["blk19", "blk_shape"], "blk4")              # [1,1,3,3]
    n("Cast", ["blk4"], "blku8", to=U8)                       # [1,1,3,3] uint8
    init("u99", np.array(99, np.uint8), np.uint8)
    init("pads", np.array([0, 0, 0, 0, 0, 0, 27, 27], np.int64), np.int64)
    n("Pad", ["blku8", "pads", "u99"], "L", mode="constant")  # [1,1,30,30] uint8

    chan = np.arange(10, dtype=np.uint8).reshape(1, 10, 1, 1)
    init("chan", chan, np.uint8)
    n("Equal", ["L", "chan"], "output")                       # [1,10,30,30] BOOL

    x = helper.make_tensor_value_info("input", F32, [1, 10, 30, 30])
    y = helper.make_tensor_value_info("output", B, [1, 10, 30, 30])
    g = helper.make_graph(nodes, "task263", [x], [y], inits)
    return helper.make_model(g, ir_version=IR_VERSION,
                             opset_imports=[helper.make_opsetid("", 11)])
