"""task251 (ARC-AGI a5313dff) — fill the enclosed band of in-grid red boxes with blue.

Rule (from the generator):
  A size x size (size in 8..12) black canvas holds 1..4 axis-aligned red boxes
  (each 4..6 wide/tall, possibly clipped one cell off any edge, non-overlapping).
  Each box is drawn as: a 1px RED outer outline, a 1px BLACK band just inside it,
  then a RED inner core.  In the OUTPUT, the 1px black band of every box that is
  FULLY inside the grid is recoloured BLUE; clipped boxes (any side off-grid) keep
  the black band black.  Everything else (red, exterior background) is unchanged.

  This is exactly the classic ENCLOSURE / hole-fill predicate:
    flood-fill from the grid border through all non-red cells; any BLACK cell that
    the flood does NOT reach is enclosed by red -> recolour it BLUE.
  A clipped box has an off-grid (hence open) side, so its band is reachable from the
  border and stays black.  Verified 0 mismatches over 3000+ fresh instances.

Encoding (bounded-iteration unrolled flood, the HARD-WALL master key):
  - The generator caps size at 12, so the whole task lives in the top-left WORK x WORK
    (WORK=13, one padding ring margin) corner.  The 30x30 padding is all background
    (channel 0) and connects to the border, so seeding the WORK-crop border reproduces
    "connected to the exterior" exactly.
  - passable = 1 - red(ch2), cropped to WORK x WORK, fp16.
  - reach0  = (border frame of the crop) AND passable.
  - per round:  reach = Min(passable, MaxPool3x3(reach, SAME))   -- 8-connected
                dilation; corners are red so 8-conn never leaks (verified == 4-conn).
                Min(passable in {0,1}, count>=0) == passable AND dilated, stays in {0,1}.
    D=11 rounds (empirical max needed 9 at WORK=13, margin 2).
  - blue = (input ch0 == 1) AND (reach == 0), padded back to 30x30.
  - output = Where(blue, blue_onehot[1,10,1,1], input)  -- one op into the FREE output.

Dominant intermediate: the 11 fp16 [1,1,13,13] reach/dil planes (~338B each) — irreducible:
the flood is inherently iterative (no closed-form), the canvas is already cropped to the
generator bound, MaxPool needs float, and 9 dilations can be required so D can't shrink.

Fresh generalization: ISOLATED arc-gen-fresh 200/200 (and 3000+ in dev).
"""

import numpy as np
from onnx import helper, numpy_helper, TensorProto

from ..harness import IR_VERSION

F16 = TensorProto.FLOAT16
I64 = TensorProto.INT64
BOOL = TensorProto.BOOL

WORK = 13
N_ROUNDS = 11


def build(task):
    inits, nodes = [], []

    def init(name, arr, dtype):
        inits.append(numpy_helper.from_array(
            np.ascontiguousarray(arr, dtype=dtype), name))
        return name

    def n(op, ins, out, **attrs):
        nodes.append(helper.make_node(op, ins, [out], **attrs))
        return out

    # ---- crop red (ch2) and background (ch0) to the WORK x WORK active corner ----
    init("axes", np.array([1, 2, 3], np.int64), np.int64)
    init("steps", np.array([1, 1, 1], np.int64), np.int64)
    init("red_s", np.array([2, 0, 0], np.int64), np.int64)
    init("red_e", np.array([3, WORK, WORK], np.int64), np.int64)
    init("bg_s", np.array([0, 0, 0], np.int64), np.int64)
    init("bg_e", np.array([1, WORK, WORK], np.int64), np.int64)

    n("Slice", ["input", "red_s", "red_e", "axes", "steps"], "red_f")   # [1,1,W,W] f32
    n("Slice", ["input", "bg_s", "bg_e", "axes", "steps"], "bg_f")      # [1,1,W,W] f32
    n("Cast", ["red_f"], "red", to=F16)
    n("Cast", ["bg_f"], "bg", to=F16)

    # passable = 1 - red  (everything that is not red)
    init("one", np.array(1.0, np.float16), np.float16)
    n("Sub", ["one", "red"], "passable")                                # [1,1,W,W] f16

    # ---- seed = border frame of the crop, intersected with passable -------------
    frame = np.zeros((1, 1, WORK, WORK), np.float16)
    frame[0, 0, 0, :] = 1.0
    frame[0, 0, -1, :] = 1.0
    frame[0, 0, :, 0] = 1.0
    frame[0, 0, :, -1] = 1.0
    init("frame", frame, np.float16)
    n("Min", ["passable", "frame"], "reach0")                           # [1,1,W,W] f16

    # ---- bounded BFS: reach = Min(passable, MaxPool3x3(reach)) -------------------
    cur = "reach0"
    for i in range(N_ROUNDS):
        dil = n("MaxPool", [cur], f"dil{i}", kernel_shape=[3, 3],
                pads=[1, 1, 1, 1], strides=[1, 1])
        cur = n("Min", ["passable", dil], f"reach{i + 1}")

    # ---- blue = background cell AND not reached ---------------------------------
    # not_reached = (reach == 0); blue = bg AND not_reached
    init("zero16", np.array(0.0, np.float16), np.float16)
    n("Equal", [cur, "zero16"], "not_reached")                          # [1,1,W,W] bool
    n("Cast", ["bg"], "bg_bool", to=BOOL)                              # bg in {0,1}
    n("And", ["bg_bool", "not_reached"], "blue_small")                  # [1,1,W,W] bool

    # ---- pad blue mask back to 30x30 -------------------------------------------
    pad_amt = 30 - WORK
    init("pads", np.array([0, 0, 0, 0, 0, 0, pad_amt, pad_amt], np.int64), np.int64)
    # Pad rejects bool -> pad a uint8 cast, then re-cast to bool for Where.
    n("Cast", ["blue_small"], "blue_u8", to=TensorProto.UINT8)
    init("zero_u8", np.array(0, np.uint8), np.uint8)
    n("Pad", ["blue_u8", "pads", "zero_u8"], "blue_pad_u8")             # [1,1,30,30] u8
    n("Cast", ["blue_pad_u8"], "blue_mask", to=BOOL)                    # [1,1,30,30] bool

    # ---- output = Where(blue, blue_onehot, input) ------------------------------
    blue_oh = np.zeros((1, 10, 1, 1), np.float32)
    blue_oh[0, 1, 0, 0] = 1.0   # blue == colour 1
    init("blue_oh", blue_oh, np.float32)
    n("Where", ["blue_mask", "blue_oh", "input"], "output")            # [1,10,30,30] f32

    x = helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 10, 30, 30])
    y = helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 10, 30, 30])
    graph = helper.make_graph(nodes, "task251", [x], [y], inits)
    return helper.make_model(graph, ir_version=IR_VERSION,
                             opset_imports=[helper.make_opsetid("", 11)])
