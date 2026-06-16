"""task354 (ARC-AGI ddf7fa4f) — recolour each gray box from its light.

Rule (from the generator, verified 5000/0 in numpy + fresh):
  size-10 grid. Row 0 holds `num_lights`(=3) coloured "light" pixels, one per
  light column (gray excluded as a colour). Below sit `num_lights` solid gray
  rectangular boxes. The generator guarantees each box's COLUMN span contains
  EXACTLY one light column ("only under the right light"); boxes never overlap
  in 2-D (but two vertically-separated boxes MAY share a column).
  OUTPUT: every light pixel passes through unchanged at row 0, and every gray
  box is recoloured to the colour of the light whose column lies inside that
  box's column span. All other in-grid cells stay background (0).

Key reduction (verified 5000/0): within ANY single row, each contiguous gray
run is exactly one box's full column extent and contains exactly one light
column. The light column is gray through the WHOLE box (the box is solid and
the light sits above it), so:

  seed[r,c] = L[c]          if (r,c) is gray AND column c holds a light
            = 0             otherwise
  (L[c] = colour at row 0 of column c, broadcast down every row)

is exactly the light colour planted on each box's light-column strip. A purely
HORIZONTAL run-fill then spreads that single seed across the box: iterate
`Where(gray, MaxPool(1x3, cur), 0)`. Gating by gray between every step blocks
leakage across the (>=1 cell) gaps between boxes, so radius-1 dilation is used.
Max box width is 5 and the seed column can sit at a box edge => up to 4 cells
of spread one way => K=4 iterations (verified: K=3 fails 143/4000, K=4 0/4000).

Encoding (tiny 10x10 canvas, single colour-index plane):
  colf = one 1x1 Conv [0..9] over the one-hot input (the only fp32 30x30
  intermediate), Slice to the active 10x10 corner, Cast->uint8.  gray=(colf==5).
  lightrow = colf row 0 ([1,1,1,10]); seed via Where(gray, lightrow, 0);
  K=4 horizontal MaxPool+Where fills.  L = Where(gray, filled, colf) so lights
  at row 0 and background pass through; Pad to 30x30 with sentinel 10 (off-grid
  => no channel), final Equal(L, arange[0..9]) -> free BOOL output.

Dominant intermediate: the [1,1,30,30] fp32 Conv output (3600B, irreducible —
the one-hot->index collapse must touch the full 30x30 input once).  Everything
else is uint8/bool at 10x10 (<=100B).
"""

import numpy as np
import onnx
from onnx import helper, numpy_helper, TensorProto

from ..harness import IR_VERSION

U8 = TensorProto.UINT8
B = TensorProto.BOOL
F32 = TensorProto.FLOAT

WORK = 10  # active grid side (size = 10)
KITER = 4  # horizontal dilation iterations (radius 1 each; max box width 5)


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
    # 1x1 channel->index kernel restricted to the row-0 light slice (f16 out via
    # a tiny weighted sum); colours <=9 are fp16-exact.
    init("kw", np.arange(10, dtype=np.float32).reshape(1, 10, 1, 1), np.float32)
    init("chan", np.arange(10, dtype=np.uint8).reshape(1, 10, 1, 1), np.uint8)
    init("f0", np.array(0.0, np.float16), np.float16)
    init("u0", np.array(0, np.uint8), np.uint8)

    # gray = input channel 5, cropped to the active 10x10 corner
    init("gray_st", np.array([5, 0, 0], np.int64), np.int64)
    init("gray_en", np.array([6, WORK, WORK], np.int64), np.int64)
    init("gray_ax", np.array([1, 2, 3], np.int64), np.int64)
    init("half", np.array(0.5, np.float32), np.float32)
    # row-0 light slice of the input one-hot: [1,10,1,10]
    init("row0_st", np.array([0, 0], np.int64), np.int64)
    init("row0_en", np.array([1, WORK], np.int64), np.int64)
    init("row0_ax", np.array([2, 3], np.int64), np.int64)
    # row index ramp on the 10x10 canvas (to mask lightrow to row 0 only)
    init("rowramp", np.arange(WORK, dtype=np.float16).reshape(1, 1, WORK, 1),
         np.float16)
    # pad L (10x10 uint8) -> 30x30 with sentinel 10
    init("padpads", np.array([0, 0, 0, 0, 0, 0, 30 - WORK, 30 - WORK], np.int64),
         np.int64)
    init("padval", np.array(10, np.uint8), np.uint8)

    # ---- gray mask on 10x10 (bool) — no full-30x30 colour plane needed ----
    n("Slice", ["input", "gray_st", "gray_en", "gray_ax"], "gray_f")  # [1,1,10,10]
    n("Greater", ["gray_f", "half"], "gray")            # [1,1,10,10] bool

    # ---- lightrow = colour index at row 0, cols 0..9 ([1,1,1,10] f16) ----
    # Slice the one-hot to row 0 ([1,10,1,10]) then contract channels with a 1x1
    # Conv weighted by [0..9]; light colours land at their columns, 0 elsewhere.
    n("Slice", ["input", "row0_st", "row0_en", "row0_ax"], "row0")  # [1,10,1,10]
    n("Conv", ["row0", "kw"], "lightrow_f")             # [1,1,1,10] f32
    n("Cast", ["lightrow_f"], "lightrow", to=TensorProto.FLOAT16)  # [1,1,1,10] f16

    # seed = L[c] on every gray cell of a light column; non-light gray -> 0.
    # lightrow ([1,1,1,10]) broadcasts down all rows; Where keeps it only on gray.
    n("Where", ["gray", "lightrow", "f0"], "cur0")      # [1,1,10,10] f16

    # ---- horizontal run-fill: KITER iterations of (1x3 MaxPool then re-gate) ---
    # Re-gating by gray between every dilation step blocks leakage across the
    # (>=1 cell) gaps between boxes. The LAST step's re-gate is folded into the
    # label Where below (which masks to gray anyway), so it is skipped here.
    cur = "cur0"
    for i in range(KITER):
        p = n("MaxPool", [cur], f"pool{i}",
              kernel_shape=[1, 3], strides=[1, 1], pads=[0, 1, 0, 1])
        if i < KITER - 1:
            cur = n("Where", ["gray", p, "f0"], f"cur{i+1}")  # [1,1,10,10] f16
        else:
            cur = p
    n("Cast", [cur], "filled_u8", to=U8)                # [1,1,10,10] uint8

    # ---- lightplane = lights at row 0, background (0) everywhere else --------
    # Non-gray in-grid cells are either a row-0 light or background. Mask the
    # row-0 colour vector to row 0 only (gray never appears at row 0).
    n("Equal", ["rowramp", "f0"], "is_row0")            # [1,1,10,1] bool (row==0)
    n("Cast", ["lightrow"], "lightrow_u8", to=U8)       # [1,1,1,10] uint8
    n("Where", ["is_row0", "lightrow_u8", "u0"], "lightplane")  # [1,1,10,10] u8

    # ---- L = filled colour on gray, else lightplane (lights/background) ----
    n("Where", ["gray", "filled_u8", "lightplane"], "L10")  # [1,1,10,10] uint8

    # ---- pad to 30x30 (sentinel 10 outside) + final Equal -> BOOL output ----
    n("Pad", ["L10", "padpads", "padval"], "L", mode="constant")  # [1,1,30,30]
    n("Equal", ["L", "chan"], "output")                 # -> free BOOL output

    x = helper.make_tensor_value_info("input", F32, [1, 10, 30, 30])
    y = helper.make_tensor_value_info("output", B, [1, 10, 30, 30])
    graph = helper.make_graph(nodes, "task354", [x], [y], inits)
    return helper.make_model(
        graph, ir_version=IR_VERSION,
        opset_imports=[helper.make_opsetid("", 11)])
