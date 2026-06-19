"""task251 (ARC-AGI a5313dff) — "fill the black gap inside each in-grid red box with blue".

Rule (from the generator task_a5313dff.py):
  Input: a black(0) canvas carrying 1px-thick RED(2) rectangle outlines (boxes), each
  of width/height in {4,5,6}. Each box may also carry an inner red core (a smaller red
  rect inset 2 cells). Between the outer red outline and the inner area the cells are
  black -- this is the 1-cell-wide "gap" ring just inside the outline.
  Output: every gap cell of a FULLY-IN-GRID box becomes BLUE(1); red stays red; boxes
  that are clipped by the grid edge (drawn with row/col = -1) keep their gap black.
  Boxes may abut/overlap, so a pure 4-direction ray-enclosure / flood test FALSE-POSITIVES
  on spurious cells between boxes -> the rule must MATCH the actual box outline template.

Approach (re-golf of the public kojimar template-match net to the true active canvas):
  Grids are always <=12x12 (size=randint(8,12)), so slice black(ch0) and red(ch2) to a
  WxW work region (W=12, vs the public 14), cast fp16, and form a SINGLE-channel detector
  input combo = red - black (red->+1, in-grid-black->-1, off-grid->0). One Conv with 9
  output channels (the 9 (wide,tall) in {4,5,6}^2 box-outline templates, 6x6 footprint;
  outline +1/(2*nred), gap -1/(2*ngap)) scores exactly 1.0 only at the top-left of a box
  whose RED outline is complete AND whose 1-cell gap ring is BLACK; the conv is padded
  bottom/right by 2 so every in-grid anchor (up to size-4) is reachable from the WxW slice.
  Greater(scores,0.99) lights true boxes (max partial 0.9375 << 0.99); a clipped box is
  missing part of its outline so never fires. A ConvTranspose with the matching 9 paint
  stamps (gap ring) deposits the blue frame, output-cropped via pads=[0,0,2,2] to WxW.
  Recombine [black-fill, fill(blue), red] and Pad DIRECTLY into the FREE output (off-canvas
  stays bg via ch0=0; no 30x30 intermediate).

  vs the public 14-base net: combo replaces the 2-ch concat (1 channel), every working plane
  is 12x12 not 14x14, and the ConvTranspose crop removes the fill slice -> 8741 -> 7101 mem,
  score 15.82 -> 16.04.

Dominant intermediate (the WALL): the detection bank scores[1,9,9,9] fp16 (1458) +
  detect_bool[1,9,9,9] bool (729) + detect[1,9,9,9] fp16 (1458) = 3645B. Irreducible:
  (1) all 9 (wide,tall) combos occur with equal probability and each needs its own template
  + paint stamp (no channel can be dropped/merged); (2) the 9x9 anchor grid is the exact set
  of valid in-grid top-lefts for a width-4 box (0..size-4=8) -> spatial size pinned;
  (3) ORT Conv/ConvTranspose have NO uint8/int8 kernel, so scores+detect are fp16 (the floor),
  and the hard threshold Greater MUST emit a bool intermediate before the fp16 cast that the
  ConvTranspose consumes. Bank floors at 3645; with the minimal cropped surround (~3456) the
  whole net floors near 7101 -> 16.04, +0.22 over 15.82 (MARGINAL, < +0.3).
"""

import numpy as np
from onnx import helper, numpy_helper, TensorProto

from ..harness import IR_VERSION

F32 = TensorProto.FLOAT
F16 = TensorProto.FLOAT16
BOOL = TensorProto.BOOL
I64 = TensorProto.INT64

W = 12   # active work region (grids are always <=12x12)
K = 6    # box-outline footprint (max wide/tall = 6)


def _templates():
    """9 detector templates + 9 paint stamps for (wide,tall) in {4,5,6}^2.

    detector_weights[t, {black,red}, K, K]:  linear correlation that == 1.0 only at the
      top-left of a red outline of exactly (wide,tall) whose 1-cell gap ring is black.
      red-on-outline contributes 1/(2*nred), black-on-gap contributes 1/(2*ngap), so a
      perfect match sums to 0.5+0.5 == 1.0; threshold 0.99 demands near-perfect on both.
    paint_weights[t, 1, K, K]: 1.0 on the gap-ring cells, so ConvTranspose stamps blue.
    """
    sizes = [4, 5, 6]
    combos = [(w, t) for w in sizes for t in sizes]  # 9
    # SINGLE-CHANNEL detector over combo = red - black: red->+1, black->-1, else 0.
    # outline cell (+1) weight +1/(2*nred); gap cell (-1) weight -1/(2*ngap) so a black
    # gap contributes +1/(2*ngap). A perfect match (full red outline AND black gap) sums
    # to 0.5+0.5 == 1.0; clipped boxes (missing outline) or non-black gaps fall short.
    det = np.zeros((9, 1, K, K), np.float32)
    paint = np.zeros((9, 1, K, K), np.float32)
    for i, (wide, tall) in enumerate(combos):
        red_cells, gap_cells = [], []
        for r in range(tall):
            for c in range(wide):
                if r == 0 or r == tall - 1 or c == 0 or c == wide - 1:
                    red_cells.append((r, c))
        for r in range(1, tall - 1):
            for c in range(1, wide - 1):
                if r == 1 or r == tall - 2 or c == 1 or c == wide - 2:
                    gap_cells.append((r, c))
        nred, ngap = len(red_cells), len(gap_cells)
        for (r, c) in red_cells:
            det[i, 0, r, c] += 1.0 / (2.0 * nred)
        for (r, c) in gap_cells:
            det[i, 0, r, c] += -1.0 / (2.0 * ngap)
        for (r, c) in gap_cells:
            paint[i, 0, r, c] = 1.0
    return det.astype(np.float16), paint.astype(np.float16)


def build(task):
    inits, nodes = [], []

    def init(name, arr, dtype):
        inits.append(numpy_helper.from_array(
            np.ascontiguousarray(arr, dtype=dtype), name))
        return name

    def n(op, ins, out, **attrs):
        nodes.append(helper.make_node(op, ins, [out], **attrs))
        return out

    det_w, paint_w = _templates()

    # ---- slice black (ch0) and red (ch2) to the WxW work region ------------
    # A 4x4 box can anchor at top-left col/row up to W-4 (=8); a no-pad 6x6 conv only
    # reaches positions 0..W-6, so pad the conv bottom/right by 2 to make every valid
    # top-left detectable (effective 14-window) while the materialized slices stay WxW.
    init("ax", np.array([1, 2, 3], np.int64), np.int64)
    init("b_s", np.array([0, 0, 0], np.int64), np.int64)
    init("b_e", np.array([1, W, W], np.int64), np.int64)
    n("Slice", ["input", "b_s", "b_e", "ax"], "black")        # [1,1,W,W] f32
    init("r_s", np.array([2, 0, 0], np.int64), np.int64)
    init("r_e", np.array([3, W, W], np.int64), np.int64)
    n("Slice", ["input", "r_s", "r_e", "ax"], "red")          # [1,1,W,W] f32

    n("Cast", ["black"], "black16", to=F16)                   # [1,1,W,W] f16
    n("Cast", ["red"], "red16", to=F16)                       # [1,1,W,W] f16
    # single-channel detector input: combo = red - black (red->+1, black->-1, else 0)
    n("Sub", ["red16", "black16"], "combo")                   # [1,1,W,W] f16

    # ---- detection bank: 9 box-outline templates -------------------------
    # pad bottom/right by 2 -> effective (W+2)=14 window -> scores [1,9,9,9]
    init("detector_weights", det_w, np.float16)
    n("Conv", ["combo", "detector_weights"], "scores",
      pads=[0, 0, 2, 2])                                      # [1,9,9,9] f16

    init("threshold", np.array([0.99], np.float16), np.float16)
    n("Greater", ["scores", "threshold"], "detect_bool")      # [1,9,9,9] bool
    n("Cast", ["detect_bool"], "detect", to=F16)              # f16

    # ---- paint blue interior frame via ConvTranspose -----------------------
    # transpose-conv of [1,9,9,9] with 6x6 kernel naturally yields 14x14; ConvTranspose
    # `pads` CROP the output, so pads=[0,0,2,2] emits the WxW canvas directly (the blue
    # only ever lands in the <=12 grid) -> no 14x14 / extra slice plane.
    init("paint_weights", paint_w, np.float16)
    n("ConvTranspose", ["detect", "paint_weights"], "fill",
      pads=[0, 0, 2, 2])                                      # [1,1,W,W] f16

    # ---- recombine channels: black' = black - fill, blue = fill, red = red --
    # Pad the 3-channel small canvas DIRECTLY into the FREE output -> no 30x30
    # intermediate ever materialises (ch3..9 zero-pad, off-grid stays bg via ch0=0).
    n("Sub", ["black16", "fill"], "out_black")                # [1,1,W,W] f16
    n("Concat", ["out_black", "fill", "red16"], "small3", axis=1)  # [1,3,W,W] f16

    init("pads", np.array([0, 0, 0, 0, 0, 7, 30 - W, 30 - W], np.int64), np.int64)
    n("Pad", ["small3", "pads"], "output", mode="constant")   # [1,10,30,30] f16

    x = helper.make_tensor_value_info("input", F32, [1, 10, 30, 30])
    y = helper.make_tensor_value_info("output", F16, [1, 10, 30, 30])
    g = helper.make_graph(nodes, "task251", [x], [y], inits)
    return helper.make_model(g, ir_version=IR_VERSION,
                             opset_imports=[helper.make_opsetid("", 11)])
