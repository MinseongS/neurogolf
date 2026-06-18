"""task182 (ARC-AGI 776ffc46) — recolour every sprite matching the boxed sprite.

Rule (from the generator):
  A 20x20 (canvas 30x30) grid holds 5-6 small sprites, each one of 10 hardcoded
  shapes, placed non-overlapping with a >=1-cell gap.  Sprite #0 is drawn in a
  "special" colour (2 or 3) and is surrounded by a 7x7 gray(5) box outline; a
  duplicate of sprite #0's SHAPE is guaranteed (idxs[1]==idxs[0]).  Every other
  sprite is colour 1.  OUTPUT: recolour EVERY colour-1 sprite whose shape exactly
  matches the boxed sprite's shape to the special colour; leave all other sprites
  (and the box) untouched.

Encoding (runtime-kernel cross-correlation, exact shape match — NOT a flood-fill):
  * colf = Sum_k k*input_k (1x1 Conv) -> fp32 colour-index plane, sliced to the
    20x20 active grid.  occ1 = (colf==1); ref = special-colour occupancy (colf in
    {2,3}).
  * The reference SHAPE is extracted into a fixed 7x7 runtime kernel PF, the
    pattern placed at offset (1,1) (so a 1-cell border surrounds it inside the
    7x7 frame).  Top-left (r0,c0) of the reference bbox comes from per-row/col
    presence + a ramp ReduceMin; PF = a 7x7 Gather window of ref at (r0-1,c0-1).
    S = ReduceSum(PF) = the reference pixel count.
  * A colour-1 sprite EXACTLY matches the reference iff at its pattern-top-left
    anchor (i,j): corr_inner = Conv(occ1, PF) == S  (all pattern pixels present)
    AND corr_ring = Conv(occ1, ring) == 0, where ring = dilate(PF) - PF is the
    immediate 8-neighbourhood of the pattern (no extra pixel touches it, which
    rejects superset shapes — the proven discriminator since same-count shapes
    are never subsets of each other and connected supersets always touch).
  * anchor = (corr_inner==S) & (corr_ring==0).  Paint: recolor = Conv(anchor,
    PF_flipped) > 0 spreads the pattern back over each matched sprite's cells.
  * Label L = where(recolor, special, colf); Pad to 30x30 (sentinel 99 off-grid)
    and one final Equal(L, arange) into the FREE bool output.
"""

import numpy as np
import onnx
from onnx import helper, numpy_helper, TensorProto

from ..harness import IR_VERSION

WORK = 20  # active grid (generator size = 20)
F = 7      # runtime kernel frame side (max shape bbox 5 + 1-cell border each side)


def build(task):
    inits, nodes = [], []

    def init(name, arr, dtype):
        inits.append(numpy_helper.from_array(
            np.ascontiguousarray(arr, dtype=dtype), name))
        return name

    def n(op, ins, out, **attrs):
        nodes.append(helper.make_node(op, ins, [out], **attrs))
        return out

    # fp16 everywhere downstream of the one fp32 colour-index entry (task377 lever:
    # counts here are tiny — S<=12, corr<=24 — so fp16 is integer-exact; ORT
    # fp16 Min/Max work under ORT_DISABLE_ALL; static scorer counts fp16 at half).
    H16 = TensorProto.FLOAT16

    # ---------- colour-index plane (1x1 Conv: Sum_k k*input_k) ----------
    ksel = np.arange(10, dtype=np.float32).reshape(1, 10, 1, 1)
    init("ksel", ksel, np.float32)
    n("Conv", ["input", "ksel"], "colf30")                 # [1,1,30,30] f32 entry

    # slice to active 20x20 then cast to fp16
    init("st20", np.array([0, 0, 0, 0], np.int64), np.int64)
    init("en20", np.array([1, 1, WORK, WORK], np.int64), np.int64)
    init("ax4", np.array([0, 1, 2, 3], np.int64), np.int64)
    n("Slice", ["colf30", "st20", "en20", "ax4"], "colf32")  # [1,1,20,20] f32
    n("Cast", ["colf32"], "colf", to=H16)                  # [1,1,20,20] fp16

    init("one", np.array(1.0, np.float16), np.float16)
    init("half", np.array(0.5, np.float16), np.float16)
    init("c1p5", np.array(1.5, np.float16), np.float16)
    init("c3p5", np.array(3.5, np.float16), np.float16)

    n("Equal", ["colf", "one"], "occ1_b")                  # bool colour-1 cells
    n("Cast", ["occ1_b"], "occ1", to=H16)                  # [1,1,20,20] fp16
    # special-colour occupancy = (1.5 < colf < 3.5)  -> colf in {2,3}
    n("Greater", ["colf", "c1p5"], "gt15")
    n("Less", ["colf", "c3p5"], "lt35")
    n("And", ["gt15", "lt35"], "spec_b")
    n("Cast", ["spec_b"], "spec_occ", to=H16)              # [1,1,20,20] fp16

    # ---------- locate the REAL reference box (complete on-grid 7x7 gray) ----------
    # gray plane; Conv with a 7x7 perimeter kernel.  A complete box outline gives a
    # response == perimeter count (24) at its top-left (valid-conv index).  Curated
    # "fakeout" boxes are always partially off-grid, so the complete box is unique.
    init("c5", np.array(5.0, np.float16), np.float16)
    n("Equal", ["colf", "c5"], "gray_b")
    n("Cast", ["gray_b"], "gray", to=H16)                  # [1,1,20,20] fp16
    peri = np.zeros((1, 1, 7, 7), np.float16)
    peri[0, 0, 0, :] = 1; peri[0, 0, 6, :] = 1
    peri[0, 0, :, 0] = 1; peri[0, 0, :, 6] = 1
    init("peri", peri, np.float16)                         # 24 ones
    n("Conv", ["gray", "peri"], "box_resp")                # valid -> [1,1,14,14] fp16
    init("c23p5", np.array(23.5, np.float16), np.float16)
    n("Greater", ["box_resp", "c23p5"], "isbox_b")         # ==24 (unique)
    n("Cast", ["isbox_b"], "isbox", to=H16)                # [1,1,14,14] fp16
    # box top-left (bi,bj) from the unique peak (valid-conv index == box top-left)
    NB = WORK - 6  # 14
    brow = np.arange(NB, dtype=np.float16).reshape(1, 1, NB, 1)
    bcol = np.arange(NB, dtype=np.float16).reshape(1, 1, 1, NB)
    init("brow", brow, np.float16)
    init("bcol", bcol, np.float16)
    n("Mul", ["isbox", "brow"], "bi_m")
    n("Mul", ["isbox", "bcol"], "bj_m")
    n("ReduceMax", ["bi_m"], "bi", axes=[1, 2, 3], keepdims=1)  # [1,1,1,1] fp16 scalar
    n("ReduceMax", ["bj_m"], "bj", axes=[1, 2, 3], keepdims=1)  # [1,1,1,1]

    # box-interior mask: rows in (bi, bi+6), cols in (bj, bj+6)
    rr30 = np.arange(WORK, dtype=np.float16).reshape(1, 1, WORK, 1)
    cc30 = np.arange(WORK, dtype=np.float16).reshape(1, 1, 1, WORK)
    init("rr30", rr30, np.float16)
    init("cc30", cc30, np.float16)
    init("c6", np.array(6.0, np.float16), np.float16)
    n("Add", ["bi", "c6"], "bi_hi")
    n("Add", ["bj", "c6"], "bj_hi")
    n("Greater", ["rr30", "bi"], "rgt")
    n("Less", ["rr30", "bi_hi"], "rlt")
    n("Greater", ["cc30", "bj"], "cgt")
    n("Less", ["cc30", "bj_hi"], "clt")
    n("And", ["rgt", "rlt"], "rin")                        # [1,1,20,1]
    n("And", ["cgt", "clt"], "cin")                        # [1,1,1,20]
    n("And", ["rin", "cin"], "inbox_b")                    # [1,1,20,20]
    n("Cast", ["inbox_b"], "inbox", to=H16)
    # ref = special occupancy gated to the box interior (drops decoy special sprites)
    n("Mul", ["spec_occ", "inbox"], "ref")                 # [1,1,20,20] fp16

    # ---------- reference bbox top-left (r0, c0) ----------
    n("ReduceMax", ["ref"], "rowhas", axes=[1, 3], keepdims=1)  # [1,1,20,1]
    n("ReduceMax", ["ref"], "colhas", axes=[1, 2], keepdims=1)  # [1,1,1,20]
    rowramp = np.arange(WORK, dtype=np.float16).reshape(1, 1, WORK, 1)
    colramp = np.arange(WORK, dtype=np.float16).reshape(1, 1, 1, WORK)
    init("rowramp", rowramp, np.float16)
    init("colramp", colramp, np.float16)
    init("big", np.array(999.0, np.float16), np.float16)
    n("Greater", ["rowhas", "half"], "rowhas_b")
    n("Greater", ["colhas", "half"], "colhas_b")
    n("Where", ["rowhas_b", "rowramp", "big"], "rmasked")  # [1,1,20,1]
    n("Where", ["colhas_b", "colramp", "big"], "cmasked")  # [1,1,1,20]
    n("ReduceMin", ["rmasked"], "r0", axes=[2], keepdims=0)  # [1,1,1]
    n("ReduceMin", ["cmasked"], "c0", axes=[3], keepdims=0)  # [1,1,1]
    # window start = (r0-1, c0-1), clipped into [0, WORK-F]
    init("onef", np.array(1.0, np.float16), np.float16)
    init("zerof", np.array(0.0, np.float16), np.float16)
    n("Sub", ["r0", "onef"], "rs_raw")
    n("Sub", ["c0", "onef"], "cs_raw")
    n("Max", ["rs_raw", "zerof"], "rs_f")                  # clip >=0
    n("Max", ["cs_raw", "zerof"], "cs_f")
    init("maxstart", np.array(float(WORK - F), np.float16), np.float16)
    n("Min", ["rs_f", "maxstart"], "rs_c")
    n("Min", ["cs_f", "maxstart"], "cs_c")
    n("Cast", ["rs_c"], "rs_i", to=TensorProto.INT64)      # scalar
    n("Cast", ["cs_c"], "cs_i", to=TensorProto.INT64)
    aF = np.arange(F, dtype=np.int64)
    init("aF", aF, np.int64)
    init("to1", np.array([1], np.int64), np.int64)
    n("Reshape", ["rs_i", "to1"], "rs_1")
    n("Reshape", ["cs_i", "to1"], "cs_1")
    n("Add", ["rs_1", "aF"], "ridx")                       # [F]
    n("Add", ["cs_1", "aF"], "cidx")                       # [F]

    # ---------- extract PF = ref[ridx, cidx] 7x7 runtime kernel ----------
    n("Gather", ["ref", "ridx"], "ref_r", axis=2)          # [1,1,F,20] fp16
    n("Gather", ["ref_r", "cidx"], "PF", axis=3)           # [1,1,F,F] fp16 (0/1)
    n("ReduceSum", ["PF"], "S", axes=[1, 2, 3], keepdims=1)  # [1,1,1] = pixel count

    # ring = dilate(PF) - PF  (immediate 8-neighbourhood)
    init("ones3", np.ones((1, 1, 3, 3), np.float16), np.float16)
    n("Conv", ["PF", "ones3"], "dil_resp", pads=[1, 1, 1, 1])   # [1,1,F,F] fp16
    n("Greater", ["dil_resp", "half"], "dil_b")
    n("Cast", ["dil_b"], "dil", to=H16)
    n("Sub", ["dil", "PF"], "ring")                        # [1,1,F,F] fp16 (0/1)

    # ---------- correlation over occ1 (runtime PF/ring kernels) ----------
    n("Conv", ["occ1", "PF"], "corr_inner", pads=[F - 1, F - 1, F - 1, F - 1])
    n("Conv", ["occ1", "ring"], "corr_ring", pads=[F - 1, F - 1, F - 1, F - 1])
    # peak lands at pattern-top-left + (F-2); crop the WORK x WORK anchor window
    init("cst", np.array([0, 0, F - 2, F - 2], np.int64), np.int64)
    init("cen", np.array([1, 1, F - 2 + WORK, F - 2 + WORK], np.int64), np.int64)
    n("Slice", ["corr_inner", "cst", "cen", "ax4"], "ci")  # [1,1,20,20] fp16
    n("Slice", ["corr_ring", "cst", "cen", "ax4"], "cr")   # [1,1,20,20] fp16

    n("Equal", ["ci", "S"], "all_present")                 # corr_inner == S
    n("Less", ["cr", "half"], "no_extra")                  # corr_ring == 0
    n("And", ["all_present", "no_extra"], "anchor_b")
    n("Cast", ["anchor_b"], "anchor", to=H16)              # [1,1,20,20] fp16

    # ---------- paint: spread PF over each matched sprite (Conv with flipped PF) ----
    init("rev_st", np.array([F - 1, F - 1], np.int64), np.int64)
    init("rev_en", np.array([-F - 1, -F - 1], np.int64), np.int64)
    init("rev_ax", np.array([2, 3], np.int64), np.int64)
    init("rev_step", np.array([-1, -1], np.int64), np.int64)
    n("Slice", ["PF", "rev_st", "rev_en", "rev_ax", "rev_step"], "PFflip")
    n("Conv", ["anchor", "PFflip"], "paint_resp",
      pads=[F - 1, F - 1, F - 1, F - 1])
    init("pst", np.array([0, 0, 1, 1], np.int64), np.int64)
    init("pen", np.array([1, 1, 1 + WORK, 1 + WORK], np.int64), np.int64)
    n("Slice", ["paint_resp", "pst", "pen", "ax4"], "paint")  # [1,1,20,20]
    n("Greater", ["paint", "half"], "recolor_b")           # bool

    # ---------- special colour scalar (from the boxed reference) ----------
    n("Mul", ["ref", "colf"], "ref_col")
    n("ReduceMax", ["ref_col"], "spec_f", axes=[1, 2, 3], keepdims=1)  # [1,1,1,1] fp16

    # ---------- label map ----------
    n("Where", ["recolor_b", "spec_f", "colf"], "L20")     # [1,1,20,20] fp16
    init("padpads",
         np.array([0, 0, 0, 0, 0, 0, 30 - WORK, 30 - WORK], np.int64), np.int64)
    init("padval", np.array(99.0, np.float16), np.float16)
    n("Pad", ["L20", "padpads", "padval"], "L30", mode="constant")  # [1,1,30,30] fp16

    chan = np.arange(10, dtype=np.float16).reshape(1, 10, 1, 1)
    init("chan", chan, np.float16)
    n("Equal", ["L30", "chan"], "output")                  # -> FREE bool output

    x = helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 10, 30, 30])
    y = helper.make_tensor_value_info("output", TensorProto.BOOL, [1, 10, 30, 30])
    graph = helper.make_graph(nodes, "graph", [x], [y], inits)
    return helper.make_model(
        graph, ir_version=IR_VERSION,
        opset_imports=[helper.make_opsetid("", 11)])
