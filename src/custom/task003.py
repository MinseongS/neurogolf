"""task003 (ARC-AGI 017c7c7b) — extend a periodic blue stencil, recolour red.

Rule (from the ARC-GEN generator, verified fresh):
  A height=6, width=3 grid holds a vertically-PERIODIC blue (colour 1) stencil
  with period `steps` in {2,3}.  For steps==2 the column pattern may FLIP
  left-right once per period (`flip`).  The output is height=9, width=3: the
  SAME stencil recoloured RED (colour 2) and extended by the same periodic rule
  to 9 rows.  Off the 9x3 grid everything is background 0.

  Output rows 0..5 == input rows 0..5 (recoloured).  Rows 6,7,8 continue the
  same offset+flip schedule the generator uses; since the input already shows
  exactly the rows the continuation reuses, no flip handling is needed:
      steps==3 (offsets 0,3,6):  out6=in0, out7=in1, out8=in2
      steps==2 (offsets 0,2,4,6,8 with flip toggling): out6=in2, out7=in3, out8=in0
  This needs only ONE scalar boolean:
      is3 = pattern is period-3  <=>  shift-by-3 matches (in3==in0,in4==in1,in5==in2)
  (For steps==2 that can only hold if all rows are equal, where both extensions
   coincide -- harmless.)

  Detection uses tiny fp16 reductions on the 6x3 blue slice; the OUTPUT copy is
  built entirely in UINT8 (one-hot is {0,1}, harness scores out>0) so the
  dominant pre-Pad [1,10,9,3] block costs 270B (uint8) not 540B (fp16).
"""

import numpy as np
from onnx import helper, numpy_helper, TensorProto

from ..harness import IR_VERSION

S = 30
H = 6      # input active height
W = 3      # width
EH = 9     # output (extended) height


def build(task):
    inits, nodes = [], []
    seen = set()

    def init(name, arr, dt):
        if name in seen:
            return name
        seen.add(name)
        inits.append(numpy_helper.from_array(np.ascontiguousarray(arr, dtype=dt), name))
        return name

    def n(op, ins, out, **attrs):
        nodes.append(helper.make_node(op, ins, [out], **attrs))
        return out

    F = TensorProto.FLOAT
    F16 = TensorProto.FLOAT16
    U8 = TensorProto.UINT8
    B = TensorProto.BOOL

    # ---- blue channel, 6x3 active slice --------------------------------------
    init("s_start", np.array([1, 0, 0], np.int64), np.int64)   # ch1, row0, col0
    init("s_end", np.array([2, H, W], np.int64), np.int64)
    init("s_ax", np.array([1, 2, 3], np.int64), np.int64)
    n("Slice", ["input", "s_start", "s_end", "s_ax"], "blue")  # [1,1,6,3] fp32

    # per-row slices in fp16 for detection (tiny)
    n("Cast", ["blue"], "blue16", to=F16)                      # [1,1,6,3] fp16

    def rowf(r):
        init(f"r{r}s", np.array([r], np.int64), np.int64)
        init(f"r{r}e", np.array([r + 1], np.int64), np.int64)
        init("ax2", np.array([2], np.int64), np.int64)
        return n("Slice", ["blue16", f"r{r}s", f"r{r}e", "ax2"], f"row{r}f")

    r0 = rowf(0)
    r1 = rowf(1)
    r2 = rowf(2)
    r3 = rowf(3)
    r4 = rowf(4)
    r5 = rowf(5)

    # ---- detection -----------------------------------------------------------
    # period-3 (steps==3) <=> shift-by-3 matches exactly: r3==r0,r4==r1,r5==r2.
    # (For steps==2 this can only hold if all rows are equal, in which case the
    #  steps==2 and steps==3 extensions coincide -- harmless.)
    init("zh", np.array(0.0, np.float16), np.float16)

    def sqsum(a, b, tag):
        n("Sub", [a, b], f"d_{tag}")
        n("Mul", [f"d_{tag}", f"d_{tag}"], f"dsq_{tag}")
        return n("ReduceSum", [f"dsq_{tag}"], f"ss_{tag}", keepdims=0)

    sqsum(r3, r0, "30")
    sqsum(r4, r1, "41")
    sqsum(r5, r2, "52")
    n("Sum", ["ss_30", "ss_41", "ss_52"], "mm3")           # period-3 mismatch
    n("Equal", ["mm3", "zh"], "is3_b")                     # ==0 => period-3

    # ---- assemble output rows in UINT8 (cheap copy block) --------------------
    # uint8 row slices of the blue channel
    n("Cast", ["blue"], "blueU", to=U8)                    # [1,1,6,3] uint8

    def rowu(r):
        return n("Slice", ["blueU", f"r{r}s", f"r{r}e", "ax2"], f"row{r}u")

    u0, u1, u2, u3 = rowu(0), rowu(1), rowu(2), rowu(3)

    # steps3: out6,7,8 = u0,u1,u2 ;  steps2: out6,7,8 = u2,u3,u0   (Where on uint8)
    n("Where", ["is3_b", u0, u2], "out6")
    n("Where", ["is3_b", u1, u3], "out7")
    n("Where", ["is3_b", u2, u0], "out8")

    # red plane = rows 0..5 (blueU) concat out6,out7,out8 -> [1,1,9,3] uint8
    n("Concat", ["blueU", "out6", "out7", "out8"], "red91", axis=2)

    # background within grid = (red==0) ; zeros = (red==2, never happens)
    init("z_u8", np.array(0, np.uint8), np.uint8)
    init("two_u8", np.array(2, np.uint8), np.uint8)
    n("Equal", ["red91", "z_u8"], "bg91_b")
    n("Cast", ["bg91_b"], "bg91", to=U8)
    n("Equal", ["red91", "two_u8"], "z91_b")               # all-false -> zeros
    n("Cast", ["z91_b"], "z91", to=U8)

    # channel order: 0=bg,1=z,2=red,3..9=z
    n("Concat", ["bg91", "z91", "red91", "z91", "z91", "z91",
                 "z91", "z91", "z91", "z91"], "block", axis=1)  # [1,10,9,3] u8

    # pad to 30x30 with 0
    init("padspec", np.array([0, 0, 0, 0, 0, 0, S - EH, S - W], np.int64), np.int64)
    init("padval0", np.array(0, np.uint8), np.uint8)
    n("Pad", ["block", "padspec", "padval0"], "output")

    x = helper.make_tensor_value_info("input", F, [1, 10, 30, 30])
    y = helper.make_tensor_value_info("output", U8, [1, 10, 30, 30])
    graph = helper.make_graph(nodes, "task003", [x], [y], inits)
    return helper.make_model(graph, ir_version=IR_VERSION,
                             opset_imports=[helper.make_opsetid("", 11)])
