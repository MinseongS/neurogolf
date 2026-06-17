"""task345 (ARC-AGI d9f24cd1) — rising red rays deflected right by gray dots.

Rule (from the generator, verified fresh in numpy):
  size-10 grid. Gray pixels (5) placed at (rows[i], cols[i]). Red rays (2) each
  start at the BOTTOM row (r=9) at column `start` and rise. At cell (r,c):
    if the cell directly ABOVE, (r-1,c), is GRAY  -> step RIGHT  (c += 1)
    else                                          -> step UP     (r -= 1)
  every visited cell becomes red. Gray is never overwritten. The INPUT already
  carries the gray dots AND the red start pixels at row 9. Rays do NOT block
  each other (only GRAY deflects).

Closed-form reduction (verified 4000/4000 vs the sim):
  The generator spaces grays so far apart that EACH ray jogs right AT MOST ONCE
  (measured max-jogs/ray = 1 / 5000). So a ray is: a vertical run up its start
  column -> one step right at the gray that blocks it -> a vertical run up the
  next column. The SECOND vertical fill, seeded by {bottom starts} U {jog
  cells}, already reproduces the first run too, so only ONE vfill is needed.

  vfill = per-column segmented OR-scan  R[r,c]=seed OR (notgray[r,c] & R[r+1,c]),
  resolved by a Hillis-Steele DOUBLING scan in ceil(log2 10)=4 rounds:
      R <- max(R, g * shup_{2^k}(R));  g <- g * shup_{2^k}(g)
  (shup permutation MatMul; g=notgray.)

  The jog seed is found WITHOUT a first full fill, via a triangular MatMul:
      sclr[r,c] = NOT any gray in rows [r,8] of col c   (T @ gray, T lower-tri)
      jogsrc[r,c] = startcol[c] & gray[r-1,c] & sclr[r,c]   (= ray hits its
                    lowest reachable gray)
      jog = shift-right(jogsrc)                            (into col c+1)
      seed = start OR jog ;  R = vfill(seed)

  startcol[c] = ReduceMax of the start mask over rows ([1,1,1,10]). All planes
  live on the tiny 10x10 active corner in fp16 {0,1} (Max==OR, Mul==AND stay
  exactly 0/1). Output: L = 5 on gray, 2 on R, else 0; padded to 30x30
  (sentinel 10 off-grid) then Equal(L, arange) -> free BOOL output.

Dominant intermediate: fp16 10x10 planes (200B). No 30x30 colour plane.
"""

import numpy as np
import onnx
from onnx import helper, numpy_helper, TensorProto

from ..harness import IR_VERSION

U8 = TensorProto.UINT8
B = TensorProto.BOOL
F16 = TensorProto.FLOAT16
F32 = TensorProto.FLOAT

WORK = 10
ROUNDS = 4


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
    init("chan", np.arange(10, dtype=np.uint8).reshape(1, 10, 1, 1), np.uint8)
    init("half", np.array(0.5, np.float32), np.float32)
    init("halfh", np.array(0.5, np.float16), np.float16)
    init("oneh", np.array(1.0, np.float16), np.float16)
    init("u0", np.array(0, np.uint8), np.uint8)
    init("u2", np.array(2, np.uint8), np.uint8)
    init("u5", np.array(5, np.uint8), np.uint8)

    # channel slices on the active 10x10 corner
    init("red_st", np.array([2, 0, 0], np.int64), np.int64)
    init("red_en", np.array([3, WORK, WORK], np.int64), np.int64)
    init("gray_st", np.array([5, 0, 0], np.int64), np.int64)
    init("gray_en", np.array([6, WORK, WORK], np.int64), np.int64)
    init("cax", np.array([1, 2, 3], np.int64), np.int64)

    # up-shift permutation matrices Sup_k @ X -> X[r+2^k, c]  (Sup_k[r,r+2^k]=1)
    for k in range(ROUNDS):
        s = 1 << k
        M = np.zeros((WORK, WORK), np.float16)
        for r in range(WORK - s):
            M[r, r + s] = 1.0
        init(f"Sup{k}", M.reshape(1, 1, WORK, WORK), np.float16)
    # gray-down shift: Sdn @ X -> X[r-1,c] ; right shift: X @ Sright -> X[r,c-1]
    Sdn = np.zeros((WORK, WORK), np.float16)
    for r in range(1, WORK):
        Sdn[r, r - 1] = 1.0
    init("Sdn", Sdn.reshape(1, 1, WORK, WORK), np.float16)
    Sright = np.zeros((WORK, WORK), np.float16)
    for c in range(1, WORK):
        Sright[c - 1, c] = 1.0
    init("Sright", Sright.reshape(1, 1, WORK, WORK), np.float16)
    # lower-triangular T[r,r']=1 for r<=r'<=WORK-2 : sum gray over rows [r, 8]
    T = np.zeros((WORK, WORK), np.float16)
    for r in range(WORK):
        for rp in range(r, WORK - 1):
            T[r, rp] = 1.0
    init("T", T.reshape(1, 1, WORK, WORK), np.float16)

    # pad L (10x10 uint8) -> 30x30 with sentinel 10 (off-grid => no channel)
    init("padpads", np.array([0, 0, 0, 0, 0, 0, 30 - WORK, 30 - WORK], np.int64),
         np.int64)
    init("padval", np.array(10, np.uint8), np.uint8)

    # ---- gray / start / notgray masks (fp16 {0,1}, 10x10 corner) ----
    n("Slice", ["input", "gray_st", "gray_en", "cax"], "gray_f")
    n("Cast", ["gray_f"], "gray_h", to=F16)
    n("Slice", ["input", "red_st", "red_en", "cax"], "start_f")
    n("Cast", ["start_f"], "start_h", to=F16)
    n("Sub", ["oneh", "gray_h"], "notgray_h")

    # startcol[c] = does column c carry a bottom-row start  ([1,1,1,10])
    n("ReduceMax", ["start_h"], "startcol", axes=[2], keepdims=1)

    # sclr[r,c] = no gray in rows [r,8] of col c  (T @ gray == 0)
    n("MatMul", ["T", "gray_h"], "anyabove")
    n("Less", ["anyabove", "halfh"], "sclr_b")        # bool
    n("Cast", ["sclr_b"], "sclr", to=F16)

    # gAbove[r,c] = gray[r-1,c] ; jogsrc = startcol & gAbove & sclr
    n("MatMul", ["Sdn", "gray_h"], "gAbove")
    n("Mul", ["gAbove", "sclr"], "tmpj")
    n("Mul", ["tmpj", "startcol"], "jogsrc")
    n("MatMul", ["jogsrc", "Sright"], "jog")          # shift right into c+1
    n("Max", ["start_h", "jog"], "seed")

    # ---- single vertical up-fill: segmented OR-scan via doubling ----
    R = "seed"
    g = "notgray_h"
    for k in range(ROUNDS):
        su = n("MatMul", [f"Sup{k}", R], f"sR{k}")    # R[r+2^k,c]
        cand = n("Mul", [g, su], f"cand{k}")
        R = n("Max", [R, cand], f"R{k}")
        if k < ROUNDS - 1:
            gu = n("MatMul", [f"Sup{k}", g], f"sg{k}")
            g = n("Mul", [g, gu], f"g{k}")

    n("Greater", [R, "halfh"], "Rb")                  # bool red
    n("Greater", ["gray_f", "half"], "gray")          # bool gray

    # ---- L = 5 on gray, 2 on R, else 0 (uint8 10x10) ----
    n("Where", ["gray", "u5", "u0"], "Lg")
    n("Where", ["Rb", "u2", "Lg"], "L10")

    n("Pad", ["L10", "padpads", "padval"], "L", mode="constant")
    n("Equal", ["L", "chan"], "output")               # free BOOL output

    x = helper.make_tensor_value_info("input", F32, [1, 10, 30, 30])
    y = helper.make_tensor_value_info("output", B, [1, 10, 30, 30])
    graph = helper.make_graph(nodes, "task345", [x], [y], inits)
    return helper.make_model(
        graph, ir_version=IR_VERSION,
        opset_imports=[helper.make_opsetid("", 11)])
