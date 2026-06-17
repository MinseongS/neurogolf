"""task212 (ARC-AGI 8d510a79) — vertical color rays bounded by a gray horizon.

Rule (generator task_8d510a79.py, size=10 fixed, grid placed top-left on 30x30):
  A gray horizon row (color 5) spans the whole grid at row `horizon` (3..6).
  Source pixels are blue (color 1, idx=0) or red (color 2, idx=1) at various
  (r,c).  Per column there is AT MOST one source above the horizon and one below
  (each chosen once per column in each half), so rays never collide.

  Each source emits a vertical ray.  Direction (generator):
    dr = -1 if blue else +1;  dr = dr if r<horizon else -dr.
  =>  BLUE always travels AWAY from the horizon (to the near grid edge);
      RED  always travels TOWARD the horizon (stopping one cell before it, since
      the horizon row is painted first and rays stop on an already-painted cell).
  The ray paints color idx+1 from the source up to (and including) the source,
  ending at a grid edge (blue) or just before the horizon (red).

  Per-column / per-color closed form (size=10 active):
    above region = rows < horizon ; below region = rows > horizon.
    above-blue  filled[i] iff a blue src at j>=i exists in above  (suffix-OR up)
    above-red   filled[i] iff a red  src at j<=i exists in above  (prefix-OR dn)
    below-blue  filled[i] iff a blue src at j<=i exists in below  (prefix-OR dn)
    below-red   filled[i] iff a red  src at j>=i exists in below  (suffix-OR up)
  Each "OR along a column" = a triangular (10x10) MatMul along the row axis;
  results restricted to their region (mask) so they never leak across the horizon.
  Within a region at most one source => OR is exact; blue/red fills are disjoint.

Encoding (route 10-ch expansion into FREE bool output via Equal(L,arange)):
  L = 1*blue_fill + 2*red_fill + 5*gray_row  (uint8, 10x10) -> Pad 255 -> Equal.
  Triangular MatMuls along rows: Tri[10,10] @ src[1,K,10,10] contracts rows.
"""

import numpy as np
import onnx
from onnx import helper, numpy_helper, TensorProto

from ..harness import IR_VERSION

F16 = TensorProto.FLOAT16
F32 = TensorProto.FLOAT
BOOL = TensorProto.BOOL
U8 = TensorProto.UINT8
I64 = TensorProto.INT64

N = 10  # active grid is always 10x10


def build(task):
    inits, nodes = [], []

    def init(name, arr, dtype):
        inits.append(numpy_helper.from_array(
            np.ascontiguousarray(arr, dtype=dtype), name))
        return name

    def n(op, ins, out, **attrs):
        nodes.append(helper.make_node(op, ins, [out], **attrs))
        return out

    # ---- slice the active 10x10 region, channels 1,2,5 -----------------------
    # blue=ch1, red=ch2, gray=ch5.  Slice as fp16 working planes.
    def chan_slice(ci, tag):
        init(f"cs_s_{tag}", np.array([ci, 0, 0], np.int64), np.int64)
        init(f"cs_e_{tag}", np.array([ci + 1, N, N], np.int64), np.int64)
        init(f"cs_ax_{tag}", np.array([1, 2, 3], np.int64), np.int64)
        n("Slice", ["input", f"cs_s_{tag}", f"cs_e_{tag}", f"cs_ax_{tag}"],
          f"{tag}_f32")                        # [1,1,10,10] f32
        n("Cast", [f"{tag}_f32"], f"{tag}", to=F16)   # [1,1,10,10] f16 {0,1}
        return f"{tag}"

    B = chan_slice(1, "blue")
    R = chan_slice(2, "red")
    G = chan_slice(5, "gray")

    # ---- horizon row mask: rowgray[r] = OR_c gray[r,c] (one-hot at horizon) ---
    n("ReduceMax", [G], "rowgray", axes=[3], keepdims=1)   # [1,1,10,1] f16

    # strict-upper / strict-lower triangular along rows (10x10)
    U = np.triu(np.ones((N, N), np.float16), k=1)   # U[r,j]=1 for j>r
    Lt = np.tril(np.ones((N, N), np.float16), k=-1)  # L[r,j]=1 for j<r
    init("UTRI", U, np.float16)
    init("LTRI", Lt, np.float16)
    # aboveMask[r] = exists gray at j>r  = (horizon>r) ; below = exists gray j<r
    # exactly one gray row => acc is already {0,1}, no thresholding needed.
    n("MatMul", ["UTRI", "rowgray"], "aboveM")   # [1,1,10,1] {0,1}
    n("MatMul", ["LTRI", "rowgray"], "belowM")

    # Each triangular MatMul accumulator is EXACTLY {0,1}: within a column there
    # is at most one source per (region,color), so the prefix/suffix sum over a
    # single 1 is 0 or 1 -> NO thresholding needed.
    #
    # inclusive prefix/suffix over rows: UINC[r,j]=1 j>=r ; LINC[r,j]=1 j<=r.
    UI = np.triu(np.ones((N, N), np.float16), k=0)
    LI = np.tril(np.ones((N, N), np.float16), k=0)
    init("UINC", UI, np.float16)
    init("LINC", LI, np.float16)

    # BLUE travels away from horizon -> source-mask to region (no result leak):
    n("Mul", [B, "aboveM"], "B_ab")
    n("Mul", [B, "belowM"], "B_be")
    n("MatMul", ["UINC", "B_ab"], "abB")   # above-blue: suffix-OR up   (j>=i)
    n("MatMul", ["LINC", "B_be"], "beB")   # below-blue: prefix-OR down (j<=i)
    n("Add", ["abB", "beB"], "blue_fill")  # disjoint regions -> {0,1}

    # RED travels toward horizon -> use full R, result-mask to region.
    # Fold the colour weight (*2) into the triangular matrices so red_fill is
    # already {0,2} (saves the explicit *2 plane).
    init("UINC2", UI * 2, np.float16)
    init("LINC2", LI * 2, np.float16)
    n("MatMul", ["LINC2", R], "abR_raw")   # above-red:  prefix-OR down (j<=i)
    n("MatMul", ["UINC2", R], "beR_raw")   # below-red:  suffix-OR up   (j>=i)
    n("Mul", ["abR_raw", "aboveM"], "abR")
    n("Mul", ["beR_raw", "belowM"], "beR")
    n("Add", ["abR", "beR"], "red_fill")   # {0,2}

    # ---- colour-index L = blue + red(*2) + gray(*5) in one Sum ---------------
    init("W5", np.array(5.0, np.float16), np.float16)
    n("Mul", [G, "W5"], "gray5")
    nodes.append(helper.make_node(
        "Sum", ["blue_fill", "red_fill", "gray5"], ["L_f16"]))  # {0,1,2,5}
    n("Cast", ["L_f16"], "L_u8", to=U8)     # [1,1,10,10] uint8

    # ---- pad to 30x30 with sentinel 255 --------------------------------------
    init("Lpads", np.array([0, 0, 0, 0, 0, 0, 30 - N, 30 - N], np.int64), np.int64)
    init("SENT", np.array(255, np.uint8), np.uint8)
    n("Pad", ["L_u8", "Lpads", "SENT"], "L30", mode="constant")  # [1,1,30,30] u8

    init("arange", np.arange(10, dtype=np.uint8).reshape(1, 10, 1, 1), np.uint8)
    n("Equal", ["L30", "arange"], "output")   # [1,10,30,30] BOOL (FREE)

    x = helper.make_tensor_value_info("input", F32, [1, 10, 30, 30])
    y = helper.make_tensor_value_info("output", BOOL, [1, 10, 30, 30])
    graph = helper.make_graph(nodes, "task212", [x], [y], inits)
    return helper.make_model(graph, ir_version=IR_VERSION,
                             opset_imports=[helper.make_opsetid("", 11)])
