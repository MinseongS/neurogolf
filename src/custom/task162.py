"""task162 (ARC-AGI 6cf79266) — "fill the 3x3 black holes with blue".

Rule (from generator task_6cf79266.py, size=20 fixed, one foreground colour
K in {2..9} on a dense black static background, 1..3 black 3x3 cutouts):
  The generator scans interior cells (r,c in [1,18]) in ROW-MAJOR order and,
  whenever the 3x3 window centred at (r,c) is currently all-black IN THE OUTPUT,
  paints that whole 3x3 block blue(=channel 1).  Because it edits the output
  in place, an earlier-scanned all-black window that overlaps a later one
  SUPPRESSES the later fill (the later window is no longer all-black).  Two
  3x3 windows overlap iff their centres are within Chebyshev distance 2.

  Exact closed-form (verified vs the in-place generator over 8000 fresh
  instances AND all stored examples, 0 mismatch):
    black   = input ch0 (one-hot black presence), cropped to the 20x20 grid
    cand    = Conv(black, 3x3 ones, pad1) >= 9        # all-9-black windows
    blocker = Conv(cand,  K, pad2) > 0                # K = the 12 row-major-
              EARLIER offsets within Chebyshev<=2 (causal half of the 5x5)
    fire    = cand AND NOT blocker                    # a window fires iff no
              earlier overlapping all-black window exists (the earlier one
              always fires, so chains never form -> purely local)
    blue    = Conv(fire,  3x3 ones, pad1) > 0         # paint each fired 3x3
    output  = Where(blue, blue_onehot, input)

NOTE: a naive parallel "fill every all-black 3x3" over-fills overlapping holes
and FAILS the stored example with two adjacent cutouts; the causal blocker is
load-bearing.

Off-grid handling: the 20x20 grid sits top-left of the 30x30 canvas; off-grid
cells are ch0=1 (black).  Slicing black to the 20x20 region BEFORE the cand
Conv means any window touching off-grid can never reach 9 -> centres are
auto-restricted to the grid interior.  blue is padded back to 30x30 with zeros
(no blue off-grid); a single Where routes the 10-ch expansion into the FREE
output (blue cells -> channel-1 one-hot, all others copy input).

Dominant intermediates: the [1,1,20,20] fp16/bool planes (~400-800B each);
the 3x3 neighbourhood ops need a per-cell plane, but the region is cropped to
20x20 and every plane is fp16/bool.
"""

import numpy as np
import onnx
from onnx import helper, numpy_helper, TensorProto

from ..harness import IR_VERSION

F32 = TensorProto.FLOAT
F16 = TensorProto.FLOAT16
BOOL = TensorProto.BOOL
I64 = TensorProto.INT64


def build(task):
    inits, nodes = [], []

    def init(name, arr, dtype):
        inits.append(numpy_helper.from_array(
            np.ascontiguousarray(arr, dtype=dtype), name))
        return name

    def n(op, ins, out, **attrs):
        nodes.append(helper.make_node(op, ins, [out], **attrs))
        return out

    # ---- black-in-grid: slice ch0 to the active 20x20 region ----------------
    # ch0 is the black one-hot.  Slicing to [0:20,0:20] drops the off-grid black
    # so centre windows touching off-grid can never reach a 3x3-all-black count.
    init("s_st", np.array([0, 0, 0], np.int64), np.int64)   # ch0, r0, c0
    init("s_en", np.array([1, 20, 20], np.int64), np.int64)
    init("s_ax", np.array([1, 2, 3], np.int64), np.int64)
    n("Slice", ["input", "s_st", "s_en", "s_ax"], "black32")  # [1,1,20,20] f32
    n("Cast", ["black32"], "black", to=F16)                    # [1,1,20,20] f16

    # ---- cand = all-9-black 3x3 windows (pad1 -> stays 20x20) ---------------
    W3 = np.ones((1, 1, 3, 3), np.float16)
    init("W3a", W3, np.float16)
    n("Conv", ["black", "W3a"], "ccount", kernel_shape=[3, 3],
      pads=[1, 1, 1, 1])                                       # [1,1,20,20] f16
    init("EIGHT5", np.array(8.5, np.float16), np.float16)
    n("Greater", ["ccount", "EIGHT5"], "cand_b")               # >=9 -> cand
    n("Cast", ["cand_b"], "cand", to=F16)                      # [1,1,20,20] f16

    # ---- blocker = any row-major-EARLIER overlapping cand (causal 5x5) ------
    # K[a,b]=1 for offsets (a-2,b-2) strictly earlier in row-major order within
    # Chebyshev<=2 (rows -2,-1 all cols; row 0 cols -2,-1) -> 12 ones.
    Kc = np.zeros((1, 1, 5, 5), np.float16)
    for a in range(5):
        for b in range(5):
            do, dc = a - 2, b - 2
            if do < 0 or (do == 0 and dc < 0):
                Kc[0, 0, a, b] = 1.0
    init("Kcausal", Kc, np.float16)
    n("Conv", ["cand", "Kcausal"], "blkcount", kernel_shape=[5, 5],
      pads=[2, 2, 2, 2])                                       # [1,1,20,20] f16
    init("HALFH", np.array(0.5, np.float16), np.float16)
    n("Less", ["blkcount", "HALFH"], "notblk_b")               # ~blocker directly
    n("And", ["cand_b", "notblk_b"], "fire_b")                 # cand & ~blocker
    n("Cast", ["fire_b"], "fire", to=F16)                      # [1,1,20,20] f16

    # ---- blue = dilate fired centres by a 3x3; threshold to u8; pad to 30x30 -
    init("W3b", W3, np.float16)
    n("Conv", ["fire", "W3b"], "bcount", kernel_shape=[3, 3],
      pads=[1, 1, 1, 1])                                       # [1,1,20,20] f16
    n("Greater", ["bcount", "HALFH"], "blue20_b")              # [1,1,20,20] bool
    n("Cast", ["blue20_b"], "blue20_u8", to=TensorProto.UINT8)
    init("bpads", np.array([0, 0, 0, 0, 0, 0, 10, 10], np.int64), np.int64)
    init("ZEROU8", np.array(0, np.uint8), np.uint8)
    n("Pad", ["blue20_u8", "bpads", "ZEROU8"], "blue30_u8", mode="constant")
    n("Cast", ["blue30_u8"], "blue_b", to=BOOL)                # [1,1,30,30] bool

    # ---- single Where -> FREE [1,10,30,30] output --------------------------
    blue_oh = np.zeros((1, 10, 1, 1), np.float32)
    blue_oh[0, 1, 0, 0] = 1.0                                  # channel 1 = blue
    init("blue_onehot", blue_oh, np.float32)
    n("Where", ["blue_b", "blue_onehot", "input"], "output")

    x = helper.make_tensor_value_info("input", F32, [1, 10, 30, 30])
    y = helper.make_tensor_value_info("output", F32, [1, 10, 30, 30])
    gr = helper.make_graph(nodes, "task162", [x], [y], inits)
    return helper.make_model(gr, ir_version=IR_VERSION,
                             opset_imports=[helper.make_opsetid("", 11)])
