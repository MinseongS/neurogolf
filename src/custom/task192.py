"""task192 (ARC-AGI 7e0986d6) — "remove the isolated static pixels, keep the boxes".

Rule (from the generator):
  The grid (background = colour 0, size 10..20 in each axis) holds 3..5 SOLID
  rectangular boxes of one colour `boxcolor` (each >=3 wide & >=3 tall, mutually
  separated by a >=1 gap) plus a sprinkling of single "static" pixels of a SECOND
  colour `color`.  The static pixels come from `remove_neighbors(random_pixels)`
  so no two static pixels are 4-adjacent, but a static pixel MAY land on a box
  cell (overwriting it to `color`) or sit next to a box.  OUTPUT = the boxes only,
  rendered in `boxcolor`: every static pixel is deleted and every box hole that a
  static pixel had punched is re-filled with `boxcolor`.

  Two deterministic facts (verified over 1000 fresh instances):
   * KEEP MASK is closed-form local: keep(r,c) iff cell (r,c) is part of at least
     one fully OCCUPIED 2x2 square (occupancy = any non-background colour).  A box
     cell (box >=3x3, and a static pixel sitting on a box cell still counts as
     occupied) is always inside a filled 2x2; an isolated static pixel — never
     4-adjacent to another static pixel and never able to fill a 2x2 from a single
     box side because the box's far side has the mandatory >=1 gap — is never in
     a filled 2x2.  Exact: 0/500 mismatch.
   * VALUE is a single scalar `boxcolor` = the most-frequent non-background colour
     (each box has >=9 cells; static pixels are sparse), recovered as
     argmax over channels 1..9 of the per-channel pixel COUNT.  Exact: 0/1000.

Encoding (floor-break, route the 10-ch expansion into the FREE output):
  occ  = (max over channels of input)>0, but channel 0 (background) is excluded by
         using colf = Conv(input, w=[0,1,..,9]) (sum_k k*input_k) > 0.5.
  keep = occ part-of-filled-2x2  (two 2x2 sum-Convs, exactly as task193).
  boxhot[1,10,1,1] = Equal(cnt, ReduceMax(cnt)) with cnt = ReduceSum(input,[2,3])
         and channel 0 zeroed -> a one-hot over the box colour channel.
  output = Where(keep, boxhot, bg_onehot)  -> [1,10,30,30] ONLY as the free output
         (boxhot/bg both [1,10,1,1] broadcast in; keep is [1,1,30,30]).  Off-grid
         cells (>=size) are non-occupied -> keep=0 -> bg_onehot (ch0=1), which is
         the correct all-zero / background padding.
  Working planes are 20x20 fp16 (size<=20), the dominant intermediate.
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

    # ---- ONE 1x1 Conv packs BOTH signals into a single 30x30 plane ---------
    # weights: ch0(background)=1, ch_k(k>=1)=10+k.  So g==0 off-grid (all-zero
    # one-hot), g==1 in-grid background, g in [11..19] in-grid coloured cell.
    #   ingrid   = g > 0.5      (grid extent, incl. background)
    #   occupied = g > 9.5      (any coloured / non-background cell)
    wpack = np.zeros((1, 10, 1, 1), np.float32)
    wpack[0, 0, 0, 0] = 1.0
    for k in range(1, 10):
        wpack[0, k, 0, 0] = 10.0 + k
    init("WPACK", wpack, np.float32)
    n("Conv", ["input", "WPACK"], "g30", kernel_shape=[1, 1])      # [1,1,30,30] f32
    # crop to 20x20 active region (max size 20) so working planes are 20x20.
    init("c20_s", np.array([0, 0], np.int64), np.int64)
    init("c20_e", np.array([20, 20], np.int64), np.int64)
    init("c20_ax", np.array([2, 3], np.int64), np.int64)
    n("Slice", ["g30", "c20_s", "c20_e", "c20_ax"], "g")           # [1,1,20,20] f32
    init("NINE5", np.array(9.5, np.float32), np.float32)
    n("Greater", ["g", "NINE5"], "occ_b")                          # coloured cell
    n("Cast", ["occ_b"], "occ", to=F16)                            # [1,1,20,20] f16 {0,1}

    # ---- block = Conv(occ, 2x2 ones), pad bottom/right ---------------------
    Wk = np.ones((1, 1, 2, 2), np.float16)
    init("Wsum", Wk, np.float16)
    n("Conv", ["occ", "Wsum"], "blockcnt",
      kernel_shape=[2, 2], pads=[0, 0, 1, 1])                      # [1,1,20,20] f16
    init("THREE5", np.array(3.5, np.float16), np.float16)
    n("Greater", ["blockcnt", "THREE5"], "blockfull_b")            # full 2x2 (==4)
    n("Cast", ["blockfull_b"], "blockfull", to=F16)

    # ---- keep = Conv(blockfull, 2x2 ones), pad top/left --------------------
    # a cell is kept iff ANY of the 4 covering 2x2 blocks is full.
    init("Wsum2", Wk, np.float16)
    n("Conv", ["blockfull", "Wsum2"], "keepcnt",
      kernel_shape=[2, 2], pads=[1, 1, 0, 0])                      # [1,1,20,20] f16
    init("HALFH", np.array(0.5, np.float16), np.float16)
    n("Greater", ["keepcnt", "HALFH"], "keep20_b")                 # [1,1,20,20] bool

    # ---- boxcolor SCALAR bc = argmax-count channel index -------------------
    # cnt = per-channel pixel count; zero ch0; one-hot the argmax channel; then
    # contract against the channel ramp to a scalar colour index.
    n("ReduceSum", ["input"], "cnt_raw", axes=[2, 3], keepdims=1)  # [1,10,1,1] f32
    mask0 = np.ones((1, 10, 1, 1), np.float32); mask0[0, 0, 0, 0] = 0.0
    init("MASK0", mask0, np.float32)
    n("Mul", ["cnt_raw", "MASK0"], "cnt")                          # [1,10,1,1] f32
    n("ReduceMax", ["cnt"], "cntmax", axes=[1], keepdims=1)        # [1,1,1,1] f32
    n("Equal", ["cnt", "cntmax"], "boxhot_b")                      # [1,10,1,1] bool
    n("Cast", ["boxhot_b"], "boxhot", to=F32)                      # [1,10,1,1] f32 one-hot
    ramp = np.arange(10, dtype=np.float32).reshape(1, 10, 1, 1)
    init("CRAMP", ramp, np.float32)
    n("Mul", ["boxhot", "CRAMP"], "bchan")                         # [1,10,1,1] f32
    n("ReduceSum", ["bchan"], "bc", axes=[1], keepdims=1)          # [1,1,1,1] f32 scalar

    # ---- in-grid mask (20x20): g > 0.5 (background==1, off-grid==0) ---------
    init("HALF32", np.array(0.5, np.float32), np.float32)
    n("Greater", ["g", "HALF32"], "ingrid_b")                      # [1,1,20,20] bool
    n("Cast", ["ingrid_b"], "ingrid_f", to=F16)                    # [1,1,20,20] f16
    n("Cast", ["keep20_b"], "keep_f", to=F16)                      # [1,1,20,20] f16
    n("Cast", ["bc"], "bc16", to=F16)                              # scalar f16

    # ---- single colour-index plane L (20x20) ------------------------------
    #   L = keep*bc - (1 - ingrid)
    #     keep            -> bc   (box colour channel)
    #     ingrid & !keep  -> 0    (background channel)
    #     off-grid        -> -1   (matches no channel -> all-zero)
    n("Mul", ["keep_f", "bc16"], "keep_bc")                        # [1,1,20,20] f16
    init("ONE16", np.array(1.0, np.float16), np.float16)
    n("Sub", ["ONE16", "ingrid_f"], "offg")                        # 1-ingrid
    n("Sub", ["keep_bc", "offg"], "L20")                           # [1,1,20,20] f16
    # pad L to 30x30 with -1 (cells >=20 are always off-grid -> match no channel)
    init("opads", np.array([0, 0, 0, 0, 0, 0, 10, 10], np.int64), np.int64)
    init("NEG1H", np.array(-1.0, np.float16), np.float16)
    n("Pad", ["L20", "opads", "NEG1H"], "L30", mode="constant")    # [1,1,30,30] f16
    # expand to the 10-channel one-hot via Equal vs the channel ramp -> the
    # 10-ch expansion lands ONLY in the FREE bool output (no full plane stored).
    crampf = np.arange(10, dtype=np.float16).reshape(1, 10, 1, 1)
    init("CRAMPF", crampf, np.float16)
    n("Equal", ["L30", "CRAMPF"], "output")                        # [1,10,30,30] bool FREE

    x = helper.make_tensor_value_info("input", F32, [1, 10, 30, 30])
    y = helper.make_tensor_value_info("output", BOOL, [1, 10, 30, 30])
    g = helper.make_graph(nodes, "task192", [x], [y], inits)
    return helper.make_model(g, ir_version=IR_VERSION,
                             opset_imports=[helper.make_opsetid("", 11)])
