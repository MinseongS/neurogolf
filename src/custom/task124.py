"""task124 (ARC-AGI 53b68214) — extend a vertically (and optionally diagonally)
periodic sprite tiling down to fill the 10x10 output.

Rule (from the ARC-GEN generator, verified fresh):
  Input is an H x 10 grid (H in 5..8) holding the TOP of a periodic pattern.
  A small sprite of height `tall` (1..3) repeats vertically with period `tall`;
  if it repeats diagonally it ALSO shifts right by `shift = (wide-1)*diag`
  (shift in {0,1,2}) every period.  The OUTPUT is the same pattern extended
  across all 10 rows:
      out(r, c) = P[r % tall][ c - shift*(r // tall) ]   (0 if the source col is
                  out of range), where P = the first `tall` rows of the input.

Recovery of (tall, shift) — confirmed exact over 8000 fresh instances:
  For each candidate tall t in {1,2,3} derive the diagonal shift geometrically as
      shift(t) = leftmost-coloured-col(block1) - leftmost-coloured-col(block0)
  (block b = rows [b*t, (b+1)*t)); then choose the SMALLEST t whose (t, shift(t))
  makes the input self-consistent (row r equals row r-t shifted right by shift,
  over occupied row pairs).  Implemented as a 3-candidate mismatch + ArgMin on a
  tie-break key (mismatch*100 + t).

Memory: the only fp32 full-grid intermediate is the colour-index Conv plane
[1,1,30,30] (3600B) and the single fp16 30x30 carrier feeding the one-hot
expansion (1800B); all detection runs on 10x10 fp16 crops and the 10-channel
expansion is routed into the FREE bool output.
"""

import numpy as np
from onnx import helper, numpy_helper, TensorProto

from ..harness import IR_VERSION

S = 30
K = 10  # active canvas


def build(task):
    inits, nodes = [], []

    def init(name, arr, dt):
        inits.append(numpy_helper.from_array(np.ascontiguousarray(arr, dtype=dt), name))
        return name

    def n(op, ins, out, **attrs):
        nodes.append(helper.make_node(op, ins, [out], **attrs))
        return out

    F = TensorProto.FLOAT
    F16 = TensorProto.FLOAT16
    I64 = TensorProto.INT64
    B = TensorProto.BOOL

    # ---- colour-index plane: colf = sum_k k*input_k (on a cropped HxW canvas,
    # channels 1..9 only — channel 0 (background) has weight 0).
    HMAX = 8  # input height <= 8; source rows used <= tall-1 <= 2
    init("chW", np.arange(1, 10, dtype=np.float32).reshape(1, 9, 1, 1), np.float32)
    init("s0", np.array([1, 0, 0], np.int64), np.int64)
    init("sHK", np.array([10, HMAX, K], np.int64), np.int64)
    init("ax123", np.array([1, 2, 3], np.int64), np.int64)
    n("Slice", ["input", "s0", "sHK", "ax123"], "inHK")    # [1,9,8,10] fp32
    n("Conv", ["inHK", "chW"], "colfK")                    # [1,1,8,10] fp32
    n("Cast", ["colfK"], "V", to=F16)                      # [1,1,8,10] fp16

    # occupancy plane (bool) and fp16 occupancy
    init("z16", np.array(0.0, np.float16), np.float16)
    n("Greater", ["V", "z16"], "occB")                     # [1,1,10,10] bool
    n("Cast", ["occB"], "occF", to=F16)                    # [1,1,10,10] fp16

    # row occupancy rowocc[r] = max over cols (>0) -> [1,1,10,1] fp16
    n("ReduceMax", ["occF"], "rowoccF", axes=[3], keepdims=1)  # already 0/1

    init("flat1", np.array([1], np.int64), np.int64)
    init("flatH", np.array([HMAX], np.int64), np.int64)

    # ---- per-row occupancy BITMASK bm[r] = sum_c 2^c * occ[r,c] (fp32, exact) ----
    n("Cast", ["occB"], "occ32", to=F)                     # [1,1,8,10] fp32
    init("w2col", (2.0 ** np.arange(10)).astype(np.float32).reshape(10, 1), np.float32)
    n("MatMul", ["occ32", "w2col"], "bm4")                 # [1,1,8,1] fp32 (contract width)
    n("Reshape", ["bm4", "flatH"], "bmv")                  # [8] fp32
    # pad bm by 3 zeros on top -> [11], so a t-row shift is a static slice
    init("padbm", np.array([3, 0], np.int64), np.int64)
    init("z32", np.array(0.0, np.float32), np.float32)
    n("Pad", ["bmv", "padbm", "z32"], "bmp")               # [11] fp32
    # row occupancy (bool over the 8 rows)
    n("Greater", ["bmv", "z32"], "rowoccB1")               # [8] bool

    # ---- per-ROW leftmost coloured column lc[r] (for shift derivation) ----
    colramp_v = np.arange(K, dtype=np.float16).reshape(1, 1, 1, K)
    init("colramp4", colramp_v, np.float16)
    init("big99", np.array(99.0, np.float16), np.float16)
    n("Where", ["occB", "colramp4", "big99"], "ci")        # [1,1,8,10] fp16
    n("ReduceMin", ["ci"], "lc", axes=[3], keepdims=0)     # [1,1,8] fp16
    n("Reshape", ["lc", "flatH"], "lcv")                   # [8] fp16
    init("idx0", np.array([0], np.int64), np.int64)
    n("Gather", ["lcv", "idx0"], "lc0", axis=0)            # [1] fp16
    init("pow2tab", np.array([1.0, 2.0, 4.0], np.float32), np.float32)  # 2^shift
    init("ax0", np.array([0], np.int64), np.int64)

    # ---- per-candidate t in {1,2,3}: shift(t) + 1-D bitmask consistency ----
    for t in (1, 2, 3):
        # shift(t) = clip(lc[t]-lc[0], 0, 2)
        init(f"idxt_{t}", np.array([t], np.int64), np.int64)
        n("Gather", ["lcv", f"idxt_{t}"], f"lct_{t}", axis=0)   # [1] fp16
        n("Sub", [f"lct_{t}", "lc0"], f"shraw_{t}")
        n("Cast", [f"shraw_{t}"], f"shrawF_{t}", to=F)
        init("shlo", np.array(0.0, np.float32), np.float32) if t == 1 else None
        init("shhi", np.array(2.0, np.float32), np.float32) if t == 1 else None
        n("Clip", [f"shrawF_{t}", "shlo", "shhi"], f"shclipF_{t}")  # [1] fp32 shift
        n("Cast", [f"shclipF_{t}"], f"shift_{t}", to=I64)       # [1] int64
        # 2^shift via table lookup
        n("Gather", ["pow2tab", f"shift_{t}"], f"pw_{t}", axis=0)   # [1] fp32
        # bm[r-t] = slice bmp[3-t : 11-t]
        init(f"rs_{t}", np.array([3 - t], np.int64), np.int64)
        init(f"re_{t}", np.array([HMAX + 3 - t], np.int64), np.int64)
        n("Slice", ["bmp", f"rs_{t}", f"re_{t}", "ax0"], f"bmsh_{t}")  # [8] fp32 = bm[r-t]
        # pred = (bm[r-t] * 2^shift) mod 1024
        n("Mul", [f"bmsh_{t}", f"pw_{t}"], f"prem_{t}")         # [8] fp32
        n("Mod", [f"prem_{t}", "mod1024"], f"pred_{t}", fmod=1)  # [8] fp32
        init("mod1024", np.array(1024.0, np.float32), np.float32) if t == 1 else None
        # gate: predecessor row r-t exists (bm[r-t]>0, excludes the first t rows
        # and beyond-H) AND current row r occupied (bm[r]>0).
        n("Greater", [f"bmsh_{t}", "z32"], f"predocc_{t}")      # [8] bool
        n("And", [f"predocc_{t}", "rowoccB1"], f"gate_{t}")     # [8] bool
        # mismatch over gated rows: bm[r] != pred
        n("Equal", ["bmv", f"pred_{t}"], f"eq_{t}")             # [8] bool
        n("Not", [f"eq_{t}"], f"ne_{t}")
        n("And", [f"ne_{t}", f"gate_{t}"], f"bad_{t}")          # [8] bool
        n("Cast", [f"bad_{t}"], f"badf_{t}", to=F)              # [8] fp32
        n("ReduceSum", [f"badf_{t}"], f"m_{t}", axes=[0], keepdims=1)  # [1] fp32

    shift_scalars = ["shift_1", "shift_2", "shift_3"]
    n("Concat", ["m_1", "m_2", "m_3"], "mvec", axis=0)     # [3] fp32

    # ---- choose tall: ArgMin of key = mismatch*100 + t -------------------
    init("tieb", np.array([1.0, 2.0, 3.0], np.float32), np.float32)
    init("c100", np.array(100.0, np.float32), np.float32)
    n("Mul", ["mvec", "c100"], "mbig")
    n("Add", ["mbig", "tieb"], "key")                      # [3] fp32
    n("ArgMin", ["key"], "bestI", axis=0, keepdims=0)      # int64 scalar  (0,1,2)
    # tall = bestI + 1
    init("oneI", np.array(1, np.int64), np.int64)
    n("Add", ["bestI", "oneI"], "tall")                    # int64 scalar

    # shift = shift_scalars[bestI] : concat the three int64 [1] shifts -> [3]
    n("Concat", shift_scalars, "shvec", axis=0)            # [3] int64
    n("Reshape", ["bestI", "flat1"], "bestIv")             # [1]
    n("Gather", ["shvec", "bestIv"], "shift", axis=0)      # [1] int64

    # ---- source index maps (output 10x10) --------------------------------
    init("rowramp", np.arange(K, dtype=np.int64), np.int64)  # [10]
    n("Reshape", ["tall", "flat1"], "tallv")               # [1]
    n("Mod", ["rowramp", "tallv"], "sr")                   # [10] r%tall
    n("Div", ["rowramp", "tallv"], "block")                # [10] r//tall
    n("Mul", ["block", "shift"], "rowoff")                 # [10] shift*block(r)

    # gather rows of V by sr -> A[r,c] = V[sr(r), c]  (sr in [0,tall-1] <= 2 < 8)
    init("kH10", np.array([HMAX, K], np.int64), np.int64)
    n("Reshape", ["V", "kH10"], "V2")                      # [8,10] fp16
    n("Gather", ["V2", "sr"], "Arows", axis=0)             # [10,10]

    # left-pad Arows by PADC zero cols so a negative source col -> background zero
    # (rowoff[r] in [0,18]; source col c-rowoff in [-18,9]; idx = PADC + c - rowoff in
    #  [0, PADC+9], always valid -> no clamp / no valid mask needed).
    NI = TensorProto.INT32
    PADC = 9
    init("padArows", np.array([0, PADC, 0, 0], np.int64), np.int64)
    init("z16b", np.array(0.0, np.float16), np.float16)
    n("Pad", ["Arows", "padArows", "z16b"], "Ap")          # [10, 10+PADC] fp16
    init("colrampR", (np.arange(K, dtype=np.int32) + PADC).reshape(1, K), np.int32)  # PADC+c
    init("rowoffshape", np.array([K, 1], np.int64), np.int64)
    n("Cast", ["rowoff"], "rowoff32", to=NI)               # [10] int32
    n("Reshape", ["rowoff32", "rowoffshape"], "rowoffCol")  # [10,1] int32
    n("Sub", ["colrampR", "rowoffCol"], "SCi")             # [10,10] int32 idx
    n("GatherElements", ["Ap", "SCi"], "outidx2", axis=1)  # [10,10] fp16

    # ---- expand to one-hot into FREE bool output ------------------------
    # carrier as uint8 (smaller than fp16): values 0..9 in grid, 99 sentinel outside
    U8 = TensorProto.UINT8
    n("Cast", ["outidx2"], "outidx2u", to=U8)              # [10,10] uint8
    init("k1110", np.array([1, 1, K, K], np.int64), np.int64)
    n("Reshape", ["outidx2u", "k1110"], "outidx4")         # [1,1,10,10] uint8
    init("padOut", np.array([0, 0, 0, 0, 0, 0, S - K, S - K], np.int64), np.int64)
    init("sent99", np.array(99, np.uint8), np.uint8)
    n("Pad", ["outidx4", "padOut", "sent99"], "outidx30")  # [1,1,30,30] uint8, 99 outside
    init("arange8", np.arange(10, dtype=np.uint8).reshape(1, 10, 1, 1), np.uint8)
    n("Equal", ["outidx30", "arange8"], "output")          # [1,10,30,30] bool FREE

    x = helper.make_tensor_value_info("input", F, [1, 10, 30, 30])
    y = helper.make_tensor_value_info("output", B, [1, 10, 30, 30])
    graph = helper.make_graph(nodes, "task124", [x], [y], inits)
    return helper.make_model(graph, ir_version=IR_VERSION,
                             opset_imports=[helper.make_opsetid("", 11)])
