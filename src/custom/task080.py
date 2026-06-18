"""task080 (ARC-AGI 39e1d7f9) — replicate the decorated "stamp" onto every center.

Rule (generator task_39e1d7f9.py, size 5..10, spacing sp = 6-(size-1)//2, period p=sp+1):
  The grid is a block lattice: each bitmap cell -> an (sp x sp) solid block, blocks
  separated by single full lines of `linecolor` (lines at row/col index == p-1 mod p).
  (size-1)//2 "pixels" each carry a CENTER block of colors[0].  In the INPUT exactly ONE
  pixel is fully decorated: its 4 orthogonal neighbour blocks = colors[1] (edge), its 4
  diagonal neighbour blocks = colors[2] (corner, only when there are 3 colours).  The
  OUTPUT decorates EVERY center the same way (clipped at the bitmap border).  Stamps never
  overlap, so fill lands only on background cells.

  Approach — collapse to BITMAP resolution (task159/task195 magnify lever), 500/500 fresh:
    colf= sum_k k*input_k (the ONE fp32 30x30 entry plane); rowsum = a no-pad 1x30 Conv
          over channels 1..9 (per-row non-bg count, no 30x30 occupancy plane);
    p   = first full-line row index + 1                          (the block period sp+1)
    B   = downsample colf to a 10x10 bitmap by gathering block-top cells (indices i*p);
          off-grid blocks (i*p >= A) masked to 0.
    on the tiny 10x10 (all 1-cell shifts via pad-once-to-12x12 + fixed Slice):
        center = occ & 4 ortho-neighbour occ; c0/c1/c2 = B at center / its up / up-left;
        seed = (B==c0)&occ; edge = ortho-dilate(seed), corner = diag-dilate(seed);
        outB = where(edge&bg,c1, where(corner&bg,c2, B))
    TWO-SENTINEL-BLOCK upscale: extend outB to a 12x12 block table (block10 = linecolor,
    block11 = 99 off-grid), and a single double-Gather with index
        uidx[i] = 11 if i>=A  else 10 if (i+1)%p==0  else i//p
    builds the WHOLE 30x30 output (line cells -> linecolor, off-grid -> 99) with NO tail
    Where and NO line / in-grid 30x30 mask planes.  outB12 is uint8 so the final Gather +
    Equal(arange) avoid an fp16->fp32 PrecisionFreeCast on the 30x30 plane.  Output is the
    FREE BOOL [1,10,30,30] = Equal(big_u8, arange).

  pts 15.22, mem 17106, params 487, fresh 500/500.  Dominant intermediate: colf32 (3600B
  fp32 entry, irreducible) + the downsample Gather Bg2 (1200B fp32, inherits colf dtype) +
  the uint8 final plane (900B).  Beats baseline 14.36 by +0.86.
"""

import numpy as np
import onnx
from onnx import helper, numpy_helper, TensorProto

from ..harness import IR_VERSION

F16 = TensorProto.FLOAT16
F32 = TensorProto.FLOAT
BOOL = TensorProto.BOOL
U8 = TensorProto.UINT8
I32 = TensorProto.INT32
I64 = TensorProto.INT64

W = 30
K = 10     # bitmap side (size <= 10)


def build(task):
    inits, nodes = [], []
    _np = {F16: np.float16, F32: np.float32, BOOL: np.bool_, U8: np.uint8,
           I32: np.int32, I64: np.int64}

    def init(name, arr, dtype):
        inits.append(numpy_helper.from_array(
            np.ascontiguousarray(arr, dtype=_np[dtype]), name))
        return name

    def n(op, ins, out, **attrs):
        nodes.append(helper.make_node(op, ins, [out], **attrs))
        return out

    # ---- colour-index plane colf [1,1,30,30] -> fp16 ----------------------
    init("KW", np.arange(10, dtype=np.float32).reshape(1, 10, 1, 1), F32)
    n("Conv", ["input", "KW"], "colf32")               # [1,1,30,30] fp32 = sum k*input_k
    init("ZH", np.array(0.0, np.float16), F16)
    init("ZF", np.array(0.0, np.float32), F32)

    # ---- row/col occupancy COUNTS as no-pad convs (no 30x30 occ plane) -----
    # rowsum[r] = # non-bg cells in row r = Conv over channels 1..9 with a 1x30 kernel.
    rk = np.zeros((1, 10, 1, W), np.float32); rk[0, 1:, 0, :] = 1.0
    init("ROWK", rk, F32)
    n("Conv", ["input", "ROWK"], "rowsum")             # [1,1,30,1] fp32 = per-row count

    # ---- period p = first full-line row index + 1 -------------------------
    n("ReduceMax", ["rowsum"], "Amax", axes=[2, 3], keepdims=1)  # = active width A
    init("HALFF", np.array(0.5, np.float32), F32)
    n("Sub", ["Amax", "HALFF"], "WTHR")
    n("Greater", ["rowsum", "WTHR"], "isrow_b")        # [1,1,30,1] bool full-line rows
    n("Cast", ["isrow_b"], "isrowf", to=F16)
    n("ArgMax", ["isrowf"], "pm1_i64", axis=2, keepdims=0)   # [1,1,1] = p-1
    n("Cast", ["pm1_i64"], "pm1_i32", to=I32)
    init("ONE_I", np.array(1, np.int32), I32)
    n("Add", ["pm1_i32", "ONE_I"], "p_i32a")           # [1,1,1]
    init("flat1", np.array([1], np.int64), I64)
    n("Reshape", ["p_i32a", "flat1"], "p_i32")         # [1]

    # ---- downsample: bitmap B = colf at block-tops (stride p) -------------
    # block-top indices i*p, CLAMPED to 29 (off-grid blocks read garbage, masked below).
    init("ar10f", np.arange(K, dtype=np.float32), F32)         # [10]
    n("Cast", ["p_i32"], "pf0", to=F32)
    n("Mul", ["ar10f", "pf0"], "bidf")                        # i*p float
    init("HI29F", np.array(float(W - 1), np.float32), F32)
    n("Min", ["bidf", "HI29F"], "bidcf")                      # clamp <=29
    n("Cast", ["bidcf"], "bidxc", to=I32)                     # [10] gather indices
    n("Gather", ["colf32", "bidxc"], "Bg2", axis=2)            # [1,1,10,30]
    n("Gather", ["Bg2", "bidxc"], "Braw32", axis=3)          # [1,1,10,10] fp32 raw colour idx
    n("Cast", ["Braw32"], "Braw", to=F16)                    # tiny bitmap -> fp16
    # block validity: block i exists iff i*p < A (else off-grid -> force 0)
    n("Cast", ["Amax"], "AmaxF", to=F32)                     # [1,1,1,1] = A
    n("Less", ["bidf", "AmaxF"], "bvalid_b")                 # [10]? broadcasts to [1,1,1,10]
    n("Cast", ["bvalid_b"], "bvalidf", to=F16)
    # bvalidf is along last gathered axis; need both row & col validity -> outer AND.
    init("v_rsh", np.array([1, 1, K, 1], np.int64), I64)
    init("v_csh", np.array([1, 1, 1, K], np.int64), I64)
    n("Reshape", ["bvalidf", "v_rsh"], "vrow")              # [1,1,10,1]
    n("Reshape", ["bvalidf", "v_csh"], "vcol")              # [1,1,1,10]
    n("Mul", ["vrow", "vcol"], "vmask")                    # [1,1,10,10] {0,1}
    n("Mul", ["Braw", "vmask"], "B")                       # off-grid blocks -> 0
    n("Greater", ["B", "ZH"], "bocc_b")
    n("Cast", ["bocc_b"], "bocc", to=F16)                     # [1,1,10,10] {0,1}

    # ---- tiny 10x10 shifts: pad each source ONCE to 12x12 (zero border), then a
    # cheap fixed Slice picks each shifted 10x10 window (out[r,c]=src[r-dr,c-dc]).
    init("padB1", np.array([0, 0, 1, 1, 0, 0, 1, 1], np.int64), I64)
    init("SLAX", np.array([2, 3], np.int64), I64)
    _padded = {}

    def padb(src):
        if src not in _padded:
            _padded[src] = n("Pad", [src, "padB1", "ZH"], f"{src}_b12", mode="constant")
        return _padded[src]

    def shift1(src, dr, dc, tag):
        pb = padb(src)
        # window start = (1-dr, 1-dc) into the 12x12 padded buffer
        s0, s1 = 1 - dr, 1 - dc
        init(f"ss_{tag}", np.array([s0, s1], np.int64), I64)
        init(f"se_{tag}", np.array([s0 + K, s1 + K], np.int64), I64)
        return n("Slice", [pb, f"ss_{tag}", f"se_{tag}", "SLAX"], tag)

    # ---- center detection: bocc & 4 ortho-neighbour bocc ------------------
    U = shift1("bocc", 1, 0, "bU")     # neighbour above brought down
    D = shift1("bocc", -1, 0, "bD")
    L = shift1("bocc", 0, 1, "bL")
    R = shift1("bocc", 0, -1, "bR")
    n("Mul", ["bocc", U], "i1")
    n("Mul", ["i1", D], "i2")
    n("Mul", ["i2", L], "i3")
    n("Mul", ["i3", R], "iscenter")    # [1,1,10,10] {0,1}, marks decorated block

    # ---- colours c0,c1,c2 (scalars) --------------------------------------
    n("Mul", ["B", "iscenter"], "c0p")
    n("ReduceMax", ["c0p"], "c0", axes=[2, 3], keepdims=1)   # [1,1,1,1] center colour
    Bup = shift1("B", 1, 0, "Bup")                           # block-above colour
    n("Mul", ["Bup", "iscenter"], "c1p")
    n("ReduceMax", ["c1p"], "c1", axes=[2, 3], keepdims=1)   # edge colour
    Bul = shift1("Bup", 0, 1, "Bul")                         # up-left colour
    n("Mul", ["Bul", "iscenter"], "c2p")
    n("ReduceMax", ["c2p"], "c2", axes=[2, 3], keepdims=1)   # corner colour

    # ---- seed = (B==c0) & bocc, dilate ----------------------------------
    n("Equal", ["B", "c0"], "seq_b")
    n("Cast", ["seq_b"], "seq", to=F16)
    n("Mul", ["seq", "bocc"], "seed")                        # [1,1,10,10] {0,1}
    sU = shift1("seed", 1, 0, "sU")
    sD = shift1("seed", -1, 0, "sD")
    sL = shift1("seed", 0, 1, "sL")
    sR = shift1("seed", 0, -1, "sR")
    n("Add", [sU, sD], "vbar")                               # vertical neighbours
    n("Add", [sL, sR], "hbar")
    n("Add", ["vbar", "hbar"], "edge_sum")
    n("Greater", ["edge_sum", "ZH"], "edge_b")              # ortho neighbours
    cL = shift1("vbar", 0, 1, "cL")                          # vbar shifted L -> diag
    cR = shift1("vbar", 0, -1, "cR")
    n("Add", [cL, cR], "corner_sum")
    n("Greater", ["corner_sum", "ZH"], "corner_b")          # diagonal neighbours

    # ---- compose outB (colour index) on 10x10 ---------------------------
    n("Not", ["bocc_b"], "bbg_b")                           # background bitmap cells
    n("And", ["edge_b", "bbg_b"], "edge_g")
    n("And", ["corner_b", "bbg_b"], "corner_g")
    n("Where", ["corner_g", "c2", "B"], "ob1")
    n("Where", ["edge_g", "c1", "ob1"], "outB")             # [1,1,10,10] fp16 colour idx

    # ---- TWO-SENTINEL-BLOCK upscale: ONE double-Gather builds the whole 30x30 output.
    # Extend outB to a 12x12 block table:
    #   blocks 0..9  : the bitmap content
    #   block  10    : LINECOLOR row/col (any line row/col reads this block)
    #   block  11    : 99 off-grid sentinel (off-grid reads this block; padded LAST so it
    #                  wins the corner where a line index meets an off-grid index)
    # Then big[r,c] = outB12[uidx[r], uidx[c]] is the final colour-index plane -- no tail
    # Where, no line/in-grid 30x30 mask planes.
    # linecolor scalar from colf[p-1, 0]
    n("Reshape", ["pm1_i32", "flat1"], "pm1v")              # [1]
    n("Gather", ["colf32", "pm1v"], "lcrow", axis=2)        # [1,1,1,30]
    init("z1", np.array([0], np.int64), I64)
    n("Gather", ["lcrow", "z1"], "lc32", axis=3)            # [1,1,1,1] linecolor fp32
    n("Cast", ["lc32"], "lc", to=F16)
    init("lc_sh", np.array([1], np.int64), I64)             # reshape to 1-elem (Pad value)
    n("Reshape", ["lc", "lc_sh"], "lc0")                    # [1] fp16 for Pad value
    init("padL", np.array([0, 0, 0, 0, 0, 0, 1, 1], np.int64), I64)
    n("Pad", ["outB", "padL", "lc0"], "outB11")            # [1,1,11,11] block10 = linecolor
    init("padO", np.array([0, 0, 0, 0, 0, 0, 1, 1], np.int64), I64)
    init("S99H0", np.array(99.0, np.float16), F16)
    n("Pad", ["outB11", "padO", "S99H0"], "outB12h")       # [1,1,12,12] block11 = 99
    # cast tiny block table to uint8: uint8 Gather/Equal avoid the fp16->fp32
    # PrecisionFreeCast on the full 30x30 plane (uint8 big = 900B, no upcast).
    n("Cast", ["outB12h"], "outB12", to=U8)                # [1,1,12,12] uint8

    # uidx[i]:  off-grid(i>=A) -> 11 ;  in-grid line((i+1)%p==0) -> 10 ;  else i//p
    init("ar30f", np.arange(W, dtype=np.float32).reshape(1, 1, W), F32)
    n("Cast", ["p_i32"], "pf", to=F32)                      # [1]
    n("Div", ["ar30f", "pf"], "qf")
    n("Floor", ["qf"], "qfl")                               # i//p (>=0)
    # line indices: (i+1) mod p == 0
    init("ar30b", np.arange(W, dtype=np.int32), I32)        # [30]
    n("Add", ["ar30b", "ONE_I"], "ip1")
    n("Mod", ["ip1", "p_i32"], "imod")
    init("ZI", np.array(0, np.int32), I32)
    n("Equal", ["imod", "ZI"], "linevec_b")                 # [30] bool
    init("lv_sh", np.array([1, 1, W], np.int64), I64)
    n("Reshape", ["linevec_b", "lv_sh"], "linevec3")        # [1,1,30]
    init("TENF", np.array(10.0, np.float32), F32)
    n("Where", ["linevec3", "TENF", "qfl"], "uidx_l")       # line -> 10 else i//p
    # off-grid override -> 11
    n("Cast", ["Amax"], "AmaxF2", to=F32)
    n("Less", ["ar30f", "AmaxF2"], "ingrid_vec")            # [1,1,30] bool i<A
    init("ELEVENF", np.array(11.0, np.float32), F32)
    n("Where", ["ingrid_vec", "uidx_l", "ELEVENF"], "uidxf")  # off-grid -> 11
    n("Cast", ["uidxf"], "uidx", to=I32)
    init("flat30", np.array([W], np.int64), I64)
    n("Reshape", ["uidx", "flat30"], "uidx30")              # [30]
    n("Gather", ["outB12", "uidx30"], "big2", axis=2)       # [1,1,30,12]
    n("Gather", ["big2", "uidx30"], "big", axis=3)          # [1,1,30,30] fp16 FINAL colours

    # ---- Equal(big, arange) -> BOOL [1,10,30,30] FREE ------------------
    init("arangeU", np.arange(10, dtype=np.uint8).reshape(1, 10, 1, 1), U8)
    n("Equal", ["big", "arangeU"], "output")

    # drop any unreferenced initializers (they still count as params)
    used = {i for node in nodes for i in node.input}
    inits = [t for t in inits if t.name in used]

    x = helper.make_tensor_value_info("input", F32, [1, 10, 30, 30])
    y = helper.make_tensor_value_info("output", BOOL, [1, 10, 30, 30])
    graph = helper.make_graph(nodes, "task080", [x], [y], inits)
    return helper.make_model(graph, ir_version=IR_VERSION,
                             opset_imports=[helper.make_opsetid("", 11)])
