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
    B   = downsample colf to a 10x10 bitmap by gathering block-top cells (indices i*p).

  PLANE-NARROWING re-golf (v2, 15.22->15.61, +0.39): the whole 10x10 bitmap stage runs in
  uint8 (value bitmap) and bool (masks).  colf32 (the one fp32 30x30 entry plane) is
  Cast->uint8 ONCE so the downsample gathers (Bg2u 300B, B 100B) and every shift/mask plane
  is 1B not 2B, halving the ~6KB fp16 10x10 working set.  Three structural collapses:
    - center detection = ONE fp16 cross-conv of occupancy == 4 (all 4 ortho neighbours
      occupied), replacing the 4-shift + 4-Greater + 3-And machinery (~1.2KB -> ~0.6KB);
    - seed dilation = two fp16 3x3 convs (cross kernel -> ortho/edge, X kernel -> diag/
      corner), replacing the 8-shift + OR machinery (~1.8KB -> ~0.8KB);
    - the uidx upscale float vectors are fp16 (values 0..29 exact) not fp32.
  The three colour scalars c0/c1/c2 still need ReduceMax (fp-only) so a tiny masked uint8
  block is cast to fp16 just for them.

  TWO-SENTINEL-BLOCK upscale: extend outB to a 12x12 uint8 block table (block10 = linecolor,
  block11 = 99 off-grid) and a single double-Gather with index
      uidx[i] = 11 if i>=A  else 10 if (i+1)%p==0  else i//p
  builds the WHOLE 30x30 output (line cells -> linecolor, off-grid -> 99) with NO tail
  Where and NO line / in-grid 30x30 mask planes.  Output is the FREE BOOL [1,10,30,30] =
  Equal(big_u8, arange).

  Dominant irreducible intermediate: colf32 (3600B fp32 entry — the 10->1 Conv reduction
  must output fp32) + its uint8 cast colfu (900B) + the final 30x30 colour plane big (900B
  uint8, feeds the FREE output Equal).  Everything else is tiny bitmap-resolution.

  pts 15.6128, mem 11468, params 467, fresh 200/200.  Beats prior adopted 15.2247 by +0.39.
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

    # ---- colour-index plane colf [1,1,30,30] fp32 (the ONE fp32 entry plane) -----
    init("KW", np.arange(10, dtype=np.float32).reshape(1, 10, 1, 1), F32)
    n("Conv", ["input", "KW"], "colf32")               # [1,1,30,30] fp32 = sum k*input_k
    n("Cast", ["colf32"], "colfu", to=U8)              # [1,1,30,30] uint8 (values 0-9)

    # ---- row occupancy COUNTS as a no-pad conv (no 30x30 occ plane) --------------
    rk = np.zeros((1, 10, 1, W), np.float32); rk[0, 1:, 0, :] = 1.0
    init("ROWK", rk, F32)
    n("Conv", ["input", "ROWK"], "rowsum")             # [1,1,30,1] fp32 = per-row count

    # ---- period p = first full-line row index + 1 -------------------------------
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

    # ---- downsample: bitmap B = colfu at block-tops (stride p), all uint8 --------
    init("ar10f", np.arange(K, dtype=np.float32), F32)         # [10]
    n("Cast", ["p_i32"], "pf0", to=F32)
    n("Mul", ["ar10f", "pf0"], "bidf")                        # i*p float
    init("HI29F", np.array(float(W - 1), np.float32), F32)
    n("Min", ["bidf", "HI29F"], "bidcf")                      # clamp <=29
    n("Cast", ["bidcf"], "bidxc", to=I32)                     # [10] gather indices
    n("Gather", ["colfu", "bidxc"], "Bg2u", axis=2)           # [1,1,10,30] uint8
    n("Gather", ["Bg2u", "bidxc"], "Braw", axis=3)           # [1,1,10,10] uint8 raw colour idx
    # block validity: block i exists iff i*p < A (else off-grid -> force 0)
    n("Cast", ["Amax"], "AmaxF", to=F32)                     # [1,1,1,1] = A
    n("Less", ["bidf", "AmaxF"], "bvalid_b")                 # [1,1,1,10] bool
    init("v_rsh", np.array([1, 1, K, 1], np.int64), I64)
    init("v_csh", np.array([1, 1, 1, K], np.int64), I64)
    n("Reshape", ["bvalid_b", "v_rsh"], "vrow")             # [1,1,10,1] bool
    n("Reshape", ["bvalid_b", "v_csh"], "vcol")             # [1,1,1,10] bool
    n("And", ["vrow", "vcol"], "vmask_b")                   # [1,1,10,10] bool
    init("U8Z", np.array(0, np.uint8), U8)
    n("Where", ["vmask_b", "Braw", "U8Z"], "B")            # off-grid blocks -> 0 uint8
    n("Greater", ["B", "U8Z"], "bocc_b")                   # [1,1,10,10] bool occupancy

    # ---- tiny 10x10 shifts: pad each uint8 source ONCE to 12x12 (zero border),
    # then a cheap fixed Slice picks each shifted 10x10 window (out[r,c]=src[r-dr,c-dc]).
    init("padB1", np.array([0, 0, 1, 1, 0, 0, 1, 1], np.int64), I64)
    init("SLAX", np.array([2, 3], np.int64), I64)
    _padded = {}

    def padb_u8(src):
        if src not in _padded:
            _padded[src] = n("Pad", [src, "padB1", "U8Z"], f"{src}_b12", mode="constant")
        return _padded[src]

    def shift_u8(src, dr, dc, tag):
        pb = padb_u8(src)
        s0, s1 = 1 - dr, 1 - dc
        init(f"ss_{tag}", np.array([s0, s1], np.int64), I64)
        init(f"se_{tag}", np.array([s0 + K, s1 + K], np.int64), I64)
        return n("Slice", [pb, f"ss_{tag}", f"se_{tag}", "SLAX"], tag)

    # ---- center detection via ONE fp16 cross-conv: a center has all 4 ortho
    # neighbours occupied -> cross-conv of occupancy == 4 (no 4-shift+And machinery).
    n("Cast", ["bocc_b"], "boccf", to=F16)                  # [1,1,10,10] fp16 {0,1}
    init("CROSSK", np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]],
                            np.float16).reshape(1, 1, 3, 3), F16)
    n("Conv", ["boccf", "CROSSK"], "occ_cnt", pads=[1, 1, 1, 1])  # ortho-occ count fp16
    init("FOURH", np.array(4.0, np.float16), F16)
    n("Equal", ["occ_cnt", "FOURH"], "all4_b")              # bool: 4 ortho neighbours
    n("And", ["bocc_b", "all4_b"], "iscenter_b")            # [1,1,10,10] bool decorated block

    # ---- colours c0,c1,c2 (scalars): mask the tiny block then fp16 ReduceMax -----
    BU = shift_u8("B", 1, 0, "BUsh")     # block above brought down (for c1)
    # c0 = center colour, c1 = block-above colour, c2 = up-left block colour.
    n("Where", ["iscenter_b", "B", "U8Z"], "c0u")
    n("Cast", ["c0u"], "c0f", to=F16)
    n("ReduceMax", ["c0f"], "c0", axes=[2, 3], keepdims=1)   # [1,1,1,1] fp16 center colour
    n("Where", ["iscenter_b", BU, "U8Z"], "c1u")
    n("Cast", ["c1u"], "c1f", to=F16)
    n("ReduceMax", ["c1f"], "c1", axes=[2, 3], keepdims=1)   # edge colour fp16
    BUL = shift_u8(BU, 0, 1, "BULsh")                        # up-left colour (uint8)
    n("Where", ["iscenter_b", BUL, "U8Z"], "c2u")
    n("Cast", ["c2u"], "c2f", to=F16)
    n("ReduceMax", ["c2f"], "c2", axes=[2, 3], keepdims=1)   # corner colour fp16

    # ---- seed = (B==c0)&occ ; dilate via two fp16 3x3 convs (ortho-cross + diag-X).
    # Conv-dilation collapses the 8-plane shift+OR machinery into 2 fp16 conv planes:
    # edge = ortho-neighbour-of-seed (cross kernel), corner = diag-neighbour (X kernel).
    n("Cast", ["c0"], "c0u8", to=U8)                        # [1,1,1,1] uint8 center colour
    n("Equal", ["B", "c0u8"], "seq_b")                      # bool
    n("And", ["seq_b", "bocc_b"], "seed_b")                 # [1,1,10,10] bool
    n("Cast", ["seed_b"], "seedf", to=F16)                  # [1,1,10,10] fp16 {0,1}
    init("DIAGK", np.array([[1, 0, 1], [0, 0, 0], [1, 0, 1]],
                           np.float16).reshape(1, 1, 3, 3), F16)
    init("ZH", np.array(0.0, np.float16), F16)
    n("Conv", ["seedf", "CROSSK"], "edge_cnt", pads=[1, 1, 1, 1])   # ortho count fp16
    n("Conv", ["seedf", "DIAGK"], "corner_cnt", pads=[1, 1, 1, 1])  # diag count fp16
    n("Greater", ["edge_cnt", "ZH"], "edge_b")             # ortho neighbours bool
    n("Greater", ["corner_cnt", "ZH"], "corner_b")         # diagonal neighbours bool

    # ---- compose outB (uint8 colour index) on 10x10 -----------------------------
    n("Not", ["bocc_b"], "bbg_b")                           # background bitmap cells
    n("And", ["edge_b", "bbg_b"], "edge_g")
    n("And", ["corner_b", "bbg_b"], "corner_g")
    # cast edge/corner colours to uint8 for the Where chain (Where keeps uint8)
    n("Cast", ["c1"], "c1u8", to=U8)
    n("Cast", ["c2"], "c2u8", to=U8)
    n("Where", ["corner_g", "c2u8", "B"], "ob1")           # uint8
    n("Where", ["edge_g", "c1u8", "ob1"], "outB")          # [1,1,10,10] uint8 colour idx

    # ---- TWO-SENTINEL-BLOCK upscale: ONE double-Gather builds the whole 30x30 output.
    #   blocks 0..9 : bitmap;  block 10 : linecolor;  block 11 : 99 off-grid sentinel.
    # linecolor scalar from colfu[p-1, 0]
    n("Reshape", ["pm1_i32", "flat1"], "pm1v")              # [1]
    n("Gather", ["colfu", "pm1v"], "lcrow", axis=2)         # [1,1,1,30] uint8
    init("z1", np.array([0], np.int64), I64)
    n("Gather", ["lcrow", "z1"], "lc0", axis=3)            # [1,1,1,1] uint8 linecolor
    init("lc_sh", np.array([1], np.int64), I64)
    n("Reshape", ["lc0", "lc_sh"], "lcv")                  # [1] uint8 for Pad value
    init("padL", np.array([0, 0, 0, 0, 0, 0, 1, 1], np.int64), I64)
    n("Pad", ["outB", "padL", "lcv"], "outB11")           # [1,1,11,11] block10 = linecolor
    init("padO", np.array([0, 0, 0, 0, 0, 0, 1, 1], np.int64), I64)
    init("S99U", np.array(99, np.uint8), U8)
    init("s99v", np.array([99], np.uint8), U8)
    n("Pad", ["outB11", "padO", "s99v"], "outB12")        # [1,1,12,12] uint8 block11 = 99

    # uidx[i]:  off-grid(i>=A) -> 11 ;  in-grid line((i+1)%p==0) -> 10 ;  else i//p
    # all the float index machinery is fp16 (values 0..29, exact) -> 60B/plane not 120B.
    init("ar30f", np.arange(W, dtype=np.float16).reshape(1, 1, W), F16)
    n("Cast", ["p_i32"], "pf", to=F16)                      # [1] fp16
    n("Div", ["ar30f", "pf"], "qf")
    n("Floor", ["qf"], "qfl")                               # i//p (>=0) fp16
    init("ar30b", np.arange(W, dtype=np.int32), I32)        # [30]
    n("Add", ["ar30b", "ONE_I"], "ip1")
    n("Mod", ["ip1", "p_i32"], "imod")
    init("ZI", np.array(0, np.int32), I32)
    n("Equal", ["imod", "ZI"], "linevec_b")                 # [30] bool
    init("lv_sh", np.array([1, 1, W], np.int64), I64)
    n("Reshape", ["linevec_b", "lv_sh"], "linevec3")        # [1,1,30]
    init("TENF", np.array(10.0, np.float16), F16)
    n("Where", ["linevec3", "TENF", "qfl"], "uidx_l")       # line -> 10 else i//p fp16
    n("Cast", ["Amax"], "AmaxF2", to=F16)
    n("Less", ["ar30f", "AmaxF2"], "ingrid_vec")            # [1,1,30] bool i<A
    init("ELEVENF", np.array(11.0, np.float16), F16)
    n("Where", ["ingrid_vec", "uidx_l", "ELEVENF"], "uidxf")  # off-grid -> 11 fp16
    n("Cast", ["uidxf"], "uidx", to=I32)
    init("flat30", np.array([W], np.int64), I64)
    n("Reshape", ["uidx", "flat30"], "uidx30")              # [30]
    n("Gather", ["outB12", "uidx30"], "big2", axis=2)       # [1,1,30,12] uint8
    n("Gather", ["big2", "uidx30"], "big", axis=3)          # [1,1,30,30] uint8 FINAL colours

    # ---- Equal(big, arange) -> BOOL [1,10,30,30] FREE ---------------------------
    init("arangeU", np.arange(10, dtype=np.uint8).reshape(1, 10, 1, 1), U8)
    n("Equal", ["big", "arangeU"], "output")

    used = {i for node in nodes for i in node.input}
    inits = [t for t in inits if t.name in used]

    x = helper.make_tensor_value_info("input", F32, [1, 10, 30, 30])
    y = helper.make_tensor_value_info("output", BOOL, [1, 10, 30, 30])
    graph = helper.make_graph(nodes, "task080", [x], [y], inits)
    return helper.make_model(graph, ir_version=IR_VERSION,
                             opset_imports=[helper.make_opsetid("", 11)])
