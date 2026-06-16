"""Task 213 (ARC-AGI 8e1813be): collapse colored stripes into an n x n block.

Rule: the input has n equally-spaced solid stripes (every 3rd line, starting at
row `offset`), each a distinct color drawn from {1,2,3,4,6,7,8,9} (never gray=5,
never black=0). A (n+2)x(n+2) gray(5)/black(0) box marker is overlaid somewhere
(pure distractor). The output is an n x n grid where, for xpose=0, output row r
is entirely the r-th stripe color (top-to-bottom); under xpose=1 the stripes are
vertical and the output is transposed so column c is the c-th stripe color.

Because stripe colors exclude 0 and 5, the box never contributes a stripe color.

LEAN ENCODING (the only [1,10,30,30] tensors are two BOOL And planes; the
10-channel expansion is routed into the FREE output as the final Or):
  colf[1,1,30,30] = sum_k k*input_k with ch0,ch5 weighted 0 (1x1 Conv)  -> masked
                    colour-index plane (stripe colour, 0 at box/gaps/background).
  per line (axis 3 -> per-row, axis 2 -> per-col):
    linecolor[30,1] = ReduceMax(colf, perp-axis)   (colour, 0 on gap lines)
    iss             = linecolor > 0   ; K = sum(iss) = #nonzero lines
    COMPACTION via geometry, NOT a [30,30] matrix: stripes sit at exactly
      offset, offset+3, ...  so offset = ReduceMin(rowidx + (1-iss)*BIG) and
      outcolor[i] = Gather(linecolor, clip(offset + 3*i, 0, 29)).
  ORIENTATION without per-colour counts: in horiz #nonzero rows = n but every
  column crosses every stripe so #nonzero cols ~= W -> K_r < K_c; in vert it
  flips. So horiz = (K_r < K_c) and the output size n = min(K_r, K_c) (= the
  smaller K, taken via Where to avoid fp16 Min). Select outcolor by orientation.
  ONE-HOT via integer Equal (opset-11 Equal supports int32, rejects float):
    chot[1,10,30,1] = Equal(int(outcolor), arange_ch) AND outcolor>0 AND slot<n.
  output[k,i,c] = Or( chot_row AND (c<n & horiz), chot_col AND (r<n & vert) ).
  Each And is [1,10,30,30] BOOL; the wrong-orientation plane is all-False (its
  perp mask is gated off) so the final Or selects the right one into output.

All math integer-valued, float32-exact (ints < 2^24).
"""

import numpy as np
from onnx import helper, numpy_helper, TensorProto

from ..builders import _model

F32 = TensorProto.FLOAT
BOOL = TensorProto.BOOL
IR_VERSION = 7


def build(task):
    inits = []
    nodes = []
    vinfos = []

    def init(name, arr, dtype=np.float32):
        inits.append(numpy_helper.from_array(
            np.ascontiguousarray(arr, dtype=dtype), name))
        return name

    def n(op, inputs, out, **attrs):
        nodes.append(helper.make_node(op, inputs, [out], **attrs))
        return out

    def vi(name, dtype, shape):
        vinfos.append(helper.make_tensor_value_info(name, dtype, shape))
        return name

    init("one", np.array(1.0, np.float32))
    init("zero", np.array(0.0, np.float32))

    # 1x1 conv weight: out[1,1] = sum_k k*input_k, with ch0,ch5 weight 0.
    w = np.zeros((1, 10, 1, 1), np.float32)
    for k in range(10):
        if k != 0 and k != 5:
            w[0, k, 0, 0] = float(k)
    init("convw", w)

    init("big", np.array(1000.0, np.float32))
    init("rowidx", np.arange(30, dtype=np.float32).reshape(30, 1))   # [30,1]
    init("step3", (3.0 * np.arange(30, dtype=np.float32)).reshape(30, 1))  # [30,1]

    arange_row = np.arange(30, dtype=np.float32).reshape(1, 1, 1, 30)
    init("ar_row", arange_row)                   # [1,1,1,30]
    arange_col = np.arange(30, dtype=np.float32).reshape(1, 1, 30, 1)
    init("ar_col", arange_col)                   # [1,1,30,1]

    init("ar_ch_i", np.arange(10).reshape(1, 10, 1, 1), np.int32)  # int32 channels

    # masked color-index plane
    n("Conv", ["input", "convw"], "colf")
    vi("colf", F32, [1, 1, 30, 30])

    def line_pipeline(tag, reduce_axis):
        # reduce_axis=3 -> per-row (vertical stripes? no: horizontal stripes occupy
        #   a row, reduce over cols=axis3 gives per-row color). line index = rows.
        if reduce_axis == 3:
            line_sh = [1, 1, 30, 1]
        else:
            line_sh = [1, 1, 1, 30]
        # per-line color (max over the perpendicular axis)
        n("ReduceMax", ["colf"], f"lc_{tag}", axes=[reduce_axis], keepdims=1)
        vi(f"lc_{tag}", F32, line_sh)
        # reshape to [30,1]
        init(f"sh301_{tag}", np.array([30, 1], np.int64), np.int64)
        n("Reshape", [f"lc_{tag}", f"sh301_{tag}"], f"lcv_{tag}")
        vi(f"lcv_{tag}", F32, [30, 1])
        # isstripe = color > 0  -> {0,1} float
        n("Greater", [f"lcv_{tag}", "zero"], f"issb_{tag}")
        vi(f"issb_{tag}", BOOL, [30, 1])
        n("Cast", [f"issb_{tag}"], f"iss_{tag}", to=F32)
        vi(f"iss_{tag}", F32, [30, 1])
        # K = total stripe count
        n("ReduceSum", [f"iss_{tag}"], f"K_{tag}", axes=[0, 1], keepdims=1)
        vi(f"K_{tag}", F32, [1, 1])
        # Stripes lie at rows offset, offset+3, offset+6, ... (fixed step 3).
        # offset = first row with a stripe = ReduceMin( row_idx + (1-iss)*BIG ).
        #   gap rows get a huge value so the min lands on the first stripe row.
        n("Sub", ["one", f"iss_{tag}"], f"gap_{tag}")            # [30,1]
        vi(f"gap_{tag}", F32, [30, 1])
        n("Mul", [f"gap_{tag}", "big"], f"gapb_{tag}")          # [30,1]
        vi(f"gapb_{tag}", F32, [30, 1])
        n("Add", [f"gapb_{tag}", "rowidx"], f"cand_{tag}")      # [30,1]
        vi(f"cand_{tag}", F32, [30, 1])
        n("ReduceMin", [f"cand_{tag}"], f"off_{tag}", axes=[0, 1], keepdims=1)
        vi(f"off_{tag}", F32, [1, 1])                            # offset scalar
        # gather indices = clip(offset + 3*arange(30), 0, 29)
        n("Add", ["step3", f"off_{tag}"], f"gidxf_{tag}")        # [30,1]+[1,1]->[30,1]
        vi(f"gidxf_{tag}", F32, [30, 1])
        n("Clip", [f"gidxf_{tag}"], f"gidxc_{tag}", min=0.0, max=29.0)
        vi(f"gidxc_{tag}", F32, [30, 1])
        n("Cast", [f"gidxc_{tag}"], f"gidx_{tag}", to=TensorProto.INT32)
        vi(f"gidx_{tag}", TensorProto.INT32, [30, 1])
        # squeeze to [30] index vector for Gather over axis 0 of lcv [30,1]
        init(f"sh30_{tag}", np.array([30], np.int64), np.int64)
        n("Reshape", [f"gidx_{tag}", f"sh30_{tag}"], f"gidxv_{tag}")
        vi(f"gidxv_{tag}", TensorProto.INT32, [30])
        n("Gather", [f"lcv_{tag}", f"gidxv_{tag}"], f"oc_{tag}", axis=0)  # [30,1]
        vi(f"oc_{tag}", F32, [30, 1])
        return f"oc_{tag}", f"K_{tag}"

    # horizontal stripes: each stripe is a full ROW -> per-row colour (reduce cols)
    #   In horiz: #nonzero rows = n (stripes); #nonzero cols ~= W (every column
    #   crosses every stripe). So K_r = n, K_c ~= W, and K_r < K_c. In vert it
    #   flips. Orientation is therefore just K_r < K_c, and n = min(K_r, K_c).
    oc_r, K_r = line_pipeline("r", reduce_axis=3)
    oc_c, K_c = line_pipeline("c", reduce_axis=2)

    # horiz = K_r < K_c  (fewer nonzero rows than nonzero cols)
    n("Less", [K_r, K_c], "horizb")                 # [1,1] bool
    vi("horizb", TensorProto.BOOL, [1, 1])
    init("sh1", np.array([1], np.int64), np.int64)
    n("Reshape", ["horizb", "sh1"], "horiz_s")
    vi("horiz_s", TensorProto.BOOL, [1])

    # selected outcolor [30,1] and output size n = min(K_r, K_c) (Min crashes
    # fp16 under DISABLE_ALL but these are fp32 scalars; use Where to avoid Min).
    n("Where", ["horiz_s", oc_r, oc_c], "outcolor")
    vi("outcolor", F32, [30, 1])
    n("Where", ["horizb", K_r, K_c], "Ksel")        # min(K_r,K_c)=n  [1,1]
    vi("Ksel", F32, [1, 1])
    n("Reshape", ["Ksel", "sh1"], "Kscalar")        # [1]
    vi("Kscalar", F32, [1])

    # ---- build the output one-hot ----
    # horizontal layout: output row i solid color outcolor[i], cols 0..K-1.
    #   colorhot_r[1,10,30,1] = (outcolor[i]==k) AND (outcolor[i]>0)
    #   output[k,i,c] = colorhot_r[k,i] AND (c < K)
    # one-hot per output slot, built once: ehot[k,i] = (outcolor[i]==k) AND
    # (outcolor[i]>0).  outcolor is [30,1] indexed by output slot i.
    #   eqf = 1 - Clip(|outcolor - k|, 0, 1)  (float eq, opset-11 Equal rejects float)
    init("sh_oc_row", np.array([1, 1, 30, 1], np.int64), np.int64)
    n("Reshape", ["outcolor", "sh_oc_row"], "oc_slot")     # [1,1,30,1]
    vi("oc_slot", F32, [1, 1, 30, 1])
    # one-hot via integer Equal (opset-11 Equal supports int32, rejects float).
    n("Cast", ["oc_slot"], "oc_int", to=TensorProto.INT32)  # [1,1,30,1]
    vi("oc_int", TensorProto.INT32, [1, 1, 30, 1])
    n("Equal", ["oc_int", "ar_ch_i"], "ehot")              # [1,10,30,1] bool
    vi("ehot", BOOL, [1, 10, 30, 1])
    n("Greater", ["oc_slot", "zero"], "valid0")            # [1,1,30,1] bool
    vi("valid0", BOOL, [1, 1, 30, 1])
    # slot index (dim2) must be < K: over-clipped Gather slots i>=K may pick up a
    # real stripe colour at the clamped row, so gate them out explicitly.
    n("Less", ["ar_col", "Kscalar"], "slotvalid")          # [1,1,30,1] (i<K)
    vi("slotvalid", BOOL, [1, 1, 30, 1])
    n("And", ["valid0", "slotvalid"], "valid")             # [1,1,30,1]
    vi("valid", BOOL, [1, 1, 30, 1])
    n("And", ["ehot", "valid"], "chot")                    # [1,10,30,1] bool
    vi("chot", BOOL, [1, 10, 30, 1])

    # perpendicular validity (slot < K) for both axes, gated by orientation so
    # the wrong-orientation plane is all-False and the final Or selects cleanly.
    n("Not", ["horiz_s"], "vert_s")                        # [1] bool
    vi("vert_s", BOOL, [1])
    n("Less", ["ar_row", "Kscalar"], "perp_col0")          # [1,1,1,30] (c<K)
    vi("perp_col0", BOOL, [1, 1, 1, 30])
    n("And", ["perp_col0", "horiz_s"], "perp_col")         # gate by horiz
    vi("perp_col", BOOL, [1, 1, 1, 30])
    n("Less", ["ar_col", "Kscalar"], "perp_row0")          # [1,1,30,1] (r<K)
    vi("perp_row0", BOOL, [1, 1, 30, 1])
    n("And", ["perp_row0", "vert_s"], "perp_row")          # gate by vert
    vi("perp_row", BOOL, [1, 1, 30, 1])

    # horiz: slot=row -> chot is [1,10,30,1] as-is; broadcast over cols<K.
    n("And", ["chot", "perp_col"], "out_horiz")            # [1,10,30,30] bool
    vi("out_horiz", BOOL, [1, 10, 30, 30])
    # vert: slot=col -> reshape chot to [1,10,1,30]; broadcast over rows<K.
    init("sh_chot_col", np.array([1, 10, 1, 30], np.int64), np.int64)
    n("Reshape", ["chot", "sh_chot_col"], "chot_c")        # [1,10,1,30] bool
    vi("chot_c", BOOL, [1, 10, 1, 30])
    n("And", ["chot_c", "perp_row"], "out_vert")           # [1,10,30,30] bool
    vi("out_vert", BOOL, [1, 10, 30, 30])

    # exactly one plane is non-empty -> Or selects it (into the FREE output)
    n("Or", ["out_horiz", "out_vert"], "output")

    model = _model(nodes, inits, vinfos)
    # declare output as BOOL
    model.graph.output[0].type.tensor_type.elem_type = BOOL
    return model
