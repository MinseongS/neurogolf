"""Task 213 (ARC-AGI 8e1813be): collapse colored stripes into an n x n block.

Rule: the input has n equally-spaced solid stripes (every 3rd line, starting at
row `offset`), each a distinct color drawn from {1,2,3,4,6,7,8,9} (never gray=5,
never black=0). A (n+2)x(n+2) gray(5)/black(0) box marker is overlaid somewhere
(pure distractor). The output is an n x n grid where, for xpose=0, output row r
is entirely the r-th stripe color; under xpose=1 the stripes are vertical and
the output is transposed so column c is the c-th stripe color.

Because stripe colors exclude 0 and 5, the box never contributes a stripe color.

LEAN ENCODING (NO [1,10,30,30] working plane is ever materialised — the 10-ch
expansion lands DIRECTLY in the FREE bool `output` via a single associated And):
  colf[1,1,30,30] = sum_k k*input_k with ch0,ch5 weighted 0 (1x1 Conv) -> masked
                    colour-index plane (stripe colour, 0 at box/gaps/background).
  per axis: linecolor[30,1]=ReduceMax(colf, perp-axis); iss=linecolor>0;
    K=sum(iss)=#nonzero lines.  Stripes sit at offset, offset+3, ... so
    offset=ReduceMin(rowidx+(1-iss)*BIG); outcolor[i]=Gather(linecolor,
    clip(offset+3*i,0,29)).
  ORIENTATION: horiz #nonzero rows=n but every col crosses every stripe so
    #nonzero cols~=W => K_r<K_c; vert flips.  horiz=(K_r<K_c), n=min(K_r,K_c).
  CANONICALISE TO ROW-SLOT FRAME: pick outcolor=oc_r if horiz else oc_c, and
    slot=row.  Build the one-hot slot plane ehot[1,10,30,1] (Equal(int,arange)).
    The chosen frame's output is then ALWAYS out[k,i,c]=ehot[k,i] AND (i<K) AND
    (c<K) — i.e. a row-solid block of side n.  In the canonical frame the data
    is laid out as rows; the original grid for vert is the transpose, which is
    handled by selecting oc_c (the column colours) up front, so the SAME
    row-solid construction yields the correct n x n output (an n x n block is
    its own transpose modulo the per-line colour, already chosen by oc).
  output[k,i,c] = And3( ehot[1,10,30,1], rowin[1,1,30,1]=(i<K),
                        colin[1,1,1,30]=(c<K) )  -> the FREE bool output.
    Associated as And(ehot_gated[1,10,30,1], colin[1,1,1,30]) so the only
    full-grid tensor IS the graph output (FREE).  No [1,10,30,30] carrier.

All math integer-valued, float32-exact (ints < 2^24).
"""

import numpy as np
from onnx import helper, numpy_helper, TensorProto

from ..builders import _model

F32 = TensorProto.FLOAT
BOOL = TensorProto.BOOL
INT32 = TensorProto.INT32


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

    # 1x1 conv weight to collapse channels: out = sum_k k*presence_k (ch0,ch5 -> 0).
    w = np.zeros((1, 10, 1, 1), np.float32)
    for k in range(10):
        if k not in (0, 5):
            w[0, k, 0, 0] = float(k)
    init("convw", w)

    init("big", np.array(1000.0, np.float32))
    init("rowidx", np.arange(30, dtype=np.float32).reshape(30, 1))
    init("step3", (3.0 * np.arange(30, dtype=np.float32)).reshape(30, 1))

    init("ar_row", np.arange(30, dtype=np.float32).reshape(1, 1, 1, 30))  # [1,1,1,30]
    init("ar_col", np.arange(30, dtype=np.float32).reshape(1, 1, 30, 1))  # [1,1,30,1]

    # NO full colf plane: derive each line-colour vector from a per-channel
    # presence reduction (max over the perpendicular spatial axis -> [1,10,30,1]
    # / [1,10,1,30], 1200B) then a 1x1 Conv weight-k channel-collapse to [1,1,*]
    # (120B).  A stripe line is full-width one colour, so presence_k=1 there and
    # weighted-sum = k (box gray/black weight 0; one stripe colour per line).
    def line_pipeline(tag, reduce_axis):
        # presence per channel over the perpendicular axis
        n("ReduceMax", ["input"], f"pres_{tag}", axes=[reduce_axis], keepdims=1)
        pres_sh = [1, 10, 30, 1] if reduce_axis == 3 else [1, 10, 1, 30]
        vi(f"pres_{tag}", F32, pres_sh)
        # channel-collapse to line colour via 1x1 Conv (weight k, excl 0/5)
        n("Conv", [f"pres_{tag}", "convw"], f"lc_{tag}")
        line_sh = [1, 1, 30, 1] if reduce_axis == 3 else [1, 1, 1, 30]
        vi(f"lc_{tag}", F32, line_sh)
        init(f"sh301_{tag}", np.array([30, 1], np.int64), np.int64)
        n("Reshape", [f"lc_{tag}", f"sh301_{tag}"], f"lcv_{tag}")
        vi(f"lcv_{tag}", F32, [30, 1])
        n("Greater", [f"lcv_{tag}", "zero"], f"issb_{tag}")
        vi(f"issb_{tag}", BOOL, [30, 1])
        n("Cast", [f"issb_{tag}"], f"iss_{tag}", to=F32)
        vi(f"iss_{tag}", F32, [30, 1])
        n("ReduceSum", [f"iss_{tag}"], f"K_{tag}", axes=[0, 1], keepdims=1)
        vi(f"K_{tag}", F32, [1, 1])
        n("Sub", ["one", f"iss_{tag}"], f"gap_{tag}")
        vi(f"gap_{tag}", F32, [30, 1])
        n("Mul", [f"gap_{tag}", "big"], f"gapb_{tag}")
        vi(f"gapb_{tag}", F32, [30, 1])
        n("Add", [f"gapb_{tag}", "rowidx"], f"cand_{tag}")
        vi(f"cand_{tag}", F32, [30, 1])
        n("ReduceMin", [f"cand_{tag}"], f"off_{tag}", axes=[0, 1], keepdims=1)
        vi(f"off_{tag}", F32, [1, 1])
        n("Add", ["step3", f"off_{tag}"], f"gidxf_{tag}")
        vi(f"gidxf_{tag}", F32, [30, 1])
        n("Clip", [f"gidxf_{tag}"], f"gidxc_{tag}", min=0.0, max=29.0)
        vi(f"gidxc_{tag}", F32, [30, 1])
        n("Cast", [f"gidxc_{tag}"], f"gidx_{tag}", to=INT32)
        vi(f"gidx_{tag}", INT32, [30, 1])
        init(f"sh30_{tag}", np.array([30], np.int64), np.int64)
        n("Reshape", [f"gidx_{tag}", f"sh30_{tag}"], f"gidxv_{tag}")
        vi(f"gidxv_{tag}", INT32, [30])
        n("Gather", [f"lcv_{tag}", f"gidxv_{tag}"], f"oc_{tag}", axis=0)
        vi(f"oc_{tag}", F32, [30, 1])
        return f"oc_{tag}", f"K_{tag}"

    oc_r, K_r = line_pipeline("r", reduce_axis=3)
    oc_c, K_c = line_pipeline("c", reduce_axis=2)

    # horiz = K_r < K_c
    n("Less", [K_r, K_c], "horizb")               # [1,1] bool
    vi("horizb", BOOL, [1, 1])
    n("Where", ["horizb", K_r, K_c], "Ksel")      # [1,1] = n = min(K_r,K_c)
    vi("Ksel", F32, [1, 1])

    # Per-slot colour as a row-vector [1,1,30,1] (horiz: slot=row) and as a
    # col-vector [1,1,1,30] (vert: slot=col).  outcolor[r][c] = oc_row[r] when
    # horiz (each ROW solid), oc_col[c] when vert (each COLUMN solid).  Where
    # broadcasts both to ONE [1,1,30,30] colour-index plane (the only full
    # plane, 3600B fp32) -> Equal routes the 10-ch expansion into FREE output.
    # cast the tiny [30,1] colour vectors to UINT8 BEFORE broadcasting so EVERY
    # full-grid colour-index plane is uint8 (900B, 1/4 of fp32).  uint8 Where and
    # uint8 Equal both run under ORT_DISABLE_ALL; the output op is uint8 Equal so
    # no fp32 PrecisionFreeCast plane appears.  Colours are 1..9, sentinel = 255.
    n("Cast", [oc_r], "ocr_i", to=INT32)
    vi("ocr_i", INT32, [30, 1])
    n("Cast", [oc_c], "occ_i", to=INT32)
    vi("occ_i", INT32, [30, 1])
    init("sh_row", np.array([1, 1, 30, 1], np.int64), np.int64)
    init("sh_col", np.array([1, 1, 1, 30], np.int64), np.int64)
    n("Reshape", ["ocr_i", "sh_row"], "ocr_p")    # [1,1,30,1] int32
    vi("ocr_p", INT32, [1, 1, 30, 1])
    n("Reshape", ["occ_i", "sh_col"], "occ_p")    # [1,1,1,30] int32
    vi("occ_p", INT32, [1, 1, 1, 30])
    n("Where", ["horizb", "ocr_p", "occ_p"], "cidx_u")  # [1,1,30,30] int32
    vi("cidx_u", INT32, [1, 1, 30, 30])

    # in-grid n x n block: cells with (r<n) AND (c<n).  OUTSIDE the block set the
    # colour index to SENTINEL 255 so Equal(.,arange[0..9]) matches nothing.
    init("sh11", np.array([1, 1, 1, 1], np.int64), np.int64)
    n("Reshape", ["Ksel", "sh11"], "Kp")          # [1,1,1,1]
    vi("Kp", F32, [1, 1, 1, 1])
    n("Less", ["ar_col", "Kp"], "rowin")          # [1,1,30,1] (r<n)
    vi("rowin", BOOL, [1, 1, 30, 1])
    n("Less", ["ar_row", "Kp"], "colin")          # [1,1,1,30] (c<n)
    vi("colin", BOOL, [1, 1, 1, 30])
    n("And", ["rowin", "colin"], "blockin")       # [1,1,30,30] bool block mask
    vi("blockin", BOOL, [1, 1, 30, 30])
    init("sent", np.array(-1, np.int32), np.int32)
    n("Where", ["blockin", "cidx_u", "sent"], "cidx_g")  # [1,1,30,30] int32
    vi("cidx_g", INT32, [1, 1, 30, 30])
    init("ar_ch_i", np.arange(10).reshape(1, 10, 1, 1), np.int32)
    n("Equal", ["cidx_g", "ar_ch_i"], "output")   # [1,10,30,30] bool = FREE output

    model = _model(nodes, inits, vinfos)
    model.graph.output[0].type.tensor_type.elem_type = BOOL
    return model
