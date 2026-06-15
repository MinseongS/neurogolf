"""task131 (ARC-AGI 56dc2b01) — move green creature adjacent to red line + cyan marker.

Rule (recovered EXACTLY on all 266 generator examples):
  The grid has exactly ONE full red(2) line (a complete column OR a complete row)
  and a green(3) "creature" on one side of it.  Output:
    * the red line stays put;
    * the green creature is TRANSLATED (shape preserved) perpendicular to the line
      so its bounding-box edge nearest the line sits exactly ONE cell from the line;
    * a full CYAN(8) line is drawn one cell beyond the creature's FAR bbox edge;
    * the original green is erased.

  Geometry (red column rc, green col-bbox [cmin,cmax]):
    green left  of red (cmax<rc): shift s = rc-1-cmax ; cyan col = cmin+s-1
    green right of red (cmin>rc): shift s = rc+1-cmin ; cyan col = cmax+s+1
  Horizontal-red is the transpose of the vertical case.

Floor-break encoding (single canonical branch + final Equal; no [1,10,30,30] tensor):
  We extract green/red/in-grid planes with 1x1 Convs (free input -> [1,1,30,30]).
  We pick a CANONICAL orientation in which the red line is a COLUMN: if `input`
  already has a red column we use the planes as-is, else we use their transpose
  (a red row becomes a red column).  We run ONE vertical-rule subgraph producing a
  uint8 label map L[30,30], then transpose L back where we transposed the input.
  The data-dependent column shift of the green plane is done with a Gather using a
  computed index vector idx[c'] = c'-s.  Final free BOOL output = Equal(L,arange).
  All values are small integers, exact in float32 / uint8.
"""

import numpy as np
import onnx
from onnx import helper, numpy_helper, TensorProto

from ..harness import IR_VERSION

F = TensorProto.FLOAT
U8 = TensorProto.UINT8
I64 = TensorProto.INT64
B = TensorProto.BOOL
N = 30


def build(task):
    inits, nodes, _seen = [], [], set()

    def init(name, arr, dtype):
        if name in _seen:
            return name
        _seen.add(name)
        inits.append(numpy_helper.from_array(
            np.ascontiguousarray(arr, dtype=dtype), name))
        return name

    def n(op, ins, out, **attrs):
        nodes.append(helper.make_node(op, ins, [out], **attrs))
        return out

    init("colvec", np.arange(N, dtype=np.float32).reshape(1, 1, 1, N), np.float32)  # [1,1,1,30]
    init("idxbase", np.arange(N, dtype=np.float32), np.float32)                      # [30]
    init("zero", np.array(0.0, np.float32), np.float32)
    init("one", np.array(1.0, np.float32), np.float32)
    init("half", np.array(0.5, np.float32), np.float32)
    init("big", np.array(1000.0, np.float32), np.float32)
    init("s11", np.array([1, 1], np.int64), np.int64)

    # --- extract green / red / in-grid planes ONCE (no [1,10,30,30] tensor) --
    gw = np.zeros((1, 10, 1, 1), np.float32); gw[0, 3, 0, 0] = 1.0
    rw = np.zeros((1, 10, 1, 1), np.float32); rw[0, 2, 0, 0] = 1.0
    init("gw", gw, np.float32)
    init("rw", rw, np.float32)
    n("Conv", ["input", "gw"], "Gp")   # [1,1,30,30] green
    n("Conv", ["input", "rw"], "Rp")   # [1,1,30,30] red
    n("ReduceMax", ["input"], "IGp", axes=[1], keepdims=1)  # [1,1,30,30] in-grid

    # --- decide canonical orientation: does `input` have a full red COLUMN? ---
    # red column c: colR[c] == colIG[c] and colR[c] > 0   (full red over in-grid rows)
    n("ReduceSum", ["Rp"], "colR0", axes=[2], keepdims=1)    # [1,1,1,30]
    n("ReduceSum", ["IGp"], "colIG0", axes=[2], keepdims=1)  # [1,1,1,30]
    n("Greater", ["colR0", "zero"], "rpos0")
    n("Sub", ["colR0", "colIG0"], "rd0"); n("Abs", ["rd0"], "rda0")
    n("Less", ["rda0", "half"], "req0")
    n("And", ["rpos0", "req0"], "rc0_b")
    n("Cast", ["rc0_b"], "rc0f", to=F)
    n("ReduceMax", ["rc0f"], "hasrcf", axes=[1, 2, 3], keepdims=1)  # [1,1,1,1]
    n("Reshape", ["hasrcf", "s11"], "hasrc11")
    n("Greater", ["hasrc11", "half"], "sel11")  # [1,1] bool

    # --- transposed planes (red row -> red column) --------------------------
    n("Transpose", ["Gp"], "GpT", perm=[0, 1, 3, 2])
    n("Transpose", ["Rp"], "RpT", perm=[0, 1, 3, 2])
    n("Transpose", ["IGp"], "IGpT", perm=[0, 1, 3, 2])
    # squeeze all to [30,30] then select canonical planes
    for nm in ["Gp", "Rp", "IGp", "GpT", "RpT", "IGpT"]:
        n("Squeeze", [nm], nm + "2", axes=[0, 1])  # [30,30]
    n("Where", ["sel11", "Gp2", "GpT2"], "G")    # [30,30] canonical green
    n("Where", ["sel11", "Rp2", "RpT2"], "R")    # canonical red
    n("Where", ["sel11", "IGp2", "IGpT2"], "IG") # canonical in-grid

    # --- vertical-rule on canonical [30,30] planes --------------------------
    # column aggregates (reduce over rows = axis 0)
    n("ReduceSum", ["R"], "colR", axes=[0], keepdims=1)    # [1,30]
    n("ReduceSum", ["IG"], "colIG", axes=[0], keepdims=1)  # [1,30]
    n("ReduceMax", ["G"], "gcol", axes=[0], keepdims=1)    # [1,30] green col presence
    init("colrow", np.arange(N, dtype=np.float32).reshape(1, N), np.float32)  # [1,30]

    # red column rc (exactly one)
    n("Greater", ["colR", "zero"], "rpos")
    n("Sub", ["colR", "colIG"], "rdc"); n("Abs", ["rdc"], "rdca")
    n("Less", ["rdca", "half"], "req")
    n("And", ["rpos", "req"], "redcol_b")
    n("Cast", ["redcol_b"], "redcol", to=F)
    n("Mul", ["redcol", "colrow"], "rcw")
    n("ReduceSum", ["rcw"], "rc", axes=[1], keepdims=1)  # [1,1]

    # green bbox cmin / cmax
    n("Mul", ["gcol", "colrow"], "cmaxw")
    n("ReduceMax", ["cmaxw"], "cmax", axes=[1], keepdims=1)  # [1,1]
    n("Sub", ["one", "gcol"], "nog"); n("Mul", ["nog", "big"], "pen")
    n("Add", ["cmaxw", "pen"], "cminw")
    n("ReduceMin", ["cminw"], "cmin", axes=[1], keepdims=1)  # [1,1]

    # direction + shift + cyan column
    n("Less", ["cmax", "rc"], "left_b")
    n("Cast", ["left_b"], "left", to=F)
    n("Sub", ["one", "left"], "right")
    n("Sub", ["rc", "one"], "rcm1"); n("Sub", ["rcm1", "cmax"], "sL")
    n("Add", ["rc", "one"], "rcp1"); n("Sub", ["rcp1", "cmin"], "sR")
    n("Mul", ["sL", "left"], "sLm"); n("Mul", ["sR", "right"], "sRm")
    n("Add", ["sLm", "sRm"], "s")  # [1,1] shift
    n("Add", ["cmin", "s"], "cycL0"); n("Sub", ["cycL0", "one"], "cycL")
    n("Add", ["cmax", "s"], "cycR0"); n("Add", ["cycR0", "one"], "cycR")
    n("Mul", ["cycL", "left"], "cycLm"); n("Mul", ["cycR", "right"], "cycRm")
    n("Add", ["cycLm", "cycRm"], "cyc")  # [1,1]

    # --- shift green plane by s along columns via Gather --------------------
    # idx[c'] = c' - s  (clamped to [0,N-1]); Gather columns of G.
    n("Reshape", ["s", "s11"], "s2")          # [1,1]
    n("Squeeze", ["s2"], "ss", axes=[0, 1])   # scalar
    n("Sub", ["idxbase", "ss"], "idxf")       # [30]
    init("zerof1", np.array(0.0, np.float32), np.float32)
    init("maxf1", np.array(float(N - 1), np.float32), np.float32)
    n("Clip", ["idxf", "zerof1", "maxf1"], "idxc")
    n("Cast", ["idxc"], "idx", to=I64)        # [30]
    # Gather along columns (axis 1) of G[30,30]
    n("Gather", ["G", "idx"], "Gs", axis=1)   # [30,30] columns picked
    # but Gather picks G[:, idx[c']] = G[:, c'-s]; this places original col c into c+s. correct.
    n("Greater", ["Gs", "half"], "Gs_b")      # [30,30] bool
    # Guard against clamp duplication at borders: a shifted-in column is valid only
    # if its source idx was within range, i.e. 0 <= c'-s <= N-1.
    init("neghalf", np.array(-0.5, np.float32), np.float32)
    init("nmhalf", np.array(float(N) - 0.5, np.float32), np.float32)
    n("Greater", ["idxf", "neghalf"], "ge0")
    n("Less", ["idxf", "nmhalf"], "le0")
    n("And", ["ge0", "le0"], "valid")         # [30]
    n("And", ["Gs_b", "valid"], "Gfin_b")     # broadcast [30,30] & [30]

    # --- cyan full column at cyc, only in-grid ------------------------------
    n("Sub", ["colrow", "cyc"], "cyd"); n("Abs", ["cyd"], "cyda")
    n("Less", ["cyda", "half"], "cymask_b")   # [1,30]
    n("Cast", ["cymask_b"], "cymask", to=F)
    n("Mul", ["cymask", "IG"], "cyplane")     # [30,30] (broadcast row)
    n("Greater", ["cyplane", "half"], "cy_b")

    n("Greater", ["R", "half"], "R_b")
    n("Greater", ["IG", "half"], "IG_b")

    # --- assemble label map -------------------------------------------------
    init("u0", np.array(0, np.uint8), np.uint8)
    init("u2", np.array(2, np.uint8), np.uint8)
    init("u3", np.array(3, np.uint8), np.uint8)
    init("u8", np.array(8, np.uint8), np.uint8)
    init("u10", np.array(10, np.uint8), np.uint8)
    n("Where", ["IG_b", "u0", "u10"], "L0")   # 0 in-grid else 10
    n("Where", ["cy_b", "u8", "L0"], "L1")
    n("Where", ["Gfin_b", "u3", "L1"], "L2")
    n("Where", ["R_b", "u2", "L2"], "Lcanon")  # [30,30] in canonical orientation

    # --- un-transpose where we used the transposed input --------------------
    n("Transpose", ["Lcanon"], "LcanonT", perm=[1, 0])
    n("Where", ["sel11", "Lcanon", "LcanonT"], "Lsel")  # [30,30]
    init("Lshape", np.array([1, 1, N, N], np.int64), np.int64)
    n("Reshape", ["Lsel", "Lshape"], "L")  # [1,1,30,30]

    chan = np.arange(10, dtype=np.uint8).reshape(1, 10, 1, 1)
    init("chan", chan, np.uint8)
    n("Equal", ["L", "chan"], "output")  # [1,10,30,30] BOOL

    x = helper.make_tensor_value_info("input", F, [1, 10, N, N])
    y = helper.make_tensor_value_info("output", B, [1, 10, N, N])
    g = helper.make_graph(nodes, "task131", [x], [y], inits)
    return helper.make_model(g, ir_version=IR_VERSION,
                             opset_imports=[helper.make_opsetid("", 11)])
