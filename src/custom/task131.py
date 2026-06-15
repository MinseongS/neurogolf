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
  Three planes (green/red/in-grid) are made with 1x1 Convs / ReduceMax from the free
  `input` ([1,1,30,30] float each).  We compute the rc / cmin / cmax / shift / cyan
  scalars from cheap 1-D row AND column aggregates of those planes, and pick the set
  belonging to whichever orientation has a full red line (column => vertical).
  The 2-D label assembly is done on uint8 planes in a CANONICAL orientation (red
  line vertical): green/red/in-grid planes are cast to uint8, transposed, and
  selected by orientation.  The green column shift is a Gather with idx[c']=c'-s.
  The uint8 label L[30,30] is transposed back where we transposed, reshaped to
  [1,1,30,30], and the free BOOL output = Equal(L, arange).  All values are small
  integers, exact in float32 / uint8.
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

    M = 20  # working canvas for 2-D ops (grids are <=18; redline/cyan/green all < 20)
    init("idxbase", np.arange(M, dtype=np.float32), np.float32)             # [M]
    init("zero", np.array(0.0, np.float32), np.float32)
    init("one", np.array(1.0, np.float32), np.float32)
    init("half", np.array(0.5, np.float32), np.float32)
    init("big", np.array(1000.0, np.float32), np.float32)
    init("neghalf", np.array(-0.5, np.float32), np.float32)
    init("nmhalf", np.array(float(N) - 0.5, np.float32), np.float32)
    init("s11", np.array([1, 1], np.int64), np.int64)


    # ===== 1-D aggregates via wide Convs (no 2-D red / in-grid plane) =======
    # column aggregate kernel [1,10,30,1] (sum over 30 rows) -> [1,1,1,30].
    # row    aggregate kernel [1,10,1,30] (sum over 30 cols) -> [1,1,30,1].
    def aggker(over_rows, channels):
        # weight 1 on the given channels, summed over the long spatial axis
        if over_rows:
            w = np.zeros((1, 10, N, 1), np.float32)
            for c in channels:
                w[0, c, :, 0] = 1.0
        else:
            w = np.zeros((1, 10, 1, N), np.float32)
            for c in channels:
                w[0, c, 0, :] = 1.0
        return w

    ALL = list(range(10))
    init("kcolR", aggker(True, [2]), np.float32)    # red, per column
    init("kcolIG", aggker(True, ALL), np.float32)   # in-grid count, per column
    init("kcolG", aggker(True, [3]), np.float32)    # green, per column
    init("krowR", aggker(False, [2]), np.float32)
    init("krowIG", aggker(False, ALL), np.float32)
    init("krowG", aggker(False, [3]), np.float32)

    def aggregates(over_rows, tag):
        kR = "kcolR" if over_rows else "krowR"
        kIG = "kcolIG" if over_rows else "krowIG"
        kG = "kcolG" if over_rows else "krowG"
        n("Conv", ["input", kR], f"{tag}Rc")    # [1,1,1,30] or [1,1,30,1]
        n("Conv", ["input", kIG], f"{tag}IGc")
        n("Conv", ["input", kG], f"{tag}Gc")
        sq = [0, 1, 2] if over_rows else [0, 1, 3]
        n("Squeeze", [f"{tag}Rc"], f"{tag}Rv", axes=sq)    # [30]
        n("Squeeze", [f"{tag}IGc"], f"{tag}IGv", axes=sq)
        n("Squeeze", [f"{tag}Gc"], f"{tag}Gv", axes=sq)
        return f"{tag}Rv", f"{tag}IGv", f"{tag}Gv"

    vR, vIG, vG = aggregates(True, "v")    # vertical-red: per-column aggregates
    hR, hIG, hG = aggregates(False, "h")   # horizontal-red: per-row aggregates

    init("linev", np.arange(N, dtype=np.float32), np.float32)  # [30]

    def scalars(Rv, IGv, Gv, tag):
        # redcol indicator over the 30 line positions
        n("Greater", [Rv, "zero"], f"{tag}rpos")
        n("Sub", [Rv, IGv], f"{tag}rd"); n("Abs", [f"{tag}rd"], f"{tag}rda")
        n("Less", [f"{tag}rda", "half"], f"{tag}req")
        n("And", [f"{tag}rpos", f"{tag}req"], f"{tag}redln_b")  # [30]
        n("Cast", [f"{tag}redln_b"], f"{tag}redln", to=F)
        n("ReduceMax", [f"{tag}redln"], f"{tag}has", axes=[0], keepdims=1)  # [1]
        n("Mul", [f"{tag}redln", "linev"], f"{tag}rcw")
        n("ReduceSum", [f"{tag}rcw"], f"{tag}rc", axes=[0], keepdims=1)  # [1]
        # green bbox (Gv is a count; convert to 0/1 presence)
        n("Greater", [Gv, "zero"], f"{tag}gp_b")
        n("Cast", [f"{tag}gp_b"], f"{tag}gp", to=F)
        n("Mul", [f"{tag}gp", "linev"], f"{tag}cmaxw")
        n("ReduceMax", [f"{tag}cmaxw"], f"{tag}cmax", axes=[0], keepdims=1)
        n("Sub", ["one", f"{tag}gp"], f"{tag}nog"); n("Mul", [f"{tag}nog", "big"], f"{tag}pen")
        n("Add", [f"{tag}cmaxw", f"{tag}pen"], f"{tag}cminw")
        n("ReduceMin", [f"{tag}cminw"], f"{tag}cmin", axes=[0], keepdims=1)
        # direction/shift/cyan
        n("Less", [f"{tag}cmax", f"{tag}rc"], f"{tag}left_b")
        n("Cast", [f"{tag}left_b"], f"{tag}left", to=F)
        n("Sub", ["one", f"{tag}left"], f"{tag}right")
        n("Sub", [f"{tag}rc", "one"], f"{tag}rcm1"); n("Sub", [f"{tag}rcm1", f"{tag}cmax"], f"{tag}sL")
        n("Add", [f"{tag}rc", "one"], f"{tag}rcp1"); n("Sub", [f"{tag}rcp1", f"{tag}cmin"], f"{tag}sR")
        n("Mul", [f"{tag}sL", f"{tag}left"], f"{tag}sLm"); n("Mul", [f"{tag}sR", f"{tag}right"], f"{tag}sRm")
        n("Add", [f"{tag}sLm", f"{tag}sRm"], f"{tag}s")  # [1]
        n("Add", [f"{tag}cmin", f"{tag}s"], f"{tag}cycL0"); n("Sub", [f"{tag}cycL0", "one"], f"{tag}cycL")
        n("Add", [f"{tag}cmax", f"{tag}s"], f"{tag}cycR0"); n("Add", [f"{tag}cycR0", "one"], f"{tag}cycR")
        n("Mul", [f"{tag}cycL", f"{tag}left"], f"{tag}cycLm"); n("Mul", [f"{tag}cycR", f"{tag}right"], f"{tag}cycRm")
        n("Add", [f"{tag}cycLm", f"{tag}cycRm"], f"{tag}cyc")  # [1]
        return f"{tag}has", f"{tag}s", f"{tag}cyc", f"{tag}rc"

    vhas, vs, vcyc, vrc = scalars(vR, vIG, vG, "vs_")
    hhas, hs, hcyc, hrc = scalars(hR, hIG, hG, "hs_")

    # grid extent (original): in-grid present per column (vIGv) / per row (hIGv).
    # W_orig = max in-grid column index +1 ; H_orig = max in-grid row index +1.
    def extent(IGv, nm):
        n("Greater", [IGv, "zero"], nm + "_b"); n("Cast", [nm + "_b"], nm + "f", to=F)
        n("Mul", [nm + "f", "linev"], nm + "w")
        n("ReduceMax", [nm + "w"], nm + "max", axes=[0], keepdims=1)
        n("Add", [nm + "max", "one"], nm)   # [1]
        return nm
    Worig = extent(vIG, "Worig")   # in-grid columns
    Horig = extent(hIG, "Horig")   # in-grid rows

    # orientation: use vertical scalars where input has a full red column, else horizontal
    n("Greater", [vhas, "half"], "usev")  # [1] bool
    n("Where", ["usev", vs, hs], "s")        # chosen shift
    n("Where", ["usev", vcyc, hcyc], "cyc")  # chosen cyan line index
    n("Where", ["usev", vrc, hrc], "rcc")    # canonical red column index
    # canonical grid dims: vertical keeps (H,W); horizontal transposes them.
    n("Where", ["usev", Horig, Worig], "Hc")  # canonical height (rows)
    n("Where", ["usev", Worig, Horig], "Wc")  # canonical width  (cols)

    # ===== canonical GREEN plane (cropped to MxM working canvas) =============
    init("u0", np.array(0, np.uint8), np.uint8)
    # Slice green channel (3) and the MxM working region directly from `input`
    # -> [1,1,M,M] float, no 30x30 intermediate.
    init("sl_starts", np.array([0, 3, 0, 0], np.int64), np.int64)
    init("sl_ends", np.array([1, 4, M, M], np.int64), np.int64)
    init("sl_axes", np.array([0, 1, 2, 3], np.int64), np.int64)
    n("Slice", ["input", "sl_starts", "sl_ends", "sl_axes"], "Gpc")  # [1,1,M,M] float
    n("Cast", ["Gpc"], "Gpu", to=U8)
    n("Squeeze", ["Gpu"], "Gpu2", axes=[0, 1])       # [M,M]
    n("Transpose", ["Gpu2"], "GpuT", perm=[1, 0])
    n("Reshape", ["usev", "s11"], "usev11")
    n("Where", ["usev11", "Gpu2", "GpuT"], "G")      # [M,M] canonical green

    # in-grid (canonical): rows < Hc AND cols < Wc  (outer product of 1-D bounds)
    init("col1", np.arange(M, dtype=np.float32).reshape(1, M), np.float32)  # [1,M]
    init("row1", np.arange(M, dtype=np.float32).reshape(M, 1), np.float32)  # [M,1]
    n("Less", ["col1", "Wc"], "colin_b")    # [1,M]
    n("Less", ["row1", "Hc"], "rowin_b")    # [M,1]
    n("And", ["colin_b", "rowin_b"], "IG_b")  # [M,M]

    # ===== shift canonical green by s via Gather ============================
    n("Reshape", ["s", "s11"], "s2"); n("Squeeze", ["s2"], "ss", axes=[0, 1])  # scalar
    n("Sub", ["idxbase", "ss"], "idxf")            # [M]
    init("zerof", np.array(0.0, np.float32), np.float32)
    init("maxf", np.array(float(M - 1), np.float32), np.float32)
    n("Clip", ["idxf", "zerof", "maxf"], "idxc")
    n("Cast", ["idxc"], "idx", to=I64)
    n("Gather", ["G", "idx"], "Gs", axis=1)        # [M,M] uint8, Gs[r,c']=G[r,c'-s]
    n("Greater", ["Gs", "u0"], "Gs_b")
    n("Greater", ["idxf", "neghalf"], "ge0")
    init("mmhalf", np.array(float(M) - 0.5, np.float32), np.float32)
    n("Less", ["idxf", "mmhalf"], "le0")
    n("And", ["ge0", "le0"], "valid")              # [M]
    n("And", ["Gs_b", "valid"], "Gfin_b")          # [M,M]

    # ===== cyan full column at cyc, red column at rcc (only in-grid) =========
    n("Sub", ["col1", "cyc"], "cyd"); n("Abs", ["cyd"], "cyda")
    n("Less", ["cyda", "half"], "cymask_b")        # [1,M]
    n("And", ["cymask_b", "IG_b"], "cy_b")         # [M,M]
    n("Sub", ["col1", "rcc"], "rdd"); n("Abs", ["rdd"], "rdda")
    n("Less", ["rdda", "half"], "rmask_b")         # [1,M]
    n("And", ["rmask_b", "IG_b"], "R_b")           # [M,M]

    # ===== assemble label map (MxM) =========================================
    init("u2", np.array(2, np.uint8), np.uint8)
    init("u3", np.array(3, np.uint8), np.uint8)
    init("u8", np.array(8, np.uint8), np.uint8)
    init("u10", np.array(10, np.uint8), np.uint8)
    n("Where", ["IG_b", "u0", "u10"], "L0")
    n("Where", ["cy_b", "u8", "L0"], "L1")
    n("Where", ["Gfin_b", "u3", "L1"], "L2")
    n("Where", ["R_b", "u2", "L2"], "Lcanon")      # [M,M]

    n("Transpose", ["Lcanon"], "LcanonT", perm=[1, 0])
    n("Where", ["usev11", "Lcanon", "LcanonT"], "Lsel")  # [M,M]
    # reshape to [1,1,M,M] then Pad to [1,1,N,N] with sentinel 10 (off-grid).
    init("Mshape", np.array([1, 1, M, M], np.int64), np.int64)
    n("Reshape", ["Lsel", "Mshape"], "Lr")  # [1,1,M,M]
    init("padcfg", np.array([0, 0, 0, 0, 0, 0, N - M, N - M], np.int64), np.int64)
    n("Pad", ["Lr", "padcfg", "u10"], "L", mode="constant")  # [1,1,N,N]

    chan = np.arange(10, dtype=np.uint8).reshape(1, 10, 1, 1)
    init("chan", chan, np.uint8)
    n("Equal", ["L", "chan"], "output")            # [1,10,30,30] BOOL

    x = helper.make_tensor_value_info("input", F, [1, 10, N, N])
    y = helper.make_tensor_value_info("output", B, [1, 10, N, N])
    g = helper.make_graph(nodes, "task131", [x], [y], inits)
    return helper.make_model(g, ir_version=IR_VERSION,
                             opset_imports=[helper.make_opsetid("", 11)])
