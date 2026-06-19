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

Lean encoding (mem 7960 / params 728 -> 15.93, vs the prior 10791/1326 -> 15.60):
  NO 2-D colour/value planes are materialised.  All rule scalars come from cheap 1-D
  line aggregates, and ALL 2-D assembly runs on a SMALL canonical canvas.
  Re-golf wins over the 15.60 version:
    * RED+GREEN per-line counts PACKED into ONE Conv output channel (packed =
      1*redcount + 6*greencount; mutually exclusive, both <6) -> halves the wide
      Conv kernels (1200 -> 600 params).  green present <=> packed>=6;
      red line <=> packed==in-grid-count AND packed>0.
    * canonical GREEN built WITHOUT an 18x18 square: green always starts at row 0,
      so the two orientation candidates are NON-SQUARE crops
      (input[ch3,0:HR,0:WR] and transpose(input[ch3,0:WR,0:HR])) selected by xpose
      -> drops the 1296B fp32 18x18 slice + 3 uint8 squares.
    * label finalisation stays 4D (no final Reshape plane); int32 Gather index.
    * Red(2)+Green(3) per-line counts: ONE 2-output-channel Conv per axis (kernel
      spans the full long axis -> per-line sum).  In-grid per-line counts: a
      zero-param ReduceSum over the FREE 10-ch input.  The squeezed [30] line
      vectors are sliced to the active extent [18] so every downstream scalar plane
      is halved.
    * rc / cmin / cmax / shift / cyan are derived per orientation and picked by
      whichever axis carries a FULL red line (column => vertical).
    * 2-D assembly is done in a CANONICAL (red-vertical, un-flipped) frame whose grid
      is at most HR=5 rows x WR=18 cols (generator: height 4..5, width 16..18).  The
      green channel is sliced to an 18x18 square, cast to uint8, transposed and
      orientation-selected, then cropped to [5,18].  The creature is slid by a Gather
      idx[c']=c'-s; cyan/red are full in-grid columns; the [5,18] label is padded to
      18x18, transposed and orientation-selected to un-canonicalise, padded to
      [1,1,30,30] with off-grid sentinel 10, and the FREE BOOL output = Equal(L, arange).
  All values are small integers, exact in float32 / uint8.  flip is handled purely by
  the data-driven left/right shift (no special casing).
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

    # Canonical (red-vertical, un-flipped) frame: the generator makes height in
    # 4..5 and width in 16..18, with red drawn as a VERTICAL line.  So the
    # canonical grid is at most HR=5 rows x WR=18 cols.  xpose=1 instances are
    # the transpose (18 rows x 5 cols); we bring them into canonical by a single
    # square (SQ x SQ) transpose, then crop to [HR, WR].
    HR, WR, SQ = 5, 18, 18
    init("idxbase", np.arange(WR, dtype=np.float32), np.float32)            # [WR]
    init("zero", np.array(0.0, np.float32), np.float32)
    init("one", np.array(1.0, np.float32), np.float32)
    init("half", np.array(0.5, np.float32), np.float32)
    init("big", np.array(1000.0, np.float32), np.float32)
    init("neghalf", np.array(-0.5, np.float32), np.float32)
    init("s11", np.array([1, 1], np.int64), np.int64)


    # ===== 1-D aggregates via wide Convs (no 2-D red / in-grid plane) =======
    # column aggregate kernel [1,10,30,1] (sum over 30 rows) -> [1,1,1,30].
    # row    aggregate kernel [1,10,1,30] (sum over 30 cols) -> [1,1,30,1].
    # Red(2) and Green(3) per-line counts are PACKED into ONE output channel:
    #   packed = 1*redcount + 6*greencount  (kernel [1,10,L,1]=300 params/axis).
    # Red is a FULL line (count == height) and green sits on one side, so red and
    # green are never on the same line; both counts <= height <= 5 < 6, so:
    #   * green present  <=> packed >= 6   (green-only line => 6*g, g>=1)
    #   * red line       <=> packed == in-grid-count AND packed > 0
    #         (red col: packed=h=ingrid; green col: packed=6g != g=ingrid for g>0).
    # In-grid per-line counts are a zero-param ReduceSum over the FREE input.
    def rgker(over_rows):
        if over_rows:
            w = np.zeros((1, 10, N, 1), np.float32)   # [out=1 ; 10 ; 30 rows ; 1]
            w[0, 2, :, 0] = 1.0   # red, per column
            w[0, 3, :, 0] = 6.0   # green, per column (packed *6)
        else:
            w = np.zeros((1, 10, 1, N), np.float32)
            w[0, 2, 0, :] = 1.0
            w[0, 3, 0, :] = 6.0
        return w

    init("kcolRG", rgker(True), np.float32)    # [1,10,30,1]
    init("krowRG", rgker(False), np.float32)   # [1,10,1,30]
    init("five5", np.array(5.5, np.float32), np.float32)

    def aggregates(over_rows, tag):
        kRG = "kcolRG" if over_rows else "krowRG"
        n("Conv", ["input", kRG], f"{tag}Pc")   # [1,1,1,30] or [1,1,30,1] packed
        # in-grid count per line: sum over channels AND the long spatial axis.
        igaxes = [1, 2] if over_rows else [1, 3]
        n("ReduceSum", ["input"], f"{tag}IGc", axes=igaxes, keepdims=1)  # [1,1,1,30]/[1,1,30,1]
        sq = [0, 1, 2] if over_rows else [0, 1, 3]
        n("Squeeze", [f"{tag}Pc"], f"{tag}Pv0", axes=sq)    # [30]
        n("Squeeze", [f"{tag}IGc"], f"{tag}IGv0", axes=sq)
        # crop the [30] line vectors to the active extent [WR]=18 (cols/rows<=18,
        # red<=16; positions >=18 are always background) -> halve every scalar plane.
        n("Slice", [f"{tag}Pv0", "lv_s", "lv_e", "lv_ax"], f"{tag}Pv")
        n("Slice", [f"{tag}IGv0", "lv_s", "lv_e", "lv_ax"], f"{tag}IGv")
        return f"{tag}Pv", f"{tag}IGv"

    init("lv_s", np.array([0], np.int64), np.int64)
    init("lv_e", np.array([WR], np.int64), np.int64)
    init("lv_ax", np.array([0], np.int64), np.int64)

    vP, vIG = aggregates(True, "v")    # vertical-red: per-column packed aggregates
    hP, hIG = aggregates(False, "h")   # horizontal-red: per-row packed aggregates

    init("linev", np.arange(WR, dtype=np.float32), np.float32)  # [WR]

    def scalars(Pv, IGv, tag):
        # packed = redcount + 6*greencount.  Red line: packed==ingrid AND packed>0.
        n("Greater", [Pv, "zero"], f"{tag}rpos")
        n("Sub", [Pv, IGv], f"{tag}rd"); n("Abs", [f"{tag}rd"], f"{tag}rda")
        n("Less", [f"{tag}rda", "half"], f"{tag}req")
        n("And", [f"{tag}rpos", f"{tag}req"], f"{tag}redln_b")  # [30]
        n("Cast", [f"{tag}redln_b"], f"{tag}redln", to=F)
        n("ReduceMax", [f"{tag}redln"], f"{tag}has", axes=[0], keepdims=1)  # [1]
        n("Mul", [f"{tag}redln", "linev"], f"{tag}rcw")
        n("ReduceSum", [f"{tag}rcw"], f"{tag}rc", axes=[0], keepdims=1)  # [1]
        # green present per line <=> packed >= 6 (green-only line => 6*g).
        n("Greater", [Pv, "five5"], f"{tag}gp_b")   # >5 i.e. >=6
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

    vhas, vs, vcyc, vrc = scalars(vP, vIG, "vs_")
    hhas, hs, hcyc, hrc = scalars(hP, hIG, "hs_")

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
    # canonical grid width (long, red-vertical axis): vertical=Worig, horizontal=Horig.
    n("Where", ["usev", Worig, Horig], "Wc")  # canonical width  (cols)
    # canonical grid height (short, perpendicular to red): vertical=Horig, horizontal=Worig.
    n("Where", ["usev", Horig, Worig], "Hc")  # canonical height (rows)

    # ===== canonical GREEN plane [HR,WR] WITHOUT an 18x18 square ============
    # The green creature ALWAYS starts at row 0 (continuous_creature begins at (0,0)
    # and is placed at grid[r][c+offset]), so in the canonical (red-vertical) frame
    # green occupies rows 0..HR-1 and some columns in 0..WR-1.  Build the two
    # orientation candidates as NON-SQUARE crops and select -- no [18,18] plane:
    #   xpose=0 canonical green  = input[ch3, 0:HR, 0:WR]            -> [HR,WR]
    #   xpose=1 canonical green  = transpose(input[ch3, 0:WR, 0:HR]) -> [HR,WR]
    init("u0", np.array(0, np.uint8), np.uint8)
    n("Reshape", ["usev", "s11"], "usev11")
    # candidate A (xpose=0): rows 0..HR, cols 0..WR
    init("gA_s", np.array([0, 3, 0, 0], np.int64), np.int64)
    init("gA_e", np.array([1, 4, HR, WR], np.int64), np.int64)
    init("gAB_ax", np.array([0, 1, 2, 3], np.int64), np.int64)
    n("Slice", ["input", "gA_s", "gA_e", "gAB_ax"], "GApc")  # [1,1,HR,WR] float
    n("Cast", ["GApc"], "GAu", to=U8)
    n("Squeeze", ["GAu"], "GA", axes=[0, 1])                 # [HR,WR] uint8
    # candidate B (xpose=1): rows 0..WR, cols 0..HR, then transpose -> [HR,WR]
    init("gB_e", np.array([1, 4, WR, HR], np.int64), np.int64)
    n("Slice", ["input", "gA_s", "gB_e", "gAB_ax"], "GBpc")  # [1,1,WR,HR] float
    n("Cast", ["GBpc"], "GBu", to=U8)
    n("Squeeze", ["GBu"], "GB0", axes=[0, 1])                # [WR,HR] uint8
    n("Transpose", ["GB0"], "GB", perm=[1, 0])               # [HR,WR] uint8
    n("Where", ["usev11", "GA", "GB"], "G")                  # [HR,WR] canonical green

    # in-grid (canonical): rows < Hc AND cols < Wc (separable, broadcast to [HR,WR]).
    init("colW", np.arange(WR, dtype=np.float32).reshape(1, WR), np.float32)  # [1,WR]
    init("rowH", np.arange(HR, dtype=np.float32).reshape(HR, 1), np.float32)  # [HR,1]
    n("Less", ["colW", "Wc"], "colin_b")    # [1,WR]
    n("Less", ["rowH", "Hc"], "rowin_b")    # [HR,1]
    n("And", ["colin_b", "rowin_b"], "IG_b")  # [HR,WR]

    # ===== shift canonical green by s via Gather (along WR cols) ============
    n("Reshape", ["s", "s11"], "s2"); n("Squeeze", ["s2"], "ss", axes=[0, 1])  # scalar
    n("Sub", ["idxbase", "ss"], "idxf")            # [WR]
    init("zerof", np.array(0.0, np.float32), np.float32)
    init("maxf", np.array(float(WR - 1), np.float32), np.float32)
    n("Clip", ["idxf", "zerof", "maxf"], "idxc")
    n("Cast", ["idxc"], "idx", to=TensorProto.INT32)
    n("Gather", ["G", "idx"], "Gs", axis=1)        # [HR,WR] uint8, Gs[r,c']=G[r,c'-s]
    n("Greater", ["Gs", "u0"], "Gs_b")             # [HR,WR] bool presence
    n("Greater", ["idxf", "neghalf"], "ge0")
    init("mmhalf", np.array(float(WR) - 0.5, np.float32), np.float32)
    n("Less", ["idxf", "mmhalf"], "le0")
    n("And", ["ge0", "le0"], "valid")              # [WR]
    n("And", ["Gs_b", "valid"], "Gfin_b")          # [HR,WR]

    # ===== cyan full column at cyc, red column at rcc =======================
    # (red+cyan are full columns over all HR canonical rows, then masked off-grid
    #  cols >= Wc; rows are always in-grid since canonical height <= HR.)
    n("Sub", ["colW", "cyc"], "cyd"); n("Abs", ["cyd"], "cyda")
    n("Less", ["cyda", "half"], "cymask_b")        # [1,WR]
    n("And", ["cymask_b", "IG_b"], "cy_b")         # [HR,WR]
    n("Sub", ["colW", "rcc"], "rdd"); n("Abs", ["rdd"], "rdda")
    n("Less", ["rdda", "half"], "rmask_b")         # [1,WR]
    n("And", ["rmask_b", "IG_b"], "R_b")           # [HR,WR]

    # ===== assemble canonical label map [HR,WR] =============================
    init("u2", np.array(2, np.uint8), np.uint8)
    init("u3", np.array(3, np.uint8), np.uint8)
    init("u8", np.array(8, np.uint8), np.uint8)
    init("u10", np.array(10, np.uint8), np.uint8)
    # in-grid background -> 0, off-grid -> 10 sentinel (all-false output)
    n("Where", ["IG_b", "u0", "u10"], "L0")        # [HR,WR]
    n("Where", ["cy_b", "u8", "L0"], "L1")         # [HR,WR]
    n("Where", ["R_b", "u2", "L1"], "Lrc")         # [HR,WR]
    n("Where", ["Gfin_b", "u3", "Lrc"], "Lcanon")  # [HR,WR] green on top

    # pad canonical [HR,WR] up to [SQ,SQ] (rows -> SQ with off-grid sentinel 10),
    # transpose, and select orientation -> uncanonicalize.  Stay 4D so no final
    # Reshape plane is needed (the 30x30 Pad consumes a [1,1,SQ,SQ] tensor).
    init("Lc4", np.array([1, 1, HR, WR], np.int64), np.int64)
    n("Reshape", ["Lcanon", "Lc4"], "Lcanon4")                      # [1,1,HR,WR]
    init("lpad", np.array([0, 0, 0, 0, 0, 0, SQ - HR, 0], np.int64), np.int64)
    n("Pad", ["Lcanon4", "lpad", "u10"], "Lcsq", mode="constant")   # [1,1,SQ,SQ]
    n("Transpose", ["Lcsq"], "LcsqT", perm=[0, 1, 3, 2])            # [1,1,SQ,SQ]
    n("Reshape", ["usev", "Mr4"], "usev4")
    n("Where", ["usev4", "Lcsq", "LcsqT"], "Lsel")                  # [1,1,SQ,SQ]
    init("Mr4", np.array([1, 1, 1, 1], np.int64), np.int64)
    init("padcfg", np.array([0, 0, 0, 0, 0, 0, N - SQ, N - SQ], np.int64), np.int64)
    n("Pad", ["Lsel", "padcfg", "u10"], "L", mode="constant")  # [1,1,N,N]

    chan = np.arange(10, dtype=np.uint8).reshape(1, 10, 1, 1)
    init("chan", chan, np.uint8)
    n("Equal", ["L", "chan"], "output")            # [1,10,30,30] BOOL

    x = helper.make_tensor_value_info("input", F, [1, 10, N, N])
    y = helper.make_tensor_value_info("output", B, [1, 10, N, N])
    g = helper.make_graph(nodes, "task131", [x], [y], inits)
    return helper.make_model(g, ir_version=IR_VERSION,
                             opset_imports=[helper.make_opsetid("", 11)])
