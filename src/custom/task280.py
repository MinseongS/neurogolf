"""Task 280 (b527c5c6): two boxes, each emits a structured "beam".

Rule (verified fresh 800/800 with a from-input reconstructor):
  Input has TWO solid green rectangles, each with ONE red dot on one of its
  edges.  One box is tall (H>=W) -> HORIZONTAL beam; the other wide (H<W) ->
  VERTICAL beam.  A beam is an axis-aligned rectangle:
    * tall box: band rows = dotrow +/- (W-1) (W=box width), centred on dot row;
      cols run from the dot to the grid wall on the side the box does NOT occupy
      (dot on left edge -> beam left, etc.).  Green band, centre row red.
    * wide box: axes swapped (band cols = dotcol +/- (H-1), rows -> wall along
      empty side, centre col red).
  Output = input + the two beams (red centre line over green band; green fills
  only previously-empty cells; clipped to the n x n in-grid region).

ONNX construction (NO 30x30 working plane; beams are SEPARABLE row (x) col
rectangles broadcast straight into the FREE Where output):
  - R = input ch2, FG = (sum_k k*input_k > 0), IN = (sum_k input_k > 0) ingrid.
  - 2 dots: ArgMax over flattened R, mask, ArgMax again -> coords [2].
  - per dot Gather its row-fg & col-fg [2,30] vectors; the contiguous run on
    each side = distance to the nearest non-fg cell, via ReduceMax/ReduceMin of
    masked position ramps.  -> Wh, Wv (run widths), horiz=Wv>=Wh, half=min-1.
  - beam row/col masks built from comparisons on the position ramp, OR'd over
    the 2-dot axis -> green-band mask & red-line mask [30,30].
  - out = Where(redmask, red1hot, Where(greenband & empty, green1hot, input))
    then AND ingrid; emitted as the 10-ch BOOL output via Equal(label, arange).
"""

import numpy as np
import onnx
from onnx import helper, numpy_helper, TensorProto

from ..harness import IR_VERSION

N = 30
F = TensorProto.FLOAT
B = TensorProto.BOOL
I64 = TensorProto.INT64


def build(task):
    inits, nodes = [], []

    def init(name, arr, dtype=None):
        a = np.ascontiguousarray(arr)
        if dtype is not None and not isinstance(dtype, int):
            a = a.astype(dtype)
        inits.append(numpy_helper.from_array(a, name))
        return name

    def nd(op, ins, out, **attrs):
        nodes.append(helper.make_node(op, list(ins), [out], **attrs))
        return out

    # ---------------- constants ----------------
    init("zerf", np.array(0.0, np.float32), F)
    init("half_c", np.array(0.5, np.float32), F)
    init("bigN", np.array(float(N), np.float32), F)
    init("one_f", np.array(1.0, np.float32), F)

    # ---------------- slice / reduce colour planes ----------------
    init("shHW", np.array([N, N], np.int64), I64)
    # input only contains colours {0 bg, 2 red, 3 green}.  Slice ch2,ch3,ch0.
    init("c0", np.array([0], np.int64), I64)
    init("c1", np.array([1], np.int64), I64)
    init("c2", np.array([2], np.int64), I64)
    init("c3", np.array([3], np.int64), I64)
    init("c4", np.array([4], np.int64), I64)
    init("ax1", np.array([1], np.int64), I64)
    nd("Slice", ["input", "c2", "c3", "ax1"], "R4")                   # [1,1,30,30]
    nd("Slice", ["input", "c3", "c4", "ax1"], "G4")
    nd("Slice", ["input", "c0", "c1", "ax1"], "BG4")
    # convert to bool FIRST (900B) then reshape bool to [30,30]
    nd("Greater", ["R4", "zerf"], "Rb4")                              # [1,1,30,30] bool
    nd("Greater", ["G4", "zerf"], "Gb4")
    nd("Greater", ["BG4", "zerf"], "BGb4")
    nd("Reshape", ["Rb4", "shHW"], "Rb")                             # [30,30] bool
    nd("Reshape", ["Gb4", "shHW"], "Gb")
    nd("Reshape", ["BGb4", "shHW"], "BGb")
    nd("Or", ["Rb", "Gb"], "FGb")                                    # [30,30] bool
    nd("Or", ["FGb", "BGb"], "INb")                                  # ingrid bool

    # ---------------- locate 2 dots (uint8 flat, 900B) ----------------
    nd("Cast", ["Rb4"], "Ru8")                                       # [1,1,30,30] uint8
    nodes[-1].attribute.append(helper.make_attribute("to", TensorProto.UINT8))
    nd("Reshape", ["Ru8", _i(inits, "sh900", [N * N], I64)], "Rflat")  # [900] uint8
    nd("ArgMax", ["Rflat"], "idx0", axis=0, keepdims=0)               # scalar int64
    nd("Reshape", ["idx0", _i(inits, "sh1", [1], I64)], "idx0v")      # [1]
    inits.append(numpy_helper.from_array(np.array([0], np.uint8), "u8zero"))
    nd("ScatterElements", ["Rflat", "idx0v", "u8zero"], "Rflat1", axis=0)
    nd("ArgMax", ["Rflat1"], "idx1", axis=0, keepdims=0)
    nd("Reshape", ["idx1", "sh1"], "idx1v")
    # coords
    nd("Concat", ["idx0v", "idx1v"], "idxs", axis=0)                  # [2] int64
    nd("Cast", ["idxs"], "idxf", to=F)                               # [2] float
    nd("Div", ["idxf", "bigN"], "dyf_raw")
    nd("Floor", ["dyf_raw"], "dyf")                                  # [2] dot rows
    nd("Mul", ["dyf", "bigN"], "dyN")
    nd("Sub", ["idxf", "dyN"], "dxf")                               # [2] dot cols

    # ---------------- gather per-dot fg rows/cols ----------------
    nd("Cast", ["dyf"], "dyi", to=I64)
    nd("Cast", ["dxf"], "dxi", to=I64)
    nd("Gather", ["FGb", "dyi"], "rowfgb", axis=0)                   # [2,30] bool
    nd("Gather", ["FGb", "dxi"], "colfgb", axis=1)                   # [30,2] bool
    nd("Cast", ["rowfgb"], "rowfg", to=F)                           # [2,30] small
    nd("Transpose", ["colfgb"], "colfgbT", perm=[1, 0])             # [2,30]
    nd("Cast", ["colfgbT"], "colfgT", to=F)

    # ---------------- contiguous run measurement ----------------
    # gap = (fg<0.5); position ramp pos[30]; per dot center dx (for rowfg), dy (col)
    # left run = center-1 - max{ j<center : gap[j] }   (=-1 if none -> run=center)
    # right run = min{ j>center : gap[j] } - center -1 (=N if none -> run=N-1-center)
    Lr = _runs(nd, init, inits, "rowfg", "dxf", "row")  # returns names (L,R)
    Ur = _runs(nd, init, inits, "colfgT", "dyf", "col")
    Lname, Rname = Lr
    Uname, Dname = Ur
    nd("Add", [Lname, Rname], "Whm1")            # Wh-1  [2]
    nd("Add", [Uname, Dname], "Wvm1")            # Wv-1  [2]
    # Wh=Whm1+1, Wv=Wvm1+1 ; horiz = Wv>=Wh  <=>  Wvm1>=Whm1  <=>  NOT(Wvm1<Whm1)
    nd("Less", ["Wvm1", "Whm1"], "vert_b")        # [2] bool: vertical beam
    nd("Cast", ["vert_b"], "vert_f", to=F)
    # half = horiz? Wh-1 : Wv-1
    nd("Where", ["vert_b", "Wvm1", "Whm1"], "half")  # [2] float

    # direction flags: horizontal -> beam left if Lr==0 (Lname==0)
    nd("Equal", [Lname, "zerf_s"], "leftedge_b")     # [2] bool (Lname==0 -> beam left)
    nd("Equal", [Uname, "zerf_s"], "topedge_b")
    _i_f(inits, "zerf_s", [0.0])

    # ---------------- build beam masks per dot, OR over the 2 axis ----------------
    # We work with shape [2, 30, 30] then ReduceMax over axis 0.
    # ramps
    init("ar_r", np.arange(N, dtype=np.float32).reshape(1, N, 1), F)   # rows [1,30,1]
    init("ar_c", np.arange(N, dtype=np.float32).reshape(1, 1, N), F)   # cols [1,1,30]
    # dot coords broadcastable [2,1,1]
    nd("Reshape", ["dyf", _i(inits, "sh211", [2, 1, 1], I64)], "dy3")
    nd("Reshape", ["dxf", "sh211"], "dx3")
    nd("Reshape", ["half", "sh211"], "half3")
    nd("Reshape", ["vert_b", "sh211"], "vert3b")         # bool [2,1,1] vertical
    nd("Not", ["vert3b"], "horiz3b")
    nd("Reshape", ["leftedge_b", "sh211"], "left3b")
    nd("Reshape", ["topedge_b", "sh211"], "top3b")

    # ---- HORIZONTAL beam contribution (per dot, only where NOT vertical) ----
    # row band: |row - dy| <= half
    nd("Sub", ["ar_r", "dy3"], "rdiff")
    nd("Abs", ["rdiff"], "rabs")
    nd("Add", ["half3", "half_c"], "halfp")        # half+0.5 for <= via <
    nd("Less", ["rabs", "halfp"], "rowband_b")     # [2,30,1] bool
    # centre row: |row-dy|<0.5
    nd("Less", ["rabs", "half_c"], "rowctr_b")     # [2,30,1]
    # col side: leftedge -> col<dx ; else col>dx  (off-grid suppressed later by ingrid)
    nd("Less", ["ar_c", "dx3"], "colLT_b")         # col<dx
    nd("Greater", ["ar_c", "dx3"], "colRight_b")   # col>dx
    # hcol = left ? colLT : colRight  (avoid bool-branch Where)
    nd("And", ["left3b", "colLT_b"], "hcol_a")
    nd("Not", ["left3b"], "notleft3b")
    nd("And", ["notleft3b", "colRight_b"], "hcol_b2")
    nd("Or", ["hcol_a", "hcol_b2"], "hcol_b")   # [2,1,30]
    # gate by horiz, then reduce over the 2-dot axis to SMALL vectors (since exactly
    # one dot is horizontal) -> no [2,30,30] product ever forms.
    nd("And", ["hcol_b", "horiz3b"], "hcolH")   # [2,1,30]
    nd("And", ["rowband_b", "horiz3b"], "rbandH")  # [2,30,1]
    nd("And", ["rowctr_b", "horiz3b"], "rctrH")    # [2,30,1]

    # ---- VERTICAL beam masks (small) ----
    nd("Sub", ["ar_c", "dx3"], "cdiff")
    nd("Abs", ["cdiff"], "cabs")
    nd("Less", ["cabs", "halfp"], "colband_b")     # [2,1,30]
    nd("Less", ["cabs", "half_c"], "colctr_b")
    nd("Less", ["ar_r", "dy3"], "rowLT_b")
    nd("Greater", ["ar_r", "dy3"], "rowDown_b")
    nd("And", ["top3b", "rowLT_b"], "vrow_a")
    nd("Not", ["top3b"], "nottop3b")
    nd("And", ["nottop3b", "rowDown_b"], "vrow_b2")
    nd("Or", ["vrow_a", "vrow_b2"], "vrow_b")   # [2,30,1]
    nd("And", ["vrow_b", "vert3b"], "vrowV")    # [2,30,1]
    nd("And", ["colband_b", "vert3b"], "cbandV") # [2,1,30]
    nd("And", ["colctr_b", "vert3b"], "cctrV")   # [2,1,30]

    # reduce each small mask over the 2-dot axis (axis 0)
    def or2(name, out, shp):
        a, b = out + "_a", out + "_b"
        nodes.append(helper.make_node("Split", [name], [a, b], axis=0))
        nd("Or", [a, b], out)
    or2("hcolH", "hcol1", None)     # [1,1,30]
    or2("rbandH", "rband1", None)   # [1,30,1]
    or2("rctrH", "rctr1", None)     # [1,30,1]
    or2("vrowV", "vrow1", None)     # [1,30,1]
    or2("cbandV", "cband1", None)   # [1,1,30]
    or2("cctrV", "cctr1", None)     # [1,1,30]

    # single outer-products -> [1,30,30]
    nd("And", ["rband1", "hcol1"], "hg")          # horiz green
    nd("And", ["rctr1", "hcol1"], "hr")
    nd("And", ["cband1", "vrow1"], "vg")          # vert green
    nd("And", ["cctr1", "vrow1"], "vr")
    nd("Or", ["hg", "vg"], "g_all3")              # [1,30,30]
    nd("Or", ["hr", "vr"], "r_all3")
    nd("Reshape", ["g_all3", "shHW"], "greenmask") # [30,30]
    nd("Reshape", ["r_all3", "shHW"], "redmask")

    # ---------------- compose output one-hot directly (bool) ----------------
    # isred  = (input-red OR beam-red) AND ingrid
    # isgreen= (input-green OR (beam-green AND NOT fg)) AND NOT red AND ingrid
    # bg     = ingrid AND NOT (red OR green)
    nd("Or", ["Rb", "redmask"], "redfull")
    nd("And", ["redfull", "INb"], "out_red")
    nd("Not", ["FGb"], "emptyc")
    nd("And", ["greenmask", "emptyc"], "gbeam_ok")
    nd("Or", ["Gb", "gbeam_ok"], "green0")
    nd("Not", ["out_red"], "notred")
    nd("And", ["green0", "notred"], "greenfull")
    nd("And", ["greenfull", "INb"], "out_green")
    nd("Or", ["out_red", "out_green"], "anyfg")
    nd("Not", ["anyfg"], "notfg")
    nd("And", ["INb", "notfg"], "out_bg")
    nd("Reshape", ["out_bg", _i(inits, "sh1130", [1, 1, N, N], I64)], "out_bg4")
    nd("Reshape", ["out_red", "sh1130"], "out_red4")
    nd("Reshape", ["out_green", "sh1130"], "out_green4")
    init("falsech", np.zeros((1, 1, N, N), dtype=bool), None)
    nodes.append(helper.make_node("Concat",
        ["out_bg4", "falsech", "out_red4", "out_green4",
         "falsech", "falsech", "falsech", "falsech", "falsech", "falsech"],
        ["output"], axis=1))

    graph = helper.make_graph(nodes, "task280",
        [helper.make_tensor_value_info("input", F, [1, 10, N, N])],
        [helper.make_tensor_value_info("output", B, [1, 10, N, N])],
        inits)
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 11)])
    model.ir_version = IR_VERSION
    return model


def _i(inits, name, vals, dtype):
    inits.append(numpy_helper.from_array(np.array(vals, dtype=np.int64), name))
    return name


def _i_f(inits, name, vals):
    inits.append(numpy_helper.from_array(np.array(vals, dtype=np.float32), name))
    return name


def _runs(nd, init, inits, vecname, ctrname, tag):
    """vecname: [2,30] fg float; ctrname: [2] center float.
    Returns (Lname, Rname) = left/right contiguous run lengths [2]."""
    # pos ramp [1,30]
    pr = f"posr_{tag}"
    inits.append(numpy_helper.from_array(
        np.arange(N, dtype=np.float32).reshape(1, N), pr))
    gap = f"gap_{tag}"
    nd("Less", [vecname, "half_c"], gap)            # bool [2,30] non-fg
    ctr2 = f"ctr2_{tag}"
    nd("Reshape", [ctrname, _i(inits, f"sh21_{tag}", [2, 1], I64)], ctr2)  # [2,1]
    # left: positions < center AND gap -> max position, else -1
    posLT = f"posLT_{tag}"; nd("Less", [pr, ctr2], posLT)      # [2,30] bool
    gL = f"gL_{tag}"; nd("And", [gap, posLT], gL)
    # masked pos = where(gL, pos, -1)
    mlp = f"mlp_{tag}"
    nd("Where", [gL, pr, _i_f(inits, f"negposL_{tag}", [-1.0])], mlp)
    blk = f"blkL_{tag}"; nd("ReduceMax", [mlp], blk, axes=[1], keepdims=0)  # [2]
    # L = center-1-blocker
    cm1 = f"cm1_{tag}"; nd("Sub", [ctrname, "one_f"], cm1)
    Lname = f"Lrun_{tag}"; nd("Sub", [cm1, blk], Lname)
    # right: positions>center AND gap -> min position else N
    posGT = f"posGT_{tag}"; nd("Greater", [pr, ctr2], posGT)
    gR = f"gR_{tag}"; nd("And", [gap, posGT], gR)
    mrp = f"mrp_{tag}"
    nd("Where", [gR, pr, _i_f(inits, f"bigposR_{tag}", [float(N)])], mrp)
    blkr = f"blkR_{tag}"; nd("ReduceMin", [mrp], blkr, axes=[1], keepdims=0)
    cp1 = f"cp1_{tag}"; nd("Add", [ctrname, "one_f"], cp1)
    Rname = f"Rrun_{tag}"; nd("Sub", [blkr, cp1], Rname)
    return Lname, Rname
