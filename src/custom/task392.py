"""Task 392 (ARC-GEN f8c80d96): complete the nested concentric square "mats".

Rule (from the ARC-GEN generator, verified fresh):
  A 10x10 grid (embedded top-left of the 30x30 one-hot canvas).  The generator
  picks a centre (row, col) -- one of the two is always 0 (the centre sits on an
  edge) -- a line thickness ``thick`` in {1,2}, a colour, and ``show`` in {2,3}.
  It draws ``size`` concentric square ring "mats": ring ``i`` is the perimeter of
  the box [A_i, B_i]^2 (offsets from the centre) with
      A_i = thick - (thick+1)*i ,   B_i = (thick+1)*i - 1 .
  The INPUT shows only the first ``show+1`` rings; the OUTPUT draws ALL rings on
  a gray (5) background.  The output is therefore a pure function of
  (row, col, thick, colour), independent of ``show``.

  Closed form for a painted output cell (verified exhaustively):
      painted(r,c)  <=>  max(need(r-row), need(c-col)) % (thick+1) == 0
      need(x) = max(x+1, thick-x)
  i.e. the full painted set is a deterministic nested-ring pattern.

Encoding (min-area-covering candidate match, exact, all-tiny intermediates):
  Because one of {row,col} is 0 and thick in {1,2}, there are only 38 distinct
  possible full output patterns.  The shown input rings are always a SUBSET of
  the (unique) full pattern, and that full pattern is the smallest-area candidate
  that COVERS the shown input -- proven unique over 60000 fresh instances.

  So: P = painted-mask of the input (10x10), then for each candidate mask C_k
  (flattened [38,100]) compute covered_k = C_k . Pvec (a single MatMul -> [38,1]),
  pick argmin( area_k + BIG*(|P| - covered_k) ); the winner is the full output
  painted mask.  Gather it back to [10,10].

  Output one-hot is routed into the FREE bool output via a uint8 label plane:
      L = colour     where painted
        = 5 (gray)   where in-grid & not painted
        = 10         off-grid (matches no channel -> all-zero, as the harness
                     encodes off-grid cells)
  final op  Equal(L, arange[0..9]) -> BOOL output, so the 10-channel expansion
  costs nothing.

  Dominant intermediate: the [38,100] candidate-mask initializer (3800 params,
  free in *memory*; it is the score floor here) and the padded label plane
  Lp uint8 [1,1,30,30] = 900 B (irreducible: must be 30x30 to broadcast against
  the 10 colour channels in the final Equal).  All scoring tensors are <=400 B.
"""

import numpy as np
from onnx import helper, numpy_helper, TensorProto

from ..harness import IR_VERSION

G = 10  # active grid size


def _need(x, thick):
    return max(x + 1, thick - x)


def _full_mask(row, col, thick):
    s = thick + 1
    M = np.zeros((G, G), np.float32)
    for r in range(G):
        for c in range(G):
            if max(_need(r - row, thick), _need(c - col, thick)) % s == 0:
                M[r, c] = 1.0
    return M


def _candidates():
    uniq = {}
    order = []
    for th in (1, 2):
        cells = [(k, 0) for k in range(G)] + [(0, k) for k in range(G)]
        for (rr, cc) in cells:
            m = _full_mask(rr, cc, th)
            key = m.tobytes()
            if key not in uniq:
                uniq[key] = m
                order.append(m)
    C = np.stack(order)  # [U,G,G]
    return C


def build(task):
    inits, nodes = [], []

    def init(name, arr, dtype):
        inits.append(numpy_helper.from_array(
            np.ascontiguousarray(arr, dtype=dtype), name))
        return name

    def n(op, ins, out, **attrs):
        nodes.append(helper.make_node(op, ins, [out], **attrs))
        return out

    C = _candidates()              # [U,G,G] float32
    U = C.shape[0]
    area = C.reshape(U, -1).sum(1)  # [U]
    Cflat = C.reshape(U, G * G)     # [U,100]

    # ---- recover painted mask P over the 10x10 grid -------------------------
    # input is [1,10,30,30]; slice the non-bg colour channels (1..9) over the
    # 10x10 grid, sum over channels -> occupancy, >0 -> painted.
    # painted = non-background.  Background is colour 0; within the 10x10 grid a
    # background cell sets channel 0, a painted cell does not (off-grid sets none).
    init("ax0123", np.array([0, 1, 2, 3], np.int64), np.int64)
    init("bg_s", np.array([0, 0, 0, 0], np.int64), np.int64)
    init("bg_e", np.array([1, 1, G, G], np.int64), np.int64)
    n("Slice", ["input", "bg_s", "bg_e", "ax0123"], "bg")     # [1,1,10,10] f32
    init("half", np.array(0.5, np.float32), np.float32)
    n("Less", ["bg", "half"], "Pbool")                        # [1,1,10,10] bool (painted)
    n("Cast", ["Pbool"], "Pf", to=TensorProto.FLOAT)          # [1,1,10,10] f32

    # ---- min-area covering candidate ---------------------------------------
    # Pvec [100,1]
    init("pvec_shape", np.array([G * G, 1], np.int64), np.int64)
    n("Reshape", ["Pf", "pvec_shape"], "Pvec")                # [100,1]
    init("Cflat", Cflat, np.float32)                          # [U,100] const
    n("MatMul", ["Cflat", "Pvec"], "covered")                 # [U,1]
    init("npaint_shape", np.array([1, 1], np.int64), np.int64)
    n("ReduceSum", ["Pf"], "np_raw", axes=[0, 1, 2, 3], keepdims=1)  # [1,1,1,1]
    n("Reshape", ["np_raw", "npaint_shape"], "npaint")        # [1,1]
    n("Sub", ["npaint", "covered"], "uncov")                  # [U,1]
    init("BIG", np.array(1000.0, np.float32), np.float32)
    n("Mul", ["uncov", "BIG"], "uncov_b")                     # [U,1]
    init("area", area.reshape(U, 1), np.float32)              # [U,1] const
    n("Add", ["uncov_b", "area"], "score")                    # [U,1]
    init("score_shape", np.array([U], np.int64), np.int64)
    n("Reshape", ["score", "score_shape"], "score_v")         # [U]
    n("ArgMin", ["score_v"], "win", axis=0, keepdims=0)       # scalar int64

    # gather winning full row of Cflat -> [100], reshape to [G,G]
    n("Gather", ["Cflat", "win"], "winrow", axis=0)           # [100] f32

    # ---- recover colour scalar ---------------------------------------------
    # per-channel presence over the whole (FREE) input, weighted by channel index.
    # The off-grid region is all-zero so it never contributes; only the one fg
    # colour channel is present.  ReduceMax output is tiny ([1,10,1,1] = 40 B).
    n("ReduceMax", ["input"], "chpres_raw", axes=[2, 3], keepdims=1)  # [1,10,1,1]
    # weight channel k by k (channel 0 weight 0 so background drops out)
    init("chw", np.arange(0, 10, dtype=np.float32).reshape(1, 10, 1, 1),
         np.float32)
    n("Mul", ["chpres_raw", "chw"], "chw_w")                  # [1,10,1,1]
    n("ReduceSum", ["chw_w"], "color_raw", axes=[0, 1, 2, 3], keepdims=0)  # scalar
    # -> colour value (float). cast to build label plane.

    # ---- build 30x30 label plane -------------------------------------------
    # winmask [G,G] -> [1,1,G,G]
    init("wm_shape", np.array([1, 1, G, G], np.int64), np.int64)
    n("Reshape", ["winrow", "wm_shape"], "wm")                # [1,1,10,10] f32
    # painted bool
    n("Greater", ["wm", "half"], "wm_bool")                   # [1,1,10,10] bool

    # colour broadcast plane (float scalar -> we make a uint8 label):
    # L_in = color if painted else 5  (over the 10x10 grid)
    init("c_shape", np.array([1, 1, 1, 1], np.int64), np.int64)
    n("Reshape", ["color_raw", "c_shape"], "color_4d")        # [1,1,1,1] f32
    init("gray5", np.array(5.0, np.float32).reshape(1, 1, 1, 1), np.float32)
    n("Where", ["wm_bool", "color_4d", "gray5"], "Lgrid_f")   # [1,1,10,10] f32
    n("Cast", ["Lgrid_f"], "Lgrid_u", to=TensorProto.UINT8)   # [1,1,10,10] u8

    # Pad to 30x30 with sentinel 10 (off-grid -> matches no channel)
    init("pads", np.array([0, 0, 0, 0, 0, 0, 20, 20], np.int64), np.int64)
    init("sent", np.array(10, np.uint8), np.uint8)
    n("Pad", ["Lgrid_u", "pads", "sent"], "Lp", mode="constant")  # [1,1,30,30] u8

    # final: Equal(Lp, arange[0..9]) -> BOOL output [1,10,30,30]
    init("arange10", np.arange(10, dtype=np.uint8).reshape(1, 10, 1, 1),
         np.uint8)
    n("Equal", ["Lp", "arange10"], "output")                  # BOOL output

    graph = helper.make_graph(
        nodes, "task392",
        [helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 10, 30, 30])],
        [helper.make_tensor_value_info("output", TensorProto.BOOL, [1, 10, 30, 30])],
        inits,
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 11)])
    model.ir_version = IR_VERSION
    return model
