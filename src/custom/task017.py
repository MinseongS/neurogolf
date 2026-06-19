"""task017 (ARC-AGI 0dfd9992) — fill black cutouts in a doubly-periodic pattern.

Rule (from the ARC-GEN generator, verified fresh):
  A size=21 grid is filled with a DOUBLY-PERIODIC pattern of period `length`
  (4..9) in BOTH axes (same offset/length on both axes):
      v(r,c) = ((rr*rr + cc*cc) % mod) + 1,  rr=(offset+r)%length-length//2,
                                             cc=(offset+c)%length-length//2
  with mod in 4..9, length in 4..mod, offset in 1..length, half = length//2.
  The INPUT has 5 black rectangles (colour 0) stamped over the pattern; the
  OUTPUT is the SAME pattern with the cutouts removed.

Approach (TEMPLATE-MATCH the global parameters, then CLOSED-FORM rebuild — exact,
no per-cell period detection or max-fold):
  There are only 106 valid (mod,length,offset,half) parameter tuples.  Precompute
  for each tuple the pattern colour at 16 fixed sample cells -> candidate_samples
  [106,16] (uint8).  Read those 16 cells' colours from the input via GatherND +
  ArgMax over the colour axis -> input_sample[16].  Score each candidate by the
  number of agreeing samples (ReduceSum of Equal) and ArgMax -> best tuple
  (cutout-robust majority vote).  Recover (mod,length,offset,half) and rebuild the
  whole 21x21 pattern by the closed-form formula.  Pad the colour-index plane to
  30x30 with a -1 sentinel and Equal -> the FREE bool one-hot output.

Tables (candidate_params, candidate_samples, sample_nd_idx) are reused verbatim
from the verified crowd net; the win over it is routing the 10-ch one-hot into the
FREE output (the crowd net materialised onehot_raw [1,10,21,21]=4410B then Pad'd
it) and keeping the match comparison narrow.
"""

import numpy as np
import onnx
from onnx import helper, numpy_helper, TensorProto

from ..harness import IR_VERSION

S = 30
G = 21
_KOJI = "/Users/minseong/project/neurogolf/networks/task017.onnx"


def _tables():
    m = onnx.load(_KOJI)
    d = {init.name: onnx.numpy_helper.to_array(init) for init in m.graph.initializer}
    return d


def build(task):
    d = _tables()
    cand_params = np.asarray(d["candidate_params"], dtype=np.uint8)      # [106,4]
    cand_samples = np.asarray(d["candidate_samples"], dtype=np.uint8)    # [1,106,16]
    sample_nd = np.asarray(d["sample_nd_idx"], dtype=np.int64)           # [10,16,4]
    row_grid = np.asarray(d["row_grid"], dtype=np.float16)               # [1,1,21,1]
    col_grid = np.asarray(d["col_grid"], dtype=np.float16)               # [1,1,1,21]

    # Drop one weak sample (greedy-selected): trims the [106,NS] match plane and
    # the GatherND index table at ~baseline robustness (~99.93% fresh vs 99.95%).
    SUB = [0, 1, 2, 3, 4, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]   # NS=15
    cand_samples = np.ascontiguousarray(cand_samples[:, :, SUB])  # [1,106,15]
    sample_nd = np.ascontiguousarray(sample_nd[:, SUB, :])        # [10,15,4]

    NC = cand_params.shape[0]
    NS = sample_nd.shape[1]

    inits, nodes = [], []
    seen = set()

    def init(name, arr, dt):
        if name in seen:
            return name
        seen.add(name)
        inits.append(numpy_helper.from_array(np.ascontiguousarray(arr, dtype=dt), name))
        return name

    def n(op, ins, out, **attrs):
        nodes.append(helper.make_node(op, ins, [out], **attrs))
        return out

    F = TensorProto.FLOAT
    H = TensorProto.FLOAT16
    U8 = TensorProto.UINT8
    NI = TensorProto.INT32
    I64 = TensorProto.INT64
    B = TensorProto.BOOL

    # ---- read the 16 sample cells' colours from the input -------------------
    init("sample_nd_idx", sample_nd, np.int64)
    n("GatherND", ["input", "sample_nd_idx"], "sample_planes")   # [10,16] fp32
    n("ArgMax", ["sample_planes"], "in_sample_i64", axis=0, keepdims=1)  # [1,16] int64
    n("Cast", ["in_sample_i64"], "in_sample_u8", to=U8)          # [1,16] uint8

    # ---- score each candidate by # agreeing samples -------------------------
    init("candidate_samples", cand_samples, np.uint8)            # [1,106,16]
    n("Equal", ["candidate_samples", "in_sample_u8"], "matches_b")   # [1,106,16] bool
    n("Cast", ["matches_b"], "matches_h", to=H)                  # [1,106,16] fp16
    n("ReduceSum", ["matches_h"], "scores", axes=[2], keepdims=0)  # [1,106] fp16
    n("ArgMax", ["scores"], "best_id", axis=1, keepdims=0)       # scalar int64

    # ---- recover (mod,length,offset,half) -----------------------------------
    init("candidate_params", cand_params, np.uint8)             # [106,4]
    n("Gather", ["candidate_params", "best_id"], "sel_u8", axis=0)  # [1,4] uint8
    for nm, j in (("mod", 0), ("length", 1), ("offset", 2), ("half", 3)):
        init(f"idx_{nm}", np.array([j], np.int64), np.int64)
        n("Gather", ["sel_u8", f"idx_{nm}"], f"{nm}_u8", axis=1)  # [1,1] uint8
        n("Cast", [f"{nm}_u8"], nm, to=H)                        # [1,1] fp16

    # ---- closed-form rebuild: v=((rr^2+cc^2)%mod)+1 -------------------------
    init("row_grid", row_grid, np.float16)                       # [1,1,21,1]
    init("col_grid", col_grid, np.float16)                       # [1,1,1,21]
    n("Add", ["row_grid", "offset"], "rpo")
    n("Add", ["col_grid", "offset"], "cpo")
    n("Mod", ["rpo", "length"], "rmod", fmod=1)
    n("Mod", ["cpo", "length"], "cmod", fmod=1)
    n("Sub", ["rmod", "half"], "r")                              # [1,1,21,1] fp16
    n("Sub", ["cmod", "half"], "c")                              # [1,1,1,21] fp16
    n("Mul", ["r", "r"], "rr")
    n("Mul", ["c", "c"], "cc")
    n("Add", ["rr", "cc"], "rrcc")                              # [1,1,21,21] fp16
    n("Mod", ["rrcc", "mod"], "pat0", fmod=1)                   # in-grid 0..mod-1
    n("Cast", ["pat0"], "pat0_u8", to=U8)                       # [1,1,21,21] uint8

    # ---- pad pat0 to 30x30 (sentinel 200), Equal -> FREE output --------------
    # The +1 of the colour formula is folded into channel_values (= arange-1 mod
    # 256 = [255,0,1,..,8]): in-grid pat0 in {0..8} -> channels 1..9; channel 0
    # (compare value 255) never appears in-grid; off-grid sentinel 200 matches no
    # channel -> all-zero (grid is exactly 21x21, off-grid must be background-free).
    init("pad30", np.array([0, 0, 0, 0, 0, 0, S - G, S - G], np.int64), np.int64)
    init("pad200", np.array(200, np.uint8), np.uint8)
    n("Pad", ["pat0_u8", "pad30", "pad200"], "labels30")       # [1,1,30,30] uint8
    chv = ((np.arange(10, dtype=np.int64) - 1) % 256).astype(np.uint8)
    init("channel_values", chv.reshape(1, 10, 1, 1), np.uint8)
    n("Equal", ["labels30", "channel_values"], "output")        # [1,10,30,30] bool FREE

    x = helper.make_tensor_value_info("input", F, [1, 10, 30, 30])
    y = helper.make_tensor_value_info("output", B, [1, 10, 30, 30])
    graph = helper.make_graph(nodes, "task017", [x], [y], inits)
    return helper.make_model(graph, ir_version=IR_VERSION,
                             opset_imports=[helper.make_opsetid("", 11)])
