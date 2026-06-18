"""task134 (ARC-AGI 5ad4f10b) — magnified conway-sprite identification.

Rule (from the ARC-GEN generator):
  * A grid (20-30 sq) holds scattered single pixels of `color` (~5% density)
    plus ONE "mega sprite": a 3x3 conway_sprite (every row & col occupied)
    drawn in `megacolor`, magnified by `magnifier` m in [2,6] -> each ON cell
    of the 3x3 becomes a solid m x m block. Offset places it anywhere in-grid.
  * OUTPUT is a 3x3 grid: output[r][c] = `color` for every ON cell (r,c) of the
    conway 3x3 pattern, background (0) elsewhere.  (Output recolours the mega
    pattern with the SCATTERED colour, not megacolor.)

Recovery (input only):
  * Exactly two nonzero colours: mega vs scatter.  The mega channel is the
    high-DENSITY one: density = cnt / (nrows*ncols) where nrows/ncols are the
    counts of occupied rows/cols.  For mega (contiguous square span 3m) this is
    on_cells/9 in [0.33,0.78]; for scatter it is <=0.28 (0/20000 overlap).
    Channel-0 (background) must be zeroed before the argmax.
  * mega bbox: rmin=ArgMax(row_any), cmin=ArgMax(col_any); span = nrows = 3m,
    so magnifier m = nrows/3.  rowoffset=rmin, coloffset=cmin.
  * scatter colour digit = the OTHER nonzero channel index (argmax of cnt with
    ch0 and mega zeroed).
  * 3x3 pattern: pattern[r][c] = mega_plane[rmin+r*m][cmin+c*m]  (top-left of
    each m x m block; solid when ON, empty when OFF).  Two chained Gathers.

Output: build [1,10,3,3] one-hot (ON->scatter channel, OFF->ch0), Cast uint8,
Pad to [1,10,30,30] with 0 (outside the 3x3 the target is all-zero); declare
output uint8 (harness scores out>0).
"""

import numpy as np
from onnx import helper, numpy_helper, TensorProto

from ..harness import IR_VERSION


def build(task):
    inits, nodes = [], []

    def init(name, arr, dt):
        inits.append(numpy_helper.from_array(np.ascontiguousarray(arr, dtype=dt), name))
        return name

    def n(op, ins, out, **attrs):
        nodes.append(helper.make_node(op, ins, [out], **attrs))
        return out

    F = TensorProto.FLOAT
    U8 = TensorProto.UINT8
    I64 = TensorProto.INT64

    # ---- per-channel reductions ----
    n("ReduceMax", ["input"], "row_any", axes=[3], keepdims=1)   # [1,10,30,1]
    n("ReduceMax", ["input"], "col_any", axes=[2], keepdims=1)   # [1,10,1,30]
    n("ReduceSum", ["input"], "cnt", axes=[2, 3], keepdims=1)    # [1,10,1,1]
    n("ReduceSum", ["row_any"], "nrows", axes=[2], keepdims=1)   # [1,10,1,1]
    n("ReduceSum", ["col_any"], "ncols", axes=[3], keepdims=1)   # [1,10,1,1]

    # ---- density = cnt / (nrows*ncols), ch0 zeroed ----
    n("Mul", ["nrows", "ncols"], "area")                          # [1,10,1,1]
    init("eps", np.array(1e-6, np.float32), np.float32)
    n("Add", ["area", "eps"], "area_s")
    n("Div", ["cnt", "area_s"], "density")
    # mask out channel 0 (background) so it cannot win argmax
    not0 = init("not_ch0", np.array([0] + [1] * 9, np.float32).reshape(1, 10, 1, 1), np.float32)
    n("Mul", ["density", not0], "density_m")
    n("ArgMax", ["density_m"], "mega_idx4", axis=1, keepdims=1)   # [1,1,1,1] int64

    # ---- scatter colour = other nonzero channel (argmax cnt w/ ch0 & mega off) ----
    # build a [1,10,1,1] mask that is 0 at ch0 and at mega channel, else 1.
    chan = init("chan_f", np.arange(10, dtype=np.float32).reshape(1, 10, 1, 1), np.float32)
    n("Cast", ["mega_idx4"], "mega_f", to=F)                      # [1,1,1,1]
    n("Equal", [chan, "mega_f"], "is_mega")                       # [1,10,1,1] bool
    n("Cast", ["is_mega"], "is_mega_f", to=F)
    one = init("one_f", np.array(1.0, np.float32), np.float32)
    n("Sub", [one, "is_mega_f"], "keep_mega")                     # 0 at mega else 1
    n("Mul", ["keep_mega", not0], "scat_keep")                    # also zero ch0
    n("Mul", ["cnt", "scat_keep"], "scat_cnt")
    n("ArgMax", ["scat_cnt"], "scat_idx4", axis=1, keepdims=1)    # [1,1,1,1] int64

    # ---- mega bbox + magnifier (scalars) ----
    n("ArgMax", ["row_any"], "rmin_ch", axis=2, keepdims=1)       # [1,10,1,1] int64
    n("ArgMax", ["col_any"], "cmin_ch", axis=3, keepdims=1)       # [1,10,1,1] int64
    # gather mega channel's values (squeeze channel axis -> scalar index)
    init("sh1", np.array([1], np.int64), np.int64)
    n("Reshape", ["mega_idx4", "sh1"], "mega_idx")               # [1] int64
    # rmin/cmin/nrows are [1,10,1,1]; gather along channel axis=1
    n("Gather", ["rmin_ch", "mega_idx"], "rmin_g", axis=1)        # [1,1,1,1]
    n("Gather", ["cmin_ch", "mega_idx"], "cmin_g", axis=1)
    n("Gather", ["nrows", "mega_idx"], "span_g", axis=1)          # span = 3m (float)
    # mag = span/3
    init("c3f", np.array(3.0, np.float32), np.float32)
    n("Div", ["span_g", "c3f"], "mag_f")                          # [1,1,1,1] float
    n("Reshape", ["mag_f", "sh1"], "mag_s")                       # [1] float
    n("Cast", ["mag_s"], "mag_i", to=I64)                        # [1] int64

    # rmin/cmin scalars (int64 [1])
    n("Reshape", ["rmin_g", "sh1"], "rmin_s")                    # [1]
    n("Reshape", ["cmin_g", "sh1"], "cmin_s")

    # row indices = rmin + mag*[0,1,2]  (and likewise cols)
    init("k012", np.array([0, 1, 2], np.int64), np.int64)
    n("Mul", ["mag_i", "k012"], "moff")                          # [3] = mag*[0,1,2]
    n("Add", ["rmin_s", "moff"], "row_idx")                      # [3]
    n("Add", ["cmin_s", "moff"], "col_idx")                      # [3]

    # ---- sample 3x3 pattern from mega plane ----
    n("Gather", ["input", "mega_idx"], "mega_plane", axis=1)     # [1,1,30,30] fp32
    n("Gather", ["mega_plane", "row_idx"], "prows", axis=2)      # [1,1,3,30]
    n("Gather", ["prows", "col_idx"], "pattern", axis=3)         # [1,1,3,3] (0/1)

    # ---- colour-index = pattern * scatter_idx ----
    n("Cast", ["scat_idx4"], "scat_f", to=F)                     # [1,1,1,1]
    n("Mul", ["pattern", "scat_f"], "L3")                        # [1,1,3,3] index

    # one-hot over channels -> [1,10,3,3] bool, cast uint8, pad to 30x30
    n("Equal", ["L3", chan], "oh3")                              # [1,10,3,3] bool
    n("Cast", ["oh3"], "oh3_u8", to=U8)
    init("pads", np.array([0, 0, 0, 0, 0, 0, 27, 27], np.int64), np.int64)
    init("zu8", np.array(0, np.uint8), np.uint8)
    n("Pad", ["oh3_u8", "pads", "zu8"], "output", mode="constant")

    out_vi = helper.make_tensor_value_info("output", U8, [1, 10, 30, 30])
    in_vi = helper.make_tensor_value_info("input", F, [1, 10, 30, 30])
    graph = helper.make_graph(nodes, "task134", [in_vi], [out_vi], inits)
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 11)])
    model.ir_version = IR_VERSION
    return model
