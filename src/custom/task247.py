"""task247 (ARC-AGI a3325580): find the box(es) with the most pixels; output a
grid `count` rows tall and `W` cols wide where W = number of boxes tied at the
max pixel count, ordered left-to-right by box column, each column solid-filled
with that box's color.

Each box is a distinct color (colors 1-9). Per-color pixel count = box size.
M = max count (in [3,6]); winners = colors whose count == M (these are exactly
the num_max "max boxes", W = num_max in [1,3]). Output column c = winner ranked
c-th by min-col; every one of the M rows is identical.

Encoding (all-narrow-dtype, tiny canvas):
  slice -> per-color count -> iswin -> min-col -> rank -> place into 6x3 uint8
  canvas -> Pad to 10x30x30 uint8 output.
"""

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

NCOL = 9      # color channels 1..9
MAXR = 9      # max count M; fresh gen gives [3,6] but hardcoded train examples reach 9
MAXW = 3      # max winners W = num_max in [1,3]


def _vi(name, dtype, shape):
    return helper.make_tensor_value_info(name, dtype, shape)


def build(task=None):
    inits = []
    nodes = []
    vis = []

    def const(name, arr):
        inits.append(numpy_helper.from_array(np.asarray(arr), name))

    # --- reductions directly on the FREE input (no 3600B slice) ---
    # per-channel pixel count over full input [1,10,1,1]
    nodes.append(helper.make_node("ReduceSum", ["input"], ["cnt10"], axes=[2, 3], keepdims=1))
    vis.append(_vi("cnt10", TensorProto.FLOAT, [1, 10, 1, 1]))
    # drop bg channel 0 -> cnt[1,9,1,1]
    const("c1s", np.array([1], dtype=np.int64))
    const("c1e", np.array([10], dtype=np.int64))
    const("c1a", np.array([1], dtype=np.int64))
    nodes.append(helper.make_node("Slice", ["cnt10", "c1s", "c1e", "c1a"], ["cnt"]))
    vis.append(_vi("cnt", TensorProto.FLOAT, [1, 9, 1, 1]))
    # M = max count
    nodes.append(helper.make_node("ReduceMax", ["cnt"], ["M"], axes=[1], keepdims=1))
    vis.append(_vi("M", TensorProto.FLOAT, [1, 1, 1, 1]))
    # iswin[1,9,1,1] = cnt==M  (exact integer equality)
    nodes.append(helper.make_node("Equal", ["cnt", "M"], ["iswin"]))
    vis.append(_vi("iswin", TensorProto.BOOL, [1, 9, 1, 1]))

    # per-channel, per-column occupancy over the free input [1,10,1,30]
    nodes.append(helper.make_node("ReduceMax", ["input"], ["occ10"], axes=[2], keepdims=1))
    vis.append(_vi("occ10", TensorProto.FLOAT, [1, 10, 1, 30]))
    # min-col = first occupied col via ArgMax directly on occ10 (trailing zero
    # cols never beat an occupied col). Then drop bg channel.
    nodes.append(helper.make_node("ArgMax", ["occ10"], ["mc10"], axis=3, keepdims=1))
    vis.append(_vi("mc10", TensorProto.INT64, [1, 10, 1, 1]))
    nodes.append(helper.make_node("Slice", ["mc10", "c1s", "c1e", "c1a"], ["mc_i"]))
    vis.append(_vi("mc_i", TensorProto.INT64, [1, 9, 1, 1]))
    nodes.append(helper.make_node("Cast", ["mc_i"], ["mc"], to=TensorProto.FLOAT))
    vis.append(_vi("mc", TensorProto.FLOAT, [1, 9, 1, 1]))
    # mcT[1,1,9,1]
    nodes.append(helper.make_node("Transpose", ["mc"], ["mcT"], perm=[0, 2, 1, 3]))
    vis.append(_vi("mcT", TensorProto.FLOAT, [1, 1, 9, 1]))
    # lt[i,j] = mc[i] < mc[j]   shape [1,9,9,1]
    nodes.append(helper.make_node("Less", ["mc", "mcT"], ["lt"]))
    vis.append(_vi("lt", TensorProto.BOOL, [1, 9, 9, 1]))
    # only winners contribute as i: lt AND iswin (broadcast iswin over axis2)
    nodes.append(helper.make_node("And", ["lt", "iswin"], ["ltw"]))
    vis.append(_vi("ltw", TensorProto.BOOL, [1, 9, 9, 1]))
    nodes.append(helper.make_node("Cast", ["ltw"], ["ltwf"], to=TensorProto.FLOAT16))
    vis.append(_vi("ltwf", TensorProto.FLOAT16, [1, 9, 9, 1]))
    # rank[j] = #winners i with mc[i]<mc[j]  -> reduce over axis1 (i)
    nodes.append(helper.make_node("ReduceSum", ["ltwf"], ["rank0"], axes=[1], keepdims=1))
    vis.append(_vi("rank0", TensorProto.FLOAT16, [1, 1, 9, 1]))
    nodes.append(helper.make_node("Transpose", ["rank0"], ["rank"], perm=[0, 2, 1, 3]))
    vis.append(_vi("rank", TensorProto.FLOAT16, [1, 9, 1, 1]))

    # colcmp[1,9,1,3]: color j at output column c  <=> rank[j]==c
    const("colramp", np.arange(MAXW, dtype=np.float16).reshape(1, 1, 1, MAXW))
    nodes.append(helper.make_node("Equal", ["rank", "colramp"], ["atcol"]))
    vis.append(_vi("atcol", TensorProto.BOOL, [1, 9, 1, MAXW]))
    # place[1,9,1,3] = iswin AND atcol -> uint8 for the Where body
    nodes.append(helper.make_node("And", ["atcol", "iswin"], ["place"]))
    vis.append(_vi("place", TensorProto.BOOL, [1, 9, 1, MAXW]))
    nodes.append(helper.make_node("Cast", ["place"], ["placeu"], to=TensorProto.UINT8))
    vis.append(_vi("placeu", TensorProto.UINT8, [1, 9, 1, MAXW]))

    # rows: r < M for r in 0..MAXR-1  -> rowmask[1,1,MAXR,1]
    const("rowramp", np.arange(MAXR, dtype=np.float32).reshape(1, 1, MAXR, 1))
    nodes.append(helper.make_node("Less", ["rowramp", "M"], ["rowmask"]))
    vis.append(_vi("rowmask", TensorProto.BOOL, [1, 1, MAXR, 1]))

    # combine: out[ch,r,c] = rowmask[r] AND place[ch,c]   -> [1,9,MAXR,MAXW] uint8
    # Where(rowmask, placeu, 0) broadcasts [1,1,MAXR,1] x [1,9,1,MAXW] -> [1,9,MAXR,MAXW]
    const("z_u8", np.array(0, dtype=np.uint8))
    nodes.append(helper.make_node("Where", ["rowmask", "placeu", "z_u8"], ["bodyu"]))
    vis.append(_vi("bodyu", TensorProto.UINT8, [1, 9, MAXR, MAXW]))

    # Pad: insert bg channel 0 at front, pad rows MAXR->30, cols MAXW->30
    const("pads", np.array([0, 1, 0, 0, 0, 0, 30 - MAXR, 30 - MAXW], dtype=np.int64))
    const("padval", np.array(0, dtype=np.uint8))
    nodes.append(helper.make_node("Pad", ["bodyu", "pads", "padval"], ["output"], mode="constant"))

    out_vi = _vi("output", TensorProto.UINT8, [1, 10, 30, 30])
    in_vi = _vi("input", TensorProto.FLOAT, [1, 10, 30, 30])

    graph = helper.make_graph(nodes, "task247", [in_vi], [out_vi], inits, value_info=vis)
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 11)])
    model.ir_version = 10
    return model


if __name__ == "__main__":
    onnx.save(build(), "/tmp/task247_test.onnx")
    print("saved")
