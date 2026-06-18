"""task050 (ARC 253bf280): cyan endpoints; fill green strictly between any two
cyan sharing a row or a column.

Rule: a bg cell becomes green iff (cyan present to its left AND right in the
same row) OR (cyan present above AND below in the same column). Cyan stays cyan.

Encoding: work on the gen-bounded 15x15 active canvas. Compute "has cyan strictly
left/right/up/down" via exclusive CumSum (fwd) + row/col totals (reverse side via
ReduceSum, avoiding 2 extra cumsum planes). Build a single colour-index plane
L = 8*cyan + 3*green, Pad to 30x30, then Equal against an arange[1,10,1,1] to route
the 10-channel one-hot expansion straight into the FREE bool output.
"""
import numpy as np
import onnx
from onnx import helper, TensorProto, numpy_helper as nh

W = 15  # generator caps width/height at 15

def _init(name, arr):
    return nh.from_array(np.asarray(arr), name=name)

def build(task=None):
    nodes = []
    inits = []

    # --- slice cyan (ch8) and bg (ch0) from the free input, to 15x15 ---
    inits.append(_init("axes_chw", np.array([1, 2, 3], dtype=np.int64)))
    inits.append(_init("c8_st", np.array([8, 0, 0], dtype=np.int64)))
    inits.append(_init("c8_en", np.array([9, W, W], dtype=np.int64)))
    nodes.append(helper.make_node("Slice", ["input", "c8_st", "c8_en", "axes_chw"], ["cyan_f"]))

    # cast cyan to fp16 for cheap cumsum work
    nodes.append(helper.make_node("Cast", ["cyan_f"], ["cyan_h"], to=TensorProto.FLOAT16))

    # in-grid mask as a separable rectangle from FREE input occupancy profiles
    # (any channel present per row / per col), sliced to the 15x15 working canvas.
    nodes.append(helper.make_node("ReduceMax", ["input"], ["rowany_f"], axes=[1, 3], keepdims=1))
    nodes.append(helper.make_node("ReduceMax", ["input"], ["colany_f"], axes=[1, 2], keepdims=1))
    inits.append(_init("r_st", np.array([0, 0, 0, 0], dtype=np.int64)))
    inits.append(_init("r_en", np.array([1, 1, W, 1], dtype=np.int64)))
    inits.append(_init("r_ax", np.array([0, 1, 2, 3], dtype=np.int64)))
    inits.append(_init("c_en", np.array([1, 1, 1, W], dtype=np.int64)))
    nodes.append(helper.make_node("Slice", ["rowany_f", "r_st", "r_en", "r_ax"], ["rowany15_f"]))
    nodes.append(helper.make_node("Slice", ["colany_f", "r_st", "c_en", "r_ax"], ["colany15_f"]))
    # off-grid = NOT(rowany AND colany) = (NOT rowany) OR (NOT colany); compute it
    # straight from the tiny [1,1,15,1] / [1,1,1,15] vectors so the only 15x15 plane
    # materialised is the broadcast offgrid mask itself.
    nodes.append(helper.make_node("Cast", ["rowany15_f"], ["rowany_b"], to=TensorProto.BOOL))
    nodes.append(helper.make_node("Cast", ["colany15_f"], ["colany_b"], to=TensorProto.BOOL))
    nodes.append(helper.make_node("Not", ["rowany_b"], ["nrow_b"]))
    nodes.append(helper.make_node("Not", ["colany_b"], ["ncol_b"]))
    nodes.append(helper.make_node("Or", ["nrow_b", "ncol_b"], ["offgrid_b"]))

    inits.append(_init("axis_r", np.array(2, dtype=np.int64)))  # rows axis (vertical)
    inits.append(_init("axis_c", np.array(3, dtype=np.int64)))  # cols axis (horizontal)

    # ONE exclusive forward cumsum per axis = # cyan strictly before this cell;
    # plus the per-line total (tiny). A bg cell is strictly between two cyan in its
    # line iff 0 < before < total (cyan strictly left AND right, since
    # before+after = total at a bg cell). Only 2 cumsums needed.
    nodes.append(helper.make_node("CumSum", ["cyan_h", "axis_c"], ["cleft_h"], exclusive=1))
    nodes.append(helper.make_node("CumSum", ["cyan_h", "axis_r"], ["cup_h"], exclusive=1))
    inits.append(_init("axes3", np.array([3], dtype=np.int64)))
    inits.append(_init("axes2", np.array([2], dtype=np.int64)))
    nodes.append(helper.make_node("ReduceSum", ["cyan_h", "axes3"], ["rowtot_h"], keepdims=1))
    nodes.append(helper.make_node("ReduceSum", ["cyan_h", "axes2"], ["coltot_h"], keepdims=1))

    inits.append(_init("zero_h", np.array(0, dtype=np.float16).reshape(1, 1, 1, 1)))
    nodes.append(helper.make_node("Greater", ["cleft_h", "zero_h"], ["rl"]))
    nodes.append(helper.make_node("Less", ["cleft_h", "rowtot_h"], ["rr"]))
    nodes.append(helper.make_node("And", ["rl", "rr"], ["brow"]))
    nodes.append(helper.make_node("Greater", ["cup_h", "zero_h"], ["cu"]))
    nodes.append(helper.make_node("Less", ["cup_h", "coltot_h"], ["cd"]))
    nodes.append(helper.make_node("And", ["cu", "cd"], ["bcol"]))
    nodes.append(helper.make_node("Or", ["brow", "bcol"], ["green_b"]))
    # note: "between" can only be true on bg cells (never on cyan: the generator
    # never places 3 collinear cyan), and the Where chain below gives cyan priority,
    # so no explicit bg-AND is needed.
    nodes.append(helper.make_node("Cast", ["cyan_f"], ["cyan_b"], to=TensorProto.BOOL))

    # Build a single uint8 colour-index plane L (15x15) via a Where priority chain:
    # cyan->8, else green->3, else offgrid->99, else bg->0.  All values disjoint and
    # distinct from each other; uint8 keeps the plane at 1 byte/elem.
    inits.append(_init("u8", np.array(0, dtype=np.uint8).reshape(1, 1, 1, 1)))
    inits.append(_init("u3", np.array(3, dtype=np.uint8).reshape(1, 1, 1, 1)))
    inits.append(_init("u8v", np.array(8, dtype=np.uint8).reshape(1, 1, 1, 1)))
    inits.append(_init("u99", np.array(99, dtype=np.uint8).reshape(1, 1, 1, 1)))
    nodes.append(helper.make_node("Where", ["offgrid_b", "u99", "u8"], ["L_a"]))
    nodes.append(helper.make_node("Where", ["green_b", "u3", "L_a"], ["L_b"]))
    nodes.append(helper.make_node("Where", ["cyan_b", "u8v", "L_b"], ["L_u"]))

    # Pad L to 30x30 with sentinel 99 (off-grid -> no channel matches)
    inits.append(_init("pad_to_30", np.array([0, 0, 0, 0, 0, 0, 30 - W, 30 - W], dtype=np.int64)))
    inits.append(_init("padval", np.array(99, dtype=np.uint8)))
    nodes.append(helper.make_node("Pad", ["L_u", "pad_to_30", "padval"], ["L30_u"], mode="constant"))

    # Equal against uint8 arange -> free bool output [1,10,30,30]
    arange = np.arange(10, dtype=np.uint8).reshape(1, 10, 1, 1)
    inits.append(_init("arange10", arange))
    nodes.append(helper.make_node("Equal", ["L30_u", "arange10"], ["output"]))

    out_vi = helper.make_tensor_value_info("output", TensorProto.BOOL, [1, 10, 30, 30])
    in_vi = helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 10, 30, 30])
    graph = helper.make_graph(nodes, "task050", [in_vi], [out_vi], inits)
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    model.ir_version = 8
    return model


if __name__ == "__main__":
    import onnx as _o
    _o.save(build(), "/tmp/task050_test.onnx")
    print("saved")
