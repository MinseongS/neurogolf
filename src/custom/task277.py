"""Task 277 (ARC-AGI b230c067) — classify repeated cyan sprites.

Generator rule, verified from `/tmp/arc-gen/tasks/task_b230c067.py`:

* A 10x10 grid contains three cyan copies derived from one sprite.
* Two copies are the full sprite (`idx=0`) and must become blue(1).
* One copy is made by deleting one column from the full sprite (`idx=1`) and
  must become red(2).
* The full sprite width is 3 or 4 in fresh data; the ARC validation examples
  also include a width-5 full sprite, so this builder keeps the h3_w5 path.

The deployed public graph is already very small.  This file reconstructs the
same semantic algorithm in source form so the task is under local builder
control: find all valid all-cyan bounding boxes for widths 3/4/5 and heights
1..4, decide the active full width, cover full-width boxes as blue, and paint
remaining cyan cells red.
"""

import numpy as np
from onnx import TensorProto, helper, numpy_helper

from ..harness import IR_VERSION


S = 30
A = 10


def build(task):
    inits = []
    nodes = []
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
    U8 = TensorProto.UINT8

    init("cyan_f_starts", np.array([8, 0, 0], np.int64), np.int64)
    init("cyan_f_ends", np.array([9, A, A], np.int64), np.int64)
    init("slice_axes", np.array([1, 2, 3], np.int64), np.int64)
    init("one_u8", np.array(1, np.uint8), np.uint8)
    init("zero_u8", np.array(0, np.uint8), np.uint8)

    n("Slice", ["input", "cyan_f_starts", "cyan_f_ends", "slice_axes"], "cyan_f")
    n("Cast", ["cyan_f"], "cyan_u8", to=U8)

    # Top-left validity for an all-cyan h x w rectangle:
    # row_bad_w_h = any horizontal w-window with a missing cyan, expanded over h.
    for w in (3, 4, 5):
        n("MaxPool", ["cyan_u8"], f"row_any_w{w}", kernel_shape=[1, w])
        n("Sub", ["one_u8", f"row_any_w{w}"], f"row_bad_w{w}_h1")
        hs = (1, 2, 3, 4) if w in (3, 4) else (3,)
        for h in hs:
            if h == 1:
                continue
            n("MaxPool", [f"row_bad_w{w}_h1"], f"row_bad_w{w}_h{h}", kernel_shape=[h, 1])

    # Symmetric column-validity terms.
    n("Sub", ["one_u8", "cyan_u8"], "col_bad_h1")
    for h in (2, 3, 4):
        n("MaxPool", ["cyan_u8"], f"col_any_h{h}", kernel_shape=[h, 1])
        n("Sub", ["one_u8", f"col_any_h{h}"], f"col_bad_h{h}")

    for h in (1, 2, 3, 4):
        for w in (3, 4):
            n("MaxPool", [f"col_bad_h{h}"], f"col_bad_h{h}_w{w}", kernel_shape=[1, w])
    n("MaxPool", ["col_bad_h3"], "col_bad_h3_w5", kernel_shape=[1, 5])

    # Combine row/col constraints into h_w_valid top-left masks.
    for h in (1, 2, 3, 4):
        for w in (3, 4):
            if h == 1:
                n("Sub", ["one_u8", f"col_bad_h1_w{w}"], f"h1_w{w}_valid")
            else:
                n("Max", [f"row_bad_w{w}_h{h}", f"col_bad_h{h}_w{w}"], f"h{h}_w{w}_bad")
                n("Sub", ["one_u8", f"h{h}_w{w}_bad"], f"h{h}_w{w}_valid")
    n("Max", ["row_bad_w5_h3", "col_bad_h3_w5"], "h3_w5_bad")
    n("Sub", ["one_u8", "h3_w5_bad"], "h3_w5_valid")

    # Presence of width-4/5 full sprites decides which cover width is active.
    for h in (1, 2, 3, 4):
        n("ReduceMax", [f"h{h}_w4_valid"], f"h{h}_w4_has", keepdims=0)
    n("Max", ["h1_w4_has", "h2_w4_has", "h3_w4_has", "h4_w4_has"], "has_w4")
    n("ReduceMax", ["h3_w5_valid"], "h3_w5_has", keepdims=0)
    n("Sub", ["one_u8", "h3_w5_has"], "not_has_w5")
    n("Sub", ["one_u8", "has_w4"], "not_has_w4")
    n("Min", ["has_w4", "not_has_w5"], "active_w4")
    n("Min", ["not_has_w4", "not_has_w5"], "active_w3")

    # Expand top-left masks back over each candidate rectangle to cover all
    # pixels belonging to full-width sprites.
    for h in (1, 2, 3, 4):
        for w in (3, 4):
            n(
                "MaxPool",
                [f"h{h}_w{w}_valid"],
                f"h{h}_w{w}_cover",
                kernel_shape=[h, w],
                pads=[h - 1, w - 1, h - 1, w - 1],
            )
    n("MaxPool", ["h3_w5_valid"], "h3_w5_cover", kernel_shape=[3, 5], pads=[2, 4, 2, 4])

    n("Max", ["h1_w3_cover", "h2_w3_cover", "h3_w3_cover", "h4_w3_cover"], "cover_w3")
    n("Max", ["h1_w4_cover", "h2_w4_cover", "h3_w4_cover", "h4_w4_cover"], "cover_w4")
    n("Min", ["cover_w3", "active_w3"], "active_cover_w3")
    n("Min", ["cover_w4", "active_w4"], "active_cover_w4")
    n("Max", ["h3_w5_cover", "active_cover_w4", "active_cover_w3"], "blue_cover")

    n("Min", ["cyan_u8", "blue_cover"], "blue")
    n("Sub", ["cyan_u8", "blue"], "red")

    # Output channels inside active 10x10: background, blue, red.  Pad the
    # remaining channels/spatial area with zeros; output tensor itself is free.
    n("Concat", ["col_bad_h1", "blue", "red"], "out3", axis=1)
    init("pad_to_output", np.array([0, 0, 0, 0, 0, 7, S - A, S - A], np.int64), np.int64)
    n("Pad", ["out3", "pad_to_output", "zero_u8"], "output", mode="constant")

    x = helper.make_tensor_value_info("input", F, [1, 10, S, S])
    y = helper.make_tensor_value_info("output", U8, [1, 10, S, S])
    graph = helper.make_graph(nodes, "task277", [x], [y], inits)
    return helper.make_model(
        graph,
        ir_version=IR_VERSION,
        opset_imports=[helper.make_opsetid("", 21)],
    )
