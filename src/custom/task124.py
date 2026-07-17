"""Task 124 — inverse-quantized periodic mask with runtime-colour renderer.

The binary channel-0 crop is quantized directly to uint8 stored values 0/2.
With activation zero-point 1 these mean foreground/background -1/+1.  A
signed two-position difference is encoded by QLinearMatMul at output
zero-point 2 and reduced modulo four, while two six-tap QLinearConv hashes
detect the period-3 controller case.

The reconstructed 10x10 mask feeds the final padded uint8 QLinearConv
directly.  Its runtime ScatterElements weights use weight zero-point 1:
background stored 2 gives +1, the selected colour stored 0 gives -1, and all
other colours stored 1 give 0.  The graph output is therefore the FREE padded
one-hot plane without a Where/Pad/Equal tail.
"""

import numpy as np
import onnx
from onnx import TensorProto, helper

from ._exact import model, tensor


def build(task):
    inits = [
        tensor("ch0_starts", np.asarray([0, 0, 0, 0], dtype=np.int64)),
        tensor("ch0_ends", np.asarray([1, 1, 5, 10], dtype=np.int64)),
        tensor("row0_idx", np.asarray([0], dtype=np.int64)),
        tensor(
            "source_offsets",
            np.asarray(
                [
                    [22, 2, 20, 0, 18],
                    [36, 8, 26, 36, 8],
                    [26, 8, 26, 8, 26],
                    [24, 5, 23, 4, 22],
                ],
                dtype=np.int32,
            ),
        ),
        tensor(
            "ng_background8",
            np.full((1, 1, 1, 8), 2, dtype=np.uint8),
        ),
        tensor(
            "ng_mask_base",
            np.asarray([2] + [1] * 9, dtype=np.uint8).reshape(10, 1, 1, 1),
        ),
        tensor(
            "ng_zero_update", np.zeros((1, 1, 1, 1), dtype=np.uint8)
        ),
        tensor("ng_scale", np.asarray(1.0, dtype=np.float32)),
        tensor("ng_x_zero_point", np.asarray(1, dtype=np.uint8)),
        tensor("ng_y_zero_point", np.asarray(0, dtype=np.uint8)),
        tensor("ten_i32", np.asarray([10], dtype=np.int32)),
        tensor("slice_axes", np.asarray([3], dtype=np.int32)),
        tensor("slice_steps", np.asarray([1], dtype=np.int32)),
        tensor("row2_idx", np.asarray([2], dtype=np.int64)),
        tensor("ng_half_scale", np.asarray(0.5, dtype=np.float32)),
        tensor(
            "ng_diff_weight",
            np.asarray([0, 2], dtype=np.uint8).reshape(2, 1),
        ),
        tensor("ng_diff_y_zero_point", np.asarray(2, dtype=np.uint8)),
        tensor("ng_four_u8", np.asarray(4, dtype=np.uint8)),
        tensor(
            "hash6",
            np.asarray([1, 2, 4, 8, 16, 32], dtype=np.uint8).reshape(
                1, 1, 1, 6
            ),
        ),
    ]
    nodes = [
        helper.make_node(
            "Slice", ["input", "ch0_starts", "ch0_ends"], ["ch0_first5"]
        ),
        helper.make_node(
            "QuantizeLinear",
            ["ch0_first5", "ng_half_scale", ""],
            ["centered5"],
        ),
        helper.make_node(
            "Gather", ["centered5", "row0_idx"], ["row0_4d"], axis=2
        ),
        helper.make_node(
            "ArgMin", ["row0_4d"], ["left0_i64"], axis=3, keepdims=0
        ),
        helper.make_node(
            "Gather", ["centered5", "row2_idx"], ["row2_4d"], axis=2
        ),
        helper.make_node(
            "ArgMin", ["row2_4d"], ["left2_i64"], axis=3, keepdims=0
        ),
        helper.make_node(
            "Cast", ["left0_i64"], ["left0_u8"], to=TensorProto.UINT8
        ),
        helper.make_node(
            "Cast", ["left2_i64"], ["left2_u8"], to=TensorProto.UINT8
        ),
        helper.make_node(
            "Concat", ["left0_u8", "left2_u8"], ["left02_u8"], axis=2
        ),
        helper.make_node(
            "QLinearMatMul",
            [
                "left02_u8",
                "ng_scale",
                "ng_y_zero_point",
                "ng_diff_weight",
                "ng_scale",
                "ng_x_zero_point",
                "ng_scale",
                "ng_diff_y_zero_point",
            ],
            ["shift_rank3_u8"],
        ),
        helper.make_node(
            "Squeeze",
            ["shift_rank3_u8"],
            ["shift_route_u8"],
            axes=[0, 1, 2],
        ),
        helper.make_node(
            "Gather", ["centered5", "slice_steps"], ["p3_rows_a"], axis=2
        ),
        helper.make_node(
            "QLinearConv",
            [
                "p3_rows_a",
                "ng_scale",
                "ng_y_zero_point",
                "hash6",
                "ng_scale",
                "ng_y_zero_point",
                "ng_scale",
                "ng_y_zero_point",
            ],
            ["p3_hash_a"],
            pads=[0, 0, 0, -4],
        ),
        helper.make_node(
            "QLinearConv",
            [
                "centered5",
                "ng_scale",
                "ng_y_zero_point",
                "hash6",
                "ng_scale",
                "ng_y_zero_point",
                "ng_scale",
                "ng_y_zero_point",
            ],
            ["p3_hash_b"],
            pads=[-4, 0, 0, -4],
        ),
        helper.make_node(
            "Equal", ["p3_hash_a", "p3_hash_b"], ["is_p3"]
        ),
        helper.make_node(
            "Squeeze",
            ["is_p3"],
            ["is_p3_scalar"],
            axes=[0, 1, 2, 3],
        ),
        helper.make_node(
            "Where",
            ["is_p3_scalar", "ng_x_zero_point", "shift_route_u8"],
            ["candidate_u8"],
        ),
        helper.make_node(
            "Mod", ["candidate_u8", "ng_four_u8"], ["candidate_mod_u8"]
        ),
        helper.make_node(
            "Cast", ["candidate_mod_u8"], ["candidate_i32"], to=TensorProto.INT32
        ),
        helper.make_node(
            "Gather",
            ["source_offsets", "candidate_i32"],
            ["source_offset"],
            axis=0,
        ),
        helper.make_node(
            "Split",
            ["source_offset"],
            [f"bottom_start_vec_{index}" for index in range(5)],
            axis=0,
            split=[1] * 5,
        ),
        *[
            helper.make_node(
                "Add",
                [f"bottom_start_vec_{index}", "ten_i32"],
                [f"bottom_end_{index}"],
            )
            for index in range(5)
        ],
        helper.make_node(
            "Concat",
            [
                "ng_background8",
                "row0_4d",
                "ng_background8",
                "p3_rows_a",
                "row2_4d",
            ],
            ["fg_pad4d"],
            axis=3,
        ),
        *[
            helper.make_node(
                "Slice",
                [
                    "fg_pad4d",
                    f"bottom_start_vec_{index}",
                    f"bottom_end_{index}",
                    "slice_axes",
                    "slice_steps",
                ],
                [f"bottom_row_{index}"],
            )
            for index in range(5)
        ],
        helper.make_node(
            "Concat",
            ["centered5"] + [f"bottom_row_{index}" for index in range(5)],
            ["ng_centered_mask"],
            axis=2,
        ),
        helper.make_node(
            "ReduceMax", ["input"], ["present_f"], axes=[2, 3], keepdims=1
        ),
        helper.make_node(
            "ArgMax",
            ["present_f"],
            ["color_idx_i64"],
            axis=1,
            keepdims=1,
            select_last_index=1,
        ),
        helper.make_node(
            "ScatterElements",
            ["ng_mask_base", "color_idx_i64", "ng_zero_update"],
            ["ng_wmask"],
            name="ng_wmask",
            axis=0,
        ),
        helper.make_node(
            "QLinearConv",
            [
                "ng_centered_mask",
                "ng_scale",
                "ng_x_zero_point",
                "ng_wmask",
                "ng_scale",
                "ng_x_zero_point",
                "ng_scale",
                "ng_y_zero_point",
            ],
            ["output"],
            name="output",
            pads=[0, 0, 20, 20],
        ),
    ]
    value_infos = [
        *[
            helper.make_tensor_value_info(
                f"bottom_row_{index}", TensorProto.UINT8, [1, 1, 1, 10]
            )
            for index in range(5)
        ],
        helper.make_tensor_value_info(
            "fg_pad4d", TensorProto.UINT8, [1, 1, 1, 46]
        ),
        helper.make_tensor_value_info(
            "p3_hash_a", TensorProto.UINT8, [1, 1, 1, 1]
        ),
        helper.make_tensor_value_info(
            "p3_hash_b", TensorProto.UINT8, [1, 1, 1, 1]
        ),
    ]
    result = model(
        "task124_live_exact",
        nodes,
        inits,
        output_dtype=TensorProto.UINT8,
        opset=12,
        value_infos=value_infos,
    )
    return onnx.shape_inference.infer_shapes(result, strict_mode=True)
