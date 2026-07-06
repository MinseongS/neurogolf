"""Task 368 probe: write recoloured gray cells directly into the final output.

The incumbent scatters a 10x10 uint8 label grid, pads it to 30x30, then uses a
final Equal to produce the one-hot output. This probe instead scatters two
scalar updates per target cell into the free input tensor:

- gray channel 5 -> 0
- target colour channel -> 1

It is expected to trade the 900B final label plane for dynamic ScatterND indices.
"""
import numpy as np
from onnx import TensorProto, helper

from src.custom import task368
from src.custom._exact import tensor


def _drop_tail(model):
    old_outputs = {
        "all_pos_i32",
        "all_colors_u8",
        "placed_color_flat_u8",
        "placed_color_u8",
        "placed_color_30_u8",
        "output",
    }
    kept = [node for node in model.graph.node if not any(o in old_outputs for o in node.output)]
    del model.graph.node[:]
    model.graph.node.extend(kept)

    kept_vi = [vi for vi in model.graph.value_info if vi.name not in old_outputs]
    del model.graph.value_info[:]
    model.graph.value_info.extend(kept_vi)


def build(task):
    m = task368.build(task)
    _drop_tail(m)

    m.graph.initializer.extend(
        [
            tensor("shape36_1_i64", np.array([36, 1], np.int64)),
            tensor("five_i64", np.array([5], np.int64)),
            tensor("zero_i64", np.array([0], np.int64)),
            tensor("ninehundred_i32", np.array([900], np.int32)),
            tensor("thirty_i32_tail", np.array([30], np.int32)),
            tensor("scatter_updates_f", np.r_[np.zeros(36, np.float32), np.ones(36, np.float32)]),
        ]
    )

    tail_nodes = [
        helper.make_node(
            "Concat",
            ["place1_pos_i32", "place2_pos_i32", "place3_pos_i32"],
            ["gray_pos10_i32"],
            axis=0,
        ),
        helper.make_node(
            "Concat",
            ["src_colors_u8", "src_colors_u8", "src_colors_u8"],
            ["gray_colors_u8"],
            axis=0,
        ),
        helper.make_node("Div", ["gray_pos10_i32", "ten_i32"], ["gray_rows_i32"]),
        helper.make_node("Mod", ["gray_pos10_i32", "ten_i32"], ["gray_cols_i32"], fmod=0),
        helper.make_node("Mul", ["gray_rows_i32", "thirty_i32_tail"], ["gray_row30_i32"]),
        helper.make_node("Add", ["gray_row30_i32", "gray_cols_i32"], ["gray_pos30_i32"]),
        helper.make_node("Cast", ["gray_rows_i32"], ["gray_rows_i64"], to=TensorProto.INT64),
        helper.make_node("Cast", ["gray_cols_i32"], ["gray_cols_i64"], to=TensorProto.INT64),
        helper.make_node("Cast", ["gray_colors_u8"], ["gray_colors_i64"], to=TensorProto.INT64),
        helper.make_node("Mul", ["gray_rows_i64", "zero_i64"], ["zero36_i64"]),
        helper.make_node("Add", ["zero36_i64", "five_i64"], ["five36_i64"]),
        helper.make_node("Reshape", ["zero36_i64", "shape36_1_i64"], ["zero36_col_i64"]),
        helper.make_node("Reshape", ["five36_i64", "shape36_1_i64"], ["five36_col_i64"]),
        helper.make_node("Reshape", ["gray_colors_i64", "shape36_1_i64"], ["color36_col_i64"]),
        helper.make_node("Reshape", ["gray_rows_i64", "shape36_1_i64"], ["row36_col_i64"]),
        helper.make_node("Reshape", ["gray_cols_i64", "shape36_1_i64"], ["col36_col_i64"]),
        helper.make_node(
            "Concat",
            ["zero36_col_i64", "five36_col_i64", "row36_col_i64", "col36_col_i64"],
            ["gray_off_indices_i64"],
            axis=1,
        ),
        helper.make_node(
            "Concat",
            ["zero36_col_i64", "color36_col_i64", "row36_col_i64", "col36_col_i64"],
            ["color_on_indices_i64"],
            axis=1,
        ),
        helper.make_node(
            "Concat",
            ["gray_off_indices_i64", "color_on_indices_i64"],
            ["scatter_indices_i64"],
            axis=0,
        ),
        helper.make_node("ScatterND", ["input", "scatter_indices_i64", "scatter_updates_f"], ["output"]),
    ]
    m.graph.node.extend(tail_nodes)

    m.graph.value_info.extend(
        [
            helper.make_tensor_value_info("gray_pos10_i32", TensorProto.INT32, [36]),
            helper.make_tensor_value_info("gray_colors_u8", TensorProto.UINT8, [36]),
            helper.make_tensor_value_info("gray_rows_i32", TensorProto.INT32, [36]),
            helper.make_tensor_value_info("gray_cols_i32", TensorProto.INT32, [36]),
            helper.make_tensor_value_info("gray_pos30_i32", TensorProto.INT32, [36]),
            helper.make_tensor_value_info("gray_rows_i64", TensorProto.INT64, [36]),
            helper.make_tensor_value_info("gray_cols_i64", TensorProto.INT64, [36]),
            helper.make_tensor_value_info("gray_colors_i64", TensorProto.INT64, [36]),
            helper.make_tensor_value_info("zero36_i64", TensorProto.INT64, [36]),
            helper.make_tensor_value_info("five36_i64", TensorProto.INT64, [36]),
            helper.make_tensor_value_info("zero36_col_i64", TensorProto.INT64, [36, 1]),
            helper.make_tensor_value_info("five36_col_i64", TensorProto.INT64, [36, 1]),
            helper.make_tensor_value_info("color36_col_i64", TensorProto.INT64, [36, 1]),
            helper.make_tensor_value_info("row36_col_i64", TensorProto.INT64, [36, 1]),
            helper.make_tensor_value_info("col36_col_i64", TensorProto.INT64, [36, 1]),
            helper.make_tensor_value_info("gray_off_indices_i64", TensorProto.INT64, [36, 4]),
            helper.make_tensor_value_info("color_on_indices_i64", TensorProto.INT64, [36, 4]),
            helper.make_tensor_value_info("scatter_indices_i64", TensorProto.INT64, [72, 4]),
        ]
    )
    m.graph.output[0].type.tensor_type.elem_type = TensorProto.FLOAT
    return m
