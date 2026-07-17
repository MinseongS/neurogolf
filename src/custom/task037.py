"""Task 037 — six exact diagonal predicates folded into the FREE output.

Each selected colour is represented by ``(lo, span, base, sign, colour)``.
For output channel k and position (r, c), its quadratic predicate is

    (r-lo)(lo+span-r) + 1/4
    - 10(c-base-sign*r)^2
    - 10[k>0](k-colour)^2.

It is positive exactly on that coloured segment and negative elsewhere.  The
generator's accepted segments do not overlap, so the product of all six
predicates is positive on background and negative on one coloured segment;
the channel sign vector flips non-background channels.  A single terminal
Einsum performs the product directly in the uncounted graph output.
"""

from __future__ import annotations

import string

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

from ._exact import model


# weight, descriptor-left, descriptor-right, row-power, column-power,
# channel-basis.  Descriptor rows are [1, lo, span, base, sign, colour].
_TERMS = [
    (0.25, 0, 0, 0, 0, 0),
    (-1.0, 1, 1, 0, 0, 0),
    (-1.0, 1, 2, 0, 0, 0),
    (-10.0, 3, 3, 0, 0, 0),
    (2.0, 1, 0, 1, 0, 0),
    (1.0, 2, 0, 1, 0, 0),
    (-20.0, 3, 4, 1, 0, 0),
    (-11.0, 0, 0, 2, 0, 0),
    (20.0, 3, 0, 0, 1, 0),
    (-10.0, 0, 0, 0, 2, 0),
    (20.0, 4, 0, 1, 1, 0),
    (-10.0, 5, 5, 0, 0, 1),
    (20.0, 5, 0, 0, 0, 2),
    (-10.0, 0, 0, 0, 0, 3),
]


def _tensor(name: str, value, dtype) -> onnx.TensorProto:
    return numpy_helper.from_array(np.asarray(value, dtype=dtype), name)


def _one_hot(indices: list[int], width: int) -> np.ndarray:
    return np.eye(width, dtype=np.float32)[np.asarray(indices, dtype=np.int64)]


def _renderer_initializers() -> list[onnx.TensorProto]:
    weights, left, right, row_power, col_power, channel_power = map(
        list, zip(*_TERMS, strict=True)
    )
    r = np.arange(30, dtype=np.float32)
    c = np.arange(30, dtype=np.float32)
    valid_r = (r < 10).astype(np.float32)
    valid_c = (c < 10).astype(np.float32)
    k = np.arange(10, dtype=np.float32)
    nonzero = (k > 0).astype(np.float32)
    result = [
        _tensor("term_weight", weights, np.float32),
        _tensor("descriptor_left", _one_hot(left, 6), np.float32),
        _tensor("descriptor_right", _one_hot(right, 6), np.float32),
        _tensor("row_select", _one_hot(row_power, 3), np.float32),
        _tensor("col_select", _one_hot(col_power, 3), np.float32),
        _tensor("channel_select", _one_hot(channel_power, 4), np.float32),
        _tensor(
            "row_basis_poly",
            np.stack([valid_r, r * valid_r, r * r * valid_r]),
            np.float32,
        ),
        _tensor(
            "col_basis_poly",
            np.stack([valid_c, c * valid_c, c * c * valid_c]),
            np.float32,
        ),
        _tensor(
            "channel_basis_poly",
            np.stack(
                [np.ones(10, dtype=np.float32), nonzero, nonzero * k, nonzero * k * k]
            ).reshape(4, 1, 10),
            np.float32,
        ),
        _tensor("output_sign", [1.0] + [-1.0] * 9, np.float32),
        _tensor("descriptor_one", np.ones((1, 1, 1, 6)), np.float32),
        _tensor("invalid_lo", 20, np.uint8),
    ]
    eye = np.eye(6, dtype=np.float32)
    result.extend(_tensor(f"slot_{index}", eye[index], np.float32) for index in range(6))
    return result


def _terminal_einsum() -> onnx.NodeProto:
    symbols = [char for char in string.ascii_letters if char not in set("NKRCXY")]
    inputs: list[str] = []
    subscripts: list[str] = []
    cursor = 0
    for slot in range(6):
        p, a, b, s, u, v, w = symbols[cursor : cursor + 7]
        cursor += 7
        inputs.extend(
            [
                "term_weight",
                "descriptor_left",
                "descriptor",
                "descriptor_right",
                "descriptor",
                f"slot_{slot}",
                "row_select",
                "row_basis_poly",
                "col_select",
                "col_basis_poly",
                "channel_select",
                "channel_basis_poly",
            ]
        )
        subscripts.extend(
            [
                p,
                p + a,
                "XY" + a + s,
                p + b,
                "XY" + b + s,
                s,
                p + u,
                u + "R",
                p + v,
                v + "C",
                p + w,
                w + "NK",
            ]
        )
    inputs.append("output_sign")
    subscripts.append("K")
    return helper.make_node(
        "Einsum",
        inputs,
        ["output"],
        name="output",
        equation=",".join(subscripts) + "->NKRC",
    )


def build(task):
    inits = [
        _tensor("I30", np.arange(30).reshape(30, 1, 1), np.float32),
        _tensor("two_f", [2], np.float32),
        _tensor("two_i8", [2], np.int8),
        _tensor("two_u8", [2], np.uint8),
        _tensor("z_u8", [0], np.uint8),
        _tensor("st", [1], np.int64),
        _tensor("en", [10], np.int64),
        _tensor("ax3", [3], np.int64),
        _tensor("topk6", [6], np.int64),
        _tensor("kvec", np.arange(1, 10).reshape(1, 1, 1, 9), np.uint8),
        *_renderer_initializers(),
    ]
    nodes = [
        helper.make_node(
            "Einsum", ["input", "I30"], ["Sr_f"], name="Sr_f", equation="bkrc,rxy->bxyk"
        ),
        helper.make_node(
            "Einsum", ["input", "I30"], ["Sc_f"], name="Sc_f", equation="bkrc,cxy->bxyk"
        ),
        helper.make_node(
            "Einsum",
            ["input", "I30", "I30", "two_f"],
            ["tS_f"],
            name="tS_f",
            equation="bkrc,rxy,cxy,z->bxyk",
        ),
        helper.make_node("Cast", ["Sr_f"], ["Sr10"], name="Sr10", to=3),
        helper.make_node("Slice", ["Sr10", "st", "en", "ax3"], ["Sr"], name="Sr"),
        helper.make_node("Cast", ["Sc_f"], ["Sc10"], name="Sc10", to=3),
        helper.make_node("Slice", ["Sc10", "st", "en", "ax3"], ["Sc"], name="Sc"),
        helper.make_node("Cast", ["tS_f"], ["tS10"], name="tS10", to=3),
        helper.make_node("Slice", ["tS10", "st", "en", "ax3"], ["tS"], name="tS"),
        helper.make_node("Mul", ["Sr", "Sc"], ["SrSc"], name="SrSc"),
        helper.make_node("Sub", ["tS", "SrSc"], ["D"], name="D"),
        helper.make_node("Abs", ["D"], ["absD"], name="absD"),
        helper.make_node("Cast", ["absD"], ["absD_h"], name="absD_h", to=10),
        helper.make_node(
            "TopK",
            ["absD_h", "topk6"],
            ["selected_square_h", "slot_idx64"],
            name="slot_topk6_fp16",
            axis=3,
            largest=1,
            sorted=0,
        ),
        helper.make_node("Sqrt", ["selected_square_h"], ["selected_span"], name="selected_span"),
        helper.make_node(
            "GatherElements", ["D", "slot_idx64"], ["selected_D"], name="selected_D", axis=3
        ),
        helper.make_node("Sign", ["selected_D"], ["selected_sgn"], name="selected_sgn"),
        helper.make_node(
            "GatherElements", ["Sr", "slot_idx64"], ["selected_Sr"], name="selected_Sr", axis=3
        ),
        helper.make_node(
            "GatherElements", ["Sc", "slot_idx64"], ["selected_Sc"], name="selected_Sc", axis=3
        ),
        helper.make_node("Cast", ["selected_span"], ["selected_n"], name="selected_n", to=2),
        helper.make_node(
            "Mul", ["selected_sgn", "selected_Sr"], ["selected_sgnSr"], name="selected_sgnSr"
        ),
        helper.make_node(
            "Sub", ["selected_Sc", "selected_sgnSr"], ["selected_num"], name="selected_num"
        ),
        helper.make_node(
            "Div", ["selected_num", "two_i8"], ["selected_base"], name="selected_base"
        ),
        helper.make_node(
            "Cast", ["selected_Sr"], ["selected_Sr_u"], name="selected_Sr_u", to=2
        ),
        helper.make_node(
            "Sub", ["selected_Sr_u", "selected_n"], ["selected_numlo"], name="selected_numlo"
        ),
        helper.make_node(
            "Div", ["selected_numlo", "two_u8"], ["selected_lo"], name="selected_lo"
        ),
        helper.make_node(
            "Greater", ["selected_n", "z_u8"], ["selected_present"], name="selected_present"
        ),
        helper.make_node(
            "GatherElements",
            ["kvec", "slot_idx64"],
            ["selected_kvec"],
            name="selected_kvec",
            axis=3,
        ),
        helper.make_node(
            "Where",
            ["selected_present", "selected_lo", "invalid_lo"],
            ["selected_lo_safe"],
            name="selected_lo_safe",
        ),
        helper.make_node("Cast", ["selected_lo_safe"], ["d_lo"], name="d_lo", to=1),
        helper.make_node("Cast", ["selected_n"], ["d_span"], name="d_span", to=1),
        helper.make_node("Cast", ["selected_base"], ["d_base"], name="d_base", to=1),
        helper.make_node("Cast", ["selected_sgn"], ["d_sign"], name="d_sign", to=1),
        helper.make_node("Cast", ["selected_kvec"], ["d_colour"], name="d_colour", to=1),
        helper.make_node(
            "Concat",
            ["descriptor_one", "d_lo", "d_span", "d_base", "d_sign", "d_colour"],
            ["descriptor"],
            name="descriptor",
            axis=2,
        ),
        _terminal_einsum(),
    ]
    result = model(
        "task037_quadratic_product_renderer",
        nodes,
        inits,
        output_dtype=TensorProto.FLOAT,
        opset=20,
        value_infos=[],
        ir_version=10,
    )
    return onnx.shape_inference.infer_shapes(result, strict_mode=True)
