"""Task 017 — periodic pattern inpainting.

The grid is a 21x21 periodic colour pattern with black cut-outs.  Recover the
pattern parameters by matching a few fixed sample cells against all unique
candidate patterns, then rebuild the full closed-form pattern:

    colour(r,c) = ((rr^2 + cc^2) % mod) + 1
    rr = (offset + r) % length - length//2
    cc = (offset + c) % length - length//2
"""

from __future__ import annotations

import numpy as np
from onnx import TensorProto, helper

from ._exact import model, tensor


SAMPLE_POSITIONS = [
    (0, 4),
    (20, 15),
    (3, 19),
    (10, 2),
    (1, 1),
    (2, 7),
    (4, 20),
    (3, 3),
    (13, 0),
    (18, 18),
    (6, 1),
    (9, 17),
    (11, 7),
]


def _pattern(mod: int, length: int, offset: int) -> np.ndarray:
    rows = np.arange(21, dtype=np.int16)[:, None]
    cols = np.arange(21, dtype=np.int16)[None, :]
    half = length // 2
    rr = (rows + offset) % length - half
    cc = (cols + offset) % length - half
    return ((rr * rr + cc * cc) % mod + 1).astype(np.uint8)


def _candidate_params() -> np.ndarray:
    rows = []
    seen = set()
    for mod in range(4, 10):
        for length in range(4, mod + 1):
            for offset in range(1, length + 1):
                pat = _pattern(mod, length, offset)
                key = pat.tobytes()
                if key in seen:
                    continue
                seen.add(key)
                rows.append([mod, length, offset])
    return np.array(rows, dtype=np.uint8)


def _candidate_samples(params: np.ndarray) -> np.ndarray:
    samples = []
    for mod, length, offset in params:
        pat = _pattern(int(mod), int(length), int(offset))
        samples.append([pat[r, c] for r, c in SAMPLE_POSITIONS])
    return np.array(samples, dtype=np.uint8)[None, :, :]


def _sample_indices() -> np.ndarray:
    # GatherND with batch_dims=2: data [1,10,30,30], indices [1,10,NS,2]
    # selecting (r, c) within the trailing H,W dims, replicated per channel.
    ns = len(SAMPLE_POSITIONS)
    idx = np.zeros((1, 10, ns, 2), dtype=np.int64)
    for ch in range(10):
        for j, (r, c) in enumerate(SAMPLE_POSITIONS):
            idx[0, ch, j] = [r, c]
    return idx


def build(task):
    params = _candidate_params()
    inits = [
        tensor("candidate_params", params),
        tensor("candidate_samples", _candidate_samples(params)),
        tensor("sample_nd_idx", _sample_indices()),
        tensor("row_grid", np.arange(21, dtype=np.float16).reshape(1, 1, 21, 1)),
        tensor("col_grid", np.arange(21, dtype=np.float16).reshape(1, 1, 1, 21)),
        tensor("idx_mod", np.array([0], dtype=np.int64)),
        tensor("idx_len", np.array([1], dtype=np.int64)),
        tensor("idx_offset", np.array([2], dtype=np.int64)),
        tensor("half_two", np.array(2.0, dtype=np.float16)),
        # channel ch (0..9) lights where pattern0 == ch-1; ch0 compares 255
        # (never matches in-grid 0..mod-1 nor off-grid sentinel 200).
        tensor("channel_values", np.array([255, 0, 1, 2, 3, 4, 5, 6, 7, 8], dtype=np.uint8).reshape(1, 10, 1, 1)),
        tensor("pads", np.array([0, 0, 0, 0, 0, 0, 9, 9], dtype=np.int64)),
        tensor("pad_value", np.array(200, dtype=np.uint8)),
    ]
    nodes = [
        helper.make_node("GatherND", ["input", "sample_nd_idx"], ["sample_planes"], batch_dims=2),
        helper.make_node("ArgMax", ["sample_planes"], ["input_sample_i64"], axis=1, keepdims=1),
        helper.make_node("Cast", ["input_sample_i64"], ["input_sample_u8"], to=TensorProto.UINT8),
        helper.make_node("Equal", ["candidate_samples", "input_sample_u8"], ["matches_b"]),
        helper.make_node("Cast", ["matches_b"], ["matches_h"], to=TensorProto.FLOAT16),
        helper.make_node("ReduceSum", ["matches_h", "idx_offset"], ["scores"], keepdims=0),
        helper.make_node("ArgMax", ["scores"], ["best_id"], axis=1, keepdims=0),
        helper.make_node("Gather", ["candidate_params", "best_id"], ["selected_params_u8"], axis=0),
        helper.make_node("Gather", ["selected_params_u8", "idx_mod"], ["mod_u8"], axis=1),
        helper.make_node("Gather", ["selected_params_u8", "idx_len"], ["length_u8"], axis=1),
        helper.make_node("Gather", ["selected_params_u8", "idx_offset"], ["offset_u8"], axis=1),
        helper.make_node("Cast", ["mod_u8"], ["mod"], to=TensorProto.FLOAT16),
        helper.make_node("Cast", ["length_u8"], ["length"], to=TensorProto.FLOAT16),
        helper.make_node("Cast", ["offset_u8"], ["offset"], to=TensorProto.FLOAT16),
        # half = floor(length / 2)
        helper.make_node("Div", ["length", "half_two"], ["length_half"]),
        helper.make_node("Floor", ["length_half"], ["half"]),
        helper.make_node("Add", ["row_grid", "offset"], ["row_plus_offset"]),
        helper.make_node("Add", ["col_grid", "offset"], ["col_plus_offset"]),
        helper.make_node("Mod", ["row_plus_offset", "length"], ["row_mod"], fmod=1),
        helper.make_node("Mod", ["col_plus_offset", "length"], ["col_mod"], fmod=1),
        helper.make_node("Sub", ["row_mod", "half"], ["r"]),
        helper.make_node("Sub", ["col_mod", "half"], ["c"]),
        helper.make_node("Mul", ["r", "r"], ["rr"]),
        helper.make_node("Mul", ["c", "c"], ["cc"]),
        helper.make_node("Add", ["rr", "cc"], ["rrcc"]),
        helper.make_node("Mod", ["rrcc", "mod"], ["pattern0"], fmod=1),
        helper.make_node("Cast", ["pattern0"], ["pattern0_u8"], to=TensorProto.UINT8),
        helper.make_node("Pad", ["pattern0_u8", "pads", "pad_value"], ["pattern0_full"], mode="constant"),
        helper.make_node("Equal", ["pattern0_full", "channel_values"], ["output"]),
    ]
    return model("task017", nodes, inits, output_dtype=TensorProto.BOOL, opset=13)
