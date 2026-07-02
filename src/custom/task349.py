"""Task 349 candidate — collapse the 5 per-radius dilation MaxPools into ONE
5-channel QLinearConv (union-by-count) + Min clip; stack the 5 run detectors
into ONE QLinearConv.

Equivalence-golf vs incumbent src/custom/task349.py:
  * hp_all = QLinearConv(m, W[5,1,1,12]) == the 5 incumbent hp planes stacked
    (identical kernels/bias, values {0,3}).
  * gsum   = QLinearConv(hp_all, ones[1,5,11,20], pads [5,14,5,5]): channel k
    (k=1..5) has ones at rows 5-k..5+k, cols (15-3k)..(14+k), i.e. exactly the
    incumbent MaxPool window (kernel [2k+1,4k], pads [k,3k-1,k,k]).  gsum > 0
    <=> max over the same windows > 0; u8 saturation is harmless because
  * gclip  = Min(gsum, 3) in {0,3} == Max(gr_1..gr_5) of the incumbent.
  * color = Max(gclip, beam, mar9); epilogue identical to incumbent.

Memory: colf 3600 + m 900 + valid 900 + hp_all 4500 + gsum 900 + gclip 900
        + beam 900 + mar9 900 + color 900 + cv 900 = 15300 (was 18000)
Params: 90 (incumbent) + 1100 (dilation weight) + 1 (three) = 1191 (was 90)
Total 16491 vs 18090.
"""
import numpy as np
from onnx import helper, numpy_helper, TensorProto

from ..harness import IR_VERSION


def _t(name, arr):
    return numpy_helper.from_array(np.asarray(arr), name)


def _run_kernel(r):
    """int8 kernel: maximal maroon run of length >= 2r emits value 3 (scaled x3)."""
    k = np.zeros((1, 1, 12), dtype=np.int8)
    k[0, 0, 0] = -100          # left veto
    k[0, 0, 1:1 + 2 * r] = 3   # 2r core cells (x3 so a hit emits 3)
    k[0, 0, 1 + 2 * r] = -100  # right veto
    return k


def build(task):
    # entry conv weight: maroon -> 1.0, in-grid background -> 0.25.
    colf_w = np.zeros((1, 10, 1, 1), dtype=np.float32)
    colf_w[0, 0, 0, 0] = 0.25
    colf_w[0, 9, 0, 0] = 1.0

    # stacked run detectors: channel c detects radius r = c+1 runs (value 3)
    hk_all = np.stack([_run_kernel(r) for r in range(1, 6)], axis=0)  # [5,1,1,12]
    hb_all = np.array([-3 * (2 * r - 1) for r in range(1, 6)], dtype=np.int32)

    # stacked dilation windows: channel c=r-1 covers rows +-r, cols [-(3r-1), +r]
    gw = np.zeros((1, 5, 11, 20), dtype=np.int8)
    for c, r in enumerate(range(1, 6)):
        gw[0, c, 5 - r:5 + r + 1, 15 - 3 * r:15 + r] = 1

    inits = [
        _t("colf_w", colf_w),
        _t("scale", np.array(1.0, dtype=np.float32)),
        _t("xz", np.array(0, dtype=np.uint8)),
        _t("wz", np.array(0, dtype=np.int8)),
        _t("sentinel", np.array(255, dtype=np.uint8)),
        _t("k", np.arange(10, dtype=np.uint8).reshape(1, 10, 1, 1)),
        _t("mar_w", np.array([[[[9]]]], dtype=np.int8)),
        _t("hk_all", hk_all),
        _t("hb_all", hb_all),
        _t("gw", gw),
        _t("three", np.array(3, dtype=np.uint8)),
    ]

    qz = ["scale", "xz"]
    nodes = [
        helper.make_node("Conv", ["input", "colf_w"], ["colf"], kernel_shape=[1, 1]),
        helper.make_node("Cast", ["colf"], ["m"], to=TensorProto.UINT8),     # maroon 0/1
        helper.make_node("Cast", ["colf"], ["valid"], to=TensorProto.BOOL),  # in-grid
        # maroon -> literal colour 9
        helper.make_node("QLinearConv",
                         ["m", *qz, "mar_w", "scale", "wz", *qz], ["mar9"],
                         kernel_shape=[1, 1]),
        # blue beam (value 1) down each column
        helper.make_node("MaxPool", ["m"], ["beam"],
                         kernel_shape=[30, 1], pads=[29, 0, 0, 0], strides=[1, 1]),
        # stacked radius detectors -> [1,5,30,30], values {0,3}
        helper.make_node("QLinearConv",
                         ["m", *qz, "hk_all", "scale", "wz", *qz, "hb_all"],
                         ["hp_all"], kernel_shape=[1, 12], pads=[0, 1, 0, 10],
                         strides=[1, 1]),
        # union of per-radius dilations by count -> [1,1,30,30]
        helper.make_node("QLinearConv",
                         ["hp_all", *qz, "gw", "scale", "wz", *qz],
                         ["gsum"], kernel_shape=[11, 20], pads=[5, 14, 5, 5],
                         strides=[1, 1]),
        helper.make_node("Min", ["gsum", "three"], ["gclip"]),
        # priority Max: green(3) beats beam(1); maroon9 beats both.
        helper.make_node("Max", ["gclip", "beam", "mar9"], ["color"]),
        # off-grid -> 255 so Equal against k=[0..9] yields all-false there.
        helper.make_node("Where", ["valid", "color", "sentinel"], ["cv"]),
        helper.make_node("Equal", ["cv", "k"], ["output"]),
    ]

    graph = helper.make_graph(
        nodes, "task349",
        [helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 10, 30, 30])],
        [helper.make_tensor_value_info("output", TensorProto.BOOL, [1, 10, 30, 30])],
        inits,
    )
    return helper.make_model(graph, ir_version=IR_VERSION,
                             opset_imports=[helper.make_opsetid("", 13)])
