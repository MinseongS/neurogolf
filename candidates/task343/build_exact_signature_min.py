"""Shave task343's exact signature graph with direct p4 scalar probes."""

from pathlib import Path

import numpy as np
import onnx
from onnx import helper, numpy_helper

from build_exact_signature import build as build_exact_signature


OUTPUT = Path(__file__).with_name("exact_signature_min.onnx")
PERIODS = {
    "period3_u8": 3,
    "period4_u8": 4,
    "period6_u8": 6,
    "period8_u8": 8,
}


def build(_task=None) -> onnx.ModelProto:
    model = build_exact_signature()
    graph = model.graph
    graph.name = "task343_exact_signature_min"

    producers = {output: node for node in graph.node for output in node.output}
    producers["p4_col0"].CopyFrom(
        helper.make_node(
            "Conv",
            ["input", "signature_kernel"],
            ["p4_col0"],
            name="signature_p4_col0",
            pads=[-2, 0, -25, -29],
        )
    )
    producers["p4_col4"].CopyFrom(
        helper.make_node(
            "Conv",
            ["input", "signature_kernel"],
            ["p4_col4"],
            name="signature_p4_col4",
            pads=[-2, -4, -25, -25],
        )
    )

    replaced = []
    for init in graph.initializer:
        if init.name in {"idx0", "idx1"}:
            continue
        if init.name == "squeeze_all_axes":
            replaced.append(
                numpy_helper.from_array(
                    np.array([0, 1, 2], dtype=np.int64), init.name
                )
            )
        elif init.name in PERIODS:
            replaced.append(
                numpy_helper.from_array(
                    np.array([PERIODS[init.name]], dtype=np.uint8), init.name
                )
            )
        else:
            replaced.append(init)
    del graph.initializer[:]
    graph.initializer.extend(replaced)
    del graph.value_info[:]

    onnx.checker.check_model(model, full_check=True)
    return onnx.shape_inference.infer_shapes(
        model, strict_mode=True, data_prop=True
    )


if __name__ == "__main__":
    candidate = build()
    onnx.save(candidate, OUTPUT)
    print(OUTPUT)
