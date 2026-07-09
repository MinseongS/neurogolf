"""Unsigned-int TopK scan: the Kaggle grader-killer check.

The Kaggle grader errors the WHOLE submission if any net feeds an unsigned
dtype (uint8/16/32/64) into TopK — invisible to local ORT/checker gates.
See memory neurogolf-uint8-topk-grader-killer + reports/submission_log.md
(submissions 54255339, 2026-07-02).

Ported (function wrapper, logic unchanged) from
reports/scripts/scan_unsigned_topk.py.
"""

from pathlib import Path

import onnx
from onnx import shape_inference

UNSIGNED = {
    onnx.TensorProto.UINT8,
    onnx.TensorProto.UINT16,
    onnx.TensorProto.UINT32,
    onnx.TensorProto.UINT64,
}


def find_unsigned_topk(model_path: Path) -> list[str]:
    """Return a list of violation descriptions (empty list = clean).

    A TopK node is a violation if its input[0] elem_type is unsigned
    (UINT8/16/32/64) OR the type is unresolved (UNKNOWN also counts as a
    violation).
    """
    model = onnx.load(str(model_path))
    try:
        model = shape_inference.infer_shapes(model)
    except Exception:
        pass
    types = {}
    g = model.graph
    for vi in list(g.value_info) + list(g.input) + list(g.output):
        types[vi.name] = vi.type.tensor_type.elem_type
    for init in g.initializer:
        types[init.name] = init.data_type
    # propagate through Cast/producers we know
    for node in g.node:
        if node.op_type == "Cast":
            for a in node.attribute:
                if a.name == "to":
                    types[node.output[0]] = a.i
    offenders = []
    for node in g.node:
        if node.op_type == "TopK":
            t = types.get(node.input[0])
            if t in UNSIGNED:
                offenders.append(
                    f"TopK {node.name or node.output[0]} input elem_type={t}"
                )
            elif t is None:
                offenders.append(
                    f"TopK {node.name or node.output[0]} input elem_type=UNKNOWN-verify-manually"
                )
    return offenders
