"""Unsupported integer TopK scan: the Kaggle grader-killer check.

The Kaggle grader errors the WHOLE submission if a net feeds an unsupported
integer dtype into TopK — invisible to local ORT/checker gates.  Unsigned
integer TopK was established on 2026-07-02; signed INT8 was independently
established by task233 submission 54418836 and again by full submission
54716353 on 2026-07-15.

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

# Keep the public function name for compatibility with the existing gate/pack
# callers, but include signed INT8: pinned local ORT accepts it while Kaggle's
# grader rejects the package.  Do not broaden this to every signed integer
# without an oracle; INT32/INT64 may have different grader support.
UNSUPPORTED_INTEGER_TOPK = UNSIGNED | {onnx.TensorProto.INT8}


def find_unsigned_topk(model_path: Path) -> list[str]:
    """Return a list of violation descriptions (empty list = clean).

    A TopK node is a violation if its input[0] elem_type is unsigned
    (UINT8/16/32/64), signed INT8, OR unresolved (UNKNOWN also counts as a
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
            if t in UNSUPPORTED_INTEGER_TOPK:
                offenders.append(
                    f"TopK {node.name or node.output[0]} input elem_type={t}"
                )
            elif t is None:
                offenders.append(
                    f"TopK {node.name or node.output[0]} input elem_type=UNKNOWN-verify-manually"
                )
    return offenders
