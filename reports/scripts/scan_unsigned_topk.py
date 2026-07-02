"""Pre-submit grader-killer scan: unsigned-int TopK inputs across ALL 400 nets.

The Kaggle grader errors the WHOLE submission if any net feeds an unsigned
dtype (uint8/16/32/64) into TopK — invisible to local ORT/checker gates.
See memory neurogolf-uint8-topk-grader-killer + reports/submission_log.md
(submissions 54255339, 2026-07-02). Run on the exact nets going into the zip:

  PYTHONPATH=. .venv/bin/python reports/scripts/scan_unsigned_topk.py [networks_dir]

Exit code 1 if any offender is found.
"""

import sys
from pathlib import Path

import onnx
from onnx import shape_inference

UNSIGNED = {
    onnx.TensorProto.UINT8,
    onnx.TensorProto.UINT16,
    onnx.TensorProto.UINT32,
    onnx.TensorProto.UINT64,
}

root = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("networks")
offenders = []
paths = sorted(root.glob("task*.onnx"))
for p in paths:
    model = onnx.load(str(p))
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
    for node in g.node:
        if node.op_type == "TopK":
            t = types.get(node.input[0])
            if t in UNSIGNED:
                offenders.append((p.name, node.name or node.output[0], t))
            elif t is None:
                offenders.append((p.name, node.name or node.output[0], "UNKNOWN-verify-manually"))

print(f"scanned {len(paths)} nets")
if offenders:
    for f, n, t in offenders:
        print(f"OFFENDER {f}: TopK {n} input elem_type={t}")
    sys.exit(1)
print("clean: no unsigned TopK inputs")
