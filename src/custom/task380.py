"""task380 (ARC-AGI ed36ccf7) — fixed geometric transform of a tiny grid.

Rule (from the ARC-GEN generator, verified fresh):
  size=3.  For every coloured pixel:  output[size-1-c][r] = input[r][c] = color.
  i.e. the map is the deterministic permutation
      out[2 - c][r] = in[r][c]   for r,c in 0..2
  which is a rotate-90 / reflect composition on the 3x3 active region.
  Everything off the 3x3 grid is background 0.

  This is a PURE one-hot spatial permutation (no colour change), so the whole
  pipeline can run in UINT8 (itemsize 1) and the OUTPUT can be declared UINT8 —
  the harness scores (out > 0), which is identical for a {0,1} one-hot.

Encoding (matches the public Slice/Transpose/Pad net but in uint8):
  - Slice the 10-ch input to the 3x3 active block with cols reversed
    (axis3 steps=-1 over 0..2)  ->  in[r][2-c]              (fp32, 360B entry)
  - Cast that 3x3x10 block to UINT8                          (90B)
  - Transpose axes (0,1,3,2) to swap the spatial axes        (uint8, 90B)
        after reverse-cols the value at spatial (r, j) is in[r][2-j];
        transpose puts it at (j, r), i.e. out[j][r] = in[r][2-j].
        Setting c = 2 - j  =>  j = 2 - c  =>  out[2-c][r] = in[r][c].  ✓
  - Pad back to 30x30 (uint8 Pad runs under ORT_DISABLE_ALL); its output IS the
    graph output, declared UINT8.

Result: mem=540 (360 fp32 crop + 90 uint8 cast + 90 uint8 transpose), params=16
(slice 8 + opset-13 Pad pads-input 8), pts=18.68.  Beats public 18.41 by +0.27 =>
MARGINAL.  This is the structural floor: the fp32 3x3x10 entry crop (360B) is
irreducible (input is fp32, all 10 colour channels can carry the random colour,
3x3 is the full active grid), the rot90 needs BOTH a reverse-step slice and a
Transpose (so two small uint8 working planes survive the cast), and uint8 Pad
forces opset-13's 8-param pads input.  Staying in fp32 to use opset-10's
attribute-Pad (0 pad-params) costs 720B mem = the original 18.41; the uint8/opset-13
trade nets only +0.27, just shy of the +0.30 bar.
"""

import numpy as np
from onnx import helper, numpy_helper, TensorProto

from ..harness import IR_VERSION

S = 30
G = 3  # active grid is always 3x3


def build(task):
    inits, nodes = [], []

    def init(name, arr):
        inits.append(numpy_helper.from_array(np.ascontiguousarray(arr), name))
        return name

    # 1) Slice to 3x3 with cols reversed: revcols[.,.,r,j] = input[.,.,r, 2-j]
    init("starts", np.array([0, 2], np.int64))
    init("ends", np.array([3, -100], np.int64))  # col 2,1,0 (neg end -> include 0)
    init("axes", np.array([2, 3], np.int64))
    init("steps", np.array([1, -1], np.int64))
    nodes.append(helper.make_node("Slice", ["input", "starts", "ends", "axes", "steps"], ["revcols"]))

    # 2) Cast to uint8
    nodes.append(helper.make_node("Cast", ["revcols"], ["u8"], to=TensorProto.UINT8))

    # 3) Transpose spatial axes -> rot[.,.,j,r] = revcols[.,.,r,j] = input[.,.,r,2-j]
    nodes.append(helper.make_node("Transpose", ["u8"], ["rot"], perm=[0, 1, 3, 2]))

    # 4) Pad back to 30x30 (uint8 Pad needs opset-13; pads is an input -> 8 params).
    init("pads", np.array([0, 0, 0, 0, 0, 0, S - G, S - G], np.int64))
    nodes.append(helper.make_node("Pad", ["rot", "pads"], ["output"], mode="constant"))

    graph = helper.make_graph(
        nodes, "task380", [helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 10, S, S])],
        [helper.make_tensor_value_info("output", TensorProto.UINT8, [1, 10, S, S])], inits,
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    model.ir_version = IR_VERSION
    return model
