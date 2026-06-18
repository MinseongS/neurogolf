"""task322 (ARC-AGI d037b0a7) — per-column downward gravity fill.

Rule (from the ARC-GEN generator, verified fresh):
  The active grid is ALWAYS 3x3 (size=3 fixed). Each column c holds exactly one
  coloured pixel at row rows[c] with colour colors[c]. The output fills that
  column from row rows[c] DOWN to the bottom (row 2) with colour colors[c]:
      output[r][c] = colors[c]  for all r >= rows[c],  else background 0.
  Off-grid cells (r>=3 or c>=3) are all-zero across every channel (the harness
  one-hot leaves them empty — NOT channel-0=1).  Every in-grid cell carries
  exactly one one-hot (bg cells set channel 0).

  This is a per-column, per-colour DOWNWARD cumulative-OR over the 3 active rows.

Encoding — ONE cross-channel height-3 Conv; its output IS the graph output (mem 0):
  Top padding 2 (pads=[2,0,0,0]) so each output row r reads input rows
  {r-2, r-1, r} (kernel positions kp=0,1,2). No column padding -> off-grid
  columns/rows whose 3-window is all-zero get the bias alone.

  The previous public net used a learned height-5 conv (510 params). Height 3 is
  sufficient: from the bottom in-grid row (r=2) the window reaches the grid top
  (row 0). The off-grid leak (a colour at rows 1-2 bleeding into output rows 3-4)
  is killed by requiring the CENTER cell to be in-grid, which is detectable
  because every in-grid cell carries a one-hot while off-grid cells are all-zero.

  out colour channel k>=1  (bias -1.5):
      W[k,k,kp=0]=W[k,k,kp=1]=1   colour-k present ABOVE centre (rows r-2,r-1)
      W[k,k,kp=2]=2               colour-k AT centre (the pixel itself)
      W[k,0,kp=2]=1               background one-hot AT centre (in-grid anchor)
    fire iff  (colour at centre)            -> 2-1.5>0
          or  (colour above) AND (bg at centre, i.e. in-grid bg-below-pixel) -> 1+1-1.5>0
    a colour above with an EMPTY centre (off-grid row 3/4) gives 1-1.5<0 -> no fire.

  out channel 0 (background, bias -0.5):
      W[0,0,kp=2]=1               bg one-hot AT centre (cell is in-grid bg)
      W[0,k,:]=-10  (all kp)      any colour at/above strongly suppresses
    fire iff  bg at centre AND no colour at/above -> in-grid unfilled cell.
    off-grid centre is empty -> -0.5<0 -> no fire.
"""

import numpy as np
from onnx import helper, numpy_helper, TensorProto

from ..harness import IR_VERSION

S = 30
K = 3  # kernel height: reach 2 rows up + self covers the whole 3-row grid


def build(task):
    W = np.zeros((10, 10, K, 1), dtype=np.float32)
    B = np.zeros((10,), dtype=np.float32)

    # colour channels: downward fill, gated by in-grid centre occupancy
    for k in range(1, 10):
        W[k, k, 0, 0] = 1.0   # colour above centre (row r-2)
        W[k, k, 1, 0] = 1.0   # colour above centre (row r-1)
        W[k, k, 2, 0] = 2.0   # colour at centre (the pixel)
        W[k, 0, 2, 0] = 1.0   # background one-hot at centre (in-grid anchor)
        B[k] = -1.5

    # background channel 0: in-grid bg cell with no coloured pixel at/above it
    W[0, 0, 2, 0] = 1.0
    for k in range(1, 10):
        W[0, k, :, 0] = -10.0
    B[0] = -0.5

    inits = [
        numpy_helper.from_array(W, "W"),
        numpy_helper.from_array(B, "B"),
    ]
    node = helper.make_node(
        "Conv", ["input", "W", "B"], ["output"],
        kernel_shape=[K, 1],
        pads=[K - 1, 0, 0, 0],   # top=2, bottom=0 -> rows {r-2,r-1,r}; no col pad
    )

    F = TensorProto.FLOAT
    graph = helper.make_graph(
        [node], "task322",
        [helper.make_tensor_value_info("input", F, [1, 10, S, S])],
        [helper.make_tensor_value_info("output", F, [1, 10, S, S])],
        inits,
    )
    model = helper.make_model(
        graph,
        opset_imports=[helper.make_opsetid("", 13)],
        ir_version=IR_VERSION,
    )
    return model
