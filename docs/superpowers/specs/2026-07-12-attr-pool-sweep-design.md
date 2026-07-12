# Attribute-Only Pool Sweep Design

## Goal

Search all 400 tasks for an exact, parameter-free, single-node MaxPool solution.
Any new hit scores 25 points because the graph input and final graph output are
free and every MaxPool setting is stored as an uncounted node attribute.

## Search Space

The input and output are the official padded `[1,10,30,30]` one-hot tensors.
Search stride-one rectangular MaxPool nodes with:

- kernel height and width from 1 through 30;
- every dilation whose effective kernel fits within 30 cells;
- every asymmetric top/bottom and left/right padding split that preserves a
  30x30 output.

AveragePool and LpPool need no separate support scan: for nonnegative one-hot
inputs and the scorer's final `output > 0` decoding, their positive support is
the same rectangular support union as MaxPool.

## Algorithm

A rectangular 2D pool is separable. For every example and channel, project the
input and target to row-presence and column-presence vectors. Enumerate all 1D
kernel/dilation/padding configurations independently on the row and column
projections. Only take the Cartesian product of row and column configurations
that match every projection, then run exact 2D support comparison.

Use train+test first. A surviving task/configuration is then verified against
all arc-gen examples. Existing cost-zero tasks are controls, not discoveries.

## Correctness and Output

- Synthetic tests cover identity, asymmetric directional dilation, dilated
  pooling, and a recoloring rejection.
- Every reported hit must match every output channel and every padded cell.
- A new hit is materialized as one ONNX MaxPool node and evaluated with the
  official harness before any adoption is considered.
- If no new hit exists, record the complete searched parameter bounds and
  0-hit result in the `fourth-25pt-hunt` four-field ledger.

All scanner and candidate artifacts live under `candidates/attr_pool_sweep/`.
Deployed files remain unchanged unless a hit later passes `ng gate` and is
adopted through `ng adopt`.
