# placement_fixed_offset — Fixed offset

Placement is a constant translation/offset.

## Optimization Question

What is the cheapest verified ONNX family for tasks in this class, and
which semantic preconditions let us avoid full-canvas intermediates?

## Candidate Tasks

| task | pts | mem | params | status | first routes | evidence |
|---:|---:|---:|---:|---|---|---|
| 261 | 19.702 | 0 | 200 | seeded_from_verified_log | `compiler_single_conv_qlinear` | stored examples have bbox-aligned translation of the nonzero mask |
| 053 | 21.599 | 0 | 30 | seeded_from_verified_log | `compiler_direct_output_algebra`, `compiler_tiny_lut_gather` | stored examples have bbox-aligned translation of the nonzero mask |

## Known Best Routes

- Pending human review.

## Kill Criteria

- Pending human review.

## Successful Applications

- Pending verification.

## Failed Applications / Walls

- Pending verification.
