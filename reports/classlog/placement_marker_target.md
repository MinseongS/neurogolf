# placement_marker_target — Marker target

Placement is controlled by markers or target slots.

## Optimization Question

What is the cheapest verified ONNX family for tasks in this class, and
which semantic preconditions let us avoid full-canvas intermediates?

## Candidate Tasks

| task | pts | mem | params | status | first routes | evidence |
|---:|---:|---:|---:|---|---|---|
| 066 | 15.256 | 16899 | 160 | seeded_from_verified_log | `compiler_direct_output_algebra`, `compiler_final_equal_overlay`, `compiler_bounded_scan` | tasklog keyword match: \b(marker\|hint).{0,50}\b(target\|slot\|place\|route\|lattice)\|\b(target\|slot).{0,50}\b(marker\|hint) |
| 145 | 15.511 | 13104 | 111 | seeded_from_verified_log | `compiler_tiny_lut_gather`, `compiler_final_equal_overlay`, `compiler_bounded_scan` | tasklog keyword match: \b(marker\|hint).{0,50}\b(target\|slot\|place\|route\|lattice)\|\b(target\|slot).{0,50}\b(marker\|hint) |
| 089 | 16.111 | 7092 | 160 | seeded_from_verified_log | `compiler_tiny_lut_gather`, `compiler_single_conv_qlinear`, `compiler_final_equal_overlay`, `compiler_sparse_scatter` | tasklog keyword match: \b(marker\|hint).{0,50}\b(target\|slot\|place\|route\|lattice)\|\b(target\|slot).{0,50}\b(marker\|hint) |
| 005 | 16.163 | 6392 | 490 | seeded_from_verified_log | `compiler_tiny_lut_gather`, `compiler_single_conv_qlinear`, `compiler_final_equal_overlay`, `compiler_qlinear_uint8` | tasklog keyword match: \b(marker\|hint).{0,50}\b(target\|slot\|place\|route\|lattice)\|\b(target\|slot).{0,50}\b(marker\|hint) |
| 383 | 16.289 | 5974 | 96 | seeded_from_verified_log | `compiler_tiny_lut_gather`, `compiler_single_conv_qlinear`, `compiler_final_equal_overlay`, `compiler_bounded_scan` | tasklog keyword match: \b(marker\|hint).{0,50}\b(target\|slot\|place\|route\|lattice)\|\b(target\|slot).{0,50}\b(marker\|hint) |
| 323 | 17.314 | 1521 | 656 | seeded_from_verified_log | `compiler_single_conv_qlinear`, `compiler_qlinear_uint8` | tasklog keyword match: \b(marker\|hint).{0,50}\b(target\|slot\|place\|route\|lattice)\|\b(target\|slot).{0,50}\b(marker\|hint) |
| 111 | 18.718 | 528 | 7 | seeded_from_verified_log | `compiler_tiny_lut_gather`, `compiler_bounded_scan` | tasklog keyword match: \b(marker\|hint).{0,50}\b(target\|slot\|place\|route\|lattice)\|\b(target\|slot).{0,50}\b(marker\|hint) |

## Known Best Routes

- Pending human review.

## Kill Criteria

- Pending human review.

## Successful Applications

- Pending verification.

## Failed Applications / Walls

- Pending verification.
