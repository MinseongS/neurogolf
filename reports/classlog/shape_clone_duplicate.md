# shape_clone_duplicate — Clone/duplicate shape

A shape is repeated one or more times.

## Optimization Question

What is the cheapest verified ONNX family for tasks in this class, and
which semantic preconditions let us avoid full-canvas intermediates?

## Candidate Tasks

| task | pts | mem | params | status | first routes | evidence |
|---:|---:|---:|---:|---|---|---|
| 107 | 16.687 | 2924 | 1154 | seeded_from_verified_log | `compiler_direct_onehot_gather`, `compiler_tiny_lut_gather`, `compiler_final_equal_overlay`, `compiler_sparse_scatter` | larger output often indicates expansion/repetition candidate |
| 398 | 17.042 | 1666 | 1193 | seeded_from_verified_log | `compiler_direct_onehot_gather`, `compiler_tiny_lut_gather`, `compiler_final_equal_overlay` | larger output often indicates expansion/repetition candidate |
| 019 | 17.063 | 2709 | 89 | seeded_from_verified_log | `compiler_tiny_lut_gather`, `compiler_single_conv_qlinear`, `compiler_final_equal_overlay`, `compiler_qlinear_uint8` | larger output often indicates expansion/repetition candidate |
| 388 | 17.178 | 2278 | 218 | seeded_from_verified_log | `compiler_final_equal_overlay`, `compiler_sparse_scatter`, `compiler_bounded_scan` | larger output often indicates expansion/repetition candidate |
| 275 | 17.394 | 1374 | 637 | seeded_from_verified_log | `compiler_direct_output_algebra`, `compiler_single_conv_qlinear`, `compiler_final_equal_overlay`, `compiler_bounded_scan` | larger output often indicates expansion/repetition candidate |
| 124 | 17.423 | 1879 | 74 | seeded_from_verified_log | `compiler_direct_onehot_gather`, `compiler_tiny_lut_gather`, `compiler_final_equal_overlay`, `compiler_bounded_scan` | larger output often indicates expansion/repetition candidate |
| 295 | 17.620 | 1557 | 47 | seeded_from_tasklog_and_inventory | `compiler_final_equal_overlay` | larger output often indicates expansion/repetition candidate |
| 304 | 17.727 | 1404 | 37 | seeded_from_tasklog_and_inventory | `compiler_single_conv_qlinear`, `compiler_final_equal_overlay` | larger output often indicates expansion/repetition candidate |
| 123 | 17.797 | 403 | 940 | seeded_from_verified_log | `compiler_direct_output_algebra`, `compiler_tiny_lut_gather`, `compiler_final_equal_overlay` | larger output often indicates expansion/repetition candidate |
| 108 | 18.122 | 864 | 107 | seeded_from_verified_log | `compiler_single_conv_qlinear`, `compiler_final_equal_overlay`, `compiler_roi_pool_crop` | larger output often indicates expansion/repetition candidate |
| 194 | 18.145 | 900 | 49 | seeded_from_verified_log | `compiler_tiny_lut_gather` | larger output often indicates expansion/repetition candidate |
| 239 | 18.147 | 897 | 50 | seeded_from_tasklog_and_inventory | `compiler_tiny_lut_gather`, `compiler_final_equal_overlay` | larger output often indicates expansion/repetition candidate |
| 114 | 18.198 | 0 | 900 | operator_evidence_only_needs_human_review | `compiler_single_conv_qlinear` | larger output often indicates expansion/repetition candidate |
| 327 | 18.219 | 810 | 71 | seeded_from_verified_log | `compiler_single_conv_qlinear`, `compiler_final_equal_overlay` | larger output often indicates expansion/repetition candidate |
| 106 | 18.233 | 776 | 93 | seeded_from_verified_log | `compiler_tiny_lut_gather`, `compiler_single_conv_qlinear`, `compiler_sparse_scatter` | larger output often indicates expansion/repetition candidate |
| 221 | 18.309 | 451 | 354 | seeded_from_tasklog_and_inventory | `compiler_direct_output_algebra`, `compiler_final_equal_overlay`, `compiler_bounded_scan` | larger output often indicates expansion/repetition candidate |
| 289 | 18.391 | 618 | 124 | seeded_from_verified_log | `compiler_direct_output_algebra`, `compiler_final_equal_overlay`, `compiler_bounded_scan` | larger output often indicates expansion/repetition candidate |
| 269 | 18.436 | 685 | 24 | seeded_from_tasklog_and_inventory | `compiler_tiny_lut_gather`, `compiler_roi_pool_crop` | larger output often indicates expansion/repetition candidate |
| 152 | 18.556 | 504 | 125 | operator_evidence_only_needs_human_review | `compiler_single_conv_qlinear`, `compiler_sparse_scatter` | larger output often indicates expansion/repetition candidate |
| 083 | 18.897 | 376 | 71 | seeded_from_verified_log | `compiler_tiny_lut_gather`, `compiler_sparse_scatter`, `compiler_bounded_scan` | larger output often indicates expansion/repetition candidate |
| 211 | 19.019 | 240 | 156 | seeded_from_verified_log | `compiler_direct_output_algebra` | larger output often indicates expansion/repetition candidate |
| 003 | 19.290 | 281 | 21 | seeded_from_verified_log | `compiler_tiny_lut_gather`, `compiler_final_equal_overlay` | larger output often indicates expansion/repetition candidate |
| 249 | 19.341 | 248 | 39 | seeded_from_verified_log | `compiler_tiny_lut_gather` | larger output often indicates expansion/repetition candidate |
| 142 | 19.358 | 144 | 138 | seeded_from_verified_log | `compiler_direct_output_algebra` | larger output often indicates expansion/repetition candidate |
| 104 | 19.463 | 197 | 57 | seeded_from_verified_log | `compiler_direct_onehot_gather`, `compiler_tiny_lut_gather`, `compiler_single_conv_qlinear`, `compiler_qlinear_uint8` | larger output often indicates expansion/repetition candidate |
| 001 | 19.519 | 0 | 240 | seeded_from_verified_log | `compiler_direct_output_algebra`, `compiler_tiny_lut_gather` | larger output often indicates expansion/repetition candidate |
| 315 | 19.562 | 0 | 230 | seeded_from_verified_log | `compiler_direct_output_algebra` | larger output often indicates expansion/repetition candidate |
| 231 | 19.611 | 180 | 39 | seeded_from_verified_log | `compiler_tiny_lut_gather` | larger output often indicates expansion/repetition candidate |
| 376 | 19.774 | 144 | 42 | operator_evidence_only_needs_human_review | `compiler_tiny_lut_gather` | larger output often indicates expansion/repetition candidate |
| 116 | 21.599 | 0 | 30 | operator_evidence_only_needs_human_review | `compiler_direct_output_algebra`, `compiler_tiny_lut_gather` | larger output often indicates expansion/repetition candidate |
| 164 | 21.599 | 0 | 30 | operator_evidence_only_needs_human_review | `compiler_direct_output_algebra`, `compiler_tiny_lut_gather` | larger output often indicates expansion/repetition candidate |
| 172 | 21.599 | 0 | 30 | operator_evidence_only_needs_human_review | `compiler_direct_output_algebra`, `compiler_tiny_lut_gather` | larger output often indicates expansion/repetition candidate |
| 210 | 21.599 | 0 | 30 | operator_evidence_only_needs_human_review | `compiler_direct_output_algebra`, `compiler_tiny_lut_gather` | larger output often indicates expansion/repetition candidate |
| 311 | 21.599 | 0 | 30 | operator_evidence_only_needs_human_review | `compiler_direct_output_algebra`, `compiler_tiny_lut_gather` | larger output often indicates expansion/repetition candidate |
| 223 | 23.391 | 0 | 5 | operator_evidence_only_needs_human_review | `compiler_direct_output_algebra`, `compiler_roi_pool_crop` | larger output often indicates expansion/repetition candidate |
| 307 | 23.391 | 0 | 5 | operator_evidence_only_needs_human_review | `compiler_direct_output_algebra`, `compiler_roi_pool_crop` | larger output often indicates expansion/repetition candidate |

## Known Best Routes

- Pending human review.

## Kill Criteria

- Pending human review.

## Successful Applications

- Pending verification.

## Failed Applications / Walls

- Pending verification.
