# action_reorder_pack — Reorder/pack/sort

Objects are sorted, packed, ranked, or rearranged.

## Optimization Question

What is the cheapest verified ONNX family for tasks in this class, and
which semantic preconditions let us avoid full-canvas intermediates?

## Candidate Tasks

| task | pts | mem | params | status | first routes | evidence |
|---:|---:|---:|---:|---|---|---|
| 396 | 15.633 | 11567 | 133 | seeded_from_verified_log | `compiler_single_conv_qlinear`, `compiler_final_equal_overlay`, `compiler_roi_pool_crop`, `compiler_bounded_scan` | tasklog keyword match: \b(sort\|reorder\|rank order\|arrange objects\|packed objects) |
| 079 | 16.947 | 3065 | 78 | seeded_from_verified_log | `compiler_single_conv_qlinear`, `compiler_final_equal_overlay`, `compiler_bounded_scan` | tasklog keyword match: \b(sort\|reorder\|rank order\|arrange objects\|packed objects) |
| 374 | 17.258 | 2262 | 42 | seeded_from_verified_log | `compiler_final_equal_overlay`, `compiler_bounded_scan` | tasklog keyword match: \b(sort\|reorder\|rank order\|arrange objects\|packed objects) |
| 012 | 17.275 | 2076 | 188 | seeded_from_verified_log | `compiler_tiny_lut_gather`, `compiler_final_equal_overlay`, `compiler_sparse_scatter` | tasklog keyword match: \b(sort\|reorder\|rank order\|arrange objects\|packed objects) |
| 271 | 17.921 | 1141 | 46 | seeded_from_verified_log | `compiler_single_conv_qlinear`, `compiler_qlinear_uint8` | tasklog keyword match: \b(sort\|reorder\|rank order\|arrange objects\|packed objects) |
| 301 | 17.960 | 1077 | 64 | seeded_from_verified_log | `compiler_tiny_lut_gather`, `compiler_final_equal_overlay`, `compiler_bounded_scan` | tasklog keyword match: \b(sort\|reorder\|rank order\|arrange objects\|packed objects) |
| 010 | 18.025 | 730 | 340 | seeded_from_verified_log | `compiler_tiny_lut_gather`, `compiler_final_equal_overlay` | tasklog keyword match: \b(sort\|reorder\|rank order\|arrange objects\|packed objects) |
| 015 | 18.198 | 0 | 900 | seeded_from_verified_log | `compiler_single_conv_qlinear` | tasklog keyword match: \b(sort\|reorder\|rank order\|arrange objects\|packed objects) |
| 263 | 18.478 | 563 | 117 | seeded_from_verified_log | `compiler_tiny_lut_gather`, `compiler_single_conv_qlinear`, `compiler_final_equal_overlay` | tasklog keyword match: \b(sort\|reorder\|rank order\|arrange objects\|packed objects) |
| 300 | 18.708 | 474 | 66 | seeded_from_verified_log | `compiler_direct_output_algebra`, `compiler_tiny_lut_gather`, `compiler_final_equal_overlay` | tasklog keyword match: \b(sort\|reorder\|rank order\|arrange objects\|packed objects) |
| 316 | 18.946 | 388 | 38 | seeded_from_verified_log | `compiler_direct_output_algebra`, `compiler_tiny_lut_gather`, `compiler_final_equal_overlay` | tasklog keyword match: \b(sort\|reorder\|rank order\|arrange objects\|packed objects) |
| 391 | 19.931 | 135 | 24 | seeded_from_verified_log | `compiler_tiny_lut_gather`, `compiler_final_equal_overlay` | tasklog keyword match: \b(sort\|reorder\|rank order\|arrange objects\|packed objects) |

## Known Best Routes

- Pending human review.

## Kill Criteria

- Pending human review.

## Successful Applications

- Pending verification.

## Failed Applications / Walls

- Pending verification.
