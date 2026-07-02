# direction_gravity — Gravity/drop direction

Objects move/drop/fall toward an edge or obstacle.

## Optimization Question

What is the cheapest verified ONNX family for tasks in this class, and
which semantic preconditions let us avoid full-canvas intermediates?

## Candidate Tasks

| task | pts | mem | params | status | first routes | evidence |
|---:|---:|---:|---:|---|---|---|
| 396 | 15.633 | 11567 | 133 | seeded_from_verified_log | `compiler_single_conv_qlinear`, `compiler_final_equal_overlay`, `compiler_roi_pool_crop`, `compiler_bounded_scan` | tasklog keyword match: \b(gravity\|fall\|falling\|settle\|settling) |
| 379 | 15.818 | 9069 | 653 | seeded_from_verified_log | `compiler_single_conv_qlinear`, `compiler_final_equal_overlay`, `compiler_bounded_scan` | tasklog keyword match: \b(gravity\|fall\|falling\|settle\|settling) |
| 382 | 16.337 | 5648 | 135 | seeded_from_verified_log | `compiler_direct_output_algebra`, `compiler_tiny_lut_gather`, `compiler_final_equal_overlay`, `compiler_bounded_scan` | tasklog keyword match: \b(gravity\|fall\|falling\|settle\|settling) |
| 117 | 16.666 | 3922 | 243 | seeded_from_verified_log | `compiler_tiny_lut_gather`, `compiler_single_conv_qlinear`, `compiler_final_equal_overlay`, `compiler_qlinear_uint8` | tasklog keyword match: \b(gravity\|fall\|falling\|settle\|settling) |
| 132 | 16.684 | 3990 | 99 | seeded_from_verified_log | `compiler_tiny_lut_gather`, `compiler_single_conv_qlinear`, `compiler_final_equal_overlay`, `compiler_bounded_scan` | tasklog keyword match: \b(gravity\|fall\|falling\|settle\|settling) |
| 093 | 16.823 | 3456 | 103 | seeded_from_verified_log | `compiler_tiny_lut_gather`, `compiler_final_equal_overlay`, `compiler_qlinear_uint8`, `compiler_bounded_scan` | tasklog keyword match: \b(gravity\|fall\|falling\|settle\|settling) |
| 310 | 16.902 | 3208 | 79 | seeded_from_verified_log | `compiler_direct_output_algebra`, `compiler_tiny_lut_gather`, `compiler_final_equal_overlay` | tasklog keyword match: \b(gravity\|fall\|falling\|settle\|settling) |
| 079 | 16.947 | 3065 | 78 | seeded_from_verified_log | `compiler_single_conv_qlinear`, `compiler_final_equal_overlay`, `compiler_bounded_scan` | tasklog keyword match: \b(gravity\|fall\|falling\|settle\|settling) |
| 246 | 17.080 | 2334 | 419 | seeded_from_verified_log | `compiler_single_conv_qlinear`, `compiler_final_equal_overlay` | tasklog keyword match: \b(gravity\|fall\|falling\|settle\|settling) |
| 012 | 17.275 | 2076 | 188 | seeded_from_verified_log | `compiler_tiny_lut_gather`, `compiler_final_equal_overlay`, `compiler_sparse_scatter` | tasklog keyword match: \b(gravity\|fall\|falling\|settle\|settling) |
| 348 | 17.353 | 1938 | 157 | seeded_from_verified_log | `compiler_final_equal_overlay` | tasklog keyword match: \b(gravity\|fall\|falling\|settle\|settling) |
| 051 | 17.503 | 1702 | 101 | seeded_from_verified_log | `compiler_direct_output_algebra`, `compiler_final_equal_overlay` | tasklog keyword match: \b(gravity\|fall\|falling\|settle\|settling) |
| 341 | 17.735 | 1394 | 35 | seeded_from_verified_log | `compiler_final_equal_overlay`, `compiler_bounded_scan` | tasklog keyword match: \b(gravity\|fall\|falling\|settle\|settling) |
| 119 | 17.781 | 1244 | 121 | seeded_from_verified_log | `compiler_tiny_lut_gather`, `compiler_sparse_scatter`, `compiler_bounded_scan` | tasklog keyword match: \b(gravity\|fall\|falling\|settle\|settling) |
| 293 | 17.839 | 1247 | 41 | seeded_from_verified_log | `compiler_tiny_lut_gather`, `compiler_final_equal_overlay`, `compiler_sparse_scatter` | tasklog keyword match: \b(gravity\|fall\|falling\|settle\|settling) |
| 183 | 17.885 | 1008 | 222 | seeded_from_verified_log | `compiler_direct_output_algebra`, `compiler_final_equal_overlay` | tasklog keyword match: \b(gravity\|fall\|falling\|settle\|settling) |
| 228 | 18.131 | 849 | 113 | seeded_from_verified_log | `compiler_direct_output_algebra`, `compiler_tiny_lut_gather`, `compiler_final_equal_overlay`, `compiler_sparse_scatter` | tasklog keyword match: \b(gravity\|fall\|falling\|settle\|settling) |
| 032 | 18.187 | 0 | 910 | seeded_from_verified_log | `compiler_single_conv_qlinear` | tasklog keyword match: \b(gravity\|fall\|falling\|settle\|settling) |
| 331 | 18.187 | 0 | 910 | seeded_from_verified_log | `compiler_single_conv_qlinear` | tasklog keyword match: \b(gravity\|fall\|falling\|settle\|settling) |
| 230 | 18.198 | 0 | 900 | seeded_from_verified_log | `compiler_tiny_lut_gather`, `compiler_single_conv_qlinear` | tasklog keyword match: \b(gravity\|fall\|falling\|settle\|settling) |
| 083 | 18.897 | 376 | 71 | seeded_from_verified_log | `compiler_tiny_lut_gather`, `compiler_sparse_scatter`, `compiler_bounded_scan` | tasklog keyword match: \b(gravity\|fall\|falling\|settle\|settling) |
| 266 | 19.142 | 315 | 35 | seeded_from_verified_log | `compiler_tiny_lut_gather` | tasklog keyword match: \b(gravity\|fall\|falling\|settle\|settling) |
| 375 | 19.142 | 108 | 242 | seeded_from_verified_log | `compiler_direct_output_algebra` | tasklog keyword match: \b(gravity\|fall\|falling\|settle\|settling) |
| 386 | 19.648 | 180 | 31 | seeded_from_verified_log |  | tasklog keyword match: \b(gravity\|fall\|falling\|settle\|settling) |
| 150 | 19.937 | 128 | 30 | seeded_from_verified_log | `compiler_tiny_lut_gather` | tasklog keyword match: \b(gravity\|fall\|falling\|settle\|settling) |
| 339 | 20.058 | 40 | 100 | seeded_from_verified_log | `compiler_direct_output_algebra` | tasklog keyword match: \b(gravity\|fall\|falling\|settle\|settling) |
| 053 | 21.599 | 0 | 30 | seeded_from_verified_log | `compiler_direct_output_algebra`, `compiler_tiny_lut_gather` | tasklog keyword match: \b(gravity\|fall\|falling\|settle\|settling) |

## Known Best Routes

- Pending human review.

## Kill Criteria

- Pending human review.

## Successful Applications

- Pending verification.

## Failed Applications / Walls

- Pending verification.
