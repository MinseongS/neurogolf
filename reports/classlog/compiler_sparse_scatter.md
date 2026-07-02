# compiler_sparse_scatter — Sparse scatter/edit stream

Use sparse coordinate updates instead of full masks where safe.

## Optimization Question

What is the cheapest verified ONNX family for tasks in this class, and
which semantic preconditions let us avoid full-canvas intermediates?

## Candidate Tasks

| task | pts | mem | params | status | first routes | evidence |
|---:|---:|---:|---:|---|---|---|
| 233 | 14.003 | 59147 | 565 | seeded_from_verified_log | `compiler_direct_onehot_gather`, `compiler_tiny_lut_gather`, `compiler_single_conv_qlinear`, `compiler_final_equal_overlay` | Scatter operator present; scatter tag |
| 018 | 14.157 | 48196 | 2987 | seeded_from_verified_log | `compiler_tiny_lut_gather`, `compiler_single_conv_qlinear`, `compiler_final_equal_overlay`, `compiler_sparse_scatter` | Scatter operator present; scatter tag |
| 366 | 14.497 | 35927 | 490 | seeded_from_verified_log | `compiler_tiny_lut_gather`, `compiler_single_conv_qlinear`, `compiler_final_equal_overlay`, `compiler_sparse_scatter` | Scatter operator present; scatter tag |
| 118 | 14.559 | 30849 | 3387 | seeded_from_verified_log | `compiler_direct_output_algebra`, `compiler_direct_onehot_gather`, `compiler_tiny_lut_gather`, `compiler_single_conv_qlinear` | Scatter operator present; scatter tag |
| 133 | 14.578 | 32294 | 1288 | seeded_from_verified_log | `compiler_single_conv_qlinear`, `compiler_final_equal_overlay`, `compiler_qlinear_uint8`, `compiler_roi_pool_crop` | Scatter operator present; scatter tag |
| 054 | 14.829 | 25885 | 238 | seeded_from_tasklog_and_inventory | `compiler_tiny_lut_gather`, `compiler_single_conv_qlinear`, `compiler_final_equal_overlay`, `compiler_sparse_scatter` | Scatter operator present; scatter tag |
| 285 | 14.848 | 25080 | 550 | seeded_from_verified_log | `compiler_tiny_lut_gather`, `compiler_single_conv_qlinear`, `compiler_final_equal_overlay`, `compiler_sparse_scatter` | Scatter operator present; scatter tag |
| 076 | 14.932 | 23296 | 292 | seeded_from_verified_log | `compiler_tiny_lut_gather`, `compiler_final_equal_overlay`, `compiler_sparse_scatter`, `compiler_bounded_scan` | Scatter operator present; scatter tag |
| 173 | 15.022 | 21430 | 112 | seeded_from_verified_log | `compiler_tiny_lut_gather`, `compiler_single_conv_qlinear`, `compiler_final_equal_overlay`, `compiler_sparse_scatter` | Scatter operator present; scatter tag |
| 101 | 15.216 | 16850 | 905 | seeded_from_verified_log | `compiler_tiny_lut_gather`, `compiler_final_equal_overlay`, `compiler_sparse_scatter`, `compiler_bounded_scan` | Scatter operator present; scatter tag |
| 370 | 15.823 | 8645 | 1024 | seeded_from_verified_log | `compiler_direct_output_algebra`, `compiler_tiny_lut_gather`, `compiler_single_conv_qlinear`, `compiler_final_equal_overlay` | Scatter operator present; scatter tag |
| 216 | 15.880 | 9048 | 87 | seeded_from_verified_log | `compiler_direct_onehot_gather`, `compiler_tiny_lut_gather`, `compiler_single_conv_qlinear`, `compiler_final_equal_overlay` | Scatter operator present; scatter tag |
| 096 | 15.905 | 8108 | 805 | seeded_from_verified_log | `compiler_tiny_lut_gather`, `compiler_final_equal_overlay`, `compiler_sparse_scatter`, `compiler_bounded_scan` | Scatter operator present; scatter tag |
| 157 | 15.972 | 7983 | 351 | seeded_from_verified_log | `compiler_tiny_lut_gather`, `compiler_final_equal_overlay`, `compiler_sparse_scatter`, `compiler_bounded_scan` | Scatter operator present; scatter tag |
| 089 | 16.111 | 7092 | 160 | seeded_from_verified_log | `compiler_tiny_lut_gather`, `compiler_single_conv_qlinear`, `compiler_final_equal_overlay`, `compiler_sparse_scatter` | Scatter operator present; scatter tag |
| 092 | 16.148 | 6805 | 182 | seeded_from_verified_log | `compiler_direct_output_algebra`, `compiler_final_equal_overlay`, `compiler_sparse_scatter`, `compiler_bounded_scan` | Scatter operator present; scatter tag |
| 368 | 16.264 | 6022 | 203 | seeded_from_verified_log | `compiler_tiny_lut_gather`, `compiler_final_equal_overlay`, `compiler_sparse_scatter` | Scatter operator present; scatter tag |
| 148 | 16.323 | 5759 | 110 | seeded_from_verified_log | `compiler_final_equal_overlay`, `compiler_sparse_scatter`, `compiler_bounded_scan` | Scatter operator present; scatter tag |
| 387 | 16.471 | 4954 | 105 | seeded_from_verified_log | `compiler_tiny_lut_gather`, `compiler_final_equal_overlay`, `compiler_sparse_scatter` | Scatter operator present; scatter tag |
| 037 | 16.535 | 4614 | 132 | seeded_from_verified_log | `compiler_tiny_lut_gather`, `compiler_final_equal_overlay`, `compiler_sparse_scatter`, `compiler_bounded_scan` | Scatter operator present; scatter tag |
| 361 | 16.547 | 4322 | 366 | seeded_from_verified_log | `compiler_tiny_lut_gather`, `compiler_single_conv_qlinear`, `compiler_final_equal_overlay`, `compiler_roi_pool_crop` | Scatter operator present; scatter tag |
| 265 | 16.619 | 4328 | 34 | seeded_from_verified_log | `compiler_tiny_lut_gather`, `compiler_final_equal_overlay`, `compiler_sparse_scatter`, `compiler_bounded_scan` | Scatter operator present; scatter tag |
| 206 | 16.657 | 4122 | 77 | seeded_from_verified_log | `compiler_single_conv_qlinear`, `compiler_final_equal_overlay`, `compiler_sparse_scatter`, `compiler_bounded_scan` | Scatter operator present; scatter tag |
| 107 | 16.687 | 2924 | 1154 | seeded_from_verified_log | `compiler_direct_onehot_gather`, `compiler_tiny_lut_gather`, `compiler_final_equal_overlay`, `compiler_sparse_scatter` | Scatter operator present; scatter tag |
| 131 | 16.709 | 3312 | 675 | seeded_from_verified_log | `compiler_sparse_scatter`, `compiler_bounded_scan` | Scatter operator present; scatter tag |
| 284 | 16.774 | 3082 | 653 | seeded_from_verified_log | `compiler_single_conv_qlinear`, `compiler_sparse_scatter` | Scatter operator present; scatter tag |
| 035 | 16.954 | 2994 | 128 | seeded_from_verified_log | `compiler_sparse_scatter` | Scatter operator present; scatter tag |
| 390 | 16.964 | 2624 | 467 | seeded_from_verified_log | `compiler_final_equal_overlay`, `compiler_sparse_scatter` | Scatter operator present; scatter tag |
| 330 | 16.986 | 2800 | 222 | seeded_from_verified_log | `compiler_final_equal_overlay`, `compiler_sparse_scatter`, `compiler_bounded_scan` | Scatter operator present; scatter tag |
| 070 | 16.991 | 2977 | 31 | seeded_from_verified_log | `compiler_tiny_lut_gather`, `compiler_sparse_scatter`, `compiler_bounded_scan` | Scatter operator present; scatter tag |
| 397 | 17.046 | 2648 | 200 | seeded_from_tasklog_and_inventory | `compiler_tiny_lut_gather`, `compiler_single_conv_qlinear`, `compiler_final_equal_overlay`, `compiler_sparse_scatter` | Scatter operator present; scatter tag |
| 388 | 17.178 | 2278 | 218 | seeded_from_verified_log | `compiler_final_equal_overlay`, `compiler_sparse_scatter`, `compiler_bounded_scan` | Scatter operator present; scatter tag |
| 012 | 17.275 | 2076 | 188 | seeded_from_verified_log | `compiler_tiny_lut_gather`, `compiler_final_equal_overlay`, `compiler_sparse_scatter` | Scatter operator present; scatter tag |
| 336 | 17.298 | 1188 | 1024 | seeded_from_verified_log | `compiler_tiny_lut_gather`, `compiler_sparse_scatter` | Scatter operator present; scatter tag |
| 260 | 17.344 | 1804 | 310 | seeded_from_verified_log | `compiler_tiny_lut_gather`, `compiler_final_equal_overlay`, `compiler_sparse_scatter`, `compiler_bounded_scan` | Scatter operator present; scatter tag |
| 084 | 17.440 | 1837 | 83 | seeded_from_verified_log | `compiler_sparse_scatter` | Scatter operator present; scatter tag |
| 099 | 17.445 | 1633 | 278 | seeded_from_verified_log | `compiler_tiny_lut_gather`, `compiler_final_equal_overlay`, `compiler_sparse_scatter`, `compiler_bounded_scan` | Scatter operator present; scatter tag |
| 199 | 17.484 | 1749 | 88 | seeded_from_verified_log | `compiler_direct_output_algebra`, `compiler_final_equal_overlay`, `compiler_sparse_scatter` | Scatter operator present; scatter tag |
| 342 | 17.632 | 1510 | 75 | seeded_from_verified_log | `compiler_direct_output_algebra`, `compiler_tiny_lut_gather`, `compiler_final_equal_overlay`, `compiler_sparse_scatter` | Scatter operator present; scatter tag |
| 020 | 17.670 | 1356 | 170 | seeded_from_verified_log | `compiler_tiny_lut_gather`, `compiler_single_conv_qlinear`, `compiler_sparse_scatter`, `compiler_bounded_scan` | Scatter operator present; scatter tag |
| 308 | 17.695 | 1385 | 103 | seeded_from_verified_log | `compiler_direct_output_algebra`, `compiler_final_equal_overlay`, `compiler_sparse_scatter`, `compiler_bounded_scan` | Scatter operator present; scatter tag |
| 119 | 17.781 | 1244 | 121 | seeded_from_verified_log | `compiler_tiny_lut_gather`, `compiler_sparse_scatter`, `compiler_bounded_scan` | Scatter operator present; scatter tag |
| 293 | 17.839 | 1247 | 41 | seeded_from_verified_log | `compiler_tiny_lut_gather`, `compiler_final_equal_overlay`, `compiler_sparse_scatter` | Scatter operator present; scatter tag |
| 346 | 17.866 | 1224 | 30 | seeded_from_verified_log | `compiler_tiny_lut_gather`, `compiler_single_conv_qlinear`, `compiler_final_equal_overlay`, `compiler_qlinear_uint8` | Scatter operator present; scatter tag |
| 329 | 18.035 | 1029 | 30 | seeded_from_verified_log | `compiler_final_equal_overlay`, `compiler_sparse_scatter` | Scatter operator present; scatter tag |
| 228 | 18.131 | 849 | 113 | seeded_from_verified_log | `compiler_direct_output_algebra`, `compiler_tiny_lut_gather`, `compiler_final_equal_overlay`, `compiler_sparse_scatter` | Scatter operator present; scatter tag |
| 288 | 18.135 | 769 | 189 | seeded_from_verified_log | `compiler_tiny_lut_gather`, `compiler_final_equal_overlay`, `compiler_sparse_scatter`, `compiler_bounded_scan` | Scatter operator present; scatter tag |
| 106 | 18.233 | 776 | 93 | seeded_from_verified_log | `compiler_tiny_lut_gather`, `compiler_single_conv_qlinear`, `compiler_sparse_scatter` | Scatter operator present; scatter tag |
| 320 | 18.354 | 448 | 322 | seeded_from_verified_log | `compiler_single_conv_qlinear`, `compiler_sparse_scatter` | Scatter operator present; scatter tag |
| 152 | 18.556 | 504 | 125 | operator_evidence_only_needs_human_review | `compiler_single_conv_qlinear`, `compiler_sparse_scatter` | Scatter operator present; scatter tag |
| 068 | 18.565 | 500 | 123 | seeded_from_verified_log | `compiler_direct_output_algebra`, `compiler_final_equal_overlay`, `compiler_sparse_scatter` | Scatter operator present; scatter tag |
| 126 | 18.595 | 590 | 15 | operator_evidence_only_needs_human_review | `compiler_direct_output_algebra`, `compiler_final_equal_overlay`, `compiler_sparse_scatter`, `compiler_bounded_scan` | Scatter operator present; scatter tag |
| 147 | 18.723 | 432 | 100 | seeded_from_verified_log | `compiler_single_conv_qlinear`, `compiler_sparse_scatter` | Scatter operator present; scatter tag |
| 362 | 18.744 | 408 | 113 | operator_evidence_only_needs_human_review | `compiler_direct_output_algebra`, `compiler_tiny_lut_gather`, `compiler_final_equal_overlay`, `compiler_sparse_scatter` | Scatter operator present; scatter tag |
| 371 | 18.854 | 376 | 91 | seeded_from_verified_log | `compiler_direct_output_algebra`, `compiler_sparse_scatter` | Scatter operator present; scatter tag |
| 083 | 18.897 | 376 | 71 | seeded_from_verified_log | `compiler_tiny_lut_gather`, `compiler_sparse_scatter`, `compiler_bounded_scan` | Scatter operator present; scatter tag |
| 292 | 18.927 | 252 | 182 | seeded_from_tasklog_and_inventory | `compiler_sparse_scatter` | Scatter operator present; scatter tag |
| 181 | 19.089 | 201 | 168 | seeded_from_verified_log | `compiler_sparse_scatter` | Scatter operator present; scatter tag |
| 353 | 19.189 | 272 | 62 | seeded_from_verified_log | `compiler_direct_output_algebra`, `compiler_sparse_scatter` | Scatter operator present; scatter tag |
| 248 | 19.983 | 122 | 29 | seeded_from_verified_log | `compiler_sparse_scatter` | Scatter operator present; scatter tag |
| 389 | 20.058 | 128 | 12 | seeded_from_verified_log | `compiler_direct_output_algebra`, `compiler_tiny_lut_gather`, `compiler_sparse_scatter`, `compiler_bounded_scan` | Scatter operator present; scatter tag |

## Known Best Routes

- Pending human review.

## Kill Criteria

- Pending human review.

## Successful Applications

- Pending verification.

## Failed Applications / Walls

- Pending verification.
