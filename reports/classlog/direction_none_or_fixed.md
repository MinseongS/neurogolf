# direction_none_or_fixed — No/fixed direction

No dynamic direction, or direction is a fixed constant.

## Optimization Question

What is the cheapest verified ONNX family for tasks in this class, and
which semantic preconditions let us avoid full-canvas intermediates?

## Candidate Tasks

| task | pts | mem | params | status | first routes | evidence |
|---:|---:|---:|---:|---|---|---|
| 366 | 14.497 | 35927 | 490 | seeded_from_verified_log | `compiler_tiny_lut_gather`, `compiler_single_conv_qlinear`, `compiler_final_equal_overlay`, `compiler_sparse_scatter` | no dynamic direction evidence found |
| 187 | 14.580 | 32850 | 665 | seeded_from_verified_log | `compiler_single_conv_qlinear`, `compiler_final_equal_overlay`, `compiler_bounded_scan` | no dynamic direction evidence found |
| 173 | 15.022 | 21430 | 112 | seeded_from_verified_log | `compiler_tiny_lut_gather`, `compiler_single_conv_qlinear`, `compiler_final_equal_overlay`, `compiler_sparse_scatter` | no dynamic direction evidence found |
| 193 | 15.361 | 15284 | 68 | seeded_from_verified_log | `compiler_tiny_lut_gather`, `compiler_single_conv_qlinear`, `compiler_final_equal_overlay` | no dynamic direction evidence found |
| 255 | 15.387 | 14663 | 293 | seeded_from_verified_log | `compiler_direct_output_algebra`, `compiler_direct_onehot_gather`, `compiler_single_conv_qlinear`, `compiler_final_equal_overlay` | no dynamic direction evidence found |
| 044 | 15.651 | 11420 | 68 | seeded_from_verified_log | `compiler_direct_onehot_gather`, `compiler_tiny_lut_gather`, `compiler_single_conv_qlinear`, `compiler_final_equal_overlay` | no dynamic direction evidence found |
| 328 | 16.179 | 4835 | 1940 | seeded_from_tasklog_and_inventory | `compiler_tiny_lut_gather`, `compiler_final_equal_overlay` | no dynamic direction evidence found |
| 330 | 16.986 | 2800 | 222 | seeded_from_verified_log | `compiler_final_equal_overlay`, `compiler_sparse_scatter`, `compiler_bounded_scan` | no dynamic direction evidence found |
| 397 | 17.046 | 2648 | 200 | seeded_from_tasklog_and_inventory | `compiler_tiny_lut_gather`, `compiler_single_conv_qlinear`, `compiler_final_equal_overlay`, `compiler_sparse_scatter` | no dynamic direction evidence found |
| 022 | 17.258 | 2004 | 298 | seeded_from_verified_log | `compiler_direct_output_algebra`, `compiler_final_equal_overlay`, `compiler_roi_pool_crop` | no dynamic direction evidence found |
| 169 | 17.579 | 1500 | 170 | seeded_from_verified_log | `compiler_tiny_lut_gather`, `compiler_single_conv_qlinear`, `compiler_final_equal_overlay`, `compiler_qlinear_uint8` | no dynamic direction evidence found |
| 295 | 17.620 | 1557 | 47 | seeded_from_tasklog_and_inventory | `compiler_final_equal_overlay` | no dynamic direction evidence found |
| 369 | 17.623 | 1300 | 299 | seeded_from_verified_log | `compiler_single_conv_qlinear`, `compiler_qlinear_uint8` | no dynamic direction evidence found |
| 217 | 17.707 | 1429 | 41 | seeded_from_tasklog_and_inventory | `compiler_tiny_lut_gather`, `compiler_single_conv_qlinear`, `compiler_final_equal_overlay`, `compiler_bounded_scan` | no dynamic direction evidence found |
| 304 | 17.727 | 1404 | 37 | seeded_from_tasklog_and_inventory | `compiler_single_conv_qlinear`, `compiler_final_equal_overlay` | no dynamic direction evidence found |
| 297 | 17.764 | 1350 | 39 | seeded_from_tasklog_and_inventory | `compiler_tiny_lut_gather`, `compiler_final_equal_overlay`, `compiler_bounded_scan` | no dynamic direction evidence found |
| 160 | 17.804 | 1164 | 170 | seeded_from_tasklog_and_inventory | `compiler_single_conv_qlinear`, `compiler_qlinear_uint8`, `compiler_bounded_scan` | no dynamic direction evidence found |
| 239 | 18.147 | 897 | 50 | seeded_from_tasklog_and_inventory | `compiler_tiny_lut_gather`, `compiler_final_equal_overlay` | no dynamic direction evidence found |
| 097 | 18.187 | 0 | 910 | seeded_from_verified_log | `compiler_single_conv_qlinear` | no dynamic direction evidence found |
| 171 | 18.187 | 0 | 910 | seeded_from_tasklog_and_inventory | `compiler_single_conv_qlinear` | no dynamic direction evidence found |
| 294 | 18.187 | 0 | 910 | seeded_from_verified_log | `compiler_single_conv_qlinear` | no dynamic direction evidence found |
| 344 | 18.187 | 0 | 910 | seeded_from_tasklog_and_inventory | `compiler_single_conv_qlinear` | no dynamic direction evidence found |
| 098 | 18.198 | 0 | 900 | operator_evidence_only_needs_human_review | `compiler_single_conv_qlinear` | no dynamic direction evidence found |
| 114 | 18.198 | 0 | 900 | operator_evidence_only_needs_human_review | `compiler_single_conv_qlinear` | no dynamic direction evidence found |
| 151 | 18.198 | 0 | 900 | operator_evidence_only_needs_human_review | `compiler_single_conv_qlinear` | no dynamic direction evidence found |
| 267 | 18.359 | 724 | 42 | seeded_from_tasklog_and_inventory | `compiler_final_equal_overlay` | no dynamic direction evidence found |
| 152 | 18.556 | 504 | 125 | operator_evidence_only_needs_human_review | `compiler_single_conv_qlinear`, `compiler_sparse_scatter` | no dynamic direction evidence found |
| 068 | 18.565 | 500 | 123 | seeded_from_verified_log | `compiler_direct_output_algebra`, `compiler_final_equal_overlay`, `compiler_sparse_scatter` | no dynamic direction evidence found |
| 126 | 18.595 | 590 | 15 | operator_evidence_only_needs_human_review | `compiler_direct_output_algebra`, `compiler_final_equal_overlay`, `compiler_sparse_scatter`, `compiler_bounded_scan` | no dynamic direction evidence found |
| 384 | 18.615 | 540 | 53 | operator_evidence_only_needs_human_review | `compiler_bounded_scan` | no dynamic direction evidence found |
| 362 | 18.744 | 408 | 113 | operator_evidence_only_needs_human_review | `compiler_direct_output_algebra`, `compiler_tiny_lut_gather`, `compiler_final_equal_overlay`, `compiler_sparse_scatter` | no dynamic direction evidence found |
| 081 | 18.860 | 392 | 72 | seeded_from_verified_log | `compiler_single_conv_qlinear`, `compiler_qlinear_uint8` | no dynamic direction evidence found |
| 352 | 18.869 | 0 | 460 | seeded_from_verified_log | `compiler_single_conv_qlinear` | no dynamic direction evidence found |
| 292 | 18.927 | 252 | 182 | seeded_from_tasklog_and_inventory | `compiler_sparse_scatter` | no dynamic direction evidence found |
| 203 | 18.986 | 408 | 1 | seeded_from_tasklog_and_inventory | `compiler_tiny_lut_gather`, `compiler_final_equal_overlay`, `compiler_bounded_scan` | no dynamic direction evidence found |
| 095 | 19.156 | 245 | 100 | seeded_from_verified_log | `compiler_single_conv_qlinear`, `compiler_qlinear_uint8` | no dynamic direction evidence found |
| 229 | 19.519 | 212 | 28 | operator_evidence_only_needs_human_review | `compiler_final_equal_overlay` | no dynamic direction evidence found |
| 176 | 19.519 | 0 | 240 | operator_evidence_only_needs_human_review | `compiler_direct_output_algebra` | no dynamic direction evidence found |
| 236 | 19.532 | 208 | 29 | operator_evidence_only_needs_human_review | `compiler_final_equal_overlay` | no dynamic direction evidence found |
| 322 | 19.702 | 144 | 56 | operator_evidence_only_needs_human_review | `compiler_single_conv_qlinear`, `compiler_final_equal_overlay`, `compiler_bounded_scan` | no dynamic direction evidence found |
| 026 | 19.702 | 0 | 200 | operator_evidence_only_needs_human_review | `compiler_single_conv_qlinear` | no dynamic direction evidence found |
| 135 | 19.702 | 0 | 200 | operator_evidence_only_needs_human_review | `compiler_single_conv_qlinear` | no dynamic direction evidence found |
| 334 | 19.717 | 189 | 8 | seeded_from_verified_log | `compiler_bounded_scan` | no dynamic direction evidence found |
| 313 | 19.743 | 0 | 192 | operator_evidence_only_needs_human_review | `compiler_direct_output_algebra` | no dynamic direction evidence found |
| 376 | 19.774 | 144 | 42 | operator_evidence_only_needs_human_review | `compiler_tiny_lut_gather` | no dynamic direction evidence found |
| 235 | 19.864 | 111 | 59 | operator_evidence_only_needs_human_review | `compiler_direct_onehot_gather`, `compiler_final_equal_overlay` | no dynamic direction evidence found |
| 395 | 19.894 | 144 | 21 | operator_evidence_only_needs_human_review |  | no dynamic direction evidence found |
| 393 | 20.030 | 125 | 19 | operator_evidence_only_needs_human_review | `compiler_direct_output_algebra`, `compiler_final_equal_overlay` | no dynamic direction evidence found |
| 347 | 20.037 | 117 | 26 | operator_evidence_only_needs_human_review | `compiler_direct_output_algebra` | no dynamic direction evidence found |
| 144 | 20.356 | 0 | 104 | operator_evidence_only_needs_human_review | `compiler_direct_output_algebra`, `compiler_single_conv_qlinear` | no dynamic direction evidence found |
| 186 | 20.365 | 80 | 23 | seeded_from_verified_log | `compiler_direct_output_algebra`, `compiler_bounded_scan` | no dynamic direction evidence found |
| 314 | 20.395 | 0 | 100 | operator_evidence_only_needs_human_review | `compiler_direct_output_algebra`, `compiler_single_conv_qlinear` | no dynamic direction evidence found |
| 399 | 20.489 | 71 | 20 | seeded_from_verified_log | `compiler_direct_output_algebra`, `compiler_tiny_lut_gather`, `compiler_final_equal_overlay` | no dynamic direction evidence found |
| 129 | 20.618 | 80 | 0 | seeded_from_verified_log | `compiler_direct_output_algebra` | no dynamic direction evidence found |
| 103 | 20.906 | 36 | 24 | operator_evidence_only_needs_human_review | `compiler_direct_output_algebra`, `compiler_final_equal_overlay`, `compiler_bounded_scan` | no dynamic direction evidence found |
| 299 | 21.088 | 0 | 50 | operator_evidence_only_needs_human_review | `compiler_direct_output_algebra` | no dynamic direction evidence found |
| 056 | 21.474 | 34 | 0 | seeded_from_verified_log | `compiler_direct_output_algebra`, `compiler_final_equal_overlay` | no dynamic direction evidence found |
| 116 | 21.599 | 0 | 30 | operator_evidence_only_needs_human_review | `compiler_direct_output_algebra`, `compiler_tiny_lut_gather` | no dynamic direction evidence found |
| 130 | 21.599 | 0 | 30 | seeded_from_verified_log | `compiler_direct_output_algebra`, `compiler_single_conv_qlinear` | no dynamic direction evidence found |
| 164 | 21.599 | 0 | 30 | operator_evidence_only_needs_human_review | `compiler_direct_output_algebra`, `compiler_tiny_lut_gather` | no dynamic direction evidence found |
| 172 | 21.599 | 0 | 30 | operator_evidence_only_needs_human_review | `compiler_direct_output_algebra`, `compiler_tiny_lut_gather` | no dynamic direction evidence found |
| 210 | 21.599 | 0 | 30 | operator_evidence_only_needs_human_review | `compiler_direct_output_algebra`, `compiler_tiny_lut_gather` | no dynamic direction evidence found |
| 311 | 21.599 | 0 | 30 | operator_evidence_only_needs_human_review | `compiler_direct_output_algebra`, `compiler_tiny_lut_gather` | no dynamic direction evidence found |
| 385 | 21.599 | 0 | 30 | operator_evidence_only_needs_human_review | `compiler_direct_output_algebra`, `compiler_tiny_lut_gather` | no dynamic direction evidence found |
| 016 | 22.697 | 0 | 10 | operator_evidence_only_needs_human_review | `compiler_direct_output_algebra`, `compiler_tiny_lut_gather` | no dynamic direction evidence found |
| 276 | 22.697 | 0 | 10 | operator_evidence_only_needs_human_review | `compiler_direct_output_algebra`, `compiler_tiny_lut_gather` | no dynamic direction evidence found |
| 309 | 22.697 | 0 | 10 | operator_evidence_only_needs_human_review | `compiler_direct_output_algebra`, `compiler_tiny_lut_gather` | no dynamic direction evidence found |
| 337 | 22.697 | 0 | 10 | operator_evidence_only_needs_human_review | `compiler_direct_output_algebra`, `compiler_tiny_lut_gather` | no dynamic direction evidence found |
| 087 | 23.391 | 0 | 5 | operator_evidence_only_needs_human_review | `compiler_direct_output_algebra`, `compiler_roi_pool_crop` | no dynamic direction evidence found |
| 223 | 23.391 | 0 | 5 | operator_evidence_only_needs_human_review | `compiler_direct_output_algebra`, `compiler_roi_pool_crop` | no dynamic direction evidence found |
| 307 | 23.391 | 0 | 5 | operator_evidence_only_needs_human_review | `compiler_direct_output_algebra`, `compiler_roi_pool_crop` | no dynamic direction evidence found |
| 179 | 25.000 | 0 | 0 | operator_evidence_only_needs_human_review | `compiler_direct_output_algebra` | no dynamic direction evidence found |
| 241 | 25.000 | 0 | 0 | operator_evidence_only_needs_human_review | `compiler_direct_output_algebra` | no dynamic direction evidence found |

## Known Best Routes

- Pending human review.

## Kill Criteria

- Pending human review.

## Successful Applications

- Pending verification.

## Failed Applications / Walls

- Pending verification.
