# cost_mem0_param_game — Mem0 param game

Memory is zero/tiny; improvements must reduce params.

## Optimization Question

What is the cheapest verified ONNX family for tasks in this class, and
which semantic preconditions let us avoid full-canvas intermediates?

## Candidate Tasks

| task | pts | mem | params | status | first routes | evidence |
|---:|---:|---:|---:|---|---|---|
| 232 | 17.749 | 0 | 1410 | seeded_from_tasklog_and_inventory | `compiler_tiny_lut_gather`, `compiler_single_conv_qlinear` | memory is 0; params dominate |
| 060 | 18.082 | 0 | 1010 | seeded_from_tasklog_and_inventory | `compiler_single_conv_qlinear` | memory is 0; params dominate |
| 032 | 18.187 | 0 | 910 | seeded_from_verified_log | `compiler_single_conv_qlinear` | memory is 0; params dominate |
| 097 | 18.187 | 0 | 910 | seeded_from_verified_log | `compiler_single_conv_qlinear` | memory is 0; params dominate |
| 120 | 18.187 | 0 | 910 | seeded_from_verified_log | `compiler_single_conv_qlinear` | memory is 0; params dominate |
| 171 | 18.187 | 0 | 910 | seeded_from_tasklog_and_inventory | `compiler_single_conv_qlinear` | memory is 0; params dominate |
| 283 | 18.187 | 0 | 910 | seeded_from_verified_log | `compiler_single_conv_qlinear` | memory is 0; params dominate |
| 294 | 18.187 | 0 | 910 | seeded_from_verified_log | `compiler_single_conv_qlinear` | memory is 0; params dominate |
| 331 | 18.187 | 0 | 910 | seeded_from_verified_log | `compiler_single_conv_qlinear` | memory is 0; params dominate |
| 344 | 18.187 | 0 | 910 | seeded_from_tasklog_and_inventory | `compiler_single_conv_qlinear` | memory is 0; params dominate |
| 015 | 18.198 | 0 | 900 | seeded_from_verified_log | `compiler_single_conv_qlinear` | memory is 0; params dominate |
| 098 | 18.198 | 0 | 900 | operator_evidence_only_needs_human_review | `compiler_single_conv_qlinear` | memory is 0; params dominate |
| 114 | 18.198 | 0 | 900 | operator_evidence_only_needs_human_review | `compiler_single_conv_qlinear` | memory is 0; params dominate |
| 151 | 18.198 | 0 | 900 | operator_evidence_only_needs_human_review | `compiler_single_conv_qlinear` | memory is 0; params dominate |
| 220 | 18.198 | 0 | 900 | seeded_from_verified_log | `compiler_single_conv_qlinear` | memory is 0; params dominate |
| 230 | 18.198 | 0 | 900 | seeded_from_verified_log | `compiler_tiny_lut_gather`, `compiler_single_conv_qlinear` | memory is 0; params dominate |
| 122 | 18.303 | 0 | 810 | seeded_from_verified_log | `compiler_single_conv_qlinear` | memory is 0; params dominate |
| 372 | 18.435 | 0 | 710 | seeded_from_verified_log | `compiler_single_conv_qlinear` | memory is 0; params dominate |
| 352 | 18.869 | 0 | 460 | seeded_from_verified_log | `compiler_single_conv_qlinear` | memory is 0; params dominate |
| 257 | 18.986 | 0 | 409 | seeded_from_verified_log | `compiler_single_conv_qlinear` | memory is 0; params dominate |
| 375 | 19.142 | 108 | 242 | seeded_from_verified_log | `compiler_direct_output_algebra` | memory is 108; params dominate |
| 360 | 19.171 | 0 | 340 | seeded_from_verified_log | `compiler_direct_output_algebra` | memory is 0; params dominate |
| 128 | 19.296 | 0 | 300 | seeded_from_verified_log | `compiler_single_conv_qlinear` | memory is 0; params dominate |
| 306 | 19.296 | 0 | 300 | seeded_from_verified_log | `compiler_direct_output_algebra`, `compiler_tiny_lut_gather` | memory is 0; params dominate |
| 321 | 19.296 | 0 | 300 | seeded_from_verified_log | `compiler_single_conv_qlinear` | memory is 0; params dominate |
| 142 | 19.358 | 144 | 138 | seeded_from_verified_log | `compiler_direct_output_algebra` | memory is 144; params dominate |
| 252 | 19.439 | 0 | 260 | seeded_from_verified_log | `compiler_direct_output_algebra` | memory is 0; params dominate |
| 001 | 19.519 | 0 | 240 | seeded_from_verified_log | `compiler_direct_output_algebra`, `compiler_tiny_lut_gather` | memory is 0; params dominate |
| 176 | 19.519 | 0 | 240 | operator_evidence_only_needs_human_review | `compiler_direct_output_algebra` | memory is 0; params dominate |
| 315 | 19.562 | 0 | 230 | seeded_from_verified_log | `compiler_direct_output_algebra` | memory is 0; params dominate |
| 274 | 19.579 | 146 | 80 | seeded_from_verified_log | `compiler_direct_output_algebra`, `compiler_final_equal_overlay` | memory is 146; params dominate |
| 180 | 19.653 | 0 | 210 | seeded_from_verified_log | `compiler_single_conv_qlinear` | memory is 0; params dominate |
| 322 | 19.702 | 144 | 56 | operator_evidence_only_needs_human_review | `compiler_single_conv_qlinear`, `compiler_final_equal_overlay`, `compiler_bounded_scan` | memory is 144; params dominate |
| 026 | 19.702 | 0 | 200 | operator_evidence_only_needs_human_review | `compiler_single_conv_qlinear` | memory is 0; params dominate |
| 135 | 19.702 | 0 | 200 | operator_evidence_only_needs_human_review | `compiler_single_conv_qlinear` | memory is 0; params dominate |
| 261 | 19.702 | 0 | 200 | seeded_from_verified_log | `compiler_single_conv_qlinear` | memory is 0; params dominate |
| 313 | 19.743 | 0 | 192 | operator_evidence_only_needs_human_review | `compiler_direct_output_algebra` | memory is 0; params dominate |
| 082 | 19.753 | 0 | 190 | seeded_from_verified_log | `compiler_single_conv_qlinear` | memory is 0; params dominate |
| 376 | 19.774 | 144 | 42 | operator_evidence_only_needs_human_review | `compiler_tiny_lut_gather` | memory is 144; params dominate |
| 235 | 19.864 | 111 | 59 | operator_evidence_only_needs_human_review | `compiler_direct_onehot_gather`, `compiler_final_equal_overlay` | memory is 111; params dominate |
| 028 | 19.876 | 0 | 168 | seeded_from_verified_log | `compiler_direct_output_algebra` | memory is 0; params dominate |
| 395 | 19.894 | 144 | 21 | operator_evidence_only_needs_human_review |  | memory is 144; params dominate |
| 258 | 19.925 | 0 | 160 | seeded_from_verified_log | `compiler_single_conv_qlinear` | memory is 0; params dominate |
| 391 | 19.931 | 135 | 24 | seeded_from_verified_log | `compiler_tiny_lut_gather`, `compiler_final_equal_overlay` | memory is 135; params dominate |
| 150 | 19.937 | 128 | 30 | seeded_from_verified_log | `compiler_tiny_lut_gather` | memory is 128; params dominate |
| 155 | 19.937 | 128 | 30 | seeded_from_verified_log | `compiler_tiny_lut_gather` | memory is 128; params dominate |
| 006 | 19.970 | 126 | 27 | seeded_from_verified_log |  | memory is 126; params dominate |
| 248 | 19.983 | 122 | 29 | seeded_from_verified_log | `compiler_sparse_scatter` | memory is 122; params dominate |
| 167 | 20.003 | 20 | 128 | seeded_from_verified_log | `compiler_direct_output_algebra` | memory is 20; params dominate |
| 317 | 20.016 | 128 | 18 | seeded_from_verified_log | `compiler_direct_output_algebra`, `compiler_roi_pool_crop` | memory is 128; params dominate |
| 393 | 20.030 | 125 | 19 | operator_evidence_only_needs_human_review | `compiler_direct_output_algebra`, `compiler_final_equal_overlay` | memory is 125; params dominate |
| 347 | 20.037 | 117 | 26 | operator_evidence_only_needs_human_review | `compiler_direct_output_algebra` | memory is 117; params dominate |
| 389 | 20.058 | 128 | 12 | seeded_from_verified_log | `compiler_direct_output_algebra`, `compiler_tiny_lut_gather`, `compiler_sparse_scatter`, `compiler_bounded_scan` | memory is 128; params dominate |
| 339 | 20.058 | 40 | 100 | seeded_from_verified_log | `compiler_direct_output_algebra` | memory is 40; params dominate |
| 149 | 20.087 | 36 | 100 | seeded_from_verified_log | `compiler_direct_output_algebra`, `compiler_single_conv_qlinear` | memory is 36; params dominate |
| 298 | 20.095 | 120 | 15 | seeded_from_verified_log | `compiler_direct_output_algebra` | memory is 120; params dominate |
| 007 | 20.156 | 0 | 127 | seeded_from_verified_log | `compiler_direct_output_algebra`, `compiler_tiny_lut_gather` | memory is 0; params dominate |
| 024 | 20.300 | 0 | 110 | seeded_from_verified_log | `compiler_direct_output_algebra` | memory is 0; params dominate |
| 262 | 20.337 | 90 | 16 | seeded_from_verified_log | `compiler_direct_output_algebra` | memory is 90; params dominate |
| 144 | 20.356 | 0 | 104 | operator_evidence_only_needs_human_review | `compiler_direct_output_algebra`, `compiler_single_conv_qlinear` | memory is 0; params dominate |
| 186 | 20.365 | 80 | 23 | seeded_from_verified_log | `compiler_direct_output_algebra`, `compiler_bounded_scan` | memory is 80; params dominate |
| 227 | 20.395 | 0 | 100 | seeded_from_verified_log | `compiler_direct_output_algebra`, `compiler_single_conv_qlinear` | memory is 0; params dominate |
| 314 | 20.395 | 0 | 100 | operator_evidence_only_needs_human_review | `compiler_direct_output_algebra`, `compiler_single_conv_qlinear` | memory is 0; params dominate |
| 318 | 20.395 | 0 | 100 | seeded_from_verified_log | `compiler_direct_output_algebra`, `compiler_single_conv_qlinear` | memory is 0; params dominate |
| 380 | 20.405 | 0 | 99 | seeded_from_tasklog_and_inventory | `compiler_direct_output_algebra` | memory is 0; params dominate |
| 399 | 20.489 | 71 | 20 | seeded_from_verified_log | `compiler_direct_output_algebra`, `compiler_tiny_lut_gather`, `compiler_final_equal_overlay` | memory is 71; params dominate |
| 129 | 20.618 | 80 | 0 | seeded_from_verified_log | `compiler_direct_output_algebra` | memory is 80; params dominate |
| 291 | 20.780 | 40 | 28 | seeded_from_verified_log | `compiler_direct_output_algebra` | memory is 40; params dominate |
| 103 | 20.906 | 36 | 24 | operator_evidence_only_needs_human_review | `compiler_direct_output_algebra`, `compiler_final_equal_overlay`, `compiler_bounded_scan` | memory is 36; params dominate |
| 373 | 20.906 | 0 | 60 | seeded_from_verified_log | `compiler_direct_output_algebra` | memory is 0; params dominate |
| 299 | 21.088 | 0 | 50 | operator_evidence_only_needs_human_review | `compiler_direct_output_algebra` | memory is 0; params dominate |
| 067 | 21.150 | 39 | 8 | seeded_from_verified_log | `compiler_direct_output_algebra` | memory is 39; params dominate |
| 073 | 21.311 | 0 | 40 | seeded_from_verified_log | `compiler_direct_output_algebra`, `compiler_single_conv_qlinear` | memory is 0; params dominate |
| 056 | 21.474 | 34 | 0 | seeded_from_verified_log | `compiler_direct_output_algebra`, `compiler_final_equal_overlay` | memory is 34; params dominate |
| 053 | 21.599 | 0 | 30 | seeded_from_verified_log | `compiler_direct_output_algebra`, `compiler_tiny_lut_gather` | memory is 0; params dominate |
| 113 | 21.599 | 0 | 30 | seeded_from_verified_log | `compiler_direct_output_algebra`, `compiler_tiny_lut_gather` | memory is 0; params dominate |
| 116 | 21.599 | 0 | 30 | operator_evidence_only_needs_human_review | `compiler_direct_output_algebra`, `compiler_tiny_lut_gather` | memory is 0; params dominate |
| 130 | 21.599 | 0 | 30 | seeded_from_verified_log | `compiler_direct_output_algebra`, `compiler_single_conv_qlinear` | memory is 0; params dominate |
| 164 | 21.599 | 0 | 30 | operator_evidence_only_needs_human_review | `compiler_direct_output_algebra`, `compiler_tiny_lut_gather` | memory is 0; params dominate |
| 172 | 21.599 | 0 | 30 | operator_evidence_only_needs_human_review | `compiler_direct_output_algebra`, `compiler_tiny_lut_gather` | memory is 0; params dominate |
| 210 | 21.599 | 0 | 30 | operator_evidence_only_needs_human_review | `compiler_direct_output_algebra`, `compiler_tiny_lut_gather` | memory is 0; params dominate |
| 311 | 21.599 | 0 | 30 | operator_evidence_only_needs_human_review | `compiler_direct_output_algebra`, `compiler_tiny_lut_gather` | memory is 0; params dominate |
| 326 | 21.599 | 0 | 30 | seeded_from_verified_log | `compiler_direct_output_algebra` | memory is 0; params dominate |
| 385 | 21.599 | 0 | 30 | operator_evidence_only_needs_human_review | `compiler_direct_output_algebra`, `compiler_tiny_lut_gather` | memory is 0; params dominate |
| 166 | 21.822 | 0 | 24 | seeded_from_verified_log | `compiler_direct_output_algebra` | memory is 0; params dominate |
| 312 | 22.004 | 0 | 20 | seeded_from_verified_log | `compiler_direct_output_algebra` | memory is 0; params dominate |
| 016 | 22.697 | 0 | 10 | operator_evidence_only_needs_human_review | `compiler_direct_output_algebra`, `compiler_tiny_lut_gather` | memory is 0; params dominate |
| 276 | 22.697 | 0 | 10 | operator_evidence_only_needs_human_review | `compiler_direct_output_algebra`, `compiler_tiny_lut_gather` | memory is 0; params dominate |
| 309 | 22.697 | 0 | 10 | operator_evidence_only_needs_human_review | `compiler_direct_output_algebra`, `compiler_tiny_lut_gather` | memory is 0; params dominate |
| 337 | 22.697 | 0 | 10 | operator_evidence_only_needs_human_review | `compiler_direct_output_algebra`, `compiler_tiny_lut_gather` | memory is 0; params dominate |
| 087 | 23.391 | 0 | 5 | operator_evidence_only_needs_human_review | `compiler_direct_output_algebra`, `compiler_roi_pool_crop` | memory is 0; params dominate |
| 140 | 23.391 | 0 | 5 | seeded_from_verified_log | `compiler_direct_output_algebra`, `compiler_tiny_lut_gather`, `compiler_roi_pool_crop` | memory is 0; params dominate |
| 223 | 23.391 | 0 | 5 | operator_evidence_only_needs_human_review | `compiler_direct_output_algebra`, `compiler_roi_pool_crop` | memory is 0; params dominate |
| 307 | 23.391 | 0 | 5 | operator_evidence_only_needs_human_review | `compiler_direct_output_algebra`, `compiler_roi_pool_crop` | memory is 0; params dominate |
| 179 | 25.000 | 0 | 0 | operator_evidence_only_needs_human_review | `compiler_direct_output_algebra` | memory is 0; params dominate |
| 241 | 25.000 | 0 | 0 | operator_evidence_only_needs_human_review | `compiler_direct_output_algebra` | memory is 0; params dominate |

## Known Best Routes

- Pending human review.

## Kill Criteria

- Pending human review.

## Successful Applications

- Pending verification.

## Failed Applications / Walls

- Pending verification.
