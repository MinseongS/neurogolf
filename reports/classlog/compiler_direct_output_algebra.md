# compiler_direct_output_algebra — Direct output algebra

Emit thresholded final output without full intermediate carriers.

## Optimization Question

What is the cheapest verified ONNX family for tasks in this class, and
which semantic preconditions let us avoid full-canvas intermediates?

## Candidate Tasks

| task | pts | mem | params | status | first routes | evidence |
|---:|---:|---:|---:|---|---|---|
| 118 | 14.559 | 30849 | 3387 | seeded_from_verified_log | `compiler_direct_output_algebra`, `compiler_direct_onehot_gather`, `compiler_tiny_lut_gather`, `compiler_single_conv_qlinear` | Einsum can be direct threshold algebra candidate |
| 319 | 15.119 | 19280 | 279 | seeded_from_verified_log | `compiler_direct_output_algebra`, `compiler_tiny_lut_gather`, `compiler_final_equal_overlay`, `compiler_bounded_scan` | Einsum can be direct threshold algebra candidate |
| 066 | 15.256 | 16899 | 160 | seeded_from_verified_log | `compiler_direct_output_algebra`, `compiler_final_equal_overlay`, `compiler_bounded_scan` | Einsum can be direct threshold algebra candidate |
| 255 | 15.387 | 14663 | 293 | seeded_from_verified_log | `compiler_direct_output_algebra`, `compiler_direct_onehot_gather`, `compiler_single_conv_qlinear`, `compiler_final_equal_overlay` | Einsum can be direct threshold algebra candidate |
| 025 | 15.448 | 13874 | 195 | seeded_from_verified_log | `compiler_direct_output_algebra`, `compiler_final_equal_overlay`, `compiler_bounded_scan` | Einsum can be direct threshold algebra candidate |
| 029 | 15.715 | 10736 | 34 | seeded_from_verified_log | `compiler_direct_output_algebra`, `compiler_tiny_lut_gather`, `compiler_final_equal_overlay`, `compiler_bounded_scan` | Einsum can be direct threshold algebra candidate |
| 370 | 15.823 | 8645 | 1024 | seeded_from_verified_log | `compiler_direct_output_algebra`, `compiler_tiny_lut_gather`, `compiler_single_conv_qlinear`, `compiler_final_equal_overlay` | Einsum can be direct threshold algebra candidate |
| 092 | 16.148 | 6805 | 182 | seeded_from_verified_log | `compiler_direct_output_algebra`, `compiler_final_equal_overlay`, `compiler_sparse_scatter`, `compiler_bounded_scan` | Einsum can be direct threshold algebra candidate |
| 264 | 16.292 | 5348 | 706 | seeded_from_verified_log | `compiler_direct_output_algebra`, `compiler_tiny_lut_gather`, `compiler_single_conv_qlinear`, `compiler_final_equal_overlay` | Einsum can be direct threshold algebra candidate |
| 280 | 16.294 | 5286 | 753 | seeded_from_verified_log | `compiler_direct_output_algebra`, `compiler_final_equal_overlay` | Einsum can be direct threshold algebra candidate |
| 382 | 16.337 | 5648 | 135 | seeded_from_verified_log | `compiler_direct_output_algebra`, `compiler_tiny_lut_gather`, `compiler_final_equal_overlay`, `compiler_bounded_scan` | Einsum can be direct threshold algebra candidate |
| 234 | 16.448 | 5113 | 66 | seeded_from_verified_log | `compiler_direct_output_algebra`, `compiler_tiny_lut_gather`, `compiler_final_equal_overlay`, `compiler_bounded_scan` | Einsum can be direct threshold algebra candidate |
| 208 | 16.539 | 4612 | 114 | seeded_from_verified_log | `compiler_direct_output_algebra`, `compiler_single_conv_qlinear`, `compiler_final_equal_overlay`, `compiler_qlinear_uint8` | Einsum can be direct threshold algebra candidate |
| 358 | 16.587 | 4332 | 174 | seeded_from_verified_log | `compiler_direct_output_algebra`, `compiler_final_equal_overlay`, `compiler_bounded_scan` | Einsum can be direct threshold algebra candidate |
| 378 | 16.600 | 4332 | 117 | seeded_from_verified_log | `compiler_direct_output_algebra`, `compiler_tiny_lut_gather`, `compiler_final_equal_overlay`, `compiler_bounded_scan` | Einsum can be direct threshold algebra candidate |
| 365 | 16.699 | 3908 | 120 | seeded_from_verified_log | `compiler_direct_output_algebra`, `compiler_tiny_lut_gather`, `compiler_single_conv_qlinear`, `compiler_final_equal_overlay` | Einsum can be direct threshold algebra candidate |
| 137 | 16.832 | 3433 | 93 | seeded_from_verified_log | `compiler_direct_output_algebra`, `compiler_final_equal_overlay` | Einsum can be direct threshold algebra candidate |
| 310 | 16.902 | 3208 | 79 | seeded_from_verified_log | `compiler_direct_output_algebra`, `compiler_tiny_lut_gather`, `compiler_final_equal_overlay` | Einsum can be direct threshold algebra candidate |
| 333 | 16.921 | 2792 | 435 | seeded_from_verified_log | `compiler_direct_output_algebra`, `compiler_tiny_lut_gather`, `compiler_final_equal_overlay`, `compiler_bounded_scan` | Einsum can be direct threshold algebra candidate |
| 270 | 16.971 | 2742 | 327 | seeded_from_verified_log | `compiler_direct_output_algebra`, `compiler_bounded_scan` | Einsum can be direct threshold algebra candidate |
| 335 | 17.001 | 2288 | 691 | seeded_from_verified_log | `compiler_direct_output_algebra`, `compiler_single_conv_qlinear`, `compiler_final_equal_overlay`, `compiler_bounded_scan` | Einsum can be direct threshold algebra candidate |
| 197 | 17.136 | 2586 | 16 | seeded_from_tasklog_and_inventory | `compiler_direct_output_algebra`, `compiler_tiny_lut_gather`, `compiler_single_conv_qlinear`, `compiler_qlinear_uint8` | Einsum can be direct threshold algebra candidate |
| 184 | 17.168 | 2460 | 60 | seeded_from_verified_log | `compiler_direct_output_algebra`, `compiler_single_conv_qlinear`, `compiler_final_equal_overlay`, `compiler_bounded_scan` | Einsum can be direct threshold algebra candidate |
| 238 | 17.178 | 2051 | 444 | seeded_from_verified_log | `compiler_direct_output_algebra`, `compiler_tiny_lut_gather`, `compiler_final_equal_overlay`, `compiler_bounded_scan` | Einsum can be direct threshold algebra candidate |
| 281 | 17.223 | 2321 | 65 | seeded_from_verified_log | `compiler_direct_output_algebra`, `compiler_final_equal_overlay`, `compiler_bounded_scan` | Einsum can be direct threshold algebra candidate |
| 022 | 17.258 | 2004 | 298 | seeded_from_verified_log | `compiler_direct_output_algebra`, `compiler_final_equal_overlay`, `compiler_roi_pool_crop` | Einsum can be direct threshold algebra candidate |
| 275 | 17.394 | 1374 | 637 | seeded_from_verified_log | `compiler_direct_output_algebra`, `compiler_single_conv_qlinear`, `compiler_final_equal_overlay`, `compiler_bounded_scan` | Einsum can be direct threshold algebra candidate |
| 287 | 17.402 | 1699 | 295 | seeded_from_verified_log | `compiler_direct_output_algebra`, `compiler_tiny_lut_gather`, `compiler_final_equal_overlay` | Einsum can be direct threshold algebra candidate |
| 392 | 17.424 | 1602 | 349 | seeded_from_verified_log | `compiler_direct_output_algebra`, `compiler_tiny_lut_gather`, `compiler_final_equal_overlay`, `compiler_bounded_scan` | Einsum can be direct threshold algebra candidate |
| 013 | 17.435 | 1869 | 61 | seeded_from_verified_log | `compiler_direct_output_algebra`, `compiler_tiny_lut_gather`, `compiler_final_equal_overlay` | Einsum can be direct threshold algebra candidate |
| 213 | 17.457 | 1838 | 49 | seeded_from_verified_log | `compiler_direct_output_algebra`, `compiler_final_equal_overlay` | Einsum can be direct threshold algebra candidate |
| 088 | 17.465 | 1791 | 81 | seeded_from_verified_log | `compiler_direct_output_algebra`, `compiler_final_equal_overlay` | Einsum can be direct threshold algebra candidate |
| 199 | 17.484 | 1749 | 88 | seeded_from_verified_log | `compiler_direct_output_algebra`, `compiler_final_equal_overlay`, `compiler_sparse_scatter` | Einsum can be direct threshold algebra candidate |
| 051 | 17.503 | 1702 | 101 | seeded_from_verified_log | `compiler_direct_output_algebra`, `compiler_final_equal_overlay` | Einsum can be direct threshold algebra candidate |
| 112 | 17.519 | 1616 | 158 | seeded_from_verified_log | `compiler_direct_output_algebra`, `compiler_tiny_lut_gather` | Einsum can be direct threshold algebra candidate |
| 156 | 17.536 | 1697 | 47 | seeded_from_tasklog_and_inventory | `compiler_direct_output_algebra`, `compiler_tiny_lut_gather`, `compiler_final_equal_overlay` | Einsum can be direct threshold algebra candidate |
| 045 | 17.578 | 1050 | 623 | seeded_from_verified_log | `compiler_direct_output_algebra`, `compiler_bounded_scan` | Einsum can be direct threshold algebra candidate |
| 303 | 17.610 | 1598 | 22 | seeded_from_verified_log | `compiler_direct_output_algebra`, `compiler_final_equal_overlay` | Einsum can be direct threshold algebra candidate |
| 342 | 17.632 | 1510 | 75 | seeded_from_verified_log | `compiler_direct_output_algebra`, `compiler_tiny_lut_gather`, `compiler_final_equal_overlay`, `compiler_sparse_scatter` | Einsum can be direct threshold algebra candidate |
| 159 | 17.642 | 1419 | 149 | seeded_from_verified_log | `compiler_direct_output_algebra`, `compiler_tiny_lut_gather`, `compiler_final_equal_overlay` | Einsum can be direct threshold algebra candidate |
| 141 | 17.683 | 1404 | 101 | seeded_from_verified_log | `compiler_direct_output_algebra`, `compiler_final_equal_overlay` | Einsum can be direct threshold algebra candidate |
| 308 | 17.695 | 1385 | 103 | seeded_from_verified_log | `compiler_direct_output_algebra`, `compiler_final_equal_overlay`, `compiler_sparse_scatter`, `compiler_bounded_scan` | Einsum can be direct threshold algebra candidate |
| 075 | 17.695 | 1326 | 161 | seeded_from_verified_log | `compiler_direct_output_algebra`, `compiler_final_equal_overlay` | Einsum can be direct threshold algebra candidate |
| 123 | 17.797 | 403 | 940 | seeded_from_verified_log | `compiler_direct_output_algebra`, `compiler_tiny_lut_gather`, `compiler_final_equal_overlay` | Einsum can be direct threshold algebra candidate |
| 183 | 17.885 | 1008 | 222 | seeded_from_verified_log | `compiler_direct_output_algebra`, `compiler_final_equal_overlay` | Einsum can be direct threshold algebra candidate |
| 021 | 17.935 | 1072 | 98 | seeded_from_tasklog_and_inventory | `compiler_direct_output_algebra` | Einsum can be direct threshold algebra candidate |
| 115 | 18.050 | 980 | 63 | seeded_from_verified_log | `compiler_direct_output_algebra`, `compiler_bounded_scan` | Einsum can be direct threshold algebra candidate |
| 228 | 18.131 | 849 | 113 | seeded_from_verified_log | `compiler_direct_output_algebra`, `compiler_tiny_lut_gather`, `compiler_final_equal_overlay`, `compiler_sparse_scatter` | Einsum can be direct threshold algebra candidate |
| 143 | 18.269 | 591 | 247 | seeded_from_verified_log | `compiler_direct_output_algebra`, `compiler_final_equal_overlay` | Einsum can be direct threshold algebra candidate |
| 221 | 18.309 | 451 | 354 | seeded_from_tasklog_and_inventory | `compiler_direct_output_algebra`, `compiler_final_equal_overlay`, `compiler_bounded_scan` | Einsum can be direct threshold algebra candidate |
| 153 | 18.322 | 741 | 54 | seeded_from_verified_log | `compiler_direct_output_algebra`, `compiler_tiny_lut_gather`, `compiler_final_equal_overlay`, `compiler_bounded_scan` | Einsum can be direct threshold algebra candidate |
| 043 | 18.323 | 588 | 206 | seeded_from_verified_log | `compiler_direct_output_algebra` | Einsum can be direct threshold algebra candidate |
| 200 | 18.372 | 570 | 186 | seeded_from_verified_log | `compiler_direct_output_algebra`, `compiler_single_conv_qlinear`, `compiler_final_equal_overlay`, `compiler_bounded_scan` | Einsum can be direct threshold algebra candidate |
| 289 | 18.391 | 618 | 124 | seeded_from_verified_log | `compiler_direct_output_algebra`, `compiler_final_equal_overlay`, `compiler_bounded_scan` | Einsum can be direct threshold algebra candidate |
| 065 | 18.537 | 582 | 59 | seeded_from_verified_log | `compiler_direct_output_algebra`, `compiler_final_equal_overlay`, `compiler_bounded_scan` | Einsum can be direct threshold algebra candidate |
| 068 | 18.565 | 500 | 123 | seeded_from_verified_log | `compiler_direct_output_algebra`, `compiler_final_equal_overlay`, `compiler_sparse_scatter` | Einsum can be direct threshold algebra candidate |
| 126 | 18.595 | 590 | 15 | operator_evidence_only_needs_human_review | `compiler_direct_output_algebra`, `compiler_final_equal_overlay`, `compiler_sparse_scatter`, `compiler_bounded_scan` | Einsum can be direct threshold algebra candidate |
| 253 | 18.616 | 540 | 52 | seeded_from_verified_log | `compiler_direct_output_algebra`, `compiler_final_equal_overlay` | Einsum can be direct threshold algebra candidate |
| 059 | 18.677 | 265 | 292 | seeded_from_verified_log | `compiler_direct_output_algebra`, `compiler_final_equal_overlay`, `compiler_bounded_scan` | Einsum can be direct threshold algebra candidate |
| 100 | 18.703 | 454 | 89 | seeded_from_verified_log | `compiler_direct_output_algebra`, `compiler_final_equal_overlay` | Einsum can be direct threshold algebra candidate |
| 300 | 18.708 | 474 | 66 | seeded_from_verified_log | `compiler_direct_output_algebra`, `compiler_tiny_lut_gather`, `compiler_final_equal_overlay` | Einsum can be direct threshold algebra candidate |
| 362 | 18.744 | 408 | 113 | operator_evidence_only_needs_human_review | `compiler_direct_output_algebra`, `compiler_tiny_lut_gather`, `compiler_final_equal_overlay`, `compiler_sparse_scatter` | Einsum can be direct threshold algebra candidate |
| 247 | 18.847 | 408 | 62 | seeded_from_tasklog_and_inventory | `compiler_direct_output_algebra`, `compiler_final_equal_overlay`, `compiler_bounded_scan` | Einsum can be direct threshold algebra candidate |
| 371 | 18.854 | 376 | 91 | seeded_from_verified_log | `compiler_direct_output_algebra`, `compiler_sparse_scatter` | Einsum can be direct threshold algebra candidate |
| 242 | 18.932 | 372 | 60 | seeded_from_verified_log | `compiler_direct_output_algebra` | Einsum can be direct threshold algebra candidate |
| 316 | 18.946 | 388 | 38 | seeded_from_verified_log | `compiler_direct_output_algebra`, `compiler_tiny_lut_gather`, `compiler_final_equal_overlay` | Einsum can be direct threshold algebra candidate |
| 211 | 19.019 | 240 | 156 | seeded_from_verified_log | `compiler_direct_output_algebra` | Einsum can be direct threshold algebra candidate |
| 049 | 19.044 | 252 | 134 | seeded_from_verified_log | `compiler_direct_output_algebra`, `compiler_single_conv_qlinear`, `compiler_final_equal_overlay`, `compiler_bounded_scan` | Einsum can be direct threshold algebra candidate |
| 040 | 19.076 | 240 | 134 | seeded_from_verified_log | `compiler_direct_output_algebra`, `compiler_bounded_scan` | Einsum can be direct threshold algebra candidate |
| 290 | 19.128 | 292 | 63 | seeded_from_verified_log | `compiler_direct_output_algebra`, `compiler_tiny_lut_gather`, `compiler_final_equal_overlay` | Einsum can be direct threshold algebra candidate |
| 375 | 19.142 | 108 | 242 | seeded_from_verified_log | `compiler_direct_output_algebra` | Einsum can be direct threshold algebra candidate |
| 121 | 19.168 | 275 | 66 | seeded_from_verified_log | `compiler_direct_output_algebra`, `compiler_tiny_lut_gather` | Einsum can be direct threshold algebra candidate |
| 360 | 19.171 | 0 | 340 | seeded_from_verified_log | `compiler_direct_output_algebra` | Einsum can be direct threshold algebra candidate |
| 353 | 19.189 | 272 | 62 | seeded_from_verified_log | `compiler_direct_output_algebra`, `compiler_sparse_scatter` | Einsum can be direct threshold algebra candidate |
| 306 | 19.296 | 0 | 300 | seeded_from_verified_log | `compiler_direct_output_algebra`, `compiler_tiny_lut_gather` | Einsum can be direct threshold algebra candidate |
| 142 | 19.358 | 144 | 138 | seeded_from_verified_log | `compiler_direct_output_algebra` | Einsum can be direct threshold algebra candidate |
| 252 | 19.439 | 0 | 260 | seeded_from_verified_log | `compiler_direct_output_algebra` | Einsum can be direct threshold algebra candidate |
| 001 | 19.519 | 0 | 240 | seeded_from_verified_log | `compiler_direct_output_algebra`, `compiler_tiny_lut_gather` | Einsum can be direct threshold algebra candidate |
| 176 | 19.519 | 0 | 240 | operator_evidence_only_needs_human_review | `compiler_direct_output_algebra` | Einsum can be direct threshold algebra candidate |
| 315 | 19.562 | 0 | 230 | seeded_from_verified_log | `compiler_direct_output_algebra` | Einsum can be direct threshold algebra candidate |
| 274 | 19.579 | 146 | 80 | seeded_from_verified_log | `compiler_direct_output_algebra`, `compiler_final_equal_overlay` | Einsum can be direct threshold algebra candidate |
| 038 | 19.639 | 163 | 50 | seeded_from_verified_log | `compiler_direct_output_algebra` | Einsum can be direct threshold algebra candidate |
| 313 | 19.743 | 0 | 192 | operator_evidence_only_needs_human_review | `compiler_direct_output_algebra` | Einsum can be direct threshold algebra candidate |
| 028 | 19.876 | 0 | 168 | seeded_from_verified_log | `compiler_direct_output_algebra` | Einsum can be direct threshold algebra candidate |
| 167 | 20.003 | 20 | 128 | seeded_from_verified_log | `compiler_direct_output_algebra` | Einsum can be direct threshold algebra candidate; high-score frontier task (20.003 pts) |
| 317 | 20.016 | 128 | 18 | seeded_from_verified_log | `compiler_direct_output_algebra`, `compiler_roi_pool_crop` | high-score frontier task (20.016 pts) |
| 393 | 20.030 | 125 | 19 | operator_evidence_only_needs_human_review | `compiler_direct_output_algebra`, `compiler_final_equal_overlay` | high-score frontier task (20.030 pts) |
| 347 | 20.037 | 117 | 26 | operator_evidence_only_needs_human_review | `compiler_direct_output_algebra` | high-score frontier task (20.037 pts) |
| 389 | 20.058 | 128 | 12 | seeded_from_verified_log | `compiler_direct_output_algebra`, `compiler_tiny_lut_gather`, `compiler_sparse_scatter`, `compiler_bounded_scan` | high-score frontier task (20.058 pts) |
| 339 | 20.058 | 40 | 100 | seeded_from_verified_log | `compiler_direct_output_algebra` | high-score frontier task (20.058 pts) |
| 149 | 20.087 | 36 | 100 | seeded_from_verified_log | `compiler_direct_output_algebra`, `compiler_single_conv_qlinear` | high-score frontier task (20.087 pts) |
| 298 | 20.095 | 120 | 15 | seeded_from_verified_log | `compiler_direct_output_algebra` | Einsum can be direct threshold algebra candidate; high-score frontier task (20.095 pts) |
| 007 | 20.156 | 0 | 127 | seeded_from_verified_log | `compiler_direct_output_algebra`, `compiler_tiny_lut_gather` | Einsum can be direct threshold algebra candidate; high-score frontier task (20.156 pts) |
| 024 | 20.300 | 0 | 110 | seeded_from_verified_log | `compiler_direct_output_algebra` | Einsum can be direct threshold algebra candidate; high-score frontier task (20.300 pts) |
| 262 | 20.337 | 90 | 16 | seeded_from_verified_log | `compiler_direct_output_algebra` | high-score frontier task (20.337 pts) |
| 144 | 20.356 | 0 | 104 | operator_evidence_only_needs_human_review | `compiler_direct_output_algebra`, `compiler_single_conv_qlinear` | high-score frontier task (20.356 pts) |
| 186 | 20.365 | 80 | 23 | seeded_from_verified_log | `compiler_direct_output_algebra`, `compiler_bounded_scan` | high-score frontier task (20.365 pts) |
| 227 | 20.395 | 0 | 100 | seeded_from_verified_log | `compiler_direct_output_algebra`, `compiler_single_conv_qlinear` | high-score frontier task (20.395 pts) |
| 314 | 20.395 | 0 | 100 | operator_evidence_only_needs_human_review | `compiler_direct_output_algebra`, `compiler_single_conv_qlinear` | high-score frontier task (20.395 pts) |
| 318 | 20.395 | 0 | 100 | seeded_from_verified_log | `compiler_direct_output_algebra`, `compiler_single_conv_qlinear` | high-score frontier task (20.395 pts) |
| 380 | 20.405 | 0 | 99 | seeded_from_tasklog_and_inventory | `compiler_direct_output_algebra` | Einsum can be direct threshold algebra candidate; high-score frontier task (20.405 pts) |
| 399 | 20.489 | 71 | 20 | seeded_from_verified_log | `compiler_direct_output_algebra`, `compiler_tiny_lut_gather`, `compiler_final_equal_overlay` | high-score frontier task (20.489 pts) |
| 129 | 20.618 | 80 | 0 | seeded_from_verified_log | `compiler_direct_output_algebra` | Einsum can be direct threshold algebra candidate; high-score frontier task (20.618 pts) |
| 291 | 20.780 | 40 | 28 | seeded_from_verified_log | `compiler_direct_output_algebra` | Einsum can be direct threshold algebra candidate; high-score frontier task (20.780 pts) |
| 103 | 20.906 | 36 | 24 | operator_evidence_only_needs_human_review | `compiler_direct_output_algebra`, `compiler_final_equal_overlay`, `compiler_bounded_scan` | high-score frontier task (20.906 pts) |
| 373 | 20.906 | 0 | 60 | seeded_from_verified_log | `compiler_direct_output_algebra` | Einsum can be direct threshold algebra candidate; high-score frontier task (20.906 pts) |
| 299 | 21.088 | 0 | 50 | operator_evidence_only_needs_human_review | `compiler_direct_output_algebra` | Einsum can be direct threshold algebra candidate; high-score frontier task (21.088 pts) |
| 067 | 21.150 | 39 | 8 | seeded_from_verified_log | `compiler_direct_output_algebra` | high-score frontier task (21.150 pts) |
| 073 | 21.311 | 0 | 40 | seeded_from_verified_log | `compiler_direct_output_algebra`, `compiler_single_conv_qlinear` | high-score frontier task (21.311 pts) |
| 056 | 21.474 | 34 | 0 | seeded_from_verified_log | `compiler_direct_output_algebra`, `compiler_final_equal_overlay` | high-score frontier task (21.474 pts) |
| 053 | 21.599 | 0 | 30 | seeded_from_verified_log | `compiler_direct_output_algebra`, `compiler_tiny_lut_gather` | high-score frontier task (21.599 pts) |
| 113 | 21.599 | 0 | 30 | seeded_from_verified_log | `compiler_direct_output_algebra`, `compiler_tiny_lut_gather` | high-score frontier task (21.599 pts) |
| 116 | 21.599 | 0 | 30 | operator_evidence_only_needs_human_review | `compiler_direct_output_algebra`, `compiler_tiny_lut_gather` | high-score frontier task (21.599 pts) |
| 130 | 21.599 | 0 | 30 | seeded_from_verified_log | `compiler_direct_output_algebra`, `compiler_single_conv_qlinear` | high-score frontier task (21.599 pts) |
| 164 | 21.599 | 0 | 30 | operator_evidence_only_needs_human_review | `compiler_direct_output_algebra`, `compiler_tiny_lut_gather` | high-score frontier task (21.599 pts) |
| 172 | 21.599 | 0 | 30 | operator_evidence_only_needs_human_review | `compiler_direct_output_algebra`, `compiler_tiny_lut_gather` | high-score frontier task (21.599 pts) |
| 210 | 21.599 | 0 | 30 | operator_evidence_only_needs_human_review | `compiler_direct_output_algebra`, `compiler_tiny_lut_gather` | high-score frontier task (21.599 pts) |
| 311 | 21.599 | 0 | 30 | operator_evidence_only_needs_human_review | `compiler_direct_output_algebra`, `compiler_tiny_lut_gather` | high-score frontier task (21.599 pts) |
| 326 | 21.599 | 0 | 30 | seeded_from_verified_log | `compiler_direct_output_algebra` | Einsum can be direct threshold algebra candidate; high-score frontier task (21.599 pts) |
| 385 | 21.599 | 0 | 30 | operator_evidence_only_needs_human_review | `compiler_direct_output_algebra`, `compiler_tiny_lut_gather` | high-score frontier task (21.599 pts) |
| 166 | 21.822 | 0 | 24 | seeded_from_verified_log | `compiler_direct_output_algebra` | Einsum can be direct threshold algebra candidate; high-score frontier task (21.822 pts) |
| 312 | 22.004 | 0 | 20 | seeded_from_verified_log | `compiler_direct_output_algebra` | Einsum can be direct threshold algebra candidate; high-score frontier task (22.004 pts) |
| 016 | 22.697 | 0 | 10 | operator_evidence_only_needs_human_review | `compiler_direct_output_algebra`, `compiler_tiny_lut_gather` | high-score frontier task (22.697 pts) |
| 276 | 22.697 | 0 | 10 | operator_evidence_only_needs_human_review | `compiler_direct_output_algebra`, `compiler_tiny_lut_gather` | high-score frontier task (22.697 pts) |
| 309 | 22.697 | 0 | 10 | operator_evidence_only_needs_human_review | `compiler_direct_output_algebra`, `compiler_tiny_lut_gather` | high-score frontier task (22.697 pts) |
| 337 | 22.697 | 0 | 10 | operator_evidence_only_needs_human_review | `compiler_direct_output_algebra`, `compiler_tiny_lut_gather` | high-score frontier task (22.697 pts) |
| 087 | 23.391 | 0 | 5 | operator_evidence_only_needs_human_review | `compiler_direct_output_algebra`, `compiler_roi_pool_crop` | high-score frontier task (23.391 pts) |
| 140 | 23.391 | 0 | 5 | seeded_from_verified_log | `compiler_direct_output_algebra`, `compiler_tiny_lut_gather`, `compiler_roi_pool_crop` | high-score frontier task (23.391 pts) |
| 223 | 23.391 | 0 | 5 | operator_evidence_only_needs_human_review | `compiler_direct_output_algebra`, `compiler_roi_pool_crop` | high-score frontier task (23.391 pts) |
| 307 | 23.391 | 0 | 5 | operator_evidence_only_needs_human_review | `compiler_direct_output_algebra`, `compiler_roi_pool_crop` | high-score frontier task (23.391 pts) |
| 179 | 25.000 | 0 | 0 | operator_evidence_only_needs_human_review | `compiler_direct_output_algebra` | high-score frontier task (25.000 pts) |
| 241 | 25.000 | 0 | 0 | operator_evidence_only_needs_human_review | `compiler_direct_output_algebra` | high-score frontier task (25.000 pts) |

## Known Best Routes

- **Single-op direct `Einsum` to `output`**: proven on tasks such as
  `001`, `007`, `024`, `166`, `299`, `312`, `326`, `373`, and `380`.
  This is the cleanest form: the only counted cost is dense selector/factor
  initializers; memory can be `0`.
- **Low-rank channel sign factorization**: task001 shows the key trick.  The
  graph need not emit exact one-hot values; it only needs the correct sign under
  the scorer's `output > 0` threshold over reachable generator states.
- **Static or periodic spatial selectors**: successful direct-output graphs
  use fixed selectors such as `[3,30]`, `[2,10]`, `[30]`, or small periodic
  bases.  The spatial relation is known before reading the input, or depends on
  tiny fixed positions that can be selected algebraically.
- **Small symbolic products**: direct algebra works when the rule is a low-degree
  product/sum over a few input cells, colour factors, row/column selectors, or
  fixed-period bases.

## Kill Criteria

- Reject this class when the rule needs a dynamic full-canvas predicate such as
  `x >= left(input) and x <= right(input)` unless that predicate can be folded
  into one final output expression without materializing the `[30,30]` mask.
- Reject when output priority is an overlay rule (`A unless B`, vertical wins,
  later object overwrites earlier object) and the blocker cannot be expressed as
  a low-degree sign separator.  Otherwise the graph usually needs a mask/Where or
  scalar label carrier.
- Reject when the rule requires connected components, flood-fill, assignment
  matching, or dynamic sparse coordinates.  These can still be optimized, but
  they are not direct-output algebra unless a separate breakthrough removes the
  dynamic state.
- Reject if the proposed direct graph merely moves work into full-size
  intermediate `Greater/Where/Mul/Add` tensors.  The class only counts when the
  final `[1,10,30,30]` output is the first full-canvas tensor.
- Be suspicious of candidates added only because `Einsum` appears in the current
  graph.  Many low-score graphs use `Einsum` for a sub-step while still relying
  on scan/scatter/label carriers.

## Successful Applications

- `task001`: replaced materialized 9x9/30x30 carriers with one direct `Einsum`
  and reachable-state channel sign factorization.  Current source is
  `mem=0, params=240`.
- `task146`: latent direct-output task that was not initially tagged by the
  class map because it uses `Conv -> dynamic Slice -> Pad`, not `Einsum`.  The
  label-map carrier had already been removed; a class probe still found a small
  win by skipping an unused block predicate.  The exactly-one-asymmetric-block
  promise means block1/block2 symmetry alone determines the selected block, so
  the checksum Conv can crop away block0.  Adopted `mem=388 -> 380`, params
  unchanged, `18.795442 -> 18.811736`, fresh `1500/1500`.
- `task007`, `task024`, `task166`, `task299`, `task312`, `task326`,
  `task373`, `task380`: current graphs are direct `Einsum` families with
  `mem=0` and small params.  They are useful archetypes for static selectors,
  low-rank symbolic products, and periodic routing.

## Failed Applications / Walls

- `task092` probe, 2026-07-01: visible rule is simple axis-aligned endpoint
  span fill, but vertical strokes overwrite horizontal strokes at crossings.
  A direct polynomial can express one horizontal or vertical interval from
  endpoint moments, but suppressing horizontal channels wherever any vertical
  interval covers the cell requires either a materialized full-canvas vertical
  mask or a high-degree sign product over vertical candidates.  The current best
  graph therefore remains a scalar label/scatter route (`mem=6805, params=182`).
  This is a strong negative boundary for the class: dynamic interval fill with
  overwrite priority is not automatically a direct-output algebra win.
- `task264` probe, 2026-07-01: semantic rule is verified fixed 9x9 glyph-chart
  stamping after detecting eight 3x3 glyph sprites and reading their colours.
  The final `eq9 [1,10,9,9]` carrier (810B) looks like a direct-output target,
  but replacing it with direct 30x30 output would require either a dense fixed
  slot-to-output selector or a 30x30 label grid, which is not cheaper than the
  current 9x9 bool carrier.  The true dominant cost is `label16 [1,1,16,16]`
  plus gray/hash detection, so this task belongs more to "delete dynamic colour
  read carrier" than to pure direct-output algebra.

## Current Class Assessment

This class is viable and high leverage, but it is narrower than "tasks with an
`Einsum` node."  It also includes direct one-hot/crop/Pad graphs like `task146`
when the model routes the free input directly to the free output without a
scalar label carrier.  The best transfer targets are tasks whose spatial mapping
is static/periodic or whose dynamic choice is scalar enough to become a low-rank
sign factor or a tiny dynamic slice.  The worst transfer targets are dynamic
spans, flood-fill, assignment, and overwrite-priority tasks, because they usually
recreate the same full-canvas carrier in another form.

Next best human-review probe should avoid overwrite priority and dynamic colour
read/detection carriers.  Prefer a task whose current graph already has the
needed scalar features and only materializes the final mask/label plane.  Good
candidate shape: one small symbolic rule, no `TopK`, no `Scatter`, no
connectivity/assignment tag, and a current cost near 500..2000 where deleting one
full or cropped output carrier could matter.  Also include latent direct-output
graphs that use `Pad` without final `Equal`; for those, look for unused
predicates/selectors under generator promises such as exactly-one-outlier.
