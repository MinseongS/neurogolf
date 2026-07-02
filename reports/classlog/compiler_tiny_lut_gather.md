# compiler_tiny_lut_gather — Tiny LUT/Gather

Use small lookup tables, Gather, or channel remap.

## Optimization Question

What is the cheapest verified ONNX family for tasks in this class, and
which semantic preconditions let us avoid full-canvas intermediates?

## Candidate Tasks

| task | pts | mem | params | status | first routes | evidence |
|---:|---:|---:|---:|---|---|---|
| 233 | 14.003 | 59147 | 565 | seeded_from_verified_log | `compiler_direct_onehot_gather`, `compiler_tiny_lut_gather`, `compiler_single_conv_qlinear`, `compiler_final_equal_overlay` | Gather/LUT operator evidence |
| 018 | 14.157 | 48196 | 2987 | seeded_from_verified_log | `compiler_tiny_lut_gather`, `compiler_single_conv_qlinear`, `compiler_final_equal_overlay`, `compiler_sparse_scatter` | Gather/LUT operator evidence |
| 286 | 14.341 | 41822 | 742 | seeded_from_verified_log | `compiler_tiny_lut_gather`, `compiler_single_conv_qlinear`, `compiler_final_equal_overlay`, `compiler_qlinear_uint8` | Gather/LUT operator evidence |
| 366 | 14.497 | 35927 | 490 | seeded_from_verified_log | `compiler_tiny_lut_gather`, `compiler_single_conv_qlinear`, `compiler_final_equal_overlay`, `compiler_sparse_scatter` | Gather/LUT operator evidence |
| 158 | 14.528 | 32979 | 2340 | seeded_from_verified_log | `compiler_tiny_lut_gather`, `compiler_single_conv_qlinear`, `compiler_final_equal_overlay`, `compiler_qlinear_uint8` | Gather/LUT operator evidence |
| 118 | 14.559 | 30849 | 3387 | seeded_from_verified_log | `compiler_direct_output_algebra`, `compiler_direct_onehot_gather`, `compiler_tiny_lut_gather`, `compiler_single_conv_qlinear` | Gather/LUT operator evidence |
| 209 | 14.620 | 32027 | 185 | seeded_from_verified_log | `compiler_tiny_lut_gather`, `compiler_single_conv_qlinear`, `compiler_final_equal_overlay`, `compiler_qlinear_uint8` | Gather/LUT operator evidence |
| 191 | 14.624 | 31252 | 841 | seeded_from_verified_log | `compiler_direct_onehot_gather`, `compiler_tiny_lut_gather`, `compiler_single_conv_qlinear`, `compiler_final_equal_overlay` | Gather/LUT operator evidence |
| 054 | 14.829 | 25885 | 238 | seeded_from_tasklog_and_inventory | `compiler_tiny_lut_gather`, `compiler_single_conv_qlinear`, `compiler_final_equal_overlay`, `compiler_sparse_scatter` | Gather/LUT operator evidence |
| 285 | 14.848 | 25080 | 550 | seeded_from_verified_log | `compiler_tiny_lut_gather`, `compiler_single_conv_qlinear`, `compiler_final_equal_overlay`, `compiler_sparse_scatter` | Gather/LUT operator evidence |
| 002 | 14.896 | 24320 | 125 | seeded_from_verified_log | `compiler_direct_onehot_gather`, `compiler_tiny_lut_gather` | Gather/LUT operator evidence |
| 076 | 14.932 | 23296 | 292 | seeded_from_verified_log | `compiler_tiny_lut_gather`, `compiler_final_equal_overlay`, `compiler_sparse_scatter`, `compiler_bounded_scan` | Gather/LUT operator evidence |
| 364 | 14.956 | 22900 | 113 | seeded_from_verified_log | `compiler_tiny_lut_gather`, `compiler_single_conv_qlinear`, `compiler_final_equal_overlay`, `compiler_qlinear_uint8` | Gather/LUT operator evidence |
| 173 | 15.022 | 21430 | 112 | seeded_from_verified_log | `compiler_tiny_lut_gather`, `compiler_single_conv_qlinear`, `compiler_final_equal_overlay`, `compiler_sparse_scatter` | Gather/LUT operator evidence |
| 367 | 15.051 | 16200 | 4733 | seeded_from_verified_log | `compiler_tiny_lut_gather`, `compiler_single_conv_qlinear`, `compiler_qlinear_uint8` | Gather/LUT operator evidence |
| 349 | 15.102 | 19800 | 90 | seeded_from_verified_log | `compiler_tiny_lut_gather`, `compiler_single_conv_qlinear`, `compiler_final_equal_overlay`, `compiler_qlinear_uint8` | Gather/LUT operator evidence |
| 319 | 15.119 | 19280 | 279 | seeded_from_verified_log | `compiler_direct_output_algebra`, `compiler_tiny_lut_gather`, `compiler_final_equal_overlay`, `compiler_bounded_scan` | Gather/LUT operator evidence |
| 219 | 15.195 | 18033 | 93 | seeded_from_verified_log | `compiler_tiny_lut_gather`, `compiler_final_equal_overlay`, `compiler_bounded_scan` | Gather/LUT operator evidence |
| 101 | 15.216 | 16850 | 905 | seeded_from_verified_log | `compiler_tiny_lut_gather`, `compiler_final_equal_overlay`, `compiler_sparse_scatter`, `compiler_bounded_scan` | Gather/LUT operator evidence |
| 193 | 15.361 | 15284 | 68 | seeded_from_verified_log | `compiler_tiny_lut_gather`, `compiler_single_conv_qlinear`, `compiler_final_equal_overlay` | Gather/LUT operator evidence |
| 138 | 15.456 | 13789 | 172 | seeded_from_verified_log | `compiler_tiny_lut_gather`, `compiler_single_conv_qlinear`, `compiler_final_equal_overlay`, `compiler_bounded_scan` | Gather/LUT operator evidence |
| 110 | 15.468 | 13155 | 634 | seeded_from_verified_log | `compiler_tiny_lut_gather`, `compiler_single_conv_qlinear`, `compiler_final_equal_overlay`, `compiler_bounded_scan` | Gather/LUT operator evidence |
| 198 | 15.484 | 13434 | 138 | seeded_from_verified_log | `compiler_tiny_lut_gather`, `compiler_single_conv_qlinear`, `compiler_final_equal_overlay`, `compiler_bounded_scan` | Gather/LUT operator evidence |
| 064 | 15.494 | 13368 | 68 | seeded_from_verified_log | `compiler_tiny_lut_gather`, `compiler_final_equal_overlay`, `compiler_bounded_scan` | Gather/LUT operator evidence |
| 145 | 15.511 | 13104 | 111 | seeded_from_verified_log | `compiler_tiny_lut_gather`, `compiler_final_equal_overlay`, `compiler_bounded_scan` | Gather/LUT operator evidence |
| 044 | 15.651 | 11420 | 68 | seeded_from_verified_log | `compiler_direct_onehot_gather`, `compiler_tiny_lut_gather`, `compiler_single_conv_qlinear`, `compiler_final_equal_overlay` | Gather/LUT operator evidence |
| 338 | 15.712 | 10140 | 666 | seeded_from_verified_log | `compiler_tiny_lut_gather`, `compiler_single_conv_qlinear`, `compiler_final_equal_overlay`, `compiler_qlinear_uint8` | Gather/LUT operator evidence |
| 029 | 15.715 | 10736 | 34 | seeded_from_verified_log | `compiler_direct_output_algebra`, `compiler_tiny_lut_gather`, `compiler_final_equal_overlay`, `compiler_bounded_scan` | Gather/LUT operator evidence |
| 080 | 15.726 | 10198 | 454 | seeded_from_verified_log | `compiler_tiny_lut_gather`, `compiler_single_conv_qlinear`, `compiler_final_equal_overlay`, `compiler_qlinear_uint8` | Gather/LUT operator evidence |
| 017 | 15.744 | 8448 | 2021 | seeded_from_verified_log | `compiler_tiny_lut_gather`, `compiler_final_equal_overlay` | Gather/LUT operator evidence |
| 370 | 15.823 | 8645 | 1024 | seeded_from_verified_log | `compiler_direct_output_algebra`, `compiler_tiny_lut_gather`, `compiler_single_conv_qlinear`, `compiler_final_equal_overlay` | Gather/LUT operator evidence |
| 216 | 15.880 | 9048 | 87 | seeded_from_verified_log | `compiler_direct_onehot_gather`, `compiler_tiny_lut_gather`, `compiler_single_conv_qlinear`, `compiler_final_equal_overlay` | Gather/LUT operator evidence |
| 074 | 15.889 | 9000 | 50 | seeded_from_verified_log | `compiler_tiny_lut_gather`, `compiler_single_conv_qlinear`, `compiler_final_equal_overlay` | Gather/LUT operator evidence |
| 350 | 15.891 | 9012 | 26 | seeded_from_verified_log | `compiler_tiny_lut_gather`, `compiler_bounded_scan` | Gather/LUT operator evidence |
| 096 | 15.905 | 8108 | 805 | seeded_from_verified_log | `compiler_tiny_lut_gather`, `compiler_final_equal_overlay`, `compiler_sparse_scatter`, `compiler_bounded_scan` | Gather/LUT operator evidence |
| 157 | 15.972 | 7983 | 351 | seeded_from_verified_log | `compiler_tiny_lut_gather`, `compiler_final_equal_overlay`, `compiler_sparse_scatter`, `compiler_bounded_scan` | Gather/LUT operator evidence |
| 377 | 15.975 | 8162 | 150 | seeded_from_verified_log | `compiler_tiny_lut_gather`, `compiler_single_conv_qlinear`, `compiler_final_equal_overlay` | Gather/LUT operator evidence |
| 014 | 16.023 | 7809 | 108 | seeded_from_verified_log | `compiler_tiny_lut_gather`, `compiler_final_equal_overlay`, `compiler_bounded_scan` | Gather/LUT operator evidence |
| 089 | 16.111 | 7092 | 160 | seeded_from_verified_log | `compiler_tiny_lut_gather`, `compiler_single_conv_qlinear`, `compiler_final_equal_overlay`, `compiler_sparse_scatter` | Gather/LUT operator evidence |
| 174 | 16.130 | 6973 | 142 | seeded_from_verified_log | `compiler_direct_onehot_gather`, `compiler_tiny_lut_gather`, `compiler_final_equal_overlay`, `compiler_bounded_scan` | Gather/LUT operator evidence |
| 009 | 16.143 | 6929 | 95 | seeded_from_verified_log | `compiler_tiny_lut_gather`, `compiler_single_conv_qlinear`, `compiler_final_equal_overlay`, `compiler_bounded_scan` | Gather/LUT operator evidence |
| 005 | 16.163 | 6392 | 490 | seeded_from_verified_log | `compiler_tiny_lut_gather`, `compiler_single_conv_qlinear`, `compiler_final_equal_overlay`, `compiler_qlinear_uint8` | Gather/LUT operator evidence |
| 182 | 16.176 | 6695 | 98 | seeded_from_verified_log | `compiler_tiny_lut_gather`, `compiler_single_conv_qlinear`, `compiler_final_equal_overlay`, `compiler_qlinear_uint8` | Gather/LUT operator evidence |
| 328 | 16.179 | 4835 | 1940 | seeded_from_tasklog_and_inventory | `compiler_tiny_lut_gather`, `compiler_final_equal_overlay` | Gather/LUT operator evidence |
| 368 | 16.264 | 6022 | 203 | seeded_from_verified_log | `compiler_tiny_lut_gather`, `compiler_final_equal_overlay`, `compiler_sparse_scatter` | Gather/LUT operator evidence |
| 165 | 16.276 | 5836 | 314 | seeded_from_verified_log | `compiler_tiny_lut_gather`, `compiler_single_conv_qlinear`, `compiler_final_equal_overlay`, `compiler_qlinear_uint8` | Gather/LUT operator evidence |
| 340 | 16.285 | 5812 | 279 | seeded_from_verified_log | `compiler_tiny_lut_gather`, `compiler_final_equal_overlay`, `compiler_qlinear_uint8`, `compiler_bounded_scan` | Gather/LUT operator evidence |
| 383 | 16.289 | 5974 | 96 | seeded_from_verified_log | `compiler_tiny_lut_gather`, `compiler_single_conv_qlinear`, `compiler_final_equal_overlay`, `compiler_bounded_scan` | Gather/LUT operator evidence |
| 264 | 16.292 | 5348 | 706 | seeded_from_verified_log | `compiler_direct_output_algebra`, `compiler_tiny_lut_gather`, `compiler_single_conv_qlinear`, `compiler_final_equal_overlay` | Gather/LUT operator evidence |
| 382 | 16.337 | 5648 | 135 | seeded_from_verified_log | `compiler_direct_output_algebra`, `compiler_tiny_lut_gather`, `compiler_final_equal_overlay`, `compiler_bounded_scan` | Gather/LUT operator evidence |
| 008 | 16.365 | 5561 | 65 | seeded_from_verified_log | `compiler_tiny_lut_gather`, `compiler_final_equal_overlay`, `compiler_bounded_scan` | Gather/LUT operator evidence |
| 279 | 16.378 | 5508 | 47 | seeded_from_tasklog_and_inventory | `compiler_tiny_lut_gather`, `compiler_single_conv_qlinear`, `compiler_qlinear_uint8` | Gather/LUT operator evidence |
| 234 | 16.448 | 5113 | 66 | seeded_from_verified_log | `compiler_direct_output_algebra`, `compiler_tiny_lut_gather`, `compiler_final_equal_overlay`, `compiler_bounded_scan` | Gather/LUT operator evidence |
| 004 | 16.454 | 5044 | 100 | seeded_from_verified_log | `compiler_tiny_lut_gather`, `compiler_single_conv_qlinear`, `compiler_final_equal_overlay`, `compiler_qlinear_uint8` | Gather/LUT operator evidence |
| 387 | 16.471 | 4954 | 105 | seeded_from_verified_log | `compiler_tiny_lut_gather`, `compiler_final_equal_overlay`, `compiler_sparse_scatter` | Gather/LUT operator evidence |
| 037 | 16.535 | 4614 | 132 | seeded_from_verified_log | `compiler_tiny_lut_gather`, `compiler_final_equal_overlay`, `compiler_sparse_scatter`, `compiler_bounded_scan` | Gather/LUT operator evidence |
| 361 | 16.547 | 4322 | 366 | seeded_from_verified_log | `compiler_tiny_lut_gather`, `compiler_single_conv_qlinear`, `compiler_final_equal_overlay`, `compiler_roi_pool_crop` | Gather/LUT operator evidence |
| 378 | 16.600 | 4332 | 117 | seeded_from_verified_log | `compiler_direct_output_algebra`, `compiler_tiny_lut_gather`, `compiler_final_equal_overlay`, `compiler_bounded_scan` | Gather/LUT operator evidence |
| 265 | 16.619 | 4328 | 34 | seeded_from_verified_log | `compiler_tiny_lut_gather`, `compiler_final_equal_overlay`, `compiler_sparse_scatter`, `compiler_bounded_scan` | Gather/LUT operator evidence |
| 277 | 16.627 | 4101 | 228 | seeded_from_verified_log | `compiler_tiny_lut_gather`, `compiler_final_equal_overlay`, `compiler_bounded_scan` | Gather/LUT operator evidence |
| 117 | 16.666 | 3922 | 243 | seeded_from_verified_log | `compiler_tiny_lut_gather`, `compiler_single_conv_qlinear`, `compiler_final_equal_overlay`, `compiler_qlinear_uint8` | Gather/LUT operator evidence |
| 132 | 16.684 | 3990 | 99 | seeded_from_verified_log | `compiler_tiny_lut_gather`, `compiler_single_conv_qlinear`, `compiler_final_equal_overlay`, `compiler_bounded_scan` | Gather/LUT operator evidence |
| 161 | 16.684 | 4055 | 33 | seeded_from_verified_log | `compiler_tiny_lut_gather`, `compiler_final_equal_overlay` | Gather/LUT operator evidence |
| 107 | 16.687 | 2924 | 1154 | seeded_from_verified_log | `compiler_direct_onehot_gather`, `compiler_tiny_lut_gather`, `compiler_final_equal_overlay`, `compiler_sparse_scatter` | Gather/LUT operator evidence |
| 365 | 16.699 | 3908 | 120 | seeded_from_verified_log | `compiler_direct_output_algebra`, `compiler_tiny_lut_gather`, `compiler_single_conv_qlinear`, `compiler_final_equal_overlay` | Gather/LUT operator evidence |
| 069 | 16.728 | 3848 | 64 | seeded_from_verified_log | `compiler_tiny_lut_gather`, `compiler_single_conv_qlinear`, `compiler_final_equal_overlay`, `compiler_bounded_scan` | Gather/LUT operator evidence |
| 093 | 16.823 | 3456 | 103 | seeded_from_verified_log | `compiler_tiny_lut_gather`, `compiler_final_equal_overlay`, `compiler_qlinear_uint8`, `compiler_bounded_scan` | Gather/LUT operator evidence |
| 201 | 16.881 | 3202 | 154 | seeded_from_verified_log | `compiler_tiny_lut_gather`, `compiler_single_conv_qlinear`, `compiler_final_equal_overlay`, `compiler_bounded_scan` | Gather/LUT operator evidence |
| 310 | 16.902 | 3208 | 79 | seeded_from_verified_log | `compiler_direct_output_algebra`, `compiler_tiny_lut_gather`, `compiler_final_equal_overlay` | Gather/LUT operator evidence |
| 333 | 16.921 | 2792 | 435 | seeded_from_verified_log | `compiler_direct_output_algebra`, `compiler_tiny_lut_gather`, `compiler_final_equal_overlay`, `compiler_bounded_scan` | Gather/LUT operator evidence |
| 250 | 16.921 | 3164 | 61 | seeded_from_verified_log | `compiler_tiny_lut_gather`, `compiler_final_equal_overlay`, `compiler_qlinear_uint8`, `compiler_bounded_scan` | Gather/LUT operator evidence |
| 086 | 16.949 | 2946 | 190 | seeded_from_verified_log | `compiler_tiny_lut_gather`, `compiler_single_conv_qlinear`, `compiler_final_equal_overlay`, `compiler_qlinear_uint8` | Gather/LUT operator evidence |
| 055 | 16.963 | 3030 | 62 | seeded_from_verified_log | `compiler_tiny_lut_gather`, `compiler_final_equal_overlay`, `compiler_qlinear_uint8`, `compiler_bounded_scan` | Gather/LUT operator evidence |
| 091 | 16.984 | 2853 | 175 | seeded_from_verified_log | `compiler_tiny_lut_gather`, `compiler_single_conv_qlinear`, `compiler_final_equal_overlay` | Gather/LUT operator evidence |
| 070 | 16.991 | 2977 | 31 | seeded_from_verified_log | `compiler_tiny_lut_gather`, `compiler_sparse_scatter`, `compiler_bounded_scan` | Gather/LUT operator evidence |
| 105 | 17.007 | 2854 | 106 | seeded_from_verified_log | `compiler_direct_onehot_gather`, `compiler_tiny_lut_gather`, `compiler_final_equal_overlay`, `compiler_bounded_scan` | Gather/LUT operator evidence |
| 398 | 17.042 | 1666 | 1193 | seeded_from_verified_log | `compiler_direct_onehot_gather`, `compiler_tiny_lut_gather`, `compiler_final_equal_overlay` | Gather/LUT operator evidence |
| 397 | 17.046 | 2648 | 200 | seeded_from_tasklog_and_inventory | `compiler_tiny_lut_gather`, `compiler_single_conv_qlinear`, `compiler_final_equal_overlay`, `compiler_sparse_scatter` | Gather/LUT operator evidence |
| 394 | 17.063 | 1871 | 929 | seeded_from_verified_log | `compiler_tiny_lut_gather`, `compiler_final_equal_overlay`, `compiler_bounded_scan` | Gather/LUT operator evidence |
| 019 | 17.063 | 2709 | 89 | seeded_from_verified_log | `compiler_tiny_lut_gather`, `compiler_single_conv_qlinear`, `compiler_final_equal_overlay`, `compiler_qlinear_uint8` | Gather/LUT operator evidence |
| 154 | 17.077 | 2661 | 99 | seeded_from_verified_log | `compiler_tiny_lut_gather`, `compiler_final_equal_overlay` | Gather/LUT operator evidence |
| 245 | 17.083 | 2604 | 139 | seeded_from_verified_log | `compiler_tiny_lut_gather`, `compiler_bounded_scan` | Gather/LUT operator evidence |
| 071 | 17.100 | 2643 | 55 | seeded_from_verified_log | `compiler_tiny_lut_gather`, `compiler_final_equal_overlay`, `compiler_bounded_scan` | Gather/LUT operator evidence |
| 197 | 17.136 | 2586 | 16 | seeded_from_tasklog_and_inventory | `compiler_direct_output_algebra`, `compiler_tiny_lut_gather`, `compiler_single_conv_qlinear`, `compiler_qlinear_uint8` | Gather/LUT operator evidence |
| 170 | 17.158 | 2216 | 329 | seeded_from_verified_log | `compiler_tiny_lut_gather`, `compiler_single_conv_qlinear`, `compiler_final_equal_overlay`, `compiler_bounded_scan` | Gather/LUT operator evidence |
| 062 | 17.162 | 2465 | 71 | seeded_from_verified_log | `compiler_tiny_lut_gather`, `compiler_final_equal_overlay`, `compiler_bounded_scan` | Gather/LUT operator evidence |
| 238 | 17.178 | 2051 | 444 | seeded_from_verified_log | `compiler_direct_output_algebra`, `compiler_tiny_lut_gather`, `compiler_final_equal_overlay`, `compiler_bounded_scan` | Gather/LUT operator evidence |
| 030 | 17.218 | 2248 | 148 | seeded_from_verified_log | `compiler_tiny_lut_gather`, `compiler_bounded_scan` | Gather/LUT operator evidence |
| 034 | 17.231 | 2198 | 167 | seeded_from_verified_log | `compiler_tiny_lut_gather`, `compiler_single_conv_qlinear`, `compiler_qlinear_uint8`, `compiler_bounded_scan` | Gather/LUT operator evidence |
| 163 | 17.240 | 2235 | 110 | seeded_from_verified_log | `compiler_tiny_lut_gather`, `compiler_single_conv_qlinear`, `compiler_final_equal_overlay`, `compiler_bounded_scan` | Gather/LUT operator evidence |
| 046 | 17.252 | 2221 | 97 | seeded_from_tasklog_and_inventory | `compiler_tiny_lut_gather`, `compiler_single_conv_qlinear`, `compiler_final_equal_overlay`, `compiler_bounded_scan` | Gather/LUT operator evidence |
| 134 | 17.266 | 2250 | 34 | seeded_from_verified_log | `compiler_tiny_lut_gather`, `compiler_bounded_scan` | Gather/LUT operator evidence |
| 012 | 17.275 | 2076 | 188 | seeded_from_verified_log | `compiler_tiny_lut_gather`, `compiler_final_equal_overlay`, `compiler_sparse_scatter` | Gather/LUT operator evidence |
| 168 | 17.286 | 2096 | 143 | seeded_from_verified_log | `compiler_tiny_lut_gather`, `compiler_single_conv_qlinear`, `compiler_qlinear_uint8`, `compiler_bounded_scan` | Gather/LUT operator evidence |
| 336 | 17.298 | 1188 | 1024 | seeded_from_verified_log | `compiler_tiny_lut_gather`, `compiler_sparse_scatter` | Gather/LUT operator evidence |
| 273 | 17.322 | 2109 | 51 | seeded_from_verified_log | `compiler_tiny_lut_gather`, `compiler_bounded_scan` | Gather/LUT operator evidence |
| 260 | 17.344 | 1804 | 310 | seeded_from_verified_log | `compiler_tiny_lut_gather`, `compiler_final_equal_overlay`, `compiler_sparse_scatter`, `compiler_bounded_scan` | Gather/LUT operator evidence |
| 011 | 17.400 | 1836 | 163 | seeded_from_verified_log | `compiler_tiny_lut_gather`, `compiler_single_conv_qlinear`, `compiler_final_equal_overlay` | Gather/LUT operator evidence |
| 287 | 17.402 | 1699 | 295 | seeded_from_verified_log | `compiler_direct_output_algebra`, `compiler_tiny_lut_gather`, `compiler_final_equal_overlay` | Gather/LUT operator evidence |
| 185 | 17.419 | 1651 | 310 | seeded_from_verified_log | `compiler_tiny_lut_gather`, `compiler_single_conv_qlinear`, `compiler_final_equal_overlay`, `compiler_bounded_scan` | Gather/LUT operator evidence |
| 124 | 17.423 | 1879 | 74 | seeded_from_verified_log | `compiler_direct_onehot_gather`, `compiler_tiny_lut_gather`, `compiler_final_equal_overlay`, `compiler_bounded_scan` | Gather/LUT operator evidence |
| 392 | 17.424 | 1602 | 349 | seeded_from_verified_log | `compiler_direct_output_algebra`, `compiler_tiny_lut_gather`, `compiler_final_equal_overlay`, `compiler_bounded_scan` | Gather/LUT operator evidence |
| 013 | 17.435 | 1869 | 61 | seeded_from_verified_log | `compiler_direct_output_algebra`, `compiler_tiny_lut_gather`, `compiler_final_equal_overlay` | Gather/LUT operator evidence |
| 099 | 17.445 | 1633 | 278 | seeded_from_verified_log | `compiler_tiny_lut_gather`, `compiler_final_equal_overlay`, `compiler_sparse_scatter`, `compiler_bounded_scan` | Gather/LUT operator evidence |
| 400 | 17.468 | 1800 | 66 | seeded_from_verified_log | `compiler_tiny_lut_gather`, `compiler_roi_pool_crop`, `compiler_bounded_scan` | Gather/LUT operator evidence |
| 109 | 17.516 | 1686 | 93 | seeded_from_verified_log | `compiler_tiny_lut_gather`, `compiler_final_equal_overlay` | Gather/LUT operator evidence |
| 112 | 17.519 | 1616 | 158 | seeded_from_verified_log | `compiler_direct_output_algebra`, `compiler_tiny_lut_gather` | Gather/LUT operator evidence |
| 156 | 17.536 | 1697 | 47 | seeded_from_tasklog_and_inventory | `compiler_direct_output_algebra`, `compiler_tiny_lut_gather`, `compiler_final_equal_overlay` | Gather/LUT operator evidence |
| 169 | 17.579 | 1500 | 170 | seeded_from_verified_log | `compiler_tiny_lut_gather`, `compiler_single_conv_qlinear`, `compiler_final_equal_overlay`, `compiler_qlinear_uint8` | Gather/LUT operator evidence |
| 061 | 17.581 | 1633 | 35 | seeded_from_verified_log | `compiler_tiny_lut_gather`, `compiler_final_equal_overlay`, `compiler_qlinear_uint8`, `compiler_bounded_scan` | Gather/LUT operator evidence |
| 342 | 17.632 | 1510 | 75 | seeded_from_verified_log | `compiler_direct_output_algebra`, `compiler_tiny_lut_gather`, `compiler_final_equal_overlay`, `compiler_sparse_scatter` | Gather/LUT operator evidence |
| 159 | 17.642 | 1419 | 149 | seeded_from_verified_log | `compiler_direct_output_algebra`, `compiler_tiny_lut_gather`, `compiler_final_equal_overlay` | Gather/LUT operator evidence |
| 020 | 17.670 | 1356 | 170 | seeded_from_verified_log | `compiler_tiny_lut_gather`, `compiler_single_conv_qlinear`, `compiler_sparse_scatter`, `compiler_bounded_scan` | Gather/LUT operator evidence |
| 345 | 17.690 | 1431 | 64 | seeded_from_verified_log | `compiler_tiny_lut_gather` | Gather/LUT operator evidence |
| 240 | 17.704 | 1393 | 82 | seeded_from_verified_log | `compiler_tiny_lut_gather`, `compiler_final_equal_overlay`, `compiler_bounded_scan` | Gather/LUT operator evidence |
| 217 | 17.707 | 1429 | 41 | seeded_from_tasklog_and_inventory | `compiler_tiny_lut_gather`, `compiler_single_conv_qlinear`, `compiler_final_equal_overlay`, `compiler_bounded_scan` | Gather/LUT operator evidence |
| 232 | 17.749 | 0 | 1410 | seeded_from_tasklog_and_inventory | `compiler_tiny_lut_gather`, `compiler_single_conv_qlinear` | Gather/LUT operator evidence |
| 297 | 17.764 | 1350 | 39 | seeded_from_tasklog_and_inventory | `compiler_tiny_lut_gather`, `compiler_final_equal_overlay`, `compiler_bounded_scan` | Gather/LUT operator evidence |
| 119 | 17.781 | 1244 | 121 | seeded_from_verified_log | `compiler_tiny_lut_gather`, `compiler_sparse_scatter`, `compiler_bounded_scan` | Gather/LUT operator evidence |
| 123 | 17.797 | 403 | 940 | seeded_from_verified_log | `compiler_direct_output_algebra`, `compiler_tiny_lut_gather`, `compiler_final_equal_overlay` | Gather/LUT operator evidence |
| 293 | 17.839 | 1247 | 41 | seeded_from_verified_log | `compiler_tiny_lut_gather`, `compiler_final_equal_overlay`, `compiler_sparse_scatter` | Gather/LUT operator evidence |
| 175 | 17.856 | 330 | 937 | seeded_from_verified_log | `compiler_tiny_lut_gather`, `compiler_final_equal_overlay`, `compiler_bounded_scan` | Gather/LUT operator evidence |
| 346 | 17.866 | 1224 | 30 | seeded_from_verified_log | `compiler_tiny_lut_gather`, `compiler_single_conv_qlinear`, `compiler_final_equal_overlay`, `compiler_qlinear_uint8` | Gather/LUT operator evidence |
| 343 | 17.892 | 1147 | 75 | seeded_from_verified_log | `compiler_tiny_lut_gather`, `compiler_single_conv_qlinear`, `compiler_final_equal_overlay`, `compiler_bounded_scan` | Gather/LUT operator evidence |
| 136 | 17.899 | 1179 | 34 | seeded_from_verified_log | `compiler_tiny_lut_gather`, `compiler_final_equal_overlay`, `compiler_bounded_scan` | Gather/LUT operator evidence |
| 244 | 17.928 | 1076 | 103 | seeded_from_verified_log | `compiler_tiny_lut_gather`, `compiler_final_equal_overlay`, `compiler_roi_pool_crop`, `compiler_bounded_scan` | Gather/LUT operator evidence |
| 301 | 17.960 | 1077 | 64 | seeded_from_verified_log | `compiler_tiny_lut_gather`, `compiler_final_equal_overlay`, `compiler_bounded_scan` | Gather/LUT operator evidence |
| 010 | 18.025 | 730 | 340 | seeded_from_verified_log | `compiler_tiny_lut_gather`, `compiler_final_equal_overlay` | Gather/LUT operator evidence |
| 225 | 18.060 | 926 | 107 | seeded_from_verified_log | `compiler_tiny_lut_gather`, `compiler_single_conv_qlinear`, `compiler_final_equal_overlay`, `compiler_bounded_scan` | Gather/LUT operator evidence |
| 058 | 18.060 | 252 | 781 | seeded_from_verified_log | `compiler_tiny_lut_gather` | Gather/LUT operator evidence |
| 228 | 18.131 | 849 | 113 | seeded_from_verified_log | `compiler_direct_output_algebra`, `compiler_tiny_lut_gather`, `compiler_final_equal_overlay`, `compiler_sparse_scatter` | Gather/LUT operator evidence |
| 288 | 18.135 | 769 | 189 | seeded_from_verified_log | `compiler_tiny_lut_gather`, `compiler_final_equal_overlay`, `compiler_sparse_scatter`, `compiler_bounded_scan` | Gather/LUT operator evidence |
| 194 | 18.145 | 900 | 49 | seeded_from_verified_log | `compiler_tiny_lut_gather` | Gather/LUT operator evidence |
| 239 | 18.147 | 897 | 50 | seeded_from_tasklog_and_inventory | `compiler_tiny_lut_gather`, `compiler_final_equal_overlay` | Gather/LUT operator evidence |
| 195 | 18.175 | 882 | 39 | seeded_from_verified_log | `compiler_tiny_lut_gather`, `compiler_bounded_scan` | Gather/LUT operator evidence |
| 259 | 18.192 | 853 | 52 | seeded_from_verified_log | `compiler_tiny_lut_gather`, `compiler_single_conv_qlinear`, `compiler_final_equal_overlay`, `compiler_bounded_scan` | Gather/LUT operator evidence |
| 230 | 18.198 | 0 | 900 | seeded_from_verified_log | `compiler_tiny_lut_gather`, `compiler_single_conv_qlinear` | Gather/LUT operator evidence |
| 106 | 18.233 | 776 | 93 | seeded_from_verified_log | `compiler_tiny_lut_gather`, `compiler_single_conv_qlinear`, `compiler_sparse_scatter` | Gather/LUT operator evidence |
| 153 | 18.322 | 741 | 54 | seeded_from_verified_log | `compiler_direct_output_algebra`, `compiler_tiny_lut_gather`, `compiler_final_equal_overlay`, `compiler_bounded_scan` | Gather/LUT operator evidence |
| 214 | 18.358 | 726 | 41 | seeded_from_verified_log | `compiler_tiny_lut_gather`, `compiler_single_conv_qlinear`, `compiler_final_equal_overlay` | Gather/LUT operator evidence |
| 178 | 18.363 | 702 | 61 | seeded_from_verified_log | `compiler_tiny_lut_gather`, `compiler_single_conv_qlinear`, `compiler_final_equal_overlay`, `compiler_bounded_scan` | Gather/LUT operator evidence |
| 215 | 18.380 | 678 | 72 | seeded_from_tasklog_and_inventory | `compiler_tiny_lut_gather`, `compiler_bounded_scan` | Gather/LUT operator evidence |
| 269 | 18.436 | 685 | 24 | seeded_from_tasklog_and_inventory | `compiler_tiny_lut_gather`, `compiler_roi_pool_crop` | Gather/LUT operator evidence |
| 263 | 18.478 | 563 | 117 | seeded_from_verified_log | `compiler_tiny_lut_gather`, `compiler_single_conv_qlinear`, `compiler_final_equal_overlay` | Gather/LUT operator evidence |
| 057 | 18.496 | 638 | 30 | seeded_from_verified_log | `compiler_tiny_lut_gather`, `compiler_final_equal_overlay`, `compiler_bounded_scan` | Gather/LUT operator evidence |
| 218 | 18.505 | 558 | 104 | seeded_from_verified_log | `compiler_tiny_lut_gather`, `compiler_single_conv_qlinear`, `compiler_final_equal_overlay` | Gather/LUT operator evidence |
| 305 | 18.545 | 296 | 340 | seeded_from_verified_log | `compiler_tiny_lut_gather` | Gather/LUT operator evidence |
| 300 | 18.708 | 474 | 66 | seeded_from_verified_log | `compiler_direct_output_algebra`, `compiler_tiny_lut_gather`, `compiler_final_equal_overlay` | Gather/LUT operator evidence |
| 111 | 18.718 | 528 | 7 | seeded_from_verified_log | `compiler_tiny_lut_gather`, `compiler_bounded_scan` | Gather/LUT operator evidence |
| 362 | 18.744 | 408 | 113 | operator_evidence_only_needs_human_review | `compiler_direct_output_algebra`, `compiler_tiny_lut_gather`, `compiler_final_equal_overlay`, `compiler_sparse_scatter` | Gather/LUT operator evidence |
| 039 | 18.826 | 424 | 56 | seeded_from_verified_log | `compiler_direct_onehot_gather`, `compiler_tiny_lut_gather` | Gather/LUT operator evidence |
| 083 | 18.897 | 376 | 71 | seeded_from_verified_log | `compiler_tiny_lut_gather`, `compiler_sparse_scatter`, `compiler_bounded_scan` | Gather/LUT operator evidence |
| 316 | 18.946 | 388 | 38 | seeded_from_verified_log | `compiler_direct_output_algebra`, `compiler_tiny_lut_gather`, `compiler_final_equal_overlay` | Gather/LUT operator evidence |
| 203 | 18.986 | 408 | 1 | seeded_from_tasklog_and_inventory | `compiler_tiny_lut_gather`, `compiler_final_equal_overlay`, `compiler_bounded_scan` | Gather/LUT operator evidence |
| 290 | 19.128 | 292 | 63 | seeded_from_verified_log | `compiler_direct_output_algebra`, `compiler_tiny_lut_gather`, `compiler_final_equal_overlay` | Gather/LUT operator evidence |
| 266 | 19.142 | 315 | 35 | seeded_from_verified_log | `compiler_tiny_lut_gather` | Gather/LUT operator evidence |
| 121 | 19.168 | 275 | 66 | seeded_from_verified_log | `compiler_direct_output_algebra`, `compiler_tiny_lut_gather` | Gather/LUT operator evidence |
| 003 | 19.290 | 281 | 21 | seeded_from_verified_log | `compiler_tiny_lut_gather`, `compiler_final_equal_overlay` | Gather/LUT operator evidence |
| 306 | 19.296 | 0 | 300 | seeded_from_verified_log | `compiler_direct_output_algebra`, `compiler_tiny_lut_gather` | Gather/LUT operator evidence |
| 249 | 19.341 | 248 | 39 | seeded_from_verified_log | `compiler_tiny_lut_gather` | Gather/LUT operator evidence |
| 104 | 19.463 | 197 | 57 | seeded_from_verified_log | `compiler_direct_onehot_gather`, `compiler_tiny_lut_gather`, `compiler_single_conv_qlinear`, `compiler_qlinear_uint8` | Gather/LUT operator evidence |
| 001 | 19.519 | 0 | 240 | seeded_from_verified_log | `compiler_direct_output_algebra`, `compiler_tiny_lut_gather` | Gather/LUT operator evidence |
| 231 | 19.611 | 180 | 39 | seeded_from_verified_log | `compiler_tiny_lut_gather` | Gather/LUT operator evidence |
| 376 | 19.774 | 144 | 42 | operator_evidence_only_needs_human_review | `compiler_tiny_lut_gather` | Gather/LUT operator evidence |
| 391 | 19.931 | 135 | 24 | seeded_from_verified_log | `compiler_tiny_lut_gather`, `compiler_final_equal_overlay` | Gather/LUT operator evidence |
| 150 | 19.937 | 128 | 30 | seeded_from_verified_log | `compiler_tiny_lut_gather` | Gather/LUT operator evidence |
| 155 | 19.937 | 128 | 30 | seeded_from_verified_log | `compiler_tiny_lut_gather` | Gather/LUT operator evidence |
| 389 | 20.058 | 128 | 12 | seeded_from_verified_log | `compiler_direct_output_algebra`, `compiler_tiny_lut_gather`, `compiler_sparse_scatter`, `compiler_bounded_scan` | Gather/LUT operator evidence |
| 007 | 20.156 | 0 | 127 | seeded_from_verified_log | `compiler_direct_output_algebra`, `compiler_tiny_lut_gather` | Gather/LUT operator evidence |
| 399 | 20.489 | 71 | 20 | seeded_from_verified_log | `compiler_direct_output_algebra`, `compiler_tiny_lut_gather`, `compiler_final_equal_overlay` | Gather/LUT operator evidence |
| 053 | 21.599 | 0 | 30 | seeded_from_verified_log | `compiler_direct_output_algebra`, `compiler_tiny_lut_gather` | Gather/LUT operator evidence |
| 113 | 21.599 | 0 | 30 | seeded_from_verified_log | `compiler_direct_output_algebra`, `compiler_tiny_lut_gather` | Gather/LUT operator evidence |
| 116 | 21.599 | 0 | 30 | operator_evidence_only_needs_human_review | `compiler_direct_output_algebra`, `compiler_tiny_lut_gather` | Gather/LUT operator evidence |
| 164 | 21.599 | 0 | 30 | operator_evidence_only_needs_human_review | `compiler_direct_output_algebra`, `compiler_tiny_lut_gather` | Gather/LUT operator evidence |
| 172 | 21.599 | 0 | 30 | operator_evidence_only_needs_human_review | `compiler_direct_output_algebra`, `compiler_tiny_lut_gather` | Gather/LUT operator evidence |
| 210 | 21.599 | 0 | 30 | operator_evidence_only_needs_human_review | `compiler_direct_output_algebra`, `compiler_tiny_lut_gather` | Gather/LUT operator evidence |
| 311 | 21.599 | 0 | 30 | operator_evidence_only_needs_human_review | `compiler_direct_output_algebra`, `compiler_tiny_lut_gather` | Gather/LUT operator evidence |
| 385 | 21.599 | 0 | 30 | operator_evidence_only_needs_human_review | `compiler_direct_output_algebra`, `compiler_tiny_lut_gather` | Gather/LUT operator evidence |
| 016 | 22.697 | 0 | 10 | operator_evidence_only_needs_human_review | `compiler_direct_output_algebra`, `compiler_tiny_lut_gather` | Gather/LUT operator evidence |
| 276 | 22.697 | 0 | 10 | operator_evidence_only_needs_human_review | `compiler_direct_output_algebra`, `compiler_tiny_lut_gather` | Gather/LUT operator evidence |
| 309 | 22.697 | 0 | 10 | operator_evidence_only_needs_human_review | `compiler_direct_output_algebra`, `compiler_tiny_lut_gather` | Gather/LUT operator evidence |
| 337 | 22.697 | 0 | 10 | operator_evidence_only_needs_human_review | `compiler_direct_output_algebra`, `compiler_tiny_lut_gather` | Gather/LUT operator evidence |
| 140 | 23.391 | 0 | 5 | seeded_from_verified_log | `compiler_direct_output_algebra`, `compiler_tiny_lut_gather`, `compiler_roi_pool_crop` | Gather/LUT operator evidence |

## Known Best Routes

- Pending human review.

## Kill Criteria

- Pending human review.

## Successful Applications

- Pending verification.

## Failed Applications / Walls

- Pending verification.
