# shape_local_stencil — Local stencil

Cell state depends mainly on a local neighbourhood.

## Optimization Question

What is the cheapest verified ONNX family for tasks in this class, and
which semantic preconditions let us avoid full-canvas intermediates?

## Candidate Tasks

| task | pts | mem | params | status | first routes | evidence |
|---:|---:|---:|---:|---|---|---|
| 233 | 14.003 | 59147 | 565 | seeded_from_verified_log | `compiler_direct_onehot_gather`, `compiler_tiny_lut_gather`, `compiler_single_conv_qlinear`, `compiler_final_equal_overlay` | Conv family present |
| 018 | 14.157 | 48196 | 2987 | seeded_from_verified_log | `compiler_tiny_lut_gather`, `compiler_single_conv_qlinear`, `compiler_final_equal_overlay`, `compiler_sparse_scatter` | Conv family present |
| 286 | 14.341 | 41822 | 742 | seeded_from_verified_log | `compiler_tiny_lut_gather`, `compiler_single_conv_qlinear`, `compiler_final_equal_overlay`, `compiler_qlinear_uint8` | Conv family present |
| 366 | 14.497 | 35927 | 490 | seeded_from_verified_log | `compiler_tiny_lut_gather`, `compiler_single_conv_qlinear`, `compiler_final_equal_overlay`, `compiler_sparse_scatter` | Conv family present |
| 158 | 14.528 | 32979 | 2340 | seeded_from_verified_log | `compiler_tiny_lut_gather`, `compiler_single_conv_qlinear`, `compiler_final_equal_overlay`, `compiler_qlinear_uint8` | Conv family present |
| 118 | 14.559 | 30849 | 3387 | seeded_from_verified_log | `compiler_direct_output_algebra`, `compiler_direct_onehot_gather`, `compiler_tiny_lut_gather`, `compiler_single_conv_qlinear` | Conv family present; local_stencil tag |
| 133 | 14.578 | 32294 | 1288 | seeded_from_verified_log | `compiler_single_conv_qlinear`, `compiler_final_equal_overlay`, `compiler_qlinear_uint8`, `compiler_roi_pool_crop` | Conv family present |
| 187 | 14.580 | 32850 | 665 | seeded_from_verified_log | `compiler_single_conv_qlinear`, `compiler_final_equal_overlay`, `compiler_bounded_scan` | Conv family present; local_stencil tag |
| 209 | 14.620 | 32027 | 185 | seeded_from_verified_log | `compiler_tiny_lut_gather`, `compiler_single_conv_qlinear`, `compiler_final_equal_overlay`, `compiler_qlinear_uint8` | Conv family present |
| 191 | 14.624 | 31252 | 841 | seeded_from_verified_log | `compiler_direct_onehot_gather`, `compiler_tiny_lut_gather`, `compiler_single_conv_qlinear`, `compiler_final_equal_overlay` | Conv family present |
| 054 | 14.829 | 25885 | 238 | seeded_from_tasklog_and_inventory | `compiler_tiny_lut_gather`, `compiler_single_conv_qlinear`, `compiler_final_equal_overlay`, `compiler_sparse_scatter` | Conv family present |
| 285 | 14.848 | 25080 | 550 | seeded_from_verified_log | `compiler_tiny_lut_gather`, `compiler_single_conv_qlinear`, `compiler_final_equal_overlay`, `compiler_sparse_scatter` | Conv family present |
| 364 | 14.956 | 22900 | 113 | seeded_from_verified_log | `compiler_tiny_lut_gather`, `compiler_single_conv_qlinear`, `compiler_final_equal_overlay`, `compiler_qlinear_uint8` | Conv family present; local_stencil tag |
| 243 | 14.972 | 22608 | 44 | seeded_from_verified_log | `compiler_single_conv_qlinear`, `compiler_qlinear_uint8` | Conv family present |
| 173 | 15.022 | 21430 | 112 | seeded_from_verified_log | `compiler_tiny_lut_gather`, `compiler_single_conv_qlinear`, `compiler_final_equal_overlay`, `compiler_sparse_scatter` | Conv family present |
| 367 | 15.051 | 16200 | 4733 | seeded_from_verified_log | `compiler_tiny_lut_gather`, `compiler_single_conv_qlinear`, `compiler_qlinear_uint8` | Conv family present; local_stencil tag |
| 349 | 15.102 | 19800 | 90 | seeded_from_verified_log | `compiler_tiny_lut_gather`, `compiler_single_conv_qlinear`, `compiler_final_equal_overlay`, `compiler_qlinear_uint8` | Conv family present; local_stencil tag |
| 205 | 15.360 | 14734 | 638 | seeded_from_verified_log | `compiler_single_conv_qlinear`, `compiler_final_equal_overlay`, `compiler_roi_pool_crop`, `compiler_bounded_scan` | Conv family present |
| 193 | 15.361 | 15284 | 68 | seeded_from_verified_log | `compiler_tiny_lut_gather`, `compiler_single_conv_qlinear`, `compiler_final_equal_overlay` | Conv family present; local_stencil tag |
| 255 | 15.387 | 14663 | 293 | seeded_from_verified_log | `compiler_direct_output_algebra`, `compiler_direct_onehot_gather`, `compiler_single_conv_qlinear`, `compiler_final_equal_overlay` | Conv family present |
| 077 | 15.393 | 14760 | 114 | seeded_from_verified_log | `compiler_single_conv_qlinear`, `compiler_qlinear_uint8`, `compiler_bounded_scan` | Conv family present; local_stencil tag |
| 138 | 15.456 | 13789 | 172 | seeded_from_verified_log | `compiler_tiny_lut_gather`, `compiler_single_conv_qlinear`, `compiler_final_equal_overlay`, `compiler_bounded_scan` | Conv family present; local_stencil tag |
| 110 | 15.468 | 13155 | 634 | seeded_from_verified_log | `compiler_tiny_lut_gather`, `compiler_single_conv_qlinear`, `compiler_final_equal_overlay`, `compiler_bounded_scan` | Conv family present |
| 198 | 15.484 | 13434 | 138 | seeded_from_verified_log | `compiler_tiny_lut_gather`, `compiler_single_conv_qlinear`, `compiler_final_equal_overlay`, `compiler_bounded_scan` | Conv family present |
| 023 | 15.616 | 11603 | 291 | seeded_from_verified_log | `compiler_single_conv_qlinear`, `compiler_final_equal_overlay`, `compiler_qlinear_uint8` | Conv family present |
| 396 | 15.633 | 11567 | 133 | seeded_from_verified_log | `compiler_single_conv_qlinear`, `compiler_final_equal_overlay`, `compiler_roi_pool_crop`, `compiler_bounded_scan` | Conv family present; local_stencil tag |
| 044 | 15.651 | 11420 | 68 | seeded_from_verified_log | `compiler_direct_onehot_gather`, `compiler_tiny_lut_gather`, `compiler_single_conv_qlinear`, `compiler_final_equal_overlay` | Conv family present |
| 338 | 15.712 | 10140 | 666 | seeded_from_verified_log | `compiler_tiny_lut_gather`, `compiler_single_conv_qlinear`, `compiler_final_equal_overlay`, `compiler_qlinear_uint8` | Conv family present; local_stencil tag |
| 080 | 15.726 | 10198 | 454 | seeded_from_verified_log | `compiler_tiny_lut_gather`, `compiler_single_conv_qlinear`, `compiler_final_equal_overlay`, `compiler_qlinear_uint8` | Conv family present |
| 204 | 15.753 | 9920 | 453 | seeded_from_verified_log | `compiler_single_conv_qlinear`, `compiler_qlinear_uint8`, `compiler_bounded_scan` | Conv family present; local_stencil tag |
| 379 | 15.818 | 9069 | 653 | seeded_from_verified_log | `compiler_single_conv_qlinear`, `compiler_final_equal_overlay`, `compiler_bounded_scan` | Conv family present |
| 370 | 15.823 | 8645 | 1024 | seeded_from_verified_log | `compiler_direct_output_algebra`, `compiler_tiny_lut_gather`, `compiler_single_conv_qlinear`, `compiler_final_equal_overlay` | Conv family present |
| 216 | 15.880 | 9048 | 87 | seeded_from_verified_log | `compiler_direct_onehot_gather`, `compiler_tiny_lut_gather`, `compiler_single_conv_qlinear`, `compiler_final_equal_overlay` | Conv family present |
| 074 | 15.889 | 9000 | 50 | seeded_from_verified_log | `compiler_tiny_lut_gather`, `compiler_single_conv_qlinear`, `compiler_final_equal_overlay` | Conv family present; local_stencil tag |
| 324 | 15.907 | 7308 | 1586 | seeded_from_verified_log | `compiler_single_conv_qlinear`, `compiler_final_equal_overlay`, `compiler_qlinear_uint8`, `compiler_bounded_scan` | Conv family present; local_stencil tag |
| 222 | 15.916 | 8736 | 78 | seeded_from_verified_log | `compiler_single_conv_qlinear`, `compiler_final_equal_overlay`, `compiler_qlinear_uint8`, `compiler_bounded_scan` | Conv family present; local_stencil tag |
| 192 | 15.938 | 8515 | 106 | seeded_from_verified_log | `compiler_single_conv_qlinear`, `compiler_final_equal_overlay` | Conv family present; local_stencil tag |
| 377 | 15.975 | 8162 | 150 | seeded_from_verified_log | `compiler_tiny_lut_gather`, `compiler_single_conv_qlinear`, `compiler_final_equal_overlay` | Conv family present; local_stencil tag |
| 202 | 16.039 | 7773 | 24 | seeded_from_verified_log | `compiler_single_conv_qlinear`, `compiler_final_equal_overlay`, `compiler_bounded_scan` | Conv family present; local_stencil tag |
| 089 | 16.111 | 7092 | 160 | seeded_from_verified_log | `compiler_tiny_lut_gather`, `compiler_single_conv_qlinear`, `compiler_final_equal_overlay`, `compiler_sparse_scatter` | Conv family present |
| 009 | 16.143 | 6929 | 95 | seeded_from_verified_log | `compiler_tiny_lut_gather`, `compiler_single_conv_qlinear`, `compiler_final_equal_overlay`, `compiler_bounded_scan` | Conv family present; local_stencil tag |
| 005 | 16.163 | 6392 | 490 | seeded_from_verified_log | `compiler_tiny_lut_gather`, `compiler_single_conv_qlinear`, `compiler_final_equal_overlay`, `compiler_qlinear_uint8` | Conv family present |
| 182 | 16.176 | 6695 | 98 | seeded_from_verified_log | `compiler_tiny_lut_gather`, `compiler_single_conv_qlinear`, `compiler_final_equal_overlay`, `compiler_qlinear_uint8` | Conv family present; local_stencil tag |
| 165 | 16.276 | 5836 | 314 | seeded_from_verified_log | `compiler_tiny_lut_gather`, `compiler_single_conv_qlinear`, `compiler_final_equal_overlay`, `compiler_qlinear_uint8` | Conv family present; local_stencil tag |
| 383 | 16.289 | 5974 | 96 | seeded_from_verified_log | `compiler_tiny_lut_gather`, `compiler_single_conv_qlinear`, `compiler_final_equal_overlay`, `compiler_bounded_scan` | Conv family present; local_stencil tag |
| 264 | 16.292 | 5348 | 706 | seeded_from_verified_log | `compiler_direct_output_algebra`, `compiler_tiny_lut_gather`, `compiler_single_conv_qlinear`, `compiler_final_equal_overlay` | Conv family present; local_stencil tag |
| 279 | 16.378 | 5508 | 47 | seeded_from_tasklog_and_inventory | `compiler_tiny_lut_gather`, `compiler_single_conv_qlinear`, `compiler_qlinear_uint8` | Conv family present; local_stencil tag |
| 278 | 16.391 | 5436 | 47 | seeded_from_verified_log | `compiler_single_conv_qlinear`, `compiler_qlinear_uint8`, `compiler_bounded_scan` | Conv family present; local_stencil tag |
| 085 | 16.409 | 4500 | 881 | seeded_from_verified_log | `compiler_single_conv_qlinear` | Conv family present; local_stencil tag |
| 004 | 16.454 | 5044 | 100 | seeded_from_verified_log | `compiler_tiny_lut_gather`, `compiler_single_conv_qlinear`, `compiler_final_equal_overlay`, `compiler_qlinear_uint8` | Conv family present; local_stencil tag |
| 208 | 16.539 | 4612 | 114 | seeded_from_verified_log | `compiler_direct_output_algebra`, `compiler_single_conv_qlinear`, `compiler_final_equal_overlay`, `compiler_qlinear_uint8` | Conv family present |
| 361 | 16.547 | 4322 | 366 | seeded_from_verified_log | `compiler_tiny_lut_gather`, `compiler_single_conv_qlinear`, `compiler_final_equal_overlay`, `compiler_roi_pool_crop` | Conv family present |
| 196 | 16.580 | 4500 | 38 | seeded_from_verified_log | `compiler_single_conv_qlinear`, `compiler_qlinear_uint8`, `compiler_bounded_scan` | Conv family present; local_stencil tag |
| 206 | 16.657 | 4122 | 77 | seeded_from_verified_log | `compiler_single_conv_qlinear`, `compiler_final_equal_overlay`, `compiler_sparse_scatter`, `compiler_bounded_scan` | Conv family present |
| 117 | 16.666 | 3922 | 243 | seeded_from_verified_log | `compiler_tiny_lut_gather`, `compiler_single_conv_qlinear`, `compiler_final_equal_overlay`, `compiler_qlinear_uint8` | Conv family present; local_stencil tag |
| 363 | 16.673 | 3839 | 296 | seeded_from_verified_log | `compiler_single_conv_qlinear`, `compiler_final_equal_overlay`, `compiler_bounded_scan` | Conv family present; local_stencil tag |
| 132 | 16.684 | 3990 | 99 | seeded_from_verified_log | `compiler_tiny_lut_gather`, `compiler_single_conv_qlinear`, `compiler_final_equal_overlay`, `compiler_bounded_scan` | Conv family present; local_stencil tag |
| 162 | 16.689 | 4024 | 44 | seeded_from_verified_log | `compiler_single_conv_qlinear`, `compiler_qlinear_uint8`, `compiler_bounded_scan` | Conv family present; local_stencil tag |
| 365 | 16.699 | 3908 | 120 | seeded_from_verified_log | `compiler_direct_output_algebra`, `compiler_tiny_lut_gather`, `compiler_single_conv_qlinear`, `compiler_final_equal_overlay` | Conv family present |
| 069 | 16.728 | 3848 | 64 | seeded_from_verified_log | `compiler_tiny_lut_gather`, `compiler_single_conv_qlinear`, `compiler_final_equal_overlay`, `compiler_bounded_scan` | Conv family present |
| 177 | 16.751 | 3692 | 130 | seeded_from_verified_log | `compiler_single_conv_qlinear`, `compiler_final_equal_overlay`, `compiler_roi_pool_crop` | Conv family present; local_stencil tag |
| 284 | 16.774 | 3082 | 653 | seeded_from_verified_log | `compiler_single_conv_qlinear`, `compiler_sparse_scatter` | Conv family present |
| 201 | 16.881 | 3202 | 154 | seeded_from_verified_log | `compiler_tiny_lut_gather`, `compiler_single_conv_qlinear`, `compiler_final_equal_overlay`, `compiler_bounded_scan` | Conv family present; local_stencil tag |
| 079 | 16.947 | 3065 | 78 | seeded_from_verified_log | `compiler_single_conv_qlinear`, `compiler_final_equal_overlay`, `compiler_bounded_scan` | Conv family present; local_stencil tag |
| 086 | 16.949 | 2946 | 190 | seeded_from_verified_log | `compiler_tiny_lut_gather`, `compiler_single_conv_qlinear`, `compiler_final_equal_overlay`, `compiler_qlinear_uint8` | Conv family present; local_stencil tag |
| 042 | 16.958 | 2792 | 318 | seeded_from_verified_log | `compiler_single_conv_qlinear` | Conv family present; local_stencil tag |
| 091 | 16.984 | 2853 | 175 | seeded_from_verified_log | `compiler_tiny_lut_gather`, `compiler_single_conv_qlinear`, `compiler_final_equal_overlay` | Conv family present; local_stencil tag |
| 335 | 17.001 | 2288 | 691 | seeded_from_verified_log | `compiler_direct_output_algebra`, `compiler_single_conv_qlinear`, `compiler_final_equal_overlay`, `compiler_bounded_scan` | Conv family present; local_stencil tag |
| 397 | 17.046 | 2648 | 200 | seeded_from_tasklog_and_inventory | `compiler_tiny_lut_gather`, `compiler_single_conv_qlinear`, `compiler_final_equal_overlay`, `compiler_sparse_scatter` | Conv family present; local_stencil tag |
| 224 | 17.052 | 2460 | 371 | seeded_from_verified_log | `compiler_single_conv_qlinear`, `compiler_bounded_scan` | Conv family present; local_stencil tag |
| 019 | 17.063 | 2709 | 89 | seeded_from_verified_log | `compiler_tiny_lut_gather`, `compiler_single_conv_qlinear`, `compiler_final_equal_overlay`, `compiler_qlinear_uint8` | Conv family present; local_stencil tag |
| 246 | 17.080 | 2334 | 419 | seeded_from_verified_log | `compiler_single_conv_qlinear`, `compiler_final_equal_overlay` | Conv family present; local_stencil tag |
| 354 | 17.080 | 2674 | 77 | seeded_from_verified_log | `compiler_single_conv_qlinear`, `compiler_final_equal_overlay`, `compiler_bounded_scan` | Conv family present; local_stencil tag |
| 190 | 17.089 | 2480 | 247 | seeded_from_verified_log | `compiler_single_conv_qlinear`, `compiler_qlinear_uint8`, `compiler_bounded_scan` | Conv family present; local_stencil tag |
| 094 | 17.094 | 2662 | 51 | seeded_from_verified_log | `compiler_single_conv_qlinear` | Conv family present; local_stencil tag |
| 197 | 17.136 | 2586 | 16 | seeded_from_tasklog_and_inventory | `compiler_direct_output_algebra`, `compiler_tiny_lut_gather`, `compiler_single_conv_qlinear`, `compiler_qlinear_uint8` | Conv family present |
| 170 | 17.158 | 2216 | 329 | seeded_from_verified_log | `compiler_tiny_lut_gather`, `compiler_single_conv_qlinear`, `compiler_final_equal_overlay`, `compiler_bounded_scan` | Conv family present; local_stencil tag |
| 102 | 17.165 | 2414 | 113 | seeded_from_verified_log | `compiler_single_conv_qlinear`, `compiler_qlinear_uint8`, `compiler_bounded_scan` | Conv family present; local_stencil tag |
| 184 | 17.168 | 2460 | 60 | seeded_from_verified_log | `compiler_direct_output_algebra`, `compiler_single_conv_qlinear`, `compiler_final_equal_overlay`, `compiler_bounded_scan` | Conv family present; local_stencil tag |
| 034 | 17.231 | 2198 | 167 | seeded_from_verified_log | `compiler_tiny_lut_gather`, `compiler_single_conv_qlinear`, `compiler_qlinear_uint8`, `compiler_bounded_scan` | Conv family present; local_stencil tag |
| 212 | 17.234 | 2178 | 182 | seeded_from_verified_log | `compiler_single_conv_qlinear`, `compiler_qlinear_uint8`, `compiler_bounded_scan` | Conv family present |
| 163 | 17.240 | 2235 | 110 | seeded_from_verified_log | `compiler_tiny_lut_gather`, `compiler_single_conv_qlinear`, `compiler_final_equal_overlay`, `compiler_bounded_scan` | Conv family present; local_stencil tag |
| 046 | 17.252 | 2221 | 97 | seeded_from_tasklog_and_inventory | `compiler_tiny_lut_gather`, `compiler_single_conv_qlinear`, `compiler_final_equal_overlay`, `compiler_bounded_scan` | Conv family present; local_stencil tag |
| 168 | 17.286 | 2096 | 143 | seeded_from_verified_log | `compiler_tiny_lut_gather`, `compiler_single_conv_qlinear`, `compiler_qlinear_uint8`, `compiler_bounded_scan` | Conv family present; local_stencil tag |
| 323 | 17.314 | 1521 | 656 | seeded_from_verified_log | `compiler_single_conv_qlinear`, `compiler_qlinear_uint8` | Conv family present; local_stencil tag |
| 325 | 17.335 | 2010 | 123 | seeded_from_verified_log | `compiler_single_conv_qlinear`, `compiler_final_equal_overlay`, `compiler_qlinear_uint8`, `compiler_bounded_scan` | Conv family present; local_stencil tag |
| 063 | 17.394 | 1728 | 283 | seeded_from_verified_log | `compiler_single_conv_qlinear`, `compiler_final_equal_overlay` | Conv family present; local_stencil tag |
| 275 | 17.394 | 1374 | 637 | seeded_from_verified_log | `compiler_direct_output_algebra`, `compiler_single_conv_qlinear`, `compiler_final_equal_overlay`, `compiler_bounded_scan` | Conv family present; local_stencil tag |
| 011 | 17.400 | 1836 | 163 | seeded_from_verified_log | `compiler_tiny_lut_gather`, `compiler_single_conv_qlinear`, `compiler_final_equal_overlay` | Conv family present; local_stencil tag |
| 185 | 17.419 | 1651 | 310 | seeded_from_verified_log | `compiler_tiny_lut_gather`, `compiler_single_conv_qlinear`, `compiler_final_equal_overlay`, `compiler_bounded_scan` | Conv family present; local_stencil tag |
| 237 | 17.441 | 1778 | 140 | seeded_from_verified_log | `compiler_single_conv_qlinear`, `compiler_final_equal_overlay`, `compiler_bounded_scan` | Conv family present; local_stencil tag |
| 041 | 17.511 | 1730 | 59 | seeded_from_verified_log | `compiler_single_conv_qlinear`, `compiler_final_equal_overlay` | Conv family present; local_stencil tag |
| 302 | 17.519 | 1216 | 558 | seeded_from_verified_log | `compiler_single_conv_qlinear`, `compiler_qlinear_uint8` | Conv family present; local_stencil tag |
| 169 | 17.579 | 1500 | 170 | seeded_from_verified_log | `compiler_tiny_lut_gather`, `compiler_single_conv_qlinear`, `compiler_final_equal_overlay`, `compiler_qlinear_uint8` | Conv family present; local_stencil tag |
| 369 | 17.623 | 1300 | 299 | seeded_from_verified_log | `compiler_single_conv_qlinear`, `compiler_qlinear_uint8` | Conv family present; local_stencil tag |
| 020 | 17.670 | 1356 | 170 | seeded_from_verified_log | `compiler_tiny_lut_gather`, `compiler_single_conv_qlinear`, `compiler_sparse_scatter`, `compiler_bounded_scan` | Conv family present; local_stencil tag |
| 217 | 17.707 | 1429 | 41 | seeded_from_tasklog_and_inventory | `compiler_tiny_lut_gather`, `compiler_single_conv_qlinear`, `compiler_final_equal_overlay`, `compiler_bounded_scan` | Conv family present; local_stencil tag |
| 304 | 17.727 | 1404 | 37 | seeded_from_tasklog_and_inventory | `compiler_single_conv_qlinear`, `compiler_final_equal_overlay` | Conv family present; local_stencil tag |
| 232 | 17.749 | 0 | 1410 | seeded_from_tasklog_and_inventory | `compiler_tiny_lut_gather`, `compiler_single_conv_qlinear` | Conv family present; local_stencil tag |
| 160 | 17.804 | 1164 | 170 | seeded_from_tasklog_and_inventory | `compiler_single_conv_qlinear`, `compiler_qlinear_uint8`, `compiler_bounded_scan` | Conv family present; local_stencil tag |
| 356 | 17.815 | 1300 | 19 | seeded_from_verified_log | `compiler_single_conv_qlinear`, `compiler_qlinear_uint8`, `compiler_bounded_scan` | Conv family present |
| 346 | 17.866 | 1224 | 30 | seeded_from_verified_log | `compiler_tiny_lut_gather`, `compiler_single_conv_qlinear`, `compiler_final_equal_overlay`, `compiler_qlinear_uint8` | Conv family present; local_stencil tag |
| 343 | 17.892 | 1147 | 75 | seeded_from_verified_log | `compiler_tiny_lut_gather`, `compiler_single_conv_qlinear`, `compiler_final_equal_overlay`, `compiler_bounded_scan` | Conv family present; local_stencil tag |
| 271 | 17.921 | 1141 | 46 | seeded_from_verified_log | `compiler_single_conv_qlinear`, `compiler_qlinear_uint8` | Conv family present; local_stencil tag |
| 188 | 17.921 | 1128 | 59 | seeded_from_verified_log | `compiler_single_conv_qlinear`, `compiler_final_equal_overlay`, `compiler_qlinear_uint8`, `compiler_bounded_scan` | Conv family present; local_stencil tag |
| 048 | 17.956 | 1030 | 116 | seeded_from_verified_log | `compiler_single_conv_qlinear`, `compiler_final_equal_overlay`, `compiler_bounded_scan` | Conv family present |
| 225 | 18.060 | 926 | 107 | seeded_from_verified_log | `compiler_tiny_lut_gather`, `compiler_single_conv_qlinear`, `compiler_final_equal_overlay`, `compiler_bounded_scan` | Conv family present; local_stencil tag |
| 060 | 18.082 | 0 | 1010 | seeded_from_tasklog_and_inventory | `compiler_single_conv_qlinear` | Conv family present; local_stencil tag |
| 108 | 18.122 | 864 | 107 | seeded_from_verified_log | `compiler_single_conv_qlinear`, `compiler_final_equal_overlay`, `compiler_roi_pool_crop` | Conv family present; local_stencil tag |
| 189 | 18.181 | 866 | 49 | seeded_from_verified_log | `compiler_single_conv_qlinear`, `compiler_final_equal_overlay`, `compiler_roi_pool_crop` | Conv family present; local_stencil tag |
| 032 | 18.187 | 0 | 910 | seeded_from_verified_log | `compiler_single_conv_qlinear` | Conv family present; local_stencil tag |
| 097 | 18.187 | 0 | 910 | seeded_from_verified_log | `compiler_single_conv_qlinear` | Conv family present; local_stencil tag |
| 120 | 18.187 | 0 | 910 | seeded_from_verified_log | `compiler_single_conv_qlinear` | Conv family present; local_stencil tag |
| 171 | 18.187 | 0 | 910 | seeded_from_tasklog_and_inventory | `compiler_single_conv_qlinear` | Conv family present; local_stencil tag |
| 283 | 18.187 | 0 | 910 | seeded_from_verified_log | `compiler_single_conv_qlinear` | Conv family present; local_stencil tag |
| 294 | 18.187 | 0 | 910 | seeded_from_verified_log | `compiler_single_conv_qlinear` | Conv family present; local_stencil tag |
| 331 | 18.187 | 0 | 910 | seeded_from_verified_log | `compiler_single_conv_qlinear` | Conv family present; local_stencil tag |
| 344 | 18.187 | 0 | 910 | seeded_from_tasklog_and_inventory | `compiler_single_conv_qlinear` | Conv family present; local_stencil tag |
| 259 | 18.192 | 853 | 52 | seeded_from_verified_log | `compiler_tiny_lut_gather`, `compiler_single_conv_qlinear`, `compiler_final_equal_overlay`, `compiler_bounded_scan` | Conv family present; local_stencil tag |
| 139 | 18.193 | 765 | 139 | seeded_from_verified_log | `compiler_single_conv_qlinear`, `compiler_qlinear_uint8`, `compiler_bounded_scan` | Conv family present |
| 015 | 18.198 | 0 | 900 | seeded_from_verified_log | `compiler_single_conv_qlinear` | Conv family present; local_stencil tag |
| 098 | 18.198 | 0 | 900 | operator_evidence_only_needs_human_review | `compiler_single_conv_qlinear` | Conv family present; local_stencil tag |
| 114 | 18.198 | 0 | 900 | operator_evidence_only_needs_human_review | `compiler_single_conv_qlinear` | Conv family present; local_stencil tag |
| 151 | 18.198 | 0 | 900 | operator_evidence_only_needs_human_review | `compiler_single_conv_qlinear` | Conv family present; local_stencil tag |
| 220 | 18.198 | 0 | 900 | seeded_from_verified_log | `compiler_single_conv_qlinear` | Conv family present; local_stencil tag |
| 230 | 18.198 | 0 | 900 | seeded_from_verified_log | `compiler_tiny_lut_gather`, `compiler_single_conv_qlinear` | Conv family present; local_stencil tag |
| 327 | 18.219 | 810 | 71 | seeded_from_verified_log | `compiler_single_conv_qlinear`, `compiler_final_equal_overlay` | Conv family present; local_stencil tag |
| 106 | 18.233 | 776 | 93 | seeded_from_verified_log | `compiler_tiny_lut_gather`, `compiler_single_conv_qlinear`, `compiler_sparse_scatter` | Conv family present; local_stencil tag |
| 122 | 18.303 | 0 | 810 | seeded_from_verified_log | `compiler_single_conv_qlinear` | Conv family present; local_stencil tag |
| 320 | 18.354 | 448 | 322 | seeded_from_verified_log | `compiler_single_conv_qlinear`, `compiler_sparse_scatter` | Conv family present; local_stencil tag |
| 214 | 18.358 | 726 | 41 | seeded_from_verified_log | `compiler_tiny_lut_gather`, `compiler_single_conv_qlinear`, `compiler_final_equal_overlay` | Conv family present; local_stencil tag |
| 178 | 18.363 | 702 | 61 | seeded_from_verified_log | `compiler_tiny_lut_gather`, `compiler_single_conv_qlinear`, `compiler_final_equal_overlay`, `compiler_bounded_scan` | Conv family present; local_stencil tag |
| 200 | 18.372 | 570 | 186 | seeded_from_verified_log | `compiler_direct_output_algebra`, `compiler_single_conv_qlinear`, `compiler_final_equal_overlay`, `compiler_bounded_scan` | Conv family present; local_stencil tag |
| 372 | 18.435 | 0 | 710 | seeded_from_verified_log | `compiler_single_conv_qlinear` | Conv family present; local_stencil tag |
| 263 | 18.478 | 563 | 117 | seeded_from_verified_log | `compiler_tiny_lut_gather`, `compiler_single_conv_qlinear`, `compiler_final_equal_overlay` | Conv family present; local_stencil tag |
| 218 | 18.505 | 558 | 104 | seeded_from_verified_log | `compiler_tiny_lut_gather`, `compiler_single_conv_qlinear`, `compiler_final_equal_overlay` | Conv family present; local_stencil tag |
| 152 | 18.556 | 504 | 125 | operator_evidence_only_needs_human_review | `compiler_single_conv_qlinear`, `compiler_sparse_scatter` | Conv family present; local_stencil tag |
| 332 | 18.667 | 429 | 134 | seeded_from_verified_log | `compiler_single_conv_qlinear`, `compiler_qlinear_uint8` | Conv family present; local_stencil tag |
| 147 | 18.723 | 432 | 100 | seeded_from_verified_log | `compiler_single_conv_qlinear`, `compiler_sparse_scatter` | Conv family present; local_stencil tag |
| 146 | 18.795 | 388 | 107 | seeded_from_verified_log | `compiler_single_conv_qlinear`, `compiler_final_equal_overlay` | Conv family present; local_stencil tag |
| 081 | 18.860 | 392 | 72 | seeded_from_verified_log | `compiler_single_conv_qlinear`, `compiler_qlinear_uint8` | Conv family present; local_stencil tag |
| 352 | 18.869 | 0 | 460 | seeded_from_verified_log | `compiler_single_conv_qlinear` | Conv family present; local_stencil tag |
| 282 | 18.900 | 407 | 39 | seeded_from_verified_log | `compiler_single_conv_qlinear`, `compiler_qlinear_uint8` | Conv family present; local_stencil tag |
| 257 | 18.986 | 0 | 409 | seeded_from_verified_log | `compiler_single_conv_qlinear` | Conv family present; local_stencil tag |
| 272 | 19.004 | 200 | 202 | seeded_from_verified_log | `compiler_single_conv_qlinear` | Conv family present; local_stencil tag |
| 127 | 19.019 | 204 | 192 | seeded_from_verified_log | `compiler_single_conv_qlinear` | Conv family present; local_stencil tag |
| 049 | 19.044 | 252 | 134 | seeded_from_verified_log | `compiler_direct_output_algebra`, `compiler_single_conv_qlinear`, `compiler_final_equal_overlay`, `compiler_bounded_scan` | Conv family present; local_stencil tag |
| 095 | 19.156 | 245 | 100 | seeded_from_verified_log | `compiler_single_conv_qlinear`, `compiler_qlinear_uint8` | Conv family present |
| 128 | 19.296 | 0 | 300 | seeded_from_verified_log | `compiler_single_conv_qlinear` | Conv family present; local_stencil tag |
| 321 | 19.296 | 0 | 300 | seeded_from_verified_log | `compiler_single_conv_qlinear` | Conv family present; local_stencil tag |
| 357 | 19.327 | 258 | 33 | seeded_from_verified_log | `compiler_single_conv_qlinear`, `compiler_final_equal_overlay`, `compiler_qlinear_uint8` | Conv family present |
| 104 | 19.463 | 197 | 57 | seeded_from_verified_log | `compiler_direct_onehot_gather`, `compiler_tiny_lut_gather`, `compiler_single_conv_qlinear`, `compiler_qlinear_uint8` | Conv family present |
| 180 | 19.653 | 0 | 210 | seeded_from_verified_log | `compiler_single_conv_qlinear` | Conv family present; local_stencil tag |
| 322 | 19.702 | 144 | 56 | operator_evidence_only_needs_human_review | `compiler_single_conv_qlinear`, `compiler_final_equal_overlay`, `compiler_bounded_scan` | Conv family present; local_stencil tag |
| 026 | 19.702 | 0 | 200 | operator_evidence_only_needs_human_review | `compiler_single_conv_qlinear` | Conv family present; local_stencil tag |
| 135 | 19.702 | 0 | 200 | operator_evidence_only_needs_human_review | `compiler_single_conv_qlinear` | Conv family present; local_stencil tag |
| 261 | 19.702 | 0 | 200 | seeded_from_verified_log | `compiler_single_conv_qlinear` | Conv family present; local_stencil tag |
| 082 | 19.753 | 0 | 190 | seeded_from_verified_log | `compiler_single_conv_qlinear` | Conv family present; local_stencil tag |
| 258 | 19.925 | 0 | 160 | seeded_from_verified_log | `compiler_single_conv_qlinear` | Conv family present; local_stencil tag |
| 149 | 20.087 | 36 | 100 | seeded_from_verified_log | `compiler_direct_output_algebra`, `compiler_single_conv_qlinear` | Conv family present; local_stencil tag |
| 144 | 20.356 | 0 | 104 | operator_evidence_only_needs_human_review | `compiler_direct_output_algebra`, `compiler_single_conv_qlinear` | Conv family present; local_stencil tag |
| 227 | 20.395 | 0 | 100 | seeded_from_verified_log | `compiler_direct_output_algebra`, `compiler_single_conv_qlinear` | Conv family present; local_stencil tag |
| 314 | 20.395 | 0 | 100 | operator_evidence_only_needs_human_review | `compiler_direct_output_algebra`, `compiler_single_conv_qlinear` | Conv family present; local_stencil tag |
| 318 | 20.395 | 0 | 100 | seeded_from_verified_log | `compiler_direct_output_algebra`, `compiler_single_conv_qlinear` | Conv family present; local_stencil tag |
| 073 | 21.311 | 0 | 40 | seeded_from_verified_log | `compiler_direct_output_algebra`, `compiler_single_conv_qlinear` | Conv family present; local_stencil tag |
| 130 | 21.599 | 0 | 30 | seeded_from_verified_log | `compiler_direct_output_algebra`, `compiler_single_conv_qlinear` | Conv family present; local_stencil tag |

## Known Best Routes

- Pending human review.

## Kill Criteria

- Pending human review.

## Successful Applications

- Pending verification.

## Failed Applications / Walls

- Pending verification.
