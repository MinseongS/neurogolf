# cost_full_label_plane_floor — Full label plane floor

A 30x30 scalar label/mask carrier likely dominates.

## Optimization Question

What is the cheapest verified ONNX family for tasks in this class, and
which semantic preconditions let us avoid full-canvas intermediates?

## Candidate Tasks

| task | pts | mem | params | status | first routes | evidence |
|---:|---:|---:|---:|---|---|---|
| 233 | 14.003 | 59147 | 565 | seeded_from_verified_log | `compiler_direct_onehot_gather`, `compiler_tiny_lut_gather`, `compiler_single_conv_qlinear`, `compiler_final_equal_overlay` | Equal/Where/Pad with nontrivial memory suggests scalar label/mask carrier |
| 018 | 14.157 | 48196 | 2987 | seeded_from_verified_log | `compiler_tiny_lut_gather`, `compiler_single_conv_qlinear`, `compiler_final_equal_overlay`, `compiler_sparse_scatter` | Equal/Where/Pad with nontrivial memory suggests scalar label/mask carrier |
| 286 | 14.341 | 41822 | 742 | seeded_from_verified_log | `compiler_tiny_lut_gather`, `compiler_single_conv_qlinear`, `compiler_final_equal_overlay`, `compiler_qlinear_uint8` | Equal/Where/Pad with nontrivial memory suggests scalar label/mask carrier |
| 366 | 14.497 | 35927 | 490 | seeded_from_verified_log | `compiler_tiny_lut_gather`, `compiler_single_conv_qlinear`, `compiler_final_equal_overlay`, `compiler_sparse_scatter` | Equal/Where/Pad with nontrivial memory suggests scalar label/mask carrier |
| 158 | 14.528 | 32979 | 2340 | seeded_from_verified_log | `compiler_tiny_lut_gather`, `compiler_single_conv_qlinear`, `compiler_final_equal_overlay`, `compiler_qlinear_uint8` | Equal/Where/Pad with nontrivial memory suggests scalar label/mask carrier |
| 118 | 14.559 | 30849 | 3387 | seeded_from_verified_log | `compiler_direct_output_algebra`, `compiler_direct_onehot_gather`, `compiler_tiny_lut_gather`, `compiler_single_conv_qlinear` | Equal/Where/Pad with nontrivial memory suggests scalar label/mask carrier |
| 133 | 14.578 | 32294 | 1288 | seeded_from_verified_log | `compiler_single_conv_qlinear`, `compiler_final_equal_overlay`, `compiler_qlinear_uint8`, `compiler_roi_pool_crop` | Equal/Where/Pad with nontrivial memory suggests scalar label/mask carrier |
| 187 | 14.580 | 32850 | 665 | seeded_from_verified_log | `compiler_single_conv_qlinear`, `compiler_final_equal_overlay`, `compiler_bounded_scan` | Equal/Where/Pad with nontrivial memory suggests scalar label/mask carrier |
| 209 | 14.620 | 32027 | 185 | seeded_from_verified_log | `compiler_tiny_lut_gather`, `compiler_single_conv_qlinear`, `compiler_final_equal_overlay`, `compiler_qlinear_uint8` | Equal/Where/Pad with nontrivial memory suggests scalar label/mask carrier |
| 191 | 14.624 | 31252 | 841 | seeded_from_verified_log | `compiler_direct_onehot_gather`, `compiler_tiny_lut_gather`, `compiler_single_conv_qlinear`, `compiler_final_equal_overlay` | Equal/Where/Pad with nontrivial memory suggests scalar label/mask carrier |
| 054 | 14.829 | 25885 | 238 | seeded_from_tasklog_and_inventory | `compiler_tiny_lut_gather`, `compiler_single_conv_qlinear`, `compiler_final_equal_overlay`, `compiler_sparse_scatter` | Equal/Where/Pad with nontrivial memory suggests scalar label/mask carrier |
| 285 | 14.848 | 25080 | 550 | seeded_from_verified_log | `compiler_tiny_lut_gather`, `compiler_single_conv_qlinear`, `compiler_final_equal_overlay`, `compiler_sparse_scatter` | Equal/Where/Pad with nontrivial memory suggests scalar label/mask carrier |
| 002 | 14.896 | 24320 | 125 | seeded_from_verified_log | `compiler_direct_onehot_gather`, `compiler_tiny_lut_gather` | Equal/Where/Pad with nontrivial memory suggests scalar label/mask carrier |
| 076 | 14.932 | 23296 | 292 | seeded_from_verified_log | `compiler_tiny_lut_gather`, `compiler_final_equal_overlay`, `compiler_sparse_scatter`, `compiler_bounded_scan` | Equal/Where/Pad with nontrivial memory suggests scalar label/mask carrier |
| 364 | 14.956 | 22900 | 113 | seeded_from_verified_log | `compiler_tiny_lut_gather`, `compiler_single_conv_qlinear`, `compiler_final_equal_overlay`, `compiler_qlinear_uint8` | Equal/Where/Pad with nontrivial memory suggests scalar label/mask carrier |
| 243 | 14.972 | 22608 | 44 | seeded_from_verified_log | `compiler_single_conv_qlinear`, `compiler_qlinear_uint8` | Equal/Where/Pad with nontrivial memory suggests scalar label/mask carrier |
| 173 | 15.022 | 21430 | 112 | seeded_from_verified_log | `compiler_tiny_lut_gather`, `compiler_single_conv_qlinear`, `compiler_final_equal_overlay`, `compiler_sparse_scatter` | Equal/Where/Pad with nontrivial memory suggests scalar label/mask carrier |
| 367 | 15.051 | 16200 | 4733 | seeded_from_verified_log | `compiler_tiny_lut_gather`, `compiler_single_conv_qlinear`, `compiler_qlinear_uint8` | Equal/Where/Pad with nontrivial memory suggests scalar label/mask carrier |
| 349 | 15.102 | 19800 | 90 | seeded_from_verified_log | `compiler_tiny_lut_gather`, `compiler_single_conv_qlinear`, `compiler_final_equal_overlay`, `compiler_qlinear_uint8` | Equal/Where/Pad with nontrivial memory suggests scalar label/mask carrier |
| 319 | 15.119 | 19280 | 279 | seeded_from_verified_log | `compiler_direct_output_algebra`, `compiler_tiny_lut_gather`, `compiler_final_equal_overlay`, `compiler_bounded_scan` | Equal/Where/Pad with nontrivial memory suggests scalar label/mask carrier |
| 219 | 15.195 | 18033 | 93 | seeded_from_verified_log | `compiler_tiny_lut_gather`, `compiler_final_equal_overlay`, `compiler_bounded_scan` | Equal/Where/Pad with nontrivial memory suggests scalar label/mask carrier |
| 101 | 15.216 | 16850 | 905 | seeded_from_verified_log | `compiler_tiny_lut_gather`, `compiler_final_equal_overlay`, `compiler_sparse_scatter`, `compiler_bounded_scan` | Equal/Where/Pad with nontrivial memory suggests scalar label/mask carrier |
| 066 | 15.256 | 16899 | 160 | seeded_from_verified_log | `compiler_direct_output_algebra`, `compiler_final_equal_overlay`, `compiler_bounded_scan` | Equal/Where/Pad with nontrivial memory suggests scalar label/mask carrier |
| 205 | 15.360 | 14734 | 638 | seeded_from_verified_log | `compiler_single_conv_qlinear`, `compiler_final_equal_overlay`, `compiler_roi_pool_crop`, `compiler_bounded_scan` | Equal/Where/Pad with nontrivial memory suggests scalar label/mask carrier |
| 193 | 15.361 | 15284 | 68 | seeded_from_verified_log | `compiler_tiny_lut_gather`, `compiler_single_conv_qlinear`, `compiler_final_equal_overlay` | Equal/Where/Pad with nontrivial memory suggests scalar label/mask carrier |
| 255 | 15.387 | 14663 | 293 | seeded_from_verified_log | `compiler_direct_output_algebra`, `compiler_direct_onehot_gather`, `compiler_single_conv_qlinear`, `compiler_final_equal_overlay` | Equal/Where/Pad with nontrivial memory suggests scalar label/mask carrier |
| 077 | 15.393 | 14760 | 114 | seeded_from_verified_log | `compiler_single_conv_qlinear`, `compiler_qlinear_uint8`, `compiler_bounded_scan` | Equal/Where/Pad with nontrivial memory suggests scalar label/mask carrier |
| 025 | 15.448 | 13874 | 195 | seeded_from_verified_log | `compiler_direct_output_algebra`, `compiler_final_equal_overlay`, `compiler_bounded_scan` | Equal/Where/Pad with nontrivial memory suggests scalar label/mask carrier |
| 138 | 15.456 | 13789 | 172 | seeded_from_verified_log | `compiler_tiny_lut_gather`, `compiler_single_conv_qlinear`, `compiler_final_equal_overlay`, `compiler_bounded_scan` | Equal/Where/Pad with nontrivial memory suggests scalar label/mask carrier |
| 110 | 15.468 | 13155 | 634 | seeded_from_verified_log | `compiler_tiny_lut_gather`, `compiler_single_conv_qlinear`, `compiler_final_equal_overlay`, `compiler_bounded_scan` | Equal/Where/Pad with nontrivial memory suggests scalar label/mask carrier |
| 198 | 15.484 | 13434 | 138 | seeded_from_verified_log | `compiler_tiny_lut_gather`, `compiler_single_conv_qlinear`, `compiler_final_equal_overlay`, `compiler_bounded_scan` | Equal/Where/Pad with nontrivial memory suggests scalar label/mask carrier |
| 064 | 15.494 | 13368 | 68 | seeded_from_verified_log | `compiler_tiny_lut_gather`, `compiler_final_equal_overlay`, `compiler_bounded_scan` | Equal/Where/Pad with nontrivial memory suggests scalar label/mask carrier |
| 145 | 15.511 | 13104 | 111 | seeded_from_verified_log | `compiler_tiny_lut_gather`, `compiler_final_equal_overlay`, `compiler_bounded_scan` | Equal/Where/Pad with nontrivial memory suggests scalar label/mask carrier |
| 023 | 15.616 | 11603 | 291 | seeded_from_verified_log | `compiler_single_conv_qlinear`, `compiler_final_equal_overlay`, `compiler_qlinear_uint8` | Equal/Where/Pad with nontrivial memory suggests scalar label/mask carrier |
| 396 | 15.633 | 11567 | 133 | seeded_from_verified_log | `compiler_single_conv_qlinear`, `compiler_final_equal_overlay`, `compiler_roi_pool_crop`, `compiler_bounded_scan` | Equal/Where/Pad with nontrivial memory suggests scalar label/mask carrier |
| 044 | 15.651 | 11420 | 68 | seeded_from_verified_log | `compiler_direct_onehot_gather`, `compiler_tiny_lut_gather`, `compiler_single_conv_qlinear`, `compiler_final_equal_overlay` | Equal/Where/Pad with nontrivial memory suggests scalar label/mask carrier |
| 338 | 15.712 | 10140 | 666 | seeded_from_verified_log | `compiler_tiny_lut_gather`, `compiler_single_conv_qlinear`, `compiler_final_equal_overlay`, `compiler_qlinear_uint8` | Equal/Where/Pad with nontrivial memory suggests scalar label/mask carrier |
| 029 | 15.715 | 10736 | 34 | seeded_from_verified_log | `compiler_direct_output_algebra`, `compiler_tiny_lut_gather`, `compiler_final_equal_overlay`, `compiler_bounded_scan` | Equal/Where/Pad with nontrivial memory suggests scalar label/mask carrier |
| 080 | 15.726 | 10198 | 454 | seeded_from_verified_log | `compiler_tiny_lut_gather`, `compiler_single_conv_qlinear`, `compiler_final_equal_overlay`, `compiler_qlinear_uint8` | Equal/Where/Pad with nontrivial memory suggests scalar label/mask carrier |
| 017 | 15.744 | 8448 | 2021 | seeded_from_verified_log | `compiler_tiny_lut_gather`, `compiler_final_equal_overlay` | Equal/Where/Pad with nontrivial memory suggests scalar label/mask carrier |
| 204 | 15.753 | 9920 | 453 | seeded_from_verified_log | `compiler_single_conv_qlinear`, `compiler_qlinear_uint8`, `compiler_bounded_scan` | Equal/Where/Pad with nontrivial memory suggests scalar label/mask carrier |
| 379 | 15.818 | 9069 | 653 | seeded_from_verified_log | `compiler_single_conv_qlinear`, `compiler_final_equal_overlay`, `compiler_bounded_scan` | Equal/Where/Pad with nontrivial memory suggests scalar label/mask carrier |
| 370 | 15.823 | 8645 | 1024 | seeded_from_verified_log | `compiler_direct_output_algebra`, `compiler_tiny_lut_gather`, `compiler_single_conv_qlinear`, `compiler_final_equal_overlay` | Equal/Where/Pad with nontrivial memory suggests scalar label/mask carrier |
| 216 | 15.880 | 9048 | 87 | seeded_from_verified_log | `compiler_direct_onehot_gather`, `compiler_tiny_lut_gather`, `compiler_single_conv_qlinear`, `compiler_final_equal_overlay` | Equal/Where/Pad with nontrivial memory suggests scalar label/mask carrier |
| 074 | 15.889 | 9000 | 50 | seeded_from_verified_log | `compiler_tiny_lut_gather`, `compiler_single_conv_qlinear`, `compiler_final_equal_overlay` | Equal/Where/Pad with nontrivial memory suggests scalar label/mask carrier |
| 350 | 15.891 | 9012 | 26 | seeded_from_verified_log | `compiler_tiny_lut_gather`, `compiler_bounded_scan` | Equal/Where/Pad with nontrivial memory suggests scalar label/mask carrier |
| 096 | 15.905 | 8108 | 805 | seeded_from_verified_log | `compiler_tiny_lut_gather`, `compiler_final_equal_overlay`, `compiler_sparse_scatter`, `compiler_bounded_scan` | Equal/Where/Pad with nontrivial memory suggests scalar label/mask carrier |
| 324 | 15.907 | 7308 | 1586 | seeded_from_verified_log | `compiler_single_conv_qlinear`, `compiler_final_equal_overlay`, `compiler_qlinear_uint8`, `compiler_bounded_scan` | Equal/Where/Pad with nontrivial memory suggests scalar label/mask carrier |
| 222 | 15.916 | 8736 | 78 | seeded_from_verified_log | `compiler_single_conv_qlinear`, `compiler_final_equal_overlay`, `compiler_qlinear_uint8`, `compiler_bounded_scan` | Equal/Where/Pad with nontrivial memory suggests scalar label/mask carrier |
| 192 | 15.938 | 8515 | 106 | seeded_from_verified_log | `compiler_single_conv_qlinear`, `compiler_final_equal_overlay` | Equal/Where/Pad with nontrivial memory suggests scalar label/mask carrier |
| 157 | 15.972 | 7983 | 351 | seeded_from_verified_log | `compiler_tiny_lut_gather`, `compiler_final_equal_overlay`, `compiler_sparse_scatter`, `compiler_bounded_scan` | Equal/Where/Pad with nontrivial memory suggests scalar label/mask carrier |
| 377 | 15.975 | 8162 | 150 | seeded_from_verified_log | `compiler_tiny_lut_gather`, `compiler_single_conv_qlinear`, `compiler_final_equal_overlay` | Equal/Where/Pad with nontrivial memory suggests scalar label/mask carrier |
| 014 | 16.023 | 7809 | 108 | seeded_from_verified_log | `compiler_tiny_lut_gather`, `compiler_final_equal_overlay`, `compiler_bounded_scan` | Equal/Where/Pad with nontrivial memory suggests scalar label/mask carrier |
| 202 | 16.039 | 7773 | 24 | seeded_from_verified_log | `compiler_single_conv_qlinear`, `compiler_final_equal_overlay`, `compiler_bounded_scan` | Equal/Where/Pad with nontrivial memory suggests scalar label/mask carrier |
| 089 | 16.111 | 7092 | 160 | seeded_from_verified_log | `compiler_tiny_lut_gather`, `compiler_single_conv_qlinear`, `compiler_final_equal_overlay`, `compiler_sparse_scatter` | Equal/Where/Pad with nontrivial memory suggests scalar label/mask carrier |
| 174 | 16.130 | 6973 | 142 | seeded_from_verified_log | `compiler_direct_onehot_gather`, `compiler_tiny_lut_gather`, `compiler_final_equal_overlay`, `compiler_bounded_scan` | Equal/Where/Pad with nontrivial memory suggests scalar label/mask carrier |
| 009 | 16.143 | 6929 | 95 | seeded_from_verified_log | `compiler_tiny_lut_gather`, `compiler_single_conv_qlinear`, `compiler_final_equal_overlay`, `compiler_bounded_scan` | Equal/Where/Pad with nontrivial memory suggests scalar label/mask carrier |
| 092 | 16.148 | 6805 | 182 | seeded_from_verified_log | `compiler_direct_output_algebra`, `compiler_final_equal_overlay`, `compiler_sparse_scatter`, `compiler_bounded_scan` | Equal/Where/Pad with nontrivial memory suggests scalar label/mask carrier |
| 005 | 16.163 | 6392 | 490 | seeded_from_verified_log | `compiler_tiny_lut_gather`, `compiler_single_conv_qlinear`, `compiler_final_equal_overlay`, `compiler_qlinear_uint8` | Equal/Where/Pad with nontrivial memory suggests scalar label/mask carrier |
| 182 | 16.176 | 6695 | 98 | seeded_from_verified_log | `compiler_tiny_lut_gather`, `compiler_single_conv_qlinear`, `compiler_final_equal_overlay`, `compiler_qlinear_uint8` | Equal/Where/Pad with nontrivial memory suggests scalar label/mask carrier |
| 328 | 16.179 | 4835 | 1940 | seeded_from_tasklog_and_inventory | `compiler_tiny_lut_gather`, `compiler_final_equal_overlay` | Equal/Where/Pad with nontrivial memory suggests scalar label/mask carrier |
| 368 | 16.264 | 6022 | 203 | seeded_from_verified_log | `compiler_tiny_lut_gather`, `compiler_final_equal_overlay`, `compiler_sparse_scatter` | Equal/Where/Pad with nontrivial memory suggests scalar label/mask carrier |
| 165 | 16.276 | 5836 | 314 | seeded_from_verified_log | `compiler_tiny_lut_gather`, `compiler_single_conv_qlinear`, `compiler_final_equal_overlay`, `compiler_qlinear_uint8` | Equal/Where/Pad with nontrivial memory suggests scalar label/mask carrier |
| 340 | 16.285 | 5812 | 279 | seeded_from_verified_log | `compiler_tiny_lut_gather`, `compiler_final_equal_overlay`, `compiler_qlinear_uint8`, `compiler_bounded_scan` | Equal/Where/Pad with nontrivial memory suggests scalar label/mask carrier |
| 383 | 16.289 | 5974 | 96 | seeded_from_verified_log | `compiler_tiny_lut_gather`, `compiler_single_conv_qlinear`, `compiler_final_equal_overlay`, `compiler_bounded_scan` | Equal/Where/Pad with nontrivial memory suggests scalar label/mask carrier |
| 264 | 16.292 | 5348 | 706 | seeded_from_verified_log | `compiler_direct_output_algebra`, `compiler_tiny_lut_gather`, `compiler_single_conv_qlinear`, `compiler_final_equal_overlay` | Equal/Where/Pad with nontrivial memory suggests scalar label/mask carrier |
| 280 | 16.294 | 5286 | 753 | seeded_from_verified_log | `compiler_direct_output_algebra`, `compiler_final_equal_overlay` | Equal/Where/Pad with nontrivial memory suggests scalar label/mask carrier |
| 148 | 16.323 | 5759 | 110 | seeded_from_verified_log | `compiler_final_equal_overlay`, `compiler_sparse_scatter`, `compiler_bounded_scan` | Equal/Where/Pad with nontrivial memory suggests scalar label/mask carrier |
| 382 | 16.337 | 5648 | 135 | seeded_from_verified_log | `compiler_direct_output_algebra`, `compiler_tiny_lut_gather`, `compiler_final_equal_overlay`, `compiler_bounded_scan` | Equal/Where/Pad with nontrivial memory suggests scalar label/mask carrier |
| 008 | 16.365 | 5561 | 65 | seeded_from_verified_log | `compiler_tiny_lut_gather`, `compiler_final_equal_overlay`, `compiler_bounded_scan` | Equal/Where/Pad with nontrivial memory suggests scalar label/mask carrier |
| 279 | 16.378 | 5508 | 47 | seeded_from_tasklog_and_inventory | `compiler_tiny_lut_gather`, `compiler_single_conv_qlinear`, `compiler_qlinear_uint8` | Equal/Where/Pad with nontrivial memory suggests scalar label/mask carrier |
| 278 | 16.391 | 5436 | 47 | seeded_from_verified_log | `compiler_single_conv_qlinear`, `compiler_qlinear_uint8`, `compiler_bounded_scan` | Equal/Where/Pad with nontrivial memory suggests scalar label/mask carrier |
| 085 | 16.409 | 4500 | 881 | seeded_from_verified_log | `compiler_single_conv_qlinear` | Equal/Where/Pad with nontrivial memory suggests scalar label/mask carrier |
| 234 | 16.448 | 5113 | 66 | seeded_from_verified_log | `compiler_direct_output_algebra`, `compiler_tiny_lut_gather`, `compiler_final_equal_overlay`, `compiler_bounded_scan` | Equal/Where/Pad with nontrivial memory suggests scalar label/mask carrier |
| 004 | 16.454 | 5044 | 100 | seeded_from_verified_log | `compiler_tiny_lut_gather`, `compiler_single_conv_qlinear`, `compiler_final_equal_overlay`, `compiler_qlinear_uint8` | Equal/Where/Pad with nontrivial memory suggests scalar label/mask carrier |
| 387 | 16.471 | 4954 | 105 | seeded_from_verified_log | `compiler_tiny_lut_gather`, `compiler_final_equal_overlay`, `compiler_sparse_scatter` | Equal/Where/Pad with nontrivial memory suggests scalar label/mask carrier |
| 251 | 16.499 | 4752 | 168 | seeded_from_verified_log | `compiler_bounded_scan` | Equal/Where/Pad with nontrivial memory suggests scalar label/mask carrier |
| 037 | 16.535 | 4614 | 132 | seeded_from_verified_log | `compiler_tiny_lut_gather`, `compiler_final_equal_overlay`, `compiler_sparse_scatter`, `compiler_bounded_scan` | Equal/Where/Pad with nontrivial memory suggests scalar label/mask carrier |
| 208 | 16.539 | 4612 | 114 | seeded_from_verified_log | `compiler_direct_output_algebra`, `compiler_single_conv_qlinear`, `compiler_final_equal_overlay`, `compiler_qlinear_uint8` | Equal/Where/Pad with nontrivial memory suggests scalar label/mask carrier |
| 361 | 16.547 | 4322 | 366 | seeded_from_verified_log | `compiler_tiny_lut_gather`, `compiler_single_conv_qlinear`, `compiler_final_equal_overlay`, `compiler_roi_pool_crop` | Equal/Where/Pad with nontrivial memory suggests scalar label/mask carrier |
| 196 | 16.580 | 4500 | 38 | seeded_from_verified_log | `compiler_single_conv_qlinear`, `compiler_qlinear_uint8`, `compiler_bounded_scan` | Equal/Where/Pad with nontrivial memory suggests scalar label/mask carrier |
| 358 | 16.587 | 4332 | 174 | seeded_from_verified_log | `compiler_direct_output_algebra`, `compiler_final_equal_overlay`, `compiler_bounded_scan` | Equal/Where/Pad with nontrivial memory suggests scalar label/mask carrier |
| 378 | 16.600 | 4332 | 117 | seeded_from_verified_log | `compiler_direct_output_algebra`, `compiler_tiny_lut_gather`, `compiler_final_equal_overlay`, `compiler_bounded_scan` | Equal/Where/Pad with nontrivial memory suggests scalar label/mask carrier |
| 265 | 16.619 | 4328 | 34 | seeded_from_verified_log | `compiler_tiny_lut_gather`, `compiler_final_equal_overlay`, `compiler_sparse_scatter`, `compiler_bounded_scan` | Equal/Where/Pad with nontrivial memory suggests scalar label/mask carrier |
| 277 | 16.627 | 4101 | 228 | seeded_from_verified_log | `compiler_tiny_lut_gather`, `compiler_final_equal_overlay`, `compiler_bounded_scan` | Equal/Where/Pad with nontrivial memory suggests scalar label/mask carrier |
| 206 | 16.657 | 4122 | 77 | seeded_from_verified_log | `compiler_single_conv_qlinear`, `compiler_final_equal_overlay`, `compiler_sparse_scatter`, `compiler_bounded_scan` | Equal/Where/Pad with nontrivial memory suggests scalar label/mask carrier |
| 117 | 16.666 | 3922 | 243 | seeded_from_verified_log | `compiler_tiny_lut_gather`, `compiler_single_conv_qlinear`, `compiler_final_equal_overlay`, `compiler_qlinear_uint8` | Equal/Where/Pad with nontrivial memory suggests scalar label/mask carrier |
| 363 | 16.673 | 3839 | 296 | seeded_from_verified_log | `compiler_single_conv_qlinear`, `compiler_final_equal_overlay`, `compiler_bounded_scan` | Equal/Where/Pad with nontrivial memory suggests scalar label/mask carrier |
| 132 | 16.684 | 3990 | 99 | seeded_from_verified_log | `compiler_tiny_lut_gather`, `compiler_single_conv_qlinear`, `compiler_final_equal_overlay`, `compiler_bounded_scan` | Equal/Where/Pad with nontrivial memory suggests scalar label/mask carrier |
| 161 | 16.684 | 4055 | 33 | seeded_from_verified_log | `compiler_tiny_lut_gather`, `compiler_final_equal_overlay` | Equal/Where/Pad with nontrivial memory suggests scalar label/mask carrier |
| 107 | 16.687 | 2924 | 1154 | seeded_from_verified_log | `compiler_direct_onehot_gather`, `compiler_tiny_lut_gather`, `compiler_final_equal_overlay`, `compiler_sparse_scatter` | Equal/Where/Pad with nontrivial memory suggests scalar label/mask carrier |
| 162 | 16.689 | 4024 | 44 | seeded_from_verified_log | `compiler_single_conv_qlinear`, `compiler_qlinear_uint8`, `compiler_bounded_scan` | Equal/Where/Pad with nontrivial memory suggests scalar label/mask carrier |
| 365 | 16.699 | 3908 | 120 | seeded_from_verified_log | `compiler_direct_output_algebra`, `compiler_tiny_lut_gather`, `compiler_single_conv_qlinear`, `compiler_final_equal_overlay` | Equal/Where/Pad with nontrivial memory suggests scalar label/mask carrier |
| 131 | 16.709 | 3312 | 675 | seeded_from_verified_log | `compiler_sparse_scatter`, `compiler_bounded_scan` | Equal/Where/Pad with nontrivial memory suggests scalar label/mask carrier |
| 069 | 16.728 | 3848 | 64 | seeded_from_verified_log | `compiler_tiny_lut_gather`, `compiler_single_conv_qlinear`, `compiler_final_equal_overlay`, `compiler_bounded_scan` | Equal/Where/Pad with nontrivial memory suggests scalar label/mask carrier |
| 050 | 16.728 | 3825 | 86 | seeded_from_verified_log | `compiler_direct_onehot_gather` | Equal/Where/Pad with nontrivial memory suggests scalar label/mask carrier |
| 359 | 16.740 | 3852 | 15 | seeded_from_verified_log | `compiler_final_equal_overlay`, `compiler_bounded_scan` | Equal/Where/Pad with nontrivial memory suggests scalar label/mask carrier |
| 177 | 16.751 | 3692 | 130 | seeded_from_verified_log | `compiler_single_conv_qlinear`, `compiler_final_equal_overlay`, `compiler_roi_pool_crop` | Equal/Where/Pad with nontrivial memory suggests scalar label/mask carrier |
| 268 | 16.816 | 3328 | 256 | seeded_from_verified_log | `compiler_final_equal_overlay` | Equal/Where/Pad with nontrivial memory suggests scalar label/mask carrier |
| 093 | 16.823 | 3456 | 103 | seeded_from_verified_log | `compiler_tiny_lut_gather`, `compiler_final_equal_overlay`, `compiler_qlinear_uint8`, `compiler_bounded_scan` | Equal/Where/Pad with nontrivial memory suggests scalar label/mask carrier |
| 137 | 16.832 | 3433 | 93 | seeded_from_verified_log | `compiler_direct_output_algebra`, `compiler_final_equal_overlay` | Equal/Where/Pad with nontrivial memory suggests scalar label/mask carrier |
| 125 | 16.850 | 3434 | 29 | seeded_from_verified_log | `compiler_final_equal_overlay`, `compiler_bounded_scan` | Equal/Where/Pad with nontrivial memory suggests scalar label/mask carrier |
| 201 | 16.881 | 3202 | 154 | seeded_from_verified_log | `compiler_tiny_lut_gather`, `compiler_single_conv_qlinear`, `compiler_final_equal_overlay`, `compiler_bounded_scan` | Equal/Where/Pad with nontrivial memory suggests scalar label/mask carrier |
| 310 | 16.902 | 3208 | 79 | seeded_from_verified_log | `compiler_direct_output_algebra`, `compiler_tiny_lut_gather`, `compiler_final_equal_overlay` | Equal/Where/Pad with nontrivial memory suggests scalar label/mask carrier |
| 333 | 16.921 | 2792 | 435 | seeded_from_verified_log | `compiler_direct_output_algebra`, `compiler_tiny_lut_gather`, `compiler_final_equal_overlay`, `compiler_bounded_scan` | Equal/Where/Pad with nontrivial memory suggests scalar label/mask carrier |
| 250 | 16.921 | 3164 | 61 | seeded_from_verified_log | `compiler_tiny_lut_gather`, `compiler_final_equal_overlay`, `compiler_qlinear_uint8`, `compiler_bounded_scan` | Equal/Where/Pad with nontrivial memory suggests scalar label/mask carrier |
| 079 | 16.947 | 3065 | 78 | seeded_from_verified_log | `compiler_single_conv_qlinear`, `compiler_final_equal_overlay`, `compiler_bounded_scan` | Equal/Where/Pad with nontrivial memory suggests scalar label/mask carrier |
| 086 | 16.949 | 2946 | 190 | seeded_from_verified_log | `compiler_tiny_lut_gather`, `compiler_single_conv_qlinear`, `compiler_final_equal_overlay`, `compiler_qlinear_uint8` | Equal/Where/Pad with nontrivial memory suggests scalar label/mask carrier |
| 035 | 16.954 | 2994 | 128 | seeded_from_verified_log | `compiler_sparse_scatter` | Equal/Where/Pad with nontrivial memory suggests scalar label/mask carrier |
| 042 | 16.958 | 2792 | 318 | seeded_from_verified_log | `compiler_single_conv_qlinear` | Equal/Where/Pad with nontrivial memory suggests scalar label/mask carrier |
| 055 | 16.963 | 3030 | 62 | seeded_from_verified_log | `compiler_tiny_lut_gather`, `compiler_final_equal_overlay`, `compiler_qlinear_uint8`, `compiler_bounded_scan` | Equal/Where/Pad with nontrivial memory suggests scalar label/mask carrier |
| 090 | 16.964 | 3009 | 82 | seeded_from_verified_log | `compiler_bounded_scan` | Equal/Where/Pad with nontrivial memory suggests scalar label/mask carrier |
| 390 | 16.964 | 2624 | 467 | seeded_from_verified_log | `compiler_final_equal_overlay`, `compiler_sparse_scatter` | Equal/Where/Pad with nontrivial memory suggests scalar label/mask carrier |
| 270 | 16.971 | 2742 | 327 | seeded_from_verified_log | `compiler_direct_output_algebra`, `compiler_bounded_scan` | Equal/Where/Pad with nontrivial memory suggests scalar label/mask carrier |
| 091 | 16.984 | 2853 | 175 | seeded_from_verified_log | `compiler_tiny_lut_gather`, `compiler_single_conv_qlinear`, `compiler_final_equal_overlay` | Equal/Where/Pad with nontrivial memory suggests scalar label/mask carrier |
| 330 | 16.986 | 2800 | 222 | seeded_from_verified_log | `compiler_final_equal_overlay`, `compiler_sparse_scatter`, `compiler_bounded_scan` | Equal/Where/Pad with nontrivial memory suggests scalar label/mask carrier |
| 070 | 16.991 | 2977 | 31 | seeded_from_verified_log | `compiler_tiny_lut_gather`, `compiler_sparse_scatter`, `compiler_bounded_scan` | Equal/Where/Pad with nontrivial memory suggests scalar label/mask carrier |
| 335 | 17.001 | 2288 | 691 | seeded_from_verified_log | `compiler_direct_output_algebra`, `compiler_single_conv_qlinear`, `compiler_final_equal_overlay`, `compiler_bounded_scan` | Equal/Where/Pad with nontrivial memory suggests scalar label/mask carrier |
| 105 | 17.007 | 2854 | 106 | seeded_from_verified_log | `compiler_direct_onehot_gather`, `compiler_tiny_lut_gather`, `compiler_final_equal_overlay`, `compiler_bounded_scan` | Equal/Where/Pad with nontrivial memory suggests scalar label/mask carrier |
| 256 | 17.011 | 2898 | 50 | seeded_from_tasklog_and_inventory | `compiler_final_equal_overlay`, `compiler_bounded_scan` | Equal/Where/Pad with nontrivial memory suggests scalar label/mask carrier |
| 398 | 17.042 | 1666 | 1193 | seeded_from_verified_log | `compiler_direct_onehot_gather`, `compiler_tiny_lut_gather`, `compiler_final_equal_overlay` | Equal/Where/Pad with nontrivial memory suggests scalar label/mask carrier |
| 397 | 17.046 | 2648 | 200 | seeded_from_tasklog_and_inventory | `compiler_tiny_lut_gather`, `compiler_single_conv_qlinear`, `compiler_final_equal_overlay`, `compiler_sparse_scatter` | Equal/Where/Pad with nontrivial memory suggests scalar label/mask carrier |
| 224 | 17.052 | 2460 | 371 | seeded_from_verified_log | `compiler_single_conv_qlinear`, `compiler_bounded_scan` | Equal/Where/Pad with nontrivial memory suggests scalar label/mask carrier |
| 394 | 17.063 | 1871 | 929 | seeded_from_verified_log | `compiler_tiny_lut_gather`, `compiler_final_equal_overlay`, `compiler_bounded_scan` | Equal/Where/Pad with nontrivial memory suggests scalar label/mask carrier |
| 019 | 17.063 | 2709 | 89 | seeded_from_verified_log | `compiler_tiny_lut_gather`, `compiler_single_conv_qlinear`, `compiler_final_equal_overlay`, `compiler_qlinear_uint8` | Equal/Where/Pad with nontrivial memory suggests scalar label/mask carrier |
| 154 | 17.077 | 2661 | 99 | seeded_from_verified_log | `compiler_tiny_lut_gather`, `compiler_final_equal_overlay` | Equal/Where/Pad with nontrivial memory suggests scalar label/mask carrier |
| 246 | 17.080 | 2334 | 419 | seeded_from_verified_log | `compiler_single_conv_qlinear`, `compiler_final_equal_overlay` | Equal/Where/Pad with nontrivial memory suggests scalar label/mask carrier |
| 354 | 17.080 | 2674 | 77 | seeded_from_verified_log | `compiler_single_conv_qlinear`, `compiler_final_equal_overlay`, `compiler_bounded_scan` | Equal/Where/Pad with nontrivial memory suggests scalar label/mask carrier |
| 245 | 17.083 | 2604 | 139 | seeded_from_verified_log | `compiler_tiny_lut_gather`, `compiler_bounded_scan` | Equal/Where/Pad with nontrivial memory suggests scalar label/mask carrier |
| 190 | 17.089 | 2480 | 247 | seeded_from_verified_log | `compiler_single_conv_qlinear`, `compiler_qlinear_uint8`, `compiler_bounded_scan` | Equal/Where/Pad with nontrivial memory suggests scalar label/mask carrier |
| 355 | 17.093 | 2688 | 27 | seeded_from_verified_log | `compiler_final_equal_overlay`, `compiler_bounded_scan` | Equal/Where/Pad with nontrivial memory suggests scalar label/mask carrier |
| 094 | 17.094 | 2662 | 51 | seeded_from_verified_log | `compiler_single_conv_qlinear` | Equal/Where/Pad with nontrivial memory suggests scalar label/mask carrier |
| 071 | 17.100 | 2643 | 55 | seeded_from_verified_log | `compiler_tiny_lut_gather`, `compiler_final_equal_overlay`, `compiler_bounded_scan` | Equal/Where/Pad with nontrivial memory suggests scalar label/mask carrier |
| 170 | 17.158 | 2216 | 329 | seeded_from_verified_log | `compiler_tiny_lut_gather`, `compiler_single_conv_qlinear`, `compiler_final_equal_overlay`, `compiler_bounded_scan` | Equal/Where/Pad with nontrivial memory suggests scalar label/mask carrier |
| 062 | 17.162 | 2465 | 71 | seeded_from_verified_log | `compiler_tiny_lut_gather`, `compiler_final_equal_overlay`, `compiler_bounded_scan` | Equal/Where/Pad with nontrivial memory suggests scalar label/mask carrier |
| 102 | 17.165 | 2414 | 113 | seeded_from_verified_log | `compiler_single_conv_qlinear`, `compiler_qlinear_uint8`, `compiler_bounded_scan` | Equal/Where/Pad with nontrivial memory suggests scalar label/mask carrier |
| 184 | 17.168 | 2460 | 60 | seeded_from_verified_log | `compiler_direct_output_algebra`, `compiler_single_conv_qlinear`, `compiler_final_equal_overlay`, `compiler_bounded_scan` | Equal/Where/Pad with nontrivial memory suggests scalar label/mask carrier |
| 388 | 17.178 | 2278 | 218 | seeded_from_verified_log | `compiler_final_equal_overlay`, `compiler_sparse_scatter`, `compiler_bounded_scan` | Equal/Where/Pad with nontrivial memory suggests scalar label/mask carrier |
| 238 | 17.178 | 2051 | 444 | seeded_from_verified_log | `compiler_direct_output_algebra`, `compiler_tiny_lut_gather`, `compiler_final_equal_overlay`, `compiler_bounded_scan` | Equal/Where/Pad with nontrivial memory suggests scalar label/mask carrier |
| 030 | 17.218 | 2248 | 148 | seeded_from_verified_log | `compiler_tiny_lut_gather`, `compiler_bounded_scan` | Equal/Where/Pad with nontrivial memory suggests scalar label/mask carrier |
| 281 | 17.223 | 2321 | 65 | seeded_from_verified_log | `compiler_direct_output_algebra`, `compiler_final_equal_overlay`, `compiler_bounded_scan` | Equal/Where/Pad with nontrivial memory suggests scalar label/mask carrier |
| 034 | 17.231 | 2198 | 167 | seeded_from_verified_log | `compiler_tiny_lut_gather`, `compiler_single_conv_qlinear`, `compiler_qlinear_uint8`, `compiler_bounded_scan` | Equal/Where/Pad with nontrivial memory suggests scalar label/mask carrier |
| 212 | 17.234 | 2178 | 182 | seeded_from_verified_log | `compiler_single_conv_qlinear`, `compiler_qlinear_uint8`, `compiler_bounded_scan` | Equal/Where/Pad with nontrivial memory suggests scalar label/mask carrier |
| 163 | 17.240 | 2235 | 110 | seeded_from_verified_log | `compiler_tiny_lut_gather`, `compiler_single_conv_qlinear`, `compiler_final_equal_overlay`, `compiler_bounded_scan` | Equal/Where/Pad with nontrivial memory suggests scalar label/mask carrier |
| 046 | 17.252 | 2221 | 97 | seeded_from_tasklog_and_inventory | `compiler_tiny_lut_gather`, `compiler_single_conv_qlinear`, `compiler_final_equal_overlay`, `compiler_bounded_scan` | Equal/Where/Pad with nontrivial memory suggests scalar label/mask carrier |
| 374 | 17.258 | 2262 | 42 | seeded_from_verified_log | `compiler_final_equal_overlay`, `compiler_bounded_scan` | Equal/Where/Pad with nontrivial memory suggests scalar label/mask carrier |
| 022 | 17.258 | 2004 | 298 | seeded_from_verified_log | `compiler_direct_output_algebra`, `compiler_final_equal_overlay`, `compiler_roi_pool_crop` | Equal/Where/Pad with nontrivial memory suggests scalar label/mask carrier |
| 134 | 17.266 | 2250 | 34 | seeded_from_verified_log | `compiler_tiny_lut_gather`, `compiler_bounded_scan` | Equal/Where/Pad with nontrivial memory suggests scalar label/mask carrier |
| 012 | 17.275 | 2076 | 188 | seeded_from_verified_log | `compiler_tiny_lut_gather`, `compiler_final_equal_overlay`, `compiler_sparse_scatter` | Equal/Where/Pad with nontrivial memory suggests scalar label/mask carrier |
| 168 | 17.286 | 2096 | 143 | seeded_from_verified_log | `compiler_tiny_lut_gather`, `compiler_single_conv_qlinear`, `compiler_qlinear_uint8`, `compiler_bounded_scan` | Equal/Where/Pad with nontrivial memory suggests scalar label/mask carrier |
| 036 | 17.314 | 2140 | 37 | seeded_from_verified_log | `compiler_final_equal_overlay`, `compiler_bounded_scan` | Equal/Where/Pad with nontrivial memory suggests scalar label/mask carrier |
| 273 | 17.322 | 2109 | 51 | seeded_from_verified_log | `compiler_tiny_lut_gather`, `compiler_bounded_scan` | Equal/Where/Pad with nontrivial memory suggests scalar label/mask carrier |
| 325 | 17.335 | 2010 | 123 | seeded_from_verified_log | `compiler_single_conv_qlinear`, `compiler_final_equal_overlay`, `compiler_qlinear_uint8`, `compiler_bounded_scan` | Equal/Where/Pad with nontrivial memory suggests scalar label/mask carrier |
| 260 | 17.344 | 1804 | 310 | seeded_from_verified_log | `compiler_tiny_lut_gather`, `compiler_final_equal_overlay`, `compiler_sparse_scatter`, `compiler_bounded_scan` | Equal/Where/Pad with nontrivial memory suggests scalar label/mask carrier |
| 348 | 17.353 | 1938 | 157 | seeded_from_verified_log | `compiler_final_equal_overlay` | Equal/Where/Pad with nontrivial memory suggests scalar label/mask carrier |
| 063 | 17.394 | 1728 | 283 | seeded_from_verified_log | `compiler_single_conv_qlinear`, `compiler_final_equal_overlay` | Equal/Where/Pad with nontrivial memory suggests scalar label/mask carrier |
| 275 | 17.394 | 1374 | 637 | seeded_from_verified_log | `compiler_direct_output_algebra`, `compiler_single_conv_qlinear`, `compiler_final_equal_overlay`, `compiler_bounded_scan` | Equal/Where/Pad with nontrivial memory suggests scalar label/mask carrier |
| 011 | 17.400 | 1836 | 163 | seeded_from_verified_log | `compiler_tiny_lut_gather`, `compiler_single_conv_qlinear`, `compiler_final_equal_overlay` | Equal/Where/Pad with nontrivial memory suggests scalar label/mask carrier |
| 287 | 17.402 | 1699 | 295 | seeded_from_verified_log | `compiler_direct_output_algebra`, `compiler_tiny_lut_gather`, `compiler_final_equal_overlay` | Equal/Where/Pad with nontrivial memory suggests scalar label/mask carrier |
| 381 | 17.418 | 1908 | 55 | seeded_from_verified_log | `compiler_direct_onehot_gather` | Equal/Where/Pad with nontrivial memory suggests scalar label/mask carrier |
| 185 | 17.419 | 1651 | 310 | seeded_from_verified_log | `compiler_tiny_lut_gather`, `compiler_single_conv_qlinear`, `compiler_final_equal_overlay`, `compiler_bounded_scan` | Equal/Where/Pad with nontrivial memory suggests scalar label/mask carrier |
| 124 | 17.423 | 1879 | 74 | seeded_from_verified_log | `compiler_direct_onehot_gather`, `compiler_tiny_lut_gather`, `compiler_final_equal_overlay`, `compiler_bounded_scan` | Equal/Where/Pad with nontrivial memory suggests scalar label/mask carrier |
| 392 | 17.424 | 1602 | 349 | seeded_from_verified_log | `compiler_direct_output_algebra`, `compiler_tiny_lut_gather`, `compiler_final_equal_overlay`, `compiler_bounded_scan` | Equal/Where/Pad with nontrivial memory suggests scalar label/mask carrier |
| 013 | 17.435 | 1869 | 61 | seeded_from_verified_log | `compiler_direct_output_algebra`, `compiler_tiny_lut_gather`, `compiler_final_equal_overlay` | Equal/Where/Pad with nontrivial memory suggests scalar label/mask carrier |
| 237 | 17.441 | 1778 | 140 | seeded_from_verified_log | `compiler_single_conv_qlinear`, `compiler_final_equal_overlay`, `compiler_bounded_scan` | Equal/Where/Pad with nontrivial memory suggests scalar label/mask carrier |
| 099 | 17.445 | 1633 | 278 | seeded_from_verified_log | `compiler_tiny_lut_gather`, `compiler_final_equal_overlay`, `compiler_sparse_scatter`, `compiler_bounded_scan` | Equal/Where/Pad with nontrivial memory suggests scalar label/mask carrier |
| 213 | 17.457 | 1838 | 49 | seeded_from_verified_log | `compiler_direct_output_algebra`, `compiler_final_equal_overlay` | Equal/Where/Pad with nontrivial memory suggests scalar label/mask carrier |
| 088 | 17.465 | 1791 | 81 | seeded_from_verified_log | `compiler_direct_output_algebra`, `compiler_final_equal_overlay` | Equal/Where/Pad with nontrivial memory suggests scalar label/mask carrier |
| 400 | 17.468 | 1800 | 66 | seeded_from_verified_log | `compiler_tiny_lut_gather`, `compiler_roi_pool_crop`, `compiler_bounded_scan` | Equal/Where/Pad with nontrivial memory suggests scalar label/mask carrier |
| 199 | 17.484 | 1749 | 88 | seeded_from_verified_log | `compiler_direct_output_algebra`, `compiler_final_equal_overlay`, `compiler_sparse_scatter` | Equal/Where/Pad with nontrivial memory suggests scalar label/mask carrier |
| 051 | 17.503 | 1702 | 101 | seeded_from_verified_log | `compiler_direct_output_algebra`, `compiler_final_equal_overlay` | Equal/Where/Pad with nontrivial memory suggests scalar label/mask carrier |
| 041 | 17.511 | 1730 | 59 | seeded_from_verified_log | `compiler_single_conv_qlinear`, `compiler_final_equal_overlay` | Equal/Where/Pad with nontrivial memory suggests scalar label/mask carrier |
| 109 | 17.516 | 1686 | 93 | seeded_from_verified_log | `compiler_tiny_lut_gather`, `compiler_final_equal_overlay` | Equal/Where/Pad with nontrivial memory suggests scalar label/mask carrier |
| 112 | 17.519 | 1616 | 158 | seeded_from_verified_log | `compiler_direct_output_algebra`, `compiler_tiny_lut_gather` | Equal/Where/Pad with nontrivial memory suggests scalar label/mask carrier |
| 156 | 17.536 | 1697 | 47 | seeded_from_tasklog_and_inventory | `compiler_direct_output_algebra`, `compiler_tiny_lut_gather`, `compiler_final_equal_overlay` | Equal/Where/Pad with nontrivial memory suggests scalar label/mask carrier |
| 169 | 17.579 | 1500 | 170 | seeded_from_verified_log | `compiler_tiny_lut_gather`, `compiler_single_conv_qlinear`, `compiler_final_equal_overlay`, `compiler_qlinear_uint8` | Equal/Where/Pad with nontrivial memory suggests scalar label/mask carrier |
| 061 | 17.581 | 1633 | 35 | seeded_from_verified_log | `compiler_tiny_lut_gather`, `compiler_final_equal_overlay`, `compiler_qlinear_uint8`, `compiler_bounded_scan` | Equal/Where/Pad with nontrivial memory suggests scalar label/mask carrier |
| 351 | 17.591 | 1632 | 19 | seeded_from_verified_log | `compiler_bounded_scan` | Equal/Where/Pad with nontrivial memory suggests scalar label/mask carrier |
| 033 | 17.598 | 1606 | 33 | seeded_from_verified_log |  | Equal/Where/Pad with nontrivial memory suggests scalar label/mask carrier |
| 226 | 17.602 | 1613 | 20 | seeded_from_verified_log | `compiler_final_equal_overlay` | Equal/Where/Pad with nontrivial memory suggests scalar label/mask carrier |
| 303 | 17.610 | 1598 | 22 | seeded_from_verified_log | `compiler_direct_output_algebra`, `compiler_final_equal_overlay` | Equal/Where/Pad with nontrivial memory suggests scalar label/mask carrier |
| 295 | 17.620 | 1557 | 47 | seeded_from_tasklog_and_inventory | `compiler_final_equal_overlay` | Equal/Where/Pad with nontrivial memory suggests scalar label/mask carrier |
| 342 | 17.632 | 1510 | 75 | seeded_from_verified_log | `compiler_direct_output_algebra`, `compiler_tiny_lut_gather`, `compiler_final_equal_overlay`, `compiler_sparse_scatter` | Equal/Where/Pad with nontrivial memory suggests scalar label/mask carrier |
| 159 | 17.642 | 1419 | 149 | seeded_from_verified_log | `compiler_direct_output_algebra`, `compiler_tiny_lut_gather`, `compiler_final_equal_overlay` | Equal/Where/Pad with nontrivial memory suggests scalar label/mask carrier |
| 141 | 17.683 | 1404 | 101 | seeded_from_verified_log | `compiler_direct_output_algebra`, `compiler_final_equal_overlay` | Equal/Where/Pad with nontrivial memory suggests scalar label/mask carrier |
| 345 | 17.690 | 1431 | 64 | seeded_from_verified_log | `compiler_tiny_lut_gather` | Equal/Where/Pad with nontrivial memory suggests scalar label/mask carrier |
| 308 | 17.695 | 1385 | 103 | seeded_from_verified_log | `compiler_direct_output_algebra`, `compiler_final_equal_overlay`, `compiler_sparse_scatter`, `compiler_bounded_scan` | Equal/Where/Pad with nontrivial memory suggests scalar label/mask carrier |
| 075 | 17.695 | 1326 | 161 | seeded_from_verified_log | `compiler_direct_output_algebra`, `compiler_final_equal_overlay` | Equal/Where/Pad with nontrivial memory suggests scalar label/mask carrier |
| 031 | 17.700 | 1434 | 46 | seeded_from_verified_log | `compiler_bounded_scan` | Equal/Where/Pad with nontrivial memory suggests scalar label/mask carrier |
| 240 | 17.704 | 1393 | 82 | seeded_from_verified_log | `compiler_tiny_lut_gather`, `compiler_final_equal_overlay`, `compiler_bounded_scan` | Equal/Where/Pad with nontrivial memory suggests scalar label/mask carrier |
| 217 | 17.707 | 1429 | 41 | seeded_from_tasklog_and_inventory | `compiler_tiny_lut_gather`, `compiler_single_conv_qlinear`, `compiler_final_equal_overlay`, `compiler_bounded_scan` | Equal/Where/Pad with nontrivial memory suggests scalar label/mask carrier |
| 304 | 17.727 | 1404 | 37 | seeded_from_tasklog_and_inventory | `compiler_single_conv_qlinear`, `compiler_final_equal_overlay` | Equal/Where/Pad with nontrivial memory suggests scalar label/mask carrier |
| 341 | 17.735 | 1394 | 35 | seeded_from_verified_log | `compiler_final_equal_overlay`, `compiler_bounded_scan` | Equal/Where/Pad with nontrivial memory suggests scalar label/mask carrier |
| 297 | 17.764 | 1350 | 39 | seeded_from_tasklog_and_inventory | `compiler_tiny_lut_gather`, `compiler_final_equal_overlay`, `compiler_bounded_scan` | Equal/Where/Pad with nontrivial memory suggests scalar label/mask carrier |
| 119 | 17.781 | 1244 | 121 | seeded_from_verified_log | `compiler_tiny_lut_gather`, `compiler_sparse_scatter`, `compiler_bounded_scan` | Equal/Where/Pad with nontrivial memory suggests scalar label/mask carrier |
| 027 | 17.822 | 1262 | 48 | seeded_from_verified_log |  | Equal/Where/Pad with nontrivial memory suggests scalar label/mask carrier |
| 078 | 17.835 | 1260 | 33 | seeded_from_tasklog_and_inventory |  | Equal/Where/Pad with nontrivial memory suggests scalar label/mask carrier |
| 293 | 17.839 | 1247 | 41 | seeded_from_verified_log | `compiler_tiny_lut_gather`, `compiler_final_equal_overlay`, `compiler_sparse_scatter` | Equal/Where/Pad with nontrivial memory suggests scalar label/mask carrier |
| 346 | 17.866 | 1224 | 30 | seeded_from_verified_log | `compiler_tiny_lut_gather`, `compiler_single_conv_qlinear`, `compiler_final_equal_overlay`, `compiler_qlinear_uint8` | Equal/Where/Pad with nontrivial memory suggests scalar label/mask carrier |
| 183 | 17.885 | 1008 | 222 | seeded_from_verified_log | `compiler_direct_output_algebra`, `compiler_final_equal_overlay` | Equal/Where/Pad with nontrivial memory suggests scalar label/mask carrier |
| 343 | 17.892 | 1147 | 75 | seeded_from_verified_log | `compiler_tiny_lut_gather`, `compiler_single_conv_qlinear`, `compiler_final_equal_overlay`, `compiler_bounded_scan` | Equal/Where/Pad with nontrivial memory suggests scalar label/mask carrier |
| 136 | 17.899 | 1179 | 34 | seeded_from_verified_log | `compiler_tiny_lut_gather`, `compiler_final_equal_overlay`, `compiler_bounded_scan` | Equal/Where/Pad with nontrivial memory suggests scalar label/mask carrier |
| 271 | 17.921 | 1141 | 46 | seeded_from_verified_log | `compiler_single_conv_qlinear`, `compiler_qlinear_uint8` | Equal/Where/Pad with nontrivial memory suggests scalar label/mask carrier |
| 188 | 17.921 | 1128 | 59 | seeded_from_verified_log | `compiler_single_conv_qlinear`, `compiler_final_equal_overlay`, `compiler_qlinear_uint8`, `compiler_bounded_scan` | Equal/Where/Pad with nontrivial memory suggests scalar label/mask carrier |
| 244 | 17.928 | 1076 | 103 | seeded_from_verified_log | `compiler_tiny_lut_gather`, `compiler_final_equal_overlay`, `compiler_roi_pool_crop`, `compiler_bounded_scan` | Equal/Where/Pad with nontrivial memory suggests scalar label/mask carrier |
| 021 | 17.935 | 1072 | 98 | seeded_from_tasklog_and_inventory | `compiler_direct_output_algebra` | Equal/Where/Pad with nontrivial memory suggests scalar label/mask carrier |
| 048 | 17.956 | 1030 | 116 | seeded_from_verified_log | `compiler_single_conv_qlinear`, `compiler_final_equal_overlay`, `compiler_bounded_scan` | Equal/Where/Pad with nontrivial memory suggests scalar label/mask carrier |
| 301 | 17.960 | 1077 | 64 | seeded_from_verified_log | `compiler_tiny_lut_gather`, `compiler_final_equal_overlay`, `compiler_bounded_scan` | Equal/Where/Pad with nontrivial memory suggests scalar label/mask carrier |
| 254 | 17.999 | 1079 | 19 | seeded_from_verified_log | `compiler_final_equal_overlay`, `compiler_bounded_scan` | Equal/Where/Pad with nontrivial memory suggests scalar label/mask carrier |
| 329 | 18.035 | 1029 | 30 | seeded_from_verified_log | `compiler_final_equal_overlay`, `compiler_sparse_scatter` | Equal/Where/Pad with nontrivial memory suggests scalar label/mask carrier |
| 115 | 18.050 | 980 | 63 | seeded_from_verified_log | `compiler_direct_output_algebra`, `compiler_bounded_scan` | Equal/Where/Pad with nontrivial memory suggests scalar label/mask carrier |
| 225 | 18.060 | 926 | 107 | seeded_from_verified_log | `compiler_tiny_lut_gather`, `compiler_single_conv_qlinear`, `compiler_final_equal_overlay`, `compiler_bounded_scan` | Equal/Where/Pad with nontrivial memory suggests scalar label/mask carrier |
| 108 | 18.122 | 864 | 107 | seeded_from_verified_log | `compiler_single_conv_qlinear`, `compiler_final_equal_overlay`, `compiler_roi_pool_crop` | Equal/Where/Pad with nontrivial memory suggests scalar label/mask carrier |
| 228 | 18.131 | 849 | 113 | seeded_from_verified_log | `compiler_direct_output_algebra`, `compiler_tiny_lut_gather`, `compiler_final_equal_overlay`, `compiler_sparse_scatter` | Equal/Where/Pad with nontrivial memory suggests scalar label/mask carrier |
| 194 | 18.145 | 900 | 49 | seeded_from_verified_log | `compiler_tiny_lut_gather` | Equal/Where/Pad with nontrivial memory suggests scalar label/mask carrier |
| 239 | 18.147 | 897 | 50 | seeded_from_tasklog_and_inventory | `compiler_tiny_lut_gather`, `compiler_final_equal_overlay` | Equal/Where/Pad with nontrivial memory suggests scalar label/mask carrier |
| 195 | 18.175 | 882 | 39 | seeded_from_verified_log | `compiler_tiny_lut_gather`, `compiler_bounded_scan` | Equal/Where/Pad with nontrivial memory suggests scalar label/mask carrier |
| 189 | 18.181 | 866 | 49 | seeded_from_verified_log | `compiler_single_conv_qlinear`, `compiler_final_equal_overlay`, `compiler_roi_pool_crop` | Equal/Where/Pad with nontrivial memory suggests scalar label/mask carrier |
| 259 | 18.192 | 853 | 52 | seeded_from_verified_log | `compiler_tiny_lut_gather`, `compiler_single_conv_qlinear`, `compiler_final_equal_overlay`, `compiler_bounded_scan` | Equal/Where/Pad with nontrivial memory suggests scalar label/mask carrier |
| 327 | 18.219 | 810 | 71 | seeded_from_verified_log | `compiler_single_conv_qlinear`, `compiler_final_equal_overlay` | Equal/Where/Pad with nontrivial memory suggests scalar label/mask carrier |
| 047 | 18.241 | 826 | 36 | seeded_from_verified_log | `compiler_bounded_scan` | Equal/Where/Pad with nontrivial memory suggests scalar label/mask carrier |

## Known Best Routes

- Pending human review.

## Kill Criteria

- Pending human review.

## Successful Applications

- Pending verification.

## Failed Applications / Walls

- Pending verification.
