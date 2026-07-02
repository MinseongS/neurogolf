# direction_axis_aligned — Axis-aligned direction

Uses horizontal/vertical rows, columns, bars, or spans.

## Optimization Question

What is the cheapest verified ONNX family for tasks in this class, and
which semantic preconditions let us avoid full-canvas intermediates?

## Candidate Tasks

| task | pts | mem | params | status | first routes | evidence |
|---:|---:|---:|---:|---|---|---|
| 233 | 14.003 | 59147 | 565 | seeded_from_verified_log | `compiler_direct_onehot_gather`, `compiler_tiny_lut_gather`, `compiler_single_conv_qlinear`, `compiler_final_equal_overlay` | tasklog keyword match: \b(row\|column\|horizontal\|vertical\|axis\|bar\|span) |
| 286 | 14.341 | 41822 | 742 | seeded_from_verified_log | `compiler_tiny_lut_gather`, `compiler_single_conv_qlinear`, `compiler_final_equal_overlay`, `compiler_qlinear_uint8` | tasklog keyword match: \b(row\|column\|horizontal\|vertical\|axis\|bar\|span) |
| 158 | 14.528 | 32979 | 2340 | seeded_from_verified_log | `compiler_tiny_lut_gather`, `compiler_single_conv_qlinear`, `compiler_final_equal_overlay`, `compiler_qlinear_uint8` | tasklog keyword match: \b(row\|column\|horizontal\|vertical\|axis\|bar\|span) |
| 118 | 14.559 | 30849 | 3387 | seeded_from_verified_log | `compiler_direct_output_algebra`, `compiler_direct_onehot_gather`, `compiler_tiny_lut_gather`, `compiler_single_conv_qlinear` | tasklog keyword match: \b(row\|column\|horizontal\|vertical\|axis\|bar\|span) |
| 133 | 14.578 | 32294 | 1288 | seeded_from_verified_log | `compiler_single_conv_qlinear`, `compiler_final_equal_overlay`, `compiler_qlinear_uint8`, `compiler_roi_pool_crop` | tasklog keyword match: \b(row\|column\|horizontal\|vertical\|axis\|bar\|span) |
| 209 | 14.620 | 32027 | 185 | seeded_from_verified_log | `compiler_tiny_lut_gather`, `compiler_single_conv_qlinear`, `compiler_final_equal_overlay`, `compiler_qlinear_uint8` | tasklog keyword match: \b(row\|column\|horizontal\|vertical\|axis\|bar\|span) |
| 191 | 14.624 | 31252 | 841 | seeded_from_verified_log | `compiler_direct_onehot_gather`, `compiler_tiny_lut_gather`, `compiler_single_conv_qlinear`, `compiler_final_equal_overlay` | tasklog keyword match: \b(row\|column\|horizontal\|vertical\|axis\|bar\|span) |
| 054 | 14.829 | 25885 | 238 | seeded_from_tasklog_and_inventory | `compiler_tiny_lut_gather`, `compiler_single_conv_qlinear`, `compiler_final_equal_overlay`, `compiler_sparse_scatter` | tasklog keyword match: \b(row\|column\|horizontal\|vertical\|axis\|bar\|span) |
| 002 | 14.896 | 24320 | 125 | seeded_from_verified_log | `compiler_direct_onehot_gather`, `compiler_tiny_lut_gather` | tasklog keyword match: \b(row\|column\|horizontal\|vertical\|axis\|bar\|span) |
| 076 | 14.932 | 23296 | 292 | seeded_from_verified_log | `compiler_tiny_lut_gather`, `compiler_final_equal_overlay`, `compiler_sparse_scatter`, `compiler_bounded_scan` | tasklog keyword match: \b(row\|column\|horizontal\|vertical\|axis\|bar\|span) |
| 364 | 14.956 | 22900 | 113 | seeded_from_verified_log | `compiler_tiny_lut_gather`, `compiler_single_conv_qlinear`, `compiler_final_equal_overlay`, `compiler_qlinear_uint8` | tasklog keyword match: \b(row\|column\|horizontal\|vertical\|axis\|bar\|span) |
| 243 | 14.972 | 22608 | 44 | seeded_from_verified_log | `compiler_single_conv_qlinear`, `compiler_qlinear_uint8` | tasklog keyword match: \b(row\|column\|horizontal\|vertical\|axis\|bar\|span) |
| 367 | 15.051 | 16200 | 4733 | seeded_from_verified_log | `compiler_tiny_lut_gather`, `compiler_single_conv_qlinear`, `compiler_qlinear_uint8` | tasklog keyword match: \b(row\|column\|horizontal\|vertical\|axis\|bar\|span) |
| 349 | 15.102 | 19800 | 90 | seeded_from_verified_log | `compiler_tiny_lut_gather`, `compiler_single_conv_qlinear`, `compiler_final_equal_overlay`, `compiler_qlinear_uint8` | tasklog keyword match: \b(row\|column\|horizontal\|vertical\|axis\|bar\|span) |
| 319 | 15.119 | 19280 | 279 | seeded_from_verified_log | `compiler_direct_output_algebra`, `compiler_tiny_lut_gather`, `compiler_final_equal_overlay`, `compiler_bounded_scan` | tasklog keyword match: \b(row\|column\|horizontal\|vertical\|axis\|bar\|span) |
| 219 | 15.195 | 18033 | 93 | seeded_from_verified_log | `compiler_tiny_lut_gather`, `compiler_final_equal_overlay`, `compiler_bounded_scan` | tasklog keyword match: \b(row\|column\|horizontal\|vertical\|axis\|bar\|span) |
| 101 | 15.216 | 16850 | 905 | seeded_from_verified_log | `compiler_tiny_lut_gather`, `compiler_final_equal_overlay`, `compiler_sparse_scatter`, `compiler_bounded_scan` | tasklog keyword match: \b(row\|column\|horizontal\|vertical\|axis\|bar\|span) |
| 066 | 15.256 | 16899 | 160 | seeded_from_verified_log | `compiler_direct_output_algebra`, `compiler_final_equal_overlay`, `compiler_bounded_scan` | tasklog keyword match: \b(row\|column\|horizontal\|vertical\|axis\|bar\|span) |
| 205 | 15.360 | 14734 | 638 | seeded_from_verified_log | `compiler_single_conv_qlinear`, `compiler_final_equal_overlay`, `compiler_roi_pool_crop`, `compiler_bounded_scan` | tasklog keyword match: \b(row\|column\|horizontal\|vertical\|axis\|bar\|span) |
| 077 | 15.393 | 14760 | 114 | seeded_from_verified_log | `compiler_single_conv_qlinear`, `compiler_qlinear_uint8`, `compiler_bounded_scan` | tasklog keyword match: \b(row\|column\|horizontal\|vertical\|axis\|bar\|span) |
| 025 | 15.448 | 13874 | 195 | seeded_from_verified_log | `compiler_direct_output_algebra`, `compiler_final_equal_overlay`, `compiler_bounded_scan` | tasklog keyword match: \b(row\|column\|horizontal\|vertical\|axis\|bar\|span) |
| 138 | 15.456 | 13789 | 172 | seeded_from_verified_log | `compiler_tiny_lut_gather`, `compiler_single_conv_qlinear`, `compiler_final_equal_overlay`, `compiler_bounded_scan` | tasklog keyword match: \b(row\|column\|horizontal\|vertical\|axis\|bar\|span) |
| 110 | 15.468 | 13155 | 634 | seeded_from_verified_log | `compiler_tiny_lut_gather`, `compiler_single_conv_qlinear`, `compiler_final_equal_overlay`, `compiler_bounded_scan` | tasklog keyword match: \b(row\|column\|horizontal\|vertical\|axis\|bar\|span) |
| 198 | 15.484 | 13434 | 138 | seeded_from_verified_log | `compiler_tiny_lut_gather`, `compiler_single_conv_qlinear`, `compiler_final_equal_overlay`, `compiler_bounded_scan` | tasklog keyword match: \b(row\|column\|horizontal\|vertical\|axis\|bar\|span) |
| 064 | 15.494 | 13368 | 68 | seeded_from_verified_log | `compiler_tiny_lut_gather`, `compiler_final_equal_overlay`, `compiler_bounded_scan` | tasklog keyword match: \b(row\|column\|horizontal\|vertical\|axis\|bar\|span) |
| 145 | 15.511 | 13104 | 111 | seeded_from_verified_log | `compiler_tiny_lut_gather`, `compiler_final_equal_overlay`, `compiler_bounded_scan` | tasklog keyword match: \b(row\|column\|horizontal\|vertical\|axis\|bar\|span) |
| 023 | 15.616 | 11603 | 291 | seeded_from_verified_log | `compiler_single_conv_qlinear`, `compiler_final_equal_overlay`, `compiler_qlinear_uint8` | tasklog keyword match: \b(row\|column\|horizontal\|vertical\|axis\|bar\|span) |
| 396 | 15.633 | 11567 | 133 | seeded_from_verified_log | `compiler_single_conv_qlinear`, `compiler_final_equal_overlay`, `compiler_roi_pool_crop`, `compiler_bounded_scan` | tasklog keyword match: \b(row\|column\|horizontal\|vertical\|axis\|bar\|span) |
| 338 | 15.712 | 10140 | 666 | seeded_from_verified_log | `compiler_tiny_lut_gather`, `compiler_single_conv_qlinear`, `compiler_final_equal_overlay`, `compiler_qlinear_uint8` | tasklog keyword match: \b(row\|column\|horizontal\|vertical\|axis\|bar\|span) |
| 029 | 15.715 | 10736 | 34 | seeded_from_verified_log | `compiler_direct_output_algebra`, `compiler_tiny_lut_gather`, `compiler_final_equal_overlay`, `compiler_bounded_scan` | tasklog keyword match: \b(row\|column\|horizontal\|vertical\|axis\|bar\|span) |
| 080 | 15.726 | 10198 | 454 | seeded_from_verified_log | `compiler_tiny_lut_gather`, `compiler_single_conv_qlinear`, `compiler_final_equal_overlay`, `compiler_qlinear_uint8` | tasklog keyword match: \b(row\|column\|horizontal\|vertical\|axis\|bar\|span) |
| 017 | 15.744 | 8448 | 2021 | seeded_from_verified_log | `compiler_tiny_lut_gather`, `compiler_final_equal_overlay` | tasklog keyword match: \b(row\|column\|horizontal\|vertical\|axis\|bar\|span) |
| 204 | 15.753 | 9920 | 453 | seeded_from_verified_log | `compiler_single_conv_qlinear`, `compiler_qlinear_uint8`, `compiler_bounded_scan` | tasklog keyword match: \b(row\|column\|horizontal\|vertical\|axis\|bar\|span) |
| 379 | 15.818 | 9069 | 653 | seeded_from_verified_log | `compiler_single_conv_qlinear`, `compiler_final_equal_overlay`, `compiler_bounded_scan` | tasklog keyword match: \b(row\|column\|horizontal\|vertical\|axis\|bar\|span) |
| 370 | 15.823 | 8645 | 1024 | seeded_from_verified_log | `compiler_direct_output_algebra`, `compiler_tiny_lut_gather`, `compiler_single_conv_qlinear`, `compiler_final_equal_overlay` | tasklog keyword match: \b(row\|column\|horizontal\|vertical\|axis\|bar\|span) |
| 216 | 15.880 | 9048 | 87 | seeded_from_verified_log | `compiler_direct_onehot_gather`, `compiler_tiny_lut_gather`, `compiler_single_conv_qlinear`, `compiler_final_equal_overlay` | tasklog keyword match: \b(row\|column\|horizontal\|vertical\|axis\|bar\|span) |
| 074 | 15.889 | 9000 | 50 | seeded_from_verified_log | `compiler_tiny_lut_gather`, `compiler_single_conv_qlinear`, `compiler_final_equal_overlay` | tasklog keyword match: \b(row\|column\|horizontal\|vertical\|axis\|bar\|span) |
| 350 | 15.891 | 9012 | 26 | seeded_from_verified_log | `compiler_tiny_lut_gather`, `compiler_bounded_scan` | tasklog keyword match: \b(row\|column\|horizontal\|vertical\|axis\|bar\|span) |
| 096 | 15.905 | 8108 | 805 | seeded_from_verified_log | `compiler_tiny_lut_gather`, `compiler_final_equal_overlay`, `compiler_sparse_scatter`, `compiler_bounded_scan` | tasklog keyword match: \b(row\|column\|horizontal\|vertical\|axis\|bar\|span) |
| 324 | 15.907 | 7308 | 1586 | seeded_from_verified_log | `compiler_single_conv_qlinear`, `compiler_final_equal_overlay`, `compiler_qlinear_uint8`, `compiler_bounded_scan` | tasklog keyword match: \b(row\|column\|horizontal\|vertical\|axis\|bar\|span) |
| 222 | 15.916 | 8736 | 78 | seeded_from_verified_log | `compiler_single_conv_qlinear`, `compiler_final_equal_overlay`, `compiler_qlinear_uint8`, `compiler_bounded_scan` | tasklog keyword match: \b(row\|column\|horizontal\|vertical\|axis\|bar\|span) |
| 192 | 15.938 | 8515 | 106 | seeded_from_verified_log | `compiler_single_conv_qlinear`, `compiler_final_equal_overlay` | tasklog keyword match: \b(row\|column\|horizontal\|vertical\|axis\|bar\|span) |
| 157 | 15.972 | 7983 | 351 | seeded_from_verified_log | `compiler_tiny_lut_gather`, `compiler_final_equal_overlay`, `compiler_sparse_scatter`, `compiler_bounded_scan` | tasklog keyword match: \b(row\|column\|horizontal\|vertical\|axis\|bar\|span) |
| 377 | 15.975 | 8162 | 150 | seeded_from_verified_log | `compiler_tiny_lut_gather`, `compiler_single_conv_qlinear`, `compiler_final_equal_overlay` | tasklog keyword match: \b(row\|column\|horizontal\|vertical\|axis\|bar\|span) |
| 014 | 16.023 | 7809 | 108 | seeded_from_verified_log | `compiler_tiny_lut_gather`, `compiler_final_equal_overlay`, `compiler_bounded_scan` | tasklog keyword match: \b(row\|column\|horizontal\|vertical\|axis\|bar\|span) |
| 202 | 16.039 | 7773 | 24 | seeded_from_verified_log | `compiler_single_conv_qlinear`, `compiler_final_equal_overlay`, `compiler_bounded_scan` | tasklog keyword match: \b(row\|column\|horizontal\|vertical\|axis\|bar\|span) |
| 089 | 16.111 | 7092 | 160 | seeded_from_verified_log | `compiler_tiny_lut_gather`, `compiler_single_conv_qlinear`, `compiler_final_equal_overlay`, `compiler_sparse_scatter` | tasklog keyword match: \b(row\|column\|horizontal\|vertical\|axis\|bar\|span) |
| 174 | 16.130 | 6973 | 142 | seeded_from_verified_log | `compiler_direct_onehot_gather`, `compiler_tiny_lut_gather`, `compiler_final_equal_overlay`, `compiler_bounded_scan` | tasklog keyword match: \b(row\|column\|horizontal\|vertical\|axis\|bar\|span) |
| 009 | 16.143 | 6929 | 95 | seeded_from_verified_log | `compiler_tiny_lut_gather`, `compiler_single_conv_qlinear`, `compiler_final_equal_overlay`, `compiler_bounded_scan` | tasklog keyword match: \b(row\|column\|horizontal\|vertical\|axis\|bar\|span) |
| 092 | 16.148 | 6805 | 182 | seeded_from_verified_log | `compiler_direct_output_algebra`, `compiler_final_equal_overlay`, `compiler_sparse_scatter`, `compiler_bounded_scan` | tasklog keyword match: \b(row\|column\|horizontal\|vertical\|axis\|bar\|span) |
| 005 | 16.163 | 6392 | 490 | seeded_from_verified_log | `compiler_tiny_lut_gather`, `compiler_single_conv_qlinear`, `compiler_final_equal_overlay`, `compiler_qlinear_uint8` | tasklog keyword match: \b(row\|column\|horizontal\|vertical\|axis\|bar\|span) |
| 182 | 16.176 | 6695 | 98 | seeded_from_verified_log | `compiler_tiny_lut_gather`, `compiler_single_conv_qlinear`, `compiler_final_equal_overlay`, `compiler_qlinear_uint8` | tasklog keyword match: \b(row\|column\|horizontal\|vertical\|axis\|bar\|span) |
| 368 | 16.264 | 6022 | 203 | seeded_from_verified_log | `compiler_tiny_lut_gather`, `compiler_final_equal_overlay`, `compiler_sparse_scatter` | tasklog keyword match: \b(row\|column\|horizontal\|vertical\|axis\|bar\|span) |
| 165 | 16.276 | 5836 | 314 | seeded_from_verified_log | `compiler_tiny_lut_gather`, `compiler_single_conv_qlinear`, `compiler_final_equal_overlay`, `compiler_qlinear_uint8` | tasklog keyword match: \b(row\|column\|horizontal\|vertical\|axis\|bar\|span) |
| 340 | 16.285 | 5812 | 279 | seeded_from_verified_log | `compiler_tiny_lut_gather`, `compiler_final_equal_overlay`, `compiler_qlinear_uint8`, `compiler_bounded_scan` | tasklog keyword match: \b(row\|column\|horizontal\|vertical\|axis\|bar\|span) |
| 383 | 16.289 | 5974 | 96 | seeded_from_verified_log | `compiler_tiny_lut_gather`, `compiler_single_conv_qlinear`, `compiler_final_equal_overlay`, `compiler_bounded_scan` | tasklog keyword match: \b(row\|column\|horizontal\|vertical\|axis\|bar\|span) |
| 264 | 16.292 | 5348 | 706 | seeded_from_verified_log | `compiler_direct_output_algebra`, `compiler_tiny_lut_gather`, `compiler_single_conv_qlinear`, `compiler_final_equal_overlay` | tasklog keyword match: \b(row\|column\|horizontal\|vertical\|axis\|bar\|span) |
| 280 | 16.294 | 5286 | 753 | seeded_from_verified_log | `compiler_direct_output_algebra`, `compiler_final_equal_overlay` | tasklog keyword match: \b(row\|column\|horizontal\|vertical\|axis\|bar\|span) |
| 148 | 16.323 | 5759 | 110 | seeded_from_verified_log | `compiler_final_equal_overlay`, `compiler_sparse_scatter`, `compiler_bounded_scan` | tasklog keyword match: \b(row\|column\|horizontal\|vertical\|axis\|bar\|span) |
| 382 | 16.337 | 5648 | 135 | seeded_from_verified_log | `compiler_direct_output_algebra`, `compiler_tiny_lut_gather`, `compiler_final_equal_overlay`, `compiler_bounded_scan` | tasklog keyword match: \b(row\|column\|horizontal\|vertical\|axis\|bar\|span) |
| 279 | 16.378 | 5508 | 47 | seeded_from_tasklog_and_inventory | `compiler_tiny_lut_gather`, `compiler_single_conv_qlinear`, `compiler_qlinear_uint8` | tasklog keyword match: \b(row\|column\|horizontal\|vertical\|axis\|bar\|span) |
| 278 | 16.391 | 5436 | 47 | seeded_from_verified_log | `compiler_single_conv_qlinear`, `compiler_qlinear_uint8`, `compiler_bounded_scan` | tasklog keyword match: \b(row\|column\|horizontal\|vertical\|axis\|bar\|span) |
| 085 | 16.409 | 4500 | 881 | seeded_from_verified_log | `compiler_single_conv_qlinear` | tasklog keyword match: \b(row\|column\|horizontal\|vertical\|axis\|bar\|span) |
| 234 | 16.448 | 5113 | 66 | seeded_from_verified_log | `compiler_direct_output_algebra`, `compiler_tiny_lut_gather`, `compiler_final_equal_overlay`, `compiler_bounded_scan` | tasklog keyword match: \b(row\|column\|horizontal\|vertical\|axis\|bar\|span) |
| 004 | 16.454 | 5044 | 100 | seeded_from_verified_log | `compiler_tiny_lut_gather`, `compiler_single_conv_qlinear`, `compiler_final_equal_overlay`, `compiler_qlinear_uint8` | tasklog keyword match: \b(row\|column\|horizontal\|vertical\|axis\|bar\|span) |
| 387 | 16.471 | 4954 | 105 | seeded_from_verified_log | `compiler_tiny_lut_gather`, `compiler_final_equal_overlay`, `compiler_sparse_scatter` | tasklog keyword match: \b(row\|column\|horizontal\|vertical\|axis\|bar\|span) |
| 251 | 16.499 | 4752 | 168 | seeded_from_verified_log | `compiler_bounded_scan` | tasklog keyword match: \b(row\|column\|horizontal\|vertical\|axis\|bar\|span) |
| 037 | 16.535 | 4614 | 132 | seeded_from_verified_log | `compiler_tiny_lut_gather`, `compiler_final_equal_overlay`, `compiler_sparse_scatter`, `compiler_bounded_scan` | tasklog keyword match: \b(row\|column\|horizontal\|vertical\|axis\|bar\|span) |
| 208 | 16.539 | 4612 | 114 | seeded_from_verified_log | `compiler_direct_output_algebra`, `compiler_single_conv_qlinear`, `compiler_final_equal_overlay`, `compiler_qlinear_uint8` | tasklog keyword match: \b(row\|column\|horizontal\|vertical\|axis\|bar\|span) |
| 361 | 16.547 | 4322 | 366 | seeded_from_verified_log | `compiler_tiny_lut_gather`, `compiler_single_conv_qlinear`, `compiler_final_equal_overlay`, `compiler_roi_pool_crop` | tasklog keyword match: \b(row\|column\|horizontal\|vertical\|axis\|bar\|span) |
| 196 | 16.580 | 4500 | 38 | seeded_from_verified_log | `compiler_single_conv_qlinear`, `compiler_qlinear_uint8`, `compiler_bounded_scan` | tasklog keyword match: \b(row\|column\|horizontal\|vertical\|axis\|bar\|span) |
| 358 | 16.587 | 4332 | 174 | seeded_from_verified_log | `compiler_direct_output_algebra`, `compiler_final_equal_overlay`, `compiler_bounded_scan` | tasklog keyword match: \b(row\|column\|horizontal\|vertical\|axis\|bar\|span) |
| 378 | 16.600 | 4332 | 117 | seeded_from_verified_log | `compiler_direct_output_algebra`, `compiler_tiny_lut_gather`, `compiler_final_equal_overlay`, `compiler_bounded_scan` | tasklog keyword match: \b(row\|column\|horizontal\|vertical\|axis\|bar\|span) |
| 265 | 16.619 | 4328 | 34 | seeded_from_verified_log | `compiler_tiny_lut_gather`, `compiler_final_equal_overlay`, `compiler_sparse_scatter`, `compiler_bounded_scan` | tasklog keyword match: \b(row\|column\|horizontal\|vertical\|axis\|bar\|span) |
| 277 | 16.627 | 4101 | 228 | seeded_from_verified_log | `compiler_tiny_lut_gather`, `compiler_final_equal_overlay`, `compiler_bounded_scan` | tasklog keyword match: \b(row\|column\|horizontal\|vertical\|axis\|bar\|span) |
| 206 | 16.657 | 4122 | 77 | seeded_from_verified_log | `compiler_single_conv_qlinear`, `compiler_final_equal_overlay`, `compiler_sparse_scatter`, `compiler_bounded_scan` | tasklog keyword match: \b(row\|column\|horizontal\|vertical\|axis\|bar\|span) |
| 117 | 16.666 | 3922 | 243 | seeded_from_verified_log | `compiler_tiny_lut_gather`, `compiler_single_conv_qlinear`, `compiler_final_equal_overlay`, `compiler_qlinear_uint8` | tasklog keyword match: \b(row\|column\|horizontal\|vertical\|axis\|bar\|span) |
| 363 | 16.673 | 3839 | 296 | seeded_from_verified_log | `compiler_single_conv_qlinear`, `compiler_final_equal_overlay`, `compiler_bounded_scan` | tasklog keyword match: \b(row\|column\|horizontal\|vertical\|axis\|bar\|span) |
| 132 | 16.684 | 3990 | 99 | seeded_from_verified_log | `compiler_tiny_lut_gather`, `compiler_single_conv_qlinear`, `compiler_final_equal_overlay`, `compiler_bounded_scan` | tasklog keyword match: \b(row\|column\|horizontal\|vertical\|axis\|bar\|span) |
| 161 | 16.684 | 4055 | 33 | seeded_from_verified_log | `compiler_tiny_lut_gather`, `compiler_final_equal_overlay` | tasklog keyword match: \b(row\|column\|horizontal\|vertical\|axis\|bar\|span) |
| 107 | 16.687 | 2924 | 1154 | seeded_from_verified_log | `compiler_direct_onehot_gather`, `compiler_tiny_lut_gather`, `compiler_final_equal_overlay`, `compiler_sparse_scatter` | tasklog keyword match: \b(row\|column\|horizontal\|vertical\|axis\|bar\|span) |
| 162 | 16.689 | 4024 | 44 | seeded_from_verified_log | `compiler_single_conv_qlinear`, `compiler_qlinear_uint8`, `compiler_bounded_scan` | tasklog keyword match: \b(row\|column\|horizontal\|vertical\|axis\|bar\|span) |
| 365 | 16.699 | 3908 | 120 | seeded_from_verified_log | `compiler_direct_output_algebra`, `compiler_tiny_lut_gather`, `compiler_single_conv_qlinear`, `compiler_final_equal_overlay` | tasklog keyword match: \b(row\|column\|horizontal\|vertical\|axis\|bar\|span) |
| 131 | 16.709 | 3312 | 675 | seeded_from_verified_log | `compiler_sparse_scatter`, `compiler_bounded_scan` | tasklog keyword match: \b(row\|column\|horizontal\|vertical\|axis\|bar\|span) |
| 069 | 16.728 | 3848 | 64 | seeded_from_verified_log | `compiler_tiny_lut_gather`, `compiler_single_conv_qlinear`, `compiler_final_equal_overlay`, `compiler_bounded_scan` | tasklog keyword match: \b(row\|column\|horizontal\|vertical\|axis\|bar\|span) |
| 050 | 16.728 | 3825 | 86 | seeded_from_verified_log | `compiler_direct_onehot_gather` | tasklog keyword match: \b(row\|column\|horizontal\|vertical\|axis\|bar\|span) |
| 359 | 16.740 | 3852 | 15 | seeded_from_verified_log | `compiler_final_equal_overlay`, `compiler_bounded_scan` | tasklog keyword match: \b(row\|column\|horizontal\|vertical\|axis\|bar\|span) |
| 177 | 16.751 | 3692 | 130 | seeded_from_verified_log | `compiler_single_conv_qlinear`, `compiler_final_equal_overlay`, `compiler_roi_pool_crop` | tasklog keyword match: \b(row\|column\|horizontal\|vertical\|axis\|bar\|span) |
| 284 | 16.774 | 3082 | 653 | seeded_from_verified_log | `compiler_single_conv_qlinear`, `compiler_sparse_scatter` | tasklog keyword match: \b(row\|column\|horizontal\|vertical\|axis\|bar\|span) |
| 268 | 16.816 | 3328 | 256 | seeded_from_verified_log | `compiler_final_equal_overlay` | tasklog keyword match: \b(row\|column\|horizontal\|vertical\|axis\|bar\|span) |
| 093 | 16.823 | 3456 | 103 | seeded_from_verified_log | `compiler_tiny_lut_gather`, `compiler_final_equal_overlay`, `compiler_qlinear_uint8`, `compiler_bounded_scan` | tasklog keyword match: \b(row\|column\|horizontal\|vertical\|axis\|bar\|span) |
| 137 | 16.832 | 3433 | 93 | seeded_from_verified_log | `compiler_direct_output_algebra`, `compiler_final_equal_overlay` | tasklog keyword match: \b(row\|column\|horizontal\|vertical\|axis\|bar\|span) |
| 125 | 16.850 | 3434 | 29 | seeded_from_verified_log | `compiler_final_equal_overlay`, `compiler_bounded_scan` | tasklog keyword match: \b(row\|column\|horizontal\|vertical\|axis\|bar\|span) |
| 201 | 16.881 | 3202 | 154 | seeded_from_verified_log | `compiler_tiny_lut_gather`, `compiler_single_conv_qlinear`, `compiler_final_equal_overlay`, `compiler_bounded_scan` | tasklog keyword match: \b(row\|column\|horizontal\|vertical\|axis\|bar\|span) |
| 310 | 16.902 | 3208 | 79 | seeded_from_verified_log | `compiler_direct_output_algebra`, `compiler_tiny_lut_gather`, `compiler_final_equal_overlay` | tasklog keyword match: \b(row\|column\|horizontal\|vertical\|axis\|bar\|span) |
| 333 | 16.921 | 2792 | 435 | seeded_from_verified_log | `compiler_direct_output_algebra`, `compiler_tiny_lut_gather`, `compiler_final_equal_overlay`, `compiler_bounded_scan` | tasklog keyword match: \b(row\|column\|horizontal\|vertical\|axis\|bar\|span) |
| 250 | 16.921 | 3164 | 61 | seeded_from_verified_log | `compiler_tiny_lut_gather`, `compiler_final_equal_overlay`, `compiler_qlinear_uint8`, `compiler_bounded_scan` | tasklog keyword match: \b(row\|column\|horizontal\|vertical\|axis\|bar\|span) |
| 079 | 16.947 | 3065 | 78 | seeded_from_verified_log | `compiler_single_conv_qlinear`, `compiler_final_equal_overlay`, `compiler_bounded_scan` | tasklog keyword match: \b(row\|column\|horizontal\|vertical\|axis\|bar\|span) |
| 086 | 16.949 | 2946 | 190 | seeded_from_verified_log | `compiler_tiny_lut_gather`, `compiler_single_conv_qlinear`, `compiler_final_equal_overlay`, `compiler_qlinear_uint8` | tasklog keyword match: \b(row\|column\|horizontal\|vertical\|axis\|bar\|span) |
| 035 | 16.954 | 2994 | 128 | seeded_from_verified_log | `compiler_sparse_scatter` | tasklog keyword match: \b(row\|column\|horizontal\|vertical\|axis\|bar\|span) |
| 055 | 16.963 | 3030 | 62 | seeded_from_verified_log | `compiler_tiny_lut_gather`, `compiler_final_equal_overlay`, `compiler_qlinear_uint8`, `compiler_bounded_scan` | tasklog keyword match: \b(row\|column\|horizontal\|vertical\|axis\|bar\|span) |
| 090 | 16.964 | 3009 | 82 | seeded_from_verified_log | `compiler_bounded_scan` | tasklog keyword match: \b(row\|column\|horizontal\|vertical\|axis\|bar\|span) |
| 390 | 16.964 | 2624 | 467 | seeded_from_verified_log | `compiler_final_equal_overlay`, `compiler_sparse_scatter` | tasklog keyword match: \b(row\|column\|horizontal\|vertical\|axis\|bar\|span) |
| 270 | 16.971 | 2742 | 327 | seeded_from_verified_log | `compiler_direct_output_algebra`, `compiler_bounded_scan` | tasklog keyword match: \b(row\|column\|horizontal\|vertical\|axis\|bar\|span) |
| 091 | 16.984 | 2853 | 175 | seeded_from_verified_log | `compiler_tiny_lut_gather`, `compiler_single_conv_qlinear`, `compiler_final_equal_overlay` | tasklog keyword match: \b(row\|column\|horizontal\|vertical\|axis\|bar\|span) |
| 070 | 16.991 | 2977 | 31 | seeded_from_verified_log | `compiler_tiny_lut_gather`, `compiler_sparse_scatter`, `compiler_bounded_scan` | tasklog keyword match: \b(row\|column\|horizontal\|vertical\|axis\|bar\|span) |
| 335 | 17.001 | 2288 | 691 | seeded_from_verified_log | `compiler_direct_output_algebra`, `compiler_single_conv_qlinear`, `compiler_final_equal_overlay`, `compiler_bounded_scan` | tasklog keyword match: \b(row\|column\|horizontal\|vertical\|axis\|bar\|span) |
| 105 | 17.007 | 2854 | 106 | seeded_from_verified_log | `compiler_direct_onehot_gather`, `compiler_tiny_lut_gather`, `compiler_final_equal_overlay`, `compiler_bounded_scan` | tasklog keyword match: \b(row\|column\|horizontal\|vertical\|axis\|bar\|span) |
| 256 | 17.011 | 2898 | 50 | seeded_from_tasklog_and_inventory | `compiler_final_equal_overlay`, `compiler_bounded_scan` | tasklog keyword match: \b(row\|column\|horizontal\|vertical\|axis\|bar\|span) |
| 398 | 17.042 | 1666 | 1193 | seeded_from_verified_log | `compiler_direct_onehot_gather`, `compiler_tiny_lut_gather`, `compiler_final_equal_overlay` | tasklog keyword match: \b(row\|column\|horizontal\|vertical\|axis\|bar\|span) |
| 224 | 17.052 | 2460 | 371 | seeded_from_verified_log | `compiler_single_conv_qlinear`, `compiler_bounded_scan` | tasklog keyword match: \b(row\|column\|horizontal\|vertical\|axis\|bar\|span) |
| 394 | 17.063 | 1871 | 929 | seeded_from_verified_log | `compiler_tiny_lut_gather`, `compiler_final_equal_overlay`, `compiler_bounded_scan` | tasklog keyword match: \b(row\|column\|horizontal\|vertical\|axis\|bar\|span) |
| 154 | 17.077 | 2661 | 99 | seeded_from_verified_log | `compiler_tiny_lut_gather`, `compiler_final_equal_overlay` | tasklog keyword match: \b(row\|column\|horizontal\|vertical\|axis\|bar\|span) |
| 246 | 17.080 | 2334 | 419 | seeded_from_verified_log | `compiler_single_conv_qlinear`, `compiler_final_equal_overlay` | tasklog keyword match: \b(row\|column\|horizontal\|vertical\|axis\|bar\|span) |
| 354 | 17.080 | 2674 | 77 | seeded_from_verified_log | `compiler_single_conv_qlinear`, `compiler_final_equal_overlay`, `compiler_bounded_scan` | tasklog keyword match: \b(row\|column\|horizontal\|vertical\|axis\|bar\|span) |
| 245 | 17.083 | 2604 | 139 | seeded_from_verified_log | `compiler_tiny_lut_gather`, `compiler_bounded_scan` | tasklog keyword match: \b(row\|column\|horizontal\|vertical\|axis\|bar\|span) |
| 190 | 17.089 | 2480 | 247 | seeded_from_verified_log | `compiler_single_conv_qlinear`, `compiler_qlinear_uint8`, `compiler_bounded_scan` | tasklog keyword match: \b(row\|column\|horizontal\|vertical\|axis\|bar\|span) |
| 355 | 17.093 | 2688 | 27 | seeded_from_verified_log | `compiler_final_equal_overlay`, `compiler_bounded_scan` | tasklog keyword match: \b(row\|column\|horizontal\|vertical\|axis\|bar\|span) |
| 094 | 17.094 | 2662 | 51 | seeded_from_verified_log | `compiler_single_conv_qlinear` | tasklog keyword match: \b(row\|column\|horizontal\|vertical\|axis\|bar\|span) |
| 071 | 17.100 | 2643 | 55 | seeded_from_verified_log | `compiler_tiny_lut_gather`, `compiler_final_equal_overlay`, `compiler_bounded_scan` | tasklog keyword match: \b(row\|column\|horizontal\|vertical\|axis\|bar\|span) |
| 197 | 17.136 | 2586 | 16 | seeded_from_tasklog_and_inventory | `compiler_direct_output_algebra`, `compiler_tiny_lut_gather`, `compiler_single_conv_qlinear`, `compiler_qlinear_uint8` | tasklog keyword match: \b(row\|column\|horizontal\|vertical\|axis\|bar\|span) |
| 170 | 17.158 | 2216 | 329 | seeded_from_verified_log | `compiler_tiny_lut_gather`, `compiler_single_conv_qlinear`, `compiler_final_equal_overlay`, `compiler_bounded_scan` | tasklog keyword match: \b(row\|column\|horizontal\|vertical\|axis\|bar\|span) |
| 062 | 17.162 | 2465 | 71 | seeded_from_verified_log | `compiler_tiny_lut_gather`, `compiler_final_equal_overlay`, `compiler_bounded_scan` | tasklog keyword match: \b(row\|column\|horizontal\|vertical\|axis\|bar\|span) |
| 102 | 17.165 | 2414 | 113 | seeded_from_verified_log | `compiler_single_conv_qlinear`, `compiler_qlinear_uint8`, `compiler_bounded_scan` | tasklog keyword match: \b(row\|column\|horizontal\|vertical\|axis\|bar\|span) |
| 184 | 17.168 | 2460 | 60 | seeded_from_verified_log | `compiler_direct_output_algebra`, `compiler_single_conv_qlinear`, `compiler_final_equal_overlay`, `compiler_bounded_scan` | tasklog keyword match: \b(row\|column\|horizontal\|vertical\|axis\|bar\|span) |
| 388 | 17.178 | 2278 | 218 | seeded_from_verified_log | `compiler_final_equal_overlay`, `compiler_sparse_scatter`, `compiler_bounded_scan` | tasklog keyword match: \b(row\|column\|horizontal\|vertical\|axis\|bar\|span) |
| 238 | 17.178 | 2051 | 444 | seeded_from_verified_log | `compiler_direct_output_algebra`, `compiler_tiny_lut_gather`, `compiler_final_equal_overlay`, `compiler_bounded_scan` | tasklog keyword match: \b(row\|column\|horizontal\|vertical\|axis\|bar\|span) |
| 030 | 17.218 | 2248 | 148 | seeded_from_verified_log | `compiler_tiny_lut_gather`, `compiler_bounded_scan` | tasklog keyword match: \b(row\|column\|horizontal\|vertical\|axis\|bar\|span) |
| 281 | 17.223 | 2321 | 65 | seeded_from_verified_log | `compiler_direct_output_algebra`, `compiler_final_equal_overlay`, `compiler_bounded_scan` | tasklog keyword match: \b(row\|column\|horizontal\|vertical\|axis\|bar\|span) |
| 034 | 17.231 | 2198 | 167 | seeded_from_verified_log | `compiler_tiny_lut_gather`, `compiler_single_conv_qlinear`, `compiler_qlinear_uint8`, `compiler_bounded_scan` | tasklog keyword match: \b(row\|column\|horizontal\|vertical\|axis\|bar\|span) |
| 212 | 17.234 | 2178 | 182 | seeded_from_verified_log | `compiler_single_conv_qlinear`, `compiler_qlinear_uint8`, `compiler_bounded_scan` | tasklog keyword match: \b(row\|column\|horizontal\|vertical\|axis\|bar\|span) |
| 163 | 17.240 | 2235 | 110 | seeded_from_verified_log | `compiler_tiny_lut_gather`, `compiler_single_conv_qlinear`, `compiler_final_equal_overlay`, `compiler_bounded_scan` | tasklog keyword match: \b(row\|column\|horizontal\|vertical\|axis\|bar\|span) |
| 046 | 17.252 | 2221 | 97 | seeded_from_tasklog_and_inventory | `compiler_tiny_lut_gather`, `compiler_single_conv_qlinear`, `compiler_final_equal_overlay`, `compiler_bounded_scan` | tasklog keyword match: \b(row\|column\|horizontal\|vertical\|axis\|bar\|span) |
| 374 | 17.258 | 2262 | 42 | seeded_from_verified_log | `compiler_final_equal_overlay`, `compiler_bounded_scan` | tasklog keyword match: \b(row\|column\|horizontal\|vertical\|axis\|bar\|span) |
| 134 | 17.266 | 2250 | 34 | seeded_from_verified_log | `compiler_tiny_lut_gather`, `compiler_bounded_scan` | tasklog keyword match: \b(row\|column\|horizontal\|vertical\|axis\|bar\|span) |
| 012 | 17.275 | 2076 | 188 | seeded_from_verified_log | `compiler_tiny_lut_gather`, `compiler_final_equal_overlay`, `compiler_sparse_scatter` | tasklog keyword match: \b(row\|column\|horizontal\|vertical\|axis\|bar\|span) |
| 168 | 17.286 | 2096 | 143 | seeded_from_verified_log | `compiler_tiny_lut_gather`, `compiler_single_conv_qlinear`, `compiler_qlinear_uint8`, `compiler_bounded_scan` | tasklog keyword match: \b(row\|column\|horizontal\|vertical\|axis\|bar\|span) |
| 336 | 17.298 | 1188 | 1024 | seeded_from_verified_log | `compiler_tiny_lut_gather`, `compiler_sparse_scatter` | tasklog keyword match: \b(row\|column\|horizontal\|vertical\|axis\|bar\|span) |
| 036 | 17.314 | 2140 | 37 | seeded_from_verified_log | `compiler_final_equal_overlay`, `compiler_bounded_scan` | tasklog keyword match: \b(row\|column\|horizontal\|vertical\|axis\|bar\|span) |
| 323 | 17.314 | 1521 | 656 | seeded_from_verified_log | `compiler_single_conv_qlinear`, `compiler_qlinear_uint8` | tasklog keyword match: \b(row\|column\|horizontal\|vertical\|axis\|bar\|span) |
| 273 | 17.322 | 2109 | 51 | seeded_from_verified_log | `compiler_tiny_lut_gather`, `compiler_bounded_scan` | tasklog keyword match: \b(row\|column\|horizontal\|vertical\|axis\|bar\|span) |
| 260 | 17.344 | 1804 | 310 | seeded_from_verified_log | `compiler_tiny_lut_gather`, `compiler_final_equal_overlay`, `compiler_sparse_scatter`, `compiler_bounded_scan` | tasklog keyword match: \b(row\|column\|horizontal\|vertical\|axis\|bar\|span) |
| 348 | 17.353 | 1938 | 157 | seeded_from_verified_log | `compiler_final_equal_overlay` | tasklog keyword match: \b(row\|column\|horizontal\|vertical\|axis\|bar\|span) |
| 063 | 17.394 | 1728 | 283 | seeded_from_verified_log | `compiler_single_conv_qlinear`, `compiler_final_equal_overlay` | tasklog keyword match: \b(row\|column\|horizontal\|vertical\|axis\|bar\|span) |
| 275 | 17.394 | 1374 | 637 | seeded_from_verified_log | `compiler_direct_output_algebra`, `compiler_single_conv_qlinear`, `compiler_final_equal_overlay`, `compiler_bounded_scan` | tasklog keyword match: \b(row\|column\|horizontal\|vertical\|axis\|bar\|span) |
| 011 | 17.400 | 1836 | 163 | seeded_from_verified_log | `compiler_tiny_lut_gather`, `compiler_single_conv_qlinear`, `compiler_final_equal_overlay` | tasklog keyword match: \b(row\|column\|horizontal\|vertical\|axis\|bar\|span) |
| 287 | 17.402 | 1699 | 295 | seeded_from_verified_log | `compiler_direct_output_algebra`, `compiler_tiny_lut_gather`, `compiler_final_equal_overlay` | tasklog keyword match: \b(row\|column\|horizontal\|vertical\|axis\|bar\|span) |
| 381 | 17.418 | 1908 | 55 | seeded_from_verified_log | `compiler_direct_onehot_gather` | tasklog keyword match: \b(row\|column\|horizontal\|vertical\|axis\|bar\|span) |
| 185 | 17.419 | 1651 | 310 | seeded_from_verified_log | `compiler_tiny_lut_gather`, `compiler_single_conv_qlinear`, `compiler_final_equal_overlay`, `compiler_bounded_scan` | tasklog keyword match: \b(row\|column\|horizontal\|vertical\|axis\|bar\|span) |
| 124 | 17.423 | 1879 | 74 | seeded_from_verified_log | `compiler_direct_onehot_gather`, `compiler_tiny_lut_gather`, `compiler_final_equal_overlay`, `compiler_bounded_scan` | tasklog keyword match: \b(row\|column\|horizontal\|vertical\|axis\|bar\|span) |
| 392 | 17.424 | 1602 | 349 | seeded_from_verified_log | `compiler_direct_output_algebra`, `compiler_tiny_lut_gather`, `compiler_final_equal_overlay`, `compiler_bounded_scan` | tasklog keyword match: \b(row\|column\|horizontal\|vertical\|axis\|bar\|span) |
| 013 | 17.435 | 1869 | 61 | seeded_from_verified_log | `compiler_direct_output_algebra`, `compiler_tiny_lut_gather`, `compiler_final_equal_overlay` | tasklog keyword match: \b(row\|column\|horizontal\|vertical\|axis\|bar\|span) |
| 084 | 17.440 | 1837 | 83 | seeded_from_verified_log | `compiler_sparse_scatter` | tasklog keyword match: \b(row\|column\|horizontal\|vertical\|axis\|bar\|span) |
| 237 | 17.441 | 1778 | 140 | seeded_from_verified_log | `compiler_single_conv_qlinear`, `compiler_final_equal_overlay`, `compiler_bounded_scan` | tasklog keyword match: \b(row\|column\|horizontal\|vertical\|axis\|bar\|span) |
| 099 | 17.445 | 1633 | 278 | seeded_from_verified_log | `compiler_tiny_lut_gather`, `compiler_final_equal_overlay`, `compiler_sparse_scatter`, `compiler_bounded_scan` | tasklog keyword match: \b(row\|column\|horizontal\|vertical\|axis\|bar\|span) |
| 213 | 17.457 | 1838 | 49 | seeded_from_verified_log | `compiler_direct_output_algebra`, `compiler_final_equal_overlay` | tasklog keyword match: \b(row\|column\|horizontal\|vertical\|axis\|bar\|span) |
| 088 | 17.465 | 1791 | 81 | seeded_from_verified_log | `compiler_direct_output_algebra`, `compiler_final_equal_overlay` | tasklog keyword match: \b(row\|column\|horizontal\|vertical\|axis\|bar\|span) |
| 400 | 17.468 | 1800 | 66 | seeded_from_verified_log | `compiler_tiny_lut_gather`, `compiler_roi_pool_crop`, `compiler_bounded_scan` | tasklog keyword match: \b(row\|column\|horizontal\|vertical\|axis\|bar\|span) |
| 199 | 17.484 | 1749 | 88 | seeded_from_verified_log | `compiler_direct_output_algebra`, `compiler_final_equal_overlay`, `compiler_sparse_scatter` | tasklog keyword match: \b(row\|column\|horizontal\|vertical\|axis\|bar\|span) |
| 051 | 17.503 | 1702 | 101 | seeded_from_verified_log | `compiler_direct_output_algebra`, `compiler_final_equal_overlay` | tasklog keyword match: \b(row\|column\|horizontal\|vertical\|axis\|bar\|span) |
| 041 | 17.511 | 1730 | 59 | seeded_from_verified_log | `compiler_single_conv_qlinear`, `compiler_final_equal_overlay` | tasklog keyword match: \b(row\|column\|horizontal\|vertical\|axis\|bar\|span) |
| 109 | 17.516 | 1686 | 93 | seeded_from_verified_log | `compiler_tiny_lut_gather`, `compiler_final_equal_overlay` | tasklog keyword match: \b(row\|column\|horizontal\|vertical\|axis\|bar\|span) |
| 112 | 17.519 | 1616 | 158 | seeded_from_verified_log | `compiler_direct_output_algebra`, `compiler_tiny_lut_gather` | tasklog keyword match: \b(row\|column\|horizontal\|vertical\|axis\|bar\|span) |
| 302 | 17.519 | 1216 | 558 | seeded_from_verified_log | `compiler_single_conv_qlinear`, `compiler_qlinear_uint8` | tasklog keyword match: \b(row\|column\|horizontal\|vertical\|axis\|bar\|span) |
| 156 | 17.536 | 1697 | 47 | seeded_from_tasklog_and_inventory | `compiler_direct_output_algebra`, `compiler_tiny_lut_gather`, `compiler_final_equal_overlay` | tasklog keyword match: \b(row\|column\|horizontal\|vertical\|axis\|bar\|span) |
| 045 | 17.578 | 1050 | 623 | seeded_from_verified_log | `compiler_direct_output_algebra`, `compiler_bounded_scan` | tasklog keyword match: \b(row\|column\|horizontal\|vertical\|axis\|bar\|span) |
| 061 | 17.581 | 1633 | 35 | seeded_from_verified_log | `compiler_tiny_lut_gather`, `compiler_final_equal_overlay`, `compiler_qlinear_uint8`, `compiler_bounded_scan` | tasklog keyword match: \b(row\|column\|horizontal\|vertical\|axis\|bar\|span) |
| 351 | 17.591 | 1632 | 19 | seeded_from_verified_log | `compiler_bounded_scan` | tasklog keyword match: \b(row\|column\|horizontal\|vertical\|axis\|bar\|span) |
| 033 | 17.598 | 1606 | 33 | seeded_from_verified_log |  | tasklog keyword match: \b(row\|column\|horizontal\|vertical\|axis\|bar\|span) |
| 226 | 17.602 | 1613 | 20 | seeded_from_verified_log | `compiler_final_equal_overlay` | tasklog keyword match: \b(row\|column\|horizontal\|vertical\|axis\|bar\|span) |
| 303 | 17.610 | 1598 | 22 | seeded_from_verified_log | `compiler_direct_output_algebra`, `compiler_final_equal_overlay` | tasklog keyword match: \b(row\|column\|horizontal\|vertical\|axis\|bar\|span) |
| 342 | 17.632 | 1510 | 75 | seeded_from_verified_log | `compiler_direct_output_algebra`, `compiler_tiny_lut_gather`, `compiler_final_equal_overlay`, `compiler_sparse_scatter` | tasklog keyword match: \b(row\|column\|horizontal\|vertical\|axis\|bar\|span) |
| 159 | 17.642 | 1419 | 149 | seeded_from_verified_log | `compiler_direct_output_algebra`, `compiler_tiny_lut_gather`, `compiler_final_equal_overlay` | tasklog keyword match: \b(row\|column\|horizontal\|vertical\|axis\|bar\|span) |
| 020 | 17.670 | 1356 | 170 | seeded_from_verified_log | `compiler_tiny_lut_gather`, `compiler_single_conv_qlinear`, `compiler_sparse_scatter`, `compiler_bounded_scan` | tasklog keyword match: \b(row\|column\|horizontal\|vertical\|axis\|bar\|span) |
| 141 | 17.683 | 1404 | 101 | seeded_from_verified_log | `compiler_direct_output_algebra`, `compiler_final_equal_overlay` | tasklog keyword match: \b(row\|column\|horizontal\|vertical\|axis\|bar\|span) |
| 345 | 17.690 | 1431 | 64 | seeded_from_verified_log | `compiler_tiny_lut_gather` | tasklog keyword match: \b(row\|column\|horizontal\|vertical\|axis\|bar\|span) |
| 308 | 17.695 | 1385 | 103 | seeded_from_verified_log | `compiler_direct_output_algebra`, `compiler_final_equal_overlay`, `compiler_sparse_scatter`, `compiler_bounded_scan` | tasklog keyword match: \b(row\|column\|horizontal\|vertical\|axis\|bar\|span) |
| 075 | 17.695 | 1326 | 161 | seeded_from_verified_log | `compiler_direct_output_algebra`, `compiler_final_equal_overlay` | tasklog keyword match: \b(row\|column\|horizontal\|vertical\|axis\|bar\|span) |
| 031 | 17.700 | 1434 | 46 | seeded_from_verified_log | `compiler_bounded_scan` | tasklog keyword match: \b(row\|column\|horizontal\|vertical\|axis\|bar\|span) |
| 240 | 17.704 | 1393 | 82 | seeded_from_verified_log | `compiler_tiny_lut_gather`, `compiler_final_equal_overlay`, `compiler_bounded_scan` | tasklog keyword match: \b(row\|column\|horizontal\|vertical\|axis\|bar\|span) |
| 341 | 17.735 | 1394 | 35 | seeded_from_verified_log | `compiler_final_equal_overlay`, `compiler_bounded_scan` | tasklog keyword match: \b(row\|column\|horizontal\|vertical\|axis\|bar\|span) |
| 232 | 17.749 | 0 | 1410 | seeded_from_tasklog_and_inventory | `compiler_tiny_lut_gather`, `compiler_single_conv_qlinear` | tasklog keyword match: \b(row\|column\|horizontal\|vertical\|axis\|bar\|span) |
| 119 | 17.781 | 1244 | 121 | seeded_from_verified_log | `compiler_tiny_lut_gather`, `compiler_sparse_scatter`, `compiler_bounded_scan` | tasklog keyword match: \b(row\|column\|horizontal\|vertical\|axis\|bar\|span) |
| 123 | 17.797 | 403 | 940 | seeded_from_verified_log | `compiler_direct_output_algebra`, `compiler_tiny_lut_gather`, `compiler_final_equal_overlay` | tasklog keyword match: \b(row\|column\|horizontal\|vertical\|axis\|bar\|span) |
| 356 | 17.815 | 1300 | 19 | seeded_from_verified_log | `compiler_single_conv_qlinear`, `compiler_qlinear_uint8`, `compiler_bounded_scan` | tasklog keyword match: \b(row\|column\|horizontal\|vertical\|axis\|bar\|span) |
| 027 | 17.822 | 1262 | 48 | seeded_from_verified_log |  | tasklog keyword match: \b(row\|column\|horizontal\|vertical\|axis\|bar\|span) |
| 078 | 17.835 | 1260 | 33 | seeded_from_tasklog_and_inventory |  | tasklog keyword match: \b(row\|column\|horizontal\|vertical\|axis\|bar\|span) |
| 293 | 17.839 | 1247 | 41 | seeded_from_verified_log | `compiler_tiny_lut_gather`, `compiler_final_equal_overlay`, `compiler_sparse_scatter` | tasklog keyword match: \b(row\|column\|horizontal\|vertical\|axis\|bar\|span) |
| 175 | 17.856 | 330 | 937 | seeded_from_verified_log | `compiler_tiny_lut_gather`, `compiler_final_equal_overlay`, `compiler_bounded_scan` | tasklog keyword match: \b(row\|column\|horizontal\|vertical\|axis\|bar\|span) |
| 346 | 17.866 | 1224 | 30 | seeded_from_verified_log | `compiler_tiny_lut_gather`, `compiler_single_conv_qlinear`, `compiler_final_equal_overlay`, `compiler_qlinear_uint8` | tasklog keyword match: \b(row\|column\|horizontal\|vertical\|axis\|bar\|span) |
| 183 | 17.885 | 1008 | 222 | seeded_from_verified_log | `compiler_direct_output_algebra`, `compiler_final_equal_overlay` | tasklog keyword match: \b(row\|column\|horizontal\|vertical\|axis\|bar\|span) |
| 343 | 17.892 | 1147 | 75 | seeded_from_verified_log | `compiler_tiny_lut_gather`, `compiler_single_conv_qlinear`, `compiler_final_equal_overlay`, `compiler_bounded_scan` | tasklog keyword match: \b(row\|column\|horizontal\|vertical\|axis\|bar\|span) |
| 136 | 17.899 | 1179 | 34 | seeded_from_verified_log | `compiler_tiny_lut_gather`, `compiler_final_equal_overlay`, `compiler_bounded_scan` | tasklog keyword match: \b(row\|column\|horizontal\|vertical\|axis\|bar\|span) |
| 271 | 17.921 | 1141 | 46 | seeded_from_verified_log | `compiler_single_conv_qlinear`, `compiler_qlinear_uint8` | tasklog keyword match: \b(row\|column\|horizontal\|vertical\|axis\|bar\|span) |
| 188 | 17.921 | 1128 | 59 | seeded_from_verified_log | `compiler_single_conv_qlinear`, `compiler_final_equal_overlay`, `compiler_qlinear_uint8`, `compiler_bounded_scan` | tasklog keyword match: \b(row\|column\|horizontal\|vertical\|axis\|bar\|span) |
| 244 | 17.928 | 1076 | 103 | seeded_from_verified_log | `compiler_tiny_lut_gather`, `compiler_final_equal_overlay`, `compiler_roi_pool_crop`, `compiler_bounded_scan` | tasklog keyword match: \b(row\|column\|horizontal\|vertical\|axis\|bar\|span) |
| 021 | 17.935 | 1072 | 98 | seeded_from_tasklog_and_inventory | `compiler_direct_output_algebra` | tasklog keyword match: \b(row\|column\|horizontal\|vertical\|axis\|bar\|span) |
| 048 | 17.956 | 1030 | 116 | seeded_from_verified_log | `compiler_single_conv_qlinear`, `compiler_final_equal_overlay`, `compiler_bounded_scan` | tasklog keyword match: \b(row\|column\|horizontal\|vertical\|axis\|bar\|span) |
| 301 | 17.960 | 1077 | 64 | seeded_from_verified_log | `compiler_tiny_lut_gather`, `compiler_final_equal_overlay`, `compiler_bounded_scan` | tasklog keyword match: \b(row\|column\|horizontal\|vertical\|axis\|bar\|span) |
| 254 | 17.999 | 1079 | 19 | seeded_from_verified_log | `compiler_final_equal_overlay`, `compiler_bounded_scan` | tasklog keyword match: \b(row\|column\|horizontal\|vertical\|axis\|bar\|span) |
| 010 | 18.025 | 730 | 340 | seeded_from_verified_log | `compiler_tiny_lut_gather`, `compiler_final_equal_overlay` | tasklog keyword match: \b(row\|column\|horizontal\|vertical\|axis\|bar\|span) |
| 329 | 18.035 | 1029 | 30 | seeded_from_verified_log | `compiler_final_equal_overlay`, `compiler_sparse_scatter` | tasklog keyword match: \b(row\|column\|horizontal\|vertical\|axis\|bar\|span) |
| 115 | 18.050 | 980 | 63 | seeded_from_verified_log | `compiler_direct_output_algebra`, `compiler_bounded_scan` | tasklog keyword match: \b(row\|column\|horizontal\|vertical\|axis\|bar\|span) |
| 225 | 18.060 | 926 | 107 | seeded_from_verified_log | `compiler_tiny_lut_gather`, `compiler_single_conv_qlinear`, `compiler_final_equal_overlay`, `compiler_bounded_scan` | tasklog keyword match: \b(row\|column\|horizontal\|vertical\|axis\|bar\|span) |
| 058 | 18.060 | 252 | 781 | seeded_from_verified_log | `compiler_tiny_lut_gather` | tasklog keyword match: \b(row\|column\|horizontal\|vertical\|axis\|bar\|span) |
| 060 | 18.082 | 0 | 1010 | seeded_from_tasklog_and_inventory | `compiler_single_conv_qlinear` | tasklog keyword match: \b(row\|column\|horizontal\|vertical\|axis\|bar\|span) |
| 108 | 18.122 | 864 | 107 | seeded_from_verified_log | `compiler_single_conv_qlinear`, `compiler_final_equal_overlay`, `compiler_roi_pool_crop` | tasklog keyword match: \b(row\|column\|horizontal\|vertical\|axis\|bar\|span) |
| 228 | 18.131 | 849 | 113 | seeded_from_verified_log | `compiler_direct_output_algebra`, `compiler_tiny_lut_gather`, `compiler_final_equal_overlay`, `compiler_sparse_scatter` | tasklog keyword match: \b(row\|column\|horizontal\|vertical\|axis\|bar\|span) |
| 288 | 18.135 | 769 | 189 | seeded_from_verified_log | `compiler_tiny_lut_gather`, `compiler_final_equal_overlay`, `compiler_sparse_scatter`, `compiler_bounded_scan` | tasklog keyword match: \b(row\|column\|horizontal\|vertical\|axis\|bar\|span) |
| 194 | 18.145 | 900 | 49 | seeded_from_verified_log | `compiler_tiny_lut_gather` | tasklog keyword match: \b(row\|column\|horizontal\|vertical\|axis\|bar\|span) |
| 195 | 18.175 | 882 | 39 | seeded_from_verified_log | `compiler_tiny_lut_gather`, `compiler_bounded_scan` | tasklog keyword match: \b(row\|column\|horizontal\|vertical\|axis\|bar\|span) |
| 189 | 18.181 | 866 | 49 | seeded_from_verified_log | `compiler_single_conv_qlinear`, `compiler_final_equal_overlay`, `compiler_roi_pool_crop` | tasklog keyword match: \b(row\|column\|horizontal\|vertical\|axis\|bar\|span) |
| 032 | 18.187 | 0 | 910 | seeded_from_verified_log | `compiler_single_conv_qlinear` | tasklog keyword match: \b(row\|column\|horizontal\|vertical\|axis\|bar\|span) |
| 120 | 18.187 | 0 | 910 | seeded_from_verified_log | `compiler_single_conv_qlinear` | tasklog keyword match: \b(row\|column\|horizontal\|vertical\|axis\|bar\|span) |
| 283 | 18.187 | 0 | 910 | seeded_from_verified_log | `compiler_single_conv_qlinear` | tasklog keyword match: \b(row\|column\|horizontal\|vertical\|axis\|bar\|span) |
| 331 | 18.187 | 0 | 910 | seeded_from_verified_log | `compiler_single_conv_qlinear` | tasklog keyword match: \b(row\|column\|horizontal\|vertical\|axis\|bar\|span) |
| 259 | 18.192 | 853 | 52 | seeded_from_verified_log | `compiler_tiny_lut_gather`, `compiler_single_conv_qlinear`, `compiler_final_equal_overlay`, `compiler_bounded_scan` | tasklog keyword match: \b(row\|column\|horizontal\|vertical\|axis\|bar\|span) |
| 139 | 18.193 | 765 | 139 | seeded_from_verified_log | `compiler_single_conv_qlinear`, `compiler_qlinear_uint8`, `compiler_bounded_scan` | tasklog keyword match: \b(row\|column\|horizontal\|vertical\|axis\|bar\|span) |
| 015 | 18.198 | 0 | 900 | seeded_from_verified_log | `compiler_single_conv_qlinear` | tasklog keyword match: \b(row\|column\|horizontal\|vertical\|axis\|bar\|span) |
| 220 | 18.198 | 0 | 900 | seeded_from_verified_log | `compiler_single_conv_qlinear` | tasklog keyword match: \b(row\|column\|horizontal\|vertical\|axis\|bar\|span) |
| 230 | 18.198 | 0 | 900 | seeded_from_verified_log | `compiler_tiny_lut_gather`, `compiler_single_conv_qlinear` | tasklog keyword match: \b(row\|column\|horizontal\|vertical\|axis\|bar\|span) |
| 106 | 18.233 | 776 | 93 | seeded_from_verified_log | `compiler_tiny_lut_gather`, `compiler_single_conv_qlinear`, `compiler_sparse_scatter` | tasklog keyword match: \b(row\|column\|horizontal\|vertical\|axis\|bar\|span) |
| 047 | 18.241 | 826 | 36 | seeded_from_verified_log | `compiler_bounded_scan` | tasklog keyword match: \b(row\|column\|horizontal\|vertical\|axis\|bar\|span) |
| 143 | 18.269 | 591 | 247 | seeded_from_verified_log | `compiler_direct_output_algebra`, `compiler_final_equal_overlay` | tasklog keyword match: \b(row\|column\|horizontal\|vertical\|axis\|bar\|span) |
| 122 | 18.303 | 0 | 810 | seeded_from_verified_log | `compiler_single_conv_qlinear` | tasklog keyword match: \b(row\|column\|horizontal\|vertical\|axis\|bar\|span) |
| 221 | 18.309 | 451 | 354 | seeded_from_tasklog_and_inventory | `compiler_direct_output_algebra`, `compiler_final_equal_overlay`, `compiler_bounded_scan` | tasklog keyword match: \b(row\|column\|horizontal\|vertical\|axis\|bar\|span) |
| 153 | 18.322 | 741 | 54 | seeded_from_verified_log | `compiler_direct_output_algebra`, `compiler_tiny_lut_gather`, `compiler_final_equal_overlay`, `compiler_bounded_scan` | tasklog keyword match: \b(row\|column\|horizontal\|vertical\|axis\|bar\|span) |
| 043 | 18.323 | 588 | 206 | seeded_from_verified_log | `compiler_direct_output_algebra` | tasklog keyword match: \b(row\|column\|horizontal\|vertical\|axis\|bar\|span) |
| 320 | 18.354 | 448 | 322 | seeded_from_verified_log | `compiler_single_conv_qlinear`, `compiler_sparse_scatter` | tasklog keyword match: \b(row\|column\|horizontal\|vertical\|axis\|bar\|span) |
| 214 | 18.358 | 726 | 41 | seeded_from_verified_log | `compiler_tiny_lut_gather`, `compiler_single_conv_qlinear`, `compiler_final_equal_overlay` | tasklog keyword match: \b(row\|column\|horizontal\|vertical\|axis\|bar\|span) |
| 178 | 18.363 | 702 | 61 | seeded_from_verified_log | `compiler_tiny_lut_gather`, `compiler_single_conv_qlinear`, `compiler_final_equal_overlay`, `compiler_bounded_scan` | tasklog keyword match: \b(row\|column\|horizontal\|vertical\|axis\|bar\|span) |
| 200 | 18.372 | 570 | 186 | seeded_from_verified_log | `compiler_direct_output_algebra`, `compiler_single_conv_qlinear`, `compiler_final_equal_overlay`, `compiler_bounded_scan` | tasklog keyword match: \b(row\|column\|horizontal\|vertical\|axis\|bar\|span) |
| 215 | 18.380 | 678 | 72 | seeded_from_tasklog_and_inventory | `compiler_tiny_lut_gather`, `compiler_bounded_scan` | tasklog keyword match: \b(row\|column\|horizontal\|vertical\|axis\|bar\|span) |
| 289 | 18.391 | 618 | 124 | seeded_from_verified_log | `compiler_direct_output_algebra`, `compiler_final_equal_overlay`, `compiler_bounded_scan` | tasklog keyword match: \b(row\|column\|horizontal\|vertical\|axis\|bar\|span) |
| 372 | 18.435 | 0 | 710 | seeded_from_verified_log | `compiler_single_conv_qlinear` | tasklog keyword match: \b(row\|column\|horizontal\|vertical\|axis\|bar\|span) |
| 269 | 18.436 | 685 | 24 | seeded_from_tasklog_and_inventory | `compiler_tiny_lut_gather`, `compiler_roi_pool_crop` | tasklog keyword match: \b(row\|column\|horizontal\|vertical\|axis\|bar\|span) |
| 263 | 18.478 | 563 | 117 | seeded_from_verified_log | `compiler_tiny_lut_gather`, `compiler_single_conv_qlinear`, `compiler_final_equal_overlay` | tasklog keyword match: \b(row\|column\|horizontal\|vertical\|axis\|bar\|span) |
| 057 | 18.496 | 638 | 30 | seeded_from_verified_log | `compiler_tiny_lut_gather`, `compiler_final_equal_overlay`, `compiler_bounded_scan` | tasklog keyword match: \b(row\|column\|horizontal\|vertical\|axis\|bar\|span) |
| 218 | 18.505 | 558 | 104 | seeded_from_verified_log | `compiler_tiny_lut_gather`, `compiler_single_conv_qlinear`, `compiler_final_equal_overlay` | tasklog keyword match: \b(row\|column\|horizontal\|vertical\|axis\|bar\|span) |
| 065 | 18.537 | 582 | 59 | seeded_from_verified_log | `compiler_direct_output_algebra`, `compiler_final_equal_overlay`, `compiler_bounded_scan` | tasklog keyword match: \b(row\|column\|horizontal\|vertical\|axis\|bar\|span) |
| 332 | 18.667 | 429 | 134 | seeded_from_verified_log | `compiler_single_conv_qlinear`, `compiler_qlinear_uint8` | tasklog keyword match: \b(row\|column\|horizontal\|vertical\|axis\|bar\|span) |
| 059 | 18.677 | 265 | 292 | seeded_from_verified_log | `compiler_direct_output_algebra`, `compiler_final_equal_overlay`, `compiler_bounded_scan` | tasklog keyword match: \b(row\|column\|horizontal\|vertical\|axis\|bar\|span) |
| 100 | 18.703 | 454 | 89 | seeded_from_verified_log | `compiler_direct_output_algebra`, `compiler_final_equal_overlay` | tasklog keyword match: \b(row\|column\|horizontal\|vertical\|axis\|bar\|span) |
| 300 | 18.708 | 474 | 66 | seeded_from_verified_log | `compiler_direct_output_algebra`, `compiler_tiny_lut_gather`, `compiler_final_equal_overlay` | tasklog keyword match: \b(row\|column\|horizontal\|vertical\|axis\|bar\|span) |
| 111 | 18.718 | 528 | 7 | seeded_from_verified_log | `compiler_tiny_lut_gather`, `compiler_bounded_scan` | tasklog keyword match: \b(row\|column\|horizontal\|vertical\|axis\|bar\|span) |
| 147 | 18.723 | 432 | 100 | seeded_from_verified_log | `compiler_single_conv_qlinear`, `compiler_sparse_scatter` | tasklog keyword match: \b(row\|column\|horizontal\|vertical\|axis\|bar\|span) |
| 146 | 18.795 | 388 | 107 | seeded_from_verified_log | `compiler_single_conv_qlinear`, `compiler_final_equal_overlay` | tasklog keyword match: \b(row\|column\|horizontal\|vertical\|axis\|bar\|span) |
| 039 | 18.826 | 424 | 56 | seeded_from_verified_log | `compiler_direct_onehot_gather`, `compiler_tiny_lut_gather` | tasklog keyword match: \b(row\|column\|horizontal\|vertical\|axis\|bar\|span) |
| 247 | 18.847 | 408 | 62 | seeded_from_tasklog_and_inventory | `compiler_direct_output_algebra`, `compiler_final_equal_overlay`, `compiler_bounded_scan` | tasklog keyword match: \b(row\|column\|horizontal\|vertical\|axis\|bar\|span) |
| 371 | 18.854 | 376 | 91 | seeded_from_verified_log | `compiler_direct_output_algebra`, `compiler_sparse_scatter` | tasklog keyword match: \b(row\|column\|horizontal\|vertical\|axis\|bar\|span) |
| 083 | 18.897 | 376 | 71 | seeded_from_verified_log | `compiler_tiny_lut_gather`, `compiler_sparse_scatter`, `compiler_bounded_scan` | tasklog keyword match: \b(row\|column\|horizontal\|vertical\|axis\|bar\|span) |
| 282 | 18.900 | 407 | 39 | seeded_from_verified_log | `compiler_single_conv_qlinear`, `compiler_qlinear_uint8` | tasklog keyword match: \b(row\|column\|horizontal\|vertical\|axis\|bar\|span) |
| 242 | 18.932 | 372 | 60 | seeded_from_verified_log | `compiler_direct_output_algebra` | tasklog keyword match: \b(row\|column\|horizontal\|vertical\|axis\|bar\|span) |
| 316 | 18.946 | 388 | 38 | seeded_from_verified_log | `compiler_direct_output_algebra`, `compiler_tiny_lut_gather`, `compiler_final_equal_overlay` | tasklog keyword match: \b(row\|column\|horizontal\|vertical\|axis\|bar\|span) |
| 072 | 18.957 | 390 | 31 | seeded_from_verified_log | `compiler_final_equal_overlay` | tasklog keyword match: \b(row\|column\|horizontal\|vertical\|axis\|bar\|span) |
| 257 | 18.986 | 0 | 409 | seeded_from_verified_log | `compiler_single_conv_qlinear` | tasklog keyword match: \b(row\|column\|horizontal\|vertical\|axis\|bar\|span) |
| 272 | 19.004 | 200 | 202 | seeded_from_verified_log | `compiler_single_conv_qlinear` | tasklog keyword match: \b(row\|column\|horizontal\|vertical\|axis\|bar\|span) |
| 211 | 19.019 | 240 | 156 | seeded_from_verified_log | `compiler_direct_output_algebra` | tasklog keyword match: \b(row\|column\|horizontal\|vertical\|axis\|bar\|span) |
| 127 | 19.019 | 204 | 192 | seeded_from_verified_log | `compiler_single_conv_qlinear` | tasklog keyword match: \b(row\|column\|horizontal\|vertical\|axis\|bar\|span) |
| 049 | 19.044 | 252 | 134 | seeded_from_verified_log | `compiler_direct_output_algebra`, `compiler_single_conv_qlinear`, `compiler_final_equal_overlay`, `compiler_bounded_scan` | tasklog keyword match: \b(row\|column\|horizontal\|vertical\|axis\|bar\|span) |
| 040 | 19.076 | 240 | 134 | seeded_from_verified_log | `compiler_direct_output_algebra`, `compiler_bounded_scan` | tasklog keyword match: \b(row\|column\|horizontal\|vertical\|axis\|bar\|span) |
| 181 | 19.089 | 201 | 168 | seeded_from_verified_log | `compiler_sparse_scatter` | tasklog keyword match: \b(row\|column\|horizontal\|vertical\|axis\|bar\|span) |
| 290 | 19.128 | 292 | 63 | seeded_from_verified_log | `compiler_direct_output_algebra`, `compiler_tiny_lut_gather`, `compiler_final_equal_overlay` | tasklog keyword match: \b(row\|column\|horizontal\|vertical\|axis\|bar\|span) |
| 266 | 19.142 | 315 | 35 | seeded_from_verified_log | `compiler_tiny_lut_gather` | tasklog keyword match: \b(row\|column\|horizontal\|vertical\|axis\|bar\|span) |
| 375 | 19.142 | 108 | 242 | seeded_from_verified_log | `compiler_direct_output_algebra` | tasklog keyword match: \b(row\|column\|horizontal\|vertical\|axis\|bar\|span) |
| 121 | 19.168 | 275 | 66 | seeded_from_verified_log | `compiler_direct_output_algebra`, `compiler_tiny_lut_gather` | tasklog keyword match: \b(row\|column\|horizontal\|vertical\|axis\|bar\|span) |
| 360 | 19.171 | 0 | 340 | seeded_from_verified_log | `compiler_direct_output_algebra` | tasklog keyword match: \b(row\|column\|horizontal\|vertical\|axis\|bar\|span) |
| 353 | 19.189 | 272 | 62 | seeded_from_verified_log | `compiler_direct_output_algebra`, `compiler_sparse_scatter` | tasklog keyword match: \b(row\|column\|horizontal\|vertical\|axis\|bar\|span) |
| 296 | 19.201 | 285 | 45 | seeded_from_verified_log | `compiler_final_equal_overlay` | tasklog keyword match: \b(row\|column\|horizontal\|vertical\|axis\|bar\|span) |
| 003 | 19.290 | 281 | 21 | seeded_from_verified_log | `compiler_tiny_lut_gather`, `compiler_final_equal_overlay` | tasklog keyword match: \b(row\|column\|horizontal\|vertical\|axis\|bar\|span) |
| 128 | 19.296 | 0 | 300 | seeded_from_verified_log | `compiler_single_conv_qlinear` | tasklog keyword match: \b(row\|column\|horizontal\|vertical\|axis\|bar\|span) |
| 306 | 19.296 | 0 | 300 | seeded_from_verified_log | `compiler_direct_output_algebra`, `compiler_tiny_lut_gather` | tasklog keyword match: \b(row\|column\|horizontal\|vertical\|axis\|bar\|span) |
| 321 | 19.296 | 0 | 300 | seeded_from_verified_log | `compiler_single_conv_qlinear` | tasklog keyword match: \b(row\|column\|horizontal\|vertical\|axis\|bar\|span) |
| 357 | 19.327 | 258 | 33 | seeded_from_verified_log | `compiler_single_conv_qlinear`, `compiler_final_equal_overlay`, `compiler_qlinear_uint8` | tasklog keyword match: \b(row\|column\|horizontal\|vertical\|axis\|bar\|span) |
| 249 | 19.341 | 248 | 39 | seeded_from_verified_log | `compiler_tiny_lut_gather` | tasklog keyword match: \b(row\|column\|horizontal\|vertical\|axis\|bar\|span) |
| 142 | 19.358 | 144 | 138 | seeded_from_verified_log | `compiler_direct_output_algebra` | tasklog keyword match: \b(row\|column\|horizontal\|vertical\|axis\|bar\|span) |
| 252 | 19.439 | 0 | 260 | seeded_from_verified_log | `compiler_direct_output_algebra` | tasklog keyword match: \b(row\|column\|horizontal\|vertical\|axis\|bar\|span) |
| 104 | 19.463 | 197 | 57 | seeded_from_verified_log | `compiler_direct_onehot_gather`, `compiler_tiny_lut_gather`, `compiler_single_conv_qlinear`, `compiler_qlinear_uint8` | tasklog keyword match: \b(row\|column\|horizontal\|vertical\|axis\|bar\|span) |
| 001 | 19.519 | 0 | 240 | seeded_from_verified_log | `compiler_direct_output_algebra`, `compiler_tiny_lut_gather` | tasklog keyword match: \b(row\|column\|horizontal\|vertical\|axis\|bar\|span) |
| 207 | 19.536 | 192 | 44 | seeded_from_verified_log |  | tasklog keyword match: \b(row\|column\|horizontal\|vertical\|axis\|bar\|span) |
| 315 | 19.562 | 0 | 230 | seeded_from_verified_log | `compiler_direct_output_algebra` | tasklog keyword match: \b(row\|column\|horizontal\|vertical\|axis\|bar\|span) |
| 052 | 19.575 | 213 | 14 | seeded_from_verified_log | `compiler_roi_pool_crop`, `compiler_bounded_scan` | tasklog keyword match: \b(row\|column\|horizontal\|vertical\|axis\|bar\|span) |
| 274 | 19.579 | 146 | 80 | seeded_from_verified_log | `compiler_direct_output_algebra`, `compiler_final_equal_overlay` | tasklog keyword match: \b(row\|column\|horizontal\|vertical\|axis\|bar\|span) |
| 231 | 19.611 | 180 | 39 | seeded_from_verified_log | `compiler_tiny_lut_gather` | tasklog keyword match: \b(row\|column\|horizontal\|vertical\|axis\|bar\|span) |
| 038 | 19.639 | 163 | 50 | seeded_from_verified_log | `compiler_direct_output_algebra` | tasklog keyword match: \b(row\|column\|horizontal\|vertical\|axis\|bar\|span) |
| 386 | 19.648 | 180 | 31 | seeded_from_verified_log |  | tasklog keyword match: \b(row\|column\|horizontal\|vertical\|axis\|bar\|span) |
| 180 | 19.653 | 0 | 210 | seeded_from_verified_log | `compiler_single_conv_qlinear` | tasklog keyword match: \b(row\|column\|horizontal\|vertical\|axis\|bar\|span) |
| 261 | 19.702 | 0 | 200 | seeded_from_verified_log | `compiler_single_conv_qlinear` | tasklog keyword match: \b(row\|column\|horizontal\|vertical\|axis\|bar\|span) |
| 082 | 19.753 | 0 | 190 | seeded_from_verified_log | `compiler_single_conv_qlinear` | tasklog keyword match: \b(row\|column\|horizontal\|vertical\|axis\|bar\|span) |
| 028 | 19.876 | 0 | 168 | seeded_from_verified_log | `compiler_direct_output_algebra` | tasklog keyword match: \b(row\|column\|horizontal\|vertical\|axis\|bar\|span) |
| 258 | 19.925 | 0 | 160 | seeded_from_verified_log | `compiler_single_conv_qlinear` | tasklog keyword match: \b(row\|column\|horizontal\|vertical\|axis\|bar\|span) |
| 391 | 19.931 | 135 | 24 | seeded_from_verified_log | `compiler_tiny_lut_gather`, `compiler_final_equal_overlay` | tasklog keyword match: \b(row\|column\|horizontal\|vertical\|axis\|bar\|span) |
| 150 | 19.937 | 128 | 30 | seeded_from_verified_log | `compiler_tiny_lut_gather` | tasklog keyword match: \b(row\|column\|horizontal\|vertical\|axis\|bar\|span) |
| 155 | 19.937 | 128 | 30 | seeded_from_verified_log | `compiler_tiny_lut_gather` | tasklog keyword match: \b(row\|column\|horizontal\|vertical\|axis\|bar\|span) |
| 006 | 19.970 | 126 | 27 | seeded_from_verified_log |  | tasklog keyword match: \b(row\|column\|horizontal\|vertical\|axis\|bar\|span) |
| 248 | 19.983 | 122 | 29 | seeded_from_verified_log | `compiler_sparse_scatter` | tasklog keyword match: \b(row\|column\|horizontal\|vertical\|axis\|bar\|span) |
| 167 | 20.003 | 20 | 128 | seeded_from_verified_log | `compiler_direct_output_algebra` | tasklog keyword match: \b(row\|column\|horizontal\|vertical\|axis\|bar\|span) |
| 317 | 20.016 | 128 | 18 | seeded_from_verified_log | `compiler_direct_output_algebra`, `compiler_roi_pool_crop` | tasklog keyword match: \b(row\|column\|horizontal\|vertical\|axis\|bar\|span) |
| 389 | 20.058 | 128 | 12 | seeded_from_verified_log | `compiler_direct_output_algebra`, `compiler_tiny_lut_gather`, `compiler_sparse_scatter`, `compiler_bounded_scan` | tasklog keyword match: \b(row\|column\|horizontal\|vertical\|axis\|bar\|span) |
| 339 | 20.058 | 40 | 100 | seeded_from_verified_log | `compiler_direct_output_algebra` | tasklog keyword match: \b(row\|column\|horizontal\|vertical\|axis\|bar\|span) |
| 149 | 20.087 | 36 | 100 | seeded_from_verified_log | `compiler_direct_output_algebra`, `compiler_single_conv_qlinear` | tasklog keyword match: \b(row\|column\|horizontal\|vertical\|axis\|bar\|span) |
| 298 | 20.095 | 120 | 15 | seeded_from_verified_log | `compiler_direct_output_algebra` | tasklog keyword match: \b(row\|column\|horizontal\|vertical\|axis\|bar\|span) |
| 007 | 20.156 | 0 | 127 | seeded_from_verified_log | `compiler_direct_output_algebra`, `compiler_tiny_lut_gather` | tasklog keyword match: \b(row\|column\|horizontal\|vertical\|axis\|bar\|span) |
| 024 | 20.300 | 0 | 110 | seeded_from_verified_log | `compiler_direct_output_algebra` | tasklog keyword match: \b(row\|column\|horizontal\|vertical\|axis\|bar\|span) |
| 262 | 20.337 | 90 | 16 | seeded_from_verified_log | `compiler_direct_output_algebra` | tasklog keyword match: \b(row\|column\|horizontal\|vertical\|axis\|bar\|span) |
| 227 | 20.395 | 0 | 100 | seeded_from_verified_log | `compiler_direct_output_algebra`, `compiler_single_conv_qlinear` | tasklog keyword match: \b(row\|column\|horizontal\|vertical\|axis\|bar\|span) |
| 318 | 20.395 | 0 | 100 | seeded_from_verified_log | `compiler_direct_output_algebra`, `compiler_single_conv_qlinear` | tasklog keyword match: \b(row\|column\|horizontal\|vertical\|axis\|bar\|span) |
| 380 | 20.405 | 0 | 99 | seeded_from_tasklog_and_inventory | `compiler_direct_output_algebra` | tasklog keyword match: \b(row\|column\|horizontal\|vertical\|axis\|bar\|span) |
| 291 | 20.780 | 40 | 28 | seeded_from_verified_log | `compiler_direct_output_algebra` | tasklog keyword match: \b(row\|column\|horizontal\|vertical\|axis\|bar\|span) |
| 373 | 20.906 | 0 | 60 | seeded_from_verified_log | `compiler_direct_output_algebra` | tasklog keyword match: \b(row\|column\|horizontal\|vertical\|axis\|bar\|span) |
| 067 | 21.150 | 39 | 8 | seeded_from_verified_log | `compiler_direct_output_algebra` | tasklog keyword match: \b(row\|column\|horizontal\|vertical\|axis\|bar\|span) |
| 073 | 21.311 | 0 | 40 | seeded_from_verified_log | `compiler_direct_output_algebra`, `compiler_single_conv_qlinear` | tasklog keyword match: \b(row\|column\|horizontal\|vertical\|axis\|bar\|span) |
| 053 | 21.599 | 0 | 30 | seeded_from_verified_log | `compiler_direct_output_algebra`, `compiler_tiny_lut_gather` | tasklog keyword match: \b(row\|column\|horizontal\|vertical\|axis\|bar\|span) |
| 113 | 21.599 | 0 | 30 | seeded_from_verified_log | `compiler_direct_output_algebra`, `compiler_tiny_lut_gather` | tasklog keyword match: \b(row\|column\|horizontal\|vertical\|axis\|bar\|span) |
| 326 | 21.599 | 0 | 30 | seeded_from_verified_log | `compiler_direct_output_algebra` | tasklog keyword match: \b(row\|column\|horizontal\|vertical\|axis\|bar\|span) |
| 166 | 21.822 | 0 | 24 | seeded_from_verified_log | `compiler_direct_output_algebra` | tasklog keyword match: \b(row\|column\|horizontal\|vertical\|axis\|bar\|span) |
| 312 | 22.004 | 0 | 20 | seeded_from_verified_log | `compiler_direct_output_algebra` | tasklog keyword match: \b(row\|column\|horizontal\|vertical\|axis\|bar\|span) |
| 140 | 23.391 | 0 | 5 | seeded_from_verified_log | `compiler_direct_output_algebra`, `compiler_tiny_lut_gather`, `compiler_roi_pool_crop` | tasklog keyword match: \b(row\|column\|horizontal\|vertical\|axis\|bar\|span) |

## Known Best Routes

- Pending human review.

## Kill Criteria

- Pending human review.

## Successful Applications

- Pending verification.

## Failed Applications / Walls

- Pending verification.
