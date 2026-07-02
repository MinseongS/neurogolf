# shape_extend_line_ray — Extend line/ray

Line, ray, or span is extended from observed marks.

## Optimization Question

What is the cheapest verified ONNX family for tasks in this class, and
which semantic preconditions let us avoid full-canvas intermediates?

## Candidate Tasks

| task | pts | mem | params | status | first routes | evidence |
|---:|---:|---:|---:|---|---|---|
| 233 | 14.003 | 59147 | 565 | seeded_from_verified_log | `compiler_direct_onehot_gather`, `compiler_tiny_lut_gather`, `compiler_single_conv_qlinear`, `compiler_final_equal_overlay` | tasklog keyword match: \b(extend\|line\|ray\|span\|connect\|between) |
| 286 | 14.341 | 41822 | 742 | seeded_from_verified_log | `compiler_tiny_lut_gather`, `compiler_single_conv_qlinear`, `compiler_final_equal_overlay`, `compiler_qlinear_uint8` | tasklog keyword match: \b(extend\|line\|ray\|span\|connect\|between) |
| 366 | 14.497 | 35927 | 490 | seeded_from_verified_log | `compiler_tiny_lut_gather`, `compiler_single_conv_qlinear`, `compiler_final_equal_overlay`, `compiler_sparse_scatter` | tasklog keyword match: \b(extend\|line\|ray\|span\|connect\|between) |
| 158 | 14.528 | 32979 | 2340 | seeded_from_verified_log | `compiler_tiny_lut_gather`, `compiler_single_conv_qlinear`, `compiler_final_equal_overlay`, `compiler_qlinear_uint8` | tasklog keyword match: \b(extend\|line\|ray\|span\|connect\|between) |
| 118 | 14.559 | 30849 | 3387 | seeded_from_verified_log | `compiler_direct_output_algebra`, `compiler_direct_onehot_gather`, `compiler_tiny_lut_gather`, `compiler_single_conv_qlinear` | tasklog keyword match: \b(extend\|line\|ray\|span\|connect\|between) |
| 133 | 14.578 | 32294 | 1288 | seeded_from_verified_log | `compiler_single_conv_qlinear`, `compiler_final_equal_overlay`, `compiler_qlinear_uint8`, `compiler_roi_pool_crop` | tasklog keyword match: \b(extend\|line\|ray\|span\|connect\|between) |
| 187 | 14.580 | 32850 | 665 | seeded_from_verified_log | `compiler_single_conv_qlinear`, `compiler_final_equal_overlay`, `compiler_bounded_scan` | tasklog keyword match: \b(extend\|line\|ray\|span\|connect\|between) |
| 209 | 14.620 | 32027 | 185 | seeded_from_verified_log | `compiler_tiny_lut_gather`, `compiler_single_conv_qlinear`, `compiler_final_equal_overlay`, `compiler_qlinear_uint8` | tasklog keyword match: \b(extend\|line\|ray\|span\|connect\|between) |
| 054 | 14.829 | 25885 | 238 | seeded_from_tasklog_and_inventory | `compiler_tiny_lut_gather`, `compiler_single_conv_qlinear`, `compiler_final_equal_overlay`, `compiler_sparse_scatter` | tasklog keyword match: \b(extend\|line\|ray\|span\|connect\|between) |
| 285 | 14.848 | 25080 | 550 | seeded_from_verified_log | `compiler_tiny_lut_gather`, `compiler_single_conv_qlinear`, `compiler_final_equal_overlay`, `compiler_sparse_scatter` | tasklog keyword match: \b(extend\|line\|ray\|span\|connect\|between) |
| 002 | 14.896 | 24320 | 125 | seeded_from_verified_log | `compiler_direct_onehot_gather`, `compiler_tiny_lut_gather` | tasklog keyword match: \b(extend\|line\|ray\|span\|connect\|between) |
| 076 | 14.932 | 23296 | 292 | seeded_from_verified_log | `compiler_tiny_lut_gather`, `compiler_final_equal_overlay`, `compiler_sparse_scatter`, `compiler_bounded_scan` | tasklog keyword match: \b(extend\|line\|ray\|span\|connect\|between) |
| 364 | 14.956 | 22900 | 113 | seeded_from_verified_log | `compiler_tiny_lut_gather`, `compiler_single_conv_qlinear`, `compiler_final_equal_overlay`, `compiler_qlinear_uint8` | tasklog keyword match: \b(extend\|line\|ray\|span\|connect\|between) |
| 243 | 14.972 | 22608 | 44 | seeded_from_verified_log | `compiler_single_conv_qlinear`, `compiler_qlinear_uint8` | tasklog keyword match: \b(extend\|line\|ray\|span\|connect\|between) |
| 367 | 15.051 | 16200 | 4733 | seeded_from_verified_log | `compiler_tiny_lut_gather`, `compiler_single_conv_qlinear`, `compiler_qlinear_uint8` | tasklog keyword match: \b(extend\|line\|ray\|span\|connect\|between) |
| 219 | 15.195 | 18033 | 93 | seeded_from_verified_log | `compiler_tiny_lut_gather`, `compiler_final_equal_overlay`, `compiler_bounded_scan` | tasklog keyword match: \b(extend\|line\|ray\|span\|connect\|between) |
| 101 | 15.216 | 16850 | 905 | seeded_from_verified_log | `compiler_tiny_lut_gather`, `compiler_final_equal_overlay`, `compiler_sparse_scatter`, `compiler_bounded_scan` | tasklog keyword match: \b(extend\|line\|ray\|span\|connect\|between) |
| 066 | 15.256 | 16899 | 160 | seeded_from_verified_log | `compiler_direct_output_algebra`, `compiler_final_equal_overlay`, `compiler_bounded_scan` | tasklog keyword match: \b(extend\|line\|ray\|span\|connect\|between) |
| 077 | 15.393 | 14760 | 114 | seeded_from_verified_log | `compiler_single_conv_qlinear`, `compiler_qlinear_uint8`, `compiler_bounded_scan` | tasklog keyword match: \b(extend\|line\|ray\|span\|connect\|between) |
| 025 | 15.448 | 13874 | 195 | seeded_from_verified_log | `compiler_direct_output_algebra`, `compiler_final_equal_overlay`, `compiler_bounded_scan` | tasklog keyword match: \b(extend\|line\|ray\|span\|connect\|between) |
| 138 | 15.456 | 13789 | 172 | seeded_from_verified_log | `compiler_tiny_lut_gather`, `compiler_single_conv_qlinear`, `compiler_final_equal_overlay`, `compiler_bounded_scan` | tasklog keyword match: \b(extend\|line\|ray\|span\|connect\|between) |
| 110 | 15.468 | 13155 | 634 | seeded_from_verified_log | `compiler_tiny_lut_gather`, `compiler_single_conv_qlinear`, `compiler_final_equal_overlay`, `compiler_bounded_scan` | tasklog keyword match: \b(extend\|line\|ray\|span\|connect\|between) |
| 198 | 15.484 | 13434 | 138 | seeded_from_verified_log | `compiler_tiny_lut_gather`, `compiler_single_conv_qlinear`, `compiler_final_equal_overlay`, `compiler_bounded_scan` | tasklog keyword match: \b(extend\|line\|ray\|span\|connect\|between) |
| 064 | 15.494 | 13368 | 68 | seeded_from_verified_log | `compiler_tiny_lut_gather`, `compiler_final_equal_overlay`, `compiler_bounded_scan` | tasklog keyword match: \b(extend\|line\|ray\|span\|connect\|between) |
| 145 | 15.511 | 13104 | 111 | seeded_from_verified_log | `compiler_tiny_lut_gather`, `compiler_final_equal_overlay`, `compiler_bounded_scan` | tasklog keyword match: \b(extend\|line\|ray\|span\|connect\|between) |
| 044 | 15.651 | 11420 | 68 | seeded_from_verified_log | `compiler_direct_onehot_gather`, `compiler_tiny_lut_gather`, `compiler_single_conv_qlinear`, `compiler_final_equal_overlay` | tasklog keyword match: \b(extend\|line\|ray\|span\|connect\|between) |
| 338 | 15.712 | 10140 | 666 | seeded_from_verified_log | `compiler_tiny_lut_gather`, `compiler_single_conv_qlinear`, `compiler_final_equal_overlay`, `compiler_qlinear_uint8` | tasklog keyword match: \b(extend\|line\|ray\|span\|connect\|between) |
| 080 | 15.726 | 10198 | 454 | seeded_from_verified_log | `compiler_tiny_lut_gather`, `compiler_single_conv_qlinear`, `compiler_final_equal_overlay`, `compiler_qlinear_uint8` | tasklog keyword match: \b(extend\|line\|ray\|span\|connect\|between) |
| 204 | 15.753 | 9920 | 453 | seeded_from_verified_log | `compiler_single_conv_qlinear`, `compiler_qlinear_uint8`, `compiler_bounded_scan` | tasklog keyword match: \b(extend\|line\|ray\|span\|connect\|between) |
| 379 | 15.818 | 9069 | 653 | seeded_from_verified_log | `compiler_single_conv_qlinear`, `compiler_final_equal_overlay`, `compiler_bounded_scan` | tasklog keyword match: \b(extend\|line\|ray\|span\|connect\|between) |
| 370 | 15.823 | 8645 | 1024 | seeded_from_verified_log | `compiler_direct_output_algebra`, `compiler_tiny_lut_gather`, `compiler_single_conv_qlinear`, `compiler_final_equal_overlay` | tasklog keyword match: \b(extend\|line\|ray\|span\|connect\|between) |
| 216 | 15.880 | 9048 | 87 | seeded_from_verified_log | `compiler_direct_onehot_gather`, `compiler_tiny_lut_gather`, `compiler_single_conv_qlinear`, `compiler_final_equal_overlay` | tasklog keyword match: \b(extend\|line\|ray\|span\|connect\|between) |
| 350 | 15.891 | 9012 | 26 | seeded_from_verified_log | `compiler_tiny_lut_gather`, `compiler_bounded_scan` | tasklog keyword match: \b(extend\|line\|ray\|span\|connect\|between) |
| 222 | 15.916 | 8736 | 78 | seeded_from_verified_log | `compiler_single_conv_qlinear`, `compiler_final_equal_overlay`, `compiler_qlinear_uint8`, `compiler_bounded_scan` | tasklog keyword match: \b(extend\|line\|ray\|span\|connect\|between) |
| 192 | 15.938 | 8515 | 106 | seeded_from_verified_log | `compiler_single_conv_qlinear`, `compiler_final_equal_overlay` | tasklog keyword match: \b(extend\|line\|ray\|span\|connect\|between) |
| 157 | 15.972 | 7983 | 351 | seeded_from_verified_log | `compiler_tiny_lut_gather`, `compiler_final_equal_overlay`, `compiler_sparse_scatter`, `compiler_bounded_scan` | tasklog keyword match: \b(extend\|line\|ray\|span\|connect\|between) |
| 014 | 16.023 | 7809 | 108 | seeded_from_verified_log | `compiler_tiny_lut_gather`, `compiler_final_equal_overlay`, `compiler_bounded_scan` | tasklog keyword match: \b(extend\|line\|ray\|span\|connect\|between) |
| 202 | 16.039 | 7773 | 24 | seeded_from_verified_log | `compiler_single_conv_qlinear`, `compiler_final_equal_overlay`, `compiler_bounded_scan` | tasklog keyword match: \b(extend\|line\|ray\|span\|connect\|between) |
| 089 | 16.111 | 7092 | 160 | seeded_from_verified_log | `compiler_tiny_lut_gather`, `compiler_single_conv_qlinear`, `compiler_final_equal_overlay`, `compiler_sparse_scatter` | tasklog keyword match: \b(extend\|line\|ray\|span\|connect\|between) |
| 009 | 16.143 | 6929 | 95 | seeded_from_verified_log | `compiler_tiny_lut_gather`, `compiler_single_conv_qlinear`, `compiler_final_equal_overlay`, `compiler_bounded_scan` | tasklog keyword match: \b(extend\|line\|ray\|span\|connect\|between) |
| 092 | 16.148 | 6805 | 182 | seeded_from_verified_log | `compiler_direct_output_algebra`, `compiler_final_equal_overlay`, `compiler_sparse_scatter`, `compiler_bounded_scan` | tasklog keyword match: \b(extend\|line\|ray\|span\|connect\|between) |
| 005 | 16.163 | 6392 | 490 | seeded_from_verified_log | `compiler_tiny_lut_gather`, `compiler_single_conv_qlinear`, `compiler_final_equal_overlay`, `compiler_qlinear_uint8` | tasklog keyword match: \b(extend\|line\|ray\|span\|connect\|between) |
| 182 | 16.176 | 6695 | 98 | seeded_from_verified_log | `compiler_tiny_lut_gather`, `compiler_single_conv_qlinear`, `compiler_final_equal_overlay`, `compiler_qlinear_uint8` | tasklog keyword match: \b(extend\|line\|ray\|span\|connect\|between) |
| 368 | 16.264 | 6022 | 203 | seeded_from_verified_log | `compiler_tiny_lut_gather`, `compiler_final_equal_overlay`, `compiler_sparse_scatter` | tasklog keyword match: \b(extend\|line\|ray\|span\|connect\|between) |
| 340 | 16.285 | 5812 | 279 | seeded_from_verified_log | `compiler_tiny_lut_gather`, `compiler_final_equal_overlay`, `compiler_qlinear_uint8`, `compiler_bounded_scan` | tasklog keyword match: \b(extend\|line\|ray\|span\|connect\|between) |
| 383 | 16.289 | 5974 | 96 | seeded_from_verified_log | `compiler_tiny_lut_gather`, `compiler_single_conv_qlinear`, `compiler_final_equal_overlay`, `compiler_bounded_scan` | tasklog keyword match: \b(extend\|line\|ray\|span\|connect\|between) |
| 264 | 16.292 | 5348 | 706 | seeded_from_verified_log | `compiler_direct_output_algebra`, `compiler_tiny_lut_gather`, `compiler_single_conv_qlinear`, `compiler_final_equal_overlay` | tasklog keyword match: \b(extend\|line\|ray\|span\|connect\|between) |
| 280 | 16.294 | 5286 | 753 | seeded_from_verified_log | `compiler_direct_output_algebra`, `compiler_final_equal_overlay` | tasklog keyword match: \b(extend\|line\|ray\|span\|connect\|between) |
| 085 | 16.409 | 4500 | 881 | seeded_from_verified_log | `compiler_single_conv_qlinear` | tasklog keyword match: \b(extend\|line\|ray\|span\|connect\|between) |
| 234 | 16.448 | 5113 | 66 | seeded_from_verified_log | `compiler_direct_output_algebra`, `compiler_tiny_lut_gather`, `compiler_final_equal_overlay`, `compiler_bounded_scan` | tasklog keyword match: \b(extend\|line\|ray\|span\|connect\|between) |
| 251 | 16.499 | 4752 | 168 | seeded_from_verified_log | `compiler_bounded_scan` | tasklog keyword match: \b(extend\|line\|ray\|span\|connect\|between) |
| 037 | 16.535 | 4614 | 132 | seeded_from_verified_log | `compiler_tiny_lut_gather`, `compiler_final_equal_overlay`, `compiler_sparse_scatter`, `compiler_bounded_scan` | tasklog keyword match: \b(extend\|line\|ray\|span\|connect\|between) |
| 196 | 16.580 | 4500 | 38 | seeded_from_verified_log | `compiler_single_conv_qlinear`, `compiler_qlinear_uint8`, `compiler_bounded_scan` | tasklog keyword match: \b(extend\|line\|ray\|span\|connect\|between) |
| 358 | 16.587 | 4332 | 174 | seeded_from_verified_log | `compiler_direct_output_algebra`, `compiler_final_equal_overlay`, `compiler_bounded_scan` | tasklog keyword match: \b(extend\|line\|ray\|span\|connect\|between) |
| 378 | 16.600 | 4332 | 117 | seeded_from_verified_log | `compiler_direct_output_algebra`, `compiler_tiny_lut_gather`, `compiler_final_equal_overlay`, `compiler_bounded_scan` | tasklog keyword match: \b(extend\|line\|ray\|span\|connect\|between) |
| 265 | 16.619 | 4328 | 34 | seeded_from_verified_log | `compiler_tiny_lut_gather`, `compiler_final_equal_overlay`, `compiler_sparse_scatter`, `compiler_bounded_scan` | tasklog keyword match: \b(extend\|line\|ray\|span\|connect\|between) |
| 277 | 16.627 | 4101 | 228 | seeded_from_verified_log | `compiler_tiny_lut_gather`, `compiler_final_equal_overlay`, `compiler_bounded_scan` | tasklog keyword match: \b(extend\|line\|ray\|span\|connect\|between) |
| 206 | 16.657 | 4122 | 77 | seeded_from_verified_log | `compiler_single_conv_qlinear`, `compiler_final_equal_overlay`, `compiler_sparse_scatter`, `compiler_bounded_scan` | tasklog keyword match: \b(extend\|line\|ray\|span\|connect\|between) |
| 117 | 16.666 | 3922 | 243 | seeded_from_verified_log | `compiler_tiny_lut_gather`, `compiler_single_conv_qlinear`, `compiler_final_equal_overlay`, `compiler_qlinear_uint8` | tasklog keyword match: \b(extend\|line\|ray\|span\|connect\|between) |
| 363 | 16.673 | 3839 | 296 | seeded_from_verified_log | `compiler_single_conv_qlinear`, `compiler_final_equal_overlay`, `compiler_bounded_scan` | tasklog keyword match: \b(extend\|line\|ray\|span\|connect\|between) |
| 132 | 16.684 | 3990 | 99 | seeded_from_verified_log | `compiler_tiny_lut_gather`, `compiler_single_conv_qlinear`, `compiler_final_equal_overlay`, `compiler_bounded_scan` | tasklog keyword match: \b(extend\|line\|ray\|span\|connect\|between) |
| 161 | 16.684 | 4055 | 33 | seeded_from_verified_log | `compiler_tiny_lut_gather`, `compiler_final_equal_overlay` | tasklog keyword match: \b(extend\|line\|ray\|span\|connect\|between) |
| 107 | 16.687 | 2924 | 1154 | seeded_from_verified_log | `compiler_direct_onehot_gather`, `compiler_tiny_lut_gather`, `compiler_final_equal_overlay`, `compiler_sparse_scatter` | tasklog keyword match: \b(extend\|line\|ray\|span\|connect\|between) |
| 365 | 16.699 | 3908 | 120 | seeded_from_verified_log | `compiler_direct_output_algebra`, `compiler_tiny_lut_gather`, `compiler_single_conv_qlinear`, `compiler_final_equal_overlay` | tasklog keyword match: \b(extend\|line\|ray\|span\|connect\|between) |
| 131 | 16.709 | 3312 | 675 | seeded_from_verified_log | `compiler_sparse_scatter`, `compiler_bounded_scan` | tasklog keyword match: \b(extend\|line\|ray\|span\|connect\|between) |
| 069 | 16.728 | 3848 | 64 | seeded_from_verified_log | `compiler_tiny_lut_gather`, `compiler_single_conv_qlinear`, `compiler_final_equal_overlay`, `compiler_bounded_scan` | tasklog keyword match: \b(extend\|line\|ray\|span\|connect\|between) |
| 050 | 16.728 | 3825 | 86 | seeded_from_verified_log | `compiler_direct_onehot_gather` | tasklog keyword match: \b(extend\|line\|ray\|span\|connect\|between) |
| 359 | 16.740 | 3852 | 15 | seeded_from_verified_log | `compiler_final_equal_overlay`, `compiler_bounded_scan` | tasklog keyword match: \b(extend\|line\|ray\|span\|connect\|between) |
| 177 | 16.751 | 3692 | 130 | seeded_from_verified_log | `compiler_single_conv_qlinear`, `compiler_final_equal_overlay`, `compiler_roi_pool_crop` | tasklog keyword match: \b(extend\|line\|ray\|span\|connect\|between) |
| 284 | 16.774 | 3082 | 653 | seeded_from_verified_log | `compiler_single_conv_qlinear`, `compiler_sparse_scatter` | tasklog keyword match: \b(extend\|line\|ray\|span\|connect\|between) |
| 093 | 16.823 | 3456 | 103 | seeded_from_verified_log | `compiler_tiny_lut_gather`, `compiler_final_equal_overlay`, `compiler_qlinear_uint8`, `compiler_bounded_scan` | tasklog keyword match: \b(extend\|line\|ray\|span\|connect\|between) |
| 125 | 16.850 | 3434 | 29 | seeded_from_verified_log | `compiler_final_equal_overlay`, `compiler_bounded_scan` | tasklog keyword match: \b(extend\|line\|ray\|span\|connect\|between) |
| 201 | 16.881 | 3202 | 154 | seeded_from_verified_log | `compiler_tiny_lut_gather`, `compiler_single_conv_qlinear`, `compiler_final_equal_overlay`, `compiler_bounded_scan` | tasklog keyword match: \b(extend\|line\|ray\|span\|connect\|between) |
| 310 | 16.902 | 3208 | 79 | seeded_from_verified_log | `compiler_direct_output_algebra`, `compiler_tiny_lut_gather`, `compiler_final_equal_overlay` | tasklog keyword match: \b(extend\|line\|ray\|span\|connect\|between) |
| 333 | 16.921 | 2792 | 435 | seeded_from_verified_log | `compiler_direct_output_algebra`, `compiler_tiny_lut_gather`, `compiler_final_equal_overlay`, `compiler_bounded_scan` | tasklog keyword match: \b(extend\|line\|ray\|span\|connect\|between) |
| 250 | 16.921 | 3164 | 61 | seeded_from_verified_log | `compiler_tiny_lut_gather`, `compiler_final_equal_overlay`, `compiler_qlinear_uint8`, `compiler_bounded_scan` | tasklog keyword match: \b(extend\|line\|ray\|span\|connect\|between) |
| 079 | 16.947 | 3065 | 78 | seeded_from_verified_log | `compiler_single_conv_qlinear`, `compiler_final_equal_overlay`, `compiler_bounded_scan` | tasklog keyword match: \b(extend\|line\|ray\|span\|connect\|between) |
| 086 | 16.949 | 2946 | 190 | seeded_from_verified_log | `compiler_tiny_lut_gather`, `compiler_single_conv_qlinear`, `compiler_final_equal_overlay`, `compiler_qlinear_uint8` | tasklog keyword match: \b(extend\|line\|ray\|span\|connect\|between) |
| 035 | 16.954 | 2994 | 128 | seeded_from_verified_log | `compiler_sparse_scatter` | tasklog keyword match: \b(extend\|line\|ray\|span\|connect\|between) |
| 055 | 16.963 | 3030 | 62 | seeded_from_verified_log | `compiler_tiny_lut_gather`, `compiler_final_equal_overlay`, `compiler_qlinear_uint8`, `compiler_bounded_scan` | tasklog keyword match: \b(extend\|line\|ray\|span\|connect\|between) |
| 270 | 16.971 | 2742 | 327 | seeded_from_verified_log | `compiler_direct_output_algebra`, `compiler_bounded_scan` | tasklog keyword match: \b(extend\|line\|ray\|span\|connect\|between) |
| 091 | 16.984 | 2853 | 175 | seeded_from_verified_log | `compiler_tiny_lut_gather`, `compiler_single_conv_qlinear`, `compiler_final_equal_overlay` | tasklog keyword match: \b(extend\|line\|ray\|span\|connect\|between) |
| 330 | 16.986 | 2800 | 222 | seeded_from_verified_log | `compiler_final_equal_overlay`, `compiler_sparse_scatter`, `compiler_bounded_scan` | tasklog keyword match: \b(extend\|line\|ray\|span\|connect\|between) |
| 335 | 17.001 | 2288 | 691 | seeded_from_verified_log | `compiler_direct_output_algebra`, `compiler_single_conv_qlinear`, `compiler_final_equal_overlay`, `compiler_bounded_scan` | tasklog keyword match: \b(extend\|line\|ray\|span\|connect\|between) |
| 105 | 17.007 | 2854 | 106 | seeded_from_verified_log | `compiler_direct_onehot_gather`, `compiler_tiny_lut_gather`, `compiler_final_equal_overlay`, `compiler_bounded_scan` | tasklog keyword match: \b(extend\|line\|ray\|span\|connect\|between) |
| 398 | 17.042 | 1666 | 1193 | seeded_from_verified_log | `compiler_direct_onehot_gather`, `compiler_tiny_lut_gather`, `compiler_final_equal_overlay` | tasklog keyword match: \b(extend\|line\|ray\|span\|connect\|between) |
| 019 | 17.063 | 2709 | 89 | seeded_from_verified_log | `compiler_tiny_lut_gather`, `compiler_single_conv_qlinear`, `compiler_final_equal_overlay`, `compiler_qlinear_uint8` | tasklog keyword match: \b(extend\|line\|ray\|span\|connect\|between) |
| 246 | 17.080 | 2334 | 419 | seeded_from_verified_log | `compiler_single_conv_qlinear`, `compiler_final_equal_overlay` | tasklog keyword match: \b(extend\|line\|ray\|span\|connect\|between) |
| 354 | 17.080 | 2674 | 77 | seeded_from_verified_log | `compiler_single_conv_qlinear`, `compiler_final_equal_overlay`, `compiler_bounded_scan` | tasklog keyword match: \b(extend\|line\|ray\|span\|connect\|between) |
| 245 | 17.083 | 2604 | 139 | seeded_from_verified_log | `compiler_tiny_lut_gather`, `compiler_bounded_scan` | tasklog keyword match: \b(extend\|line\|ray\|span\|connect\|between) |
| 190 | 17.089 | 2480 | 247 | seeded_from_verified_log | `compiler_single_conv_qlinear`, `compiler_qlinear_uint8`, `compiler_bounded_scan` | tasklog keyword match: \b(extend\|line\|ray\|span\|connect\|between) |
| 355 | 17.093 | 2688 | 27 | seeded_from_verified_log | `compiler_final_equal_overlay`, `compiler_bounded_scan` | tasklog keyword match: \b(extend\|line\|ray\|span\|connect\|between) |
| 071 | 17.100 | 2643 | 55 | seeded_from_verified_log | `compiler_tiny_lut_gather`, `compiler_final_equal_overlay`, `compiler_bounded_scan` | tasklog keyword match: \b(extend\|line\|ray\|span\|connect\|between) |
| 062 | 17.162 | 2465 | 71 | seeded_from_verified_log | `compiler_tiny_lut_gather`, `compiler_final_equal_overlay`, `compiler_bounded_scan` | tasklog keyword match: \b(extend\|line\|ray\|span\|connect\|between) |
| 184 | 17.168 | 2460 | 60 | seeded_from_verified_log | `compiler_direct_output_algebra`, `compiler_single_conv_qlinear`, `compiler_final_equal_overlay`, `compiler_bounded_scan` | tasklog keyword match: \b(extend\|line\|ray\|span\|connect\|between) |
| 388 | 17.178 | 2278 | 218 | seeded_from_verified_log | `compiler_final_equal_overlay`, `compiler_sparse_scatter`, `compiler_bounded_scan` | tasklog keyword match: \b(extend\|line\|ray\|span\|connect\|between) |
| 238 | 17.178 | 2051 | 444 | seeded_from_verified_log | `compiler_direct_output_algebra`, `compiler_tiny_lut_gather`, `compiler_final_equal_overlay`, `compiler_bounded_scan` | tasklog keyword match: \b(extend\|line\|ray\|span\|connect\|between) |
| 030 | 17.218 | 2248 | 148 | seeded_from_verified_log | `compiler_tiny_lut_gather`, `compiler_bounded_scan` | tasklog keyword match: \b(extend\|line\|ray\|span\|connect\|between) |
| 281 | 17.223 | 2321 | 65 | seeded_from_verified_log | `compiler_direct_output_algebra`, `compiler_final_equal_overlay`, `compiler_bounded_scan` | tasklog keyword match: \b(extend\|line\|ray\|span\|connect\|between) |
| 212 | 17.234 | 2178 | 182 | seeded_from_verified_log | `compiler_single_conv_qlinear`, `compiler_qlinear_uint8`, `compiler_bounded_scan` | tasklog keyword match: \b(extend\|line\|ray\|span\|connect\|between) |
| 163 | 17.240 | 2235 | 110 | seeded_from_verified_log | `compiler_tiny_lut_gather`, `compiler_single_conv_qlinear`, `compiler_final_equal_overlay`, `compiler_bounded_scan` | tasklog keyword match: \b(extend\|line\|ray\|span\|connect\|between) |
| 046 | 17.252 | 2221 | 97 | seeded_from_tasklog_and_inventory | `compiler_tiny_lut_gather`, `compiler_single_conv_qlinear`, `compiler_final_equal_overlay`, `compiler_bounded_scan` | tasklog keyword match: \b(extend\|line\|ray\|span\|connect\|between) |
| 374 | 17.258 | 2262 | 42 | seeded_from_verified_log | `compiler_final_equal_overlay`, `compiler_bounded_scan` | tasklog keyword match: \b(extend\|line\|ray\|span\|connect\|between) |
| 134 | 17.266 | 2250 | 34 | seeded_from_verified_log | `compiler_tiny_lut_gather`, `compiler_bounded_scan` | tasklog keyword match: \b(extend\|line\|ray\|span\|connect\|between) |
| 168 | 17.286 | 2096 | 143 | seeded_from_verified_log | `compiler_tiny_lut_gather`, `compiler_single_conv_qlinear`, `compiler_qlinear_uint8`, `compiler_bounded_scan` | tasklog keyword match: \b(extend\|line\|ray\|span\|connect\|between) |
| 336 | 17.298 | 1188 | 1024 | seeded_from_verified_log | `compiler_tiny_lut_gather`, `compiler_sparse_scatter` | tasklog keyword match: \b(extend\|line\|ray\|span\|connect\|between) |
| 036 | 17.314 | 2140 | 37 | seeded_from_verified_log | `compiler_final_equal_overlay`, `compiler_bounded_scan` | tasklog keyword match: \b(extend\|line\|ray\|span\|connect\|between) |
| 325 | 17.335 | 2010 | 123 | seeded_from_verified_log | `compiler_single_conv_qlinear`, `compiler_final_equal_overlay`, `compiler_qlinear_uint8`, `compiler_bounded_scan` | tasklog keyword match: \b(extend\|line\|ray\|span\|connect\|between) |
| 348 | 17.353 | 1938 | 157 | seeded_from_verified_log | `compiler_final_equal_overlay` | tasklog keyword match: \b(extend\|line\|ray\|span\|connect\|between) |
| 063 | 17.394 | 1728 | 283 | seeded_from_verified_log | `compiler_single_conv_qlinear`, `compiler_final_equal_overlay` | tasklog keyword match: \b(extend\|line\|ray\|span\|connect\|between) |
| 011 | 17.400 | 1836 | 163 | seeded_from_verified_log | `compiler_tiny_lut_gather`, `compiler_single_conv_qlinear`, `compiler_final_equal_overlay` | tasklog keyword match: \b(extend\|line\|ray\|span\|connect\|between) |
| 381 | 17.418 | 1908 | 55 | seeded_from_verified_log | `compiler_direct_onehot_gather` | tasklog keyword match: \b(extend\|line\|ray\|span\|connect\|between) |
| 185 | 17.419 | 1651 | 310 | seeded_from_verified_log | `compiler_tiny_lut_gather`, `compiler_single_conv_qlinear`, `compiler_final_equal_overlay`, `compiler_bounded_scan` | tasklog keyword match: \b(extend\|line\|ray\|span\|connect\|between) |
| 124 | 17.423 | 1879 | 74 | seeded_from_verified_log | `compiler_direct_onehot_gather`, `compiler_tiny_lut_gather`, `compiler_final_equal_overlay`, `compiler_bounded_scan` | tasklog keyword match: \b(extend\|line\|ray\|span\|connect\|between) |
| 013 | 17.435 | 1869 | 61 | seeded_from_verified_log | `compiler_direct_output_algebra`, `compiler_tiny_lut_gather`, `compiler_final_equal_overlay` | tasklog keyword match: \b(extend\|line\|ray\|span\|connect\|between) |
| 237 | 17.441 | 1778 | 140 | seeded_from_verified_log | `compiler_single_conv_qlinear`, `compiler_final_equal_overlay`, `compiler_bounded_scan` | tasklog keyword match: \b(extend\|line\|ray\|span\|connect\|between) |
| 213 | 17.457 | 1838 | 49 | seeded_from_verified_log | `compiler_direct_output_algebra`, `compiler_final_equal_overlay` | tasklog keyword match: \b(extend\|line\|ray\|span\|connect\|between) |
| 088 | 17.465 | 1791 | 81 | seeded_from_verified_log | `compiler_direct_output_algebra`, `compiler_final_equal_overlay` | tasklog keyword match: \b(extend\|line\|ray\|span\|connect\|between) |
| 400 | 17.468 | 1800 | 66 | seeded_from_verified_log | `compiler_tiny_lut_gather`, `compiler_roi_pool_crop`, `compiler_bounded_scan` | tasklog keyword match: \b(extend\|line\|ray\|span\|connect\|between) |
| 199 | 17.484 | 1749 | 88 | seeded_from_verified_log | `compiler_direct_output_algebra`, `compiler_final_equal_overlay`, `compiler_sparse_scatter` | tasklog keyword match: \b(extend\|line\|ray\|span\|connect\|between) |
| 051 | 17.503 | 1702 | 101 | seeded_from_verified_log | `compiler_direct_output_algebra`, `compiler_final_equal_overlay` | tasklog keyword match: \b(extend\|line\|ray\|span\|connect\|between) |
| 041 | 17.511 | 1730 | 59 | seeded_from_verified_log | `compiler_single_conv_qlinear`, `compiler_final_equal_overlay` | tasklog keyword match: \b(extend\|line\|ray\|span\|connect\|between) |
| 109 | 17.516 | 1686 | 93 | seeded_from_verified_log | `compiler_tiny_lut_gather`, `compiler_final_equal_overlay` | tasklog keyword match: \b(extend\|line\|ray\|span\|connect\|between) |
| 302 | 17.519 | 1216 | 558 | seeded_from_verified_log | `compiler_single_conv_qlinear`, `compiler_qlinear_uint8` | tasklog keyword match: \b(extend\|line\|ray\|span\|connect\|between) |
| 045 | 17.578 | 1050 | 623 | seeded_from_verified_log | `compiler_direct_output_algebra`, `compiler_bounded_scan` | tasklog keyword match: \b(extend\|line\|ray\|span\|connect\|between) |
| 169 | 17.579 | 1500 | 170 | seeded_from_verified_log | `compiler_tiny_lut_gather`, `compiler_single_conv_qlinear`, `compiler_final_equal_overlay`, `compiler_qlinear_uint8` | tasklog keyword match: \b(extend\|line\|ray\|span\|connect\|between) |
| 061 | 17.581 | 1633 | 35 | seeded_from_verified_log | `compiler_tiny_lut_gather`, `compiler_final_equal_overlay`, `compiler_qlinear_uint8`, `compiler_bounded_scan` | tasklog keyword match: \b(extend\|line\|ray\|span\|connect\|between) |
| 351 | 17.591 | 1632 | 19 | seeded_from_verified_log | `compiler_bounded_scan` | tasklog keyword match: \b(extend\|line\|ray\|span\|connect\|between) |
| 033 | 17.598 | 1606 | 33 | seeded_from_verified_log |  | tasklog keyword match: \b(extend\|line\|ray\|span\|connect\|between) |
| 226 | 17.602 | 1613 | 20 | seeded_from_verified_log | `compiler_final_equal_overlay` | tasklog keyword match: \b(extend\|line\|ray\|span\|connect\|between) |
| 303 | 17.610 | 1598 | 22 | seeded_from_verified_log | `compiler_direct_output_algebra`, `compiler_final_equal_overlay` | tasklog keyword match: \b(extend\|line\|ray\|span\|connect\|between) |
| 369 | 17.623 | 1300 | 299 | seeded_from_verified_log | `compiler_single_conv_qlinear`, `compiler_qlinear_uint8` | tasklog keyword match: \b(extend\|line\|ray\|span\|connect\|between) |
| 342 | 17.632 | 1510 | 75 | seeded_from_verified_log | `compiler_direct_output_algebra`, `compiler_tiny_lut_gather`, `compiler_final_equal_overlay`, `compiler_sparse_scatter` | tasklog keyword match: \b(extend\|line\|ray\|span\|connect\|between) |
| 159 | 17.642 | 1419 | 149 | seeded_from_verified_log | `compiler_direct_output_algebra`, `compiler_tiny_lut_gather`, `compiler_final_equal_overlay` | tasklog keyword match: \b(extend\|line\|ray\|span\|connect\|between) |
| 141 | 17.683 | 1404 | 101 | seeded_from_verified_log | `compiler_direct_output_algebra`, `compiler_final_equal_overlay` | tasklog keyword match: \b(extend\|line\|ray\|span\|connect\|between) |
| 345 | 17.690 | 1431 | 64 | seeded_from_verified_log | `compiler_tiny_lut_gather` | tasklog keyword match: \b(extend\|line\|ray\|span\|connect\|between) |
| 031 | 17.700 | 1434 | 46 | seeded_from_verified_log | `compiler_bounded_scan` | tasklog keyword match: \b(extend\|line\|ray\|span\|connect\|between) |
| 240 | 17.704 | 1393 | 82 | seeded_from_verified_log | `compiler_tiny_lut_gather`, `compiler_final_equal_overlay`, `compiler_bounded_scan` | tasklog keyword match: \b(extend\|line\|ray\|span\|connect\|between) |
| 341 | 17.735 | 1394 | 35 | seeded_from_verified_log | `compiler_final_equal_overlay`, `compiler_bounded_scan` | tasklog keyword match: \b(extend\|line\|ray\|span\|connect\|between) |
| 232 | 17.749 | 0 | 1410 | seeded_from_tasklog_and_inventory | `compiler_tiny_lut_gather`, `compiler_single_conv_qlinear` | tasklog keyword match: \b(extend\|line\|ray\|span\|connect\|between) |
| 119 | 17.781 | 1244 | 121 | seeded_from_verified_log | `compiler_tiny_lut_gather`, `compiler_sparse_scatter`, `compiler_bounded_scan` | tasklog keyword match: \b(extend\|line\|ray\|span\|connect\|between) |
| 123 | 17.797 | 403 | 940 | seeded_from_verified_log | `compiler_direct_output_algebra`, `compiler_tiny_lut_gather`, `compiler_final_equal_overlay` | tasklog keyword match: \b(extend\|line\|ray\|span\|connect\|between) |
| 356 | 17.815 | 1300 | 19 | seeded_from_verified_log | `compiler_single_conv_qlinear`, `compiler_qlinear_uint8`, `compiler_bounded_scan` | tasklog keyword match: \b(extend\|line\|ray\|span\|connect\|between) |
| 293 | 17.839 | 1247 | 41 | seeded_from_verified_log | `compiler_tiny_lut_gather`, `compiler_final_equal_overlay`, `compiler_sparse_scatter` | tasklog keyword match: \b(extend\|line\|ray\|span\|connect\|between) |
| 343 | 17.892 | 1147 | 75 | seeded_from_verified_log | `compiler_tiny_lut_gather`, `compiler_single_conv_qlinear`, `compiler_final_equal_overlay`, `compiler_bounded_scan` | tasklog keyword match: \b(extend\|line\|ray\|span\|connect\|between) |
| 136 | 17.899 | 1179 | 34 | seeded_from_verified_log | `compiler_tiny_lut_gather`, `compiler_final_equal_overlay`, `compiler_bounded_scan` | tasklog keyword match: \b(extend\|line\|ray\|span\|connect\|between) |
| 188 | 17.921 | 1128 | 59 | seeded_from_verified_log | `compiler_single_conv_qlinear`, `compiler_final_equal_overlay`, `compiler_qlinear_uint8`, `compiler_bounded_scan` | tasklog keyword match: \b(extend\|line\|ray\|span\|connect\|between) |
| 244 | 17.928 | 1076 | 103 | seeded_from_verified_log | `compiler_tiny_lut_gather`, `compiler_final_equal_overlay`, `compiler_roi_pool_crop`, `compiler_bounded_scan` | tasklog keyword match: \b(extend\|line\|ray\|span\|connect\|between) |
| 021 | 17.935 | 1072 | 98 | seeded_from_tasklog_and_inventory | `compiler_direct_output_algebra` | tasklog keyword match: \b(extend\|line\|ray\|span\|connect\|between) |
| 048 | 17.956 | 1030 | 116 | seeded_from_verified_log | `compiler_single_conv_qlinear`, `compiler_final_equal_overlay`, `compiler_bounded_scan` | tasklog keyword match: \b(extend\|line\|ray\|span\|connect\|between) |
| 058 | 18.060 | 252 | 781 | seeded_from_verified_log | `compiler_tiny_lut_gather` | tasklog keyword match: \b(extend\|line\|ray\|span\|connect\|between) |
| 288 | 18.135 | 769 | 189 | seeded_from_verified_log | `compiler_tiny_lut_gather`, `compiler_final_equal_overlay`, `compiler_sparse_scatter`, `compiler_bounded_scan` | tasklog keyword match: \b(extend\|line\|ray\|span\|connect\|between) |
| 195 | 18.175 | 882 | 39 | seeded_from_verified_log | `compiler_tiny_lut_gather`, `compiler_bounded_scan` | tasklog keyword match: \b(extend\|line\|ray\|span\|connect\|between) |
| 032 | 18.187 | 0 | 910 | seeded_from_verified_log | `compiler_single_conv_qlinear` | tasklog keyword match: \b(extend\|line\|ray\|span\|connect\|between) |
| 097 | 18.187 | 0 | 910 | seeded_from_verified_log | `compiler_single_conv_qlinear` | tasklog keyword match: \b(extend\|line\|ray\|span\|connect\|between) |
| 120 | 18.187 | 0 | 910 | seeded_from_verified_log | `compiler_single_conv_qlinear` | tasklog keyword match: \b(extend\|line\|ray\|span\|connect\|between) |
| 283 | 18.187 | 0 | 910 | seeded_from_verified_log | `compiler_single_conv_qlinear` | tasklog keyword match: \b(extend\|line\|ray\|span\|connect\|between) |
| 259 | 18.192 | 853 | 52 | seeded_from_verified_log | `compiler_tiny_lut_gather`, `compiler_single_conv_qlinear`, `compiler_final_equal_overlay`, `compiler_bounded_scan` | tasklog keyword match: \b(extend\|line\|ray\|span\|connect\|between) |
| 139 | 18.193 | 765 | 139 | seeded_from_verified_log | `compiler_single_conv_qlinear`, `compiler_qlinear_uint8`, `compiler_bounded_scan` | tasklog keyword match: \b(extend\|line\|ray\|span\|connect\|between) |
| 015 | 18.198 | 0 | 900 | seeded_from_verified_log | `compiler_single_conv_qlinear` | tasklog keyword match: \b(extend\|line\|ray\|span\|connect\|between) |
| 220 | 18.198 | 0 | 900 | seeded_from_verified_log | `compiler_single_conv_qlinear` | tasklog keyword match: \b(extend\|line\|ray\|span\|connect\|between) |
| 230 | 18.198 | 0 | 900 | seeded_from_verified_log | `compiler_tiny_lut_gather`, `compiler_single_conv_qlinear` | tasklog keyword match: \b(extend\|line\|ray\|span\|connect\|between) |
| 327 | 18.219 | 810 | 71 | seeded_from_verified_log | `compiler_single_conv_qlinear`, `compiler_final_equal_overlay` | tasklog keyword match: \b(extend\|line\|ray\|span\|connect\|between) |
| 047 | 18.241 | 826 | 36 | seeded_from_verified_log | `compiler_bounded_scan` | tasklog keyword match: \b(extend\|line\|ray\|span\|connect\|between) |
| 122 | 18.303 | 0 | 810 | seeded_from_verified_log | `compiler_single_conv_qlinear` | tasklog keyword match: \b(extend\|line\|ray\|span\|connect\|between) |
| 153 | 18.322 | 741 | 54 | seeded_from_verified_log | `compiler_direct_output_algebra`, `compiler_tiny_lut_gather`, `compiler_final_equal_overlay`, `compiler_bounded_scan` | tasklog keyword match: \b(extend\|line\|ray\|span\|connect\|between) |
| 043 | 18.323 | 588 | 206 | seeded_from_verified_log | `compiler_direct_output_algebra` | tasklog keyword match: \b(extend\|line\|ray\|span\|connect\|between) |
| 178 | 18.363 | 702 | 61 | seeded_from_verified_log | `compiler_tiny_lut_gather`, `compiler_single_conv_qlinear`, `compiler_final_equal_overlay`, `compiler_bounded_scan` | tasklog keyword match: \b(extend\|line\|ray\|span\|connect\|between) |
| 200 | 18.372 | 570 | 186 | seeded_from_verified_log | `compiler_direct_output_algebra`, `compiler_single_conv_qlinear`, `compiler_final_equal_overlay`, `compiler_bounded_scan` | tasklog keyword match: \b(extend\|line\|ray\|span\|connect\|between) |
| 269 | 18.436 | 685 | 24 | seeded_from_tasklog_and_inventory | `compiler_tiny_lut_gather`, `compiler_roi_pool_crop` | tasklog keyword match: \b(extend\|line\|ray\|span\|connect\|between) |
| 263 | 18.478 | 563 | 117 | seeded_from_verified_log | `compiler_tiny_lut_gather`, `compiler_single_conv_qlinear`, `compiler_final_equal_overlay` | tasklog keyword match: \b(extend\|line\|ray\|span\|connect\|between) |
| 057 | 18.496 | 638 | 30 | seeded_from_verified_log | `compiler_tiny_lut_gather`, `compiler_final_equal_overlay`, `compiler_bounded_scan` | tasklog keyword match: \b(extend\|line\|ray\|span\|connect\|between) |
| 218 | 18.505 | 558 | 104 | seeded_from_verified_log | `compiler_tiny_lut_gather`, `compiler_single_conv_qlinear`, `compiler_final_equal_overlay` | tasklog keyword match: \b(extend\|line\|ray\|span\|connect\|between) |
| 065 | 18.537 | 582 | 59 | seeded_from_verified_log | `compiler_direct_output_algebra`, `compiler_final_equal_overlay`, `compiler_bounded_scan` | tasklog keyword match: \b(extend\|line\|ray\|span\|connect\|between) |
| 100 | 18.703 | 454 | 89 | seeded_from_verified_log | `compiler_direct_output_algebra`, `compiler_final_equal_overlay` | tasklog keyword match: \b(extend\|line\|ray\|span\|connect\|between) |
| 300 | 18.708 | 474 | 66 | seeded_from_verified_log | `compiler_direct_output_algebra`, `compiler_tiny_lut_gather`, `compiler_final_equal_overlay` | tasklog keyword match: \b(extend\|line\|ray\|span\|connect\|between) |
| 147 | 18.723 | 432 | 100 | seeded_from_verified_log | `compiler_single_conv_qlinear`, `compiler_sparse_scatter` | tasklog keyword match: \b(extend\|line\|ray\|span\|connect\|between) |
| 039 | 18.826 | 424 | 56 | seeded_from_verified_log | `compiler_direct_onehot_gather`, `compiler_tiny_lut_gather` | tasklog keyword match: \b(extend\|line\|ray\|span\|connect\|between) |
| 081 | 18.860 | 392 | 72 | seeded_from_verified_log | `compiler_single_conv_qlinear`, `compiler_qlinear_uint8` | tasklog keyword match: \b(extend\|line\|ray\|span\|connect\|between) |
| 282 | 18.900 | 407 | 39 | seeded_from_verified_log | `compiler_single_conv_qlinear`, `compiler_qlinear_uint8` | tasklog keyword match: \b(extend\|line\|ray\|span\|connect\|between) |
| 242 | 18.932 | 372 | 60 | seeded_from_verified_log | `compiler_direct_output_algebra` | tasklog keyword match: \b(extend\|line\|ray\|span\|connect\|between) |
| 272 | 19.004 | 200 | 202 | seeded_from_verified_log | `compiler_single_conv_qlinear` | tasklog keyword match: \b(extend\|line\|ray\|span\|connect\|between) |
| 127 | 19.019 | 204 | 192 | seeded_from_verified_log | `compiler_single_conv_qlinear` | tasklog keyword match: \b(extend\|line\|ray\|span\|connect\|between) |
| 049 | 19.044 | 252 | 134 | seeded_from_verified_log | `compiler_direct_output_algebra`, `compiler_single_conv_qlinear`, `compiler_final_equal_overlay`, `compiler_bounded_scan` | tasklog keyword match: \b(extend\|line\|ray\|span\|connect\|between) |
| 040 | 19.076 | 240 | 134 | seeded_from_verified_log | `compiler_direct_output_algebra`, `compiler_bounded_scan` | tasklog keyword match: \b(extend\|line\|ray\|span\|connect\|between) |
| 290 | 19.128 | 292 | 63 | seeded_from_verified_log | `compiler_direct_output_algebra`, `compiler_tiny_lut_gather`, `compiler_final_equal_overlay` | tasklog keyword match: \b(extend\|line\|ray\|span\|connect\|between) |
| 375 | 19.142 | 108 | 242 | seeded_from_verified_log | `compiler_direct_output_algebra` | tasklog keyword match: \b(extend\|line\|ray\|span\|connect\|between) |
| 296 | 19.201 | 285 | 45 | seeded_from_verified_log | `compiler_final_equal_overlay` | tasklog keyword match: \b(extend\|line\|ray\|span\|connect\|between) |
| 003 | 19.290 | 281 | 21 | seeded_from_verified_log | `compiler_tiny_lut_gather`, `compiler_final_equal_overlay` | tasklog keyword match: \b(extend\|line\|ray\|span\|connect\|between) |
| 128 | 19.296 | 0 | 300 | seeded_from_verified_log | `compiler_single_conv_qlinear` | tasklog keyword match: \b(extend\|line\|ray\|span\|connect\|between) |
| 357 | 19.327 | 258 | 33 | seeded_from_verified_log | `compiler_single_conv_qlinear`, `compiler_final_equal_overlay`, `compiler_qlinear_uint8` | tasklog keyword match: \b(extend\|line\|ray\|span\|connect\|between) |
| 104 | 19.463 | 197 | 57 | seeded_from_verified_log | `compiler_direct_onehot_gather`, `compiler_tiny_lut_gather`, `compiler_single_conv_qlinear`, `compiler_qlinear_uint8` | tasklog keyword match: \b(extend\|line\|ray\|span\|connect\|between) |
| 001 | 19.519 | 0 | 240 | seeded_from_verified_log | `compiler_direct_output_algebra`, `compiler_tiny_lut_gather` | tasklog keyword match: \b(extend\|line\|ray\|span\|connect\|between) |
| 315 | 19.562 | 0 | 230 | seeded_from_verified_log | `compiler_direct_output_algebra` | tasklog keyword match: \b(extend\|line\|ray\|span\|connect\|between) |
| 231 | 19.611 | 180 | 39 | seeded_from_verified_log | `compiler_tiny_lut_gather` | tasklog keyword match: \b(extend\|line\|ray\|span\|connect\|between) |
| 180 | 19.653 | 0 | 210 | seeded_from_verified_log | `compiler_single_conv_qlinear` | tasklog keyword match: \b(extend\|line\|ray\|span\|connect\|between) |
| 258 | 19.925 | 0 | 160 | seeded_from_verified_log | `compiler_single_conv_qlinear` | tasklog keyword match: \b(extend\|line\|ray\|span\|connect\|between) |
| 006 | 19.970 | 126 | 27 | seeded_from_verified_log |  | tasklog keyword match: \b(extend\|line\|ray\|span\|connect\|between) |
| 317 | 20.016 | 128 | 18 | seeded_from_verified_log | `compiler_direct_output_algebra`, `compiler_roi_pool_crop` | tasklog keyword match: \b(extend\|line\|ray\|span\|connect\|between) |
| 007 | 20.156 | 0 | 127 | seeded_from_verified_log | `compiler_direct_output_algebra`, `compiler_tiny_lut_gather` | tasklog keyword match: \b(extend\|line\|ray\|span\|connect\|between) |
| 024 | 20.300 | 0 | 110 | seeded_from_verified_log | `compiler_direct_output_algebra` | tasklog keyword match: \b(extend\|line\|ray\|span\|connect\|between) |
| 073 | 21.311 | 0 | 40 | seeded_from_verified_log | `compiler_direct_output_algebra`, `compiler_single_conv_qlinear` | tasklog keyword match: \b(extend\|line\|ray\|span\|connect\|between) |
| 053 | 21.599 | 0 | 30 | seeded_from_verified_log | `compiler_direct_output_algebra`, `compiler_tiny_lut_gather` | tasklog keyword match: \b(extend\|line\|ray\|span\|connect\|between) |
| 130 | 21.599 | 0 | 30 | seeded_from_verified_log | `compiler_direct_output_algebra`, `compiler_single_conv_qlinear` | tasklog keyword match: \b(extend\|line\|ray\|span\|connect\|between) |

## Known Best Routes

- Pending human review.

## Kill Criteria

- Pending human review.

## Successful Applications

- Pending verification.

## Failed Applications / Walls

- Pending verification.
