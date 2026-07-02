# shape_bbox_rect — Bounding box/rectangle

Shape is a bbox, solid rectangle, frame, or rectangular interior.

## Optimization Question

What is the cheapest verified ONNX family for tasks in this class, and
which semantic preconditions let us avoid full-canvas intermediates?

## Candidate Tasks

| task | pts | mem | params | status | first routes | evidence |
|---:|---:|---:|---:|---|---|---|
| 233 | 14.003 | 59147 | 565 | seeded_from_verified_log | `compiler_direct_onehot_gather`, `compiler_tiny_lut_gather`, `compiler_single_conv_qlinear`, `compiler_final_equal_overlay` | 3/3 output masks are bbox-dense; tasklog keyword match: \b(bbox\|bounding\|rectangle\|rectangular\|box\|frame) |
| 018 | 14.157 | 48196 | 2987 | seeded_from_verified_log | `compiler_tiny_lut_gather`, `compiler_single_conv_qlinear`, `compiler_final_equal_overlay`, `compiler_sparse_scatter` | tasklog keyword match: \b(bbox\|bounding\|rectangle\|rectangular\|box\|frame) |
| 286 | 14.341 | 41822 | 742 | seeded_from_verified_log | `compiler_tiny_lut_gather`, `compiler_single_conv_qlinear`, `compiler_final_equal_overlay`, `compiler_qlinear_uint8` | 1/2 output masks are bbox-dense |
| 366 | 14.497 | 35927 | 490 | seeded_from_verified_log | `compiler_tiny_lut_gather`, `compiler_single_conv_qlinear`, `compiler_final_equal_overlay`, `compiler_sparse_scatter` | 2/3 output masks are bbox-dense; tasklog keyword match: \b(bbox\|bounding\|rectangle\|rectangular\|box\|frame) |
| 158 | 14.528 | 32979 | 2340 | seeded_from_verified_log | `compiler_tiny_lut_gather`, `compiler_single_conv_qlinear`, `compiler_final_equal_overlay`, `compiler_qlinear_uint8` | 3/3 output masks are bbox-dense; tasklog keyword match: \b(bbox\|bounding\|rectangle\|rectangular\|box\|frame) |
| 187 | 14.580 | 32850 | 665 | seeded_from_verified_log | `compiler_single_conv_qlinear`, `compiler_final_equal_overlay`, `compiler_bounded_scan` | 3/3 output masks are bbox-dense; tasklog keyword match: \b(bbox\|bounding\|rectangle\|rectangular\|box\|frame) |
| 209 | 14.620 | 32027 | 185 | seeded_from_verified_log | `compiler_tiny_lut_gather`, `compiler_single_conv_qlinear`, `compiler_final_equal_overlay`, `compiler_qlinear_uint8` | tasklog keyword match: \b(bbox\|bounding\|rectangle\|rectangular\|box\|frame) |
| 191 | 14.624 | 31252 | 841 | seeded_from_verified_log | `compiler_direct_onehot_gather`, `compiler_tiny_lut_gather`, `compiler_single_conv_qlinear`, `compiler_final_equal_overlay` | tasklog keyword match: \b(bbox\|bounding\|rectangle\|rectangular\|box\|frame) |
| 054 | 14.829 | 25885 | 238 | seeded_from_tasklog_and_inventory | `compiler_tiny_lut_gather`, `compiler_single_conv_qlinear`, `compiler_final_equal_overlay`, `compiler_sparse_scatter` | 3/3 output masks are bbox-dense; tasklog keyword match: \b(bbox\|bounding\|rectangle\|rectangular\|box\|frame) |
| 285 | 14.848 | 25080 | 550 | seeded_from_verified_log | `compiler_tiny_lut_gather`, `compiler_single_conv_qlinear`, `compiler_final_equal_overlay`, `compiler_sparse_scatter` | tasklog keyword match: \b(bbox\|bounding\|rectangle\|rectangular\|box\|frame) |
| 002 | 14.896 | 24320 | 125 | seeded_from_verified_log | `compiler_direct_onehot_gather`, `compiler_tiny_lut_gather` | tasklog keyword match: \b(bbox\|bounding\|rectangle\|rectangular\|box\|frame) |
| 364 | 14.956 | 22900 | 113 | seeded_from_verified_log | `compiler_tiny_lut_gather`, `compiler_single_conv_qlinear`, `compiler_final_equal_overlay`, `compiler_qlinear_uint8` | tasklog keyword match: \b(bbox\|bounding\|rectangle\|rectangular\|box\|frame) |
| 243 | 14.972 | 22608 | 44 | seeded_from_verified_log | `compiler_single_conv_qlinear`, `compiler_qlinear_uint8` | 3/3 output masks are bbox-dense; tasklog keyword match: \b(bbox\|bounding\|rectangle\|rectangular\|box\|frame) |
| 367 | 15.051 | 16200 | 4733 | seeded_from_verified_log | `compiler_tiny_lut_gather`, `compiler_single_conv_qlinear`, `compiler_qlinear_uint8` | tasklog keyword match: \b(bbox\|bounding\|rectangle\|rectangular\|box\|frame) |
| 319 | 15.119 | 19280 | 279 | seeded_from_verified_log | `compiler_direct_output_algebra`, `compiler_tiny_lut_gather`, `compiler_final_equal_overlay`, `compiler_bounded_scan` | 4/4 output masks are bbox-dense; tasklog keyword match: \b(bbox\|bounding\|rectangle\|rectangular\|box\|frame) |
| 101 | 15.216 | 16850 | 905 | seeded_from_verified_log | `compiler_tiny_lut_gather`, `compiler_final_equal_overlay`, `compiler_sparse_scatter`, `compiler_bounded_scan` | tasklog keyword match: \b(bbox\|bounding\|rectangle\|rectangular\|box\|frame) |
| 205 | 15.360 | 14734 | 638 | seeded_from_verified_log | `compiler_single_conv_qlinear`, `compiler_final_equal_overlay`, `compiler_roi_pool_crop`, `compiler_bounded_scan` | 3/3 output masks are bbox-dense; tasklog keyword match: \b(bbox\|bounding\|rectangle\|rectangular\|box\|frame) |
| 193 | 15.361 | 15284 | 68 | seeded_from_verified_log | `compiler_tiny_lut_gather`, `compiler_single_conv_qlinear`, `compiler_final_equal_overlay` | 1/3 output masks are bbox-dense |
| 255 | 15.387 | 14663 | 293 | seeded_from_verified_log | `compiler_direct_output_algebra`, `compiler_direct_onehot_gather`, `compiler_single_conv_qlinear`, `compiler_final_equal_overlay` | tasklog keyword match: \b(bbox\|bounding\|rectangle\|rectangular\|box\|frame) |
| 077 | 15.393 | 14760 | 114 | seeded_from_verified_log | `compiler_single_conv_qlinear`, `compiler_qlinear_uint8`, `compiler_bounded_scan` | tasklog keyword match: \b(bbox\|bounding\|rectangle\|rectangular\|box\|frame) |
| 138 | 15.456 | 13789 | 172 | seeded_from_verified_log | `compiler_tiny_lut_gather`, `compiler_single_conv_qlinear`, `compiler_final_equal_overlay`, `compiler_bounded_scan` | 1/3 output masks are bbox-dense; tasklog keyword match: \b(bbox\|bounding\|rectangle\|rectangular\|box\|frame) |
| 110 | 15.468 | 13155 | 634 | seeded_from_verified_log | `compiler_tiny_lut_gather`, `compiler_single_conv_qlinear`, `compiler_final_equal_overlay`, `compiler_bounded_scan` | 3/3 output masks are bbox-dense; tasklog keyword match: \b(bbox\|bounding\|rectangle\|rectangular\|box\|frame) |
| 198 | 15.484 | 13434 | 138 | seeded_from_verified_log | `compiler_tiny_lut_gather`, `compiler_single_conv_qlinear`, `compiler_final_equal_overlay`, `compiler_bounded_scan` | 3/3 output masks are bbox-dense |
| 064 | 15.494 | 13368 | 68 | seeded_from_verified_log | `compiler_tiny_lut_gather`, `compiler_final_equal_overlay`, `compiler_bounded_scan` | 4/4 output masks are bbox-dense; tasklog keyword match: \b(bbox\|bounding\|rectangle\|rectangular\|box\|frame) |
| 145 | 15.511 | 13104 | 111 | seeded_from_verified_log | `compiler_tiny_lut_gather`, `compiler_final_equal_overlay`, `compiler_bounded_scan` | tasklog keyword match: \b(bbox\|bounding\|rectangle\|rectangular\|box\|frame) |
| 023 | 15.616 | 11603 | 291 | seeded_from_verified_log | `compiler_single_conv_qlinear`, `compiler_final_equal_overlay`, `compiler_qlinear_uint8` | tasklog keyword match: \b(bbox\|bounding\|rectangle\|rectangular\|box\|frame) |
| 396 | 15.633 | 11567 | 133 | seeded_from_verified_log | `compiler_single_conv_qlinear`, `compiler_final_equal_overlay`, `compiler_roi_pool_crop`, `compiler_bounded_scan` | tasklog keyword match: \b(bbox\|bounding\|rectangle\|rectangular\|box\|frame) |
| 044 | 15.651 | 11420 | 68 | seeded_from_verified_log | `compiler_direct_onehot_gather`, `compiler_tiny_lut_gather`, `compiler_single_conv_qlinear`, `compiler_final_equal_overlay` | tasklog keyword match: \b(bbox\|bounding\|rectangle\|rectangular\|box\|frame) |
| 338 | 15.712 | 10140 | 666 | seeded_from_verified_log | `compiler_tiny_lut_gather`, `compiler_single_conv_qlinear`, `compiler_final_equal_overlay`, `compiler_qlinear_uint8` | 2/3 output masks are bbox-dense; tasklog keyword match: \b(bbox\|bounding\|rectangle\|rectangular\|box\|frame) |
| 029 | 15.715 | 10736 | 34 | seeded_from_verified_log | `compiler_direct_output_algebra`, `compiler_tiny_lut_gather`, `compiler_final_equal_overlay`, `compiler_bounded_scan` | tasklog keyword match: \b(bbox\|bounding\|rectangle\|rectangular\|box\|frame) |
| 017 | 15.744 | 8448 | 2021 | seeded_from_verified_log | `compiler_tiny_lut_gather`, `compiler_final_equal_overlay` | 3/3 output masks are bbox-dense; tasklog keyword match: \b(bbox\|bounding\|rectangle\|rectangular\|box\|frame) |
| 204 | 15.753 | 9920 | 453 | seeded_from_verified_log | `compiler_single_conv_qlinear`, `compiler_qlinear_uint8`, `compiler_bounded_scan` | tasklog keyword match: \b(bbox\|bounding\|rectangle\|rectangular\|box\|frame) |
| 379 | 15.818 | 9069 | 653 | seeded_from_verified_log | `compiler_single_conv_qlinear`, `compiler_final_equal_overlay`, `compiler_bounded_scan` | tasklog keyword match: \b(bbox\|bounding\|rectangle\|rectangular\|box\|frame) |
| 370 | 15.823 | 8645 | 1024 | seeded_from_verified_log | `compiler_direct_output_algebra`, `compiler_tiny_lut_gather`, `compiler_single_conv_qlinear`, `compiler_final_equal_overlay` | 3/3 output masks are bbox-dense |
| 216 | 15.880 | 9048 | 87 | seeded_from_verified_log | `compiler_direct_onehot_gather`, `compiler_tiny_lut_gather`, `compiler_single_conv_qlinear`, `compiler_final_equal_overlay` | 3/3 output masks are bbox-dense; tasklog keyword match: \b(bbox\|bounding\|rectangle\|rectangular\|box\|frame) |
| 074 | 15.889 | 9000 | 50 | seeded_from_verified_log | `compiler_tiny_lut_gather`, `compiler_single_conv_qlinear`, `compiler_final_equal_overlay` | tasklog keyword match: \b(bbox\|bounding\|rectangle\|rectangular\|box\|frame) |
| 350 | 15.891 | 9012 | 26 | seeded_from_verified_log | `compiler_tiny_lut_gather`, `compiler_bounded_scan` | tasklog keyword match: \b(bbox\|bounding\|rectangle\|rectangular\|box\|frame) |
| 096 | 15.905 | 8108 | 805 | seeded_from_verified_log | `compiler_tiny_lut_gather`, `compiler_final_equal_overlay`, `compiler_sparse_scatter`, `compiler_bounded_scan` | 3/3 output masks are bbox-dense; tasklog keyword match: \b(bbox\|bounding\|rectangle\|rectangular\|box\|frame) |
| 324 | 15.907 | 7308 | 1586 | seeded_from_verified_log | `compiler_single_conv_qlinear`, `compiler_final_equal_overlay`, `compiler_qlinear_uint8`, `compiler_bounded_scan` | 3/3 output masks are bbox-dense |
| 222 | 15.916 | 8736 | 78 | seeded_from_verified_log | `compiler_single_conv_qlinear`, `compiler_final_equal_overlay`, `compiler_qlinear_uint8`, `compiler_bounded_scan` | 3/3 output masks are bbox-dense; tasklog keyword match: \b(bbox\|bounding\|rectangle\|rectangular\|box\|frame) |
| 192 | 15.938 | 8515 | 106 | seeded_from_verified_log | `compiler_single_conv_qlinear`, `compiler_final_equal_overlay` | tasklog keyword match: \b(bbox\|bounding\|rectangle\|rectangular\|box\|frame) |
| 157 | 15.972 | 7983 | 351 | seeded_from_verified_log | `compiler_tiny_lut_gather`, `compiler_final_equal_overlay`, `compiler_sparse_scatter`, `compiler_bounded_scan` | tasklog keyword match: \b(bbox\|bounding\|rectangle\|rectangular\|box\|frame) |
| 377 | 15.975 | 8162 | 150 | seeded_from_verified_log | `compiler_tiny_lut_gather`, `compiler_single_conv_qlinear`, `compiler_final_equal_overlay` | 3/3 output masks are bbox-dense |
| 014 | 16.023 | 7809 | 108 | seeded_from_verified_log | `compiler_tiny_lut_gather`, `compiler_final_equal_overlay`, `compiler_bounded_scan` | tasklog keyword match: \b(bbox\|bounding\|rectangle\|rectangular\|box\|frame) |
| 202 | 16.039 | 7773 | 24 | seeded_from_verified_log | `compiler_single_conv_qlinear`, `compiler_final_equal_overlay`, `compiler_bounded_scan` | 4/4 output masks are bbox-dense |
| 089 | 16.111 | 7092 | 160 | seeded_from_verified_log | `compiler_tiny_lut_gather`, `compiler_single_conv_qlinear`, `compiler_final_equal_overlay`, `compiler_sparse_scatter` | tasklog keyword match: \b(bbox\|bounding\|rectangle\|rectangular\|box\|frame) |
| 174 | 16.130 | 6973 | 142 | seeded_from_verified_log | `compiler_direct_onehot_gather`, `compiler_tiny_lut_gather`, `compiler_final_equal_overlay`, `compiler_bounded_scan` | 1/3 output masks are bbox-dense; tasklog keyword match: \b(bbox\|bounding\|rectangle\|rectangular\|box\|frame) |
| 092 | 16.148 | 6805 | 182 | seeded_from_verified_log | `compiler_direct_output_algebra`, `compiler_final_equal_overlay`, `compiler_sparse_scatter`, `compiler_bounded_scan` | tasklog keyword match: \b(bbox\|bounding\|rectangle\|rectangular\|box\|frame) |
| 182 | 16.176 | 6695 | 98 | seeded_from_verified_log | `compiler_tiny_lut_gather`, `compiler_single_conv_qlinear`, `compiler_final_equal_overlay`, `compiler_qlinear_uint8` | tasklog keyword match: \b(bbox\|bounding\|rectangle\|rectangular\|box\|frame) |
| 368 | 16.264 | 6022 | 203 | seeded_from_verified_log | `compiler_tiny_lut_gather`, `compiler_final_equal_overlay`, `compiler_sparse_scatter` | tasklog keyword match: \b(bbox\|bounding\|rectangle\|rectangular\|box\|frame) |
| 340 | 16.285 | 5812 | 279 | seeded_from_verified_log | `compiler_tiny_lut_gather`, `compiler_final_equal_overlay`, `compiler_qlinear_uint8`, `compiler_bounded_scan` | tasklog keyword match: \b(bbox\|bounding\|rectangle\|rectangular\|box\|frame) |
| 383 | 16.289 | 5974 | 96 | seeded_from_verified_log | `compiler_tiny_lut_gather`, `compiler_single_conv_qlinear`, `compiler_final_equal_overlay`, `compiler_bounded_scan` | tasklog keyword match: \b(bbox\|bounding\|rectangle\|rectangular\|box\|frame) |
| 264 | 16.292 | 5348 | 706 | seeded_from_verified_log | `compiler_direct_output_algebra`, `compiler_tiny_lut_gather`, `compiler_single_conv_qlinear`, `compiler_final_equal_overlay` | 2/2 output masks are bbox-dense |
| 280 | 16.294 | 5286 | 753 | seeded_from_verified_log | `compiler_direct_output_algebra`, `compiler_final_equal_overlay` | tasklog keyword match: \b(bbox\|bounding\|rectangle\|rectangular\|box\|frame) |
| 382 | 16.337 | 5648 | 135 | seeded_from_verified_log | `compiler_direct_output_algebra`, `compiler_tiny_lut_gather`, `compiler_final_equal_overlay`, `compiler_bounded_scan` | tasklog keyword match: \b(bbox\|bounding\|rectangle\|rectangular\|box\|frame) |
| 008 | 16.365 | 5561 | 65 | seeded_from_verified_log | `compiler_tiny_lut_gather`, `compiler_final_equal_overlay`, `compiler_bounded_scan` | tasklog keyword match: \b(bbox\|bounding\|rectangle\|rectangular\|box\|frame) |
| 279 | 16.378 | 5508 | 47 | seeded_from_tasklog_and_inventory | `compiler_tiny_lut_gather`, `compiler_single_conv_qlinear`, `compiler_qlinear_uint8` | 4/4 output masks are bbox-dense; tasklog keyword match: \b(bbox\|bounding\|rectangle\|rectangular\|box\|frame) |
| 278 | 16.391 | 5436 | 47 | seeded_from_verified_log | `compiler_single_conv_qlinear`, `compiler_qlinear_uint8`, `compiler_bounded_scan` | tasklog keyword match: \b(bbox\|bounding\|rectangle\|rectangular\|box\|frame) |
| 085 | 16.409 | 4500 | 881 | seeded_from_verified_log | `compiler_single_conv_qlinear` | tasklog keyword match: \b(bbox\|bounding\|rectangle\|rectangular\|box\|frame) |
| 234 | 16.448 | 5113 | 66 | seeded_from_verified_log | `compiler_direct_output_algebra`, `compiler_tiny_lut_gather`, `compiler_final_equal_overlay`, `compiler_bounded_scan` | 3/3 output masks are bbox-dense; tasklog keyword match: \b(bbox\|bounding\|rectangle\|rectangular\|box\|frame) |
| 004 | 16.454 | 5044 | 100 | seeded_from_verified_log | `compiler_tiny_lut_gather`, `compiler_single_conv_qlinear`, `compiler_final_equal_overlay`, `compiler_qlinear_uint8` | tasklog keyword match: \b(bbox\|bounding\|rectangle\|rectangular\|box\|frame) |
| 387 | 16.471 | 4954 | 105 | seeded_from_verified_log | `compiler_tiny_lut_gather`, `compiler_final_equal_overlay`, `compiler_sparse_scatter` | tasklog keyword match: \b(bbox\|bounding\|rectangle\|rectangular\|box\|frame) |
| 251 | 16.499 | 4752 | 168 | seeded_from_verified_log | `compiler_bounded_scan` | 1/3 output masks are bbox-dense; tasklog keyword match: \b(bbox\|bounding\|rectangle\|rectangular\|box\|frame) |
| 208 | 16.539 | 4612 | 114 | seeded_from_verified_log | `compiler_direct_output_algebra`, `compiler_single_conv_qlinear`, `compiler_final_equal_overlay`, `compiler_qlinear_uint8` | tasklog keyword match: \b(bbox\|bounding\|rectangle\|rectangular\|box\|frame) |
| 361 | 16.547 | 4322 | 366 | seeded_from_verified_log | `compiler_tiny_lut_gather`, `compiler_single_conv_qlinear`, `compiler_final_equal_overlay`, `compiler_roi_pool_crop` | tasklog keyword match: \b(bbox\|bounding\|rectangle\|rectangular\|box\|frame) |
| 196 | 16.580 | 4500 | 38 | seeded_from_verified_log | `compiler_single_conv_qlinear`, `compiler_qlinear_uint8`, `compiler_bounded_scan` | tasklog keyword match: \b(bbox\|bounding\|rectangle\|rectangular\|box\|frame) |
| 358 | 16.587 | 4332 | 174 | seeded_from_verified_log | `compiler_direct_output_algebra`, `compiler_final_equal_overlay`, `compiler_bounded_scan` | tasklog keyword match: \b(bbox\|bounding\|rectangle\|rectangular\|box\|frame) |
| 378 | 16.600 | 4332 | 117 | seeded_from_verified_log | `compiler_direct_output_algebra`, `compiler_tiny_lut_gather`, `compiler_final_equal_overlay`, `compiler_bounded_scan` | tasklog keyword match: \b(bbox\|bounding\|rectangle\|rectangular\|box\|frame) |
| 277 | 16.627 | 4101 | 228 | seeded_from_verified_log | `compiler_tiny_lut_gather`, `compiler_final_equal_overlay`, `compiler_bounded_scan` | tasklog keyword match: \b(bbox\|bounding\|rectangle\|rectangular\|box\|frame) |
| 206 | 16.657 | 4122 | 77 | seeded_from_verified_log | `compiler_single_conv_qlinear`, `compiler_final_equal_overlay`, `compiler_sparse_scatter`, `compiler_bounded_scan` | tasklog keyword match: \b(bbox\|bounding\|rectangle\|rectangular\|box\|frame) |
| 117 | 16.666 | 3922 | 243 | seeded_from_verified_log | `compiler_tiny_lut_gather`, `compiler_single_conv_qlinear`, `compiler_final_equal_overlay`, `compiler_qlinear_uint8` | tasklog keyword match: \b(bbox\|bounding\|rectangle\|rectangular\|box\|frame) |
| 363 | 16.673 | 3839 | 296 | seeded_from_verified_log | `compiler_single_conv_qlinear`, `compiler_final_equal_overlay`, `compiler_bounded_scan` | 1/3 output masks are bbox-dense; tasklog keyword match: \b(bbox\|bounding\|rectangle\|rectangular\|box\|frame) |
| 132 | 16.684 | 3990 | 99 | seeded_from_verified_log | `compiler_tiny_lut_gather`, `compiler_single_conv_qlinear`, `compiler_final_equal_overlay`, `compiler_bounded_scan` | 2/4 output masks are bbox-dense; tasklog keyword match: \b(bbox\|bounding\|rectangle\|rectangular\|box\|frame) |
| 161 | 16.684 | 4055 | 33 | seeded_from_verified_log | `compiler_tiny_lut_gather`, `compiler_final_equal_overlay` | tasklog keyword match: \b(bbox\|bounding\|rectangle\|rectangular\|box\|frame) |
| 107 | 16.687 | 2924 | 1154 | seeded_from_verified_log | `compiler_direct_onehot_gather`, `compiler_tiny_lut_gather`, `compiler_final_equal_overlay`, `compiler_sparse_scatter` | tasklog keyword match: \b(bbox\|bounding\|rectangle\|rectangular\|box\|frame) |
| 365 | 16.699 | 3908 | 120 | seeded_from_verified_log | `compiler_direct_output_algebra`, `compiler_tiny_lut_gather`, `compiler_single_conv_qlinear`, `compiler_final_equal_overlay` | 3/3 output masks are bbox-dense; tasklog keyword match: \b(bbox\|bounding\|rectangle\|rectangular\|box\|frame) |
| 131 | 16.709 | 3312 | 675 | seeded_from_verified_log | `compiler_sparse_scatter`, `compiler_bounded_scan` | 1/3 output masks are bbox-dense; tasklog keyword match: \b(bbox\|bounding\|rectangle\|rectangular\|box\|frame) |
| 069 | 16.728 | 3848 | 64 | seeded_from_verified_log | `compiler_tiny_lut_gather`, `compiler_single_conv_qlinear`, `compiler_final_equal_overlay`, `compiler_bounded_scan` | tasklog keyword match: \b(bbox\|bounding\|rectangle\|rectangular\|box\|frame) |
| 050 | 16.728 | 3825 | 86 | seeded_from_verified_log | `compiler_direct_onehot_gather` | tasklog keyword match: \b(bbox\|bounding\|rectangle\|rectangular\|box\|frame) |
| 359 | 16.740 | 3852 | 15 | seeded_from_verified_log | `compiler_final_equal_overlay`, `compiler_bounded_scan` | 3/3 output masks are bbox-dense |
| 177 | 16.751 | 3692 | 130 | seeded_from_verified_log | `compiler_single_conv_qlinear`, `compiler_final_equal_overlay`, `compiler_roi_pool_crop` | 3/3 output masks are bbox-dense; tasklog keyword match: \b(bbox\|bounding\|rectangle\|rectangular\|box\|frame) |
| 284 | 16.774 | 3082 | 653 | seeded_from_verified_log | `compiler_single_conv_qlinear`, `compiler_sparse_scatter` | tasklog keyword match: \b(bbox\|bounding\|rectangle\|rectangular\|box\|frame) |
| 268 | 16.816 | 3328 | 256 | seeded_from_verified_log | `compiler_final_equal_overlay` | 1/3 output masks are bbox-dense; tasklog keyword match: \b(bbox\|bounding\|rectangle\|rectangular\|box\|frame) |
| 125 | 16.850 | 3434 | 29 | seeded_from_verified_log | `compiler_final_equal_overlay`, `compiler_bounded_scan` | 2/2 output masks are bbox-dense; tasklog keyword match: \b(bbox\|bounding\|rectangle\|rectangular\|box\|frame) |
| 201 | 16.881 | 3202 | 154 | seeded_from_verified_log | `compiler_tiny_lut_gather`, `compiler_single_conv_qlinear`, `compiler_final_equal_overlay`, `compiler_bounded_scan` | tasklog keyword match: \b(bbox\|bounding\|rectangle\|rectangular\|box\|frame) |
| 310 | 16.902 | 3208 | 79 | seeded_from_verified_log | `compiler_direct_output_algebra`, `compiler_tiny_lut_gather`, `compiler_final_equal_overlay` | 3/3 output masks are bbox-dense; tasklog keyword match: \b(bbox\|bounding\|rectangle\|rectangular\|box\|frame) |
| 333 | 16.921 | 2792 | 435 | seeded_from_verified_log | `compiler_direct_output_algebra`, `compiler_tiny_lut_gather`, `compiler_final_equal_overlay`, `compiler_bounded_scan` | tasklog keyword match: \b(bbox\|bounding\|rectangle\|rectangular\|box\|frame) |
| 250 | 16.921 | 3164 | 61 | seeded_from_verified_log | `compiler_tiny_lut_gather`, `compiler_final_equal_overlay`, `compiler_qlinear_uint8`, `compiler_bounded_scan` | tasklog keyword match: \b(bbox\|bounding\|rectangle\|rectangular\|box\|frame) |
| 086 | 16.949 | 2946 | 190 | seeded_from_verified_log | `compiler_tiny_lut_gather`, `compiler_single_conv_qlinear`, `compiler_final_equal_overlay`, `compiler_qlinear_uint8` | 1/3 output masks are bbox-dense; tasklog keyword match: \b(bbox\|bounding\|rectangle\|rectangular\|box\|frame) |
| 035 | 16.954 | 2994 | 128 | seeded_from_verified_log | `compiler_sparse_scatter` | tasklog keyword match: \b(bbox\|bounding\|rectangle\|rectangular\|box\|frame) |
| 042 | 16.958 | 2792 | 318 | seeded_from_verified_log | `compiler_single_conv_qlinear` | tasklog keyword match: \b(bbox\|bounding\|rectangle\|rectangular\|box\|frame) |
| 090 | 16.964 | 3009 | 82 | seeded_from_verified_log | `compiler_bounded_scan` | tasklog keyword match: \b(bbox\|bounding\|rectangle\|rectangular\|box\|frame) |
| 390 | 16.964 | 2624 | 467 | seeded_from_verified_log | `compiler_final_equal_overlay`, `compiler_sparse_scatter` | tasklog keyword match: \b(bbox\|bounding\|rectangle\|rectangular\|box\|frame) |
| 091 | 16.984 | 2853 | 175 | seeded_from_verified_log | `compiler_tiny_lut_gather`, `compiler_single_conv_qlinear`, `compiler_final_equal_overlay` | tasklog keyword match: \b(bbox\|bounding\|rectangle\|rectangular\|box\|frame) |
| 330 | 16.986 | 2800 | 222 | seeded_from_verified_log | `compiler_final_equal_overlay`, `compiler_sparse_scatter`, `compiler_bounded_scan` | tasklog keyword match: \b(bbox\|bounding\|rectangle\|rectangular\|box\|frame) |
| 070 | 16.991 | 2977 | 31 | seeded_from_verified_log | `compiler_tiny_lut_gather`, `compiler_sparse_scatter`, `compiler_bounded_scan` | tasklog keyword match: \b(bbox\|bounding\|rectangle\|rectangular\|box\|frame) |
| 335 | 17.001 | 2288 | 691 | seeded_from_verified_log | `compiler_direct_output_algebra`, `compiler_single_conv_qlinear`, `compiler_final_equal_overlay`, `compiler_bounded_scan` | tasklog keyword match: \b(bbox\|bounding\|rectangle\|rectangular\|box\|frame) |
| 105 | 17.007 | 2854 | 106 | seeded_from_verified_log | `compiler_direct_onehot_gather`, `compiler_tiny_lut_gather`, `compiler_final_equal_overlay`, `compiler_bounded_scan` | tasklog keyword match: \b(bbox\|bounding\|rectangle\|rectangular\|box\|frame) |
| 256 | 17.011 | 2898 | 50 | seeded_from_tasklog_and_inventory | `compiler_final_equal_overlay`, `compiler_bounded_scan` | tasklog keyword match: \b(bbox\|bounding\|rectangle\|rectangular\|box\|frame) |
| 398 | 17.042 | 1666 | 1193 | seeded_from_verified_log | `compiler_direct_onehot_gather`, `compiler_tiny_lut_gather`, `compiler_final_equal_overlay` | tasklog keyword match: \b(bbox\|bounding\|rectangle\|rectangular\|box\|frame) |
| 397 | 17.046 | 2648 | 200 | seeded_from_tasklog_and_inventory | `compiler_tiny_lut_gather`, `compiler_single_conv_qlinear`, `compiler_final_equal_overlay`, `compiler_sparse_scatter` | tasklog keyword match: \b(bbox\|bounding\|rectangle\|rectangular\|box\|frame) |
| 224 | 17.052 | 2460 | 371 | seeded_from_verified_log | `compiler_single_conv_qlinear`, `compiler_bounded_scan` | tasklog keyword match: \b(bbox\|bounding\|rectangle\|rectangular\|box\|frame) |
| 394 | 17.063 | 1871 | 929 | seeded_from_verified_log | `compiler_tiny_lut_gather`, `compiler_final_equal_overlay`, `compiler_bounded_scan` | 3/3 output masks are bbox-dense |
| 019 | 17.063 | 2709 | 89 | seeded_from_verified_log | `compiler_tiny_lut_gather`, `compiler_single_conv_qlinear`, `compiler_final_equal_overlay`, `compiler_qlinear_uint8` | tasklog keyword match: \b(bbox\|bounding\|rectangle\|rectangular\|box\|frame) |
| 154 | 17.077 | 2661 | 99 | seeded_from_verified_log | `compiler_tiny_lut_gather`, `compiler_final_equal_overlay` | tasklog keyword match: \b(bbox\|bounding\|rectangle\|rectangular\|box\|frame) |
| 354 | 17.080 | 2674 | 77 | seeded_from_verified_log | `compiler_single_conv_qlinear`, `compiler_final_equal_overlay`, `compiler_bounded_scan` | tasklog keyword match: \b(bbox\|bounding\|rectangle\|rectangular\|box\|frame) |
| 245 | 17.083 | 2604 | 139 | seeded_from_verified_log | `compiler_tiny_lut_gather`, `compiler_bounded_scan` | tasklog keyword match: \b(bbox\|bounding\|rectangle\|rectangular\|box\|frame) |
| 190 | 17.089 | 2480 | 247 | seeded_from_verified_log | `compiler_single_conv_qlinear`, `compiler_qlinear_uint8`, `compiler_bounded_scan` | tasklog keyword match: \b(bbox\|bounding\|rectangle\|rectangular\|box\|frame) |
| 355 | 17.093 | 2688 | 27 | seeded_from_verified_log | `compiler_final_equal_overlay`, `compiler_bounded_scan` | 4/4 output masks are bbox-dense; tasklog keyword match: \b(bbox\|bounding\|rectangle\|rectangular\|box\|frame) |
| 094 | 17.094 | 2662 | 51 | seeded_from_verified_log | `compiler_single_conv_qlinear` | 2/2 output masks are bbox-dense; tasklog keyword match: \b(bbox\|bounding\|rectangle\|rectangular\|box\|frame) |
| 071 | 17.100 | 2643 | 55 | seeded_from_verified_log | `compiler_tiny_lut_gather`, `compiler_final_equal_overlay`, `compiler_bounded_scan` | tasklog keyword match: \b(bbox\|bounding\|rectangle\|rectangular\|box\|frame) |
| 170 | 17.158 | 2216 | 329 | seeded_from_verified_log | `compiler_tiny_lut_gather`, `compiler_single_conv_qlinear`, `compiler_final_equal_overlay`, `compiler_bounded_scan` | tasklog keyword match: \b(bbox\|bounding\|rectangle\|rectangular\|box\|frame) |
| 062 | 17.162 | 2465 | 71 | seeded_from_verified_log | `compiler_tiny_lut_gather`, `compiler_final_equal_overlay`, `compiler_bounded_scan` | 4/4 output masks are bbox-dense |
| 102 | 17.165 | 2414 | 113 | seeded_from_verified_log | `compiler_single_conv_qlinear`, `compiler_qlinear_uint8`, `compiler_bounded_scan` | tasklog keyword match: \b(bbox\|bounding\|rectangle\|rectangular\|box\|frame) |
| 184 | 17.168 | 2460 | 60 | seeded_from_verified_log | `compiler_direct_output_algebra`, `compiler_single_conv_qlinear`, `compiler_final_equal_overlay`, `compiler_bounded_scan` | 3/3 output masks are bbox-dense |
| 388 | 17.178 | 2278 | 218 | seeded_from_verified_log | `compiler_final_equal_overlay`, `compiler_sparse_scatter`, `compiler_bounded_scan` | tasklog keyword match: \b(bbox\|bounding\|rectangle\|rectangular\|box\|frame) |
| 238 | 17.178 | 2051 | 444 | seeded_from_verified_log | `compiler_direct_output_algebra`, `compiler_tiny_lut_gather`, `compiler_final_equal_overlay`, `compiler_bounded_scan` | tasklog keyword match: \b(bbox\|bounding\|rectangle\|rectangular\|box\|frame) |
| 030 | 17.218 | 2248 | 148 | seeded_from_verified_log | `compiler_tiny_lut_gather`, `compiler_bounded_scan` | 1/3 output masks are bbox-dense |
| 281 | 17.223 | 2321 | 65 | seeded_from_verified_log | `compiler_direct_output_algebra`, `compiler_final_equal_overlay`, `compiler_bounded_scan` | 3/3 output masks are bbox-dense; tasklog keyword match: \b(bbox\|bounding\|rectangle\|rectangular\|box\|frame) |
| 022 | 17.258 | 2004 | 298 | seeded_from_verified_log | `compiler_direct_output_algebra`, `compiler_final_equal_overlay`, `compiler_roi_pool_crop` | 3/3 output masks are bbox-dense |
| 134 | 17.266 | 2250 | 34 | seeded_from_verified_log | `compiler_tiny_lut_gather`, `compiler_bounded_scan` | tasklog keyword match: \b(bbox\|bounding\|rectangle\|rectangular\|box\|frame) |
| 012 | 17.275 | 2076 | 188 | seeded_from_verified_log | `compiler_tiny_lut_gather`, `compiler_final_equal_overlay`, `compiler_sparse_scatter` | tasklog keyword match: \b(bbox\|bounding\|rectangle\|rectangular\|box\|frame) |
| 336 | 17.298 | 1188 | 1024 | seeded_from_verified_log | `compiler_tiny_lut_gather`, `compiler_sparse_scatter` | 1/2 output masks are bbox-dense; tasklog keyword match: \b(bbox\|bounding\|rectangle\|rectangular\|box\|frame) |
| 036 | 17.314 | 2140 | 37 | seeded_from_verified_log | `compiler_final_equal_overlay`, `compiler_bounded_scan` | 1/2 output masks are bbox-dense; tasklog keyword match: \b(bbox\|bounding\|rectangle\|rectangular\|box\|frame) |
| 273 | 17.322 | 2109 | 51 | seeded_from_verified_log | `compiler_tiny_lut_gather`, `compiler_bounded_scan` | tasklog keyword match: \b(bbox\|bounding\|rectangle\|rectangular\|box\|frame) |
| 325 | 17.335 | 2010 | 123 | seeded_from_verified_log | `compiler_single_conv_qlinear`, `compiler_final_equal_overlay`, `compiler_qlinear_uint8`, `compiler_bounded_scan` | tasklog keyword match: \b(bbox\|bounding\|rectangle\|rectangular\|box\|frame) |
| 063 | 17.394 | 1728 | 283 | seeded_from_verified_log | `compiler_single_conv_qlinear`, `compiler_final_equal_overlay` | 1/3 output masks are bbox-dense |
| 275 | 17.394 | 1374 | 637 | seeded_from_verified_log | `compiler_direct_output_algebra`, `compiler_single_conv_qlinear`, `compiler_final_equal_overlay`, `compiler_bounded_scan` | tasklog keyword match: \b(bbox\|bounding\|rectangle\|rectangular\|box\|frame) |
| 011 | 17.400 | 1836 | 163 | seeded_from_verified_log | `compiler_tiny_lut_gather`, `compiler_single_conv_qlinear`, `compiler_final_equal_overlay` | tasklog keyword match: \b(bbox\|bounding\|rectangle\|rectangular\|box\|frame) |
| 287 | 17.402 | 1699 | 295 | seeded_from_verified_log | `compiler_direct_output_algebra`, `compiler_tiny_lut_gather`, `compiler_final_equal_overlay` | 4/4 output masks are bbox-dense; tasklog keyword match: \b(bbox\|bounding\|rectangle\|rectangular\|box\|frame) |
| 381 | 17.418 | 1908 | 55 | seeded_from_verified_log | `compiler_direct_onehot_gather` | tasklog keyword match: \b(bbox\|bounding\|rectangle\|rectangular\|box\|frame) |
| 185 | 17.419 | 1651 | 310 | seeded_from_verified_log | `compiler_tiny_lut_gather`, `compiler_single_conv_qlinear`, `compiler_final_equal_overlay`, `compiler_bounded_scan` | tasklog keyword match: \b(bbox\|bounding\|rectangle\|rectangular\|box\|frame) |
| 124 | 17.423 | 1879 | 74 | seeded_from_verified_log | `compiler_direct_onehot_gather`, `compiler_tiny_lut_gather`, `compiler_final_equal_overlay`, `compiler_bounded_scan` | 1/3 output masks are bbox-dense |
| 392 | 17.424 | 1602 | 349 | seeded_from_verified_log | `compiler_direct_output_algebra`, `compiler_tiny_lut_gather`, `compiler_final_equal_overlay`, `compiler_bounded_scan` | 3/3 output masks are bbox-dense; tasklog keyword match: \b(bbox\|bounding\|rectangle\|rectangular\|box\|frame) |
| 084 | 17.440 | 1837 | 83 | seeded_from_verified_log | `compiler_sparse_scatter` | 1/3 output masks are bbox-dense |
| 099 | 17.445 | 1633 | 278 | seeded_from_verified_log | `compiler_tiny_lut_gather`, `compiler_final_equal_overlay`, `compiler_sparse_scatter`, `compiler_bounded_scan` | 1/3 output masks are bbox-dense; tasklog keyword match: \b(bbox\|bounding\|rectangle\|rectangular\|box\|frame) |
| 213 | 17.457 | 1838 | 49 | seeded_from_verified_log | `compiler_direct_output_algebra`, `compiler_final_equal_overlay` | 3/3 output masks are bbox-dense; tasklog keyword match: \b(bbox\|bounding\|rectangle\|rectangular\|box\|frame) |
| 088 | 17.465 | 1791 | 81 | seeded_from_verified_log | `compiler_direct_output_algebra`, `compiler_final_equal_overlay` | tasklog keyword match: \b(bbox\|bounding\|rectangle\|rectangular\|box\|frame) |
| 400 | 17.468 | 1800 | 66 | seeded_from_verified_log | `compiler_tiny_lut_gather`, `compiler_roi_pool_crop`, `compiler_bounded_scan` | 2/3 output masks are bbox-dense |
| 199 | 17.484 | 1749 | 88 | seeded_from_verified_log | `compiler_direct_output_algebra`, `compiler_final_equal_overlay`, `compiler_sparse_scatter` | 1/3 output masks are bbox-dense |
| 051 | 17.503 | 1702 | 101 | seeded_from_verified_log | `compiler_direct_output_algebra`, `compiler_final_equal_overlay` | tasklog keyword match: \b(bbox\|bounding\|rectangle\|rectangular\|box\|frame) |
| 112 | 17.519 | 1616 | 158 | seeded_from_verified_log | `compiler_direct_output_algebra`, `compiler_tiny_lut_gather` | tasklog keyword match: \b(bbox\|bounding\|rectangle\|rectangular\|box\|frame) |
| 302 | 17.519 | 1216 | 558 | seeded_from_verified_log | `compiler_single_conv_qlinear`, `compiler_qlinear_uint8` | tasklog keyword match: \b(bbox\|bounding\|rectangle\|rectangular\|box\|frame) |
| 156 | 17.536 | 1697 | 47 | seeded_from_tasklog_and_inventory | `compiler_direct_output_algebra`, `compiler_tiny_lut_gather`, `compiler_final_equal_overlay` | tasklog keyword match: \b(bbox\|bounding\|rectangle\|rectangular\|box\|frame) |
| 169 | 17.579 | 1500 | 170 | seeded_from_verified_log | `compiler_tiny_lut_gather`, `compiler_single_conv_qlinear`, `compiler_final_equal_overlay`, `compiler_qlinear_uint8` | tasklog keyword match: \b(bbox\|bounding\|rectangle\|rectangular\|box\|frame) |
| 061 | 17.581 | 1633 | 35 | seeded_from_verified_log | `compiler_tiny_lut_gather`, `compiler_final_equal_overlay`, `compiler_qlinear_uint8`, `compiler_bounded_scan` | 4/4 output masks are bbox-dense; tasklog keyword match: \b(bbox\|bounding\|rectangle\|rectangular\|box\|frame) |
| 351 | 17.591 | 1632 | 19 | seeded_from_verified_log | `compiler_bounded_scan` | 3/3 output masks are bbox-dense |
| 226 | 17.602 | 1613 | 20 | seeded_from_verified_log | `compiler_final_equal_overlay` | tasklog keyword match: \b(bbox\|bounding\|rectangle\|rectangular\|box\|frame) |
| 303 | 17.610 | 1598 | 22 | seeded_from_verified_log | `compiler_direct_output_algebra`, `compiler_final_equal_overlay` | 1/3 output masks are bbox-dense; tasklog keyword match: \b(bbox\|bounding\|rectangle\|rectangular\|box\|frame) |
| 369 | 17.623 | 1300 | 299 | seeded_from_verified_log | `compiler_single_conv_qlinear`, `compiler_qlinear_uint8` | 3/3 output masks are bbox-dense |
| 342 | 17.632 | 1510 | 75 | seeded_from_verified_log | `compiler_direct_output_algebra`, `compiler_tiny_lut_gather`, `compiler_final_equal_overlay`, `compiler_sparse_scatter` | 3/3 output masks are bbox-dense; tasklog keyword match: \b(bbox\|bounding\|rectangle\|rectangular\|box\|frame) |
| 159 | 17.642 | 1419 | 149 | seeded_from_verified_log | `compiler_direct_output_algebra`, `compiler_tiny_lut_gather`, `compiler_final_equal_overlay` | 1/3 output masks are bbox-dense; tasklog keyword match: \b(bbox\|bounding\|rectangle\|rectangular\|box\|frame) |
| 308 | 17.695 | 1385 | 103 | seeded_from_verified_log | `compiler_direct_output_algebra`, `compiler_final_equal_overlay`, `compiler_sparse_scatter`, `compiler_bounded_scan` | 3/3 output masks are bbox-dense; tasklog keyword match: \b(bbox\|bounding\|rectangle\|rectangular\|box\|frame) |
| 031 | 17.700 | 1434 | 46 | seeded_from_verified_log | `compiler_bounded_scan` | tasklog keyword match: \b(bbox\|bounding\|rectangle\|rectangular\|box\|frame) |
| 232 | 17.749 | 0 | 1410 | seeded_from_tasklog_and_inventory | `compiler_tiny_lut_gather`, `compiler_single_conv_qlinear` | 1/3 output masks are bbox-dense |
| 297 | 17.764 | 1350 | 39 | seeded_from_tasklog_and_inventory | `compiler_tiny_lut_gather`, `compiler_final_equal_overlay`, `compiler_bounded_scan` | 3/3 output masks are bbox-dense |
| 123 | 17.797 | 403 | 940 | seeded_from_verified_log | `compiler_direct_output_algebra`, `compiler_tiny_lut_gather`, `compiler_final_equal_overlay` | 3/3 output masks are bbox-dense |
| 160 | 17.804 | 1164 | 170 | seeded_from_tasklog_and_inventory | `compiler_single_conv_qlinear`, `compiler_qlinear_uint8`, `compiler_bounded_scan` | tasklog keyword match: \b(bbox\|bounding\|rectangle\|rectangular\|box\|frame) |
| 078 | 17.835 | 1260 | 33 | seeded_from_tasklog_and_inventory |  | 1/3 output masks are bbox-dense |
| 293 | 17.839 | 1247 | 41 | seeded_from_verified_log | `compiler_tiny_lut_gather`, `compiler_final_equal_overlay`, `compiler_sparse_scatter` | tasklog keyword match: \b(bbox\|bounding\|rectangle\|rectangular\|box\|frame) |
| 175 | 17.856 | 330 | 937 | seeded_from_verified_log | `compiler_tiny_lut_gather`, `compiler_final_equal_overlay`, `compiler_bounded_scan` | 3/3 output masks are bbox-dense; tasklog keyword match: \b(bbox\|bounding\|rectangle\|rectangular\|box\|frame) |
| 346 | 17.866 | 1224 | 30 | seeded_from_verified_log | `compiler_tiny_lut_gather`, `compiler_single_conv_qlinear`, `compiler_final_equal_overlay`, `compiler_qlinear_uint8` | 4/4 output masks are bbox-dense |
| 183 | 17.885 | 1008 | 222 | seeded_from_verified_log | `compiler_direct_output_algebra`, `compiler_final_equal_overlay` | tasklog keyword match: \b(bbox\|bounding\|rectangle\|rectangular\|box\|frame) |
| 343 | 17.892 | 1147 | 75 | seeded_from_verified_log | `compiler_tiny_lut_gather`, `compiler_single_conv_qlinear`, `compiler_final_equal_overlay`, `compiler_bounded_scan` | 1/3 output masks are bbox-dense |
| 136 | 17.899 | 1179 | 34 | seeded_from_verified_log | `compiler_tiny_lut_gather`, `compiler_final_equal_overlay`, `compiler_bounded_scan` | tasklog keyword match: \b(bbox\|bounding\|rectangle\|rectangular\|box\|frame) |
| 271 | 17.921 | 1141 | 46 | seeded_from_verified_log | `compiler_single_conv_qlinear`, `compiler_qlinear_uint8` | 4/4 output masks are bbox-dense; tasklog keyword match: \b(bbox\|bounding\|rectangle\|rectangular\|box\|frame) |
| 188 | 17.921 | 1128 | 59 | seeded_from_verified_log | `compiler_single_conv_qlinear`, `compiler_final_equal_overlay`, `compiler_qlinear_uint8`, `compiler_bounded_scan` | 3/3 output masks are bbox-dense; tasklog keyword match: \b(bbox\|bounding\|rectangle\|rectangular\|box\|frame) |
| 021 | 17.935 | 1072 | 98 | seeded_from_tasklog_and_inventory | `compiler_direct_output_algebra` | 3/3 output masks are bbox-dense; tasklog keyword match: \b(bbox\|bounding\|rectangle\|rectangular\|box\|frame) |
| 048 | 17.956 | 1030 | 116 | seeded_from_verified_log | `compiler_single_conv_qlinear`, `compiler_final_equal_overlay`, `compiler_bounded_scan` | 3/6 output masks are bbox-dense; tasklog keyword match: \b(bbox\|bounding\|rectangle\|rectangular\|box\|frame) |
| 301 | 17.960 | 1077 | 64 | seeded_from_verified_log | `compiler_tiny_lut_gather`, `compiler_final_equal_overlay`, `compiler_bounded_scan` | tasklog keyword match: \b(bbox\|bounding\|rectangle\|rectangular\|box\|frame) |
| 329 | 18.035 | 1029 | 30 | seeded_from_verified_log | `compiler_final_equal_overlay`, `compiler_sparse_scatter` | 2/3 output masks are bbox-dense; tasklog keyword match: \b(bbox\|bounding\|rectangle\|rectangular\|box\|frame) |
| 115 | 18.050 | 980 | 63 | seeded_from_verified_log | `compiler_direct_output_algebra`, `compiler_bounded_scan` | 3/3 output masks are bbox-dense; tasklog keyword match: \b(bbox\|bounding\|rectangle\|rectangular\|box\|frame) |
| 225 | 18.060 | 926 | 107 | seeded_from_verified_log | `compiler_tiny_lut_gather`, `compiler_single_conv_qlinear`, `compiler_final_equal_overlay`, `compiler_bounded_scan` | tasklog keyword match: \b(bbox\|bounding\|rectangle\|rectangular\|box\|frame) |
| 058 | 18.060 | 252 | 781 | seeded_from_verified_log | `compiler_tiny_lut_gather` | tasklog keyword match: \b(bbox\|bounding\|rectangle\|rectangular\|box\|frame) |
| 060 | 18.082 | 0 | 1010 | seeded_from_tasklog_and_inventory | `compiler_single_conv_qlinear` | 2/2 output masks are bbox-dense |
| 228 | 18.131 | 849 | 113 | seeded_from_verified_log | `compiler_direct_output_algebra`, `compiler_tiny_lut_gather`, `compiler_final_equal_overlay`, `compiler_sparse_scatter` | tasklog keyword match: \b(bbox\|bounding\|rectangle\|rectangular\|box\|frame) |
| 194 | 18.145 | 900 | 49 | seeded_from_verified_log | `compiler_tiny_lut_gather` | 2/3 output masks are bbox-dense |
| 195 | 18.175 | 882 | 39 | seeded_from_verified_log | `compiler_tiny_lut_gather`, `compiler_bounded_scan` | tasklog keyword match: \b(bbox\|bounding\|rectangle\|rectangular\|box\|frame) |
| 189 | 18.181 | 866 | 49 | seeded_from_verified_log | `compiler_single_conv_qlinear`, `compiler_final_equal_overlay`, `compiler_roi_pool_crop` | tasklog keyword match: \b(bbox\|bounding\|rectangle\|rectangular\|box\|frame) |
| 120 | 18.187 | 0 | 910 | seeded_from_verified_log | `compiler_single_conv_qlinear` | tasklog keyword match: \b(bbox\|bounding\|rectangle\|rectangular\|box\|frame) |
| 171 | 18.187 | 0 | 910 | seeded_from_tasklog_and_inventory | `compiler_single_conv_qlinear` | 2/4 output masks are bbox-dense; tasklog keyword match: \b(bbox\|bounding\|rectangle\|rectangular\|box\|frame) |
| 283 | 18.187 | 0 | 910 | seeded_from_verified_log | `compiler_single_conv_qlinear` | tasklog keyword match: \b(bbox\|bounding\|rectangle\|rectangular\|box\|frame) |
| 294 | 18.187 | 0 | 910 | seeded_from_verified_log | `compiler_single_conv_qlinear` | 1/2 output masks are bbox-dense |
| 344 | 18.187 | 0 | 910 | seeded_from_tasklog_and_inventory | `compiler_single_conv_qlinear` | tasklog keyword match: \b(bbox\|bounding\|rectangle\|rectangular\|box\|frame) |
| 259 | 18.192 | 853 | 52 | seeded_from_verified_log | `compiler_tiny_lut_gather`, `compiler_single_conv_qlinear`, `compiler_final_equal_overlay`, `compiler_bounded_scan` | 1/3 output masks are bbox-dense; tasklog keyword match: \b(bbox\|bounding\|rectangle\|rectangular\|box\|frame) |
| 139 | 18.193 | 765 | 139 | seeded_from_verified_log | `compiler_single_conv_qlinear`, `compiler_qlinear_uint8`, `compiler_bounded_scan` | tasklog keyword match: \b(bbox\|bounding\|rectangle\|rectangular\|box\|frame) |
| 114 | 18.198 | 0 | 900 | operator_evidence_only_needs_human_review | `compiler_single_conv_qlinear` | 2/3 output masks are bbox-dense |
| 220 | 18.198 | 0 | 900 | seeded_from_verified_log | `compiler_single_conv_qlinear` | 2/4 output masks are bbox-dense |
| 106 | 18.233 | 776 | 93 | seeded_from_verified_log | `compiler_tiny_lut_gather`, `compiler_single_conv_qlinear`, `compiler_sparse_scatter` | 3/3 output masks are bbox-dense |
| 143 | 18.269 | 591 | 247 | seeded_from_verified_log | `compiler_direct_output_algebra`, `compiler_final_equal_overlay` | tasklog keyword match: \b(bbox\|bounding\|rectangle\|rectangular\|box\|frame) |
| 122 | 18.303 | 0 | 810 | seeded_from_verified_log | `compiler_single_conv_qlinear` | tasklog keyword match: \b(bbox\|bounding\|rectangle\|rectangular\|box\|frame) |
| 153 | 18.322 | 741 | 54 | seeded_from_verified_log | `compiler_direct_output_algebra`, `compiler_tiny_lut_gather`, `compiler_final_equal_overlay`, `compiler_bounded_scan` | 3/3 output masks are bbox-dense; tasklog keyword match: \b(bbox\|bounding\|rectangle\|rectangular\|box\|frame) |
| 214 | 18.358 | 726 | 41 | seeded_from_verified_log | `compiler_tiny_lut_gather`, `compiler_single_conv_qlinear`, `compiler_final_equal_overlay` | 3/3 output masks are bbox-dense |
| 178 | 18.363 | 702 | 61 | seeded_from_verified_log | `compiler_tiny_lut_gather`, `compiler_single_conv_qlinear`, `compiler_final_equal_overlay`, `compiler_bounded_scan` | 5/5 output masks are bbox-dense |
| 289 | 18.391 | 618 | 124 | seeded_from_verified_log | `compiler_direct_output_algebra`, `compiler_final_equal_overlay`, `compiler_bounded_scan` | tasklog keyword match: \b(bbox\|bounding\|rectangle\|rectangular\|box\|frame) |
| 372 | 18.435 | 0 | 710 | seeded_from_verified_log | `compiler_single_conv_qlinear` | tasklog keyword match: \b(bbox\|bounding\|rectangle\|rectangular\|box\|frame) |
| 263 | 18.478 | 563 | 117 | seeded_from_verified_log | `compiler_tiny_lut_gather`, `compiler_single_conv_qlinear`, `compiler_final_equal_overlay` | 2/4 output masks are bbox-dense |
| 057 | 18.496 | 638 | 30 | seeded_from_verified_log | `compiler_tiny_lut_gather`, `compiler_final_equal_overlay`, `compiler_bounded_scan` | tasklog keyword match: \b(bbox\|bounding\|rectangle\|rectangular\|box\|frame) |
| 218 | 18.505 | 558 | 104 | seeded_from_verified_log | `compiler_tiny_lut_gather`, `compiler_single_conv_qlinear`, `compiler_final_equal_overlay` | 3/3 output masks are bbox-dense; tasklog keyword match: \b(bbox\|bounding\|rectangle\|rectangular\|box\|frame) |
| 065 | 18.537 | 582 | 59 | seeded_from_verified_log | `compiler_direct_output_algebra`, `compiler_final_equal_overlay`, `compiler_bounded_scan` | 3/3 output masks are bbox-dense |
| 305 | 18.545 | 296 | 340 | seeded_from_verified_log | `compiler_tiny_lut_gather` | 3/3 output masks are bbox-dense; tasklog keyword match: \b(bbox\|bounding\|rectangle\|rectangular\|box\|frame) |
| 152 | 18.556 | 504 | 125 | operator_evidence_only_needs_human_review | `compiler_single_conv_qlinear`, `compiler_sparse_scatter` | 4/4 output masks are bbox-dense |
| 068 | 18.565 | 500 | 123 | seeded_from_verified_log | `compiler_direct_output_algebra`, `compiler_final_equal_overlay`, `compiler_sparse_scatter` | 3/3 output masks are bbox-dense; tasklog keyword match: \b(bbox\|bounding\|rectangle\|rectangular\|box\|frame) |
| 253 | 18.616 | 540 | 52 | seeded_from_verified_log | `compiler_direct_output_algebra`, `compiler_final_equal_overlay` | tasklog keyword match: \b(bbox\|bounding\|rectangle\|rectangular\|box\|frame) |
| 059 | 18.677 | 265 | 292 | seeded_from_verified_log | `compiler_direct_output_algebra`, `compiler_final_equal_overlay`, `compiler_bounded_scan` | tasklog keyword match: \b(bbox\|bounding\|rectangle\|rectangular\|box\|frame) |
| 100 | 18.703 | 454 | 89 | seeded_from_verified_log | `compiler_direct_output_algebra`, `compiler_final_equal_overlay` | 3/3 output masks are bbox-dense; tasklog keyword match: \b(bbox\|bounding\|rectangle\|rectangular\|box\|frame) |
| 300 | 18.708 | 474 | 66 | seeded_from_verified_log | `compiler_direct_output_algebra`, `compiler_tiny_lut_gather`, `compiler_final_equal_overlay` | 2/4 output masks are bbox-dense; tasklog keyword match: \b(bbox\|bounding\|rectangle\|rectangular\|box\|frame) |
| 111 | 18.718 | 528 | 7 | seeded_from_verified_log | `compiler_tiny_lut_gather`, `compiler_bounded_scan` | tasklog keyword match: \b(bbox\|bounding\|rectangle\|rectangular\|box\|frame) |
| 146 | 18.795 | 388 | 107 | seeded_from_verified_log | `compiler_single_conv_qlinear`, `compiler_final_equal_overlay` | 4/4 output masks are bbox-dense; tasklog keyword match: \b(bbox\|bounding\|rectangle\|rectangular\|box\|frame) |
| 039 | 18.826 | 424 | 56 | seeded_from_verified_log | `compiler_direct_onehot_gather`, `compiler_tiny_lut_gather` | tasklog keyword match: \b(bbox\|bounding\|rectangle\|rectangular\|box\|frame) |
| 247 | 18.847 | 408 | 62 | seeded_from_tasklog_and_inventory | `compiler_direct_output_algebra`, `compiler_final_equal_overlay`, `compiler_bounded_scan` | 6/6 output masks are bbox-dense |
| 352 | 18.869 | 0 | 460 | seeded_from_verified_log | `compiler_single_conv_qlinear` | 1/3 output masks are bbox-dense |
| 282 | 18.900 | 407 | 39 | seeded_from_verified_log | `compiler_single_conv_qlinear`, `compiler_qlinear_uint8` | tasklog keyword match: \b(bbox\|bounding\|rectangle\|rectangular\|box\|frame) |
| 242 | 18.932 | 372 | 60 | seeded_from_verified_log | `compiler_direct_output_algebra` | 3/3 output masks are bbox-dense |
| 316 | 18.946 | 388 | 38 | seeded_from_verified_log | `compiler_direct_output_algebra`, `compiler_tiny_lut_gather`, `compiler_final_equal_overlay` | 3/3 output masks are bbox-dense |
| 203 | 18.986 | 408 | 1 | seeded_from_tasklog_and_inventory | `compiler_tiny_lut_gather`, `compiler_final_equal_overlay`, `compiler_bounded_scan` | 4/4 output masks are bbox-dense |
| 257 | 18.986 | 0 | 409 | seeded_from_verified_log | `compiler_single_conv_qlinear` | 5/6 output masks are bbox-dense; tasklog keyword match: \b(bbox\|bounding\|rectangle\|rectangular\|box\|frame) |
| 127 | 19.019 | 204 | 192 | seeded_from_verified_log | `compiler_single_conv_qlinear` | 4/4 output masks are bbox-dense |
| 049 | 19.044 | 252 | 134 | seeded_from_verified_log | `compiler_direct_output_algebra`, `compiler_single_conv_qlinear`, `compiler_final_equal_overlay`, `compiler_bounded_scan` | 5/5 output masks are bbox-dense; tasklog keyword match: \b(bbox\|bounding\|rectangle\|rectangular\|box\|frame) |
| 290 | 19.128 | 292 | 63 | seeded_from_verified_log | `compiler_direct_output_algebra`, `compiler_tiny_lut_gather`, `compiler_final_equal_overlay` | 3/3 output masks are bbox-dense; tasklog keyword match: \b(bbox\|bounding\|rectangle\|rectangular\|box\|frame) |
| 095 | 19.156 | 245 | 100 | seeded_from_verified_log | `compiler_single_conv_qlinear`, `compiler_qlinear_uint8` | tasklog keyword match: \b(bbox\|bounding\|rectangle\|rectangular\|box\|frame) |
| 353 | 19.189 | 272 | 62 | seeded_from_verified_log | `compiler_direct_output_algebra`, `compiler_sparse_scatter` | tasklog keyword match: \b(bbox\|bounding\|rectangle\|rectangular\|box\|frame) |
| 128 | 19.296 | 0 | 300 | seeded_from_verified_log | `compiler_single_conv_qlinear` | tasklog keyword match: \b(bbox\|bounding\|rectangle\|rectangular\|box\|frame) |
| 321 | 19.296 | 0 | 300 | seeded_from_verified_log | `compiler_single_conv_qlinear` | 4/5 output masks are bbox-dense; tasklog keyword match: \b(bbox\|bounding\|rectangle\|rectangular\|box\|frame) |
| 357 | 19.327 | 258 | 33 | seeded_from_verified_log | `compiler_single_conv_qlinear`, `compiler_final_equal_overlay`, `compiler_qlinear_uint8` | 3/3 output masks are bbox-dense |
| 142 | 19.358 | 144 | 138 | seeded_from_verified_log | `compiler_direct_output_algebra` | 2/3 output masks are bbox-dense |
| 104 | 19.463 | 197 | 57 | seeded_from_verified_log | `compiler_direct_onehot_gather`, `compiler_tiny_lut_gather`, `compiler_single_conv_qlinear`, `compiler_qlinear_uint8` | tasklog keyword match: \b(bbox\|bounding\|rectangle\|rectangular\|box\|frame) |
| 001 | 19.519 | 0 | 240 | seeded_from_verified_log | `compiler_direct_output_algebra`, `compiler_tiny_lut_gather` | tasklog keyword match: \b(bbox\|bounding\|rectangle\|rectangular\|box\|frame) |
| 229 | 19.519 | 212 | 28 | operator_evidence_only_needs_human_review | `compiler_final_equal_overlay` | 4/4 output masks are bbox-dense |
| 315 | 19.562 | 0 | 230 | seeded_from_verified_log | `compiler_direct_output_algebra` | tasklog keyword match: \b(bbox\|bounding\|rectangle\|rectangular\|box\|frame) |
| 052 | 19.575 | 213 | 14 | seeded_from_verified_log | `compiler_roi_pool_crop`, `compiler_bounded_scan` | 4/4 output masks are bbox-dense |
| 274 | 19.579 | 146 | 80 | seeded_from_verified_log | `compiler_direct_output_algebra`, `compiler_final_equal_overlay` | 5/6 output masks are bbox-dense |
| 231 | 19.611 | 180 | 39 | seeded_from_verified_log | `compiler_tiny_lut_gather` | 3/3 output masks are bbox-dense |
| 038 | 19.639 | 163 | 50 | seeded_from_verified_log | `compiler_direct_output_algebra` | 3/3 output masks are bbox-dense; tasklog keyword match: \b(bbox\|bounding\|rectangle\|rectangular\|box\|frame) |
| 180 | 19.653 | 0 | 210 | seeded_from_verified_log | `compiler_single_conv_qlinear` | 5/5 output masks are bbox-dense |
| 322 | 19.702 | 144 | 56 | operator_evidence_only_needs_human_review | `compiler_single_conv_qlinear`, `compiler_final_equal_overlay`, `compiler_bounded_scan` | 2/3 output masks are bbox-dense |
| 026 | 19.702 | 0 | 200 | operator_evidence_only_needs_human_review | `compiler_single_conv_qlinear` | 3/5 output masks are bbox-dense |
| 261 | 19.702 | 0 | 200 | seeded_from_verified_log | `compiler_single_conv_qlinear` | 3/3 output masks are bbox-dense |
| 313 | 19.743 | 0 | 192 | operator_evidence_only_needs_human_review | `compiler_direct_output_algebra` | 3/3 output masks are bbox-dense |
| 235 | 19.864 | 111 | 59 | operator_evidence_only_needs_human_review | `compiler_direct_onehot_gather`, `compiler_final_equal_overlay` | 4/4 output masks are bbox-dense |
| 028 | 19.876 | 0 | 168 | seeded_from_verified_log | `compiler_direct_output_algebra` | tasklog keyword match: \b(bbox\|bounding\|rectangle\|rectangular\|box\|frame) |
| 391 | 19.931 | 135 | 24 | seeded_from_verified_log | `compiler_tiny_lut_gather`, `compiler_final_equal_overlay` | 4/4 output masks are bbox-dense |
| 150 | 19.937 | 128 | 30 | seeded_from_verified_log | `compiler_tiny_lut_gather` | 3/3 output masks are bbox-dense |
| 155 | 19.937 | 128 | 30 | seeded_from_verified_log | `compiler_tiny_lut_gather` | 3/3 output masks are bbox-dense |
| 006 | 19.970 | 126 | 27 | seeded_from_verified_log |  | 1/3 output masks are bbox-dense |
| 167 | 20.003 | 20 | 128 | seeded_from_verified_log | `compiler_direct_output_algebra` | 2/5 output masks are bbox-dense |
| 393 | 20.030 | 125 | 19 | operator_evidence_only_needs_human_review | `compiler_direct_output_algebra`, `compiler_final_equal_overlay` | 3/3 output masks are bbox-dense |
| 339 | 20.058 | 40 | 100 | seeded_from_verified_log | `compiler_direct_output_algebra` | 4/4 output masks are bbox-dense |
| 298 | 20.095 | 120 | 15 | seeded_from_verified_log | `compiler_direct_output_algebra` | 2/3 output masks are bbox-dense |
| 007 | 20.156 | 0 | 127 | seeded_from_verified_log | `compiler_direct_output_algebra`, `compiler_tiny_lut_gather` | 3/3 output masks are bbox-dense |
| 262 | 20.337 | 90 | 16 | seeded_from_verified_log | `compiler_direct_output_algebra` | 4/4 output masks are bbox-dense |
| 186 | 20.365 | 80 | 23 | seeded_from_verified_log | `compiler_direct_output_algebra`, `compiler_bounded_scan` | 7/10 output masks are bbox-dense |
| 318 | 20.395 | 0 | 100 | seeded_from_verified_log | `compiler_direct_output_algebra`, `compiler_single_conv_qlinear` | 2/4 output masks are bbox-dense |
| 399 | 20.489 | 71 | 20 | seeded_from_verified_log | `compiler_direct_output_algebra`, `compiler_tiny_lut_gather`, `compiler_final_equal_overlay` | tasklog keyword match: \b(bbox\|bounding\|rectangle\|rectangular\|box\|frame) |
| 129 | 20.618 | 80 | 0 | seeded_from_verified_log | `compiler_direct_output_algebra` | 3/3 output masks are bbox-dense |
| 291 | 20.780 | 40 | 28 | seeded_from_verified_log | `compiler_direct_output_algebra` | 3/3 output masks are bbox-dense; tasklog keyword match: \b(bbox\|bounding\|rectangle\|rectangular\|box\|frame) |
| 103 | 20.906 | 36 | 24 | operator_evidence_only_needs_human_review | `compiler_direct_output_algebra`, `compiler_final_equal_overlay`, `compiler_bounded_scan` | 6/6 output masks are bbox-dense |
| 373 | 20.906 | 0 | 60 | seeded_from_verified_log | `compiler_direct_output_algebra` | 2/2 output masks are bbox-dense |
| 067 | 21.150 | 39 | 8 | seeded_from_verified_log | `compiler_direct_output_algebra` | 2/3 output masks are bbox-dense |
| 056 | 21.474 | 34 | 0 | seeded_from_verified_log | `compiler_direct_output_algebra`, `compiler_final_equal_overlay` | 7/7 output masks are bbox-dense |
| 053 | 21.599 | 0 | 30 | seeded_from_verified_log | `compiler_direct_output_algebra`, `compiler_tiny_lut_gather` | 2/4 output masks are bbox-dense |
| 116 | 21.599 | 0 | 30 | operator_evidence_only_needs_human_review | `compiler_direct_output_algebra`, `compiler_tiny_lut_gather` | 4/4 output masks are bbox-dense |
| 130 | 21.599 | 0 | 30 | seeded_from_verified_log | `compiler_direct_output_algebra`, `compiler_single_conv_qlinear` | tasklog keyword match: \b(bbox\|bounding\|rectangle\|rectangular\|box\|frame) |
| 164 | 21.599 | 0 | 30 | operator_evidence_only_needs_human_review | `compiler_direct_output_algebra`, `compiler_tiny_lut_gather` | 4/4 output masks are bbox-dense |
| 172 | 21.599 | 0 | 30 | operator_evidence_only_needs_human_review | `compiler_direct_output_algebra`, `compiler_tiny_lut_gather` | 4/4 output masks are bbox-dense |
| 210 | 21.599 | 0 | 30 | operator_evidence_only_needs_human_review | `compiler_direct_output_algebra`, `compiler_tiny_lut_gather` | 1/3 output masks are bbox-dense |
| 311 | 21.599 | 0 | 30 | operator_evidence_only_needs_human_review | `compiler_direct_output_algebra`, `compiler_tiny_lut_gather` | 1/3 output masks are bbox-dense |
| 326 | 21.599 | 0 | 30 | seeded_from_verified_log | `compiler_direct_output_algebra` | 2/3 output masks are bbox-dense |
| 166 | 21.822 | 0 | 24 | seeded_from_verified_log | `compiler_direct_output_algebra` | 3/3 output masks are bbox-dense; tasklog keyword match: \b(bbox\|bounding\|rectangle\|rectangular\|box\|frame) |
| 312 | 22.004 | 0 | 20 | seeded_from_verified_log | `compiler_direct_output_algebra` | tasklog keyword match: \b(bbox\|bounding\|rectangle\|rectangular\|box\|frame) |
| 016 | 22.697 | 0 | 10 | operator_evidence_only_needs_human_review | `compiler_direct_output_algebra`, `compiler_tiny_lut_gather` | 4/4 output masks are bbox-dense |
| 276 | 22.697 | 0 | 10 | operator_evidence_only_needs_human_review | `compiler_direct_output_algebra`, `compiler_tiny_lut_gather` | 3/3 output masks are bbox-dense |
| 309 | 22.697 | 0 | 10 | operator_evidence_only_needs_human_review | `compiler_direct_output_algebra`, `compiler_tiny_lut_gather` | 3/3 output masks are bbox-dense |
| 337 | 22.697 | 0 | 10 | operator_evidence_only_needs_human_review | `compiler_direct_output_algebra`, `compiler_tiny_lut_gather` | 3/3 output masks are bbox-dense |
| 087 | 23.391 | 0 | 5 | operator_evidence_only_needs_human_review | `compiler_direct_output_algebra`, `compiler_roi_pool_crop` | 4/4 output masks are bbox-dense |
| 307 | 23.391 | 0 | 5 | operator_evidence_only_needs_human_review | `compiler_direct_output_algebra`, `compiler_roi_pool_crop` | 2/3 output masks are bbox-dense |
| 179 | 25.000 | 0 | 0 | operator_evidence_only_needs_human_review | `compiler_direct_output_algebra` | 4/4 output masks are bbox-dense |

## Known Best Routes

- Pending human review.

## Kill Criteria

- Pending human review.

## Successful Applications

- Pending verification.

## Failed Applications / Walls

- Pending verification.
