# color_recolor_rule — Recolour by rule

Input colours are mapped to different output colours.

## Optimization Question

What is the cheapest verified ONNX family for tasks in this class, and
which semantic preconditions let us avoid full-canvas intermediates?

## Candidate Tasks

| task | pts | mem | params | status | first routes | evidence |
|---:|---:|---:|---:|---|---|---|
| 158 | 14.528 | 32979 | 2340 | seeded_from_verified_log | `compiler_tiny_lut_gather`, `compiler_single_conv_qlinear`, `compiler_final_equal_overlay`, `compiler_qlinear_uint8` | same nonzero mask but different colours |
| 118 | 14.559 | 30849 | 3387 | seeded_from_verified_log | `compiler_direct_output_algebra`, `compiler_direct_onehot_gather`, `compiler_tiny_lut_gather`, `compiler_single_conv_qlinear` | output introduces colours not present in input; same nonzero mask but different colours |
| 191 | 14.624 | 31252 | 841 | seeded_from_verified_log | `compiler_direct_onehot_gather`, `compiler_tiny_lut_gather`, `compiler_single_conv_qlinear`, `compiler_final_equal_overlay` | tasklog keyword match: \b(recolou?r\|recolor\|mapped? to\|colour rank\|color rank) |
| 054 | 14.829 | 25885 | 238 | seeded_from_tasklog_and_inventory | `compiler_tiny_lut_gather`, `compiler_single_conv_qlinear`, `compiler_final_equal_overlay`, `compiler_sparse_scatter` | same nonzero mask but different colours |
| 002 | 14.896 | 24320 | 125 | seeded_from_verified_log | `compiler_direct_onehot_gather`, `compiler_tiny_lut_gather` | output introduces colours not present in input |
| 364 | 14.956 | 22900 | 113 | seeded_from_verified_log | `compiler_tiny_lut_gather`, `compiler_single_conv_qlinear`, `compiler_final_equal_overlay`, `compiler_qlinear_uint8` | same nonzero mask but different colours; tasklog keyword match: \b(recolou?r\|recolor\|mapped? to\|colour rank\|color rank) |
| 243 | 14.972 | 22608 | 44 | seeded_from_verified_log | `compiler_single_conv_qlinear`, `compiler_qlinear_uint8` | tasklog keyword match: \b(recolou?r\|recolor\|mapped? to\|colour rank\|color rank) |
| 367 | 15.051 | 16200 | 4733 | seeded_from_verified_log | `compiler_tiny_lut_gather`, `compiler_single_conv_qlinear`, `compiler_qlinear_uint8` | output introduces colours not present in input |
| 349 | 15.102 | 19800 | 90 | seeded_from_verified_log | `compiler_tiny_lut_gather`, `compiler_single_conv_qlinear`, `compiler_final_equal_overlay`, `compiler_qlinear_uint8` | output introduces colours not present in input |
| 219 | 15.195 | 18033 | 93 | seeded_from_verified_log | `compiler_tiny_lut_gather`, `compiler_final_equal_overlay`, `compiler_bounded_scan` | output introduces colours not present in input; tasklog keyword match: \b(recolou?r\|recolor\|mapped? to\|colour rank\|color rank) |
| 255 | 15.387 | 14663 | 293 | seeded_from_verified_log | `compiler_direct_output_algebra`, `compiler_direct_onehot_gather`, `compiler_single_conv_qlinear`, `compiler_final_equal_overlay` | output introduces colours not present in input |
| 077 | 15.393 | 14760 | 114 | seeded_from_verified_log | `compiler_single_conv_qlinear`, `compiler_qlinear_uint8`, `compiler_bounded_scan` | output introduces colours not present in input; same nonzero mask but different colours |
| 110 | 15.468 | 13155 | 634 | seeded_from_verified_log | `compiler_tiny_lut_gather`, `compiler_single_conv_qlinear`, `compiler_final_equal_overlay`, `compiler_bounded_scan` | tasklog keyword match: \b(recolou?r\|recolor\|mapped? to\|colour rank\|color rank) |
| 064 | 15.494 | 13368 | 68 | seeded_from_verified_log | `compiler_tiny_lut_gather`, `compiler_final_equal_overlay`, `compiler_bounded_scan` | same nonzero mask but different colours |
| 145 | 15.511 | 13104 | 111 | seeded_from_verified_log | `compiler_tiny_lut_gather`, `compiler_final_equal_overlay`, `compiler_bounded_scan` | output introduces colours not present in input |
| 023 | 15.616 | 11603 | 291 | seeded_from_verified_log | `compiler_single_conv_qlinear`, `compiler_final_equal_overlay`, `compiler_qlinear_uint8` | same nonzero mask but different colours |
| 396 | 15.633 | 11567 | 133 | seeded_from_verified_log | `compiler_single_conv_qlinear`, `compiler_final_equal_overlay`, `compiler_roi_pool_crop`, `compiler_bounded_scan` | tasklog keyword match: \b(recolou?r\|recolor\|mapped? to\|colour rank\|color rank) |
| 204 | 15.753 | 9920 | 453 | seeded_from_verified_log | `compiler_single_conv_qlinear`, `compiler_qlinear_uint8`, `compiler_bounded_scan` | output introduces colours not present in input |
| 370 | 15.823 | 8645 | 1024 | seeded_from_verified_log | `compiler_direct_output_algebra`, `compiler_tiny_lut_gather`, `compiler_single_conv_qlinear`, `compiler_final_equal_overlay` | same nonzero mask but different colours; tasklog keyword match: \b(recolou?r\|recolor\|mapped? to\|colour rank\|color rank) |
| 074 | 15.889 | 9000 | 50 | seeded_from_verified_log | `compiler_tiny_lut_gather`, `compiler_single_conv_qlinear`, `compiler_final_equal_overlay` | tasklog keyword match: \b(recolou?r\|recolor\|mapped? to\|colour rank\|color rank) |
| 350 | 15.891 | 9012 | 26 | seeded_from_verified_log | `compiler_tiny_lut_gather`, `compiler_bounded_scan` | output introduces colours not present in input |
| 324 | 15.907 | 7308 | 1586 | seeded_from_verified_log | `compiler_single_conv_qlinear`, `compiler_final_equal_overlay`, `compiler_qlinear_uint8`, `compiler_bounded_scan` | same nonzero mask but different colours; tasklog keyword match: \b(recolou?r\|recolor\|mapped? to\|colour rank\|color rank) |
| 192 | 15.938 | 8515 | 106 | seeded_from_verified_log | `compiler_single_conv_qlinear`, `compiler_final_equal_overlay` | tasklog keyword match: \b(recolou?r\|recolor\|mapped? to\|colour rank\|color rank) |
| 202 | 16.039 | 7773 | 24 | seeded_from_verified_log | `compiler_single_conv_qlinear`, `compiler_final_equal_overlay`, `compiler_bounded_scan` | tasklog keyword match: \b(recolou?r\|recolor\|mapped? to\|colour rank\|color rank) |
| 174 | 16.130 | 6973 | 142 | seeded_from_verified_log | `compiler_direct_onehot_gather`, `compiler_tiny_lut_gather`, `compiler_final_equal_overlay`, `compiler_bounded_scan` | tasklog keyword match: \b(recolou?r\|recolor\|mapped? to\|colour rank\|color rank) |
| 182 | 16.176 | 6695 | 98 | seeded_from_verified_log | `compiler_tiny_lut_gather`, `compiler_single_conv_qlinear`, `compiler_final_equal_overlay`, `compiler_qlinear_uint8` | same nonzero mask but different colours; tasklog keyword match: \b(recolou?r\|recolor\|mapped? to\|colour rank\|color rank) |
| 368 | 16.264 | 6022 | 203 | seeded_from_verified_log | `compiler_tiny_lut_gather`, `compiler_final_equal_overlay`, `compiler_sparse_scatter` | same nonzero mask but different colours; tasklog keyword match: \b(recolou?r\|recolor\|mapped? to\|colour rank\|color rank) |
| 264 | 16.292 | 5348 | 706 | seeded_from_verified_log | `compiler_direct_output_algebra`, `compiler_tiny_lut_gather`, `compiler_single_conv_qlinear`, `compiler_final_equal_overlay` | tasklog keyword match: \b(recolou?r\|recolor\|mapped? to\|colour rank\|color rank) |
| 148 | 16.323 | 5759 | 110 | seeded_from_verified_log | `compiler_final_equal_overlay`, `compiler_sparse_scatter`, `compiler_bounded_scan` | output introduces colours not present in input |
| 279 | 16.378 | 5508 | 47 | seeded_from_tasklog_and_inventory | `compiler_tiny_lut_gather`, `compiler_single_conv_qlinear`, `compiler_qlinear_uint8` | output introduces colours not present in input; same nonzero mask but different colours |
| 278 | 16.391 | 5436 | 47 | seeded_from_verified_log | `compiler_single_conv_qlinear`, `compiler_qlinear_uint8`, `compiler_bounded_scan` | output introduces colours not present in input |
| 234 | 16.448 | 5113 | 66 | seeded_from_verified_log | `compiler_direct_output_algebra`, `compiler_tiny_lut_gather`, `compiler_final_equal_overlay`, `compiler_bounded_scan` | tasklog keyword match: \b(recolou?r\|recolor\|mapped? to\|colour rank\|color rank) |
| 387 | 16.471 | 4954 | 105 | seeded_from_verified_log | `compiler_tiny_lut_gather`, `compiler_final_equal_overlay`, `compiler_sparse_scatter` | output introduces colours not present in input |
| 251 | 16.499 | 4752 | 168 | seeded_from_verified_log | `compiler_bounded_scan` | output introduces colours not present in input; tasklog keyword match: \b(recolou?r\|recolor\|mapped? to\|colour rank\|color rank) |
| 196 | 16.580 | 4500 | 38 | seeded_from_verified_log | `compiler_single_conv_qlinear`, `compiler_qlinear_uint8`, `compiler_bounded_scan` | output introduces colours not present in input; same nonzero mask but different colours |
| 265 | 16.619 | 4328 | 34 | seeded_from_verified_log | `compiler_tiny_lut_gather`, `compiler_final_equal_overlay`, `compiler_sparse_scatter`, `compiler_bounded_scan` | output introduces colours not present in input |
| 277 | 16.627 | 4101 | 228 | seeded_from_verified_log | `compiler_tiny_lut_gather`, `compiler_final_equal_overlay`, `compiler_bounded_scan` | same nonzero mask but different colours |
| 107 | 16.687 | 2924 | 1154 | seeded_from_verified_log | `compiler_direct_onehot_gather`, `compiler_tiny_lut_gather`, `compiler_final_equal_overlay`, `compiler_sparse_scatter` | output introduces colours not present in input |
| 162 | 16.689 | 4024 | 44 | seeded_from_verified_log | `compiler_single_conv_qlinear`, `compiler_qlinear_uint8`, `compiler_bounded_scan` | output introduces colours not present in input |
| 131 | 16.709 | 3312 | 675 | seeded_from_verified_log | `compiler_sparse_scatter`, `compiler_bounded_scan` | output introduces colours not present in input |
| 069 | 16.728 | 3848 | 64 | seeded_from_verified_log | `compiler_tiny_lut_gather`, `compiler_single_conv_qlinear`, `compiler_final_equal_overlay`, `compiler_bounded_scan` | tasklog keyword match: \b(recolou?r\|recolor\|mapped? to\|colour rank\|color rank) |
| 050 | 16.728 | 3825 | 86 | seeded_from_verified_log | `compiler_direct_onehot_gather` | output introduces colours not present in input |
| 359 | 16.740 | 3852 | 15 | seeded_from_verified_log | `compiler_final_equal_overlay`, `compiler_bounded_scan` | same nonzero mask but different colours |
| 268 | 16.816 | 3328 | 256 | seeded_from_verified_log | `compiler_final_equal_overlay` | output introduces colours not present in input |
| 125 | 16.850 | 3434 | 29 | seeded_from_verified_log | `compiler_final_equal_overlay`, `compiler_bounded_scan` | output introduces colours not present in input; same nonzero mask but different colours |
| 035 | 16.954 | 2994 | 128 | seeded_from_verified_log | `compiler_sparse_scatter` | same nonzero mask but different colours |
| 042 | 16.958 | 2792 | 318 | seeded_from_verified_log | `compiler_single_conv_qlinear` | output introduces colours not present in input |
| 055 | 16.963 | 3030 | 62 | seeded_from_verified_log | `compiler_tiny_lut_gather`, `compiler_final_equal_overlay`, `compiler_qlinear_uint8`, `compiler_bounded_scan` | output introduces colours not present in input |
| 090 | 16.964 | 3009 | 82 | seeded_from_verified_log | `compiler_bounded_scan` | output introduces colours not present in input; tasklog keyword match: \b(recolou?r\|recolor\|mapped? to\|colour rank\|color rank) |
| 330 | 16.986 | 2800 | 222 | seeded_from_verified_log | `compiler_final_equal_overlay`, `compiler_sparse_scatter`, `compiler_bounded_scan` | same nonzero mask but different colours; tasklog keyword match: \b(recolou?r\|recolor\|mapped? to\|colour rank\|color rank) |
| 070 | 16.991 | 2977 | 31 | seeded_from_verified_log | `compiler_tiny_lut_gather`, `compiler_sparse_scatter`, `compiler_bounded_scan` | output introduces colours not present in input; same nonzero mask but different colours |
| 335 | 17.001 | 2288 | 691 | seeded_from_verified_log | `compiler_direct_output_algebra`, `compiler_single_conv_qlinear`, `compiler_final_equal_overlay`, `compiler_bounded_scan` | output introduces colours not present in input |
| 105 | 17.007 | 2854 | 106 | seeded_from_verified_log | `compiler_direct_onehot_gather`, `compiler_tiny_lut_gather`, `compiler_final_equal_overlay`, `compiler_bounded_scan` | output introduces colours not present in input |
| 256 | 17.011 | 2898 | 50 | seeded_from_tasklog_and_inventory | `compiler_final_equal_overlay`, `compiler_bounded_scan` | output introduces colours not present in input |
| 397 | 17.046 | 2648 | 200 | seeded_from_tasklog_and_inventory | `compiler_tiny_lut_gather`, `compiler_single_conv_qlinear`, `compiler_final_equal_overlay`, `compiler_sparse_scatter` | output introduces colours not present in input |
| 019 | 17.063 | 2709 | 89 | seeded_from_verified_log | `compiler_tiny_lut_gather`, `compiler_single_conv_qlinear`, `compiler_final_equal_overlay`, `compiler_qlinear_uint8` | output introduces colours not present in input |
| 246 | 17.080 | 2334 | 419 | seeded_from_verified_log | `compiler_single_conv_qlinear`, `compiler_final_equal_overlay` | output introduces colours not present in input |
| 354 | 17.080 | 2674 | 77 | seeded_from_verified_log | `compiler_single_conv_qlinear`, `compiler_final_equal_overlay`, `compiler_bounded_scan` | same nonzero mask but different colours; tasklog keyword match: \b(recolou?r\|recolor\|mapped? to\|colour rank\|color rank) |
| 094 | 17.094 | 2662 | 51 | seeded_from_verified_log | `compiler_single_conv_qlinear` | output introduces colours not present in input; same nonzero mask but different colours |
| 062 | 17.162 | 2465 | 71 | seeded_from_verified_log | `compiler_tiny_lut_gather`, `compiler_final_equal_overlay`, `compiler_bounded_scan` | tasklog keyword match: \b(recolou?r\|recolor\|mapped? to\|colour rank\|color rank) |
| 102 | 17.165 | 2414 | 113 | seeded_from_verified_log | `compiler_single_conv_qlinear`, `compiler_qlinear_uint8`, `compiler_bounded_scan` | output introduces colours not present in input; tasklog keyword match: \b(recolou?r\|recolor\|mapped? to\|colour rank\|color rank) |
| 388 | 17.178 | 2278 | 218 | seeded_from_verified_log | `compiler_final_equal_overlay`, `compiler_sparse_scatter`, `compiler_bounded_scan` | output introduces colours not present in input |
| 238 | 17.178 | 2051 | 444 | seeded_from_verified_log | `compiler_direct_output_algebra`, `compiler_tiny_lut_gather`, `compiler_final_equal_overlay`, `compiler_bounded_scan` | tasklog keyword match: \b(recolou?r\|recolor\|mapped? to\|colour rank\|color rank) |
| 046 | 17.252 | 2221 | 97 | seeded_from_tasklog_and_inventory | `compiler_tiny_lut_gather`, `compiler_single_conv_qlinear`, `compiler_final_equal_overlay`, `compiler_bounded_scan` | tasklog keyword match: \b(recolou?r\|recolor\|mapped? to\|colour rank\|color rank) |
| 374 | 17.258 | 2262 | 42 | seeded_from_verified_log | `compiler_final_equal_overlay`, `compiler_bounded_scan` | same nonzero mask but different colours; tasklog keyword match: \b(recolou?r\|recolor\|mapped? to\|colour rank\|color rank) |
| 134 | 17.266 | 2250 | 34 | seeded_from_verified_log | `compiler_tiny_lut_gather`, `compiler_bounded_scan` | tasklog keyword match: \b(recolou?r\|recolor\|mapped? to\|colour rank\|color rank) |
| 336 | 17.298 | 1188 | 1024 | seeded_from_verified_log | `compiler_tiny_lut_gather`, `compiler_sparse_scatter` | output introduces colours not present in input |
| 323 | 17.314 | 1521 | 656 | seeded_from_verified_log | `compiler_single_conv_qlinear`, `compiler_qlinear_uint8` | output introduces colours not present in input |
| 273 | 17.322 | 2109 | 51 | seeded_from_verified_log | `compiler_tiny_lut_gather`, `compiler_bounded_scan` | output introduces colours not present in input |
| 348 | 17.353 | 1938 | 157 | seeded_from_verified_log | `compiler_final_equal_overlay` | output introduces colours not present in input |
| 063 | 17.394 | 1728 | 283 | seeded_from_verified_log | `compiler_single_conv_qlinear`, `compiler_final_equal_overlay` | output introduces colours not present in input |
| 287 | 17.402 | 1699 | 295 | seeded_from_verified_log | `compiler_direct_output_algebra`, `compiler_tiny_lut_gather`, `compiler_final_equal_overlay` | same nonzero mask but different colours; tasklog keyword match: \b(recolou?r\|recolor\|mapped? to\|colour rank\|color rank) |
| 381 | 17.418 | 1908 | 55 | seeded_from_verified_log | `compiler_direct_onehot_gather` | output introduces colours not present in input |
| 084 | 17.440 | 1837 | 83 | seeded_from_verified_log | `compiler_sparse_scatter` | output introduces colours not present in input |
| 088 | 17.465 | 1791 | 81 | seeded_from_verified_log | `compiler_direct_output_algebra`, `compiler_final_equal_overlay` | tasklog keyword match: \b(recolou?r\|recolor\|mapped? to\|colour rank\|color rank) |
| 199 | 17.484 | 1749 | 88 | seeded_from_verified_log | `compiler_direct_output_algebra`, `compiler_final_equal_overlay`, `compiler_sparse_scatter` | output introduces colours not present in input |
| 302 | 17.519 | 1216 | 558 | seeded_from_verified_log | `compiler_single_conv_qlinear`, `compiler_qlinear_uint8` | output introduces colours not present in input |
| 156 | 17.536 | 1697 | 47 | seeded_from_tasklog_and_inventory | `compiler_direct_output_algebra`, `compiler_tiny_lut_gather`, `compiler_final_equal_overlay` | output introduces colours not present in input; same nonzero mask but different colours |
| 169 | 17.579 | 1500 | 170 | seeded_from_verified_log | `compiler_tiny_lut_gather`, `compiler_single_conv_qlinear`, `compiler_final_equal_overlay`, `compiler_qlinear_uint8` | same nonzero mask but different colours; tasklog keyword match: \b(recolou?r\|recolor\|mapped? to\|colour rank\|color rank) |
| 226 | 17.602 | 1613 | 20 | seeded_from_verified_log | `compiler_final_equal_overlay` | output introduces colours not present in input |
| 303 | 17.610 | 1598 | 22 | seeded_from_verified_log | `compiler_direct_output_algebra`, `compiler_final_equal_overlay` | output introduces colours not present in input; tasklog keyword match: \b(recolou?r\|recolor\|mapped? to\|colour rank\|color rank) |
| 369 | 17.623 | 1300 | 299 | seeded_from_verified_log | `compiler_single_conv_qlinear`, `compiler_qlinear_uint8` | tasklog keyword match: \b(recolou?r\|recolor\|mapped? to\|colour rank\|color rank) |
| 304 | 17.727 | 1404 | 37 | seeded_from_tasklog_and_inventory | `compiler_single_conv_qlinear`, `compiler_final_equal_overlay` | output introduces colours not present in input |
| 341 | 17.735 | 1394 | 35 | seeded_from_verified_log | `compiler_final_equal_overlay`, `compiler_bounded_scan` | output introduces colours not present in input |
| 232 | 17.749 | 0 | 1410 | seeded_from_tasklog_and_inventory | `compiler_tiny_lut_gather`, `compiler_single_conv_qlinear` | output introduces colours not present in input |
| 119 | 17.781 | 1244 | 121 | seeded_from_verified_log | `compiler_tiny_lut_gather`, `compiler_sparse_scatter`, `compiler_bounded_scan` | output introduces colours not present in input |
| 160 | 17.804 | 1164 | 170 | seeded_from_tasklog_and_inventory | `compiler_single_conv_qlinear`, `compiler_qlinear_uint8`, `compiler_bounded_scan` | output introduces colours not present in input |
| 027 | 17.822 | 1262 | 48 | seeded_from_verified_log |  | output introduces colours not present in input |
| 293 | 17.839 | 1247 | 41 | seeded_from_verified_log | `compiler_tiny_lut_gather`, `compiler_final_equal_overlay`, `compiler_sparse_scatter` | same nonzero mask but different colours |
| 183 | 17.885 | 1008 | 222 | seeded_from_verified_log | `compiler_direct_output_algebra`, `compiler_final_equal_overlay` | tasklog keyword match: \b(recolou?r\|recolor\|mapped? to\|colour rank\|color rank) |
| 254 | 17.999 | 1079 | 19 | seeded_from_verified_log | `compiler_final_equal_overlay`, `compiler_bounded_scan` | tasklog keyword match: \b(recolou?r\|recolor\|mapped? to\|colour rank\|color rank) |
| 010 | 18.025 | 730 | 340 | seeded_from_verified_log | `compiler_tiny_lut_gather`, `compiler_final_equal_overlay` | same nonzero mask but different colours; tasklog keyword match: \b(recolou?r\|recolor\|mapped? to\|colour rank\|color rank) |
| 058 | 18.060 | 252 | 781 | seeded_from_verified_log | `compiler_tiny_lut_gather` | output introduces colours not present in input |
| 060 | 18.082 | 0 | 1010 | seeded_from_tasklog_and_inventory | `compiler_single_conv_qlinear` | output introduces colours not present in input |
| 239 | 18.147 | 897 | 50 | seeded_from_tasklog_and_inventory | `compiler_tiny_lut_gather`, `compiler_final_equal_overlay` | output introduces colours not present in input |
| 189 | 18.181 | 866 | 49 | seeded_from_verified_log | `compiler_single_conv_qlinear`, `compiler_final_equal_overlay`, `compiler_roi_pool_crop` | tasklog keyword match: \b(recolou?r\|recolor\|mapped? to\|colour rank\|color rank) |
| 120 | 18.187 | 0 | 910 | seeded_from_verified_log | `compiler_single_conv_qlinear` | output introduces colours not present in input; same nonzero mask but different colours |
| 171 | 18.187 | 0 | 910 | seeded_from_tasklog_and_inventory | `compiler_single_conv_qlinear` | output introduces colours not present in input |
| 283 | 18.187 | 0 | 910 | seeded_from_verified_log | `compiler_single_conv_qlinear` | same nonzero mask but different colours; tasklog keyword match: \b(recolou?r\|recolor\|mapped? to\|colour rank\|color rank) |
| 294 | 18.187 | 0 | 910 | seeded_from_verified_log | `compiler_single_conv_qlinear` | output introduces colours not present in input; same nonzero mask but different colours |
| 331 | 18.187 | 0 | 910 | seeded_from_verified_log | `compiler_single_conv_qlinear` | output introduces colours not present in input |
| 139 | 18.193 | 765 | 139 | seeded_from_verified_log | `compiler_single_conv_qlinear`, `compiler_qlinear_uint8`, `compiler_bounded_scan` | output introduces colours not present in input |
| 015 | 18.198 | 0 | 900 | seeded_from_verified_log | `compiler_single_conv_qlinear` | output introduces colours not present in input |
| 114 | 18.198 | 0 | 900 | operator_evidence_only_needs_human_review | `compiler_single_conv_qlinear` | output introduces colours not present in input |
| 151 | 18.198 | 0 | 900 | operator_evidence_only_needs_human_review | `compiler_single_conv_qlinear` | output introduces colours not present in input |
| 220 | 18.198 | 0 | 900 | seeded_from_verified_log | `compiler_single_conv_qlinear` | output introduces colours not present in input; tasklog keyword match: \b(recolou?r\|recolor\|mapped? to\|colour rank\|color rank) |
| 230 | 18.198 | 0 | 900 | seeded_from_verified_log | `compiler_tiny_lut_gather`, `compiler_single_conv_qlinear` | output introduces colours not present in input |
| 047 | 18.241 | 826 | 36 | seeded_from_verified_log | `compiler_bounded_scan` | output introduces colours not present in input |
| 143 | 18.269 | 591 | 247 | seeded_from_verified_log | `compiler_direct_output_algebra`, `compiler_final_equal_overlay` | same nonzero mask but different colours; tasklog keyword match: \b(recolou?r\|recolor\|mapped? to\|colour rank\|color rank) |
| 043 | 18.323 | 588 | 206 | seeded_from_verified_log | `compiler_direct_output_algebra` | output introduces colours not present in input |
| 320 | 18.354 | 448 | 322 | seeded_from_verified_log | `compiler_single_conv_qlinear`, `compiler_sparse_scatter` | output introduces colours not present in input; same nonzero mask but different colours |
| 200 | 18.372 | 570 | 186 | seeded_from_verified_log | `compiler_direct_output_algebra`, `compiler_single_conv_qlinear`, `compiler_final_equal_overlay`, `compiler_bounded_scan` | output introduces colours not present in input |
| 126 | 18.595 | 590 | 15 | operator_evidence_only_needs_human_review | `compiler_direct_output_algebra`, `compiler_final_equal_overlay`, `compiler_sparse_scatter`, `compiler_bounded_scan` | output introduces colours not present in input |
| 332 | 18.667 | 429 | 134 | seeded_from_verified_log | `compiler_single_conv_qlinear`, `compiler_qlinear_uint8` | output introduces colours not present in input; same nonzero mask but different colours |
| 147 | 18.723 | 432 | 100 | seeded_from_verified_log | `compiler_single_conv_qlinear`, `compiler_sparse_scatter` | output introduces colours not present in input; same nonzero mask but different colours |
| 371 | 18.854 | 376 | 91 | seeded_from_verified_log | `compiler_direct_output_algebra`, `compiler_sparse_scatter` | output introduces colours not present in input |
| 081 | 18.860 | 392 | 72 | seeded_from_verified_log | `compiler_single_conv_qlinear`, `compiler_qlinear_uint8` | output introduces colours not present in input |
| 352 | 18.869 | 0 | 460 | seeded_from_verified_log | `compiler_single_conv_qlinear` | output introduces colours not present in input |
| 282 | 18.900 | 407 | 39 | seeded_from_verified_log | `compiler_single_conv_qlinear`, `compiler_qlinear_uint8` | output introduces colours not present in input |
| 292 | 18.927 | 252 | 182 | seeded_from_tasklog_and_inventory | `compiler_sparse_scatter` | output introduces colours not present in input; same nonzero mask but different colours |
| 203 | 18.986 | 408 | 1 | seeded_from_tasklog_and_inventory | `compiler_tiny_lut_gather`, `compiler_final_equal_overlay`, `compiler_bounded_scan` | same nonzero mask but different colours |
| 272 | 19.004 | 200 | 202 | seeded_from_verified_log | `compiler_single_conv_qlinear` | output introduces colours not present in input; same nonzero mask but different colours |
| 127 | 19.019 | 204 | 192 | seeded_from_verified_log | `compiler_single_conv_qlinear` | tasklog keyword match: \b(recolou?r\|recolor\|mapped? to\|colour rank\|color rank) |
| 040 | 19.076 | 240 | 134 | seeded_from_verified_log | `compiler_direct_output_algebra`, `compiler_bounded_scan` | same nonzero mask but different colours; tasklog keyword match: \b(recolou?r\|recolor\|mapped? to\|colour rank\|color rank) |
| 266 | 19.142 | 315 | 35 | seeded_from_verified_log | `compiler_tiny_lut_gather` | tasklog keyword match: \b(recolou?r\|recolor\|mapped? to\|colour rank\|color rank) |
| 095 | 19.156 | 245 | 100 | seeded_from_verified_log | `compiler_single_conv_qlinear`, `compiler_qlinear_uint8` | output introduces colours not present in input |
| 003 | 19.290 | 281 | 21 | seeded_from_verified_log | `compiler_tiny_lut_gather`, `compiler_final_equal_overlay` | tasklog keyword match: \b(recolou?r\|recolor\|mapped? to\|colour rank\|color rank) |
| 252 | 19.439 | 0 | 260 | seeded_from_verified_log | `compiler_direct_output_algebra` | output introduces colours not present in input; same nonzero mask but different colours |
| 229 | 19.519 | 212 | 28 | operator_evidence_only_needs_human_review | `compiler_final_equal_overlay` | same nonzero mask but different colours |
| 176 | 19.519 | 0 | 240 | operator_evidence_only_needs_human_review | `compiler_direct_output_algebra` | output introduces colours not present in input |
| 313 | 19.743 | 0 | 192 | operator_evidence_only_needs_human_review | `compiler_direct_output_algebra` | same nonzero mask but different colours |
| 258 | 19.925 | 0 | 160 | seeded_from_verified_log | `compiler_single_conv_qlinear` | output introduces colours not present in input |
| 150 | 19.937 | 128 | 30 | seeded_from_verified_log | `compiler_tiny_lut_gather` | same nonzero mask but different colours |
| 155 | 19.937 | 128 | 30 | seeded_from_verified_log | `compiler_tiny_lut_gather` | same nonzero mask but different colours |
| 389 | 20.058 | 128 | 12 | seeded_from_verified_log | `compiler_direct_output_algebra`, `compiler_tiny_lut_gather`, `compiler_sparse_scatter`, `compiler_bounded_scan` | tasklog keyword match: \b(recolou?r\|recolor\|mapped? to\|colour rank\|color rank) |
| 298 | 20.095 | 120 | 15 | seeded_from_verified_log | `compiler_direct_output_algebra` | tasklog keyword match: \b(recolou?r\|recolor\|mapped? to\|colour rank\|color rank) |
| 314 | 20.395 | 0 | 100 | operator_evidence_only_needs_human_review | `compiler_direct_output_algebra`, `compiler_single_conv_qlinear` | same nonzero mask but different colours |
| 318 | 20.395 | 0 | 100 | seeded_from_verified_log | `compiler_direct_output_algebra`, `compiler_single_conv_qlinear` | tasklog keyword match: \b(recolou?r\|recolor\|mapped? to\|colour rank\|color rank) |
| 373 | 20.906 | 0 | 60 | seeded_from_verified_log | `compiler_direct_output_algebra` | same nonzero mask but different colours; tasklog keyword match: \b(recolou?r\|recolor\|mapped? to\|colour rank\|color rank) |
| 299 | 21.088 | 0 | 50 | operator_evidence_only_needs_human_review | `compiler_direct_output_algebra` | output introduces colours not present in input |
| 166 | 21.822 | 0 | 24 | seeded_from_verified_log | `compiler_direct_output_algebra` | output introduces colours not present in input |
| 312 | 22.004 | 0 | 20 | seeded_from_verified_log | `compiler_direct_output_algebra` | same nonzero mask but different colours; tasklog keyword match: \b(recolou?r\|recolor\|mapped? to\|colour rank\|color rank) |
| 016 | 22.697 | 0 | 10 | operator_evidence_only_needs_human_review | `compiler_direct_output_algebra`, `compiler_tiny_lut_gather` | same nonzero mask but different colours |
| 276 | 22.697 | 0 | 10 | operator_evidence_only_needs_human_review | `compiler_direct_output_algebra`, `compiler_tiny_lut_gather` | same nonzero mask but different colours |
| 309 | 22.697 | 0 | 10 | operator_evidence_only_needs_human_review | `compiler_direct_output_algebra`, `compiler_tiny_lut_gather` | same nonzero mask but different colours |
| 337 | 22.697 | 0 | 10 | operator_evidence_only_needs_human_review | `compiler_direct_output_algebra`, `compiler_tiny_lut_gather` | same nonzero mask but different colours |
| 087 | 23.391 | 0 | 5 | operator_evidence_only_needs_human_review | `compiler_direct_output_algebra`, `compiler_roi_pool_crop` | same nonzero mask but different colours |
| 179 | 25.000 | 0 | 0 | operator_evidence_only_needs_human_review | `compiler_direct_output_algebra` | same nonzero mask but different colours |

## Known Best Routes

- Pending human review.

## Kill Criteria

- Pending human review.

## Successful Applications

- Pending verification.

## Failed Applications / Walls

- Pending verification.
