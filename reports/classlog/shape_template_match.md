# shape_template_match — Template match

Small template/sprite is matched, rotated, or stamped.

## Optimization Question

What is the cheapest verified ONNX family for tasks in this class, and
which semantic preconditions let us avoid full-canvas intermediates?

## Candidate Tasks

| task | pts | mem | params | status | first routes | evidence |
|---:|---:|---:|---:|---|---|---|
| 233 | 14.003 | 59147 | 565 | seeded_from_verified_log | `compiler_direct_onehot_gather`, `compiler_tiny_lut_gather`, `compiler_single_conv_qlinear`, `compiler_final_equal_overlay` | tasklog keyword match: \b(template\|sprite\|stamp\|glyph\|dihedral\|rotation\|rotated) |
| 018 | 14.157 | 48196 | 2987 | seeded_from_verified_log | `compiler_tiny_lut_gather`, `compiler_single_conv_qlinear`, `compiler_final_equal_overlay`, `compiler_sparse_scatter` | tasklog keyword match: \b(template\|sprite\|stamp\|glyph\|dihedral\|rotation\|rotated) |
| 366 | 14.497 | 35927 | 490 | seeded_from_verified_log | `compiler_tiny_lut_gather`, `compiler_single_conv_qlinear`, `compiler_final_equal_overlay`, `compiler_sparse_scatter` | tasklog keyword match: \b(template\|sprite\|stamp\|glyph\|dihedral\|rotation\|rotated) |
| 158 | 14.528 | 32979 | 2340 | seeded_from_verified_log | `compiler_tiny_lut_gather`, `compiler_single_conv_qlinear`, `compiler_final_equal_overlay`, `compiler_qlinear_uint8` | tasklog keyword match: \b(template\|sprite\|stamp\|glyph\|dihedral\|rotation\|rotated) |
| 118 | 14.559 | 30849 | 3387 | seeded_from_verified_log | `compiler_direct_output_algebra`, `compiler_direct_onehot_gather`, `compiler_tiny_lut_gather`, `compiler_single_conv_qlinear` | tasklog keyword match: \b(template\|sprite\|stamp\|glyph\|dihedral\|rotation\|rotated) |
| 133 | 14.578 | 32294 | 1288 | seeded_from_verified_log | `compiler_single_conv_qlinear`, `compiler_final_equal_overlay`, `compiler_qlinear_uint8`, `compiler_roi_pool_crop` | tasklog keyword match: \b(template\|sprite\|stamp\|glyph\|dihedral\|rotation\|rotated) |
| 209 | 14.620 | 32027 | 185 | seeded_from_verified_log | `compiler_tiny_lut_gather`, `compiler_single_conv_qlinear`, `compiler_final_equal_overlay`, `compiler_qlinear_uint8` | tasklog keyword match: \b(template\|sprite\|stamp\|glyph\|dihedral\|rotation\|rotated) |
| 191 | 14.624 | 31252 | 841 | seeded_from_verified_log | `compiler_direct_onehot_gather`, `compiler_tiny_lut_gather`, `compiler_single_conv_qlinear`, `compiler_final_equal_overlay` | tasklog keyword match: \b(template\|sprite\|stamp\|glyph\|dihedral\|rotation\|rotated) |
| 054 | 14.829 | 25885 | 238 | seeded_from_tasklog_and_inventory | `compiler_tiny_lut_gather`, `compiler_single_conv_qlinear`, `compiler_final_equal_overlay`, `compiler_sparse_scatter` | tasklog keyword match: \b(template\|sprite\|stamp\|glyph\|dihedral\|rotation\|rotated) |
| 285 | 14.848 | 25080 | 550 | seeded_from_verified_log | `compiler_tiny_lut_gather`, `compiler_single_conv_qlinear`, `compiler_final_equal_overlay`, `compiler_sparse_scatter` | tasklog keyword match: \b(template\|sprite\|stamp\|glyph\|dihedral\|rotation\|rotated) |
| 076 | 14.932 | 23296 | 292 | seeded_from_verified_log | `compiler_tiny_lut_gather`, `compiler_final_equal_overlay`, `compiler_sparse_scatter`, `compiler_bounded_scan` | tasklog keyword match: \b(template\|sprite\|stamp\|glyph\|dihedral\|rotation\|rotated) |
| 364 | 14.956 | 22900 | 113 | seeded_from_verified_log | `compiler_tiny_lut_gather`, `compiler_single_conv_qlinear`, `compiler_final_equal_overlay`, `compiler_qlinear_uint8` | tasklog keyword match: \b(template\|sprite\|stamp\|glyph\|dihedral\|rotation\|rotated) |
| 173 | 15.022 | 21430 | 112 | seeded_from_verified_log | `compiler_tiny_lut_gather`, `compiler_single_conv_qlinear`, `compiler_final_equal_overlay`, `compiler_sparse_scatter` | tasklog keyword match: \b(template\|sprite\|stamp\|glyph\|dihedral\|rotation\|rotated) |
| 367 | 15.051 | 16200 | 4733 | seeded_from_verified_log | `compiler_tiny_lut_gather`, `compiler_single_conv_qlinear`, `compiler_qlinear_uint8` | tasklog keyword match: \b(template\|sprite\|stamp\|glyph\|dihedral\|rotation\|rotated) |
| 319 | 15.119 | 19280 | 279 | seeded_from_verified_log | `compiler_direct_output_algebra`, `compiler_tiny_lut_gather`, `compiler_final_equal_overlay`, `compiler_bounded_scan` | tasklog keyword match: \b(template\|sprite\|stamp\|glyph\|dihedral\|rotation\|rotated) |
| 219 | 15.195 | 18033 | 93 | seeded_from_verified_log | `compiler_tiny_lut_gather`, `compiler_final_equal_overlay`, `compiler_bounded_scan` | tasklog keyword match: \b(template\|sprite\|stamp\|glyph\|dihedral\|rotation\|rotated) |
| 101 | 15.216 | 16850 | 905 | seeded_from_verified_log | `compiler_tiny_lut_gather`, `compiler_final_equal_overlay`, `compiler_sparse_scatter`, `compiler_bounded_scan` | tasklog keyword match: \b(template\|sprite\|stamp\|glyph\|dihedral\|rotation\|rotated) |
| 066 | 15.256 | 16899 | 160 | seeded_from_verified_log | `compiler_direct_output_algebra`, `compiler_final_equal_overlay`, `compiler_bounded_scan` | tasklog keyword match: \b(template\|sprite\|stamp\|glyph\|dihedral\|rotation\|rotated) |
| 044 | 15.651 | 11420 | 68 | seeded_from_verified_log | `compiler_direct_onehot_gather`, `compiler_tiny_lut_gather`, `compiler_single_conv_qlinear`, `compiler_final_equal_overlay` | tasklog keyword match: \b(template\|sprite\|stamp\|glyph\|dihedral\|rotation\|rotated) |
| 080 | 15.726 | 10198 | 454 | seeded_from_verified_log | `compiler_tiny_lut_gather`, `compiler_single_conv_qlinear`, `compiler_final_equal_overlay`, `compiler_qlinear_uint8` | tasklog keyword match: \b(template\|sprite\|stamp\|glyph\|dihedral\|rotation\|rotated) |
| 017 | 15.744 | 8448 | 2021 | seeded_from_verified_log | `compiler_tiny_lut_gather`, `compiler_final_equal_overlay` | tasklog keyword match: \b(template\|sprite\|stamp\|glyph\|dihedral\|rotation\|rotated) |
| 379 | 15.818 | 9069 | 653 | seeded_from_verified_log | `compiler_single_conv_qlinear`, `compiler_final_equal_overlay`, `compiler_bounded_scan` | tasklog keyword match: \b(template\|sprite\|stamp\|glyph\|dihedral\|rotation\|rotated) |
| 370 | 15.823 | 8645 | 1024 | seeded_from_verified_log | `compiler_direct_output_algebra`, `compiler_tiny_lut_gather`, `compiler_single_conv_qlinear`, `compiler_final_equal_overlay` | tasklog keyword match: \b(template\|sprite\|stamp\|glyph\|dihedral\|rotation\|rotated) |
| 074 | 15.889 | 9000 | 50 | seeded_from_verified_log | `compiler_tiny_lut_gather`, `compiler_single_conv_qlinear`, `compiler_final_equal_overlay` | tasklog keyword match: \b(template\|sprite\|stamp\|glyph\|dihedral\|rotation\|rotated) |
| 096 | 15.905 | 8108 | 805 | seeded_from_verified_log | `compiler_tiny_lut_gather`, `compiler_final_equal_overlay`, `compiler_sparse_scatter`, `compiler_bounded_scan` | tasklog keyword match: \b(template\|sprite\|stamp\|glyph\|dihedral\|rotation\|rotated) |
| 157 | 15.972 | 7983 | 351 | seeded_from_verified_log | `compiler_tiny_lut_gather`, `compiler_final_equal_overlay`, `compiler_sparse_scatter`, `compiler_bounded_scan` | tasklog keyword match: \b(template\|sprite\|stamp\|glyph\|dihedral\|rotation\|rotated) |
| 089 | 16.111 | 7092 | 160 | seeded_from_verified_log | `compiler_tiny_lut_gather`, `compiler_single_conv_qlinear`, `compiler_final_equal_overlay`, `compiler_sparse_scatter` | tasklog keyword match: \b(template\|sprite\|stamp\|glyph\|dihedral\|rotation\|rotated) |
| 174 | 16.130 | 6973 | 142 | seeded_from_verified_log | `compiler_direct_onehot_gather`, `compiler_tiny_lut_gather`, `compiler_final_equal_overlay`, `compiler_bounded_scan` | tasklog keyword match: \b(template\|sprite\|stamp\|glyph\|dihedral\|rotation\|rotated) |
| 005 | 16.163 | 6392 | 490 | seeded_from_verified_log | `compiler_tiny_lut_gather`, `compiler_single_conv_qlinear`, `compiler_final_equal_overlay`, `compiler_qlinear_uint8` | tasklog keyword match: \b(template\|sprite\|stamp\|glyph\|dihedral\|rotation\|rotated) |
| 182 | 16.176 | 6695 | 98 | seeded_from_verified_log | `compiler_tiny_lut_gather`, `compiler_single_conv_qlinear`, `compiler_final_equal_overlay`, `compiler_qlinear_uint8` | tasklog keyword match: \b(template\|sprite\|stamp\|glyph\|dihedral\|rotation\|rotated) |
| 328 | 16.179 | 4835 | 1940 | seeded_from_tasklog_and_inventory | `compiler_tiny_lut_gather`, `compiler_final_equal_overlay` | tasklog keyword match: \b(template\|sprite\|stamp\|glyph\|dihedral\|rotation\|rotated) |
| 264 | 16.292 | 5348 | 706 | seeded_from_verified_log | `compiler_direct_output_algebra`, `compiler_tiny_lut_gather`, `compiler_single_conv_qlinear`, `compiler_final_equal_overlay` | tasklog keyword match: \b(template\|sprite\|stamp\|glyph\|dihedral\|rotation\|rotated) |
| 382 | 16.337 | 5648 | 135 | seeded_from_verified_log | `compiler_direct_output_algebra`, `compiler_tiny_lut_gather`, `compiler_final_equal_overlay`, `compiler_bounded_scan` | tasklog keyword match: \b(template\|sprite\|stamp\|glyph\|dihedral\|rotation\|rotated) |
| 278 | 16.391 | 5436 | 47 | seeded_from_verified_log | `compiler_single_conv_qlinear`, `compiler_qlinear_uint8`, `compiler_bounded_scan` | tasklog keyword match: \b(template\|sprite\|stamp\|glyph\|dihedral\|rotation\|rotated) |
| 251 | 16.499 | 4752 | 168 | seeded_from_verified_log | `compiler_bounded_scan` | tasklog keyword match: \b(template\|sprite\|stamp\|glyph\|dihedral\|rotation\|rotated) |
| 208 | 16.539 | 4612 | 114 | seeded_from_verified_log | `compiler_direct_output_algebra`, `compiler_single_conv_qlinear`, `compiler_final_equal_overlay`, `compiler_qlinear_uint8` | tasklog keyword match: \b(template\|sprite\|stamp\|glyph\|dihedral\|rotation\|rotated) |
| 361 | 16.547 | 4322 | 366 | seeded_from_verified_log | `compiler_tiny_lut_gather`, `compiler_single_conv_qlinear`, `compiler_final_equal_overlay`, `compiler_roi_pool_crop` | tasklog keyword match: \b(template\|sprite\|stamp\|glyph\|dihedral\|rotation\|rotated) |
| 277 | 16.627 | 4101 | 228 | seeded_from_verified_log | `compiler_tiny_lut_gather`, `compiler_final_equal_overlay`, `compiler_bounded_scan` | tasklog keyword match: \b(template\|sprite\|stamp\|glyph\|dihedral\|rotation\|rotated) |
| 206 | 16.657 | 4122 | 77 | seeded_from_verified_log | `compiler_single_conv_qlinear`, `compiler_final_equal_overlay`, `compiler_sparse_scatter`, `compiler_bounded_scan` | tasklog keyword match: \b(template\|sprite\|stamp\|glyph\|dihedral\|rotation\|rotated) |
| 117 | 16.666 | 3922 | 243 | seeded_from_verified_log | `compiler_tiny_lut_gather`, `compiler_single_conv_qlinear`, `compiler_final_equal_overlay`, `compiler_qlinear_uint8` | tasklog keyword match: \b(template\|sprite\|stamp\|glyph\|dihedral\|rotation\|rotated) |
| 363 | 16.673 | 3839 | 296 | seeded_from_verified_log | `compiler_single_conv_qlinear`, `compiler_final_equal_overlay`, `compiler_bounded_scan` | tasklog keyword match: \b(template\|sprite\|stamp\|glyph\|dihedral\|rotation\|rotated) |
| 132 | 16.684 | 3990 | 99 | seeded_from_verified_log | `compiler_tiny_lut_gather`, `compiler_single_conv_qlinear`, `compiler_final_equal_overlay`, `compiler_bounded_scan` | tasklog keyword match: \b(template\|sprite\|stamp\|glyph\|dihedral\|rotation\|rotated) |
| 162 | 16.689 | 4024 | 44 | seeded_from_verified_log | `compiler_single_conv_qlinear`, `compiler_qlinear_uint8`, `compiler_bounded_scan` | tasklog keyword match: \b(template\|sprite\|stamp\|glyph\|dihedral\|rotation\|rotated) |
| 069 | 16.728 | 3848 | 64 | seeded_from_verified_log | `compiler_tiny_lut_gather`, `compiler_single_conv_qlinear`, `compiler_final_equal_overlay`, `compiler_bounded_scan` | tasklog keyword match: \b(template\|sprite\|stamp\|glyph\|dihedral\|rotation\|rotated) |
| 359 | 16.740 | 3852 | 15 | seeded_from_verified_log | `compiler_final_equal_overlay`, `compiler_bounded_scan` | tasklog keyword match: \b(template\|sprite\|stamp\|glyph\|dihedral\|rotation\|rotated) |
| 284 | 16.774 | 3082 | 653 | seeded_from_verified_log | `compiler_single_conv_qlinear`, `compiler_sparse_scatter` | tasklog keyword match: \b(template\|sprite\|stamp\|glyph\|dihedral\|rotation\|rotated) |
| 201 | 16.881 | 3202 | 154 | seeded_from_verified_log | `compiler_tiny_lut_gather`, `compiler_single_conv_qlinear`, `compiler_final_equal_overlay`, `compiler_bounded_scan` | tasklog keyword match: \b(template\|sprite\|stamp\|glyph\|dihedral\|rotation\|rotated) |
| 079 | 16.947 | 3065 | 78 | seeded_from_verified_log | `compiler_single_conv_qlinear`, `compiler_final_equal_overlay`, `compiler_bounded_scan` | tasklog keyword match: \b(template\|sprite\|stamp\|glyph\|dihedral\|rotation\|rotated) |
| 086 | 16.949 | 2946 | 190 | seeded_from_verified_log | `compiler_tiny_lut_gather`, `compiler_single_conv_qlinear`, `compiler_final_equal_overlay`, `compiler_qlinear_uint8` | tasklog keyword match: \b(template\|sprite\|stamp\|glyph\|dihedral\|rotation\|rotated) |
| 042 | 16.958 | 2792 | 318 | seeded_from_verified_log | `compiler_single_conv_qlinear` | tasklog keyword match: \b(template\|sprite\|stamp\|glyph\|dihedral\|rotation\|rotated) |
| 330 | 16.986 | 2800 | 222 | seeded_from_verified_log | `compiler_final_equal_overlay`, `compiler_sparse_scatter`, `compiler_bounded_scan` | tasklog keyword match: \b(template\|sprite\|stamp\|glyph\|dihedral\|rotation\|rotated) |
| 245 | 17.083 | 2604 | 139 | seeded_from_verified_log | `compiler_tiny_lut_gather`, `compiler_bounded_scan` | tasklog keyword match: \b(template\|sprite\|stamp\|glyph\|dihedral\|rotation\|rotated) |
| 355 | 17.093 | 2688 | 27 | seeded_from_verified_log | `compiler_final_equal_overlay`, `compiler_bounded_scan` | tasklog keyword match: \b(template\|sprite\|stamp\|glyph\|dihedral\|rotation\|rotated) |
| 071 | 17.100 | 2643 | 55 | seeded_from_verified_log | `compiler_tiny_lut_gather`, `compiler_final_equal_overlay`, `compiler_bounded_scan` | tasklog keyword match: \b(template\|sprite\|stamp\|glyph\|dihedral\|rotation\|rotated) |
| 170 | 17.158 | 2216 | 329 | seeded_from_verified_log | `compiler_tiny_lut_gather`, `compiler_single_conv_qlinear`, `compiler_final_equal_overlay`, `compiler_bounded_scan` | tasklog keyword match: \b(template\|sprite\|stamp\|glyph\|dihedral\|rotation\|rotated) |
| 062 | 17.162 | 2465 | 71 | seeded_from_verified_log | `compiler_tiny_lut_gather`, `compiler_final_equal_overlay`, `compiler_bounded_scan` | tasklog keyword match: \b(template\|sprite\|stamp\|glyph\|dihedral\|rotation\|rotated) |
| 102 | 17.165 | 2414 | 113 | seeded_from_verified_log | `compiler_single_conv_qlinear`, `compiler_qlinear_uint8`, `compiler_bounded_scan` | tasklog keyword match: \b(template\|sprite\|stamp\|glyph\|dihedral\|rotation\|rotated) |
| 238 | 17.178 | 2051 | 444 | seeded_from_verified_log | `compiler_direct_output_algebra`, `compiler_tiny_lut_gather`, `compiler_final_equal_overlay`, `compiler_bounded_scan` | tasklog keyword match: \b(template\|sprite\|stamp\|glyph\|dihedral\|rotation\|rotated) |
| 163 | 17.240 | 2235 | 110 | seeded_from_verified_log | `compiler_tiny_lut_gather`, `compiler_single_conv_qlinear`, `compiler_final_equal_overlay`, `compiler_bounded_scan` | tasklog keyword match: \b(template\|sprite\|stamp\|glyph\|dihedral\|rotation\|rotated) |
| 022 | 17.258 | 2004 | 298 | seeded_from_verified_log | `compiler_direct_output_algebra`, `compiler_final_equal_overlay`, `compiler_roi_pool_crop` | tasklog keyword match: \b(template\|sprite\|stamp\|glyph\|dihedral\|rotation\|rotated) |
| 134 | 17.266 | 2250 | 34 | seeded_from_verified_log | `compiler_tiny_lut_gather`, `compiler_bounded_scan` | tasklog keyword match: \b(template\|sprite\|stamp\|glyph\|dihedral\|rotation\|rotated) |
| 012 | 17.275 | 2076 | 188 | seeded_from_verified_log | `compiler_tiny_lut_gather`, `compiler_final_equal_overlay`, `compiler_sparse_scatter` | tasklog keyword match: \b(template\|sprite\|stamp\|glyph\|dihedral\|rotation\|rotated) |
| 323 | 17.314 | 1521 | 656 | seeded_from_verified_log | `compiler_single_conv_qlinear`, `compiler_qlinear_uint8` | tasklog keyword match: \b(template\|sprite\|stamp\|glyph\|dihedral\|rotation\|rotated) |
| 185 | 17.419 | 1651 | 310 | seeded_from_verified_log | `compiler_tiny_lut_gather`, `compiler_single_conv_qlinear`, `compiler_final_equal_overlay`, `compiler_bounded_scan` | tasklog keyword match: \b(template\|sprite\|stamp\|glyph\|dihedral\|rotation\|rotated) |
| 124 | 17.423 | 1879 | 74 | seeded_from_verified_log | `compiler_direct_onehot_gather`, `compiler_tiny_lut_gather`, `compiler_final_equal_overlay`, `compiler_bounded_scan` | tasklog keyword match: \b(template\|sprite\|stamp\|glyph\|dihedral\|rotation\|rotated) |
| 392 | 17.424 | 1602 | 349 | seeded_from_verified_log | `compiler_direct_output_algebra`, `compiler_tiny_lut_gather`, `compiler_final_equal_overlay`, `compiler_bounded_scan` | tasklog keyword match: \b(template\|sprite\|stamp\|glyph\|dihedral\|rotation\|rotated) |
| 088 | 17.465 | 1791 | 81 | seeded_from_verified_log | `compiler_direct_output_algebra`, `compiler_final_equal_overlay` | tasklog keyword match: \b(template\|sprite\|stamp\|glyph\|dihedral\|rotation\|rotated) |
| 400 | 17.468 | 1800 | 66 | seeded_from_verified_log | `compiler_tiny_lut_gather`, `compiler_roi_pool_crop`, `compiler_bounded_scan` | tasklog keyword match: \b(template\|sprite\|stamp\|glyph\|dihedral\|rotation\|rotated) |
| 199 | 17.484 | 1749 | 88 | seeded_from_verified_log | `compiler_direct_output_algebra`, `compiler_final_equal_overlay`, `compiler_sparse_scatter` | tasklog keyword match: \b(template\|sprite\|stamp\|glyph\|dihedral\|rotation\|rotated) |
| 109 | 17.516 | 1686 | 93 | seeded_from_verified_log | `compiler_tiny_lut_gather`, `compiler_final_equal_overlay` | tasklog keyword match: \b(template\|sprite\|stamp\|glyph\|dihedral\|rotation\|rotated) |
| 112 | 17.519 | 1616 | 158 | seeded_from_verified_log | `compiler_direct_output_algebra`, `compiler_tiny_lut_gather` | tasklog keyword match: \b(template\|sprite\|stamp\|glyph\|dihedral\|rotation\|rotated) |
| 169 | 17.579 | 1500 | 170 | seeded_from_verified_log | `compiler_tiny_lut_gather`, `compiler_single_conv_qlinear`, `compiler_final_equal_overlay`, `compiler_qlinear_uint8` | tasklog keyword match: \b(template\|sprite\|stamp\|glyph\|dihedral\|rotation\|rotated) |
| 033 | 17.598 | 1606 | 33 | seeded_from_verified_log |  | tasklog keyword match: \b(template\|sprite\|stamp\|glyph\|dihedral\|rotation\|rotated) |
| 159 | 17.642 | 1419 | 149 | seeded_from_verified_log | `compiler_direct_output_algebra`, `compiler_tiny_lut_gather`, `compiler_final_equal_overlay` | tasklog keyword match: \b(template\|sprite\|stamp\|glyph\|dihedral\|rotation\|rotated) |
| 020 | 17.670 | 1356 | 170 | seeded_from_verified_log | `compiler_tiny_lut_gather`, `compiler_single_conv_qlinear`, `compiler_sparse_scatter`, `compiler_bounded_scan` | tasklog keyword match: \b(template\|sprite\|stamp\|glyph\|dihedral\|rotation\|rotated) |
| 075 | 17.695 | 1326 | 161 | seeded_from_verified_log | `compiler_direct_output_algebra`, `compiler_final_equal_overlay` | tasklog keyword match: \b(template\|sprite\|stamp\|glyph\|dihedral\|rotation\|rotated) |
| 217 | 17.707 | 1429 | 41 | seeded_from_tasklog_and_inventory | `compiler_tiny_lut_gather`, `compiler_single_conv_qlinear`, `compiler_final_equal_overlay`, `compiler_bounded_scan` | tasklog keyword match: \b(template\|sprite\|stamp\|glyph\|dihedral\|rotation\|rotated) |
| 160 | 17.804 | 1164 | 170 | seeded_from_tasklog_and_inventory | `compiler_single_conv_qlinear`, `compiler_qlinear_uint8`, `compiler_bounded_scan` | tasklog keyword match: \b(template\|sprite\|stamp\|glyph\|dihedral\|rotation\|rotated) |
| 027 | 17.822 | 1262 | 48 | seeded_from_verified_log |  | tasklog keyword match: \b(template\|sprite\|stamp\|glyph\|dihedral\|rotation\|rotated) |
| 346 | 17.866 | 1224 | 30 | seeded_from_verified_log | `compiler_tiny_lut_gather`, `compiler_single_conv_qlinear`, `compiler_final_equal_overlay`, `compiler_qlinear_uint8` | tasklog keyword match: \b(template\|sprite\|stamp\|glyph\|dihedral\|rotation\|rotated) |
| 225 | 18.060 | 926 | 107 | seeded_from_verified_log | `compiler_tiny_lut_gather`, `compiler_single_conv_qlinear`, `compiler_final_equal_overlay`, `compiler_bounded_scan` | tasklog keyword match: \b(template\|sprite\|stamp\|glyph\|dihedral\|rotation\|rotated) |
| 228 | 18.131 | 849 | 113 | seeded_from_verified_log | `compiler_direct_output_algebra`, `compiler_tiny_lut_gather`, `compiler_final_equal_overlay`, `compiler_sparse_scatter` | tasklog keyword match: \b(template\|sprite\|stamp\|glyph\|dihedral\|rotation\|rotated) |
| 288 | 18.135 | 769 | 189 | seeded_from_verified_log | `compiler_tiny_lut_gather`, `compiler_final_equal_overlay`, `compiler_sparse_scatter`, `compiler_bounded_scan` | tasklog keyword match: \b(template\|sprite\|stamp\|glyph\|dihedral\|rotation\|rotated) |
| 194 | 18.145 | 900 | 49 | seeded_from_verified_log | `compiler_tiny_lut_gather` | tasklog keyword match: \b(template\|sprite\|stamp\|glyph\|dihedral\|rotation\|rotated) |
| 195 | 18.175 | 882 | 39 | seeded_from_verified_log | `compiler_tiny_lut_gather`, `compiler_bounded_scan` | tasklog keyword match: \b(template\|sprite\|stamp\|glyph\|dihedral\|rotation\|rotated) |
| 331 | 18.187 | 0 | 910 | seeded_from_verified_log | `compiler_single_conv_qlinear` | tasklog keyword match: \b(template\|sprite\|stamp\|glyph\|dihedral\|rotation\|rotated) |
| 259 | 18.192 | 853 | 52 | seeded_from_verified_log | `compiler_tiny_lut_gather`, `compiler_single_conv_qlinear`, `compiler_final_equal_overlay`, `compiler_bounded_scan` | tasklog keyword match: \b(template\|sprite\|stamp\|glyph\|dihedral\|rotation\|rotated) |
| 139 | 18.193 | 765 | 139 | seeded_from_verified_log | `compiler_single_conv_qlinear`, `compiler_qlinear_uint8`, `compiler_bounded_scan` | tasklog keyword match: \b(template\|sprite\|stamp\|glyph\|dihedral\|rotation\|rotated) |
| 015 | 18.198 | 0 | 900 | seeded_from_verified_log | `compiler_single_conv_qlinear` | tasklog keyword match: \b(template\|sprite\|stamp\|glyph\|dihedral\|rotation\|rotated) |
| 220 | 18.198 | 0 | 900 | seeded_from_verified_log | `compiler_single_conv_qlinear` | tasklog keyword match: \b(template\|sprite\|stamp\|glyph\|dihedral\|rotation\|rotated) |
| 230 | 18.198 | 0 | 900 | seeded_from_verified_log | `compiler_tiny_lut_gather`, `compiler_single_conv_qlinear` | tasklog keyword match: \b(template\|sprite\|stamp\|glyph\|dihedral\|rotation\|rotated) |
| 106 | 18.233 | 776 | 93 | seeded_from_verified_log | `compiler_tiny_lut_gather`, `compiler_single_conv_qlinear`, `compiler_sparse_scatter` | tasklog keyword match: \b(template\|sprite\|stamp\|glyph\|dihedral\|rotation\|rotated) |
| 143 | 18.269 | 591 | 247 | seeded_from_verified_log | `compiler_direct_output_algebra`, `compiler_final_equal_overlay` | tasklog keyword match: \b(template\|sprite\|stamp\|glyph\|dihedral\|rotation\|rotated) |
| 122 | 18.303 | 0 | 810 | seeded_from_verified_log | `compiler_single_conv_qlinear` | tasklog keyword match: \b(template\|sprite\|stamp\|glyph\|dihedral\|rotation\|rotated) |
| 214 | 18.358 | 726 | 41 | seeded_from_verified_log | `compiler_tiny_lut_gather`, `compiler_single_conv_qlinear`, `compiler_final_equal_overlay` | tasklog keyword match: \b(template\|sprite\|stamp\|glyph\|dihedral\|rotation\|rotated) |
| 263 | 18.478 | 563 | 117 | seeded_from_verified_log | `compiler_tiny_lut_gather`, `compiler_single_conv_qlinear`, `compiler_final_equal_overlay` | tasklog keyword match: \b(template\|sprite\|stamp\|glyph\|dihedral\|rotation\|rotated) |
| 057 | 18.496 | 638 | 30 | seeded_from_verified_log | `compiler_tiny_lut_gather`, `compiler_final_equal_overlay`, `compiler_bounded_scan` | tasklog keyword match: \b(template\|sprite\|stamp\|glyph\|dihedral\|rotation\|rotated) |
| 068 | 18.565 | 500 | 123 | seeded_from_verified_log | `compiler_direct_output_algebra`, `compiler_final_equal_overlay`, `compiler_sparse_scatter` | tasklog keyword match: \b(template\|sprite\|stamp\|glyph\|dihedral\|rotation\|rotated) |
| 300 | 18.708 | 474 | 66 | seeded_from_verified_log | `compiler_direct_output_algebra`, `compiler_tiny_lut_gather`, `compiler_final_equal_overlay` | tasklog keyword match: \b(template\|sprite\|stamp\|glyph\|dihedral\|rotation\|rotated) |
| 111 | 18.718 | 528 | 7 | seeded_from_verified_log | `compiler_tiny_lut_gather`, `compiler_bounded_scan` | tasklog keyword match: \b(template\|sprite\|stamp\|glyph\|dihedral\|rotation\|rotated) |
| 039 | 18.826 | 424 | 56 | seeded_from_verified_log | `compiler_direct_onehot_gather`, `compiler_tiny_lut_gather` | tasklog keyword match: \b(template\|sprite\|stamp\|glyph\|dihedral\|rotation\|rotated) |
| 371 | 18.854 | 376 | 91 | seeded_from_verified_log | `compiler_direct_output_algebra`, `compiler_sparse_scatter` | tasklog keyword match: \b(template\|sprite\|stamp\|glyph\|dihedral\|rotation\|rotated) |
| 083 | 18.897 | 376 | 71 | seeded_from_verified_log | `compiler_tiny_lut_gather`, `compiler_sparse_scatter`, `compiler_bounded_scan` | tasklog keyword match: \b(template\|sprite\|stamp\|glyph\|dihedral\|rotation\|rotated) |
| 282 | 18.900 | 407 | 39 | seeded_from_verified_log | `compiler_single_conv_qlinear`, `compiler_qlinear_uint8` | tasklog keyword match: \b(template\|sprite\|stamp\|glyph\|dihedral\|rotation\|rotated) |
| 181 | 19.089 | 201 | 168 | seeded_from_verified_log | `compiler_sparse_scatter` | tasklog keyword match: \b(template\|sprite\|stamp\|glyph\|dihedral\|rotation\|rotated) |
| 266 | 19.142 | 315 | 35 | seeded_from_verified_log | `compiler_tiny_lut_gather` | tasklog keyword match: \b(template\|sprite\|stamp\|glyph\|dihedral\|rotation\|rotated) |
| 095 | 19.156 | 245 | 100 | seeded_from_verified_log | `compiler_single_conv_qlinear`, `compiler_qlinear_uint8` | tasklog keyword match: \b(template\|sprite\|stamp\|glyph\|dihedral\|rotation\|rotated) |
| 121 | 19.168 | 275 | 66 | seeded_from_verified_log | `compiler_direct_output_algebra`, `compiler_tiny_lut_gather` | tasklog keyword match: \b(template\|sprite\|stamp\|glyph\|dihedral\|rotation\|rotated) |
| 128 | 19.296 | 0 | 300 | seeded_from_verified_log | `compiler_single_conv_qlinear` | tasklog keyword match: \b(template\|sprite\|stamp\|glyph\|dihedral\|rotation\|rotated) |
| 306 | 19.296 | 0 | 300 | seeded_from_verified_log | `compiler_direct_output_algebra`, `compiler_tiny_lut_gather` | tasklog keyword match: \b(template\|sprite\|stamp\|glyph\|dihedral\|rotation\|rotated) |
| 001 | 19.519 | 0 | 240 | seeded_from_verified_log | `compiler_direct_output_algebra`, `compiler_tiny_lut_gather` | tasklog keyword match: \b(template\|sprite\|stamp\|glyph\|dihedral\|rotation\|rotated) |
| 207 | 19.536 | 192 | 44 | seeded_from_verified_log |  | tasklog keyword match: \b(template\|sprite\|stamp\|glyph\|dihedral\|rotation\|rotated) |
| 315 | 19.562 | 0 | 230 | seeded_from_verified_log | `compiler_direct_output_algebra` | tasklog keyword match: \b(template\|sprite\|stamp\|glyph\|dihedral\|rotation\|rotated) |
| 261 | 19.702 | 0 | 200 | seeded_from_verified_log | `compiler_single_conv_qlinear` | tasklog keyword match: \b(template\|sprite\|stamp\|glyph\|dihedral\|rotation\|rotated) |
| 082 | 19.753 | 0 | 190 | seeded_from_verified_log | `compiler_single_conv_qlinear` | tasklog keyword match: \b(template\|sprite\|stamp\|glyph\|dihedral\|rotation\|rotated) |
| 028 | 19.876 | 0 | 168 | seeded_from_verified_log | `compiler_direct_output_algebra` | tasklog keyword match: \b(template\|sprite\|stamp\|glyph\|dihedral\|rotation\|rotated) |
| 258 | 19.925 | 0 | 160 | seeded_from_verified_log | `compiler_single_conv_qlinear` | tasklog keyword match: \b(template\|sprite\|stamp\|glyph\|dihedral\|rotation\|rotated) |
| 248 | 19.983 | 122 | 29 | seeded_from_verified_log | `compiler_sparse_scatter` | tasklog keyword match: \b(template\|sprite\|stamp\|glyph\|dihedral\|rotation\|rotated) |
| 339 | 20.058 | 40 | 100 | seeded_from_verified_log | `compiler_direct_output_algebra` | tasklog keyword match: \b(template\|sprite\|stamp\|glyph\|dihedral\|rotation\|rotated) |
| 056 | 21.474 | 34 | 0 | seeded_from_verified_log | `compiler_direct_output_algebra`, `compiler_final_equal_overlay` | tasklog keyword match: \b(template\|sprite\|stamp\|glyph\|dihedral\|rotation\|rotated) |
| 140 | 23.391 | 0 | 5 | seeded_from_verified_log | `compiler_direct_output_algebra`, `compiler_tiny_lut_gather`, `compiler_roi_pool_crop` | tasklog keyword match: \b(template\|sprite\|stamp\|glyph\|dihedral\|rotation\|rotated) |

## Known Best Routes

- Pending human review.

## Kill Criteria

- Pending human review.

## Successful Applications

- Pending verification.

## Failed Applications / Walls

- Pending verification.
