# shape_copy_object — Copy object shape

Shape is copied from an input object/component.

## Optimization Question

What is the cheapest verified ONNX family for tasks in this class, and
which semantic preconditions let us avoid full-canvas intermediates?

## Candidate Tasks

| task | pts | mem | params | status | first routes | evidence |
|---:|---:|---:|---:|---|---|---|
| 158 | 14.528 | 32979 | 2340 | seeded_from_verified_log | `compiler_tiny_lut_gather`, `compiler_single_conv_qlinear`, `compiler_final_equal_overlay`, `compiler_qlinear_uint8` | same nonzero cell count with preserved colours suggests object-copy candidate |
| 133 | 14.578 | 32294 | 1288 | seeded_from_verified_log | `compiler_single_conv_qlinear`, `compiler_final_equal_overlay`, `compiler_qlinear_uint8`, `compiler_roi_pool_crop` | tasklog keyword match: \b(copy object\|copy the object\|object shape\|same shape\|identical shape) |
| 209 | 14.620 | 32027 | 185 | seeded_from_verified_log | `compiler_tiny_lut_gather`, `compiler_single_conv_qlinear`, `compiler_final_equal_overlay`, `compiler_qlinear_uint8` | tasklog keyword match: \b(copy object\|copy the object\|object shape\|same shape\|identical shape) |
| 054 | 14.829 | 25885 | 238 | seeded_from_tasklog_and_inventory | `compiler_tiny_lut_gather`, `compiler_single_conv_qlinear`, `compiler_final_equal_overlay`, `compiler_sparse_scatter` | same nonzero cell count with preserved colours suggests object-copy candidate |
| 064 | 15.494 | 13368 | 68 | seeded_from_verified_log | `compiler_tiny_lut_gather`, `compiler_final_equal_overlay`, `compiler_bounded_scan` | same nonzero cell count with preserved colours suggests object-copy candidate |
| 044 | 15.651 | 11420 | 68 | seeded_from_verified_log | `compiler_direct_onehot_gather`, `compiler_tiny_lut_gather`, `compiler_single_conv_qlinear`, `compiler_final_equal_overlay` | same nonzero cell count with preserved colours suggests object-copy candidate; tasklog keyword match: \b(copy object\|copy the object\|object shape\|same shape\|identical shape) |
| 370 | 15.823 | 8645 | 1024 | seeded_from_verified_log | `compiler_direct_output_algebra`, `compiler_tiny_lut_gather`, `compiler_single_conv_qlinear`, `compiler_final_equal_overlay` | same nonzero cell count with preserved colours suggests object-copy candidate |
| 324 | 15.907 | 7308 | 1586 | seeded_from_verified_log | `compiler_single_conv_qlinear`, `compiler_final_equal_overlay`, `compiler_qlinear_uint8`, `compiler_bounded_scan` | same nonzero cell count with preserved colours suggests object-copy candidate |
| 182 | 16.176 | 6695 | 98 | seeded_from_verified_log | `compiler_tiny_lut_gather`, `compiler_single_conv_qlinear`, `compiler_final_equal_overlay`, `compiler_qlinear_uint8` | same nonzero cell count with preserved colours suggests object-copy candidate |
| 368 | 16.264 | 6022 | 203 | seeded_from_verified_log | `compiler_tiny_lut_gather`, `compiler_final_equal_overlay`, `compiler_sparse_scatter` | same nonzero cell count with preserved colours suggests object-copy candidate |
| 008 | 16.365 | 5561 | 65 | seeded_from_verified_log | `compiler_tiny_lut_gather`, `compiler_final_equal_overlay`, `compiler_bounded_scan` | same nonzero cell count with preserved colours suggests object-copy candidate |
| 004 | 16.454 | 5044 | 100 | seeded_from_verified_log | `compiler_tiny_lut_gather`, `compiler_single_conv_qlinear`, `compiler_final_equal_overlay`, `compiler_qlinear_uint8` | same nonzero cell count with preserved colours suggests object-copy candidate |
| 363 | 16.673 | 3839 | 296 | seeded_from_verified_log | `compiler_single_conv_qlinear`, `compiler_final_equal_overlay`, `compiler_bounded_scan` | tasklog keyword match: \b(copy object\|copy the object\|object shape\|same shape\|identical shape) |
| 359 | 16.740 | 3852 | 15 | seeded_from_verified_log | `compiler_final_equal_overlay`, `compiler_bounded_scan` | same nonzero cell count with preserved colours suggests object-copy candidate |
| 093 | 16.823 | 3456 | 103 | seeded_from_verified_log | `compiler_tiny_lut_gather`, `compiler_final_equal_overlay`, `compiler_qlinear_uint8`, `compiler_bounded_scan` | same nonzero cell count with preserved colours suggests object-copy candidate |
| 250 | 16.921 | 3164 | 61 | seeded_from_verified_log | `compiler_tiny_lut_gather`, `compiler_final_equal_overlay`, `compiler_qlinear_uint8`, `compiler_bounded_scan` | same nonzero cell count with preserved colours suggests object-copy candidate |
| 035 | 16.954 | 2994 | 128 | seeded_from_verified_log | `compiler_sparse_scatter` | same nonzero cell count with preserved colours suggests object-copy candidate |
| 390 | 16.964 | 2624 | 467 | seeded_from_verified_log | `compiler_final_equal_overlay`, `compiler_sparse_scatter` | same nonzero cell count with preserved colours suggests object-copy candidate |
| 270 | 16.971 | 2742 | 327 | seeded_from_verified_log | `compiler_direct_output_algebra`, `compiler_bounded_scan` | same nonzero cell count with preserved colours suggests object-copy candidate |
| 154 | 17.077 | 2661 | 99 | seeded_from_verified_log | `compiler_tiny_lut_gather`, `compiler_final_equal_overlay` | same nonzero cell count with preserved colours suggests object-copy candidate |
| 354 | 17.080 | 2674 | 77 | seeded_from_verified_log | `compiler_single_conv_qlinear`, `compiler_final_equal_overlay`, `compiler_bounded_scan` | same nonzero cell count with preserved colours suggests object-copy candidate |
| 245 | 17.083 | 2604 | 139 | seeded_from_verified_log | `compiler_tiny_lut_gather`, `compiler_bounded_scan` | same nonzero cell count with preserved colours suggests object-copy candidate |
| 030 | 17.218 | 2248 | 148 | seeded_from_verified_log | `compiler_tiny_lut_gather`, `compiler_bounded_scan` | same nonzero cell count with preserved colours suggests object-copy candidate |
| 287 | 17.402 | 1699 | 295 | seeded_from_verified_log | `compiler_direct_output_algebra`, `compiler_tiny_lut_gather`, `compiler_final_equal_overlay` | same nonzero cell count with preserved colours suggests object-copy candidate |
| 088 | 17.465 | 1791 | 81 | seeded_from_verified_log | `compiler_direct_output_algebra`, `compiler_final_equal_overlay` | tasklog keyword match: \b(copy object\|copy the object\|object shape\|same shape\|identical shape) |
| 078 | 17.835 | 1260 | 33 | seeded_from_tasklog_and_inventory |  | same nonzero cell count with preserved colours suggests object-copy candidate |
| 293 | 17.839 | 1247 | 41 | seeded_from_verified_log | `compiler_tiny_lut_gather`, `compiler_final_equal_overlay`, `compiler_sparse_scatter` | same nonzero cell count with preserved colours suggests object-copy candidate |
| 301 | 17.960 | 1077 | 64 | seeded_from_verified_log | `compiler_tiny_lut_gather`, `compiler_final_equal_overlay`, `compiler_bounded_scan` | same nonzero cell count with preserved colours suggests object-copy candidate |
| 228 | 18.131 | 849 | 113 | seeded_from_verified_log | `compiler_direct_output_algebra`, `compiler_tiny_lut_gather`, `compiler_final_equal_overlay`, `compiler_sparse_scatter` | same nonzero cell count with preserved colours suggests object-copy candidate |
| 032 | 18.187 | 0 | 910 | seeded_from_verified_log | `compiler_single_conv_qlinear` | same nonzero cell count with preserved colours suggests object-copy candidate |
| 143 | 18.269 | 591 | 247 | seeded_from_verified_log | `compiler_direct_output_algebra`, `compiler_final_equal_overlay` | same nonzero cell count with preserved colours suggests object-copy candidate |
| 122 | 18.303 | 0 | 810 | seeded_from_verified_log | `compiler_single_conv_qlinear` | same nonzero cell count with preserved colours suggests object-copy candidate |
| 203 | 18.986 | 408 | 1 | seeded_from_tasklog_and_inventory | `compiler_tiny_lut_gather`, `compiler_final_equal_overlay`, `compiler_bounded_scan` | same nonzero cell count with preserved colours suggests object-copy candidate |
| 040 | 19.076 | 240 | 134 | seeded_from_verified_log | `compiler_direct_output_algebra`, `compiler_bounded_scan` | same nonzero cell count with preserved colours suggests object-copy candidate |
| 353 | 19.189 | 272 | 62 | seeded_from_verified_log | `compiler_direct_output_algebra`, `compiler_sparse_scatter` | same nonzero cell count with preserved colours suggests object-copy candidate |
| 128 | 19.296 | 0 | 300 | seeded_from_verified_log | `compiler_single_conv_qlinear` | same nonzero cell count with preserved colours suggests object-copy candidate |
| 249 | 19.341 | 248 | 39 | seeded_from_verified_log | `compiler_tiny_lut_gather` | tasklog keyword match: \b(copy object\|copy the object\|object shape\|same shape\|identical shape) |
| 313 | 19.743 | 0 | 192 | operator_evidence_only_needs_human_review | `compiler_direct_output_algebra` | same nonzero cell count with preserved colours suggests object-copy candidate |
| 150 | 19.937 | 128 | 30 | seeded_from_verified_log | `compiler_tiny_lut_gather` | same nonzero cell count with preserved colours suggests object-copy candidate |
| 155 | 19.937 | 128 | 30 | seeded_from_verified_log | `compiler_tiny_lut_gather` | same nonzero cell count with preserved colours suggests object-copy candidate |
| 314 | 20.395 | 0 | 100 | operator_evidence_only_needs_human_review | `compiler_direct_output_algebra`, `compiler_single_conv_qlinear` | same nonzero cell count with preserved colours suggests object-copy candidate |
| 380 | 20.405 | 0 | 99 | seeded_from_tasklog_and_inventory | `compiler_direct_output_algebra` | same nonzero cell count with preserved colours suggests object-copy candidate |
| 373 | 20.906 | 0 | 60 | seeded_from_verified_log | `compiler_direct_output_algebra` | same nonzero cell count with preserved colours suggests object-copy candidate |
| 053 | 21.599 | 0 | 30 | seeded_from_verified_log | `compiler_direct_output_algebra`, `compiler_tiny_lut_gather` | same nonzero cell count with preserved colours suggests object-copy candidate |
| 312 | 22.004 | 0 | 20 | seeded_from_verified_log | `compiler_direct_output_algebra` | same nonzero cell count with preserved colours suggests object-copy candidate |
| 337 | 22.697 | 0 | 10 | operator_evidence_only_needs_human_review | `compiler_direct_output_algebra`, `compiler_tiny_lut_gather` | same nonzero cell count with preserved colours suggests object-copy candidate |
| 087 | 23.391 | 0 | 5 | operator_evidence_only_needs_human_review | `compiler_direct_output_algebra`, `compiler_roi_pool_crop` | same nonzero cell count with preserved colours suggests object-copy candidate |
| 140 | 23.391 | 0 | 5 | seeded_from_verified_log | `compiler_direct_output_algebra`, `compiler_tiny_lut_gather`, `compiler_roi_pool_crop` | same nonzero cell count with preserved colours suggests object-copy candidate |
| 179 | 25.000 | 0 | 0 | operator_evidence_only_needs_human_review | `compiler_direct_output_algebra` | same nonzero cell count with preserved colours suggests object-copy candidate |
| 241 | 25.000 | 0 | 0 | operator_evidence_only_needs_human_review | `compiler_direct_output_algebra` | same nonzero cell count with preserved colours suggests object-copy candidate |

## Known Best Routes

- Pending human review.

## Kill Criteria

- Pending human review.

## Successful Applications

- Pending verification.

## Failed Applications / Walls

- Pending verification.
