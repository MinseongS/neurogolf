# action_clone_repeat — Clone/repeat

Object/template is repeated or tiled.

## Optimization Question

What is the cheapest verified ONNX family for tasks in this class, and
which semantic preconditions let us avoid full-canvas intermediates?

## Candidate Tasks

| task | pts | mem | params | status | first routes | evidence |
|---:|---:|---:|---:|---|---|---|
| 233 | 14.003 | 59147 | 565 | seeded_from_verified_log | `compiler_direct_onehot_gather`, `compiler_tiny_lut_gather`, `compiler_single_conv_qlinear`, `compiler_final_equal_overlay` | tasklog keyword match: \b(repeat\|tile\|clone\|duplicate\|copies of itself\|every \d+ cells) |
| 018 | 14.157 | 48196 | 2987 | seeded_from_verified_log | `compiler_tiny_lut_gather`, `compiler_single_conv_qlinear`, `compiler_final_equal_overlay`, `compiler_sparse_scatter` | tasklog keyword match: \b(repeat\|tile\|clone\|duplicate\|copies of itself\|every \d+ cells) |
| 366 | 14.497 | 35927 | 490 | seeded_from_verified_log | `compiler_tiny_lut_gather`, `compiler_single_conv_qlinear`, `compiler_final_equal_overlay`, `compiler_sparse_scatter` | tasklog keyword match: \b(repeat\|tile\|clone\|duplicate\|copies of itself\|every \d+ cells) |
| 158 | 14.528 | 32979 | 2340 | seeded_from_verified_log | `compiler_tiny_lut_gather`, `compiler_single_conv_qlinear`, `compiler_final_equal_overlay`, `compiler_qlinear_uint8` | tasklog keyword match: \b(repeat\|tile\|clone\|duplicate\|copies of itself\|every \d+ cells) |
| 054 | 14.829 | 25885 | 238 | seeded_from_tasklog_and_inventory | `compiler_tiny_lut_gather`, `compiler_single_conv_qlinear`, `compiler_final_equal_overlay`, `compiler_sparse_scatter` | tasklog keyword match: \b(repeat\|tile\|clone\|duplicate\|copies of itself\|every \d+ cells) |
| 285 | 14.848 | 25080 | 550 | seeded_from_verified_log | `compiler_tiny_lut_gather`, `compiler_single_conv_qlinear`, `compiler_final_equal_overlay`, `compiler_sparse_scatter` | tasklog keyword match: \b(repeat\|tile\|clone\|duplicate\|copies of itself\|every \d+ cells) |
| 364 | 14.956 | 22900 | 113 | seeded_from_verified_log | `compiler_tiny_lut_gather`, `compiler_single_conv_qlinear`, `compiler_final_equal_overlay`, `compiler_qlinear_uint8` | tasklog keyword match: \b(repeat\|tile\|clone\|duplicate\|copies of itself\|every \d+ cells) |
| 219 | 15.195 | 18033 | 93 | seeded_from_verified_log | `compiler_tiny_lut_gather`, `compiler_final_equal_overlay`, `compiler_bounded_scan` | tasklog keyword match: \b(repeat\|tile\|clone\|duplicate\|copies of itself\|every \d+ cells) |
| 066 | 15.256 | 16899 | 160 | seeded_from_verified_log | `compiler_direct_output_algebra`, `compiler_final_equal_overlay`, `compiler_bounded_scan` | tasklog keyword match: \b(repeat\|tile\|clone\|duplicate\|copies of itself\|every \d+ cells) |
| 110 | 15.468 | 13155 | 634 | seeded_from_verified_log | `compiler_tiny_lut_gather`, `compiler_single_conv_qlinear`, `compiler_final_equal_overlay`, `compiler_bounded_scan` | tasklog keyword match: \b(repeat\|tile\|clone\|duplicate\|copies of itself\|every \d+ cells) |
| 023 | 15.616 | 11603 | 291 | seeded_from_verified_log | `compiler_single_conv_qlinear`, `compiler_final_equal_overlay`, `compiler_qlinear_uint8` | tasklog keyword match: \b(repeat\|tile\|clone\|duplicate\|copies of itself\|every \d+ cells) |
| 338 | 15.712 | 10140 | 666 | seeded_from_verified_log | `compiler_tiny_lut_gather`, `compiler_single_conv_qlinear`, `compiler_final_equal_overlay`, `compiler_qlinear_uint8` | tasklog keyword match: \b(repeat\|tile\|clone\|duplicate\|copies of itself\|every \d+ cells) |
| 370 | 15.823 | 8645 | 1024 | seeded_from_verified_log | `compiler_direct_output_algebra`, `compiler_tiny_lut_gather`, `compiler_single_conv_qlinear`, `compiler_final_equal_overlay` | tasklog keyword match: \b(repeat\|tile\|clone\|duplicate\|copies of itself\|every \d+ cells) |
| 222 | 15.916 | 8736 | 78 | seeded_from_verified_log | `compiler_single_conv_qlinear`, `compiler_final_equal_overlay`, `compiler_qlinear_uint8`, `compiler_bounded_scan` | tasklog keyword match: \b(repeat\|tile\|clone\|duplicate\|copies of itself\|every \d+ cells) |
| 014 | 16.023 | 7809 | 108 | seeded_from_verified_log | `compiler_tiny_lut_gather`, `compiler_final_equal_overlay`, `compiler_bounded_scan` | tasklog keyword match: \b(repeat\|tile\|clone\|duplicate\|copies of itself\|every \d+ cells) |
| 202 | 16.039 | 7773 | 24 | seeded_from_verified_log | `compiler_single_conv_qlinear`, `compiler_final_equal_overlay`, `compiler_bounded_scan` | tasklog keyword match: \b(repeat\|tile\|clone\|duplicate\|copies of itself\|every \d+ cells) |
| 092 | 16.148 | 6805 | 182 | seeded_from_verified_log | `compiler_direct_output_algebra`, `compiler_final_equal_overlay`, `compiler_sparse_scatter`, `compiler_bounded_scan` | tasklog keyword match: \b(repeat\|tile\|clone\|duplicate\|copies of itself\|every \d+ cells) |
| 005 | 16.163 | 6392 | 490 | seeded_from_verified_log | `compiler_tiny_lut_gather`, `compiler_single_conv_qlinear`, `compiler_final_equal_overlay`, `compiler_qlinear_uint8` | tasklog keyword match: \b(repeat\|tile\|clone\|duplicate\|copies of itself\|every \d+ cells) |
| 182 | 16.176 | 6695 | 98 | seeded_from_verified_log | `compiler_tiny_lut_gather`, `compiler_single_conv_qlinear`, `compiler_final_equal_overlay`, `compiler_qlinear_uint8` | tasklog keyword match: \b(repeat\|tile\|clone\|duplicate\|copies of itself\|every \d+ cells) |
| 368 | 16.264 | 6022 | 203 | seeded_from_verified_log | `compiler_tiny_lut_gather`, `compiler_final_equal_overlay`, `compiler_sparse_scatter` | tasklog keyword match: \b(repeat\|tile\|clone\|duplicate\|copies of itself\|every \d+ cells) |
| 340 | 16.285 | 5812 | 279 | seeded_from_verified_log | `compiler_tiny_lut_gather`, `compiler_final_equal_overlay`, `compiler_qlinear_uint8`, `compiler_bounded_scan` | tasklog keyword match: \b(repeat\|tile\|clone\|duplicate\|copies of itself\|every \d+ cells) |
| 264 | 16.292 | 5348 | 706 | seeded_from_verified_log | `compiler_direct_output_algebra`, `compiler_tiny_lut_gather`, `compiler_single_conv_qlinear`, `compiler_final_equal_overlay` | tasklog keyword match: \b(repeat\|tile\|clone\|duplicate\|copies of itself\|every \d+ cells) |
| 148 | 16.323 | 5759 | 110 | seeded_from_verified_log | `compiler_final_equal_overlay`, `compiler_sparse_scatter`, `compiler_bounded_scan` | tasklog keyword match: \b(repeat\|tile\|clone\|duplicate\|copies of itself\|every \d+ cells) |
| 279 | 16.378 | 5508 | 47 | seeded_from_tasklog_and_inventory | `compiler_tiny_lut_gather`, `compiler_single_conv_qlinear`, `compiler_qlinear_uint8` | tasklog keyword match: \b(repeat\|tile\|clone\|duplicate\|copies of itself\|every \d+ cells) |
| 004 | 16.454 | 5044 | 100 | seeded_from_verified_log | `compiler_tiny_lut_gather`, `compiler_single_conv_qlinear`, `compiler_final_equal_overlay`, `compiler_qlinear_uint8` | tasklog keyword match: \b(repeat\|tile\|clone\|duplicate\|copies of itself\|every \d+ cells) |
| 117 | 16.666 | 3922 | 243 | seeded_from_verified_log | `compiler_tiny_lut_gather`, `compiler_single_conv_qlinear`, `compiler_final_equal_overlay`, `compiler_qlinear_uint8` | tasklog keyword match: \b(repeat\|tile\|clone\|duplicate\|copies of itself\|every \d+ cells) |
| 359 | 16.740 | 3852 | 15 | seeded_from_verified_log | `compiler_final_equal_overlay`, `compiler_bounded_scan` | tasklog keyword match: \b(repeat\|tile\|clone\|duplicate\|copies of itself\|every \d+ cells) |
| 093 | 16.823 | 3456 | 103 | seeded_from_verified_log | `compiler_tiny_lut_gather`, `compiler_final_equal_overlay`, `compiler_qlinear_uint8`, `compiler_bounded_scan` | tasklog keyword match: \b(repeat\|tile\|clone\|duplicate\|copies of itself\|every \d+ cells) |
| 394 | 17.063 | 1871 | 929 | seeded_from_verified_log | `compiler_tiny_lut_gather`, `compiler_final_equal_overlay`, `compiler_bounded_scan` | tasklog keyword match: \b(repeat\|tile\|clone\|duplicate\|copies of itself\|every \d+ cells) |
| 355 | 17.093 | 2688 | 27 | seeded_from_verified_log | `compiler_final_equal_overlay`, `compiler_bounded_scan` | tasklog keyword match: \b(repeat\|tile\|clone\|duplicate\|copies of itself\|every \d+ cells) |
| 388 | 17.178 | 2278 | 218 | seeded_from_verified_log | `compiler_final_equal_overlay`, `compiler_sparse_scatter`, `compiler_bounded_scan` | tasklog keyword match: \b(repeat\|tile\|clone\|duplicate\|copies of itself\|every \d+ cells) |
| 275 | 17.394 | 1374 | 637 | seeded_from_verified_log | `compiler_direct_output_algebra`, `compiler_single_conv_qlinear`, `compiler_final_equal_overlay`, `compiler_bounded_scan` | tasklog keyword match: \b(repeat\|tile\|clone\|duplicate\|copies of itself\|every \d+ cells) |
| 011 | 17.400 | 1836 | 163 | seeded_from_verified_log | `compiler_tiny_lut_gather`, `compiler_single_conv_qlinear`, `compiler_final_equal_overlay` | tasklog keyword match: \b(repeat\|tile\|clone\|duplicate\|copies of itself\|every \d+ cells) |
| 124 | 17.423 | 1879 | 74 | seeded_from_verified_log | `compiler_direct_onehot_gather`, `compiler_tiny_lut_gather`, `compiler_final_equal_overlay`, `compiler_bounded_scan` | tasklog keyword match: \b(repeat\|tile\|clone\|duplicate\|copies of itself\|every \d+ cells) |
| 169 | 17.579 | 1500 | 170 | seeded_from_verified_log | `compiler_tiny_lut_gather`, `compiler_single_conv_qlinear`, `compiler_final_equal_overlay`, `compiler_qlinear_uint8` | tasklog keyword match: \b(repeat\|tile\|clone\|duplicate\|copies of itself\|every \d+ cells) |
| 033 | 17.598 | 1606 | 33 | seeded_from_verified_log |  | tasklog keyword match: \b(repeat\|tile\|clone\|duplicate\|copies of itself\|every \d+ cells) |
| 240 | 17.704 | 1393 | 82 | seeded_from_verified_log | `compiler_tiny_lut_gather`, `compiler_final_equal_overlay`, `compiler_bounded_scan` | tasklog keyword match: \b(repeat\|tile\|clone\|duplicate\|copies of itself\|every \d+ cells) |
| 304 | 17.727 | 1404 | 37 | seeded_from_tasklog_and_inventory | `compiler_single_conv_qlinear`, `compiler_final_equal_overlay` | integer scale suggests tiling/repetition candidate |
| 123 | 17.797 | 403 | 940 | seeded_from_verified_log | `compiler_direct_output_algebra`, `compiler_tiny_lut_gather`, `compiler_final_equal_overlay` | integer scale suggests tiling/repetition candidate |
| 027 | 17.822 | 1262 | 48 | seeded_from_verified_log |  | tasklog keyword match: \b(repeat\|tile\|clone\|duplicate\|copies of itself\|every \d+ cells) |
| 175 | 17.856 | 330 | 937 | seeded_from_verified_log | `compiler_tiny_lut_gather`, `compiler_final_equal_overlay`, `compiler_bounded_scan` | tasklog keyword match: \b(repeat\|tile\|clone\|duplicate\|copies of itself\|every \d+ cells) |
| 343 | 17.892 | 1147 | 75 | seeded_from_verified_log | `compiler_tiny_lut_gather`, `compiler_single_conv_qlinear`, `compiler_final_equal_overlay`, `compiler_bounded_scan` | tasklog keyword match: \b(repeat\|tile\|clone\|duplicate\|copies of itself\|every \d+ cells) |
| 188 | 17.921 | 1128 | 59 | seeded_from_verified_log | `compiler_single_conv_qlinear`, `compiler_final_equal_overlay`, `compiler_qlinear_uint8`, `compiler_bounded_scan` | tasklog keyword match: \b(repeat\|tile\|clone\|duplicate\|copies of itself\|every \d+ cells) |
| 115 | 18.050 | 980 | 63 | seeded_from_verified_log | `compiler_direct_output_algebra`, `compiler_bounded_scan` | tasklog keyword match: \b(repeat\|tile\|clone\|duplicate\|copies of itself\|every \d+ cells) |
| 108 | 18.122 | 864 | 107 | seeded_from_verified_log | `compiler_single_conv_qlinear`, `compiler_final_equal_overlay`, `compiler_roi_pool_crop` | integer scale suggests tiling/repetition candidate |
| 194 | 18.145 | 900 | 49 | seeded_from_verified_log | `compiler_tiny_lut_gather` | integer scale suggests tiling/repetition candidate |
| 195 | 18.175 | 882 | 39 | seeded_from_verified_log | `compiler_tiny_lut_gather`, `compiler_bounded_scan` | tasklog keyword match: \b(repeat\|tile\|clone\|duplicate\|copies of itself\|every \d+ cells) |
| 327 | 18.219 | 810 | 71 | seeded_from_verified_log | `compiler_single_conv_qlinear`, `compiler_final_equal_overlay` | integer scale suggests tiling/repetition candidate; tasklog keyword match: \b(repeat\|tile\|clone\|duplicate\|copies of itself\|every \d+ cells) |
| 122 | 18.303 | 0 | 810 | seeded_from_verified_log | `compiler_single_conv_qlinear` | tasklog keyword match: \b(repeat\|tile\|clone\|duplicate\|copies of itself\|every \d+ cells) |
| 263 | 18.478 | 563 | 117 | seeded_from_verified_log | `compiler_tiny_lut_gather`, `compiler_single_conv_qlinear`, `compiler_final_equal_overlay` | tasklog keyword match: \b(repeat\|tile\|clone\|duplicate\|copies of itself\|every \d+ cells) |
| 057 | 18.496 | 638 | 30 | seeded_from_verified_log | `compiler_tiny_lut_gather`, `compiler_final_equal_overlay`, `compiler_bounded_scan` | tasklog keyword match: \b(repeat\|tile\|clone\|duplicate\|copies of itself\|every \d+ cells) |
| 152 | 18.556 | 504 | 125 | operator_evidence_only_needs_human_review | `compiler_single_conv_qlinear`, `compiler_sparse_scatter` | integer scale suggests tiling/repetition candidate |
| 059 | 18.677 | 265 | 292 | seeded_from_verified_log | `compiler_direct_output_algebra`, `compiler_final_equal_overlay`, `compiler_bounded_scan` | tasklog keyword match: \b(repeat\|tile\|clone\|duplicate\|copies of itself\|every \d+ cells) |
| 083 | 18.897 | 376 | 71 | seeded_from_verified_log | `compiler_tiny_lut_gather`, `compiler_sparse_scatter`, `compiler_bounded_scan` | integer scale suggests tiling/repetition candidate |
| 316 | 18.946 | 388 | 38 | seeded_from_verified_log | `compiler_direct_output_algebra`, `compiler_tiny_lut_gather`, `compiler_final_equal_overlay` | tasklog keyword match: \b(repeat\|tile\|clone\|duplicate\|copies of itself\|every \d+ cells) |
| 211 | 19.019 | 240 | 156 | seeded_from_verified_log | `compiler_direct_output_algebra` | integer scale suggests tiling/repetition candidate; tasklog keyword match: \b(repeat\|tile\|clone\|duplicate\|copies of itself\|every \d+ cells) |
| 181 | 19.089 | 201 | 168 | seeded_from_verified_log | `compiler_sparse_scatter` | tasklog keyword match: \b(repeat\|tile\|clone\|duplicate\|copies of itself\|every \d+ cells) |
| 296 | 19.201 | 285 | 45 | seeded_from_verified_log | `compiler_final_equal_overlay` | tasklog keyword match: \b(repeat\|tile\|clone\|duplicate\|copies of itself\|every \d+ cells) |
| 306 | 19.296 | 0 | 300 | seeded_from_verified_log | `compiler_direct_output_algebra`, `compiler_tiny_lut_gather` | tasklog keyword match: \b(repeat\|tile\|clone\|duplicate\|copies of itself\|every \d+ cells) |
| 249 | 19.341 | 248 | 39 | seeded_from_verified_log | `compiler_tiny_lut_gather` | tasklog keyword match: \b(repeat\|tile\|clone\|duplicate\|copies of itself\|every \d+ cells) |
| 142 | 19.358 | 144 | 138 | seeded_from_verified_log | `compiler_direct_output_algebra` | integer scale suggests tiling/repetition candidate |
| 104 | 19.463 | 197 | 57 | seeded_from_verified_log | `compiler_direct_onehot_gather`, `compiler_tiny_lut_gather`, `compiler_single_conv_qlinear`, `compiler_qlinear_uint8` | integer scale suggests tiling/repetition candidate |
| 001 | 19.519 | 0 | 240 | seeded_from_verified_log | `compiler_direct_output_algebra`, `compiler_tiny_lut_gather` | integer scale suggests tiling/repetition candidate; tasklog keyword match: \b(repeat\|tile\|clone\|duplicate\|copies of itself\|every \d+ cells) |
| 315 | 19.562 | 0 | 230 | seeded_from_verified_log | `compiler_direct_output_algebra` | integer scale suggests tiling/repetition candidate |
| 231 | 19.611 | 180 | 39 | seeded_from_verified_log | `compiler_tiny_lut_gather` | tasklog keyword match: \b(repeat\|tile\|clone\|duplicate\|copies of itself\|every \d+ cells) |
| 386 | 19.648 | 180 | 31 | seeded_from_verified_log |  | tasklog keyword match: \b(repeat\|tile\|clone\|duplicate\|copies of itself\|every \d+ cells) |
| 116 | 21.599 | 0 | 30 | operator_evidence_only_needs_human_review | `compiler_direct_output_algebra`, `compiler_tiny_lut_gather` | integer scale suggests tiling/repetition candidate |
| 130 | 21.599 | 0 | 30 | seeded_from_verified_log | `compiler_direct_output_algebra`, `compiler_single_conv_qlinear` | tasklog keyword match: \b(repeat\|tile\|clone\|duplicate\|copies of itself\|every \d+ cells) |
| 164 | 21.599 | 0 | 30 | operator_evidence_only_needs_human_review | `compiler_direct_output_algebra`, `compiler_tiny_lut_gather` | integer scale suggests tiling/repetition candidate |
| 172 | 21.599 | 0 | 30 | operator_evidence_only_needs_human_review | `compiler_direct_output_algebra`, `compiler_tiny_lut_gather` | integer scale suggests tiling/repetition candidate |
| 210 | 21.599 | 0 | 30 | operator_evidence_only_needs_human_review | `compiler_direct_output_algebra`, `compiler_tiny_lut_gather` | integer scale suggests tiling/repetition candidate |
| 311 | 21.599 | 0 | 30 | operator_evidence_only_needs_human_review | `compiler_direct_output_algebra`, `compiler_tiny_lut_gather` | integer scale suggests tiling/repetition candidate |
| 326 | 21.599 | 0 | 30 | seeded_from_verified_log | `compiler_direct_output_algebra` | tasklog keyword match: \b(repeat\|tile\|clone\|duplicate\|copies of itself\|every \d+ cells) |
| 223 | 23.391 | 0 | 5 | operator_evidence_only_needs_human_review | `compiler_direct_output_algebra`, `compiler_roi_pool_crop` | integer scale suggests tiling/repetition candidate |

## Known Best Routes

- Pending human review.

## Kill Criteria

- Pending human review.

## Successful Applications

- Pending verification.

## Failed Applications / Walls

- Pending verification.
