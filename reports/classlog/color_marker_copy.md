# color_marker_copy — Copy marker/object colour

Colour is selected from a marker, hint, or source object.

## Optimization Question

What is the cheapest verified ONNX family for tasks in this class, and
which semantic preconditions let us avoid full-canvas intermediates?

## Candidate Tasks

| task | pts | mem | params | status | first routes | evidence |
|---:|---:|---:|---:|---|---|---|
| 018 | 14.157 | 48196 | 2987 | seeded_from_verified_log | `compiler_tiny_lut_gather`, `compiler_single_conv_qlinear`, `compiler_final_equal_overlay`, `compiler_sparse_scatter` | tasklog keyword match: \b(marker\|hint).{0,40}\b(colou?r\|palette)\|preserv(?:e\|ing).{0,40}(hint\|marker) colou?r |
| 118 | 14.559 | 30849 | 3387 | seeded_from_verified_log | `compiler_direct_output_algebra`, `compiler_direct_onehot_gather`, `compiler_tiny_lut_gather`, `compiler_single_conv_qlinear` | tasklog keyword match: \b(marker\|hint).{0,40}\b(colou?r\|palette)\|preserv(?:e\|ing).{0,40}(hint\|marker) colou?r |
| 044 | 15.651 | 11420 | 68 | seeded_from_verified_log | `compiler_direct_onehot_gather`, `compiler_tiny_lut_gather`, `compiler_single_conv_qlinear`, `compiler_final_equal_overlay` | tasklog keyword match: \b(marker\|hint).{0,40}\b(colou?r\|palette)\|preserv(?:e\|ing).{0,40}(hint\|marker) colou?r |
| 370 | 15.823 | 8645 | 1024 | seeded_from_verified_log | `compiler_direct_output_algebra`, `compiler_tiny_lut_gather`, `compiler_single_conv_qlinear`, `compiler_final_equal_overlay` | tasklog keyword match: \b(marker\|hint).{0,40}\b(colou?r\|palette)\|preserv(?:e\|ing).{0,40}(hint\|marker) colou?r |
| 089 | 16.111 | 7092 | 160 | seeded_from_verified_log | `compiler_tiny_lut_gather`, `compiler_single_conv_qlinear`, `compiler_final_equal_overlay`, `compiler_sparse_scatter` | tasklog keyword match: \b(marker\|hint).{0,40}\b(colou?r\|palette)\|preserv(?:e\|ing).{0,40}(hint\|marker) colou?r |
| 005 | 16.163 | 6392 | 490 | seeded_from_verified_log | `compiler_tiny_lut_gather`, `compiler_single_conv_qlinear`, `compiler_final_equal_overlay`, `compiler_qlinear_uint8` | tasklog keyword match: \b(marker\|hint).{0,40}\b(colou?r\|palette)\|preserv(?:e\|ing).{0,40}(hint\|marker) colou?r |
| 383 | 16.289 | 5974 | 96 | seeded_from_verified_log | `compiler_tiny_lut_gather`, `compiler_single_conv_qlinear`, `compiler_final_equal_overlay`, `compiler_bounded_scan` | tasklog keyword match: \b(marker\|hint).{0,40}\b(colou?r\|palette)\|preserv(?:e\|ing).{0,40}(hint\|marker) colou?r |
| 161 | 16.684 | 4055 | 33 | seeded_from_verified_log | `compiler_tiny_lut_gather`, `compiler_final_equal_overlay` | tasklog keyword match: \b(marker\|hint).{0,40}\b(colou?r\|palette)\|preserv(?:e\|ing).{0,40}(hint\|marker) colou?r |
| 069 | 16.728 | 3848 | 64 | seeded_from_verified_log | `compiler_tiny_lut_gather`, `compiler_single_conv_qlinear`, `compiler_final_equal_overlay`, `compiler_bounded_scan` | tasklog keyword match: \b(marker\|hint).{0,40}\b(colou?r\|palette)\|preserv(?:e\|ing).{0,40}(hint\|marker) colou?r |
| 355 | 17.093 | 2688 | 27 | seeded_from_verified_log | `compiler_final_equal_overlay`, `compiler_bounded_scan` | tasklog keyword match: \b(marker\|hint).{0,40}\b(colou?r\|palette)\|preserv(?:e\|ing).{0,40}(hint\|marker) colou?r |
| 260 | 17.344 | 1804 | 310 | seeded_from_verified_log | `compiler_tiny_lut_gather`, `compiler_final_equal_overlay`, `compiler_sparse_scatter`, `compiler_bounded_scan` | tasklog keyword match: \b(marker\|hint).{0,40}\b(colou?r\|palette)\|preserv(?:e\|ing).{0,40}(hint\|marker) colou?r |
| 088 | 17.465 | 1791 | 81 | seeded_from_verified_log | `compiler_direct_output_algebra`, `compiler_final_equal_overlay` | tasklog keyword match: \b(marker\|hint).{0,40}\b(colou?r\|palette)\|preserv(?:e\|ing).{0,40}(hint\|marker) colou?r |
| 230 | 18.198 | 0 | 900 | seeded_from_verified_log | `compiler_tiny_lut_gather`, `compiler_single_conv_qlinear` | tasklog keyword match: \b(marker\|hint).{0,40}\b(colou?r\|palette)\|preserv(?:e\|ing).{0,40}(hint\|marker) colou?r |
| 121 | 19.168 | 275 | 66 | seeded_from_verified_log | `compiler_direct_output_algebra`, `compiler_tiny_lut_gather` | tasklog keyword match: \b(marker\|hint).{0,40}\b(colou?r\|palette)\|preserv(?:e\|ing).{0,40}(hint\|marker) colou?r |

## Known Best Routes

- Pending human review.

## Kill Criteria

- Pending human review.

## Successful Applications

- Pending verification.

## Failed Applications / Walls

- Pending verification.
