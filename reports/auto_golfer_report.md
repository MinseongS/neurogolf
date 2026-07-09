# Auto Golfer Report

## Probe Wins
| delta | task | cost | candidate |
|---:|---:|---:|---|
| +0.001126 | 354 | 2666->2663 | `reports/candidates/task354/task354_dedupe_initializers.onnx` |

## Probe Summary
| probe | seconds | wins | returncode |
|---|---:|---:|---:|
| `reports/candidates/dedupe_initializers_active_probe.py` | 0.551 | 1 | 0 |
| `reports/candidates/prune_dead_constants_active_probe.py` | 0.144 | 0 | 0 |
| `reports/candidates/noop_reshape_active_probe.py` | 0.207 | 0 | 0 |
| `reports/candidates/dynamic_cse_active_probe.py` | 45 | 0 | None |
| `reports/candidates/cast_dtype_batch_probe.py` | 19.63 | 0 | 0 |
| `reports/candidates/zero_concat_to_pad_active_probe.py` | 0.607 | 0 | 0 |
| `reports/candidates/contiguous_gather_to_slice_active_probe.py` | 0.143 | 0 | 0 |
| `reports/candidates/negative_pad_normalize_probe.py` | 45 | 0 | None |
| `reports/candidates/sign_argmax_uint8_active_probe.py` | 0.148 | 0 | 0 |
| `reports/candidates/defer_widening_cast_shape_probe.py` | 12.266 | 0 | 0 |
| `reports/candidates/remove_gather_index_cast_probe.py` | 0.443 | 0 | 0 |

## Semantic Compiler Targets
| rank | task | score | cost | penalty | dominant tags | top tensors |
|---:|---:|---:|---:|---:|---|---|
| 1 | 096 | 756.5 | 7682 | 2700.0 | full_canvas:3300, producer_bound_candidate:2400, other:1668, final_output_welded:1604 | `row_sum_full:ReduceSum:1200:full_canvas/producer_bound_candidate; col_sum_full:ReduceSum:1200:full_canvas/producer_bound_candidate; padded_color:Pad:900:final_output_welded/full_canvas; radius_i32:Slice:484:index_or_profile` |
| 2 | 019 | 486.2 | 2793 | 0.0 | final_output_welded:1343, other:1154, full_canvas:900, pad_concat_carrier:900 | `safe_name_45:Pad:900:final_output_welded/full_canvas; safe_name_15:Slice:144:other; safe_name_30:And:144:other; safe_name_37:Gather:144:other` |
| 3 | 154 | 471.0 | 2759 | 0.0 | other:1521, final_output_welded:1092, full_canvas:900, pad_concat_carrier:900 | `class30:Pad:900:final_output_welded/full_canvas; red_f:Slice:400:other; class15:Pad:225:other; red_u8:Cast:100:other` |
| 4 | 295 | 420.8 | 1604 | 0.0 | final_output_welded:1269, full_canvas:900, pad_concat_carrier:900, other:229 | `full_label:Pad:900:final_output_welded/full_canvas; filled:Less:162:final_output_welded; default_label:Where:162:final_output_welded; small_label:Where:162:other` |
| 5 | 187 | 250.0 | 8322 | 3500.0 | final_output_welded:5000, producer_bound_candidate:5000 | `t2:Conv:5000:final_output_welded/producer_bound_candidate` |
| 6 | 197 | 235.0 | 1201 | 0.0 | index_or_profile:600, other:420, final_output_welded:260 | `g1_sliced:Slice:400:index_or_profile; masked:Where:200:other; indices:Concat:120:final_output_welded/index_or_profile; eq_mat:Equal:100:final_output_welded` |
| 7 | 008 | 220.5 | 3198 | 0.0 | final_output_welded:1998, other:1066, full_canvas:900, pad_concat_carrier:900 | `lab30:Pad:900:final_output_welded/full_canvas; m8:Pad:256:final_output_welded; redplaced:Pad:256:final_output_welded; base:Where:256:final_output_welded` |
| 8 | 232 | 0.0 | 1410 | 0.0 |  | `` |
| 9 | 060 | 0.0 | 1010 | 0.0 |  | `` |
| 10 | 294 | 0.0 | 910 | 0.0 |  | `` |
| 11 | 344 | 0.0 | 910 | 0.0 |  | `` |
| 12 | 098 | 0.0 | 900 | 0.0 |  | `` |
| 13 | 114 | 0.0 | 900 | 0.0 |  | `` |
| 14 | 151 | 0.0 | 900 | 0.0 |  | `` |
| 15 | 176 | 0.0 | 210 | 0.0 |  | `` |
| 16 | 026 | 0.0 | 200 | 0.0 |  | `` |
| 17 | 135 | 0.0 | 200 | 0.0 |  | `` |
| 18 | 229 | 0.0 | 200 | 0.0 |  | `` |
| 19 | 313 | 0.0 | 188 | 0.0 |  | `` |
| 20 | 144 | 0.0 | 104 | 0.0 |  | `` |
| 21 | 314 | 0.0 | 100 | 0.0 |  | `` |
| 22 | 380 | 0.0 | 99 | 0.0 |  | `` |
| 23 | 152 | 0.0 | 90 | 0.0 |  | `` |
| 24 | 299 | 0.0 | 50 | 0.0 |  | `` |
| 25 | 116 | 0.0 | 30 | 0.0 |  | `` |
| 26 | 164 | 0.0 | 30 | 0.0 |  | `` |
| 27 | 172 | 0.0 | 30 | 0.0 |  | `` |
| 28 | 210 | 0.0 | 30 | 0.0 |  | `` |
| 29 | 311 | 0.0 | 30 | 0.0 |  | `` |
| 30 | 385 | 0.0 | 30 | 0.0 |  | `` |
