# Known Insight Coverage

Active-overfit audit: `submission/overfit_nets/` x `reports/insight_registry.yaml`.

States are heuristic triage labels, not proof of applicability.

## Top Actionable Gaps

| rank | task | insight | state | priority | cost | pts | tags |
|---:|---:|---|---|---:|---:|---:|---|
| 1 | 133 | `threshold_linearize_pairwise_onehot_and` | actionable | 26047.6 | 21322 | 15.033 | assignment_wall,connectivity_wall,conv_heavy,documented_wall,free_fp32_input_quantize_required,high_memory,lut_selection,onehot_final_equal |
| 2 | 054 | `pad_compensated_spatial_crop` | actionable | 25313.5 | 20131 | 15.090 | free_fp32_input_quantize_required,gather_heavy,high_memory,onehot_final_equal,scan,scatter |
| 3 | 138 | `walk_einsum_iteration_collapse` | actionable | 15992.5 | 11716 | 15.631 | conv_heavy,documented_wall,free_fp32_input_quantize_required,high_memory,local_stencil,onehot_final_equal |
| 4 | 198 | `strided_conv_fixed_block_counts` | actionable | 15666.7 | 11410 | 15.658 | connectivity_wall,documented_wall,free_fp32_input_quantize_required,gather_heavy,high_memory,local_stencil,onehot_final_equal,scan |
| 5 | 198 | `walk_einsum_iteration_collapse` | actionable | 15666.7 | 11410 | 15.658 | connectivity_wall,documented_wall,free_fp32_input_quantize_required,gather_heavy,high_memory,local_stencil,onehot_final_equal,scan |
| 6 | 173 | `walk_einsum_iteration_collapse` | actionable | 15570.7 | 11320 | 15.666 | free_fp32_input_quantize_required,gather_heavy,high_memory,lut_selection,onehot_final_equal,scatter |
| 7 | 350 | `walk_einsum_iteration_collapse` | actionable | 13117.7 | 9036 | 15.891 | connectivity_wall,documented_wall,lut_selection |
| 8 | 255 | `pad_compensated_spatial_crop` | actionable | 12097.0 | 7597 | 16.064 | conv_heavy,onehot_final_equal,qlinear |
| 9 | 255 | `zero_concat_tail_to_pad` | actionable | 12097.0 | 7597 | 16.064 | conv_heavy,onehot_final_equal,qlinear |
| 10 | 255 | `band_profile_contraction_final_equal` | actionable | 11597.0 | 7597 | 16.064 | conv_heavy,onehot_final_equal,qlinear |
| 11 | 255 | `free_final_onehot_equal` | actionable | 11597.0 | 7597 | 16.064 | conv_heavy,onehot_final_equal,qlinear |
| 12 | 255 | `solid_marker_profile_reconstruction` | actionable | 11597.0 | 7597 | 16.064 | conv_heavy,onehot_final_equal,qlinear |
| 13 | 219 | `uint8_presence_for_argmax` | actionable | 11210.0 | 7210 | 16.117 | assignment_wall,connectivity_wall,documented_wall,lut_selection,onehot_final_equal,scan |
| 14 | 157 | `strided_conv_fixed_block_counts` | actionable | 10851.0 | 6851 | 16.168 | assignment_wall,connectivity_wall,documented_wall,gather_heavy,lut_selection,onehot_final_equal,scatter |
| 15 | 089 | `threshold_linearize_pairwise_onehot_and` | actionable | 10715.0 | 6715 | 16.188 | assignment_wall,connectivity_wall,documented_wall,free_fp32_input_quantize_required,onehot_final_equal,scatter |
| 16 | 328 | `strided_conv_fixed_block_counts` | actionable | 10631.0 | 6631 | 16.200 | documented_wall,lut_selection,onehot_final_equal |
| 17 | 255 | `high_score_frontier_final_output_only` | actionable | 10597.0 | 7597 | 16.064 | conv_heavy,onehot_final_equal,qlinear |
| 18 | 023 | `band_profile_contraction_final_equal` | actionable | 10348.0 | 6348 | 16.244 | conv_heavy,local_stencil,onehot_final_equal,qlinear |
| 19 | 182 | `strided_conv_fixed_block_counts` | actionable | 10069.0 | 6069 | 16.289 | assignment_wall,connectivity_wall,conv_heavy,documented_wall,free_fp32_input_quantize_required,local_stencil,lut_selection,onehot_final_equal |
| 20 | 192 | `high_score_frontier_final_output_only` | actionable | 9833.0 | 6833 | 16.170 | documented_wall,onehot_final_equal,qlinear |
| 21 | 382 | `band_profile_contraction_final_equal` | actionable | 9626.0 | 5626 | 16.365 | onehot_final_equal,scan |
| 22 | 279 | `band_profile_contraction_final_equal` | actionable | 9546.0 | 5546 | 16.379 | conv_heavy,local_stencil,lut_selection,qlinear |
| 23 | 279 | `category_augmented_separable_lut` | actionable | 9546.0 | 5546 | 16.379 | conv_heavy,local_stencil,lut_selection,qlinear |
| 24 | 279 | `threshold_linearize_pairwise_onehot_and` | actionable | 9546.0 | 5546 | 16.379 | conv_heavy,local_stencil,lut_selection,qlinear |
| 25 | 023 | `high_score_frontier_final_output_only` | actionable | 9348.0 | 6348 | 16.244 | conv_heavy,local_stencil,onehot_final_equal,qlinear |
| 26 | 396 | `public_teacher_qlinear_conv_rewrite` | actionable | 8926.0 | 4926 | 16.498 | conv_heavy,local_stencil,onehot_final_equal |
| 27 | 148 | `threshold_linearize_pairwise_onehot_and` | actionable | 8744.0 | 4744 | 16.535 | documented_wall,onehot_final_equal,scatter |
| 28 | 382 | `high_score_frontier_final_output_only` | actionable | 8626.0 | 5626 | 16.365 | onehot_final_equal,scan |
| 29 | 377 | `category_augmented_separable_lut` | actionable | 8567.0 | 4567 | 16.573 | onehot_final_equal,scan |
| 30 | 377 | `free_final_onehot_equal` | actionable | 8567.0 | 4567 | 16.573 | onehot_final_equal,scan |
| 31 | 377 | `solid_marker_profile_reconstruction` | actionable | 8567.0 | 4567 | 16.573 | onehot_final_equal,scan |
| 32 | 377 | `uint8_presence_for_argmax` | actionable | 8567.0 | 4567 | 16.573 | onehot_final_equal,scan |
| 33 | 279 | `high_score_frontier_final_output_only` | actionable | 8546.0 | 5546 | 16.379 | conv_heavy,local_stencil,lut_selection,qlinear |
| 34 | 278 | `high_score_frontier_final_output_only` | actionable | 8010.0 | 5010 | 16.481 | conv_heavy,free_fp32_input_quantize_required,local_stencil,lut_selection,qlinear |
| 35 | 277 | `threshold_linearize_pairwise_onehot_and` | actionable | 7741.0 | 3741 | 16.773 | connectivity_wall,lut_selection,maxpool_scan,onehot_final_equal |
| 36 | 093 | `solid_marker_profile_reconstruction` | actionable | 7531.0 | 3531 | 16.831 | connectivity_wall,onehot_final_equal,qlinear |
| 37 | 125 | `solid_marker_profile_reconstruction` | actionable | 7463.0 | 3463 | 16.850 | connectivity_wall,documented_wall,maxpool_scan,onehot_final_equal |
| 38 | 154 | `direct_onehot_gather_output` | actionable | 7243.0 | 2743 | 17.083 | onehot_final_equal |
| 39 | 154 | `zero_concat_tail_to_pad` | actionable | 7243.0 | 2743 | 17.083 | onehot_final_equal |
| 40 | 310 | `strided_conv_fixed_block_counts` | actionable | 7227.0 | 3227 | 16.921 | connectivity_wall,documented_wall,onehot_final_equal |
| 41 | 201 | `threshold_linearize_pairwise_onehot_and` | actionable | 7183.0 | 3183 | 16.934 | assignment_wall,free_fp32_input_quantize_required,local_stencil,onehot_final_equal |
| 42 | 008 | `strided_conv_fixed_block_counts` | actionable | 7150.0 | 3150 | 16.945 | lut_selection,onehot_final_equal |
| 43 | 330 | `solid_marker_profile_reconstruction` | actionable | 7022.0 | 3022 | 16.986 | connectivity_wall,documented_wall,onehot_final_equal,scatter |
| 44 | 330 | `threshold_linearize_pairwise_onehot_and` | actionable | 7022.0 | 3022 | 16.986 | connectivity_wall,documented_wall,onehot_final_equal,scatter |
| 45 | 398 | `qlinear_uint8_lut_or_matmul` | actionable | 6813.0 | 2813 | 17.058 | matmul,onehot_final_equal |
| 46 | 398 | `strided_conv_fixed_block_counts` | actionable | 6813.0 | 2813 | 17.058 | matmul,onehot_final_equal |
| 47 | 234 | `strided_conv_fixed_block_counts` | actionable | 6780.0 | 2780 | 17.070 | connectivity_wall,onehot_final_equal |
| 48 | 397 | `threshold_linearize_pairwise_onehot_and` | actionable | 6751.0 | 2751 | 17.080 | free_fp32_input_quantize_required,local_stencil,onehot_final_equal,scatter |
| 49 | 154 | `category_augmented_separable_lut` | actionable | 6743.0 | 2743 | 17.083 | onehot_final_equal |
| 50 | 154 | `self_einsum_axis_activity_gate` | actionable | 6743.0 | 2743 | 17.083 | onehot_final_equal |
| 51 | 154 | `solid_marker_profile_reconstruction` | actionable | 6743.0 | 2743 | 17.083 | onehot_final_equal |
| 52 | 154 | `threshold_linearize_pairwise_onehot_and` | actionable | 6743.0 | 2743 | 17.083 | onehot_final_equal |
| 53 | 094 | `public_teacher_qlinear_conv_rewrite` | actionable | 6677.0 | 2677 | 17.108 | connectivity_wall,conv_heavy,documented_wall,local_stencil |
| 54 | 354 | `public_teacher_qlinear_conv_rewrite` | actionable | 6644.0 | 2644 | 17.120 | conv_heavy,local_stencil,onehot_final_equal |
| 55 | 256 | `zero_concat_tail_to_pad` | actionable | 6563.0 | 2063 | 17.368 | onehot_final_equal |
| 56 | 170 | `threshold_linearize_pairwise_onehot_and` | actionable | 6537.0 | 2537 | 17.161 | assignment_wall,conv_heavy,documented_wall,free_fp32_input_quantize_required,local_stencil,lut_selection,onehot_final_equal |
| 57 | 062 | `threshold_linearize_pairwise_onehot_and` | actionable | 6524.0 | 2524 | 17.166 | connectivity_wall,onehot_final_equal |
| 58 | 019 | `threshold_linearize_pairwise_onehot_and` | actionable | 6503.0 | 2503 | 17.175 | local_stencil,onehot_final_equal,qlinear |
| 59 | 185 | `direct_onehot_gather_output` | actionable | 6455.0 | 1955 | 17.422 | conv_heavy,documented_wall,free_fp32_input_quantize_required,local_stencil,onehot_final_equal |
| 60 | 185 | `sparse_conv_single_op_floor` | actionable | 6455.0 | 1955 | 17.422 | conv_heavy,documented_wall,free_fp32_input_quantize_required,local_stencil,onehot_final_equal |

## Insight Coverage Summary

| insight | source | logged | graph | actionable | blocked | unlowered | probe_no_win | candidate_logged | top tasks |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `category_augmented_separable_lut` | 1 | 194 | 0 | 3 | 23 | 0 | 0 | 1 | 279:actionable:9546, 255:candidate_logged:8597, 377:actionable:8567, 154:actionable:6743, 349:known_blocked:-10151 |
| `high_score_frontier_final_output_only` | 10 | 149 | 0 | 6 | 20 | 0 | 0 | 0 | 255:actionable:10597, 192:actionable:9833, 023:actionable:9348, 382:actionable:8626, 279:actionable:8546 |
| `threshold_linearize_pairwise_onehot_and` | 1 | 27 | 93 | 26 | 154 | 1 | 0 | 1 | 133:actionable:26047, 286:candidate_graphevidence:18380, 089:actionable:10715, 279:actionable:9546, 148:actionable:8744 |
| `qlinear_uint8_lut_or_matmul` | 4 | 56 | 43 | 2 | 2 | 0 | 0 | 0 | 398:actionable:6813, 124:actionable:5953, 174:known_blocked:-21153, 105:known_blocked:-22651 |
| `free_final_onehot_equal` | 3 | 178 | 1 | 5 | 151 | 0 | 0 | 2 | 255:actionable:11597, 377:actionable:8567, 185:actionable:5955, 304:actionable:5441, 197:actionable:4904 |
| `scan_dtype_and_shift_compression` | 2 | 21 | 108 | 0 | 19 | 0 | 0 | 0 | 173:candidate_graphevidence:10070, 324:candidate_graphevidence:7429, 382:candidate_graphevidence:4126, 377:candidate_graphevidence:3067, 233:byte_negative:2212 |
| `sparse_conv_single_op_floor` | 3 | 148 | 47 | 1 | 45 | 0 | 0 | 4 | 255:candidate_logged:9097, 185:actionable:6455, 232:candidate_logged:2910, 160:candidate_logged:2834, 344:candidate_logged:2410 |
| `exact_preserve_to_semantic_rewrite` | 4 | 85 | 0 | 0 | 0 | 0 | 0 | 0 |  |
| `dihedral_template_match_stacked_conv` | 1 | 21 | 0 | 0 | 0 | 0 | 0 | 0 |  |
| `bounded_crop_before_connectivity_scan` | 1 | 199 | 0 | 0 | 17 | 0 | 0 | 0 | 233:byte_negative:3212, 366:byte_negative:2078, 018:known_blocked:-316, 133:byte_negative:-8452, 158:byte_negative:-9481 |
| `public_teacher_bitwise_scan_replacement` | 2 | 65 | 0 | 0 | 6 | 0 | 0 | 0 | 233:byte_negative:2212, 018:known_blocked:-1316, 133:byte_negative:-9452, 158:byte_negative:-10481, 285:byte_negative:-11213 |
| `public_teacher_qlinear_conv_rewrite` | 8 | 24 | 0 | 3 | 2 | 0 | 0 | 0 | 396:actionable:8926, 094:actionable:6677, 354:actionable:6644, 363:known_blocked:-21466, 042:byte_negative:-27390 |
| `marker_routed_hidden_path_compiler` | 1 | 89 | 0 | 0 | 0 | 0 | 0 | 0 |  |
| `rotation_component_template_scatter` | 1 | 92 | 0 | 0 | 5 | 0 | 0 | 0 | 018:known_blocked:-1316, 054:byte_negative:-10686, 285:byte_negative:-11213, 286:byte_negative:-12619, 367:byte_negative:-15104 |
| `band_profile_contraction_final_equal` | 1 | 201 | 0 | 4 | 25 | 0 | 0 | 0 | 255:actionable:11597, 023:actionable:10348, 382:actionable:9626, 279:actionable:9546, 018:known_blocked:-316 |
| `zero_concat_tail_to_pad` | 0 | 117 | 0 | 5 | 147 | 0 | 0 | 1 | 255:actionable:12097, 154:actionable:7243, 256:actionable:6563, 185:actionable:6455, 304:actionable:5941 |
| `direct_onehot_gather_output` | 0 | 79 | 23 | 3 | 179 | 0 | 0 | 4 | 255:candidate_logged:9097, 154:actionable:7243, 185:actionable:6455, 304:actionable:5941, 233:byte_negative:3712 |
| `uint8_topk_compact_label_grid` | 0 | 99 | 25 | 0 | 15 | 0 | 0 | 0 | 366:byte_negative:2078, 101:byte_negative:-16379, 233:kaggle_falsified:-16787, 368:known_blocked:-20370, 361:known_blocked:-20812 |
| `strided_conv_fixed_block_counts` | 1 | 48 | 0 | 12 | 18 | 0 | 0 | 0 | 198:actionable:15666, 157:actionable:10851, 328:actionable:10631, 182:actionable:10069, 310:actionable:7227 |
| `solid_marker_profile_reconstruction` | 0 | 141 | 0 | 11 | 136 | 1 | 0 | 1 | 255:actionable:11597, 377:actionable:8567, 093:actionable:7531, 125:actionable:7463, 330:actionable:7022 |
| `walk_einsum_iteration_collapse` | 5 | 78 | 0 | 4 | 19 | 0 | 0 | 0 | 138:actionable:15992, 198:actionable:15666, 173:actionable:15570, 350:actionable:13117, 233:byte_negative:3212 |
| `self_einsum_axis_activity_gate` | 1 | 34 | 0 | 10 | 197 | 0 | 0 | 1 | 255:candidate_logged:8597, 154:actionable:6743, 256:actionable:6063, 185:actionable:5955, 304:actionable:5441 |
| `label_pad_vs_onehot_pad_ordering` | 0 | 8 | 0 | 0 | 193 | 0 | 19 | 0 | 054:probe_no_win:8813, 233:byte_negative:3712, 366:byte_negative:2578, 018:known_blocked:183, 255:probe_no_win:-4403 |
| `uint8_presence_for_argmax` | 1 | 137 | 0 | 3 | 14 | 0 | 0 | 0 | 219:actionable:11210, 377:actionable:8567, 046:actionable:6274, 366:byte_negative:2078, 054:byte_negative:-9686 |
| `signed_rect_priority_overlay` | 3 | 262 | 0 | 0 | 0 | 0 | 0 | 0 |  |
| `gridsample_warp_render` | 1 | 119 | 0 | 0 | 0 | 0 | 0 | 0 |  |
| `einsum_vs_free_input_reduction` | 3 | 201 | 0 | 0 | 0 | 0 | 0 | 0 |  |
| `qlinearconv_signed_renderer` | 1 | 39 | 44 | 0 | 0 | 0 | 0 | 0 |  |
| `dynamic_bundled_cse_rewire` | 4 | 187 | 0 | 0 | 24 | 0 | 0 | 0 | 366:byte_negative:2078, 133:byte_negative:-8452, 364:known_blocked:-8825, 158:byte_negative:-9481, 054:byte_negative:-9686 |
| `dedupe_byte_identical_initializers` | 0 | 11 | 0 | 0 | 155 | 0 | 169 | 0 | 233:byte_negative:3712, 366:byte_negative:2578, 018:known_blocked:183, 145:probe_no_win:-1314, 379:probe_no_win:-2818 |
| `pad_compensated_spatial_crop` | 0 | 42 | 130 | 2 | 43 | 0 | 0 | 0 | 054:actionable:25313, 076:candidate_graphevidence:13202, 255:actionable:12097, 233:byte_negative:3712, 366:byte_negative:2578 |
| `zero_compare_to_bool_cast` | 2 | 18 | 0 | 0 | 141 | 0 | 106 | 0 | 233:byte_negative:3712, 366:byte_negative:2578, 018:known_blocked:183, 379:probe_no_win:-2818, 255:probe_no_win:-4403 |
| `branch_einsum_copy_edit_epilogue` | 2 | 100 | 57 | 0 | 55 | 0 | 0 | 0 | 233:byte_negative:3212, 366:byte_negative:2078, 018:known_blocked:-316, 133:byte_negative:-8452, 364:known_blocked:-8825 |
| `residual_spatialop_to_free_einsum_collapse` | 4 | 189 | 59 | 0 | 15 | 0 | 0 | 0 | 233:byte_negative:3212, 366:byte_negative:2078, 018:known_blocked:-316, 133:byte_negative:-8452, 158:byte_negative:-9481 |
| `spatial_reducesum_to_einsum_profile_tail` | 5 | 182 | 35 | 11 | 79 | 0 | 0 | 0 | 256:actionable:6063, 297:actionable:5389, 020:actionable:5261, 301:actionable:5141, 215:actionable:4739 |

## Highest-Cost Task Coverage

| task | cost | pts | candidate insights | state counts |
|---:|---:|---:|---|---|
| 233 | 32667 | 14.606 |  | {'logged': 13, 'byte_negative': 17, 'source': 2, 'kaggle_falsified': 3} |
| 366 | 31559 | 14.640 |  | {'logged': 13, 'byte_negative': 20, 'source': 1, 'none': 1} |
| 018 | 24358 | 14.899 |  | {'none': 4, 'known_blocked': 21, 'logged': 7, 'source': 1, 'kaggle_falsified': 2} |
| 133 | 21322 | 15.033 | `threshold_linearize_pairwise_onehot_and` | {'logged': 13, 'actionable': 1, 'graph_evidence': 2, 'byte_negative': 16, 'none': 2, 'source': 1} |
| 158 | 20329 | 15.080 |  | {'logged': 13, 'byte_negative': 18, 'graph_evidence': 3, 'none': 1} |
| 054 | 20131 | 15.090 | `pad_compensated_spatial_crop` | {'logged': 8, 'byte_negative': 20, 'none': 5, 'probe_no_win': 1, 'actionable': 1} |
| 285 | 19623 | 15.116 |  | {'logged': 14, 'none': 2, 'byte_negative': 17, 'kaggle_falsified': 2} |
| 286 | 18271 | 15.187 | `threshold_linearize_pairwise_onehot_and` | {'logged': 10, 'none': 9, 'candidate_graphevidence': 1, 'byte_negative': 14, 'source': 1} |
| 364 | 16157 | 15.310 |  | {'none': 4, 'known_blocked': 12, 'graph_evidence': 3, 'logged': 15, 'source': 1} |
| 367 | 15890 | 15.327 |  | {'logged': 15, 'byte_negative': 12, 'graph_evidence': 2, 'none': 6} |
| 349 | 14892 | 15.391 |  | {'known_blocked': 19, 'graph_evidence': 3, 'logged': 7, 'none': 5, 'source': 1} |
| 101 | 13725 | 15.473 |  | {'logged': 14, 'none': 2, 'byte_negative': 18, 'graph_evidence': 1} |
| 076 | 12856 | 15.538 | `pad_compensated_spatial_crop` | {'logged': 9, 'none': 9, 'known_blocked': 13, 'graph_evidence': 1, 'kaggle_falsified': 2, 'candidate_graphevidence': 1} |
| 118 | 12282 | 15.584 |  | {'none': 9, 'known_blocked': 14, 'graph_evidence': 4, 'logged': 7, 'kaggle_falsified': 1} |
| 138 | 11716 | 15.631 | `walk_einsum_iteration_collapse` | {'byte_negative': 18, 'none': 11, 'graph_evidence': 1, 'logged': 4, 'actionable': 1} |
| 198 | 11410 | 15.658 | `strided_conv_fixed_block_counts`, `walk_einsum_iteration_collapse` | {'logged': 7, 'byte_negative': 17, 'none': 9, 'actionable': 2} |
| 173 | 11320 | 15.666 | `scan_dtype_and_shift_compression`, `walk_einsum_iteration_collapse` | {'none': 6, 'known_blocked': 15, 'candidate_graphevidence': 1, 'logged': 10, 'kaggle_falsified': 2, 'actionable': 1} |
| 191 | 11044 | 15.690 |  | {'none': 3, 'byte_negative': 17, 'graph_evidence': 2, 'logged': 12, 'source': 1} |
| 145 | 10492 | 15.742 |  | {'logged': 18, 'byte_negative': 9, 'graph_evidence': 2, 'none': 5, 'probe_no_win': 1} |
| 204 | 10232 | 15.767 |  | {'none': 10, 'logged': 18, 'graph_evidence': 2, 'known_blocked': 5} |
| 066 | 10121 | 15.778 |  | {'byte_negative': 16, 'logged': 9, 'none': 6, 'source': 1, 'graph_evidence': 3} |
| 025 | 9817 | 15.808 |  | {'logged': 16, 'byte_negative': 12, 'graph_evidence': 3, 'none': 4} |
| 080 | 9713 | 15.819 |  | {'byte_negative': 14, 'source': 1, 'graph_evidence': 2, 'none': 5, 'logged': 13} |
| 064 | 9504 | 15.841 |  | {'logged': 11, 'byte_negative': 11, 'none': 9, 'graph_evidence': 4} |
| 379 | 9095 | 15.885 |  | {'logged': 11, 'none': 13, 'known_blocked': 8, 'graph_evidence': 1, 'probe_no_win': 2} |
| 074 | 9050 | 15.889 |  | {'logged': 12, 'byte_negative': 11, 'none': 12} |
| 350 | 9036 | 15.891 | `walk_einsum_iteration_collapse` | {'logged': 20, 'none': 10, 'byte_negative': 3, 'actionable': 1, 'graph_evidence': 1} |
| 324 | 8862 | 15.910 | `scan_dtype_and_shift_compression` | {'logged': 9, 'known_blocked': 13, 'graph_evidence': 4, 'candidate_graphevidence': 1, 'none': 7, 'source': 1} |
| 338 | 8848 | 15.912 |  | {'logged': 12, 'none': 7, 'byte_negative': 13, 'source': 2, 'probe_no_win': 1} |
| 216 | 8660 | 15.934 |  | {'logged': 13, 'none': 6, 'byte_negative': 12, 'graph_evidence': 3, 'source': 1} |
| 370 | 8099 | 16.001 |  | {'logged': 8, 'byte_negative': 17, 'graph_evidence': 2, 'none': 7, 'source': 1} |
| 209 | 7775 | 16.041 |  | {'logged': 15, 'none': 6, 'byte_negative': 11, 'source': 2, 'graph_evidence': 1} |
| 096 | 7682 | 16.053 |  | {'logged': 6, 'byte_negative': 15, 'none': 12, 'graph_evidence': 2} |
| 187 | 7622 | 16.061 |  | {'none': 7, 'byte_negative': 2, 'logged': 22, 'graph_evidence': 1, 'source': 3} |
| 255 | 7597 | 16.064 | `category_augmented_separable_lut`, `high_score_frontier_final_output_only`, `threshold_linearize_pairwise_onehot_and`, `free_final_onehot_equal`, `sparse_conv_single_op_floor`, `band_profile_contraction_final_equal`, `zero_concat_tail_to_pad`, `direct_onehot_gather_output` | {'candidate_logged': 5, 'actionable': 6, 'source': 3, 'graph_evidence': 4, 'none': 11, 'logged': 3, 'probe_no_win': 3} |
| 219 | 7210 | 16.117 | `uint8_presence_for_argmax` | {'none': 13, 'known_blocked': 11, 'logged': 9, 'actionable': 1, 'graph_evidence': 1} |
| 157 | 6851 | 16.168 | `strided_conv_fixed_block_counts` | {'none': 14, 'unlowered_semantic_compiler': 2, 'known_blocked': 10, 'logged': 5, 'kaggle_falsified': 2, 'actionable': 1, 'graph_evidence': 1} |
| 192 | 6833 | 16.170 | `high_score_frontier_final_output_only` | {'byte_negative': 12, 'actionable': 1, 'none': 15, 'logged': 6, 'graph_evidence': 1} |
| 089 | 6715 | 16.188 | `threshold_linearize_pairwise_onehot_and` | {'logged': 10, 'actionable': 1, 'none': 13, 'known_blocked': 9, 'graph_evidence': 1, 'probe_no_win': 1} |
| 009 | 6698 | 16.190 |  | {'logged': 17, 'none': 8, 'byte_negative': 9, 'probe_no_win': 1} |
