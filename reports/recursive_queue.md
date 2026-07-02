# Recursive improvement queue

Generated from `reports/global_layer_inventory.json` and `reports/insight_registry.yaml`.

Use this as a global queue.  When a deep task attempt creates a new mechanism, add it to the registry and regenerate this file.

## category_augmented_separable_lut

Fold row/column-separable override planes into an augmented LUT

- source tasks: 055
- candidates: 11
- expected: {'gain_type': 'memory_by_deleting_full_canvas_overrides', 'risk': 'low-medium when override predicates are row/column separable and categories are mutually exclusive', 'verification': 'stored eval plus fresh/adopt; inspect separator/off-grid category priority in LUT'}

| rank | task | score | pts | mem | params | source | tags |
|---:|---:|---:|---:|---:|---:|---|---|
| 1 | 349 | 18659.4 | 15.102 | 19800 | 90 | semantic_or_handbuilt | conv_heavy,documented_wall,high_memory,local_stencil,lut_selection,maxpool_scan,open_angle,qlinear |
| 2 | 193 | 15543.7 | 15.361 | 15284 | 68 | semantic_or_handbuilt | high_memory,local_stencil,lut_selection,onehot_final_equal,qlinear |
| 3 | 222 | 8839.2 | 15.916 | 8736 | 78 | unknown | conv_heavy,custom_win,local_stencil,maxpool_scan,onehot_final_equal,open_angle,qlinear |
| 4 | 080 | 8734.1 | 15.726 | 10198 | 454 | exact_preserve | conv_heavy,documented_wall,exact_preserve,high_memory,lut_selection,open_angle,qlinear |
| 5 | 328 | 6275.0 | 16.179 | 4835 | 1940 | exact_preserve | exact_preserve,lut_selection |
| 6 | 279 | 5555.0 | 16.378 | 5508 | 47 | semantic_or_handbuilt | conv_heavy,local_stencil,lut_selection,qlinear |
| 7 | 008 | 5126.0 | 16.365 | 5561 | 65 | exact_preserve | exact_preserve,lut_selection |
| 8 | 208 | 4226.0 | 16.539 | 4612 | 114 | exact_preserve | exact_preserve,onehot_final_equal,open_angle,qlinear |
| 9 | 117 | 3665.0 | 16.666 | 3922 | 243 | exact_preserve | exact_preserve,local_stencil,lut_selection,open_angle,qlinear |
| 10 | 091 | 2528.0 | 16.984 | 2853 | 175 | exact_preserve | exact_preserve,local_stencil,lut_selection,open_angle |
| 11 | 029 | -29644.6 | 15.715 | 10736 | 34 | exact_preserve | exact_preserve,high_memory,marginal_wall,onehot_final_equal,open_angle |

## sparse_edit_stream_without_mask_planes

Replace monotonic full-canvas edit-mask cascades with sparse label updates when duplicate inactive writes are impossible

- source tasks: 054
- candidates: 0
- expected: {'gain_type': 'memory_if_monotonic', 'risk': 'high unless duplicate scatter overwrite behaviour is proven irrelevant', 'verification': 'compare against incumbent on stored and fresh; inspect duplicate indices before adoption'}

| rank | task | score | pts | mem | params | source | tags |
|---:|---:|---:|---:|---:|---:|---|---|

## high_score_frontier_final_output_only

Prioritize tasks that can reach the 20+ frontier by avoiding every full-canvas intermediate

- source tasks: 087 140 223 307 016 276 309 337 056 092
- candidates: 32
- expected: {'gain_type': 'frontier_research', 'risk': 'medium-high because it often requires semantic recompilation rather than graph surgery', 'verification': 'prove no counted full-canvas intermediates in mem profile, then fresh eval before adoption'}

| rank | task | score | pts | mem | params | source | tags |
|---:|---:|---:|---:|---:|---:|---|---|
| 1 | 191 | 32005.9 | 14.624 | 31252 | 841 | exact_preserve | assignment_wall,conv_heavy,custom_win,exact_preserve,high_memory,low_score,matmul,open_angle |
| 2 | 054 | 25974.2 | 14.829 | 25885 | 238 | exact_preserve | exact_preserve,gather_heavy,high_memory,low_score,scan,scatter |
| 3 | 173 | 21335.3 | 15.022 | 21430 | 112 | exact_preserve | exact_preserve,gather_heavy,high_memory,scatter |
| 4 | 349 | 18659.4 | 15.102 | 19800 | 90 | semantic_or_handbuilt | conv_heavy,documented_wall,high_memory,local_stencil,lut_selection,maxpool_scan,open_angle,qlinear |
| 5 | 066 | 16782.3 | 15.256 | 16899 | 160 | exact_preserve | exact_preserve,high_memory,marker_routed_path,open_angle,scan |
| 6 | 193 | 15543.7 | 15.361 | 15284 | 68 | semantic_or_handbuilt | high_memory,local_stencil,lut_selection,onehot_final_equal,qlinear |
| 7 | 025 | 13734.5 | 15.448 | 13874 | 195 | exact_preserve | exact_preserve,high_memory,onehot_final_equal,open_angle,scan |
| 8 | 138 | 12124.2 | 15.456 | 13789 | 172 | exact_preserve | conv_heavy,documented_wall,exact_preserve,high_memory,local_stencil,open_angle |
| 9 | 396 | 11810.2 | 15.633 | 11567 | 133 | semantic_or_handbuilt | conv_heavy,high_memory,local_stencil |
| 10 | 023 | 11509.1 | 15.616 | 11603 | 291 | exact_preserve | conv_heavy,exact_preserve,high_memory,open_angle,qlinear |
| 11 | 017 | 10545.9 | 15.744 | 8448 | 2021 | semantic_or_handbuilt | custom_win,onehot_final_equal,open_angle |
| 12 | 370 | 9222.0 | 15.823 | 8645 | 1024 | exact_preserve | conv_heavy,exact_preserve,qlinear,scan,scatter |
| 13 | 222 | 8839.2 | 15.916 | 8736 | 78 | unknown | conv_heavy,custom_win,local_stencil,maxpool_scan,onehot_final_equal,open_angle,qlinear |
| 14 | 080 | 8734.1 | 15.726 | 10198 | 454 | exact_preserve | conv_heavy,documented_wall,exact_preserve,high_memory,lut_selection,open_angle,qlinear |
| 15 | 074 | 8583.2 | 15.889 | 9000 | 50 | exact_preserve | exact_preserve,local_stencil,open_angle |
| 16 | 377 | 7819.6 | 15.975 | 8162 | 150 | exact_preserve | exact_preserve,local_stencil |
| 17 | 096 | 6941.6 | 15.905 | 8108 | 805 | exact_preserve | documented_wall,exact_preserve,open_angle,scatter |
| 18 | 324 | 6921.9 | 15.907 | 7308 | 1586 | exact_preserve | conv_heavy,documented_wall,exact_preserve,local_stencil,qlinear,scan |
| 19 | 192 | 6639.6 | 15.938 | 8515 | 106 | exact_preserve | documented_wall,exact_preserve,local_stencil,open_angle |
| 20 | 328 | 6275.0 | 16.179 | 4835 | 1940 | exact_preserve | exact_preserve,lut_selection |
| 21 | 279 | 5555.0 | 16.378 | 5508 | 47 | semantic_or_handbuilt | conv_heavy,local_stencil,lut_selection,qlinear |
| 22 | 264 | 5554.0 | 16.292 | 5348 | 706 | exact_preserve | exact_preserve,local_stencil,lut_selection,onehot_final_equal |
| 23 | 278 | 5483.0 | 16.391 | 5436 | 47 | semantic_or_handbuilt | local_stencil,qlinear |
| 24 | 382 | 5283.0 | 16.337 | 5648 | 135 | exact_preserve | exact_preserve,open_angle,scan |
| 25 | 008 | 5126.0 | 16.365 | 5561 | 65 | exact_preserve | exact_preserve,lut_selection |

## threshold_linearize_pairwise_onehot_and

Replace pairwise one-hot AND/Kronecker products with direct thresholded product scores

- source tasks: 001
- candidates: 66
- expected: {'gain_type': 'memory_by_deleting_boolean_product_carrier', 'risk': 'medium; relies on one-hot inputs, positive-threshold scoring, and class-specific margins', 'verification': 'stored eval plus fresh/adopt when generator samples exist; inspect off-footprint scores are non-positive'}

| rank | task | score | pts | mem | params | source | tags |
|---:|---:|---:|---:|---:|---:|---|---|
| 1 | 366 | 36367.8 | 14.497 | 35927 | 490 | exact_preserve | connectivity_wall,exact_preserve,gather_heavy,high_memory,low_score,lut_selection,open_angle,scan |
| 2 | 002 | 22776.3 | 14.896 | 24320 | 125 | exact_preserve | bitwise_program,connectivity_wall,documented_wall,exact_preserve,gather_heavy,high_memory,low_score,lut_selection |
| 3 | 367 | 19717.7 | 15.051 | 16200 | 4733 | semantic_or_handbuilt | connectivity_wall,conv_heavy,documented_wall,high_memory,local_stencil,lut_selection,open_angle,qlinear |
| 4 | 349 | 18659.4 | 15.102 | 19800 | 90 | semantic_or_handbuilt | conv_heavy,documented_wall,high_memory,local_stencil,lut_selection,maxpool_scan,open_angle,qlinear |
| 5 | 219 | 16367.5 | 15.195 | 18033 | 93 | exact_preserve | assignment_wall,connectivity_wall,documented_wall,exact_preserve,gather_heavy,high_memory,lut_selection,open_angle |
| 6 | 193 | 15543.7 | 15.361 | 15284 | 68 | semantic_or_handbuilt | high_memory,local_stencil,lut_selection,onehot_final_equal,qlinear |
| 7 | 025 | 13734.5 | 15.448 | 13874 | 195 | exact_preserve | exact_preserve,high_memory,onehot_final_equal,open_angle,scan |
| 8 | 064 | 11587.7 | 15.494 | 13368 | 68 | exact_preserve | connectivity_wall,documented_wall,exact_preserve,high_memory,onehot_final_equal,open_angle |
| 9 | 145 | 11361.7 | 15.511 | 13104 | 111 | exact_preserve | connectivity_wall,custom_win,documented_wall,exact_preserve,high_memory,lut_selection,maxpool_scan,open_angle |
| 10 | 017 | 10545.9 | 15.744 | 8448 | 2021 | semantic_or_handbuilt | custom_win,onehot_final_equal,open_angle |
| 11 | 222 | 8839.2 | 15.916 | 8736 | 78 | unknown | conv_heavy,custom_win,local_stencil,maxpool_scan,onehot_final_equal,open_angle,qlinear |
| 12 | 080 | 8734.1 | 15.726 | 10198 | 454 | exact_preserve | conv_heavy,documented_wall,exact_preserve,high_memory,lut_selection,open_angle,qlinear |
| 13 | 202 | 7797.0 | 16.039 | 7773 | 24 | semantic_or_handbuilt | connectivity_wall,local_stencil,onehot_final_equal,open_angle,qlinear |
| 14 | 157 | 6342.4 | 15.972 | 7983 | 351 | exact_preserve | assignment_wall,connectivity_wall,documented_wall,exact_preserve,gather_heavy,lut_selection,open_angle,scatter |
| 15 | 328 | 6275.0 | 16.179 | 4835 | 1940 | exact_preserve | exact_preserve,lut_selection |
| 16 | 368 | 5725.0 | 16.264 | 6022 | 203 | exact_preserve | connectivity_wall,exact_preserve,gather_heavy,onehot_final_equal,scatter |
| 17 | 279 | 5555.0 | 16.378 | 5508 | 47 | semantic_or_handbuilt | conv_heavy,local_stencil,lut_selection,qlinear |
| 18 | 264 | 5554.0 | 16.292 | 5348 | 706 | exact_preserve | exact_preserve,local_stencil,lut_selection,onehot_final_equal |
| 19 | 234 | 5179.0 | 16.448 | 5113 | 66 | semantic_or_handbuilt | connectivity_wall,gather_heavy,onehot_final_equal |
| 20 | 008 | 5126.0 | 16.365 | 5561 | 65 | exact_preserve | exact_preserve,lut_selection |
| 21 | 009 | 5024.0 | 16.143 | 6929 | 95 | exact_preserve | connectivity_wall,documented_wall,exact_preserve,local_stencil,lut_selection,onehot_final_equal,open_angle |
| 22 | 005 | 4882.0 | 16.163 | 6392 | 490 | exact_preserve | conv_heavy,documented_wall,exact_preserve,lut_selection,qlinear |
| 23 | 182 | 4793.0 | 16.176 | 6695 | 98 | exact_preserve | assignment_wall,connectivity_wall,conv_heavy,documented_wall,exact_preserve,local_stencil,lut_selection,open_angle |
| 24 | 383 | 4570.0 | 16.289 | 5974 | 96 | unknown | assignment_wall,connectivity_wall,documented_wall,local_stencil,onehot_final_equal,open_angle |
| 25 | 037 | 4246.0 | 16.535 | 4614 | 132 | exact_preserve | connectivity_wall,exact_preserve,lut_selection,scatter |

## qlinear_uint8_lut_or_matmul

Replace fp16/fp32 one-hot LUT/MatMul selection with uint8 QLinearMatMul/QLinearConv where exact

- source tasks: 055 080 338
- candidates: 1
- expected: {'gain_type': 'memory', 'risk': 'medium', 'verification': 'stored eval, fresh eval if generator available, then adopt gate'}

| rank | task | score | pts | mem | params | source | tags |
|---:|---:|---:|---:|---:|---:|---|---|
| 1 | 002 | 22776.3 | 14.896 | 24320 | 125 | exact_preserve | bitwise_program,connectivity_wall,documented_wall,exact_preserve,gather_heavy,high_memory,low_score,lut_selection |

## free_final_onehot_equal

Delay 10-channel expansion to final Equal/Where output

- source tasks: 017 095 146 381
- candidates: 25
- expected: {'gain_type': 'memory', 'risk': 'low-medium', 'verification': 'output equivalence before adoption'}

| rank | task | score | pts | mem | params | source | tags |
|---:|---:|---:|---:|---:|---:|---|---|
| 1 | 193 | 15543.7 | 15.361 | 15284 | 68 | semantic_or_handbuilt | high_memory,local_stencil,lut_selection,onehot_final_equal,qlinear |
| 2 | 025 | 13734.5 | 15.448 | 13874 | 195 | exact_preserve | exact_preserve,high_memory,onehot_final_equal,open_angle,scan |
| 3 | 064 | 11587.7 | 15.494 | 13368 | 68 | exact_preserve | connectivity_wall,documented_wall,exact_preserve,high_memory,onehot_final_equal,open_angle |
| 4 | 017 | 10545.9 | 15.744 | 8448 | 2021 | semantic_or_handbuilt | custom_win,onehot_final_equal,open_angle |
| 5 | 222 | 8839.2 | 15.916 | 8736 | 78 | unknown | conv_heavy,custom_win,local_stencil,maxpool_scan,onehot_final_equal,open_angle,qlinear |
| 6 | 202 | 7797.0 | 16.039 | 7773 | 24 | semantic_or_handbuilt | connectivity_wall,local_stencil,onehot_final_equal,open_angle,qlinear |
| 7 | 368 | 5725.0 | 16.264 | 6022 | 203 | exact_preserve | connectivity_wall,exact_preserve,gather_heavy,onehot_final_equal,scatter |
| 8 | 264 | 5554.0 | 16.292 | 5348 | 706 | exact_preserve | exact_preserve,local_stencil,lut_selection,onehot_final_equal |
| 9 | 234 | 5179.0 | 16.448 | 5113 | 66 | semantic_or_handbuilt | connectivity_wall,gather_heavy,onehot_final_equal |
| 10 | 009 | 5024.0 | 16.143 | 6929 | 95 | exact_preserve | connectivity_wall,documented_wall,exact_preserve,local_stencil,lut_selection,onehot_final_equal,open_angle |
| 11 | 383 | 4570.0 | 16.289 | 5974 | 96 | unknown | assignment_wall,connectivity_wall,documented_wall,local_stencil,onehot_final_equal,open_angle |
| 12 | 208 | 4226.0 | 16.539 | 4612 | 114 | exact_preserve | exact_preserve,onehot_final_equal,open_angle,qlinear |
| 13 | 363 | 3635.0 | 16.673 | 3839 | 296 | exact_preserve | connectivity_wall,conv_heavy,exact_preserve,local_stencil,onehot_final_equal,open_angle |
| 14 | 107 | 3578.0 | 16.687 | 2924 | 1154 | exact_preserve | exact_preserve,lut_selection,matmul,onehot_final_equal,open_angle,qlinear,scatter |
| 15 | 177 | 3322.0 | 16.751 | 3692 | 130 | exact_preserve | connectivity_wall,conv_heavy,exact_preserve,local_stencil,onehot_final_equal,open_angle |
| 16 | 036 | 1677.0 | 17.314 | 2140 | 37 | exact_preserve | connectivity_wall,exact_preserve,onehot_final_equal,open_angle |
| 17 | 392 | 1451.0 | 17.424 | 1602 | 349 | exact_preserve | custom_win,exact_preserve,onehot_final_equal,open_angle |
| 18 | 213 | 1387.0 | 17.457 | 1838 | 49 | exact_preserve | exact_preserve,onehot_final_equal,open_angle |
| 19 | 333 | 1227.0 | 16.921 | 2792 | 435 | exact_preserve | connectivity_wall,documented_wall,exact_preserve,onehot_final_equal,open_angle |
| 20 | 303 | 1120.0 | 17.610 | 1598 | 22 | exact_preserve | exact_preserve,onehot_final_equal,open_angle |
| 21 | 308 | 988.0 | 17.695 | 1385 | 103 | exact_preserve | connectivity_wall,exact_preserve,onehot_final_equal,open_angle,scatter |
| 22 | 071 | 698.0 | 17.100 | 2643 | 55 | exact_preserve | connectivity_wall,documented_wall,exact_preserve,onehot_final_equal,open_angle |
| 23 | 170 | 545.0 | 17.158 | 2216 | 329 | exact_preserve | assignment_wall,conv_heavy,documented_wall,exact_preserve,local_stencil,lut_selection,onehot_final_equal,open_angle |
| 24 | 184 | 520.0 | 17.168 | 2460 | 60 | exact_preserve | connectivity_wall,conv_heavy,documented_wall,exact_preserve,local_stencil,onehot_final_equal,open_angle,scan |
| 25 | 029 | -29644.6 | 15.715 | 10736 | 34 | exact_preserve | exact_preserve,high_memory,marginal_wall,onehot_final_equal,open_angle |

## scan_dtype_and_shift_compression

Compress scan-style MaxPool/CumSum/Hillis-Steele pipelines with lower dtype or shared shifts

- source tasks: 046 216
- candidates: 36
- expected: {'gain_type': 'memory', 'risk': 'medium-high', 'verification': 'stored/fresh eval and compare against incumbent'}

| rank | task | score | pts | mem | params | source | tags |
|---:|---:|---:|---:|---:|---:|---|---|
| 1 | 233 | 58311.2 | 14.003 | 59147 | 565 | exact_preserve | assignment_wall,connectivity_wall,conv_heavy,documented_wall,exact_preserve,gather_heavy,high_memory,low_score |
| 2 | 366 | 36367.8 | 14.497 | 35927 | 490 | exact_preserve | connectivity_wall,exact_preserve,gather_heavy,high_memory,low_score,lut_selection,open_angle,scan |
| 3 | 187 | 33940.9 | 14.580 | 32850 | 665 | semantic_or_handbuilt | connectivity_wall,high_memory,local_stencil,low_score,maxpool_scan |
| 4 | 133 | 32008.5 | 14.578 | 32294 | 1288 | exact_preserve | assignment_wall,connectivity_wall,conv_heavy,documented_wall,exact_preserve,gather_heavy,heuristic,high_memory |
| 5 | 054 | 25974.2 | 14.829 | 25885 | 238 | exact_preserve | exact_preserve,gather_heavy,high_memory,low_score,scan,scatter |
| 6 | 285 | 23975.5 | 14.848 | 25080 | 550 | exact_preserve | connectivity_wall,documented_wall,exact_preserve,gather_heavy,high_memory,low_score,open_angle,scatter |
| 7 | 364 | 22826.1 | 14.956 | 22900 | 113 | exact_preserve | connectivity_wall,exact_preserve,high_memory,local_stencil,low_score,maxpool_scan,open_angle,qlinear |
| 8 | 173 | 21335.3 | 15.022 | 21430 | 112 | exact_preserve | exact_preserve,gather_heavy,high_memory,scatter |
| 9 | 349 | 18659.4 | 15.102 | 19800 | 90 | semantic_or_handbuilt | conv_heavy,documented_wall,high_memory,local_stencil,lut_selection,maxpool_scan,open_angle,qlinear |
| 10 | 319 | 17823.4 | 15.119 | 19280 | 279 | exact_preserve | assignment_wall,documented_wall,exact_preserve,gather_heavy,high_memory,scan |
| 11 | 101 | 17490.3 | 15.216 | 16850 | 905 | exact_preserve | connectivity_wall,exact_preserve,gather_heavy,high_memory,scatter |
| 12 | 066 | 16782.3 | 15.256 | 16899 | 160 | exact_preserve | exact_preserve,high_memory,marker_routed_path,open_angle,scan |
| 13 | 219 | 16367.5 | 15.195 | 18033 | 93 | exact_preserve | assignment_wall,connectivity_wall,documented_wall,exact_preserve,gather_heavy,high_memory,lut_selection,open_angle |
| 14 | 025 | 13734.5 | 15.448 | 13874 | 195 | exact_preserve | exact_preserve,high_memory,onehot_final_equal,open_angle,scan |
| 15 | 110 | 11948.5 | 15.468 | 13155 | 634 | exact_preserve | connectivity_wall,conv_heavy,documented_wall,exact_preserve,high_memory,maxpool_scan,open_angle |
| 16 | 198 | 11726.7 | 15.484 | 13434 | 138 | exact_preserve | connectivity_wall,documented_wall,exact_preserve,gather_heavy,high_memory,open_angle,scan |
| 17 | 145 | 11361.7 | 15.511 | 13104 | 111 | exact_preserve | connectivity_wall,custom_win,documented_wall,exact_preserve,high_memory,lut_selection,maxpool_scan,open_angle |
| 18 | 370 | 9222.0 | 15.823 | 8645 | 1024 | exact_preserve | conv_heavy,exact_preserve,qlinear,scan,scatter |
| 19 | 222 | 8839.2 | 15.916 | 8736 | 78 | unknown | conv_heavy,custom_win,local_stencil,maxpool_scan,onehot_final_equal,open_angle,qlinear |
| 20 | 204 | 8447.1 | 15.753 | 9920 | 453 | exact_preserve | connectivity_wall,conv_heavy,custom_win,documented_wall,exact_preserve,local_stencil,maxpool_scan,open_angle |
| 21 | 324 | 6921.9 | 15.907 | 7308 | 1586 | exact_preserve | conv_heavy,documented_wall,exact_preserve,local_stencil,qlinear,scan |
| 22 | 157 | 6342.4 | 15.972 | 7983 | 351 | exact_preserve | assignment_wall,connectivity_wall,documented_wall,exact_preserve,gather_heavy,lut_selection,open_angle,scatter |
| 23 | 382 | 5283.0 | 16.337 | 5648 | 135 | exact_preserve | exact_preserve,open_angle,scan |
| 24 | 234 | 5179.0 | 16.448 | 5113 | 66 | semantic_or_handbuilt | connectivity_wall,gather_heavy,onehot_final_equal |
| 25 | 277 | 3829.0 | 16.627 | 4101 | 228 | exact_preserve | connectivity_wall,exact_preserve,lut_selection,maxpool_scan |

## sparse_conv_single_op_floor

Collapse local neighborhood rules to one sparse Conv/QLinearConv when output is thresholded

- source tasks: 015 095 230 294
- candidates: 68
- expected: {'gain_type': 'memory+params', 'risk': 'low when rule is truly local', 'verification': 'fresh process eval; beware one-process mem0 false signals'}

| rank | task | score | pts | mem | params | source | tags |
|---:|---:|---:|---:|---:|---:|---|---|
| 1 | 349 | 18659.4 | 15.102 | 19800 | 90 | semantic_or_handbuilt | conv_heavy,documented_wall,high_memory,local_stencil,lut_selection,maxpool_scan,open_angle,qlinear |
| 2 | 193 | 15543.7 | 15.361 | 15284 | 68 | semantic_or_handbuilt | high_memory,local_stencil,lut_selection,onehot_final_equal,qlinear |
| 3 | 255 | 14639.9 | 15.387 | 14663 | 293 | exact_preserve | conv_heavy,exact_preserve,high_memory,matmul,qlinear |
| 4 | 138 | 12124.2 | 15.456 | 13789 | 172 | exact_preserve | conv_heavy,documented_wall,exact_preserve,high_memory,local_stencil,open_angle |
| 5 | 396 | 11810.2 | 15.633 | 11567 | 133 | semantic_or_handbuilt | conv_heavy,high_memory,local_stencil |
| 6 | 023 | 11509.1 | 15.616 | 11603 | 291 | exact_preserve | conv_heavy,exact_preserve,high_memory,open_angle,qlinear |
| 7 | 370 | 9222.0 | 15.823 | 8645 | 1024 | exact_preserve | conv_heavy,exact_preserve,qlinear,scan,scatter |
| 8 | 222 | 8839.2 | 15.916 | 8736 | 78 | unknown | conv_heavy,custom_win,local_stencil,maxpool_scan,onehot_final_equal,open_angle,qlinear |
| 9 | 080 | 8734.1 | 15.726 | 10198 | 454 | exact_preserve | conv_heavy,documented_wall,exact_preserve,high_memory,lut_selection,open_angle,qlinear |
| 10 | 074 | 8583.2 | 15.889 | 9000 | 50 | exact_preserve | exact_preserve,local_stencil,open_angle |
| 11 | 377 | 7819.6 | 15.975 | 8162 | 150 | exact_preserve | exact_preserve,local_stencil |
| 12 | 324 | 6921.9 | 15.907 | 7308 | 1586 | exact_preserve | conv_heavy,documented_wall,exact_preserve,local_stencil,qlinear,scan |
| 13 | 192 | 6639.6 | 15.938 | 8515 | 106 | exact_preserve | documented_wall,exact_preserve,local_stencil,open_angle |
| 14 | 279 | 5555.0 | 16.378 | 5508 | 47 | semantic_or_handbuilt | conv_heavy,local_stencil,lut_selection,qlinear |
| 15 | 264 | 5554.0 | 16.292 | 5348 | 706 | exact_preserve | exact_preserve,local_stencil,lut_selection,onehot_final_equal |
| 16 | 278 | 5483.0 | 16.391 | 5436 | 47 | semantic_or_handbuilt | local_stencil,qlinear |
| 17 | 005 | 4882.0 | 16.163 | 6392 | 490 | exact_preserve | conv_heavy,documented_wall,exact_preserve,lut_selection,qlinear |
| 18 | 085 | 4881.0 | 16.409 | 4500 | 881 | exact_preserve | exact_preserve,local_stencil,open_angle |
| 19 | 117 | 3665.0 | 16.666 | 3922 | 243 | exact_preserve | exact_preserve,local_stencil,lut_selection,open_angle,qlinear |
| 20 | 132 | 3589.0 | 16.684 | 3990 | 99 | exact_preserve | bitwise_program,exact_preserve,gather_heavy,local_stencil,open_angle |
| 21 | 162 | 3568.0 | 16.689 | 4024 | 44 | exact_preserve | exact_preserve,local_stencil,open_angle,qlinear |
| 22 | 284 | 3235.0 | 16.774 | 3082 | 653 | exact_preserve | conv_heavy,custom_win,exact_preserve,open_angle,scatter |
| 23 | 335 | 2979.0 | 17.001 | 2288 | 691 | semantic_or_handbuilt | conv_heavy,local_stencil,open_angle |
| 24 | 354 | 2751.0 | 17.080 | 2674 | 77 | semantic_or_handbuilt | conv_heavy,local_stencil,open_angle |
| 25 | 079 | 2643.0 | 16.947 | 3065 | 78 | exact_preserve | custom_win,exact_preserve,local_stencil,open_angle |

## exact_preserve_to_semantic_rewrite

Prioritize exact-preserve source builders for semantic replacement

- source tasks: 002 018 286 366
- candidates: 33
- expected: {'gain_type': 'research', 'risk': 'high', 'verification': 'document semantic mechanism before implementation'}

| rank | task | score | pts | mem | params | source | tags |
|---:|---:|---:|---:|---:|---:|---|---|
| 1 | 233 | 58311.2 | 14.003 | 59147 | 565 | exact_preserve | assignment_wall,connectivity_wall,conv_heavy,documented_wall,exact_preserve,gather_heavy,high_memory,low_score |
| 2 | 286 | 41061.6 | 14.341 | 41822 | 742 | exact_preserve | bitwise_program,connectivity_wall,documented_wall,exact_preserve,gather_heavy,high_memory,low_score,qlinear |
| 3 | 366 | 36367.8 | 14.497 | 35927 | 490 | exact_preserve | connectivity_wall,exact_preserve,gather_heavy,high_memory,low_score,lut_selection,open_angle,scan |
| 4 | 133 | 32008.5 | 14.578 | 32294 | 1288 | exact_preserve | assignment_wall,connectivity_wall,conv_heavy,documented_wall,exact_preserve,gather_heavy,heuristic,high_memory |
| 5 | 191 | 32005.9 | 14.624 | 31252 | 841 | exact_preserve | assignment_wall,conv_heavy,custom_win,exact_preserve,high_memory,low_score,matmul,open_angle |
| 6 | 054 | 25974.2 | 14.829 | 25885 | 238 | exact_preserve | exact_preserve,gather_heavy,high_memory,low_score,scan,scatter |
| 7 | 285 | 23975.5 | 14.848 | 25080 | 550 | exact_preserve | connectivity_wall,documented_wall,exact_preserve,gather_heavy,high_memory,low_score,open_angle,scatter |
| 8 | 364 | 22826.1 | 14.956 | 22900 | 113 | exact_preserve | connectivity_wall,exact_preserve,high_memory,local_stencil,low_score,maxpool_scan,open_angle,qlinear |
| 9 | 002 | 22776.3 | 14.896 | 24320 | 125 | exact_preserve | bitwise_program,connectivity_wall,documented_wall,exact_preserve,gather_heavy,high_memory,low_score,lut_selection |
| 10 | 076 | 21908.5 | 14.932 | 23296 | 292 | exact_preserve | assignment_wall,connectivity_wall,documented_wall,exact_preserve,high_memory,low_score,open_angle,scatter |
| 11 | 173 | 21335.3 | 15.022 | 21430 | 112 | exact_preserve | exact_preserve,gather_heavy,high_memory,scatter |
| 12 | 243 | 20960.4 | 14.972 | 22608 | 44 | exact_preserve | connectivity_wall,conv_heavy,custom_win,documented_wall,exact_preserve,high_memory,low_score,qlinear |
| 13 | 319 | 17823.4 | 15.119 | 19280 | 279 | exact_preserve | assignment_wall,documented_wall,exact_preserve,gather_heavy,high_memory,scan |
| 14 | 101 | 17490.3 | 15.216 | 16850 | 905 | exact_preserve | connectivity_wall,exact_preserve,gather_heavy,high_memory,scatter |
| 15 | 066 | 16782.3 | 15.256 | 16899 | 160 | exact_preserve | exact_preserve,high_memory,marker_routed_path,open_angle,scan |
| 16 | 219 | 16367.5 | 15.195 | 18033 | 93 | exact_preserve | assignment_wall,connectivity_wall,documented_wall,exact_preserve,gather_heavy,high_memory,lut_selection,open_angle |
| 17 | 205 | 15064.1 | 15.360 | 14734 | 638 | exact_preserve | connectivity_wall,conv_heavy,exact_preserve,high_memory,open_angle,qlinear |
| 18 | 255 | 14639.9 | 15.387 | 14663 | 293 | exact_preserve | conv_heavy,exact_preserve,high_memory,matmul,qlinear |
| 19 | 025 | 13734.5 | 15.448 | 13874 | 195 | exact_preserve | exact_preserve,high_memory,onehot_final_equal,open_angle,scan |
| 20 | 138 | 12124.2 | 15.456 | 13789 | 172 | exact_preserve | conv_heavy,documented_wall,exact_preserve,high_memory,local_stencil,open_angle |
| 21 | 110 | 11948.5 | 15.468 | 13155 | 634 | exact_preserve | connectivity_wall,conv_heavy,documented_wall,exact_preserve,high_memory,maxpool_scan,open_angle |
| 22 | 198 | 11726.7 | 15.484 | 13434 | 138 | exact_preserve | connectivity_wall,documented_wall,exact_preserve,gather_heavy,high_memory,open_angle,scan |
| 23 | 064 | 11587.7 | 15.494 | 13368 | 68 | exact_preserve | connectivity_wall,documented_wall,exact_preserve,high_memory,onehot_final_equal,open_angle |
| 24 | 023 | 11509.1 | 15.616 | 11603 | 291 | exact_preserve | conv_heavy,exact_preserve,high_memory,open_angle,qlinear |
| 25 | 145 | 11361.7 | 15.511 | 13104 | 111 | exact_preserve | connectivity_wall,custom_win,documented_wall,exact_preserve,high_memory,lut_selection,maxpool_scan,open_angle |

## dihedral_template_match_stacked_conv

Use one stacked Conv for all dihedral orientations of a small runtime-extracted template

- source tasks: 191
- candidates: 0
- expected: {'gain_type': 'memory+accuracy', 'risk': 'medium', 'verification': 'fresh exact eval; compare against incumbent and propagate if template extraction is deterministic'}

| rank | task | score | pts | mem | params | source | tags |
|---:|---:|---:|---:|---:|---:|---|---|

## bounded_crop_before_connectivity_scan

Run iterative connectivity/flood-fill on the generator's true max canvas, not the 30x30 harness canvas

- source tasks: 187
- candidates: 34
- expected: {'gain_type': 'memory+generalization', 'risk': 'medium', 'verification': 'confirm generator max height/width, stored eval, fresh adopt gate'}

| rank | task | score | pts | mem | params | source | tags |
|---:|---:|---:|---:|---:|---:|---|---|
| 1 | 233 | 58311.2 | 14.003 | 59147 | 565 | exact_preserve | assignment_wall,connectivity_wall,conv_heavy,documented_wall,exact_preserve,gather_heavy,high_memory,low_score |
| 2 | 286 | 41061.6 | 14.341 | 41822 | 742 | exact_preserve | bitwise_program,connectivity_wall,documented_wall,exact_preserve,gather_heavy,high_memory,low_score,qlinear |
| 3 | 366 | 36367.8 | 14.497 | 35927 | 490 | exact_preserve | connectivity_wall,exact_preserve,gather_heavy,high_memory,low_score,lut_selection,open_angle,scan |
| 4 | 187 | 33940.9 | 14.580 | 32850 | 665 | semantic_or_handbuilt | connectivity_wall,high_memory,local_stencil,low_score,maxpool_scan |
| 5 | 133 | 32008.5 | 14.578 | 32294 | 1288 | exact_preserve | assignment_wall,connectivity_wall,conv_heavy,documented_wall,exact_preserve,gather_heavy,heuristic,high_memory |
| 6 | 191 | 32005.9 | 14.624 | 31252 | 841 | exact_preserve | assignment_wall,conv_heavy,custom_win,exact_preserve,high_memory,low_score,matmul,open_angle |
| 7 | 054 | 25974.2 | 14.829 | 25885 | 238 | exact_preserve | exact_preserve,gather_heavy,high_memory,low_score,scan,scatter |
| 8 | 285 | 23975.5 | 14.848 | 25080 | 550 | exact_preserve | connectivity_wall,documented_wall,exact_preserve,gather_heavy,high_memory,low_score,open_angle,scatter |
| 9 | 364 | 22826.1 | 14.956 | 22900 | 113 | exact_preserve | connectivity_wall,exact_preserve,high_memory,local_stencil,low_score,maxpool_scan,open_angle,qlinear |
| 10 | 076 | 21908.5 | 14.932 | 23296 | 292 | exact_preserve | assignment_wall,connectivity_wall,documented_wall,exact_preserve,high_memory,low_score,open_angle,scatter |
| 11 | 173 | 21335.3 | 15.022 | 21430 | 112 | exact_preserve | exact_preserve,gather_heavy,high_memory,scatter |
| 12 | 349 | 18659.4 | 15.102 | 19800 | 90 | semantic_or_handbuilt | conv_heavy,documented_wall,high_memory,local_stencil,lut_selection,maxpool_scan,open_angle,qlinear |
| 13 | 319 | 17823.4 | 15.119 | 19280 | 279 | exact_preserve | assignment_wall,documented_wall,exact_preserve,gather_heavy,high_memory,scan |
| 14 | 101 | 17490.3 | 15.216 | 16850 | 905 | exact_preserve | connectivity_wall,exact_preserve,gather_heavy,high_memory,scatter |
| 15 | 066 | 16782.3 | 15.256 | 16899 | 160 | exact_preserve | exact_preserve,high_memory,marker_routed_path,open_angle,scan |
| 16 | 219 | 16367.5 | 15.195 | 18033 | 93 | exact_preserve | assignment_wall,connectivity_wall,documented_wall,exact_preserve,gather_heavy,high_memory,lut_selection,open_angle |
| 17 | 205 | 15064.1 | 15.360 | 14734 | 638 | exact_preserve | connectivity_wall,conv_heavy,exact_preserve,high_memory,open_angle,qlinear |
| 18 | 255 | 14639.9 | 15.387 | 14663 | 293 | exact_preserve | conv_heavy,exact_preserve,high_memory,matmul,qlinear |
| 19 | 025 | 13734.5 | 15.448 | 13874 | 195 | exact_preserve | exact_preserve,high_memory,onehot_final_equal,open_angle,scan |
| 20 | 138 | 12124.2 | 15.456 | 13789 | 172 | exact_preserve | conv_heavy,documented_wall,exact_preserve,high_memory,local_stencil,open_angle |
| 21 | 110 | 11948.5 | 15.468 | 13155 | 634 | exact_preserve | connectivity_wall,conv_heavy,documented_wall,exact_preserve,high_memory,maxpool_scan,open_angle |
| 22 | 396 | 11810.2 | 15.633 | 11567 | 133 | semantic_or_handbuilt | conv_heavy,high_memory,local_stencil |
| 23 | 198 | 11726.7 | 15.484 | 13434 | 138 | exact_preserve | connectivity_wall,documented_wall,exact_preserve,gather_heavy,high_memory,open_angle,scan |
| 24 | 064 | 11587.7 | 15.494 | 13368 | 68 | exact_preserve | connectivity_wall,documented_wall,exact_preserve,high_memory,onehot_final_equal,open_angle |
| 25 | 023 | 11509.1 | 15.616 | 11603 | 291 | exact_preserve | conv_heavy,exact_preserve,high_memory,open_angle,qlinear |

## public_teacher_bitwise_scan_replacement

Replace repeated MaxPool/Max/Min scan stacks with bitwise shift/mask routing when the state is binary

- source tasks: 002 209
- candidates: 15
- expected: {'gain_type': 'memory', 'risk': 'medium-high', 'verification': 'stored eval, fresh eval when generator available, public-probe if strict fresh is known weak'}

| rank | task | score | pts | mem | params | source | tags |
|---:|---:|---:|---:|---:|---:|---|---|
| 1 | 233 | 58311.2 | 14.003 | 59147 | 565 | exact_preserve | assignment_wall,connectivity_wall,conv_heavy,documented_wall,exact_preserve,gather_heavy,high_memory,low_score |
| 2 | 187 | 33940.9 | 14.580 | 32850 | 665 | semantic_or_handbuilt | connectivity_wall,high_memory,local_stencil,low_score,maxpool_scan |
| 3 | 133 | 32008.5 | 14.578 | 32294 | 1288 | exact_preserve | assignment_wall,connectivity_wall,conv_heavy,documented_wall,exact_preserve,gather_heavy,heuristic,high_memory |
| 4 | 191 | 32005.9 | 14.624 | 31252 | 841 | exact_preserve | assignment_wall,conv_heavy,custom_win,exact_preserve,high_memory,low_score,matmul,open_angle |
| 5 | 285 | 23975.5 | 14.848 | 25080 | 550 | exact_preserve | connectivity_wall,documented_wall,exact_preserve,gather_heavy,high_memory,low_score,open_angle,scatter |
| 6 | 364 | 22826.1 | 14.956 | 22900 | 113 | exact_preserve | connectivity_wall,exact_preserve,high_memory,local_stencil,low_score,maxpool_scan,open_angle,qlinear |
| 7 | 076 | 21908.5 | 14.932 | 23296 | 292 | exact_preserve | assignment_wall,connectivity_wall,documented_wall,exact_preserve,high_memory,low_score,open_angle,scatter |
| 8 | 173 | 21335.3 | 15.022 | 21430 | 112 | exact_preserve | exact_preserve,gather_heavy,high_memory,scatter |
| 9 | 349 | 18659.4 | 15.102 | 19800 | 90 | semantic_or_handbuilt | conv_heavy,documented_wall,high_memory,local_stencil,lut_selection,maxpool_scan,open_angle,qlinear |
| 10 | 066 | 16782.3 | 15.256 | 16899 | 160 | exact_preserve | exact_preserve,high_memory,marker_routed_path,open_angle,scan |
| 11 | 219 | 16367.5 | 15.195 | 18033 | 93 | exact_preserve | assignment_wall,connectivity_wall,documented_wall,exact_preserve,gather_heavy,high_memory,lut_selection,open_angle |
| 12 | 158 | -6239.3 | 14.528 | 32979 | 2340 | exact_preserve | assignment_wall,conv_heavy,documented_wall,exact_preserve,gather_heavy,high_memory,low_score,marginal_wall |
| 13 | 018 | -50264.1 | 14.157 | 48196 | 2987 | exact_preserve | ambiguity_wall,documented_wall,exact_preserve,gather_heavy,high_memory,low_score,maxpool_scan,scatter |
| 14 | 118 | -67331.7 | 14.559 | 30849 | 3387 | exact_preserve | connectivity_wall,conv_heavy,documented_wall,exact_preserve,heuristic,high_memory,infeasible_exact_wall,information_loss_wall |
| 15 | 209 | -69374.0 | 14.620 | 32027 | 185 | exact_preserve | ambiguity_wall,assignment_wall,connectivity_wall,conv_heavy,documented_wall,exact_preserve,high_memory,low_score |

## public_teacher_qlinear_conv_rewrite

Convert repeated binary/count Conv/ConvTranspose towers to QLinearConv/uint8 routes

- source tasks: 023 349 182 233 255 364 338 004
- candidates: 38
- expected: {'gain_type': 'memory+params', 'risk': 'medium', 'verification': 'stored eval plus fresh; inspect for output-range saturation before adoption'}

| rank | task | score | pts | mem | params | source | tags |
|---:|---:|---:|---:|---:|---:|---|---|
| 1 | 187 | 33940.9 | 14.580 | 32850 | 665 | semantic_or_handbuilt | connectivity_wall,high_memory,local_stencil,low_score,maxpool_scan |
| 2 | 349 | 18659.4 | 15.102 | 19800 | 90 | semantic_or_handbuilt | conv_heavy,documented_wall,high_memory,local_stencil,lut_selection,maxpool_scan,open_angle,qlinear |
| 3 | 193 | 15543.7 | 15.361 | 15284 | 68 | semantic_or_handbuilt | high_memory,local_stencil,lut_selection,onehot_final_equal,qlinear |
| 4 | 205 | 15064.1 | 15.360 | 14734 | 638 | exact_preserve | connectivity_wall,conv_heavy,exact_preserve,high_memory,open_angle,qlinear |
| 5 | 255 | 14639.9 | 15.387 | 14663 | 293 | exact_preserve | conv_heavy,exact_preserve,high_memory,matmul,qlinear |
| 6 | 138 | 12124.2 | 15.456 | 13789 | 172 | exact_preserve | conv_heavy,documented_wall,exact_preserve,high_memory,local_stencil,open_angle |
| 7 | 110 | 11948.5 | 15.468 | 13155 | 634 | exact_preserve | connectivity_wall,conv_heavy,documented_wall,exact_preserve,high_memory,maxpool_scan,open_angle |
| 8 | 396 | 11810.2 | 15.633 | 11567 | 133 | semantic_or_handbuilt | conv_heavy,high_memory,local_stencil |
| 9 | 379 | 9276.6 | 15.818 | 9069 | 653 | exact_preserve | connectivity_wall,conv_heavy,custom_win,exact_preserve,open_angle |
| 10 | 222 | 8839.2 | 15.916 | 8736 | 78 | unknown | conv_heavy,custom_win,local_stencil,maxpool_scan,onehot_final_equal,open_angle,qlinear |
| 11 | 080 | 8734.1 | 15.726 | 10198 | 454 | exact_preserve | conv_heavy,documented_wall,exact_preserve,high_memory,lut_selection,open_angle,qlinear |
| 12 | 074 | 8583.2 | 15.889 | 9000 | 50 | exact_preserve | exact_preserve,local_stencil,open_angle |
| 13 | 377 | 7819.6 | 15.975 | 8162 | 150 | exact_preserve | exact_preserve,local_stencil |
| 14 | 202 | 7797.0 | 16.039 | 7773 | 24 | semantic_or_handbuilt | connectivity_wall,local_stencil,onehot_final_equal,open_angle,qlinear |
| 15 | 324 | 6921.9 | 15.907 | 7308 | 1586 | exact_preserve | conv_heavy,documented_wall,exact_preserve,local_stencil,qlinear,scan |
| 16 | 192 | 6639.6 | 15.938 | 8515 | 106 | exact_preserve | documented_wall,exact_preserve,local_stencil,open_angle |
| 17 | 264 | 5554.0 | 16.292 | 5348 | 706 | exact_preserve | exact_preserve,local_stencil,lut_selection,onehot_final_equal |
| 18 | 004 | 5144.0 | 16.454 | 5044 | 100 | semantic_or_handbuilt | connectivity_wall,conv_heavy,local_stencil,open_angle,qlinear |
| 19 | 009 | 5024.0 | 16.143 | 6929 | 95 | exact_preserve | connectivity_wall,documented_wall,exact_preserve,local_stencil,lut_selection,onehot_final_equal,open_angle |
| 20 | 005 | 4882.0 | 16.163 | 6392 | 490 | exact_preserve | conv_heavy,documented_wall,exact_preserve,lut_selection,qlinear |
| 21 | 085 | 4881.0 | 16.409 | 4500 | 881 | exact_preserve | exact_preserve,local_stencil,open_angle |
| 22 | 165 | 4150.0 | 16.276 | 5836 | 314 | exact_preserve | connectivity_wall,documented_wall,exact_preserve,local_stencil,open_angle,qlinear |
| 23 | 363 | 3635.0 | 16.673 | 3839 | 296 | exact_preserve | connectivity_wall,conv_heavy,exact_preserve,local_stencil,onehot_final_equal,open_angle |
| 24 | 132 | 3589.0 | 16.684 | 3990 | 99 | exact_preserve | bitwise_program,exact_preserve,gather_heavy,local_stencil,open_angle |
| 25 | 177 | 3322.0 | 16.751 | 3692 | 130 | exact_preserve | connectivity_wall,conv_heavy,exact_preserve,local_stencil,onehot_final_equal,open_angle |

## marker_routed_hidden_path_compiler

Compile endpoint-pair hidden paths from deliberate marker pixels instead of replaying scan-heavy exact graphs

- source tasks: 066
- candidates: 1
- expected: {'gain_type': 'semantic_rewrite_memory', 'risk': 'medium-high until tie-break is exact', 'verification': 'stored validation plus large fresh generator eval before ONNX adoption'}

| rank | task | score | pts | mem | params | source | tags |
|---:|---:|---:|---:|---:|---:|---|---|
| 1 | 066 | 16782.3 | 15.256 | 16899 | 160 | exact_preserve | exact_preserve,high_memory,marker_routed_path,open_angle,scan |

## rotation_component_template_scatter

Replace scan-heavy in-context sprite reconstruction with rotation-only 3x3 component template scatter

- source tasks: 233
- candidates: 15
- expected: {'gain_type': 'semantic_rewrite_memory', 'risk': 'medium-high until bbox/component edge cases are fully fresh-verified', 'verification': 'Python reference stored plus large fresh, then source-owned ONNX compiler and adopt gate'}

| rank | task | score | pts | mem | params | source | tags |
|---:|---:|---:|---:|---:|---:|---|---|
| 1 | 286 | 41061.6 | 14.341 | 41822 | 742 | exact_preserve | bitwise_program,connectivity_wall,documented_wall,exact_preserve,gather_heavy,high_memory,low_score,qlinear |
| 2 | 366 | 36367.8 | 14.497 | 35927 | 490 | exact_preserve | connectivity_wall,exact_preserve,gather_heavy,high_memory,low_score,lut_selection,open_angle,scan |
| 3 | 187 | 33940.9 | 14.580 | 32850 | 665 | semantic_or_handbuilt | connectivity_wall,high_memory,local_stencil,low_score,maxpool_scan |
| 4 | 054 | 25974.2 | 14.829 | 25885 | 238 | exact_preserve | exact_preserve,gather_heavy,high_memory,low_score,scan,scatter |
| 5 | 285 | 23975.5 | 14.848 | 25080 | 550 | exact_preserve | connectivity_wall,documented_wall,exact_preserve,gather_heavy,high_memory,low_score,open_angle,scatter |
| 6 | 364 | 22826.1 | 14.956 | 22900 | 113 | exact_preserve | connectivity_wall,exact_preserve,high_memory,local_stencil,low_score,maxpool_scan,open_angle,qlinear |
| 7 | 002 | 22776.3 | 14.896 | 24320 | 125 | exact_preserve | bitwise_program,connectivity_wall,documented_wall,exact_preserve,gather_heavy,high_memory,low_score,lut_selection |
| 8 | 173 | 21335.3 | 15.022 | 21430 | 112 | exact_preserve | exact_preserve,gather_heavy,high_memory,scatter |
| 9 | 243 | 20960.4 | 14.972 | 22608 | 44 | exact_preserve | connectivity_wall,conv_heavy,custom_win,documented_wall,exact_preserve,high_memory,low_score,qlinear |
| 10 | 367 | 19717.7 | 15.051 | 16200 | 4733 | semantic_or_handbuilt | connectivity_wall,conv_heavy,documented_wall,high_memory,local_stencil,lut_selection,open_angle,qlinear |
| 11 | 349 | 18659.4 | 15.102 | 19800 | 90 | semantic_or_handbuilt | conv_heavy,documented_wall,high_memory,local_stencil,lut_selection,maxpool_scan,open_angle,qlinear |
| 12 | 101 | 17490.3 | 15.216 | 16850 | 905 | exact_preserve | connectivity_wall,exact_preserve,gather_heavy,high_memory,scatter |
| 13 | 066 | 16782.3 | 15.256 | 16899 | 160 | exact_preserve | exact_preserve,high_memory,marker_routed_path,open_angle,scan |
| 14 | 193 | 15543.7 | 15.361 | 15284 | 68 | semantic_or_handbuilt | high_memory,local_stencil,lut_selection,onehot_final_equal,qlinear |
| 15 | 018 | -50264.1 | 14.157 | 48196 | 2987 | exact_preserve | ambiguity_wall,documented_wall,exact_preserve,gather_heavy,high_memory,low_score,maxpool_scan,scatter |

## band_profile_contraction_final_equal

Use distinct-colour row/column profiles to avoid full connectivity for band-fill tasks

- source tasks: 202
- candidates: 25
- expected: {'gain_type': 'semantic_rewrite_memory_or_source_ownership', 'risk': 'medium; requires generator proof that colours are distinct per band/profile', 'verification': 'stored eval plus large fresh generator eval; source/live reconcile after rewrite'}

| rank | task | score | pts | mem | params | source | tags |
|---:|---:|---:|---:|---:|---:|---|---|
| 1 | 054 | 25974.2 | 14.829 | 25885 | 238 | exact_preserve | exact_preserve,gather_heavy,high_memory,low_score,scan,scatter |
| 2 | 173 | 21335.3 | 15.022 | 21430 | 112 | exact_preserve | exact_preserve,gather_heavy,high_memory,scatter |
| 3 | 349 | 18659.4 | 15.102 | 19800 | 90 | semantic_or_handbuilt | conv_heavy,documented_wall,high_memory,local_stencil,lut_selection,maxpool_scan,open_angle,qlinear |
| 4 | 066 | 16782.3 | 15.256 | 16899 | 160 | exact_preserve | exact_preserve,high_memory,marker_routed_path,open_angle,scan |
| 5 | 193 | 15543.7 | 15.361 | 15284 | 68 | semantic_or_handbuilt | high_memory,local_stencil,lut_selection,onehot_final_equal,qlinear |
| 6 | 255 | 14639.9 | 15.387 | 14663 | 293 | exact_preserve | conv_heavy,exact_preserve,high_memory,matmul,qlinear |
| 7 | 025 | 13734.5 | 15.448 | 13874 | 195 | exact_preserve | exact_preserve,high_memory,onehot_final_equal,open_angle,scan |
| 8 | 138 | 12124.2 | 15.456 | 13789 | 172 | exact_preserve | conv_heavy,documented_wall,exact_preserve,high_memory,local_stencil,open_angle |
| 9 | 396 | 11810.2 | 15.633 | 11567 | 133 | semantic_or_handbuilt | conv_heavy,high_memory,local_stencil |
| 10 | 023 | 11509.1 | 15.616 | 11603 | 291 | exact_preserve | conv_heavy,exact_preserve,high_memory,open_angle,qlinear |
| 11 | 017 | 10545.9 | 15.744 | 8448 | 2021 | semantic_or_handbuilt | custom_win,onehot_final_equal,open_angle |
| 12 | 222 | 8839.2 | 15.916 | 8736 | 78 | unknown | conv_heavy,custom_win,local_stencil,maxpool_scan,onehot_final_equal,open_angle,qlinear |
| 13 | 080 | 8734.1 | 15.726 | 10198 | 454 | exact_preserve | conv_heavy,documented_wall,exact_preserve,high_memory,lut_selection,open_angle,qlinear |
| 14 | 074 | 8583.2 | 15.889 | 9000 | 50 | exact_preserve | exact_preserve,local_stencil,open_angle |
| 15 | 377 | 7819.6 | 15.975 | 8162 | 150 | exact_preserve | exact_preserve,local_stencil |
| 16 | 324 | 6921.9 | 15.907 | 7308 | 1586 | exact_preserve | conv_heavy,documented_wall,exact_preserve,local_stencil,qlinear,scan |
| 17 | 192 | 6639.6 | 15.938 | 8515 | 106 | exact_preserve | documented_wall,exact_preserve,local_stencil,open_angle |
| 18 | 279 | 5555.0 | 16.378 | 5508 | 47 | semantic_or_handbuilt | conv_heavy,local_stencil,lut_selection,qlinear |
| 19 | 264 | 5554.0 | 16.292 | 5348 | 706 | exact_preserve | exact_preserve,local_stencil,lut_selection,onehot_final_equal |
| 20 | 278 | 5483.0 | 16.391 | 5436 | 47 | semantic_or_handbuilt | local_stencil,qlinear |
| 21 | 085 | 4881.0 | 16.409 | 4500 | 881 | exact_preserve | exact_preserve,local_stencil,open_angle |
| 22 | 208 | 4226.0 | 16.539 | 4612 | 114 | exact_preserve | exact_preserve,onehot_final_equal,open_angle,qlinear |
| 23 | 162 | 3568.0 | 16.689 | 4024 | 44 | exact_preserve | exact_preserve,local_stencil,open_angle,qlinear |
| 24 | 029 | -29644.6 | 15.715 | 10736 | 34 | exact_preserve | exact_preserve,high_memory,marginal_wall,onehot_final_equal,open_angle |
| 25 | 018 | -50264.1 | 14.157 | 48196 | 2987 | exact_preserve | ambiguity_wall,documented_wall,exact_preserve,gather_heavy,high_memory,low_score,maxpool_scan,scatter |

## bounded_multicopy_slot_topk_floor

Do not shrink multi-copy reconstruction candidate lists below generator extrema

- source tasks: 101
- candidates: 0
- expected: {'gain_type': 'negative_filter', 'risk': 'high if candidate-list bounds are inferred from stored examples only', 'verification': 'measure generator extrema before reducing TopK/list widths; compare direct channel slices against colour-index entry plane when palette is fixed'}

| rank | task | score | pts | mem | params | source | tags |
|---:|---:|---:|---:|---:|---:|---|---|

## sparse_initializer_not_dense_operand

Sparse initializers cannot be used as params loopholes for dense ONNX operands

- source tasks: 123 232
- candidates: 0
- expected: {'gain_type': 'negative_filter', 'risk': 'certain rejection by shape inference/ORT for dense operator operands', 'verification': 'do not pursue sparse_initializer substitutions for standard dense inputs; use algebraic operator rewrites instead'}

| rank | task | score | pts | mem | params | source | tags |
|---:|---:|---:|---:|---:|---:|---|---|

## generator_extrema_before_topk_shrink

Prove generator extrema before shrinking sparse TopK candidate lists

- source tasks: 173 285
- candidates: 0
- expected: {'gain_type': 'negative_filter', 'risk': 'high if list width is reduced from empirical samples only', 'verification': 'derive the generator maximum and, when possible, build an explicit adversarial fresh example before reducing TopK widths'}

| rank | task | score | pts | mem | params | source | tags |
|---:|---:|---:|---:|---:|---:|---|---|

## variable_size_mask_requires_onehot_input

Variable-size output masks may require the original one-hot input, not compact labels

- source tasks: 173
- candidates: 0
- expected: {'gain_type': 'negative_filter', 'risk': 'medium-high for crop rewrites that remove one-hot validity masks', 'verification': 'check whether outside-grid all-zero padding must differ from true black channel-0 cells before replacing input-based size masks'}

| rank | task | score | pts | mem | params | source | tags |
|---:|---:|---:|---:|---:|---:|---|---|

## flood_fill_iteration_count_lb_tradeoff

Flood-fill iteration count is a correctness and score tradeoff, not sample slack

- source tasks: 187
- candidates: 0
- expected: {'gain_type': 'negative_filter', 'risk': 'rare fresh/LB failures when iteration count is below generator path-distance maximum', 'verification': 'measure exterior/reachable distance on adversarial or large fresh samples before reducing scan depth; adoption requires stored and fresh tradeoff decision'}

| rank | task | score | pts | mem | params | source | tags |
|---:|---:|---:|---:|---:|---:|---|---|

## zero_concat_tail_to_pad

Replace zero-mask Concat tails with Pad when extending an active crop to the harness canvas

- source tasks: 066
- candidates: 189
- expected: {'gain_type': 'memory+params', 'risk': 'low when concatenated tails are constant zero/false and Pad fill defaults to zero', 'verification': 'stored eval plus side-by-side fresh output equivalence against incumbent'}

| rank | task | score | pts | mem | params | source | tags |
|---:|---:|---:|---:|---:|---:|---|---|
| 1 | 233 | 58311.2 | 14.003 | 59147 | 565 | exact_preserve | assignment_wall,connectivity_wall,conv_heavy,documented_wall,exact_preserve,gather_heavy,high_memory,low_score |
| 2 | 286 | 41061.6 | 14.341 | 41822 | 742 | exact_preserve | bitwise_program,connectivity_wall,documented_wall,exact_preserve,gather_heavy,high_memory,low_score,qlinear |
| 3 | 366 | 36367.8 | 14.497 | 35927 | 490 | exact_preserve | connectivity_wall,exact_preserve,gather_heavy,high_memory,low_score,lut_selection,open_angle,scan |
| 4 | 187 | 33940.9 | 14.580 | 32850 | 665 | semantic_or_handbuilt | connectivity_wall,high_memory,local_stencil,low_score,maxpool_scan |
| 5 | 133 | 32008.5 | 14.578 | 32294 | 1288 | exact_preserve | assignment_wall,connectivity_wall,conv_heavy,documented_wall,exact_preserve,gather_heavy,heuristic,high_memory |
| 6 | 191 | 32005.9 | 14.624 | 31252 | 841 | exact_preserve | assignment_wall,conv_heavy,custom_win,exact_preserve,high_memory,low_score,matmul,open_angle |
| 7 | 054 | 25974.2 | 14.829 | 25885 | 238 | exact_preserve | exact_preserve,gather_heavy,high_memory,low_score,scan,scatter |
| 8 | 285 | 23975.5 | 14.848 | 25080 | 550 | exact_preserve | connectivity_wall,documented_wall,exact_preserve,gather_heavy,high_memory,low_score,open_angle,scatter |
| 9 | 364 | 22826.1 | 14.956 | 22900 | 113 | exact_preserve | connectivity_wall,exact_preserve,high_memory,local_stencil,low_score,maxpool_scan,open_angle,qlinear |
| 10 | 002 | 22776.3 | 14.896 | 24320 | 125 | exact_preserve | bitwise_program,connectivity_wall,documented_wall,exact_preserve,gather_heavy,high_memory,low_score,lut_selection |
| 11 | 076 | 21908.5 | 14.932 | 23296 | 292 | exact_preserve | assignment_wall,connectivity_wall,documented_wall,exact_preserve,high_memory,low_score,open_angle,scatter |
| 12 | 173 | 21335.3 | 15.022 | 21430 | 112 | exact_preserve | exact_preserve,gather_heavy,high_memory,scatter |
| 13 | 243 | 20960.4 | 14.972 | 22608 | 44 | exact_preserve | connectivity_wall,conv_heavy,custom_win,documented_wall,exact_preserve,high_memory,low_score,qlinear |
| 14 | 367 | 19717.7 | 15.051 | 16200 | 4733 | semantic_or_handbuilt | connectivity_wall,conv_heavy,documented_wall,high_memory,local_stencil,lut_selection,open_angle,qlinear |
| 15 | 349 | 18659.4 | 15.102 | 19800 | 90 | semantic_or_handbuilt | conv_heavy,documented_wall,high_memory,local_stencil,lut_selection,maxpool_scan,open_angle,qlinear |
| 16 | 319 | 17823.4 | 15.119 | 19280 | 279 | exact_preserve | assignment_wall,documented_wall,exact_preserve,gather_heavy,high_memory,scan |
| 17 | 101 | 17490.3 | 15.216 | 16850 | 905 | exact_preserve | connectivity_wall,exact_preserve,gather_heavy,high_memory,scatter |
| 18 | 066 | 16782.3 | 15.256 | 16899 | 160 | exact_preserve | exact_preserve,high_memory,marker_routed_path,open_angle,scan |
| 19 | 219 | 16367.5 | 15.195 | 18033 | 93 | exact_preserve | assignment_wall,connectivity_wall,documented_wall,exact_preserve,gather_heavy,high_memory,lut_selection,open_angle |
| 20 | 193 | 15543.7 | 15.361 | 15284 | 68 | semantic_or_handbuilt | high_memory,local_stencil,lut_selection,onehot_final_equal,qlinear |
| 21 | 205 | 15064.1 | 15.360 | 14734 | 638 | exact_preserve | connectivity_wall,conv_heavy,exact_preserve,high_memory,open_angle,qlinear |
| 22 | 255 | 14639.9 | 15.387 | 14663 | 293 | exact_preserve | conv_heavy,exact_preserve,high_memory,matmul,qlinear |
| 23 | 025 | 13734.5 | 15.448 | 13874 | 195 | exact_preserve | exact_preserve,high_memory,onehot_final_equal,open_angle,scan |
| 24 | 138 | 12124.2 | 15.456 | 13789 | 172 | exact_preserve | conv_heavy,documented_wall,exact_preserve,high_memory,local_stencil,open_angle |
| 25 | 110 | 11948.5 | 15.468 | 13155 | 634 | exact_preserve | connectivity_wall,conv_heavy,documented_wall,exact_preserve,high_memory,maxpool_scan,open_angle |

## bounded_exact_cover_without_mode_background

Use bounded exact-cover over sparse generator objects instead of mode-colour background assumptions

- source tasks: 366
- candidates: 0
- expected: {'gain_type': 'semantic_rewrite_research', 'risk': 'high until the exact-cover search is lowered to a compact ONNX graph', 'verification': 'derive generator object-count and sparse-colour bounds, validate a Python reference on large fresh samples, then compare any ONNX lowering against stored and fresh'}

| rank | task | score | pts | mem | params | source | tags |
|---:|---:|---:|---:|---:|---:|---|---|

## direct_onehot_gather_output

Route original one-hot input directly to graph output for crop/periodic remaps

- source tasks: 343
- candidates: 211
- expected: {'gain_type': 'memory+params', 'risk': 'low when every output on-cell is copied from the original one-hot input and off-grid can point at an all-zero padded input coordinate', 'verification': 'stored eval plus fresh side-by-side; confirm black in-grid cells and off-grid all-zero semantics are both preserved'}

| rank | task | score | pts | mem | params | source | tags |
|---:|---:|---:|---:|---:|---:|---|---|
| 1 | 366 | 36367.8 | 14.497 | 35927 | 490 | exact_preserve | connectivity_wall,exact_preserve,gather_heavy,high_memory,low_score,lut_selection,open_angle,scan |
| 2 | 133 | 32008.5 | 14.578 | 32294 | 1288 | exact_preserve | assignment_wall,connectivity_wall,conv_heavy,documented_wall,exact_preserve,gather_heavy,heuristic,high_memory |
| 3 | 191 | 32005.9 | 14.624 | 31252 | 841 | exact_preserve | assignment_wall,conv_heavy,custom_win,exact_preserve,high_memory,low_score,matmul,open_angle |
| 4 | 285 | 23975.5 | 14.848 | 25080 | 550 | exact_preserve | connectivity_wall,documented_wall,exact_preserve,gather_heavy,high_memory,low_score,open_angle,scatter |
| 5 | 364 | 22826.1 | 14.956 | 22900 | 113 | exact_preserve | connectivity_wall,exact_preserve,high_memory,local_stencil,low_score,maxpool_scan,open_angle,qlinear |
| 6 | 002 | 22776.3 | 14.896 | 24320 | 125 | exact_preserve | bitwise_program,connectivity_wall,documented_wall,exact_preserve,gather_heavy,high_memory,low_score,lut_selection |
| 7 | 076 | 21908.5 | 14.932 | 23296 | 292 | exact_preserve | assignment_wall,connectivity_wall,documented_wall,exact_preserve,high_memory,low_score,open_angle,scatter |
| 8 | 367 | 19717.7 | 15.051 | 16200 | 4733 | semantic_or_handbuilt | connectivity_wall,conv_heavy,documented_wall,high_memory,local_stencil,lut_selection,open_angle,qlinear |
| 9 | 349 | 18659.4 | 15.102 | 19800 | 90 | semantic_or_handbuilt | conv_heavy,documented_wall,high_memory,local_stencil,lut_selection,maxpool_scan,open_angle,qlinear |
| 10 | 066 | 16782.3 | 15.256 | 16899 | 160 | exact_preserve | exact_preserve,high_memory,marker_routed_path,open_angle,scan |
| 11 | 219 | 16367.5 | 15.195 | 18033 | 93 | exact_preserve | assignment_wall,connectivity_wall,documented_wall,exact_preserve,gather_heavy,high_memory,lut_selection,open_angle |
| 12 | 193 | 15543.7 | 15.361 | 15284 | 68 | semantic_or_handbuilt | high_memory,local_stencil,lut_selection,onehot_final_equal,qlinear |
| 13 | 205 | 15064.1 | 15.360 | 14734 | 638 | exact_preserve | connectivity_wall,conv_heavy,exact_preserve,high_memory,open_angle,qlinear |
| 14 | 025 | 13734.5 | 15.448 | 13874 | 195 | exact_preserve | exact_preserve,high_memory,onehot_final_equal,open_angle,scan |
| 15 | 138 | 12124.2 | 15.456 | 13789 | 172 | exact_preserve | conv_heavy,documented_wall,exact_preserve,high_memory,local_stencil,open_angle |
| 16 | 110 | 11948.5 | 15.468 | 13155 | 634 | exact_preserve | connectivity_wall,conv_heavy,documented_wall,exact_preserve,high_memory,maxpool_scan,open_angle |
| 17 | 198 | 11726.7 | 15.484 | 13434 | 138 | exact_preserve | connectivity_wall,documented_wall,exact_preserve,gather_heavy,high_memory,open_angle,scan |
| 18 | 064 | 11587.7 | 15.494 | 13368 | 68 | exact_preserve | connectivity_wall,documented_wall,exact_preserve,high_memory,onehot_final_equal,open_angle |
| 19 | 023 | 11509.1 | 15.616 | 11603 | 291 | exact_preserve | conv_heavy,exact_preserve,high_memory,open_angle,qlinear |
| 20 | 145 | 11361.7 | 15.511 | 13104 | 111 | exact_preserve | connectivity_wall,custom_win,documented_wall,exact_preserve,high_memory,lut_selection,maxpool_scan,open_angle |
| 21 | 017 | 10545.9 | 15.744 | 8448 | 2021 | semantic_or_handbuilt | custom_win,onehot_final_equal,open_angle |
| 22 | 379 | 9276.6 | 15.818 | 9069 | 653 | exact_preserve | connectivity_wall,conv_heavy,custom_win,exact_preserve,open_angle |
| 23 | 338 | 8892.4 | 15.712 | 10140 | 666 | exact_preserve | connectivity_wall,documented_wall,exact_preserve,high_memory,local_stencil,open_angle,qlinear |
| 24 | 222 | 8839.2 | 15.916 | 8736 | 78 | unknown | conv_heavy,custom_win,local_stencil,maxpool_scan,onehot_final_equal,open_angle,qlinear |
| 25 | 080 | 8734.1 | 15.726 | 10198 | 454 | exact_preserve | conv_heavy,documented_wall,exact_preserve,high_memory,lut_selection,open_angle,qlinear |

## uint8_topk_compact_label_grid

Run TopK directly on compact uint8 label grids when only nonzero-cell enumeration is needed

- source tasks: 285 366
- candidates: 96
- expected: {'gain_type': 'memory', 'risk': 'medium because TopK tie ordering and dtype support must be checked in ORT', 'verification': 'stored eval plus large side-by-side incumbent equivalence; truth-fresh failure shared with incumbent is not by itself a reject'}

| rank | task | score | pts | mem | params | source | tags |
|---:|---:|---:|---:|---:|---:|---|---|
| 1 | 233 | 58311.2 | 14.003 | 59147 | 565 | exact_preserve | assignment_wall,connectivity_wall,conv_heavy,documented_wall,exact_preserve,gather_heavy,high_memory,low_score |
| 2 | 286 | 41061.6 | 14.341 | 41822 | 742 | exact_preserve | bitwise_program,connectivity_wall,documented_wall,exact_preserve,gather_heavy,high_memory,low_score,qlinear |
| 3 | 366 | 36367.8 | 14.497 | 35927 | 490 | exact_preserve | connectivity_wall,exact_preserve,gather_heavy,high_memory,low_score,lut_selection,open_angle,scan |
| 4 | 187 | 33940.9 | 14.580 | 32850 | 665 | semantic_or_handbuilt | connectivity_wall,high_memory,local_stencil,low_score,maxpool_scan |
| 5 | 133 | 32008.5 | 14.578 | 32294 | 1288 | exact_preserve | assignment_wall,connectivity_wall,conv_heavy,documented_wall,exact_preserve,gather_heavy,heuristic,high_memory |
| 6 | 191 | 32005.9 | 14.624 | 31252 | 841 | exact_preserve | assignment_wall,conv_heavy,custom_win,exact_preserve,high_memory,low_score,matmul,open_angle |
| 7 | 054 | 25974.2 | 14.829 | 25885 | 238 | exact_preserve | exact_preserve,gather_heavy,high_memory,low_score,scan,scatter |
| 8 | 285 | 23975.5 | 14.848 | 25080 | 550 | exact_preserve | connectivity_wall,documented_wall,exact_preserve,gather_heavy,high_memory,low_score,open_angle,scatter |
| 9 | 364 | 22826.1 | 14.956 | 22900 | 113 | exact_preserve | connectivity_wall,exact_preserve,high_memory,local_stencil,low_score,maxpool_scan,open_angle,qlinear |
| 10 | 002 | 22776.3 | 14.896 | 24320 | 125 | exact_preserve | bitwise_program,connectivity_wall,documented_wall,exact_preserve,gather_heavy,high_memory,low_score,lut_selection |
| 11 | 076 | 21908.5 | 14.932 | 23296 | 292 | exact_preserve | assignment_wall,connectivity_wall,documented_wall,exact_preserve,high_memory,low_score,open_angle,scatter |
| 12 | 173 | 21335.3 | 15.022 | 21430 | 112 | exact_preserve | exact_preserve,gather_heavy,high_memory,scatter |
| 13 | 243 | 20960.4 | 14.972 | 22608 | 44 | exact_preserve | connectivity_wall,conv_heavy,custom_win,documented_wall,exact_preserve,high_memory,low_score,qlinear |
| 14 | 367 | 19717.7 | 15.051 | 16200 | 4733 | semantic_or_handbuilt | connectivity_wall,conv_heavy,documented_wall,high_memory,local_stencil,lut_selection,open_angle,qlinear |
| 15 | 349 | 18659.4 | 15.102 | 19800 | 90 | semantic_or_handbuilt | conv_heavy,documented_wall,high_memory,local_stencil,lut_selection,maxpool_scan,open_angle,qlinear |
| 16 | 319 | 17823.4 | 15.119 | 19280 | 279 | exact_preserve | assignment_wall,documented_wall,exact_preserve,gather_heavy,high_memory,scan |
| 17 | 101 | 17490.3 | 15.216 | 16850 | 905 | exact_preserve | connectivity_wall,exact_preserve,gather_heavy,high_memory,scatter |
| 18 | 066 | 16782.3 | 15.256 | 16899 | 160 | exact_preserve | exact_preserve,high_memory,marker_routed_path,open_angle,scan |
| 19 | 219 | 16367.5 | 15.195 | 18033 | 93 | exact_preserve | assignment_wall,connectivity_wall,documented_wall,exact_preserve,gather_heavy,high_memory,lut_selection,open_angle |
| 20 | 193 | 15543.7 | 15.361 | 15284 | 68 | semantic_or_handbuilt | high_memory,local_stencil,lut_selection,onehot_final_equal,qlinear |
| 21 | 205 | 15064.1 | 15.360 | 14734 | 638 | exact_preserve | connectivity_wall,conv_heavy,exact_preserve,high_memory,open_angle,qlinear |
| 22 | 255 | 14639.9 | 15.387 | 14663 | 293 | exact_preserve | conv_heavy,exact_preserve,high_memory,matmul,qlinear |
| 23 | 025 | 13734.5 | 15.448 | 13874 | 195 | exact_preserve | exact_preserve,high_memory,onehot_final_equal,open_angle,scan |
| 24 | 138 | 12124.2 | 15.456 | 13789 | 172 | exact_preserve | conv_heavy,documented_wall,exact_preserve,high_memory,local_stencil,open_angle |
| 25 | 110 | 11948.5 | 15.468 | 13155 | 634 | exact_preserve | connectivity_wall,conv_heavy,documented_wall,exact_preserve,high_memory,maxpool_scan,open_angle |

## strided_conv_fixed_block_counts

Replace many fixed-block Slice+ReduceSum branches with one strided Conv

- source tasks: 011
- candidates: 205
- expected: {'gain_type': 'memory_when_repeated_block_reads_share_one_channel_or_kernel', 'risk': 'low-medium; Conv weight params can outweigh memory savings when few blocks are read or the input crop is already tiny', 'verification': 'stored eval plus fresh side-by-side; inspect Conv output shape because full 30x30 input may produce trailing off-grid stride positions that must be sliced away'}

| rank | task | score | pts | mem | params | source | tags |
|---:|---:|---:|---:|---:|---:|---|---|
| 1 | 233 | 58311.2 | 14.003 | 59147 | 565 | exact_preserve | assignment_wall,connectivity_wall,conv_heavy,documented_wall,exact_preserve,gather_heavy,high_memory,low_score |
| 2 | 286 | 41061.6 | 14.341 | 41822 | 742 | exact_preserve | bitwise_program,connectivity_wall,documented_wall,exact_preserve,gather_heavy,high_memory,low_score,qlinear |
| 3 | 366 | 36367.8 | 14.497 | 35927 | 490 | exact_preserve | connectivity_wall,exact_preserve,gather_heavy,high_memory,low_score,lut_selection,open_angle,scan |
| 4 | 187 | 33940.9 | 14.580 | 32850 | 665 | semantic_or_handbuilt | connectivity_wall,high_memory,local_stencil,low_score,maxpool_scan |
| 5 | 133 | 32008.5 | 14.578 | 32294 | 1288 | exact_preserve | assignment_wall,connectivity_wall,conv_heavy,documented_wall,exact_preserve,gather_heavy,heuristic,high_memory |
| 6 | 191 | 32005.9 | 14.624 | 31252 | 841 | exact_preserve | assignment_wall,conv_heavy,custom_win,exact_preserve,high_memory,low_score,matmul,open_angle |
| 7 | 054 | 25974.2 | 14.829 | 25885 | 238 | exact_preserve | exact_preserve,gather_heavy,high_memory,low_score,scan,scatter |
| 8 | 285 | 23975.5 | 14.848 | 25080 | 550 | exact_preserve | connectivity_wall,documented_wall,exact_preserve,gather_heavy,high_memory,low_score,open_angle,scatter |
| 9 | 364 | 22826.1 | 14.956 | 22900 | 113 | exact_preserve | connectivity_wall,exact_preserve,high_memory,local_stencil,low_score,maxpool_scan,open_angle,qlinear |
| 10 | 002 | 22776.3 | 14.896 | 24320 | 125 | exact_preserve | bitwise_program,connectivity_wall,documented_wall,exact_preserve,gather_heavy,high_memory,low_score,lut_selection |
| 11 | 076 | 21908.5 | 14.932 | 23296 | 292 | exact_preserve | assignment_wall,connectivity_wall,documented_wall,exact_preserve,high_memory,low_score,open_angle,scatter |
| 12 | 173 | 21335.3 | 15.022 | 21430 | 112 | exact_preserve | exact_preserve,gather_heavy,high_memory,scatter |
| 13 | 243 | 20960.4 | 14.972 | 22608 | 44 | exact_preserve | connectivity_wall,conv_heavy,custom_win,documented_wall,exact_preserve,high_memory,low_score,qlinear |
| 14 | 367 | 19717.7 | 15.051 | 16200 | 4733 | semantic_or_handbuilt | connectivity_wall,conv_heavy,documented_wall,high_memory,local_stencil,lut_selection,open_angle,qlinear |
| 15 | 319 | 17823.4 | 15.119 | 19280 | 279 | exact_preserve | assignment_wall,documented_wall,exact_preserve,gather_heavy,high_memory,scan |
| 16 | 101 | 17490.3 | 15.216 | 16850 | 905 | exact_preserve | connectivity_wall,exact_preserve,gather_heavy,high_memory,scatter |
| 17 | 066 | 16782.3 | 15.256 | 16899 | 160 | exact_preserve | exact_preserve,high_memory,marker_routed_path,open_angle,scan |
| 18 | 219 | 16367.5 | 15.195 | 18033 | 93 | exact_preserve | assignment_wall,connectivity_wall,documented_wall,exact_preserve,gather_heavy,high_memory,lut_selection,open_angle |
| 19 | 193 | 15543.7 | 15.361 | 15284 | 68 | semantic_or_handbuilt | high_memory,local_stencil,lut_selection,onehot_final_equal,qlinear |
| 20 | 205 | 15064.1 | 15.360 | 14734 | 638 | exact_preserve | connectivity_wall,conv_heavy,exact_preserve,high_memory,open_angle,qlinear |
| 21 | 255 | 14639.9 | 15.387 | 14663 | 293 | exact_preserve | conv_heavy,exact_preserve,high_memory,matmul,qlinear |
| 22 | 025 | 13734.5 | 15.448 | 13874 | 195 | exact_preserve | exact_preserve,high_memory,onehot_final_equal,open_angle,scan |
| 23 | 110 | 11948.5 | 15.468 | 13155 | 634 | exact_preserve | connectivity_wall,conv_heavy,documented_wall,exact_preserve,high_memory,maxpool_scan,open_angle |
| 24 | 396 | 11810.2 | 15.633 | 11567 | 133 | semantic_or_handbuilt | conv_heavy,high_memory,local_stencil |
| 25 | 198 | 11726.7 | 15.484 | 13434 | 138 | exact_preserve | connectivity_wall,documented_wall,exact_preserve,gather_heavy,high_memory,open_angle,scan |
