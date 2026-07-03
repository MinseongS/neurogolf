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
| 1 | 349 | 15204.2 | 15.289 | 15300 | 1191 | semantic_or_handbuilt | conv_heavy,documented_wall,high_memory,local_stencil,lut_selection,open_angle,qlinear |
| 2 | 080 | 8642.5 | 15.735 | 10109 | 454 | exact_preserve | conv_heavy,documented_wall,exact_preserve,high_memory,lut_selection,open_angle,qlinear |
| 3 | 222 | 7588.0 | 16.002 | 5136 | 2952 | exact_preserve | conv_heavy,custom_win,exact_preserve,local_stencil,maxpool_scan,onehot_final_equal,open_angle,qlinear |
| 4 | 328 | 6275.0 | 16.179 | 4835 | 1940 | exact_preserve | exact_preserve,lut_selection |
| 5 | 279 | 5546.0 | 16.379 | 5508 | 38 | semantic_or_handbuilt | conv_heavy,local_stencil,lut_selection,qlinear |
| 6 | 008 | 4413.0 | 16.500 | 4809 | 104 | exact_preserve | exact_preserve,lut_selection |
| 7 | 208 | 4226.0 | 16.539 | 4612 | 114 | exact_preserve | exact_preserve,onehot_final_equal,open_angle,qlinear |
| 8 | 117 | 3665.0 | 16.666 | 3922 | 243 | exact_preserve | exact_preserve,local_stencil,lut_selection,open_angle,qlinear |
| 9 | 280 | 2646.0 | 16.670 | 3363 | 783 | semantic_or_handbuilt | custom_win,documented_wall,lut_selection |
| 10 | 091 | 2528.0 | 16.984 | 2853 | 175 | exact_preserve | exact_preserve,local_stencil,lut_selection,open_angle |
| 11 | 029 | -34975.0 | 16.383 | 5436 | 89 | exact_preserve | exact_preserve,marginal_wall,onehot_final_equal,open_angle |

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
- candidates: 30
- expected: {'gain_type': 'frontier_research', 'risk': 'medium-high because it often requires semantic recompilation rather than graph surgery', 'verification': 'prove no counted full-canvas intermediates in mem profile, then fresh eval before adoption'}

| rank | task | score | pts | mem | params | source | tags |
|---:|---:|---:|---:|---:|---:|---|---|
| 1 | 054 | 20655.7 | 15.078 | 20091 | 288 | unknown | gather_heavy,high_memory,scan,scatter |
| 2 | 173 | 17379.5 | 15.222 | 17534 | 112 | exact_preserve | exact_preserve,gather_heavy,high_memory,scatter |
| 3 | 349 | 15204.2 | 15.289 | 15300 | 1191 | semantic_or_handbuilt | conv_heavy,documented_wall,high_memory,local_stencil,lut_selection,open_angle,qlinear |
| 4 | 191 | 12733.9 | 15.520 | 12919 | 171 | exact_preserve | assignment_wall,conv_heavy,custom_win,exact_preserve,high_memory,open_angle,qlinear |
| 5 | 025 | 11273.3 | 15.636 | 11520 | 144 | exact_preserve | exact_preserve,high_memory,onehot_final_equal,open_angle,scan |
| 6 | 138 | 10811.3 | 15.552 | 12215 | 462 | exact_preserve | conv_heavy,documented_wall,exact_preserve,high_memory,local_stencil,open_angle |
| 7 | 066 | 10187.7 | 15.778 | 9955 | 166 | semantic_or_handbuilt | marker_routed_path,open_angle,scan |
| 8 | 370 | 9222.0 | 15.823 | 8645 | 1024 | exact_preserve | conv_heavy,exact_preserve,qlinear,scan,scatter |
| 9 | 396 | 9207.6 | 15.825 | 7842 | 1813 | exact_preserve | conv_heavy,exact_preserve,local_stencil |
| 10 | 080 | 8642.5 | 15.735 | 10109 | 454 | exact_preserve | conv_heavy,documented_wall,exact_preserve,high_memory,lut_selection,open_angle,qlinear |
| 11 | 074 | 8583.2 | 15.889 | 9000 | 50 | exact_preserve | exact_preserve,local_stencil,open_angle |
| 12 | 377 | 7766.8 | 15.981 | 8111 | 150 | exact_preserve | exact_preserve,local_stencil |
| 13 | 222 | 7588.0 | 16.002 | 5136 | 2952 | exact_preserve | conv_heavy,custom_win,exact_preserve,local_stencil,maxpool_scan,onehot_final_equal,open_angle,qlinear |
| 14 | 017 | 7301.0 | 16.038 | 5997 | 1804 | exact_preserve | custom_win,exact_preserve,onehot_final_equal,open_angle |
| 15 | 324 | 6921.9 | 15.907 | 7308 | 1586 | exact_preserve | conv_heavy,documented_wall,exact_preserve,local_stencil,qlinear,scan |
| 16 | 096 | 6812.3 | 15.919 | 7987 | 801 | exact_preserve | documented_wall,exact_preserve,open_angle,scatter |
| 17 | 023 | 6412.0 | 16.234 | 6163 | 249 | semantic_or_handbuilt | conv_heavy,local_stencil,open_angle,qlinear |
| 18 | 328 | 6275.0 | 16.179 | 4835 | 1940 | exact_preserve | exact_preserve,lut_selection |
| 19 | 279 | 5546.0 | 16.379 | 5508 | 38 | semantic_or_handbuilt | conv_heavy,local_stencil,lut_selection,qlinear |
| 20 | 278 | 5483.0 | 16.391 | 5436 | 47 | semantic_or_handbuilt | local_stencil,qlinear |
| 21 | 382 | 5283.0 | 16.337 | 5648 | 135 | exact_preserve | exact_preserve,open_angle,scan |
| 22 | 192 | 4996.0 | 16.147 | 5745 | 1251 | exact_preserve | conv_heavy,documented_wall,exact_preserve,local_stencil,open_angle,qlinear |
| 23 | 264 | 4908.0 | 16.404 | 4700 | 708 | exact_preserve | exact_preserve,local_stencil,lut_selection,qlinear |
| 24 | 005 | 4882.0 | 16.163 | 6392 | 490 | exact_preserve | conv_heavy,documented_wall,exact_preserve,lut_selection,qlinear |
| 25 | 085 | 4881.0 | 16.409 | 4500 | 881 | exact_preserve | exact_preserve,local_stencil,open_angle |

## threshold_linearize_pairwise_onehot_and

Replace pairwise one-hot AND/Kronecker products with direct thresholded product scores

- source tasks: 001
- candidates: 69
- expected: {'gain_type': 'memory_by_deleting_boolean_product_carrier', 'risk': 'medium; relies on one-hot inputs, positive-threshold scoring, and class-specific margins', 'verification': 'stored eval plus fresh/adopt when generator samples exist; inspect off-footprint scores are non-positive'}

| rank | task | score | pts | mem | params | source | tags |
|---:|---:|---:|---:|---:|---:|---|---|
| 1 | 366 | 31466.9 | 14.640 | 30983 | 576 | exact_preserve | assignment_wall,connectivity_wall,exact_preserve,gather_heavy,high_memory,low_score,lut_selection,open_angle |
| 2 | 367 | 18288.8 | 15.121 | 14800 | 4725 | semantic_or_handbuilt | connectivity_wall,conv_heavy,documented_wall,high_memory,local_stencil,lut_selection,open_angle,qlinear |
| 3 | 349 | 15204.2 | 15.289 | 15300 | 1191 | semantic_or_handbuilt | conv_heavy,documented_wall,high_memory,local_stencil,lut_selection,open_angle,qlinear |
| 4 | 101 | 14620.5 | 15.422 | 13573 | 874 | semantic_or_handbuilt | connectivity_wall,gather_heavy,high_memory,onehot_final_equal,scatter |
| 5 | 025 | 11273.3 | 15.636 | 11520 | 144 | exact_preserve | exact_preserve,high_memory,onehot_final_equal,open_angle,scan |
| 6 | 145 | 9069.5 | 15.742 | 9244 | 1248 | semantic_or_handbuilt | connectivity_wall,custom_win,documented_wall,lut_selection,open_angle |
| 7 | 080 | 8642.5 | 15.735 | 10109 | 454 | exact_preserve | conv_heavy,documented_wall,exact_preserve,high_memory,lut_selection,open_angle,qlinear |
| 8 | 064 | 8548.7 | 15.791 | 9852 | 134 | semantic_or_handbuilt | connectivity_wall,documented_wall,onehot_final_equal,open_angle |
| 9 | 222 | 7588.0 | 16.002 | 5136 | 2952 | exact_preserve | conv_heavy,custom_win,exact_preserve,local_stencil,maxpool_scan,onehot_final_equal,open_angle,qlinear |
| 10 | 017 | 7301.0 | 16.038 | 5997 | 1804 | exact_preserve | custom_win,exact_preserve,onehot_final_equal,open_angle |
| 11 | 157 | 6342.4 | 15.972 | 7983 | 351 | exact_preserve | assignment_wall,connectivity_wall,documented_wall,exact_preserve,gather_heavy,lut_selection,open_angle,scatter |
| 12 | 187 | 6330.0 | 15.973 | 5000 | 3322 | exact_preserve | connectivity_wall,documented_wall,exact_preserve,local_stencil,lut_selection |
| 13 | 328 | 6275.0 | 16.179 | 4835 | 1940 | exact_preserve | exact_preserve,lut_selection |
| 14 | 368 | 5725.0 | 16.264 | 6022 | 203 | exact_preserve | connectivity_wall,exact_preserve,gather_heavy,onehot_final_equal,scatter |
| 15 | 219 | 5710.0 | 16.117 | 7078 | 132 | unknown | assignment_wall,connectivity_wall,documented_wall,lut_selection,onehot_final_equal,open_angle,scan |
| 16 | 279 | 5546.0 | 16.379 | 5508 | 38 | semantic_or_handbuilt | conv_heavy,local_stencil,lut_selection,qlinear |
| 17 | 002 | 5189.0 | 16.192 | 6100 | 589 | semantic_or_handbuilt | connectivity_wall,documented_wall,lut_selection |
| 18 | 009 | 5024.0 | 16.143 | 6929 | 95 | exact_preserve | connectivity_wall,documented_wall,exact_preserve,local_stencil,lut_selection,onehot_final_equal,open_angle |
| 19 | 182 | 4942.0 | 16.229 | 6345 | 97 | semantic_or_handbuilt | assignment_wall,connectivity_wall,conv_heavy,documented_wall,local_stencil,lut_selection,onehot_final_equal,open_angle |
| 20 | 264 | 4908.0 | 16.404 | 4700 | 708 | exact_preserve | exact_preserve,local_stencil,lut_selection,qlinear |
| 21 | 005 | 4882.0 | 16.163 | 6392 | 490 | exact_preserve | conv_heavy,documented_wall,exact_preserve,lut_selection,qlinear |
| 22 | 004 | 4637.0 | 16.456 | 5044 | 93 | exact_preserve | connectivity_wall,exact_preserve,local_stencil,lut_selection,open_angle,qlinear |
| 23 | 383 | 4570.0 | 16.289 | 5974 | 96 | unknown | assignment_wall,connectivity_wall,documented_wall,local_stencil,onehot_final_equal,open_angle |
| 24 | 008 | 4413.0 | 16.500 | 4809 | 104 | exact_preserve | exact_preserve,lut_selection |
| 25 | 174 | 4389.0 | 16.319 | 5743 | 146 | semantic_or_handbuilt | connectivity_wall,documented_wall,matmul,onehot_final_equal,open_angle |

## qlinear_uint8_lut_or_matmul

Replace fp16/fp32 one-hot LUT/MatMul selection with uint8 QLinearMatMul/QLinearConv where exact

- source tasks: 055 080 338
- candidates: 1
- expected: {'gain_type': 'memory', 'risk': 'medium', 'verification': 'stored eval, fresh eval if generator available, then adopt gate'}

| rank | task | score | pts | mem | params | source | tags |
|---:|---:|---:|---:|---:|---:|---|---|
| 1 | 174 | 4389.0 | 16.319 | 5743 | 146 | semantic_or_handbuilt | connectivity_wall,documented_wall,matmul,onehot_final_equal,open_angle |

## free_final_onehot_equal

Delay 10-channel expansion to final Equal/Where output

- source tasks: 017 095 146 381
- candidates: 25
- expected: {'gain_type': 'memory', 'risk': 'low-medium', 'verification': 'output equivalence before adoption'}

| rank | task | score | pts | mem | params | source | tags |
|---:|---:|---:|---:|---:|---:|---|---|
| 1 | 101 | 14620.5 | 15.422 | 13573 | 874 | semantic_or_handbuilt | connectivity_wall,gather_heavy,high_memory,onehot_final_equal,scatter |
| 2 | 025 | 11273.3 | 15.636 | 11520 | 144 | exact_preserve | exact_preserve,high_memory,onehot_final_equal,open_angle,scan |
| 3 | 064 | 8548.7 | 15.791 | 9852 | 134 | semantic_or_handbuilt | connectivity_wall,documented_wall,onehot_final_equal,open_angle |
| 4 | 222 | 7588.0 | 16.002 | 5136 | 2952 | exact_preserve | conv_heavy,custom_win,exact_preserve,local_stencil,maxpool_scan,onehot_final_equal,open_angle,qlinear |
| 5 | 017 | 7301.0 | 16.038 | 5997 | 1804 | exact_preserve | custom_win,exact_preserve,onehot_final_equal,open_angle |
| 6 | 368 | 5725.0 | 16.264 | 6022 | 203 | exact_preserve | connectivity_wall,exact_preserve,gather_heavy,onehot_final_equal,scatter |
| 7 | 219 | 5710.0 | 16.117 | 7078 | 132 | unknown | assignment_wall,connectivity_wall,documented_wall,lut_selection,onehot_final_equal,open_angle,scan |
| 8 | 009 | 5024.0 | 16.143 | 6929 | 95 | exact_preserve | connectivity_wall,documented_wall,exact_preserve,local_stencil,lut_selection,onehot_final_equal,open_angle |
| 9 | 182 | 4942.0 | 16.229 | 6345 | 97 | semantic_or_handbuilt | assignment_wall,connectivity_wall,conv_heavy,documented_wall,local_stencil,lut_selection,onehot_final_equal,open_angle |
| 10 | 383 | 4570.0 | 16.289 | 5974 | 96 | unknown | assignment_wall,connectivity_wall,documented_wall,local_stencil,onehot_final_equal,open_angle |
| 11 | 174 | 4389.0 | 16.319 | 5743 | 146 | semantic_or_handbuilt | connectivity_wall,documented_wall,matmul,onehot_final_equal,open_angle |
| 12 | 208 | 4226.0 | 16.539 | 4612 | 114 | exact_preserve | exact_preserve,onehot_final_equal,open_angle,qlinear |
| 13 | 363 | 3635.0 | 16.673 | 3839 | 296 | exact_preserve | connectivity_wall,conv_heavy,exact_preserve,local_stencil,onehot_final_equal,open_angle |
| 14 | 107 | 3578.0 | 16.687 | 2924 | 1154 | exact_preserve | exact_preserve,lut_selection,matmul,onehot_final_equal,open_angle,qlinear,scatter |
| 15 | 202 | 3287.0 | 16.902 | 3253 | 34 | unknown | connectivity_wall,onehot_final_equal,open_angle,qlinear |
| 16 | 234 | 3208.0 | 16.927 | 3144 | 64 | semantic_or_handbuilt | connectivity_wall,onehot_final_equal |
| 17 | 177 | 3121.0 | 16.805 | 3500 | 121 | exact_preserve | connectivity_wall,conv_heavy,exact_preserve,local_stencil,onehot_final_equal,open_angle |
| 18 | 036 | 1677.0 | 17.314 | 2140 | 37 | exact_preserve | connectivity_wall,exact_preserve,onehot_final_equal,open_angle |
| 19 | 392 | 1451.0 | 17.424 | 1602 | 349 | exact_preserve | custom_win,exact_preserve,onehot_final_equal,open_angle |
| 20 | 213 | 1382.0 | 17.460 | 1833 | 49 | exact_preserve | exact_preserve,onehot_final_equal,open_angle |
| 21 | 333 | 1227.0 | 16.921 | 2792 | 435 | exact_preserve | connectivity_wall,documented_wall,exact_preserve,onehot_final_equal,open_angle |
| 22 | 308 | 988.0 | 17.695 | 1385 | 103 | exact_preserve | connectivity_wall,exact_preserve,onehot_final_equal,open_angle,scatter |
| 23 | 071 | 698.0 | 17.100 | 2643 | 55 | exact_preserve | connectivity_wall,documented_wall,exact_preserve,onehot_final_equal,open_angle |
| 24 | 170 | 545.0 | 17.158 | 2216 | 329 | exact_preserve | assignment_wall,conv_heavy,documented_wall,exact_preserve,local_stencil,lut_selection,onehot_final_equal,open_angle |
| 25 | 029 | -34975.0 | 16.383 | 5436 | 89 | exact_preserve | exact_preserve,marginal_wall,onehot_final_equal,open_angle |

## scan_dtype_and_shift_compression

Compress scan-style MaxPool/CumSum/Hillis-Steele pipelines with lower dtype or shared shifts

- source tasks: 046 216
- candidates: 28
- expected: {'gain_type': 'memory', 'risk': 'medium-high', 'verification': 'stored/fresh eval and compare against incumbent'}

| rank | task | score | pts | mem | params | source | tags |
|---:|---:|---:|---:|---:|---:|---|---|
| 1 | 233 | 32165.5 | 14.588 | 32796 | 446 | semantic_or_handbuilt | assignment_wall,connectivity_wall,conv_heavy,documented_wall,gather_heavy,high_memory,low_score,matmul |
| 2 | 366 | 31466.9 | 14.640 | 30983 | 576 | exact_preserve | assignment_wall,connectivity_wall,exact_preserve,gather_heavy,high_memory,low_score,lut_selection,open_angle |
| 3 | 133 | 28025.0 | 14.703 | 27333 | 2303 | exact_preserve | assignment_wall,connectivity_wall,conv_heavy,documented_wall,exact_preserve,gather_heavy,high_memory,low_score |
| 4 | 285 | 23574.8 | 14.864 | 24684 | 550 | exact_preserve | connectivity_wall,documented_wall,exact_preserve,gather_heavy,high_memory,low_score,open_angle,scatter |
| 5 | 054 | 20655.7 | 15.078 | 20091 | 288 | unknown | gather_heavy,high_memory,scan,scatter |
| 6 | 173 | 17379.5 | 15.222 | 17534 | 112 | exact_preserve | exact_preserve,gather_heavy,high_memory,scatter |
| 7 | 319 | 17004.7 | 15.161 | 18645 | 108 | exact_preserve | assignment_wall,documented_wall,exact_preserve,gather_heavy,high_memory,scan |
| 8 | 198 | 11720.6 | 15.485 | 13428 | 138 | exact_preserve | connectivity_wall,documented_wall,exact_preserve,gather_heavy,high_memory,open_angle,scan |
| 9 | 025 | 11273.3 | 15.636 | 11520 | 144 | exact_preserve | exact_preserve,high_memory,onehot_final_equal,open_angle,scan |
| 10 | 066 | 10187.7 | 15.778 | 9955 | 166 | semantic_or_handbuilt | marker_routed_path,open_angle,scan |
| 11 | 370 | 9222.0 | 15.823 | 8645 | 1024 | exact_preserve | conv_heavy,exact_preserve,qlinear,scan,scatter |
| 12 | 204 | 8302.0 | 15.767 | 9780 | 452 | exact_preserve | connectivity_wall,conv_heavy,custom_win,documented_wall,exact_preserve,local_stencil,maxpool_scan,open_angle |
| 13 | 222 | 7588.0 | 16.002 | 5136 | 2952 | exact_preserve | conv_heavy,custom_win,exact_preserve,local_stencil,maxpool_scan,onehot_final_equal,open_angle,qlinear |
| 14 | 324 | 6921.9 | 15.907 | 7308 | 1586 | exact_preserve | conv_heavy,documented_wall,exact_preserve,local_stencil,qlinear,scan |
| 15 | 157 | 6342.4 | 15.972 | 7983 | 351 | exact_preserve | assignment_wall,connectivity_wall,documented_wall,exact_preserve,gather_heavy,lut_selection,open_angle,scatter |
| 16 | 219 | 5710.0 | 16.117 | 7078 | 132 | unknown | assignment_wall,connectivity_wall,documented_wall,lut_selection,onehot_final_equal,open_angle,scan |
| 17 | 382 | 5283.0 | 16.337 | 5648 | 135 | exact_preserve | exact_preserve,open_angle,scan |
| 18 | 277 | 3245.0 | 16.772 | 3701 | 44 | exact_preserve | connectivity_wall,exact_preserve,lut_selection,maxpool_scan |
| 19 | 055 | 3092.0 | 16.963 | 3030 | 62 | semantic_or_handbuilt | connectivity_wall,lut_selection,open_angle,qlinear,scan |
| 20 | 090 | 2572.0 | 16.970 | 2990 | 82 | exact_preserve | exact_preserve,scan |
| 21 | 196 | 2538.0 | 16.580 | 4500 | 38 | exact_preserve | connectivity_wall,custom_win,documented_wall,exact_preserve,local_stencil,maxpool_scan,open_angle,qlinear |
| 22 | 273 | 1660.0 | 17.322 | 2109 | 51 | exact_preserve | connectivity_wall,exact_preserve,gather_heavy,open_angle |
| 23 | 125 | 1463.0 | 16.850 | 3434 | 29 | exact_preserve | connectivity_wall,documented_wall,exact_preserve,maxpool_scan,open_angle |
| 24 | 046 | 818.0 | 17.252 | 2221 | 97 | heuristic | connectivity_wall,documented_wall,heuristic,local_stencil,scan |
| 25 | 184 | 403.0 | 17.216 | 2340 | 63 | exact_preserve | connectivity_wall,conv_heavy,documented_wall,exact_preserve,local_stencil,open_angle,qlinear,scan |

## sparse_conv_single_op_floor

Collapse local neighborhood rules to one sparse Conv/QLinearConv when output is thresholded

- source tasks: 015 095 230 294
- candidates: 64
- expected: {'gain_type': 'memory+params', 'risk': 'low when rule is truly local', 'verification': 'fresh process eval; beware one-process mem0 false signals'}

| rank | task | score | pts | mem | params | source | tags |
|---:|---:|---:|---:|---:|---:|---|---|
| 1 | 349 | 15204.2 | 15.289 | 15300 | 1191 | semantic_or_handbuilt | conv_heavy,documented_wall,high_memory,local_stencil,lut_selection,open_angle,qlinear |
| 2 | 255 | 14639.9 | 15.387 | 14663 | 293 | exact_preserve | conv_heavy,exact_preserve,high_memory,matmul,qlinear |
| 3 | 138 | 10811.3 | 15.552 | 12215 | 462 | exact_preserve | conv_heavy,documented_wall,exact_preserve,high_memory,local_stencil,open_angle |
| 4 | 370 | 9222.0 | 15.823 | 8645 | 1024 | exact_preserve | conv_heavy,exact_preserve,qlinear,scan,scatter |
| 5 | 396 | 9207.6 | 15.825 | 7842 | 1813 | exact_preserve | conv_heavy,exact_preserve,local_stencil |
| 6 | 080 | 8642.5 | 15.735 | 10109 | 454 | exact_preserve | conv_heavy,documented_wall,exact_preserve,high_memory,lut_selection,open_angle,qlinear |
| 7 | 074 | 8583.2 | 15.889 | 9000 | 50 | exact_preserve | exact_preserve,local_stencil,open_angle |
| 8 | 377 | 7766.8 | 15.981 | 8111 | 150 | exact_preserve | exact_preserve,local_stencil |
| 9 | 222 | 7588.0 | 16.002 | 5136 | 2952 | exact_preserve | conv_heavy,custom_win,exact_preserve,local_stencil,maxpool_scan,onehot_final_equal,open_angle,qlinear |
| 10 | 324 | 6921.9 | 15.907 | 7308 | 1586 | exact_preserve | conv_heavy,documented_wall,exact_preserve,local_stencil,qlinear,scan |
| 11 | 023 | 6412.0 | 16.234 | 6163 | 249 | semantic_or_handbuilt | conv_heavy,local_stencil,open_angle,qlinear |
| 12 | 279 | 5546.0 | 16.379 | 5508 | 38 | semantic_or_handbuilt | conv_heavy,local_stencil,lut_selection,qlinear |
| 13 | 278 | 5483.0 | 16.391 | 5436 | 47 | semantic_or_handbuilt | local_stencil,qlinear |
| 14 | 192 | 4996.0 | 16.147 | 5745 | 1251 | exact_preserve | conv_heavy,documented_wall,exact_preserve,local_stencil,open_angle,qlinear |
| 15 | 264 | 4908.0 | 16.404 | 4700 | 708 | exact_preserve | exact_preserve,local_stencil,lut_selection,qlinear |
| 16 | 005 | 4882.0 | 16.163 | 6392 | 490 | exact_preserve | conv_heavy,documented_wall,exact_preserve,lut_selection,qlinear |
| 17 | 085 | 4881.0 | 16.409 | 4500 | 881 | exact_preserve | exact_preserve,local_stencil,open_angle |
| 18 | 117 | 3665.0 | 16.666 | 3922 | 243 | exact_preserve | exact_preserve,local_stencil,lut_selection,open_angle,qlinear |
| 19 | 132 | 3589.0 | 16.684 | 3990 | 99 | exact_preserve | bitwise_program,exact_preserve,gather_heavy,local_stencil,open_angle |
| 20 | 162 | 3568.0 | 16.689 | 4024 | 44 | exact_preserve | exact_preserve,local_stencil,open_angle,qlinear |
| 21 | 284 | 3235.0 | 16.774 | 3082 | 653 | exact_preserve | conv_heavy,custom_win,exact_preserve,open_angle,scatter |
| 22 | 354 | 2751.0 | 17.080 | 2674 | 77 | semantic_or_handbuilt | conv_heavy,local_stencil,open_angle |
| 23 | 079 | 2643.0 | 16.947 | 3065 | 78 | exact_preserve | custom_win,exact_preserve,local_stencil,open_angle |
| 24 | 042 | 2610.0 | 16.958 | 2792 | 318 | exact_preserve | conv_heavy,exact_preserve,local_stencil,qlinear |
| 25 | 091 | 2528.0 | 16.984 | 2853 | 175 | exact_preserve | exact_preserve,local_stencil,lut_selection,open_angle |

## exact_preserve_to_semantic_rewrite

Prioritize exact-preserve source builders for semantic replacement

- source tasks: 002 018 286 366
- candidates: 17
- expected: {'gain_type': 'research', 'risk': 'high', 'verification': 'document semantic mechanism before implementation'}

| rank | task | score | pts | mem | params | source | tags |
|---:|---:|---:|---:|---:|---:|---|---|
| 1 | 366 | 31466.9 | 14.640 | 30983 | 576 | exact_preserve | assignment_wall,connectivity_wall,exact_preserve,gather_heavy,high_memory,low_score,lut_selection,open_angle |
| 2 | 133 | 28025.0 | 14.703 | 27333 | 2303 | exact_preserve | assignment_wall,connectivity_wall,conv_heavy,documented_wall,exact_preserve,gather_heavy,high_memory,low_score |
| 3 | 285 | 23574.8 | 14.864 | 24684 | 550 | exact_preserve | connectivity_wall,documented_wall,exact_preserve,gather_heavy,high_memory,low_score,open_angle,scatter |
| 4 | 364 | 19396.5 | 15.115 | 18500 | 1131 | exact_preserve | connectivity_wall,exact_preserve,high_memory,local_stencil,open_angle,qlinear |
| 5 | 076 | 18064.0 | 15.107 | 19456 | 340 | exact_preserve | assignment_wall,connectivity_wall,conv_heavy,documented_wall,exact_preserve,high_memory,open_angle,scatter |
| 6 | 173 | 17379.5 | 15.222 | 17534 | 112 | exact_preserve | exact_preserve,gather_heavy,high_memory,scatter |
| 7 | 319 | 17004.7 | 15.161 | 18645 | 108 | exact_preserve | assignment_wall,documented_wall,exact_preserve,gather_heavy,high_memory,scan |
| 8 | 255 | 14639.9 | 15.387 | 14663 | 293 | exact_preserve | conv_heavy,exact_preserve,high_memory,matmul,qlinear |
| 9 | 191 | 12733.9 | 15.520 | 12919 | 171 | exact_preserve | assignment_wall,conv_heavy,custom_win,exact_preserve,high_memory,open_angle,qlinear |
| 10 | 205 | 11856.6 | 15.588 | 11598 | 635 | exact_preserve | connectivity_wall,conv_heavy,exact_preserve,high_memory,open_angle,qlinear |
| 11 | 198 | 11720.6 | 15.485 | 13428 | 138 | exact_preserve | connectivity_wall,documented_wall,exact_preserve,gather_heavy,high_memory,open_angle,scan |
| 12 | 025 | 11273.3 | 15.636 | 11520 | 144 | exact_preserve | exact_preserve,high_memory,onehot_final_equal,open_angle,scan |
| 13 | 138 | 10811.3 | 15.552 | 12215 | 462 | exact_preserve | conv_heavy,documented_wall,exact_preserve,high_memory,local_stencil,open_angle |
| 14 | 044 | 9213.9 | 15.684 | 11050 | 69 | exact_preserve | assignment_wall,connectivity_wall,conv_heavy,documented_wall,exact_preserve,high_memory,matmul |
| 15 | 080 | 8642.5 | 15.735 | 10109 | 454 | exact_preserve | conv_heavy,documented_wall,exact_preserve,high_memory,lut_selection,open_angle,qlinear |
| 16 | 158 | -20805.6 | 15.052 | 18263 | 2647 | exact_preserve | assignment_wall,conv_heavy,documented_wall,exact_preserve,gather_heavy,high_memory,marginal_wall,open_angle |
| 17 | 018 | -70610.6 | 14.659 | 29807 | 1180 | exact_preserve | ambiguity_wall,documented_wall,exact_preserve,gather_heavy,high_memory,low_score,scatter |

## dihedral_template_match_stacked_conv

Use one stacked Conv for all dihedral orientations of a small runtime-extracted template

- source tasks: 191
- candidates: 1
- expected: {'gain_type': 'memory+accuracy', 'risk': 'medium', 'verification': 'fresh exact eval; compare against incumbent and propagate if template extraction is deterministic'}

| rank | task | score | pts | mem | params | source | tags |
|---:|---:|---:|---:|---:|---:|---|---|
| 1 | 182 | 4942.0 | 16.229 | 6345 | 97 | semantic_or_handbuilt | assignment_wall,connectivity_wall,conv_heavy,documented_wall,local_stencil,lut_selection,onehot_final_equal,open_angle |

## bounded_crop_before_connectivity_scan

Run iterative connectivity/flood-fill on the generator's true max canvas, not the 30x30 harness canvas

- source tasks: 187
- candidates: 21
- expected: {'gain_type': 'memory+generalization', 'risk': 'medium', 'verification': 'confirm generator max height/width, stored eval, fresh adopt gate'}

| rank | task | score | pts | mem | params | source | tags |
|---:|---:|---:|---:|---:|---:|---|---|
| 1 | 233 | 32165.5 | 14.588 | 32796 | 446 | semantic_or_handbuilt | assignment_wall,connectivity_wall,conv_heavy,documented_wall,gather_heavy,high_memory,low_score,matmul |
| 2 | 366 | 31466.9 | 14.640 | 30983 | 576 | exact_preserve | assignment_wall,connectivity_wall,exact_preserve,gather_heavy,high_memory,low_score,lut_selection,open_angle |
| 3 | 133 | 28025.0 | 14.703 | 27333 | 2303 | exact_preserve | assignment_wall,connectivity_wall,conv_heavy,documented_wall,exact_preserve,gather_heavy,high_memory,low_score |
| 4 | 285 | 23574.8 | 14.864 | 24684 | 550 | exact_preserve | connectivity_wall,documented_wall,exact_preserve,gather_heavy,high_memory,low_score,open_angle,scatter |
| 5 | 054 | 20655.7 | 15.078 | 20091 | 288 | unknown | gather_heavy,high_memory,scan,scatter |
| 6 | 076 | 18064.0 | 15.107 | 19456 | 340 | exact_preserve | assignment_wall,connectivity_wall,conv_heavy,documented_wall,exact_preserve,high_memory,open_angle,scatter |
| 7 | 173 | 17379.5 | 15.222 | 17534 | 112 | exact_preserve | exact_preserve,gather_heavy,high_memory,scatter |
| 8 | 319 | 17004.7 | 15.161 | 18645 | 108 | exact_preserve | assignment_wall,documented_wall,exact_preserve,gather_heavy,high_memory,scan |
| 9 | 349 | 15204.2 | 15.289 | 15300 | 1191 | semantic_or_handbuilt | conv_heavy,documented_wall,high_memory,local_stencil,lut_selection,open_angle,qlinear |
| 10 | 255 | 14639.9 | 15.387 | 14663 | 293 | exact_preserve | conv_heavy,exact_preserve,high_memory,matmul,qlinear |
| 11 | 191 | 12733.9 | 15.520 | 12919 | 171 | exact_preserve | assignment_wall,conv_heavy,custom_win,exact_preserve,high_memory,open_angle,qlinear |
| 12 | 205 | 11856.6 | 15.588 | 11598 | 635 | exact_preserve | connectivity_wall,conv_heavy,exact_preserve,high_memory,open_angle,qlinear |
| 13 | 198 | 11720.6 | 15.485 | 13428 | 138 | exact_preserve | connectivity_wall,documented_wall,exact_preserve,gather_heavy,high_memory,open_angle,scan |
| 14 | 025 | 11273.3 | 15.636 | 11520 | 144 | exact_preserve | exact_preserve,high_memory,onehot_final_equal,open_angle,scan |
| 15 | 138 | 10811.3 | 15.552 | 12215 | 462 | exact_preserve | conv_heavy,documented_wall,exact_preserve,high_memory,local_stencil,open_angle |
| 16 | 338 | 9248.4 | 15.725 | 10000 | 666 | semantic_or_handbuilt | connectivity_wall,documented_wall,high_memory,local_stencil,open_angle,qlinear |
| 17 | 044 | 9213.9 | 15.684 | 11050 | 69 | exact_preserve | assignment_wall,connectivity_wall,conv_heavy,documented_wall,exact_preserve,high_memory,matmul |
| 18 | 080 | 8642.5 | 15.735 | 10109 | 454 | exact_preserve | conv_heavy,documented_wall,exact_preserve,high_memory,lut_selection,open_angle,qlinear |
| 19 | 018 | -70610.6 | 14.659 | 29807 | 1180 | exact_preserve | ambiguity_wall,documented_wall,exact_preserve,gather_heavy,high_memory,low_score,scatter |
| 20 | 118 | -75711.7 | 14.856 | 24722 | 723 | semantic_or_handbuilt | connectivity_wall,conv_heavy,documented_wall,high_memory,infeasible_exact_wall,information_loss_wall,local_stencil,low_score |
| 21 | 209 | -78474.8 | 14.969 | 21298 | 1418 | unknown | ambiguity_wall,assignment_wall,connectivity_wall,conv_heavy,documented_wall,high_memory,low_score,lut_selection |

## public_teacher_bitwise_scan_replacement

Replace repeated MaxPool/Max/Min scan stacks with bitwise shift/mask routing when the state is binary

- source tasks: 002 209
- candidates: 11
- expected: {'gain_type': 'memory', 'risk': 'medium-high', 'verification': 'stored eval, fresh eval when generator available, public-probe if strict fresh is known weak'}

| rank | task | score | pts | mem | params | source | tags |
|---:|---:|---:|---:|---:|---:|---|---|
| 1 | 233 | 32165.5 | 14.588 | 32796 | 446 | semantic_or_handbuilt | assignment_wall,connectivity_wall,conv_heavy,documented_wall,gather_heavy,high_memory,low_score,matmul |
| 2 | 133 | 28025.0 | 14.703 | 27333 | 2303 | exact_preserve | assignment_wall,connectivity_wall,conv_heavy,documented_wall,exact_preserve,gather_heavy,high_memory,low_score |
| 3 | 285 | 23574.8 | 14.864 | 24684 | 550 | exact_preserve | connectivity_wall,documented_wall,exact_preserve,gather_heavy,high_memory,low_score,open_angle,scatter |
| 4 | 054 | 20655.7 | 15.078 | 20091 | 288 | unknown | gather_heavy,high_memory,scan,scatter |
| 5 | 076 | 18064.0 | 15.107 | 19456 | 340 | exact_preserve | assignment_wall,connectivity_wall,conv_heavy,documented_wall,exact_preserve,high_memory,open_angle,scatter |
| 6 | 173 | 17379.5 | 15.222 | 17534 | 112 | exact_preserve | exact_preserve,gather_heavy,high_memory,scatter |
| 7 | 349 | 15204.2 | 15.289 | 15300 | 1191 | semantic_or_handbuilt | conv_heavy,documented_wall,high_memory,local_stencil,lut_selection,open_angle,qlinear |
| 8 | 158 | -20805.6 | 15.052 | 18263 | 2647 | exact_preserve | assignment_wall,conv_heavy,documented_wall,exact_preserve,gather_heavy,high_memory,marginal_wall,open_angle |
| 9 | 018 | -70610.6 | 14.659 | 29807 | 1180 | exact_preserve | ambiguity_wall,documented_wall,exact_preserve,gather_heavy,high_memory,low_score,scatter |
| 10 | 118 | -75711.7 | 14.856 | 24722 | 723 | semantic_or_handbuilt | connectivity_wall,conv_heavy,documented_wall,high_memory,infeasible_exact_wall,information_loss_wall,local_stencil,low_score |
| 11 | 209 | -78474.8 | 14.969 | 21298 | 1418 | unknown | ambiguity_wall,assignment_wall,connectivity_wall,conv_heavy,documented_wall,high_memory,low_score,lut_selection |

## public_teacher_qlinear_conv_rewrite

Convert repeated binary/count Conv/ConvTranspose towers to QLinearConv/uint8 routes

- source tasks: 023 349 182 233 255 364 338 004
- candidates: 32
- expected: {'gain_type': 'memory+params', 'risk': 'medium', 'verification': 'stored eval plus fresh; inspect for output-range saturation before adoption'}

| rank | task | score | pts | mem | params | source | tags |
|---:|---:|---:|---:|---:|---:|---|---|
| 1 | 286 | 22796.4 | 14.915 | 21090 | 2881 | semantic_or_handbuilt | connectivity_wall,conv_heavy,documented_wall,high_memory,local_stencil,low_score,qlinear |
| 2 | 349 | 15204.2 | 15.289 | 15300 | 1191 | semantic_or_handbuilt | conv_heavy,documented_wall,high_memory,local_stencil,lut_selection,open_angle,qlinear |
| 3 | 255 | 14639.9 | 15.387 | 14663 | 293 | exact_preserve | conv_heavy,exact_preserve,high_memory,matmul,qlinear |
| 4 | 205 | 11856.6 | 15.588 | 11598 | 635 | exact_preserve | connectivity_wall,conv_heavy,exact_preserve,high_memory,open_angle,qlinear |
| 5 | 138 | 10811.3 | 15.552 | 12215 | 462 | exact_preserve | conv_heavy,documented_wall,exact_preserve,high_memory,local_stencil,open_angle |
| 6 | 396 | 9207.6 | 15.825 | 7842 | 1813 | exact_preserve | conv_heavy,exact_preserve,local_stencil |
| 7 | 080 | 8642.5 | 15.735 | 10109 | 454 | exact_preserve | conv_heavy,documented_wall,exact_preserve,high_memory,lut_selection,open_angle,qlinear |
| 8 | 074 | 8583.2 | 15.889 | 9000 | 50 | exact_preserve | exact_preserve,local_stencil,open_angle |
| 9 | 377 | 7766.8 | 15.981 | 8111 | 150 | exact_preserve | exact_preserve,local_stencil |
| 10 | 222 | 7588.0 | 16.002 | 5136 | 2952 | exact_preserve | conv_heavy,custom_win,exact_preserve,local_stencil,maxpool_scan,onehot_final_equal,open_angle,qlinear |
| 11 | 324 | 6921.9 | 15.907 | 7308 | 1586 | exact_preserve | conv_heavy,documented_wall,exact_preserve,local_stencil,qlinear,scan |
| 12 | 187 | 6330.0 | 15.973 | 5000 | 3322 | exact_preserve | connectivity_wall,documented_wall,exact_preserve,local_stencil,lut_selection |
| 13 | 009 | 5024.0 | 16.143 | 6929 | 95 | exact_preserve | connectivity_wall,documented_wall,exact_preserve,local_stencil,lut_selection,onehot_final_equal,open_angle |
| 14 | 192 | 4996.0 | 16.147 | 5745 | 1251 | exact_preserve | conv_heavy,documented_wall,exact_preserve,local_stencil,open_angle,qlinear |
| 15 | 005 | 4882.0 | 16.163 | 6392 | 490 | exact_preserve | conv_heavy,documented_wall,exact_preserve,lut_selection,qlinear |
| 16 | 085 | 4881.0 | 16.409 | 4500 | 881 | exact_preserve | exact_preserve,local_stencil,open_angle |
| 17 | 004 | 4637.0 | 16.456 | 5044 | 93 | exact_preserve | connectivity_wall,exact_preserve,local_stencil,lut_selection,open_angle,qlinear |
| 18 | 165 | 4058.0 | 16.291 | 5828 | 230 | exact_preserve | connectivity_wall,documented_wall,exact_preserve,local_stencil,open_angle,qlinear |
| 19 | 363 | 3635.0 | 16.673 | 3839 | 296 | exact_preserve | connectivity_wall,conv_heavy,exact_preserve,local_stencil,onehot_final_equal,open_angle |
| 20 | 132 | 3589.0 | 16.684 | 3990 | 99 | exact_preserve | bitwise_program,exact_preserve,gather_heavy,local_stencil,open_angle |
| 21 | 284 | 3235.0 | 16.774 | 3082 | 653 | exact_preserve | conv_heavy,custom_win,exact_preserve,open_angle,scatter |
| 22 | 177 | 3121.0 | 16.805 | 3500 | 121 | exact_preserve | connectivity_wall,conv_heavy,exact_preserve,local_stencil,onehot_final_equal,open_angle |
| 23 | 354 | 2751.0 | 17.080 | 2674 | 77 | semantic_or_handbuilt | conv_heavy,local_stencil,open_angle |
| 24 | 079 | 2643.0 | 16.947 | 3065 | 78 | exact_preserve | custom_win,exact_preserve,local_stencil,open_angle |
| 25 | 042 | 2610.0 | 16.958 | 2792 | 318 | exact_preserve | conv_heavy,exact_preserve,local_stencil,qlinear |

## marker_routed_hidden_path_compiler

Compile endpoint-pair hidden paths from deliberate marker pixels instead of replaying scan-heavy exact graphs

- source tasks: 066
- candidates: 1
- expected: {'gain_type': 'semantic_rewrite_memory', 'risk': 'medium-high until tie-break is exact', 'verification': 'stored validation plus large fresh generator eval before ONNX adoption'}

| rank | task | score | pts | mem | params | source | tags |
|---:|---:|---:|---:|---:|---:|---|---|
| 1 | 066 | 10187.7 | 15.778 | 9955 | 166 | semantic_or_handbuilt | marker_routed_path,open_angle,scan |

## rotation_component_template_scatter

Replace scan-heavy in-context sprite reconstruction with rotation-only 3x3 component template scatter

- source tasks: 233
- candidates: 7
- expected: {'gain_type': 'semantic_rewrite_memory', 'risk': 'medium-high until bbox/component edge cases are fully fresh-verified', 'verification': 'Python reference stored plus large fresh, then source-owned ONNX compiler and adopt gate'}

| rank | task | score | pts | mem | params | source | tags |
|---:|---:|---:|---:|---:|---:|---|---|
| 1 | 285 | 23574.8 | 14.864 | 24684 | 550 | exact_preserve | connectivity_wall,documented_wall,exact_preserve,gather_heavy,high_memory,low_score,open_angle,scatter |
| 2 | 286 | 22796.4 | 14.915 | 21090 | 2881 | semantic_or_handbuilt | connectivity_wall,conv_heavy,documented_wall,high_memory,local_stencil,low_score,qlinear |
| 3 | 054 | 20655.7 | 15.078 | 20091 | 288 | unknown | gather_heavy,high_memory,scan,scatter |
| 4 | 364 | 19396.5 | 15.115 | 18500 | 1131 | exact_preserve | connectivity_wall,exact_preserve,high_memory,local_stencil,open_angle,qlinear |
| 5 | 173 | 17379.5 | 15.222 | 17534 | 112 | exact_preserve | exact_preserve,gather_heavy,high_memory,scatter |
| 6 | 349 | 15204.2 | 15.289 | 15300 | 1191 | semantic_or_handbuilt | conv_heavy,documented_wall,high_memory,local_stencil,lut_selection,open_angle,qlinear |
| 7 | 018 | -70610.6 | 14.659 | 29807 | 1180 | exact_preserve | ambiguity_wall,documented_wall,exact_preserve,gather_heavy,high_memory,low_score,scatter |

## band_profile_contraction_final_equal

Use distinct-colour row/column profiles to avoid full connectivity for band-fill tasks

- source tasks: 202
- candidates: 23
- expected: {'gain_type': 'semantic_rewrite_memory_or_source_ownership', 'risk': 'medium; requires generator proof that colours are distinct per band/profile', 'verification': 'stored eval plus large fresh generator eval; source/live reconcile after rewrite'}

| rank | task | score | pts | mem | params | source | tags |
|---:|---:|---:|---:|---:|---:|---|---|
| 1 | 054 | 20655.7 | 15.078 | 20091 | 288 | unknown | gather_heavy,high_memory,scan,scatter |
| 2 | 173 | 17379.5 | 15.222 | 17534 | 112 | exact_preserve | exact_preserve,gather_heavy,high_memory,scatter |
| 3 | 349 | 15204.2 | 15.289 | 15300 | 1191 | semantic_or_handbuilt | conv_heavy,documented_wall,high_memory,local_stencil,lut_selection,open_angle,qlinear |
| 4 | 255 | 14639.9 | 15.387 | 14663 | 293 | exact_preserve | conv_heavy,exact_preserve,high_memory,matmul,qlinear |
| 5 | 025 | 11273.3 | 15.636 | 11520 | 144 | exact_preserve | exact_preserve,high_memory,onehot_final_equal,open_angle,scan |
| 6 | 138 | 10811.3 | 15.552 | 12215 | 462 | exact_preserve | conv_heavy,documented_wall,exact_preserve,high_memory,local_stencil,open_angle |
| 7 | 396 | 9207.6 | 15.825 | 7842 | 1813 | exact_preserve | conv_heavy,exact_preserve,local_stencil |
| 8 | 080 | 8642.5 | 15.735 | 10109 | 454 | exact_preserve | conv_heavy,documented_wall,exact_preserve,high_memory,lut_selection,open_angle,qlinear |
| 9 | 074 | 8583.2 | 15.889 | 9000 | 50 | exact_preserve | exact_preserve,local_stencil,open_angle |
| 10 | 377 | 7766.8 | 15.981 | 8111 | 150 | exact_preserve | exact_preserve,local_stencil |
| 11 | 222 | 7588.0 | 16.002 | 5136 | 2952 | exact_preserve | conv_heavy,custom_win,exact_preserve,local_stencil,maxpool_scan,onehot_final_equal,open_angle,qlinear |
| 12 | 017 | 7301.0 | 16.038 | 5997 | 1804 | exact_preserve | custom_win,exact_preserve,onehot_final_equal,open_angle |
| 13 | 324 | 6921.9 | 15.907 | 7308 | 1586 | exact_preserve | conv_heavy,documented_wall,exact_preserve,local_stencil,qlinear,scan |
| 14 | 023 | 6412.0 | 16.234 | 6163 | 249 | semantic_or_handbuilt | conv_heavy,local_stencil,open_angle,qlinear |
| 15 | 279 | 5546.0 | 16.379 | 5508 | 38 | semantic_or_handbuilt | conv_heavy,local_stencil,lut_selection,qlinear |
| 16 | 278 | 5483.0 | 16.391 | 5436 | 47 | semantic_or_handbuilt | local_stencil,qlinear |
| 17 | 192 | 4996.0 | 16.147 | 5745 | 1251 | exact_preserve | conv_heavy,documented_wall,exact_preserve,local_stencil,open_angle,qlinear |
| 18 | 264 | 4908.0 | 16.404 | 4700 | 708 | exact_preserve | exact_preserve,local_stencil,lut_selection,qlinear |
| 19 | 085 | 4881.0 | 16.409 | 4500 | 881 | exact_preserve | exact_preserve,local_stencil,open_angle |
| 20 | 208 | 4226.0 | 16.539 | 4612 | 114 | exact_preserve | exact_preserve,onehot_final_equal,open_angle,qlinear |
| 21 | 162 | 3568.0 | 16.689 | 4024 | 44 | exact_preserve | exact_preserve,local_stencil,open_angle,qlinear |
| 22 | 029 | -34975.0 | 16.383 | 5436 | 89 | exact_preserve | exact_preserve,marginal_wall,onehot_final_equal,open_angle |
| 23 | 018 | -70610.6 | 14.659 | 29807 | 1180 | exact_preserve | ambiguity_wall,documented_wall,exact_preserve,gather_heavy,high_memory,low_score,scatter |

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
- candidates: 178
- expected: {'gain_type': 'memory+params', 'risk': 'low when concatenated tails are constant zero/false and Pad fill defaults to zero', 'verification': 'stored eval plus side-by-side fresh output equivalence against incumbent'}

| rank | task | score | pts | mem | params | source | tags |
|---:|---:|---:|---:|---:|---:|---|---|
| 1 | 233 | 32165.5 | 14.588 | 32796 | 446 | semantic_or_handbuilt | assignment_wall,connectivity_wall,conv_heavy,documented_wall,gather_heavy,high_memory,low_score,matmul |
| 2 | 366 | 31466.9 | 14.640 | 30983 | 576 | exact_preserve | assignment_wall,connectivity_wall,exact_preserve,gather_heavy,high_memory,low_score,lut_selection,open_angle |
| 3 | 133 | 28025.0 | 14.703 | 27333 | 2303 | exact_preserve | assignment_wall,connectivity_wall,conv_heavy,documented_wall,exact_preserve,gather_heavy,high_memory,low_score |
| 4 | 285 | 23574.8 | 14.864 | 24684 | 550 | exact_preserve | connectivity_wall,documented_wall,exact_preserve,gather_heavy,high_memory,low_score,open_angle,scatter |
| 5 | 286 | 22796.4 | 14.915 | 21090 | 2881 | semantic_or_handbuilt | connectivity_wall,conv_heavy,documented_wall,high_memory,local_stencil,low_score,qlinear |
| 6 | 054 | 20655.7 | 15.078 | 20091 | 288 | unknown | gather_heavy,high_memory,scan,scatter |
| 7 | 364 | 19396.5 | 15.115 | 18500 | 1131 | exact_preserve | connectivity_wall,exact_preserve,high_memory,local_stencil,open_angle,qlinear |
| 8 | 367 | 18288.8 | 15.121 | 14800 | 4725 | semantic_or_handbuilt | connectivity_wall,conv_heavy,documented_wall,high_memory,local_stencil,lut_selection,open_angle,qlinear |
| 9 | 076 | 18064.0 | 15.107 | 19456 | 340 | exact_preserve | assignment_wall,connectivity_wall,conv_heavy,documented_wall,exact_preserve,high_memory,open_angle,scatter |
| 10 | 173 | 17379.5 | 15.222 | 17534 | 112 | exact_preserve | exact_preserve,gather_heavy,high_memory,scatter |
| 11 | 319 | 17004.7 | 15.161 | 18645 | 108 | exact_preserve | assignment_wall,documented_wall,exact_preserve,gather_heavy,high_memory,scan |
| 12 | 349 | 15204.2 | 15.289 | 15300 | 1191 | semantic_or_handbuilt | conv_heavy,documented_wall,high_memory,local_stencil,lut_selection,open_angle,qlinear |
| 13 | 255 | 14639.9 | 15.387 | 14663 | 293 | exact_preserve | conv_heavy,exact_preserve,high_memory,matmul,qlinear |
| 14 | 101 | 14620.5 | 15.422 | 13573 | 874 | semantic_or_handbuilt | connectivity_wall,gather_heavy,high_memory,onehot_final_equal,scatter |
| 15 | 191 | 12733.9 | 15.520 | 12919 | 171 | exact_preserve | assignment_wall,conv_heavy,custom_win,exact_preserve,high_memory,open_angle,qlinear |
| 16 | 205 | 11856.6 | 15.588 | 11598 | 635 | exact_preserve | connectivity_wall,conv_heavy,exact_preserve,high_memory,open_angle,qlinear |
| 17 | 198 | 11720.6 | 15.485 | 13428 | 138 | exact_preserve | connectivity_wall,documented_wall,exact_preserve,gather_heavy,high_memory,open_angle,scan |
| 18 | 025 | 11273.3 | 15.636 | 11520 | 144 | exact_preserve | exact_preserve,high_memory,onehot_final_equal,open_angle,scan |
| 19 | 138 | 10811.3 | 15.552 | 12215 | 462 | exact_preserve | conv_heavy,documented_wall,exact_preserve,high_memory,local_stencil,open_angle |
| 20 | 066 | 10187.7 | 15.778 | 9955 | 166 | semantic_or_handbuilt | marker_routed_path,open_angle,scan |
| 21 | 379 | 9487.9 | 15.847 | 9309 | 133 | semantic_or_handbuilt | connectivity_wall,custom_win,open_angle |
| 22 | 338 | 9248.4 | 15.725 | 10000 | 666 | semantic_or_handbuilt | connectivity_wall,documented_wall,high_memory,local_stencil,open_angle,qlinear |
| 23 | 370 | 9222.0 | 15.823 | 8645 | 1024 | exact_preserve | conv_heavy,exact_preserve,qlinear,scan,scatter |
| 24 | 044 | 9213.9 | 15.684 | 11050 | 69 | exact_preserve | assignment_wall,connectivity_wall,conv_heavy,documented_wall,exact_preserve,high_memory,matmul |
| 25 | 145 | 9069.5 | 15.742 | 9244 | 1248 | semantic_or_handbuilt | connectivity_wall,custom_win,documented_wall,lut_selection,open_angle |

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
- candidates: 206
- expected: {'gain_type': 'memory+params', 'risk': 'low when every output on-cell is copied from the original one-hot input and off-grid can point at an all-zero padded input coordinate', 'verification': 'stored eval plus fresh side-by-side; confirm black in-grid cells and off-grid all-zero semantics are both preserved'}

| rank | task | score | pts | mem | params | source | tags |
|---:|---:|---:|---:|---:|---:|---|---|
| 1 | 366 | 31466.9 | 14.640 | 30983 | 576 | exact_preserve | assignment_wall,connectivity_wall,exact_preserve,gather_heavy,high_memory,low_score,lut_selection,open_angle |
| 2 | 133 | 28025.0 | 14.703 | 27333 | 2303 | exact_preserve | assignment_wall,connectivity_wall,conv_heavy,documented_wall,exact_preserve,gather_heavy,high_memory,low_score |
| 3 | 285 | 23574.8 | 14.864 | 24684 | 550 | exact_preserve | connectivity_wall,documented_wall,exact_preserve,gather_heavy,high_memory,low_score,open_angle,scatter |
| 4 | 364 | 19396.5 | 15.115 | 18500 | 1131 | exact_preserve | connectivity_wall,exact_preserve,high_memory,local_stencil,open_angle,qlinear |
| 5 | 076 | 18064.0 | 15.107 | 19456 | 340 | exact_preserve | assignment_wall,connectivity_wall,conv_heavy,documented_wall,exact_preserve,high_memory,open_angle,scatter |
| 6 | 349 | 15204.2 | 15.289 | 15300 | 1191 | semantic_or_handbuilt | conv_heavy,documented_wall,high_memory,local_stencil,lut_selection,open_angle,qlinear |
| 7 | 101 | 14620.5 | 15.422 | 13573 | 874 | semantic_or_handbuilt | connectivity_wall,gather_heavy,high_memory,onehot_final_equal,scatter |
| 8 | 191 | 12733.9 | 15.520 | 12919 | 171 | exact_preserve | assignment_wall,conv_heavy,custom_win,exact_preserve,high_memory,open_angle,qlinear |
| 9 | 205 | 11856.6 | 15.588 | 11598 | 635 | exact_preserve | connectivity_wall,conv_heavy,exact_preserve,high_memory,open_angle,qlinear |
| 10 | 198 | 11720.6 | 15.485 | 13428 | 138 | exact_preserve | connectivity_wall,documented_wall,exact_preserve,gather_heavy,high_memory,open_angle,scan |
| 11 | 025 | 11273.3 | 15.636 | 11520 | 144 | exact_preserve | exact_preserve,high_memory,onehot_final_equal,open_angle,scan |
| 12 | 138 | 10811.3 | 15.552 | 12215 | 462 | exact_preserve | conv_heavy,documented_wall,exact_preserve,high_memory,local_stencil,open_angle |
| 13 | 066 | 10187.7 | 15.778 | 9955 | 166 | semantic_or_handbuilt | marker_routed_path,open_angle,scan |
| 14 | 379 | 9487.9 | 15.847 | 9309 | 133 | semantic_or_handbuilt | connectivity_wall,custom_win,open_angle |
| 15 | 338 | 9248.4 | 15.725 | 10000 | 666 | semantic_or_handbuilt | connectivity_wall,documented_wall,high_memory,local_stencil,open_angle,qlinear |
| 16 | 145 | 9069.5 | 15.742 | 9244 | 1248 | semantic_or_handbuilt | connectivity_wall,custom_win,documented_wall,lut_selection,open_angle |
| 17 | 080 | 8642.5 | 15.735 | 10109 | 454 | exact_preserve | conv_heavy,documented_wall,exact_preserve,high_memory,lut_selection,open_angle,qlinear |
| 18 | 074 | 8583.2 | 15.889 | 9000 | 50 | exact_preserve | exact_preserve,local_stencil,open_angle |
| 19 | 064 | 8548.7 | 15.791 | 9852 | 134 | semantic_or_handbuilt | connectivity_wall,documented_wall,onehot_final_equal,open_angle |
| 20 | 204 | 8302.0 | 15.767 | 9780 | 452 | exact_preserve | connectivity_wall,conv_heavy,custom_win,documented_wall,exact_preserve,local_stencil,maxpool_scan,open_angle |
| 21 | 222 | 7588.0 | 16.002 | 5136 | 2952 | exact_preserve | conv_heavy,custom_win,exact_preserve,local_stencil,maxpool_scan,onehot_final_equal,open_angle,qlinear |
| 22 | 017 | 7301.0 | 16.038 | 5997 | 1804 | exact_preserve | custom_win,exact_preserve,onehot_final_equal,open_angle |
| 23 | 096 | 6812.3 | 15.919 | 7987 | 801 | exact_preserve | documented_wall,exact_preserve,open_angle,scatter |
| 24 | 216 | 6679.9 | 15.934 | 8584 | 76 | exact_preserve | connectivity_wall,documented_wall,exact_preserve,open_angle,qlinear,scatter |
| 25 | 023 | 6412.0 | 16.234 | 6163 | 249 | semantic_or_handbuilt | conv_heavy,local_stencil,open_angle,qlinear |

## uint8_topk_compact_label_grid

Run TopK directly on compact uint8 label grids when only nonzero-cell enumeration is needed

- source tasks: 285 366
- candidates: 85
- expected: {'gain_type': 'memory', 'risk': 'medium because TopK tie ordering and dtype support must be checked in ORT', 'verification': 'stored eval plus large side-by-side incumbent equivalence; truth-fresh failure shared with incumbent is not by itself a reject'}

| rank | task | score | pts | mem | params | source | tags |
|---:|---:|---:|---:|---:|---:|---|---|
| 1 | 233 | 32165.5 | 14.588 | 32796 | 446 | semantic_or_handbuilt | assignment_wall,connectivity_wall,conv_heavy,documented_wall,gather_heavy,high_memory,low_score,matmul |
| 2 | 366 | 31466.9 | 14.640 | 30983 | 576 | exact_preserve | assignment_wall,connectivity_wall,exact_preserve,gather_heavy,high_memory,low_score,lut_selection,open_angle |
| 3 | 133 | 28025.0 | 14.703 | 27333 | 2303 | exact_preserve | assignment_wall,connectivity_wall,conv_heavy,documented_wall,exact_preserve,gather_heavy,high_memory,low_score |
| 4 | 285 | 23574.8 | 14.864 | 24684 | 550 | exact_preserve | connectivity_wall,documented_wall,exact_preserve,gather_heavy,high_memory,low_score,open_angle,scatter |
| 5 | 286 | 22796.4 | 14.915 | 21090 | 2881 | semantic_or_handbuilt | connectivity_wall,conv_heavy,documented_wall,high_memory,local_stencil,low_score,qlinear |
| 6 | 054 | 20655.7 | 15.078 | 20091 | 288 | unknown | gather_heavy,high_memory,scan,scatter |
| 7 | 364 | 19396.5 | 15.115 | 18500 | 1131 | exact_preserve | connectivity_wall,exact_preserve,high_memory,local_stencil,open_angle,qlinear |
| 8 | 367 | 18288.8 | 15.121 | 14800 | 4725 | semantic_or_handbuilt | connectivity_wall,conv_heavy,documented_wall,high_memory,local_stencil,lut_selection,open_angle,qlinear |
| 9 | 076 | 18064.0 | 15.107 | 19456 | 340 | exact_preserve | assignment_wall,connectivity_wall,conv_heavy,documented_wall,exact_preserve,high_memory,open_angle,scatter |
| 10 | 173 | 17379.5 | 15.222 | 17534 | 112 | exact_preserve | exact_preserve,gather_heavy,high_memory,scatter |
| 11 | 319 | 17004.7 | 15.161 | 18645 | 108 | exact_preserve | assignment_wall,documented_wall,exact_preserve,gather_heavy,high_memory,scan |
| 12 | 349 | 15204.2 | 15.289 | 15300 | 1191 | semantic_or_handbuilt | conv_heavy,documented_wall,high_memory,local_stencil,lut_selection,open_angle,qlinear |
| 13 | 255 | 14639.9 | 15.387 | 14663 | 293 | exact_preserve | conv_heavy,exact_preserve,high_memory,matmul,qlinear |
| 14 | 101 | 14620.5 | 15.422 | 13573 | 874 | semantic_or_handbuilt | connectivity_wall,gather_heavy,high_memory,onehot_final_equal,scatter |
| 15 | 191 | 12733.9 | 15.520 | 12919 | 171 | exact_preserve | assignment_wall,conv_heavy,custom_win,exact_preserve,high_memory,open_angle,qlinear |
| 16 | 205 | 11856.6 | 15.588 | 11598 | 635 | exact_preserve | connectivity_wall,conv_heavy,exact_preserve,high_memory,open_angle,qlinear |
| 17 | 198 | 11720.6 | 15.485 | 13428 | 138 | exact_preserve | connectivity_wall,documented_wall,exact_preserve,gather_heavy,high_memory,open_angle,scan |
| 18 | 025 | 11273.3 | 15.636 | 11520 | 144 | exact_preserve | exact_preserve,high_memory,onehot_final_equal,open_angle,scan |
| 19 | 138 | 10811.3 | 15.552 | 12215 | 462 | exact_preserve | conv_heavy,documented_wall,exact_preserve,high_memory,local_stencil,open_angle |
| 20 | 066 | 10187.7 | 15.778 | 9955 | 166 | semantic_or_handbuilt | marker_routed_path,open_angle,scan |
| 21 | 379 | 9487.9 | 15.847 | 9309 | 133 | semantic_or_handbuilt | connectivity_wall,custom_win,open_angle |
| 22 | 338 | 9248.4 | 15.725 | 10000 | 666 | semantic_or_handbuilt | connectivity_wall,documented_wall,high_memory,local_stencil,open_angle,qlinear |
| 23 | 370 | 9222.0 | 15.823 | 8645 | 1024 | exact_preserve | conv_heavy,exact_preserve,qlinear,scan,scatter |
| 24 | 044 | 9213.9 | 15.684 | 11050 | 69 | exact_preserve | assignment_wall,connectivity_wall,conv_heavy,documented_wall,exact_preserve,high_memory,matmul |
| 25 | 145 | 9069.5 | 15.742 | 9244 | 1248 | semantic_or_handbuilt | connectivity_wall,custom_win,documented_wall,lut_selection,open_angle |

## strided_conv_fixed_block_counts

Replace many fixed-block Slice+ReduceSum branches with one strided Conv

- source tasks: 011
- candidates: 181
- expected: {'gain_type': 'memory_when_repeated_block_reads_share_one_channel_or_kernel', 'risk': 'low-medium; Conv weight params can outweigh memory savings when few blocks are read or the input crop is already tiny', 'verification': 'stored eval plus fresh side-by-side; inspect Conv output shape because full 30x30 input may produce trailing off-grid stride positions that must be sliced away'}

| rank | task | score | pts | mem | params | source | tags |
|---:|---:|---:|---:|---:|---:|---|---|
| 1 | 366 | 31466.9 | 14.640 | 30983 | 576 | exact_preserve | assignment_wall,connectivity_wall,exact_preserve,gather_heavy,high_memory,low_score,lut_selection,open_angle |
| 2 | 133 | 28025.0 | 14.703 | 27333 | 2303 | exact_preserve | assignment_wall,connectivity_wall,conv_heavy,documented_wall,exact_preserve,gather_heavy,high_memory,low_score |
| 3 | 285 | 23574.8 | 14.864 | 24684 | 550 | exact_preserve | connectivity_wall,documented_wall,exact_preserve,gather_heavy,high_memory,low_score,open_angle,scatter |
| 4 | 364 | 19396.5 | 15.115 | 18500 | 1131 | exact_preserve | connectivity_wall,exact_preserve,high_memory,local_stencil,open_angle,qlinear |
| 5 | 367 | 18288.8 | 15.121 | 14800 | 4725 | semantic_or_handbuilt | connectivity_wall,conv_heavy,documented_wall,high_memory,local_stencil,lut_selection,open_angle,qlinear |
| 6 | 076 | 18064.0 | 15.107 | 19456 | 340 | exact_preserve | assignment_wall,connectivity_wall,conv_heavy,documented_wall,exact_preserve,high_memory,open_angle,scatter |
| 7 | 173 | 17379.5 | 15.222 | 17534 | 112 | exact_preserve | exact_preserve,gather_heavy,high_memory,scatter |
| 8 | 319 | 17004.7 | 15.161 | 18645 | 108 | exact_preserve | assignment_wall,documented_wall,exact_preserve,gather_heavy,high_memory,scan |
| 9 | 255 | 14639.9 | 15.387 | 14663 | 293 | exact_preserve | conv_heavy,exact_preserve,high_memory,matmul,qlinear |
| 10 | 101 | 14620.5 | 15.422 | 13573 | 874 | semantic_or_handbuilt | connectivity_wall,gather_heavy,high_memory,onehot_final_equal,scatter |
| 11 | 191 | 12733.9 | 15.520 | 12919 | 171 | exact_preserve | assignment_wall,conv_heavy,custom_win,exact_preserve,high_memory,open_angle,qlinear |
| 12 | 205 | 11856.6 | 15.588 | 11598 | 635 | exact_preserve | connectivity_wall,conv_heavy,exact_preserve,high_memory,open_angle,qlinear |
| 13 | 198 | 11720.6 | 15.485 | 13428 | 138 | exact_preserve | connectivity_wall,documented_wall,exact_preserve,gather_heavy,high_memory,open_angle,scan |
| 14 | 025 | 11273.3 | 15.636 | 11520 | 144 | exact_preserve | exact_preserve,high_memory,onehot_final_equal,open_angle,scan |
| 15 | 338 | 9248.4 | 15.725 | 10000 | 666 | semantic_or_handbuilt | connectivity_wall,documented_wall,high_memory,local_stencil,open_angle,qlinear |
| 16 | 370 | 9222.0 | 15.823 | 8645 | 1024 | exact_preserve | conv_heavy,exact_preserve,qlinear,scan,scatter |
| 17 | 044 | 9213.9 | 15.684 | 11050 | 69 | exact_preserve | assignment_wall,connectivity_wall,conv_heavy,documented_wall,exact_preserve,high_memory,matmul |
| 18 | 396 | 9207.6 | 15.825 | 7842 | 1813 | exact_preserve | conv_heavy,exact_preserve,local_stencil |
| 19 | 080 | 8642.5 | 15.735 | 10109 | 454 | exact_preserve | conv_heavy,documented_wall,exact_preserve,high_memory,lut_selection,open_angle,qlinear |
| 20 | 064 | 8548.7 | 15.791 | 9852 | 134 | semantic_or_handbuilt | connectivity_wall,documented_wall,onehot_final_equal,open_angle |
| 21 | 204 | 8302.0 | 15.767 | 9780 | 452 | exact_preserve | connectivity_wall,conv_heavy,custom_win,documented_wall,exact_preserve,local_stencil,maxpool_scan,open_angle |
| 22 | 377 | 7766.8 | 15.981 | 8111 | 150 | exact_preserve | exact_preserve,local_stencil |
| 23 | 017 | 7301.0 | 16.038 | 5997 | 1804 | exact_preserve | custom_win,exact_preserve,onehot_final_equal,open_angle |
| 24 | 324 | 6921.9 | 15.907 | 7308 | 1586 | exact_preserve | conv_heavy,documented_wall,exact_preserve,local_stencil,qlinear,scan |
| 25 | 096 | 6812.3 | 15.919 | 7987 | 801 | exact_preserve | documented_wall,exact_preserve,open_angle,scatter |

## solid_marker_profile_reconstruction

Replace full solid-marker channel slices with row/column profile contractions

- source tasks: 008
- candidates: 25
- expected: {'gain_type': 'memory_by_deleting_full_marker_plane', 'risk': 'low when the marker is generator-proven solid and separable; medium when the marker may touch borders or share rows/cols with other same-colour pixels', 'verification': 'stored eval plus fresh generation; sample generator border cases because flip/xpose can invalidate assumed crop margins'}

| rank | task | score | pts | mem | params | source | tags |
|---:|---:|---:|---:|---:|---:|---|---|
| 1 | 101 | 14620.5 | 15.422 | 13573 | 874 | semantic_or_handbuilt | connectivity_wall,gather_heavy,high_memory,onehot_final_equal,scatter |
| 2 | 025 | 11273.3 | 15.636 | 11520 | 144 | exact_preserve | exact_preserve,high_memory,onehot_final_equal,open_angle,scan |
| 3 | 064 | 8548.7 | 15.791 | 9852 | 134 | semantic_or_handbuilt | connectivity_wall,documented_wall,onehot_final_equal,open_angle |
| 4 | 222 | 7588.0 | 16.002 | 5136 | 2952 | exact_preserve | conv_heavy,custom_win,exact_preserve,local_stencil,maxpool_scan,onehot_final_equal,open_angle,qlinear |
| 5 | 017 | 7301.0 | 16.038 | 5997 | 1804 | exact_preserve | custom_win,exact_preserve,onehot_final_equal,open_angle |
| 6 | 368 | 5725.0 | 16.264 | 6022 | 203 | exact_preserve | connectivity_wall,exact_preserve,gather_heavy,onehot_final_equal,scatter |
| 7 | 219 | 5710.0 | 16.117 | 7078 | 132 | unknown | assignment_wall,connectivity_wall,documented_wall,lut_selection,onehot_final_equal,open_angle,scan |
| 8 | 009 | 5024.0 | 16.143 | 6929 | 95 | exact_preserve | connectivity_wall,documented_wall,exact_preserve,local_stencil,lut_selection,onehot_final_equal,open_angle |
| 9 | 182 | 4942.0 | 16.229 | 6345 | 97 | semantic_or_handbuilt | assignment_wall,connectivity_wall,conv_heavy,documented_wall,local_stencil,lut_selection,onehot_final_equal,open_angle |
| 10 | 383 | 4570.0 | 16.289 | 5974 | 96 | unknown | assignment_wall,connectivity_wall,documented_wall,local_stencil,onehot_final_equal,open_angle |
| 11 | 174 | 4389.0 | 16.319 | 5743 | 146 | semantic_or_handbuilt | connectivity_wall,documented_wall,matmul,onehot_final_equal,open_angle |
| 12 | 208 | 4226.0 | 16.539 | 4612 | 114 | exact_preserve | exact_preserve,onehot_final_equal,open_angle,qlinear |
| 13 | 363 | 3635.0 | 16.673 | 3839 | 296 | exact_preserve | connectivity_wall,conv_heavy,exact_preserve,local_stencil,onehot_final_equal,open_angle |
| 14 | 107 | 3578.0 | 16.687 | 2924 | 1154 | exact_preserve | exact_preserve,lut_selection,matmul,onehot_final_equal,open_angle,qlinear,scatter |
| 15 | 202 | 3287.0 | 16.902 | 3253 | 34 | unknown | connectivity_wall,onehot_final_equal,open_angle,qlinear |
| 16 | 234 | 3208.0 | 16.927 | 3144 | 64 | semantic_or_handbuilt | connectivity_wall,onehot_final_equal |
| 17 | 177 | 3121.0 | 16.805 | 3500 | 121 | exact_preserve | connectivity_wall,conv_heavy,exact_preserve,local_stencil,onehot_final_equal,open_angle |
| 18 | 036 | 1677.0 | 17.314 | 2140 | 37 | exact_preserve | connectivity_wall,exact_preserve,onehot_final_equal,open_angle |
| 19 | 392 | 1451.0 | 17.424 | 1602 | 349 | exact_preserve | custom_win,exact_preserve,onehot_final_equal,open_angle |
| 20 | 213 | 1382.0 | 17.460 | 1833 | 49 | exact_preserve | exact_preserve,onehot_final_equal,open_angle |
| 21 | 333 | 1227.0 | 16.921 | 2792 | 435 | exact_preserve | connectivity_wall,documented_wall,exact_preserve,onehot_final_equal,open_angle |
| 22 | 308 | 988.0 | 17.695 | 1385 | 103 | exact_preserve | connectivity_wall,exact_preserve,onehot_final_equal,open_angle,scatter |
| 23 | 071 | 698.0 | 17.100 | 2643 | 55 | exact_preserve | connectivity_wall,documented_wall,exact_preserve,onehot_final_equal,open_angle |
| 24 | 170 | 545.0 | 17.158 | 2216 | 329 | exact_preserve | assignment_wall,conv_heavy,documented_wall,exact_preserve,local_stencil,lut_selection,onehot_final_equal,open_angle |
| 25 | 029 | -34975.0 | 16.383 | 5436 | 89 | exact_preserve | exact_preserve,marginal_wall,onehot_final_equal,open_angle |

## walk_einsum_iteration_collapse

Collapse iterative flood/scan/propagation into ONE multi-operand Einsum via scaled walk counting (internal computation uncounted)

- source tasks: 187 313 110 243 077
- candidates: 31
- expected: {'gain_type': 'memory_by_deleting_all_iteration_planes', 'risk': 'medium: verify connectivity model (4 vs 8-conn!) matches incumbent semantics on fresh; verify step-count covers measured max BFS distance + margin; keep operands nonnegative or prove no cancellation. REJECT-CHECKS (task204 S8): parallel fan-out MaxPools (per-size anchor dilations) are NOT collapsible chains; uint8 conv banks of sub-400B planes are einsum-proof (fp32 entry ticket 1600-3600B + step params loses)', 'verification': 'numpy walk-vs-BFS on 20000 fresh, stored eval, fresh_verify candidate<=incumbent, A/B submission for grader safety on first use'}

| rank | task | score | pts | mem | params | source | tags |
|---:|---:|---:|---:|---:|---:|---|---|
| 1 | 233 | 32165.5 | 14.588 | 32796 | 446 | semantic_or_handbuilt | assignment_wall,connectivity_wall,conv_heavy,documented_wall,gather_heavy,high_memory,low_score,matmul |
| 2 | 366 | 31466.9 | 14.640 | 30983 | 576 | exact_preserve | assignment_wall,connectivity_wall,exact_preserve,gather_heavy,high_memory,low_score,lut_selection,open_angle |
| 3 | 133 | 28025.0 | 14.703 | 27333 | 2303 | exact_preserve | assignment_wall,connectivity_wall,conv_heavy,documented_wall,exact_preserve,gather_heavy,high_memory,low_score |
| 4 | 285 | 23574.8 | 14.864 | 24684 | 550 | exact_preserve | connectivity_wall,documented_wall,exact_preserve,gather_heavy,high_memory,low_score,open_angle,scatter |
| 5 | 054 | 20655.7 | 15.078 | 20091 | 288 | unknown | gather_heavy,high_memory,scan,scatter |
| 6 | 076 | 18064.0 | 15.107 | 19456 | 340 | exact_preserve | assignment_wall,connectivity_wall,conv_heavy,documented_wall,exact_preserve,high_memory,open_angle,scatter |
| 7 | 173 | 17379.5 | 15.222 | 17534 | 112 | exact_preserve | exact_preserve,gather_heavy,high_memory,scatter |
| 8 | 319 | 17004.7 | 15.161 | 18645 | 108 | exact_preserve | assignment_wall,documented_wall,exact_preserve,gather_heavy,high_memory,scan |
| 9 | 349 | 15204.2 | 15.289 | 15300 | 1191 | semantic_or_handbuilt | conv_heavy,documented_wall,high_memory,local_stencil,lut_selection,open_angle,qlinear |
| 10 | 255 | 14639.9 | 15.387 | 14663 | 293 | exact_preserve | conv_heavy,exact_preserve,high_memory,matmul,qlinear |
| 11 | 101 | 14620.5 | 15.422 | 13573 | 874 | semantic_or_handbuilt | connectivity_wall,gather_heavy,high_memory,onehot_final_equal,scatter |
| 12 | 191 | 12733.9 | 15.520 | 12919 | 171 | exact_preserve | assignment_wall,conv_heavy,custom_win,exact_preserve,high_memory,open_angle,qlinear |
| 13 | 205 | 11856.6 | 15.588 | 11598 | 635 | exact_preserve | connectivity_wall,conv_heavy,exact_preserve,high_memory,open_angle,qlinear |
| 14 | 198 | 11720.6 | 15.485 | 13428 | 138 | exact_preserve | connectivity_wall,documented_wall,exact_preserve,gather_heavy,high_memory,open_angle,scan |
| 15 | 025 | 11273.3 | 15.636 | 11520 | 144 | exact_preserve | exact_preserve,high_memory,onehot_final_equal,open_angle,scan |
| 16 | 138 | 10811.3 | 15.552 | 12215 | 462 | exact_preserve | conv_heavy,documented_wall,exact_preserve,high_memory,local_stencil,open_angle |
| 17 | 066 | 10187.7 | 15.778 | 9955 | 166 | semantic_or_handbuilt | marker_routed_path,open_angle,scan |
| 18 | 379 | 9487.9 | 15.847 | 9309 | 133 | semantic_or_handbuilt | connectivity_wall,custom_win,open_angle |
| 19 | 338 | 9248.4 | 15.725 | 10000 | 666 | semantic_or_handbuilt | connectivity_wall,documented_wall,high_memory,local_stencil,open_angle,qlinear |
| 20 | 370 | 9222.0 | 15.823 | 8645 | 1024 | exact_preserve | conv_heavy,exact_preserve,qlinear,scan,scatter |
| 21 | 044 | 9213.9 | 15.684 | 11050 | 69 | exact_preserve | assignment_wall,connectivity_wall,conv_heavy,documented_wall,exact_preserve,high_memory,matmul |
| 22 | 145 | 9069.5 | 15.742 | 9244 | 1248 | semantic_or_handbuilt | connectivity_wall,custom_win,documented_wall,lut_selection,open_angle |
| 23 | 080 | 8642.5 | 15.735 | 10109 | 454 | exact_preserve | conv_heavy,documented_wall,exact_preserve,high_memory,lut_selection,open_angle,qlinear |
| 24 | 064 | 8548.7 | 15.791 | 9852 | 134 | semantic_or_handbuilt | connectivity_wall,documented_wall,onehot_final_equal,open_angle |
| 25 | 204 | 8302.0 | 15.767 | 9780 | 452 | exact_preserve | connectivity_wall,conv_heavy,custom_win,documented_wall,exact_preserve,local_stencil,maxpool_scan,open_angle |
