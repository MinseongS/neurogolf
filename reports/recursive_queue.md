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
| 3 | 328 | 6275.0 | 16.179 | 4835 | 1940 | exact_preserve | exact_preserve,lut_selection |
| 4 | 279 | 5546.0 | 16.379 | 5508 | 38 | semantic_or_handbuilt | conv_heavy,local_stencil,lut_selection,qlinear |
| 5 | 396 | 4444.0 | 16.494 | 4854 | 90 | exact_preserve | conv_heavy,exact_preserve,local_stencil,onehot_final_equal |
| 6 | 008 | 4413.0 | 16.500 | 4809 | 104 | exact_preserve | exact_preserve,lut_selection |
| 7 | 208 | 4226.0 | 16.539 | 4612 | 114 | exact_preserve | exact_preserve,onehot_final_equal,open_angle,qlinear |
| 8 | 117 | 3665.0 | 16.666 | 3922 | 243 | exact_preserve | exact_preserve,local_stencil,lut_selection,open_angle,qlinear |
| 9 | 280 | 2646.0 | 16.670 | 3363 | 783 | semantic_or_handbuilt | custom_win,documented_wall,lut_selection |
| 10 | 091 | 2528.0 | 16.984 | 2853 | 175 | exact_preserve | exact_preserve,local_stencil,lut_selection,open_angle |
| 11 | 029 | -35201.0 | 16.425 | 5212 | 87 | exact_preserve | exact_preserve,marginal_wall,onehot_final_equal,open_angle |

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
| 4 | 191 | 11496.8 | 15.617 | 11741 | 141 | exact_preserve | assignment_wall,conv_heavy,custom_win,exact_preserve,high_memory,open_angle,qlinear |
| 5 | 138 | 10811.3 | 15.552 | 12215 | 462 | exact_preserve | conv_heavy,documented_wall,exact_preserve,high_memory,local_stencil,open_angle |
| 6 | 066 | 10187.7 | 15.778 | 9955 | 166 | semantic_or_handbuilt | marker_routed_path,open_angle,scan |
| 7 | 025 | 9999.6 | 15.748 | 10320 | 104 | exact_preserve | exact_preserve,high_memory,onehot_final_equal,open_angle,scan |
| 8 | 080 | 8642.5 | 15.735 | 10109 | 454 | exact_preserve | conv_heavy,documented_wall,exact_preserve,high_memory,lut_selection,open_angle,qlinear |
| 9 | 370 | 8587.8 | 15.944 | 7477 | 1094 | semantic_or_handbuilt | qlinear,scatter |
| 10 | 074 | 8583.2 | 15.889 | 9000 | 50 | exact_preserve | exact_preserve,local_stencil,open_angle |
| 11 | 377 | 7712.9 | 15.987 | 8075 | 134 | exact_preserve | exact_preserve,local_stencil |
| 12 | 324 | 6921.9 | 15.907 | 7308 | 1586 | exact_preserve | assignment_wall,conv_heavy,documented_wall,exact_preserve,local_stencil,qlinear,scan |
| 13 | 017 | 6504.0 | 16.146 | 5192 | 1812 | exact_preserve | custom_win,exact_preserve,matmul,onehot_final_equal,open_angle |
| 14 | 023 | 6412.0 | 16.234 | 6163 | 249 | semantic_or_handbuilt | conv_heavy,local_stencil,open_angle,qlinear |
| 15 | 096 | 6380.8 | 15.967 | 7953 | 418 | exact_preserve | documented_wall,exact_preserve,open_angle,scatter |
| 16 | 328 | 6275.0 | 16.179 | 4835 | 1940 | exact_preserve | exact_preserve,lut_selection |
| 17 | 222 | 5703.0 | 16.267 | 5992 | 211 | exact_preserve | custom_win,exact_preserve,local_stencil,maxpool_scan,open_angle,qlinear |
| 18 | 279 | 5546.0 | 16.379 | 5508 | 38 | semantic_or_handbuilt | conv_heavy,local_stencil,lut_selection,qlinear |
| 19 | 278 | 5483.0 | 16.391 | 5436 | 47 | semantic_or_handbuilt | local_stencil,qlinear |
| 20 | 382 | 5241.0 | 16.345 | 5606 | 135 | exact_preserve | exact_preserve,open_angle,scan |
| 21 | 192 | 4996.0 | 16.147 | 5745 | 1251 | exact_preserve | conv_heavy,documented_wall,exact_preserve,local_stencil,open_angle,qlinear |
| 22 | 264 | 4908.0 | 16.404 | 4700 | 708 | exact_preserve | exact_preserve,local_stencil,lut_selection,qlinear |
| 23 | 085 | 4881.0 | 16.409 | 4500 | 881 | exact_preserve | exact_preserve,local_stencil,open_angle |
| 24 | 005 | 4801.0 | 16.175 | 6311 | 490 | exact_preserve | conv_heavy,documented_wall,exact_preserve,lut_selection,qlinear |
| 25 | 396 | 4444.0 | 16.494 | 4854 | 90 | exact_preserve | conv_heavy,exact_preserve,local_stencil,onehot_final_equal |

## threshold_linearize_pairwise_onehot_and

Replace pairwise one-hot AND/Kronecker products with direct thresholded product scores

- source tasks: 001
- candidates: 69
- expected: {'gain_type': 'memory_by_deleting_boolean_product_carrier', 'risk': 'medium; relies on one-hot inputs, positive-threshold scoring, and class-specific margins', 'verification': 'stored eval plus fresh/adopt when generator samples exist; inspect off-footprint scores are non-positive'}

| rank | task | score | pts | mem | params | source | tags |
|---:|---:|---:|---:|---:|---:|---|---|
| 1 | 366 | 31466.9 | 14.640 | 30983 | 576 | exact_preserve | assignment_wall,connectivity_wall,exact_preserve,gather_heavy,high_memory,low_score,lut_selection,open_angle |
| 2 | 133 | 28025.0 | 14.703 | 27333 | 2303 | exact_preserve | assignment_wall,connectivity_wall,conv_heavy,documented_wall,exact_preserve,gather_heavy,high_memory,low_score |
| 3 | 367 | 18288.8 | 15.121 | 14800 | 4725 | semantic_or_handbuilt | connectivity_wall,conv_heavy,documented_wall,high_memory,local_stencil,lut_selection,open_angle,qlinear |
| 4 | 349 | 15204.2 | 15.289 | 15300 | 1191 | semantic_or_handbuilt | conv_heavy,documented_wall,high_memory,local_stencil,lut_selection,open_angle,qlinear |
| 5 | 101 | 14620.5 | 15.422 | 13573 | 874 | semantic_or_handbuilt | assignment_wall,connectivity_wall,gather_heavy,high_memory,onehot_final_equal,scatter |
| 6 | 025 | 9999.6 | 15.748 | 10320 | 104 | exact_preserve | exact_preserve,high_memory,onehot_final_equal,open_angle,scan |
| 7 | 145 | 9069.5 | 15.742 | 9244 | 1248 | semantic_or_handbuilt | connectivity_wall,custom_win,documented_wall,lut_selection,open_angle |
| 8 | 080 | 8642.5 | 15.735 | 10109 | 454 | exact_preserve | conv_heavy,documented_wall,exact_preserve,high_memory,lut_selection,open_angle,qlinear |
| 9 | 064 | 8548.7 | 15.791 | 9852 | 134 | semantic_or_handbuilt | connectivity_wall,documented_wall,onehot_final_equal,open_angle |
| 10 | 017 | 6504.0 | 16.146 | 5192 | 1812 | exact_preserve | custom_win,exact_preserve,matmul,onehot_final_equal,open_angle |
| 11 | 187 | 6330.0 | 15.973 | 5000 | 3322 | exact_preserve | connectivity_wall,documented_wall,exact_preserve,local_stencil,lut_selection |
| 12 | 328 | 6275.0 | 16.179 | 4835 | 1940 | exact_preserve | exact_preserve,lut_selection |
| 13 | 219 | 5710.0 | 16.117 | 7078 | 132 | unknown | assignment_wall,connectivity_wall,documented_wall,lut_selection,onehot_final_equal,open_angle,scan |
| 14 | 368 | 5693.0 | 16.269 | 5990 | 203 | exact_preserve | connectivity_wall,exact_preserve,gather_heavy,onehot_final_equal,scatter |
| 15 | 279 | 5546.0 | 16.379 | 5508 | 38 | semantic_or_handbuilt | conv_heavy,local_stencil,lut_selection,qlinear |
| 16 | 157 | 5243.0 | 16.112 | 6975 | 268 | exact_preserve | assignment_wall,connectivity_wall,documented_wall,exact_preserve,gather_heavy,lut_selection,open_angle,scatter |
| 17 | 002 | 5189.0 | 16.192 | 6100 | 589 | semantic_or_handbuilt | connectivity_wall,documented_wall,lut_selection |
| 18 | 009 | 5024.0 | 16.143 | 6929 | 95 | exact_preserve | connectivity_wall,documented_wall,exact_preserve,local_stencil,lut_selection,onehot_final_equal,open_angle |
| 19 | 182 | 4942.0 | 16.229 | 6345 | 97 | semantic_or_handbuilt | assignment_wall,connectivity_wall,conv_heavy,documented_wall,local_stencil,lut_selection,onehot_final_equal,open_angle |
| 20 | 264 | 4908.0 | 16.404 | 4700 | 708 | exact_preserve | exact_preserve,local_stencil,lut_selection,qlinear |
| 21 | 005 | 4801.0 | 16.175 | 6311 | 490 | exact_preserve | conv_heavy,documented_wall,exact_preserve,lut_selection,qlinear |
| 22 | 383 | 4570.0 | 16.289 | 5974 | 96 | unknown | assignment_wall,connectivity_wall,documented_wall,local_stencil,onehot_final_equal,open_angle |
| 23 | 396 | 4444.0 | 16.494 | 4854 | 90 | exact_preserve | conv_heavy,exact_preserve,local_stencil,onehot_final_equal |
| 24 | 008 | 4413.0 | 16.500 | 4809 | 104 | exact_preserve | exact_preserve,lut_selection |
| 25 | 208 | 4226.0 | 16.539 | 4612 | 114 | exact_preserve | exact_preserve,onehot_final_equal,open_angle,qlinear |

## qlinear_uint8_lut_or_matmul

Replace fp16/fp32 one-hot LUT/MatMul selection with uint8 QLinearMatMul/QLinearConv where exact

- source tasks: 055 080 338
- candidates: 0
- expected: {'gain_type': 'memory', 'risk': 'medium', 'verification': 'stored eval, fresh eval if generator available, then adopt gate'}

| rank | task | score | pts | mem | params | source | tags |
|---:|---:|---:|---:|---:|---:|---|---|

## free_final_onehot_equal

Delay 10-channel expansion to final Equal/Where output

- source tasks: 017 095 146 381
- candidates: 25
- expected: {'gain_type': 'memory', 'risk': 'low-medium', 'verification': 'output equivalence before adoption'}

| rank | task | score | pts | mem | params | source | tags |
|---:|---:|---:|---:|---:|---:|---|---|
| 1 | 101 | 14620.5 | 15.422 | 13573 | 874 | semantic_or_handbuilt | assignment_wall,connectivity_wall,gather_heavy,high_memory,onehot_final_equal,scatter |
| 2 | 025 | 9999.6 | 15.748 | 10320 | 104 | exact_preserve | exact_preserve,high_memory,onehot_final_equal,open_angle,scan |
| 3 | 064 | 8548.7 | 15.791 | 9852 | 134 | semantic_or_handbuilt | connectivity_wall,documented_wall,onehot_final_equal,open_angle |
| 4 | 017 | 6504.0 | 16.146 | 5192 | 1812 | exact_preserve | custom_win,exact_preserve,matmul,onehot_final_equal,open_angle |
| 5 | 219 | 5710.0 | 16.117 | 7078 | 132 | unknown | assignment_wall,connectivity_wall,documented_wall,lut_selection,onehot_final_equal,open_angle,scan |
| 6 | 368 | 5693.0 | 16.269 | 5990 | 203 | exact_preserve | connectivity_wall,exact_preserve,gather_heavy,onehot_final_equal,scatter |
| 7 | 009 | 5024.0 | 16.143 | 6929 | 95 | exact_preserve | connectivity_wall,documented_wall,exact_preserve,local_stencil,lut_selection,onehot_final_equal,open_angle |
| 8 | 182 | 4942.0 | 16.229 | 6345 | 97 | semantic_or_handbuilt | assignment_wall,connectivity_wall,conv_heavy,documented_wall,local_stencil,lut_selection,onehot_final_equal,open_angle |
| 9 | 383 | 4570.0 | 16.289 | 5974 | 96 | unknown | assignment_wall,connectivity_wall,documented_wall,local_stencil,onehot_final_equal,open_angle |
| 10 | 396 | 4444.0 | 16.494 | 4854 | 90 | exact_preserve | conv_heavy,exact_preserve,local_stencil,onehot_final_equal |
| 11 | 208 | 4226.0 | 16.539 | 4612 | 114 | exact_preserve | exact_preserve,onehot_final_equal,open_angle,qlinear |
| 12 | 363 | 3635.0 | 16.673 | 3839 | 296 | exact_preserve | connectivity_wall,conv_heavy,exact_preserve,local_stencil,onehot_final_equal,open_angle |
| 13 | 107 | 3313.0 | 16.754 | 3242 | 571 | exact_preserve | exact_preserve,lut_selection,matmul,onehot_final_equal,open_angle,qlinear,scatter |
| 14 | 202 | 3287.0 | 16.902 | 3253 | 34 | unknown | connectivity_wall,onehot_final_equal,open_angle,qlinear |
| 15 | 177 | 3121.0 | 16.805 | 3500 | 121 | exact_preserve | connectivity_wall,conv_heavy,exact_preserve,local_stencil,onehot_final_equal,open_angle |
| 16 | 234 | 2950.0 | 17.010 | 2886 | 64 | semantic_or_handbuilt | connectivity_wall,onehot_final_equal |
| 17 | 036 | 1677.0 | 17.314 | 2140 | 37 | exact_preserve | connectivity_wall,exact_preserve,onehot_final_equal,open_angle |
| 18 | 392 | 1451.0 | 17.424 | 1602 | 349 | exact_preserve | custom_win,exact_preserve,onehot_final_equal,open_angle |
| 19 | 213 | 1358.0 | 17.473 | 1809 | 49 | exact_preserve | exact_preserve,onehot_final_equal,open_angle |
| 20 | 333 | 1227.0 | 16.921 | 2792 | 435 | exact_preserve | connectivity_wall,documented_wall,exact_preserve,onehot_final_equal,open_angle |
| 21 | 308 | 988.0 | 17.695 | 1385 | 103 | exact_preserve | connectivity_wall,exact_preserve,onehot_final_equal,open_angle,scatter |
| 22 | 071 | 698.0 | 17.100 | 2643 | 55 | exact_preserve | connectivity_wall,documented_wall,exact_preserve,onehot_final_equal,open_angle |
| 23 | 170 | 545.0 | 17.158 | 2216 | 329 | exact_preserve | assignment_wall,conv_heavy,documented_wall,exact_preserve,local_stencil,lut_selection,onehot_final_equal,open_angle |
| 24 | 184 | 160.0 | 17.322 | 2100 | 60 | exact_preserve | connectivity_wall,conv_heavy,documented_wall,exact_preserve,local_stencil,onehot_final_equal,open_angle,qlinear |
| 25 | 029 | -35201.0 | 16.425 | 5212 | 87 | exact_preserve | exact_preserve,marginal_wall,onehot_final_equal,open_angle |

## scan_dtype_and_shift_compression

Compress scan-style MaxPool/CumSum/Hillis-Steele pipelines with lower dtype or shared shifts

- source tasks: 046 216
- candidates: 29
- expected: {'gain_type': 'memory', 'risk': 'medium-high', 'verification': 'stored/fresh eval and compare against incumbent'}

| rank | task | score | pts | mem | params | source | tags |
|---:|---:|---:|---:|---:|---:|---|---|
| 1 | 233 | 32165.5 | 14.588 | 32796 | 446 | semantic_or_handbuilt | assignment_wall,connectivity_wall,conv_heavy,documented_wall,gather_heavy,high_memory,low_score,matmul |
| 2 | 366 | 31466.9 | 14.640 | 30983 | 576 | exact_preserve | assignment_wall,connectivity_wall,exact_preserve,gather_heavy,high_memory,low_score,lut_selection,open_angle |
| 3 | 133 | 28025.0 | 14.703 | 27333 | 2303 | exact_preserve | assignment_wall,connectivity_wall,conv_heavy,documented_wall,exact_preserve,gather_heavy,high_memory,low_score |
| 4 | 285 | 23574.8 | 14.864 | 24684 | 550 | exact_preserve | connectivity_wall,documented_wall,exact_preserve,gather_heavy,high_memory,low_score,open_angle,scatter |
| 5 | 054 | 20655.7 | 15.078 | 20091 | 288 | unknown | gather_heavy,high_memory,scan,scatter |
| 6 | 173 | 17379.5 | 15.222 | 17534 | 112 | exact_preserve | exact_preserve,gather_heavy,high_memory,scatter |
| 7 | 319 | 15367.8 | 15.251 | 16090 | 1053 | exact_preserve | assignment_wall,documented_wall,exact_preserve,gather_heavy,high_memory,scan |
| 8 | 198 | 10943.3 | 15.542 | 12670 | 136 | exact_preserve | connectivity_wall,documented_wall,exact_preserve,gather_heavy,high_memory,open_angle,scan |
| 9 | 066 | 10187.7 | 15.778 | 9955 | 166 | semantic_or_handbuilt | marker_routed_path,open_angle,scan |
| 10 | 025 | 9999.6 | 15.748 | 10320 | 104 | exact_preserve | exact_preserve,high_memory,onehot_final_equal,open_angle,scan |
| 11 | 204 | 8302.0 | 15.767 | 9780 | 452 | exact_preserve | connectivity_wall,conv_heavy,custom_win,documented_wall,exact_preserve,local_stencil,maxpool_scan,open_angle |
| 12 | 044 | 7161.7 | 15.881 | 8972 | 154 | exact_preserve | assignment_wall,connectivity_wall,documented_wall,exact_preserve,scan |
| 13 | 324 | 6921.9 | 15.907 | 7308 | 1586 | exact_preserve | assignment_wall,conv_heavy,documented_wall,exact_preserve,local_stencil,qlinear,scan |
| 14 | 219 | 5710.0 | 16.117 | 7078 | 132 | unknown | assignment_wall,connectivity_wall,documented_wall,lut_selection,onehot_final_equal,open_angle,scan |
| 15 | 222 | 5703.0 | 16.267 | 5992 | 211 | exact_preserve | custom_win,exact_preserve,local_stencil,maxpool_scan,open_angle,qlinear |
| 16 | 157 | 5243.0 | 16.112 | 6975 | 268 | exact_preserve | assignment_wall,connectivity_wall,documented_wall,exact_preserve,gather_heavy,lut_selection,open_angle,scatter |
| 17 | 382 | 5241.0 | 16.345 | 5606 | 135 | exact_preserve | exact_preserve,open_angle,scan |
| 18 | 277 | 3245.0 | 16.772 | 3701 | 44 | exact_preserve | connectivity_wall,exact_preserve,lut_selection,maxpool_scan |
| 19 | 055 | 3092.0 | 16.963 | 3030 | 62 | semantic_or_handbuilt | connectivity_wall,lut_selection,open_angle,qlinear,scan |
| 20 | 090 | 2572.0 | 16.970 | 2990 | 82 | exact_preserve | exact_preserve,scan |
| 21 | 174 | 2563.0 | 16.574 | 4166 | 397 | exact_preserve | connectivity_wall,documented_wall,exact_preserve,matmul,open_angle,scan |
| 22 | 196 | 2538.0 | 16.580 | 4500 | 38 | exact_preserve | connectivity_wall,custom_win,documented_wall,exact_preserve,local_stencil,maxpool_scan,open_angle,qlinear |
| 23 | 273 | 1660.0 | 17.322 | 2109 | 51 | exact_preserve | connectivity_wall,exact_preserve,gather_heavy,open_angle |
| 24 | 125 | 1463.0 | 16.850 | 3434 | 29 | exact_preserve | connectivity_wall,documented_wall,exact_preserve,maxpool_scan,open_angle |
| 25 | 046 | 285.0 | 17.266 | 2189 | 96 | exact_preserve | connectivity_wall,documented_wall,exact_preserve,local_stencil,scan |

## sparse_conv_single_op_floor

Collapse local neighborhood rules to one sparse Conv/QLinearConv when output is thresholded

- source tasks: 015 095 230 294
- candidates: 62
- expected: {'gain_type': 'memory+params', 'risk': 'low when rule is truly local', 'verification': 'fresh process eval; beware one-process mem0 false signals'}

| rank | task | score | pts | mem | params | source | tags |
|---:|---:|---:|---:|---:|---:|---|---|
| 1 | 349 | 15204.2 | 15.289 | 15300 | 1191 | semantic_or_handbuilt | conv_heavy,documented_wall,high_memory,local_stencil,lut_selection,open_angle,qlinear |
| 2 | 138 | 10811.3 | 15.552 | 12215 | 462 | exact_preserve | conv_heavy,documented_wall,exact_preserve,high_memory,local_stencil,open_angle |
| 3 | 255 | 10395.4 | 15.712 | 10450 | 359 | exact_preserve | conv_heavy,exact_preserve,high_memory,matmul,qlinear |
| 4 | 080 | 8642.5 | 15.735 | 10109 | 454 | exact_preserve | conv_heavy,documented_wall,exact_preserve,high_memory,lut_selection,open_angle,qlinear |
| 5 | 074 | 8583.2 | 15.889 | 9000 | 50 | exact_preserve | exact_preserve,local_stencil,open_angle |
| 6 | 377 | 7712.9 | 15.987 | 8075 | 134 | exact_preserve | exact_preserve,local_stencil |
| 7 | 023 | 6412.0 | 16.234 | 6163 | 249 | semantic_or_handbuilt | conv_heavy,local_stencil,open_angle,qlinear |
| 8 | 222 | 5703.0 | 16.267 | 5992 | 211 | exact_preserve | custom_win,exact_preserve,local_stencil,maxpool_scan,open_angle,qlinear |
| 9 | 279 | 5546.0 | 16.379 | 5508 | 38 | semantic_or_handbuilt | conv_heavy,local_stencil,lut_selection,qlinear |
| 10 | 278 | 5483.0 | 16.391 | 5436 | 47 | semantic_or_handbuilt | local_stencil,qlinear |
| 11 | 192 | 4996.0 | 16.147 | 5745 | 1251 | exact_preserve | conv_heavy,documented_wall,exact_preserve,local_stencil,open_angle,qlinear |
| 12 | 264 | 4908.0 | 16.404 | 4700 | 708 | exact_preserve | exact_preserve,local_stencil,lut_selection,qlinear |
| 13 | 085 | 4881.0 | 16.409 | 4500 | 881 | exact_preserve | exact_preserve,local_stencil,open_angle |
| 14 | 005 | 4801.0 | 16.175 | 6311 | 490 | exact_preserve | conv_heavy,documented_wall,exact_preserve,lut_selection,qlinear |
| 15 | 396 | 4444.0 | 16.494 | 4854 | 90 | exact_preserve | conv_heavy,exact_preserve,local_stencil,onehot_final_equal |
| 16 | 117 | 3665.0 | 16.666 | 3922 | 243 | exact_preserve | exact_preserve,local_stencil,lut_selection,open_angle,qlinear |
| 17 | 132 | 3589.0 | 16.684 | 3990 | 99 | exact_preserve | bitwise_program,exact_preserve,gather_heavy,local_stencil,open_angle |
| 18 | 162 | 3568.0 | 16.689 | 4024 | 44 | exact_preserve | exact_preserve,local_stencil,open_angle,qlinear |
| 19 | 284 | 3149.0 | 16.798 | 2996 | 653 | exact_preserve | conv_heavy,custom_win,exact_preserve,open_angle,scatter |
| 20 | 354 | 2751.0 | 17.080 | 2674 | 77 | semantic_or_handbuilt | conv_heavy,local_stencil,open_angle |
| 21 | 079 | 2643.0 | 16.947 | 3065 | 78 | exact_preserve | custom_win,exact_preserve,local_stencil,open_angle |
| 22 | 042 | 2610.0 | 16.958 | 2792 | 318 | exact_preserve | conv_heavy,exact_preserve,local_stencil,qlinear |
| 23 | 091 | 2528.0 | 16.984 | 2853 | 175 | exact_preserve | exact_preserve,local_stencil,lut_selection,open_angle |
| 24 | 224 | 2331.0 | 17.052 | 2460 | 371 | exact_preserve | conv_heavy,exact_preserve,local_stencil,open_angle |
| 25 | 019 | 2298.0 | 17.063 | 2709 | 89 | exact_preserve | exact_preserve,local_stencil,qlinear |

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
| 5 | 173 | 17379.5 | 15.222 | 17534 | 112 | exact_preserve | exact_preserve,gather_heavy,high_memory,scatter |
| 6 | 319 | 15367.8 | 15.251 | 16090 | 1053 | exact_preserve | assignment_wall,documented_wall,exact_preserve,gather_heavy,high_memory,scan |
| 7 | 076 | 14771.4 | 15.285 | 16254 | 303 | exact_preserve | assignment_wall,connectivity_wall,documented_wall,exact_preserve,high_memory,open_angle,scatter |
| 8 | 191 | 11496.8 | 15.617 | 11741 | 141 | exact_preserve | assignment_wall,conv_heavy,custom_win,exact_preserve,high_memory,open_angle,qlinear |
| 9 | 198 | 10943.3 | 15.542 | 12670 | 136 | exact_preserve | connectivity_wall,documented_wall,exact_preserve,gather_heavy,high_memory,open_angle,scan |
| 10 | 138 | 10811.3 | 15.552 | 12215 | 462 | exact_preserve | conv_heavy,documented_wall,exact_preserve,high_memory,local_stencil,open_angle |
| 11 | 205 | 10806.3 | 15.676 | 10570 | 639 | exact_preserve | connectivity_wall,conv_heavy,exact_preserve,high_memory,open_angle,qlinear |
| 12 | 255 | 10395.4 | 15.712 | 10450 | 359 | exact_preserve | conv_heavy,exact_preserve,high_memory,matmul,qlinear |
| 13 | 025 | 9999.6 | 15.748 | 10320 | 104 | exact_preserve | exact_preserve,high_memory,onehot_final_equal,open_angle,scan |
| 14 | 080 | 8642.5 | 15.735 | 10109 | 454 | exact_preserve | conv_heavy,documented_wall,exact_preserve,high_memory,lut_selection,open_angle,qlinear |
| 15 | 158 | -20805.6 | 15.052 | 18263 | 2647 | exact_preserve | assignment_wall,conv_heavy,documented_wall,exact_preserve,gather_heavy,high_memory,marginal_wall,open_angle |
| 16 | 018 | -74016.2 | 14.774 | 26919 | 697 | exact_preserve | ambiguity_wall,documented_wall,exact_preserve,gather_heavy,high_memory,low_score,maxpool_scan,scatter |
| 17 | 118 | -82244.7 | 15.122 | 19153 | 339 | exact_preserve | connectivity_wall,conv_heavy,documented_wall,exact_preserve,high_memory,infeasible_exact_wall,information_loss_wall,local_stencil |

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
- candidates: 20
- expected: {'gain_type': 'memory+generalization', 'risk': 'medium', 'verification': 'confirm generator max height/width, stored eval, fresh adopt gate'}

| rank | task | score | pts | mem | params | source | tags |
|---:|---:|---:|---:|---:|---:|---|---|
| 1 | 233 | 32165.5 | 14.588 | 32796 | 446 | semantic_or_handbuilt | assignment_wall,connectivity_wall,conv_heavy,documented_wall,gather_heavy,high_memory,low_score,matmul |
| 2 | 366 | 31466.9 | 14.640 | 30983 | 576 | exact_preserve | assignment_wall,connectivity_wall,exact_preserve,gather_heavy,high_memory,low_score,lut_selection,open_angle |
| 3 | 133 | 28025.0 | 14.703 | 27333 | 2303 | exact_preserve | assignment_wall,connectivity_wall,conv_heavy,documented_wall,exact_preserve,gather_heavy,high_memory,low_score |
| 4 | 285 | 23574.8 | 14.864 | 24684 | 550 | exact_preserve | connectivity_wall,documented_wall,exact_preserve,gather_heavy,high_memory,low_score,open_angle,scatter |
| 5 | 054 | 20655.7 | 15.078 | 20091 | 288 | unknown | gather_heavy,high_memory,scan,scatter |
| 6 | 173 | 17379.5 | 15.222 | 17534 | 112 | exact_preserve | exact_preserve,gather_heavy,high_memory,scatter |
| 7 | 319 | 15367.8 | 15.251 | 16090 | 1053 | exact_preserve | assignment_wall,documented_wall,exact_preserve,gather_heavy,high_memory,scan |
| 8 | 349 | 15204.2 | 15.289 | 15300 | 1191 | semantic_or_handbuilt | conv_heavy,documented_wall,high_memory,local_stencil,lut_selection,open_angle,qlinear |
| 9 | 076 | 14771.4 | 15.285 | 16254 | 303 | exact_preserve | assignment_wall,connectivity_wall,documented_wall,exact_preserve,high_memory,open_angle,scatter |
| 10 | 191 | 11496.8 | 15.617 | 11741 | 141 | exact_preserve | assignment_wall,conv_heavy,custom_win,exact_preserve,high_memory,open_angle,qlinear |
| 11 | 198 | 10943.3 | 15.542 | 12670 | 136 | exact_preserve | connectivity_wall,documented_wall,exact_preserve,gather_heavy,high_memory,open_angle,scan |
| 12 | 138 | 10811.3 | 15.552 | 12215 | 462 | exact_preserve | conv_heavy,documented_wall,exact_preserve,high_memory,local_stencil,open_angle |
| 13 | 205 | 10806.3 | 15.676 | 10570 | 639 | exact_preserve | connectivity_wall,conv_heavy,exact_preserve,high_memory,open_angle,qlinear |
| 14 | 255 | 10395.4 | 15.712 | 10450 | 359 | exact_preserve | conv_heavy,exact_preserve,high_memory,matmul,qlinear |
| 15 | 025 | 9999.6 | 15.748 | 10320 | 104 | exact_preserve | exact_preserve,high_memory,onehot_final_equal,open_angle,scan |
| 16 | 338 | 9248.4 | 15.725 | 10000 | 666 | semantic_or_handbuilt | connectivity_wall,documented_wall,high_memory,local_stencil,open_angle,qlinear |
| 17 | 080 | 8642.5 | 15.735 | 10109 | 454 | exact_preserve | conv_heavy,documented_wall,exact_preserve,high_memory,lut_selection,open_angle,qlinear |
| 18 | 018 | -74016.2 | 14.774 | 26919 | 697 | exact_preserve | ambiguity_wall,documented_wall,exact_preserve,gather_heavy,high_memory,low_score,maxpool_scan,scatter |
| 19 | 209 | -78474.8 | 14.969 | 21298 | 1418 | unknown | ambiguity_wall,assignment_wall,connectivity_wall,conv_heavy,documented_wall,high_memory,low_score,lut_selection |
| 20 | 118 | -82244.7 | 15.122 | 19153 | 339 | exact_preserve | connectivity_wall,conv_heavy,documented_wall,exact_preserve,high_memory,infeasible_exact_wall,information_loss_wall,local_stencil |

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
| 5 | 173 | 17379.5 | 15.222 | 17534 | 112 | exact_preserve | exact_preserve,gather_heavy,high_memory,scatter |
| 6 | 349 | 15204.2 | 15.289 | 15300 | 1191 | semantic_or_handbuilt | conv_heavy,documented_wall,high_memory,local_stencil,lut_selection,open_angle,qlinear |
| 7 | 076 | 14771.4 | 15.285 | 16254 | 303 | exact_preserve | assignment_wall,connectivity_wall,documented_wall,exact_preserve,high_memory,open_angle,scatter |
| 8 | 158 | -20805.6 | 15.052 | 18263 | 2647 | exact_preserve | assignment_wall,conv_heavy,documented_wall,exact_preserve,gather_heavy,high_memory,marginal_wall,open_angle |
| 9 | 018 | -74016.2 | 14.774 | 26919 | 697 | exact_preserve | ambiguity_wall,documented_wall,exact_preserve,gather_heavy,high_memory,low_score,maxpool_scan,scatter |
| 10 | 209 | -78474.8 | 14.969 | 21298 | 1418 | unknown | ambiguity_wall,assignment_wall,connectivity_wall,conv_heavy,documented_wall,high_memory,low_score,lut_selection |
| 11 | 118 | -82244.7 | 15.122 | 19153 | 339 | exact_preserve | connectivity_wall,conv_heavy,documented_wall,exact_preserve,high_memory,infeasible_exact_wall,information_loss_wall,local_stencil |

## public_teacher_qlinear_conv_rewrite

Convert repeated binary/count Conv/ConvTranspose towers to QLinearConv/uint8 routes

- source tasks: 023 349 182 233 255 364 338 004
- candidates: 32
- expected: {'gain_type': 'memory+params', 'risk': 'medium', 'verification': 'stored eval plus fresh; inspect for output-range saturation before adoption'}

| rank | task | score | pts | mem | params | source | tags |
|---:|---:|---:|---:|---:|---:|---|---|
| 1 | 286 | 22796.4 | 14.915 | 21090 | 2881 | semantic_or_handbuilt | connectivity_wall,conv_heavy,documented_wall,high_memory,local_stencil,low_score,qlinear |
| 2 | 349 | 15204.2 | 15.289 | 15300 | 1191 | semantic_or_handbuilt | conv_heavy,documented_wall,high_memory,local_stencil,lut_selection,open_angle,qlinear |
| 3 | 138 | 10811.3 | 15.552 | 12215 | 462 | exact_preserve | conv_heavy,documented_wall,exact_preserve,high_memory,local_stencil,open_angle |
| 4 | 205 | 10806.3 | 15.676 | 10570 | 639 | exact_preserve | connectivity_wall,conv_heavy,exact_preserve,high_memory,open_angle,qlinear |
| 5 | 255 | 10395.4 | 15.712 | 10450 | 359 | exact_preserve | conv_heavy,exact_preserve,high_memory,matmul,qlinear |
| 6 | 379 | 8668.9 | 15.880 | 7860 | 1273 | exact_preserve | connectivity_wall,conv_heavy,custom_win,exact_preserve,open_angle |
| 7 | 080 | 8642.5 | 15.735 | 10109 | 454 | exact_preserve | conv_heavy,documented_wall,exact_preserve,high_memory,lut_selection,open_angle,qlinear |
| 8 | 074 | 8583.2 | 15.889 | 9000 | 50 | exact_preserve | exact_preserve,local_stencil,open_angle |
| 9 | 377 | 7712.9 | 15.987 | 8075 | 134 | exact_preserve | exact_preserve,local_stencil |
| 10 | 187 | 6330.0 | 15.973 | 5000 | 3322 | exact_preserve | connectivity_wall,documented_wall,exact_preserve,local_stencil,lut_selection |
| 11 | 222 | 5703.0 | 16.267 | 5992 | 211 | exact_preserve | custom_win,exact_preserve,local_stencil,maxpool_scan,open_angle,qlinear |
| 12 | 009 | 5024.0 | 16.143 | 6929 | 95 | exact_preserve | connectivity_wall,documented_wall,exact_preserve,local_stencil,lut_selection,onehot_final_equal,open_angle |
| 13 | 192 | 4996.0 | 16.147 | 5745 | 1251 | exact_preserve | conv_heavy,documented_wall,exact_preserve,local_stencil,open_angle,qlinear |
| 14 | 085 | 4881.0 | 16.409 | 4500 | 881 | exact_preserve | exact_preserve,local_stencil,open_angle |
| 15 | 005 | 4801.0 | 16.175 | 6311 | 490 | exact_preserve | conv_heavy,documented_wall,exact_preserve,lut_selection,qlinear |
| 16 | 396 | 4444.0 | 16.494 | 4854 | 90 | exact_preserve | conv_heavy,exact_preserve,local_stencil,onehot_final_equal |
| 17 | 165 | 4058.0 | 16.291 | 5828 | 230 | exact_preserve | connectivity_wall,documented_wall,exact_preserve,local_stencil,open_angle,qlinear |
| 18 | 363 | 3635.0 | 16.673 | 3839 | 296 | exact_preserve | connectivity_wall,conv_heavy,exact_preserve,local_stencil,onehot_final_equal,open_angle |
| 19 | 132 | 3589.0 | 16.684 | 3990 | 99 | exact_preserve | bitwise_program,exact_preserve,gather_heavy,local_stencil,open_angle |
| 20 | 284 | 3149.0 | 16.798 | 2996 | 653 | exact_preserve | conv_heavy,custom_win,exact_preserve,open_angle,scatter |
| 21 | 004 | 3137.0 | 16.456 | 5044 | 93 | exact_preserve | connectivity_wall,documented_wall,exact_preserve,local_stencil,lut_selection,open_angle,qlinear |
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
| 7 | 018 | -74016.2 | 14.774 | 26919 | 697 | exact_preserve | ambiguity_wall,documented_wall,exact_preserve,gather_heavy,high_memory,low_score,maxpool_scan,scatter |

## band_profile_contraction_final_equal

Use distinct-colour row/column profiles to avoid full connectivity for band-fill tasks

- source tasks: 202
- candidates: 22
- expected: {'gain_type': 'semantic_rewrite_memory_or_source_ownership', 'risk': 'medium; requires generator proof that colours are distinct per band/profile', 'verification': 'stored eval plus large fresh generator eval; source/live reconcile after rewrite'}

| rank | task | score | pts | mem | params | source | tags |
|---:|---:|---:|---:|---:|---:|---|---|
| 1 | 054 | 20655.7 | 15.078 | 20091 | 288 | unknown | gather_heavy,high_memory,scan,scatter |
| 2 | 173 | 17379.5 | 15.222 | 17534 | 112 | exact_preserve | exact_preserve,gather_heavy,high_memory,scatter |
| 3 | 349 | 15204.2 | 15.289 | 15300 | 1191 | semantic_or_handbuilt | conv_heavy,documented_wall,high_memory,local_stencil,lut_selection,open_angle,qlinear |
| 4 | 138 | 10811.3 | 15.552 | 12215 | 462 | exact_preserve | conv_heavy,documented_wall,exact_preserve,high_memory,local_stencil,open_angle |
| 5 | 255 | 10395.4 | 15.712 | 10450 | 359 | exact_preserve | conv_heavy,exact_preserve,high_memory,matmul,qlinear |
| 6 | 025 | 9999.6 | 15.748 | 10320 | 104 | exact_preserve | exact_preserve,high_memory,onehot_final_equal,open_angle,scan |
| 7 | 080 | 8642.5 | 15.735 | 10109 | 454 | exact_preserve | conv_heavy,documented_wall,exact_preserve,high_memory,lut_selection,open_angle,qlinear |
| 8 | 074 | 8583.2 | 15.889 | 9000 | 50 | exact_preserve | exact_preserve,local_stencil,open_angle |
| 9 | 377 | 7712.9 | 15.987 | 8075 | 134 | exact_preserve | exact_preserve,local_stencil |
| 10 | 017 | 6504.0 | 16.146 | 5192 | 1812 | exact_preserve | custom_win,exact_preserve,matmul,onehot_final_equal,open_angle |
| 11 | 023 | 6412.0 | 16.234 | 6163 | 249 | semantic_or_handbuilt | conv_heavy,local_stencil,open_angle,qlinear |
| 12 | 222 | 5703.0 | 16.267 | 5992 | 211 | exact_preserve | custom_win,exact_preserve,local_stencil,maxpool_scan,open_angle,qlinear |
| 13 | 279 | 5546.0 | 16.379 | 5508 | 38 | semantic_or_handbuilt | conv_heavy,local_stencil,lut_selection,qlinear |
| 14 | 278 | 5483.0 | 16.391 | 5436 | 47 | semantic_or_handbuilt | local_stencil,qlinear |
| 15 | 192 | 4996.0 | 16.147 | 5745 | 1251 | exact_preserve | conv_heavy,documented_wall,exact_preserve,local_stencil,open_angle,qlinear |
| 16 | 264 | 4908.0 | 16.404 | 4700 | 708 | exact_preserve | exact_preserve,local_stencil,lut_selection,qlinear |
| 17 | 085 | 4881.0 | 16.409 | 4500 | 881 | exact_preserve | exact_preserve,local_stencil,open_angle |
| 18 | 396 | 4444.0 | 16.494 | 4854 | 90 | exact_preserve | conv_heavy,exact_preserve,local_stencil,onehot_final_equal |
| 19 | 208 | 4226.0 | 16.539 | 4612 | 114 | exact_preserve | exact_preserve,onehot_final_equal,open_angle,qlinear |
| 20 | 162 | 3568.0 | 16.689 | 4024 | 44 | exact_preserve | exact_preserve,local_stencil,open_angle,qlinear |
| 21 | 029 | -35201.0 | 16.425 | 5212 | 87 | exact_preserve | exact_preserve,marginal_wall,onehot_final_equal,open_angle |
| 22 | 018 | -74016.2 | 14.774 | 26919 | 697 | exact_preserve | ambiguity_wall,documented_wall,exact_preserve,gather_heavy,high_memory,low_score,maxpool_scan,scatter |

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
- candidates: 174
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
| 9 | 173 | 17379.5 | 15.222 | 17534 | 112 | exact_preserve | exact_preserve,gather_heavy,high_memory,scatter |
| 10 | 319 | 15367.8 | 15.251 | 16090 | 1053 | exact_preserve | assignment_wall,documented_wall,exact_preserve,gather_heavy,high_memory,scan |
| 11 | 349 | 15204.2 | 15.289 | 15300 | 1191 | semantic_or_handbuilt | conv_heavy,documented_wall,high_memory,local_stencil,lut_selection,open_angle,qlinear |
| 12 | 076 | 14771.4 | 15.285 | 16254 | 303 | exact_preserve | assignment_wall,connectivity_wall,documented_wall,exact_preserve,high_memory,open_angle,scatter |
| 13 | 101 | 14620.5 | 15.422 | 13573 | 874 | semantic_or_handbuilt | assignment_wall,connectivity_wall,gather_heavy,high_memory,onehot_final_equal,scatter |
| 14 | 191 | 11496.8 | 15.617 | 11741 | 141 | exact_preserve | assignment_wall,conv_heavy,custom_win,exact_preserve,high_memory,open_angle,qlinear |
| 15 | 198 | 10943.3 | 15.542 | 12670 | 136 | exact_preserve | connectivity_wall,documented_wall,exact_preserve,gather_heavy,high_memory,open_angle,scan |
| 16 | 138 | 10811.3 | 15.552 | 12215 | 462 | exact_preserve | conv_heavy,documented_wall,exact_preserve,high_memory,local_stencil,open_angle |
| 17 | 205 | 10806.3 | 15.676 | 10570 | 639 | exact_preserve | connectivity_wall,conv_heavy,exact_preserve,high_memory,open_angle,qlinear |
| 18 | 255 | 10395.4 | 15.712 | 10450 | 359 | exact_preserve | conv_heavy,exact_preserve,high_memory,matmul,qlinear |
| 19 | 066 | 10187.7 | 15.778 | 9955 | 166 | semantic_or_handbuilt | marker_routed_path,open_angle,scan |
| 20 | 025 | 9999.6 | 15.748 | 10320 | 104 | exact_preserve | exact_preserve,high_memory,onehot_final_equal,open_angle,scan |
| 21 | 338 | 9248.4 | 15.725 | 10000 | 666 | semantic_or_handbuilt | connectivity_wall,documented_wall,high_memory,local_stencil,open_angle,qlinear |
| 22 | 145 | 9069.5 | 15.742 | 9244 | 1248 | semantic_or_handbuilt | connectivity_wall,custom_win,documented_wall,lut_selection,open_angle |
| 23 | 379 | 8668.9 | 15.880 | 7860 | 1273 | exact_preserve | connectivity_wall,conv_heavy,custom_win,exact_preserve,open_angle |
| 24 | 080 | 8642.5 | 15.735 | 10109 | 454 | exact_preserve | conv_heavy,documented_wall,exact_preserve,high_memory,lut_selection,open_angle,qlinear |
| 25 | 370 | 8587.8 | 15.944 | 7477 | 1094 | semantic_or_handbuilt | qlinear,scatter |

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
- candidates: 203
- expected: {'gain_type': 'memory+params', 'risk': 'low when every output on-cell is copied from the original one-hot input and off-grid can point at an all-zero padded input coordinate', 'verification': 'stored eval plus fresh side-by-side; confirm black in-grid cells and off-grid all-zero semantics are both preserved'}

| rank | task | score | pts | mem | params | source | tags |
|---:|---:|---:|---:|---:|---:|---|---|
| 1 | 366 | 31466.9 | 14.640 | 30983 | 576 | exact_preserve | assignment_wall,connectivity_wall,exact_preserve,gather_heavy,high_memory,low_score,lut_selection,open_angle |
| 2 | 133 | 28025.0 | 14.703 | 27333 | 2303 | exact_preserve | assignment_wall,connectivity_wall,conv_heavy,documented_wall,exact_preserve,gather_heavy,high_memory,low_score |
| 3 | 285 | 23574.8 | 14.864 | 24684 | 550 | exact_preserve | connectivity_wall,documented_wall,exact_preserve,gather_heavy,high_memory,low_score,open_angle,scatter |
| 4 | 364 | 19396.5 | 15.115 | 18500 | 1131 | exact_preserve | connectivity_wall,exact_preserve,high_memory,local_stencil,open_angle,qlinear |
| 5 | 349 | 15204.2 | 15.289 | 15300 | 1191 | semantic_or_handbuilt | conv_heavy,documented_wall,high_memory,local_stencil,lut_selection,open_angle,qlinear |
| 6 | 076 | 14771.4 | 15.285 | 16254 | 303 | exact_preserve | assignment_wall,connectivity_wall,documented_wall,exact_preserve,high_memory,open_angle,scatter |
| 7 | 101 | 14620.5 | 15.422 | 13573 | 874 | semantic_or_handbuilt | assignment_wall,connectivity_wall,gather_heavy,high_memory,onehot_final_equal,scatter |
| 8 | 191 | 11496.8 | 15.617 | 11741 | 141 | exact_preserve | assignment_wall,conv_heavy,custom_win,exact_preserve,high_memory,open_angle,qlinear |
| 9 | 198 | 10943.3 | 15.542 | 12670 | 136 | exact_preserve | connectivity_wall,documented_wall,exact_preserve,gather_heavy,high_memory,open_angle,scan |
| 10 | 138 | 10811.3 | 15.552 | 12215 | 462 | exact_preserve | conv_heavy,documented_wall,exact_preserve,high_memory,local_stencil,open_angle |
| 11 | 205 | 10806.3 | 15.676 | 10570 | 639 | exact_preserve | connectivity_wall,conv_heavy,exact_preserve,high_memory,open_angle,qlinear |
| 12 | 066 | 10187.7 | 15.778 | 9955 | 166 | semantic_or_handbuilt | marker_routed_path,open_angle,scan |
| 13 | 025 | 9999.6 | 15.748 | 10320 | 104 | exact_preserve | exact_preserve,high_memory,onehot_final_equal,open_angle,scan |
| 14 | 338 | 9248.4 | 15.725 | 10000 | 666 | semantic_or_handbuilt | connectivity_wall,documented_wall,high_memory,local_stencil,open_angle,qlinear |
| 15 | 145 | 9069.5 | 15.742 | 9244 | 1248 | semantic_or_handbuilt | connectivity_wall,custom_win,documented_wall,lut_selection,open_angle |
| 16 | 379 | 8668.9 | 15.880 | 7860 | 1273 | exact_preserve | connectivity_wall,conv_heavy,custom_win,exact_preserve,open_angle |
| 17 | 080 | 8642.5 | 15.735 | 10109 | 454 | exact_preserve | conv_heavy,documented_wall,exact_preserve,high_memory,lut_selection,open_angle,qlinear |
| 18 | 074 | 8583.2 | 15.889 | 9000 | 50 | exact_preserve | exact_preserve,local_stencil,open_angle |
| 19 | 064 | 8548.7 | 15.791 | 9852 | 134 | semantic_or_handbuilt | connectivity_wall,documented_wall,onehot_final_equal,open_angle |
| 20 | 204 | 8302.0 | 15.767 | 9780 | 452 | exact_preserve | connectivity_wall,conv_heavy,custom_win,documented_wall,exact_preserve,local_stencil,maxpool_scan,open_angle |
| 21 | 216 | 6679.9 | 15.934 | 8584 | 76 | exact_preserve | connectivity_wall,documented_wall,exact_preserve,open_angle,qlinear,scatter |
| 22 | 017 | 6504.0 | 16.146 | 5192 | 1812 | exact_preserve | custom_win,exact_preserve,matmul,onehot_final_equal,open_angle |
| 23 | 023 | 6412.0 | 16.234 | 6163 | 249 | semantic_or_handbuilt | conv_heavy,local_stencil,open_angle,qlinear |
| 24 | 096 | 6380.8 | 15.967 | 7953 | 418 | exact_preserve | documented_wall,exact_preserve,open_angle,scatter |
| 25 | 328 | 6275.0 | 16.179 | 4835 | 1940 | exact_preserve | exact_preserve,lut_selection |

## uint8_topk_compact_label_grid

Run TopK directly on compact uint8 label grids when only nonzero-cell enumeration is needed

- source tasks: 285 366
- candidates: 83
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
| 9 | 173 | 17379.5 | 15.222 | 17534 | 112 | exact_preserve | exact_preserve,gather_heavy,high_memory,scatter |
| 10 | 319 | 15367.8 | 15.251 | 16090 | 1053 | exact_preserve | assignment_wall,documented_wall,exact_preserve,gather_heavy,high_memory,scan |
| 11 | 349 | 15204.2 | 15.289 | 15300 | 1191 | semantic_or_handbuilt | conv_heavy,documented_wall,high_memory,local_stencil,lut_selection,open_angle,qlinear |
| 12 | 076 | 14771.4 | 15.285 | 16254 | 303 | exact_preserve | assignment_wall,connectivity_wall,documented_wall,exact_preserve,high_memory,open_angle,scatter |
| 13 | 101 | 14620.5 | 15.422 | 13573 | 874 | semantic_or_handbuilt | assignment_wall,connectivity_wall,gather_heavy,high_memory,onehot_final_equal,scatter |
| 14 | 191 | 11496.8 | 15.617 | 11741 | 141 | exact_preserve | assignment_wall,conv_heavy,custom_win,exact_preserve,high_memory,open_angle,qlinear |
| 15 | 198 | 10943.3 | 15.542 | 12670 | 136 | exact_preserve | connectivity_wall,documented_wall,exact_preserve,gather_heavy,high_memory,open_angle,scan |
| 16 | 138 | 10811.3 | 15.552 | 12215 | 462 | exact_preserve | conv_heavy,documented_wall,exact_preserve,high_memory,local_stencil,open_angle |
| 17 | 205 | 10806.3 | 15.676 | 10570 | 639 | exact_preserve | connectivity_wall,conv_heavy,exact_preserve,high_memory,open_angle,qlinear |
| 18 | 255 | 10395.4 | 15.712 | 10450 | 359 | exact_preserve | conv_heavy,exact_preserve,high_memory,matmul,qlinear |
| 19 | 066 | 10187.7 | 15.778 | 9955 | 166 | semantic_or_handbuilt | marker_routed_path,open_angle,scan |
| 20 | 025 | 9999.6 | 15.748 | 10320 | 104 | exact_preserve | exact_preserve,high_memory,onehot_final_equal,open_angle,scan |
| 21 | 338 | 9248.4 | 15.725 | 10000 | 666 | semantic_or_handbuilt | connectivity_wall,documented_wall,high_memory,local_stencil,open_angle,qlinear |
| 22 | 145 | 9069.5 | 15.742 | 9244 | 1248 | semantic_or_handbuilt | connectivity_wall,custom_win,documented_wall,lut_selection,open_angle |
| 23 | 379 | 8668.9 | 15.880 | 7860 | 1273 | exact_preserve | connectivity_wall,conv_heavy,custom_win,exact_preserve,open_angle |
| 24 | 080 | 8642.5 | 15.735 | 10109 | 454 | exact_preserve | conv_heavy,documented_wall,exact_preserve,high_memory,lut_selection,open_angle,qlinear |
| 25 | 370 | 8587.8 | 15.944 | 7477 | 1094 | semantic_or_handbuilt | qlinear,scatter |

## strided_conv_fixed_block_counts

Replace many fixed-block Slice+ReduceSum branches with one strided Conv

- source tasks: 011
- candidates: 175
- expected: {'gain_type': 'memory_when_repeated_block_reads_share_one_channel_or_kernel', 'risk': 'low-medium; Conv weight params can outweigh memory savings when few blocks are read or the input crop is already tiny', 'verification': 'stored eval plus fresh side-by-side; inspect Conv output shape because full 30x30 input may produce trailing off-grid stride positions that must be sliced away'}

| rank | task | score | pts | mem | params | source | tags |
|---:|---:|---:|---:|---:|---:|---|---|
| 1 | 366 | 31466.9 | 14.640 | 30983 | 576 | exact_preserve | assignment_wall,connectivity_wall,exact_preserve,gather_heavy,high_memory,low_score,lut_selection,open_angle |
| 2 | 133 | 28025.0 | 14.703 | 27333 | 2303 | exact_preserve | assignment_wall,connectivity_wall,conv_heavy,documented_wall,exact_preserve,gather_heavy,high_memory,low_score |
| 3 | 285 | 23574.8 | 14.864 | 24684 | 550 | exact_preserve | connectivity_wall,documented_wall,exact_preserve,gather_heavy,high_memory,low_score,open_angle,scatter |
| 4 | 364 | 19396.5 | 15.115 | 18500 | 1131 | exact_preserve | connectivity_wall,exact_preserve,high_memory,local_stencil,open_angle,qlinear |
| 5 | 367 | 18288.8 | 15.121 | 14800 | 4725 | semantic_or_handbuilt | connectivity_wall,conv_heavy,documented_wall,high_memory,local_stencil,lut_selection,open_angle,qlinear |
| 6 | 173 | 17379.5 | 15.222 | 17534 | 112 | exact_preserve | exact_preserve,gather_heavy,high_memory,scatter |
| 7 | 319 | 15367.8 | 15.251 | 16090 | 1053 | exact_preserve | assignment_wall,documented_wall,exact_preserve,gather_heavy,high_memory,scan |
| 8 | 076 | 14771.4 | 15.285 | 16254 | 303 | exact_preserve | assignment_wall,connectivity_wall,documented_wall,exact_preserve,high_memory,open_angle,scatter |
| 9 | 101 | 14620.5 | 15.422 | 13573 | 874 | semantic_or_handbuilt | assignment_wall,connectivity_wall,gather_heavy,high_memory,onehot_final_equal,scatter |
| 10 | 191 | 11496.8 | 15.617 | 11741 | 141 | exact_preserve | assignment_wall,conv_heavy,custom_win,exact_preserve,high_memory,open_angle,qlinear |
| 11 | 198 | 10943.3 | 15.542 | 12670 | 136 | exact_preserve | connectivity_wall,documented_wall,exact_preserve,gather_heavy,high_memory,open_angle,scan |
| 12 | 205 | 10806.3 | 15.676 | 10570 | 639 | exact_preserve | connectivity_wall,conv_heavy,exact_preserve,high_memory,open_angle,qlinear |
| 13 | 255 | 10395.4 | 15.712 | 10450 | 359 | exact_preserve | conv_heavy,exact_preserve,high_memory,matmul,qlinear |
| 14 | 025 | 9999.6 | 15.748 | 10320 | 104 | exact_preserve | exact_preserve,high_memory,onehot_final_equal,open_angle,scan |
| 15 | 338 | 9248.4 | 15.725 | 10000 | 666 | semantic_or_handbuilt | connectivity_wall,documented_wall,high_memory,local_stencil,open_angle,qlinear |
| 16 | 379 | 8668.9 | 15.880 | 7860 | 1273 | exact_preserve | connectivity_wall,conv_heavy,custom_win,exact_preserve,open_angle |
| 17 | 080 | 8642.5 | 15.735 | 10109 | 454 | exact_preserve | conv_heavy,documented_wall,exact_preserve,high_memory,lut_selection,open_angle,qlinear |
| 18 | 064 | 8548.7 | 15.791 | 9852 | 134 | semantic_or_handbuilt | connectivity_wall,documented_wall,onehot_final_equal,open_angle |
| 19 | 204 | 8302.0 | 15.767 | 9780 | 452 | exact_preserve | connectivity_wall,conv_heavy,custom_win,documented_wall,exact_preserve,local_stencil,maxpool_scan,open_angle |
| 20 | 377 | 7712.9 | 15.987 | 8075 | 134 | exact_preserve | exact_preserve,local_stencil |
| 21 | 044 | 7161.7 | 15.881 | 8972 | 154 | exact_preserve | assignment_wall,connectivity_wall,documented_wall,exact_preserve,scan |
| 22 | 324 | 6921.9 | 15.907 | 7308 | 1586 | exact_preserve | assignment_wall,conv_heavy,documented_wall,exact_preserve,local_stencil,qlinear,scan |
| 23 | 216 | 6679.9 | 15.934 | 8584 | 76 | exact_preserve | connectivity_wall,documented_wall,exact_preserve,open_angle,qlinear,scatter |
| 24 | 023 | 6412.0 | 16.234 | 6163 | 249 | semantic_or_handbuilt | conv_heavy,local_stencil,open_angle,qlinear |
| 25 | 096 | 6380.8 | 15.967 | 7953 | 418 | exact_preserve | documented_wall,exact_preserve,open_angle,scatter |

## solid_marker_profile_reconstruction

Replace full solid-marker channel slices with row/column profile contractions

- source tasks: 008
- candidates: 25
- expected: {'gain_type': 'memory_by_deleting_full_marker_plane', 'risk': 'low when the marker is generator-proven solid and separable; medium when the marker may touch borders or share rows/cols with other same-colour pixels', 'verification': 'stored eval plus fresh generation; sample generator border cases because flip/xpose can invalidate assumed crop margins'}

| rank | task | score | pts | mem | params | source | tags |
|---:|---:|---:|---:|---:|---:|---|---|
| 1 | 101 | 14620.5 | 15.422 | 13573 | 874 | semantic_or_handbuilt | assignment_wall,connectivity_wall,gather_heavy,high_memory,onehot_final_equal,scatter |
| 2 | 025 | 9999.6 | 15.748 | 10320 | 104 | exact_preserve | exact_preserve,high_memory,onehot_final_equal,open_angle,scan |
| 3 | 064 | 8548.7 | 15.791 | 9852 | 134 | semantic_or_handbuilt | connectivity_wall,documented_wall,onehot_final_equal,open_angle |
| 4 | 017 | 6504.0 | 16.146 | 5192 | 1812 | exact_preserve | custom_win,exact_preserve,matmul,onehot_final_equal,open_angle |
| 5 | 219 | 5710.0 | 16.117 | 7078 | 132 | unknown | assignment_wall,connectivity_wall,documented_wall,lut_selection,onehot_final_equal,open_angle,scan |
| 6 | 368 | 5693.0 | 16.269 | 5990 | 203 | exact_preserve | connectivity_wall,exact_preserve,gather_heavy,onehot_final_equal,scatter |
| 7 | 009 | 5024.0 | 16.143 | 6929 | 95 | exact_preserve | connectivity_wall,documented_wall,exact_preserve,local_stencil,lut_selection,onehot_final_equal,open_angle |
| 8 | 182 | 4942.0 | 16.229 | 6345 | 97 | semantic_or_handbuilt | assignment_wall,connectivity_wall,conv_heavy,documented_wall,local_stencil,lut_selection,onehot_final_equal,open_angle |
| 9 | 383 | 4570.0 | 16.289 | 5974 | 96 | unknown | assignment_wall,connectivity_wall,documented_wall,local_stencil,onehot_final_equal,open_angle |
| 10 | 396 | 4444.0 | 16.494 | 4854 | 90 | exact_preserve | conv_heavy,exact_preserve,local_stencil,onehot_final_equal |
| 11 | 208 | 4226.0 | 16.539 | 4612 | 114 | exact_preserve | exact_preserve,onehot_final_equal,open_angle,qlinear |
| 12 | 363 | 3635.0 | 16.673 | 3839 | 296 | exact_preserve | connectivity_wall,conv_heavy,exact_preserve,local_stencil,onehot_final_equal,open_angle |
| 13 | 107 | 3313.0 | 16.754 | 3242 | 571 | exact_preserve | exact_preserve,lut_selection,matmul,onehot_final_equal,open_angle,qlinear,scatter |
| 14 | 202 | 3287.0 | 16.902 | 3253 | 34 | unknown | connectivity_wall,onehot_final_equal,open_angle,qlinear |
| 15 | 177 | 3121.0 | 16.805 | 3500 | 121 | exact_preserve | connectivity_wall,conv_heavy,exact_preserve,local_stencil,onehot_final_equal,open_angle |
| 16 | 234 | 2950.0 | 17.010 | 2886 | 64 | semantic_or_handbuilt | connectivity_wall,onehot_final_equal |
| 17 | 036 | 1677.0 | 17.314 | 2140 | 37 | exact_preserve | connectivity_wall,exact_preserve,onehot_final_equal,open_angle |
| 18 | 392 | 1451.0 | 17.424 | 1602 | 349 | exact_preserve | custom_win,exact_preserve,onehot_final_equal,open_angle |
| 19 | 213 | 1358.0 | 17.473 | 1809 | 49 | exact_preserve | exact_preserve,onehot_final_equal,open_angle |
| 20 | 333 | 1227.0 | 16.921 | 2792 | 435 | exact_preserve | connectivity_wall,documented_wall,exact_preserve,onehot_final_equal,open_angle |
| 21 | 308 | 988.0 | 17.695 | 1385 | 103 | exact_preserve | connectivity_wall,exact_preserve,onehot_final_equal,open_angle,scatter |
| 22 | 071 | 698.0 | 17.100 | 2643 | 55 | exact_preserve | connectivity_wall,documented_wall,exact_preserve,onehot_final_equal,open_angle |
| 23 | 170 | 545.0 | 17.158 | 2216 | 329 | exact_preserve | assignment_wall,conv_heavy,documented_wall,exact_preserve,local_stencil,lut_selection,onehot_final_equal,open_angle |
| 24 | 184 | 160.0 | 17.322 | 2100 | 60 | exact_preserve | connectivity_wall,conv_heavy,documented_wall,exact_preserve,local_stencil,onehot_final_equal,open_angle,qlinear |
| 25 | 029 | -35201.0 | 16.425 | 5212 | 87 | exact_preserve | exact_preserve,marginal_wall,onehot_final_equal,open_angle |

## walk_einsum_iteration_collapse

Collapse iterative flood/scan/propagation into ONE multi-operand Einsum via scaled walk counting (internal computation uncounted)

- source tasks: 187 313 110 243 077
- candidates: 29
- expected: {'gain_type': 'memory_by_deleting_all_iteration_planes', 'risk': 'medium: verify connectivity model (4 vs 8-conn!) matches incumbent semantics on fresh; verify step-count covers measured max BFS distance + margin; keep operands nonnegative or prove no cancellation. REJECT-CHECKS (task204 S8): parallel fan-out MaxPools (per-size anchor dilations) are NOT collapsible chains; uint8 conv banks of sub-400B planes are einsum-proof (fp32 entry ticket 1600-3600B + step params loses)', 'verification': 'numpy walk-vs-BFS on 20000 fresh, stored eval, fresh_verify candidate<=incumbent, A/B submission for grader safety on first use'}

| rank | task | score | pts | mem | params | source | tags |
|---:|---:|---:|---:|---:|---:|---|---|
| 1 | 233 | 32165.5 | 14.588 | 32796 | 446 | semantic_or_handbuilt | assignment_wall,connectivity_wall,conv_heavy,documented_wall,gather_heavy,high_memory,low_score,matmul |
| 2 | 366 | 31466.9 | 14.640 | 30983 | 576 | exact_preserve | assignment_wall,connectivity_wall,exact_preserve,gather_heavy,high_memory,low_score,lut_selection,open_angle |
| 3 | 133 | 28025.0 | 14.703 | 27333 | 2303 | exact_preserve | assignment_wall,connectivity_wall,conv_heavy,documented_wall,exact_preserve,gather_heavy,high_memory,low_score |
| 4 | 285 | 23574.8 | 14.864 | 24684 | 550 | exact_preserve | connectivity_wall,documented_wall,exact_preserve,gather_heavy,high_memory,low_score,open_angle,scatter |
| 5 | 054 | 20655.7 | 15.078 | 20091 | 288 | unknown | gather_heavy,high_memory,scan,scatter |
| 6 | 173 | 17379.5 | 15.222 | 17534 | 112 | exact_preserve | exact_preserve,gather_heavy,high_memory,scatter |
| 7 | 319 | 15367.8 | 15.251 | 16090 | 1053 | exact_preserve | assignment_wall,documented_wall,exact_preserve,gather_heavy,high_memory,scan |
| 8 | 349 | 15204.2 | 15.289 | 15300 | 1191 | semantic_or_handbuilt | conv_heavy,documented_wall,high_memory,local_stencil,lut_selection,open_angle,qlinear |
| 9 | 076 | 14771.4 | 15.285 | 16254 | 303 | exact_preserve | assignment_wall,connectivity_wall,documented_wall,exact_preserve,high_memory,open_angle,scatter |
| 10 | 101 | 14620.5 | 15.422 | 13573 | 874 | semantic_or_handbuilt | assignment_wall,connectivity_wall,gather_heavy,high_memory,onehot_final_equal,scatter |
| 11 | 191 | 11496.8 | 15.617 | 11741 | 141 | exact_preserve | assignment_wall,conv_heavy,custom_win,exact_preserve,high_memory,open_angle,qlinear |
| 12 | 198 | 10943.3 | 15.542 | 12670 | 136 | exact_preserve | connectivity_wall,documented_wall,exact_preserve,gather_heavy,high_memory,open_angle,scan |
| 13 | 138 | 10811.3 | 15.552 | 12215 | 462 | exact_preserve | conv_heavy,documented_wall,exact_preserve,high_memory,local_stencil,open_angle |
| 14 | 205 | 10806.3 | 15.676 | 10570 | 639 | exact_preserve | connectivity_wall,conv_heavy,exact_preserve,high_memory,open_angle,qlinear |
| 15 | 255 | 10395.4 | 15.712 | 10450 | 359 | exact_preserve | conv_heavy,exact_preserve,high_memory,matmul,qlinear |
| 16 | 066 | 10187.7 | 15.778 | 9955 | 166 | semantic_or_handbuilt | marker_routed_path,open_angle,scan |
| 17 | 025 | 9999.6 | 15.748 | 10320 | 104 | exact_preserve | exact_preserve,high_memory,onehot_final_equal,open_angle,scan |
| 18 | 338 | 9248.4 | 15.725 | 10000 | 666 | semantic_or_handbuilt | connectivity_wall,documented_wall,high_memory,local_stencil,open_angle,qlinear |
| 19 | 145 | 9069.5 | 15.742 | 9244 | 1248 | semantic_or_handbuilt | connectivity_wall,custom_win,documented_wall,lut_selection,open_angle |
| 20 | 080 | 8642.5 | 15.735 | 10109 | 454 | exact_preserve | conv_heavy,documented_wall,exact_preserve,high_memory,lut_selection,open_angle,qlinear |
| 21 | 064 | 8548.7 | 15.791 | 9852 | 134 | semantic_or_handbuilt | connectivity_wall,documented_wall,onehot_final_equal,open_angle |
| 22 | 204 | 8302.0 | 15.767 | 9780 | 452 | exact_preserve | connectivity_wall,conv_heavy,custom_win,documented_wall,exact_preserve,local_stencil,maxpool_scan,open_angle |
| 23 | 044 | 7161.7 | 15.881 | 8972 | 154 | exact_preserve | assignment_wall,connectivity_wall,documented_wall,exact_preserve,scan |
| 24 | 216 | 6679.9 | 15.934 | 8584 | 76 | exact_preserve | connectivity_wall,documented_wall,exact_preserve,open_angle,qlinear,scatter |
| 25 | 158 | -20805.6 | 15.052 | 18263 | 2647 | exact_preserve | assignment_wall,conv_heavy,documented_wall,exact_preserve,gather_heavy,high_memory,marginal_wall,open_angle |

## self_einsum_axis_activity_gate

Use the input twice in one Einsum to gate final output by axis activity

- source tasks: 067
- candidates: 233
- expected: {'gain_type': 'frontier_by_deleting_explicit_mask_and_constants', 'risk': 'medium: repeated-input Einsum is exact only when the axis activity predicate is positive for every kept coordinate and zero for every rejected coordinate; verify colour-0 cells still produce one-hot occupancy', 'verification': 'stored eval plus uncached fresh generation; inspect equation letters carefully because a wrong reused index silently gates the wrong axis'}

| rank | task | score | pts | mem | params | source | tags |
|---:|---:|---:|---:|---:|---:|---|---|
| 1 | 366 | 31466.9 | 14.640 | 30983 | 576 | exact_preserve | assignment_wall,connectivity_wall,exact_preserve,gather_heavy,high_memory,low_score,lut_selection,open_angle |
| 2 | 133 | 28025.0 | 14.703 | 27333 | 2303 | exact_preserve | assignment_wall,connectivity_wall,conv_heavy,documented_wall,exact_preserve,gather_heavy,high_memory,low_score |
| 3 | 285 | 23574.8 | 14.864 | 24684 | 550 | exact_preserve | connectivity_wall,documented_wall,exact_preserve,gather_heavy,high_memory,low_score,open_angle,scatter |
| 4 | 364 | 19396.5 | 15.115 | 18500 | 1131 | exact_preserve | connectivity_wall,exact_preserve,high_memory,local_stencil,open_angle,qlinear |
| 5 | 367 | 18288.8 | 15.121 | 14800 | 4725 | semantic_or_handbuilt | connectivity_wall,conv_heavy,documented_wall,high_memory,local_stencil,lut_selection,open_angle,qlinear |
| 6 | 349 | 15204.2 | 15.289 | 15300 | 1191 | semantic_or_handbuilt | conv_heavy,documented_wall,high_memory,local_stencil,lut_selection,open_angle,qlinear |
| 7 | 076 | 14771.4 | 15.285 | 16254 | 303 | exact_preserve | assignment_wall,connectivity_wall,documented_wall,exact_preserve,high_memory,open_angle,scatter |
| 8 | 101 | 14620.5 | 15.422 | 13573 | 874 | semantic_or_handbuilt | assignment_wall,connectivity_wall,gather_heavy,high_memory,onehot_final_equal,scatter |
| 9 | 191 | 11496.8 | 15.617 | 11741 | 141 | exact_preserve | assignment_wall,conv_heavy,custom_win,exact_preserve,high_memory,open_angle,qlinear |
| 10 | 198 | 10943.3 | 15.542 | 12670 | 136 | exact_preserve | connectivity_wall,documented_wall,exact_preserve,gather_heavy,high_memory,open_angle,scan |
| 11 | 138 | 10811.3 | 15.552 | 12215 | 462 | exact_preserve | conv_heavy,documented_wall,exact_preserve,high_memory,local_stencil,open_angle |
| 12 | 205 | 10806.3 | 15.676 | 10570 | 639 | exact_preserve | connectivity_wall,conv_heavy,exact_preserve,high_memory,open_angle,qlinear |
| 13 | 066 | 10187.7 | 15.778 | 9955 | 166 | semantic_or_handbuilt | marker_routed_path,open_angle,scan |
| 14 | 025 | 9999.6 | 15.748 | 10320 | 104 | exact_preserve | exact_preserve,high_memory,onehot_final_equal,open_angle,scan |
| 15 | 338 | 9248.4 | 15.725 | 10000 | 666 | semantic_or_handbuilt | connectivity_wall,documented_wall,high_memory,local_stencil,open_angle,qlinear |
| 16 | 145 | 9069.5 | 15.742 | 9244 | 1248 | semantic_or_handbuilt | connectivity_wall,custom_win,documented_wall,lut_selection,open_angle |
| 17 | 379 | 8668.9 | 15.880 | 7860 | 1273 | exact_preserve | connectivity_wall,conv_heavy,custom_win,exact_preserve,open_angle |
| 18 | 080 | 8642.5 | 15.735 | 10109 | 454 | exact_preserve | conv_heavy,documented_wall,exact_preserve,high_memory,lut_selection,open_angle,qlinear |
| 19 | 064 | 8548.7 | 15.791 | 9852 | 134 | semantic_or_handbuilt | connectivity_wall,documented_wall,onehot_final_equal,open_angle |
| 20 | 204 | 8302.0 | 15.767 | 9780 | 452 | exact_preserve | connectivity_wall,conv_heavy,custom_win,documented_wall,exact_preserve,local_stencil,maxpool_scan,open_angle |
| 21 | 216 | 6679.9 | 15.934 | 8584 | 76 | exact_preserve | connectivity_wall,documented_wall,exact_preserve,open_angle,qlinear,scatter |
| 22 | 017 | 6504.0 | 16.146 | 5192 | 1812 | exact_preserve | custom_win,exact_preserve,matmul,onehot_final_equal,open_angle |
| 23 | 023 | 6412.0 | 16.234 | 6163 | 249 | semantic_or_handbuilt | conv_heavy,local_stencil,open_angle,qlinear |
| 24 | 096 | 6380.8 | 15.967 | 7953 | 418 | exact_preserve | documented_wall,exact_preserve,open_angle,scatter |
| 25 | 219 | 5710.0 | 16.117 | 7078 | 132 | unknown | assignment_wall,connectivity_wall,documented_wall,lut_selection,onehot_final_equal,open_angle,scan |

## label_pad_vs_onehot_pad_ordering

Choose label-pad vs onehot-pad order by compact spatial area

- source tasks: 308 382
- candidates: 29
- expected: {'gain_type': 'memory_by_swapping_final_label_expansion_order', 'risk': 'low: purely algebraic if pad value is outside channel ids and Equal broadcasts identically; verify ORT Pad supports the bool output dtype/opset when using onehot-before-pad', 'verification': 'stored eval and memory measurement for the single task; static break-even is compact_area * channel_count versus padded_label_area'}

| rank | task | score | pts | mem | params | source | tags |
|---:|---:|---:|---:|---:|---:|---|---|
| 1 | 101 | 14620.5 | 15.422 | 13573 | 874 | semantic_or_handbuilt | assignment_wall,connectivity_wall,gather_heavy,high_memory,onehot_final_equal,scatter |
| 2 | 025 | 9999.6 | 15.748 | 10320 | 104 | exact_preserve | exact_preserve,high_memory,onehot_final_equal,open_angle,scan |
| 3 | 064 | 8548.7 | 15.791 | 9852 | 134 | semantic_or_handbuilt | connectivity_wall,documented_wall,onehot_final_equal,open_angle |
| 4 | 017 | 6504.0 | 16.146 | 5192 | 1812 | exact_preserve | custom_win,exact_preserve,matmul,onehot_final_equal,open_angle |
| 5 | 219 | 5710.0 | 16.117 | 7078 | 132 | unknown | assignment_wall,connectivity_wall,documented_wall,lut_selection,onehot_final_equal,open_angle,scan |
| 6 | 368 | 5693.0 | 16.269 | 5990 | 203 | exact_preserve | connectivity_wall,exact_preserve,gather_heavy,onehot_final_equal,scatter |
| 7 | 009 | 5024.0 | 16.143 | 6929 | 95 | exact_preserve | connectivity_wall,documented_wall,exact_preserve,local_stencil,lut_selection,onehot_final_equal,open_angle |
| 8 | 182 | 4942.0 | 16.229 | 6345 | 97 | semantic_or_handbuilt | assignment_wall,connectivity_wall,conv_heavy,documented_wall,local_stencil,lut_selection,onehot_final_equal,open_angle |
| 9 | 383 | 4570.0 | 16.289 | 5974 | 96 | unknown | assignment_wall,connectivity_wall,documented_wall,local_stencil,onehot_final_equal,open_angle |
| 10 | 396 | 4444.0 | 16.494 | 4854 | 90 | exact_preserve | conv_heavy,exact_preserve,local_stencil,onehot_final_equal |
| 11 | 208 | 4226.0 | 16.539 | 4612 | 114 | exact_preserve | exact_preserve,onehot_final_equal,open_angle,qlinear |
| 12 | 363 | 3635.0 | 16.673 | 3839 | 296 | exact_preserve | connectivity_wall,conv_heavy,exact_preserve,local_stencil,onehot_final_equal,open_angle |
| 13 | 107 | 3313.0 | 16.754 | 3242 | 571 | exact_preserve | exact_preserve,lut_selection,matmul,onehot_final_equal,open_angle,qlinear,scatter |
| 14 | 202 | 3287.0 | 16.902 | 3253 | 34 | unknown | connectivity_wall,onehot_final_equal,open_angle,qlinear |
| 15 | 177 | 3121.0 | 16.805 | 3500 | 121 | exact_preserve | connectivity_wall,conv_heavy,exact_preserve,local_stencil,onehot_final_equal,open_angle |
| 16 | 234 | 2950.0 | 17.010 | 2886 | 64 | semantic_or_handbuilt | connectivity_wall,onehot_final_equal |
| 17 | 036 | 1677.0 | 17.314 | 2140 | 37 | exact_preserve | connectivity_wall,exact_preserve,onehot_final_equal,open_angle |
| 18 | 392 | 1451.0 | 17.424 | 1602 | 349 | exact_preserve | custom_win,exact_preserve,onehot_final_equal,open_angle |
| 19 | 213 | 1358.0 | 17.473 | 1809 | 49 | exact_preserve | exact_preserve,onehot_final_equal,open_angle |
| 20 | 333 | 1227.0 | 16.921 | 2792 | 435 | exact_preserve | connectivity_wall,documented_wall,exact_preserve,onehot_final_equal,open_angle |
| 21 | 308 | 988.0 | 17.695 | 1385 | 103 | exact_preserve | connectivity_wall,exact_preserve,onehot_final_equal,open_angle,scatter |
| 22 | 175 | 767.0 | 17.856 | 330 | 937 | exact_preserve | exact_preserve,lut_selection,onehot_final_equal,open_angle |
| 23 | 267 | 766.0 | 18.359 | 724 | 42 | semantic_or_handbuilt | onehot_final_equal |
| 24 | 071 | 698.0 | 17.100 | 2643 | 55 | exact_preserve | connectivity_wall,documented_wall,exact_preserve,onehot_final_equal,open_angle |
| 25 | 170 | 545.0 | 17.158 | 2216 | 329 | exact_preserve | assignment_wall,conv_heavy,documented_wall,exact_preserve,local_stencil,lut_selection,onehot_final_equal,open_angle |

## uint8_presence_for_argmax

Replace fp32 Sign presence vectors feeding ArgMax with Greater+Cast(uint8)

- source tasks: 234
- candidates: 239
- expected: {'gain_type': 'memory_by_halving_presence_profile_dtype', 'risk': 'low-medium: exact only when downstream ordering needs binary nonzero presence, not counts or signed magnitude; bool must be cast to uint8 because ORT ArgMax rejects bool', 'verification': 'stored eval and fresh verify for the single task; inspect each Sign consumer before replacing'}

| rank | task | score | pts | mem | params | source | tags |
|---:|---:|---:|---:|---:|---:|---|---|
| 1 | 233 | 32165.5 | 14.588 | 32796 | 446 | semantic_or_handbuilt | assignment_wall,connectivity_wall,conv_heavy,documented_wall,gather_heavy,high_memory,low_score,matmul |
| 2 | 366 | 31466.9 | 14.640 | 30983 | 576 | exact_preserve | assignment_wall,connectivity_wall,exact_preserve,gather_heavy,high_memory,low_score,lut_selection,open_angle |
| 3 | 133 | 28025.0 | 14.703 | 27333 | 2303 | exact_preserve | assignment_wall,connectivity_wall,conv_heavy,documented_wall,exact_preserve,gather_heavy,high_memory,low_score |
| 4 | 285 | 23574.8 | 14.864 | 24684 | 550 | exact_preserve | connectivity_wall,documented_wall,exact_preserve,gather_heavy,high_memory,low_score,open_angle,scatter |
| 5 | 054 | 20655.7 | 15.078 | 20091 | 288 | unknown | gather_heavy,high_memory,scan,scatter |
| 6 | 364 | 19396.5 | 15.115 | 18500 | 1131 | exact_preserve | connectivity_wall,exact_preserve,high_memory,local_stencil,open_angle,qlinear |
| 7 | 367 | 18288.8 | 15.121 | 14800 | 4725 | semantic_or_handbuilt | connectivity_wall,conv_heavy,documented_wall,high_memory,local_stencil,lut_selection,open_angle,qlinear |
| 8 | 319 | 15367.8 | 15.251 | 16090 | 1053 | exact_preserve | assignment_wall,documented_wall,exact_preserve,gather_heavy,high_memory,scan |
| 9 | 349 | 15204.2 | 15.289 | 15300 | 1191 | semantic_or_handbuilt | conv_heavy,documented_wall,high_memory,local_stencil,lut_selection,open_angle,qlinear |
| 10 | 076 | 14771.4 | 15.285 | 16254 | 303 | exact_preserve | assignment_wall,connectivity_wall,documented_wall,exact_preserve,high_memory,open_angle,scatter |
| 11 | 191 | 11496.8 | 15.617 | 11741 | 141 | exact_preserve | assignment_wall,conv_heavy,custom_win,exact_preserve,high_memory,open_angle,qlinear |
| 12 | 198 | 10943.3 | 15.542 | 12670 | 136 | exact_preserve | connectivity_wall,documented_wall,exact_preserve,gather_heavy,high_memory,open_angle,scan |
| 13 | 138 | 10811.3 | 15.552 | 12215 | 462 | exact_preserve | conv_heavy,documented_wall,exact_preserve,high_memory,local_stencil,open_angle |
| 14 | 205 | 10806.3 | 15.676 | 10570 | 639 | exact_preserve | connectivity_wall,conv_heavy,exact_preserve,high_memory,open_angle,qlinear |
| 15 | 066 | 10187.7 | 15.778 | 9955 | 166 | semantic_or_handbuilt | marker_routed_path,open_angle,scan |
| 16 | 025 | 9999.6 | 15.748 | 10320 | 104 | exact_preserve | exact_preserve,high_memory,onehot_final_equal,open_angle,scan |
| 17 | 338 | 9248.4 | 15.725 | 10000 | 666 | semantic_or_handbuilt | connectivity_wall,documented_wall,high_memory,local_stencil,open_angle,qlinear |
| 18 | 145 | 9069.5 | 15.742 | 9244 | 1248 | semantic_or_handbuilt | connectivity_wall,custom_win,documented_wall,lut_selection,open_angle |
| 19 | 379 | 8668.9 | 15.880 | 7860 | 1273 | exact_preserve | connectivity_wall,conv_heavy,custom_win,exact_preserve,open_angle |
| 20 | 080 | 8642.5 | 15.735 | 10109 | 454 | exact_preserve | conv_heavy,documented_wall,exact_preserve,high_memory,lut_selection,open_angle,qlinear |
| 21 | 074 | 8583.2 | 15.889 | 9000 | 50 | exact_preserve | exact_preserve,local_stencil,open_angle |
| 22 | 064 | 8548.7 | 15.791 | 9852 | 134 | semantic_or_handbuilt | connectivity_wall,documented_wall,onehot_final_equal,open_angle |
| 23 | 204 | 8302.0 | 15.767 | 9780 | 452 | exact_preserve | connectivity_wall,conv_heavy,custom_win,documented_wall,exact_preserve,local_stencil,maxpool_scan,open_angle |
| 24 | 044 | 7161.7 | 15.881 | 8972 | 154 | exact_preserve | assignment_wall,connectivity_wall,documented_wall,exact_preserve,scan |
| 25 | 324 | 6921.9 | 15.907 | 7308 | 1586 | exact_preserve | assignment_wall,conv_heavy,documented_wall,exact_preserve,local_stencil,qlinear,scan |
