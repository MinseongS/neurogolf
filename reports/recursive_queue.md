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
| 1 | 349 | 13227.6 | 15.381 | 13860 | 1182 | exact_preserve | conv_heavy,documented_wall,exact_preserve,high_memory,local_stencil,lut_selection,open_angle,qlinear |
| 2 | 080 | 7892.1 | 15.806 | 9387 | 447 | exact_preserve | conv_heavy,documented_wall,exact_preserve,local_stencil,lut_selection,open_angle,qlinear |
| 3 | 279 | 5546.0 | 16.379 | 5508 | 38 | semantic_or_handbuilt | conv_heavy,local_stencil,lut_selection,qlinear |
| 4 | 328 | 4634.0 | 16.200 | 4691 | 1943 | exact_preserve | documented_wall,exact_preserve,lut_selection |
| 5 | 278 | 4512.0 | 16.480 | 4788 | 224 | exact_preserve | conv_heavy,exact_preserve,local_stencil,lut_selection,qlinear |
| 6 | 396 | 4444.0 | 16.494 | 4854 | 90 | exact_preserve | conv_heavy,exact_preserve,local_stencil,onehot_final_equal |
| 7 | 117 | 3665.0 | 16.666 | 3922 | 243 | exact_preserve | exact_preserve,local_stencil,lut_selection,open_angle,qlinear |
| 8 | 008 | 2722.0 | 16.922 | 3088 | 134 | exact_preserve | exact_preserve,lut_selection |
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
- candidates: 28
- expected: {'gain_type': 'frontier_research', 'risk': 'medium-high because it often requires semantic recompilation rather than graph surgery', 'verification': 'prove no counted full-canvas intermediates in mem profile, then fresh eval before adoption'}

| rank | task | score | pts | mem | params | source | tags |
|---:|---:|---:|---:|---:|---:|---|---|
| 1 | 054 | 20655.7 | 15.078 | 20091 | 288 | unknown | gather_heavy,high_memory,scan,scatter |
| 2 | 349 | 13227.6 | 15.381 | 13860 | 1182 | exact_preserve | conv_heavy,documented_wall,exact_preserve,high_memory,local_stencil,lut_selection,open_angle,qlinear |
| 3 | 173 | 13034.5 | 15.498 | 13307 | 77 | exact_preserve | exact_preserve,gather_heavy,high_memory,lut_selection,scatter |
| 4 | 138 | 10811.3 | 15.552 | 12215 | 462 | exact_preserve | conv_heavy,documented_wall,exact_preserve,high_memory,local_stencil,open_angle |
| 5 | 191 | 10686.2 | 15.686 | 11009 | 83 | exact_preserve | assignment_wall,conv_heavy,custom_win,exact_preserve,high_memory,open_angle,qlinear |
| 6 | 066 | 10187.7 | 15.778 | 9955 | 166 | semantic_or_handbuilt | marker_routed_path,open_angle,scan |
| 7 | 025 | 9806.1 | 15.766 | 10132 | 104 | exact_preserve | exact_preserve,high_memory,onehot_final_equal,open_angle,scan |
| 8 | 074 | 8583.2 | 15.889 | 9000 | 50 | exact_preserve | exact_preserve,local_stencil,open_angle |
| 9 | 080 | 7892.1 | 15.806 | 9387 | 447 | exact_preserve | conv_heavy,documented_wall,exact_preserve,local_stencil,lut_selection,open_angle,qlinear |
| 10 | 377 | 7712.9 | 15.987 | 8075 | 134 | exact_preserve | exact_preserve,local_stencil |
| 11 | 370 | 7604.0 | 16.000 | 7331 | 773 | exact_preserve | conv_heavy,exact_preserve,qlinear,scatter |
| 12 | 324 | 6921.9 | 15.907 | 7308 | 1586 | exact_preserve | assignment_wall,conv_heavy,documented_wall,exact_preserve,local_stencil,qlinear,scan |
| 13 | 023 | 6412.0 | 16.234 | 6163 | 249 | semantic_or_handbuilt | conv_heavy,local_stencil,open_angle,qlinear |
| 14 | 017 | 5955.0 | 16.227 | 4749 | 1706 | exact_preserve | custom_win,exact_preserve,matmul,onehot_final_equal,open_angle |
| 15 | 096 | 5682.0 | 16.053 | 7261 | 421 | exact_preserve | documented_wall,exact_preserve,open_angle,scatter |
| 16 | 279 | 5546.0 | 16.379 | 5508 | 38 | semantic_or_handbuilt | conv_heavy,local_stencil,lut_selection,qlinear |
| 17 | 382 | 5232.0 | 16.346 | 5606 | 126 | exact_preserve | exact_preserve,open_angle,scan |
| 18 | 264 | 4908.0 | 16.404 | 4700 | 708 | exact_preserve | exact_preserve,local_stencil,lut_selection,qlinear |
| 19 | 085 | 4881.0 | 16.409 | 4500 | 881 | exact_preserve | exact_preserve,local_stencil,open_angle |
| 20 | 192 | 4837.0 | 16.170 | 6193 | 644 | exact_preserve | documented_wall,exact_preserve,open_angle,qlinear |
| 21 | 005 | 4763.0 | 16.181 | 6284 | 479 | exact_preserve | conv_heavy,documented_wall,exact_preserve,lut_selection,qlinear |
| 22 | 328 | 4634.0 | 16.200 | 4691 | 1943 | exact_preserve | documented_wall,exact_preserve,lut_selection |
| 23 | 278 | 4512.0 | 16.480 | 4788 | 224 | exact_preserve | conv_heavy,exact_preserve,local_stencil,lut_selection,qlinear |
| 24 | 396 | 4444.0 | 16.494 | 4854 | 90 | exact_preserve | conv_heavy,exact_preserve,local_stencil,onehot_final_equal |
| 25 | 092 | 3899.0 | 16.406 | 4940 | 459 | semantic_or_handbuilt | documented_wall,open_angle |

## threshold_linearize_pairwise_onehot_and

Replace pairwise one-hot AND/Kronecker products with direct thresholded product scores

- source tasks: 001
- candidates: 69
- expected: {'gain_type': 'memory_by_deleting_boolean_product_carrier', 'risk': 'medium; relies on one-hot inputs, positive-threshold scoring, and class-specific margins', 'verification': 'stored eval plus fresh/adopt when generator samples exist; inspect off-footprint scores are non-positive'}

| rank | task | score | pts | mem | params | source | tags |
|---:|---:|---:|---:|---:|---:|---|---|
| 1 | 233 | 32165.5 | 14.588 | 32796 | 446 | semantic_or_handbuilt | assignment_wall,connectivity_wall,conv_heavy,documented_wall,gather_heavy,high_memory,low_score,lut_selection |
| 2 | 366 | 29966.9 | 14.640 | 30983 | 576 | exact_preserve | assignment_wall,connectivity_wall,documented_wall,exact_preserve,gather_heavy,high_memory,low_score,lut_selection |
| 3 | 133 | 19819.1 | 15.023 | 20510 | 1016 | exact_preserve | assignment_wall,connectivity_wall,conv_heavy,documented_wall,exact_preserve,high_memory,lut_selection,open_angle |
| 4 | 101 | 14620.5 | 15.422 | 13573 | 874 | semantic_or_handbuilt | assignment_wall,connectivity_wall,gather_heavy,high_memory,lut_selection,onehot_final_equal,qlinear,scatter |
| 5 | 367 | 14092.0 | 15.327 | 15300 | 590 | exact_preserve | connectivity_wall,conv_heavy,documented_wall,exact_preserve,high_memory,local_stencil,lut_selection,open_angle |
| 6 | 349 | 13227.6 | 15.381 | 13860 | 1182 | exact_preserve | conv_heavy,documented_wall,exact_preserve,high_memory,local_stencil,lut_selection,open_angle,qlinear |
| 7 | 173 | 13034.5 | 15.498 | 13307 | 77 | exact_preserve | exact_preserve,gather_heavy,high_memory,lut_selection,scatter |
| 8 | 025 | 9806.1 | 15.766 | 10132 | 104 | exact_preserve | exact_preserve,high_memory,onehot_final_equal,open_angle,scan |
| 9 | 145 | 9069.5 | 15.742 | 9244 | 1248 | semantic_or_handbuilt | connectivity_wall,custom_win,documented_wall,lut_selection,open_angle |
| 10 | 064 | 8548.7 | 15.791 | 9852 | 134 | semantic_or_handbuilt | connectivity_wall,documented_wall,onehot_final_equal,open_angle |
| 11 | 080 | 7892.1 | 15.806 | 9387 | 447 | exact_preserve | conv_heavy,documented_wall,exact_preserve,local_stencil,lut_selection,open_angle,qlinear |
| 12 | 187 | 6330.0 | 15.973 | 5000 | 3322 | exact_preserve | connectivity_wall,documented_wall,exact_preserve,local_stencil,lut_selection |
| 13 | 017 | 5955.0 | 16.227 | 4749 | 1706 | exact_preserve | custom_win,exact_preserve,matmul,onehot_final_equal,open_angle |
| 14 | 219 | 5710.0 | 16.117 | 7078 | 132 | unknown | assignment_wall,connectivity_wall,documented_wall,lut_selection,onehot_final_equal,open_angle,scan |
| 15 | 279 | 5546.0 | 16.379 | 5508 | 38 | semantic_or_handbuilt | conv_heavy,local_stencil,lut_selection,qlinear |
| 16 | 157 | 5024.0 | 16.143 | 6768 | 256 | exact_preserve | assignment_wall,connectivity_wall,documented_wall,exact_preserve,gather_heavy,lut_selection,open_angle,scatter |
| 17 | 264 | 4908.0 | 16.404 | 4700 | 708 | exact_preserve | exact_preserve,local_stencil,lut_selection,qlinear |
| 18 | 005 | 4763.0 | 16.181 | 6284 | 479 | exact_preserve | conv_heavy,documented_wall,exact_preserve,lut_selection,qlinear |
| 19 | 009 | 4699.0 | 16.190 | 6529 | 170 | exact_preserve | connectivity_wall,documented_wall,exact_preserve,local_stencil,lut_selection,onehot_final_equal,open_angle |
| 20 | 328 | 4634.0 | 16.200 | 4691 | 1943 | exact_preserve | documented_wall,exact_preserve,lut_selection |
| 21 | 278 | 4512.0 | 16.480 | 4788 | 224 | exact_preserve | conv_heavy,exact_preserve,local_stencil,lut_selection,qlinear |
| 22 | 396 | 4444.0 | 16.494 | 4854 | 90 | exact_preserve | conv_heavy,exact_preserve,local_stencil,onehot_final_equal |
| 23 | 182 | 4100.0 | 16.284 | 6024 | 76 | exact_preserve | assignment_wall,connectivity_wall,conv_heavy,documented_wall,exact_preserve,local_stencil,lut_selection,open_angle |
| 24 | 383 | 4014.0 | 16.298 | 5918 | 96 | exact_preserve | assignment_wall,connectivity_wall,documented_wall,exact_preserve,local_stencil,onehot_final_equal,open_angle |
| 25 | 002 | 3889.0 | 16.319 | 5300 | 589 | exact_preserve | connectivity_wall,documented_wall,exact_preserve,lut_selection |

## qlinear_uint8_lut_or_matmul

Replace fp16/fp32 one-hot LUT/MatMul selection with uint8 QLinearMatMul/QLinearConv where exact

- source tasks: 055 080 338 255
- candidates: 0
- expected: {'gain_type': 'memory', 'risk': 'medium', 'verification': 'stored eval, fresh eval if generator available, then adopt gate'}

| rank | task | score | pts | mem | params | source | tags |
|---:|---:|---:|---:|---:|---:|---|---|

## free_final_onehot_equal

Delay 10-channel expansion to final Equal/Where output

- source tasks: 017 095 146 381
- candidates: 21
- expected: {'gain_type': 'memory', 'risk': 'low-medium', 'verification': 'output equivalence before adoption'}

| rank | task | score | pts | mem | params | source | tags |
|---:|---:|---:|---:|---:|---:|---|---|
| 1 | 101 | 14620.5 | 15.422 | 13573 | 874 | semantic_or_handbuilt | assignment_wall,connectivity_wall,gather_heavy,high_memory,lut_selection,onehot_final_equal,qlinear,scatter |
| 2 | 025 | 9806.1 | 15.766 | 10132 | 104 | exact_preserve | exact_preserve,high_memory,onehot_final_equal,open_angle,scan |
| 3 | 064 | 8548.7 | 15.791 | 9852 | 134 | semantic_or_handbuilt | connectivity_wall,documented_wall,onehot_final_equal,open_angle |
| 4 | 017 | 5955.0 | 16.227 | 4749 | 1706 | exact_preserve | custom_win,exact_preserve,matmul,onehot_final_equal,open_angle |
| 5 | 219 | 5710.0 | 16.117 | 7078 | 132 | unknown | assignment_wall,connectivity_wall,documented_wall,lut_selection,onehot_final_equal,open_angle,scan |
| 6 | 009 | 4699.0 | 16.190 | 6529 | 170 | exact_preserve | connectivity_wall,documented_wall,exact_preserve,local_stencil,lut_selection,onehot_final_equal,open_angle |
| 7 | 396 | 4444.0 | 16.494 | 4854 | 90 | exact_preserve | conv_heavy,exact_preserve,local_stencil,onehot_final_equal |
| 8 | 383 | 4014.0 | 16.298 | 5918 | 96 | exact_preserve | assignment_wall,connectivity_wall,documented_wall,exact_preserve,local_stencil,onehot_final_equal,open_angle |
| 9 | 363 | 3635.0 | 16.673 | 3839 | 296 | exact_preserve | connectivity_wall,conv_heavy,exact_preserve,local_stencil,onehot_final_equal,open_angle |
| 10 | 202 | 3287.0 | 16.902 | 3253 | 34 | unknown | connectivity_wall,onehot_final_equal,open_angle,qlinear |
| 11 | 107 | 3194.0 | 16.786 | 3327 | 367 | exact_preserve | exact_preserve,lut_selection,matmul,onehot_final_equal,open_angle,qlinear,scatter |
| 12 | 177 | 3121.0 | 16.805 | 3500 | 121 | exact_preserve | connectivity_wall,conv_heavy,exact_preserve,local_stencil,onehot_final_equal,open_angle |
| 13 | 234 | 2434.0 | 17.016 | 2869 | 65 | exact_preserve | connectivity_wall,exact_preserve,onehot_final_equal |
| 14 | 036 | 1600.0 | 17.350 | 2061 | 39 | exact_preserve | connectivity_wall,exact_preserve,onehot_final_equal,open_angle |
| 15 | 392 | 1451.0 | 17.424 | 1602 | 349 | exact_preserve | custom_win,exact_preserve,onehot_final_equal,open_angle |
| 16 | 213 | 1351.0 | 17.477 | 1802 | 49 | exact_preserve | exact_preserve,onehot_final_equal,open_angle |
| 17 | 333 | 1223.0 | 16.922 | 2788 | 435 | exact_preserve | connectivity_wall,documented_wall,exact_preserve,onehot_final_equal,open_angle |
| 18 | 071 | 698.0 | 17.100 | 2643 | 55 | exact_preserve | connectivity_wall,documented_wall,exact_preserve,onehot_final_equal,open_angle |
| 19 | 170 | 545.0 | 17.158 | 2216 | 329 | exact_preserve | assignment_wall,conv_heavy,documented_wall,exact_preserve,local_stencil,lut_selection,onehot_final_equal,open_angle |
| 20 | 184 | 41.0 | 17.379 | 1920 | 121 | exact_preserve | connectivity_wall,conv_heavy,documented_wall,exact_preserve,local_stencil,onehot_final_equal,open_angle,qlinear |
| 21 | 029 | -35201.0 | 16.425 | 5212 | 87 | exact_preserve | exact_preserve,marginal_wall,onehot_final_equal,open_angle |

## scan_dtype_and_shift_compression

Compress scan-style MaxPool/CumSum/Hillis-Steele pipelines with lower dtype or shared shifts

- source tasks: 046 216
- candidates: 27
- expected: {'gain_type': 'memory', 'risk': 'medium-high', 'verification': 'stored/fresh eval and compare against incumbent'}

| rank | task | score | pts | mem | params | source | tags |
|---:|---:|---:|---:|---:|---:|---|---|
| 1 | 233 | 32165.5 | 14.588 | 32796 | 446 | semantic_or_handbuilt | assignment_wall,connectivity_wall,conv_heavy,documented_wall,gather_heavy,high_memory,low_score,lut_selection |
| 2 | 366 | 29966.9 | 14.640 | 30983 | 576 | exact_preserve | assignment_wall,connectivity_wall,documented_wall,exact_preserve,gather_heavy,high_memory,low_score,lut_selection |
| 3 | 285 | 23574.8 | 14.864 | 24684 | 550 | exact_preserve | connectivity_wall,documented_wall,exact_preserve,gather_heavy,high_memory,low_score,open_angle,scatter |
| 4 | 054 | 20655.7 | 15.078 | 20091 | 288 | unknown | gather_heavy,high_memory,scan,scatter |
| 5 | 319 | 15367.8 | 15.251 | 16090 | 1053 | exact_preserve | assignment_wall,documented_wall,exact_preserve,gather_heavy,high_memory,scan |
| 6 | 173 | 13034.5 | 15.498 | 13307 | 77 | exact_preserve | exact_preserve,gather_heavy,high_memory,lut_selection,scatter |
| 7 | 066 | 10187.7 | 15.778 | 9955 | 166 | semantic_or_handbuilt | marker_routed_path,open_angle,scan |
| 8 | 025 | 9806.1 | 15.766 | 10132 | 104 | exact_preserve | exact_preserve,high_memory,onehot_final_equal,open_angle,scan |
| 9 | 198 | 9514.7 | 15.658 | 11305 | 107 | exact_preserve | connectivity_wall,documented_wall,exact_preserve,gather_heavy,high_memory,local_stencil,open_angle,scan |
| 10 | 204 | 8302.0 | 15.767 | 9780 | 452 | exact_preserve | connectivity_wall,conv_heavy,custom_win,documented_wall,exact_preserve,local_stencil,maxpool_scan,open_angle |
| 11 | 338 | 7289.7 | 15.868 | 8581 | 669 | exact_preserve | connectivity_wall,documented_wall,exact_preserve,local_stencil,open_angle,qlinear,scan |
| 12 | 324 | 6921.9 | 15.907 | 7308 | 1586 | exact_preserve | assignment_wall,conv_heavy,documented_wall,exact_preserve,local_stencil,qlinear,scan |
| 13 | 219 | 5710.0 | 16.117 | 7078 | 132 | unknown | assignment_wall,connectivity_wall,documented_wall,lut_selection,onehot_final_equal,open_angle,scan |
| 14 | 382 | 5232.0 | 16.346 | 5606 | 126 | exact_preserve | exact_preserve,open_angle,scan |
| 15 | 157 | 5024.0 | 16.143 | 6768 | 256 | exact_preserve | assignment_wall,connectivity_wall,documented_wall,exact_preserve,gather_heavy,lut_selection,open_angle,scatter |
| 16 | 222 | 3298.0 | 16.425 | 5096 | 202 | exact_preserve | custom_win,documented_wall,exact_preserve,local_stencil,maxpool_scan,open_angle,qlinear |
| 17 | 277 | 3245.0 | 16.772 | 3701 | 44 | exact_preserve | connectivity_wall,exact_preserve,lut_selection,maxpool_scan |
| 18 | 055 | 3092.0 | 16.963 | 3030 | 62 | semantic_or_handbuilt | connectivity_wall,lut_selection,open_angle,qlinear,scan |
| 19 | 044 | 2984.0 | 16.486 | 4550 | 434 | exact_preserve | assignment_wall,connectivity_wall,documented_wall,exact_preserve,scan |
| 20 | 090 | 2572.0 | 16.970 | 2990 | 82 | exact_preserve | exact_preserve,scan |
| 21 | 196 | 2538.0 | 16.580 | 4500 | 38 | exact_preserve | connectivity_wall,custom_win,documented_wall,exact_preserve,local_stencil,maxpool_scan,open_angle,qlinear |
| 22 | 174 | 2378.0 | 16.616 | 3982 | 396 | exact_preserve | connectivity_wall,documented_wall,exact_preserve,matmul,open_angle,scan |
| 23 | 273 | 1660.0 | 17.322 | 2109 | 51 | exact_preserve | connectivity_wall,exact_preserve,gather_heavy,open_angle |
| 24 | 125 | 1463.0 | 16.850 | 3434 | 29 | exact_preserve | connectivity_wall,documented_wall,exact_preserve,maxpool_scan,open_angle |
| 25 | 046 | 285.0 | 17.266 | 2189 | 96 | exact_preserve | connectivity_wall,documented_wall,exact_preserve,local_stencil,scan |

## sparse_conv_single_op_floor

Collapse local neighborhood rules to one sparse Conv/QLinearConv when output is thresholded

- source tasks: 015 095 230 294
- candidates: 61
- expected: {'gain_type': 'memory+params', 'risk': 'low when rule is truly local', 'verification': 'fresh process eval; beware one-process mem0 false signals'}

| rank | task | score | pts | mem | params | source | tags |
|---:|---:|---:|---:|---:|---:|---|---|
| 1 | 349 | 13227.6 | 15.381 | 13860 | 1182 | exact_preserve | conv_heavy,documented_wall,exact_preserve,high_memory,local_stencil,lut_selection,open_angle,qlinear |
| 2 | 138 | 10811.3 | 15.552 | 12215 | 462 | exact_preserve | conv_heavy,documented_wall,exact_preserve,high_memory,local_stencil,open_angle |
| 3 | 074 | 8583.2 | 15.889 | 9000 | 50 | exact_preserve | exact_preserve,local_stencil,open_angle |
| 4 | 255 | 8508.8 | 15.897 | 8625 | 353 | exact_preserve | conv_heavy,exact_preserve,matmul,qlinear |
| 5 | 080 | 7892.1 | 15.806 | 9387 | 447 | exact_preserve | conv_heavy,documented_wall,exact_preserve,local_stencil,lut_selection,open_angle,qlinear |
| 6 | 377 | 7712.9 | 15.987 | 8075 | 134 | exact_preserve | exact_preserve,local_stencil |
| 7 | 370 | 7604.0 | 16.000 | 7331 | 773 | exact_preserve | conv_heavy,exact_preserve,qlinear,scatter |
| 8 | 023 | 6412.0 | 16.234 | 6163 | 249 | semantic_or_handbuilt | conv_heavy,local_stencil,open_angle,qlinear |
| 9 | 279 | 5546.0 | 16.379 | 5508 | 38 | semantic_or_handbuilt | conv_heavy,local_stencil,lut_selection,qlinear |
| 10 | 264 | 4908.0 | 16.404 | 4700 | 708 | exact_preserve | exact_preserve,local_stencil,lut_selection,qlinear |
| 11 | 085 | 4881.0 | 16.409 | 4500 | 881 | exact_preserve | exact_preserve,local_stencil,open_angle |
| 12 | 005 | 4763.0 | 16.181 | 6284 | 479 | exact_preserve | conv_heavy,documented_wall,exact_preserve,lut_selection,qlinear |
| 13 | 278 | 4512.0 | 16.480 | 4788 | 224 | exact_preserve | conv_heavy,exact_preserve,local_stencil,lut_selection,qlinear |
| 14 | 396 | 4444.0 | 16.494 | 4854 | 90 | exact_preserve | conv_heavy,exact_preserve,local_stencil,onehot_final_equal |
| 15 | 117 | 3665.0 | 16.666 | 3922 | 243 | exact_preserve | exact_preserve,local_stencil,lut_selection,open_angle,qlinear |
| 16 | 132 | 3589.0 | 16.684 | 3990 | 99 | exact_preserve | bitwise_program,exact_preserve,gather_heavy,local_stencil,open_angle |
| 17 | 162 | 3568.0 | 16.689 | 4024 | 44 | exact_preserve | exact_preserve,local_stencil,open_angle,qlinear |
| 18 | 222 | 3298.0 | 16.425 | 5096 | 202 | exact_preserve | custom_win,documented_wall,exact_preserve,local_stencil,maxpool_scan,open_angle,qlinear |
| 19 | 284 | 3149.0 | 16.798 | 2996 | 653 | exact_preserve | conv_heavy,custom_win,exact_preserve,open_angle,scatter |
| 20 | 079 | 2633.0 | 16.950 | 3055 | 78 | exact_preserve | custom_win,exact_preserve,local_stencil,open_angle |
| 21 | 042 | 2610.0 | 16.958 | 2792 | 318 | exact_preserve | conv_heavy,exact_preserve,local_stencil,qlinear |
| 22 | 091 | 2528.0 | 16.984 | 2853 | 175 | exact_preserve | exact_preserve,local_stencil,lut_selection,open_angle |
| 23 | 019 | 2298.0 | 17.063 | 2709 | 89 | exact_preserve | exact_preserve,local_stencil,qlinear |
| 24 | 397 | 2282.0 | 17.069 | 2582 | 200 | exact_preserve | exact_preserve,local_stencil,scatter |
| 25 | 224 | 2272.0 | 17.073 | 2400 | 372 | exact_preserve | conv_heavy,exact_preserve,local_stencil,open_angle |

## exact_preserve_to_semantic_rewrite

Prioritize exact-preserve source builders for semantic replacement

- source tasks: 002 018 286 366
- candidates: 16
- expected: {'gain_type': 'research', 'risk': 'high', 'verification': 'document semantic mechanism before implementation'}

| rank | task | score | pts | mem | params | source | tags |
|---:|---:|---:|---:|---:|---:|---|---|
| 1 | 366 | 29966.9 | 14.640 | 30983 | 576 | exact_preserve | assignment_wall,connectivity_wall,documented_wall,exact_preserve,gather_heavy,high_memory,low_score,lut_selection |
| 2 | 285 | 23574.8 | 14.864 | 24684 | 550 | exact_preserve | connectivity_wall,documented_wall,exact_preserve,gather_heavy,high_memory,low_score,open_angle,scatter |
| 3 | 133 | 19819.1 | 15.023 | 20510 | 1016 | exact_preserve | assignment_wall,connectivity_wall,conv_heavy,documented_wall,exact_preserve,high_memory,lut_selection,open_angle |
| 4 | 364 | 16713.1 | 15.260 | 15860 | 1131 | exact_preserve | connectivity_wall,exact_preserve,high_memory,local_stencil,open_angle,qlinear |
| 5 | 319 | 15367.8 | 15.251 | 16090 | 1053 | exact_preserve | assignment_wall,documented_wall,exact_preserve,gather_heavy,high_memory,scan |
| 6 | 076 | 14134.8 | 15.324 | 15629 | 303 | exact_preserve | assignment_wall,connectivity_wall,documented_wall,exact_preserve,high_memory,open_angle,scatter |
| 7 | 367 | 14092.0 | 15.327 | 15300 | 590 | exact_preserve | connectivity_wall,conv_heavy,documented_wall,exact_preserve,high_memory,local_stencil,lut_selection,open_angle |
| 8 | 349 | 13227.6 | 15.381 | 13860 | 1182 | exact_preserve | conv_heavy,documented_wall,exact_preserve,high_memory,local_stencil,lut_selection,open_angle,qlinear |
| 9 | 173 | 13034.5 | 15.498 | 13307 | 77 | exact_preserve | exact_preserve,gather_heavy,high_memory,lut_selection,scatter |
| 10 | 138 | 10811.3 | 15.552 | 12215 | 462 | exact_preserve | conv_heavy,documented_wall,exact_preserve,high_memory,local_stencil,open_angle |
| 11 | 191 | 10686.2 | 15.686 | 11009 | 83 | exact_preserve | assignment_wall,conv_heavy,custom_win,exact_preserve,high_memory,open_angle,qlinear |
| 12 | 025 | 9806.1 | 15.766 | 10132 | 104 | exact_preserve | exact_preserve,high_memory,onehot_final_equal,open_angle,scan |
| 13 | 198 | 9514.7 | 15.658 | 11305 | 107 | exact_preserve | connectivity_wall,documented_wall,exact_preserve,gather_heavy,high_memory,local_stencil,open_angle,scan |
| 14 | 158 | -20983.1 | 15.060 | 18089 | 2646 | exact_preserve | assignment_wall,conv_heavy,documented_wall,exact_preserve,gather_heavy,high_memory,marginal_wall,open_angle |
| 15 | 018 | -76211.7 | 14.856 | 24029 | 1416 | exact_preserve | ambiguity_wall,documented_wall,exact_preserve,gather_heavy,high_memory,low_score,maxpool_scan,scatter |
| 16 | 118 | -89592.2 | 15.584 | 12153 | 130 | exact_preserve | connectivity_wall,conv_heavy,documented_wall,exact_preserve,high_memory,infeasible_exact_wall,information_loss_wall,local_stencil |

## dihedral_template_match_stacked_conv

Use one stacked Conv for all dihedral orientations of a small runtime-extracted template

- source tasks: 191
- candidates: 0
- expected: {'gain_type': 'memory+accuracy', 'risk': 'medium', 'verification': 'fresh exact eval; compare against incumbent and propagate if template extraction is deterministic'}

| rank | task | score | pts | mem | params | source | tags |
|---:|---:|---:|---:|---:|---:|---|---|

## bounded_crop_before_connectivity_scan

Run full-grid logic on the generator's true max canvas, not the 30x30 harness canvas

- source tasks: 187 173
- candidates: 19
- expected: {'gain_type': 'memory+generalization', 'risk': 'medium', 'verification': 'confirm bundled/generator max height/width, stored eval, and final Pad-to-30 sentinel semantics; fresh is diagnostic only in 8000 mode'}

| rank | task | score | pts | mem | params | source | tags |
|---:|---:|---:|---:|---:|---:|---|---|
| 1 | 233 | 32165.5 | 14.588 | 32796 | 446 | semantic_or_handbuilt | assignment_wall,connectivity_wall,conv_heavy,documented_wall,gather_heavy,high_memory,low_score,lut_selection |
| 2 | 366 | 29966.9 | 14.640 | 30983 | 576 | exact_preserve | assignment_wall,connectivity_wall,documented_wall,exact_preserve,gather_heavy,high_memory,low_score,lut_selection |
| 3 | 285 | 23574.8 | 14.864 | 24684 | 550 | exact_preserve | connectivity_wall,documented_wall,exact_preserve,gather_heavy,high_memory,low_score,open_angle,scatter |
| 4 | 286 | 22796.4 | 14.915 | 21090 | 2881 | semantic_or_handbuilt | connectivity_wall,conv_heavy,documented_wall,high_memory,local_stencil,low_score,qlinear |
| 5 | 054 | 20655.7 | 15.078 | 20091 | 288 | unknown | gather_heavy,high_memory,scan,scatter |
| 6 | 133 | 19819.1 | 15.023 | 20510 | 1016 | exact_preserve | assignment_wall,connectivity_wall,conv_heavy,documented_wall,exact_preserve,high_memory,lut_selection,open_angle |
| 7 | 319 | 15367.8 | 15.251 | 16090 | 1053 | exact_preserve | assignment_wall,documented_wall,exact_preserve,gather_heavy,high_memory,scan |
| 8 | 101 | 14620.5 | 15.422 | 13573 | 874 | semantic_or_handbuilt | assignment_wall,connectivity_wall,gather_heavy,high_memory,lut_selection,onehot_final_equal,qlinear,scatter |
| 9 | 076 | 14134.8 | 15.324 | 15629 | 303 | exact_preserve | assignment_wall,connectivity_wall,documented_wall,exact_preserve,high_memory,open_angle,scatter |
| 10 | 367 | 14092.0 | 15.327 | 15300 | 590 | exact_preserve | connectivity_wall,conv_heavy,documented_wall,exact_preserve,high_memory,local_stencil,lut_selection,open_angle |
| 11 | 349 | 13227.6 | 15.381 | 13860 | 1182 | exact_preserve | conv_heavy,documented_wall,exact_preserve,high_memory,local_stencil,lut_selection,open_angle,qlinear |
| 12 | 173 | 13034.5 | 15.498 | 13307 | 77 | exact_preserve | exact_preserve,gather_heavy,high_memory,lut_selection,scatter |
| 13 | 138 | 10811.3 | 15.552 | 12215 | 462 | exact_preserve | conv_heavy,documented_wall,exact_preserve,high_memory,local_stencil,open_angle |
| 14 | 191 | 10686.2 | 15.686 | 11009 | 83 | exact_preserve | assignment_wall,conv_heavy,custom_win,exact_preserve,high_memory,open_angle,qlinear |
| 15 | 025 | 9806.1 | 15.766 | 10132 | 104 | exact_preserve | exact_preserve,high_memory,onehot_final_equal,open_angle,scan |
| 16 | 198 | 9514.7 | 15.658 | 11305 | 107 | exact_preserve | connectivity_wall,documented_wall,exact_preserve,gather_heavy,high_memory,local_stencil,open_angle,scan |
| 17 | 158 | -20983.1 | 15.060 | 18089 | 2646 | exact_preserve | assignment_wall,conv_heavy,documented_wall,exact_preserve,gather_heavy,high_memory,marginal_wall,open_angle |
| 18 | 018 | -76211.7 | 14.856 | 24029 | 1416 | exact_preserve | ambiguity_wall,documented_wall,exact_preserve,gather_heavy,high_memory,low_score,maxpool_scan,scatter |
| 19 | 118 | -89592.2 | 15.584 | 12153 | 130 | exact_preserve | connectivity_wall,conv_heavy,documented_wall,exact_preserve,high_memory,infeasible_exact_wall,information_loss_wall,local_stencil |

## public_teacher_bitwise_scan_replacement

Replace repeated MaxPool/Max/Min scan stacks with bitwise shift/mask routing when the state is binary

- source tasks: 002 209
- candidates: 8
- expected: {'gain_type': 'memory', 'risk': 'medium-high', 'verification': 'stored eval, fresh eval when generator available, public-probe if strict fresh is known weak'}

| rank | task | score | pts | mem | params | source | tags |
|---:|---:|---:|---:|---:|---:|---|---|
| 1 | 233 | 32165.5 | 14.588 | 32796 | 446 | semantic_or_handbuilt | assignment_wall,connectivity_wall,conv_heavy,documented_wall,gather_heavy,high_memory,low_score,lut_selection |
| 2 | 285 | 23574.8 | 14.864 | 24684 | 550 | exact_preserve | connectivity_wall,documented_wall,exact_preserve,gather_heavy,high_memory,low_score,open_angle,scatter |
| 3 | 054 | 20655.7 | 15.078 | 20091 | 288 | unknown | gather_heavy,high_memory,scan,scatter |
| 4 | 133 | 19819.1 | 15.023 | 20510 | 1016 | exact_preserve | assignment_wall,connectivity_wall,conv_heavy,documented_wall,exact_preserve,high_memory,lut_selection,open_angle |
| 5 | 076 | 14134.8 | 15.324 | 15629 | 303 | exact_preserve | assignment_wall,connectivity_wall,documented_wall,exact_preserve,high_memory,open_angle,scatter |
| 6 | 367 | 14092.0 | 15.327 | 15300 | 590 | exact_preserve | connectivity_wall,conv_heavy,documented_wall,exact_preserve,high_memory,local_stencil,lut_selection,open_angle |
| 7 | 158 | -20983.1 | 15.060 | 18089 | 2646 | exact_preserve | assignment_wall,conv_heavy,documented_wall,exact_preserve,gather_heavy,high_memory,marginal_wall,open_angle |
| 8 | 018 | -76211.7 | 14.856 | 24029 | 1416 | exact_preserve | ambiguity_wall,documented_wall,exact_preserve,gather_heavy,high_memory,low_score,maxpool_scan,scatter |

## public_teacher_qlinear_conv_rewrite

Convert repeated binary/count Conv/ConvTranspose towers to QLinearConv/uint8 routes

- source tasks: 023 349 182 233 255 364 338 004
- candidates: 31
- expected: {'gain_type': 'memory+params', 'risk': 'medium', 'verification': 'stored eval plus fresh; inspect for output-range saturation before adoption'}

| rank | task | score | pts | mem | params | source | tags |
|---:|---:|---:|---:|---:|---:|---|---|
| 1 | 286 | 22796.4 | 14.915 | 21090 | 2881 | semantic_or_handbuilt | connectivity_wall,conv_heavy,documented_wall,high_memory,local_stencil,low_score,qlinear |
| 2 | 138 | 10811.3 | 15.552 | 12215 | 462 | exact_preserve | conv_heavy,documented_wall,exact_preserve,high_memory,local_stencil,open_angle |
| 3 | 205 | 9544.6 | 15.791 | 9498 | 484 | exact_preserve | connectivity_wall,conv_heavy,exact_preserve,open_angle,qlinear |
| 4 | 198 | 9514.7 | 15.658 | 11305 | 107 | exact_preserve | connectivity_wall,documented_wall,exact_preserve,gather_heavy,high_memory,local_stencil,open_angle,scan |
| 5 | 379 | 8668.9 | 15.880 | 7860 | 1273 | exact_preserve | connectivity_wall,conv_heavy,custom_win,exact_preserve,open_angle |
| 6 | 074 | 8583.2 | 15.889 | 9000 | 50 | exact_preserve | exact_preserve,local_stencil,open_angle |
| 7 | 080 | 7892.1 | 15.806 | 9387 | 447 | exact_preserve | conv_heavy,documented_wall,exact_preserve,local_stencil,lut_selection,open_angle,qlinear |
| 8 | 377 | 7712.9 | 15.987 | 8075 | 134 | exact_preserve | exact_preserve,local_stencil |
| 9 | 338 | 7289.7 | 15.868 | 8581 | 669 | exact_preserve | connectivity_wall,documented_wall,exact_preserve,local_stencil,open_angle,qlinear,scan |
| 10 | 187 | 6330.0 | 15.973 | 5000 | 3322 | exact_preserve | connectivity_wall,documented_wall,exact_preserve,local_stencil,lut_selection |
| 11 | 085 | 4881.0 | 16.409 | 4500 | 881 | exact_preserve | exact_preserve,local_stencil,open_angle |
| 12 | 005 | 4763.0 | 16.181 | 6284 | 479 | exact_preserve | conv_heavy,documented_wall,exact_preserve,lut_selection,qlinear |
| 13 | 009 | 4699.0 | 16.190 | 6529 | 170 | exact_preserve | connectivity_wall,documented_wall,exact_preserve,local_stencil,lut_selection,onehot_final_equal,open_angle |
| 14 | 278 | 4512.0 | 16.480 | 4788 | 224 | exact_preserve | conv_heavy,exact_preserve,local_stencil,lut_selection,qlinear |
| 15 | 396 | 4444.0 | 16.494 | 4854 | 90 | exact_preserve | conv_heavy,exact_preserve,local_stencil,onehot_final_equal |
| 16 | 363 | 3635.0 | 16.673 | 3839 | 296 | exact_preserve | connectivity_wall,conv_heavy,exact_preserve,local_stencil,onehot_final_equal,open_angle |
| 17 | 132 | 3589.0 | 16.684 | 3990 | 99 | exact_preserve | bitwise_program,exact_preserve,gather_heavy,local_stencil,open_angle |
| 18 | 222 | 3298.0 | 16.425 | 5096 | 202 | exact_preserve | custom_win,documented_wall,exact_preserve,local_stencil,maxpool_scan,open_angle,qlinear |
| 19 | 284 | 3149.0 | 16.798 | 2996 | 653 | exact_preserve | conv_heavy,custom_win,exact_preserve,open_angle,scatter |
| 20 | 004 | 3121.0 | 16.459 | 5028 | 93 | exact_preserve | connectivity_wall,documented_wall,exact_preserve,local_stencil,lut_selection,open_angle,qlinear |
| 21 | 177 | 3121.0 | 16.805 | 3500 | 121 | exact_preserve | connectivity_wall,conv_heavy,exact_preserve,local_stencil,onehot_final_equal,open_angle |
| 22 | 079 | 2633.0 | 16.950 | 3055 | 78 | exact_preserve | custom_win,exact_preserve,local_stencil,open_angle |
| 23 | 042 | 2610.0 | 16.958 | 2792 | 318 | exact_preserve | conv_heavy,exact_preserve,local_stencil,qlinear |
| 24 | 091 | 2528.0 | 16.984 | 2853 | 175 | exact_preserve | exact_preserve,local_stencil,lut_selection,open_angle |
| 25 | 397 | 2282.0 | 17.069 | 2582 | 200 | exact_preserve | exact_preserve,local_stencil,scatter |

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
- candidates: 6
- expected: {'gain_type': 'semantic_rewrite_memory', 'risk': 'medium-high until bbox/component edge cases are fully fresh-verified', 'verification': 'Python reference stored plus large fresh, then source-owned ONNX compiler and adopt gate'}

| rank | task | score | pts | mem | params | source | tags |
|---:|---:|---:|---:|---:|---:|---|---|
| 1 | 285 | 23574.8 | 14.864 | 24684 | 550 | exact_preserve | connectivity_wall,documented_wall,exact_preserve,gather_heavy,high_memory,low_score,open_angle,scatter |
| 2 | 286 | 22796.4 | 14.915 | 21090 | 2881 | semantic_or_handbuilt | connectivity_wall,conv_heavy,documented_wall,high_memory,local_stencil,low_score,qlinear |
| 3 | 054 | 20655.7 | 15.078 | 20091 | 288 | unknown | gather_heavy,high_memory,scan,scatter |
| 4 | 364 | 16713.1 | 15.260 | 15860 | 1131 | exact_preserve | connectivity_wall,exact_preserve,high_memory,local_stencil,open_angle,qlinear |
| 5 | 367 | 14092.0 | 15.327 | 15300 | 590 | exact_preserve | connectivity_wall,conv_heavy,documented_wall,exact_preserve,high_memory,local_stencil,lut_selection,open_angle |
| 6 | 018 | -76211.7 | 14.856 | 24029 | 1416 | exact_preserve | ambiguity_wall,documented_wall,exact_preserve,gather_heavy,high_memory,low_score,maxpool_scan,scatter |

## band_profile_contraction_final_equal

Use distinct-colour row/column profiles to avoid full connectivity for band-fill tasks

- source tasks: 202
- candidates: 19
- expected: {'gain_type': 'semantic_rewrite_memory_or_source_ownership', 'risk': 'medium; requires generator proof that colours are distinct per band/profile', 'verification': 'stored eval plus large fresh generator eval; source/live reconcile after rewrite'}

| rank | task | score | pts | mem | params | source | tags |
|---:|---:|---:|---:|---:|---:|---|---|
| 1 | 054 | 20655.7 | 15.078 | 20091 | 288 | unknown | gather_heavy,high_memory,scan,scatter |
| 2 | 349 | 13227.6 | 15.381 | 13860 | 1182 | exact_preserve | conv_heavy,documented_wall,exact_preserve,high_memory,local_stencil,lut_selection,open_angle,qlinear |
| 3 | 173 | 13034.5 | 15.498 | 13307 | 77 | exact_preserve | exact_preserve,gather_heavy,high_memory,lut_selection,scatter |
| 4 | 138 | 10811.3 | 15.552 | 12215 | 462 | exact_preserve | conv_heavy,documented_wall,exact_preserve,high_memory,local_stencil,open_angle |
| 5 | 025 | 9806.1 | 15.766 | 10132 | 104 | exact_preserve | exact_preserve,high_memory,onehot_final_equal,open_angle,scan |
| 6 | 074 | 8583.2 | 15.889 | 9000 | 50 | exact_preserve | exact_preserve,local_stencil,open_angle |
| 7 | 080 | 7892.1 | 15.806 | 9387 | 447 | exact_preserve | conv_heavy,documented_wall,exact_preserve,local_stencil,lut_selection,open_angle,qlinear |
| 8 | 377 | 7712.9 | 15.987 | 8075 | 134 | exact_preserve | exact_preserve,local_stencil |
| 9 | 023 | 6412.0 | 16.234 | 6163 | 249 | semantic_or_handbuilt | conv_heavy,local_stencil,open_angle,qlinear |
| 10 | 017 | 5955.0 | 16.227 | 4749 | 1706 | exact_preserve | custom_win,exact_preserve,matmul,onehot_final_equal,open_angle |
| 11 | 279 | 5546.0 | 16.379 | 5508 | 38 | semantic_or_handbuilt | conv_heavy,local_stencil,lut_selection,qlinear |
| 12 | 264 | 4908.0 | 16.404 | 4700 | 708 | exact_preserve | exact_preserve,local_stencil,lut_selection,qlinear |
| 13 | 085 | 4881.0 | 16.409 | 4500 | 881 | exact_preserve | exact_preserve,local_stencil,open_angle |
| 14 | 278 | 4512.0 | 16.480 | 4788 | 224 | exact_preserve | conv_heavy,exact_preserve,local_stencil,lut_selection,qlinear |
| 15 | 396 | 4444.0 | 16.494 | 4854 | 90 | exact_preserve | conv_heavy,exact_preserve,local_stencil,onehot_final_equal |
| 16 | 162 | 3568.0 | 16.689 | 4024 | 44 | exact_preserve | exact_preserve,local_stencil,open_angle,qlinear |
| 17 | 222 | 3298.0 | 16.425 | 5096 | 202 | exact_preserve | custom_win,documented_wall,exact_preserve,local_stencil,maxpool_scan,open_angle,qlinear |
| 18 | 029 | -35201.0 | 16.425 | 5212 | 87 | exact_preserve | exact_preserve,marginal_wall,onehot_final_equal,open_angle |
| 19 | 018 | -76211.7 | 14.856 | 24029 | 1416 | exact_preserve | ambiguity_wall,documented_wall,exact_preserve,gather_heavy,high_memory,low_score,maxpool_scan,scatter |

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
- candidates: 171
- expected: {'gain_type': 'memory+params', 'risk': 'low when concatenated tails are constant zero/false and Pad fill defaults to zero', 'verification': 'stored eval plus side-by-side fresh output equivalence against incumbent'}

| rank | task | score | pts | mem | params | source | tags |
|---:|---:|---:|---:|---:|---:|---|---|
| 1 | 233 | 32165.5 | 14.588 | 32796 | 446 | semantic_or_handbuilt | assignment_wall,connectivity_wall,conv_heavy,documented_wall,gather_heavy,high_memory,low_score,lut_selection |
| 2 | 366 | 29966.9 | 14.640 | 30983 | 576 | exact_preserve | assignment_wall,connectivity_wall,documented_wall,exact_preserve,gather_heavy,high_memory,low_score,lut_selection |
| 3 | 285 | 23574.8 | 14.864 | 24684 | 550 | exact_preserve | connectivity_wall,documented_wall,exact_preserve,gather_heavy,high_memory,low_score,open_angle,scatter |
| 4 | 286 | 22796.4 | 14.915 | 21090 | 2881 | semantic_or_handbuilt | connectivity_wall,conv_heavy,documented_wall,high_memory,local_stencil,low_score,qlinear |
| 5 | 054 | 20655.7 | 15.078 | 20091 | 288 | unknown | gather_heavy,high_memory,scan,scatter |
| 6 | 133 | 19819.1 | 15.023 | 20510 | 1016 | exact_preserve | assignment_wall,connectivity_wall,conv_heavy,documented_wall,exact_preserve,high_memory,lut_selection,open_angle |
| 7 | 364 | 16713.1 | 15.260 | 15860 | 1131 | exact_preserve | connectivity_wall,exact_preserve,high_memory,local_stencil,open_angle,qlinear |
| 8 | 319 | 15367.8 | 15.251 | 16090 | 1053 | exact_preserve | assignment_wall,documented_wall,exact_preserve,gather_heavy,high_memory,scan |
| 9 | 101 | 14620.5 | 15.422 | 13573 | 874 | semantic_or_handbuilt | assignment_wall,connectivity_wall,gather_heavy,high_memory,lut_selection,onehot_final_equal,qlinear,scatter |
| 10 | 076 | 14134.8 | 15.324 | 15629 | 303 | exact_preserve | assignment_wall,connectivity_wall,documented_wall,exact_preserve,high_memory,open_angle,scatter |
| 11 | 367 | 14092.0 | 15.327 | 15300 | 590 | exact_preserve | connectivity_wall,conv_heavy,documented_wall,exact_preserve,high_memory,local_stencil,lut_selection,open_angle |
| 12 | 349 | 13227.6 | 15.381 | 13860 | 1182 | exact_preserve | conv_heavy,documented_wall,exact_preserve,high_memory,local_stencil,lut_selection,open_angle,qlinear |
| 13 | 173 | 13034.5 | 15.498 | 13307 | 77 | exact_preserve | exact_preserve,gather_heavy,high_memory,lut_selection,scatter |
| 14 | 138 | 10811.3 | 15.552 | 12215 | 462 | exact_preserve | conv_heavy,documented_wall,exact_preserve,high_memory,local_stencil,open_angle |
| 15 | 191 | 10686.2 | 15.686 | 11009 | 83 | exact_preserve | assignment_wall,conv_heavy,custom_win,exact_preserve,high_memory,open_angle,qlinear |
| 16 | 066 | 10187.7 | 15.778 | 9955 | 166 | semantic_or_handbuilt | marker_routed_path,open_angle,scan |
| 17 | 025 | 9806.1 | 15.766 | 10132 | 104 | exact_preserve | exact_preserve,high_memory,onehot_final_equal,open_angle,scan |
| 18 | 205 | 9544.6 | 15.791 | 9498 | 484 | exact_preserve | connectivity_wall,conv_heavy,exact_preserve,open_angle,qlinear |
| 19 | 198 | 9514.7 | 15.658 | 11305 | 107 | exact_preserve | connectivity_wall,documented_wall,exact_preserve,gather_heavy,high_memory,local_stencil,open_angle,scan |
| 20 | 145 | 9069.5 | 15.742 | 9244 | 1248 | semantic_or_handbuilt | connectivity_wall,custom_win,documented_wall,lut_selection,open_angle |
| 21 | 379 | 8668.9 | 15.880 | 7860 | 1273 | exact_preserve | connectivity_wall,conv_heavy,custom_win,exact_preserve,open_angle |
| 22 | 064 | 8548.7 | 15.791 | 9852 | 134 | semantic_or_handbuilt | connectivity_wall,documented_wall,onehot_final_equal,open_angle |
| 23 | 204 | 8302.0 | 15.767 | 9780 | 452 | exact_preserve | connectivity_wall,conv_heavy,custom_win,documented_wall,exact_preserve,local_stencil,maxpool_scan,open_angle |
| 24 | 080 | 7892.1 | 15.806 | 9387 | 447 | exact_preserve | conv_heavy,documented_wall,exact_preserve,local_stencil,lut_selection,open_angle,qlinear |
| 25 | 370 | 7604.0 | 16.000 | 7331 | 773 | exact_preserve | conv_heavy,exact_preserve,qlinear,scatter |

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
- candidates: 205
- expected: {'gain_type': 'memory+params', 'risk': 'low when every output on-cell is copied from the original one-hot input and off-grid can point at an all-zero padded input coordinate', 'verification': 'stored eval plus fresh side-by-side; confirm black in-grid cells and off-grid all-zero semantics are both preserved'}

| rank | task | score | pts | mem | params | source | tags |
|---:|---:|---:|---:|---:|---:|---|---|
| 1 | 233 | 32165.5 | 14.588 | 32796 | 446 | semantic_or_handbuilt | assignment_wall,connectivity_wall,conv_heavy,documented_wall,gather_heavy,high_memory,low_score,lut_selection |
| 2 | 366 | 29966.9 | 14.640 | 30983 | 576 | exact_preserve | assignment_wall,connectivity_wall,documented_wall,exact_preserve,gather_heavy,high_memory,low_score,lut_selection |
| 3 | 285 | 23574.8 | 14.864 | 24684 | 550 | exact_preserve | connectivity_wall,documented_wall,exact_preserve,gather_heavy,high_memory,low_score,open_angle,scatter |
| 4 | 133 | 19819.1 | 15.023 | 20510 | 1016 | exact_preserve | assignment_wall,connectivity_wall,conv_heavy,documented_wall,exact_preserve,high_memory,lut_selection,open_angle |
| 5 | 364 | 16713.1 | 15.260 | 15860 | 1131 | exact_preserve | connectivity_wall,exact_preserve,high_memory,local_stencil,open_angle,qlinear |
| 6 | 101 | 14620.5 | 15.422 | 13573 | 874 | semantic_or_handbuilt | assignment_wall,connectivity_wall,gather_heavy,high_memory,lut_selection,onehot_final_equal,qlinear,scatter |
| 7 | 076 | 14134.8 | 15.324 | 15629 | 303 | exact_preserve | assignment_wall,connectivity_wall,documented_wall,exact_preserve,high_memory,open_angle,scatter |
| 8 | 367 | 14092.0 | 15.327 | 15300 | 590 | exact_preserve | connectivity_wall,conv_heavy,documented_wall,exact_preserve,high_memory,local_stencil,lut_selection,open_angle |
| 9 | 349 | 13227.6 | 15.381 | 13860 | 1182 | exact_preserve | conv_heavy,documented_wall,exact_preserve,high_memory,local_stencil,lut_selection,open_angle,qlinear |
| 10 | 173 | 13034.5 | 15.498 | 13307 | 77 | exact_preserve | exact_preserve,gather_heavy,high_memory,lut_selection,scatter |
| 11 | 138 | 10811.3 | 15.552 | 12215 | 462 | exact_preserve | conv_heavy,documented_wall,exact_preserve,high_memory,local_stencil,open_angle |
| 12 | 191 | 10686.2 | 15.686 | 11009 | 83 | exact_preserve | assignment_wall,conv_heavy,custom_win,exact_preserve,high_memory,open_angle,qlinear |
| 13 | 066 | 10187.7 | 15.778 | 9955 | 166 | semantic_or_handbuilt | marker_routed_path,open_angle,scan |
| 14 | 025 | 9806.1 | 15.766 | 10132 | 104 | exact_preserve | exact_preserve,high_memory,onehot_final_equal,open_angle,scan |
| 15 | 205 | 9544.6 | 15.791 | 9498 | 484 | exact_preserve | connectivity_wall,conv_heavy,exact_preserve,open_angle,qlinear |
| 16 | 198 | 9514.7 | 15.658 | 11305 | 107 | exact_preserve | connectivity_wall,documented_wall,exact_preserve,gather_heavy,high_memory,local_stencil,open_angle,scan |
| 17 | 145 | 9069.5 | 15.742 | 9244 | 1248 | semantic_or_handbuilt | connectivity_wall,custom_win,documented_wall,lut_selection,open_angle |
| 18 | 379 | 8668.9 | 15.880 | 7860 | 1273 | exact_preserve | connectivity_wall,conv_heavy,custom_win,exact_preserve,open_angle |
| 19 | 074 | 8583.2 | 15.889 | 9000 | 50 | exact_preserve | exact_preserve,local_stencil,open_angle |
| 20 | 064 | 8548.7 | 15.791 | 9852 | 134 | semantic_or_handbuilt | connectivity_wall,documented_wall,onehot_final_equal,open_angle |
| 21 | 204 | 8302.0 | 15.767 | 9780 | 452 | exact_preserve | connectivity_wall,conv_heavy,custom_win,documented_wall,exact_preserve,local_stencil,maxpool_scan,open_angle |
| 22 | 080 | 7892.1 | 15.806 | 9387 | 447 | exact_preserve | conv_heavy,documented_wall,exact_preserve,local_stencil,lut_selection,open_angle,qlinear |
| 23 | 338 | 7289.7 | 15.868 | 8581 | 669 | exact_preserve | connectivity_wall,documented_wall,exact_preserve,local_stencil,open_angle,qlinear,scan |
| 24 | 216 | 6679.9 | 15.934 | 8584 | 76 | exact_preserve | connectivity_wall,documented_wall,exact_preserve,open_angle,qlinear,scatter |
| 25 | 023 | 6412.0 | 16.234 | 6163 | 249 | semantic_or_handbuilt | conv_heavy,local_stencil,open_angle,qlinear |

## uint8_topk_compact_label_grid

Run TopK directly on compact uint8 label grids when only nonzero-cell enumeration is needed

- source tasks: 285 366 233 173 076 018
- candidates: 20
- expected: {'gain_type': 'memory', 'risk': 'medium because TopK tie ordering and dtype support must be checked in ORT', 'verification': 'stored eval plus large side-by-side incumbent equivalence; truth-fresh failure shared with incumbent is not by itself a reject'}

| rank | task | score | pts | mem | params | source | tags |
|---:|---:|---:|---:|---:|---:|---|---|
| 1 | 233 | 32165.5 | 14.588 | 32796 | 446 | semantic_or_handbuilt | assignment_wall,connectivity_wall,conv_heavy,documented_wall,gather_heavy,high_memory,low_score,lut_selection |
| 2 | 366 | 29966.9 | 14.640 | 30983 | 576 | exact_preserve | assignment_wall,connectivity_wall,documented_wall,exact_preserve,gather_heavy,high_memory,low_score,lut_selection |
| 3 | 285 | 23574.8 | 14.864 | 24684 | 550 | exact_preserve | connectivity_wall,documented_wall,exact_preserve,gather_heavy,high_memory,low_score,open_angle,scatter |
| 4 | 319 | 15367.8 | 15.251 | 16090 | 1053 | exact_preserve | assignment_wall,documented_wall,exact_preserve,gather_heavy,high_memory,scan |
| 5 | 101 | 14620.5 | 15.422 | 13573 | 874 | semantic_or_handbuilt | assignment_wall,connectivity_wall,gather_heavy,high_memory,lut_selection,onehot_final_equal,qlinear,scatter |
| 6 | 076 | 14134.8 | 15.324 | 15629 | 303 | exact_preserve | assignment_wall,connectivity_wall,documented_wall,exact_preserve,high_memory,open_angle,scatter |
| 7 | 173 | 13034.5 | 15.498 | 13307 | 77 | exact_preserve | exact_preserve,gather_heavy,high_memory,lut_selection,scatter |
| 8 | 025 | 9806.1 | 15.766 | 10132 | 104 | exact_preserve | exact_preserve,high_memory,onehot_final_equal,open_angle,scan |
| 9 | 064 | 8548.7 | 15.791 | 9852 | 134 | semantic_or_handbuilt | connectivity_wall,documented_wall,onehot_final_equal,open_angle |
| 10 | 096 | 5682.0 | 16.053 | 7261 | 421 | exact_preserve | documented_wall,exact_preserve,open_angle,scatter |
| 11 | 157 | 5024.0 | 16.143 | 6768 | 256 | exact_preserve | assignment_wall,connectivity_wall,documented_wall,exact_preserve,gather_heavy,lut_selection,open_angle,scatter |
| 12 | 368 | 4647.0 | 16.454 | 4960 | 187 | exact_preserve | connectivity_wall,exact_preserve,scatter |
| 13 | 132 | 3589.0 | 16.684 | 3990 | 99 | exact_preserve | bitwise_program,exact_preserve,gather_heavy,local_stencil,open_angle |
| 14 | 131 | 3477.0 | 16.712 | 3302 | 675 | exact_preserve | exact_preserve,open_angle,scatter |
| 15 | 037 | 3085.0 | 16.815 | 3384 | 201 | exact_preserve | connectivity_wall,exact_preserve,lut_selection,scatter |
| 16 | 079 | 2633.0 | 16.950 | 3055 | 78 | exact_preserve | custom_win,exact_preserve,local_stencil,open_angle |
| 17 | 174 | 2378.0 | 16.616 | 3982 | 396 | exact_preserve | connectivity_wall,documented_wall,exact_preserve,matmul,open_angle,scan |
| 18 | 365 | 1932.0 | 16.723 | 3808 | 124 | exact_preserve | connectivity_wall,custom_win,documented_wall,exact_preserve,gather_heavy,open_angle,qlinear |
| 19 | 361 | -37312.0 | 16.547 | 4322 | 366 | exact_preserve | documented_wall,exact_preserve,lut_selection,marginal_wall,open_angle,scatter |
| 20 | 018 | -76211.7 | 14.856 | 24029 | 1416 | exact_preserve | ambiguity_wall,documented_wall,exact_preserve,gather_heavy,high_memory,low_score,maxpool_scan,scatter |

## strided_conv_fixed_block_counts

Replace many fixed-block Slice+ReduceSum branches with one strided Conv

- source tasks: 011
- candidates: 36
- expected: {'gain_type': 'memory_when_repeated_block_reads_share_one_channel_or_kernel', 'risk': 'low-medium; Conv weight params can outweigh memory savings when few blocks are read or the input crop is already tiny', 'verification': 'stored eval plus fresh side-by-side; inspect Conv output shape because full 30x30 input may produce trailing off-grid stride positions that must be sliced away'}

| rank | task | score | pts | mem | params | source | tags |
|---:|---:|---:|---:|---:|---:|---|---|
| 1 | 366 | 29966.9 | 14.640 | 30983 | 576 | exact_preserve | assignment_wall,connectivity_wall,documented_wall,exact_preserve,gather_heavy,high_memory,low_score,lut_selection |
| 2 | 101 | 14620.5 | 15.422 | 13573 | 874 | semantic_or_handbuilt | assignment_wall,connectivity_wall,gather_heavy,high_memory,lut_selection,onehot_final_equal,qlinear,scatter |
| 3 | 191 | 10686.2 | 15.686 | 11009 | 83 | exact_preserve | assignment_wall,conv_heavy,custom_win,exact_preserve,high_memory,open_angle,qlinear |
| 4 | 198 | 9514.7 | 15.658 | 11305 | 107 | exact_preserve | connectivity_wall,documented_wall,exact_preserve,gather_heavy,high_memory,local_stencil,open_angle,scan |
| 5 | 377 | 7712.9 | 15.987 | 8075 | 134 | exact_preserve | exact_preserve,local_stencil |
| 6 | 370 | 7604.0 | 16.000 | 7331 | 773 | exact_preserve | conv_heavy,exact_preserve,qlinear,scatter |
| 7 | 382 | 5232.0 | 16.346 | 5606 | 126 | exact_preserve | exact_preserve,open_angle,scan |
| 8 | 157 | 5024.0 | 16.143 | 6768 | 256 | exact_preserve | assignment_wall,connectivity_wall,documented_wall,exact_preserve,gather_heavy,lut_selection,open_angle,scatter |
| 9 | 328 | 4634.0 | 16.200 | 4691 | 1943 | exact_preserve | documented_wall,exact_preserve,lut_selection |
| 10 | 396 | 4444.0 | 16.494 | 4854 | 90 | exact_preserve | conv_heavy,exact_preserve,local_stencil,onehot_final_equal |
| 11 | 182 | 4100.0 | 16.284 | 6024 | 76 | exact_preserve | assignment_wall,connectivity_wall,conv_heavy,documented_wall,exact_preserve,local_stencil,lut_selection,open_angle |
| 12 | 208 | 3709.0 | 16.655 | 4072 | 137 | exact_preserve | exact_preserve,open_angle,qlinear,scatter |
| 13 | 363 | 3635.0 | 16.673 | 3839 | 296 | exact_preserve | connectivity_wall,conv_heavy,exact_preserve,local_stencil,onehot_final_equal,open_angle |
| 14 | 014 | 3617.0 | 16.677 | 4036 | 81 | exact_preserve | exact_preserve |
| 15 | 107 | 3194.0 | 16.786 | 3327 | 367 | exact_preserve | exact_preserve,lut_selection,matmul,onehot_final_equal,open_angle,qlinear,scatter |
| 16 | 008 | 2722.0 | 16.922 | 3088 | 134 | exact_preserve | exact_preserve,lut_selection |
| 17 | 270 | 2569.0 | 16.971 | 2742 | 327 | exact_preserve | assignment_wall,bitwise_program,connectivity_wall,exact_preserve,open_angle,scan |
| 18 | 105 | 2443.0 | 17.013 | 2838 | 105 | exact_preserve | custom_win,exact_preserve,matmul,open_angle |
| 19 | 234 | 2434.0 | 17.016 | 2869 | 65 | exact_preserve | connectivity_wall,exact_preserve,onehot_final_equal |
| 20 | 398 | 2359.0 | 17.042 | 1666 | 1193 | exact_preserve | exact_preserve,matmul,open_angle |
| 21 | 250 | 2042.0 | 17.159 | 2488 | 54 | exact_preserve | exact_preserve,open_angle,qlinear |
| 22 | 134 | 1784.0 | 17.266 | 2250 | 34 | exact_preserve | exact_preserve,open_angle |
| 23 | 374 | 1784.0 | 17.266 | 2242 | 42 | exact_preserve | exact_preserve,open_angle |
| 24 | 012 | 1759.0 | 17.277 | 2071 | 188 | exact_preserve | exact_preserve,lut_selection,open_angle,scatter |
| 25 | 256 | 1751.0 | 17.281 | 2196 | 55 | exact_preserve | exact_preserve |

## solid_marker_profile_reconstruction

Replace full solid-marker channel slices with row/column profile contractions

- source tasks: 008
- candidates: 21
- expected: {'gain_type': 'memory_by_deleting_full_marker_plane', 'risk': 'low when the marker is generator-proven solid and separable; medium when the marker may touch borders or share rows/cols with other same-colour pixels', 'verification': 'stored eval plus fresh generation; sample generator border cases because flip/xpose can invalidate assumed crop margins'}

| rank | task | score | pts | mem | params | source | tags |
|---:|---:|---:|---:|---:|---:|---|---|
| 1 | 101 | 14620.5 | 15.422 | 13573 | 874 | semantic_or_handbuilt | assignment_wall,connectivity_wall,gather_heavy,high_memory,lut_selection,onehot_final_equal,qlinear,scatter |
| 2 | 025 | 9806.1 | 15.766 | 10132 | 104 | exact_preserve | exact_preserve,high_memory,onehot_final_equal,open_angle,scan |
| 3 | 064 | 8548.7 | 15.791 | 9852 | 134 | semantic_or_handbuilt | connectivity_wall,documented_wall,onehot_final_equal,open_angle |
| 4 | 017 | 5955.0 | 16.227 | 4749 | 1706 | exact_preserve | custom_win,exact_preserve,matmul,onehot_final_equal,open_angle |
| 5 | 219 | 5710.0 | 16.117 | 7078 | 132 | unknown | assignment_wall,connectivity_wall,documented_wall,lut_selection,onehot_final_equal,open_angle,scan |
| 6 | 009 | 4699.0 | 16.190 | 6529 | 170 | exact_preserve | connectivity_wall,documented_wall,exact_preserve,local_stencil,lut_selection,onehot_final_equal,open_angle |
| 7 | 396 | 4444.0 | 16.494 | 4854 | 90 | exact_preserve | conv_heavy,exact_preserve,local_stencil,onehot_final_equal |
| 8 | 383 | 4014.0 | 16.298 | 5918 | 96 | exact_preserve | assignment_wall,connectivity_wall,documented_wall,exact_preserve,local_stencil,onehot_final_equal,open_angle |
| 9 | 363 | 3635.0 | 16.673 | 3839 | 296 | exact_preserve | connectivity_wall,conv_heavy,exact_preserve,local_stencil,onehot_final_equal,open_angle |
| 10 | 202 | 3287.0 | 16.902 | 3253 | 34 | unknown | connectivity_wall,onehot_final_equal,open_angle,qlinear |
| 11 | 107 | 3194.0 | 16.786 | 3327 | 367 | exact_preserve | exact_preserve,lut_selection,matmul,onehot_final_equal,open_angle,qlinear,scatter |
| 12 | 177 | 3121.0 | 16.805 | 3500 | 121 | exact_preserve | connectivity_wall,conv_heavy,exact_preserve,local_stencil,onehot_final_equal,open_angle |
| 13 | 234 | 2434.0 | 17.016 | 2869 | 65 | exact_preserve | connectivity_wall,exact_preserve,onehot_final_equal |
| 14 | 036 | 1600.0 | 17.350 | 2061 | 39 | exact_preserve | connectivity_wall,exact_preserve,onehot_final_equal,open_angle |
| 15 | 392 | 1451.0 | 17.424 | 1602 | 349 | exact_preserve | custom_win,exact_preserve,onehot_final_equal,open_angle |
| 16 | 213 | 1351.0 | 17.477 | 1802 | 49 | exact_preserve | exact_preserve,onehot_final_equal,open_angle |
| 17 | 333 | 1223.0 | 16.922 | 2788 | 435 | exact_preserve | connectivity_wall,documented_wall,exact_preserve,onehot_final_equal,open_angle |
| 18 | 071 | 698.0 | 17.100 | 2643 | 55 | exact_preserve | connectivity_wall,documented_wall,exact_preserve,onehot_final_equal,open_angle |
| 19 | 170 | 545.0 | 17.158 | 2216 | 329 | exact_preserve | assignment_wall,conv_heavy,documented_wall,exact_preserve,local_stencil,lut_selection,onehot_final_equal,open_angle |
| 20 | 184 | 41.0 | 17.379 | 1920 | 121 | exact_preserve | connectivity_wall,conv_heavy,documented_wall,exact_preserve,local_stencil,onehot_final_equal,open_angle,qlinear |
| 21 | 029 | -35201.0 | 16.425 | 5212 | 87 | exact_preserve | exact_preserve,marginal_wall,onehot_final_equal,open_angle |

## walk_einsum_iteration_collapse

Collapse iterative flood/scan/propagation into ONE multi-operand Einsum via scaled walk counting (internal computation uncounted)

- source tasks: 187 313 110 243 077
- candidates: 26
- expected: {'gain_type': 'memory_by_deleting_all_iteration_planes', 'risk': 'medium: verify connectivity model (4 vs 8-conn!) matches incumbent semantics on fresh; verify step-count covers measured max BFS distance + margin; keep operands nonnegative or prove no cancellation. REJECT-CHECKS (task204 S8): parallel fan-out MaxPools (per-size anchor dilations) are NOT collapsible chains; uint8 conv banks of sub-400B planes are einsum-proof (fp32 entry ticket 1600-3600B + step params loses)', 'verification': 'numpy walk-vs-BFS on 20000 fresh, stored eval, fresh_verify candidate<=incumbent, A/B submission for grader safety on first use'}

| rank | task | score | pts | mem | params | source | tags |
|---:|---:|---:|---:|---:|---:|---|---|
| 1 | 233 | 32165.5 | 14.588 | 32796 | 446 | semantic_or_handbuilt | assignment_wall,connectivity_wall,conv_heavy,documented_wall,gather_heavy,high_memory,low_score,lut_selection |
| 2 | 366 | 29966.9 | 14.640 | 30983 | 576 | exact_preserve | assignment_wall,connectivity_wall,documented_wall,exact_preserve,gather_heavy,high_memory,low_score,lut_selection |
| 3 | 285 | 23574.8 | 14.864 | 24684 | 550 | exact_preserve | connectivity_wall,documented_wall,exact_preserve,gather_heavy,high_memory,low_score,open_angle,scatter |
| 4 | 054 | 20655.7 | 15.078 | 20091 | 288 | unknown | gather_heavy,high_memory,scan,scatter |
| 5 | 133 | 19819.1 | 15.023 | 20510 | 1016 | exact_preserve | assignment_wall,connectivity_wall,conv_heavy,documented_wall,exact_preserve,high_memory,lut_selection,open_angle |
| 6 | 319 | 15367.8 | 15.251 | 16090 | 1053 | exact_preserve | assignment_wall,documented_wall,exact_preserve,gather_heavy,high_memory,scan |
| 7 | 101 | 14620.5 | 15.422 | 13573 | 874 | semantic_or_handbuilt | assignment_wall,connectivity_wall,gather_heavy,high_memory,lut_selection,onehot_final_equal,qlinear,scatter |
| 8 | 076 | 14134.8 | 15.324 | 15629 | 303 | exact_preserve | assignment_wall,connectivity_wall,documented_wall,exact_preserve,high_memory,open_angle,scatter |
| 9 | 367 | 14092.0 | 15.327 | 15300 | 590 | exact_preserve | connectivity_wall,conv_heavy,documented_wall,exact_preserve,high_memory,local_stencil,lut_selection,open_angle |
| 10 | 349 | 13227.6 | 15.381 | 13860 | 1182 | exact_preserve | conv_heavy,documented_wall,exact_preserve,high_memory,local_stencil,lut_selection,open_angle,qlinear |
| 11 | 173 | 13034.5 | 15.498 | 13307 | 77 | exact_preserve | exact_preserve,gather_heavy,high_memory,lut_selection,scatter |
| 12 | 138 | 10811.3 | 15.552 | 12215 | 462 | exact_preserve | conv_heavy,documented_wall,exact_preserve,high_memory,local_stencil,open_angle |
| 13 | 191 | 10686.2 | 15.686 | 11009 | 83 | exact_preserve | assignment_wall,conv_heavy,custom_win,exact_preserve,high_memory,open_angle,qlinear |
| 14 | 066 | 10187.7 | 15.778 | 9955 | 166 | semantic_or_handbuilt | marker_routed_path,open_angle,scan |
| 15 | 025 | 9806.1 | 15.766 | 10132 | 104 | exact_preserve | exact_preserve,high_memory,onehot_final_equal,open_angle,scan |
| 16 | 205 | 9544.6 | 15.791 | 9498 | 484 | exact_preserve | connectivity_wall,conv_heavy,exact_preserve,open_angle,qlinear |
| 17 | 198 | 9514.7 | 15.658 | 11305 | 107 | exact_preserve | connectivity_wall,documented_wall,exact_preserve,gather_heavy,high_memory,local_stencil,open_angle,scan |
| 18 | 145 | 9069.5 | 15.742 | 9244 | 1248 | semantic_or_handbuilt | connectivity_wall,custom_win,documented_wall,lut_selection,open_angle |
| 19 | 064 | 8548.7 | 15.791 | 9852 | 134 | semantic_or_handbuilt | connectivity_wall,documented_wall,onehot_final_equal,open_angle |
| 20 | 204 | 8302.0 | 15.767 | 9780 | 452 | exact_preserve | connectivity_wall,conv_heavy,custom_win,documented_wall,exact_preserve,local_stencil,maxpool_scan,open_angle |
| 21 | 338 | 7289.7 | 15.868 | 8581 | 669 | exact_preserve | connectivity_wall,documented_wall,exact_preserve,local_stencil,open_angle,qlinear,scan |
| 22 | 216 | 6679.9 | 15.934 | 8584 | 76 | exact_preserve | connectivity_wall,documented_wall,exact_preserve,open_angle,qlinear,scatter |
| 23 | 158 | -20983.1 | 15.060 | 18089 | 2646 | exact_preserve | assignment_wall,conv_heavy,documented_wall,exact_preserve,gather_heavy,high_memory,marginal_wall,open_angle |
| 24 | 350 | -32931.3 | 15.891 | 9012 | 24 | exact_preserve | connectivity_wall,documented_wall,exact_preserve,lut_selection,marginal_wall,open_angle |
| 25 | 018 | -76211.7 | 14.856 | 24029 | 1416 | exact_preserve | ambiguity_wall,documented_wall,exact_preserve,gather_heavy,high_memory,low_score,maxpool_scan,scatter |

## self_einsum_axis_activity_gate

Use the input twice in one Einsum to gate final output by axis activity

- source tasks: 067
- candidates: 231
- expected: {'gain_type': 'frontier_by_deleting_explicit_mask_and_constants', 'risk': 'medium: repeated-input Einsum is exact only when the axis activity predicate is positive for every kept coordinate and zero for every rejected coordinate; verify colour-0 cells still produce one-hot occupancy', 'verification': 'stored eval plus uncached fresh generation; inspect equation letters carefully because a wrong reused index silently gates the wrong axis'}

| rank | task | score | pts | mem | params | source | tags |
|---:|---:|---:|---:|---:|---:|---|---|
| 1 | 366 | 29966.9 | 14.640 | 30983 | 576 | exact_preserve | assignment_wall,connectivity_wall,documented_wall,exact_preserve,gather_heavy,high_memory,low_score,lut_selection |
| 2 | 285 | 23574.8 | 14.864 | 24684 | 550 | exact_preserve | connectivity_wall,documented_wall,exact_preserve,gather_heavy,high_memory,low_score,open_angle,scatter |
| 3 | 133 | 19819.1 | 15.023 | 20510 | 1016 | exact_preserve | assignment_wall,connectivity_wall,conv_heavy,documented_wall,exact_preserve,high_memory,lut_selection,open_angle |
| 4 | 364 | 16713.1 | 15.260 | 15860 | 1131 | exact_preserve | connectivity_wall,exact_preserve,high_memory,local_stencil,open_angle,qlinear |
| 5 | 101 | 14620.5 | 15.422 | 13573 | 874 | semantic_or_handbuilt | assignment_wall,connectivity_wall,gather_heavy,high_memory,lut_selection,onehot_final_equal,qlinear,scatter |
| 6 | 076 | 14134.8 | 15.324 | 15629 | 303 | exact_preserve | assignment_wall,connectivity_wall,documented_wall,exact_preserve,high_memory,open_angle,scatter |
| 7 | 367 | 14092.0 | 15.327 | 15300 | 590 | exact_preserve | connectivity_wall,conv_heavy,documented_wall,exact_preserve,high_memory,local_stencil,lut_selection,open_angle |
| 8 | 349 | 13227.6 | 15.381 | 13860 | 1182 | exact_preserve | conv_heavy,documented_wall,exact_preserve,high_memory,local_stencil,lut_selection,open_angle,qlinear |
| 9 | 138 | 10811.3 | 15.552 | 12215 | 462 | exact_preserve | conv_heavy,documented_wall,exact_preserve,high_memory,local_stencil,open_angle |
| 10 | 191 | 10686.2 | 15.686 | 11009 | 83 | exact_preserve | assignment_wall,conv_heavy,custom_win,exact_preserve,high_memory,open_angle,qlinear |
| 11 | 066 | 10187.7 | 15.778 | 9955 | 166 | semantic_or_handbuilt | marker_routed_path,open_angle,scan |
| 12 | 025 | 9806.1 | 15.766 | 10132 | 104 | exact_preserve | exact_preserve,high_memory,onehot_final_equal,open_angle,scan |
| 13 | 205 | 9544.6 | 15.791 | 9498 | 484 | exact_preserve | connectivity_wall,conv_heavy,exact_preserve,open_angle,qlinear |
| 14 | 198 | 9514.7 | 15.658 | 11305 | 107 | exact_preserve | connectivity_wall,documented_wall,exact_preserve,gather_heavy,high_memory,local_stencil,open_angle,scan |
| 15 | 145 | 9069.5 | 15.742 | 9244 | 1248 | semantic_or_handbuilt | connectivity_wall,custom_win,documented_wall,lut_selection,open_angle |
| 16 | 379 | 8668.9 | 15.880 | 7860 | 1273 | exact_preserve | connectivity_wall,conv_heavy,custom_win,exact_preserve,open_angle |
| 17 | 064 | 8548.7 | 15.791 | 9852 | 134 | semantic_or_handbuilt | connectivity_wall,documented_wall,onehot_final_equal,open_angle |
| 18 | 204 | 8302.0 | 15.767 | 9780 | 452 | exact_preserve | connectivity_wall,conv_heavy,custom_win,documented_wall,exact_preserve,local_stencil,maxpool_scan,open_angle |
| 19 | 080 | 7892.1 | 15.806 | 9387 | 447 | exact_preserve | conv_heavy,documented_wall,exact_preserve,local_stencil,lut_selection,open_angle,qlinear |
| 20 | 338 | 7289.7 | 15.868 | 8581 | 669 | exact_preserve | connectivity_wall,documented_wall,exact_preserve,local_stencil,open_angle,qlinear,scan |
| 21 | 216 | 6679.9 | 15.934 | 8584 | 76 | exact_preserve | connectivity_wall,documented_wall,exact_preserve,open_angle,qlinear,scatter |
| 22 | 023 | 6412.0 | 16.234 | 6163 | 249 | semantic_or_handbuilt | conv_heavy,local_stencil,open_angle,qlinear |
| 23 | 017 | 5955.0 | 16.227 | 4749 | 1706 | exact_preserve | custom_win,exact_preserve,matmul,onehot_final_equal,open_angle |
| 24 | 219 | 5710.0 | 16.117 | 7078 | 132 | unknown | assignment_wall,connectivity_wall,documented_wall,lut_selection,onehot_final_equal,open_angle,scan |
| 25 | 096 | 5682.0 | 16.053 | 7261 | 421 | exact_preserve | documented_wall,exact_preserve,open_angle,scatter |

## label_pad_vs_onehot_pad_ordering

Choose label-pad vs onehot-pad order by compact spatial area

- source tasks: 308 382
- candidates: 25
- expected: {'gain_type': 'memory_by_swapping_final_label_expansion_order', 'risk': 'low: purely algebraic if pad value is outside channel ids and Equal broadcasts identically; verify ORT Pad supports the bool output dtype/opset when using onehot-before-pad', 'verification': 'stored eval and memory measurement for the single task; static break-even is compact_area * channel_count versus padded_label_area'}

| rank | task | score | pts | mem | params | source | tags |
|---:|---:|---:|---:|---:|---:|---|---|
| 1 | 101 | 14620.5 | 15.422 | 13573 | 874 | semantic_or_handbuilt | assignment_wall,connectivity_wall,gather_heavy,high_memory,lut_selection,onehot_final_equal,qlinear,scatter |
| 2 | 025 | 9806.1 | 15.766 | 10132 | 104 | exact_preserve | exact_preserve,high_memory,onehot_final_equal,open_angle,scan |
| 3 | 064 | 8548.7 | 15.791 | 9852 | 134 | semantic_or_handbuilt | connectivity_wall,documented_wall,onehot_final_equal,open_angle |
| 4 | 017 | 5955.0 | 16.227 | 4749 | 1706 | exact_preserve | custom_win,exact_preserve,matmul,onehot_final_equal,open_angle |
| 5 | 219 | 5710.0 | 16.117 | 7078 | 132 | unknown | assignment_wall,connectivity_wall,documented_wall,lut_selection,onehot_final_equal,open_angle,scan |
| 6 | 009 | 4699.0 | 16.190 | 6529 | 170 | exact_preserve | connectivity_wall,documented_wall,exact_preserve,local_stencil,lut_selection,onehot_final_equal,open_angle |
| 7 | 396 | 4444.0 | 16.494 | 4854 | 90 | exact_preserve | conv_heavy,exact_preserve,local_stencil,onehot_final_equal |
| 8 | 383 | 4014.0 | 16.298 | 5918 | 96 | exact_preserve | assignment_wall,connectivity_wall,documented_wall,exact_preserve,local_stencil,onehot_final_equal,open_angle |
| 9 | 363 | 3635.0 | 16.673 | 3839 | 296 | exact_preserve | connectivity_wall,conv_heavy,exact_preserve,local_stencil,onehot_final_equal,open_angle |
| 10 | 202 | 3287.0 | 16.902 | 3253 | 34 | unknown | connectivity_wall,onehot_final_equal,open_angle,qlinear |
| 11 | 107 | 3194.0 | 16.786 | 3327 | 367 | exact_preserve | exact_preserve,lut_selection,matmul,onehot_final_equal,open_angle,qlinear,scatter |
| 12 | 177 | 3121.0 | 16.805 | 3500 | 121 | exact_preserve | connectivity_wall,conv_heavy,exact_preserve,local_stencil,onehot_final_equal,open_angle |
| 13 | 234 | 2434.0 | 17.016 | 2869 | 65 | exact_preserve | connectivity_wall,exact_preserve,onehot_final_equal |
| 14 | 036 | 1600.0 | 17.350 | 2061 | 39 | exact_preserve | connectivity_wall,exact_preserve,onehot_final_equal,open_angle |
| 15 | 392 | 1451.0 | 17.424 | 1602 | 349 | exact_preserve | custom_win,exact_preserve,onehot_final_equal,open_angle |
| 16 | 213 | 1351.0 | 17.477 | 1802 | 49 | exact_preserve | exact_preserve,onehot_final_equal,open_angle |
| 17 | 333 | 1223.0 | 16.922 | 2788 | 435 | exact_preserve | connectivity_wall,documented_wall,exact_preserve,onehot_final_equal,open_angle |
| 18 | 267 | 766.0 | 18.359 | 724 | 42 | semantic_or_handbuilt | onehot_final_equal |
| 19 | 175 | 755.0 | 17.865 | 318 | 937 | exact_preserve | exact_preserve,lut_selection,onehot_final_equal,open_angle |
| 20 | 071 | 698.0 | 17.100 | 2643 | 55 | exact_preserve | connectivity_wall,documented_wall,exact_preserve,onehot_final_equal,open_angle |
| 21 | 170 | 545.0 | 17.158 | 2216 | 329 | exact_preserve | assignment_wall,conv_heavy,documented_wall,exact_preserve,local_stencil,lut_selection,onehot_final_equal,open_angle |
| 22 | 218 | 149.0 | 18.525 | 552 | 97 | exact_preserve | connectivity_wall,conv_heavy,exact_preserve,local_stencil,onehot_final_equal,open_angle |
| 23 | 184 | 41.0 | 17.379 | 1920 | 121 | exact_preserve | connectivity_wall,conv_heavy,documented_wall,exact_preserve,local_stencil,onehot_final_equal,open_angle,qlinear |
| 24 | 290 | -145.0 | 19.128 | 292 | 63 | exact_preserve | custom_win,exact_preserve,onehot_final_equal,open_angle |
| 25 | 029 | -35201.0 | 16.425 | 5212 | 87 | exact_preserve | exact_preserve,marginal_wall,onehot_final_equal,open_angle |

## uint8_presence_for_argmax

Replace fp32 Sign presence vectors feeding ArgMax with Greater+Cast(uint8)

- source tasks: 234
- candidates: 238
- expected: {'gain_type': 'memory_by_halving_presence_profile_dtype', 'risk': 'low-medium: exact only when downstream ordering needs binary nonzero presence, not counts or signed magnitude; bool must be cast to uint8 because ORT ArgMax rejects bool', 'verification': 'stored eval and fresh verify for the single task; inspect each Sign consumer before replacing'}

| rank | task | score | pts | mem | params | source | tags |
|---:|---:|---:|---:|---:|---:|---|---|
| 1 | 233 | 32165.5 | 14.588 | 32796 | 446 | semantic_or_handbuilt | assignment_wall,connectivity_wall,conv_heavy,documented_wall,gather_heavy,high_memory,low_score,lut_selection |
| 2 | 366 | 29966.9 | 14.640 | 30983 | 576 | exact_preserve | assignment_wall,connectivity_wall,documented_wall,exact_preserve,gather_heavy,high_memory,low_score,lut_selection |
| 3 | 285 | 23574.8 | 14.864 | 24684 | 550 | exact_preserve | connectivity_wall,documented_wall,exact_preserve,gather_heavy,high_memory,low_score,open_angle,scatter |
| 4 | 054 | 20655.7 | 15.078 | 20091 | 288 | unknown | gather_heavy,high_memory,scan,scatter |
| 5 | 133 | 19819.1 | 15.023 | 20510 | 1016 | exact_preserve | assignment_wall,connectivity_wall,conv_heavy,documented_wall,exact_preserve,high_memory,lut_selection,open_angle |
| 6 | 364 | 16713.1 | 15.260 | 15860 | 1131 | exact_preserve | connectivity_wall,exact_preserve,high_memory,local_stencil,open_angle,qlinear |
| 7 | 319 | 15367.8 | 15.251 | 16090 | 1053 | exact_preserve | assignment_wall,documented_wall,exact_preserve,gather_heavy,high_memory,scan |
| 8 | 076 | 14134.8 | 15.324 | 15629 | 303 | exact_preserve | assignment_wall,connectivity_wall,documented_wall,exact_preserve,high_memory,open_angle,scatter |
| 9 | 367 | 14092.0 | 15.327 | 15300 | 590 | exact_preserve | connectivity_wall,conv_heavy,documented_wall,exact_preserve,high_memory,local_stencil,lut_selection,open_angle |
| 10 | 349 | 13227.6 | 15.381 | 13860 | 1182 | exact_preserve | conv_heavy,documented_wall,exact_preserve,high_memory,local_stencil,lut_selection,open_angle,qlinear |
| 11 | 138 | 10811.3 | 15.552 | 12215 | 462 | exact_preserve | conv_heavy,documented_wall,exact_preserve,high_memory,local_stencil,open_angle |
| 12 | 191 | 10686.2 | 15.686 | 11009 | 83 | exact_preserve | assignment_wall,conv_heavy,custom_win,exact_preserve,high_memory,open_angle,qlinear |
| 13 | 066 | 10187.7 | 15.778 | 9955 | 166 | semantic_or_handbuilt | marker_routed_path,open_angle,scan |
| 14 | 025 | 9806.1 | 15.766 | 10132 | 104 | exact_preserve | exact_preserve,high_memory,onehot_final_equal,open_angle,scan |
| 15 | 205 | 9544.6 | 15.791 | 9498 | 484 | exact_preserve | connectivity_wall,conv_heavy,exact_preserve,open_angle,qlinear |
| 16 | 198 | 9514.7 | 15.658 | 11305 | 107 | exact_preserve | connectivity_wall,documented_wall,exact_preserve,gather_heavy,high_memory,local_stencil,open_angle,scan |
| 17 | 145 | 9069.5 | 15.742 | 9244 | 1248 | semantic_or_handbuilt | connectivity_wall,custom_win,documented_wall,lut_selection,open_angle |
| 18 | 379 | 8668.9 | 15.880 | 7860 | 1273 | exact_preserve | connectivity_wall,conv_heavy,custom_win,exact_preserve,open_angle |
| 19 | 074 | 8583.2 | 15.889 | 9000 | 50 | exact_preserve | exact_preserve,local_stencil,open_angle |
| 20 | 064 | 8548.7 | 15.791 | 9852 | 134 | semantic_or_handbuilt | connectivity_wall,documented_wall,onehot_final_equal,open_angle |
| 21 | 204 | 8302.0 | 15.767 | 9780 | 452 | exact_preserve | connectivity_wall,conv_heavy,custom_win,documented_wall,exact_preserve,local_stencil,maxpool_scan,open_angle |
| 22 | 080 | 7892.1 | 15.806 | 9387 | 447 | exact_preserve | conv_heavy,documented_wall,exact_preserve,local_stencil,lut_selection,open_angle,qlinear |
| 23 | 338 | 7289.7 | 15.868 | 8581 | 669 | exact_preserve | connectivity_wall,documented_wall,exact_preserve,local_stencil,open_angle,qlinear,scan |
| 24 | 324 | 6921.9 | 15.907 | 7308 | 1586 | exact_preserve | assignment_wall,conv_heavy,documented_wall,exact_preserve,local_stencil,qlinear,scan |
| 25 | 216 | 6679.9 | 15.934 | 8584 | 76 | exact_preserve | connectivity_wall,documented_wall,exact_preserve,open_angle,qlinear,scatter |

## signed_rect_priority_overlay

Route separable axis-aligned rect fills / overlap priority through a signed-weight Einsum instead of a materialized label/priority carrier

- source tasks: 092 234 335
- candidates: 0
- expected: {'gain_type': 'memory_by_deleting_priority_label_carrier', 'risk': "medium: only applies to LABEL/priority carrier cost, not assignment/detection cost (task233 kill); verify the grader's (out>0.0) decode still holds and overlap suppression signs are correct at every crossing", 'verification': 'stored eval, fresh_verify bit-identical vs incumbent, hand-built crossing/overlap-heavy grids'}

| rank | task | score | pts | mem | params | source | tags |
|---:|---:|---:|---:|---:|---:|---|---|

## gridsample_warp_render

Replace explicit gather+mask+zero-pad addressing with a terminal GridSample against the raw input

- source tasks: 209
- candidates: 0
- expected: {'gain_type': 'memory_by_deleting_explicit_gather_and_mask_planes', 'risk': "medium: verify ORT GridSample padding_mode/align_corners semantics match the incumbent's boundary behaviour exactly", 'verification': 'stored eval, fresh_verify bit-identical vs incumbent'}

| rank | task | score | pts | mem | params | source | tags |
|---:|---:|---:|---:|---:|---:|---|---|

## einsum_vs_free_input_reduction

Replace a materialized mask/profile plane that only gets REDUCED with an Einsum/ReduceSum contracted directly against the FREE input

- source tasks: 255 208 370
- candidates: 0
- expected: {'gain_type': 'memory_by_deleting_materialized_reduced_mask_plane', 'risk': "low-medium: only safe when the plane's sole consumer is a reduction; verify no other op reads it before deleting", 'verification': 'stored eval, fresh_verify bit-identical vs incumbent'}

| rank | task | score | pts | mem | params | source | tags |
|---:|---:|---:|---:|---:|---:|---|---|

## qlinearconv_signed_renderer

Replace fp32/fp16 stamp-and-compare towers with a QLinearConv/ConvInteger signed uint8 renderer

- source tasks: 133
- candidates: 0
- expected: {'gain_type': 'memory_by_replacing_fp_stamp_tower_with_uint8_render', 'risk': 'medium: quantized zero-point/scale must be verified exact (x_zero_point=1) so integer rendering matches fp semantics bit-for-bit', 'verification': 'stored eval, fresh_verify bit-identical vs incumbent'}

| rank | task | score | pts | mem | params | source | tags |
|---:|---:|---:|---:|---:|---:|---|---|

## runtime_conv_template_anchor

Compile runtime-extracted sprite/template masks into non-initializer Conv weights

- source tasks: 089 101
- candidates: 0
- expected: {'gain_type': 'compiler_rewrite_by_deleting_sparse_TopK_enumeration_and_coverage_chains', 'risk': 'high: the QLinear no-suppression compiler has byte margin, but scale suppression / false-positive filtering can recreate the incumbent coverage cost. Avoid fp32 ConvTranspose; use runtime-weight QLinearConv for matching and reversed-kernel stamping.', 'verification': 'first build task101 QLinear anchor-map candidate under reports/candidates/task101; compare against active overlay on bundled data; fresh_verify only diagnostic in 8000 mode; inspect whether remaining false positives fit in a small exception filter'}

| rank | task | score | pts | mem | params | source | tags |
|---:|---:|---:|---:|---:|---:|---|---|

## dynamic_bundled_cse_rewire

Use bundled runtime signatures to remove duplicate active overlay intermediates

- source tasks: 173 054 158 133 233 205 092 234 023 319
- candidates: 47
- expected: {'gain_type': 'memory_by_rewiring_duplicate_node_outputs', 'risk': 'medium: runtime equality over bundled examples is an overfit-mode proof only; require static shape/dtype match and bundled fail=0 after each rewrite', 'verification': 'run reports/candidates/dynamic_cse_active_probe.py against active submission/overfit_nets; adopt only evaluated fail=0 lower-cost candidates, then scan_unsigned_topk and pack'}

| rank | task | score | pts | mem | params | source | tags |
|---:|---:|---:|---:|---:|---:|---|---|
| 1 | 233 | 32165.5 | 14.588 | 32796 | 446 | semantic_or_handbuilt | assignment_wall,connectivity_wall,conv_heavy,documented_wall,gather_heavy,high_memory,low_score,lut_selection |
| 2 | 366 | 29966.9 | 14.640 | 30983 | 576 | exact_preserve | assignment_wall,connectivity_wall,documented_wall,exact_preserve,gather_heavy,high_memory,low_score,lut_selection |
| 3 | 285 | 23574.8 | 14.864 | 24684 | 550 | exact_preserve | connectivity_wall,documented_wall,exact_preserve,gather_heavy,high_memory,low_score,open_angle,scatter |
| 4 | 286 | 22796.4 | 14.915 | 21090 | 2881 | semantic_or_handbuilt | connectivity_wall,conv_heavy,documented_wall,high_memory,local_stencil,low_score,qlinear |
| 5 | 054 | 20655.7 | 15.078 | 20091 | 288 | unknown | gather_heavy,high_memory,scan,scatter |
| 6 | 133 | 19819.1 | 15.023 | 20510 | 1016 | exact_preserve | assignment_wall,connectivity_wall,conv_heavy,documented_wall,exact_preserve,high_memory,lut_selection,open_angle |
| 7 | 364 | 16713.1 | 15.260 | 15860 | 1131 | exact_preserve | connectivity_wall,exact_preserve,high_memory,local_stencil,open_angle,qlinear |
| 8 | 319 | 15367.8 | 15.251 | 16090 | 1053 | exact_preserve | assignment_wall,documented_wall,exact_preserve,gather_heavy,high_memory,scan |
| 9 | 101 | 14620.5 | 15.422 | 13573 | 874 | semantic_or_handbuilt | assignment_wall,connectivity_wall,gather_heavy,high_memory,lut_selection,onehot_final_equal,qlinear,scatter |
| 10 | 076 | 14134.8 | 15.324 | 15629 | 303 | exact_preserve | assignment_wall,connectivity_wall,documented_wall,exact_preserve,high_memory,open_angle,scatter |
| 11 | 367 | 14092.0 | 15.327 | 15300 | 590 | exact_preserve | connectivity_wall,conv_heavy,documented_wall,exact_preserve,high_memory,local_stencil,lut_selection,open_angle |
| 12 | 349 | 13227.6 | 15.381 | 13860 | 1182 | exact_preserve | conv_heavy,documented_wall,exact_preserve,high_memory,local_stencil,lut_selection,open_angle,qlinear |
| 13 | 173 | 13034.5 | 15.498 | 13307 | 77 | exact_preserve | exact_preserve,gather_heavy,high_memory,lut_selection,scatter |
| 14 | 138 | 10811.3 | 15.552 | 12215 | 462 | exact_preserve | conv_heavy,documented_wall,exact_preserve,high_memory,local_stencil,open_angle |
| 15 | 191 | 10686.2 | 15.686 | 11009 | 83 | exact_preserve | assignment_wall,conv_heavy,custom_win,exact_preserve,high_memory,open_angle,qlinear |
| 16 | 025 | 9806.1 | 15.766 | 10132 | 104 | exact_preserve | exact_preserve,high_memory,onehot_final_equal,open_angle,scan |
| 17 | 205 | 9544.6 | 15.791 | 9498 | 484 | exact_preserve | connectivity_wall,conv_heavy,exact_preserve,open_angle,qlinear |
| 18 | 198 | 9514.7 | 15.658 | 11305 | 107 | exact_preserve | connectivity_wall,documented_wall,exact_preserve,gather_heavy,high_memory,local_stencil,open_angle,scan |
| 19 | 379 | 8668.9 | 15.880 | 7860 | 1273 | exact_preserve | connectivity_wall,conv_heavy,custom_win,exact_preserve,open_angle |
| 20 | 074 | 8583.2 | 15.889 | 9000 | 50 | exact_preserve | exact_preserve,local_stencil,open_angle |
| 21 | 255 | 8508.8 | 15.897 | 8625 | 353 | exact_preserve | conv_heavy,exact_preserve,matmul,qlinear |
| 22 | 204 | 8302.0 | 15.767 | 9780 | 452 | exact_preserve | connectivity_wall,conv_heavy,custom_win,documented_wall,exact_preserve,local_stencil,maxpool_scan,open_angle |
| 23 | 080 | 7892.1 | 15.806 | 9387 | 447 | exact_preserve | conv_heavy,documented_wall,exact_preserve,local_stencil,lut_selection,open_angle,qlinear |
| 24 | 377 | 7712.9 | 15.987 | 8075 | 134 | exact_preserve | exact_preserve,local_stencil |
| 25 | 370 | 7604.0 | 16.000 | 7331 | 773 | exact_preserve | conv_heavy,exact_preserve,qlinear,scatter |

## dedupe_byte_identical_initializers

Deduplicate byte-identical initializers after overlay and graph surgery

- source tasks: 054 234 348 335 037 280 064 173 018 233 138 285 390
- candidates: 326
- expected: {'gain_type': 'params_by_deleting_duplicate_initializers', 'risk': 'low: byte-identical constants are equivalent; still require bundled fail=0 and lower measured cost because shape inference/scorer quirks can reject rewritten models', 'verification': 'run reports/candidates/dedupe_initializers_active_probe.py after overlay or graph-surgery passes; adopt only evaluated fail=0 lower-cost candidates'}

| rank | task | score | pts | mem | params | source | tags |
|---:|---:|---:|---:|---:|---:|---|---|
| 1 | 233 | 32165.5 | 14.588 | 32796 | 446 | semantic_or_handbuilt | assignment_wall,connectivity_wall,conv_heavy,documented_wall,gather_heavy,high_memory,low_score,lut_selection |
| 2 | 366 | 29966.9 | 14.640 | 30983 | 576 | exact_preserve | assignment_wall,connectivity_wall,documented_wall,exact_preserve,gather_heavy,high_memory,low_score,lut_selection |
| 3 | 285 | 23574.8 | 14.864 | 24684 | 550 | exact_preserve | connectivity_wall,documented_wall,exact_preserve,gather_heavy,high_memory,low_score,open_angle,scatter |
| 4 | 286 | 22796.4 | 14.915 | 21090 | 2881 | semantic_or_handbuilt | connectivity_wall,conv_heavy,documented_wall,high_memory,local_stencil,low_score,qlinear |
| 5 | 054 | 20655.7 | 15.078 | 20091 | 288 | unknown | gather_heavy,high_memory,scan,scatter |
| 6 | 133 | 19819.1 | 15.023 | 20510 | 1016 | exact_preserve | assignment_wall,connectivity_wall,conv_heavy,documented_wall,exact_preserve,high_memory,lut_selection,open_angle |
| 7 | 364 | 16713.1 | 15.260 | 15860 | 1131 | exact_preserve | connectivity_wall,exact_preserve,high_memory,local_stencil,open_angle,qlinear |
| 8 | 319 | 15367.8 | 15.251 | 16090 | 1053 | exact_preserve | assignment_wall,documented_wall,exact_preserve,gather_heavy,high_memory,scan |
| 9 | 101 | 14620.5 | 15.422 | 13573 | 874 | semantic_or_handbuilt | assignment_wall,connectivity_wall,gather_heavy,high_memory,lut_selection,onehot_final_equal,qlinear,scatter |
| 10 | 076 | 14134.8 | 15.324 | 15629 | 303 | exact_preserve | assignment_wall,connectivity_wall,documented_wall,exact_preserve,high_memory,open_angle,scatter |
| 11 | 367 | 14092.0 | 15.327 | 15300 | 590 | exact_preserve | connectivity_wall,conv_heavy,documented_wall,exact_preserve,high_memory,local_stencil,lut_selection,open_angle |
| 12 | 349 | 13227.6 | 15.381 | 13860 | 1182 | exact_preserve | conv_heavy,documented_wall,exact_preserve,high_memory,local_stencil,lut_selection,open_angle,qlinear |
| 13 | 173 | 13034.5 | 15.498 | 13307 | 77 | exact_preserve | exact_preserve,gather_heavy,high_memory,lut_selection,scatter |
| 14 | 138 | 10811.3 | 15.552 | 12215 | 462 | exact_preserve | conv_heavy,documented_wall,exact_preserve,high_memory,local_stencil,open_angle |
| 15 | 191 | 10686.2 | 15.686 | 11009 | 83 | exact_preserve | assignment_wall,conv_heavy,custom_win,exact_preserve,high_memory,open_angle,qlinear |
| 16 | 066 | 10187.7 | 15.778 | 9955 | 166 | semantic_or_handbuilt | marker_routed_path,open_angle,scan |
| 17 | 025 | 9806.1 | 15.766 | 10132 | 104 | exact_preserve | exact_preserve,high_memory,onehot_final_equal,open_angle,scan |
| 18 | 205 | 9544.6 | 15.791 | 9498 | 484 | exact_preserve | connectivity_wall,conv_heavy,exact_preserve,open_angle,qlinear |
| 19 | 198 | 9514.7 | 15.658 | 11305 | 107 | exact_preserve | connectivity_wall,documented_wall,exact_preserve,gather_heavy,high_memory,local_stencil,open_angle,scan |
| 20 | 145 | 9069.5 | 15.742 | 9244 | 1248 | semantic_or_handbuilt | connectivity_wall,custom_win,documented_wall,lut_selection,open_angle |
| 21 | 379 | 8668.9 | 15.880 | 7860 | 1273 | exact_preserve | connectivity_wall,conv_heavy,custom_win,exact_preserve,open_angle |
| 22 | 074 | 8583.2 | 15.889 | 9000 | 50 | exact_preserve | exact_preserve,local_stencil,open_angle |
| 23 | 064 | 8548.7 | 15.791 | 9852 | 134 | semantic_or_handbuilt | connectivity_wall,documented_wall,onehot_final_equal,open_angle |
| 24 | 255 | 8508.8 | 15.897 | 8625 | 353 | exact_preserve | conv_heavy,exact_preserve,matmul,qlinear |
| 25 | 204 | 8302.0 | 15.767 | 9780 | 452 | exact_preserve | connectivity_wall,conv_heavy,custom_win,documented_wall,exact_preserve,local_stencil,maxpool_scan,open_angle |

## signed_int8_topk_compact_feeds

Signed INT8 TopK feeds are local-valid but Kaggle-rejected

- source tasks: 233 173 285 076 316 397 368 018 308 366 037
- candidates: 0
- expected: {'gain_type': 'negative_filter', 'risk': 'certain Kaggle submission rejection based on full, group, and single-task oracle submissions', 'verification': 'do not adopt signed INT8 TopK feeds unless a future Kaggle oracle explicitly disproves this rejection; local bundled fail=0 and unsigned scan clean are insufficient'}

| rank | task | score | pts | mem | params | source | tags |
|---:|---:|---:|---:|---:|---:|---|---|

## topk_k2_to_argmax_uint8_with_exception_patch

ArgMax(uint8) binary-match replacement is local-valid but Kaggle-unsafe

- source tasks: 233
- candidates: 0
- expected: {'gain_type': 'negative_filter', 'risk': 'certain LB score collapse for the tested task233 route despite local bundled fail=0', 'verification': 'do not adopt ArgMax(uint8) binary-match replacements unless a future Kaggle oracle explicitly proves the target task scores'}

| rank | task | score | pts | mem | params | source | tags |
|---:|---:|---:|---:|---:|---:|---|---|

## pad_compensated_spatial_crop

Shrink dead Conv/QLinearConv borders and compensate in the sole spatial consumer padding

- source tasks: 349
- candidates: 50
- expected: {'gain_type': 'memory_by_removing_dead_spatial_border', 'risk': 'low in 8000 mode only after bundled intermediate probe plus bundled fail=0; crop limit is task-specific', 'verification': 'sweep producer output width/height with consumer pad compensation, adopt only exact bundled fail=0 and lower cost'}

| rank | task | score | pts | mem | params | source | tags |
|---:|---:|---:|---:|---:|---:|---|---|
| 1 | 233 | 32165.5 | 14.588 | 32796 | 446 | semantic_or_handbuilt | assignment_wall,connectivity_wall,conv_heavy,documented_wall,gather_heavy,high_memory,low_score,lut_selection |
| 2 | 366 | 29966.9 | 14.640 | 30983 | 576 | exact_preserve | assignment_wall,connectivity_wall,documented_wall,exact_preserve,gather_heavy,high_memory,low_score,lut_selection |
| 3 | 285 | 23574.8 | 14.864 | 24684 | 550 | exact_preserve | connectivity_wall,documented_wall,exact_preserve,gather_heavy,high_memory,low_score,open_angle,scatter |
| 4 | 286 | 22796.4 | 14.915 | 21090 | 2881 | semantic_or_handbuilt | connectivity_wall,conv_heavy,documented_wall,high_memory,local_stencil,low_score,qlinear |
| 5 | 054 | 20655.7 | 15.078 | 20091 | 288 | unknown | gather_heavy,high_memory,scan,scatter |
| 6 | 133 | 19819.1 | 15.023 | 20510 | 1016 | exact_preserve | assignment_wall,connectivity_wall,conv_heavy,documented_wall,exact_preserve,high_memory,lut_selection,open_angle |
| 7 | 364 | 16713.1 | 15.260 | 15860 | 1131 | exact_preserve | connectivity_wall,exact_preserve,high_memory,local_stencil,open_angle,qlinear |
| 8 | 076 | 14134.8 | 15.324 | 15629 | 303 | exact_preserve | assignment_wall,connectivity_wall,documented_wall,exact_preserve,high_memory,open_angle,scatter |
| 9 | 367 | 14092.0 | 15.327 | 15300 | 590 | exact_preserve | connectivity_wall,conv_heavy,documented_wall,exact_preserve,high_memory,local_stencil,lut_selection,open_angle |
| 10 | 349 | 13227.6 | 15.381 | 13860 | 1182 | exact_preserve | conv_heavy,documented_wall,exact_preserve,high_memory,local_stencil,lut_selection,open_angle,qlinear |
| 11 | 173 | 13034.5 | 15.498 | 13307 | 77 | exact_preserve | exact_preserve,gather_heavy,high_memory,lut_selection,scatter |
| 12 | 138 | 10811.3 | 15.552 | 12215 | 462 | exact_preserve | conv_heavy,documented_wall,exact_preserve,high_memory,local_stencil,open_angle |
| 13 | 191 | 10686.2 | 15.686 | 11009 | 83 | exact_preserve | assignment_wall,conv_heavy,custom_win,exact_preserve,high_memory,open_angle,qlinear |
| 14 | 205 | 9544.6 | 15.791 | 9498 | 484 | exact_preserve | connectivity_wall,conv_heavy,exact_preserve,open_angle,qlinear |
| 15 | 198 | 9514.7 | 15.658 | 11305 | 107 | exact_preserve | connectivity_wall,documented_wall,exact_preserve,gather_heavy,high_memory,local_stencil,open_angle,scan |
| 16 | 379 | 8668.9 | 15.880 | 7860 | 1273 | exact_preserve | connectivity_wall,conv_heavy,custom_win,exact_preserve,open_angle |
| 17 | 074 | 8583.2 | 15.889 | 9000 | 50 | exact_preserve | exact_preserve,local_stencil,open_angle |
| 18 | 255 | 8508.8 | 15.897 | 8625 | 353 | exact_preserve | conv_heavy,exact_preserve,matmul,qlinear |
| 19 | 204 | 8302.0 | 15.767 | 9780 | 452 | exact_preserve | connectivity_wall,conv_heavy,custom_win,documented_wall,exact_preserve,local_stencil,maxpool_scan,open_angle |
| 20 | 080 | 7892.1 | 15.806 | 9387 | 447 | exact_preserve | conv_heavy,documented_wall,exact_preserve,local_stencil,lut_selection,open_angle,qlinear |
| 21 | 377 | 7712.9 | 15.987 | 8075 | 134 | exact_preserve | exact_preserve,local_stencil |
| 22 | 370 | 7604.0 | 16.000 | 7331 | 773 | exact_preserve | conv_heavy,exact_preserve,qlinear,scatter |
| 23 | 338 | 7289.7 | 15.868 | 8581 | 669 | exact_preserve | connectivity_wall,documented_wall,exact_preserve,local_stencil,open_angle,qlinear,scan |
| 24 | 324 | 6921.9 | 15.907 | 7308 | 1586 | exact_preserve | assignment_wall,conv_heavy,documented_wall,exact_preserve,local_stencil,qlinear,scan |
| 25 | 023 | 6412.0 | 16.234 | 6163 | 249 | semantic_or_handbuilt | conv_heavy,local_stencil,open_angle,qlinear |

## zero_compare_to_bool_cast

Replace nonnegative x > 0 masks with Cast(x -> bool)

- source tasks: 009 044 077 134 224 281 319 233 234 308 355 363 397
- candidates: 270
- expected: {'gain_type': 'params_by_deleting_zero_scalar_initializers', 'risk': 'low after bundled eval; only valid when the compared tensor is known nonnegative and false iff exactly zero', 'verification': 'candidate eval fail=0 and lower cost; do not infer from syntax alone when scores may be negative'}

| rank | task | score | pts | mem | params | source | tags |
|---:|---:|---:|---:|---:|---:|---|---|
| 1 | 233 | 32165.5 | 14.588 | 32796 | 446 | semantic_or_handbuilt | assignment_wall,connectivity_wall,conv_heavy,documented_wall,gather_heavy,high_memory,low_score,lut_selection |
| 2 | 366 | 29966.9 | 14.640 | 30983 | 576 | exact_preserve | assignment_wall,connectivity_wall,documented_wall,exact_preserve,gather_heavy,high_memory,low_score,lut_selection |
| 3 | 285 | 23574.8 | 14.864 | 24684 | 550 | exact_preserve | connectivity_wall,documented_wall,exact_preserve,gather_heavy,high_memory,low_score,open_angle,scatter |
| 4 | 286 | 22796.4 | 14.915 | 21090 | 2881 | semantic_or_handbuilt | connectivity_wall,conv_heavy,documented_wall,high_memory,local_stencil,low_score,qlinear |
| 5 | 054 | 20655.7 | 15.078 | 20091 | 288 | unknown | gather_heavy,high_memory,scan,scatter |
| 6 | 133 | 19819.1 | 15.023 | 20510 | 1016 | exact_preserve | assignment_wall,connectivity_wall,conv_heavy,documented_wall,exact_preserve,high_memory,lut_selection,open_angle |
| 7 | 364 | 16713.1 | 15.260 | 15860 | 1131 | exact_preserve | connectivity_wall,exact_preserve,high_memory,local_stencil,open_angle,qlinear |
| 8 | 319 | 15367.8 | 15.251 | 16090 | 1053 | exact_preserve | assignment_wall,documented_wall,exact_preserve,gather_heavy,high_memory,scan |
| 9 | 101 | 14620.5 | 15.422 | 13573 | 874 | semantic_or_handbuilt | assignment_wall,connectivity_wall,gather_heavy,high_memory,lut_selection,onehot_final_equal,qlinear,scatter |
| 10 | 076 | 14134.8 | 15.324 | 15629 | 303 | exact_preserve | assignment_wall,connectivity_wall,documented_wall,exact_preserve,high_memory,open_angle,scatter |
| 11 | 367 | 14092.0 | 15.327 | 15300 | 590 | exact_preserve | connectivity_wall,conv_heavy,documented_wall,exact_preserve,high_memory,local_stencil,lut_selection,open_angle |
| 12 | 349 | 13227.6 | 15.381 | 13860 | 1182 | exact_preserve | conv_heavy,documented_wall,exact_preserve,high_memory,local_stencil,lut_selection,open_angle,qlinear |
| 13 | 173 | 13034.5 | 15.498 | 13307 | 77 | exact_preserve | exact_preserve,gather_heavy,high_memory,lut_selection,scatter |
| 14 | 138 | 10811.3 | 15.552 | 12215 | 462 | exact_preserve | conv_heavy,documented_wall,exact_preserve,high_memory,local_stencil,open_angle |
| 15 | 191 | 10686.2 | 15.686 | 11009 | 83 | exact_preserve | assignment_wall,conv_heavy,custom_win,exact_preserve,high_memory,open_angle,qlinear |
| 16 | 066 | 10187.7 | 15.778 | 9955 | 166 | semantic_or_handbuilt | marker_routed_path,open_angle,scan |
| 17 | 025 | 9806.1 | 15.766 | 10132 | 104 | exact_preserve | exact_preserve,high_memory,onehot_final_equal,open_angle,scan |
| 18 | 205 | 9544.6 | 15.791 | 9498 | 484 | exact_preserve | connectivity_wall,conv_heavy,exact_preserve,open_angle,qlinear |
| 19 | 198 | 9514.7 | 15.658 | 11305 | 107 | exact_preserve | connectivity_wall,documented_wall,exact_preserve,gather_heavy,high_memory,local_stencil,open_angle,scan |
| 20 | 145 | 9069.5 | 15.742 | 9244 | 1248 | semantic_or_handbuilt | connectivity_wall,custom_win,documented_wall,lut_selection,open_angle |
| 21 | 379 | 8668.9 | 15.880 | 7860 | 1273 | exact_preserve | connectivity_wall,conv_heavy,custom_win,exact_preserve,open_angle |
| 22 | 074 | 8583.2 | 15.889 | 9000 | 50 | exact_preserve | exact_preserve,local_stencil,open_angle |
| 23 | 064 | 8548.7 | 15.791 | 9852 | 134 | semantic_or_handbuilt | connectivity_wall,documented_wall,onehot_final_equal,open_angle |
| 24 | 204 | 8302.0 | 15.767 | 9780 | 452 | exact_preserve | connectivity_wall,conv_heavy,custom_win,documented_wall,exact_preserve,local_stencil,maxpool_scan,open_angle |
| 25 | 080 | 7892.1 | 15.806 | 9387 | 447 | exact_preserve | conv_heavy,documented_wall,exact_preserve,local_stencil,lut_selection,open_angle,qlinear |

## branch_einsum_copy_edit_epilogue

Fold copy/edit Where epilogues into one free-output branch Einsum

- source tasks: 077 187
- candidates: 55
- expected: {'gain_type': 'memory_by_deleting_mask_and_copy_edit_epilogue', 'risk': 'medium: branch identity count can overpower the subtract branch; sweep subtract scale and require bundled fail=0', 'verification': 'build branch-index final output op, sweep subtract scale, compare bundled fail=0 and cost; inspect multi-hot failures'}

| rank | task | score | pts | mem | params | source | tags |
|---:|---:|---:|---:|---:|---:|---|---|
| 1 | 233 | 32165.5 | 14.588 | 32796 | 446 | semantic_or_handbuilt | assignment_wall,connectivity_wall,conv_heavy,documented_wall,gather_heavy,high_memory,low_score,lut_selection |
| 2 | 366 | 29966.9 | 14.640 | 30983 | 576 | exact_preserve | assignment_wall,connectivity_wall,documented_wall,exact_preserve,gather_heavy,high_memory,low_score,lut_selection |
| 3 | 285 | 23574.8 | 14.864 | 24684 | 550 | exact_preserve | connectivity_wall,documented_wall,exact_preserve,gather_heavy,high_memory,low_score,open_angle,scatter |
| 4 | 286 | 22796.4 | 14.915 | 21090 | 2881 | semantic_or_handbuilt | connectivity_wall,conv_heavy,documented_wall,high_memory,local_stencil,low_score,qlinear |
| 5 | 054 | 20655.7 | 15.078 | 20091 | 288 | unknown | gather_heavy,high_memory,scan,scatter |
| 6 | 133 | 19819.1 | 15.023 | 20510 | 1016 | exact_preserve | assignment_wall,connectivity_wall,conv_heavy,documented_wall,exact_preserve,high_memory,lut_selection,open_angle |
| 7 | 364 | 16713.1 | 15.260 | 15860 | 1131 | exact_preserve | connectivity_wall,exact_preserve,high_memory,local_stencil,open_angle,qlinear |
| 8 | 319 | 15367.8 | 15.251 | 16090 | 1053 | exact_preserve | assignment_wall,documented_wall,exact_preserve,gather_heavy,high_memory,scan |
| 9 | 101 | 14620.5 | 15.422 | 13573 | 874 | semantic_or_handbuilt | assignment_wall,connectivity_wall,gather_heavy,high_memory,lut_selection,onehot_final_equal,qlinear,scatter |
| 10 | 076 | 14134.8 | 15.324 | 15629 | 303 | exact_preserve | assignment_wall,connectivity_wall,documented_wall,exact_preserve,high_memory,open_angle,scatter |
| 11 | 367 | 14092.0 | 15.327 | 15300 | 590 | exact_preserve | connectivity_wall,conv_heavy,documented_wall,exact_preserve,high_memory,local_stencil,lut_selection,open_angle |
| 12 | 349 | 13227.6 | 15.381 | 13860 | 1182 | exact_preserve | conv_heavy,documented_wall,exact_preserve,high_memory,local_stencil,lut_selection,open_angle,qlinear |
| 13 | 173 | 13034.5 | 15.498 | 13307 | 77 | exact_preserve | exact_preserve,gather_heavy,high_memory,lut_selection,scatter |
| 14 | 138 | 10811.3 | 15.552 | 12215 | 462 | exact_preserve | conv_heavy,documented_wall,exact_preserve,high_memory,local_stencil,open_angle |
| 15 | 191 | 10686.2 | 15.686 | 11009 | 83 | exact_preserve | assignment_wall,conv_heavy,custom_win,exact_preserve,high_memory,open_angle,qlinear |
| 16 | 066 | 10187.7 | 15.778 | 9955 | 166 | semantic_or_handbuilt | marker_routed_path,open_angle,scan |
| 17 | 025 | 9806.1 | 15.766 | 10132 | 104 | exact_preserve | exact_preserve,high_memory,onehot_final_equal,open_angle,scan |
| 18 | 198 | 9514.7 | 15.658 | 11305 | 107 | exact_preserve | connectivity_wall,documented_wall,exact_preserve,gather_heavy,high_memory,local_stencil,open_angle,scan |
| 19 | 370 | 7604.0 | 16.000 | 7331 | 773 | exact_preserve | conv_heavy,exact_preserve,qlinear,scatter |
| 20 | 338 | 7289.7 | 15.868 | 8581 | 669 | exact_preserve | connectivity_wall,documented_wall,exact_preserve,local_stencil,open_angle,qlinear,scan |
| 21 | 324 | 6921.9 | 15.907 | 7308 | 1586 | exact_preserve | assignment_wall,conv_heavy,documented_wall,exact_preserve,local_stencil,qlinear,scan |
| 22 | 216 | 6679.9 | 15.934 | 8584 | 76 | exact_preserve | connectivity_wall,documented_wall,exact_preserve,open_angle,qlinear,scatter |
| 23 | 219 | 5710.0 | 16.117 | 7078 | 132 | unknown | assignment_wall,connectivity_wall,documented_wall,lut_selection,onehot_final_equal,open_angle,scan |
| 24 | 096 | 5682.0 | 16.053 | 7261 | 421 | exact_preserve | documented_wall,exact_preserve,open_angle,scatter |
| 25 | 382 | 5232.0 | 16.346 | 5606 | 126 | exact_preserve | exact_preserve,open_angle,scan |

## residual_spatialop_to_free_einsum_collapse

Collapse residual spatial ops (Conv/GridSample/TopK/MaxPool/wide index planes) into free-input Einsum contractions

- source tasks: 011 022 036 281
- candidates: 17
- expected: {'gain_type': 'memory_by_contracting_spatial_planes_into_free_input_einsum', 'risk': 'medium: not every spatial op is separable; some planes are load-bearing detector banks (all-live over 266 bundled instances). gate strictly on isolated bundled fail=0 + cheaper cost', 'verification': 'op-census our overfit_nets for residual Conv/GridSample/TopK/MaxPool/wide-index planes; for each, min-merge vs any public dump OR hand-rebuild as free-input Einsum; isolated evaluate bundled fail=0 + lower mem+params'}

| rank | task | score | pts | mem | params | source | tags |
|---:|---:|---:|---:|---:|---:|---|---|
| 1 | 233 | 32165.5 | 14.588 | 32796 | 446 | semantic_or_handbuilt | assignment_wall,connectivity_wall,conv_heavy,documented_wall,gather_heavy,high_memory,low_score,lut_selection |
| 2 | 366 | 29966.9 | 14.640 | 30983 | 576 | exact_preserve | assignment_wall,connectivity_wall,documented_wall,exact_preserve,gather_heavy,high_memory,low_score,lut_selection |
| 3 | 285 | 23574.8 | 14.864 | 24684 | 550 | exact_preserve | connectivity_wall,documented_wall,exact_preserve,gather_heavy,high_memory,low_score,open_angle,scatter |
| 4 | 286 | 22796.4 | 14.915 | 21090 | 2881 | semantic_or_handbuilt | connectivity_wall,conv_heavy,documented_wall,high_memory,local_stencil,low_score,qlinear |
| 5 | 054 | 20655.7 | 15.078 | 20091 | 288 | unknown | gather_heavy,high_memory,scan,scatter |
| 6 | 133 | 19819.1 | 15.023 | 20510 | 1016 | exact_preserve | assignment_wall,connectivity_wall,conv_heavy,documented_wall,exact_preserve,high_memory,lut_selection,open_angle |
| 7 | 319 | 15367.8 | 15.251 | 16090 | 1053 | exact_preserve | assignment_wall,documented_wall,exact_preserve,gather_heavy,high_memory,scan |
| 8 | 101 | 14620.5 | 15.422 | 13573 | 874 | semantic_or_handbuilt | assignment_wall,connectivity_wall,gather_heavy,high_memory,lut_selection,onehot_final_equal,qlinear,scatter |
| 9 | 076 | 14134.8 | 15.324 | 15629 | 303 | exact_preserve | assignment_wall,connectivity_wall,documented_wall,exact_preserve,high_memory,open_angle,scatter |
| 10 | 349 | 13227.6 | 15.381 | 13860 | 1182 | exact_preserve | conv_heavy,documented_wall,exact_preserve,high_memory,local_stencil,lut_selection,open_angle,qlinear |
| 11 | 173 | 13034.5 | 15.498 | 13307 | 77 | exact_preserve | exact_preserve,gather_heavy,high_memory,lut_selection,scatter |
| 12 | 138 | 10811.3 | 15.552 | 12215 | 462 | exact_preserve | conv_heavy,documented_wall,exact_preserve,high_memory,local_stencil,open_angle |
| 13 | 025 | 9806.1 | 15.766 | 10132 | 104 | exact_preserve | exact_preserve,high_memory,onehot_final_equal,open_angle,scan |
| 14 | 198 | 9514.7 | 15.658 | 11305 | 107 | exact_preserve | connectivity_wall,documented_wall,exact_preserve,gather_heavy,high_memory,local_stencil,open_angle,scan |
| 15 | 158 | -20983.1 | 15.060 | 18089 | 2646 | exact_preserve | assignment_wall,conv_heavy,documented_wall,exact_preserve,gather_heavy,high_memory,marginal_wall,open_angle |
| 16 | 018 | -76211.7 | 14.856 | 24029 | 1416 | exact_preserve | ambiguity_wall,documented_wall,exact_preserve,gather_heavy,high_memory,low_score,maxpool_scan,scatter |
| 17 | 118 | -89592.2 | 15.584 | 12153 | 130 | exact_preserve | connectivity_wall,conv_heavy,documented_wall,exact_preserve,high_memory,infeasible_exact_wall,information_loss_wall,local_stencil |

## spatial_reducesum_to_einsum_profile_tail

Replace spatial ReduceSum colour-count profiles with free-input Einsum profiles when scorer prices the axes initializer

- source tasks: 012 018 037 203 239 324 378 388 396
- candidates: 98
- expected: {'gain_type': 'params_tail_by_deleting_reduce_axes_initializer', 'risk': 'low-medium: local bundled semantics are exact, but tiny op substitutions should still be submitted freely and watched for Kaggle scorer gaps', 'verification': 'run reports/candidates/reducesum_spatial_to_einsum_probe.py; adopt only bundled fail=0 + lower memory+params + scan_unsigned_topk clean'}

| rank | task | score | pts | mem | params | source | tags |
|---:|---:|---:|---:|---:|---:|---|---|
| 1 | 233 | 32165.5 | 14.588 | 32796 | 446 | semantic_or_handbuilt | assignment_wall,connectivity_wall,conv_heavy,documented_wall,gather_heavy,high_memory,low_score,lut_selection |
| 2 | 366 | 29966.9 | 14.640 | 30983 | 576 | exact_preserve | assignment_wall,connectivity_wall,documented_wall,exact_preserve,gather_heavy,high_memory,low_score,lut_selection |
| 3 | 054 | 20655.7 | 15.078 | 20091 | 288 | unknown | gather_heavy,high_memory,scan,scatter |
| 4 | 133 | 19819.1 | 15.023 | 20510 | 1016 | exact_preserve | assignment_wall,connectivity_wall,conv_heavy,documented_wall,exact_preserve,high_memory,lut_selection,open_angle |
| 5 | 319 | 15367.8 | 15.251 | 16090 | 1053 | exact_preserve | assignment_wall,documented_wall,exact_preserve,gather_heavy,high_memory,scan |
| 6 | 101 | 14620.5 | 15.422 | 13573 | 874 | semantic_or_handbuilt | assignment_wall,connectivity_wall,gather_heavy,high_memory,lut_selection,onehot_final_equal,qlinear,scatter |
| 7 | 191 | 10686.2 | 15.686 | 11009 | 83 | exact_preserve | assignment_wall,conv_heavy,custom_win,exact_preserve,high_memory,open_angle,qlinear |
| 8 | 025 | 9806.1 | 15.766 | 10132 | 104 | exact_preserve | exact_preserve,high_memory,onehot_final_equal,open_angle,scan |
| 9 | 198 | 9514.7 | 15.658 | 11305 | 107 | exact_preserve | connectivity_wall,documented_wall,exact_preserve,gather_heavy,high_memory,local_stencil,open_angle,scan |
| 10 | 064 | 8548.7 | 15.791 | 9852 | 134 | semantic_or_handbuilt | connectivity_wall,documented_wall,onehot_final_equal,open_angle |
| 11 | 377 | 7712.9 | 15.987 | 8075 | 134 | exact_preserve | exact_preserve,local_stencil |
| 12 | 370 | 7604.0 | 16.000 | 7331 | 773 | exact_preserve | conv_heavy,exact_preserve,qlinear,scatter |
| 13 | 324 | 6921.9 | 15.907 | 7308 | 1586 | exact_preserve | assignment_wall,conv_heavy,documented_wall,exact_preserve,local_stencil,qlinear,scan |
| 14 | 096 | 5682.0 | 16.053 | 7261 | 421 | exact_preserve | documented_wall,exact_preserve,open_angle,scatter |
| 15 | 382 | 5232.0 | 16.346 | 5606 | 126 | exact_preserve | exact_preserve,open_angle,scan |
| 16 | 157 | 5024.0 | 16.143 | 6768 | 256 | exact_preserve | assignment_wall,connectivity_wall,documented_wall,exact_preserve,gather_heavy,lut_selection,open_angle,scatter |
| 17 | 192 | 4837.0 | 16.170 | 6193 | 644 | exact_preserve | documented_wall,exact_preserve,open_angle,qlinear |
| 18 | 328 | 4634.0 | 16.200 | 4691 | 1943 | exact_preserve | documented_wall,exact_preserve,lut_selection |
| 19 | 396 | 4444.0 | 16.494 | 4854 | 90 | exact_preserve | conv_heavy,exact_preserve,local_stencil,onehot_final_equal |
| 20 | 182 | 4100.0 | 16.284 | 6024 | 76 | exact_preserve | assignment_wall,connectivity_wall,conv_heavy,documented_wall,exact_preserve,local_stencil,lut_selection,open_angle |
| 21 | 378 | 3932.0 | 16.603 | 4316 | 116 | exact_preserve | exact_preserve,open_angle |
| 22 | 092 | 3899.0 | 16.406 | 4940 | 459 | semantic_or_handbuilt | documented_wall,open_angle |
| 23 | 208 | 3709.0 | 16.655 | 4072 | 137 | exact_preserve | exact_preserve,open_angle,qlinear,scatter |
| 24 | 363 | 3635.0 | 16.673 | 3839 | 296 | exact_preserve | connectivity_wall,conv_heavy,exact_preserve,local_stencil,onehot_final_equal,open_angle |
| 25 | 014 | 3617.0 | 16.677 | 4036 | 81 | exact_preserve | exact_preserve |

## output_coupled_fp16_free_output_recast

Recast fp32 masks feeding only the free graph output to fp16

- source tasks: 222 205 355 377
- candidates: 0
- expected: {'gain_type': 'memory_by_recasting_output-coupled_fp32_masks_to_fp16', 'risk': 'medium: ONNX homogeneous ops require every float operand in the chain to match dtype; stale value_info can also pin types. Shared upstream tensors may feed input-co-bound fp32 paths, so prefer local casts immediately before the final-output operand chain when possible. Clear/regenerate value_info and bundled-evaluate.', 'verification': 'run reports/candidates/fresh_sweep/scan_deployed_fp16.py; for each candidate, rebuild with all homogeneous operands adjusted, bundled evaluate fail=0 + lower memory+params, then scan_unsigned_topk'}

| rank | task | score | pts | mem | params | source | tags |
|---:|---:|---:|---:|---:|---:|---|---|
