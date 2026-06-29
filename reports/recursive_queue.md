# Recursive improvement queue

Generated from `reports/global_layer_inventory.json` and `reports/insight_registry.yaml`.

Use this as a global queue.  When a deep task attempt creates a new mechanism, add it to the registry and regenerate this file.

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
- candidates: 33
- expected: {'gain_type': 'frontier_research', 'risk': 'medium-high because it often requires semantic recompilation rather than graph surgery', 'verification': 'prove no counted full-canvas intermediates in mem profile, then fresh eval before adoption'}

| rank | task | score | pts | mem | params | source | tags |
|---:|---:|---:|---:|---:|---:|---|---|
| 1 | 366 | 36367.8 | 14.497 | 35927 | 490 | exact_preserve | exact_preserve,gather_heavy,high_memory,low_score,lut_selection,open_angle,scan,scatter |
| 2 | 191 | 32030.1 | 14.623 | 31276 | 841 | exact_preserve | assignment_wall,conv_heavy,custom_win,exact_preserve,high_memory,low_score,matmul,open_angle |
| 3 | 054 | 26977.4 | 14.792 | 26877 | 238 | exact_preserve | exact_preserve,gather_heavy,high_memory,low_score,scan,scatter |
| 4 | 285 | 25475.5 | 14.848 | 25080 | 550 | exact_preserve | exact_preserve,gather_heavy,high_memory,low_score,scatter |
| 5 | 349 | 24541.9 | 14.827 | 26100 | 90 | exact_preserve | conv_heavy,documented_wall,exact_preserve,high_memory,local_stencil,low_score,lut_selection,maxpool_scan |
| 6 | 173 | 22962.9 | 14.950 | 23036 | 112 | exact_preserve | exact_preserve,gather_heavy,high_memory,low_score,scatter |
| 7 | 066 | 18161.3 | 15.179 | 17763 | 652 | exact_preserve | exact_preserve,high_memory,marker_routed_path,open_angle,scan |
| 8 | 101 | 17581.8 | 15.211 | 16940 | 905 | exact_preserve | exact_preserve,gather_heavy,high_memory,scatter |
| 9 | 025 | 13926.5 | 15.435 | 14062 | 195 | exact_preserve | exact_preserve,high_memory,onehot_final_equal,open_angle,scan |
| 10 | 396 | 12327.6 | 15.551 | 12557 | 136 | exact_preserve | conv_heavy,exact_preserve,high_memory,local_stencil |
| 11 | 138 | 12124.2 | 15.456 | 13789 | 172 | exact_preserve | conv_heavy,documented_wall,exact_preserve,high_memory,local_stencil,open_angle |
| 12 | 017 | 11828.7 | 15.631 | 9330 | 2388 | semantic_or_handbuilt | custom_win,onehot_final_equal,open_angle |
| 13 | 370 | 9968.7 | 15.751 | 9127 | 1267 | exact_preserve | conv_heavy,exact_preserve,qlinear,scan,scatter |
| 14 | 080 | 9387.4 | 15.669 | 10834 | 454 | exact_preserve | conv_heavy,documented_wall,exact_preserve,high_memory,lut_selection,open_angle,qlinear |
| 15 | 222 | 8839.2 | 15.916 | 8736 | 78 | unknown | conv_heavy,custom_win,local_stencil,maxpool_scan,onehot_final_equal,open_angle,qlinear |
| 16 | 074 | 8583.2 | 15.889 | 9000 | 50 | exact_preserve | exact_preserve,local_stencil,open_angle |
| 17 | 014 | 7485.0 | 16.015 | 7878 | 107 | exact_preserve | exact_preserve,open_angle |
| 18 | 096 | 6941.6 | 15.905 | 8108 | 805 | exact_preserve | documented_wall,exact_preserve,open_angle,scatter |
| 19 | 324 | 6921.9 | 15.907 | 7308 | 1586 | exact_preserve | conv_heavy,documented_wall,exact_preserve,local_stencil,qlinear,scan |
| 20 | 192 | 6639.6 | 15.938 | 8515 | 106 | exact_preserve | documented_wall,exact_preserve,local_stencil,open_angle |
| 21 | 328 | 6411.0 | 16.159 | 4971 | 1940 | exact_preserve | exact_preserve,lut_selection |
| 22 | 208 | 5583.0 | 16.287 | 5974 | 109 | exact_preserve | exact_preserve,onehot_final_equal,open_angle,qlinear |
| 23 | 279 | 5555.0 | 16.378 | 5508 | 47 | semantic_or_handbuilt | conv_heavy,local_stencil,qlinear |
| 24 | 382 | 5283.0 | 16.337 | 5648 | 135 | exact_preserve | exact_preserve,open_angle,scan |
| 25 | 092 | 5007.0 | 16.145 | 6825 | 182 | exact_preserve | documented_wall,exact_preserve,open_angle,scatter |

## qlinear_uint8_lut_or_matmul

Replace fp16/fp32 one-hot LUT/MatMul selection with uint8 QLinearMatMul/QLinearConv where exact

- source tasks: 055 080
- candidates: 2
- expected: {'gain_type': 'memory', 'risk': 'medium', 'verification': 'stored eval, fresh eval if generator available, then adopt gate'}

| rank | task | score | pts | mem | params | source | tags |
|---:|---:|---:|---:|---:|---:|---|---|
| 1 | 107 | 3578.0 | 16.687 | 2924 | 1154 | exact_preserve | exact_preserve,lut_selection,matmul,onehot_final_equal,open_angle,scatter |
| 2 | 381 | 198.0 | 17.305 | 2128 | 70 | exact_preserve | connectivity_wall,documented_wall,exact_preserve,matmul,onehot_final_equal,open_angle |

## free_final_onehot_equal

Delay 10-channel expansion to final Equal/Where output

- source tasks: 017 095 146
- candidates: 23
- expected: {'gain_type': 'memory', 'risk': 'low-medium', 'verification': 'output equivalence before adoption'}

| rank | task | score | pts | mem | params | source | tags |
|---:|---:|---:|---:|---:|---:|---|---|
| 1 | 025 | 13926.5 | 15.435 | 14062 | 195 | exact_preserve | exact_preserve,high_memory,onehot_final_equal,open_angle,scan |
| 2 | 017 | 11828.7 | 15.631 | 9330 | 2388 | semantic_or_handbuilt | custom_win,onehot_final_equal,open_angle |
| 3 | 064 | 11587.7 | 15.494 | 13368 | 68 | exact_preserve | connectivity_wall,documented_wall,exact_preserve,high_memory,onehot_final_equal,open_angle |
| 4 | 222 | 8839.2 | 15.916 | 8736 | 78 | unknown | conv_heavy,custom_win,local_stencil,maxpool_scan,onehot_final_equal,open_angle,qlinear |
| 5 | 055 | 5838.0 | 16.328 | 5790 | 48 | semantic_or_handbuilt | connectivity_wall,lut_selection,onehot_final_equal,open_angle,qlinear,scan |
| 6 | 208 | 5583.0 | 16.287 | 5974 | 109 | exact_preserve | exact_preserve,onehot_final_equal,open_angle,qlinear |
| 7 | 009 | 5124.0 | 16.129 | 7029 | 95 | exact_preserve | connectivity_wall,documented_wall,exact_preserve,local_stencil,lut_selection,onehot_final_equal,open_angle |
| 8 | 383 | 4570.0 | 16.289 | 5974 | 96 | unknown | assignment_wall,connectivity_wall,documented_wall,local_stencil,onehot_final_equal,open_angle |
| 9 | 368 | 4225.0 | 16.264 | 6022 | 203 | exact_preserve | assignment_wall,connectivity_wall,documented_wall,exact_preserve,gather_heavy,onehot_final_equal,open_angle,scatter |
| 10 | 264 | 4054.0 | 16.292 | 5348 | 706 | exact_preserve | assignment_wall,documented_wall,exact_preserve,local_stencil,lut_selection,onehot_final_equal,open_angle |
| 11 | 107 | 3578.0 | 16.687 | 2924 | 1154 | exact_preserve | exact_preserve,lut_selection,matmul,onehot_final_equal,open_angle,scatter |
| 12 | 177 | 3322.0 | 16.751 | 3692 | 130 | exact_preserve | connectivity_wall,conv_heavy,exact_preserve,local_stencil,onehot_final_equal,open_angle |
| 13 | 036 | 1677.0 | 17.314 | 2140 | 37 | exact_preserve | connectivity_wall,exact_preserve,onehot_final_equal,open_angle |
| 14 | 392 | 1451.0 | 17.424 | 1602 | 349 | exact_preserve | custom_win,exact_preserve,onehot_final_equal,open_angle |
| 15 | 213 | 1387.0 | 17.457 | 1838 | 49 | exact_preserve | exact_preserve,onehot_final_equal,open_angle |
| 16 | 333 | 1227.0 | 16.921 | 2792 | 435 | exact_preserve | connectivity_wall,documented_wall,exact_preserve,onehot_final_equal,open_angle |
| 17 | 303 | 1120.0 | 17.610 | 1598 | 22 | exact_preserve | exact_preserve,onehot_final_equal,open_angle |
| 18 | 308 | 988.0 | 17.695 | 1385 | 103 | exact_preserve | connectivity_wall,exact_preserve,onehot_final_equal,open_angle,scatter |
| 19 | 071 | 698.0 | 17.100 | 2643 | 55 | exact_preserve | connectivity_wall,documented_wall,exact_preserve,onehot_final_equal,open_angle |
| 20 | 170 | 545.0 | 17.158 | 2216 | 329 | exact_preserve | assignment_wall,conv_heavy,documented_wall,exact_preserve,local_stencil,lut_selection,onehot_final_equal,open_angle |
| 21 | 184 | 520.0 | 17.168 | 2460 | 60 | exact_preserve | connectivity_wall,conv_heavy,documented_wall,exact_preserve,local_stencil,onehot_final_equal,open_angle,scan |
| 22 | 381 | 198.0 | 17.305 | 2128 | 70 | exact_preserve | connectivity_wall,documented_wall,exact_preserve,matmul,onehot_final_equal,open_angle |
| 23 | 029 | -29644.6 | 15.715 | 10736 | 34 | exact_preserve | exact_preserve,high_memory,marginal_wall,onehot_final_equal,open_angle |

## scan_dtype_and_shift_compression

Compress scan-style MaxPool/CumSum/Hillis-Steele pipelines with lower dtype or shared shifts

- source tasks: 046 216
- candidates: 36
- expected: {'gain_type': 'memory', 'risk': 'medium-high', 'verification': 'stored/fresh eval and compare against incumbent'}

| rank | task | score | pts | mem | params | source | tags |
|---:|---:|---:|---:|---:|---:|---|---|
| 1 | 233 | 59811.2 | 14.003 | 59147 | 565 | exact_preserve | assignment_wall,connectivity_wall,conv_heavy,exact_preserve,gather_heavy,high_memory,low_score,matmul |
| 2 | 366 | 36367.8 | 14.497 | 35927 | 490 | exact_preserve | exact_preserve,gather_heavy,high_memory,low_score,lut_selection,open_angle,scan,scatter |
| 3 | 187 | 33940.9 | 14.580 | 32850 | 665 | semantic_or_handbuilt | connectivity_wall,high_memory,local_stencil,low_score,maxpool_scan |
| 4 | 133 | 32008.5 | 14.578 | 32294 | 1288 | exact_preserve | assignment_wall,connectivity_wall,conv_heavy,documented_wall,exact_preserve,gather_heavy,heuristic,high_memory |
| 5 | 054 | 26977.4 | 14.792 | 26877 | 238 | exact_preserve | exact_preserve,gather_heavy,high_memory,low_score,scan,scatter |
| 6 | 285 | 25475.5 | 14.848 | 25080 | 550 | exact_preserve | exact_preserve,gather_heavy,high_memory,low_score,scatter |
| 7 | 349 | 24541.9 | 14.827 | 26100 | 90 | exact_preserve | conv_heavy,documented_wall,exact_preserve,high_memory,local_stencil,low_score,lut_selection,maxpool_scan |
| 8 | 364 | 24160.9 | 14.900 | 24220 | 111 | exact_preserve | connectivity_wall,exact_preserve,high_memory,local_stencil,low_score,maxpool_scan,open_angle |
| 9 | 173 | 22962.9 | 14.950 | 23036 | 112 | exact_preserve | exact_preserve,gather_heavy,high_memory,low_score,scatter |
| 10 | 319 | 20404.0 | 14.997 | 21834 | 269 | exact_preserve | assignment_wall,documented_wall,exact_preserve,gather_heavy,high_memory,low_score,scan |
| 11 | 066 | 18161.3 | 15.179 | 17763 | 652 | exact_preserve | exact_preserve,high_memory,marker_routed_path,open_angle,scan |
| 12 | 101 | 17581.8 | 15.211 | 16940 | 905 | exact_preserve | exact_preserve,gather_heavy,high_memory,scatter |
| 13 | 219 | 16976.3 | 15.162 | 18633 | 92 | exact_preserve | assignment_wall,connectivity_wall,documented_wall,exact_preserve,gather_heavy,high_memory,lut_selection,open_angle |
| 14 | 025 | 13926.5 | 15.435 | 14062 | 195 | exact_preserve | exact_preserve,high_memory,onehot_final_equal,open_angle,scan |
| 15 | 110 | 11948.5 | 15.468 | 13155 | 634 | exact_preserve | connectivity_wall,conv_heavy,documented_wall,exact_preserve,high_memory,maxpool_scan,open_angle |
| 16 | 198 | 11726.7 | 15.484 | 13434 | 138 | exact_preserve | connectivity_wall,documented_wall,exact_preserve,gather_heavy,high_memory,open_angle,scan |
| 17 | 145 | 11361.7 | 15.511 | 13104 | 111 | exact_preserve | connectivity_wall,custom_win,documented_wall,exact_preserve,high_memory,lut_selection,maxpool_scan,open_angle |
| 18 | 370 | 9968.7 | 15.751 | 9127 | 1267 | exact_preserve | conv_heavy,exact_preserve,qlinear,scan,scatter |
| 19 | 222 | 8839.2 | 15.916 | 8736 | 78 | unknown | conv_heavy,custom_win,local_stencil,maxpool_scan,onehot_final_equal,open_angle,qlinear |
| 20 | 204 | 8780.3 | 15.722 | 10244 | 453 | exact_preserve | connectivity_wall,conv_heavy,custom_win,documented_wall,exact_preserve,high_memory,local_stencil,maxpool_scan |
| 21 | 324 | 6921.9 | 15.907 | 7308 | 1586 | exact_preserve | conv_heavy,documented_wall,exact_preserve,local_stencil,qlinear,scan |
| 22 | 157 | 6342.4 | 15.972 | 7983 | 351 | exact_preserve | assignment_wall,connectivity_wall,documented_wall,exact_preserve,gather_heavy,lut_selection,open_angle,scatter |
| 23 | 055 | 5838.0 | 16.328 | 5790 | 48 | semantic_or_handbuilt | connectivity_wall,lut_selection,onehot_final_equal,open_angle,qlinear,scan |
| 24 | 382 | 5283.0 | 16.337 | 5648 | 135 | exact_preserve | exact_preserve,open_angle,scan |
| 25 | 234 | 4854.0 | 16.167 | 6809 | 45 | exact_preserve | documented_wall,exact_preserve,gather_heavy,open_angle |

## sparse_conv_single_op_floor

Collapse local neighborhood rules to one sparse Conv/QLinearConv when output is thresholded

- source tasks: 015 095 230 294
- candidates: 63
- expected: {'gain_type': 'memory+params', 'risk': 'low when rule is truly local', 'verification': 'fresh process eval; beware one-process mem0 false signals'}

| rank | task | score | pts | mem | params | source | tags |
|---:|---:|---:|---:|---:|---:|---|---|
| 1 | 349 | 24541.9 | 14.827 | 26100 | 90 | exact_preserve | conv_heavy,documented_wall,exact_preserve,high_memory,local_stencil,low_score,lut_selection,maxpool_scan |
| 2 | 023 | 12778.9 | 15.517 | 12843 | 291 | exact_preserve | conv_heavy,exact_preserve,high_memory,qlinear |
| 3 | 396 | 12327.6 | 15.551 | 12557 | 136 | exact_preserve | conv_heavy,exact_preserve,high_memory,local_stencil |
| 4 | 138 | 12124.2 | 15.456 | 13789 | 172 | exact_preserve | conv_heavy,documented_wall,exact_preserve,high_memory,local_stencil,open_angle |
| 5 | 370 | 9968.7 | 15.751 | 9127 | 1267 | exact_preserve | conv_heavy,exact_preserve,qlinear,scan,scatter |
| 6 | 080 | 9387.4 | 15.669 | 10834 | 454 | exact_preserve | conv_heavy,documented_wall,exact_preserve,high_memory,lut_selection,open_angle,qlinear |
| 7 | 222 | 8839.2 | 15.916 | 8736 | 78 | unknown | conv_heavy,custom_win,local_stencil,maxpool_scan,onehot_final_equal,open_angle,qlinear |
| 8 | 074 | 8583.2 | 15.889 | 9000 | 50 | exact_preserve | exact_preserve,local_stencil,open_angle |
| 9 | 324 | 6921.9 | 15.907 | 7308 | 1586 | exact_preserve | conv_heavy,documented_wall,exact_preserve,local_stencil,qlinear,scan |
| 10 | 192 | 6639.6 | 15.938 | 8515 | 106 | exact_preserve | documented_wall,exact_preserve,local_stencil,open_angle |
| 11 | 005 | 6382.0 | 16.163 | 6392 | 490 | exact_preserve | conv_heavy,exact_preserve,qlinear |
| 12 | 279 | 5555.0 | 16.378 | 5508 | 47 | semantic_or_handbuilt | conv_heavy,local_stencil,qlinear |
| 13 | 085 | 4881.0 | 16.409 | 4500 | 881 | exact_preserve | exact_preserve,local_stencil,open_angle |
| 14 | 278 | 4131.0 | 16.279 | 6084 | 47 | exact_preserve | documented_wall,exact_preserve,local_stencil,open_angle,qlinear |
| 15 | 117 | 3665.0 | 16.666 | 3922 | 243 | exact_preserve | exact_preserve,local_stencil,lut_selection,open_angle,qlinear |
| 16 | 132 | 3589.0 | 16.684 | 3990 | 99 | exact_preserve | bitwise_program,exact_preserve,gather_heavy,local_stencil,open_angle |
| 17 | 162 | 3568.0 | 16.689 | 4024 | 44 | exact_preserve | exact_preserve,local_stencil,open_angle,qlinear |
| 18 | 284 | 3235.0 | 16.774 | 3082 | 653 | exact_preserve | conv_heavy,custom_win,exact_preserve,open_angle,scatter |
| 19 | 335 | 3001.0 | 16.839 | 2840 | 661 | exact_preserve | conv_heavy,exact_preserve,local_stencil,open_angle |
| 20 | 079 | 2643.0 | 16.947 | 3065 | 78 | exact_preserve | custom_win,exact_preserve,local_stencil,open_angle |
| 21 | 042 | 2610.0 | 16.958 | 2792 | 318 | exact_preserve | conv_heavy,exact_preserve,local_stencil |
| 22 | 354 | 2577.0 | 16.968 | 2998 | 79 | exact_preserve | conv_heavy,exact_preserve,local_stencil,open_angle |
| 23 | 091 | 2528.0 | 16.984 | 2853 | 175 | exact_preserve | exact_preserve,local_stencil,lut_selection,open_angle |
| 24 | 397 | 2348.0 | 17.046 | 2648 | 200 | exact_preserve | exact_preserve,local_stencil,scatter |
| 25 | 224 | 2331.0 | 17.052 | 2460 | 371 | exact_preserve | conv_heavy,exact_preserve,local_stencil,open_angle |

## exact_preserve_to_semantic_rewrite

Prioritize exact-preserve source builders for semantic replacement

- source tasks: 002 018 286 366
- candidates: 38
- expected: {'gain_type': 'research', 'risk': 'high', 'verification': 'document semantic mechanism before implementation'}

| rank | task | score | pts | mem | params | source | tags |
|---:|---:|---:|---:|---:|---:|---|---|
| 1 | 233 | 59811.2 | 14.003 | 59147 | 565 | exact_preserve | assignment_wall,connectivity_wall,conv_heavy,exact_preserve,gather_heavy,high_memory,low_score,matmul |
| 2 | 286 | 45540.5 | 14.242 | 46272 | 741 | exact_preserve | bitwise_program,connectivity_wall,documented_wall,exact_preserve,gather_heavy,high_memory,low_score,qlinear |
| 3 | 366 | 36367.8 | 14.497 | 35927 | 490 | exact_preserve | exact_preserve,gather_heavy,high_memory,low_score,lut_selection,open_angle,scan,scatter |
| 4 | 191 | 32030.1 | 14.623 | 31276 | 841 | exact_preserve | assignment_wall,conv_heavy,custom_win,exact_preserve,high_memory,low_score,matmul,open_angle |
| 5 | 133 | 32008.5 | 14.578 | 32294 | 1288 | exact_preserve | assignment_wall,connectivity_wall,conv_heavy,documented_wall,exact_preserve,gather_heavy,heuristic,high_memory |
| 6 | 054 | 26977.4 | 14.792 | 26877 | 238 | exact_preserve | exact_preserve,gather_heavy,high_memory,low_score,scan,scatter |
| 7 | 285 | 25475.5 | 14.848 | 25080 | 550 | exact_preserve | exact_preserve,gather_heavy,high_memory,low_score,scatter |
| 8 | 367 | 25263.0 | 14.800 | 21268 | 5635 | exact_preserve | connectivity_wall,conv_heavy,documented_wall,exact_preserve,high_memory,local_stencil,low_score,lut_selection |
| 9 | 349 | 24541.9 | 14.827 | 26100 | 90 | exact_preserve | conv_heavy,documented_wall,exact_preserve,high_memory,local_stencil,low_score,lut_selection,maxpool_scan |
| 10 | 364 | 24160.9 | 14.900 | 24220 | 111 | exact_preserve | connectivity_wall,exact_preserve,high_memory,local_stencil,low_score,maxpool_scan,open_angle |
| 11 | 173 | 22962.9 | 14.950 | 23036 | 112 | exact_preserve | exact_preserve,gather_heavy,high_memory,low_score,scatter |
| 12 | 002 | 22778.3 | 14.896 | 24320 | 127 | exact_preserve | bitwise_program,connectivity_wall,documented_wall,exact_preserve,gather_heavy,high_memory,low_score,matmul |
| 13 | 076 | 21908.5 | 14.932 | 23296 | 292 | exact_preserve | assignment_wall,connectivity_wall,documented_wall,exact_preserve,high_memory,low_score,open_angle,scatter |
| 14 | 255 | 21703.0 | 14.940 | 23123 | 262 | exact_preserve | connectivity_wall,conv_heavy,documented_wall,exact_preserve,high_memory,low_score,matmul,open_angle |
| 15 | 243 | 20960.4 | 14.972 | 22608 | 44 | exact_preserve | connectivity_wall,conv_heavy,custom_win,documented_wall,exact_preserve,high_memory,low_score,qlinear |
| 16 | 319 | 20404.0 | 14.997 | 21834 | 269 | exact_preserve | assignment_wall,documented_wall,exact_preserve,gather_heavy,high_memory,low_score,scan |
| 17 | 066 | 18161.3 | 15.179 | 17763 | 652 | exact_preserve | exact_preserve,high_memory,marker_routed_path,open_angle,scan |
| 18 | 101 | 17581.8 | 15.211 | 16940 | 905 | exact_preserve | exact_preserve,gather_heavy,high_memory,scatter |
| 19 | 219 | 16976.3 | 15.162 | 18633 | 92 | exact_preserve | assignment_wall,connectivity_wall,documented_wall,exact_preserve,gather_heavy,high_memory,lut_selection,open_angle |
| 20 | 205 | 15064.1 | 15.360 | 14734 | 638 | exact_preserve | connectivity_wall,conv_heavy,exact_preserve,high_memory,open_angle |
| 21 | 025 | 13926.5 | 15.435 | 14062 | 195 | exact_preserve | exact_preserve,high_memory,onehot_final_equal,open_angle,scan |
| 22 | 023 | 12778.9 | 15.517 | 12843 | 291 | exact_preserve | conv_heavy,exact_preserve,high_memory,qlinear |
| 23 | 338 | 12732.7 | 15.414 | 13890 | 667 | exact_preserve | connectivity_wall,documented_wall,exact_preserve,high_memory,local_stencil,open_angle,qlinear |
| 24 | 396 | 12327.6 | 15.551 | 12557 | 136 | exact_preserve | conv_heavy,exact_preserve,high_memory,local_stencil |
| 25 | 138 | 12124.2 | 15.456 | 13789 | 172 | exact_preserve | conv_heavy,documented_wall,exact_preserve,high_memory,local_stencil,open_angle |

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
- candidates: 36
- expected: {'gain_type': 'memory+generalization', 'risk': 'medium', 'verification': 'confirm generator max height/width, stored eval, fresh adopt gate'}

| rank | task | score | pts | mem | params | source | tags |
|---:|---:|---:|---:|---:|---:|---|---|
| 1 | 233 | 59811.2 | 14.003 | 59147 | 565 | exact_preserve | assignment_wall,connectivity_wall,conv_heavy,exact_preserve,gather_heavy,high_memory,low_score,matmul |
| 2 | 286 | 45540.5 | 14.242 | 46272 | 741 | exact_preserve | bitwise_program,connectivity_wall,documented_wall,exact_preserve,gather_heavy,high_memory,low_score,qlinear |
| 3 | 366 | 36367.8 | 14.497 | 35927 | 490 | exact_preserve | exact_preserve,gather_heavy,high_memory,low_score,lut_selection,open_angle,scan,scatter |
| 4 | 187 | 33940.9 | 14.580 | 32850 | 665 | semantic_or_handbuilt | connectivity_wall,high_memory,local_stencil,low_score,maxpool_scan |
| 5 | 191 | 32030.1 | 14.623 | 31276 | 841 | exact_preserve | assignment_wall,conv_heavy,custom_win,exact_preserve,high_memory,low_score,matmul,open_angle |
| 6 | 133 | 32008.5 | 14.578 | 32294 | 1288 | exact_preserve | assignment_wall,connectivity_wall,conv_heavy,documented_wall,exact_preserve,gather_heavy,heuristic,high_memory |
| 7 | 054 | 26977.4 | 14.792 | 26877 | 238 | exact_preserve | exact_preserve,gather_heavy,high_memory,low_score,scan,scatter |
| 8 | 285 | 25475.5 | 14.848 | 25080 | 550 | exact_preserve | exact_preserve,gather_heavy,high_memory,low_score,scatter |
| 9 | 349 | 24541.9 | 14.827 | 26100 | 90 | exact_preserve | conv_heavy,documented_wall,exact_preserve,high_memory,local_stencil,low_score,lut_selection,maxpool_scan |
| 10 | 364 | 24160.9 | 14.900 | 24220 | 111 | exact_preserve | connectivity_wall,exact_preserve,high_memory,local_stencil,low_score,maxpool_scan,open_angle |
| 11 | 173 | 22962.9 | 14.950 | 23036 | 112 | exact_preserve | exact_preserve,gather_heavy,high_memory,low_score,scatter |
| 12 | 076 | 21908.5 | 14.932 | 23296 | 292 | exact_preserve | assignment_wall,connectivity_wall,documented_wall,exact_preserve,high_memory,low_score,open_angle,scatter |
| 13 | 255 | 21703.0 | 14.940 | 23123 | 262 | exact_preserve | connectivity_wall,conv_heavy,documented_wall,exact_preserve,high_memory,low_score,matmul,open_angle |
| 14 | 319 | 20404.0 | 14.997 | 21834 | 269 | exact_preserve | assignment_wall,documented_wall,exact_preserve,gather_heavy,high_memory,low_score,scan |
| 15 | 066 | 18161.3 | 15.179 | 17763 | 652 | exact_preserve | exact_preserve,high_memory,marker_routed_path,open_angle,scan |
| 16 | 101 | 17581.8 | 15.211 | 16940 | 905 | exact_preserve | exact_preserve,gather_heavy,high_memory,scatter |
| 17 | 219 | 16976.3 | 15.162 | 18633 | 92 | exact_preserve | assignment_wall,connectivity_wall,documented_wall,exact_preserve,gather_heavy,high_memory,lut_selection,open_angle |
| 18 | 205 | 15064.1 | 15.360 | 14734 | 638 | exact_preserve | connectivity_wall,conv_heavy,exact_preserve,high_memory,open_angle |
| 19 | 025 | 13926.5 | 15.435 | 14062 | 195 | exact_preserve | exact_preserve,high_memory,onehot_final_equal,open_angle,scan |
| 20 | 023 | 12778.9 | 15.517 | 12843 | 291 | exact_preserve | conv_heavy,exact_preserve,high_memory,qlinear |
| 21 | 338 | 12732.7 | 15.414 | 13890 | 667 | exact_preserve | connectivity_wall,documented_wall,exact_preserve,high_memory,local_stencil,open_angle,qlinear |
| 22 | 396 | 12327.6 | 15.551 | 12557 | 136 | exact_preserve | conv_heavy,exact_preserve,high_memory,local_stencil |
| 23 | 138 | 12124.2 | 15.456 | 13789 | 172 | exact_preserve | conv_heavy,documented_wall,exact_preserve,high_memory,local_stencil,open_angle |
| 24 | 110 | 11948.5 | 15.468 | 13155 | 634 | exact_preserve | connectivity_wall,conv_heavy,documented_wall,exact_preserve,high_memory,maxpool_scan,open_angle |
| 25 | 198 | 11726.7 | 15.484 | 13434 | 138 | exact_preserve | connectivity_wall,documented_wall,exact_preserve,gather_heavy,high_memory,open_angle,scan |

## public_teacher_bitwise_scan_replacement

Replace repeated MaxPool/Max/Min scan stacks with bitwise shift/mask routing when the state is binary

- source tasks: 002 209
- candidates: 15
- expected: {'gain_type': 'memory', 'risk': 'medium-high', 'verification': 'stored eval, fresh eval when generator available, public-probe if strict fresh is known weak'}

| rank | task | score | pts | mem | params | source | tags |
|---:|---:|---:|---:|---:|---:|---|---|
| 1 | 233 | 59811.2 | 14.003 | 59147 | 565 | exact_preserve | assignment_wall,connectivity_wall,conv_heavy,exact_preserve,gather_heavy,high_memory,low_score,matmul |
| 2 | 187 | 33940.9 | 14.580 | 32850 | 665 | semantic_or_handbuilt | connectivity_wall,high_memory,local_stencil,low_score,maxpool_scan |
| 3 | 191 | 32030.1 | 14.623 | 31276 | 841 | exact_preserve | assignment_wall,conv_heavy,custom_win,exact_preserve,high_memory,low_score,matmul,open_angle |
| 4 | 133 | 32008.5 | 14.578 | 32294 | 1288 | exact_preserve | assignment_wall,connectivity_wall,conv_heavy,documented_wall,exact_preserve,gather_heavy,heuristic,high_memory |
| 5 | 285 | 25475.5 | 14.848 | 25080 | 550 | exact_preserve | exact_preserve,gather_heavy,high_memory,low_score,scatter |
| 6 | 349 | 24541.9 | 14.827 | 26100 | 90 | exact_preserve | conv_heavy,documented_wall,exact_preserve,high_memory,local_stencil,low_score,lut_selection,maxpool_scan |
| 7 | 364 | 24160.9 | 14.900 | 24220 | 111 | exact_preserve | connectivity_wall,exact_preserve,high_memory,local_stencil,low_score,maxpool_scan,open_angle |
| 8 | 173 | 22962.9 | 14.950 | 23036 | 112 | exact_preserve | exact_preserve,gather_heavy,high_memory,low_score,scatter |
| 9 | 076 | 21908.5 | 14.932 | 23296 | 292 | exact_preserve | assignment_wall,connectivity_wall,documented_wall,exact_preserve,high_memory,low_score,open_angle,scatter |
| 10 | 066 | 18161.3 | 15.179 | 17763 | 652 | exact_preserve | exact_preserve,high_memory,marker_routed_path,open_angle,scan |
| 11 | 219 | 16976.3 | 15.162 | 18633 | 92 | exact_preserve | assignment_wall,connectivity_wall,documented_wall,exact_preserve,gather_heavy,high_memory,lut_selection,open_angle |
| 12 | 158 | -6231.3 | 14.528 | 32987 | 2340 | exact_preserve | assignment_wall,conv_heavy,documented_wall,exact_preserve,gather_heavy,high_memory,low_score,marginal_wall |
| 13 | 018 | -50264.1 | 14.157 | 48196 | 2987 | exact_preserve | ambiguity_wall,documented_wall,exact_preserve,gather_heavy,high_memory,low_score,maxpool_scan,scatter |
| 14 | 118 | -67129.9 | 14.553 | 31049 | 3387 | exact_preserve | connectivity_wall,conv_heavy,documented_wall,exact_preserve,heuristic,high_memory,infeasible_exact_wall,information_loss_wall |
| 15 | 209 | -69374.0 | 14.620 | 32027 | 185 | exact_preserve | ambiguity_wall,assignment_wall,connectivity_wall,conv_heavy,documented_wall,exact_preserve,high_memory,low_score |

## public_teacher_qlinear_conv_rewrite

Convert repeated binary/count Conv/ConvTranspose towers to QLinearConv/uint8 routes

- source tasks: 023 349 182 233 255
- candidates: 37
- expected: {'gain_type': 'memory+params', 'risk': 'medium', 'verification': 'stored eval plus fresh; inspect for output-range saturation before adoption'}

| rank | task | score | pts | mem | params | source | tags |
|---:|---:|---:|---:|---:|---:|---|---|
| 1 | 187 | 33940.9 | 14.580 | 32850 | 665 | semantic_or_handbuilt | connectivity_wall,high_memory,local_stencil,low_score,maxpool_scan |
| 2 | 364 | 24160.9 | 14.900 | 24220 | 111 | exact_preserve | connectivity_wall,exact_preserve,high_memory,local_stencil,low_score,maxpool_scan,open_angle |
| 3 | 255 | 21703.0 | 14.940 | 23123 | 262 | exact_preserve | connectivity_wall,conv_heavy,documented_wall,exact_preserve,high_memory,low_score,matmul,open_angle |
| 4 | 205 | 15064.1 | 15.360 | 14734 | 638 | exact_preserve | connectivity_wall,conv_heavy,exact_preserve,high_memory,open_angle |
| 5 | 338 | 12732.7 | 15.414 | 13890 | 667 | exact_preserve | connectivity_wall,documented_wall,exact_preserve,high_memory,local_stencil,open_angle,qlinear |
| 6 | 396 | 12327.6 | 15.551 | 12557 | 136 | exact_preserve | conv_heavy,exact_preserve,high_memory,local_stencil |
| 7 | 138 | 12124.2 | 15.456 | 13789 | 172 | exact_preserve | conv_heavy,documented_wall,exact_preserve,high_memory,local_stencil,open_angle |
| 8 | 110 | 11948.5 | 15.468 | 13155 | 634 | exact_preserve | connectivity_wall,conv_heavy,documented_wall,exact_preserve,high_memory,maxpool_scan,open_angle |
| 9 | 202 | 10876.2 | 15.669 | 11253 | 24 | exact_preserve | connectivity_wall,exact_preserve,high_memory,local_stencil,open_angle,qlinear |
| 10 | 080 | 9387.4 | 15.669 | 10834 | 454 | exact_preserve | conv_heavy,documented_wall,exact_preserve,high_memory,lut_selection,open_angle,qlinear |
| 11 | 379 | 9276.6 | 15.818 | 9069 | 653 | exact_preserve | connectivity_wall,conv_heavy,custom_win,exact_preserve,open_angle |
| 12 | 222 | 8839.2 | 15.916 | 8736 | 78 | unknown | conv_heavy,custom_win,local_stencil,maxpool_scan,onehot_final_equal,open_angle,qlinear |
| 13 | 074 | 8583.2 | 15.889 | 9000 | 50 | exact_preserve | exact_preserve,local_stencil,open_angle |
| 14 | 377 | 8019.5 | 15.952 | 8351 | 154 | exact_preserve | connectivity_wall,exact_preserve,local_stencil,open_angle |
| 15 | 324 | 6921.9 | 15.907 | 7308 | 1586 | exact_preserve | conv_heavy,documented_wall,exact_preserve,local_stencil,qlinear,scan |
| 16 | 192 | 6639.6 | 15.938 | 8515 | 106 | exact_preserve | documented_wall,exact_preserve,local_stencil,open_angle |
| 17 | 005 | 6382.0 | 16.163 | 6392 | 490 | exact_preserve | conv_heavy,exact_preserve,qlinear |
| 18 | 009 | 5124.0 | 16.129 | 7029 | 95 | exact_preserve | connectivity_wall,documented_wall,exact_preserve,local_stencil,lut_selection,onehot_final_equal,open_angle |
| 19 | 004 | 4894.0 | 16.407 | 5300 | 94 | exact_preserve | connectivity_wall,exact_preserve,local_stencil,open_angle |
| 20 | 085 | 4881.0 | 16.409 | 4500 | 881 | exact_preserve | exact_preserve,local_stencil,open_angle |
| 21 | 165 | 4150.0 | 16.276 | 5836 | 314 | exact_preserve | connectivity_wall,documented_wall,exact_preserve,local_stencil,open_angle,qlinear |
| 22 | 363 | 4136.0 | 16.558 | 4339 | 297 | exact_preserve | connectivity_wall,conv_heavy,exact_preserve,local_stencil,open_angle |
| 23 | 132 | 3589.0 | 16.684 | 3990 | 99 | exact_preserve | bitwise_program,exact_preserve,gather_heavy,local_stencil,open_angle |
| 24 | 177 | 3322.0 | 16.751 | 3692 | 130 | exact_preserve | connectivity_wall,conv_heavy,exact_preserve,local_stencil,onehot_final_equal,open_angle |
| 25 | 284 | 3235.0 | 16.774 | 3082 | 653 | exact_preserve | conv_heavy,custom_win,exact_preserve,open_angle,scatter |

## marker_routed_hidden_path_compiler

Compile endpoint-pair hidden paths from deliberate marker pixels instead of replaying scan-heavy exact graphs

- source tasks: 066
- candidates: 1
- expected: {'gain_type': 'semantic_rewrite_memory', 'risk': 'medium-high until tie-break is exact', 'verification': 'stored validation plus large fresh generator eval before ONNX adoption'}

| rank | task | score | pts | mem | params | source | tags |
|---:|---:|---:|---:|---:|---:|---|---|
| 1 | 066 | 18161.3 | 15.179 | 17763 | 652 | exact_preserve | exact_preserve,high_memory,marker_routed_path,open_angle,scan |

## rotation_component_template_scatter

Replace scan-heavy in-context sprite reconstruction with rotation-only 3x3 component template scatter

- source tasks: 233
- candidates: 15
- expected: {'gain_type': 'semantic_rewrite_memory', 'risk': 'medium-high until bbox/component edge cases are fully fresh-verified', 'verification': 'Python reference stored plus large fresh, then source-owned ONNX compiler and adopt gate'}

| rank | task | score | pts | mem | params | source | tags |
|---:|---:|---:|---:|---:|---:|---|---|
| 1 | 286 | 45540.5 | 14.242 | 46272 | 741 | exact_preserve | bitwise_program,connectivity_wall,documented_wall,exact_preserve,gather_heavy,high_memory,low_score,qlinear |
| 2 | 366 | 36367.8 | 14.497 | 35927 | 490 | exact_preserve | exact_preserve,gather_heavy,high_memory,low_score,lut_selection,open_angle,scan,scatter |
| 3 | 187 | 33940.9 | 14.580 | 32850 | 665 | semantic_or_handbuilt | connectivity_wall,high_memory,local_stencil,low_score,maxpool_scan |
| 4 | 054 | 26977.4 | 14.792 | 26877 | 238 | exact_preserve | exact_preserve,gather_heavy,high_memory,low_score,scan,scatter |
| 5 | 285 | 25475.5 | 14.848 | 25080 | 550 | exact_preserve | exact_preserve,gather_heavy,high_memory,low_score,scatter |
| 6 | 367 | 25263.0 | 14.800 | 21268 | 5635 | exact_preserve | connectivity_wall,conv_heavy,documented_wall,exact_preserve,high_memory,local_stencil,low_score,lut_selection |
| 7 | 349 | 24541.9 | 14.827 | 26100 | 90 | exact_preserve | conv_heavy,documented_wall,exact_preserve,high_memory,local_stencil,low_score,lut_selection,maxpool_scan |
| 8 | 364 | 24160.9 | 14.900 | 24220 | 111 | exact_preserve | connectivity_wall,exact_preserve,high_memory,local_stencil,low_score,maxpool_scan,open_angle |
| 9 | 173 | 22962.9 | 14.950 | 23036 | 112 | exact_preserve | exact_preserve,gather_heavy,high_memory,low_score,scatter |
| 10 | 002 | 22778.3 | 14.896 | 24320 | 127 | exact_preserve | bitwise_program,connectivity_wall,documented_wall,exact_preserve,gather_heavy,high_memory,low_score,matmul |
| 11 | 255 | 21703.0 | 14.940 | 23123 | 262 | exact_preserve | connectivity_wall,conv_heavy,documented_wall,exact_preserve,high_memory,low_score,matmul,open_angle |
| 12 | 243 | 20960.4 | 14.972 | 22608 | 44 | exact_preserve | connectivity_wall,conv_heavy,custom_win,documented_wall,exact_preserve,high_memory,low_score,qlinear |
| 13 | 066 | 18161.3 | 15.179 | 17763 | 652 | exact_preserve | exact_preserve,high_memory,marker_routed_path,open_angle,scan |
| 14 | 101 | 17581.8 | 15.211 | 16940 | 905 | exact_preserve | exact_preserve,gather_heavy,high_memory,scatter |
| 15 | 018 | -50264.1 | 14.157 | 48196 | 2987 | exact_preserve | ambiguity_wall,documented_wall,exact_preserve,gather_heavy,high_memory,low_score,maxpool_scan,scatter |
