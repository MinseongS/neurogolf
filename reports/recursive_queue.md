# Recursive improvement queue

Generated from `reports/global_layer_inventory.json` and `reports/insight_registry.yaml`.

Use this as a global queue.  When a deep task attempt creates a new mechanism, add it to the registry and regenerate this file.

## qlinear_uint8_lut_or_matmul

Replace fp16/fp32 one-hot LUT/MatMul selection with uint8 QLinearMatMul/QLinearConv where exact

- source tasks: 055 080
- candidates: 3
- expected: {'gain_type': 'memory', 'risk': 'medium', 'verification': 'stored eval, fresh eval if generator available, then adopt gate'}

| rank | task | score | pts | mem | params | source | tags |
|---:|---:|---:|---:|---:|---:|---|---|
| 1 | 107 | 4078.0 | 16.687 | 2924 | 1154 | semantic_or_handbuilt | lut_selection,matmul,onehot_final_equal,open_angle,scatter |
| 2 | 105 | 2962.0 | 17.006 | 2856 | 106 | unknown | custom_win,matmul,onehot_final_equal,open_angle |
| 3 | 124 | 1953.0 | 17.423 | 1879 | 74 | unknown | lut_selection,matmul,open_angle |

## free_final_onehot_equal

Delay 10-channel expansion to final Equal/Where output

- source tasks: 017 095 146
- candidates: 26
- expected: {'gain_type': 'memory', 'risk': 'low-medium', 'verification': 'output equivalence before adoption'}

| rank | task | score | pts | mem | params | source | tags |
|---:|---:|---:|---:|---:|---:|---|---|
| 1 | 064 | 12087.7 | 15.494 | 13368 | 68 | semantic_or_handbuilt | connectivity_wall,documented_wall,high_memory,onehot_final_equal,open_angle |
| 2 | 017 | 11828.7 | 15.631 | 9330 | 2388 | semantic_or_handbuilt | custom_win,onehot_final_equal,open_angle |
| 3 | 202 | 11376.2 | 15.669 | 11253 | 24 | semantic_or_handbuilt | connectivity_wall,high_memory,local_stencil,onehot_final_equal,open_angle,qlinear |
| 4 | 222 | 8839.2 | 15.916 | 8736 | 78 | unknown | conv_heavy,custom_win,local_stencil,maxpool_scan,onehot_final_equal,open_angle,qlinear |
| 5 | 328 | 6912.0 | 16.159 | 4971 | 1941 | unknown | onehot_final_equal |
| 6 | 208 | 6083.0 | 16.287 | 5974 | 109 | unknown | onehot_final_equal,open_angle,qlinear |
| 7 | 055 | 5838.0 | 16.328 | 5790 | 48 | semantic_or_handbuilt | connectivity_wall,lut_selection,onehot_final_equal,open_angle,qlinear,scan |
| 8 | 009 | 5624.0 | 16.129 | 7029 | 95 | unknown | connectivity_wall,documented_wall,local_stencil,lut_selection,onehot_final_equal,open_angle |
| 9 | 165 | 4650.0 | 16.276 | 5836 | 314 | unknown | connectivity_wall,documented_wall,local_stencil,onehot_final_equal,open_angle,qlinear |
| 10 | 383 | 4570.0 | 16.289 | 5974 | 96 | unknown | assignment_wall,connectivity_wall,documented_wall,local_stencil,onehot_final_equal,open_angle |
| 11 | 378 | 4489.0 | 16.591 | 4372 | 117 | unknown | onehot_final_equal,open_angle |
| 12 | 107 | 4078.0 | 16.687 | 2924 | 1154 | semantic_or_handbuilt | lut_selection,matmul,onehot_final_equal,open_angle,scatter |
| 13 | 387 | 3559.0 | 16.471 | 4954 | 105 | unknown | connectivity_wall,documented_wall,onehot_final_equal,open_angle,scatter |
| 14 | 335 | 3501.0 | 16.839 | 2840 | 661 | semantic_or_handbuilt | conv_heavy,local_stencil,onehot_final_equal,open_angle |
| 15 | 105 | 2962.0 | 17.006 | 2856 | 106 | unknown | custom_win,matmul,onehot_final_equal,open_angle |
| 16 | 397 | 2848.0 | 17.046 | 2648 | 200 | semantic_or_handbuilt | local_stencil,onehot_final_equal,scatter |
| 17 | 069 | 2412.0 | 16.728 | 3848 | 64 | unknown | connectivity_wall,documented_wall,onehot_final_equal,open_angle |
| 18 | 051 | 1803.0 | 17.503 | 1702 | 101 | semantic_or_handbuilt | onehot_final_equal,open_angle |
| 19 | 086 | 1636.0 | 16.949 | 2946 | 190 | unknown | assignment_wall,connectivity_wall,documented_wall,local_stencil,onehot_final_equal,open_angle,qlinear |
| 20 | 303 | 1620.0 | 17.610 | 1598 | 22 | semantic_or_handbuilt | onehot_final_equal,open_angle |
| 21 | 141 | 1505.0 | 17.683 | 1404 | 101 | unknown | onehot_final_equal,open_angle |
| 22 | 346 | 1254.0 | 17.866 | 1224 | 30 | unknown | connectivity_wall,local_stencil,lut_selection,onehot_final_equal,open_angle,qlinear,scatter |
| 23 | 239 | 947.0 | 18.147 | 897 | 50 | semantic_or_handbuilt | onehot_final_equal |
| 24 | 048 | -354.0 | 17.956 | 1030 | 116 | unknown | bitwise_program,connectivity_wall,documented_wall,onehot_final_equal,open_angle |
| 25 | 209 | -69374.0 | 14.620 | 32027 | 185 | exact_preserve | ambiguity_wall,assignment_wall,connectivity_wall,conv_heavy,documented_wall,exact_preserve,heuristic,high_memory |

## scan_dtype_and_shift_compression

Compress scan-style MaxPool/CumSum/Hillis-Steele pipelines with lower dtype or shared shifts

- source tasks: 046 216
- candidates: 36
- expected: {'gain_type': 'memory', 'risk': 'medium-high', 'verification': 'stored/fresh eval and compare against incumbent'}

| rank | task | score | pts | mem | params | source | tags |
|---:|---:|---:|---:|---:|---:|---|---|
| 1 | 233 | 60342.8 | 13.994 | 59587 | 654 | exact_preserve | conv_heavy,exact_preserve,gather_heavy,high_memory,low_score,matmul,qlinear,scan |
| 2 | 366 | 34867.8 | 14.497 | 35927 | 490 | exact_preserve | assignment_wall,connectivity_wall,documented_wall,exact_preserve,gather_heavy,heuristic,high_memory,low_score |
| 3 | 187 | 33940.9 | 14.580 | 32850 | 665 | semantic_or_handbuilt | connectivity_wall,high_memory,local_stencil,low_score,maxpool_scan |
| 4 | 133 | 32008.5 | 14.578 | 32294 | 1288 | exact_preserve | assignment_wall,connectivity_wall,conv_heavy,documented_wall,exact_preserve,gather_heavy,heuristic,high_memory |
| 5 | 054 | 26994.5 | 14.792 | 26894 | 238 | exact_preserve | exact_preserve,gather_heavy,heuristic,high_memory,low_score,scan,scatter |
| 6 | 285 | 25477.5 | 14.848 | 25080 | 552 | exact_preserve | exact_preserve,gather_heavy,heuristic,high_memory,low_score,scatter |
| 7 | 349 | 25041.9 | 14.827 | 26100 | 90 | unknown | conv_heavy,documented_wall,high_memory,local_stencil,low_score,lut_selection,maxpool_scan,open_angle |
| 8 | 364 | 24660.9 | 14.900 | 24220 | 111 | unknown | connectivity_wall,high_memory,local_stencil,low_score,maxpool_scan,open_angle,template_match |
| 9 | 173 | 23087.5 | 14.945 | 23159 | 112 | exact_preserve | exact_preserve,gather_heavy,heuristic,high_memory,low_score,scatter |
| 10 | 319 | 20544.9 | 14.990 | 21973 | 269 | exact_preserve | assignment_wall,documented_wall,exact_preserve,gather_heavy,heuristic,high_memory,low_score,scan |
| 11 | 066 | 18161.3 | 15.179 | 17763 | 652 | exact_preserve | exact_preserve,heuristic,high_memory,scan |
| 12 | 101 | 17662.2 | 15.206 | 17015 | 909 | exact_preserve | exact_preserve,gather_heavy,heuristic,high_memory,scatter |
| 13 | 219 | 16976.3 | 15.162 | 18633 | 92 | exact_preserve | assignment_wall,connectivity_wall,documented_wall,exact_preserve,gather_heavy,heuristic,high_memory,lut_selection |
| 14 | 025 | 14436.7 | 15.434 | 14072 | 195 | semantic_or_handbuilt | high_memory,open_angle,scan |
| 15 | 110 | 12448.5 | 15.468 | 13155 | 634 | semantic_or_handbuilt | connectivity_wall,conv_heavy,documented_wall,high_memory,maxpool_scan,open_angle |
| 16 | 198 | 12226.7 | 15.484 | 13434 | 138 | semantic_or_handbuilt | connectivity_wall,documented_wall,gather_heavy,high_memory,open_angle,scan |
| 17 | 145 | 11861.7 | 15.511 | 13104 | 111 | unknown | connectivity_wall,custom_win,documented_wall,high_memory,lut_selection,maxpool_scan,open_angle |
| 18 | 370 | 10486.2 | 15.749 | 9144 | 1267 | semantic_or_handbuilt | conv_heavy,qlinear,scan,scatter |
| 19 | 204 | 9280.3 | 15.722 | 10244 | 453 | semantic_or_handbuilt | connectivity_wall,conv_heavy,custom_win,documented_wall,high_memory,local_stencil,maxpool_scan,open_angle |
| 20 | 222 | 8839.2 | 15.916 | 8736 | 78 | unknown | conv_heavy,custom_win,local_stencil,maxpool_scan,onehot_final_equal,open_angle,qlinear |
| 21 | 324 | 7444.7 | 15.904 | 7330 | 1586 | unknown | conv_heavy,documented_wall,local_stencil,qlinear,scan |
| 22 | 157 | 6342.4 | 15.972 | 7983 | 351 | exact_preserve | assignment_wall,connectivity_wall,documented_wall,exact_preserve,gather_heavy,heuristic,lut_selection,open_angle |
| 23 | 055 | 5838.0 | 16.328 | 5790 | 48 | semantic_or_handbuilt | connectivity_wall,lut_selection,onehot_final_equal,open_angle,qlinear,scan |
| 24 | 382 | 5783.0 | 16.337 | 5648 | 135 | semantic_or_handbuilt | lut_selection,open_angle,scan |
| 25 | 234 | 5354.0 | 16.167 | 6809 | 45 | semantic_or_handbuilt | documented_wall,gather_heavy,open_angle |

## sparse_conv_single_op_floor

Collapse local neighborhood rules to one sparse Conv/QLinearConv when output is thresholded

- source tasks: 015 095 230 294
- candidates: 64
- expected: {'gain_type': 'memory+params', 'risk': 'low when rule is truly local', 'verification': 'fresh process eval; beware one-process mem0 false signals'}

| rank | task | score | pts | mem | params | source | tags |
|---:|---:|---:|---:|---:|---:|---|---|
| 1 | 233 | 60342.8 | 13.994 | 59587 | 654 | exact_preserve | conv_heavy,exact_preserve,gather_heavy,high_memory,low_score,matmul,qlinear,scan |
| 2 | 349 | 25041.9 | 14.827 | 26100 | 90 | unknown | conv_heavy,documented_wall,high_memory,local_stencil,low_score,lut_selection,maxpool_scan,open_angle |
| 3 | 023 | 13278.9 | 15.517 | 12843 | 291 | semantic_or_handbuilt | conv_heavy,high_memory,qlinear |
| 4 | 396 | 12827.6 | 15.551 | 12557 | 136 | unknown | conv_heavy,high_memory,local_stencil |
| 5 | 138 | 12626.2 | 15.456 | 13812 | 151 | unknown | conv_heavy,documented_wall,high_memory,local_stencil,open_angle |
| 6 | 370 | 10486.2 | 15.749 | 9144 | 1267 | semantic_or_handbuilt | conv_heavy,qlinear,scan,scatter |
| 7 | 080 | 9887.4 | 15.669 | 10834 | 454 | unknown | conv_heavy,documented_wall,high_memory,lut_selection,open_angle,qlinear |
| 8 | 074 | 9083.2 | 15.889 | 9000 | 50 | semantic_or_handbuilt | local_stencil,open_angle,template_match |
| 9 | 222 | 8839.2 | 15.916 | 8736 | 78 | unknown | conv_heavy,custom_win,local_stencil,maxpool_scan,onehot_final_equal,open_angle,qlinear |
| 10 | 324 | 7444.7 | 15.904 | 7330 | 1586 | unknown | conv_heavy,documented_wall,local_stencil,qlinear,scan |
| 11 | 192 | 7139.6 | 15.938 | 8515 | 106 | unknown | documented_wall,local_stencil,open_angle |
| 12 | 005 | 6382.0 | 16.163 | 6392 | 490 | exact_preserve | conv_heavy,exact_preserve,heuristic,qlinear |
| 13 | 279 | 5555.0 | 16.378 | 5508 | 47 | semantic_or_handbuilt | conv_heavy,local_stencil,qlinear |
| 14 | 085 | 5381.0 | 16.409 | 4500 | 881 | unknown | local_stencil,open_angle |
| 15 | 278 | 4631.0 | 16.279 | 6084 | 47 | semantic_or_handbuilt | documented_wall,local_stencil,open_angle,qlinear |
| 16 | 117 | 4165.0 | 16.666 | 3922 | 243 | unknown | local_stencil,lut_selection,open_angle,qlinear |
| 17 | 132 | 4089.0 | 16.684 | 3990 | 99 | unknown | bitwise_program,gather_heavy,local_stencil,open_angle |
| 18 | 162 | 4068.0 | 16.689 | 4024 | 44 | unknown | local_stencil,open_angle,qlinear |
| 19 | 284 | 3735.0 | 16.774 | 3082 | 653 | semantic_or_handbuilt | conv_heavy,custom_win,open_angle,scatter |
| 20 | 335 | 3501.0 | 16.839 | 2840 | 661 | semantic_or_handbuilt | conv_heavy,local_stencil,onehot_final_equal,open_angle |
| 21 | 079 | 3144.0 | 16.947 | 3065 | 79 | unknown | custom_win,local_stencil,open_angle |
| 22 | 042 | 3110.0 | 16.958 | 2792 | 318 | semantic_or_handbuilt | conv_heavy,local_stencil |
| 23 | 354 | 3077.0 | 16.968 | 2998 | 79 | unknown | conv_heavy,local_stencil,open_angle |
| 24 | 091 | 3028.0 | 16.984 | 2853 | 175 | semantic_or_handbuilt | local_stencil,open_angle |
| 25 | 397 | 2848.0 | 17.046 | 2648 | 200 | semantic_or_handbuilt | local_stencil,onehot_final_equal,scatter |

## exact_preserve_to_semantic_rewrite

Prioritize exact-preserve source builders for semantic replacement

- source tasks: 002 018 286 366
- candidates: 18
- expected: {'gain_type': 'research', 'risk': 'high', 'verification': 'document semantic mechanism before implementation'}

| rank | task | score | pts | mem | params | source | tags |
|---:|---:|---:|---:|---:|---:|---|---|
| 1 | 233 | 60342.8 | 13.994 | 59587 | 654 | exact_preserve | conv_heavy,exact_preserve,gather_heavy,high_memory,low_score,matmul,qlinear,scan |
| 2 | 286 | 45540.5 | 14.242 | 46272 | 741 | exact_preserve | bitwise_program,connectivity_wall,documented_wall,exact_preserve,gather_heavy,heuristic,high_memory,low_score |
| 3 | 366 | 34867.8 | 14.497 | 35927 | 490 | exact_preserve | assignment_wall,connectivity_wall,documented_wall,exact_preserve,gather_heavy,heuristic,high_memory,low_score |
| 4 | 133 | 32008.5 | 14.578 | 32294 | 1288 | exact_preserve | assignment_wall,connectivity_wall,conv_heavy,documented_wall,exact_preserve,gather_heavy,heuristic,high_memory |
| 5 | 054 | 26994.5 | 14.792 | 26894 | 238 | exact_preserve | exact_preserve,gather_heavy,heuristic,high_memory,low_score,scan,scatter |
| 6 | 285 | 25477.5 | 14.848 | 25080 | 552 | exact_preserve | exact_preserve,gather_heavy,heuristic,high_memory,low_score,scatter |
| 7 | 173 | 23087.5 | 14.945 | 23159 | 112 | exact_preserve | exact_preserve,gather_heavy,heuristic,high_memory,low_score,scatter |
| 8 | 002 | 22778.3 | 14.896 | 24320 | 127 | exact_preserve | bitwise_program,connectivity_wall,documented_wall,exact_preserve,gather_heavy,heuristic,high_memory,low_score |
| 9 | 076 | 21922.7 | 14.931 | 23296 | 306 | exact_preserve | assignment_wall,connectivity_wall,documented_wall,exact_preserve,heuristic,high_memory,low_score,open_angle |
| 10 | 255 | 21703.0 | 14.940 | 23123 | 262 | exact_preserve | connectivity_wall,conv_heavy,documented_wall,exact_preserve,heuristic,high_memory,low_score,matmul |
| 11 | 319 | 20544.9 | 14.990 | 21973 | 269 | exact_preserve | assignment_wall,documented_wall,exact_preserve,gather_heavy,heuristic,high_memory,low_score,scan |
| 12 | 066 | 18161.3 | 15.179 | 17763 | 652 | exact_preserve | exact_preserve,heuristic,high_memory,scan |
| 13 | 101 | 17662.2 | 15.206 | 17015 | 909 | exact_preserve | exact_preserve,gather_heavy,heuristic,high_memory,scatter |
| 14 | 219 | 16976.3 | 15.162 | 18633 | 92 | exact_preserve | assignment_wall,connectivity_wall,documented_wall,exact_preserve,gather_heavy,heuristic,high_memory,lut_selection |
| 15 | 044 | 9592.7 | 15.651 | 11420 | 68 | exact_preserve | assignment_wall,connectivity_wall,conv_heavy,documented_wall,exact_preserve,heuristic,high_memory,matmul |
| 16 | 018 | -50264.1 | 14.157 | 48196 | 2987 | exact_preserve | ambiguity_wall,documented_wall,exact_preserve,gather_heavy,heuristic,high_memory,low_score,maxpool_scan |
| 17 | 118 | -67129.9 | 14.553 | 31049 | 3387 | exact_preserve | connectivity_wall,conv_heavy,documented_wall,exact_preserve,heuristic,high_memory,infeasible_exact_wall,information_loss_wall |
| 18 | 209 | -69374.0 | 14.620 | 32027 | 185 | exact_preserve | ambiguity_wall,assignment_wall,connectivity_wall,conv_heavy,documented_wall,exact_preserve,heuristic,high_memory |

## dihedral_template_match_stacked_conv

Use one stacked Conv for all dihedral orientations of a small runtime-extracted template

- source tasks: 191
- candidates: 7
- expected: {'gain_type': 'memory+accuracy', 'risk': 'medium', 'verification': 'fresh exact eval; compare against incumbent and propagate if template extraction is deterministic'}

| rank | task | score | pts | mem | params | source | tags |
|---:|---:|---:|---:|---:|---:|---|---|
| 1 | 191 | 32530.1 | 14.623 | 31276 | 841 | unknown | assignment_wall,conv_heavy,custom_win,high_memory,low_score,matmul,open_angle,template_match |
| 2 | 364 | 24660.9 | 14.900 | 24220 | 111 | unknown | connectivity_wall,high_memory,local_stencil,low_score,maxpool_scan,open_angle,template_match |
| 3 | 074 | 9083.2 | 15.889 | 9000 | 50 | semantic_or_handbuilt | local_stencil,open_angle,template_match |
| 4 | 363 | 4637.0 | 16.558 | 4339 | 298 | semantic_or_handbuilt | connectivity_wall,conv_heavy,local_stencil,open_angle,template_match |
| 5 | 287 | 1994.0 | 17.402 | 1699 | 295 | semantic_or_handbuilt | open_angle,template_match |
| 6 | 400 | 1866.0 | 17.468 | 1800 | 66 | semantic_or_handbuilt | assignment_wall,connectivity_wall,open_angle,template_match |
| 7 | 158 | -5661.7 | 14.526 | 33053 | 2343 | unknown | assignment_wall,conv_heavy,documented_wall,gather_heavy,high_memory,low_score,marginal_wall,open_angle |

## bounded_crop_before_connectivity_scan

Run iterative connectivity/flood-fill on the generator's true max canvas, not the 30x30 harness canvas

- source tasks: 187
- candidates: 36
- expected: {'gain_type': 'memory+generalization', 'risk': 'medium', 'verification': 'confirm generator max height/width, stored eval, fresh adopt gate'}

| rank | task | score | pts | mem | params | source | tags |
|---:|---:|---:|---:|---:|---:|---|---|
| 1 | 233 | 60342.8 | 13.994 | 59587 | 654 | exact_preserve | conv_heavy,exact_preserve,gather_heavy,high_memory,low_score,matmul,qlinear,scan |
| 2 | 286 | 45540.5 | 14.242 | 46272 | 741 | exact_preserve | bitwise_program,connectivity_wall,documented_wall,exact_preserve,gather_heavy,heuristic,high_memory,low_score |
| 3 | 366 | 34867.8 | 14.497 | 35927 | 490 | exact_preserve | assignment_wall,connectivity_wall,documented_wall,exact_preserve,gather_heavy,heuristic,high_memory,low_score |
| 4 | 187 | 33940.9 | 14.580 | 32850 | 665 | semantic_or_handbuilt | connectivity_wall,high_memory,local_stencil,low_score,maxpool_scan |
| 5 | 191 | 32530.1 | 14.623 | 31276 | 841 | unknown | assignment_wall,conv_heavy,custom_win,high_memory,low_score,matmul,open_angle,template_match |
| 6 | 133 | 32008.5 | 14.578 | 32294 | 1288 | exact_preserve | assignment_wall,connectivity_wall,conv_heavy,documented_wall,exact_preserve,gather_heavy,heuristic,high_memory |
| 7 | 054 | 26994.5 | 14.792 | 26894 | 238 | exact_preserve | exact_preserve,gather_heavy,heuristic,high_memory,low_score,scan,scatter |
| 8 | 285 | 25477.5 | 14.848 | 25080 | 552 | exact_preserve | exact_preserve,gather_heavy,heuristic,high_memory,low_score,scatter |
| 9 | 349 | 25041.9 | 14.827 | 26100 | 90 | unknown | conv_heavy,documented_wall,high_memory,local_stencil,low_score,lut_selection,maxpool_scan,open_angle |
| 10 | 364 | 24660.9 | 14.900 | 24220 | 111 | unknown | connectivity_wall,high_memory,local_stencil,low_score,maxpool_scan,open_angle,template_match |
| 11 | 173 | 23087.5 | 14.945 | 23159 | 112 | exact_preserve | exact_preserve,gather_heavy,heuristic,high_memory,low_score,scatter |
| 12 | 076 | 21922.7 | 14.931 | 23296 | 306 | exact_preserve | assignment_wall,connectivity_wall,documented_wall,exact_preserve,heuristic,high_memory,low_score,open_angle |
| 13 | 255 | 21703.0 | 14.940 | 23123 | 262 | exact_preserve | connectivity_wall,conv_heavy,documented_wall,exact_preserve,heuristic,high_memory,low_score,matmul |
| 14 | 319 | 20544.9 | 14.990 | 21973 | 269 | exact_preserve | assignment_wall,documented_wall,exact_preserve,gather_heavy,heuristic,high_memory,low_score,scan |
| 15 | 066 | 18161.3 | 15.179 | 17763 | 652 | exact_preserve | exact_preserve,heuristic,high_memory,scan |
| 16 | 101 | 17662.2 | 15.206 | 17015 | 909 | exact_preserve | exact_preserve,gather_heavy,heuristic,high_memory,scatter |
| 17 | 219 | 16976.3 | 15.162 | 18633 | 92 | exact_preserve | assignment_wall,connectivity_wall,documented_wall,exact_preserve,gather_heavy,heuristic,high_memory,lut_selection |
| 18 | 205 | 15564.1 | 15.360 | 14734 | 638 | semantic_or_handbuilt | connectivity_wall,conv_heavy,high_memory,lut_selection,open_angle |
| 19 | 025 | 14436.7 | 15.434 | 14072 | 195 | semantic_or_handbuilt | high_memory,open_angle,scan |
| 20 | 023 | 13278.9 | 15.517 | 12843 | 291 | semantic_or_handbuilt | conv_heavy,high_memory,qlinear |
| 21 | 338 | 13232.7 | 15.414 | 13890 | 667 | unknown | connectivity_wall,documented_wall,high_memory,local_stencil,open_angle,qlinear |
| 22 | 396 | 12827.6 | 15.551 | 12557 | 136 | unknown | conv_heavy,high_memory,local_stencil |
| 23 | 138 | 12626.2 | 15.456 | 13812 | 151 | unknown | conv_heavy,documented_wall,high_memory,local_stencil,open_angle |
| 24 | 110 | 12448.5 | 15.468 | 13155 | 634 | semantic_or_handbuilt | connectivity_wall,conv_heavy,documented_wall,high_memory,maxpool_scan,open_angle |
| 25 | 198 | 12226.7 | 15.484 | 13434 | 138 | semantic_or_handbuilt | connectivity_wall,documented_wall,gather_heavy,high_memory,open_angle,scan |

## public_teacher_bitwise_scan_replacement

Replace repeated MaxPool/Max/Min scan stacks with bitwise shift/mask routing when the state is binary

- source tasks: 002 209
- candidates: 15
- expected: {'gain_type': 'memory', 'risk': 'medium-high', 'verification': 'stored eval, fresh eval when generator available, public-probe if strict fresh is known weak'}

| rank | task | score | pts | mem | params | source | tags |
|---:|---:|---:|---:|---:|---:|---|---|
| 1 | 233 | 60342.8 | 13.994 | 59587 | 654 | exact_preserve | conv_heavy,exact_preserve,gather_heavy,high_memory,low_score,matmul,qlinear,scan |
| 2 | 187 | 33940.9 | 14.580 | 32850 | 665 | semantic_or_handbuilt | connectivity_wall,high_memory,local_stencil,low_score,maxpool_scan |
| 3 | 191 | 32530.1 | 14.623 | 31276 | 841 | unknown | assignment_wall,conv_heavy,custom_win,high_memory,low_score,matmul,open_angle,template_match |
| 4 | 133 | 32008.5 | 14.578 | 32294 | 1288 | exact_preserve | assignment_wall,connectivity_wall,conv_heavy,documented_wall,exact_preserve,gather_heavy,heuristic,high_memory |
| 5 | 285 | 25477.5 | 14.848 | 25080 | 552 | exact_preserve | exact_preserve,gather_heavy,heuristic,high_memory,low_score,scatter |
| 6 | 349 | 25041.9 | 14.827 | 26100 | 90 | unknown | conv_heavy,documented_wall,high_memory,local_stencil,low_score,lut_selection,maxpool_scan,open_angle |
| 7 | 364 | 24660.9 | 14.900 | 24220 | 111 | unknown | connectivity_wall,high_memory,local_stencil,low_score,maxpool_scan,open_angle,template_match |
| 8 | 173 | 23087.5 | 14.945 | 23159 | 112 | exact_preserve | exact_preserve,gather_heavy,heuristic,high_memory,low_score,scatter |
| 9 | 076 | 21922.7 | 14.931 | 23296 | 306 | exact_preserve | assignment_wall,connectivity_wall,documented_wall,exact_preserve,heuristic,high_memory,low_score,open_angle |
| 10 | 066 | 18161.3 | 15.179 | 17763 | 652 | exact_preserve | exact_preserve,heuristic,high_memory,scan |
| 11 | 219 | 16976.3 | 15.162 | 18633 | 92 | exact_preserve | assignment_wall,connectivity_wall,documented_wall,exact_preserve,gather_heavy,heuristic,high_memory,lut_selection |
| 12 | 158 | -5661.7 | 14.526 | 33053 | 2343 | unknown | assignment_wall,conv_heavy,documented_wall,gather_heavy,high_memory,low_score,marginal_wall,open_angle |
| 13 | 018 | -50264.1 | 14.157 | 48196 | 2987 | exact_preserve | ambiguity_wall,documented_wall,exact_preserve,gather_heavy,heuristic,high_memory,low_score,maxpool_scan |
| 14 | 118 | -67129.9 | 14.553 | 31049 | 3387 | exact_preserve | connectivity_wall,conv_heavy,documented_wall,exact_preserve,heuristic,high_memory,infeasible_exact_wall,information_loss_wall |
| 15 | 209 | -69374.0 | 14.620 | 32027 | 185 | exact_preserve | ambiguity_wall,assignment_wall,connectivity_wall,conv_heavy,documented_wall,exact_preserve,heuristic,high_memory |

## public_teacher_qlinear_conv_rewrite

Convert repeated binary/count Conv/ConvTranspose towers to QLinearConv/uint8 routes

- source tasks: 023 349 182 233 255
- candidates: 38
- expected: {'gain_type': 'memory+params', 'risk': 'medium', 'verification': 'stored eval plus fresh; inspect for output-range saturation before adoption'}

| rank | task | score | pts | mem | params | source | tags |
|---:|---:|---:|---:|---:|---:|---|---|
| 1 | 233 | 60342.8 | 13.994 | 59587 | 654 | exact_preserve | conv_heavy,exact_preserve,gather_heavy,high_memory,low_score,matmul,qlinear,scan |
| 2 | 187 | 33940.9 | 14.580 | 32850 | 665 | semantic_or_handbuilt | connectivity_wall,high_memory,local_stencil,low_score,maxpool_scan |
| 3 | 364 | 24660.9 | 14.900 | 24220 | 111 | unknown | connectivity_wall,high_memory,local_stencil,low_score,maxpool_scan,open_angle,template_match |
| 4 | 255 | 21703.0 | 14.940 | 23123 | 262 | exact_preserve | connectivity_wall,conv_heavy,documented_wall,exact_preserve,heuristic,high_memory,low_score,matmul |
| 5 | 205 | 15564.1 | 15.360 | 14734 | 638 | semantic_or_handbuilt | connectivity_wall,conv_heavy,high_memory,lut_selection,open_angle |
| 6 | 338 | 13232.7 | 15.414 | 13890 | 667 | unknown | connectivity_wall,documented_wall,high_memory,local_stencil,open_angle,qlinear |
| 7 | 396 | 12827.6 | 15.551 | 12557 | 136 | unknown | conv_heavy,high_memory,local_stencil |
| 8 | 138 | 12626.2 | 15.456 | 13812 | 151 | unknown | conv_heavy,documented_wall,high_memory,local_stencil,open_angle |
| 9 | 110 | 12448.5 | 15.468 | 13155 | 634 | semantic_or_handbuilt | connectivity_wall,conv_heavy,documented_wall,high_memory,maxpool_scan,open_angle |
| 10 | 202 | 11376.2 | 15.669 | 11253 | 24 | semantic_or_handbuilt | connectivity_wall,high_memory,local_stencil,onehot_final_equal,open_angle,qlinear |
| 11 | 080 | 9887.4 | 15.669 | 10834 | 454 | unknown | conv_heavy,documented_wall,high_memory,lut_selection,open_angle,qlinear |
| 12 | 379 | 9776.6 | 15.818 | 9069 | 653 | unknown | connectivity_wall,conv_heavy,custom_win,open_angle |
| 13 | 074 | 9083.2 | 15.889 | 9000 | 50 | semantic_or_handbuilt | local_stencil,open_angle,template_match |
| 14 | 222 | 8839.2 | 15.916 | 8736 | 78 | unknown | conv_heavy,custom_win,local_stencil,maxpool_scan,onehot_final_equal,open_angle,qlinear |
| 15 | 377 | 8519.5 | 15.952 | 8351 | 154 | semantic_or_handbuilt | connectivity_wall,local_stencil,open_angle |
| 16 | 324 | 7444.7 | 15.904 | 7330 | 1586 | unknown | conv_heavy,documented_wall,local_stencil,qlinear,scan |
| 17 | 192 | 7139.6 | 15.938 | 8515 | 106 | unknown | documented_wall,local_stencil,open_angle |
| 18 | 005 | 6382.0 | 16.163 | 6392 | 490 | exact_preserve | conv_heavy,exact_preserve,heuristic,qlinear |
| 19 | 009 | 5624.0 | 16.129 | 7029 | 95 | unknown | connectivity_wall,documented_wall,local_stencil,lut_selection,onehot_final_equal,open_angle |
| 20 | 004 | 5394.0 | 16.407 | 5300 | 94 | unknown | connectivity_wall,local_stencil,open_angle |
| 21 | 085 | 5381.0 | 16.409 | 4500 | 881 | unknown | local_stencil,open_angle |
| 22 | 165 | 4650.0 | 16.276 | 5836 | 314 | unknown | connectivity_wall,documented_wall,local_stencil,onehot_final_equal,open_angle,qlinear |
| 23 | 363 | 4637.0 | 16.558 | 4339 | 298 | semantic_or_handbuilt | connectivity_wall,conv_heavy,local_stencil,open_angle,template_match |
| 24 | 132 | 4089.0 | 16.684 | 3990 | 99 | unknown | bitwise_program,gather_heavy,local_stencil,open_angle |
| 25 | 177 | 3822.0 | 16.751 | 3692 | 130 | unknown | connectivity_wall,conv_heavy,local_stencil,open_angle |
