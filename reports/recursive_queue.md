# Recursive improvement queue

Generated from `reports/global_layer_inventory.json` and `reports/insight_registry.yaml`.

Use this as a global queue.  When a deep task attempt creates a new mechanism, add it to the registry and regenerate this file.

## qlinear_uint8_lut_or_matmul

Replace fp16/fp32 one-hot LUT/MatMul selection with uint8 QLinearMatMul/QLinearConv where exact

- source tasks: 055 080
- candidates: 47
- expected: {'gain_type': 'memory', 'risk': 'medium', 'verification': 'stored eval, fresh eval if generator available, then adopt gate'}

| rank | task | score | pts | mem | params | source | tags |
|---:|---:|---:|---:|---:|---:|---|---|
| 1 | 158 | 34338.3 | 14.526 | 33053 | 2343 | unknown | assignment_wall,conv_heavy,documented_wall,gather_heavy,high_memory,low_score,open_angle,qlinear |
| 2 | 118 | 32870.1 | 14.553 | 31049 | 3387 | exact_preserve | connectivity_wall,conv_heavy,documented_wall,exact_preserve,heuristic,high_memory,local_stencil,low_score |
| 3 | 191 | 32530.1 | 14.623 | 31276 | 841 | unknown | assignment_wall,conv_heavy,high_memory,low_score,lut_selection,matmul,open_angle |
| 4 | 133 | 32008.5 | 14.578 | 32294 | 1288 | exact_preserve | assignment_wall,connectivity_wall,conv_heavy,documented_wall,exact_preserve,gather_heavy,heuristic,high_memory |
| 5 | 367 | 25763.0 | 14.800 | 21268 | 5635 | unknown | connectivity_wall,conv_heavy,documented_wall,high_memory,local_stencil,low_score,lut_selection,open_angle |
| 6 | 243 | 21460.4 | 14.972 | 22608 | 44 | unknown | connectivity_wall,conv_heavy,documented_wall,high_memory,low_score,qlinear |
| 7 | 202 | 13743.1 | 15.483 | 13563 | 25 | semantic_or_handbuilt | connectivity_wall,high_memory,open_angle,qlinear |
| 8 | 077 | 13556.2 | 15.393 | 14760 | 114 | heuristic | connectivity_wall,documented_wall,heuristic,high_memory,local_stencil,maxpool_scan,open_angle,qlinear |
| 9 | 338 | 13232.7 | 15.414 | 13890 | 667 | unknown | connectivity_wall,documented_wall,high_memory,local_stencil,open_angle,qlinear |
| 10 | 370 | 10486.2 | 15.749 | 9144 | 1267 | semantic_or_handbuilt | conv_heavy,qlinear,scan,scatter |
| 11 | 080 | 9887.4 | 15.669 | 10834 | 454 | unknown | conv_heavy,documented_wall,high_memory,lut_selection,open_angle,qlinear |
| 12 | 044 | 9797.9 | 15.634 | 11620 | 68 | exact_preserve | assignment_wall,connectivity_wall,conv_heavy,documented_wall,exact_preserve,heuristic,high_memory,lut_selection |
| 13 | 204 | 9280.3 | 15.722 | 10244 | 453 | semantic_or_handbuilt | connectivity_wall,conv_heavy,documented_wall,high_memory,local_stencil,maxpool_scan,open_angle,qlinear |
| 14 | 222 | 8839.2 | 15.916 | 8736 | 78 | unknown | conv_heavy,local_stencil,maxpool_scan,onehot_final_equal,open_angle,qlinear |
| 15 | 324 | 7444.7 | 15.904 | 7330 | 1586 | unknown | conv_heavy,documented_wall,local_stencil,qlinear,scan |
| 16 | 216 | 7171.0 | 15.880 | 9048 | 87 | exact_preserve | connectivity_wall,documented_wall,exact_preserve,heuristic,matmul,open_angle,qlinear,scatter |
| 17 | 005 | 6600.0 | 16.132 | 6610 | 490 | exact_preserve | conv_heavy,exact_preserve,heuristic,qlinear |
| 18 | 208 | 6083.0 | 16.287 | 5974 | 109 | unknown | onehot_final_equal,open_angle,qlinear |
| 19 | 055 | 5838.0 | 16.328 | 5790 | 48 | semantic_or_handbuilt | connectivity_wall,lut_selection,onehot_final_equal,open_angle,qlinear,scan |
| 20 | 174 | 5655.0 | 16.124 | 7013 | 142 | unknown | connectivity_wall,documented_wall,lut_selection,matmul,open_angle |
| 21 | 279 | 5555.0 | 16.378 | 5508 | 47 | semantic_or_handbuilt | conv_heavy,local_stencil,qlinear |
| 22 | 340 | 4639.0 | 16.278 | 5860 | 279 | unknown | documented_wall,open_angle,qlinear |
| 23 | 278 | 4631.0 | 16.279 | 6084 | 47 | semantic_or_handbuilt | documented_wall,local_stencil,open_angle,qlinear |
| 24 | 117 | 4165.0 | 16.666 | 3922 | 243 | unknown | local_stencil,lut_selection,open_angle,qlinear |
| 25 | 107 | 4078.0 | 16.687 | 2924 | 1154 | semantic_or_handbuilt | lut_selection,matmul,onehot_final_equal,open_angle,scatter |

## free_final_onehot_equal

Delay 10-channel expansion to final Equal/Where output

- source tasks: 017 095 146
- candidates: 25
- expected: {'gain_type': 'memory', 'risk': 'low-medium', 'verification': 'output equivalence before adoption'}

| rank | task | score | pts | mem | params | source | tags |
|---:|---:|---:|---:|---:|---:|---|---|
| 1 | 209 | 49701.7 | 14.158 | 50951 | 198 | exact_preserve | assignment_wall,connectivity_wall,conv_heavy,documented_wall,exact_preserve,heuristic,high_memory,low_score |
| 2 | 017 | 12870.6 | 15.548 | 10931 | 1804 | semantic_or_handbuilt | high_memory,onehot_final_equal,open_angle |
| 3 | 064 | 12087.7 | 15.494 | 13368 | 68 | semantic_or_handbuilt | connectivity_wall,documented_wall,high_memory,onehot_final_equal,open_angle |
| 4 | 222 | 8839.2 | 15.916 | 8736 | 78 | unknown | conv_heavy,local_stencil,maxpool_scan,onehot_final_equal,open_angle,qlinear |
| 5 | 328 | 6912.0 | 16.159 | 4971 | 1941 | unknown | onehot_final_equal |
| 6 | 208 | 6083.0 | 16.287 | 5974 | 109 | unknown | onehot_final_equal,open_angle,qlinear |
| 7 | 055 | 5838.0 | 16.328 | 5790 | 48 | semantic_or_handbuilt | connectivity_wall,lut_selection,onehot_final_equal,open_angle,qlinear,scan |
| 8 | 009 | 5624.0 | 16.129 | 7029 | 95 | unknown | connectivity_wall,documented_wall,local_stencil,lut_selection,onehot_final_equal,open_angle |
| 9 | 165 | 4956.0 | 16.227 | 6144 | 312 | unknown | connectivity_wall,documented_wall,local_stencil,onehot_final_equal,open_angle |
| 10 | 383 | 4570.0 | 16.289 | 5974 | 96 | unknown | assignment_wall,connectivity_wall,documented_wall,local_stencil,onehot_final_equal,open_angle |
| 11 | 378 | 4489.0 | 16.591 | 4372 | 117 | unknown | onehot_final_equal,open_angle |
| 12 | 335 | 4216.0 | 16.653 | 3546 | 670 | semantic_or_handbuilt | conv_heavy,local_stencil,onehot_final_equal,open_angle |
| 13 | 107 | 4078.0 | 16.687 | 2924 | 1154 | semantic_or_handbuilt | lut_selection,matmul,onehot_final_equal,open_angle,scatter |
| 14 | 387 | 3559.0 | 16.471 | 4954 | 105 | unknown | connectivity_wall,documented_wall,onehot_final_equal,open_angle,scatter |
| 15 | 105 | 2962.0 | 17.006 | 2856 | 106 | unknown | lut_selection,matmul,onehot_final_equal,open_angle |
| 16 | 397 | 2848.0 | 17.046 | 2648 | 200 | semantic_or_handbuilt | local_stencil,onehot_final_equal,scatter |
| 17 | 069 | 2412.0 | 16.728 | 3848 | 64 | unknown | connectivity_wall,documented_wall,onehot_final_equal,open_angle |
| 18 | 051 | 1803.0 | 17.503 | 1702 | 101 | semantic_or_handbuilt | onehot_final_equal,open_angle |
| 19 | 086 | 1636.0 | 16.949 | 2946 | 190 | unknown | assignment_wall,connectivity_wall,documented_wall,local_stencil,onehot_final_equal,open_angle,qlinear |
| 20 | 303 | 1620.0 | 17.610 | 1598 | 22 | semantic_or_handbuilt | onehot_final_equal,open_angle |
| 21 | 141 | 1505.0 | 17.683 | 1404 | 101 | unknown | onehot_final_equal,open_angle |
| 22 | 346 | 1254.0 | 17.866 | 1224 | 30 | unknown | connectivity_wall,local_stencil,lut_selection,onehot_final_equal,open_angle,qlinear,scatter |
| 23 | 329 | 1059.0 | 18.035 | 1029 | 30 | semantic_or_handbuilt | onehot_final_equal,open_angle,scatter |
| 24 | 239 | 947.0 | 18.147 | 897 | 50 | semantic_or_handbuilt | onehot_final_equal |
| 25 | 048 | -354.0 | 17.956 | 1030 | 116 | unknown | bitwise_program,connectivity_wall,documented_wall,onehot_final_equal,open_angle |

## scan_dtype_and_shift_compression

Compress scan-style MaxPool/CumSum/Hillis-Steele pipelines with lower dtype or shared shifts

- source tasks: 046 216
- candidates: 36
- expected: {'gain_type': 'memory', 'risk': 'medium-high', 'verification': 'stored/fresh eval and compare against incumbent'}

| rank | task | score | pts | mem | params | source | tags |
|---:|---:|---:|---:|---:|---:|---|---|
| 1 | 018 | 64925.6 | 13.898 | 63257 | 3038 | exact_preserve | documented_wall,exact_preserve,gather_heavy,heuristic,high_memory,low_score,maxpool_scan,scatter |
| 2 | 209 | 49701.7 | 14.158 | 50951 | 198 | exact_preserve | assignment_wall,connectivity_wall,conv_heavy,documented_wall,exact_preserve,heuristic,high_memory,low_score |
| 3 | 187 | 43124.1 | 14.340 | 41700 | 926 | semantic_or_handbuilt | high_memory,local_stencil,low_score,maxpool_scan |
| 4 | 002 | 38595.9 | 14.400 | 40084 | 32 | exact_preserve | connectivity_wall,documented_wall,exact_preserve,heuristic,high_memory,low_score,maxpool_scan |
| 5 | 349 | 37967.4 | 14.429 | 37800 | 1196 | unknown | documented_wall,high_memory,local_stencil,low_score,lut_selection,open_angle,scan |
| 6 | 366 | 35905.2 | 14.469 | 36955 | 491 | exact_preserve | assignment_wall,connectivity_wall,documented_wall,exact_preserve,gather_heavy,heuristic,high_memory,low_score |
| 7 | 133 | 32008.5 | 14.578 | 32294 | 1288 | exact_preserve | assignment_wall,connectivity_wall,conv_heavy,documented_wall,exact_preserve,gather_heavy,heuristic,high_memory |
| 8 | 364 | 27334.8 | 14.797 | 26860 | 114 | unknown | connectivity_wall,high_memory,local_stencil,low_score,maxpool_scan,open_angle |
| 9 | 054 | 26994.5 | 14.792 | 26894 | 238 | exact_preserve | exact_preserve,gather_heavy,heuristic,high_memory,low_score,scan,scatter |
| 10 | 285 | 25477.5 | 14.848 | 25080 | 552 | exact_preserve | exact_preserve,gather_heavy,heuristic,high_memory,low_score,scatter |
| 11 | 219 | 24569.2 | 14.826 | 26133 | 84 | exact_preserve | assignment_wall,connectivity_wall,documented_wall,exact_preserve,gather_heavy,heuristic,high_memory,low_score |
| 12 | 173 | 23087.5 | 14.945 | 23159 | 112 | exact_preserve | exact_preserve,gather_heavy,heuristic,high_memory,low_score,scatter |
| 13 | 319 | 20544.9 | 14.990 | 21973 | 269 | exact_preserve | assignment_wall,documented_wall,exact_preserve,gather_heavy,heuristic,high_memory,low_score,scan |
| 14 | 066 | 19532.5 | 15.108 | 19113 | 652 | exact_preserve | exact_preserve,heuristic,high_memory,matmul,scan |
| 15 | 101 | 17662.2 | 15.206 | 17015 | 909 | exact_preserve | exact_preserve,gather_heavy,heuristic,high_memory,scatter |
| 16 | 198 | 15821.0 | 15.253 | 16961 | 136 | semantic_or_handbuilt | connectivity_wall,documented_wall,gather_heavy,high_memory,open_angle,scan |
| 17 | 025 | 14436.7 | 15.434 | 14072 | 195 | semantic_or_handbuilt | high_memory,open_angle,scan |
| 18 | 077 | 13556.2 | 15.393 | 14760 | 114 | heuristic | connectivity_wall,documented_wall,heuristic,high_memory,local_stencil,maxpool_scan,open_angle,qlinear |
| 19 | 110 | 12448.5 | 15.468 | 13155 | 634 | semantic_or_handbuilt | connectivity_wall,conv_heavy,documented_wall,high_memory,maxpool_scan,open_angle |
| 20 | 145 | 11861.7 | 15.511 | 13104 | 111 | unknown | connectivity_wall,documented_wall,high_memory,lut_selection,maxpool_scan,open_angle |
| 21 | 157 | 10840.9 | 15.550 | 12118 | 588 | exact_preserve | assignment_wall,connectivity_wall,documented_wall,exact_preserve,gather_heavy,heuristic,high_memory,open_angle |
| 22 | 370 | 10486.2 | 15.749 | 9144 | 1267 | semantic_or_handbuilt | conv_heavy,qlinear,scan,scatter |
| 23 | 204 | 9280.3 | 15.722 | 10244 | 453 | semantic_or_handbuilt | connectivity_wall,conv_heavy,documented_wall,high_memory,local_stencil,maxpool_scan,open_angle,qlinear |
| 24 | 222 | 8839.2 | 15.916 | 8736 | 78 | unknown | conv_heavy,local_stencil,maxpool_scan,onehot_final_equal,open_angle,qlinear |
| 25 | 324 | 7444.7 | 15.904 | 7330 | 1586 | unknown | conv_heavy,documented_wall,local_stencil,qlinear,scan |

## sparse_conv_single_op_floor

Collapse local neighborhood rules to one sparse Conv/QLinearConv when output is thresholded

- source tasks: 015 095 230 294
- candidates: 63
- expected: {'gain_type': 'memory+params', 'risk': 'low when rule is truly local', 'verification': 'fresh process eval; beware one-process mem0 false signals'}

| rank | task | score | pts | mem | params | source | tags |
|---:|---:|---:|---:|---:|---:|---|---|
| 1 | 187 | 43124.1 | 14.340 | 41700 | 926 | semantic_or_handbuilt | high_memory,local_stencil,low_score,maxpool_scan |
| 2 | 349 | 37967.4 | 14.429 | 37800 | 1196 | unknown | documented_wall,high_memory,local_stencil,low_score,lut_selection,open_angle,scan |
| 3 | 023 | 22690.9 | 14.984 | 22275 | 111 | semantic_or_handbuilt | conv_heavy,high_memory,local_stencil,low_score |
| 4 | 396 | 13289.1 | 15.516 | 13007 | 137 | unknown | conv_heavy,high_memory |
| 5 | 138 | 13003.1 | 15.430 | 14106 | 226 | unknown | conv_heavy,documented_wall,high_memory,local_stencil,open_angle |
| 6 | 370 | 10486.2 | 15.749 | 9144 | 1267 | semantic_or_handbuilt | conv_heavy,qlinear,scan,scatter |
| 7 | 080 | 9887.4 | 15.669 | 10834 | 454 | unknown | conv_heavy,documented_wall,high_memory,lut_selection,open_angle,qlinear |
| 8 | 074 | 9315.5 | 15.865 | 7462 | 1813 | semantic_or_handbuilt | local_stencil,open_angle |
| 9 | 222 | 8839.2 | 15.916 | 8736 | 78 | unknown | conv_heavy,local_stencil,maxpool_scan,onehot_final_equal,open_angle,qlinear |
| 10 | 324 | 7444.7 | 15.904 | 7330 | 1586 | unknown | conv_heavy,documented_wall,local_stencil,qlinear,scan |
| 11 | 192 | 7139.6 | 15.938 | 8515 | 106 | unknown | documented_wall,local_stencil,open_angle |
| 12 | 005 | 6600.0 | 16.132 | 6610 | 490 | exact_preserve | conv_heavy,exact_preserve,heuristic,qlinear |
| 13 | 279 | 5555.0 | 16.378 | 5508 | 47 | semantic_or_handbuilt | conv_heavy,local_stencil,qlinear |
| 14 | 085 | 5381.0 | 16.409 | 4500 | 881 | unknown | local_stencil,open_angle |
| 15 | 278 | 4631.0 | 16.279 | 6084 | 47 | semantic_or_handbuilt | documented_wall,local_stencil,open_angle,qlinear |
| 16 | 335 | 4216.0 | 16.653 | 3546 | 670 | semantic_or_handbuilt | conv_heavy,local_stencil,onehot_final_equal,open_angle |
| 17 | 117 | 4165.0 | 16.666 | 3922 | 243 | unknown | local_stencil,lut_selection,open_angle,qlinear |
| 18 | 132 | 4089.0 | 16.684 | 3990 | 99 | unknown | bitwise_program,gather_heavy,local_stencil,open_angle |
| 19 | 162 | 4068.0 | 16.689 | 4024 | 44 | unknown | local_stencil,open_angle,qlinear |
| 20 | 284 | 3735.0 | 16.774 | 3082 | 653 | semantic_or_handbuilt | conv_heavy,open_angle,scatter |
| 21 | 079 | 3144.0 | 16.947 | 3065 | 79 | unknown | local_stencil,open_angle |
| 22 | 042 | 3110.0 | 16.958 | 2792 | 318 | semantic_or_handbuilt | conv_heavy,local_stencil |
| 23 | 019 | 3079.0 | 16.968 | 2997 | 82 | semantic_or_handbuilt | local_stencil |
| 24 | 354 | 3077.0 | 16.968 | 2998 | 79 | unknown | conv_heavy,local_stencil,open_angle |
| 25 | 091 | 3028.0 | 16.984 | 2853 | 175 | semantic_or_handbuilt | local_stencil,open_angle |

## exact_preserve_to_semantic_rewrite

Prioritize exact-preserve source builders for semantic replacement

- source tasks: 002 018 286 366
- candidates: 18
- expected: {'gain_type': 'research', 'risk': 'high', 'verification': 'document semantic mechanism before implementation'}

| rank | task | score | pts | mem | params | source | tags |
|---:|---:|---:|---:|---:|---:|---|---|
| 1 | 018 | 64925.6 | 13.898 | 63257 | 3038 | exact_preserve | documented_wall,exact_preserve,gather_heavy,heuristic,high_memory,low_score,maxpool_scan,scatter |
| 2 | 286 | 50572.8 | 14.141 | 51276 | 739 | exact_preserve | bitwise_program,connectivity_wall,documented_wall,exact_preserve,gather_heavy,heuristic,high_memory,low_score |
| 3 | 209 | 49701.7 | 14.158 | 50951 | 198 | exact_preserve | assignment_wall,connectivity_wall,conv_heavy,documented_wall,exact_preserve,heuristic,high_memory,low_score |
| 4 | 002 | 38595.9 | 14.400 | 40084 | 32 | exact_preserve | connectivity_wall,documented_wall,exact_preserve,heuristic,high_memory,low_score,maxpool_scan |
| 5 | 366 | 35905.2 | 14.469 | 36955 | 491 | exact_preserve | assignment_wall,connectivity_wall,documented_wall,exact_preserve,gather_heavy,heuristic,high_memory,low_score |
| 6 | 118 | 32870.1 | 14.553 | 31049 | 3387 | exact_preserve | connectivity_wall,conv_heavy,documented_wall,exact_preserve,heuristic,high_memory,local_stencil,low_score |
| 7 | 133 | 32008.5 | 14.578 | 32294 | 1288 | exact_preserve | assignment_wall,connectivity_wall,conv_heavy,documented_wall,exact_preserve,gather_heavy,heuristic,high_memory |
| 8 | 054 | 26994.5 | 14.792 | 26894 | 238 | exact_preserve | exact_preserve,gather_heavy,heuristic,high_memory,low_score,scan,scatter |
| 9 | 285 | 25477.5 | 14.848 | 25080 | 552 | exact_preserve | exact_preserve,gather_heavy,heuristic,high_memory,low_score,scatter |
| 10 | 219 | 24569.2 | 14.826 | 26133 | 84 | exact_preserve | assignment_wall,connectivity_wall,documented_wall,exact_preserve,gather_heavy,heuristic,high_memory,low_score |
| 11 | 076 | 24009.9 | 14.847 | 25375 | 289 | exact_preserve | assignment_wall,connectivity_wall,documented_wall,exact_preserve,heuristic,high_memory,low_score,open_angle |
| 12 | 255 | 23399.7 | 14.871 | 24746 | 315 | exact_preserve | connectivity_wall,conv_heavy,documented_wall,exact_preserve,heuristic,high_memory,low_score,matmul |
| 13 | 173 | 23087.5 | 14.945 | 23159 | 112 | exact_preserve | exact_preserve,gather_heavy,heuristic,high_memory,low_score,scatter |
| 14 | 319 | 20544.9 | 14.990 | 21973 | 269 | exact_preserve | assignment_wall,documented_wall,exact_preserve,gather_heavy,heuristic,high_memory,low_score,scan |
| 15 | 066 | 19532.5 | 15.108 | 19113 | 652 | exact_preserve | exact_preserve,heuristic,high_memory,matmul,scan |
| 16 | 101 | 17662.2 | 15.206 | 17015 | 909 | exact_preserve | exact_preserve,gather_heavy,heuristic,high_memory,scatter |
| 17 | 157 | 10840.9 | 15.550 | 12118 | 588 | exact_preserve | assignment_wall,connectivity_wall,documented_wall,exact_preserve,gather_heavy,heuristic,high_memory,open_angle |
| 18 | 044 | 9797.9 | 15.634 | 11620 | 68 | exact_preserve | assignment_wall,connectivity_wall,conv_heavy,documented_wall,exact_preserve,heuristic,high_memory,lut_selection |
