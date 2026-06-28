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
- candidates: 25
- expected: {'gain_type': 'memory', 'risk': 'low-medium', 'verification': 'output equivalence before adoption'}

| rank | task | score | pts | mem | params | source | tags |
|---:|---:|---:|---:|---:|---:|---|---|
| 1 | 209 | 49701.7 | 14.158 | 50951 | 198 | exact_preserve | assignment_wall,connectivity_wall,conv_heavy,documented_wall,exact_preserve,heuristic,high_memory,low_score |
| 2 | 017 | 12870.6 | 15.548 | 10931 | 1804 | semantic_or_handbuilt | custom_win,high_memory,onehot_final_equal,open_angle |
| 3 | 064 | 12087.7 | 15.494 | 13368 | 68 | semantic_or_handbuilt | connectivity_wall,documented_wall,high_memory,onehot_final_equal,open_angle |
| 4 | 222 | 8839.2 | 15.916 | 8736 | 78 | unknown | conv_heavy,custom_win,local_stencil,maxpool_scan,onehot_final_equal,open_angle,qlinear |
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
| 15 | 105 | 2962.0 | 17.006 | 2856 | 106 | unknown | custom_win,matmul,onehot_final_equal,open_angle |
| 16 | 397 | 2848.0 | 17.046 | 2648 | 200 | semantic_or_handbuilt | local_stencil,onehot_final_equal,scatter |
| 17 | 069 | 2412.0 | 16.728 | 3848 | 64 | unknown | connectivity_wall,documented_wall,onehot_final_equal,open_angle |
| 18 | 051 | 1803.0 | 17.503 | 1702 | 101 | semantic_or_handbuilt | onehot_final_equal,open_angle |
| 19 | 086 | 1636.0 | 16.949 | 2946 | 190 | unknown | assignment_wall,connectivity_wall,documented_wall,local_stencil,onehot_final_equal,open_angle,qlinear |
| 20 | 303 | 1620.0 | 17.610 | 1598 | 22 | semantic_or_handbuilt | onehot_final_equal,open_angle |
| 21 | 141 | 1505.0 | 17.683 | 1404 | 101 | unknown | onehot_final_equal,open_angle |
| 22 | 346 | 1254.0 | 17.866 | 1224 | 30 | unknown | connectivity_wall,local_stencil,lut_selection,onehot_final_equal,open_angle,qlinear,scatter |
| 23 | 329 | 1059.0 | 18.035 | 1029 | 30 | semantic_or_handbuilt | custom_win,onehot_final_equal,open_angle,scatter |
| 24 | 239 | 947.0 | 18.147 | 897 | 50 | semantic_or_handbuilt | onehot_final_equal |
| 25 | 048 | -354.0 | 17.956 | 1030 | 116 | unknown | bitwise_program,connectivity_wall,documented_wall,onehot_final_equal,open_angle |

## scan_dtype_and_shift_compression

Compress scan-style MaxPool/CumSum/Hillis-Steele pipelines with lower dtype or shared shifts

- source tasks: 046 216
- candidates: 37
- expected: {'gain_type': 'memory', 'risk': 'medium-high', 'verification': 'stored/fresh eval and compare against incumbent'}

| rank | task | score | pts | mem | params | source | tags |
|---:|---:|---:|---:|---:|---:|---|---|
| 1 | 233 | 70736.4 | 13.835 | 69943 | 644 | exact_preserve | conv_heavy,exact_preserve,gather_heavy,high_memory,low_score,matmul,scan,scatter |
| 2 | 018 | 64925.6 | 13.898 | 63257 | 3038 | exact_preserve | documented_wall,exact_preserve,gather_heavy,heuristic,high_memory,low_score,maxpool_scan,scatter |
| 3 | 209 | 49701.7 | 14.158 | 50951 | 198 | exact_preserve | assignment_wall,connectivity_wall,conv_heavy,documented_wall,exact_preserve,heuristic,high_memory,low_score |
| 4 | 187 | 43124.1 | 14.340 | 41700 | 926 | semantic_or_handbuilt | high_memory,local_stencil,low_score,maxpool_scan |
| 5 | 002 | 38595.9 | 14.400 | 40084 | 32 | exact_preserve | connectivity_wall,documented_wall,exact_preserve,heuristic,high_memory,low_score,maxpool_scan |
| 6 | 349 | 37967.4 | 14.429 | 37800 | 1196 | unknown | documented_wall,high_memory,local_stencil,low_score,lut_selection,open_angle,scan |
| 7 | 366 | 35905.2 | 14.469 | 36955 | 491 | exact_preserve | assignment_wall,connectivity_wall,documented_wall,exact_preserve,gather_heavy,heuristic,high_memory,low_score |
| 8 | 133 | 32008.5 | 14.578 | 32294 | 1288 | exact_preserve | assignment_wall,connectivity_wall,conv_heavy,documented_wall,exact_preserve,gather_heavy,heuristic,high_memory |
| 9 | 364 | 27334.8 | 14.797 | 26860 | 114 | unknown | connectivity_wall,high_memory,local_stencil,low_score,maxpool_scan,open_angle,template_match |
| 10 | 054 | 26994.5 | 14.792 | 26894 | 238 | exact_preserve | exact_preserve,gather_heavy,heuristic,high_memory,low_score,scan,scatter |
| 11 | 285 | 25477.5 | 14.848 | 25080 | 552 | exact_preserve | exact_preserve,gather_heavy,heuristic,high_memory,low_score,scatter |
| 12 | 219 | 24569.2 | 14.826 | 26133 | 84 | exact_preserve | assignment_wall,connectivity_wall,documented_wall,exact_preserve,gather_heavy,heuristic,high_memory,low_score |
| 13 | 173 | 23087.5 | 14.945 | 23159 | 112 | exact_preserve | exact_preserve,gather_heavy,heuristic,high_memory,low_score,scatter |
| 14 | 319 | 20544.9 | 14.990 | 21973 | 269 | exact_preserve | assignment_wall,documented_wall,exact_preserve,gather_heavy,heuristic,high_memory,low_score,scan |
| 15 | 066 | 19532.5 | 15.108 | 19113 | 652 | exact_preserve | exact_preserve,heuristic,high_memory,matmul,scan |
| 16 | 101 | 17662.2 | 15.206 | 17015 | 909 | exact_preserve | exact_preserve,gather_heavy,heuristic,high_memory,scatter |
| 17 | 198 | 15821.0 | 15.253 | 16961 | 136 | semantic_or_handbuilt | connectivity_wall,documented_wall,gather_heavy,high_memory,open_angle,scan |
| 18 | 025 | 14436.7 | 15.434 | 14072 | 195 | semantic_or_handbuilt | high_memory,open_angle,scan |
| 19 | 077 | 13556.2 | 15.393 | 14760 | 114 | heuristic | connectivity_wall,documented_wall,heuristic,high_memory,local_stencil,maxpool_scan,open_angle,qlinear |
| 20 | 110 | 12448.5 | 15.468 | 13155 | 634 | semantic_or_handbuilt | connectivity_wall,conv_heavy,documented_wall,high_memory,maxpool_scan,open_angle |
| 21 | 145 | 11861.7 | 15.511 | 13104 | 111 | unknown | connectivity_wall,custom_win,documented_wall,high_memory,lut_selection,maxpool_scan,open_angle |
| 22 | 157 | 10840.9 | 15.550 | 12118 | 588 | exact_preserve | assignment_wall,connectivity_wall,documented_wall,exact_preserve,gather_heavy,heuristic,high_memory,lut_selection |
| 23 | 370 | 10486.2 | 15.749 | 9144 | 1267 | semantic_or_handbuilt | conv_heavy,qlinear,scan,scatter |
| 24 | 204 | 9280.3 | 15.722 | 10244 | 453 | semantic_or_handbuilt | connectivity_wall,conv_heavy,custom_win,documented_wall,high_memory,local_stencil,maxpool_scan,open_angle |
| 25 | 222 | 8839.2 | 15.916 | 8736 | 78 | unknown | conv_heavy,custom_win,local_stencil,maxpool_scan,onehot_final_equal,open_angle,qlinear |

## sparse_conv_single_op_floor

Collapse local neighborhood rules to one sparse Conv/QLinearConv when output is thresholded

- source tasks: 015 095 230 294
- candidates: 64
- expected: {'gain_type': 'memory+params', 'risk': 'low when rule is truly local', 'verification': 'fresh process eval; beware one-process mem0 false signals'}

| rank | task | score | pts | mem | params | source | tags |
|---:|---:|---:|---:|---:|---:|---|---|
| 1 | 233 | 70736.4 | 13.835 | 69943 | 644 | exact_preserve | conv_heavy,exact_preserve,gather_heavy,high_memory,low_score,matmul,scan,scatter |
| 2 | 187 | 43124.1 | 14.340 | 41700 | 926 | semantic_or_handbuilt | high_memory,local_stencil,low_score,maxpool_scan |
| 3 | 349 | 37967.4 | 14.429 | 37800 | 1196 | unknown | documented_wall,high_memory,local_stencil,low_score,lut_selection,open_angle,scan |
| 4 | 023 | 22690.9 | 14.984 | 22275 | 111 | semantic_or_handbuilt | conv_heavy,high_memory,local_stencil,low_score |
| 5 | 396 | 13289.1 | 15.516 | 13007 | 137 | unknown | conv_heavy,high_memory |
| 6 | 138 | 13003.1 | 15.430 | 14106 | 226 | unknown | conv_heavy,documented_wall,high_memory,local_stencil,open_angle |
| 7 | 370 | 10486.2 | 15.749 | 9144 | 1267 | semantic_or_handbuilt | conv_heavy,qlinear,scan,scatter |
| 8 | 080 | 9887.4 | 15.669 | 10834 | 454 | unknown | conv_heavy,documented_wall,high_memory,lut_selection,open_angle,qlinear |
| 9 | 074 | 9315.5 | 15.865 | 7462 | 1813 | semantic_or_handbuilt | local_stencil,open_angle,template_match |
| 10 | 222 | 8839.2 | 15.916 | 8736 | 78 | unknown | conv_heavy,custom_win,local_stencil,maxpool_scan,onehot_final_equal,open_angle,qlinear |
| 11 | 324 | 7444.7 | 15.904 | 7330 | 1586 | unknown | conv_heavy,documented_wall,local_stencil,qlinear,scan |
| 12 | 192 | 7139.6 | 15.938 | 8515 | 106 | unknown | documented_wall,local_stencil,open_angle |
| 13 | 005 | 6600.0 | 16.132 | 6610 | 490 | exact_preserve | conv_heavy,exact_preserve,heuristic,qlinear |
| 14 | 279 | 5555.0 | 16.378 | 5508 | 47 | semantic_or_handbuilt | conv_heavy,local_stencil,qlinear |
| 15 | 085 | 5381.0 | 16.409 | 4500 | 881 | unknown | local_stencil,open_angle |
| 16 | 278 | 4631.0 | 16.279 | 6084 | 47 | semantic_or_handbuilt | documented_wall,local_stencil,open_angle,qlinear |
| 17 | 335 | 4216.0 | 16.653 | 3546 | 670 | semantic_or_handbuilt | conv_heavy,local_stencil,onehot_final_equal,open_angle |
| 18 | 117 | 4165.0 | 16.666 | 3922 | 243 | unknown | local_stencil,lut_selection,open_angle,qlinear |
| 19 | 132 | 4089.0 | 16.684 | 3990 | 99 | unknown | bitwise_program,gather_heavy,local_stencil,open_angle |
| 20 | 162 | 4068.0 | 16.689 | 4024 | 44 | unknown | local_stencil,open_angle,qlinear |
| 21 | 284 | 3735.0 | 16.774 | 3082 | 653 | semantic_or_handbuilt | conv_heavy,custom_win,open_angle,scatter |
| 22 | 079 | 3144.0 | 16.947 | 3065 | 79 | unknown | custom_win,local_stencil,open_angle |
| 23 | 042 | 3110.0 | 16.958 | 2792 | 318 | semantic_or_handbuilt | conv_heavy,local_stencil |
| 24 | 019 | 3079.0 | 16.968 | 2997 | 82 | semantic_or_handbuilt | local_stencil |
| 25 | 354 | 3077.0 | 16.968 | 2998 | 79 | unknown | conv_heavy,local_stencil,open_angle |

## exact_preserve_to_semantic_rewrite

Prioritize exact-preserve source builders for semantic replacement

- source tasks: 002 018 286 366
- candidates: 19
- expected: {'gain_type': 'research', 'risk': 'high', 'verification': 'document semantic mechanism before implementation'}

| rank | task | score | pts | mem | params | source | tags |
|---:|---:|---:|---:|---:|---:|---|---|
| 1 | 233 | 70736.4 | 13.835 | 69943 | 644 | exact_preserve | conv_heavy,exact_preserve,gather_heavy,high_memory,low_score,matmul,scan,scatter |
| 2 | 018 | 64925.6 | 13.898 | 63257 | 3038 | exact_preserve | documented_wall,exact_preserve,gather_heavy,heuristic,high_memory,low_score,maxpool_scan,scatter |
| 3 | 286 | 50572.8 | 14.141 | 51276 | 739 | exact_preserve | bitwise_program,connectivity_wall,documented_wall,exact_preserve,gather_heavy,heuristic,high_memory,low_score |
| 4 | 209 | 49701.7 | 14.158 | 50951 | 198 | exact_preserve | assignment_wall,connectivity_wall,conv_heavy,documented_wall,exact_preserve,heuristic,high_memory,low_score |
| 5 | 002 | 38595.9 | 14.400 | 40084 | 32 | exact_preserve | connectivity_wall,documented_wall,exact_preserve,heuristic,high_memory,low_score,maxpool_scan |
| 6 | 366 | 35905.2 | 14.469 | 36955 | 491 | exact_preserve | assignment_wall,connectivity_wall,documented_wall,exact_preserve,gather_heavy,heuristic,high_memory,low_score |
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
| 17 | 157 | 10840.9 | 15.550 | 12118 | 588 | exact_preserve | assignment_wall,connectivity_wall,documented_wall,exact_preserve,gather_heavy,heuristic,high_memory,lut_selection |
| 18 | 044 | 9797.9 | 15.634 | 11620 | 68 | exact_preserve | assignment_wall,connectivity_wall,conv_heavy,documented_wall,exact_preserve,heuristic,high_memory,matmul |
| 19 | 118 | 2870.1 | 14.553 | 31049 | 3387 | exact_preserve | connectivity_wall,conv_heavy,documented_wall,exact_preserve,heuristic,high_memory,infeasible_exact_wall,information_loss_wall |

## dihedral_template_match_stacked_conv

Use one stacked Conv for all dihedral orientations of a small runtime-extracted template

- source tasks: 191
- candidates: 7
- expected: {'gain_type': 'memory+accuracy', 'risk': 'medium', 'verification': 'fresh exact eval; compare against incumbent and propagate if template extraction is deterministic'}

| rank | task | score | pts | mem | params | source | tags |
|---:|---:|---:|---:|---:|---:|---|---|
| 1 | 191 | 32530.1 | 14.623 | 31276 | 841 | unknown | assignment_wall,conv_heavy,custom_win,high_memory,low_score,matmul,open_angle,template_match |
| 2 | 364 | 27334.8 | 14.797 | 26860 | 114 | unknown | connectivity_wall,high_memory,local_stencil,low_score,maxpool_scan,open_angle,template_match |
| 3 | 074 | 9315.5 | 15.865 | 7462 | 1813 | semantic_or_handbuilt | local_stencil,open_angle,template_match |
| 4 | 363 | 5126.0 | 16.458 | 4843 | 283 | semantic_or_handbuilt | connectivity_wall,local_stencil,open_angle,template_match |
| 5 | 287 | 1994.0 | 17.402 | 1699 | 295 | semantic_or_handbuilt | open_angle,template_match |
| 6 | 400 | 1866.0 | 17.468 | 1800 | 66 | semantic_or_handbuilt | assignment_wall,connectivity_wall,open_angle,template_match |
| 7 | 158 | -5661.7 | 14.526 | 33053 | 2343 | unknown | assignment_wall,conv_heavy,documented_wall,gather_heavy,high_memory,low_score,marginal_wall,open_angle |
