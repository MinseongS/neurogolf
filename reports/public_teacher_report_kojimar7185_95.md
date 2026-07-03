# Public teacher report

Public ONNX files are teacher artifacts.  Do not adopt blindly; extract source and mechanisms first.

- candidates scanned: 410

| task | public pts | live pts | src pts | Δpts | Δmem | Δparams | flags | recommendation | path |
|---:|---:|---:|---:|---:|---:|---:|---|---|---|
| 001 | 18.902 | 19.519 | 19.519 | -0.617 | 228 | -23 | mechanism_teacher,params_down | extract_mechanism_before_any_adopt | `public_candidates/kojimar7185_95/base_submission/task001.onnx` |
| 002 | 15.044 | 16.192 | 16.192 | -1.148 | 14780 | -381 | mechanism_teacher,params_down | extract_mechanism_before_any_adopt | `public_candidates/kojimar7185_95/base_submission/task002.onnx` |
| 003 | 19.290 | 19.290 | 19.290 | +0.000 | 0 | 0 |  | archive_low_priority | `public_candidates/kojimar7185_95/base_submission/task003.onnx` |
| 004 | 16.456 | 16.454 | 16.454 | +0.001 | 0 | -7 | params_down,score_up | extract_source_then_consider_adopt | `public_candidates/kojimar7185_95/base_submission/task004.onnx` |
| 005 | 16.163 | 16.163 | 16.163 | +0.000 | 0 | 0 |  | archive_low_priority | `public_candidates/kojimar7185_95/base_submission/task005.onnx` |
| 006 | 19.970 | 19.970 | 19.970 | +0.000 | 0 | 0 |  | archive_low_priority | `public_candidates/kojimar7185_95/base_submission/task006.onnx` |
| 007 | 20.156 | 20.156 | 20.156 | +0.000 | 0 | 0 |  | archive_low_priority | `public_candidates/kojimar7185_95/base_submission/task007.onnx` |
| 008 | 16.365 | 16.365 | 16.365 | +0.000 | 0 | 0 |  | archive_low_priority | `public_candidates/kojimar7185_95/base_submission/task008.onnx` |
| 009 | 15.963 | 16.143 | 16.143 | -0.180 | 400 | 986 |  | archive_low_priority | `public_candidates/kojimar7185_95/base_submission/task009.onnx` |
| 010 | 18.025 | 18.025 | 18.025 | +0.000 | 0 | 0 |  | archive_low_priority | `public_candidates/kojimar7185_95/base_submission/task010.onnx` |
| 011 | 17.344 | 17.400 | 17.400 | -0.056 | 155 | -40 | mechanism_teacher,params_down | extract_mechanism_before_any_adopt | `public_candidates/kojimar7185_95/base_submission/task011.onnx` |
| 012 | 17.277 | 17.275 | 17.275 | +0.002 | -5 | 0 | memory_down,score_up | extract_source_then_consider_adopt | `public_candidates/kojimar7185_95/base_submission/task012.onnx` |
| 013 | 17.435 | 17.435 | 17.435 | +0.000 | 0 | 0 |  | archive_low_priority | `public_candidates/kojimar7185_95/base_submission/task013.onnx` |
| 014 | 16.664 | 16.664 | 16.664 | +0.000 | 0 | 0 |  | archive_low_priority | `public_candidates/kojimar7185_95/base_submission/task014.onnx` |
| 015 | 18.198 | 18.198 | 18.198 | +0.000 | 0 | 0 |  | archive_low_priority | `public_candidates/kojimar7185_95/base_submission/task015.onnx` |
| 016 | 22.697 | 22.697 | 22.697 | +0.000 | 0 | 0 |  | archive_low_priority | `public_candidates/kojimar7185_95/base_submission/task016.onnx` |
| 017 | 16.038 | 15.744 | 15.744 | +0.294 | -2451 | -217 | memory_down,params_down,score_up | extract_source_then_consider_adopt | `public_candidates/kojimar7185_95/base_submission/task017.onnx` |
| 018 | 14.448 | 14.659 | 14.659 | -0.210 | 7422 | -162 | mechanism_teacher,params_down | extract_mechanism_before_any_adopt | `public_candidates/kojimar7185_95/base_submission/task018.onnx` |
| 019 | 17.063 | 17.063 | 17.063 | +0.000 | 0 | 0 |  | archive_low_priority | `public_candidates/kojimar7185_95/base_submission/task019.onnx` |
| 020 | 17.670 | 17.670 | 17.670 | +0.000 | 0 | 0 |  | archive_low_priority | `public_candidates/kojimar7185_95/base_submission/task020.onnx` |
| 021 | 18.514 | 18.400 | 18.400 | +0.114 | -78 | -1 | memory_down,params_down,score_up | extract_source_then_consider_adopt | `public_candidates/kojimar7185_95/base_submission/task021.onnx` |
| 021 | 19.219 | 18.400 | 18.400 | +0.819 | -411 | 0 | memory_down,score_up | extract_source_then_consider_adopt | `public_candidates/kojimar7185_95/overrides/task021.onnx` |
| 022 | 17.258 | 17.258 | 17.258 | +0.000 | 0 | 0 |  | archive_low_priority | `public_candidates/kojimar7185_95/base_submission/task022.onnx` |
| 023 | 15.597 | 16.234 | 16.234 | -0.637 | 5672 | 42 | new_qlinear_mechanism | extract_mechanism_before_any_adopt | `public_candidates/kojimar7185_95/base_submission/task023.onnx` |
| 024 | 20.300 | 20.300 | 20.300 | +0.000 | 0 | 0 |  | archive_low_priority | `public_candidates/kojimar7185_95/base_submission/task024.onnx` |
| 025 | 15.636 | 15.448 | 15.448 | +0.187 | -2354 | -51 | memory_down,params_down,score_up | extract_source_then_consider_adopt | `public_candidates/kojimar7185_95/base_submission/task025.onnx` |
| 026 | 19.702 | 19.702 | 19.702 | +0.000 | 0 | 0 |  | archive_low_priority | `public_candidates/kojimar7185_95/base_submission/task026.onnx` |
| 027 | 17.822 | 17.822 | 17.822 | +0.000 | 0 | 0 |  | archive_low_priority | `public_candidates/kojimar7185_95/base_submission/task027.onnx` |
| 028 | 19.876 | 19.876 | 19.876 | +0.000 | 0 | 0 |  | archive_low_priority | `public_candidates/kojimar7185_95/base_submission/task028.onnx` |
| 029 | 16.383 | 16.383 | 16.383 | +0.000 | 0 | 0 |  | archive_low_priority | `public_candidates/kojimar7185_95/base_submission/task029.onnx` |
| 030 | 17.218 | 17.218 | 17.218 | +0.000 | 0 | 0 |  | archive_low_priority | `public_candidates/kojimar7185_95/base_submission/task030.onnx` |
| 031 | 18.466 | 18.466 | 18.466 | +0.000 | 0 | 0 |  | archive_low_priority | `public_candidates/kojimar7185_95/base_submission/task031.onnx` |
| 032 | 18.187 | 18.187 | 18.187 | +0.000 | 0 | 0 |  | archive_low_priority | `public_candidates/kojimar7185_95/base_submission/task032.onnx` |
| 033 | 17.598 | 17.598 | 17.598 | +0.000 | 0 | 0 |  | archive_low_priority | `public_candidates/kojimar7185_95/base_submission/task033.onnx` |
| 034 | 17.231 | 17.231 | 17.231 | +0.000 | 0 | 0 |  | archive_low_priority | `public_candidates/kojimar7185_95/base_submission/task034.onnx` |
| 035 | 17.276 | 17.276 | 17.276 | +0.000 | 0 | 0 |  | archive_low_priority | `public_candidates/kojimar7185_95/base_submission/task035.onnx` |
| 036 | 17.314 | 17.314 | 17.314 | +0.000 | 0 | 0 |  | archive_low_priority | `public_candidates/kojimar7185_95/base_submission/task036.onnx` |
| 037 | 16.535 | 16.815 | 16.815 | -0.281 | 1230 | -69 | mechanism_teacher,params_down | extract_mechanism_before_any_adopt | `public_candidates/kojimar7185_95/base_submission/task037.onnx` |
| 038 | 19.639 | 19.639 | 19.639 | +0.000 | 0 | 0 |  | archive_low_priority | `public_candidates/kojimar7185_95/base_submission/task038.onnx` |
| 039 | 18.826 | 18.826 | 18.826 | +0.000 | 0 | 0 |  | archive_low_priority | `public_candidates/kojimar7185_95/base_submission/task039.onnx` |
| 040 | 19.076 | 19.076 | 19.076 | +0.000 | 0 | 0 |  | archive_low_priority | `public_candidates/kojimar7185_95/base_submission/task040.onnx` |
| 041 | 17.511 | 17.511 | 17.511 | +0.000 | 0 | 0 |  | archive_low_priority | `public_candidates/kojimar7185_95/base_submission/task041.onnx` |
| 042 | 16.958 | 16.958 | 16.958 | +0.000 | 0 | 0 |  | archive_low_priority | `public_candidates/kojimar7185_95/base_submission/task042.onnx` |
| 043 | 18.323 | 18.323 | 18.323 | +0.000 | 0 | 0 |  | archive_low_priority | `public_candidates/kojimar7185_95/base_submission/task043.onnx` |
| 044 | 15.684 | 15.684 | 15.684 | +0.000 | 0 | 0 |  | archive_low_priority | `public_candidates/kojimar7185_95/base_submission/task044.onnx` |
| 045 | 17.578 | 17.578 | 17.578 | +0.000 | 0 | 0 |  | archive_low_priority | `public_candidates/kojimar7185_95/base_submission/task045.onnx` |
| 046 | 17.252 | 17.252 | 17.252 | +0.000 | 0 | 0 |  | archive_low_priority | `public_candidates/kojimar7185_95/base_submission/task046.onnx` |
| 047 | 18.241 | 18.241 | 18.241 | +0.000 | 0 | 0 |  | archive_low_priority | `public_candidates/kojimar7185_95/base_submission/task047.onnx` |
| 048 | 17.956 | 17.956 | 17.956 | +0.000 | 0 | 0 |  | archive_low_priority | `public_candidates/kojimar7185_95/base_submission/task048.onnx` |
| 049 | 19.044 | 19.044 | 19.044 | +0.000 | 0 | 0 |  | archive_low_priority | `public_candidates/kojimar7185_95/base_submission/task049.onnx` |
| 050 | 16.728 | 16.728 | 16.728 | +0.000 | 0 | 0 |  | archive_low_priority | `public_candidates/kojimar7185_95/base_submission/task050.onnx` |
| 051 | 17.503 | 17.503 | 17.503 | +0.000 | 0 | 0 |  | archive_low_priority | `public_candidates/kojimar7185_95/base_submission/task051.onnx` |
| 052 | 19.575 | 19.575 | 19.575 | +0.000 | 0 | 0 |  | archive_low_priority | `public_candidates/kojimar7185_95/base_submission/task052.onnx` |
| 053 | 21.599 | 21.599 | 21.599 | +0.000 | 0 | 0 |  | archive_low_priority | `public_candidates/kojimar7185_95/base_submission/task053.onnx` |
| 054 | 14.829 | 15.078 | 15.078 | -0.249 | 5808 | -50 | mechanism_teacher,params_down | extract_mechanism_before_any_adopt | `public_candidates/kojimar7185_95/base_submission/task054.onnx` |
| 055 | 15.757 | 16.963 | 16.963 | -1.206 | 7260 | -25 | mechanism_teacher,params_down | extract_mechanism_before_any_adopt | `public_candidates/kojimar7185_95/base_submission/task055.onnx` |
| 056 | 21.474 | 21.474 | 21.474 | +0.000 | 0 | 0 |  | archive_low_priority | `public_candidates/kojimar7185_95/base_submission/task056.onnx` |
| 057 | 18.496 | 18.496 | 18.496 | +0.000 | 0 | 0 |  | archive_low_priority | `public_candidates/kojimar7185_95/base_submission/task057.onnx` |
| 058 | 18.060 | 18.060 | 18.060 | +0.000 | 0 | 0 |  | archive_low_priority | `public_candidates/kojimar7185_95/base_submission/task058.onnx` |
| 059 | 18.677 | 18.677 | 18.677 | +0.000 | 0 | 0 |  | archive_low_priority | `public_candidates/kojimar7185_95/base_submission/task059.onnx` |
| 060 | 18.082 | 18.082 | 18.082 | +0.000 | 0 | 0 |  | archive_low_priority | `public_candidates/kojimar7185_95/base_submission/task060.onnx` |
| 061 | 17.581 | 17.581 | 17.581 | +0.000 | 0 | 0 |  | archive_low_priority | `public_candidates/kojimar7185_95/base_submission/task061.onnx` |
| 062 | 17.162 | 17.162 | 17.162 | +0.000 | 0 | 0 |  | archive_low_priority | `public_candidates/kojimar7185_95/base_submission/task062.onnx` |
| 063 | 17.394 | 17.394 | 17.394 | +0.000 | 0 | 0 |  | archive_low_priority | `public_candidates/kojimar7185_95/base_submission/task063.onnx` |
| 064 | 15.494 | 15.791 | 15.791 | -0.297 | 3516 | -66 | mechanism_teacher,params_down | extract_mechanism_before_any_adopt | `public_candidates/kojimar7185_95/base_submission/task064.onnx` |
| 065 | 18.537 | 18.537 | 18.537 | +0.000 | 0 | 0 |  | archive_low_priority | `public_candidates/kojimar7185_95/base_submission/task065.onnx` |
| 066 | 15.181 | 15.778 | 15.778 | -0.597 | 7790 | 468 |  | archive_low_priority | `public_candidates/kojimar7185_95/base_submission/task066.onnx` |
| 067 | 21.150 | 21.150 | 21.150 | +0.000 | 0 | 0 |  | archive_low_priority | `public_candidates/kojimar7185_95/base_submission/task067.onnx` |
| 068 | 18.565 | 18.565 | 18.565 | +0.000 | 0 | 0 |  | archive_low_priority | `public_candidates/kojimar7185_95/base_submission/task068.onnx` |
| 069 | 16.728 | 16.728 | 16.728 | +0.000 | 0 | 0 |  | archive_low_priority | `public_candidates/kojimar7185_95/base_submission/task069.onnx` |
| 070 | 16.991 | 16.991 | 16.991 | +0.000 | 0 | 0 |  | archive_low_priority | `public_candidates/kojimar7185_95/base_submission/task070.onnx` |
| 071 | 17.100 | 17.100 | 17.100 | +0.000 | 0 | 0 |  | archive_low_priority | `public_candidates/kojimar7185_95/base_submission/task071.onnx` |
| 072 | 18.957 | 18.957 | 18.957 | +0.000 | 0 | 0 |  | archive_low_priority | `public_candidates/kojimar7185_95/base_submission/task072.onnx` |
| 073 | 21.311 | 21.311 | 21.311 | +0.000 | 0 | 0 |  | archive_low_priority | `public_candidates/kojimar7185_95/base_submission/task073.onnx` |
| 074 | 15.889 | 15.889 | 15.889 | +0.000 | 0 | 0 |  | archive_low_priority | `public_candidates/kojimar7185_95/base_submission/task074.onnx` |
| 075 | 17.695 | 17.695 | 17.695 | +0.000 | 0 | 0 |  | archive_low_priority | `public_candidates/kojimar7185_95/base_submission/task075.onnx` |
| 076 | 15.107 | 15.107 | 15.107 | +0.000 | 0 | 0 |  | archive_low_priority | `public_candidates/kojimar7185_95/base_submission/task076.onnx` |
| 077 | 15.393 | 16.247 | 16.247 | -0.854 | 10260 | -1717 | mechanism_teacher,new_qlinear_mechanism,params_down | extract_mechanism_before_any_adopt | `public_candidates/kojimar7185_95/base_submission/task077.onnx` |
| 078 | 17.835 | 17.835 | 17.835 | +0.000 | 0 | 0 |  | archive_low_priority | `public_candidates/kojimar7185_95/base_submission/task078.onnx` |
| 079 | 16.947 | 16.947 | 16.947 | +0.000 | 0 | 0 |  | archive_low_priority | `public_candidates/kojimar7185_95/base_submission/task079.onnx` |
| 080 | 15.251 | 15.735 | 15.735 | -0.484 | 6915 | -342 | mechanism_teacher,params_down | extract_mechanism_before_any_adopt | `public_candidates/kojimar7185_95/base_submission/task080.onnx` |
| 081 | 18.860 | 18.860 | 18.860 | +0.000 | 0 | 0 |  | archive_low_priority | `public_candidates/kojimar7185_95/base_submission/task081.onnx` |
| 082 | 19.753 | 19.753 | 19.753 | +0.000 | 0 | 0 |  | archive_low_priority | `public_candidates/kojimar7185_95/base_submission/task082.onnx` |
| 083 | 18.897 | 18.960 | 18.960 | -0.062 | 376 | -349 | mechanism_teacher,params_down | extract_mechanism_before_any_adopt | `public_candidates/kojimar7185_95/base_submission/task083.onnx` |
| 084 | 17.380 | 17.453 | 17.453 | -0.073 | -20 | 163 | mechanism_teacher,memory_down | extract_mechanism_before_any_adopt | `public_candidates/kojimar7185_95/base_submission/task084.onnx` |
| 085 | 16.409 | 16.409 | 16.409 | +0.000 | 0 | 0 |  | archive_low_priority | `public_candidates/kojimar7185_95/base_submission/task085.onnx` |
| 086 | 16.949 | 16.949 | 16.949 | +0.000 | 0 | 0 |  | archive_low_priority | `public_candidates/kojimar7185_95/base_submission/task086.onnx` |
| 087 | 23.391 | 23.391 | 23.391 | +0.000 | 0 | 0 |  | archive_low_priority | `public_candidates/kojimar7185_95/base_submission/task087.onnx` |
| 088 | 17.465 | 17.465 | 17.465 | +0.000 | 0 | 0 |  | archive_low_priority | `public_candidates/kojimar7185_95/base_submission/task088.onnx` |
| 089 | 16.111 | 16.135 | 16.135 | -0.024 | 169 | 0 |  | archive_low_priority | `public_candidates/kojimar7185_95/base_submission/task089.onnx` |
| 090 | 16.970 | 16.964 | 16.964 | +0.006 | -19 | 0 | memory_down,score_up | extract_source_then_consider_adopt | `public_candidates/kojimar7185_95/base_submission/task090.onnx` |
| 090 | 16.964 | 16.964 | 16.964 | +0.000 | -1 | 0 | memory_down,score_up | extract_source_then_consider_adopt | `public_candidates/kojimar7185_95/overrides/task090.onnx` |
| 091 | 16.984 | 16.984 | 16.984 | +0.000 | 0 | 0 |  | archive_low_priority | `public_candidates/kojimar7185_95/base_submission/task091.onnx` |
| 092 | 16.165 | 16.165 | 16.165 | +0.000 | 0 | 0 |  | archive_low_priority | `public_candidates/kojimar7185_95/base_submission/task092.onnx` |
| 093 | 16.823 | 16.823 | 16.823 | +0.000 | 0 | 0 |  | archive_low_priority | `public_candidates/kojimar7185_95/base_submission/task093.onnx` |
| 094 | 17.096 | 17.094 | 17.094 | +0.001 | 0 | -4 | params_down,score_up | extract_source_then_consider_adopt | `public_candidates/kojimar7185_95/base_submission/task094.onnx` |
| 095 | 19.156 | 19.156 | 19.156 | +0.000 | 0 | 0 |  | archive_low_priority | `public_candidates/kojimar7185_95/base_submission/task095.onnx` |
| 096 | 15.908 | 15.919 | 15.919 | -0.011 | 96 | 3 |  | archive_low_priority | `public_candidates/kojimar7185_95/base_submission/task096.onnx` |
| 097 | 18.187 | 18.187 | 18.187 | +0.000 | 0 | 0 |  | archive_low_priority | `public_candidates/kojimar7185_95/base_submission/task097.onnx` |
| 098 | 18.198 | 18.198 | 18.198 | +0.000 | 0 | 0 |  | archive_low_priority | `public_candidates/kojimar7185_95/base_submission/task098.onnx` |
| 099 | 17.445 | 17.445 | 17.445 | +0.000 | 0 | 0 |  | archive_low_priority | `public_candidates/kojimar7185_95/base_submission/task099.onnx` |
| 100 | 18.703 | 18.703 | 18.703 | +0.000 | 0 | 0 |  | archive_low_priority | `public_candidates/kojimar7185_95/base_submission/task100.onnx` |
| 101 | 15.226 | 15.422 | 15.422 | -0.196 | 3100 | 31 |  | archive_low_priority | `public_candidates/kojimar7185_95/base_submission/task101.onnx` |
| 102 | 17.165 | 17.165 | 17.165 | +0.000 | 0 | 0 |  | archive_low_priority | `public_candidates/kojimar7185_95/base_submission/task102.onnx` |
| 103 | 20.906 | 20.906 | 20.906 | +0.000 | 0 | 0 |  | archive_low_priority | `public_candidates/kojimar7185_95/base_submission/task103.onnx` |
| 104 | 19.463 | 19.463 | 19.463 | +0.000 | 0 | 0 |  | archive_low_priority | `public_candidates/kojimar7185_95/base_submission/task104.onnx` |
| 105 | 17.007 | 17.007 | 17.007 | +0.000 | 0 | 0 |  | archive_low_priority | `public_candidates/kojimar7185_95/base_submission/task105.onnx` |
| 106 | 18.233 | 18.233 | 18.233 | +0.000 | 0 | 0 |  | archive_low_priority | `public_candidates/kojimar7185_95/base_submission/task106.onnx` |
| 106 | 18.561 | 18.233 | 18.233 | +0.328 | -324 | 81 | memory_down,score_up | extract_source_then_consider_adopt | `public_candidates/kojimar7185_95/overrides/task106.onnx` |
| 107 | 16.687 | 16.687 | 16.687 | +0.000 | 0 | 0 |  | archive_low_priority | `public_candidates/kojimar7185_95/base_submission/task107.onnx` |
| 108 | 19.296 | 19.296 | 19.296 | +0.000 | 0 | 0 |  | archive_low_priority | `public_candidates/kojimar7185_95/base_submission/task108.onnx` |
| 109 | 17.516 | 17.516 | 17.516 | +0.000 | 0 | 0 |  | archive_low_priority | `public_candidates/kojimar7185_95/base_submission/task109.onnx` |
| 110 | 15.469 | 16.332 | 16.332 | -0.864 | 12891 | -4915 | mechanism_teacher,params_down | extract_mechanism_before_any_adopt | `public_candidates/kojimar7185_95/base_submission/task110.onnx` |
| 111 | 18.718 | 18.718 | 18.718 | +0.000 | 0 | 0 |  | archive_low_priority | `public_candidates/kojimar7185_95/base_submission/task111.onnx` |
| 112 | 17.519 | 17.519 | 17.519 | +0.000 | 0 | 0 |  | archive_low_priority | `public_candidates/kojimar7185_95/base_submission/task112.onnx` |
| 113 | 21.599 | 21.599 | 21.599 | +0.000 | 0 | 0 |  | archive_low_priority | `public_candidates/kojimar7185_95/base_submission/task113.onnx` |
| 114 | 18.198 | 18.198 | 18.198 | +0.000 | 0 | 0 |  | archive_low_priority | `public_candidates/kojimar7185_95/base_submission/task114.onnx` |
| 115 | 18.050 | 18.050 | 18.050 | +0.000 | 0 | 0 |  | archive_low_priority | `public_candidates/kojimar7185_95/base_submission/task115.onnx` |
| 116 | 21.599 | 21.599 | 21.599 | +0.000 | 0 | 0 |  | archive_low_priority | `public_candidates/kojimar7185_95/base_submission/task116.onnx` |
| 117 | 16.666 | 16.666 | 16.666 | +0.000 | 0 | 0 |  | archive_low_priority | `public_candidates/kojimar7185_95/base_submission/task117.onnx` |
| 118 | 14.509 | 14.856 | 14.856 | -0.347 | 10159 | 382 |  | archive_low_priority | `public_candidates/kojimar7185_95/base_submission/task118.onnx` |
| 119 | 17.783 | 17.781 | 17.781 | +0.001 | -1 | -1 | memory_down,params_down,score_up | extract_source_then_consider_adopt | `public_candidates/kojimar7185_95/base_submission/task119.onnx` |
| 120 | 18.187 | 18.187 | 18.187 | +0.000 | 0 | 0 |  | archive_low_priority | `public_candidates/kojimar7185_95/base_submission/task120.onnx` |
| 121 | 19.168 | 19.168 | 19.168 | +0.000 | 0 | 0 |  | archive_low_priority | `public_candidates/kojimar7185_95/base_submission/task121.onnx` |
| 122 | 18.303 | 18.303 | 18.303 | +0.000 | 0 | 0 |  | archive_low_priority | `public_candidates/kojimar7185_95/base_submission/task122.onnx` |
| 123 | 17.797 | 17.797 | 17.797 | +0.000 | 0 | 0 |  | archive_low_priority | `public_candidates/kojimar7185_95/base_submission/task123.onnx` |
| 124 | 17.423 | 17.423 | 17.423 | +0.000 | 0 | 0 |  | archive_low_priority | `public_candidates/kojimar7185_95/base_submission/task124.onnx` |
| 125 | 16.850 | 16.850 | 16.850 | +0.000 | 0 | 0 |  | archive_low_priority | `public_candidates/kojimar7185_95/base_submission/task125.onnx` |
| 126 | 18.595 | 18.595 | 18.595 | +0.000 | 0 | 0 |  | archive_low_priority | `public_candidates/kojimar7185_95/base_submission/task126.onnx` |
| 127 | 19.019 | 19.019 | 19.019 | +0.000 | 0 | 0 |  | archive_low_priority | `public_candidates/kojimar7185_95/base_submission/task127.onnx` |
| 128 | 18.826 | 19.296 | 19.296 | -0.470 | 0 | 180 |  | archive_low_priority | `public_candidates/kojimar7185_95/base_submission/task128.onnx` |
| 129 | 20.618 | 20.618 | 20.618 | +0.000 | 0 | 0 |  | archive_low_priority | `public_candidates/kojimar7185_95/base_submission/task129.onnx` |
| 130 | 21.599 | 21.599 | 21.599 | +0.000 | 0 | 0 |  | archive_low_priority | `public_candidates/kojimar7185_95/base_submission/task130.onnx` |
| 131 | 16.709 | 16.709 | 16.709 | +0.000 | 0 | 0 |  | archive_low_priority | `public_candidates/kojimar7185_95/base_submission/task131.onnx` |
| 132 | 16.684 | 16.684 | 16.684 | +0.000 | 0 | 0 |  | archive_low_priority | `public_candidates/kojimar7185_95/base_submission/task132.onnx` |
| 133 | 14.579 | 14.703 | 14.703 | -0.124 | 4953 | -1025 | mechanism_teacher,params_down | extract_mechanism_before_any_adopt | `public_candidates/kojimar7185_95/base_submission/task133.onnx` |
| 134 | 17.266 | 17.266 | 17.266 | +0.000 | 0 | 0 |  | archive_low_priority | `public_candidates/kojimar7185_95/base_submission/task134.onnx` |
| 135 | 19.702 | 19.702 | 19.702 | +0.000 | 0 | 0 |  | archive_low_priority | `public_candidates/kojimar7185_95/base_submission/task135.onnx` |
| 136 | 17.899 | 17.899 | 17.899 | +0.000 | 0 | 0 |  | archive_low_priority | `public_candidates/kojimar7185_95/base_submission/task136.onnx` |
| 137 | 16.832 | 16.832 | 16.832 | +0.000 | 0 | 0 |  | archive_low_priority | `public_candidates/kojimar7185_95/base_submission/task137.onnx` |
| 138 | 15.456 | 15.552 | 15.552 | -0.096 | 1574 | -290 | mechanism_teacher,params_down | extract_mechanism_before_any_adopt | `public_candidates/kojimar7185_95/base_submission/task138.onnx` |
| 139 | 18.193 | 18.193 | 18.193 | +0.000 | 0 | 0 |  | archive_low_priority | `public_candidates/kojimar7185_95/base_submission/task139.onnx` |
| 140 | 23.391 | 23.391 | 23.391 | +0.000 | 0 | 0 |  | archive_low_priority | `public_candidates/kojimar7185_95/base_submission/task140.onnx` |
| 141 | 17.683 | 17.683 | 17.683 | +0.000 | 0 | 0 |  | archive_low_priority | `public_candidates/kojimar7185_95/base_submission/task141.onnx` |
| 142 | 19.358 | 19.807 | 19.807 | -0.449 | 144 | -42 | mechanism_teacher,params_down | extract_mechanism_before_any_adopt | `public_candidates/kojimar7185_95/base_submission/task142.onnx` |
| 143 | 18.269 | 18.269 | 18.269 | +0.000 | 0 | 0 |  | archive_low_priority | `public_candidates/kojimar7185_95/base_submission/task143.onnx` |
| 144 | 20.356 | 20.356 | 20.356 | +0.000 | 0 | 0 |  | archive_low_priority | `public_candidates/kojimar7185_95/base_submission/task144.onnx` |
| 145 | 15.511 | 15.742 | 15.742 | -0.231 | 3860 | -1137 | mechanism_teacher,params_down | extract_mechanism_before_any_adopt | `public_candidates/kojimar7185_95/base_submission/task145.onnx` |
| 146 | 18.795 | 18.812 | 18.812 | -0.016 | 8 | 0 |  | archive_low_priority | `public_candidates/kojimar7185_95/base_submission/task146.onnx` |
| 147 | 18.723 | 18.723 | 18.723 | +0.000 | 0 | 0 |  | archive_low_priority | `public_candidates/kojimar7185_95/base_submission/task147.onnx` |
| 148 | 16.299 | 16.323 | 16.323 | -0.023 | 0 | 138 |  | archive_low_priority | `public_candidates/kojimar7185_95/base_submission/task148.onnx` |
| 149 | 20.087 | 20.087 | 20.087 | +0.000 | 0 | 0 |  | archive_low_priority | `public_candidates/kojimar7185_95/base_submission/task149.onnx` |
| 150 | 19.937 | 20.073 | 20.073 | -0.135 | -8 | 28 | mechanism_teacher,memory_down | extract_mechanism_before_any_adopt | `public_candidates/kojimar7185_95/base_submission/task150.onnx` |
| 151 | 18.198 | 18.198 | 18.198 | +0.000 | 0 | 0 |  | archive_low_priority | `public_candidates/kojimar7185_95/base_submission/task151.onnx` |
| 152 | 18.873 | 19.807 | 19.807 | -0.934 | 360 | -82 | mechanism_teacher,params_down | extract_mechanism_before_any_adopt | `public_candidates/kojimar7185_95/base_submission/task152.onnx` |
| 153 | 18.322 | 18.322 | 18.322 | +0.000 | 0 | 0 |  | archive_low_priority | `public_candidates/kojimar7185_95/base_submission/task153.onnx` |
| 154 | 17.077 | 17.077 | 17.077 | +0.000 | 0 | 0 |  | archive_low_priority | `public_candidates/kojimar7185_95/base_submission/task154.onnx` |
| 155 | 20.044 | 20.044 | 20.044 | +0.000 | 0 | 0 |  | archive_low_priority | `public_candidates/kojimar7185_95/base_submission/task155.onnx` |
| 156 | 17.536 | 17.536 | 17.536 | +0.000 | 0 | 0 |  | archive_low_priority | `public_candidates/kojimar7185_95/base_submission/task156.onnx` |
| 157 | 15.973 | 15.972 | 15.972 | +0.001 | -5 | 0 | memory_down,score_up | extract_source_then_consider_adopt | `public_candidates/kojimar7185_95/base_submission/task157.onnx` |
| 157 | 15.972 | 15.972 | 15.972 | +0.000 | -2 | 0 | memory_down,score_up | extract_source_then_consider_adopt | `public_candidates/kojimar7185_95/overrides/task157.onnx` |
| 158 | 14.528 | 15.052 | 15.052 | -0.524 | 14724 | -307 | mechanism_teacher,new_qlinear_mechanism,params_down | extract_mechanism_before_any_adopt | `public_candidates/kojimar7185_95/base_submission/task158.onnx` |
| 159 | 17.642 | 17.642 | 17.642 | +0.000 | 0 | 0 |  | archive_low_priority | `public_candidates/kojimar7185_95/base_submission/task159.onnx` |
| 160 | 17.804 | 17.804 | 17.804 | +0.000 | 0 | 0 |  | archive_low_priority | `public_candidates/kojimar7185_95/base_submission/task160.onnx` |
| 161 | 17.148 | 17.148 | 17.148 | +0.000 | 0 | 0 |  | archive_low_priority | `public_candidates/kojimar7185_95/base_submission/task161.onnx` |
| 162 | 16.689 | 16.689 | 16.689 | +0.000 | 0 | 0 |  | archive_low_priority | `public_candidates/kojimar7185_95/base_submission/task162.onnx` |
| 163 | 17.240 | 17.506 | 17.506 | -0.266 | 580 | -32 | mechanism_teacher,params_down | extract_mechanism_before_any_adopt | `public_candidates/kojimar7185_95/base_submission/task163.onnx` |
| 164 | 21.599 | 21.599 | 21.599 | +0.000 | 0 | 0 |  | archive_low_priority | `public_candidates/kojimar7185_95/base_submission/task164.onnx` |
| 165 | 16.291 | 16.290 | 16.290 | +0.001 | -8 | 0 | memory_down,score_up | extract_source_then_consider_adopt | `public_candidates/kojimar7185_95/base_submission/task165.onnx` |
| 166 | 21.822 | 21.822 | 21.822 | +0.000 | 0 | 0 |  | archive_low_priority | `public_candidates/kojimar7185_95/base_submission/task166.onnx` |
| 167 | 20.003 | 20.003 | 20.003 | +0.000 | 0 | 0 |  | archive_low_priority | `public_candidates/kojimar7185_95/base_submission/task167.onnx` |
| 168 | 17.286 | 17.286 | 17.286 | +0.000 | 0 | 0 |  | archive_low_priority | `public_candidates/kojimar7185_95/base_submission/task168.onnx` |
| 169 | 17.579 | 17.579 | 17.579 | +0.000 | 0 | 0 |  | archive_low_priority | `public_candidates/kojimar7185_95/base_submission/task169.onnx` |
| 170 | 17.158 | 17.158 | 17.158 | +0.000 | 0 | 0 |  | archive_low_priority | `public_candidates/kojimar7185_95/base_submission/task170.onnx` |
| 171 | 18.187 | 18.187 | 18.187 | +0.000 | 0 | 0 |  | archive_low_priority | `public_candidates/kojimar7185_95/base_submission/task171.onnx` |
| 172 | 21.599 | 21.599 | 21.599 | +0.000 | 0 | 0 |  | archive_low_priority | `public_candidates/kojimar7185_95/base_submission/task172.onnx` |
| 173 | 15.222 | 15.139 | 15.139 | +0.083 | -1183 | -347 | memory_down,params_down,score_up | extract_source_then_consider_adopt | `public_candidates/kojimar7185_95/base_submission/task173.onnx` |
| 174 | 15.902 | 16.130 | 16.130 | -0.228 | 1786 | 36 |  | archive_low_priority | `public_candidates/kojimar7185_95/base_submission/task174.onnx` |
| 175 | 17.856 | 17.856 | 17.856 | +0.000 | 0 | 0 |  | archive_low_priority | `public_candidates/kojimar7185_95/base_submission/task175.onnx` |
| 176 | 19.519 | 19.519 | 19.519 | +0.000 | 0 | 0 |  | archive_low_priority | `public_candidates/kojimar7185_95/base_submission/task176.onnx` |
| 177 | 16.751 | 16.751 | 16.751 | +0.000 | 0 | 0 |  | archive_low_priority | `public_candidates/kojimar7185_95/base_submission/task177.onnx` |
| 178 | 18.363 | 18.363 | 18.363 | +0.000 | 0 | 0 |  | archive_low_priority | `public_candidates/kojimar7185_95/base_submission/task178.onnx` |
| 179 | 25.000 | 25.000 | 25.000 | +0.000 | 0 | 0 |  | archive_low_priority | `public_candidates/kojimar7185_95/base_submission/task179.onnx` |
| 180 | 19.653 | 19.653 | 19.653 | +0.000 | 0 | 0 |  | archive_low_priority | `public_candidates/kojimar7185_95/base_submission/task180.onnx` |
| 181 | 19.092 | 19.089 | 19.089 | +0.003 | 0 | -1 | params_down,score_up | extract_source_then_consider_adopt | `public_candidates/kojimar7185_95/base_submission/task181.onnx` |
| 182 | 16.176 | 16.229 | 16.229 | -0.053 | 350 | 1 |  | archive_low_priority | `public_candidates/kojimar7185_95/base_submission/task182.onnx` |
| 183 | 17.885 | 17.885 | 17.885 | +0.000 | 0 | 0 |  | archive_low_priority | `public_candidates/kojimar7185_95/base_submission/task183.onnx` |
| 184 | 17.168 | 17.168 | 17.168 | +0.000 | 0 | 0 |  | archive_low_priority | `public_candidates/kojimar7185_95/base_submission/task184.onnx` |
| 185 | 17.419 | 17.419 | 17.419 | +0.000 | 0 | 0 |  | archive_low_priority | `public_candidates/kojimar7185_95/base_submission/task185.onnx` |
| 186 | 20.365 | 20.365 | 20.365 | +0.000 | 0 | 0 |  | archive_low_priority | `public_candidates/kojimar7185_95/base_submission/task186.onnx` |
| 187 | 14.486 | 15.973 | 15.973 | -1.487 | 31150 | -2658 | mechanism_teacher,params_down | extract_mechanism_before_any_adopt | `public_candidates/kojimar7185_95/base_submission/task187.onnx` |
| 188 | 17.923 | 17.921 | 17.921 | +0.002 | -2 | 0 | memory_down,score_up | extract_source_then_consider_adopt | `public_candidates/kojimar7185_95/base_submission/task188.onnx` |
| 189 | 18.183 | 18.181 | 18.181 | +0.002 | 0 | -2 | params_down,score_up | extract_source_then_consider_adopt | `public_candidates/kojimar7185_95/base_submission/task189.onnx` |
| 190 | 17.165 | 17.165 | 17.165 | +0.000 | 0 | 0 |  | archive_low_priority | `public_candidates/kojimar7185_95/base_submission/task190.onnx` |
| 191 | 15.502 | 15.065 | 15.065 | +0.438 | -911 | -6405 | memory_down,new_qlinear_mechanism,params_down,score_up | extract_source_then_consider_adopt | `public_candidates/kojimar7185_95/base_submission/task191.onnx` |
| 191 | 15.502 | 15.065 | 15.065 | +0.437 | -911 | -6404 | memory_down,new_qlinear_mechanism,params_down,score_up | extract_source_then_consider_adopt | `public_candidates/kojimar7185_95/overrides/task191.onnx` |
| 192 | 15.938 | 16.147 | 16.147 | -0.209 | 2770 | -1145 | mechanism_teacher,params_down | extract_mechanism_before_any_adopt | `public_candidates/kojimar7185_95/base_submission/task192.onnx` |
| 193 | 18.187 | 15.630 | 15.630 | +2.556 | -9984 | -836 | memory_down,params_down,score_up | extract_source_then_consider_adopt | `public_candidates/kojimar7185_95/base_submission/task193.onnx` |
| 193 | 18.187 | 15.630 | 15.630 | +2.556 | -9984 | -836 | memory_down,params_down,score_up | extract_source_then_consider_adopt | `public_candidates/kojimar7185_95/overrides/task193.onnx` |
| 194 | 18.145 | 18.145 | 18.145 | +0.000 | 0 | 0 |  | archive_low_priority | `public_candidates/kojimar7185_95/base_submission/task194.onnx` |
| 195 | 18.175 | 18.175 | 18.175 | +0.000 | 0 | 0 |  | archive_low_priority | `public_candidates/kojimar7185_95/base_submission/task195.onnx` |
| 196 | 16.580 | 16.580 | 16.580 | +0.000 | 0 | 0 |  | archive_low_priority | `public_candidates/kojimar7185_95/base_submission/task196.onnx` |
| 197 | 17.136 | 17.136 | 17.136 | +0.000 | 0 | 0 |  | archive_low_priority | `public_candidates/kojimar7185_95/base_submission/task197.onnx` |
| 198 | 15.485 | 15.484 | 15.484 | +0.000 | -6 | 0 | memory_down,score_up | extract_source_then_consider_adopt | `public_candidates/kojimar7185_95/base_submission/task198.onnx` |
| 199 | 17.484 | 17.484 | 17.484 | +0.000 | 0 | 0 |  | archive_low_priority | `public_candidates/kojimar7185_95/base_submission/task199.onnx` |
| 200 | 18.372 | 18.372 | 18.372 | +0.000 | 0 | 0 |  | archive_low_priority | `public_candidates/kojimar7185_95/base_submission/task200.onnx` |
| 201 | 16.563 | 16.881 | 16.881 | -0.318 | 1274 | -16 | mechanism_teacher,params_down | extract_mechanism_before_any_adopt | `public_candidates/kojimar7185_95/base_submission/task201.onnx` |
| 202 | 15.669 | 16.902 | 16.902 | -1.233 | 8000 | -10 | mechanism_teacher,params_down | extract_mechanism_before_any_adopt | `public_candidates/kojimar7185_95/base_submission/task202.onnx` |
| 203 | 18.986 | 18.986 | 18.986 | +0.000 | 0 | 0 |  | archive_low_priority | `public_candidates/kojimar7185_95/base_submission/task203.onnx` |
| 204 | 15.650 | 15.767 | 15.767 | -0.116 | 1260 | 2 |  | archive_low_priority | `public_candidates/kojimar7185_95/base_submission/task204.onnx` |
| 205 | 15.577 | 15.588 | 15.588 | -0.011 | 136 | 5 |  | archive_low_priority | `public_candidates/kojimar7185_95/base_submission/task205.onnx` |
| 206 | 16.660 | 16.657 | 16.657 | +0.003 | -12 | -1 | memory_down,params_down,score_up | extract_source_then_consider_adopt | `public_candidates/kojimar7185_95/base_submission/task206.onnx` |
| 207 | 19.536 | 19.536 | 19.536 | +0.000 | 0 | 0 |  | archive_low_priority | `public_candidates/kojimar7185_95/base_submission/task207.onnx` |
| 208 | 16.287 | 16.539 | 16.539 | -0.252 | 1362 | -5 | mechanism_teacher,params_down | extract_mechanism_before_any_adopt | `public_candidates/kojimar7185_95/base_submission/task208.onnx` |
| 209 | 14.744 | 14.969 | 14.969 | -0.225 | 6939 | -1215 | mechanism_teacher,params_down | extract_mechanism_before_any_adopt | `public_candidates/kojimar7185_95/base_submission/task209.onnx` |
| 210 | 21.599 | 21.599 | 21.599 | +0.000 | 0 | 0 |  | archive_low_priority | `public_candidates/kojimar7185_95/base_submission/task210.onnx` |
| 211 | 19.019 | 19.296 | 19.296 | -0.278 | 240 | -144 | mechanism_teacher,params_down | extract_mechanism_before_any_adopt | `public_candidates/kojimar7185_95/base_submission/task211.onnx` |
| 212 | 17.234 | 17.234 | 17.234 | +0.000 | 0 | 0 |  | archive_low_priority | `public_candidates/kojimar7185_95/base_submission/task212.onnx` |
| 213 | 17.460 | 17.457 | 17.457 | +0.003 | -5 | 0 | memory_down,score_up | extract_source_then_consider_adopt | `public_candidates/kojimar7185_95/base_submission/task213.onnx` |
| 214 | 18.358 | 18.358 | 18.358 | +0.000 | 0 | 0 |  | archive_low_priority | `public_candidates/kojimar7185_95/base_submission/task214.onnx` |
| 215 | 18.380 | 18.380 | 18.380 | +0.000 | 0 | 0 |  | archive_low_priority | `public_candidates/kojimar7185_95/base_submission/task215.onnx` |
| 216 | 15.881 | 15.934 | 15.934 | -0.053 | 456 | 11 |  | archive_low_priority | `public_candidates/kojimar7185_95/base_submission/task216.onnx` |
| 217 | 17.707 | 17.707 | 17.707 | +0.000 | 0 | 0 |  | archive_low_priority | `public_candidates/kojimar7185_95/base_submission/task217.onnx` |
| 218 | 18.511 | 18.511 | 18.511 | +0.000 | 0 | 0 |  | archive_low_priority | `public_candidates/kojimar7185_95/base_submission/task218.onnx` |
| 219 | 15.174 | 16.117 | 16.117 | -0.943 | 11341 | -39 | mechanism_teacher,params_down | extract_mechanism_before_any_adopt | `public_candidates/kojimar7185_95/base_submission/task219.onnx` |
| 219 | 15.191 | 16.117 | 16.117 | -0.926 | 11033 | -41 | mechanism_teacher,params_down | extract_mechanism_before_any_adopt | `public_candidates/kojimar7185_95/overrides/task219.onnx` |
| 220 | 0.000 | 0.000 | 0.000 | +0.000 | 0 | 0 | public_fails_stored | reject_or_debug_public_candidate | `public_candidates/kojimar7185_95/base_submission/task220.onnx` |
| 221 | 18.309 | 18.309 | 18.309 | +0.000 | 0 | 0 |  | archive_low_priority | `public_candidates/kojimar7185_95/base_submission/task221.onnx` |
| 222 | 15.876 | 16.002 | 16.002 | -0.126 | 3992 | -2905 | mechanism_teacher,params_down | extract_mechanism_before_any_adopt | `public_candidates/kojimar7185_95/base_submission/task222.onnx` |
| 223 | 23.391 | 23.391 | 23.391 | +0.000 | 0 | 0 |  | archive_low_priority | `public_candidates/kojimar7185_95/base_submission/task223.onnx` |
| 224 | 17.052 | 17.052 | 17.052 | +0.000 | 0 | 0 |  | archive_low_priority | `public_candidates/kojimar7185_95/base_submission/task224.onnx` |
| 225 | 18.060 | 18.060 | 18.060 | +0.000 | 0 | 0 |  | archive_low_priority | `public_candidates/kojimar7185_95/base_submission/task225.onnx` |
| 226 | 17.602 | 17.602 | 17.602 | +0.000 | 0 | 0 |  | archive_low_priority | `public_candidates/kojimar7185_95/base_submission/task226.onnx` |
| 227 | 20.395 | 20.395 | 20.395 | +0.000 | 0 | 0 |  | archive_low_priority | `public_candidates/kojimar7185_95/base_submission/task227.onnx` |
| 228 | 18.131 | 18.131 | 18.131 | +0.000 | 0 | 0 |  | archive_low_priority | `public_candidates/kojimar7185_95/base_submission/task228.onnx` |
| 229 | 19.519 | 19.519 | 19.519 | +0.000 | 0 | 0 |  | archive_low_priority | `public_candidates/kojimar7185_95/base_submission/task229.onnx` |
| 230 | 0.000 | 0.000 | 0.000 | +0.000 | 0 | 0 | public_fails_stored | reject_or_debug_public_candidate | `public_candidates/kojimar7185_95/base_submission/task230.onnx` |
| 231 | 19.611 | 19.611 | 19.611 | +0.000 | 0 | 0 |  | archive_low_priority | `public_candidates/kojimar7185_95/base_submission/task231.onnx` |
| 232 | 17.749 | 17.749 | 17.749 | +0.000 | 0 | 0 |  | archive_low_priority | `public_candidates/kojimar7185_95/base_submission/task232.onnx` |
| 233 | 14.368 | 14.588 | 14.588 | -0.221 | 7930 | 271 | new_qlinear_mechanism | extract_mechanism_before_any_adopt | `public_candidates/kojimar7185_95/base_submission/task233.onnx` |
| 234 | 16.169 | 16.755 | 16.755 | -0.586 | 3056 | -19 | mechanism_teacher,params_down | extract_mechanism_before_any_adopt | `public_candidates/kojimar7185_95/base_submission/task234.onnx` |
| 235 | 19.864 | 19.864 | 19.864 | +0.000 | 0 | 0 |  | archive_low_priority | `public_candidates/kojimar7185_95/base_submission/task235.onnx` |
| 236 | 19.532 | 19.532 | 19.532 | +0.000 | 0 | 0 |  | archive_low_priority | `public_candidates/kojimar7185_95/base_submission/task236.onnx` |
| 237 | 17.450 | 17.450 | 17.450 | +0.000 | 0 | 0 |  | archive_low_priority | `public_candidates/kojimar7185_95/base_submission/task237.onnx` |
| 238 | 17.178 | 17.178 | 17.178 | +0.000 | 0 | 0 |  | archive_low_priority | `public_candidates/kojimar7185_95/base_submission/task238.onnx` |
| 239 | 18.147 | 18.147 | 18.147 | +0.000 | 0 | 0 |  | archive_low_priority | `public_candidates/kojimar7185_95/base_submission/task239.onnx` |
| 240 | 17.704 | 17.704 | 17.704 | +0.000 | 0 | 0 |  | archive_low_priority | `public_candidates/kojimar7185_95/base_submission/task240.onnx` |
| 241 | 25.000 | 25.000 | 25.000 | +0.000 | 0 | 0 |  | archive_low_priority | `public_candidates/kojimar7185_95/base_submission/task241.onnx` |
| 242 | 18.932 | 18.932 | 18.932 | +0.000 | 0 | 0 |  | archive_low_priority | `public_candidates/kojimar7185_95/base_submission/task242.onnx` |
| 243 | 14.972 | 16.298 | 16.298 | -1.326 | 17496 | -859 | mechanism_teacher,new_qlinear_mechanism,params_down | extract_mechanism_before_any_adopt | `public_candidates/kojimar7185_95/base_submission/task243.onnx` |
| 243 | 14.972 | 16.298 | 16.298 | -1.326 | 17496 | -857 | mechanism_teacher,new_qlinear_mechanism,params_down | extract_mechanism_before_any_adopt | `public_candidates/kojimar7185_95/overrides/task243.onnx` |
| 244 | 17.928 | 17.928 | 17.928 | +0.000 | 0 | 0 |  | archive_low_priority | `public_candidates/kojimar7185_95/base_submission/task244.onnx` |
| 245 | 17.083 | 17.083 | 17.083 | +0.000 | 0 | 0 |  | archive_low_priority | `public_candidates/kojimar7185_95/base_submission/task245.onnx` |
| 246 | 17.080 | 17.080 | 17.080 | +0.000 | 0 | 0 |  | archive_low_priority | `public_candidates/kojimar7185_95/base_submission/task246.onnx` |
| 247 | 18.847 | 18.847 | 18.847 | +0.000 | 0 | 0 |  | archive_low_priority | `public_candidates/kojimar7185_95/base_submission/task247.onnx` |
| 248 | 19.983 | 19.983 | 19.983 | +0.000 | 0 | 0 |  | archive_low_priority | `public_candidates/kojimar7185_95/base_submission/task248.onnx` |
| 249 | 19.341 | 19.341 | 19.341 | +0.000 | 0 | 0 |  | archive_low_priority | `public_candidates/kojimar7185_95/base_submission/task249.onnx` |
| 250 | 16.725 | 16.921 | 16.921 | -0.196 | 600 | 100 |  | archive_low_priority | `public_candidates/kojimar7185_95/base_submission/task250.onnx` |
| 251 | 16.499 | 16.979 | 16.979 | -0.480 | 1980 | -104 | mechanism_teacher,params_down | extract_mechanism_before_any_adopt | `public_candidates/kojimar7185_95/base_submission/task251.onnx` |
| 252 | 19.439 | 19.439 | 19.439 | +0.000 | 0 | 0 |  | archive_low_priority | `public_candidates/kojimar7185_95/base_submission/task252.onnx` |
| 253 | 18.616 | 18.616 | 18.616 | +0.000 | 0 | 0 |  | archive_low_priority | `public_candidates/kojimar7185_95/base_submission/task253.onnx` |
| 254 | 17.999 | 17.999 | 17.999 | +0.000 | 0 | 0 |  | archive_low_priority | `public_candidates/kojimar7185_95/base_submission/task254.onnx` |
| 255 | 14.967 | 15.387 | 15.387 | -0.420 | 7847 | -32 | mechanism_teacher,params_down | extract_mechanism_before_any_adopt | `public_candidates/kojimar7185_95/base_submission/task255.onnx` |
| 256 | 17.011 | 17.269 | 17.269 | -0.257 | 675 | -6 | mechanism_teacher,params_down | extract_mechanism_before_any_adopt | `public_candidates/kojimar7185_95/base_submission/task256.onnx` |
| 257 | 18.986 | 18.986 | 18.986 | +0.000 | 0 | 0 |  | archive_low_priority | `public_candidates/kojimar7185_95/base_submission/task257.onnx` |
| 258 | 19.925 | 19.925 | 19.925 | +0.000 | 0 | 0 |  | archive_low_priority | `public_candidates/kojimar7185_95/base_submission/task258.onnx` |
| 259 | 18.192 | 18.192 | 18.192 | +0.000 | 0 | 0 |  | archive_low_priority | `public_candidates/kojimar7185_95/base_submission/task259.onnx` |
| 260 | 17.349 | 17.349 | 17.349 | +0.000 | 0 | 0 |  | archive_low_priority | `public_candidates/kojimar7185_95/base_submission/task260.onnx` |
| 261 | 19.702 | 19.702 | 19.702 | +0.000 | 0 | 0 |  | archive_low_priority | `public_candidates/kojimar7185_95/base_submission/task261.onnx` |
| 262 | 20.337 | 20.337 | 20.337 | +0.000 | 0 | 0 |  | archive_low_priority | `public_candidates/kojimar7185_95/base_submission/task262.onnx` |
| 263 | 18.478 | 18.478 | 18.478 | +0.000 | 0 | 0 |  | archive_low_priority | `public_candidates/kojimar7185_95/base_submission/task263.onnx` |
| 264 | 16.292 | 16.292 | 16.292 | +0.000 | 0 | 0 |  | archive_low_priority | `public_candidates/kojimar7185_95/base_submission/task264.onnx` |
| 265 | 16.619 | 16.619 | 16.619 | +0.000 | 0 | 0 |  | archive_low_priority | `public_candidates/kojimar7185_95/base_submission/task265.onnx` |
| 266 | 19.142 | 19.142 | 19.142 | +0.000 | 0 | 0 |  | archive_low_priority | `public_candidates/kojimar7185_95/base_submission/task266.onnx` |
| 267 | 18.351 | 18.359 | 18.359 | -0.008 | 10 | -4 | mechanism_teacher,params_down | extract_mechanism_before_any_adopt | `public_candidates/kojimar7185_95/base_submission/task267.onnx` |
| 268 | 16.816 | 16.816 | 16.816 | +0.000 | 0 | 0 |  | archive_low_priority | `public_candidates/kojimar7185_95/base_submission/task268.onnx` |
| 269 | 18.436 | 18.436 | 18.436 | +0.000 | 0 | 0 |  | archive_low_priority | `public_candidates/kojimar7185_95/base_submission/task269.onnx` |
| 270 | 16.971 | 16.971 | 16.971 | +0.000 | 0 | 0 |  | archive_low_priority | `public_candidates/kojimar7185_95/base_submission/task270.onnx` |
| 271 | 17.921 | 17.921 | 17.921 | +0.000 | 0 | 0 |  | archive_low_priority | `public_candidates/kojimar7185_95/base_submission/task271.onnx` |
| 272 | 19.004 | 19.004 | 19.004 | +0.000 | 0 | 0 |  | archive_low_priority | `public_candidates/kojimar7185_95/base_submission/task272.onnx` |
| 273 | 17.322 | 17.322 | 17.322 | +0.000 | 0 | 0 |  | archive_low_priority | `public_candidates/kojimar7185_95/base_submission/task273.onnx` |
| 274 | 19.579 | 19.579 | 19.579 | +0.000 | 0 | 0 |  | archive_low_priority | `public_candidates/kojimar7185_95/base_submission/task274.onnx` |
| 275 | 17.394 | 17.394 | 17.394 | +0.000 | 0 | 0 |  | archive_low_priority | `public_candidates/kojimar7185_95/base_submission/task275.onnx` |
| 276 | 22.697 | 22.697 | 22.697 | +0.000 | 0 | 0 |  | archive_low_priority | `public_candidates/kojimar7185_95/base_submission/task276.onnx` |
| 277 | 16.725 | 16.725 | 16.725 | +0.000 | 0 | 0 |  | archive_low_priority | `public_candidates/kojimar7185_95/base_submission/task277.onnx` |
| 277 | 16.772 | 16.725 | 16.725 | +0.047 | 0 | -180 | params_down,score_up | extract_source_then_consider_adopt | `public_candidates/kojimar7185_95/overrides/task277.onnx` |
| 278 | 16.279 | 16.391 | 16.391 | -0.112 | 648 | 0 |  | archive_low_priority | `public_candidates/kojimar7185_95/base_submission/task278.onnx` |
| 279 | 16.378 | 16.379 | 16.379 | -0.002 | 0 | 9 |  | archive_low_priority | `public_candidates/kojimar7185_95/base_submission/task279.onnx` |
| 280 | 16.295 | 16.670 | 16.670 | -0.375 | 1919 | -30 | mechanism_teacher,params_down | extract_mechanism_before_any_adopt | `public_candidates/kojimar7185_95/base_submission/task280.onnx` |
| 281 | 17.223 | 17.223 | 17.223 | +0.000 | 0 | 0 |  | archive_low_priority | `public_candidates/kojimar7185_95/base_submission/task281.onnx` |
| 282 | 18.900 | 18.900 | 18.900 | +0.000 | 0 | 0 |  | archive_low_priority | `public_candidates/kojimar7185_95/base_submission/task282.onnx` |
| 283 | 18.187 | 18.187 | 18.187 | +0.000 | 0 | 0 |  | archive_low_priority | `public_candidates/kojimar7185_95/base_submission/task283.onnx` |
| 284 | 16.774 | 16.774 | 16.774 | +0.000 | 0 | 0 |  | archive_low_priority | `public_candidates/kojimar7185_95/base_submission/task284.onnx` |
| 285 | 14.864 | 14.864 | 14.864 | +0.000 | 0 | 0 |  | archive_low_priority | `public_candidates/kojimar7185_95/base_submission/task285.onnx` |
| 286 | 14.250 | 14.915 | 14.915 | -0.666 | 24822 | -2140 | mechanism_teacher,new_qlinear_mechanism,params_down | extract_mechanism_before_any_adopt | `public_candidates/kojimar7185_95/base_submission/task286.onnx` |
| 287 | 17.402 | 17.402 | 17.402 | +0.000 | 0 | 0 |  | archive_low_priority | `public_candidates/kojimar7185_95/base_submission/task287.onnx` |
| 288 | 18.135 | 18.135 | 18.135 | +0.000 | 0 | 0 |  | archive_low_priority | `public_candidates/kojimar7185_95/base_submission/task288.onnx` |
| 289 | 18.391 | 18.391 | 18.391 | +0.000 | 0 | 0 |  | archive_low_priority | `public_candidates/kojimar7185_95/base_submission/task289.onnx` |
| 290 | 19.128 | 19.128 | 19.128 | +0.000 | 0 | 0 |  | archive_low_priority | `public_candidates/kojimar7185_95/base_submission/task290.onnx` |
| 291 | 20.780 | 20.780 | 20.780 | +0.000 | 0 | 0 |  | archive_low_priority | `public_candidates/kojimar7185_95/base_submission/task291.onnx` |
| 292 | 18.927 | 18.927 | 18.927 | +0.000 | 0 | 0 |  | archive_low_priority | `public_candidates/kojimar7185_95/base_submission/task292.onnx` |
| 293 | 17.839 | 17.839 | 17.839 | +0.000 | 0 | 0 |  | archive_low_priority | `public_candidates/kojimar7185_95/base_submission/task293.onnx` |
| 294 | 0.000 | 0.000 | 0.000 | +0.000 | 0 | 0 | public_fails_stored | reject_or_debug_public_candidate | `public_candidates/kojimar7185_95/base_submission/task294.onnx` |
| 295 | 17.620 | 17.620 | 17.620 | +0.000 | 0 | 0 |  | archive_low_priority | `public_candidates/kojimar7185_95/base_submission/task295.onnx` |
| 296 | 19.201 | 19.201 | 19.201 | +0.000 | 0 | 0 |  | archive_low_priority | `public_candidates/kojimar7185_95/base_submission/task296.onnx` |
| 297 | 17.764 | 17.764 | 17.764 | +0.000 | 0 | 0 |  | archive_low_priority | `public_candidates/kojimar7185_95/base_submission/task297.onnx` |
| 298 | 20.095 | 20.095 | 20.095 | +0.000 | 0 | 0 |  | archive_low_priority | `public_candidates/kojimar7185_95/base_submission/task298.onnx` |
| 299 | 21.088 | 21.088 | 21.088 | +0.000 | 0 | 0 |  | archive_low_priority | `public_candidates/kojimar7185_95/base_submission/task299.onnx` |
| 300 | 18.708 | 18.708 | 18.708 | +0.000 | 0 | 0 |  | archive_low_priority | `public_candidates/kojimar7185_95/base_submission/task300.onnx` |
| 301 | 17.960 | 17.960 | 17.960 | +0.000 | 0 | 0 |  | archive_low_priority | `public_candidates/kojimar7185_95/base_submission/task301.onnx` |
| 302 | 17.519 | 17.519 | 17.519 | +0.000 | 0 | 0 |  | archive_low_priority | `public_candidates/kojimar7185_95/base_submission/task302.onnx` |
| 303 | 17.719 | 17.719 | 17.719 | +0.000 | 0 | 0 |  | archive_low_priority | `public_candidates/kojimar7185_95/base_submission/task303.onnx` |
| 304 | 17.727 | 17.727 | 17.727 | +0.000 | 0 | 0 |  | archive_low_priority | `public_candidates/kojimar7185_95/base_submission/task304.onnx` |
| 305 | 18.545 | 18.545 | 18.545 | +0.000 | 0 | 0 |  | archive_low_priority | `public_candidates/kojimar7185_95/base_submission/task305.onnx` |
| 306 | 19.296 | 19.296 | 19.296 | +0.000 | 0 | 0 |  | archive_low_priority | `public_candidates/kojimar7185_95/base_submission/task306.onnx` |
| 307 | 23.391 | 23.391 | 23.391 | +0.000 | 0 | 0 |  | archive_low_priority | `public_candidates/kojimar7185_95/base_submission/task307.onnx` |
| 308 | 17.695 | 17.695 | 17.695 | +0.000 | 0 | 0 |  | archive_low_priority | `public_candidates/kojimar7185_95/base_submission/task308.onnx` |
| 309 | 22.697 | 22.697 | 22.697 | +0.000 | 0 | 0 |  | archive_low_priority | `public_candidates/kojimar7185_95/base_submission/task309.onnx` |
| 310 | 16.902 | 16.902 | 16.902 | +0.000 | 0 | 0 |  | archive_low_priority | `public_candidates/kojimar7185_95/base_submission/task310.onnx` |
| 311 | 21.599 | 21.599 | 21.599 | +0.000 | 0 | 0 |  | archive_low_priority | `public_candidates/kojimar7185_95/base_submission/task311.onnx` |
| 312 | 22.004 | 22.004 | 22.004 | +0.000 | 0 | 0 |  | archive_low_priority | `public_candidates/kojimar7185_95/base_submission/task312.onnx` |
| 313 | 19.764 | 19.764 | 19.764 | +0.000 | 0 | 0 |  | archive_low_priority | `public_candidates/kojimar7185_95/base_submission/task313.onnx` |
| 314 | 20.395 | 20.395 | 20.395 | +0.000 | 0 | 0 |  | archive_low_priority | `public_candidates/kojimar7185_95/base_submission/task314.onnx` |
| 315 | 19.562 | 19.562 | 19.562 | +0.000 | 0 | 0 |  | archive_low_priority | `public_candidates/kojimar7185_95/base_submission/task315.onnx` |
| 316 | 18.946 | 18.946 | 18.946 | +0.000 | 0 | 0 |  | archive_low_priority | `public_candidates/kojimar7185_95/base_submission/task316.onnx` |
| 317 | 20.016 | 20.016 | 20.016 | +0.000 | 0 | 0 |  | archive_low_priority | `public_candidates/kojimar7185_95/base_submission/task317.onnx` |
| 318 | 20.395 | 20.395 | 20.395 | +0.000 | 0 | 0 |  | archive_low_priority | `public_candidates/kojimar7185_95/base_submission/task318.onnx` |
| 319 | 15.005 | 15.161 | 15.161 | -0.155 | 3032 | 123 |  | archive_low_priority | `public_candidates/kojimar7185_95/base_submission/task319.onnx` |
| 320 | 18.354 | 18.354 | 18.354 | +0.000 | 0 | 0 |  | archive_low_priority | `public_candidates/kojimar7185_95/base_submission/task320.onnx` |
| 321 | 19.296 | 19.296 | 19.296 | +0.000 | 0 | 0 |  | archive_low_priority | `public_candidates/kojimar7185_95/base_submission/task321.onnx` |
| 322 | 19.702 | 19.702 | 19.702 | +0.000 | 0 | 0 |  | archive_low_priority | `public_candidates/kojimar7185_95/base_submission/task322.onnx` |
| 323 | 17.314 | 17.314 | 17.314 | +0.000 | 0 | 0 |  | archive_low_priority | `public_candidates/kojimar7185_95/base_submission/task323.onnx` |
| 324 | 15.907 | 15.907 | 15.907 | +0.000 | 0 | 0 |  | archive_low_priority | `public_candidates/kojimar7185_95/base_submission/task324.onnx` |
| 325 | 17.335 | 17.335 | 17.335 | +0.000 | 0 | 0 |  | archive_low_priority | `public_candidates/kojimar7185_95/base_submission/task325.onnx` |
| 326 | 21.599 | 21.599 | 21.599 | +0.000 | 0 | 0 |  | archive_low_priority | `public_candidates/kojimar7185_95/base_submission/task326.onnx` |
| 327 | 18.219 | 18.219 | 18.219 | +0.000 | 0 | 0 |  | archive_low_priority | `public_candidates/kojimar7185_95/base_submission/task327.onnx` |
| 328 | 16.160 | 16.179 | 16.179 | -0.019 | 127 | 0 |  | archive_low_priority | `public_candidates/kojimar7185_95/base_submission/task328.onnx` |
| 329 | 18.035 | 18.035 | 18.035 | +0.000 | 0 | 0 |  | archive_low_priority | `public_candidates/kojimar7185_95/base_submission/task329.onnx` |
| 330 | 16.986 | 16.986 | 16.986 | +0.000 | 0 | 0 |  | archive_low_priority | `public_candidates/kojimar7185_95/base_submission/task330.onnx` |
| 330 | 16.986 | 16.986 | 16.986 | -0.001 | 0 | 2 | possible_final_equal_route | extract_mechanism_before_any_adopt | `public_candidates/kojimar7185_95/overrides/task330.onnx` |
| 331 | 18.187 | 18.187 | 18.187 | +0.000 | 0 | 0 |  | archive_low_priority | `public_candidates/kojimar7185_95/base_submission/task331.onnx` |
| 332 | 18.667 | 18.667 | 18.667 | +0.000 | 0 | 0 |  | archive_low_priority | `public_candidates/kojimar7185_95/base_submission/task332.onnx` |
| 333 | 16.921 | 16.921 | 16.921 | +0.000 | 0 | 0 |  | archive_low_priority | `public_candidates/kojimar7185_95/base_submission/task333.onnx` |
| 334 | 19.717 | 19.717 | 19.717 | +0.000 | 0 | 0 |  | archive_low_priority | `public_candidates/kojimar7185_95/base_submission/task334.onnx` |
| 335 | 16.839 | 17.297 | 17.297 | -0.458 | 734 | 552 |  | archive_low_priority | `public_candidates/kojimar7185_95/base_submission/task335.onnx` |
| 336 | 17.429 | 17.429 | 17.429 | +0.000 | 0 | 0 |  | archive_low_priority | `public_candidates/kojimar7185_95/base_submission/task336.onnx` |
| 337 | 22.697 | 22.697 | 22.697 | +0.000 | 0 | 0 |  | archive_low_priority | `public_candidates/kojimar7185_95/base_submission/task337.onnx` |
| 338 | 15.268 | 15.725 | 15.725 | -0.457 | 6146 | 33 |  | archive_low_priority | `public_candidates/kojimar7185_95/base_submission/task338.onnx` |
| 339 | 20.058 | 20.058 | 20.058 | +0.000 | 0 | 0 |  | archive_low_priority | `public_candidates/kojimar7185_95/base_submission/task339.onnx` |
| 340 | 15.691 | 16.285 | 16.285 | -0.595 | 5098 | -150 | mechanism_teacher,params_down | extract_mechanism_before_any_adopt | `public_candidates/kojimar7185_95/base_submission/task340.onnx` |
| 341 | 17.735 | 17.735 | 17.735 | +0.000 | 0 | 0 |  | archive_low_priority | `public_candidates/kojimar7185_95/base_submission/task341.onnx` |
| 342 | 17.632 | 17.632 | 17.632 | +0.000 | 0 | 0 |  | archive_low_priority | `public_candidates/kojimar7185_95/base_submission/task342.onnx` |
| 343 | 17.381 | 17.892 | 17.892 | -0.511 | 780 | 35 |  | archive_low_priority | `public_candidates/kojimar7185_95/base_submission/task343.onnx` |
| 344 | 18.187 | 18.187 | 18.187 | +0.000 | 0 | 0 |  | archive_low_priority | `public_candidates/kojimar7185_95/base_submission/task344.onnx` |
| 345 | 17.690 | 17.690 | 17.690 | +0.000 | 0 | 0 |  | archive_low_priority | `public_candidates/kojimar7185_95/base_submission/task345.onnx` |
| 346 | 17.866 | 17.866 | 17.866 | +0.000 | 0 | 0 |  | archive_low_priority | `public_candidates/kojimar7185_95/base_submission/task346.onnx` |
| 347 | 20.037 | 20.037 | 20.037 | +0.000 | 0 | 0 |  | archive_low_priority | `public_candidates/kojimar7185_95/base_submission/task347.onnx` |
| 348 | 17.271 | 17.452 | 17.452 | -0.181 | 280 | 97 |  | archive_low_priority | `public_candidates/kojimar7185_95/base_submission/task348.onnx` |
| 349 | 15.015 | 15.289 | 15.289 | -0.274 | 6300 | -1103 | mechanism_teacher,new_qlinear_mechanism,params_down | extract_mechanism_before_any_adopt | `public_candidates/kojimar7185_95/base_submission/task349.onnx` |
| 350 | 15.891 | 15.891 | 15.891 | +0.000 | 0 | -2 | params_down,score_up | extract_source_then_consider_adopt | `public_candidates/kojimar7185_95/base_submission/task350.onnx` |
| 351 | 17.591 | 18.077 | 18.077 | -0.486 | 676 | -40 | mechanism_teacher,params_down | extract_mechanism_before_any_adopt | `public_candidates/kojimar7185_95/base_submission/task351.onnx` |
| 352 | 0.000 | 0.000 | 0.000 | +0.000 | 0 | 0 | public_fails_stored | reject_or_debug_public_candidate | `public_candidates/kojimar7185_95/base_submission/task352.onnx` |
| 353 | 19.189 | 19.189 | 19.189 | +0.000 | 0 | 0 |  | archive_low_priority | `public_candidates/kojimar7185_95/base_submission/task353.onnx` |
| 354 | 16.968 | 17.080 | 17.080 | -0.112 | 324 | 2 |  | archive_low_priority | `public_candidates/kojimar7185_95/base_submission/task354.onnx` |
| 355 | 17.079 | 17.093 | 17.093 | -0.015 | 40 | 0 |  | archive_low_priority | `public_candidates/kojimar7185_95/base_submission/task355.onnx` |
| 356 | 17.815 | 17.815 | 17.815 | +0.000 | 0 | 0 |  | archive_low_priority | `public_candidates/kojimar7185_95/base_submission/task356.onnx` |
| 357 | 19.327 | 19.327 | 19.327 | +0.000 | 0 | 0 |  | archive_low_priority | `public_candidates/kojimar7185_95/base_submission/task357.onnx` |
| 358 | 16.609 | 16.609 | 16.609 | +0.000 | 0 | 0 |  | archive_low_priority | `public_candidates/kojimar7185_95/base_submission/task358.onnx` |
| 359 | 16.596 | 16.740 | 16.740 | -0.144 | 600 | -1 | mechanism_teacher,params_down | extract_mechanism_before_any_adopt | `public_candidates/kojimar7185_95/base_submission/task359.onnx` |
| 360 | 19.171 | 19.171 | 19.171 | +0.000 | 0 | 0 |  | archive_low_priority | `public_candidates/kojimar7185_95/base_submission/task360.onnx` |
| 361 | 16.516 | 16.547 | 16.547 | -0.032 | 150 | 1 |  | archive_low_priority | `public_candidates/kojimar7185_95/base_submission/task361.onnx` |
| 362 | 18.744 | 18.744 | 18.744 | +0.000 | 0 | 0 |  | archive_low_priority | `public_candidates/kojimar7185_95/base_submission/task362.onnx` |
| 363 | 16.558 | 16.673 | 16.673 | -0.114 | 500 | 1 |  | archive_low_priority | `public_candidates/kojimar7185_95/base_submission/task363.onnx` |
| 364 | 14.956 | 15.115 | 15.115 | -0.159 | 4400 | -1018 | mechanism_teacher,params_down | extract_mechanism_before_any_adopt | `public_candidates/kojimar7185_95/base_submission/task364.onnx` |
| 365 | 16.699 | 16.699 | 16.699 | +0.000 | 0 | 0 |  | archive_low_priority | `public_candidates/kojimar7185_95/base_submission/task365.onnx` |
| 366 | 14.498 | 14.640 | 14.640 | -0.142 | 4914 | -88 | mechanism_teacher,params_down | extract_mechanism_before_any_adopt | `public_candidates/kojimar7185_95/base_submission/task366.onnx` |
| 367 | 14.917 | 15.121 | 15.121 | -0.203 | 3500 | 902 |  | archive_low_priority | `public_candidates/kojimar7185_95/base_submission/task367.onnx` |
| 368 | 16.264 | 16.264 | 16.264 | +0.000 | 0 | 0 |  | archive_low_priority | `public_candidates/kojimar7185_95/base_submission/task368.onnx` |
| 369 | 17.623 | 17.623 | 17.623 | +0.000 | 0 | 0 |  | archive_low_priority | `public_candidates/kojimar7185_95/base_submission/task369.onnx` |
| 370 | 15.753 | 15.823 | 15.823 | -0.070 | 461 | 243 |  | archive_low_priority | `public_candidates/kojimar7185_95/base_submission/task370.onnx` |
| 371 | 18.854 | 18.854 | 18.854 | +0.000 | 0 | 0 |  | archive_low_priority | `public_candidates/kojimar7185_95/base_submission/task371.onnx` |
| 372 | 18.435 | 18.435 | 18.435 | +0.000 | 0 | 0 |  | archive_low_priority | `public_candidates/kojimar7185_95/base_submission/task372.onnx` |
| 373 | 20.906 | 20.906 | 20.906 | +0.000 | 0 | 0 |  | archive_low_priority | `public_candidates/kojimar7185_95/base_submission/task373.onnx` |
| 374 | 17.258 | 17.258 | 17.258 | +0.000 | 0 | 0 |  | archive_low_priority | `public_candidates/kojimar7185_95/base_submission/task374.onnx` |
| 375 | 19.142 | 19.142 | 19.142 | +0.000 | 0 | 0 |  | archive_low_priority | `public_candidates/kojimar7185_95/base_submission/task375.onnx` |
| 376 | 19.774 | 19.774 | 19.774 | +0.000 | 0 | 0 |  | archive_low_priority | `public_candidates/kojimar7185_95/base_submission/task376.onnx` |
| 377 | 15.952 | 15.981 | 15.981 | -0.029 | 240 | 4 |  | archive_low_priority | `public_candidates/kojimar7185_95/base_submission/task377.onnx` |
| 378 | 16.591 | 16.600 | 16.600 | -0.009 | 40 | 0 |  | archive_low_priority | `public_candidates/kojimar7185_95/base_submission/task378.onnx` |
| 379 | 15.713 | 15.847 | 15.847 | -0.134 | 211 | 1141 |  | archive_low_priority | `public_candidates/kojimar7185_95/base_submission/task379.onnx` |
| 380 | 20.405 | 20.405 | 20.405 | +0.000 | 0 | 0 |  | archive_low_priority | `public_candidates/kojimar7185_95/base_submission/task380.onnx` |
| 381 | 17.305 | 17.418 | 17.418 | -0.113 | 220 | 15 |  | archive_low_priority | `public_candidates/kojimar7185_95/base_submission/task381.onnx` |
| 382 | 16.337 | 16.337 | 16.337 | +0.000 | 0 | 0 |  | archive_low_priority | `public_candidates/kojimar7185_95/base_submission/task382.onnx` |
| 383 | 16.102 | 16.289 | 16.289 | -0.187 | 1272 | -22 | mechanism_teacher,params_down | extract_mechanism_before_any_adopt | `public_candidates/kojimar7185_95/base_submission/task383.onnx` |
| 384 | 18.615 | 18.615 | 18.615 | +0.000 | 0 | 0 |  | archive_low_priority | `public_candidates/kojimar7185_95/base_submission/task384.onnx` |
| 385 | 21.599 | 21.599 | 21.599 | +0.000 | 0 | 0 |  | archive_low_priority | `public_candidates/kojimar7185_95/base_submission/task385.onnx` |
| 386 | 19.648 | 19.648 | 19.648 | +0.000 | 0 | 0 |  | archive_low_priority | `public_candidates/kojimar7185_95/base_submission/task386.onnx` |
| 387 | 16.471 | 16.471 | 16.471 | +0.000 | 0 | 0 |  | archive_low_priority | `public_candidates/kojimar7185_95/base_submission/task387.onnx` |
| 388 | 17.178 | 17.178 | 17.178 | +0.000 | 0 | 0 |  | archive_low_priority | `public_candidates/kojimar7185_95/base_submission/task388.onnx` |
| 389 | 20.058 | 20.058 | 20.058 | +0.000 | 0 | 0 |  | archive_low_priority | `public_candidates/kojimar7185_95/base_submission/task389.onnx` |
| 390 | 16.964 | 16.964 | 16.964 | +0.000 | -1 | 0 | memory_down,score_up | extract_source_then_consider_adopt | `public_candidates/kojimar7185_95/base_submission/task390.onnx` |
| 391 | 19.931 | 19.931 | 19.931 | +0.000 | 0 | 0 |  | archive_low_priority | `public_candidates/kojimar7185_95/base_submission/task391.onnx` |
| 392 | 17.424 | 17.424 | 17.424 | +0.000 | 0 | 0 |  | archive_low_priority | `public_candidates/kojimar7185_95/base_submission/task392.onnx` |
| 393 | 20.030 | 20.030 | 20.030 | +0.000 | 0 | 0 |  | archive_low_priority | `public_candidates/kojimar7185_95/base_submission/task393.onnx` |
| 394 | 17.063 | 17.063 | 17.063 | +0.000 | 0 | 0 |  | archive_low_priority | `public_candidates/kojimar7185_95/base_submission/task394.onnx` |
| 395 | 19.894 | 19.894 | 19.894 | +0.000 | 0 | 0 |  | archive_low_priority | `public_candidates/kojimar7185_95/base_submission/task395.onnx` |
| 396 | 15.634 | 15.825 | 15.825 | -0.191 | 3705 | -1675 | mechanism_teacher,new_qlinear_mechanism,params_down | extract_mechanism_before_any_adopt | `public_candidates/kojimar7185_95/base_submission/task396.onnx` |
| 397 | 17.069 | 17.068 | 17.068 | +0.001 | -3 | 0 | memory_down,score_up | extract_source_then_consider_adopt | `public_candidates/kojimar7185_95/base_submission/task397.onnx` |
| 398 | 16.399 | 17.042 | 17.042 | -0.643 | 3681 | -1101 | mechanism_teacher,params_down | extract_mechanism_before_any_adopt | `public_candidates/kojimar7185_95/base_submission/task398.onnx` |
| 399 | 20.489 | 20.489 | 20.489 | +0.000 | 0 | 0 |  | archive_low_priority | `public_candidates/kojimar7185_95/base_submission/task399.onnx` |
| 400 | 17.468 | 17.468 | 17.468 | +0.000 | 0 | 0 |  | archive_low_priority | `public_candidates/kojimar7185_95/base_submission/task400.onnx` |
