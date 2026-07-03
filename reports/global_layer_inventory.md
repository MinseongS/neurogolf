# Global layer inventory summary

- tasks indexed: 400
- failures: 0
- source-controlled build(): 400/400
- no build(): none

## Source classes

- `exact_preserve`: 328
- `semantic_or_handbuilt`: 62
- `unknown`: 9
- `heuristic`: 1

## Largest op families

- `Cast`: 291 tasks
- `Slice`: 228 tasks
- `Pad`: 223 tasks
- `Equal`: 213 tasks
- `Where`: 202 tasks
- `Concat`: 178 tasks
- `Add`: 154 tasks
- `Gather`: 144 tasks
- `Sub`: 132 tasks
- `Einsum`: 130 tasks
- `ArgMax`: 120 tasks
- `And`: 119 tasks
- `ReduceMax`: 118 tasks
- `Conv`: 113 tasks
- `Greater`: 111 tasks
- `ReduceSum`: 107 tasks
- `Mul`: 102 tasks
- `Less`: 93 tasks
- `Div`: 71 tasks
- `Unsqueeze`: 71 tasks
- `Reshape`: 66 tasks
- `Mod`: 62 tasks
- `Or`: 60 tasks
- `Max`: 53 tasks
- `QLinearConv`: 48 tasks

## Key tags

- `exact_preserve`: 328 tasks — 003 004 005 006 007 008 009 010 011 012 013 014 015 017 018 019 020 021 022 024
- `open_angle`: 287 tasks — 001 003 004 007 009 010 011 012 013 015 017 020 022 023 024 025 027 028 029 030
- `local_stencil`: 123 tasks — 004 009 011 015 019 020 023 026 032 034 041 042 046 049 060 063 073 074 079 081
- `documented_wall`: 110 tasks — 001 002 003 005 007 009 011 018 024 044 046 048 057 064 069 070 071 076 077 080
- `connectivity_wall`: 107 tasks — 002 004 009 011 022 024 031 036 037 038 041 044 046 047 048 050 052 055 058 059
- `qlinear`: 62 tasks — 004 005 019 023 034 042 055 061 080 081 086 093 102 107 117 118 133 158 160 162
- `lut_selection`: 60 tasks — 001 002 003 004 005 007 008 009 010 011 012 013 034 037 039 041 053 055 058 061
- `scatter`: 58 tasks — 012 018 020 035 037 054 068 070 076 084 089 096 099 101 106 107 118 119 126 131
- `conv_heavy`: 56 tasks — 005 020 023 034 042 044 063 076 080 081 094 102 118 133 138 149 158 160 168 169
- `mem0_single_op`: 47 tasks — 015 016 026 032 053 060 073 082 097 098 113 114 116 120 122 128 130 135 144 151
- `onehot_final_equal`: 29 tasks — 009 017 025 029 036 064 071 101 107 170 174 175 177 182 202 208 213 218 219 222
- `custom_win`: 27 tasks — 011 017 049 056 079 081 093 105 123 145 161 183 191 196 204 222 243 280 284 290
- `high_memory`: 26 tasks — 018 025 044 054 076 080 101 118 133 138 158 173 191 198 205 209 233 255 285 286
- `assignment_wall`: 25 tasks — 044 076 089 101 133 143 157 158 163 170 182 191 201 209 219 233 240 253 263 270
- `scan`: 18 tasks — 025 046 054 055 066 090 184 188 198 209 219 233 270 319 324 366 370 382
- `gather_heavy`: 17 tasks — 018 054 101 132 133 157 158 173 198 233 273 285 319 342 365 366 368
- `matmul`: 14 tasks — 039 044 050 104 105 107 118 124 174 233 235 255 381 398
- `bitwise_program`: 11 tasks — 041 047 048 058 065 068 132 139 156 270 305
- `low_score`: 8 tasks — 018 118 133 209 233 285 286 366
- `maxpool_scan`: 7 tasks — 125 196 204 209 222 240 277
- `marginal_wall`: 5 tasks — 029 158 181 350 361
- `ambiguity_wall`: 4 tasks — 018 077 209 329
- `heuristic`: 1 tasks — 046
- `infeasible_exact_wall`: 1 tasks — 118
- `information_loss_wall`: 1 tasks — 118
- `marker_routed_path`: 1 tasks — 066
- `template_match`: 1 tasks — 182
