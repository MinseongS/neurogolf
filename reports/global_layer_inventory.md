# Global layer inventory summary

- tasks indexed: 400
- failures: 0
- source-controlled build(): 400/400
- no build(): none

## Source classes

- `exact_preserve`: 345
- `semantic_or_handbuilt`: 49
- `unknown`: 6

## Largest op families

- `Cast`: 292 tasks
- `Pad`: 218 tasks
- `Equal`: 212 tasks
- `Slice`: 211 tasks
- `Where`: 193 tasks
- `Concat`: 175 tasks
- `Einsum`: 158 tasks
- `Add`: 154 tasks
- `Gather`: 139 tasks
- `Sub`: 133 tasks
- `ArgMax`: 116 tasks
- `ReduceMax`: 110 tasks
- `And`: 108 tasks
- `Conv`: 108 tasks
- `Greater`: 106 tasks
- `Mul`: 102 tasks
- `ReduceSum`: 98 tasks
- `Less`: 91 tasks
- `Div`: 74 tasks
- `Reshape`: 71 tasks
- `Unsqueeze`: 64 tasks
- `Mod`: 58 tasks
- `Or`: 56 tasks
- `Max`: 52 tasks
- `TopK`: 45 tasks

## Key tags

- `exact_preserve`: 345 tasks — 002 003 004 005 006 007 008 009 010 011 012 013 014 015 017 018 019 020 021 022
- `open_angle`: 287 tasks — 001 003 004 007 009 010 011 012 013 015 017 020 022 023 024 025 027 028 029 030
- `local_stencil`: 120 tasks — 004 009 011 015 019 020 023 026 032 034 041 042 046 049 060 073 074 079 080 081
- `documented_wall`: 115 tasks — 001 002 003 004 005 007 009 011 018 024 044 046 048 057 064 069 070 071 076 077
- `connectivity_wall`: 107 tasks — 002 004 009 011 022 024 031 036 037 038 041 044 046 047 048 050 052 055 058 059
- `lut_selection`: 65 tasks — 001 002 003 004 005 007 008 009 010 011 012 013 034 037 041 053 055 058 061 080
- `qlinear`: 65 tasks — 001 004 005 019 023 034 042 055 061 080 081 086 093 101 102 107 117 118 133 156
- `scatter`: 56 tasks — 012 018 020 035 037 054 068 070 076 084 089 096 099 101 106 107 119 126 131 133
- `conv_heavy`: 54 tasks — 005 020 023 034 042 080 081 094 102 118 133 138 149 158 160 165 168 169 170 171
- `mem0_single_op`: 45 tasks — 015 016 026 032 053 060 073 082 097 098 113 114 116 120 122 130 135 144 151 164
- `assignment_wall`: 28 tasks — 001 044 076 089 101 133 143 157 158 163 170 182 191 201 209 219 233 240 253 263
- `custom_win`: 27 tasks — 011 017 049 056 079 081 093 105 123 145 161 183 191 196 204 222 243 280 284 290
- `onehot_final_equal`: 25 tasks — 009 017 025 029 036 064 071 101 107 170 175 177 184 202 213 218 219 234 267 290
- `high_memory`: 20 tasks — 018 025 054 076 101 118 133 138 158 173 191 198 233 285 286 319 349 364 366 367
- `scan`: 18 tasks — 025 044 046 054 055 066 090 174 188 198 219 233 270 319 324 338 366 382
- `gather_heavy`: 15 tasks — 018 054 101 132 157 158 173 198 233 273 285 319 342 365 366
- `bitwise_program`: 11 tasks — 041 047 048 058 065 068 132 139 156 270 305
- `matmul`: 10 tasks — 017 105 107 124 174 233 235 255 381 398
- `maxpool_scan`: 7 tasks — 018 125 196 204 222 240 277
- `marginal_wall`: 6 tasks — 001 029 158 181 350 361
- `low_score`: 5 tasks — 018 233 285 286 366
- `ambiguity_wall`: 4 tasks — 018 077 209 329
- `infeasible_exact_wall`: 1 tasks — 118
- `information_loss_wall`: 1 tasks — 118
- `marker_routed_path`: 1 tasks — 066
