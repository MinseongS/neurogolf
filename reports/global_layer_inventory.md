# Global layer inventory summary

- tasks indexed: 400
- failures: 0
- source-controlled build(): 400/400
- no build(): none

## Source classes

- `semantic_or_handbuilt`: 183
- `unknown`: 179
- `exact_preserve`: 36
- `heuristic`: 2

## Largest op families

- `Cast`: 296 tasks
- `Slice`: 246 tasks
- `Pad`: 230 tasks
- `Equal`: 219 tasks
- `Where`: 205 tasks
- `Concat`: 182 tasks
- `Add`: 154 tasks
- `Gather`: 151 tasks
- `ReduceMax`: 135 tasks
- `Sub`: 131 tasks
- `And`: 129 tasks
- `Conv`: 125 tasks
- `ArgMax`: 124 tasks
- `Greater`: 110 tasks
- `ReduceSum`: 108 tasks
- `Mul`: 100 tasks
- `Less`: 95 tasks
- `Einsum`: 89 tasks
- `Unsqueeze`: 72 tasks
- `Reshape`: 68 tasks
- `Div`: 67 tasks
- `Or`: 64 tasks
- `Mod`: 63 tasks
- `Not`: 51 tasks
- `Max`: 50 tasks

## Key tags

- `open_angle`: 294 tasks — 001 003 004 007 009 010 011 012 013 014 015 017 020 022 024 025 027 028 029 030
- `local_stencil`: 125 tasks — 004 009 015 019 020 026 032 034 041 042 046 049 060 063 073 074 077 079 081 082
- `documented_wall`: 112 tasks — 001 002 003 007 009 011 018 024 037 044 046 048 057 064 069 070 071 076 077 080
- `connectivity_wall`: 106 tasks — 002 004 009 011 022 024 031 036 037 038 041 044 046 047 048 050 052 055 058 059
- `scatter`: 61 tasks — 012 018 020 035 037 054 068 070 076 083 084 089 092 096 099 101 106 107 118 119
- `conv_heavy`: 56 tasks — 005 020 023 034 042 044 063 080 081 094 102 110 118 133 138 149 158 160 168 169
- `qlinear`: 48 tasks — 005 023 034 055 061 077 080 081 086 093 102 117 118 133 158 160 162 168 169 182
- `mem0_single_op`: 47 tasks — 015 016 026 032 053 060 073 082 097 098 113 114 116 120 122 128 130 135 144 151
- `lut_selection`: 44 tasks — 003 009 020 034 047 053 055 058 061 080 094 104 107 117 124 140 145 157 168 169
- `high_memory`: 40 tasks — 002 018 023 025 029 044 054 064 066 076 077 080 101 110 118 133 138 145 158 173
- `exact_preserve`: 36 tasks — 002 005 015 018 044 054 066 076 083 098 101 118 120 122 127 133 135 151 157 171
- `onehot_final_equal`: 33 tasks — 009 017 048 051 055 056 064 069 086 100 105 107 141 143 165 202 203 208 209 222
- `custom_win`: 27 tasks — 011 017 049 056 079 081 093 105 123 145 161 183 191 196 204 222 243 280 284 290
- `assignment_wall`: 26 tasks — 044 076 086 089 133 143 157 158 163 170 182 191 201 209 219 240 253 263 264 270
- `heuristic`: 22 tasks — 002 005 018 044 046 054 066 076 077 101 118 133 157 173 209 216 219 255 285 286
- `gather_heavy`: 21 tasks — 002 018 054 101 132 133 157 158 173 198 219 233 234 273 285 286 319 342 365 366
- `low_score`: 21 tasks — 002 018 054 076 118 133 158 173 187 191 209 233 243 255 285 286 319 349 364 366
- `matmul`: 18 tasks — 002 039 044 050 066 104 105 107 118 124 174 191 216 233 235 255 381 398
- `scan`: 17 tasks — 025 046 054 055 066 090 184 198 209 219 233 270 319 324 366 370 382
- `maxpool_scan`: 14 tasks — 018 077 110 125 145 187 196 204 222 240 251 277 349 364
- `bitwise_program`: 13 tasks — 002 041 047 048 058 065 068 132 139 156 270 286 305
- `template_match`: 8 tasks — 074 158 191 251 287 363 364 400
- `marginal_wall`: 5 tasks — 029 158 181 350 361
- `ambiguity_wall`: 4 tasks — 018 077 209 329
- `infeasible_exact_wall`: 1 tasks — 118
- `information_loss_wall`: 1 tasks — 118
