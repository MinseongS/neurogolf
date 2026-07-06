# Global layer inventory summary

- tasks indexed: 400
- failures: 0
- source-controlled build(): 400/400
- no build(): none

## Source classes

- `exact_preserve`: 333
- `semantic_or_handbuilt`: 59
- `unknown`: 8

## Largest op families

- `Cast`: 291 tasks
- `Slice`: 220 tasks
- `Pad`: 219 tasks
- `Equal`: 213 tasks
- `Where`: 195 tasks
- `Concat`: 171 tasks
- `Add`: 152 tasks
- `Einsum`: 150 tasks
- `Gather`: 139 tasks
- `Sub`: 134 tasks
- `And`: 118 tasks
- `ArgMax`: 116 tasks
- `ReduceMax`: 115 tasks
- `Conv`: 111 tasks
- `Greater`: 106 tasks
- `Mul`: 99 tasks
- `ReduceSum`: 99 tasks
- `Less`: 91 tasks
- `Div`: 72 tasks
- `Reshape`: 69 tasks
- `Unsqueeze`: 62 tasks
- `Mod`: 61 tasks
- `Or`: 61 tasks
- `Max`: 50 tasks
- `QLinearConv`: 46 tasks

## Key tags

- `exact_preserve`: 333 tasks — 003 004 005 006 007 008 009 010 011 012 013 014 015 017 018 019 020 021 022 024
- `open_angle`: 287 tasks — 001 003 004 007 009 010 011 012 013 015 017 020 022 023 024 025 027 028 029 030
- `local_stencil`: 119 tasks — 004 009 011 015 019 020 023 026 032 034 041 042 046 049 060 073 074 079 081 082
- `documented_wall`: 112 tasks — 001 002 003 004 005 007 009 011 018 024 044 046 048 057 064 069 070 071 076 077
- `connectivity_wall`: 107 tasks — 002 004 009 011 022 024 031 036 037 038 041 044 046 047 048 050 052 055 058 059
- `qlinear`: 63 tasks — 001 004 005 019 023 034 042 055 061 080 081 086 093 102 107 117 118 133 156 158
- `lut_selection`: 60 tasks — 001 002 003 004 005 007 008 009 010 011 012 013 034 037 041 053 055 058 061 080
- `scatter`: 55 tasks — 012 018 020 035 037 054 068 070 076 084 089 096 099 101 106 107 119 126 131 133
- `conv_heavy`: 50 tasks — 005 020 023 034 042 080 081 094 102 118 133 138 149 158 160 168 169 170 177 178
- `mem0_single_op`: 46 tasks — 015 016 026 032 053 060 073 082 097 098 113 114 116 120 122 130 135 144 151 164
- `onehot_final_equal`: 29 tasks — 009 017 025 029 036 064 071 101 107 170 175 177 182 184 202 208 213 218 219 234
- `assignment_wall`: 28 tasks — 001 044 076 089 101 133 143 157 158 163 170 182 191 201 209 219 233 240 253 263
- `custom_win`: 27 tasks — 011 017 049 056 079 081 093 105 123 145 161 183 191 196 204 222 243 280 284 290
- `high_memory`: 25 tasks — 018 025 054 076 080 101 118 133 138 158 173 191 198 205 209 233 255 285 286 319
- `scan`: 19 tasks — 025 044 046 054 055 066 090 174 184 188 198 209 219 233 270 319 324 366 382
- `gather_heavy`: 17 tasks — 018 054 101 132 133 157 158 173 198 233 273 285 319 342 365 366 368
- `bitwise_program`: 11 tasks — 041 047 048 058 065 068 132 139 156 270 305
- `matmul`: 11 tasks — 017 050 105 107 124 174 233 235 255 381 398
- `maxpool_scan`: 8 tasks — 018 125 196 204 209 222 240 277
- `low_score`: 7 tasks — 018 133 209 233 285 286 366
- `marginal_wall`: 6 tasks — 001 029 158 181 350 361
- `ambiguity_wall`: 4 tasks — 018 077 209 329
- `infeasible_exact_wall`: 1 tasks — 118
- `information_loss_wall`: 1 tasks — 118
- `marker_routed_path`: 1 tasks — 066
- `template_match`: 1 tasks — 182
