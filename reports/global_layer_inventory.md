# Global layer inventory summary

- tasks indexed: 400
- failures: 0
- source-controlled build(): 396/400
- no build(): 140 233 241 311

## Source classes

- `semantic_or_handbuilt`: 180
- `unknown`: 179
- `exact_preserve`: 35
- `no_build`: 4
- `heuristic`: 2

## Largest op families

- `Cast`: 296 tasks
- `Slice`: 245 tasks
- `Pad`: 229 tasks
- `Equal`: 217 tasks
- `Where`: 207 tasks
- `Concat`: 183 tasks
- `Add`: 153 tasks
- `Gather`: 150 tasks
- `ReduceMax`: 134 tasks
- `Sub`: 134 tasks
- `And`: 126 tasks
- `Conv`: 126 tasks
- `ArgMax`: 125 tasks
- `ReduceSum`: 110 tasks
- `Greater`: 108 tasks
- `Mul`: 99 tasks
- `Less`: 95 tasks
- `Einsum`: 90 tasks
- `Unsqueeze`: 72 tasks
- `Reshape`: 68 tasks
- `Div`: 66 tasks
- `Mod`: 64 tasks
- `Or`: 64 tasks
- `Max`: 53 tasks
- `Not`: 49 tasks

## Key tags

- `open_angle`: 294 tasks — 001 003 004 007 009 010 011 012 013 014 015 017 020 022 024 025 027 028 029 030
- `local_stencil`: 125 tasks — 004 009 015 019 020 023 026 032 034 041 042 046 049 060 063 073 074 077 079 081
- `documented_wall`: 112 tasks — 001 002 003 007 009 011 018 024 037 044 046 048 057 064 069 070 071 076 077 080
- `connectivity_wall`: 105 tasks — 002 004 009 011 022 024 031 036 037 038 041 044 046 047 048 050 052 055 058 059
- `scatter`: 61 tasks — 012 018 020 035 037 054 068 070 076 083 084 089 092 096 099 101 106 107 118 119
- `conv_heavy`: 55 tasks — 005 020 023 034 042 044 063 080 081 094 102 110 118 133 138 149 158 160 168 169
- `mem0_single_op`: 47 tasks — 015 016 026 032 053 060 073 082 097 098 113 114 116 120 122 128 130 135 144 151
- `lut_selection`: 44 tasks — 003 009 020 034 047 053 055 058 061 080 094 104 107 117 124 140 145 157 168 169
- `qlinear`: 43 tasks — 005 034 055 061 077 080 081 086 093 102 117 118 133 158 160 162 168 169 188 190
- `high_memory`: 42 tasks — 002 017 018 023 025 029 044 054 064 066 076 077 080 101 110 118 133 138 145 157
- `exact_preserve`: 35 tasks — 002 005 015 018 044 054 066 076 083 098 101 118 120 122 127 133 135 151 157 171
- `onehot_final_equal`: 32 tasks — 009 017 048 051 055 056 064 069 086 100 105 107 141 143 165 203 208 209 222 239
- `custom_win`: 27 tasks — 011 017 049 056 079 081 093 105 123 145 161 183 191 196 204 222 243 280 284 290
- `assignment_wall`: 26 tasks — 044 076 086 089 133 143 157 158 163 170 182 191 201 209 219 240 253 263 264 270
- `low_score`: 23 tasks — 002 018 023 054 076 118 133 158 173 187 191 209 219 233 243 255 285 286 319 349
- `heuristic`: 22 tasks — 002 005 018 044 046 054 066 076 077 101 118 133 157 173 209 216 219 255 285 286
- `gather_heavy`: 20 tasks — 018 054 101 132 133 157 158 173 198 219 233 234 273 285 286 319 342 365 366 368
- `scan`: 18 tasks — 025 046 054 055 066 090 184 198 209 219 233 270 319 324 349 366 370 382
- `matmul`: 17 tasks — 039 044 050 066 104 105 107 118 124 174 191 216 233 235 255 381 398
- `maxpool_scan`: 14 tasks — 002 018 077 110 125 145 187 196 204 222 240 251 277 364
- `bitwise_program`: 12 tasks — 041 047 048 058 065 068 132 139 156 270 286 305
- `template_match`: 8 tasks — 074 158 191 251 287 363 364 400
- `marginal_wall`: 5 tasks — 029 158 181 350 361
- `no_build`: 4 tasks — 140 233 241 311
- `infeasible_exact_wall`: 1 tasks — 118
- `information_loss_wall`: 1 tasks — 118
