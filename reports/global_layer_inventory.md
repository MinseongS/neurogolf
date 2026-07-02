# Global layer inventory summary

- tasks indexed: 400
- failures: 0
- source-controlled build(): 400/400
- no build(): none

## Source classes

- `exact_preserve`: 343
- `semantic_or_handbuilt`: 48
- `unknown`: 7
- `heuristic`: 2

## Largest op families

- `Cast`: 297 tasks
- `Slice`: 244 tasks
- `Pad`: 227 tasks
- `Equal`: 218 tasks
- `Where`: 204 tasks
- `Concat`: 180 tasks
- `Add`: 154 tasks
- `Gather`: 152 tasks
- `ReduceMax`: 133 tasks
- `Sub`: 131 tasks
- `And`: 125 tasks
- `ArgMax`: 123 tasks
- `Conv`: 122 tasks
- `Greater`: 109 tasks
- `ReduceSum`: 108 tasks
- `Mul`: 101 tasks
- `Einsum`: 96 tasks
- `Less`: 95 tasks
- `Unsqueeze`: 71 tasks
- `Reshape`: 69 tasks
- `Div`: 68 tasks
- `Mod`: 63 tasks
- `Or`: 62 tasks
- `Not`: 50 tasks
- `Max`: 49 tasks

## Key tags

- `exact_preserve`: 343 tasks — 002 003 005 006 007 008 009 010 011 012 013 014 015 018 019 020 021 022 023 024
- `open_angle`: 287 tasks — 001 003 004 007 009 010 011 012 013 015 017 020 022 023 024 025 027 028 029 030
- `local_stencil`: 128 tasks — 004 009 011 015 019 020 026 032 034 041 042 046 049 060 063 073 074 077 079 081
- `documented_wall`: 109 tasks — 001 002 003 005 007 009 011 018 024 044 046 048 057 064 069 070 071 076 077 080
- `connectivity_wall`: 107 tasks — 002 004 009 011 022 024 031 036 037 038 041 044 046 047 048 050 052 055 058 059
- `scatter`: 61 tasks — 012 018 020 035 037 054 068 070 076 083 084 089 092 096 099 101 106 107 118 119
- `qlinear`: 59 tasks — 004 005 019 023 034 042 055 061 077 080 081 086 093 102 107 117 118 133 158 160
- `conv_heavy`: 58 tasks — 004 005 020 023 034 042 044 063 080 081 094 102 110 118 133 138 149 158 160 168
- `lut_selection`: 55 tasks — 001 002 003 005 007 008 009 010 011 012 013 034 037 039 053 055 058 061 080 091
- `mem0_single_op`: 46 tasks — 015 016 026 032 053 060 073 082 097 098 113 114 116 120 122 128 130 135 144 151
- `high_memory`: 39 tasks — 002 018 023 025 029 044 054 064 066 076 077 080 101 110 118 133 138 145 158 173
- `onehot_final_equal`: 29 tasks — 009 017 025 029 036 064 071 107 170 175 177 184 193 202 208 213 218 222 234 264
- `custom_win`: 27 tasks — 011 017 049 056 079 081 093 105 123 145 161 183 191 196 204 222 243 280 284 290
- `assignment_wall`: 23 tasks — 044 076 089 133 143 157 158 163 170 182 191 201 209 219 233 240 253 263 270 319
- `gather_heavy`: 21 tasks — 002 018 054 101 132 133 157 158 173 198 219 233 234 273 285 286 319 342 365 366
- `matmul`: 17 tasks — 002 039 044 050 104 105 107 118 124 174 191 216 233 235 255 381 398
- `scan`: 17 tasks — 025 046 054 055 066 090 184 198 209 219 233 270 319 324 366 370 382
- `low_score`: 16 tasks — 002 018 054 076 118 133 158 187 191 209 233 243 285 286 364 366
- `maxpool_scan`: 14 tasks — 018 077 110 125 145 187 196 204 222 240 251 277 349 364
- `bitwise_program`: 13 tasks — 002 041 047 048 058 065 068 132 139 156 270 286 305
- `heuristic`: 5 tasks — 046 077 118 133 216
- `marginal_wall`: 5 tasks — 029 158 181 350 361
- `ambiguity_wall`: 4 tasks — 018 077 209 329
- `infeasible_exact_wall`: 1 tasks — 118
- `information_loss_wall`: 1 tasks — 118
- `marker_routed_path`: 1 tasks — 066
- `template_match`: 1 tasks — 367
