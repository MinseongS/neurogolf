# Frontier Mechanism Audit

Compares current high-score/tiny-cost active ONNX mechanisms against high-cost active tasks.

## Frontier Tasks

| task | pts | mem | params | cost | nodes | output | tags | top ops |
|---:|---:|---:|---:|---:|---:|---|---|---|
| 067 | 25.000 | 0 | 0 | 0 | 1 | Einsum | mem0, frontier_memory, einsum_direct_output | Einsum:1 |
| 179 | 25.000 | 0 | 0 | 0 | 1 | Transpose | mem0, frontier_memory | Transpose:1 |
| 241 | 25.000 | 0 | 0 | 0 | 1 | Transpose | mem0, frontier_memory | Transpose:1 |
| 087 | 23.391 | 0 | 5 | 5 | 1 | RoiAlign | mem0, frontier_memory | RoiAlign:1 |
| 140 | 23.391 | 0 | 5 | 5 | 1 | RoiAlign | mem0, frontier_memory | RoiAlign:1 |
| 223 | 23.391 | 0 | 5 | 5 | 1 | MaxRoiPool | mem0, frontier_memory | MaxRoiPool:1 |
| 307 | 23.391 | 0 | 5 | 5 | 1 | MaxRoiPool | mem0, frontier_memory | MaxRoiPool:1 |
| 016 | 22.697 | 0 | 10 | 10 | 1 | Gather | mem0, frontier_memory, direct_gather | Gather:1 |
| 276 | 22.697 | 0 | 10 | 10 | 1 | Gather | mem0, frontier_memory, direct_gather | Gather:1 |
| 309 | 22.697 | 0 | 10 | 10 | 1 | Gather | mem0, frontier_memory, direct_gather | Gather:1 |
| 337 | 22.697 | 0 | 10 | 10 | 1 | Gather | mem0, frontier_memory, direct_gather | Gather:1 |
| 312 | 22.004 | 0 | 20 | 20 | 1 | Einsum | mem0, frontier_memory, einsum_direct_output | Einsum:1 |
| 166 | 21.822 | 0 | 24 | 24 | 1 | Einsum | mem0, frontier_memory, einsum_direct_output | Einsum:1 |
| 053 | 21.599 | 0 | 30 | 30 | 1 | Gather | mem0, frontier_memory, direct_gather | Gather:1 |
| 113 | 21.599 | 0 | 30 | 30 | 1 | Gather | mem0, frontier_memory, direct_gather | Gather:1 |
| 116 | 21.599 | 0 | 30 | 30 | 1 | Gather | mem0, frontier_memory, direct_gather | Gather:1 |
| 130 | 21.599 | 0 | 30 | 30 | 1 | Conv | mem0, frontier_memory, single_op_stencil | Conv:1 |
| 164 | 21.599 | 0 | 30 | 30 | 1 | Gather | mem0, frontier_memory, direct_gather | Gather:1 |
| 172 | 21.599 | 0 | 30 | 30 | 1 | Gather | mem0, frontier_memory, direct_gather | Gather:1 |
| 210 | 21.599 | 0 | 30 | 30 | 1 | Gather | mem0, frontier_memory, direct_gather | Gather:1 |
| 311 | 21.599 | 0 | 30 | 30 | 1 | Gather | mem0, frontier_memory, direct_gather | Gather:1 |
| 326 | 21.599 | 0 | 30 | 30 | 1 | Einsum | mem0, frontier_memory, einsum_direct_output | Einsum:1 |
| 385 | 21.599 | 0 | 30 | 30 | 1 | Gather | mem0, frontier_memory, direct_gather | Gather:1 |
| 056 | 21.474 | 34 | 0 | 34 | 13 | Pad | frontier_memory, final_equal | And:3, Cast:2, Greater:2, Slice:2, Concat:1 |
| 073 | 21.311 | 0 | 40 | 40 | 1 | Conv | mem0, frontier_memory, single_op_stencil | Conv:1 |
| 299 | 21.088 | 0 | 50 | 50 | 1 | Einsum | mem0, frontier_memory, einsum_direct_output | Einsum:1 |
| 103 | 20.906 | 36 | 24 | 60 | 7 | Pad | frontier_memory, final_equal | Slice:2, Concat:1, Equal:1, Not:1, Pad:1 |
| 373 | 20.906 | 0 | 60 | 60 | 1 | Einsum | mem0, frontier_memory, einsum_direct_output | Einsum:1 |
| 399 | 20.906 | 31 | 29 | 60 | 4 | Pad | frontier_memory, final_equal | Einsum:1, Equal:1, Greater:1, Pad:1 |
| 291 | 20.780 | 40 | 28 | 68 | 2 | Pad | frontier_memory | Einsum:1, Pad:1 |
| 129 | 20.618 | 80 | 0 | 80 | 3 | Einsum | frontier_memory | Einsum:1, GlobalAveragePool:1, Hardmax:1 |
| 292 | 20.618 | 0 | 80 | 80 | 1 | Einsum | mem0, frontier_memory, einsum_direct_output | Einsum:1 |
| 142 | 20.500 | 0 | 90 | 90 | 1 | Einsum | mem0, frontier_memory, einsum_direct_output | Einsum:1 |
| 152 | 20.500 | 0 | 90 | 90 | 1 | Einsum | mem0, frontier_memory, einsum_direct_output | Einsum:1 |
| 380 | 20.405 | 0 | 99 | 99 | 1 | Einsum | mem0, frontier_memory, einsum_direct_output | Einsum:1 |
| 227 | 20.395 | 0 | 100 | 100 | 1 | Conv | mem0, frontier_memory, single_op_stencil | Conv:1 |
| 314 | 20.395 | 0 | 100 | 100 | 1 | Conv | mem0, frontier_memory, single_op_stencil | Conv:1 |
| 318 | 20.395 | 0 | 100 | 100 | 1 | Conv | mem0, frontier_memory, single_op_stencil | Conv:1 |
| 186 | 20.365 | 80 | 23 | 103 | 5 | Pad | frontier_memory | GlobalLpPool:1, Less:1, Pad:1, ReduceMax:1, Where:1 |
| 144 | 20.356 | 0 | 104 | 104 | 1 | Conv | mem0, frontier_memory, single_op_stencil | Conv:1 |
| 262 | 20.337 | 90 | 16 | 106 | 6 | Pad | frontier_memory | Concat:2, Cast:1, Pad:1, Slice:1, Split:1 |
| 024 | 20.300 | 0 | 110 | 110 | 1 | Einsum | mem0, frontier_memory, einsum_direct_output | Einsum:1 |
| 236 | 20.164 | 96 | 30 | 126 | 4 | ConvInteger | frontier_memory | Cast:1, ConvInteger:1, ConvTranspose:1, Shrink:1 |
| 007 | 20.156 | 0 | 127 | 127 | 1 | Einsum | mem0, frontier_memory, einsum_direct_output | Einsum:1 |
| 389 | 20.132 | 108 | 22 | 130 | 8 | Gather | frontier_memory, scatter | Concat:2, Cast:1, Einsum:1, Gather:1, ReduceMax:1 |
| 298 | 20.095 | 120 | 15 | 135 | 2 | Einsum | frontier_memory | Einsum:1, Slice:1 |
| 128 | 20.087 | 0 | 136 | 136 | 1 | Einsum | mem0, frontier_memory, einsum_direct_output | Einsum:1 |
| 149 | 20.087 | 36 | 100 | 136 | 2 | Conv | frontier_memory | Conv:2 |
| 150 | 20.073 | 136 | 2 | 138 | 6 | Gather | frontier_memory | Add:1, Cast:1, Gather:1, Range:1, ReduceL2:1 |
| 339 | 20.058 | 40 | 100 | 140 | 2 | ConvTranspose | frontier_memory | ConvTranspose:1, GlobalAveragePool:1 |

## High-Cost Tasks

| task | pts | mem | params | cost | nodes | output | tags | largest counted tensors |
|---:|---:|---:|---:|---:|---:|---|---|---|
| 233 | 14.605 | 32243 | 445 | 32688 | 256 | Equal | scatter, topk | 3600B float32 [1, 1, 30, 30] Conv; 3240B float16 [5, 324] Cast; 3136B float32 [1, 1, 28, 28] Conv; 1620B bool [5, 324] Equal |
| 366 | 14.640 | 30983 | 576 | 31559 | 597 | Equal | scatter, topk | 3600B float32 [1, 1, 30, 30] Conv; 1020B int32 [1, 255] Cast; 900B uint8 [1, 1, 30, 30] Cast; 900B uint8 [1, 1, 30, 30] Pad |
| 018 | 14.898 | 23688 | 696 | 24384 | 675 | Equal | scatter, topk | 2304B float32 [1, 1, 24, 24] Conv; 1152B float16 [576] Where; 1152B float16 [576] Where; 900B uint8 [30, 30] Pad |
| 133 | 15.025 | 20474 | 1016 | 21490 | 154 | Equal | scatter | 3600B float32 [1, 1, 30, 30] Conv; 1200B float32 [1, 10, 30, 1] ReduceMax; 1200B float32 [1, 10, 1, 30] ReduceMax; 900B uint8 [1, 1, 30, 30] Cast |
| 158 | 15.063 | 18041 | 2646 | 20687 | 109 | Equal |  | 2600B float32 [1, 1, 26, 25] Conv; 2208B uint8 [1, 4, 24, 23] QLinearConv; 1680B uint8 [1, 4, 21, 20] QLinearConv; 1300B bool [1, 2, 26, 25] Equal |
| 054 | 15.087 | 19904 | 280 | 20184 | 262 | Equal | scatter | 3600B float32 [1, 1, 30, 30] Conv; 1024B int64 [4, 8, 4] Concat; 900B uint8 [1, 1, 30, 30] Cast; 900B uint8 [1, 1, 30, 30] ScatterND |
| 285 | 15.109 | 19323 | 420 | 19743 | 111 | Equal | scatter, topk | 3600B float32 [1, 1, 30, 30] Conv; 1800B float16 [900] Cast; 1024B uint8 [1024] Pad; 992B int32 [8, 31] Add |
| 286 | 15.149 | 16090 | 2881 | 18971 | 21 | Equal |  | 2500B float32 [1, 1, 25, 25] Conv; 2500B float32 [1, 25, 25] Einsum; 2500B float32 [1, 25, 25] Einsum; 2500B float32 [1, 25, 25] Einsum |
| 364 | 15.260 | 15860 | 1131 | 16991 | 17 | Equal |  | 3520B float32 [1, 2, 20, 22] Slice; 1760B int32 [1, 1, 20, 22] Cast; 1760B float16 [1, 2, 20, 22] Cast; 900B uint8 [1, 1, 30, 30] Pad |
| 367 | 15.327 | 15300 | 590 | 15890 | 21 | Where | free_input_overlay | 4000B uint8 [1, 10, 20, 20] QLinearConv; 3200B float32 [1, 2, 20, 20] Slice; 900B bool [1, 1, 30, 30] Pad; 800B uint8 [1, 2, 20, 20] Cast |
| 349 | 15.381 | 13860 | 1182 | 15042 | 15 | Equal |  | 4500B uint8 [1, 5, 30, 30] QLinearConv; 3600B float32 [1, 1, 30, 30] Gather; 900B uint8 [1, 1, 30, 30] Cast; 900B uint8 [1, 1, 30, 30] QLinearConv |
| 101 | 15.471 | 12928 | 822 | 13750 | 283 | Where | free_input_overlay, scatter, topk | 1360B float32 [1, 1, 17, 20] Slice; 1360B float32 [1, 1, 17, 20] Slice; 900B bool [1, 1, 30, 30] Pad; 680B float16 [340] Where |
| 076 | 15.514 | 13080 | 89 | 13169 | 123 | Equal | scatter, topk | 900B float32 [1, 1, 15, 15] Conv; 900B uint8 [1, 1, 30, 30] Pad; 504B int32 [3, 7, 6] Cast; 450B float16 [225] Cast |
| 138 | 15.552 | 12215 | 462 | 12677 | 55 | Equal |  | 2600B float32 [1, 1, 26, 25] Conv; 900B uint8 [1, 1, 30, 30] Pad; 729B uint8 [1, 1, 27, 27] Pad; 650B uint8 [1, 1, 26, 25] Cast |
| 118 | 15.584 | 12153 | 130 | 12283 | 20 | Where | free_input_overlay | 2800B float32 [1, 1, 25, 28] Conv; 900B bool [1, 1, 30, 30] Pad; 700B uint8 [1, 1, 25, 28] Cast; 700B uint8 [1, 1, 25, 28] QLinearConv |
| 173 | 15.635 | 11232 | 438 | 11670 | 101 | Equal | scatter, topk | 2500B float32 [1, 1, 25, 25] Conv; 1250B float16 [625] Cast; 900B uint8 [1, 1, 30, 30] Pad; 625B uint8 [1, 1, 25, 25] Cast |
| 198 | 15.658 | 11305 | 107 | 11412 | 51 | Equal |  | 3600B float32 [1, 1, 30, 30] Conv; 900B uint8 [1, 1, 30, 30] Cast; 900B bool [1, 1, 30, 30] Equal; 900B uint8 [1, 1, 30, 30] Gather |
| 191 | 15.686 | 11009 | 81 | 11090 | 64 | Equal |  | 3174B uint8 [1, 6, 23, 23] QLinearConv; 2116B float32 [1, 1, 23, 23] Slice; 900B uint8 [1, 1, 30, 30] Pad; 625B uint8 [1, 1, 25, 25] Pad |
| 145 | 15.742 | 9244 | 1248 | 10492 | 23 | Equal |  | 1600B float32 [1, 1, 20, 20] Slice; 900B uint8 [1, 1, 30, 30] Pad; 800B float16 [1, 1, 20, 20] Cast; 800B float16 [1, 1, 20, 20] Einsum |
| 025 | 15.766 | 10132 | 104 | 10236 | 60 | Einsum | topk | 780B float16 [1, 13, 30] Concat; 780B float16 [1, 13, 30] Concat; 480B float32 [1, 4, 30] OneHot; 480B float32 [1, 4, 30] CumSum |
| 204 | 15.767 | 9780 | 452 | 10232 | 30 | QLinearConv |  | 1600B float32 [1, 1, 20, 20] Slice; 1600B uint8 [1, 4, 20, 20] Concat; 400B uint8 [1, 1, 20, 20] Cast; 400B uint8 [1, 1, 20, 20] Pad |
| 066 | 15.778 | 9955 | 166 | 10121 | 220 | Where | free_input_overlay | 900B bool [1, 30, 30] Pad; 800B float16 [1, 20, 20] Einsum; 480B float32 [1, 4, 30] Cast; 480B float32 [1, 4, 30] Einsum |
| 064 | 15.791 | 9852 | 133 | 9985 | 80 | Where | free_input_overlay, topk | 1200B float32 [1, 10, 30, 1] ReduceMax; 1200B float32 [1, 10, 1, 30] ReduceMax; 900B bool [1, 1, 30, 30] Pad; 576B uint8 [1, 1, 24, 24] Sub |
| 080 | 15.813 | 9327 | 446 | 9773 | 57 | Equal |  | 3600B float32 [1, 1, 30, 30] Conv; 900B uint8 [1, 1, 30, 30] Cast; 900B uint8 [1, 1, 30, 30] Gather; 360B uint8 [1, 1, 30, 12] Gather |
| 379 | 15.885 | 7840 | 1255 | 9095 | 91 | Equal |  | 900B uint8 [1, 1, 30, 30] Pad; 400B bool [1, 1, 20, 20] And; 400B bool [1, 1, 20, 20] And; 400B bool [1, 1, 20, 20] GreaterOrEqual |
| 074 | 15.889 | 9000 | 50 | 9050 | 8 | Equal |  | 3600B float32 [1, 1, 30, 30] Conv; 900B uint8 [1, 1, 30, 30] Cast; 900B uint8 [1, 1, 30, 30] Transpose; 900B uint8 [1, 1, 30, 30] Gather |
| 350 | 15.891 | 9012 | 24 | 9036 | 12 | Where | free_input_overlay | 2496B float32 [1, 1, 26, 24] Slice; 900B bool [1, 1, 30, 30] Pad; 624B uint8 [1, 1, 26, 24] Cast; 624B uint8 [1, 1, 26, 24] MaxPool |
| 324 | 15.910 | 7308 | 1556 | 8864 | 45 | Equal | topk | 1600B float32 [1, 1, 20, 20] Conv; 900B uint8 [1, 1, 30, 30] Pad; 400B uint8 [1, 1, 20, 20] Cast; 400B bool [1, 1, 20, 20] Equal |
| 338 | 15.912 | 8581 | 269 | 8850 | 27 | Equal |  | 2208B float32 [1, 1, 23, 24] Conv; 1104B float16 [1, 1, 23, 24] Cast; 1104B float16 [1, 1, 23, 24] CumSum; 900B uint8 [1, 1, 30, 30] Pad |
| 216 | 15.934 | 8584 | 76 | 8660 | 76 | Pad | scatter | 3200B float32 [1, 2, 20, 20] Slice; 800B uint8 [1, 2, 20, 20] Cast; 648B uint8 [1, 2, 18, 18] Slice; 400B uint8 [1, 1, 20, 20] QLinearConv |
| 187 | 15.973 | 5000 | 3322 | 8322 | 2 | Einsum |  | 5000B float32 [1, 2, 25, 25] Conv |
| 370 | 16.000 | 7331 | 773 | 8104 | 76 | Where | free_input_overlay, scatter | 900B bool [1, 1, 30, 30] Pad; 400B uint8 [1, 1, 20, 20] Pad; 400B uint8 [1, 1, 20, 20] QLinearConv; 400B uint8 [1, 1, 20, 20] QLinearConv |
| 209 | 16.030 | 7584 | 278 | 7862 | 169 | Equal |  | 900B uint8 [1, 1, 30, 30] Gather; 600B float32 [1, 10, 3, 5] GridSample; 600B float32 [1, 10, 3, 5] GridSample; 270B uint8 [1, 1, 30, 9] Gather |
| 096 | 16.053 | 7261 | 421 | 7682 | 114 | Equal | scatter, topk | 1200B float32 [10, 30] ReduceSum; 1200B float32 [10, 30] ReduceSum; 900B uint8 [30, 30] Pad; 484B int32 [11, 11] Slice |
| 255 | 16.064 | 7245 | 352 | 7597 | 115 | Where | free_input_overlay | 900B uint8 [1, 1, 30, 30] QLinearMatMul; 900B bool [1, 1, 30, 30] Greater; 240B bool [1, 1, 30, 8] Concat; 240B bool [1, 1, 8, 30] Concat |
| 219 | 16.117 | 7078 | 132 | 7210 | 73 | Equal |  | 900B uint8 [1, 1, 30, 30] Pad; 600B float32 [1, 1, 15, 10] Slice; 540B float16 [6, 3, 15] Cast; 360B float16 [1, 1, 18, 10] Pad |
| 157 | 16.148 | 6732 | 258 | 6990 | 518 | Equal | scatter, topk | 900B uint8 [1, 1, 30, 30] Pad; 240B float32 [1, 1, 4, 15] Slice; 150B bool [1, 1, 10, 15] Pad; 150B uint8 [1, 1, 10, 15] Where |
| 192 | 16.170 | 6193 | 642 | 6835 | 17 | Equal |  | 1600B float32 [1, 1, 20, 20] Einsum; 900B uint8 [1, 1, 30, 30] Pad; 400B bool [1, 1, 20, 20] Equal; 400B uint8 [1, 1, 20, 20] Where |
| 005 | 16.181 | 6284 | 479 | 6763 | 177 | Equal |  | 900B float32 [1, 1, 15, 15] Conv; 900B uint8 [1, 1, 30, 30] Pad; 529B uint8 [1, 1, 23, 23] Concat; 441B uint8 [1, 1, 21, 21] QLinearConv |
| 089 | 16.183 | 6585 | 160 | 6745 | 66 | Equal | scatter | 900B uint8 [1, 1, 30, 30] Pad; 676B float32 [1, 1, 13, 13] Conv; 384B int32 [96] Concat; 289B uint8 [1, 1, 17, 17] Pad |

## Rewrite Leads

- task233: largest tensors are counted full-canvas carriers, unlike the 20+ frontier. Candidate mechanism: semantic direct-output rewrite; graph surgery unlikely to be enough..
- task366: largest tensors are counted full-canvas carriers, unlike the 20+ frontier. Candidate mechanism: semantic direct-output rewrite; graph surgery unlikely to be enough..
- task018: largest tensors are counted full-canvas carriers, unlike the 20+ frontier. Candidate mechanism: semantic direct-output rewrite; graph surgery unlikely to be enough..
- task133: largest tensors are counted full-canvas carriers, unlike the 20+ frontier. Candidate mechanism: semantic direct-output rewrite; graph surgery unlikely to be enough..
- task158: largest tensors are counted full-canvas carriers, unlike the 20+ frontier. Candidate mechanism: semantic direct-output rewrite; graph surgery unlikely to be enough..
- task054: scatter-heavy edit stream; active net pays for many full-canvas edit masks. Candidate mechanism: sparse edit stream only if inactive duplicate writes are proven harmless..
- task285: largest tensors are counted full-canvas carriers, unlike the 20+ frontier. Candidate mechanism: semantic direct-output rewrite; graph surgery unlikely to be enough..
- task286: largest tensors are counted full-canvas carriers, unlike the 20+ frontier. Candidate mechanism: semantic direct-output rewrite; graph surgery unlikely to be enough..
- task364: largest tensors are counted full-canvas carriers, unlike the 20+ frontier. Candidate mechanism: semantic direct-output rewrite; graph surgery unlikely to be enough..
- task367: largest tensors are counted full-canvas carriers, unlike the 20+ frontier. Candidate mechanism: semantic direct-output rewrite; graph surgery unlikely to be enough..
- task349: local-stencil family with expensive counted canvas carriers; recursive queue already maps it to sparse_conv_single_op_floor. Candidate mechanism: replace staged detector planes with one direct-output QLinearConv/Conv or final Where(input overlay)..
- task101: one-hot/final-equal family; expensive label or mask planes may be delayed into graph output. Candidate mechanism: free_final_onehot_equal or free-input overlay..
- task076: largest tensors are counted full-canvas carriers, unlike the 20+ frontier. Candidate mechanism: semantic direct-output rewrite; graph surgery unlikely to be enough..
- task138: local-stencil family with expensive counted canvas carriers; recursive queue already maps it to sparse_conv_single_op_floor. Candidate mechanism: replace staged detector planes with one direct-output QLinearConv/Conv or final Where(input overlay)..
- task118: largest tensors are counted full-canvas carriers, unlike the 20+ frontier. Candidate mechanism: semantic direct-output rewrite; graph surgery unlikely to be enough..
- task173: largest tensors are counted full-canvas carriers, unlike the 20+ frontier. Candidate mechanism: semantic direct-output rewrite; graph surgery unlikely to be enough..
- task198: largest tensors are counted full-canvas carriers, unlike the 20+ frontier. Candidate mechanism: semantic direct-output rewrite; graph surgery unlikely to be enough..
- task191: largest tensors are counted full-canvas carriers, unlike the 20+ frontier. Candidate mechanism: semantic direct-output rewrite; graph surgery unlikely to be enough..
- task145: largest tensors are counted full-canvas carriers, unlike the 20+ frontier. Candidate mechanism: semantic direct-output rewrite; graph surgery unlikely to be enough..
- task025: one-hot/final-equal family; expensive label or mask planes may be delayed into graph output. Candidate mechanism: free_final_onehot_equal or free-input overlay..
