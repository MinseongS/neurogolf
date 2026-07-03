# Crop-bounds scan (all 400 tasks)

> **CAVEAT (S10 fan-out result):** top-11 flags ALL priced FLOOR — the scanner cannot distinguish counted entry reads from output-welded planes (free-output axis dims). Treat remaining flags as presumed false-positive unless the flagged plane is a counted ENTRY read. est_pts ignores re-embed seam costs.

SCAN-ONLY. FLAG = net carries counted intermediate planes larger than any grid the generator can emit (sampled + bundled/arc-gen max).

`est_saving_bytes` = sum over oversized planes of `bytes * (1 - bound^2/dim^2)`. `est_pts` = ln(M/(M-S)) with M = current mem+params, S = est saving capped at counted mem. Both are RANKING heuristics, NOT promises: only counted ENTRY reads benefit; free-input walk-einsum planes do not (task077 refutation), and shrinking a plane usually needs Slice/Pad plumbing that costs params. Verify per-hit before touching a net.

Sampling caveat: gen_bound from ~300 samples + bundled max; conditional branches can exceed it (see grid_crop_bounds.md) -- re-verify the bound with a large fresh sample before landing any crop.

- tasks scanned: 400
- FLAGGED: 172 (excluding already_handled: 163)
- generator sampling failures/timeouts (bound may be underestimated): 7

## Flagged tasks (ranked by est_pts, then est_saving_bytes)

| rank | task | arcid | net_dim | gen_bound | gen_in | gen_out | bundled(i/o) | #planes | oversized_B | est_save_B | mem+params | est_pts | top ops | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | task329 | d23f8c26 | 30 | 9 | 9x9 | 9x9 | 9/9 | 4 | 990 | 900 | 1029+30 | 1.902 | Unsqueezex2 ArgMax Cast Equal | ok |
| 2 | task150 | 67a3c6ac | 30 | 9 | 9x9 | 9x9 | 9/9 | 1 | 120 | 109 | 136+2 | 1.567 | Cast Gather ReduceL2 Sub | ok |
| 3 | task155 | 68b16354 | 30 | 8 | 8x8 | 8x8 | 8/8 | 1 | 120 | 111 | 140+2 | 1.538 | Cast Gather ReduceL2 Sub | ok |
| 4 | task289 | b91ae062 | 30 | 15 | 3x3 | 15x15 | 3/15 | 4 | 570 | 427 | 618+124 | 0.858 | Castx2 Div Einsum Equal | ok |
| 5 | task341 | d6ad076f | 30 | 10 | 10x10 | 10x10 | 10/10 | 1 | 900 | 800 | 1394+35 | 0.821 | Slicex9 Andx8 Equalx4 Orx3 | ok |
| 6 | task239 | 9af7a82c | 30 | 12 | 4x4 | 12x4 | 4/12 | 3 | 630 | 529 | 897+50 | 0.818 | Subx3 Concatx2 Wherex2 Cast | ok |
| 7 | task043 | 2281f1f4 | 30 | 10 | 10x10 | 10x10 | 10/10 | 4 | 480 | 426 | 588+206 | 0.771 | Castx2 Concatx2 Padx2 Slicex2 | ok |
| 8 | task345 | d9f24cd1 | 30 | 10 | 10x10 | 10x10 | 10/10 | 1 | 900 | 800 | 1431+64 | 0.766 | Castx6 Slicex6 Andx5 Gatherx5 | ok |
| 9 | task068 | 31aa019c | 30 | 10 | 10x10 | 10x10 | 10/10 | 3 | 360 | 320 | 500+123 | 0.721 | Castx3 Einsumx3 Addx2 BitwiseAndx2 | ok |
| 10 | task335 | d4a91cb9 | 30 | 20 | 20x20 | 20x20 | 20/20 | 21 | 2010 | 1116 | 2106+109 | 0.702 | Equalx4 Greaterx4 Slicex4 Andx2 | ok |
| 11 | task340 | d687bc17 | 30 | 20 | 20x20 | 20x20 | 20/20 | 39 | 5460 | 3033 | 5812+279 | 0.689 | Castx18 Wherex12 Greaterx8 ReduceSumx8 | ok |
| 12 | task075 | 363442ee | 30 | 13 | 9x13 | 9x13 | 13/13 | 1 | 900 | 731 | 1326+161 | 0.676 | Concatx3 Castx2 DepthToSpace Einsum | ok |
| 13 | task355 | de1cd16c | 30 | 20 | 20x20 | 1x1 | 20/1 | 2 | 2400 | 1333 | 2688+27 | 0.675 | ReduceSumx3 ReduceMaxx2 ArgMax Cast | ok |
| 14 | task215 | 8eb1be9a | 30 | 20 | 20x20 | 20x20 | 20/20 | 6 | 660 | 366 | 678+72 | 0.671 | Slicex3 Reshapex2 Wherex2 Cast | ok |
| 15 | task033 | 1e32b0e9 | 30 | 17 | 17x17 | 17x17 | 17/17 | 4 | 1170 | 794 | 1606+33 | 0.663 | Slicex10 Lessx8 Concatx4 Where | ok |
| 16 | task051 | 25d487eb | 30 | 20 | 20x20 | 20x20 | 20/20 | 15 | 1560 | 866 | 1702+101 | 0.655 | Castx8 Mulx6 Einsumx5 Andx3 | ok |
| 17 | task281 | b548a754 | 30 | 13 | 13x13 | 13x13 | 13/13 | 9 | 1410 | 1145 | 2321+65 | 0.654 | Wherex8 Castx6 ArgMaxx4 Lessx3 | ok |
| 18 | task067 | 2dee498d | 30 | 15 | 5x15 | 5x5 | 15/5 | 1 | 30 | 22 | 39+8 | 0.651 | Greater Pad ReduceSum Where | ok |
| 19 | task126 | 54d82841 | 30 | 20 | 10x20 | 10x20 | 20/20 | 6 | 510 | 283 | 590+15 | 0.632 | Castx3 Concatx3 Einsum Equal | ok |
| 20 | task301 | beb8660c | 30 | 12 | 12x9 | 12x9 | 12/12 | 3 | 630 | 529 | 1077+64 | 0.623 | Addx3 Subx3 Castx2 Concatx2 | ok |
| 21 | task141 | 623ea044 | 30 | 21 | 21x21 | 21x21 | 21/21 | 9 | 1320 | 673 | 1404+101 | 0.593 | Castx4 Einsumx4 Wherex3 Absx2 | ok |
| 22 | task041 | 22168020 | 30 | 10 | 10x10 | 10x10 | 10/10 | 1 | 900 | 800 | 1730+59 | 0.593 | BitwiseXorx7 BitwiseOrx6 Cast Concat | ok |
| 23 | task063 | 2bee17df | 30 | 14 | 14x14 | 14x14 | 14/14 | 3 | 1140 | 891 | 1728+283 | 0.586 | Andx2 Castx2 Convx2 Equalx2 | ok |
| 24 | task270 | ae3edfdc | 30 | 15 | 15x15 | 15x15 | 15/15 | 16 | 1800 | 1350 | 2742+327 | 0.580 | Castx12 Padx12 BitwiseAndx8 GatherElementsx8 | ok |
| 25 | task099 | 444801d8 | 30 | 10 | 10x10 | 10x10 | 10/10 | 3 | 940 | 830 | 1633+278 | 0.570 | Maxx14 Minx8 Gatherx7 ReduceMaxx7 | ok |
| 26 | task163 | 6d0160f0 | 30 | 11 | 11x11 | 11x11 | 11/11 | 1 | 900 | 779 | 1655+142 | 0.568 | Gatherx4 Castx3 ArgMaxx2 Modx2 | ok |
| 27 | task240 | 9d9215db | 30 | 19 | 19x19 | 19x19 | 19/19 | 6 | 1050 | 628 | 1393+82 | 0.556 | ArgMaxx7 Castx7 MaxPoolx7 Concatx6 | ok |
| 28 | task348 | db3e9e38 | 30 | 10 | 10x10 | 10x10 | 10/10 | 1 | 900 | 800 | 1824+72 | 0.548 | Castx4 Andx3 Slicex3 Equalx2 | ok |
| 29 | task035 | 1f642eb9 | 28 | 10 | 10x10 | 10x10 | 10/10 | 5 | 1092 | 952 | 2132+129 | 0.547 | Wherex8 Concatx7 ArgMaxx6 GatherElementsx6 | ok |
| 30 | task065 | 2dc579da | 30 | 15 | 15x15 | 7x7 | 15/7 | 3 | 360 | 270 | 582+59 | 0.547 | Wherex5 Einsumx4 Equalx4 BitwiseAndx2 | ok |
| 31 | task287 | b8825c91 | 30 | 16 | 16x16 | 16x16 | 16/16 | 2 | 1170 | 837 | 1699+295 | 0.544 | Divx2 Gatherx2 Cast Einsum | ok |
| 32 | task392 | f8c80d96 | 30 | 10 | 10x10 | 10x10 | 10/10 | 3 | 926 | 810 | 1602+349 | 0.537 | Gatherx5 Castx4 Equalx3 Modx3 | ok |
| 33 | task109 | 47c1f68c | 30 | 13 | 13x13 | 12x12 | 13/12 | 1 | 900 | 731 | 1686+93 | 0.529 | Wherex5 Castx3 Gatherx3 Concatx2 | ok |
| 34 | task124 | 53b68214 | 30 | 10 | 8x10 | 10x10 | 10/10 | 1 | 900 | 800 | 1879+74 | 0.527 | Gatherx5 Castx4 Reshapex3 ArgMaxx2 | ok |
| 35 | task362 | e48d4e1a | 30 | 10 | 10x10 | 10x10 | 10/10 | 2 | 240 | 213 | 408+113 | 0.527 | Castx3 Einsumx3 ScatterElementsx2 Subx2 | ok |
| 36 | task381 | ef135b50 | 30 | 10 | 10x10 | 10x10 | 10/10 | 1 | 900 | 800 | 1908+55 | 0.523 | Castx3 Andx2 GreaterOrEqual MatMul | ok |
| 37 | task256 | a65b410d | 30 | 13 | 13x13 | 13x13 | 13/13 | 3 | 1140 | 925 | 2223+56 | 0.521 | Andx4 Castx2 Lessx2 Slicex2 | ok |
| 38 | task119 | 508bd3b6 | 20 | 12 | 12x12 | 12x12 | 12/12 | 5 | 840 | 537 | 1244+121 | 0.501 | Castx12 Wherex8 Subx6 ArgMaxx4 | ok |
| 39 | task260 | a78176bb | 30 | 10 | 10x10 | 10x10 | 10/10 | 3 | 963 | 821 | 1804+299 | 0.495 | Castx9 Subx5 Wherex5 Addx4 | ok |
| 40 | task011 | 09629e4f | 30 | 11 | 11x11 | 11x11 | 11/11 | 1 | 900 | 779 | 1836+163 | 0.494 | Gatherx4 Unsqueezex4 Addx2 ArgMaxx2 | ok |
| 41 | task047 | 23581191 | 30 | 9 | 9x9 | 9x9 | 9/9 | 3 | 360 | 327 | 826+36 | 0.478 | ReduceMaxx4 Wherex4 Concatx3 BitwiseAndx2 | ok |
| 42 | task249 | a416b8f3 | 30 | 10 | 5x5 | 5x10 | 5/10 | 1 | 120 | 106 | 248+39 | 0.465 | Gatherx2 Cast Pad ReduceSum | ok |
| 43 | task394 | f9012d9b | 14 | 7 | 7x7 | 3x3 | 7/3 | 3 | 1386 | 1039 | 1871+929 | 0.464 | Castx4 Wherex4 Equalx3 Gatherx3 | ok |
| 44 | task273 | af902bf9 | 30 | 10 | 10x10 | 10x10 | 10/10 | 1 | 900 | 800 | 2109+51 | 0.463 | Gatherx9 Andx6 Greaterx4 Lessx4 | ok |
| 45 | task228 | 952a094c | 16 | 10 | 10x10 | 10x10 | 10/10 | 3 | 589 | 356 | 849+113 | 0.463 | Castx15 Wherex6 Addx2 ArgMaxx2 | ok |
| 46 | task199 | 834ec97d | 30 | 15 | 15x15 | 15x15 | 15/15 | 1 | 900 | 675 | 1749+88 | 0.458 | Wherex4 Castx3 Einsumx3 Divx2 | ok |
| 47 | task012 | 0962bcdd | 30 | 12 | 12x12 | 12x12 | 12/12 | 2 | 1036 | 824 | 2076+188 | 0.453 | Castx6 Concatx4 Addx3 ArgMaxx2 | ok |
| 48 | task077 | 36fdfd69 | 30 | 21 | 20x21 | 20x21 | 21/21 | 2 | 4500 | 2295 | 4500+1831 | 0.450 | MaxPoolx9 Wherex9 Lessx5 Andx3 | already_handled |
| 49 | task295 | bbc9ae5d | 30 | 18 | 1x18 | 9x18 | 18/18 | 1 | 900 | 576 | 1557+47 | 0.445 | Wherex4 Castx3 Lessx3 ReduceSumx3 | ok |
| 50 | task234 | 98cf29f8 | 30 | 20 | 20x20 | 20x20 | 20/20 | 18 | 2460 | 1366 | 3744+64 | 0.445 | Castx20 Wherex19 Reshapex18 Subx13 | ok |
| 51 | task168 | 6e19193c | 30 | 10 | 10x10 | 10x10 | 10/10 | 1 | 900 | 800 | 2096+143 | 0.442 | QLinearConvx8 Castx2 Max Mul | ok |
| 52 | task397 | fcc82909 | 30 | 10 | 10x10 | 10x10 | 10/10 | 4 | 1116 | 978 | 2585+200 | 0.433 | Reshapex10 Andx6 Equalx6 Orx6 | ok |
| 53 | task224 | 928ad970 | 30 | 16 | 16x16 | 16x16 | 16/16 | 5 | 1380 | 987 | 2460+371 | 0.429 | Andx6 ArgMaxx4 Castx4 Greaterx4 | ok |
| 54 | task034 | 1f0c79e5 | 30 | 9 | 9x9 | 9x9 | 9/9 | 1 | 900 | 819 | 2198+167 | 0.425 | QLinearConvx8 Castx3 Padx3 Concat | ok |
| 55 | task388 | f5b8619d | 30 | 12 | 6x6 | 12x12 | 6/12 | 4 | 1044 | 864 | 2278+218 | 0.425 | Castx8 Addx6 Lessx5 Mulx3 | ok |
| 56 | task061 | 29ec7d0e | 30 | 18 | 18x18 | 18x18 | 18/18 | 1 | 900 | 576 | 1633+35 | 0.424 | Modx2 ArgMax Cast Equal | ok |
| 57 | task342 | d89b689b | 16 | 10 | 10x10 | 10x10 | 10/10 | 4 | 896 | 546 | 1510+75 | 0.422 | Castx14 Concatx9 GatherElementsx8 Andx6 | ok |
| 58 | task361 | e40b9e2f | 30 | 10 | 10x10 | 10x10 | 10/10 | 25 | 2300 | 1592 | 4322+366 | 0.415 | Addx10 Castx7 Subx7 Mulx5 | ok |
| 59 | task319 | ce602527 | 30 | 19 | 19x19 | 5x5 | 19/5 | 19 | 10320 | 6180 | 18645+108 | 0.400 | Gatherx40 Equalx38 Castx37 Addx36 | ok |
| 60 | task376 | eb281b96 | 30 | 21 | 6x17 | 21x17 | 17/21 | 1 | 120 | 61 | 144+42 | 0.399 | Gatherx2 Cast Mod ReduceSum | ok |
| 61 | task190 | 7ddcd7ec | 30 | 10 | 10x10 | 10x10 | 10/10 | 1 | 900 | 800 | 2280+247 | 0.381 | QLinearConvx5 Maxx3 Castx2 MaxPool | ok |
| 62 | task062 | 2bcee788 | 30 | 10 | 10x10 | 10x10 | 10/10 | 1 | 900 | 800 | 2465+71 | 0.379 | Addx6 ReduceMaxx6 Wherex6 ArgMaxx5 | ok |
| 63 | task316 | cdecee7f | 30 | 10 | 10x10 | 3x3 | 10/3 | 2 | 150 | 133 | 388+38 | 0.375 | Castx3 Gatherx2 Einsum Equal | ok |
| 64 | task178 | 746b3537 | 30 | 14 | 14x14 | 5x5 | 14/5 | 4 | 300 | 234 | 702+61 | 0.368 | Castx6 Slicex6 Wherex5 Equalx3 | ok |
| 65 | task288 | b8cdaf2b | 12 | 9 | 9x9 | 9x9 | 9/9 | 3 | 672 | 294 | 769+189 | 0.367 | ArgMaxx2 Castx2 Equalx2 Concat | ok |
| 66 | task231 | 963e52fc | 30 | 20 | 5x10 | 5x20 | 10/20 | 1 | 120 | 66 | 180+39 | 0.363 | Concat Gather Less ReduceSum | ok |
| 67 | task102 | 44d8ac46 | 30 | 12 | 12x12 | 12x12 | 12/12 | 1 | 900 | 756 | 2414+113 | 0.355 | QLinearConvx4 MaxPoolx3 Castx2 Max | ok |
| 68 | task354 | ddf7fa4f | 30 | 10 | 10x10 | 10x10 | 10/10 | 1 | 900 | 800 | 2674+77 | 0.344 | Castx14 Addx9 Slicex6 Wherex6 | ok |
| 69 | task174 | 72ca375d | 30 | 10 | 10x10 | 5x5 | 10/5 | 4 | 2460 | 2066 | 6973+142 | 0.343 | Greaterx9 Castx8 Wherex7 Notx6 | ok |
| 70 | task096 | 4290ef0e | 30 | 19 | 19x19 | 11x11 | 19/11 | 7 | 4200 | 2515 | 7987+801 | 0.337 | Wherex18 Castx16 Addx11 Subx9 | ok |
| 71 | task268 | aba27056 | 30 | 10 | 10x10 | 10x10 | 10/10 | 3 | 1140 | 1013 | 3328+256 | 0.332 | Andx11 Orx11 Castx9 Greaterx8 | ok |
| 72 | task200 | 8403a5d5 | 30 | 10 | 10x10 | 10x10 | 10/10 | 1 | 240 | 213 | 570+186 | 0.332 | Castx2 Concatx2 Convx2 Einsum | ok |
| 73 | task019 | 10fcaaa3 | 30 | 12 | 6x6 | 12x12 | 6/12 | 1 | 900 | 756 | 2709+89 | 0.315 | Wherex4 Castx3 ReduceSumx3 Addx2 | ok |
| 74 | task305 | c3f564a4 | 30 | 16 | 16x16 | 16x16 | 16/16 | 1 | 240 | 171 | 296+340 | 0.315 | BitwiseAndx2 ArgMax Gather GlobalMaxPool | ok |
| 75 | task042 | 22233c11 | 30 | 10 | 10x10 | 10x10 | 10/10 | 1 | 900 | 800 | 2792+318 | 0.297 | Convx3 Cast ConvTranspose Greater | ok |
| 76 | task010 | 08ed6ac7 | 30 | 9 | 9x9 | 9x9 | 9/9 | 1 | 300 | 273 | 730+340 | 0.295 | Castx3 Concatx3 ReduceSumx2 Splitx2 | ok |
| 77 | task290 | b94a9452 | 30 | 15 | 15x15 | 6x6 | 15/6 | 1 | 120 | 90 | 292+63 | 0.292 | Einsumx2 Gatherx2 Cast Equal | ok |
| 78 | task091 | 3f7978a0 | 30 | 15 | 15x15 | 14x14 | 15/14 | 2 | 1020 | 765 | 2846+175 | 0.292 | Castx10 Unsqueezex9 Concatx7 Addx5 | ok |
| 79 | task161 | 6cdd2623 | 30 | 25 | 20x25 | 20x25 | 25/25 | 15 | 2100 | 641 | 2471+99 | 0.287 | Greaterx6 ReduceSumx5 Wherex4 Slicex3 | ok |
| 80 | task029 | 1c786137 | 30 | 25 | 25x25 | 23x22 | 25/23 | 15 | 4500 | 1375 | 5436+89 | 0.286 | ArgMaxx5 Castx5 Gatherx4 Squeezex4 | already_handled |
| 81 | task094 | 41e4d17e | 30 | 15 | 15x15 | 15x15 | 15/15 | 1 | 900 | 675 | 2662+51 | 0.286 | Castx3 Greaterx3 AveragePoolx2 Concatx2 | ok |
| 82 | task251 | a5313dff | 30 | 12 | 12x12 | 12x12 | 12/12 | 1 | 900 | 756 | 2772+272 | 0.285 | Mulx10 MaxPoolx8 Castx2 Slicex2 | ok |
| 83 | task250 | a48eeaf7 | 30 | 10 | 10x10 | 10x10 | 10/10 | 1 | 900 | 800 | 3164+61 | 0.285 | Greaterx4 Wherex4 Castx3 Equalx3 | ok |
| 84 | task333 | d43fd935 | 30 | 10 | 10x10 | 10x10 | 10/10 | 1 | 900 | 800 | 2792+435 | 0.285 | Wherex12 Castx11 Addx8 Divx8 | ok |
| 85 | task154 | 6855a6e4 | 30 | 15 | 15x15 | 15x15 | 15/15 | 1 | 900 | 675 | 2661+99 | 0.281 | Castx7 Slicex7 Wherex6 Gatherx4 | ok |
| 86 | task132 | 56ff96f3 | 30 | 15 | 15x15 | 15x15 | 15/15 | 7 | 1340 | 999 | 3990+99 | 0.280 | Gatherx10 Andx6 Castx5 BitShiftx4 | ok |
| 87 | task086 | 3befdf3e | 30 | 12 | 12x12 | 12x12 | 12/12 | 1 | 900 | 756 | 2946+190 | 0.276 | Castx4 Addx3 Wherex3 Gatherx2 | ok |
| 88 | task071 | 3345333e | 30 | 16 | 16x16 | 16x16 | 16/16 | 1 | 900 | 644 | 2643+55 | 0.273 | Addx6 ArgMaxx3 Castx3 Reshapex3 | ok |
| 89 | task018 | 0e206a2e | 30 | 24 | 24x24 | 24x24 | 24/24 | 22 | 20130 | 7246 | 29807+1180 | 0.266 | Andx123 Addx112 Mulx102 Castx86 | ok |
| 90 | task089 | 3e980e27 | 30 | 13 | 13x13 | 13x13 | 13/13 | 19 | 2438 | 1649 | 6923+160 | 0.265 | Castx11 Addx8 Reshapex7 Wherex7 | ok |
| 91 | task246 | a2fd1cf0 | 30 | 20 | 20x20 | 20x20 | 20/20 | 3 | 1140 | 633 | 2334+419 | 0.261 | Andx4 Slicex4 Castx2 Convx2 | ok |
| 92 | task037 | 1f876c06 | 30 | 10 | 10x10 | 10x10 | 10/10 | 1 | 900 | 800 | 3384+201 | 0.253 | Castx7 Wherex5 GatherElementsx4 Addx3 | ok |
| 93 | task064 | 2c608aff | 30 | 24 | 24x24 | 24x24 | 24/24 | 27 | 6120 | 2203 | 9852+134 | 0.249 | Castx13 ArgMaxx9 Gatherx9 Slicex9 | ok |
| 94 | task390 | f8a8fe49 | 30 | 15 | 15x15 | 15x15 | 15/15 | 1 | 900 | 675 | 2624+467 | 0.246 | Andx17 Slicex15 Castx11 Wherex11 | ok |
| 95 | task367 | e73095fd | 30 | 20 | 20x20 | 20x20 | 20/20 | 3 | 10800 | 4240 | 14800+4725 | 0.245 | Castx2 QLinearConvx2 Pad Slice | ok |
| 96 | task066 | 2dd70a9a | 30 | 20 | 20x20 | 20x20 | 20/20 | 11 | 3930 | 2183 | 9955+166 | 0.243 | Andx51 Orx24 Greaterx18 Equalx17 | ok |
| 97 | task358 | e21d9049 | 30 | 21 | 21x20 | 21x20 | 21/21 | 15 | 1860 | 948 | 4256+150 | 0.242 | Castx12 Equalx6 ArgMaxx5 Addx4 | ok |
| 98 | task363 | e5062a87 | 30 | 10 | 10x10 | 10x10 | 10/10 | 2 | 1069 | 869 | 3839+296 | 0.236 | Castx10 Equalx5 Mulx4 Slicex4 | ok |
| 99 | task069 | 321b1fc6 | 30 | 10 | 10x10 | 10x10 | 10/10 | 1 | 900 | 800 | 3848+64 | 0.229 | Unsqueezex16 Subx10 ArgMaxx9 Castx9 | ok |
| 100 | task070 | 32597951 | 30 | 17 | 17x17 | 17x17 | 17/17 | 1 | 900 | 611 | 2977+31 | 0.227 | ReduceMaxx2 And Cast Gather | ok |
| 101 | task093 | 4093f84a | 30 | 14 | 14x14 | 14x14 | 14/14 | 1 | 900 | 704 | 3456+103 | 0.220 | Wherex13 Castx9 Lessx6 QLinearMatMulx4 | ok |
| 102 | task368 | e76a88a6 | 30 | 10 | 10x10 | 10x10 | 10/10 | 26 | 2341 | 1224 | 6022+203 | 0.219 | Castx18 Addx9 Gatherx9 Concatx5 | ok |
| 103 | task125 | 543a7ed5 | 30 | 15 | 15x15 | 15x15 | 15/15 | 1 | 900 | 675 | 3434+29 | 0.217 | MaxPoolx5 Addx2 Padx2 Cast | ok |
| 104 | task359 | e26a3af2 | 30 | 27 | 25x27 | 25x27 | 26/26 | 21 | 3840 | 729 | 3852+15 | 0.209 | Castx5 Mulx4 ReduceSumx4 Subx4 | ok |
| 105 | task382 | f15e1fac | 30 | 20 | 20x20 | 20x20 | 20/20 | 9 | 1896 | 1029 | 5648+135 | 0.196 | Slicex28 Wherex12 Castx5 Padx5 | ok |
| 106 | task208 | 890034e9 | 30 | 21 | 21x21 | 21x21 | 21/21 | 19 | 1620 | 826 | 4612+114 | 0.192 | Castx7 Equalx7 Subx7 Wherex7 | ok |
| 107 | task088 | 3de23699 | 30 | 24 | 24x24 | 10x10 | 24/10 | 1 | 900 | 324 | 1791+81 | 0.190 | Castx8 Subx8 Addx4 Divx4 | ok |
| 108 | task050 | 253bf280 | 30 | 15 | 15x15 | 15x15 | 15/15 | 1 | 900 | 675 | 3825+86 | 0.190 | Castx4 Andx2 ArgMaxx2 Greaterx2 | ok |
| 109 | task378 | ec883f72 | 30 | 12 | 12x12 | 12x12 | 12/12 | 1 | 900 | 756 | 4332+117 | 0.186 | Subx13 Andx10 Castx10 Wherex9 | ok |
| 110 | task117 | 4c5c2cf0 | 30 | 15 | 15x15 | 15x15 | 15/15 | 1 | 900 | 675 | 3922+243 | 0.177 | Subx9 Slicex7 Concatx5 Addx4 | ok |
| 111 | task213 | 8e1813be | 30 | 24 | 24x24 | 7x7 | 24/7 | 10 | 840 | 302 | 1838+49 | 0.175 | Slicex12 Castx10 Orx8 Wherex8 | ok |
| 112 | task045 | 22eb0ac0 | 30 | 10 | 10x10 | 10x10 | 10/10 | 1 | 300 | 266 | 1050+623 | 0.174 | Castx2 Slicex2 Splitx2 Concat | ok |
| 113 | task379 | ecdecbb3 | 30 | 20 | 20x20 | 20x20 | 20/20 | 15 | 2580 | 1433 | 9309+133 | 0.165 | Wherex18 Equalx11 Andx9 Greaterx9 | ok |
| 114 | task101 | 447fd412 | 30 | 21 | 17x14 | 17x14 | 21/21 | 67 | 4409 | 2156 | 13573+874 | 0.162 | Andx48 Addx47 Castx32 Gatherx30 | ok |
| 115 | task196 | 810b9b61 | 30 | 15 | 15x15 | 15x15 | 15/15 | 1 | 900 | 675 | 4500+38 | 0.161 | MaxPoolx5 Minx4 Cast Less | ok |
| 116 | task046 | 234bbc79 | 30 | 20 | 3x20 | 3x16 | 20/16 | 5 | 570 | 316 | 2221+97 | 0.147 | Wherex15 Equalx7 Slicex6 Castx5 | ok |
| 117 | task265 | a8d7556c | 30 | 18 | 18x18 | 18x18 | 18/18 | 1 | 900 | 576 | 4328+34 | 0.142 | Addx2 MaxPoolx2 Subx2 Cast | ok |
| 118 | task058 | 28e73c20 | 30 | 20 | 20x20 | 20x20 | 20/20 | 1 | 240 | 133 | 252+781 | 0.138 | BitwiseAnd Cast Gather ReduceL2 | ok |
| 119 | task004 | 025d127b | 30 | 16 | 16x16 | 16x16 | 16/16 | 1 | 900 | 644 | 5044+100 | 0.134 | Wherex4 Castx2 Equalx2 Gatherx2 | ok |
| 120 | task206 | 88a10436 | 19 | 12 | 12x12 | 12x12 | 12/12 | 9 | 874 | 525 | 4122+77 | 0.134 | Castx11 Unsqueezex9 Concatx8 ReduceMaxx6 | ok |
| 121 | task162 | 6cf79266 | 30 | 20 | 20x20 | 20x20 | 20/20 | 1 | 900 | 500 | 4024+44 | 0.131 | Cast Greater MaxPool Pad | ok |
| 122 | task279 | b2862040 | 30 | 16 | 16x16 | 16x16 | 16/16 | 1 | 900 | 644 | 5508+38 | 0.123 | QLinearConvx10 Minx2 Cast Greater | ok |
| 123 | task219 | 90f3ed37 | 30 | 15 | 15x10 | 15x10 | 15/15 | 6 | 1404 | 829 | 7078+132 | 0.122 | Unsqueezex47 Mulx35 Gatherx30 ReduceSumx20 | ok |
| 124 | task008 | 05f2a901 | 30 | 16 | 16x16 | 16x16 | 16/16 | 1 | 900 | 644 | 5561+65 | 0.122 | Castx10 Wherex10 Subx6 Lessx5 | ok |
| 125 | task387 | f35d900a | 30 | 18 | 18x18 | 18x18 | 18/18 | 1 | 900 | 576 | 4954+105 | 0.121 | Addx17 Reshapex8 Subx8 Castx7 | ok |
| 126 | task278 | b27ca6d3 | 30 | 18 | 18x18 | 18x18 | 18/18 | 1 | 900 | 576 | 5436+47 | 0.111 | Castx3 Slicex2 MaxPool Min | ok |
| 127 | task182 | 776ffc46 | 30 | 20 | 20x20 | 20x20 | 20/20 | 7 | 1300 | 644 | 6345+97 | 0.105 | Castx5 Addx4 Reshapex4 ArgMaxx2 | ok |
| 128 | task048 | 239be575 | 30 | 8 | 8x8 | 1x1 | 8/1 | 1 | 120 | 111 | 1030+116 | 0.102 | BitwiseOrx36 Slicex19 BitShiftx18 BitwiseAndx10 | ok |
| 129 | task398 | feca6190 | 30 | 25 | 1x5 | 25x25 | 5/25 | 1 | 900 | 275 | 1666+1193 | 0.101 | Gatherx3 Castx2 Slicex2 Squeezex2 | ok |
| 130 | task243 | 9edfc990 | 30 | 18 | 18x18 | 18x18 | 18/18 | 1 | 900 | 576 | 5112+901 | 0.101 | Wherex29 QLinearConvx28 Castx3 Slicex2 | already_handled |
| 131 | task209 | 8a004b2b | 30 | 20 | 20x20 | 16x20 | 20/20 | 35 | 3900 | 2166 | 21298+1418 | 0.100 | Castx20 Andx17 Subx17 Wherex17 | ok |
| 132 | task192 | 7e0986d6 | 30 | 20 | 20x20 | 20x20 | 20/20 | 5 | 1200 | 666 | 5745+1251 | 0.100 | Wherex4 Greaterx3 ReduceSumx3 Equalx2 | already_handled |
| 133 | task157 | 6a1e5592 | 30 | 15 | 10x15 | 10x15 | 15/15 | 19 | 1514 | 785 | 7983+351 | 0.099 | Andx70 Gatherx57 Concatx48 Castx39 | hard_timeout_after_15 |
| 134 | task005 | 045e512c | 30 | 21 | 21x21 | 21x21 | 21/21 | 13 | 1958 | 635 | 6392+490 | 0.097 | Slicex101 Maxx44 Concatx13 Mulx9 | ok |
| 135 | task014 | 0b148d64 | 30 | 25 | 25x25 | 16x16 | 25/18 | 7 | 1260 | 385 | 4088+83 | 0.097 | Castx6 Addx4 ArgMaxx4 Gatherx3 | already_handled |
| 136 | task173 | 72322fa7 | 30 | 25 | 25x25 | 25x25 | 25/25 | 10 | 6246 | 1662 | 18717+459 | 0.091 | Gatherx19 Equalx14 Addx13 Castx11 | already_handled |
| 137 | task049 | 23b5c85d | 30 | 20 | 20x20 | 12x15 | 20/15 | 2 | 60 | 33 | 252+134 | 0.090 | Castx8 Divx3 Einsumx3 Subx3 | ok |
| 138 | task328 | d22278a0 | 30 | 18 | 18x18 | 18x18 | 18/18 | 1 | 900 | 576 | 4835+1940 | 0.089 | Castx20 Reshapex8 Concatx7 Addx6 | ok |
| 139 | task165 | 6d58a25d | 30 | 20 | 20x20 | 20x20 | 20/20 | 1 | 900 | 500 | 5836+230 | 0.086 | Castx5 Addx2 Andx2 ArgMaxx2 | ok |
| 140 | task222 | 91714a58 | 30 | 16 | 16x16 | 16x16 | 16/16 | 1 | 900 | 644 | 5136+2952 | 0.083 | MaxPoolx6 QLinearConvx6 Castx5 Equalx3 | already_handled |
| 141 | task002 | 00d62c1b | 30 | 20 | 20x20 | 20x20 | 20/20 | 1 | 900 | 500 | 6100+589 | 0.078 | BitwiseOrx80 BitShiftx40 Gatherx40 BitwiseAndx22 | ok |
| 142 | task343 | d8c310e9 | 30 | 15 | 5x15 | 5x15 | 15/15 | 1 | 120 | 90 | 1147+75 | 0.076 | Castx7 Gatherx7 Wherex4 Equalx3 | ok |
| 143 | task238 | 9aec4887 | 30 | 16 | 16x16 | 7x7 | 16/7 | 2 | 240 | 171 | 2051+444 | 0.071 | Castx8 Einsumx6 Wherex6 ArgMaxx4 | ok |
| 144 | task370 | e8dc4411 | 30 | 20 | 20x20 | 20x20 | 20/20 | 3 | 1140 | 633 | 8645+1024 | 0.068 | Greaterx8 Castx7 Gatherx6 Wherex6 | ok |
| 145 | task145 | 6455b5f5 | 30 | 20 | 20x20 | 20x20 | 20/20 | 5 | 1200 | 666 | 9244+1248 | 0.066 | Wherex9 MaxPoolx6 Castx3 Equalx3 | ok |
| 146 | task237 | 99fa7670 | 30 | 9 | 9x9 | 9x9 | 9/9 | 1 | 120 | 109 | 1763+138 | 0.059 | Castx9 Slicex7 Wherex6 Padx4 | ok |
| 147 | task324 | d07ae81c | 30 | 20 | 20x20 | 20x20 | 20/20 | 1 | 900 | 500 | 7308+1586 | 0.058 | ReduceMaxx11 Wherex6 Andx5 Notx5 | hard_timeout_after_193 |
| 148 | task148 | 673ef223 | 30 | 24 | 24x12 | 24x12 | 24/24 | 1 | 900 | 324 | 5759+110 | 0.057 | Castx15 Slicex10 Andx8 Mulx4 | ok |
| 149 | task198 | 83302e8f | 30 | 29 | 29x29 | 29x29 | 29/29 | 26 | 11400 | 747 | 13434+138 | 0.057 | Castx11 Gatherx10 Equalx9 Slicex7 | already_handled |
| 150 | task383 | f1cefba8 | 30 | 24 | 24x24 | 24x24 | 24/24 | 1 | 900 | 324 | 5974+96 | 0.055 | Gatherx8 Castx7 Reshapex6 Wherex6 | ok |
| 151 | task191 | 7df24a62 | 30 | 23 | 23x23 | 23x23 | 23/23 | 15 | 2400 | 989 | 14070+6576 | 0.049 | Castx12 Reshapex12 Mulx10 Greaterx8 | ok |
| 152 | task017 | 0dfd9992 | 30 | 21 | 21x21 | 21x21 | 21/21 | 1 | 900 | 459 | 8448+2021 | 0.045 | Castx6 Gatherx4 Addx3 Modx3 | ok |
| 153 | task170 | 6ecd11f4 | 30 | 28 | 28x28 | 4x4 | 28/4 | 6 | 810 | 104 | 2216+329 | 0.042 | Unsqueezex9 Castx8 Concatx6 Addx4 | ok |
| 154 | task177 | 7468f01a | 30 | 20 | 20x20 | 8x8 | 20/8 | 2 | 240 | 133 | 3692+130 | 0.035 | Convx3 Addx2 ArgMaxx2 Castx2 | ok |
| 155 | task338 | d5d6de2d | 30 | 25 | 25x25 | 25x25 | 25/25 | 5 | 1200 | 366 | 10000+666 | 0.035 | Greaterx4 Slicex3 Wherex3 ReduceMaxx2 | ok |
| 156 | task284 | b7249182 | 30 | 24 | 24x24 | 24x24 | 24/24 | 6 | 352 | 116 | 3082+653 | 0.032 | Addx20 Mulx12 Concatx11 Unsqueezex11 | ok |
| 157 | task138 | 5daaa586 | 30 | 26 | 26x25 | 22x23 | 26/23 | 5 | 2490 | 381 | 12215+462 | 0.031 | Gatherx7 Wherex7 Castx5 ArgMaxx4 | already_handled |
| 158 | task350 | dbc1a6ce | 30 | 26 | 26x24 | 26x24 | 26/26 | 1 | 900 | 224 | 9012+26 | 0.025 | MaxPoolx4 Minx2 Cast Greater | ok |
| 159 | task201 | 846bdb03 | 16 | 13 | 13x13 | 6x8 | 13/8 | 1 | 240 | 81 | 3202+154 | 0.025 | Wherex8 ArgMaxx6 LessOrEqualx6 Equalx5 | ok |
| 160 | task225 | 93b581b8 | 25 | 6 | 6x6 | 6x6 | 6/6 | 1 | 25 | 23 | 926+107 | 0.023 | Addx4 Gatherx4 Castx3 ArgMinx2 | ok |
| 161 | task364 | e509e548 | 30 | 22 | 20x22 | 20x22 | 22/22 | 1 | 900 | 416 | 18500+1131 | 0.021 | MaxPoolx14 Mulx14 Castx3 Equalx3 | ok |
| 162 | task336 | d4f3cd78 | 14 | 10 | 10x10 | 10x10 | 10/10 | 3 | 79 | 33 | 755+1186 | 0.017 | Concat Gather GatherND Gemm | ok |
| 163 | task158 | 6aa20dc0 | 30 | 26 | 26x25 | 26x25 | 26/26 | 7 | 1440 | 358 | 18263+2647 | 0.017 | Gatherx13 Equalx11 QLinearConvx9 Castx7 | ok |
| 164 | task204 | 868de0fa | 30 | 20 | 20x20 | 20x20 | 20/20 | 4 | 300 | 166 | 9780+452 | 0.016 | QLinearConvx9 MaxPoolx7 Castx3 Slicex3 | ok |
| 165 | task123 | 539a4f51 | 11 | 10 | 5x5 | 10x10 | 5/10 | 2 | 121 | 21 | 403+940 | 0.016 | Gatherx3 Castx2 Concat Einsum | ok |
| 166 | task286 | b782dc8a | 30 | 25 | 25x25 | 25x25 | 25/25 | 1 | 900 | 275 | 21090+2881 | 0.011 | BitwiseAndx1067 BitwiseOrx965 BitShiftx926 Gatherx48 | ok |
| 167 | task131 | 56dc2b01 | 24 | 18 | 18x18 | 18x18 | 18/18 | 1 | 96 | 42 | 3312+675 | 0.011 | Wherex21 Castx15 Unsqueezex11 Addx9 | ok |
| 168 | task020 | 11852cab | 12 | 10 | 10x10 | 10x10 | 10/10 | 1 | 48 | 14 | 1356+170 | 0.010 | Addx8 ArgMaxx3 Gatherx3 Reshapex3 | ok |
| 169 | task153 | 681b3aeb | 12 | 10 | 10x10 | 3x3 | 10/3 | 2 | 24 | 7 | 741+54 | 0.009 | Castx7 Reshapex6 Addx5 Modx4 | ok |
| 170 | task009 | 06df4c85 | 30 | 29 | 29x29 | 29x29 | 29/29 | 1 | 900 | 59 | 6929+95 | 0.008 | Wherex7 Addx4 MaxPoolx4 Modx4 | ok |
| 171 | task293 | ba97ae07 | 16 | 15 | 15x15 | 15x15 | 15/15 | 1 | 64 | 7 | 1247+41 | 0.006 | Addx5 Expandx5 Concatx4 Squeezex4 | ok |
| 172 | task118 | 50846271 | 30 | 28 | 25x28 | 25x28 | 28/28 | 1 | 900 | 116 | 24722+723 | 0.005 | Castx9 Addx3 Einsumx3 Mulx3 | ok |

## Generator sampling failures / timeouts (blind spots)

Bounds for these tasks come from partial samples + bundled data and may be underestimates -- treat their flags with suspicion.

| task | arcid | sample_n | status |
|---|---|---|---|
| task076 | 36d67576 | 111 | hard_timeout_after_111 |
| task157 | 6a1e5592 | 15 | hard_timeout_after_15 |
| task216 | 8efcae92 | 234 | timeout_after_234 |
| task233 | 97a05b5b | 50 | already_handled|timeout_after_50 |
| task264 | a8c38be5 | 57 | timeout_after_57 |
| task324 | d07ae81c | 193 | hard_timeout_after_193 |
| task330 | d2abd087 | 69 | timeout_after_69 |
