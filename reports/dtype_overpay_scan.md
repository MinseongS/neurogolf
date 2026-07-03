# fp32 dtype-overpayment scan (global value-range audit)

Scanned **326** deployed nets with counted memory>0 (manifest mem==0 skipped). Errors: **0**.

**Total potential delta_points (headline, FP16_SAFE+U8_CANDIDATE recasts): +9.688** across all tasks.

Headline savings EXCLUDE `PRODUCER_BOUND` tensors (producer eats the free fp32 graph input directly — recast is net-worse; needs producer-replacement surgery). PRODUCER_BOUND would-be bytes across all tasks: 184044.

> **Observed-integer on bundled examples is NECESSARY, not SUFFICIENT.** A plane can be integer/binary on every bundled example yet exceed the range on fresh arc-gen instances. Every adoption still needs per-task proof + fresh arc-gen gating (`reports/scripts/fresh_verify.py`), and bit-identity vs the incumbent `networks/taskNNN.onnx`.

## Counted fp32 tensor classes (tensor-level tally)

| class | count | meaning |
|---|---|---|
| FP16_SAFE | 155 | integer, |v|<=2048, no dilated-MaxPool; save bytes/2 (u8 blocked by einsum/topk) |
| U8_CANDIDATE | 677 | integer 0..255, no einsum/topk/dilated-MaxPool; save 3*bytes/4 |
| PRODUCER_BOUND | 764 | recastable range BUT producer consumes free input directly; excluded from headline |
| BLOCKED | 0 | integer&in-range but dilated-MaxPool consumer forbids any recast |
| FLOOR | 213 | non-integer OR |v|>2048; genuinely needs fp32 |

## Top-30 tasks by delta_points (175 have delta>0)

| task | mem | params | cur pts | save B | new pts | delta | fp16 logged? | main tensor (class, dims, saveB, max) |
|---|---|---|---|---|---|---|---|---|
| 184 | 2340 | 63 | 17.2155 | 900 | 17.6848 | +0.4693 | yes | safe_name_20 (FP16_SAFE, [1, 1, 3, 30], 180B, max=1.0) |
| 84 | 1817 | 78 | 17.453 | 696 | 17.9108 | +0.4577 | no | updates (U8_CANDIDATE, [1, 5, 2, 21], 630B, max=1.0) |
| 234 | 3744 | 64 | 16.7551 | 1387 | 17.2081 | +0.4529 | no | rsel (FP16_SAFE, [3, 30], 180B, max=1.0) |
| 126 | 590 | 15 | 18.5948 | 186 | 18.9621 | +0.3674 | no | updates (U8_CANDIDATE, [2, 30], 180B, max=1.0) |
| 292 | 252 | 182 | 18.927 | 126 | 19.2699 | +0.3429 | no | updates (U8_CANDIDATE, [1, 2, 3, 7], 126B, max=1.0) |
| 129 | 80 | 0 | 20.618 | 20 | 20.9057 | +0.2877 | no | winner (FP16_SAFE, [1, 10, 1, 1], 20B, max=1.0) |
| 289 | 618 | 124 | 18.3907 | 183 | 18.6739 | +0.2832 | yes | mask_float (FP16_SAFE, [30, 3], 180B, max=1.0) |
| 188 | 869 | 149 | 18.0744 | 246 | 18.351 | +0.2766 | yes | row_mask (FP16_SAFE, [1, 1, 30, 1], 60B, max=1.0) |
| 255 | 14663 | 293 | 15.3871 | 3256 | 15.6327 | +0.2455 | yes | z_top_edge (U8_CANDIDATE, [1, 1, 4, 30], 360B, max=1.0) |
| 181 | 201 | 167 | 19.0919 | 80 | 19.337 | +0.2451 | no | updates (FP16_SAFE, [4, 1, 3, 3], 72B, max=1.0) |
| 174 | 6973 | 142 | 16.13 | 1433 | 16.3549 | +0.2249 | yes | mismatch_f (U8_CANDIDATE, [1, 10, 1, 10], 300B, max=1.0) |
| 202 | 3253 | 34 | 16.9023 | 660 | 17.1264 | +0.2241 | yes | den_r1 (U8_CANDIDATE, [1, 30, 1], 90B, max=30.0) |
| 320 | 448 | 322 | 18.3536 | 144 | 18.5606 | +0.2070 | yes | updates (FP16_SAFE, [1, 2, 9, 4], 144B, max=9.0) |
| 106 | 452 | 174 | 18.5606 | 108 | 18.75 | +0.1894 | no | cast_14 (U8_CANDIDATE, [1, 1, 6, 6], 108B, max=9.0) |
| 203 | 408 | 1 | 18.9863 | 66 | 19.1623 | +0.1760 | no | target (U8_CANDIDATE, [1, 10], 30B, max=72.0) |
| 40 | 240 | 134 | 19.0757 | 60 | 19.2506 | +0.1749 | no | color_rows (FP16_SAFE, [3, 10, 1, 1], 60B, max=1.0) |
| 364 | 18500 | 1131 | 15.1151 | 2640 | 15.2596 | +0.1444 | yes | seedV (FP16_SAFE, [1, 1, 20, 22], 880B, max=1.0) |
| 353 | 272 | 62 | 19.1889 | 44 | 19.3301 | +0.1413 | yes | rows_f (U8_CANDIDATE, [4, 1], 12B, max=13.0) |
| 153 | 741 | 54 | 18.3217 | 97 | 18.4518 | +0.1301 | yes | presence_nonzero (U8_CANDIDATE, [1, 9, 1, 1], 27B, max=1.0) |
| 2 | 6100 | 589 | 16.1918 | 800 | 16.3192 | +0.1274 | no | t (FP16_SAFE, [1, 1, 20, 20], 800B, max=1.0) |
| 280 | 3363 | 783 | 16.6701 | 460 | 16.7877 | +0.1176 | no | row_terms (FP16_SAFE, [5, 20], 200B, max=1.0) |
| 25 | 11520 | 144 | 15.6357 | 1282 | 15.7522 | +0.1164 | yes | pos_onehot (FP16_SAFE, [1, 4, 30], 240B, max=1.0) |
| 379 | 9309 | 133 | 15.8471 | 1032 | 15.9628 | +0.1157 | yes | cyperrowH30 (U8_CANDIDATE, [1, 1, 30, 1], 90B, max=20.0) |
| 336 | 755 | 1186 | 17.429 | 208 | 17.5424 | +0.1133 | yes | updates (FP16_SAFE, [2, 52], 208B, max=24.0) |
| 375 | 108 | 242 | 19.1421 | 35 | 19.2474 | +0.1054 | no | channel (FP16_SAFE, [1, 10, 1, 1], 20B, max=1.0) |
| 251 | 2772 | 272 | 16.9791 | 288 | 17.0785 | +0.0994 | yes | non (FP16_SAFE, [1, 1, 12, 12], 288B, max=1.0) |
| 391 | 135 | 24 | 19.9311 | 15 | 20.0302 | +0.0991 | no | tv (U8_CANDIDATE, [5], 15B, max=170.0) |
| 300 | 474 | 66 | 18.7084 | 47 | 18.7995 | +0.0911 | yes | selector_f (FP16_SAFE, [1, 10, 1, 1], 20B, max=1.0) |
| 259 | 853 | 52 | 18.1921 | 75 | 18.2786 | +0.0865 | yes | cv_f (U8_CANDIDATE, [1, 1, 3, 3], 27B, max=9.0) |
| 340 | 5812 | 279 | 16.2854 | 480 | 16.3675 | +0.0821 | yes | tc_cc (U8_CANDIDATE, [1, 1, 1, 30], 90B, max=2.0) |

## PRODUCER_BOUND seam (separate — needs producer surgery, not recast) — top 15

| task | mem | producer-bound would-be bytes | note |
|---|---|---|---|
| 233 | 32796 | 5052 | producers: Conv |
| 118 | 24722 | 4200 | producers: Slice |
| 133 | 27333 | 3300 | producers: Conv,Einsum |
| 286 | 21090 | 3185 | producers: Conv,Einsum |
| 205 | 11598 | 3180 | producers: Conv |
| 44 | 11050 | 3000 | producers: Slice |
| 18 | 29807 | 2900 | producers: Conv,ReduceMax,ReduceSum |
| 54 | 20091 | 2900 | producers: Conv,Einsum,ReduceSum |
| 319 | 18645 | 2900 | producers: Einsum,ReduceMax,ReduceSum |
| 173 | 17534 | 2880 | producers: Conv,ReduceMax |
| 80 | 10109 | 2790 | producers: Conv |
| 285 | 24684 | 2790 | producers: Conv,Slice |
| 366 | 30983 | 2708 | producers: Conv,Einsum |
| 74 | 9000 | 2700 | producers: Conv |
| 198 | 13428 | 2700 | producers: Conv |
