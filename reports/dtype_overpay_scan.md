# fp32 dtype-overpayment scan (global value-range audit)

Scanned **323** deployed nets with counted memory>0 (manifest mem==0 skipped). Errors: **1**.

**Total potential delta_points (headline, FP16_SAFE+U8_CANDIDATE recasts): +7.349** across all tasks.

Headline savings EXCLUDE `PRODUCER_BOUND` tensors (producer eats the free fp32 graph input directly — recast is net-worse; needs producer-replacement surgery). PRODUCER_BOUND would-be bytes across all tasks: 166420.

> **Observed-integer on bundled examples is NECESSARY, not SUFFICIENT.** A plane can be integer/binary on every bundled example yet exceed the range on fresh arc-gen instances. Every adoption still needs per-task proof + fresh arc-gen gating (`reports/scripts/fresh_verify.py`), and bit-identity vs the incumbent `networks/taskNNN.onnx`.

## Counted fp32 tensor classes (tensor-level tally)

| class | count | meaning |
|---|---|---|
| FP16_SAFE | 159 | integer, |v|<=2048, no dilated-MaxPool; save bytes/2 (u8 blocked by einsum/topk) |
| U8_CANDIDATE | 583 | integer 0..255, no einsum/topk/dilated-MaxPool; save 3*bytes/4 |
| PRODUCER_BOUND | 795 | recastable range BUT producer consumes free input directly; excluded from headline |
| BLOCKED | 0 | integer&in-range but dilated-MaxPool consumer forbids any recast |
| FLOOR | 280 | non-integer OR |v|>2048; genuinely needs fp32 |

## Top-30 tasks by delta_points (162 have delta>0)

| task | mem | params | cur pts | save B | new pts | delta | fp16 logged? | main tensor (class, dims, saveB, max) |
|---|---|---|---|---|---|---|---|---|
| 84 | 1817 | 78 | 17.453 | 696 | 17.9108 | +0.4577 | no | updates (U8_CANDIDATE, [1, 5, 2, 21], 630B, max=1.0) |
| 126 | 590 | 15 | 18.5948 | 186 | 18.9621 | +0.3674 | yes | updates (U8_CANDIDATE, [2, 30], 180B, max=1.0) |
| 184 | 1920 | 121 | 17.3788 | 540 | 17.6861 | +0.3073 | yes | col_onehot (FP16_SAFE, [1, 1, 3, 30], 180B, max=1.0) |
| 129 | 80 | 0 | 20.618 | 20 | 20.9057 | +0.2877 | yes | winner (FP16_SAFE, [1, 10, 1, 1], 20B, max=1.0) |
| 289 | 618 | 124 | 18.3907 | 183 | 18.6739 | +0.2832 | yes | mask_float (FP16_SAFE, [30, 3], 180B, max=1.0) |
| 188 | 869 | 149 | 18.0744 | 246 | 18.351 | +0.2766 | yes | row_mask (FP16_SAFE, [1, 1, 30, 1], 60B, max=1.0) |
| 181 | 201 | 167 | 19.0919 | 80 | 19.337 | +0.2451 | no | updates (FP16_SAFE, [4, 1, 3, 3], 72B, max=1.0) |
| 202 | 3253 | 34 | 16.9023 | 660 | 17.1264 | +0.2241 | yes | den_r1 (U8_CANDIDATE, [1, 30, 1], 90B, max=30.0) |
| 320 | 448 | 322 | 18.3536 | 144 | 18.5606 | +0.2070 | yes | updates (FP16_SAFE, [1, 2, 9, 4], 144B, max=9.0) |
| 106 | 452 | 174 | 18.5606 | 108 | 18.75 | +0.1894 | no | cast_14 (U8_CANDIDATE, [1, 1, 6, 6], 108B, max=9.0) |
| 25 | 10132 | 104 | 15.7663 | 1504 | 15.9253 | +0.1589 | yes | left (U8_CANDIDATE, [1, 4, 30], 360B, max=1.0) |
| 336 | 764 | 1056 | 17.4934 | 208 | 17.6148 | +0.1214 | yes | updates (FP16_SAFE, [2, 52], 208B, max=100.0) |
| 332 | 389 | 172 | 18.6703 | 63 | 18.7894 | +0.1191 | yes | crop (U8_CANDIDATE, [1, 1, 1, 20], 60B, max=3.0) |
| 280 | 3363 | 783 | 16.6701 | 460 | 16.7877 | +0.1176 | no | row_terms (FP16_SAFE, [5, 20], 200B, max=1.0) |
| 234 | 2869 | 65 | 17.0159 | 313 | 17.1287 | +0.1128 | yes | g0 (FP16_SAFE, [1, 10], 20B, max=1.0) |
| 375 | 119 | 208 | 19.21 | 34 | 19.3198 | +0.1098 | no | channel (FP16_SAFE, [1, 10, 1, 1], 20B, max=1.0) |
| 251 | 2772 | 272 | 16.9791 | 288 | 17.0785 | +0.0994 | yes | non (FP16_SAFE, [1, 1, 12, 12], 288B, max=1.0) |
| 391 | 135 | 24 | 19.9311 | 15 | 20.0302 | +0.0991 | no | tv (U8_CANDIDATE, [5], 15B, max=170.0) |
| 153 | 724 | 54 | 18.3433 | 70 | 18.4376 | +0.0943 | yes | selected_mask (U8_CANDIDATE, [1, 1, 3, 3], 27B, max=1.0) |
| 300 | 467 | 61 | 18.7309 | 47 | 18.8241 | +0.0932 | yes | selector_f (FP16_SAFE, [1, 10, 1, 1], 20B, max=1.0) |
| 254 | 650 | 37 | 18.4677 | 60 | 18.5591 | +0.0914 | yes | slice_6 (U8_CANDIDATE, [1, 9], 27B, max=9.0) |
| 259 | 844 | 50 | 18.2043 | 75 | 18.2919 | +0.0876 | yes | cv_f (U8_CANDIDATE, [1, 1, 3, 3], 27B, max=9.0) |
| 255 | 8625 | 353 | 15.8975 | 706 | 15.9794 | +0.0819 | yes | safe_name_43 (U8_CANDIDATE, [1, 30], 90B, max=26.0) |
| 398 | 1666 | 1193 | 17.0418 | 210 | 17.1181 | +0.0763 | yes | row3 (U8_CANDIDATE, [10, 5], 150B, max=1.0) |
| 20 | 1176 | 170 | 17.7951 | 97 | 17.8699 | +0.0748 | yes | row_blank_min (U8_CANDIDATE, [1, 1, 8, 1], 24B, max=1.0) |
| 78 | 1260 | 33 | 17.8353 | 90 | 17.9074 | +0.0721 | yes | top_count (U8_CANDIDATE, [1, 1, 1, 10], 30B, max=5.0) |
| 65 | 582 | 59 | 18.537 | 41 | 18.6031 | +0.0661 | no | dot_mask_f (FP16_SAFE, [1, 10], 20B, max=1.0) |
| 119 | 1243 | 120 | 17.7826 | 81 | 17.8438 | +0.0613 | yes | scatter_updates (U8_CANDIDATE, [20], 60B, max=1.0) |
| 66 | 9955 | 166 | 15.7776 | 600 | 15.8387 | +0.0611 | yes | cy_norm32 (U8_CANDIDATE, [1, 4, 30], 360B, max=10.0) |
| 55 | 3030 | 62 | 16.9634 | 180 | 17.0234 | +0.0600 | yes | rowband (U8_CANDIDATE, [1, 1, 30, 1], 90B, max=2.0) |

## PRODUCER_BOUND seam (separate — needs producer surgery, not recast) — top 15

| task | mem | producer-bound would-be bytes | note |
|---|---|---|---|
| 233 | 32796 | 5052 | producers: Conv |
| 133 | 20510 | 4520 | producers: Conv,ReduceMax,ReduceSum |
| 205 | 9498 | 3390 | producers: Conv,Einsum,Slice |
| 286 | 21090 | 3185 | producers: Conv,Einsum |
| 54 | 20091 | 2900 | producers: Conv,Einsum,ReduceSum |
| 173 | 13307 | 2880 | producers: Conv,ReduceMax |
| 349 | 13860 | 2880 | producers: Gather,ReduceMax |
| 80 | 9387 | 2790 | producers: Conv |
| 285 | 24684 | 2790 | producers: Conv,Slice |
| 366 | 30983 | 2708 | producers: Conv,Einsum |
| 74 | 9000 | 2700 | producers: Conv |
| 198 | 11305 | 2700 | producers: Conv |
| 364 | 15860 | 2640 | producers: Slice |
| 187 | 5000 | 2500 | producers: Conv |
| 367 | 15300 | 2400 | producers: Slice |
