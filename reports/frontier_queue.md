# Frontier review queue

Ranked list of sub-20 tasks to human-review for a semantic shortcut into the 20+ band (counted cost <= ~148B).

- Target points at 148B: `20.0028` (= 25 - ln(148))
- `gain_to_20` = target - current points (clamped >=0; tasks >=20 excluded from queue).
- `simplicity` = median_oracle_len(233.5) / oracle_len; >1 = simpler-than-median rule.
- `rank_score` = gain_to_20 * sqrt(simplicity).
- classes abbreviate `compiler_*` as `c:` and `cost_*` as `$:`.

## Summary

- Tasks in queue (sub-20): **350** of 400; already 20+: **50**.
- Current-points bands (queued tasks): 13-16 = **41**, 16-18 = **177**, 18-20 = **132**.
- Total `gain_to_20` available across all 400: **853.95** points.
- Missing oracle files: none (all 400 covered).

## Top 60 by rank_score

| task | pts | mem | params | cost | method | gain | ora_len | simpl | rank | classes(c:/$:) | flags |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 162 | 16.69 | 4024 | 44 | 4068 | task162 | 3.314 | 119 | 1.96 | 4.642 | c:bounded_scan,c:qlinear_uint8,c:s |  |
| 84 | 17.45 | 1817 | 78 | 1895 | task084 | 2.550 | 73 | 3.20 | 4.560 | c:sparse_scatter |  |
| 41 | 17.51 | 1730 | 59 | 1789 | task041 | 2.492 | 73 | 3.20 | 4.457 | c:final_equal_overlay,c:single_con |  |
| 118 | 14.86 | 24722 | 723 | 25445 | task118 | 5.147 | 357 | 0.65 | 4.163 | c:bounded_scan,c:direct_onehot_gat | wall |
| 177 | 16.81 | 3500 | 121 | 3621 | task177 | 3.197 | 138 | 1.69 | 4.159 | c:final_equal_overlay,c:roi_pool_c |  |
| 4 | 16.46 | 5044 | 93 | 5137 | task004 | 3.547 | 182 | 1.28 | 4.018 | c:bounded_scan,c:final_equal_overl | wall |
| 377 | 15.98 | 8111 | 150 | 8261 | task377 | 4.022 | 255 | 0.92 | 3.849 | c:final_equal_overlay,c:single_con |  |
| 29 | 16.38 | 5436 | 89 | 5525 | task029 | 3.620 | 209 | 1.12 | 3.826 | c:bounded_scan,c:direct_output_alg |  |
| 350 | 15.89 | 9012 | 24 | 9036 | task350 | 4.112 | 272 | 0.86 | 3.810 | c:bounded_scan,c:tiny_lut_gather,$ |  |
| 243 | 16.30 | 5112 | 901 | 6013 | task243 | 3.704 | 222 | 1.05 | 3.799 | c:qlinear_uint8,c:single_conv_qlin |  |
| 338 | 15.73 | 10000 | 666 | 10666 | task338 | 4.278 | 304 | 0.77 | 3.749 | c:bounded_scan,c:final_equal_overl |  |
| 398 | 17.04 | 1666 | 1193 | 2859 | task398 | 2.961 | 149 | 1.57 | 3.707 | c:direct_onehot_gather,c:final_equ |  |
| 233 | 14.59 | 32796 | 446 | 33242 | task233 | 5.414 | 499 | 0.47 | 3.704 | c:bounded_scan,c:direct_onehot_gat | frontier_seed |
| 297 | 17.76 | 1350 | 39 | 1389 | task297 | 2.239 | 89 | 2.62 | 3.627 | c:bounded_scan,c:final_equal_overl |  |
| 187 | 15.97 | 5000 | 3322 | 8322 | task187 | 4.029 | 292 | 0.80 | 3.603 | c:bounded_scan,c:final_equal_overl |  |
| 92 | 16.16 | 6690 | 182 | 6872 | task092 | 3.838 | 267 | 0.87 | 3.589 | c:bounded_scan,c:direct_output_alg | frontier_seed |
| 256 | 17.27 | 2223 | 56 | 2279 | task256 | 2.734 | 136 | 1.72 | 3.583 | c:bounded_scan,c:final_equal_overl |  |
| 45 | 17.58 | 1050 | 623 | 1673 | task045 | 2.425 | 107 | 2.18 | 3.583 | c:bounded_scan,c:direct_output_alg |  |
| 14 | 16.66 | 4088 | 83 | 4171 | task014 | 3.339 | 206 | 1.13 | 3.555 | c:bounded_scan,c:final_equal_overl |  |
| 2 | 16.19 | 6100 | 589 | 6689 | task002 | 3.811 | 271 | 0.86 | 3.538 | c:direct_onehot_gather,c:tiny_lut_ | wall |
| 199 | 17.48 | 1749 | 88 | 1837 | task199 | 2.519 | 125 | 1.87 | 3.442 | c:direct_output_algebra,c:einsum_s |  |
| 368 | 16.26 | 6022 | 203 | 6225 | task368 | 3.739 | 277 | 0.84 | 3.433 | c:final_equal_overlay,c:sparse_sca |  |
| 69 | 16.73 | 3848 | 64 | 3912 | task069 | 3.275 | 215 | 1.09 | 3.413 | c:bounded_scan,c:final_equal_overl |  |
| 85 | 16.41 | 4500 | 881 | 5381 | task085 | 3.593 | 261 | 0.89 | 3.399 | c:single_conv_qlinear,$:exact_pres |  |
| 388 | 17.18 | 2278 | 218 | 2496 | task388 | 2.825 | 163 | 1.43 | 3.381 | c:bounded_scan,c:final_equal_overl |  |
| 232 | 17.75 | 0 | 1410 | 1410 | task232 | 2.254 | 106 | 2.20 | 3.346 | c:single_conv_qlinear,c:tiny_lut_g |  |
| 17 | 16.04 | 5997 | 1804 | 7801 | task017 | 3.965 | 329 | 0.71 | 3.340 | c:final_equal_overlay,c:tiny_lut_g | frontier_seed |
| 10 | 18.02 | 730 | 340 | 1070 | task010 | 1.978 | 84 | 2.78 | 3.298 | c:final_equal_overlay,c:tiny_lut_g |  |
| 37 | 16.82 | 3384 | 201 | 3585 | task037 | 3.187 | 222 | 1.05 | 3.269 | c:bounded_scan,c:final_equal_overl |  |
| 359 | 16.74 | 3852 | 15 | 3867 | task359 | 3.263 | 236 | 0.99 | 3.246 | c:bounded_scan,c:final_equal_overl |  |
| 329 | 18.03 | 1029 | 30 | 1059 | task329 | 1.968 | 86 | 2.72 | 3.243 | c:final_equal_overlay,c:sparse_sca | dead_crop |
| 132 | 16.68 | 3990 | 99 | 4089 | task132 | 3.319 | 247 | 0.95 | 3.227 | c:bounded_scan,c:final_equal_overl |  |
| 267 | 18.36 | 724 | 42 | 766 | task267 | 1.644 | 61 | 3.83 | 3.216 | c:final_equal_overlay |  |
| 50 | 16.73 | 3825 | 86 | 3911 | task050 | 3.274 | 243 | 0.96 | 3.210 | c:direct_onehot_gather,$:connectiv |  |
| 246 | 17.08 | 2334 | 419 | 2753 | task246 | 2.923 | 194 | 1.20 | 3.207 | c:final_equal_overlay,c:single_con |  |
| 32 | 18.19 | 0 | 910 | 910 | task032 | 1.816 | 75 | 3.11 | 3.205 | c:single_conv_qlinear,$:exact_pres |  |
| 365 | 16.72 | 3808 | 124 | 3932 | task365 | 3.280 | 245 | 0.95 | 3.202 | c:direct_output_algebra,c:einsum_s |  |
| 91 | 16.98 | 2853 | 175 | 3028 | task091 | 3.018 | 211 | 1.11 | 3.175 | c:final_equal_overlay,c:single_con |  |
| 397 | 17.07 | 2582 | 200 | 2782 | task397 | 2.934 | 203 | 1.15 | 3.146 | c:final_equal_overlay,c:single_con |  |
| 278 | 16.39 | 5436 | 47 | 5483 | task278 | 3.612 | 308 | 0.76 | 3.145 | c:bounded_scan,c:qlinear_uint8,c:s |  |
| 354 | 17.08 | 2674 | 77 | 2751 | task354 | 2.922 | 204 | 1.14 | 3.127 | c:bounded_scan,c:final_equal_overl |  |
| 345 | 17.69 | 1431 | 64 | 1495 | task345 | 2.313 | 128 | 1.82 | 3.124 | c:tiny_lut_gather,$:connectivity_w | dead_crop |
| 381 | 17.42 | 1908 | 55 | 1963 | task381 | 2.585 | 160 | 1.46 | 3.123 | c:direct_onehot_gather,$:connectiv |  |
| 171 | 18.19 | 0 | 910 | 910 | task171 | 1.816 | 79 | 2.96 | 3.123 | c:single_conv_qlinear,$:exact_pres |  |
| 295 | 17.62 | 1557 | 47 | 1604 | task295 | 2.383 | 138 | 1.69 | 3.100 | c:final_equal_overlay,$:exact_pres |  |
| 110 | 16.33 | 264 | 5547 | 5811 | task110 | 3.670 | 329 | 0.71 | 3.092 | c:bounded_scan,c:final_equal_overl |  |
| 335 | 17.30 | 2106 | 109 | 2215 | task335 | 2.706 | 179 | 1.30 | 3.090 | c:bounded_scan,c:direct_output_alg | dead_crop |
| 22 | 17.26 | 2004 | 298 | 2302 | task022 | 2.744 | 185 | 1.26 | 3.083 | c:direct_output_algebra,c:einsum_s |  |
| 131 | 16.71 | 3312 | 675 | 3987 | task131 | 3.294 | 268 | 0.87 | 3.074 | c:bounded_scan,c:sparse_scatter,$: |  |
| 372 | 18.43 | 0 | 710 | 710 | task372 | 1.568 | 61 | 3.83 | 3.068 | c:single_conv_qlinear,$:exact_pres |  |
| 70 | 16.99 | 2977 | 31 | 3008 | task070 | 3.012 | 228 | 1.02 | 3.048 | c:bounded_scan,c:sparse_scatter,c: |  |
| 145 | 15.74 | 9244 | 1248 | 10492 | task145 | 4.261 | 461 | 0.51 | 3.033 | c:bounded_scan,c:final_equal_overl |  |
| 237 | 17.45 | 1763 | 138 | 1901 | task237 | 2.553 | 166 | 1.41 | 3.028 | c:bounded_scan,c:final_equal_overl |  |
| 275 | 17.39 | 1374 | 637 | 2011 | task275 | 2.609 | 176 | 1.33 | 3.005 | c:bounded_scan,c:direct_output_alg |  |
| 396 | 15.82 | 7842 | 1813 | 9655 | task396 | 4.178 | 452 | 0.52 | 3.003 | c:bounded_scan,c:final_equal_overl | frontier_seed |
| 12 | 17.28 | 2071 | 188 | 2259 | task012 | 2.725 | 194 | 1.20 | 2.990 | c:final_equal_overlay,c:sparse_sca |  |
| 19 | 17.06 | 2709 | 89 | 2798 | task019 | 2.939 | 228 | 1.02 | 2.975 | c:bounded_scan,c:final_equal_overl |  |
| 9 | 16.14 | 6929 | 95 | 7024 | task009 | 3.860 | 394 | 0.59 | 2.971 | c:bounded_scan,c:final_equal_overl |  |
| 286 | 14.92 | 21090 | 2881 | 23971 | task286 | 5.087 | 694 | 0.34 | 2.951 | c:bounded_scan,c:final_equal_overl |  |
| 310 | 16.90 | 3208 | 79 | 3287 | task310 | 3.100 | 258 | 0.91 | 2.950 | c:direct_output_algebra,c:einsum_s |  |

## Frontier seeds (named in HIGH_SCORE_FRONTIER.md)

Shown regardless of rank; some may already be 20+ (then absent from the main queue).

| task | pts | mem | params | cost | method | gain | ora_len | simpl | rank | classes(c:/$:) | flags |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 233 | 14.59 | 32796 | 446 | 33242 | task233 | 5.414 | 499 | 0.47 | 3.704 | c:bounded_scan,c:direct_onehot_gat | frontier_seed |
| 92 | 16.16 | 6690 | 182 | 6872 | task092 | 3.838 | 267 | 0.87 | 3.589 | c:bounded_scan,c:direct_output_alg | frontier_seed |
| 17 | 16.04 | 5997 | 1804 | 7801 | task017 | 3.965 | 329 | 0.71 | 3.340 | c:final_equal_overlay,c:tiny_lut_g | frontier_seed |
| 396 | 15.82 | 7842 | 1813 | 9655 | task396 | 4.178 | 452 | 0.52 | 3.003 | c:bounded_scan,c:final_equal_overl | frontier_seed |
| 222 | 16.00 | 5136 | 2952 | 8088 | task222 | 4.001 | 478 | 0.49 | 2.796 | c:bounded_scan,c:final_equal_overl | frontier_seed |
| 74 | 15.89 | 9000 | 50 | 9050 | task074 | 4.113 | 512 | 0.46 | 2.778 | c:final_equal_overlay,c:single_con | frontier_seed |
| 55 | 16.96 | 3030 | 62 | 3092 | task055 | 3.039 | 303 | 0.77 | 2.668 | c:bounded_scan,c:final_equal_overl | frontier_seed |
| 204 | 15.77 | 9780 | 452 | 10232 | task204 | 4.236 | 591 | 0.40 | 2.663 | c:bounded_scan,c:qlinear_uint8,c:s | frontier_seed |
| 64 | 15.79 | 9852 | 134 | 9986 | task064 | 4.212 | 752 | 0.31 | 2.347 | c:bounded_scan,c:final_equal_overl | frontier_seed |
| 349 | 15.29 | 15300 | 1191 | 16491 | task349 | 4.713 | 1164 | 0.20 | 2.111 | c:bounded_scan,c:final_equal_overl | frontier_seed |
| 367 | 15.12 | 14800 | 4725 | 19525 | task367 | 4.882 | 1389 | 0.17 | 2.002 | c:qlinear_uint8,c:single_conv_qlin | frontier_seed |
| 328 | 16.18 | 4835 | 1940 | 6775 | task328 | 3.824 | 1083 | 0.22 | 1.776 | c:final_equal_overlay,c:tiny_lut_g | frontier_seed |
| 208 | 16.54 | 4612 | 114 | 4726 | task208 | 3.464 | 924 | 0.25 | 1.741 | c:bounded_scan,c:direct_output_alg | frontier_seed |
| 202 | 16.90 | 3253 | 34 | 3287 | task202 | 3.100 | 788 | 0.30 | 1.688 | c:bounded_scan,c:final_equal_overl | frontier_seed |
| 138 | 15.55 | 12215 | 462 | 12677 | task138 | 4.450 | 2323 | 0.10 | 1.411 | c:bounded_scan,c:final_equal_overl | frontier_seed |
| 25 | 15.64 | 11520 | 144 | 11664 | task025 | 4.367 | 3378 | 0.07 | 1.148 | c:bounded_scan,c:direct_output_alg | frontier_seed |
| 366 | 14.64 | 30983 | 576 | 31559 | task366 | 5.362 | 6846 | 0.03 | 0.990 | c:bounded_scan,c:final_equal_overl | frontier_seed |

