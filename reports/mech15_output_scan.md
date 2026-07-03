# Mechanism 15 output-decomposition scan (signed-channel priority overlay)

Structural, bundled-only (train+test), numpy screen for playbook mechanism 15.
**NECESSARY-NOT-SUFFICIENT**: bundled decomposition can under-count rects and
miss data-dependent colours the generator emits. A qualifier still needs
per-task mechanism verification (render generator outputs) before adopting.

- RMAX rects/example = 8; headroom floor = 1500B; min cost = 1200; max points = 20.0
- Excluded (S11 killed/floored + hits + walls): [2, 4, 18, 41, 44, 54, 76, 84, 92, 118, 133, 162, 177, 209, 233, 234, 285, 335, 366, 370]
- **Structural qualifiers after filters: 67**

Roster class: CONSTANT = literal fill colours identical across examples (plain signed W). DATA_DEPENDENT = HARDER (colour must be read from input via tiny einsum, not discarded). Overlap: OVERLAP needs the signed-priority trick; DISJOINT = plain signed routing.

> CAVEAT — the overlap flag is a WEAK proxy: it intersects the bounding boxes of the greedy rects of the *resolved final output*, so a true paint-order overlap that is already resolved in the output reads as DISJOINT. All 3 known hits (092/234/335) show DISJOINT here. Do not use it to reject a candidate; use it only as a hint that OVERLAP tasks are definitely in the priority-trick regime.

> The 3 hand-found hits 092/234/335 all pass this structural screen (092 R_max=8 DATA_DEPENDENT, 234 R_max=5 DATA_DEPENDENT, 335 R_max=2 CONSTANT) — that is the validation that the screen catches the mechanism. They are excluded below only because they already use it.

| task | cost | mem | params | pts | R_max | roster | overlap | shape | headroom | method |
|---|---|---|---|---|---|---|---|---|---|---|
| 367 | 19525 | 14800 | 4725 | 15.12 | 4 | CONSTANT | DISJOINT | delta | 18025 | custom:task367 |
| 255 | 14956 | 14663 | 293 | 15.39 | 5 | CONSTANT | DISJOINT | delta | 13456 | custom:task255 |
| 101 | 14447 | 13573 | 874 | 15.42 | 6 | CONSTANT | DISJOINT | delta | 12947 | custom:task101 |
| 145 | 10492 | 9244 | 1248 | 15.74 | 4 | CONSTANT | DISJOINT | delta | 8992 | custom:task145 |
| 204 | 10232 | 9780 | 452 | 15.77 | 5 | CONSTANT | DISJOINT | delta | 8732 | custom:task204 |
| 66 | 10121 | 9955 | 166 | 15.78 | 3 | CONSTANT | DISJOINT | delta | 8621 | custom:task066 |
| 64 | 9986 | 9852 | 134 | 15.79 | 5 | DATA_DEPENDENT | DISJOINT | delta | 8486 | custom:task064 |
| 182 | 6442 | 6345 | 97 | 16.23 | 6 | DATA_DEPENDENT | DISJOINT | delta | 4942 | custom:task182 |
| 23 | 6412 | 6163 | 249 | 16.23 | 7 | CONSTANT | DISJOINT | delta | 4912 | custom:task023 |
| 165 | 6058 | 5828 | 230 | 16.29 | 7 | DATA_DEPENDENT | DISJOINT | delta | 4558 | custom:task165 |
| 174 | 5889 | 5743 | 146 | 16.32 | 5 | DATA_DEPENDENT | DISJOINT | whole | 4389 | custom:task174 |
| 148 | 5869 | 5759 | 110 | 16.32 | 8 | CONSTANT | DISJOINT | delta | 4369 | custom:task148 |
| 8 | 4913 | 4809 | 104 | 16.50 | 6 | CONSTANT | DISJOINT | delta | 3413 | custom:task008 |
| 208 | 4726 | 4612 | 114 | 16.54 | 4 | DATA_DEPENDENT | DISJOINT | delta | 3226 | custom:task208 |
| 378 | 4449 | 4332 | 117 | 16.60 | 6 | DATA_DEPENDENT | DISJOINT | delta | 2949 | custom:task378 |
| 265 | 4362 | 4328 | 34 | 16.62 | 5 | CONSTANT | DISJOINT | delta | 2862 | custom:task265 |
| 206 | 4186 | 4110 | 76 | 16.66 | 5 | DATA_DEPENDENT | DISJOINT | delta | 2686 | custom:task206 |
| 280 | 4146 | 3363 | 783 | 16.67 | 6 | CONSTANT | DISJOINT | delta | 2646 | custom:task280 |
| 132 | 4089 | 3990 | 99 | 16.68 | 6 | DATA_DEPENDENT | DISJOINT | delta | 2589 | custom:task132 |
| 50 | 3911 | 3825 | 86 | 16.73 | 3 | CONSTANT | DISJOINT | delta | 2411 | custom:task050 |
| 333 | 3227 | 2792 | 435 | 16.92 | 4 | DATA_DEPENDENT | DISJOINT | delta | 1727 | custom:task333 |
| 250 | 3225 | 3164 | 61 | 16.92 | 8 | CONSTANT | DISJOINT | delta | 1725 | custom:task250 |
| 42 | 3110 | 2792 | 318 | 16.96 | 4 | CONSTANT | DISJOINT | delta | 1610 | custom:task042 |
| 55 | 3092 | 3030 | 62 | 16.96 | 5 | CONSTANT | DISJOINT | delta | 1592 | custom:task055 |
| 390 | 3090 | 2623 | 467 | 16.96 | 8 | CONSTANT | DISJOINT | delta | 1590 | custom:task390 |
| 90 | 3072 | 2990 | 82 | 16.97 | 1 | CONSTANT | DISJOINT | delta | 1572 | custom:task090 |
| 224 | 2831 | 2460 | 371 | 17.05 | 4 | DATA_DEPENDENT | DISJOINT | delta | 1331 | custom:task224 |
| 394 | 2800 | 1871 | 929 | 17.06 | 7 | DATA_DEPENDENT | DISJOINT | whole | 1300 | custom:task394 |
| 397 | 2782 | 2582 | 200 | 17.07 | 3 | CONSTANT | DISJOINT | delta | 1282 | custom:task397 |
| 246 | 2753 | 2334 | 419 | 17.08 | 2 | CONSTANT | DISJOINT | delta | 1253 | custom:task246 |
| 354 | 2751 | 2674 | 77 | 17.08 | 3 | DATA_DEPENDENT | DISJOINT | delta | 1251 | custom:task354 |
| 355 | 2708 | 2688 | 20 | 17.10 | 1 | DATA_DEPENDENT | DISJOINT | whole | 1208 | custom:task355 |
| 71 | 2698 | 2643 | 55 | 17.10 | 8 | DATA_DEPENDENT | DISJOINT | delta | 1198 | custom:task071 |
| 102 | 2527 | 2414 | 113 | 17.17 | 3 | CONSTANT | DISJOINT | delta | 1027 | custom:task102 |
| 190 | 2527 | 2280 | 247 | 17.17 | 8 | DATA_DEPENDENT | DISJOINT | delta | 1027 | custom:task190 |
| 184 | 2403 | 2340 | 63 | 17.22 | 8 | DATA_DEPENDENT | DISJOINT | whole | 903 | custom:task184 |
| 30 | 2396 | 2248 | 148 | 17.22 | 8 | CONSTANT | DISJOINT | delta | 896 | custom:task030 |
| 281 | 2386 | 2321 | 65 | 17.22 | 4 | DATA_DEPENDENT | DISJOINT | delta | 886 | custom:task281 |
| 374 | 2304 | 2262 | 42 | 17.26 | 3 | CONSTANT | DISJOINT | delta | 804 | custom:task374 |
| 22 | 2302 | 2004 | 298 | 17.26 | 7 | DATA_DEPENDENT | DISJOINT | whole | 802 | custom:task022 |
| 134 | 2284 | 2250 | 34 | 17.27 | 8 | DATA_DEPENDENT | DISJOINT | whole | 784 | custom:task134 |
| 256 | 2279 | 2223 | 56 | 17.27 | 6 | CONSTANT | DISJOINT | delta | 779 | custom:task256 |
| 35 | 2261 | 2132 | 129 | 17.28 | 7 | DATA_DEPENDENT | DISJOINT | delta | 761 | custom:task035 |
| 168 | 2239 | 2096 | 143 | 17.29 | 8 | DATA_DEPENDENT | DISJOINT | delta | 739 | custom:task168 |
| 36 | 2177 | 2140 | 37 | 17.31 | 8 | DATA_DEPENDENT | DISJOINT | whole | 677 | custom:task036 |
| 273 | 2160 | 2109 | 51 | 17.32 | 2 | CONSTANT | DISJOINT | delta | 660 | custom:task273 |
| 381 | 1963 | 1908 | 55 | 17.42 | 4 | CONSTANT | DISJOINT | delta | 463 | custom:task381 |
| 185 | 1961 | 1651 | 310 | 17.42 | 7 | DATA_DEPENDENT | DISJOINT | whole | 461 | custom:task185 |
| 336 | 1941 | 755 | 1186 | 17.43 | 3 | CONSTANT | DISJOINT | delta | 441 | custom:task336 |
| 237 | 1901 | 1763 | 138 | 17.45 | 6 | DATA_DEPENDENT | DISJOINT | delta | 401 | custom:task237 |
| 348 | 1896 | 1824 | 72 | 17.45 | 8 | CONSTANT | DISJOINT | delta | 396 | custom:task348 |
| 213 | 1882 | 1833 | 49 | 17.46 | 7 | DATA_DEPENDENT | DISJOINT | whole | 382 | custom:task213 |
| 199 | 1837 | 1749 | 88 | 17.48 | 7 | DATA_DEPENDENT | DISJOINT | delta | 337 | custom:task199 |
| 51 | 1803 | 1702 | 101 | 17.50 | 1 | DATA_DEPENDENT | DISJOINT | delta | 303 | custom:task051 |
| 302 | 1774 | 1216 | 558 | 17.52 | 3 | DATA_DEPENDENT | DISJOINT | delta | 274 | custom:task302 |
| 156 | 1744 | 1697 | 47 | 17.54 | 2 | CONSTANT | DISJOINT | delta | 244 | custom:task156 |
| 45 | 1673 | 1050 | 623 | 17.58 | 2 | DATA_DEPENDENT | DISJOINT | delta | 173 | custom:task045 |
| 226 | 1633 | 1613 | 20 | 17.60 | 3 | CONSTANT | DISJOINT | delta | 133 | custom:task226 |
| 342 | 1585 | 1510 | 75 | 17.63 | 8 | DATA_DEPENDENT | DISJOINT | delta | 85 | custom:task342 |
| 20 | 1526 | 1356 | 170 | 17.67 | 3 | DATA_DEPENDENT | DISJOINT | delta | 26 | custom:task020 |
| 345 | 1495 | 1431 | 64 | 17.69 | 5 | CONSTANT | DISJOINT | delta | -5 | custom:task345 |
| 341 | 1429 | 1394 | 35 | 17.74 | 1 | CONSTANT | DISJOINT | delta | -71 | custom:task341 |
| 356 | 1319 | 1300 | 19 | 17.82 | 4 | CONSTANT | DISJOINT | delta | -181 | custom:task356 |
| 27 | 1310 | 1262 | 48 | 17.82 | 4 | CONSTANT | DISJOINT | delta | -190 | custom:task027 |
| 293 | 1288 | 1247 | 41 | 17.84 | 1 | DATA_DEPENDENT | DISJOINT | delta | -212 | custom:task293 |
| 346 | 1254 | 1224 | 30 | 17.87 | 1 | DATA_DEPENDENT | DISJOINT | whole | -246 | custom:task346 |
| 343 | 1222 | 1147 | 75 | 17.89 | 6 | DATA_DEPENDENT | DISJOINT | delta | -278 | custom:task343 |

## Top-10 by cost — per-example detail

### task367  (cost 19525, CONSTANT, DISJOINT, R_max 4)
- roster union (literal colours): [4]
  - ex0 [delta] R=4 colours=[4]
  - ex1 [delta] R=2 colours=[4]
  - ex2 [delta] R=3 colours=[4]
  - ex3 [delta] R=4 colours=[4]

### task255  (cost 14956, CONSTANT, DISJOINT, R_max 5)
- roster union (literal colours): [3]
  - ex0 [delta] R=3 colours=[3]
  - ex1 [delta] R=3 colours=[3]
  - ex2 [delta] R=2 colours=[3]
  - ex3 [delta] R=5 colours=[3]

### task101  (cost 14447, CONSTANT, DISJOINT, R_max 6)
- roster union (literal colours): [1]
  - ex0 [delta] R=4 colours=[1]
  - ex1 [delta] R=4 colours=[1]
  - ex2 [delta] R=1 colours=[1]
  - ex3 [delta] R=6 colours=[1]

### task145  (cost 10492, CONSTANT, DISJOINT, R_max 4)
- roster union (literal colours): [1, 8]
  - ex0 [delta] R=3 colours=[1, 8]
  - ex1 [delta] R=2 colours=[1, 8]
  - ex2 [delta] R=4 colours=[1, 8]
  - ex3 [delta] R=2 colours=[1, 8]
  - ex4 [delta] R=4 colours=[1, 8]

### task204  (cost 10232, CONSTANT, DISJOINT, R_max 5)
- roster union (literal colours): [2, 7]
  - ex0 [delta] R=3 colours=[2, 7]
  - ex1 [delta] R=2 colours=[2, 7]
  - ex2 [delta] R=5 colours=[2, 7]
  - ex3 [delta] R=3 colours=[2, 7]
  - ex4 [delta] R=2 colours=[2, 7]
  - ex5 [delta] R=4 colours=[2, 7]

### task066  (cost 10121, CONSTANT, DISJOINT, R_max 3)
- roster union (literal colours): [3]
  - ex0 [delta] R=3 colours=[3]
  - ex1 [delta] R=2 colours=[3]
  - ex2 [delta] R=3 colours=[3]
  - ex3 [delta] R=3 colours=[3]

### task064  (cost 9986, DATA_DEPENDENT, DISJOINT, R_max 5)
- roster union (literal colours): [1, 2, 4, 8]
  - ex0 [delta] R=1 colours=[4]
  - ex1 [delta] R=1 colours=[8]
  - ex2 [delta] R=2 colours=[2]
  - ex3 [delta] R=3 colours=[4]
  - ex4 [delta] R=5 colours=[1]

### task182  (cost 6442, DATA_DEPENDENT, DISJOINT, R_max 6)
- roster union (literal colours): [2, 3]
  - ex0 [delta] R=6 colours=[2]
  - ex1 [delta] R=2 colours=[3]
  - ex2 [delta] R=2 colours=[2]
  - ex3 [delta] R=1 colours=[3]
  - ex4 [delta] R=6 colours=[2]

### task023  (cost 6412, CONSTANT, DISJOINT, R_max 7)
- roster union (literal colours): [2, 8]
  - ex0 [delta] R=6 colours=[2, 8]
  - ex1 [delta] R=6 colours=[2, 8]
  - ex2 [delta] R=4 colours=[2, 8]
  - ex3 [delta] R=7 colours=[2, 8]

### task165  (cost 6058, DATA_DEPENDENT, DISJOINT, R_max 7)
- roster union (literal colours): [2, 3, 6, 8]
  - ex0 [delta] R=2 colours=[8]
  - ex1 [delta] R=4 colours=[2]
  - ex2 [delta] R=7 colours=[3]
  - ex3 [delta] R=7 colours=[6]
