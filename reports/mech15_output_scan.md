# Mechanism 15 output-decomposition scan (signed-channel priority overlay)

Structural, bundled-only (train+test), numpy screen for playbook mechanism 15.
**NECESSARY-NOT-SUFFICIENT**: bundled decomposition can under-count rects and
miss data-dependent colours the generator emits. A qualifier still needs
per-task mechanism verification (render generator outputs) before adopting.

- RMAX rects/example = 8; headroom floor = 1500B; min cost = 1200; max points = 20.0
- Excluded (S11 killed/floored + hits + walls): [2, 4, 18, 41, 44, 54, 76, 84, 92, 118, 133, 162, 177, 209, 233, 234, 285, 335, 366, 370]
- **Structural qualifiers after filters: 65**

Roster class: CONSTANT = literal fill colours identical across examples (plain signed W). DATA_DEPENDENT = HARDER (colour must be read from input via tiny einsum, not discarded). Overlap: OVERLAP needs the signed-priority trick; DISJOINT = plain signed routing.

> CAVEAT — the overlap flag is a WEAK proxy: it intersects the bounding boxes of the greedy rects of the *resolved final output*, so a true paint-order overlap that is already resolved in the output reads as DISJOINT. All 3 known hits (092/234/335) show DISJOINT here. Do not use it to reject a candidate; use it only as a hint that OVERLAP tasks are definitely in the priority-trick regime.

> The 3 hand-found hits 092/234/335 all pass this structural screen (092 R_max=8 DATA_DEPENDENT, 234 R_max=5 DATA_DEPENDENT, 335 R_max=2 CONSTANT) — that is the validation that the screen catches the mechanism. They are excluded below only because they already use it.

| task | cost | mem | params | pts | R_max | roster | overlap | shape | headroom | method |
|---|---|---|---|---|---|---|---|---|---|---|
| 367 | 15890 | 15300 | 590 | 15.33 | 4 | CONSTANT | DISJOINT | delta | 14390 | custom:task367 |
| 101 | 14447 | 13573 | 874 | 15.42 | 6 | CONSTANT | DISJOINT | delta | 12947 | custom:task101 |
| 145 | 10492 | 9244 | 1248 | 15.74 | 4 | CONSTANT | DISJOINT | delta | 8992 | custom:task145 |
| 204 | 10232 | 9780 | 452 | 15.77 | 5 | CONSTANT | DISJOINT | delta | 8732 | custom:task204 |
| 66 | 10121 | 9955 | 166 | 15.78 | 3 | CONSTANT | DISJOINT | delta | 8621 | custom:task066 |
| 64 | 9986 | 9852 | 134 | 15.79 | 5 | DATA_DEPENDENT | DISJOINT | delta | 8486 | custom:task064 |
| 255 | 8978 | 8625 | 353 | 15.90 | 5 | CONSTANT | DISJOINT | delta | 7478 | custom:task255 |
| 23 | 6412 | 6163 | 249 | 16.23 | 7 | CONSTANT | DISJOINT | delta | 4912 | custom:task023 |
| 182 | 6100 | 6024 | 76 | 16.28 | 6 | DATA_DEPENDENT | DISJOINT | delta | 4600 | custom:task182 |
| 148 | 4889 | 4635 | 254 | 16.51 | 8 | CONSTANT | DISJOINT | delta | 3389 | custom:task148 |
| 165 | 4840 | 4635 | 205 | 16.52 | 7 | DATA_DEPENDENT | DISJOINT | delta | 3340 | custom:task165 |
| 378 | 4432 | 4316 | 116 | 16.60 | 6 | DATA_DEPENDENT | DISJOINT | delta | 2932 | custom:task378 |
| 174 | 4378 | 3982 | 396 | 16.62 | 5 | DATA_DEPENDENT | DISJOINT | whole | 2878 | custom:task174 |
| 265 | 4362 | 4328 | 34 | 16.62 | 5 | CONSTANT | DISJOINT | delta | 2862 | custom:task265 |
| 208 | 4209 | 4072 | 137 | 16.66 | 4 | DATA_DEPENDENT | DISJOINT | delta | 2709 | custom:task208 |
| 280 | 4146 | 3363 | 783 | 16.67 | 6 | CONSTANT | DISJOINT | delta | 2646 | custom:task280 |
| 132 | 4089 | 3990 | 99 | 16.68 | 6 | DATA_DEPENDENT | DISJOINT | delta | 2589 | custom:task132 |
| 50 | 3852 | 3825 | 27 | 16.74 | 3 | CONSTANT | DISJOINT | delta | 2352 | custom:task050 |
| 333 | 3223 | 2788 | 435 | 16.92 | 4 | DATA_DEPENDENT | DISJOINT | delta | 1723 | custom:task333 |
| 8 | 3222 | 3088 | 134 | 16.92 | 6 | CONSTANT | DISJOINT | delta | 1722 | custom:task8 |
| 42 | 3110 | 2792 | 318 | 16.96 | 4 | CONSTANT | DISJOINT | delta | 1610 | custom:task042 |
| 55 | 3092 | 3030 | 62 | 16.96 | 5 | CONSTANT | DISJOINT | delta | 1592 | custom:task055 |
| 390 | 3090 | 2623 | 467 | 16.96 | 8 | CONSTANT | DISJOINT | delta | 1590 | custom:task390 |
| 90 | 3072 | 2990 | 82 | 16.97 | 1 | CONSTANT | DISJOINT | delta | 1572 | custom:task090 |
| 397 | 2782 | 2582 | 200 | 17.07 | 3 | CONSTANT | DISJOINT | delta | 1282 | custom:task397 |
| 394 | 2780 | 1898 | 882 | 17.07 | 7 | DATA_DEPENDENT | DISJOINT | whole | 1280 | custom:task394 |
| 224 | 2772 | 2400 | 372 | 17.07 | 4 | DATA_DEPENDENT | DISJOINT | delta | 1272 | custom:task224 |
| 355 | 2704 | 2678 | 26 | 17.10 | 1 | DATA_DEPENDENT | DISJOINT | whole | 1204 | custom:task355 |
| 71 | 2698 | 2643 | 55 | 17.10 | 8 | DATA_DEPENDENT | DISJOINT | delta | 1198 | custom:task071 |
| 354 | 2692 | 2615 | 77 | 17.10 | 3 | DATA_DEPENDENT | DISJOINT | delta | 1192 | custom:task354 |
| 246 | 2637 | 2220 | 417 | 17.12 | 2 | CONSTANT | DISJOINT | delta | 1137 | custom:task246 |
| 250 | 2542 | 2488 | 54 | 17.16 | 8 | CONSTANT | DISJOINT | delta | 1042 | custom:task250 |
| 102 | 2527 | 2414 | 113 | 17.17 | 3 | CONSTANT | DISJOINT | delta | 1027 | custom:task102 |
| 190 | 2527 | 2280 | 247 | 17.17 | 8 | DATA_DEPENDENT | DISJOINT | delta | 1027 | custom:task190 |
| 30 | 2396 | 2248 | 148 | 17.22 | 8 | CONSTANT | DISJOINT | delta | 896 | custom:task030 |
| 281 | 2386 | 2321 | 65 | 17.22 | 4 | DATA_DEPENDENT | DISJOINT | delta | 886 | custom:task281 |
| 22 | 2302 | 2004 | 298 | 17.26 | 7 | DATA_DEPENDENT | DISJOINT | whole | 802 | custom:task022 |
| 134 | 2284 | 2250 | 34 | 17.27 | 8 | DATA_DEPENDENT | DISJOINT | whole | 784 | custom:task134 |
| 374 | 2284 | 2242 | 42 | 17.27 | 3 | CONSTANT | DISJOINT | delta | 784 | custom:task374 |
| 256 | 2251 | 2196 | 55 | 17.28 | 6 | CONSTANT | DISJOINT | delta | 751 | custom:task256 |
| 35 | 2249 | 2125 | 124 | 17.28 | 7 | DATA_DEPENDENT | DISJOINT | delta | 749 | custom:task35 |
| 168 | 2239 | 2096 | 143 | 17.29 | 8 | DATA_DEPENDENT | DISJOINT | delta | 739 | custom:task168 |
| 273 | 2160 | 2109 | 51 | 17.32 | 2 | CONSTANT | DISJOINT | delta | 660 | custom:task273 |
| 36 | 2100 | 2061 | 39 | 17.35 | 8 | DATA_DEPENDENT | DISJOINT | whole | 600 | custom:task036 |
| 184 | 2041 | 1920 | 121 | 17.38 | 8 | DATA_DEPENDENT | DISJOINT | whole | 541 | custom:task184 |
| 381 | 1963 | 1908 | 55 | 17.42 | 4 | CONSTANT | DISJOINT | delta | 463 | custom:task381 |
| 185 | 1961 | 1651 | 310 | 17.42 | 7 | DATA_DEPENDENT | DISJOINT | whole | 461 | custom:task185 |
| 348 | 1896 | 1824 | 72 | 17.45 | 8 | CONSTANT | DISJOINT | delta | 396 | custom:task348 |
| 213 | 1851 | 1802 | 49 | 17.48 | 7 | DATA_DEPENDENT | DISJOINT | whole | 351 | custom:task213 |
| 237 | 1838 | 1700 | 138 | 17.48 | 6 | DATA_DEPENDENT | DISJOINT | delta | 338 | custom:task237 |
| 199 | 1829 | 1741 | 88 | 17.49 | 7 | DATA_DEPENDENT | DISJOINT | delta | 329 | custom:task199 |
| 336 | 1820 | 764 | 1056 | 17.49 | 3 | CONSTANT | DISJOINT | delta | 320 | custom:task336 |
| 206 | 1795 | 1660 | 135 | 17.51 | 5 | DATA_DEPENDENT | DISJOINT | delta | 295 | custom:task206 |
| 302 | 1774 | 1216 | 558 | 17.52 | 3 | DATA_DEPENDENT | DISJOINT | delta | 274 | custom:task302 |
| 51 | 1744 | 1643 | 101 | 17.54 | 1 | DATA_DEPENDENT | DISJOINT | delta | 244 | custom:task051 |
| 45 | 1673 | 1050 | 623 | 17.58 | 2 | DATA_DEPENDENT | DISJOINT | delta | 173 | custom:task045 |
| 156 | 1564 | 1505 | 59 | 17.64 | 2 | CONSTANT | DISJOINT | delta | 64 | custom:task156 |
| 226 | 1534 | 1513 | 21 | 17.66 | 3 | CONSTANT | DISJOINT | delta | 34 | custom:task226 |
| 345 | 1495 | 1431 | 64 | 17.69 | 5 | CONSTANT | DISJOINT | delta | -5 | custom:task345 |
| 341 | 1429 | 1394 | 35 | 17.74 | 1 | CONSTANT | DISJOINT | delta | -71 | custom:task341 |
| 20 | 1346 | 1176 | 170 | 17.80 | 3 | DATA_DEPENDENT | DISJOINT | delta | -154 | custom:task020 |
| 356 | 1319 | 1300 | 19 | 17.82 | 4 | CONSTANT | DISJOINT | delta | -181 | custom:task356 |
| 27 | 1310 | 1262 | 48 | 17.82 | 4 | CONSTANT | DISJOINT | delta | -190 | custom:task027 |
| 346 | 1254 | 1224 | 30 | 17.87 | 1 | DATA_DEPENDENT | DISJOINT | whole | -246 | custom:task346 |
| 343 | 1222 | 1147 | 75 | 17.89 | 6 | DATA_DEPENDENT | DISJOINT | delta | -278 | custom:task343 |

## Top-10 by cost — per-example detail

### task367  (cost 15890, CONSTANT, DISJOINT, R_max 4)
- roster union (literal colours): [4]
  - ex0 [delta] R=4 colours=[4]
  - ex1 [delta] R=2 colours=[4]
  - ex2 [delta] R=3 colours=[4]
  - ex3 [delta] R=4 colours=[4]

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

### task255  (cost 8978, CONSTANT, DISJOINT, R_max 5)
- roster union (literal colours): [3]
  - ex0 [delta] R=3 colours=[3]
  - ex1 [delta] R=3 colours=[3]
  - ex2 [delta] R=2 colours=[3]
  - ex3 [delta] R=5 colours=[3]

### task023  (cost 6412, CONSTANT, DISJOINT, R_max 7)
- roster union (literal colours): [2, 8]
  - ex0 [delta] R=6 colours=[2, 8]
  - ex1 [delta] R=6 colours=[2, 8]
  - ex2 [delta] R=4 colours=[2, 8]
  - ex3 [delta] R=7 colours=[2, 8]

### task182  (cost 6100, DATA_DEPENDENT, DISJOINT, R_max 6)
- roster union (literal colours): [2, 3]
  - ex0 [delta] R=6 colours=[2]
  - ex1 [delta] R=2 colours=[3]
  - ex2 [delta] R=2 colours=[2]
  - ex3 [delta] R=1 colours=[3]
  - ex4 [delta] R=6 colours=[2]

### task148  (cost 4889, CONSTANT, DISJOINT, R_max 8)
- roster union (literal colours): [4, 8]
  - ex0 [delta] R=3 colours=[4, 8]
  - ex1 [delta] R=6 colours=[4, 8]
  - ex2 [delta] R=8 colours=[4, 8]
  - ex3 [delta] R=8 colours=[4, 8]
