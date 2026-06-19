# LB status (stored ↔ real LB gap tracker)

- **stored (local, optimistic):** 7135.87
- **last confirmed LB:** 7107.01  (stored 7135.87 @ 2026-06-19 7k-harvest confirmed UTC)
- **structural gap (stored−LB at anchor):** 28.86  ← base-net overcount, ~stable
- **PROJECTED current LB:** 7107.01  (= stored − gap; +-0.00 of un-submitted wins since anchor)
- next submit at +5 adopted wins re-anchors this.

## Gap attribution (genverify n=40 batch — rough, false-neg prone; isolated n=200 is truth)
Estimated overcount across 13 sub-100% tasks ≈ **35.7** pts (cf. gap 28.86).
Top offenders (replacing these with generalizing nets closes the gap directly):

| task | stored | fresh rate | est. overcount | method |
|---|---|---|---|---|
| 219 | 15.00 | 0.00 | 15.00 | None |
| 255 | 13.95 | 0.00 | 13.95 | None |
| 157 | 15.35 | 0.93 | 1.15 | ext:sajayr7k |
| 2 | 13.51 | 0.93 | 1.01 | gen:vyank6322 |
| 319 | 14.58 | 0.95 | 0.73 | gen:thbdh6332 |
| 366 | 13.50 | 0.95 | 0.68 | gen:vyank6322 |
| 118 | 13.34 | 0.95 | 0.67 | gen:seddik |
| 233 | 13.25 | 0.95 | 0.66 | gen:thbdh6332 |
| 151 | 18.19 | 0.97 | 0.45 | gen:thbdh6332 |
| 44 | 15.55 | 0.97 | 0.39 | ext:sajayr7k |
| 23 | 13.86 | 0.97 | 0.35 | gen:galaxy |
| 76 | 13.71 | 0.97 | 0.34 | gen:thbdh6332 |
| 18 | 13.34 | 0.97 | 0.33 | gen:thbdh6332 |
