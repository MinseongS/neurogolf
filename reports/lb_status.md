# LB status (stored ↔ real LB gap tracker)

- **stored (local, optimistic):** 6551.71
- **last confirmed LB:** 6486.73  (stored 6548.00 @ 2026-06-16 16:14 UTC)
- **structural gap (stored−LB at anchor):** 61.27  ← base-net overcount, ~stable
- **PROJECTED current LB:** 6490.44  (= stored − gap; +3.71 of un-submitted wins since anchor)
- next submit at +5 adopted wins re-anchors this.

## Gap attribution (genverify n=40 batch — rough, false-neg prone; isolated n=200 is truth)
Estimated overcount across 11 sub-100% tasks ≈ **34.3** pts (cf. gap 61.27).
Top offenders (replacing these with generalizing nets closes the gap directly):

| task | stored | fresh rate | est. overcount | method |
|---|---|---|---|---|
| 219 | 15.00 | 0.00 | 15.00 | None |
| 255 | 13.95 | 0.00 | 13.95 | None |
| 209 | 13.36 | 0.88 | 1.67 | gen:wguesdon6315 |
| 118 | 13.34 | 0.93 | 1.00 | gen:seddik |
| 2 | 13.51 | 0.95 | 0.68 | gen:vyank6322 |
| 90 | 13.97 | 0.97 | 0.35 | gen:thbdh6332 |
| 157 | 13.76 | 0.97 | 0.34 | gen:wguesdon6315 |
| 366 | 13.50 | 0.97 | 0.34 | gen:vyank6322 |
| 251 | 13.48 | 0.97 | 0.34 | gen:thbdh6332 |
| 18 | 13.34 | 0.97 | 0.33 | gen:thbdh6332 |
| 101 | 12.68 | 0.97 | 0.32 | gen:thbdh6332 |
