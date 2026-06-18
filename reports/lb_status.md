# LB status (stored ↔ real LB gap tracker)

- **stored (local, optimistic):** 6615.71
- **last confirmed LB:** 6585.61  (stored 6614.56 @ 2026-06-18 15:30 UTC)
- **structural gap (stored−LB at anchor):** 28.95  ← base-net overcount, ~stable
- **PROJECTED current LB:** 6586.76  (= stored − gap; +1.15 of un-submitted wins since anchor)
- next submit at +5 adopted wins re-anchors this.

## Gap attribution (genverify n=40 batch — rough, false-neg prone; isolated n=200 is truth)
Estimated overcount across 12 sub-100% tasks ≈ **36.5** pts (cf. gap 28.95).
Top offenders (replacing these with generalizing nets closes the gap directly):

| task | stored | fresh rate | est. overcount | method |
|---|---|---|---|---|
| 219 | 15.00 | 0.00 | 15.00 | None |
| 255 | 13.95 | 0.00 | 13.95 | None |
| 157 | 13.76 | 0.88 | 1.72 | gen:wguesdon6315 |
| 118 | 13.34 | 0.88 | 1.67 | gen:seddik |
| 23 | 13.86 | 0.93 | 1.04 | gen:galaxy |
| 209 | 13.36 | 0.93 | 1.00 | gen:wguesdon6315 |
| 277 | 15.53 | 0.97 | 0.39 | ext:biohack_new |
| 76 | 13.71 | 0.97 | 0.34 | gen:thbdh6332 |
| 2 | 13.51 | 0.97 | 0.34 | gen:vyank6322 |
| 366 | 13.50 | 0.97 | 0.34 | gen:vyank6322 |
| 18 | 13.34 | 0.97 | 0.33 | gen:thbdh6332 |
| 233 | 13.25 | 0.97 | 0.33 | gen:thbdh6332 |
