# LB status (stored ↔ real LB gap tracker)

- **stored (local, optimistic):** 6580.48
- **last confirmed LB:** 6543.68  (stored 6572.64 @ 2026-06-18 02:05 UTC)
- **structural gap (stored−LB at anchor):** 28.96  ← base-net overcount, ~stable
- **PROJECTED current LB:** 6551.52  (= stored − gap; +7.84 of un-submitted wins since anchor)
- next submit at +5 adopted wins re-anchors this.

## Gap attribution (genverify n=40 batch — rough, false-neg prone; isolated n=200 is truth)
Estimated overcount across 9 sub-100% tasks ≈ **33.2** pts (cf. gap 28.96).
Top offenders (replacing these with generalizing nets closes the gap directly):

| task | stored | fresh rate | est. overcount | method |
|---|---|---|---|---|
| 219 | 15.00 | 0.03 | 14.62 | None |
| 255 | 13.95 | 0.00 | 13.95 | None |
| 157 | 13.76 | 0.90 | 1.38 | gen:wguesdon6315 |
| 332 | 17.00 | 0.95 | 0.85 | custom:task332 |
| 209 | 13.36 | 0.95 | 0.67 | gen:wguesdon6315 |
| 18 | 13.34 | 0.95 | 0.67 | gen:thbdh6332 |
| 23 | 13.86 | 0.97 | 0.35 | gen:galaxy |
| 2 | 13.51 | 0.97 | 0.34 | gen:vyank6322 |
| 366 | 13.50 | 0.97 | 0.34 | gen:vyank6322 |
