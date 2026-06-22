# LB status (stored ↔ real LB gap tracker)

- **stored (local, optimistic):** 7155.60
- **last confirmed LB:** 7150.13  (2026-06-22 REBASE onto franksunp 7141.14 base + 25 exact custom overlays; CONFIRMED 7150.13)
- **structural gap (stored−LB at anchor):** 5.47  ← franksunp base local≈real (gap collapsed from 30.82); stored now a tight proxy
- **PROJECTED current LB:** 7150.13  (= stored − gap; +0.00 of un-submitted wins since anchor)
- next submit at +5 adopted wins re-anchors this.
- prior anchors: 7134.40 (kokinnwakashuu 7125.30 base+22 overlays, gap 30.82), 7123.43.

## Gap attribution (genverify n=40 batch — rough, false-neg prone; isolated n=200 is truth)
Estimated overcount across 13 sub-100% tasks ≈ **35.9** pts (cf. gap 30.82).
Top offenders (replacing these with generalizing nets closes the gap directly):

| task | stored | fresh rate | est. overcount | method |
|---|---|---|---|---|
| 219 | 14.61 | 0.00 | 14.61 | ext:kojimar7113 |
| 255 | 14.26 | 0.00 | 14.26 | ext:kojimar7113 |
| 157 | 15.35 | 0.93 | 1.15 | ext:kojimar7113 |
| 2 | 14.32 | 0.93 | 1.07 | ext:kojimar7113 |
| 319 | 14.97 | 0.95 | 0.75 | ext:kojimar7113 |
| 366 | 14.38 | 0.95 | 0.72 | ext:kojimar7113 |
| 118 | 14.17 | 0.95 | 0.71 | ext:kojimar7113 |
| 233 | 13.76 | 0.95 | 0.69 | ext:kojimar7113 |
| 151 | 18.20 | 0.97 | 0.45 | ext:kojimar7113 |
| 23 | 16.31 | 0.97 | 0.41 | ext:kojimar7113 |
| 44 | 15.56 | 0.97 | 0.39 | ext:kojimar7113 |
| 76 | 14.80 | 0.97 | 0.37 | ext:kojimar7113 |
| 18 | 13.82 | 0.97 | 0.35 | ext:kojimar7113 |
