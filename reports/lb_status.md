# LB status (stored ↔ real LB gap tracker)

- **stored (local, optimistic):** 7170.49
- **last confirmed LB:** 7170.59  (stored 7170.49 @ 2026-06-27 18:47 submission 54118286 COMPLETE UTC)
- **structural gap (stored−LB at anchor):** -0.10  ← base-net overcount, ~stable
- **PROJECTED current LB:** 7170.59  (= stored − gap; +0.00 of un-submitted wins since anchor)
- next submit at +5 adopted wins re-anchors this.

## Gap attribution (genverify n=40 batch — rough, false-neg prone; isolated n=200 is truth)
Estimated overcount across 13 sub-100% tasks ≈ **36.8** pts (cf. gap -0.10).
Top offenders (replacing these with generalizing nets closes the gap directly):

| task | stored | fresh rate | est. overcount | method |
|---|---|---|---|---|
| 255 | 14.87 | 0.00 | 14.87 | ext:franksunp7166_68 |
| 219 | 14.83 | 0.00 | 14.83 | ext:franksunp7166_65 |
| 157 | 15.55 | 0.93 | 1.17 | ext:franksunp7166_68 |
| 2 | 14.40 | 0.93 | 1.08 | ext:franksunp7166_65 |
| 319 | 14.99 | 0.95 | 0.75 | ext:franksunp7166_68 |
| 118 | 14.55 | 0.95 | 0.73 | custom:task118_tail_where+onnxsim |
| 366 | 14.47 | 0.95 | 0.72 | ext:franksunp7166_65 |
| 233 | 13.84 | 0.95 | 0.69 | ext:franksunp7166_68 |
| 151 | 18.20 | 0.97 | 0.45 | ext:franksunp7166_65 |
| 44 | 15.63 | 0.97 | 0.39 | ext:franksunp7166_68 |
| 23 | 14.98 | 0.97 | 0.37 | ext:franksunp7166_65 |
| 76 | 14.85 | 0.97 | 0.37 | ext:franksunp7166_68 |
| 18 | 13.90 | 0.97 | 0.35 | ext:franksunp7166_68 |
