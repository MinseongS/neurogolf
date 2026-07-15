---
deployed_cost: 3850
logged_costs_match: stale-likely
migrated: 2026-07-09
---

# task050 — 253bf280

**Rule:** Cyan (8) pixels are scattered endpoints. For every two cyan pixels sharing a
row, fill green (3) strictly between them; likewise for every two cyan sharing a column.
Cyan stays cyan, background stays background. (Generator never places 3+ collinear cyan,
so a cyan cell is never itself "between" two others.)
**Current (prior):** 15.892 pts, ext:biohack_new, mem 8775, params 250
**Target tier:** detection/A — separable per-line prefix-scan; closed-form, no flood-fill.

## Attempts
| # | angle | tier | mem | params | pts | fresh | outcome |
|---|---|---|---|---|---|---|---|
| 1 | Equal(arange) route, no off-grid sentinel | - | - | - | 0 | - | bg wrong off-grid (off-grid must be ALL-zero, not bg) |
| 2 | + sentinel 99 + fp16 pad-to-30 plane | A | 12210 | 41 | 15.59 | - | worse (30x30 fp16 plane) |
| 3 | uint8 Where-chain L + uint8 Pad + Equal | A | 8325 | 40 | 15.97 | - | beats by +0.08 (4 cumsum) |
| 4 | in-grid via ReduceMax row/col profiles (drop bg fp32 slice) | A | 7365 | 50 | 16.09 | - | +0.20 |
| 5 | offgrid = (¬rowany)∨(¬colany) in one op | A | 7170 | 50 | 16.12 | - | +0.22 |
| 6 | **2 cumsums: between ⇔ 0<before<linetotal** | A | **6330** | **53** | **16.24** | **200/200** | **+0.347 — adopted** |

## Best achieved
16.239 @ mem 6330 params 53 — beats prior 15.892 by **+0.347** (≥+0.3). fresh 200/200.

## Irreducible-floor analysis
Two 900B planes dominate: (a) cyan fp32 channel Slice [1,1,15,15] (Slice always
returns the fp32 input dtype; needed before the fp16 cumsum cast) and (b) the uint8
colour-index carrier Pad'd to [1,1,30,30] (the Equal-vs-arange carrier; uint8 1-byte is
already the dtype floor and the output canvas is fixed 30×30). The 2 fp16 CumSum planes
(450 each) are the per-axis prefix-counts. Everything else is bool/uint8 15×15 (225) or
tiny profile vectors. Going below ~6.3KB would require removing one of the two 900B
planes, which the algorithm structurally needs.

## OPEN ANGLES (re-attack backlog)
- Collapse the 3-Where L-build to 2 Where (~225B): no clean way found — 4 disjoint
  outcomes {0,3,8,99} need a depth-2 select; uint8 Add/Max work but still need a Where
  per scaled mask.
- Replace cyan fp32 Slice (900) with a fp16 path: blocked — Slice inherits fp32 input
  dtype; casting the 10-ch input first costs 18000B.

## INSIGHT (transferable)
⭐ "STRICTLY-BETWEEN two same-colour marks in a row/col" needs only ONE exclusive
forward CumSum per axis, NOT two (no reverse cumsum / no second prefix): at a background
cell `before + after = linetotal`, so `has-mark-both-sides ⇔ 0 < before < linetotal`
(`Greater(before,0) ∧ Less(before, ReduceSum-total)`). Halves the cumsum count vs the
fwd+reverse idiom. Also ⭐ the off-grid region of a Pad-to-30 / Equal-vs-arange routing
must map to a SENTINEL value (≠ any channel index), because the harness scores off-grid
cells as ALL-channels-zero — emitting bg (index 0) there fails every example; derive the
in-grid rectangle for free from `ReduceMax(input, axes=[1,3])`/`[1,2]` occupancy profiles
(no fp32 channel slice for bg).


## S15b (2026-07-06) — ADOPTED from prvsiyan 7235.05 min-merge: 3911 -> 3852 (+0.015); gate inc/cand=0/0 (safe). See [[neurogolf-urad-7225-bundle-vein]].
## ADOPTED 20260709T041340Z
- cost: 3850 -> 3849 (points 16.7444)
- source: candidates/public_dumps/20260709/7261-53-lb-compact-onnx-artifact-starter/nets/task050.onnx
- note: min-merge from nets

## ADOPTED 20260712T043900Z
- cost: 3849 -> 1896 (points 17.4525)
- source: candidates/task050/cand.onnx
- note: axis-separable interval-fill as single free-output einsum (mem=0, zero counted intermediates): 4 additive deg-2 terms {X0,X8,BR,BC}; bg=X0-BR-BC (uses input ch0 as bg base -> input reads FACTOR, no ingrid-sum), green=BR+BC, cyan=X8; BR/BC=strict-between counts via one strict-lower-tri T reused transposed; 3849->1896 (+0.707); fresh 2200 div0 bit-identical. TRANSFERABLE: bg=input-ch0-minus-fill

## ADOPTED 20260713T133635Z
- cost: 1896 -> 448 (points 18.8952)
- source: candidates/task050/from_task350.onnx
- note: task350 endpoint-fill formula port with cyan source and green strict interior; bundled 271/0

## ADOPTED 20260715T032343Z
- cost: 448 -> 438 (points 18.9178)
- source: candidates/task050/factor_r2_exact.onnx
- note: direct exact shared rank-2 R2 factorization

## ADOPTED 20260715T053337Z
- cost: 438 -> 436 (points 18.9224)
- source: candidates/task050/fold_endpoint_sign.onnx
- note: runtime-safe exact fold: absorb G[x] sign into distinct Fout[x,o], removes one operand and 2 params
