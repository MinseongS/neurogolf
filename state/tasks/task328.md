---
deployed_cost: 6631
logged_costs_match: match
migrated: 2026-07-09
---

# task328 — d22278a0

2026-06-28 status: installed/source-owned exact baseline = 16.159130 pts, mem
4971, params 1940.  `src/custom/task328.py` was re-synced from the live ONNX
because the previous semantic source scored only 14.939 / mem 23296 / params
113.

## Rule summary

The input is a square grid with 2-4 coloured anchor pixels at distinct corners.
For each in-grid cell, choose the uniquely nearest anchor by Manhattan distance;
if the winning anchor's Chebyshev distance is even, paint that anchor colour,
otherwise background.

## Current live mechanism

The installed graph uses a compact packed-template/lookup strategy rather than
the older explicit 18x18 distance-field semantic implementation.  It has lower
memory but high params from the packed template bank.

## Frontier status

Not immediately a 20+ candidate.  The old semantic source proves low params are
possible, but it pays many 18x18 distance/parity planes.  The live graph proves
lower memory is possible, but pays ~1940 params.  A real frontier breakthrough
would need a hybrid: keep the live packed-output route while replacing the
template bank with scalar/vector distance logic that does not materialize the
four 18x18 corner fields.  No safe rewrite has been identified yet.


## S16 adoption (2026-07-06) — yuu111111111 public-bundle net (+0.021)
- Source: yuu111111111/neurogolf-6-failure-modes notebook (total 7235.05, embedded 400-net archive; MINED per-task despite lower total).
- New grader cost = 6634 (mem 4691 + params 1943), fail=0 bundled.
- Fresh-gate 1500: incumbent fail = 0 | candidate fail = 0 | candidate != incumbent = 0  -> cand_fail <= incumbent_fail (safe rule PASS).
- Mechanism: structural golf: fewer counted node-output intermediates (graph rewrite, functionally equal on fresh).

## 2026-07-07 auto-golfer bank-factor probe KILL

Auto-golfer reopened this because active cost is still template-heavy:
`4688+1943=6631`, with top counted tensors `label_pad` 900B,
`pair_codes_i32` 648B, `hi_sh` 360B, and `label_pairs` 324B.  The active
mechanism is a 1717-entry int64 `template_bank` addressed by
`size/subset/row`, then radix-64 unpacked into two labels per code and finally
decoded through the corner-colour map.

Oracle check: decoding the full bank exactly matches the semantic rule for all
sizes 6..18 and all 11 valid active-corner subsets: unique nearest Manhattan
corner, Chebyshev-even gate, background on ties/odd distance.  So the bank is
not mysterious public overfit data; it is a packed compiler output for the known
rule.

Cost wall: the older semantic source had only 113 params, but materialized
18x18 distance/parity/winner planes and scored `23296+113`, far worse.  Simple
bank compression also loses: the 1717 rows collapse to 754 unique packed rows,
but unique-bank + index-table would cost `754+1717` params plus another Gather.
Direct bool output would delete the 900B `label_pad`, but creates a 10-channel
18x18 bool plane before Pad/reshape, which is much more expensive.

Conclusion: do not retry plain bank factorization, unique-row compression, or
label-pad-to-bool-output for this task.  A real win still needs a new lowering
that generates packed row codes from scalar/vector logic without materializing
18x18 corner distance fields.

## ADOPTED 20260709T041326Z
- cost: 6631 -> 6304 (points 16.2511)
- source: candidates/public_dumps/20260709/7261-53-lb-compact-onnx-artifact-starter/nets/task328.onnx
- note: min-merge from nets

## 2026-07-12 — Manhattan-nearest-corner free-output-einsum rebuild → NO-BUILD (kill bar +0.1)
⚠️ STALE-MD CORRECTION: prior entries describing a "1717-entry int64 template bank" are STALE (pre-6304).
The 20260709 min-merge replaced it — current deployed = compact 69-node semantic distance-decomposition
net (matches src/custom/task328.py): params only 123 elements (ramps aH/aW, scalars, colorIdx); cost is
node-output MEMORY, not a param bank. Read the net, not this md, for future 328 work.
ran: full rule verify (arc-gen task_d22278a0) + grader ORT trace cost breakdown + leaner argmin/mask-product
variants byte-priced. Rule: each cell -> color of unique Manhattan-nearest ACTIVE corner (data-dependent
2-4 subset), bg on ties or winner Chebyshev odd, grid 6-18, ch0=1 for in-grid bg (dynamic boundary masked).
verdict: 6304 = mem6181+params123. Breakdown: 900B labelPad [1,1,30,30]u8 (Pad->Equal channel expansion,
already optimal — one-hot-first = 3240B worse) + 4536B = fourteen [1,1,18,18] decision planes (dbestT/dbestB,
tie masks, colorField, cheb parity, inside, label) + ~640B 1D vectors + corner reads. The ~14 genuinely-2D
planes are non-linear (argmin/tie/parity) AND non-separable (nested-square parity rings, diagonal
boundaries) => cannot fold into free-output einsum (348 M-expand trick fails) nor rank-decode. Size<=18
forces 18x18. Architectural hard core = 14*324+900 = 5436B; realistic best ~5600-5900 > 5700 bar. From-scratch
materialized variants floor ~8kB (worse). The 6304 net is near-optimal.
tool+date: opus agent + grader trace, onnx 1.21/ort 1.26, 2026-07-12.
reopen: a separable/linear reformulation of the argmin+parity core (none found); an op that emits Chebyshev
parity rings cheaply; a leaked <5000 net.
falsification history: recon audit's "fp32-field-floor is conditional, ramps avoid it" premise was CORRECT
about the ramps but the net ALREADY uses them — residual is the 14 non-foldable 2D decision planes, a
different (real) floor than the stale fp32-field one.
