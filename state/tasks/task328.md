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
