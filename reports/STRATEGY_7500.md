# Strategy to 7500 (fundamental analysis, 2026-06-16)

## The honest math (reports/lb_status.py + manifest distribution)
- stored 6469.9, avg **16.17**/task. Target 7500 = avg **18.75**. Gap = **+1030 stored** (+2.58/task).
- 294 tasks <17pt. If ALL→18.75: +1038 (just barely 7500). So EVERY sub-17 feasible task must reach ~18.75.
- **Tier-B (label-map, ~16.8) is BELOW the 18.75 target.** Incremental floor-break of public-base tasks to
  Tier-B CANNOT reach 7500 — it plateaus ~6500-6650 (matches our trajectory: LB 6393→6400→6406, +~1.5/task
  on thinning headroom). This is the core realization: **we have been winning the wrong tier.**

## Where 7500 must come from (the unlocks, in priority)
**(A) Tier-S on the feasible majority — THE lever (~+600-900).** To score 19-20 a net needs mem+params ≲ 250
(e^(25-19.5)). That means NO [1,1,30,30] intermediate at all (the 900B label-map caps you at 16.8). Output must
be produced DIRECTLY into the free `output` by ~1 op with near-zero intermediates:
  - pure geometric remap (mirror/tile/transpose/rotate/permute/crop/Kronecker) = ONE `Gather`/`Transpose`/
    `ConvTranspose` into output, mem ~0 → **18-25**. Many such tasks currently sit at 14-16 on bloated PUBLIC
    nets — re-encoding them Tier-S is +3-7 EACH (much bigger than the +1-2 Tier-B wins we've been taking).
  - per-cell recolor where output = `Where(mask, color_onehot, input)` with mask a single Conv (no label map).
  - **NEW TARGET CLASS: hunt tasks whose output is a FIXED (content-independent) permutation of input cells.**
    Detector: generate 2 instances, check if output = input[fixed_index_map] for a map that's identical across
    instances. If yes → single Gather, mem 0. (Build reports/tier_s_scan.py.)
**(B) The "infeasible" ~100 (flood/connectivity/sprite/random) — likely a genuine ceiling (~475 drag).** NCA /
unrolled-Conv CAN compute flood-fill/connectivity but cumulative memory (K×900B bool, K~30) floors them ~14.8 —
NO better than memorizer under this scorer. So these probably cap ~14 for EVERYONE incl #1. Spend little here;
verify the ceiling on 1-2 (unrolled-dilation test) then move on. Exception: connectivity that's separable or
log-depth-expressible. Random/non-deterministic (255/219) are hard 0 — skip forever.
**(C) base-net gap recovery (~34 stored, real LB).** Replace genverify-flagged non-gen nets (209/118/2/157) with
generalizing customs — closes the gap directly. Borderline/hard but real LB value.

## Revised targeting (loop pivot)
1. **PRIMARY: Tier-S hunt.** Before building label-map, ALWAYS test if the task is a fixed permutation / pure
   geometric remap → Tier-S (mem 0). Re-scan ALL adopted Tier-B customs + public 14-16 tasks for Tier-S
   re-encoding (+3-7 each). This is the 7500 path; Tier-B grind is the floor, not the goal.
2. Keep the Tier-B incremental grind running in parallel (steady floor), but bias new agents to attempt Tier-S
   FIRST and fall back to Tier-B only when the rule is genuinely content-dependent-per-cell.
3. Quantify reality: if after a Tier-S sweep the projected LB plateaus <6800, 7500 likely needs cracking the
   infeasible-100, which may be scorer-impossible — escalate to the user with the evidence.

## Self-improvement loop (trial-and-error → systematize)
- Every adopted win logs an INSIGHT (tasklog). Recurring insights graduate to BUILD_PROMPT.md levers (done: 9
  collapsible-detection forms). When a PROCEDURE recurs with judgment (not just a lever), create a skill via
  skill-creator (candidate: "tier-s-detect" if the permutation-scan becomes routine). Don't force skills.
- Track the Tier distribution over time (how many tasks at S/A/B/detection) as the real progress metric, not
  just stored. Add to lb_status or a new tier_census.

## ⭐ SCAN RESULT (2026-06-16) — lever (A) is mostly EMPTY; recalibrated ceiling ~6900.
`reports/tier_s_scan.py` found only **13 pure-fixed-transform tasks, 12 already at 18-25** (public kojimar
nets already encode the obvious Tier-S geometry). Only task1 (kron_self, 17.10) has mild headroom. So there is
NO untapped pure-transform reservoir. The 14-16 tasks are content-dependent → label-map → capped ~17.5-18.1
(the 900B L[1,1,30,30] feeding Equal is the floor; sub-900B output is only possible for local-Conv/permutation
rules, which ARE the pure-transform tasks already taken). Idealized ceiling if EVERY feasible task is stripped
to its label-map floor: **~6825-6975**. So re-encoding alone tops ~6900 — it does NOT reach 7500.

## WHERE 7500 ACTUALLY LIVES (the ~600pt gap from ~6900):
1. **The "infeasible ~100" (flood/connectivity/sprite/correspondence).** This is the only reservoir big enough.
   #1's 7700 almost certainly cracks many of these. Two sub-questions: (a) are they truly scorer-capped (NCA
   unroll = K×900B cumulative ≈ 14.8, no better than memorizer)? (b) is the "infeasible" list OVER-broad —
   mislabeled-feasible tasks (like 191/264 were recoverable)? **ACTION: re-triage all ~100 with fresh eyes +
   the closed-form lenses + the count-parametric/separable/equivariance tricks.** Highest-value open work.
2. **base-net gap recovery (~34 real LB):** replace genverify-flagged non-gen 209/118/2/157.
3. Strip content-dependent tasks to ~18 (floor grind, +0.5-1 each) — keeps climbing toward the ~6900 ceiling.

## HONEST STANCE: re-encoding → ~6900 (confident). 6900→7500 needs the infeasible-100 to be partly
feasible (unknown) OR a fundamentally cheaper output encoding than 900B (searched, none found for non-local
rules). Pursue the infeasible re-triage as the 7500 bet; if it's a wall, ~6900 is the real ceiling and that
must be reported, not pretended past.

## Research notes
- NCA for ARC: arxiv 2506.15746 (iterated local Conv). CompressARC: per-puzzle equivariant net, arxiv 2512.06104
  (equivariance to color-perm + D4 + pair-order — we already exploit gravity/transpose equivariance).
- Connected-component labeling = iterated flood/label-propagation, fixed-iteration unroll possible but
  cumulative-memory-expensive under this scorer.

## RE-TRIAGE RESULT (2026-06-16): 50/122 mislabeled-FEASIBLE (+85.6 est), 72 genuine-infeasible.
The BAIL list was 41% over-broad. Feasible reservoir: 5 Tier-S + 21 Tier-A + 24 Tier-B/count, est +85.6
stored (→ ~6495 LB at 1:1, climbing toward the ~6900 ceiling as the rest of the floor grind lands).
CONFIRMS the thesis: this reservoir does NOT reach 7500 — the 72 genuine-infeasible (flood-fill/enclosure,
box-connectivity, sprite-correspondence/rotation-matching, ray-tracing, per-object CC-size, shape-classify)
are the hard wall, scorer-capped ~14 for everyone. Queue: reports/retriage_build_queue.json (ranked by gain).
NET 7500 VERDICT: re-encoding + this reservoir → realistic ~6800-6900 ceiling. 6900→7500 would require
cracking flood-fill/connectivity/correspondence cheaply, which appears scorer-impossible (cumulative memory).
Pursue the 50 feasible (big, real), then report the honest ceiling.
