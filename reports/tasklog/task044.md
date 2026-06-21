# task044 (ARC 228f6490) — INFEASIBLE: shape-correspondence assignment wall

Deployed: kojimar 15.56 pts, 12444 B, 109 params, 180 nodes
(ArgMax×9 + GatherND + ScatterND + 2 Conv — heavy connected-component / assignment machinery).

## Rule (derived exactly)
Input: two hollow GRAY (5) boxes; two colored "sprites" (connected creatures) sitting
OUTSIDE their boxes; one "dust" color of scattered (mostly isolated) pixels.
Output: each sprite is ERASED from its location and STAMPED into a box interior at the
box's interior top-left corner `(brow+1, bcol+1)` (same relative shape). Dust + gray
unchanged. (Generator draws creature identically at sprite loc and box interior; the
in-box black marker is color-0 = invisible.)

## Why INFEASIBLE for a closed-form net
The transform needs: (1) connected-component creature extraction to tell sprites from
dust, (2) a per-sprite→box ASSIGNMENT, (3) per-sprite data-dependent 2-D translation of
an arbitrary connected shape into its box. (1)+(3) alone are already near the deployed
cost; (2) is the true wall:

- The assignment is NOT recoverable from any simple input feature. Measured over fresh
  instances: bbox-fit is ambiguous in **91%** (710/778) of cases; sprite-pixel-count→box-
  area rank is **wrong 27%** (133/485); largest-count→largest-box wrong 27%.
- The only local predicate that helps — "stamp creature at box-interior corner; valid iff
  it fits the interior AND lands only on background cells" — still yields a NON-UNIQUE
  matching in **195/3000 (6.5%)** of instances (dust can validly stamp by chance; two
  creatures can each validly stamp into either box).
- Dust-vs-sprite by connectivity (most-fragmented color) is **wrong 136/3000 (4.5%)**;
  count-2 sprite vs count-2 dust collide.
- No exact-fit disambiguator: the creature bbox equals its box interior in only
  2958/6000 (49%) of true pairs, so "tightest fit" cannot break the tie.
- These error sources don't even compose to an EXACT numpy reference (124/3000 fail with
  first-valid-permutation), so an EXACT closed-form ONNX net is impossible — the residual
  ~5% ambiguity is decided solely by the generator's hidden creature→box `idx` pairing,
  which leaves no signal in the input.

This is the documented multi-object shape-correspondence / ambiguous-template-recovery
wall (cf. task158, task279). Allowed ops (no Loop/NonZero/Unique/Compress) cannot do the
connected-component assignment + arbitrary-shape data-dependent translation exactly, and
even the deployed kojimar net (which can, with 180 nodes) only reaches 15.56. A from-
scratch closed-form net cannot reach EXACTNESS, let alone beat the score.

VERDICT: INFEASIBLE (assignment is input-underdetermined in ~5% of instances).
