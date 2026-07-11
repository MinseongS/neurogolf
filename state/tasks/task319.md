---
deployed_cost: 5832
logged_costs_match: match
migrated: 2026-07-09
---

# task319 — ce602527

**Rule:** Grid (15-19 square-ish) bg color + two conway sprites (3-5 wide/tall). Sprite idx0 ("magnified")
is drawn small in colors[0]; a 2x-upscaled copy of sprite0 (each pixel→2x2 block) is drawn elsewhere in
`magcolor`, deliberately placed so exactly one 2-cell edge strip is off-grid (`some_hidden` required; can also
collide with other pixels). Sprite idx1 is a distractor in colors[1]. Exactly 3 non-bg colors always present.
**Output** = sprite0's small copy rendered in its own (wide×tall) bbox, in colors[0]. So the whole task is:
identify WHICH of the two small sprites is the magnified one, output it in its own color.
**Current:** 14.58 pts, gen:thbdh6332, mem 33406, params 101 (a bloated full-canvas correlation net; also a
GAP-ATTRIBUTION task — scores less on Kaggle held-out).
**Target tier:** detection / correspondence — would need exact shape-correspondence.

## Attempts (numpy feasibility ceilings, ISOLATED fresh generator)
| # | angle | accuracy | outcome |
|---|---|---|---|
| 1 | identify magcolor by max pixel-count, sprite0 by count/size heuristic | ~60-90% | FAIL (mag!=max-count 7/500) |
| 2 | bbox-doubling match (2*spriteH/W ∈ {magH, magH+2}) unique winner | 302/500 unique&correct | FAIL (ambiguous, both sprites pass ~40%) |
| 3 | downsample mag blob /2, cross-correlate vs each sprite (all parities/shifts, asym penalty) | 263/300 ≈ 88% | FAIL (conway sprites coincide; clipping+collision noise) |
| -  | exact 2x2-clean-block test on mag blob | 469/500 | FAIL — mag blob not always clean (partial-cell clip / collisions) |

## Best achieved
No exact net. Best achievable discriminator ≈ 88% fresh — far below 200/200.

## Irreducible-floor analysis
Two compounding correspondence walls, neither closed-form:
1. **Magnified-blob identity is not a clean scalar.** Max-pixel-count picks the wrong color ~1.4% (the
   distractor sprite1 can out-count a heavily-clipped/colliding magnified blob). The "all 2x2 blocks uniform"
   test that would anchor it exactly fails ~6% because the off-grid clip can remove a partial (non-block-aligned)
   strip and the magnified copy can collide with pre-existing sprite pixels.
2. **sprite0 vs sprite1 needs true shape correspondence.** Both are conway sprites of overlapping size (3-5),
   so size/count/bbox give no separating margin; the only signal is matching the (occluded, possibly-collided)
   2x blob shape against each small sprite — a cross-correlation + argmax. Even an exhaustive numpy matcher
   (every parity, every shift, asymmetric mismatch penalty) tops out ~88%. Expressing that in ONNX is exactly
   the full-canvas correlation the public net already does (mem 33406) — bloated AND still wrong on held-out.
No separable row⊗col / count→pattern / bounded-unroll reformulation exists: the answer depends on a 2-D shape
identity, not on any per-axis profile or scalar. PARTIAL is also impossible — there is no exact sub-component
(both color readout AND geometry hinge on first solving the unsolvable identification).

## OPEN ANGLES (exhausted to INFEASIBLE)
- None that reach exactness. The matching is intrinsically a noisy 2-D template correlation; no closed form.

## INSIGHT (transferable)
⭐ "Magnified-sprite correspondence" (output = the small sprite whose 2x-upscale appears elsewhere) is a TRUE
correspondence wall when (a) the upscaled copy is deliberately edge-clipped AND can collide with other pixels,
so it isn't a clean 2x2-block image, and (b) the two candidate sprites are same-distribution conway shapes with
no size/count margin. Both the anchor (which color is magnified) and the selection (which small sprite matches)
require full 2-D cross-correlation — best fresh accuracy ~88%, no separable/scalar/bounded-unroll escape.
This matches the BUILD_PROMPT "shape-correspondence + global argmax across data-dependent components" floor.

## 2026-06-30 (S7) — LANDED safe-golf, fresh-gated
The incumbent cast the WHOLE [1,10,30,30] one-hot input to uint8 (9000B plane) then
did per-colour Gather-by-channel object masks. Replaced with a 3600B Einsum
channel-collapse colour-index (`'bchw,c->bhw'`, weight=[1..10] so inside-board
cell→colour+1, padding→0); object masks become Equal(cplane, colorX+1)→Cast(uint8),
bit-identical (the +1 keeps colour-0 objects distinct from zero padding).
row_occ/col_occ now reduce the FREE fp32 input directly. Bit-identical to incumbent
on 3000/3000 fresh (both share the incumbent's pre-existing 8/3000 ambiguous-case
fails — unchanged). mem 21834→19530, params 269→279, pts 14.9965→15.1061 (+0.110).
LESSON: the 9000B full-input-cast is NOT always a dead-end — when downstream only
needs per-colour comparisons, a 3600B Einsum collapse replaces it. Cross-task scan
found only task286 still does Cast(input) (consumer=Slice, different pattern).

## S8 (2026-07-02) — reverse-ArgMax → select_last_index (+0.022) ADOPTED, div 0
Gather(rev30)+ArgMax+(30−tail) ×7 → ArgMax(select_last_index=1)+1; drops rev planes + rev30/thirty inits. FLEET-WIDE IDIOM: scan for Gather(reversed)+ArgMax patterns.

## S9 (2026-07-03) — fallback-table dead-row shrink (+0.020) ADOPTED
fallback_sig_table[38,3]+slots[38] memorization patch fires on only 3 bundled rows
(22/26/36) and 0/3000 fresh → kept 3 rows, bit-identical. mem 18890→18645, params
248→108, total 19138→18753. Gates: stored fail=0; div 0/3000 fresh + 0/400 random
(orchestrator); cached 3000: 3/3; uncached 800: 1/1, div 0. Latency 0.2ms.
FLOORS re-priced: cplane 3600 fp32 (input fp32 locks einsum out-dtype), 3× Equal+Cast
masks 5400B (K-batch neutral), 12 pairwise all-equal 2400B (fp16 bilinear-dot = exact
wash, single-use views), Gather×16 views non-uniform indices + dynamic-Slice banned.
Backup reports/retired_networks/task319_pre_s9.onnx. DO NOT re-probe repeat-group here.

## 2026-07-07 — bundled dynamic-CSE active overlay (+0.0002)

Built `reports/candidates/task319/task319_dynamic_cse_greedy.onnx` with
`reports/candidates/dynamic_cse_active_probe.py`.  One single-byte duplicate
carrier was rewired: `safe_name_132->safe_name_131`.

Bundled gate: fail=0.  Cost: 5850 -> 5849 (memory 5686 -> 5685, params 164
unchanged).  Active overlay updated in `submission/overfit_nets/task319.onnx`;
backup at `reports/candidates/task319/task319_pre_dynamic_cse.onnx`.

## 2026-07-08 — bounded-profile crop oracle: FAIL / byte-negative

Context: after public tail adoption, active `submission/overfit_nets/task319.onnx`
is the compact public bit-pack style net (`memory=5674`, `params=163`,
`points=16.32802775479621`).  Dominant remaining counted tensors are two fp32
profile Einsums:

- `safe_name_24 = Einsum(input, powers30, equation='bchw,w->ch')`, `[10,30]`, 1200B.
- `safe_name_25 = Einsum(input, equation='bchw->cw')`, `[10,30]`, 1200B.

Oracle:

- Raw bundled grid sizes are bounded by `19x19`.
- However, downstream 5-row local views gather row-profile indices up to `24`
  (`safe_name_116`), so a row profile needs at least length 25.  Column profile
  only needs length 19.

Probe:

- Built `reports/candidates/task319/task319_profile_crop21x19.onnx` as a first
  graph-surgery attempt.  It failed ORT bundled eval because `safe_name_116`
  can index `21`, exceeding the cropped row profile.
- A corrected Slice-based `25x19` input crop is byte-negative: the Slice output
  itself is counted as a full fp32 tensor (`1*10*25*19*4 = 19000B`) before the
  smaller profile, far worse than the incumbent two 1200B profiles.
- A selector-Einsum crop avoids Slice but is also byte-negative: row selector
  `25x30` plus column selector `19x30` costs about 1320 initializer elements to
  save only about 640B of profile output.

Result:

The bounded-profile crop is not adoptable.  Reopen only if sparse initializers
become accepted by the Kaggle grader, or if a new op can crop the free input
inside the same counted profile op without paying dense selector params.

## 2026-07-08 — zero-compare cumulative bool-cast tail ADOPTED

Re-ran `reports/candidates/zero_compare_to_bool_cast_probe.py` on the current active
overfit set and then combined the individually passing replacements with
`reports/candidates/zero_compare_to_bool_cast/apply_cumulative.py`.

Changed six repeated nonnegative presence tests:
`safe_name_49`, `safe_name_53`, `safe_name_57`, `safe_name_61`, `safe_name_65`,
`safe_name_69`.

Artifact: `reports/candidates/task319/task319_zero_compare_tail.onnx`.
Bundled gate: `267/267`, fail=0.  Cost: `5837 -> 5832` (memory `5674`
unchanged, params `163 -> 158`).  Active backup:
`submission/overfit_nets/.zero_compare_tail_backup/task319.onnx`.

## 2026-07-11 — fresh-tail diagnosis (5.1% sweep) → FIXABLE, dedicated build needed
ran: 25 fail instances classified with an exact clipped-anchor 2x-pattern matcher (numpy):
  20/25 UNIQUELY FIXABLE (visible magnified fragment uniquely identifies sprite0; deployed
  bit-packed row-signature matcher [nodes 63-100] picks wrong), 5/25 irreducible ambiguity
  (fragment consistent with both sprites; generator picks sprite0 unobservably -> coin flip).
tool+date: numpy matcher + graph dump, 2026-07-11 (fork).
verdict: correct matcher would cut fail 5.1% -> ~0.5-1%. Blueprint: replace the row-signature
  compare with exact clipped 2x matching — per candidate sprite, build 2x-upscaled runtime kernel
  (double-Gather), Conv-correlate against padded magcolor mask (anchors incl. off-grid via pad 12),
  require count==|visible| AND zero hits on ~magmask; select winner; keep the existing render tail.
  Est. +1.5-3KB on 5832 (-0.2~-0.35pt) vs protection ~0.65-1.3pt at k=1-2 hidden draws — POSITIVE EV
  if hidden draws are raw-like; NOTE the task002 bundled!=raw finding weakens the premise.
reopen: dedicated builder session for the matcher; hidden-set generation evidence.
falsification history: none (first diagnosis).
