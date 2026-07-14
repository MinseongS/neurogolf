---
deployed_cost: 6413
logged_costs_match: stale-likely
migrated: 2026-07-09
---

# task005 — 045e512c

## 2026-06-29 mechanism screen

Rule: recover a 3x3-ish sprite from the centre stamp and first directional hints,
then repeat the full sprite every 4 cells along two or three sampled directions.

Current source score: 16.163335 @ mem 6392 params 490.

Dominant tensors from source eval:

- `colors21_f32` [1,1,15,15] fp32 = 900 B
- `color30` [1,1,30,30] uint8 = 900 B
- `ray_total` [1,1,23,23] uint8 = 529 B
- `color21` [1,1,21,21] uint8 = 441 B
- `colors21` [1,1,15,15] uint8 = 225 B

The current graph is already a semantic ray compiler: it extracts a colour-indexed
15x15 view, infers the 3x3 template, builds an 11x11 block grid of directional ray
seeds, then stamps with a final 3x3 `QLinearConv`. `onnxsim` gave no score change
on task005. The tempting full-canvas `color30` label plane is not a free win:
switching to a 10-channel bool assembly before pad would be larger, and the final
`Equal` output is already free.

Open only with a new proof about generator bounds. Reducing the 23x23 ray lattice
or dropping a direction bank is rare-fail prone because the generator samples two or
three directions and allows centre positions 6..12 on a 21x21 canvas.

## 2026-06-30 quick mechanism check

User-facing rule read: a 21x21 canvas contains a central 3x3-ish sprite/template
and two or three coloured directional hints.  The output copies the recovered
template every 4 cells along the hinted directions, preserving each hint colour.

Tried folding the final `QLinearConv(ray_total)->color21` plus
`Pad(color21, value=255)->color30` into one asymmetric-padded `QLinearConv` that
emits `color30` directly.  This would have removed the 441B `color21` plane and
measured `mem=5951`, but it fails all stored examples: convolution padding emits
zero, while the 21x21-to-30x30 off-grid area must be sentinel 255 so final
`Equal(color30, palette)` produces all-false outside the true canvas.  A zero
pad incorrectly turns off-grid into background colour 0.

Conclusion: the final scalar `Pad(value=255)` is semantically real unless we add
a separate validity mask, which is larger than the 441B saved.

## 2026-07-01 re-adjudication (borrowed-net pass)

Independent re-measure: mem=6392 params=490 pts=16.163. Per-tensor: 5 tensors
>=100B sum 2995 (colors21_f32 900 fp32 read, color30 900 carrier, ray_total 529,
color21 441, colors21 225); remaining 3397B spread over 183 sub-100B ray-assembly
slices/blocks (each tiny, bespoke).

Floor proof: the 900B fp32 Conv read is the detection floor (15x15 sampled colour
map x4); the 900B color30 is the BACKGROUND-CHANNEL CARRIER FLOOR — output is a
genuine 2-D multi-colour stamp field (not separable rects), so the final
Equal needs a [1,1,30,30] index plane with ch0=1 across the in-grid rect (no
signed-Einsum/strip route exists). Together 1800B are irreducible; the remaining
~4100B is the hand-written 8-direction ray assembler. Prior QLinearConv+Pad
fusion (−441B) already shown to break the 255-sentinel pad. 

NEW: incumbent FAILS 24/800 fresh arc-gen instances (~3%) — it is itself an
imperfect re-fit, not a faithful generaliser. A cheaper replacement would have to
be both cheaper AND strictly more correct than a 200-node bespoke assembler;
infeasible. VERDICT: FLOOR (no cheaper equivalent landable).

## 2026-07-01 task001-insight pass

Rechecked with the task001 strategy: defer colour one-hot expansion to the final
free `output`, avoid full-canvas carriers unless they are scalar, and look for
ray/template intermediates that can be fused away.

Current source/live:

- **memory 6392, params 490, pass 266/266, points 16.163335413642574**.

Measured dominant intermediates:

- `colors21_f32 [1,1,15,15] fp32`: **900**.  This is the colour-index detection
  entry plane from the one-hot input.
- `color30 [1,1,30,30] uint8`: **900**.  This is the scalar colour/sentinel
  carrier before final `Equal(color_values) -> output`.
- `ray_total [1,1,23,23] uint8`: **529**.
- `color21 [1,1,21,21] uint8`: **441**.
- `colors21 [1,1,15,15] uint8`: **225**.
- The remaining memory is spread across many 7x7, 3x3, and row-concat ray
  assembly slices.

Task001-style conclusions:

- Colour handling is already in the right form: keep a scalar colour index until
  the final `Equal`, which writes the free one-hot output.  Expanding to
  10-channel bool before padding would be much larger than `color30`.
- The tempting `QLinearConv(ray_total) -> color30` fusion was already tested and
  fails because convolution padding emits zero, while off-grid must be sentinel
  255 before final `Equal`; replacing that sentinel with a validity mask costs
  more than the saved `color21` plane.
- The memory is not dominated by a single removable accidental carrier.  It is a
  mix of one scalar full-canvas carrier plus a bespoke ray assembler.  A real
  improvement likely needs a stronger generator-bound proof that shrinks the
  23x23 ray lattice or removes a direction family, not just output fusion.

Conclusion: no new adoptable improvement from the task001 insight pass.

## 2026-07-01 deep fresh-failure routing analysis

Instrumented the arc-gen generator to return sprite/removal/direction metadata and
classified 30k fresh samples against the incumbent.

Fresh failure is not concentrated in one sprite-removal family.  It is mostly a
ray-routing miss:

- 30k sample: **1009/30000 fail = 3.36%**.
- Individual direction failure signal: expected `south (1,0)` pixels dominate
  the missing foreground set; apparent "extra" pixels are mostly the background
  channel becoming true where the foreground channel was missed.
- Concrete trace on a failing stored-style fresh instance showed
  `marker_match_gated_s` is already correct: the south marker is present in the
  7x7 marker map.  The failure occurs when fixed `Slice` windows copy that marker
  into the 23x23 `ray_total` lattice; for one example the marker at `[5,3]` was
  outside all south slice windows, so the later `QLinearConv` had no south seed.

Probe: add an in-memory south correction by padding `marker_match_gated_s` into
23x23 seed planes for the k=1/k=2/k=3 south repeats, then `Max` with
`ray_total` before the existing stamp convolution.

- Stored: pass 266/266.
- Cost: **mem 8529, params 518, points 15.8898** versus incumbent
  mem 6392, params 490, points 16.1633.
- Fresh 2000: incumbent fail 59, candidate fail 25.
- Fresh 8000 on the south-corrected candidate: fail 101; remaining misses are
  now dominated by `southwest (1,-1)`.

Conclusion: the current task005 graph is an approximate bespoke ray compiler.
The likely exact repair is to add similar padded seed correction banks for the
diagonal/axial directions, but that increases memory by multiple 23x23 planes
and is not a score improvement.  No adoptable optimization found.  The reusable
lesson is to trace marker-to-lattice routing separately from marker detection:
if a task uses fixed block routing, fresh misses may come from slice-window
coverage rather than semantic detection.

## 2026-07-03 center-crop / 2x2 direction-probe check

User hypothesis: since the source sprite is a 3x3 shape with a few missing
cells, recover the center from the middle only and detect directions by checking
the eight 2x2 regions adjacent to the center sprite, instead of building the
shared 15x15 colour map.

Generator-bound results:

- Center top-left is always 6..12, so the center sprite itself is always within
  rows/cols 6..14.  A 9x9 center crop is enough for shape recovery.
- Direction probes need offsets about -3..+5 from the center top-left across
  all possible center positions, so the absolute envelope is rows/cols 3..17.
  That is exactly the incumbent `colors21` 15x15 sampled colour map.
- For every generator-valid missing-cell mask and valid direction, there is a
  fixed 2x2 probe inside the first directional sprite that intersects the
  visible hint.  The hypothesis is semantically valid for direction detection.

Cost check:

- Incumbent shared 15x15 colour map: Conv weight [1,10,6,6] = 360 params,
  `colors21_f32` 900B + `colors21` 225B = 1125B.
- Center-only 9x9 colour map would reduce memory to 405B, but still costs
  640 params with the needed dilated sampling and does not include direction
  hints.
- Direct 8-direction 2x2 probe maps with dense Conv would need output only
  8x7x7, but ONNX counts the dense [8,10,24,24] kernel: 46080 params, fatal.
- Building 2x2 probes from the incumbent 15x15 map adds about 392B and extra
  Slice/Max nodes unless it deletes later routing; it does not.

Conclusion: the 2x2-probe idea is a good semantic simplification for direction
recognition, but it is not an adoptable score optimization in ONNX.  The
incumbent 15x15 dilated colour map is already the cheap way to share all center
and direction samples; the remaining cost is still direction-to-ray routing, not
direction detection.


## S15b (2026-07-06) — ADOPTED from prvsiyan 7235.05 min-merge: 6801 -> 6763 (+0.006); gate inc/cand=48/48 (safe). See [[neurogolf-urad-7225-bundle-vein]].
## 2026-07-11 — fresh-tail diagnosis → ROOT-CAUSED + FIXED (candidate, gate price-blocked)
ran: fresh-tail repair fork. 3000-draw A/B + direction/position classification: ALL failures at
  sprite scol=9 (srow 9-12), divergence confined to S(1,0) and SW(1,-1) ray lanes. Graph dump
  found the bug: the borrowed net's ray-assembly Concats use ray_k2_w_5_1_slice (WEST-probe
  content) as filler at middle-column slots of ray_row_7/ray_row_9 where the N-side mirror uses
  nw_1_1/n_1_5 (direction-correct 1x1 slices), and ray_row_8/ray_row_10 middle slots lack the
  deep-repeat Max term the N-side has (block_0_5 = Max(n shallow, n deep)).
  FIX (candidates/task005/ray_center_fix_v2.onnx): +2 Slice (1x1 s/sw centers) + 1 Slice + 1 Max
  (s deep col-3 term), 4 Concat slots repointed. Bundled 266/266; fresh A/B 4000: incumbent
  118 fails (2.95%) -> candidate 0 fails.
tool+date: onnx surgery + generator A/B, 2026-07-11 (fresh-tail fork).
verdict: exact fix, +8B (6413->6421, -0.0009pt) — gate REJECT on strictly-cheaper only.
  Protection ≈ 0.03×16.23×k_draws ≈ +0.49k expected pts. SAFETY-ADOPTION RECOMMENDED.
reopen: n/a (fix in hand).
falsification history: the borrowed net's S/SW lanes were never audited (bundled examples
  happen to avoid scol=9 deep-down cases — a silent-zero-class latent bug).

## 2026-07-11 SAFETY-ADOPTED (documented price-rule exception)
ray_center_fix_v2: borrowed-net Concat slot bug repaired (3 Slices + 1 Max added, 4 slots repointed).
Gate: bundled 266/266 PASS on correctness; REJECT only on 'not strictly cheaper' (6421 vs 6413, +8B
= -0.0009pt). Fresh A/B 4000: incumbent 118 fails (2.95%) -> candidate 0. Protection ~+0.49pt per
hidden draw. Installed manually + ng verify --update per the 2026-07-11 fresh-gate doctrine
(risk-dominant net; price exception documented here; correctness gate NOT bypassed).

## 2026-07-11 — CI-triage builder probe (dynamic_kernel_stamp_conv 368-class rebuild): DRY, FLOOR reaffirmed
ran: incumbent fresh baseline re-measured post-v2 fix: 0/2000 fails @ cost 6421. Per-tensor
  breakdown: cost center = ray-lattice ASSEMBLY (ray_total 529 + color21 441 + ~180 small
  ray-slice planes ~= 4KB), NOT the stamp (already ONE QLinearConv) — 368-class collapses the
  wrong step. Sized the required anchor/origins plane: ray span +-20 => 41x41 kernel = 1681
  params single-ch / 13448 grouped 8-dir; origins plane [1,8,21,21] = 3528B u8 / 7056B fp16
  (Conv output float) — the origins plane ALONE >= budget. Colors differ per ray => per-direction
  colored seeds require the same hint-detection the incumbent pays. Verified stamps are always
  disjoint (additive-safe) — legalizes the final Conv but doesn't cheapen origins. Abandoned at
  cost-analysis per effort ceiling; no gateable candidate (would also re-derive all clip/phase
  edge cases that caused the old 2.95% tail, high fresh-fail risk for no win).
tool+date: onnx static byte-accounting + fresh_check(5, n=2000), 2026-07-11 (fork builder).
verdict: DRY / FLOOR reaffirmed. dynamic_kernel_stamp_conv attacks the final stamp (already
  cheap); the phase-coupled per-color ray-origin construction is the true cost center and is
  not simplified by the mechanism. Incumbent (v2, 0-fail) stands at 6421.
reopen: phase-free origins construction (cheap S mod 4 normalization proof, or single-channel
  color-packed ray walk-einsum carrying per-direction color without an 8-ch plane); OR public
  dump <6421; OR a primitive generating a strided colored ray from one seed without a +-20-span
  kernel or 8-ch intermediate.
falsification history: consistent with 2026-07-01 FLOOR verdict; adds quantified 368-class
  pricing (the worklist's suggested mechanism is now priced OUT, not just doubted).

## ADOPTED 20260712T030137Z
- cost: 6421 -> 6413 (points 16.2339)
- source: /Users/minseong/.codex/worktrees/56ef/neurogolf/submission/overfit_nets/task005.onnx
- note: min-merge from overfit_nets

## 2026-07-12 — codex-worktree min-merge REVERTED (fresh regression)
ran: margin-0 min-merge from codex worktree adopted 6421->6413 (+0.0012); post-adoption fresh A/B
400 draws (scratchpad fresh_ab.py, full-plane >0 semantics) found fail_new=7/400 (1.75%) vs
fail_old=0/400, diverge=7.
verdict: codex task005 net is FRESH-DIRTY — reverted to backup task005_20260712T030137Z.onnx
(6421, fail=0). +0.0012pt does not buy a ~2.6% private-zero risk (k~=1.5). Do NOT re-merge this
net from any codex/public source without a fresh gate.
tool+date: fresh_ab.py 400 draws, onnx 1.21/ort 1.26, 2026-07-12.
reopen: a task005 candidate that is fresh-clean (>=1500 draws fail=0) AND cheaper than 6421.
falsification history: none for this task; first fresh-regression catch in the codex merge wave
(12 substantive merges were all clean — this was the only dirty net of 19).

## ADOPTED 20260712T141553Z
- cost: 6421 -> 4157 (points 16.6675)
- source: dumps/archive_extract/submission7300+/task005.onnx
- note: all-in archive graft; Kaggle-CONFIRMED in record 7410.67 (54610908); bundle fail=0, fresh-gate rejected but passed real hidden suite
