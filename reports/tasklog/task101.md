# task101 — 447fd412

## 2026-06-29 semantic rewrite screen

Rule: the generator draws 2..4 copies of one small connected sprite.  The first
copy has `bmag=1` and shows both blue(1) and red(2) pixels.  Later copies show
red pixels in the input but hide blue pixels; the output restores the hidden
blue pixels at the copy's magnification (`bmag` in 1..3).  Dimensions are
bounded: ordinary fresh grids are 12x14 or 14x17, while stored/test stress can
reach 21 columns.

Current source/live graph is a source-owned exact replay with:

- score `15.210521364075738`
- memory `16940`
- params `905`
- fresh check `1000/1000`

Dominant tensors from a traced mem profile:

| tensor | bytes | note |
|---|---:|---|
| `c1cf`, `c2cf` | 1428 each | fp32 channel slices over the 17x21 work grid |
| `crop3` | 1071 | final 3-channel bool crop before Pad to output |
| `rawpos_score` | 714 | fp16 score vector for TopK over red candidate pixels |
| 17x21 bool masks | 357 each | copy-slot coverage, blue mask, output channel masks |

Mechanism tested: transfer the task351/task201 copy compiler pattern
(`1x1 colour-index plane -> small gathers -> final one-hot`).  Reject for this
task: input colours are only 1/2, so the current direct ch1/ch2 `Slice` pair
costs `2 * 1428 + bool casts`, while a full colour-index entry plane would cost
`3600` before the same work-grid logic.  The usual colour-index copy lever is
not cheaper when the useful palette is already two fixed channels.

TopK-width probe: current graph keeps `k_raw=29` red candidate positions.  Stored
max red count is 22, but 20k fresh samples reached 30 red cells:

| source | max red count |
|---|---:|
| stored train/test/arc-gen | 22 |
| fresh 20k | 30 |

Adopt decision: **no rewrite**.  Reducing `k_raw` is unsafe under the generator,
and replacing channel slices with a colour-index plane is a net loss.  Current
fresh generalization is good (`1000/1000`), so leave the source graph unchanged.

Transferable negative insight: bounded multi-copy reconstruction tasks can look
like pure spatial-copy wins, but if the palette is fixed to a few channels,
direct channel slices can beat the standard colour-index entry plane.  Also,
candidate-list widths must be bounded from generator extremes, not only stored
examples.

## 2026-06-30 component/scale anchor research

User hypothesis: the hard part across copy tasks is not the visual rule, but
cheaply recovering component count, scale, and anchor.  Task101 is a good probe
because the generator has a small connected reference object, then places
upscaled copies where only the indicator/red cells are visible.

Generator source check (`generate_447fd412` from RE-ARC): reference object is a
connected object with bbox at most 4x4.  Later occurrences use
`outobj = upscale(obj, fac)` and input only `indic` cells; output restores the
hidden `main` cells.  This confirms that a component-level mechanism should be
possible in principle, but scale and anchor must be exact.

Stored/bundled analysis:

| probe | result |
|---|---:|
| bundled examples inspected | 266 |
| 4-connected red component count | max 8, unstable |
| 8-connected red component count | max 8, unstable |
| red cells | max 22 in bundled examples |
| naive red-to-blue offset spray | rejects: massive false positives |
| top-left template scan over reference red cells | 225/266 |
| red-anchored template scan, all red template cells visible | 212/266, miss 0 but 141 extra blue cells |
| + 4-neighbor separation filter | 262/266, miss 0 but 5 extra blue cells |
| + maximal-scale/coverage suppression | **266/266**, extra 0, miss 0 |

Mechanism candidate: replace raw red-cell enumeration with a **maximal
reference-template anchor map**:

1. Extract the reference component containing both colours.
2. Build dynamic red-template cells and blue-template cells for scale 1/2/3.
3. Candidate anchor is any placement where all in-template red cells are present.
4. Reject anchors whose 4-neighborhood touches unrelated nonzero cells, matching
   the generator's spacing invariant.
5. Process larger-scale matches first; suppress smaller matches whose red
   coverage is already covered by a larger accepted match.  This removes false
   positives caused by scale-1 submatches inside scale-2/3 red blocks.
6. Scatter only the blue-template cells for accepted anchors.

Why this matters: the current graph carries `TopK(k_raw=29)` over all red cells
and then walks copy slots with many `Gather`/coverage ops.  The Python oracle's
valid template candidates averaged 3.29 per example, max 6 on bundled data.  If
the dynamic template anchor map can be compiled cheaply, it may remove much of
the 29-wide raw candidate path.

Risk: this is not yet an ONNX improvement.  Dynamic template matching may still
be expensive because the reference red/blue masks are input-dependent, and the
maximal-coverage suppression may recreate a smaller version of the current
coverage chain.  Fresh generator is not available locally in this workspace, so
the 266/266 result is stored/bundled only; earlier tasklog fresh evidence still
applies only to the incumbent graph.

Next attack: prototype a source-owned candidate that keeps the existing 4x4
reference extraction but replaces `rawpos_score`/`TopK(k_raw=29)` with scale
template anchor candidates.  Reject unless stored passes and memory meaningfully
drops below the current `16940`.

## 2026-06-30 adoption attempts after anchor research

Goal: turn the anchor research into an actual score improvement.  No candidate
was adopted.

| candidate | stored | memory | params | points | outcome |
|---|---:|---:|---:|---:|---|
| incumbent | 266/266 | 16940 | 905 | 15.210521 | keep |
| output tail `uint8 class_idx -> Equal(output)` | 0/266 | 16769 | 917 | 0 | failed: padded/outside-grid cells became colour 0 instead of all-zero |
| output tail with invalid sentinel outside grid | 266/266 | 17126 | 918 | 15.199432 | correct but worse |
| `p1/p2` TopK score vectors as uint8 | 23/266 | 16898 | 906 | 0 | failed: ordering changed; original uses `0,-1,-2...` fp16 scores |
| `p1/p2` TopK score vectors as int8 | load fail | - | - | 0 | ORT has no int8 `Where` kernel here |
| omit TopK value output | load fail | - | - | 0 | ONNX TopK value output cannot be empty |
| `onnxsim` compression sweep | 266/266 | 16940 | 905 | 15.210521 | no gain |
| single Conv fit (`k=1,3,5`) | fail | - | - | 0 | channel 0 not separable |
| local public candidates | 266/266 | 16940 | 905 | 15.210521 | same as incumbent; URAD was slightly worse |

Conclusion: the cheap local edits around the existing graph do not improve the
score.  A real improvement still requires replacing the `rawpos_score` /
`TopK(k_raw=29)` copy-finding path with the maximal reference-template anchor
map, not just dtype or output-tail surgery.  That rewrite is nontrivial because
the ONNX graph must compile dynamic red/blue reference offsets and scale
suppression without recreating an equally large candidate/coverage chain.

## S3 re-fit pass (2026-06-30) — drop 10 no-op nodes, bit-identical
Removed 10 provably-true/no-op nodes from the copy1/copy2/copy3 scale-detection bounds-checks:
5× `Equal(x, x)` (tautological True on int tensors: copy{1,2,3}_scale_idx_rok/cok) and the
dependent `And(seen, ib)` chains where `ib`≡True, so `seenib≡seen` (copy1/2) and copy3's chain
collapses to its real row-clamp `rok`. Codegen placeholder bounds-checks that never clamp here.
- Before: mem 16940 / params 905 / 15.2105 pts.  After: mem **16850** / params 905 / **15.2156 pts (+0.0051)**.
- Gates: evaluate fail=0 (266/266); ORT_ENABLE_ALL fail=0; fresh_verify 2500 instances = candidate
  bit-identical to incumbent (0 divergence, including identical pre-existing OOB behavior), fail-vs-GT=0.
- Rest of graph is FLOOR: no other dead/dup nodes; the two 1428B fp32 channel slices are the proven
  per-cell detection floor; the two 357-elem arrays are mem↔params fungible (zero net change). LANDED.

## S8 (2026-07-02) — counting-model rebuild + CRASH FIX (+0.206) ADOPTED
Free-input einsum profiles for blue bbox; 4×4 reference patch via 'bchw,c,uh,vw->uv' with
OneHot selectors (OOB rows read zeros); nzc plane dropped; epilogue = single
Where(blue_mask30, e1_vals, input). 13573+874 vs 16880+875 → 15.216→15.422.
REAL BUG FIXED: incumbent HARD-ERRORS ORT on ~0.1% fresh (red target at rows 15-16 →
scale-probe Gather idx 357+ OOB; S3 refit removed non-tautological bounds checks) — silent
private-LB risk removed. Candidate pads c2_flat 357→401 + column-overflow guards (+476B).
Fresh (crash-tolerant gate): cached 2500 inc 2, cand 0; uncached 10000 inc 11, cand 0 —
all divergences = incumbent crash instances. TRAPS: ORT OneHot needs i32 depth (i64 depth
kernel unregistered); stock fresh_verify aborts on incumbent inference errors — scratchpad
fresh_gate.py counts them as fails (consider upstreaming).

## S11 (2026-07-03) — mech-15/pointer scout: KILL — output = data-dependent blue sprite-template scatter at searched anchors (scale 1-3, per-instance shape); cost = detection slices + TopK anchor search, no carrier. Same bucket as 233/285 (assignment/detection).
