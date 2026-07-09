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

## 2026-07-07 — compiler-mechanism re-open: runtime Conv template anchors

New cross-task mechanism found from task089: ORT accepts a runtime tensor as the
second input to `Conv`, so a template recovered from the input can become a
non-initializer convolution kernel.  A local sanity probe is stored at
`reports/candidates/task101/dynamic_template_anchor_probe.py` and confirms the
runtime-weight path in the current environment.

This invalidates the strongest old blocker for the maximal reference-template
anchor map: we do not necessarily need to enumerate red cells with
`TopK(k_raw=29)` and then run the copy1/copy2/copy3 coverage chain.  Candidate
compiler shape:

1. keep the current 4x4 reference extraction for `p1_patch`/`p2_patch`;
2. expand the dynamic red template to scale 1/2/3 kernels;
3. run `Conv(red_plane, red_kernel_s)` and compare with the dynamic red-cell
   count to get anchor maps;
4. apply the existing 4-neighbor separation and maximal-scale suppression on
   anchor maps, not on a 29-wide red-cell list;
5. stamp hidden blue from dynamic blue kernels into the free final
   `Where(blue_mask30, e1_vals, input)` output.

Expected win, if it lowers cleanly: delete `rawpos_score`/`TopK(k_raw=29)` and
much of the sequential raw-candidate coverage arithmetic.  Remaining risks:
dynamic `Conv` uses fp32 tensors in the proven path, scale-2/3 kernels are bigger
than the current sparse gather arithmetic, and the maximal-scale suppression may
recreate enough full-map masks to lose the byte trade.  This is now a real
compiler candidate, not a floor proof.

## 2026-07-07 — direct 30x30 mask scatter tail probe KILL

Candidate: `reports/candidates/task101/task101_direct_30_mask_scatter.onnx`.

Hypothesis: replace the tail
`ScatterElements(false[340]) -> Reshape[1,1,17,20] -> Pad[1,1,30,30] -> Where(output)`
with a direct scatter into a 30x30 flat mask, deleting the counted Pad output.

Result: bundled fail=0 but cost worsened badly: active 12928+822 -> candidate
16768+1385.  The 17x20->30x30 flat-index retarget requires full 136-vector
`Div/Mod/Mul/Add/Where` int32 carriers plus a larger false base initializer, so
the Pad deletion is more than erased.

Conclusion: tail-only sparse overlay is not the lever.  A real win still needs
the runtime-template compiler to remove the raw red-candidate TopK/coverage
path before the tail.

## 2026-07-08 S32 — runtime-template compiler byte analysis

Artifacts:

- `reports/candidates/task101/dynamic_qlinearconv_probe.py`
- `reports/candidates/task101/dynamic_convtranspose_probe.py`
- `reports/candidates/task101/runtime_template_cost_model.py`
- `reports/candidates/task101/runtime_template_compiler_analysis.md`

New primitive checks:

- Runtime tensor as `ConvTranspose` weight works in ORT, but fp32 stamping is
  too expensive if used literally.
- Runtime tensor as `QLinearConv` weight also works.  With scale=1 and zp=0 it
  returns exact uint8 template-correlation counts; task101 red-template counts
  are safely below 255.

Byte model result:

- fp32 runtime Conv + ConvTranspose, no suppression: estimated memory `16070`
  vs active `12928` (bad).
- fp32 with maximal suppression: `21310` (dead).
- QLinear runtime matching + QLinear reversed-kernel stamping, no suppression:
  estimated memory `11696` (about `-1232B`, potentially good).
- QLinear with full maximal suppression: `15308` (bad).

Conclusion: the compiler is real, but the viable form is narrow.  It must use
runtime-weight `QLinearConv` for both matching and stamping; full maximal
coverage suppression erases the win.  Next target is a cheap-filter candidate
matching the earlier 4-neighbor/separation oracle, then patch only the remaining
bundled false positives if they fit under the ~1232B margin.

Follow-up oracle (`qlinear_anchor_oracle.py`):

- raw anchor maps: fail `79`, all failures are extra blue (no misses after
  allowing negative/off-grid anchors).
- 4-neighbor separation: fail `19`, extra blue `83`, no misses.
- scale-priority overlap: fail `0`.
- full maximal suppression: fail `0`.

The 19 separated failures are small-scale submatches inside larger red copies.
Correctness does not require the current sequential copy claim machinery; a
scale-priority overlap rule is enough.  But the obvious ONNX lowerings are still
too expensive: direct full-grid red coverage and an anchor-coordinate bank both
estimate around `15.3KB`, worse than the `12.9KB` active.  A score win needs a
sub-1232B suppression lowering specialized to the `<=2` reference-red-cell
case; otherwise the QLinear compiler is a correct-but-not-cheaper rewrite.

## 2026-07-08 S33 — anchor-coordinate raw-bridge suppression

New oracle artifact:

- `reports/candidates/task101/anchor_coordinate_suppression_oracle.py`

Result: the full scale-priority overlap rule can be narrowed.  It is enough to
suppress only:

1. raw scale-3 anchor coordinates -> scale-2 anchor coordinates;
2. raw scale-2 anchor coordinates -> scale-1 anchor coordinates.

The second stage deliberately uses **raw** scale-2 anchors, not accepted scale-2
anchors.  This creates a bridge: a scale-3 match suppresses a scale-2 submatch,
and that raw scale-2 submatch suppresses the scale-1 submatch.  The oracle is
bundled `266/266`, extra `0`, miss `0`.  It rejects only 30 candidates on the
bundle (`3->2`: 1, `2->1`: 29).

Updated byte model:

| QLinear compiler estimate | memory | vs active |
|---|---:|---:|
| no suppression | 11696 | -1232 |
| two-stage anchor-coordinate suppression | 12914 | -14 |
| full maximal suppression | 15308 | +2380 |

This is the first correct suppression design that is plausibly under active,
but the margin is only about 14B in the dense-kernel model.  An ONNX candidate
must reuse existing padded anchor maps and QLinear scales, avoid a direct
scale-3 -> scale-1 map, and avoid any extra full-grid carrier.  Otherwise it
will be correct but not a score win.

## 2026-07-08 S33b — real ONNX QLinear splice KILL

Artifact:

- `reports/candidates/task101/build_qlinear_anchor_splice.py`
- `reports/candidates/task101/task101_qlinear_anchor_splice_no_suppression.onnx`
- `reports/candidates/task101/task101_qlinear_anchor_splice_full_coverage.onnx`

Implemented a real runtime-QLinear anchor/stamp compiler by splicing the
incumbent graph up to reference extraction, then replacing raw TopK/copy
claims/Scatter tail with dynamic QLinearConv anchor maps.  Correctness required
several details that the hand byte model underpriced:

- red input pad left/right/bottom for negative/off-work anchors;
- boundary-aware separation with `K+2` kernels, because false submatches can
  touch red just outside the template bbox;
- scale-1 dynamic reference-bbox red-overlap exclusion;
- reversed dynamic kernels for QLinearConv stamping.

Measured results:

| candidate | bundled | memory | params | verdict |
|---|---:|---:|---:|---|
| no suppression | 247/266 | 26140 | 278 | fail 19 |
| full coverage suppression | 266/266 | 31886 | 278 | correct but worse |
| active | 266/266 | 12928 | 822 | keep |

Conclusion: this lowering is dead for score.  The earlier no-suppression
estimate (`~11696B`) was falsified by the actual scorer; even with free
suppression the measured base is already far above active.  Do not keep trying
minor suppression-bank variants in this graph shape.  A future revival would
need to avoid the dynamic anchor/stamp carriers themselves, not just suppress
them more cheaply.

## 2026-07-08 S35 — REJECTED dynamic-CSE micro tail

Artifact: `reports/candidates/task101/task101_dynamic_cse_greedy.onnx`.

This does **not** revive the QLinear anchor compiler. It is a mechanical active-net peephole:
the dynamic CSE pass merged duplicate live carriers (`safe_name_100 -> safe_name_98`) without
changing semantics. Bundled gate: `266/266`, memory `12908 -> 12904`, params unchanged `817`,
cost `13725 -> 13721`. The combined micro-tail submission **54451532** completed at publicScore
`7248.82` versus best `7264.29`; the later full-sweep submission **54451744** was also bad
(`7248.83`). The overlay was reverted from
`submission/overfit_nets/.micro_tail_backup_20260708/task101.onnx`.
