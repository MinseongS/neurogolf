---
deployed_cost: 11410
logged_costs_match: stale-likely
migrated: 2026-07-09
---

# task198 — 83302e8f

**Rule:** size×size cell grid, each cell minisize×minisize px, cells separated by 1-px
`color` lines (pitch p=minisize+1, actual_size=size·p−1≤29). Input: black canvas + color
grid-lines, some line pixels punched back to black ("permeable points"). Output: green
canvas, same lines; every permeable point → YELLOW(4); a cell interior → YELLOW(4) if ANY
of its 4 walls has a gap, else GREEN(3). DEPTH-1 (each gap marks exactly the 2 cells it
separates) — NOT a transitive flood. (Memory's "task198 = flood wall" referred to our OLD
flood net; the actual ARC task is closed-form & separable.)

**Current (deployed):** 15.17 pts, ext:kojimar7113 (crowd MaxPool×8 net, f32→f16→bool entry
triplication), mem≈18600.
**Target tier:** A/B closed-form (separable cell-mark via selector MatMuls + Gather upsample).

## Attempts
| # | angle | tier | mem | params | pts | fresh | outcome |
|---|---|---|---|---|---|---|---|
| pre | existing custom (fp16 everywhere) | B | 30460 | 135 | 14.67 | — | below P |
| v2 | uint8 carriers + uint8 Equal output | B | 25206 | 136 | 14.86 | — | below P |
| v3 | shared isblack16, separable gap downsample | B | 21786 | 140 | 15.00 | — | below P |
| v6 | double-Gather uint8 upsample (kills fp16 1800 upsample plane) | B | 19487 | 143 | 15.12 | 200/200 | **best, still <P** |
| v9/11 | sentinel-interiorL to drop gridL/ingrid | B | 19701/19821 | — | 15.10 | — | net-worse (online needs 3 planes; line spans orthogonal off-grid axis) |

## Best achieved
15.115 @ mem 19487 params 143 — adopted? N (do-not-adopt). Beats prior 15.17? **NO** (−0.05).

## Irreducible-floor analysis
Three structural plane groups, all required for an EXACT net:
1. **Entry blackness plane = 3600B fp32** (Slice ch0). Required because the OUTPUT marks every
   permeable point (black-on-a-line, arbitrary (r,c)) YELLOW → a full 2-D `isblack∧online`
   plane is unavoidable. Conv on fp32 input yields fp32 (probed: declaring fp16 output →
   ORT type error); Cast(input→fp16) is 18000B. So one 3600B fp32 entry is the floor.
2. **Cell-mark gap pipeline ≈ 4320B**: isblack16 fp16 (1800, shared) + 6×[1,1,S,30] fp16
   selectors/matmuls (2520). fp32 variant (use ch0 directly, no isblack16) costs MORE
   (6×840 fp32 small > 1800+6×420). The gap "any-gap-in-cell" aggregation genuinely needs
   the cell-space reduction (separable double-MatMul Rsel@isblack16@Csel).
3. **Compose = 6 full planes ≈ 5400B**: online_b, ingrid_b, interiorL, lineL, gridL, L
   (all uint8/bool 900). Cannot drop below 6: `online=(rline|cline)&ingrid` needs 3 full
   planes because a line on an in-grid row STILL spans off-grid columns (per-axis in-grid
   folding into rline/cline is INCORRECT — verified it leaks colour into off-grid). The
   sentinel-interiorL trick (off-grid→99 via padded Gather) lets the L-gate drop but the
   correct `online` then needs onv+onh+or = 3 planes → wash.
Total ≈ 19.5K → 15.12, structurally ~0.05 BELOW the kojimar MaxPool net (15.17) and +0.35
below the +0.3 win threshold (needs mem ≤ 13772 ⇒ would have to delete the fp32 entry AND
halve both the cell-mark and compose — not possible for an exact net).

## OPEN ANGLES (re-attack backlog)
- Gather-downsample the gaps at WALL columns only (wall_cols=(i+1)·p−1 via runtime ramp) to
  shrink the 30-wide gap intermediates to S-wide — saves maybe ~1KB of small planes, NOT
  enough to clear +0.3.
- Match kojimar's localized-MaxPool-fill formulation (avoids the cell-space round-trip) — but
  that net is ALREADY 15.17; replicating it can't beat it by +0.3.
- The +0.3 gap is structural: an exact closed-form net for this rule floors ~15.1-15.2.

## INSIGHT (transferable)
⭐ **Gather-upsample beats MatMul-upsample for a uint8 cell→pixel expansion**: a cell-space
value plane interiorCell[S,S] upsamples to pixel space as `Gather(Gather(cell,Ridx,ax2),Cidx,ax3)`
with int32 index vectors — Gather PRESERVES uint8, so the [1,1,30,30] result is 900B and the
only full plane, vs the double-MatMul which forces a 1800B fp16 plane + a 900B uint8 cast
(task198 v3→v6: 21786→19487, +0.12). Clip indices in fp16 (Clip rejects int) then Cast int32.
⭐ **uint8 Equal/Where/Gather all run under ORT_DISABLE_ALL** (probed) — declare the final
colour-index carrier uint8 (900B) feeding `Equal(L_u8, chan_u8)→bool output` (free); 2× cheaper
than the fp16-Equal carrier.
⭐ **Per-axis in-grid folding into line masks is INCORRECT when lines span the orthogonal axis**:
`online=(rline&ringrid)|(cline&cingrid)` leaks colour at (in-grid-line-row, off-grid-col)
because rline alone is true there. A correct in-grid line gate needs the full cross-axis AND
(3 planes). Watch this on any line/grid task with an in-grid mask.
⭐ Memory's "task198 = flood wall (infeasible)" was a FALSE label tied to our old flood net —
the real ARC task is closed-form depth-1 wall-gap marking, fully separable. Re-triage
"flood/connectivity wall" labels against the GENERATOR, not the deployed net's op signature.

## S10 (2026-07-03) — bobmyers7186 teacher ADOPTED (+0.000)
**Mechanism (op-census diff):** Zero-masking recast from **uint8-`Mul`** (`zero_u8`) to **fp16-`Where`** (`where_mask_zero_f16`): Cast 11→9, Mul 6→3, Where 1→4. −6B.
**Old→new:** mem 13434→13428, params 138→138.
**Gate:** bundled cand fail=0; fresh N=2000 inc_fail=0 cand_fail=0. No TopK reject.
Backup `reports/retired_networks/task198_pre_s10.onnx`; source `public_candidates/bobmyers7186/task198.onnx`. Gate data: scratchpad/gate_small/results.jsonl.
No transferable mechanism — minor trim.


## S15 (2026-07-06) — ADOPTED from urad public bundle 7225.82 (submission 54367833): 12806 -> 12298 (+0.040)
Mechanism: value_info Slice crop + CumSum.
Gate (fresh_verify, inc/cand fail on 1500-2000): 0/0 -> adopted under safe rule (cand fail <= inc fail AND cheaper).
Source-owned via live_to_exact_source --write-src; re-measured grader-side fail=0. Backup in scratchpad/backup_networks.
See memory [[neurogolf-urad-7225-bundle-vein]]. 

## S15b (2026-07-06) — RE-ADOPTED from prvsiyan 7235.05 min-merge notebook (further golf): 12298 -> 11412 (+0.075)
Gate fresh_verify 1500: inc=0/0 (cand<=inc, safe rule). prvsiyan bundle = min-merge of public sources, had a cheaper variant than my prior net. Source-owned via live_to_exact_source, re-measured fail=0. See [[neurogolf-urad-7225-bundle-vein]]. 

## 2026-07-07 auto-golfer scalar-line-colour probe KILL

Candidate: `reports/candidates/task198/scalar_line_color_probe.py` writes
`reports/candidates/task198/task198_scalar_line_color.onnx`.

Hypothesis: active `Conv(input,W)->G` already has the colour-index plane, so
delete the full `G_u8` cast plane.  Build `eq1` directly from `G`, infer pitch
from a small `eq1` slice, sample the one grid-line colour as scalar `G[0,p-1]`,
and use `vals=Where(eq1, cellexp, line_colour_scalar)`.

Result: bundled fail 266/266.  Cost would have improved (`11305+107` to
`10530+108`) if correct, so the byte model was valid.  Semantic failure:
`G=0` represents outside the real grid on the 30x30 padded canvas, while `G=1`
is black inside the real grid.  The scalar-line replay paints every non-`eq1`
cell, including off-grid zeros, as the line colour.  Preserving off-grid zeros
requires another full mask/replay plane, which erases the `G_u8` deletion.

Do not retry scalar line-colour replay unless a free-output construction can
distinguish off-grid `G=0` from line pixels without a counted full plane.

## ADOPTED 20260712T140209Z
- cost: 11410 -> 7922 (points 16.0226)
- source: /Users/minseong/project/neurogolf/dumps/archive_extract/submission7300+/task198.onnx
- note: archive.zip submission7300+ net; fresh 2000/0 fail; mechanism-graft

## 2026-07-13 REGIME-CRACK attempt on `labf` (Conv fp32 3600B) — VERDICT: FLOOR (fp32-co-bind)
- **Ran:** graph dump + semantics decode; built free-output base-recolour einsum
  (`candidates/task198/build_regime.py` -> `regime.onnx`); gated both; ORT mixed-dtype probe.
- **Tool+date:** `uv run ng gate` (onnx 1.21/ort 1.26) + ORT session-build probe, 2026-07-13, opus.
- **Structure:** `labf = Conv1x1(input)` fp32 3600B (~45%) -> `lab` u8 900B = per-pixel base
  colour-index READ of the one-hot fp32 input (black->green, line passthrough, ch0 merges ch3);
  `out_label = Max(lab, room_u8, doors)` u8 900B; `output = Equal(out_label, colors)` = FREE.
- **Why floor (both routes >=3600B fp32):**
  1. Scalar-index route (current): any linear reduction of the *fp32* one-hot input to a
     per-pixel index inherits fp32 by co-bind => [1,1,30,30] fp32 = 3600B (=labf), then Cast u8.
     No 1-byte einsum/conv (harness feeds input fp32 always -> no u8-input escape; ORT has no u8 Einsum).
  2. Free-output-einsum route: **FACT A (verified)** base recolour alone folds to the FREE output
     via `einsum('bkrc,kj->bjrc', input, M)` — measured **memory=0, cost=100**, but bundled
     **fail=266/266** (gap pixels stay green): the data-dependent overlay (bg AND gap -> yellow)
     is essential and is an ADDITIVE second einsum term with an incompatible contraction skeleton.
     Merging base+overlay into one free-output node needs an augmentation gate operand
     `G[t>=2,30,30]`; **FACT B (verified)** ORT REJECTS an fp16 gate co-bound with fp32 input
     (uniform-T rule) so G is forced fp32 = 7200B > the 3600B it replaces. Overlay's
     separable(doors)+block(room 7x7) structure only shrinks G's upstream build, not the
     materialized [2,30,30] fp32 operand. Net cost ~= 9722 > 7922 => LOSE.
- **Taxonomy:** fp32-co-bind economics deep floor (playbook free-output-einsum.md); CONV-FP32
  arsenal member where the counted plane is a genuine per-pixel fp32 colour READ, not a bool
  Where routing mask. Consistent with the prior "scalar-line-colour" KILL (off-grid 0 disambiguation).
- **Reopen trigger:** ORT gains a mixed-T Einsum/Conv kernel emitting a 1-byte carrier from an
  fp32 operand, OR harness begins feeding `input` as a 1-byte dtype, OR a new public dump ships a
  cheaper task198 net. Falsification history: none yet for this specific fp32-co-bind claim.
- **Evidence:** `candidates/task198/build_regime.py` (FACT A/B reproducible); base-only
  `regime.onnx` gates fail=266 cost=100. NOT adopted (deployed 7922 unchanged).

## ADOPTED 20260715T180737Z
- cost: 7922 -> 7915 (points 16.0235)
- source: candidates/task198/missingfreeops_shared_threshold.onnx
- note: shared threshold: rescale rm32 families and reuse cthr=15
