# task133 — 57aa92db

**Rule:** A shared 3×3 "creature" shape S (`continuous_creature`, seed pixels[0]=(0,0) →
every cell offset (dr,dc)∈{0,1,2}², (0,0)=signature). 2–4 sprites; sprite idx has anchor
(brow,bcol), magnifier bmag∈[1,4] (bmag[0]=1), a distinct colour, and a "show" cell. The
OUTPUT draws, for every sprite, the FULL shape S magnified by m (each cell → solid m×m
block), signature cell in `pcolor`, all other cells in the sprite's colour. The INPUT draws
sprite 0 fully (the m=1 template) but for every other sprite only TWO blocks (signature in
pcolor + one cell in its colour). So OUTPUT = INPUT + the missing magnified blocks per sprite.
**Current:** 12.84 pts, `gen:thbdh6332`, mem 191249, params 206 — GENERALIZES (fresh_pass 60/60).
**Target tier:** B (closed-form exact reconstruction).

## ⚠️ PRIOR "INFEASIBLE / correspondence wall" VERDICT WAS FALSE
The earlier tasklog called this a connectivity/correspondence wall ("pcolor↔sprite matching
needs flood-fill; variable-count components; anchor not derivable"). **All of that is wrong.**
Everything is LOCAL and closed-form, verified EXACT 3000/3000 fresh + ISOLATED fresh 200/200:
- anchor = pcolor-block TOP-LEFT (P & ¬up & ¬left) — no matching needed.
- m per sprite = rightward SOLID RUN of the pcolor block (count-separable, blocks never touch).
- sprite colour = the colour 4-adjacent to the pcolor block, dilated over the block (≤3 steps).
- S (offset set) = OR-over-anchors of "any colour at anchor+(dr·m,dc·m)"; the m=1 template
  reveals all of S. Stamp = Where(shift(spriteColour_m, off·m)>0 ∧ Smask, that, out).
There is NO sprite enumeration, NO component labelling, NO `show`↔anchor correspondence — the
output is INPUT plus translated colour blocks. The bail intuition was the documented-FALSE one.

## Attempts
| # | angle | mem | params | stored | fresh | outcome |
|---|---|---|---|---|---|---|
| 1 | shift-stamp fp16, offsets {0,1,2} | 1.27M | 2784 | — | 200/200 | correct, over mem |
| 2 | uint8 + slice-shifts | 888K | 1582 | 11.0 | 200/200 | correct, over |
| 3 | fp16 adjacency, p=1 pads, merged Where-per-m stamp, drop redundant P | 642K | 1424 | 11.63 | 200/200 | correct, over |

## Best achieved
11.63 @ mem 641650 params 1424 — adopted? **N**. Beats 12.84? **NO (−1.21 stored)**.
EXACT + GENERALIZES (ISOLATED fresh 200/200, arc-gen 262/262), but heavier than the deployed
Gather net AND fails the 5 ARC-AGI ORIGINALS (train+test) → `evaluate.ok=False`.

## Irreducible-floor analysis (at-floor vs the DEPLOYED net)
The deployed 12.84 net is itself a generalizing exact solver using the **task195/159
magnify-Gather** (inspected: 18× Gather + Floor/Div/Clip = `gidx=clip(floor((i−anchor)/m))`,
per-sprite-slot peel) at 191K / 206 params — the EFFICIENT form of this reconstruction.
My shift-stamp net floors ~11.6 because a multi-sprite variable-magnify STAMP intrinsically
needs ~370 full 30×30 planes: stamp loop 4 m × 8 offsets (32), Smask OR 8×4 (32 back-slices +
ands), run/m-plane/colour dilations (~100), pcolor adjacency (one fp32 4-nbr Conv 36000 + fp16
[10,900] reshapes ~90K). Even all-uint8 working planes (900B) + merged stamp pin it at ~340K in
planes alone ⇒ ≤~11.9 < 12.84. **m is count-separable, placement is positional/deterministic
(no random bijection)** — fully closed-form — but per-offset×per-m SHIFTING is the wrong (heavy)
form; the deployed double-Gather already sits at the byte floor.

## OPEN ANGLES (re-attack backlog)
- ONLY path to beat 12.84: reimplement the magnify-Gather per-sprite-slot (task195/159) and
  shave under 141K. Low odds — would have to undercut an already-tuned 191K/206 gen export.
  Not attempted (re-deriving the deployed approach to win a few KB is poor EV).
- The 5 ARC-AGI ORIGINALS use WIDER creatures (validate ex1 width 4: S col-offset +3, pixels[0]
  not at the corner ⇒ offsets span −2..3). Extending the offset window to cover them ~4× the
  stamp/Smask loops (prohibitive) — and they are OUT-OF-DISTRIBUTION for the fresh-graded
  generator (which always has pixels[0]=(0,0), offsets {0,1,2}). So evaluate.ok is unattainable
  cheaply even ignoring the byte floor.

## INSIGHT (transferable)
⭐ "Variable-magnify is SEPARABLE" held — and this whole rule is closed-form & LOCAL (m = solid-run
scalar; per-sprite anchor = pcolor-block corner; colour propagates within the block) — so the prior
"correspondence/flood-fill wall" verdict was a FALSE bail. BUT separability ≠ a winning byte count.
A MULTI-OBJECT variable-magnify STAMP (≤K sprites × {1..4} mag × ≤9 offsets) costs ~O(K·mag·offset)
full 30×30 planes via shift-stamping (~340K+), which floors BELOW a deployed **magnify-Gather**
export. When a gen-import already uses 18× Gather + Floor/Div and `fresh_pass` confirms it
generalizes, it is at the EFFICIENT FLOOR — re-deriving by shifts LOSES. The right efficient form
is the per-sprite-slot double-Gather `out[i,j]=S[clip(floor((i−anc)/m))…]`, not per-offset×per-m shifts.
⭐ pcolor = the unique colour 4-adjacent to ≥2 DISTINCT other colours (component-count / corner-count
fail ~70% — a creature colour splits into ≥2 blobs / has ≥2 top-left corners). Build via 4-nbr
dilation-MatMul → [10,10], zero row0+col0 (bg neighbours everything).
⭐ This harness pads OFF-grid input AND target with ALL-ZERO channels (NOT ch0=1) → gate the output
one-hot by an in-grid mask = ReduceMax(input, axes=[1])>0, else off-grid ch0=1 fails every example.
