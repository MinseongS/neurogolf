# task018 — 0e206a2e

**Rule:** 1–2 "creature" sprites (continuous_creature, 6–12 px in a wide×tall box, wide+tall=9), grid 12–24.
Each sprite is placed TWICE. Placement A ("original"): full sprite = a mode-colour body + exactly 3 uniquely-
coloured marker pixels (color_list[0,1,2]); drawn in both input and output. Placement B ("clone"): a RANDOMLY
ROTATED copy (rot∈{1,2,3,4}); in the INPUT only its 3 markers are shown (mode-colour body hidden), in the
OUTPUT the full rotated body+markers are shown. The output ERASES the original entirely and shows only the
reconstructed clone(s). To solve from input you must: group the 3 clone markers per sprite, find the rigid
transform (rotation+offset) mapping the original's markers to the clone's markers, then stamp the rotated full
body at the clone. mode = most-frequent colour (shared body colour). With 2 sprites both sprites share the SAME
3 marker colours, so colour does NOT identify which sprite — grouping is spatial.

**Current:** 13.34 pts, deployed net = 1395 nodes (Conv/ArgMax/TopK/ScatterND/Mod/Gather…), mem-heavy.
**Target tier:** detection/reconstruction — but see floor analysis: this is an INFO-BOTTLENECK WALL.

## Attempts
| # | angle | tier | mem | params | pts | fresh | outcome |
|---|---|---|---|---|---|---|---|
| 1 | full numpy reference solver (CC-label originals, group clone markers, enumerate rot×offset, verify) | — | — | — | — | 297/300 | the 3 fails are GENUINELY AMBIGUOUS inputs (markers→multiple distinct valid outputs) |
| 2 | deterministic tiebreak (min-rot / max-rot) on ambiguous cases | — | — | — | — | n/a | rejected — true rot is uniform random, min-rot matches truth only 42% |
| 3 | bounded-iteration unrolling | — | — | — | — | n/a | N/A — blocker is not iteration depth; the rotation is simply not encoded in the input |

## Best achieved
No improvement attempted/adopted. Deployed net measured at fresh **496/500 = 99.2%** — already at the
information ceiling. Beats prior? N (no better net is possible).

## Irreducible-floor analysis
NOT a memory floor — an **information floor**. The generator chooses `rotates[i]` (and, for 2 sprites, the
clone marker grouping) UNIFORMLY AT RANDOM, and the input only shows the clone's 3 markers + the original's
full shape. Measured irreducible ambiguity: **1-sprite ≈0.7%, 2-sprite ≈2.0%** of fresh instances admit
≥2 valid (rotation, grouping) interpretations that produce DIFFERENT outputs. In ambiguous cases the true
rotation is split across rot∈{1,2,3} (18/9/4) with no deterministic tiebreak recovering it. Hence NO net —
exact or otherwise — can pass fresh 200/200; the achievable ceiling is ~98–99%, exactly where the deployed
net already sits (99.2%/500). The "4/200 failures" are the irreducible ambiguous cases, not a fixable bug.

## OPEN ANGLES
- None that change the verdict. (Confirmed: no marker-colour-order signal, no body-orientation signal, no
  tiebreak. The disambiguating bit was discarded by the generator.)

## INSIGHT (transferable)
⭐ task18 is a TRUE info-bottleneck WALL (same family as 219/255): the deployed 1395-node net already operates
AT the ~99% information ceiling, so the blank ledger note was a true wall, NOT a false-positive. Method lesson
for sprite-rotation-reconstruction tasks: before investing in an ONNX rebuild, run a Python *candidate
enumerator* over fresh instances and check whether the markers/visible-pixels UNIQUELY determine the answer.
When wide==tall or markers are rotation-symmetric, a 90°-rotation set leaves multiple valid placements ⇒
unsolvable. Don't chase fresh 200/200 on tasks whose generator picks an unobservable random rigid transform.
