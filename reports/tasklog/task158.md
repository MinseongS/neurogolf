# task158 — 6aa20dc0

**Rule:** One reference sprite (a transpose-symmetric 3×3 pattern: corner cells c0@(0,0) and c1@(2,2)
distinct colours, body colour c2 at 2–3 sampled cells of {(1,0),(1,1),(2,0),(2,1)} mirrored across the
diagonal) is placed at 2–4 "mega" positions, each with its OWN magnification mag∈{1,2,3} and an independent
dihedral hflip/vflip. The INPUT shows the FIRST mega (idx 0, the reference) FULLY drawn, but every other
mega only shows its two diagonal CORNER blocks (the magnified c0 and c1 cells). The OUTPUT fills in every
mega completely (full flipped-magnified sprite at each position). bg = input[0][0].

**Current:** 12.85 pts, gen:biohack_new, mem 189350, params 366

**Target tier:** detection/multi-object reconstruction — would only beat floor if cleanly closed-form.

## Determinism verdict (the key question)
- The binding is **POSITIONAL/deterministic, NOT a random bijection.** Each output mega is reconstructed
  independently from its own two visible corner blocks. Mag is recoverable per-object from corner-block
  SIZE; flip is recoverable from which corner block is colour c0 (c0≠c1 ⇒ c0-corner position uniquely
  determines (h,v)). So the prompt's "variable-mag + flip is separable" hypothesis is CORRECT here in
  principle — this is not the classic random-dictionary wall.

## What WORKS (verified, unrestricted numpy)
1. **Colour recovery, exact:** c2 = the non-bg colour with MINIMUM bounding-box span (0/400 fails) — it
   appears only in the reference body. mag(reference) = minimum solid run-length of the c2 mask (0/1000).
2. **Stamp-by-correlation is EXACT given the true sprite: 2000/2000.** For each (mag∈{1,2,3}, h, v) build
   the flipped-magnified tile T (3·mag²) and a "visible" mask = base cells (0,0),(2,2) flipped; stamp T
   wherever the window equals T on the visible cells AND is strictly bg elsewhere in the 3·mag×3·mag
   footprint. The reference (full body) fails the "rest is bg" test ⇒ correctly left intact. No pairing,
   no NonZero, no component labelling needed. This part IS ONNX-expressible (12 fixed correlation passes).

## Where it FAILS (the genuine wall)
**Reference-sprite recovery is ambiguous.** Multiple candidate (origin,mag) windows downsample to a
valid transpose-symmetric sprite that also stamps consistently with the input. Selecting the candidate by
"minimal input-cell contradiction after re-stamp" picks the correct flip-equivalence class only ~88% of
the time; ties (bad=0 for both a correct and a wrong small-mag sprite) are irreducible without the OUTPUT.
Full numpy solver: ~46% (918/2000) — and that is the *unrestricted* ceiling. Translating the candidate
enumeration + re-stamp-and-compare selection into Loop/NonZero/argmax-free ONNX is not cleanly expressible
(it is itself a data-dependent search), and ~46% can never reach the required fresh 200/200.

## Attempts
| # | angle | tier | outcome |
|---|---|---|---|
| 1 | 8-conn component peel + per-object bbox | — | FAILS: non-ref corner blocks are disconnected, cannot pair |
| 2 | stamp-by-correlation w/ ORACLE sprite | A | 2000/2000 EXACT — reconstruction is closed-form |
| 3 | + sprite recovery (min-span c2, runlen mag, window enum, restamp tie-break) | — | ~46% — recovery selection ambiguous, not fresh-passing & not ONNX-clean |

## Best achieved
No generalizing net. Not adopted. Does NOT beat 12.85.

## Irreducible reason
The reconstruction (stamp) is exact and separable, but **recovering the canonical reference sprite requires
a data-dependent search over candidate (origin,mag) windows with a tie-break that needs the output** — a
search not expressible in the Loop/NonZero/argmax-free op set, and even unrestricted it caps ~46%. The wall
is the SPRITE-RECOVERY/disambiguation step, not the magnify (which IS separable) and not the binding (which
IS positional).

## OPEN ANGLES (re-attack backlog)
- A cleaner canonical-sprite recovery that is unique by construction (e.g. derive the base PATTERN directly
  from c2-cell base-coordinates via phase = c2cells mod mag, placing c0/c1 by transpose-symmetry, WITHOUT
  enumerating windows). If a unique closed-form sprite recovery exists, the rest (stamp 12×) is exact and
  ONNX-buildable at ~30–60KB → ~14.5–15.3 pts (+1.7–2.5). This is the single blocker; worth one more agent.
- 12 strict-correlation stamp passes with a RUNTIME-recovered sprite tile is the buildable backbone if
  recovery is solved.

## ⭐ RECOVERY SOLVED + STAMP CORRECTED (2026-06-19 follow-up agent) — algorithm EXACT, ONNX not yet built
The prior agent's "single blocker = recovery" is RESOLVED unique-by-construction, AND a correction was found
to the stamp ("2000/2000 exact" was a sampling artifact). Full pipeline verified EXACT in numpy:
**solver-prop 50000/50000, fixed-unroll(4-round) 30000/30000, naive-prop 20000/20000.**
Reference solver lives in `src/custom/task158.py::solve()` (repo-import verified 5000/5000 fresh).

THE RECOVERY (unique by construction — prior 46% "ceiling" was wrong):
1. The REFERENCE is ALWAYS mag=1. Generator: `mag = 1 if not mags else randint(1,3)`; `mags` is empty on the
   first mega ⇒ idx-0 reference is mag=1 every time. (verified)
2. Body colour c2 appears ONLY inside the reference 3×3. (verified 3000/3000) Non-ref megas DRAW ONLY their
   two corner blocks (c0/c1) — `if idx>0 and (row!=col or row==1): continue` keeps only canonical (0,0)&(2,2).
   So c2 cells DETERMINISTICALLY locate the reference. NO window search, NO output needed.
3. c2 = non-bg colour with MIN bbox span (confined to 3×3; c0/c1 spread across megas).
4. Canonical sprite = unflip the ref 3×3 by its own (h,v): the valid unflip has c0@(0,0),c1@(2,2)
   (distinct, ≠bg, ≠c2) and a transpose-symmetric c2 body. Unique. (180°-symmetric bodies → c0/c1 label
   ambiguous, but IRRELEVANT: the 180-rotated canon is reached by the same 4 stamp passes.)

THE STAMP CORRECTION (bare 12-pass is NOT exact, even with the ORACLE sprite):
- Measured: bare 12-strict-correlation stamp with ORACLE canon = **19984/20000 (0.08% FAIL)**, NOT 2000/2000.
  Prior agent's 2000-sample test was lucky (~1.6 expected fails). PHANTOM BRIDGES: a c0 corner-block of mega A
  and a c1 corner-block of mega B can align as the two diagonal corners of a (false) tile with bg between.
  Phantom rate after isolation alone = 142/300000 (0.047%), by phantom-mag {mag1:28, mag2:99, mag3:15}.
  Sequential-overwrite stamp = 0.046% (no help); largest-mag-first consume = 0.0067%; exact-cover prop = 0.
- bg via grid[0,0] is wrong 0.85% (a mega can sit at (0,0)) ⇒ use MODE colour.
- FIX = isolation ring (reject footprints whose 1-cell border isn't all-bg) + EXACT-COVER: the true non-ref
  megas are the unique set of isolated placements whose visible corner-block cells partition (all non-bg minus
  the reference region). Resolve by NAKED-SINGLES propagation: pick every candidate that uniquely owns some
  target cell, remove its cells, kill overlappers; repeat. Terminates in ≤3 rounds (≤3 non-ref megas).

ONNX FEASIBILITY (proven): the ≤4-round propagation is BOUNDED ⇒ unrollable to a fixed tensor pipeline
(stacked bool masks + sum-reductions + elementwise; no Loop/NonZero/argmax). Whole task is opset-10 buildable.
Build plan in `src/custom/task158.py` docstring. NOT yet implemented (large graph: recovery + 12 stamps +
4-round per-cell exact-cover). `build()` is a NotImplemented stub. Next agent: build the ONNX from solve().

⚠️ CORRECTION to prior INSIGHT: "per-object variable-mag stamping is EXACT via 12 strict-correlation passes"
is FALSE for tasks where corner-blocks of distinct objects can alias into a phantom tile. The stamp needs an
EXACT-COVER over candidate placements (bounded naked-singles propagation, ≤K rounds for ≤K objects), not just
correlation. The "rest is bg" + visible-cell match is necessary but NOT sufficient.

## INSIGHT (transferable)
⭐ "Per-object variable magnify + dihedral flip" reconstruction is **separable and exact via 12 fixed
(mag×flip) strict-correlation stamp passes** — stamp T where window matches T on the VISIBLE cells AND is
strictly bg elsewhere; the fully-drawn reference auto-excludes itself via the "rest is bg" test. The wall
in such tasks is usually the SPRITE/TEMPLATE RECOVERY (locating + disambiguating the reference), not the
stamping. Recover the body colour as the min-bbox-span non-bg colour and the magnification as the min solid
run-length of that colour's mask — both exact and ONNX-cheap.
