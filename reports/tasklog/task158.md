# task158 — 6aa20dc0 (variable-mag dihedral sprite re-stamping)

**Rule:** A canonical 3x3 sprite (c0@(0,0), c1@(2,2) distinct corner colours; transpose-symmetric
c2 body on off-diagonal cells; bg=4th colour). 2-4 "megas" are placed, each with its own
magnification mag∈{1,2,3} and own (hflip,vflip), non-overlapping with margin 2. Mega 0 (the
REFERENCE, always mag=1) is fully drawn; every other mega shows ONLY its two magnified diagonal
CORNER blocks. OUTPUT fills every mega with its full flipped-magnified sprite.
**Current (prior):** 12.85 pts (public net).
**Target tier:** detection/B — non-local stamp+correspondence; the wall was wrongly believed to be
sprite RECOVERY (prior agent capped ~46%). It is NOT — recovery is closed-form and stamping is separable.

## Attempts
| # | angle | tier | mem | params | pts | fresh | outcome |
|---|---|---|---|---|---|---|---|
| 1 | full 30x30 fp32, 4 role-spread ConvTranspose | det | 933215 | 8882 | 11.24 | 266/266 | exact but heavy |
| 2 | fp16 working planes + combined write | det | 368081 | 8893 | 12.16 | 200/200 | — |
| 3 | channel-contracting Convs (Fcount/value as [1,12,9,9] weight) | det | 193489 | 8887 | 12.78 | 200/200 | — |
| 4 | forward-spread = Conv (pad-topleft) not ConvTranspose; no 38x38 | det | 193489 | — | 12.78 | 200/200 | — |
| 5 | crop pipeline to WORK 26x25 active region (generator size cap) | det | 152769 | 8902 | **13.01** | 200/200 | EXACT, adopted |
| 6 | (alt) write-all, drop exact-cover confirmation | det | 102719 | 6958 | 13.39 | 200/200 | 0.04% LATENT LEAK — NOT adopted |

## Best achieved
**13.01 @ mem 152769 params 8902 — EXACT, generalizes (ISOLATED fresh 200/200; ONNX==target 2000/2000;
solve()==target 50000/50000).** Beats prior 12.85 by +0.16 (MARGINAL by the strict +0.3 rule, but a
TRUE fully-generalizing gap-closer). A write-all variant scores 13.39 but fails 0.04% of instances
(phantom over-stamp) so it would zero those on the LB — rejected in favour of exactness.

## Algorithm (the prior "wall" dissolved)
1. bg = MODE colour (per-channel pixel-count ArgMax). c2 = non-bg colour with MIN bbox span
   (it is confined to the reference 3x3). The two remaining present colours are {c0,c1}.
2. refpos = (min row, min col) of {c2 cells} ∪ {non-bg-non-c2 cells within Chebyshev-2 of the
   c2 bbox} — deterministic, NO window search (verified 2000/2000). canon = the 3x3 window at
   refpos, hflipped iff the "special" corners lie on the anti-diagonal (→ c0,c1 on main diagonal).
3. For each (mag,h,v): match where the two corner blocks == c0/c1 AND the rest of the (3·mag)²
   footprint == bg (a Conv on [eq_bg,eq_c0,eq_c1] with a STATIC [12,3,9,9] kernel; off-grid sentinel).
4. EXACT-COVER collapses to ONE SPATIAL pass (no Loop, no isolation ring, no ref self-exclusion):
   Fcount = Σ_configs forward-spread(match by the Sz² block); uniq = (Fcount==1); a placement is
   CONFIRMED iff its visible mask overlaps a uniq cell. Write confirmed tiles. (20000/20000 numpy.)

## Irreducible-floor analysis
Dominant: four fp16 [1,12,26,25] planes (sat, match, vsat, confirmed ≈15.6KB each) + three bool
[1,12,26,25] gates (≈7.8KB each). The 12 channels = 3 mags × 4 flips are ALL distinct (verified
3000/3000 — a transpose-symmetric body still has 4 distinct flips), the active canvas is at its
26x25 floor (generator caps width≤25, height≤26 — verified maxH=26,maxW=25 over 1000 fresh), and the
exact-cover REQUIRES per-config gating (the value-write needs `confirmed[config]` to know which tile
to stamp), so none of the four 12-ch planes can be contracted. Forward-spread Conv (pad-topleft)
instead of ConvTranspose removed all 38x38 planes; channel-contracting [1,12,9,9] weights removed the
post-spread 12-ch planes. That is the architectural floor for the EXACT net.

## OPEN ANGLES
- Drop exact-cover → 13.39 pts but 0.04% phantom-overstamp leak (write-all). Reconsider only if the
  LB tolerates a 0.04% per-instance leak (it does not — a held-out failing instance → 0 on that one).
- A 2-stage match (shared per-mag footprint plane + cheap corner-colour disambiguation) MIGHT cut
  the flip dimension from the dominant `sat`/`match` planes (untried; risky, est ~−15KB).

## INSIGHT (transferable)
⭐ "Variable-mag dihedral sprite re-stamping" is NOT a correspondence/recovery wall. The canonical
sprite recovers closed-form (min-bbox-span colour locates the fully-drawn reference; a single hflip
fixes orientation), and per-object magnify+flip STAMPING is fully separable as 12 fixed strict-
correlation passes. The exact-cover (deduping phantom corner-block alignments) collapses to ONE
spatial pass: footprint-coverage count → uniq cells → confirm placements whose visible mask owns a
uniq cell — NO iterative naked-singles, NO Loop/NonZero. ⭐ FORWARD-SPREAD (stamp a tile at every
marked top-left) = a Conv with the kernel 180-flipped at the BOTTOM-RIGHT + pads=[k-1,k-1,0,0] (NOT
ConvTranspose) → output stays NxN, no (N+k-1)² plane. ⭐ A grouped per-config spread Conv that must be
SUMMED over configs is better written as a single NON-grouped Conv with a [1,Cfg,k,k] weight — it
contracts the config axis in one op and never materialises the [1,Cfg,N,N] plane. ⭐ A data-
INDEPENDENT crop to the generator's max grid (here 26x25) shrinks every working plane with zero risk.
⭐ ORT (current build) keeps fp16 for Conv/ConvTranspose/Equal/ReduceSum/Where/Min/Max under
ORT_DISABLE_ALL — declare every full-grid plane fp16. ⭐ np.ascontiguousarray inflates a 0-dim numpy
scalar to shape [1]; build Gather index scalars WITHOUT it (true 0-dim init; prod([])=1, params-safe)
so the gathered axis is REMOVED (a [1]-index keeps a stray dim → out-of-bounds on the next Gather).

---

## 2026-06-19 RE-PROBE (current deployed = ext:kojimar7113, 14.53 pts) — MARGINAL, did not beat
The crowd net SUPERSEDED our 13.01 EXACT build: kojimar mem=33059 params=2343 (vs our 152769/8902)
by running the 12-pass stamp pipeline on SMALL all-uint8 per-config blocks (`pair_u81/82/83`
[1,4,24,23]/[1,4,21,20]/[1,4,18,17] — one per mag, 4 flips in the channel axis — not our four
[1,12,26,25] fp16 planes). It PASSES fresh 200/200 (re-verified). So kojimar already applied the
uint8 + plane-reduction levers past our recorded "architectural floor."

To beat +0.3 from 14.525 needs total ≤ 26226 ⇒ cut 9176B. Measured cost (static value_info):
- `color_f` f32 [1,1,30,30]=3600B — the colour-index ENTRY, `Conv(input,w_color)`, consumed ONLY
  by `Cast->color_u8`. **fp32-LOCKED**: Conv reads the fixed fp32 graph input, ORT forces weight
  dtype==input dtype, output inherits fp32; fp16 would need an 18000B input cast. IRREDUCIBLE here.
  (The "Conv keeps fp16" lever needs the Conv input to already be a narrow WORKING plane, not entry.)
- `pair_u8*` (2208+1680+1224) + `ab_u8` 1300 + ~90 small 650B/552B uint8/bool planes = the 12
  (mag×dihedral) stamp passes; already uint8, the documented sole buildable backbone.
ONLY safe lever = dedup ~3 bool/u8 near-duplicate pairs (nonbg_bool+nonbg_u8, mask_a_*, mask_b_*)
≈1950B ⇒ pts ~14.58, gain ~+0.057 << +0.3.
⇒ MARGINAL. Algorithm is SOLVED + generalizing; this is a MEM floor, not an accuracy wall, and the
fp32 entry plane plus the irreducible 12-pass uint8 stamp planes pin it. No beating net written.
