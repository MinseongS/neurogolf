# Sparse-edit family worklist (2026-07-12) — pending task018 mechanism verdict

Premise: community intel (thread 724263) proves a GENERAL ~4850 net exists for task018
(ours 24358). If the 018 rebuild lands, its idiom (runtime-computed edit list → clamped
ScatterND(data=free-input) → free output; or centroid/affine-warp render) should be fanned
out to the rest of the old hash-scatter family — all sparse-edit tasks where the incumbent
is a 1000+-node exact solver:

| task | cost | pts | fresh-fail (2026-07-12 isolated) | rule shape |
|------|------|-----|----------------------------------|------------|
| 018 | 24358 | 14.90 | 2.0% (irreducible ~2%) | sprite rotation-reconstruction (fork building) |
| 101 | 13725 | 15.47 | 0.85% | (see tasklog) |
| 076 | 12825 | 15.54 | (timeout in sweeps; gen slow) | rainbow sprites, rot/flip completion |
| 118 | 12282 | 15.58 | (7.67% deployed) | cross-stamp on static (red/cyan by collision) |
| 173 | 11320 | 15.66 | 0.2% | sprite completion, ~1% cells change; delta already ScatterElements on free input |
| 066 | 10121 | 15.77 | 0.25% | (see tasklog) |
| 025 | 9817 | 15.80 | 0.1% | dots move onto adjacent-to-line cells |

Dispatch rule: only after the 018 fork reports a WORKING mechanism (else this is the
worklist-is-wall-residue pattern — these incumbents are optimized exact solvers, and the
26/26 C/I discriminator says rebuilds only win vs LEGACY chains; the 018 datum is the only
evidence the wall is soft). If 018 fails to land, ledger this list as dormant with reopen
trigger = post-deadline writeup of the 4850 mechanism.
