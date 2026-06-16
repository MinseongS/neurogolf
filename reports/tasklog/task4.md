# task4 — per-shape un-slant shear (was pending-retry/stall)

**Rule:** 0–4 variable-count shapes at data-dependent positions get a per-row "un-slant"
shear. Looks segmentation-bound but is fully LOCAL when shapes are row-separable: classify
rows by 1-cell vertical occupancy neighbours, characterise the edge-case pixel by its direct
vertical neighbour (NO flood-fill / rightmost-scan), then collapse the colour-copy remap into
ONE value plane L=shiftR1(colf·shiftmask)+colf·copymask (collision-free → Add==Or).

**Target tier:** A reached. pts 14.08 → **14.87** (+0.78). mem 25152, params 38.
fresh 200/200 + 500/500.

**Feasibility:** the prior pending-retry/stall — feasibility check passed FAST, NOT a wall.
Row-separability makes the whole transform a local per-row shift → closed-form tier-A.

**Dominant intermediate:** two 3600B fp32 entry planes (colf30 colour-index Conv +
ingrid30 channel-ReduceMax, both irreducible) + 3600B int32 Equal plane (opset-10 Equal
rejects fp16/uint8).

**OPEN ANGLES:** fold in-grid detection into colour Conv to delete ingrid30 (~+0.15);
eliminate int32 cast by routing uint8 one-hot Pad into FREE output (worse at W=17, recheck smaller W).

**INSIGHT:** a per-shape shear that looks segmentation-bound is LOCAL when shapes are
row-separable — classify by vertical occupancy neighbours, never flood/scan. Collision-free
colour remaps let Add substitute for Or. Crop 30→17 active region for the byte win.
