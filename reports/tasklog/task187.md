# task187 — 7b6016b9 bounded flood-fill crop

## Rule

Preserve the non-black line drawing.  Black cells connected to the outside become
green(3).  Black cells enclosed by boxes become red(2).  Off-grid cells are all
false.

Generator inspection shows height/width are always 20..25.

## Result

| attempt | mechanism | stored pts | memory | params | fresh | outcome |
|---|---|---:|---:|---:|---|---|
| incumbent | 30x30 label-map flood-fill | 14.340 | 41700 | 926 | failed adopt fresh gate | replaced |
| crop2 | reduce to 1-channel label/masks, crop to 25x25, run flood-fill, pad label back to 30x30 | 14.580 | 32850 | 665 | passed adopt fresh gate | adopted |

Local stored gain is +0.240 pts.  The larger practical gain is that the incumbent
was non-generalizing under `src.adopt`, while the crop2 source passed the fresh
generator gate.

## Transferable insight

For connectivity/flood-fill tasks with a generator-bounded canvas smaller than
30x30, do not crop the 10-channel fp32 input directly.  First reduce to
one-channel label/mask planes, then crop those planes and run the iterative scan
there.  Pad the final one-channel label map back to 30x30 with a sentinel label
before final Equal.
