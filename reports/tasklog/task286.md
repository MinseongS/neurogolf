# task286 — b782dc8a

## Current live

`memory=46272`, `params=741`, `points=14.241820561808675`.
`src/custom/task286.py` was updated to a live-exact source builder on 2026-06-28.

## Semantic rule

Input is a cyan wall maze (`8`) with black corridors and an adjacent seed pair of two
non-cyan colours. Output flood-fills the 4-connected black component containing the seeds.
The fill colour alternates by checkerboard parity:

`out[r,c] = seed_color_for_parity[(r+c) % 2]`

Cyan remains cyan; black corridors outside the seed component remain black.

## Architecture assessment

The deployed graph is already the right low-level family: bit-parallel flood fill.
Rows are packed into 25-bit `uint32` bitsets, propagated with `BitShift`, `BitwiseAnd`,
and `BitwiseOr`, then expanded back to a 25×25 mask.

Major bottlenecks:

- full one-hot input cast: about 9000B;
- cropped 10-channel task area: about 6250B;
- bitset pack/seed/final expansion planes: about 2500B each;
- many scalar uint32 bit-propagation nodes.

## Low-level opportunities

- Source-owning the deployed `QLinearConv` colour-index extraction is done; this reproduced live
  `46272` memory and fixes the older source/live mismatch.
- Beating live materially is unlikely without a new connectivity representation. The remaining
  plausible tweak is projecting/gathering required channels before cropping the 10-channel plane,
  but expected gain is only ~1–2KB and may be offset by added maps.

Do not re-try MaxPool flood, pointer-jump connected components, or byte-chunked bitsets without a
specific new reason; they are expected to be worse under static-shape ONNX memory accounting.
