# task078 — 3906de3d

## 2026-06-29 compact-column-fill screen

Current source score: 17.835280 @ mem 1260 params 33.

Rule: blue top column heights and red bottom column heights are rearranged into
a top-blue / middle-red / background column fill in a 10x10 canvas.

The current graph slices only the relevant fixed colour channels, reduces each
column to top/bottom counts, builds three 10x10 bool masks (`background`, `blue`,
`red`), concatenates them as [1,3,10,10], and pads directly to the free output.

`conv_fit.py 78` failed for k=1/3/5.  No rewrite adopted.  The output height per
column is data-dependent; the compact three-channel pad is already cheaper than a
full-canvas label-map path.
