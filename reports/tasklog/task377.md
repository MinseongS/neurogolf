# task377 (ARC eb5a1d5d) — nested rectangles -> concentric square rings

## Result
- **14.83 -> 15.28 (+0.45)**, mem 25924 -> 15639, params 116 -> 1004, fresh 500/500.
- Tier: B (closed-form scalar/sequence recovery + tiny-canvas figure + Pad-to-free-output).

## Rule
Input = N strictly-nested axis-aligned rectangles (rect0 fills grid; each later rect
strictly inside, corner moving SE), colours[0..N-1] (adjacent rings differ, non-adjacent
may repeat). Output = (2N-1)x(2N-1) concentric square rings at top-left; ring idx
(Chebyshev dist from border) = colours[idx]; rest of canvas all-zero. WHOLE output is
determined by scalar N + length-N colour SEQUENCE.

## What worked (escape = tiny-canvas figure + scalar/sequence recovery)
1. **Build the figure on a fixed 15x15 (W=2K-1, K=8) canvas, not 30x30.** ring(i,j)=
   min(i,j,m-i,m-j), m=2N-2; L=cv[ring] is [1,1,15,15]; off-figure = uint8-safe sentinel
   250. Cast L->uint8, **Pad 15x15->30x30 (opset-11 Pad accepts uint8), then Equal(Lpad,
   arange0..9) emits the 10-ch one-hot BOOL into the FREE graph output.** The 30x30
   output-side tensor is a single 900B uint8 plane (not the old 3600 int32 + 1800 L +
   1800 ring). Channel-0 and 250 sentinel match no channel -> all-off = target.
2. **Colour SEQUENCE off ONE deepest row (killed the entire vertical 30x30 scan, ~4350B).**
   N = max rowdepth alone (every rect has row- AND col-extent >=3 so the deepest row hits
   all N rects; verified 4000/4000). rowselN = 1-hot first-row-of-depth-N; deeprow =
   rowselN @ colf [1,1,1,30] reads the nested palindrome c0..c_{N-1}..c0. 1-D transition
   conv -> upper-tri MatMul prefix-sum -> per-column segment index -> cv[d] = deeprow at
   first column of segment d. All <= [1,1,8,30]. (Deep-row cv verified 5000/5000.)

## Floor / dominant intermediates (why ~15.6KB remains)
- colf32 [1,1,30,30] fp32 3600B = the 10->1 reduction off the FREE fp32 input (Conv keeps
  input dtype) — irreducible entry plane.
- colf fp16 1800B (cast so the one scan plane is fp16).
- horizontal depth scan hdiff(1740) + h_eq(870 bool) + h_eqf(1740 fp16 cast for ReduceSum,
  which rejects bool/uint8) ~= 4350B — the ONE un-removable 30-wide scan; needed to find
  per-row depth so we can locate the deepest row + N.
- idx_i 900B int32 (Gather index floor), Lpad 900B uint8.

## Levers / lessons (transferable)
- ORT now accepts: **uint8 Pad** (opset-11) and **fp16/uint8 Equal** (stale "int32-only"
  claim is false) — so a tiny one-hot can be uint8-Padded into the free output with NO
  int32 30x30 carrier.
- ReduceSum still rejects bool/uint8 (need the fp16 cast); CumSum rejects fp16 under
  ORT_DISABLE_ALL -> use an upper-triangular MatMul for a 1-D prefix-sum (900 params,
  log-cheap; the only param bump).
- **Read a parametric figure's defining SEQUENCE off a single selected line** (here the
  deepest row) instead of two full-grid scans: 1-hot line-selector @ plane -> [1,1,1,W],
  then a tiny 1-D segment/prefix recovery. Halved the scan cost.
- Sentinel choice 250 (uint8-safe) instead of -1 lets the padded value plane stay uint8.
