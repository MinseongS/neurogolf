# Build-agent standing instructions (read this fully before building any task)

You build a compact ONNX net for ONE NeuroGolf task. Repo /Users/minseong/project/neurogolf,
`.venv/bin/python` from repo root. You are given: a task number N and its current points P.
GOAL: produce `src/custom/taskNNN.py` (3-digit, `build(task)` convention) that BEATS P by **≥+0.3**
stored AND generalizes (ISOLATED fresh 200/200). If you can only reach <P+0.3, report MARGINAL.
AMBITION: push to the minimal tier the rule admits — do NOT settle at the first working encoding.

## Step 1 — rule
`.venv/bin/python -m src.show N --gen`; read the generator fully; write the exact rule down.
EARLY FEASIBILITY CHECK: if it's a non-local detection task (rays/counts/flood/connectivity/shape-
correspondence) where the public net is already near its floor and you can't beat +0.3, report
INFEASIBLE/MARGINAL FAST. BUT first test whether an apparent "detection" is really one of the collapsible
forms below — most "detection" tasks this sweep turned out to be closed-form and beat the floor.
CONFIRMED FLOOR (do NOT burn 40+ min here): "segment into components + GLOBAL-ArgMax over them +
variable-size crop" floors near ~13.4 for everyone — the public CumSum-scan net is already exact at floor.
(T,L)=(top-of-col-run,left-of-row-run) prefix/suffix-MAX scans give a unique per-box label and per-box
red-counts reduce to a 2-D integral image (4 Gathers, no flood-fill), but the constant factor still lands
at-floor; if the rule needs a global argmax across data-dependent-count components, BAIL fast (task216).
BUT the OPPOSITE case is closed-form & tier-A, NOT a BAIL: "emit the FIXED-SIZE (e.g. 3×3) box with the
most X pixels" — a 3×3 all-ones sum-Conv gives per-top-left counts AND box-validity (occ-conv==9) in one
pass; the UNIQUE-argmax position recovers as a scalar (minrow,mincol)=ReduceMax(iswin·rowramp)/
ReduceMax(iswin·colramp) with NO NonZero/ArgMax op; data-dependent crop = Add scalar offset to a [0..k]
index const + chained Gather(axis2)·Gather(axis3), staying small (9×9/7×7) under the 30×30 label floor
(task271). The discriminator: fixed small box + unique winner ⇒ feasible; variable-size components + global
argmax ⇒ wall. MIDDLE case (SOLID axis-aligned rects, variable size, DISTINCT counts): also feasible
without flood-fill — per-component reductions become contiguous-run all-reduces (segmented doubling) and
the unique-max winner's bbox falls out of one Equal-to-max; BUT a 2-D segmented SUM costs 4 one-directional
sweeps (~13.6KB fp16) which caps the score ~15.1 (marginal) — reaching B≈16.8 needs a CumSum integral image
with a per-cell rectangle read, blocked by needing a data-dependent GatherND (task365, marginal +0.27).

## Step 2 — read these
- reports/SWEEP_SYSTEM.md (tier ladder S>A>B>detection; minimal-tier mindset; MARGINAL threshold ≥+0.3)
- reports/FLOOR_BREAK_GUIDE.md (scoring model + toolkit)
- 3-4 worked task logs in reports/tasklog/ (e.g. task290 count-parametric, task188 un-duplicate-crop,
  task195 Kronecker, task061 closed-form-arithmetic, task250 double-MatMul-scatter, task035 slide-copy,
  task119 ray=closed-form-diagonals, task341 orientation-equivariance) + 1-2 src/custom/task*.py for idiom.

## PROVEN LEVERS (reuse — most wins came from spotting one of these)
- ⭐ ROW/COL SUM-PROFILE AS ONE NO-PAD CONV: a full-row/col sum over a channel-subset is ONE no-pad Conv —
  `Conv(input[1,10,30,30], W[1,10,1,30]) -> [1,1,30,1]` (ch0 kernel weight = 0 to drop background) folds
  "drop a channel + collapse a spatial axis" into one op, dodging the 1200B [1,10,30,1] intermediate that
  ReduceSum(input,[3]) forces; cost is only a 300-elem kernel per axis as params (task141).
- ⭐ 1×1 CONV WITH RUNTIME ONE-HOT WEIGHT = pick channel k + extract its 30×30 plane in ONE op: ORT accepts a
  non-initializer (runtime-computed) Conv weight, so a [1,10,1,1] selector one-hot built from a scalar k
  collapses "select channel + materialise its plane" into a single 3600B Conv — strictly cheaper than
  per-channel ReduceMax-occupancy + gate-Mul (task049).
- ⭐ COUNT-RANK = pure rank function, NO sort/argmax-loop: "sort/label the K rare colours by count" → cnt=
  ReduceSum(input,[2,3]) [1,K,1,1]; rank_k=ReduceSum(Greater(cnt,cnt')) over a tiny [K,K]; route into FREE
  bool output via Equal(rank, target-ramp)∧mask. ⚠️ PITFALL: MUST mask channel-0 (canvas background) to 0
  FIRST — ch0 has the largest pixel count and silently steals rank-0, shifting every label by one (task391).
- ⭐ SCALARS FROM PIXEL-COUNT VECTOR (cutout-robust): when a plane is parameterised by a few scalars and the
  input is partially corrupted (cutout/specks), derive the scalars from cnt=ReduceSum(input,[2,3]) ([1,K,1,1],
  40B) instead of a single-cell readout — a dominant colour's count survives ~125 cut cells so ArgMax(cnt)
  stays exact; fold any "+1"/offset into the channel constant (chan[k]=k−1) to drop an Add plane (task175).
- ⭐ WHERE-CHAIN PRIORITY: in a stacked `Where(maskN, colorN, ...)` priority chain you can DROP all ¬-masking
  of lower-priority masks — the higher-priority Where overwrites any over-marked cells, so compute each mask
  WITHOUT subtracting the masks above it (frees the AND/NOT ops, ~1.5kB) (task125).
- ⭐ RUNTIME-FACTOR MAGNIFY (data-dependent Kronecker upscale): to upscale a recovered K×K pattern S by a
  RUNTIME factor m, use two small gathers `out[i,j]=S[gidx[i],gidx[j]]` (Gather axis0 then axis1,
  gidx[i]=clip(floor((i-1)/m),0,k-1)) — avoids both the int64 flat-index plane and the fp32 index-arithmetic
  plane (generalises const-index Kronecker task195); recover m as a scalar from a perimeter/ring count (task159).
- ⭐ FIXED-SHAPE STAMP AT MARKER MIDPOINT (orientation-free, closed-form, NOT a detection wall): centre =
  channel centroid (Σcoord·profile/count) and a local shape (e.g. PLUS = radius-1 L1 ball `(|r−mr|+|c−mc|)<1.5`)
  builds as ONE fp16 outer-sum plane routed into the FREE Where output; existing dots survive since they sit at
  L1≥3 (task371).
- ⭐ PER-DIRECTION ENCLOSURE = NOT a flood-fill wall on 2-color input: an enclosed hole = bg cell with shape
  pixels in all 4 directions `aL∧aR∧aU∧aD`, each a strict-triangular prefix/suffix-OR MatMul applied
  PER-DIRECTION so it never merges separated boxes (task125).
- GEOMETRY BOUNDS BEAT DTYPE TRICKS: read generator coord bounds, shrink working canvas to the true active
  region, crop OUT unused color channels, slice colour channel to exact extent (fp16 can ADD bytes vs a
  tight fp32 slice). If the generator grid is a fixed full size, the canvas is all in-grid → delete the bg
  slice + in-grid Or chain (the 30×30 Pad sentinel alone zeroes off-canvas).
- REFUSE TO MATERIALIZE PLANES: rule params (color/pos/flag/size) are usually SCALARS via channel/axis
  reductions (ReduceMax(input,[2,3])→[1,10,1,1]; per-channel pixel COUNTS ReduceSum axes=[2,3]=40B; 1-D
  ReduceMin/Max occupancy) — they never need a per-cell plane. ⭐ `ReduceMax(input, axes=[1,3])`→[1,1,30,1]
  (and [1,2]→[1,1,1,30]) recovers grid HEIGHT/WIDTH as a 120B vector with NO 30×30 occupancy plane —
  retrofit this into ANY task that recovers H/W. ⭐ For "argmin/argmax over K candidates with a unique winner
  + per-candidate label", PACK count+label into ONE additive accumulation `base·cnt + label` (e.g. each
  minimiser emits 100, +colour if even) read back by MAGNITUDE BANDS — collapses the whole ismin/count/
  select/colour plane army into a single Sum (task328). ⭐ fp16/bool/uint8 do NOT shrink a FULL-GRID
  plane in the ORT trace (ORT upcasts via PrecisionFreeCast to fp32) — the lever is FEWER full planes, not
  narrower ones; dtype tricks only help SMALL working planes. "which scattered colour is the magnified
  sprite" = presence-DENSITY argmax cnt/(nrows·ncols), exact & cheap, beats bbox-area/span heuristics (task134). Recover SHAPE from counts, skipping the
  3600B colour-Conv. Collapse 2-D detection to 1-D ROW/COL PROFILES (tip/centroid/spans/in-grid rect =
  rowany⊗colany), zero 2-D mask planes. fp16/bool the small working planes. PER-CHANNEL BATCHED MATVEC:
  `MatMul(input, vec[1,10,30,1])` contracts an axis of the FREE fp32 input directly (operand order picks
  the contracted axis, no transpose copy) → eliminates the [1,10,30,30] materialization floor entirely;
  TWO independent per-row scalars come from ONE MatMul(input[1,10,30,30], W[1,10,30,2]) by packing each as
  a contraction-weight COLUMN (col0=k → weighted-count colour, col1=col-ramp gated k≥1 → start position),
  then ReduceSum over channels + Slice — kills the colf plane and a duplicate contraction (task232);
  params count ELEMENTS (cheap) so a small matrix often beats arange+compare arithmetic (task025).
- COLLAPSIBLE "DETECTION" forms (check each): copy/slide of a row/col/edge (task035); closed-form
  arithmetic of a recovered scalar — replace lookup tables, fp16 Mod is integer-exact <2048 and 4× cheaper
  than int32, fold offsets into the final Equal constant (task061); count-parametric shape rebuild
  (task290); periodic arm/profile reconstruction is flip/offset/colour-AGNOSTIC — recover ONE period from
  the n shown consecutive cells and extend with a 1-D modular Gather(arm, first+(i−first) mod n), replacing
  a 30×30 equality/MatMul matrix; use an indicator-Conv [0,1,…,1] for counts + reduced-one-hot slices for
  colours to never materialize a colour-index plane (task358); horizontal PERIODIC tile to K× width = the
  ENTIRE output is ONE free Gather(input, srcIdx, axis=3) (pad cols zeroed via an empty pad col; period from
  colsig[c]==colsig[c+p]) (task231); D4-symmetric OCCLUDED fill = 8 dihedral pullbacks {I,T}×{none,flipR,
  flipC,flipRC} (0-param Transpose + step −1 Slices), reconstruct by elementwise MAX with occluded=−1
  sentinel, gathering the K×K crop from each orbit + maxing tiny blocks not 8 full planes (task400);
  fractal self-tiling = Kronecker kron(S,S) via macro=(u//3)*3+v//3 & micro=(u%3)*3+v%3 index
  maps — NOT naive outer product (task195); un-duplicate CROP = Where(rowmask∧colmask,input,0), dup axis
  from generator range constraints (task188); data-dependent row/col-independent coordinate remap =
  boolean double MatMul Rmat@src@CmatT, fp16 {0,1} exact (task250); stamp a recovered K-row sub-pattern
  into a REGULAR CELL TILING = M=Srow@P@ScolT, Srow[R,dr]=(R%stride==dr+1), auto-zero on gaps/lines/off-grid
  (no in-grid mask); "odd-one-out by pixel COUNT then K× upscale" is closed-form tier-B — unique cell =
  `count<threshold` (NO ReduceMin/ArgMax when exactly one differs), select = `Sum_{R,C}(count<thr)·block`,
  upscale via the task195 const-index Gather (task011); "segment into a tall×wide grid of patches separated
  by single all-bg lines + label each block" is a fully SEPARABLE rows×cols partition (NOT flood-fill):
  per-axis block index = EXCLUSIVE CumSum of the all-bg-line indicator (gated to in-grid extent), tiny
  [1,1,K,30] one-hot selectors, downsample by double MatMul Snum=Rsel@colf@Csel / Sden=Rsel@occ@Csel
  (colour=Snum/Sden exact); the data-dependent tall×wide OUTPUT shape falls out free — R≥tall/C≥wide get
  Sden=0 → sentinel → all-zero, so fixed K×K→Pad(99)→Equal needs no NonZero (task184); "distractor box +
  REGULARLY-SPACED stripes → compacted n×n" is closed-form: regular spacing makes compaction a strided
  Gather(linecolor, clip(offset+stride·i)) — kills the whole [30,30] prefix/eqpos matrix chain (~43KB→~1KB);
  orientation = K_r<K_c (every col crosses every horizontal stripe so nonzero-cols≈W vs nonzero-rows=n),
  n=min(K_r,K_c) (task213). integer Equal(int32) builds a colour one-hot in ONE op vs a 5-op Sub/Abs/Clip/Greater float chain; ADD one *dynamic* colour at masked positions via output=Where(cond[1,1,30,30],
  color_onehot[1,10,1,1],input) (broadcast lands in Where's FREE output, recover colour by slicing a
  guaranteed-hit position) — never build a [1,10,H,W] delta (task033); recover ORIENTATION (xpose) with
  ZERO per-cell planes via total peak-match mass (peak_col=Σ_c max_ch colcount vs peak_row=Σ_r max_ch rowcount,
  correct axis maximizes it), then `Where(scalar[1,1,1,1], A[1,1,1,W], B[1,1,H,1])` broadcasts to [1,1,H,W]
  in ONE op — selecting orientation AND broadcasting the chosen per-line vector at once, no two candidate
  planes (task359); an orientation-dependent (xpose) 1-D reflection folds the per-axis gating into the tiny
  [W] remap VECTOR (rvec=Where(active, refl, identity_i) → inactive axis becomes identity, ONE Equal builds
  the matrix), killing the separate EYE init and the full-matrix Where selects; recover the reflection axes
  from FULL-edge detection (per-row count ≥ box width), which detects orientation for free (task390);
  ray/bounce = union of 45° diagonals
  r+c==a OR r-c==b through a vertex (task119); FILL-ENCLOSED-REGION is NOT a connectivity wall — interior =
  PARITY of horizontal-wall crossings above each cell (lower-triangular MatMul + Mod-2), a per-box fill
  attribute like side-length parity = (topwall_row+botwall_row) mod 2 via index-weighted MaxPools (task204); FILL between two same-colour endpoints along a 45° line is
  NOT a connectivity bail — it is a direction-separable per-channel diagonal prefix-OR ∧ suffix-OR; when the
  generator BOUNDS segment length it collapses to a single bounded K×K diagonal Conv + >0 (cheaper than
  doubling-shift chains or a [100,100] reachability matmul); reshape the 9 colour channels onto the BATCH
  axis so one small kernel serves all, and reuse each diagonal kernel for both opposite directions by
  swapping the asymmetric SAME-pad side (no flips) (task037); complete a Cn ROTATIONAL symmetry (input=k-of-n copies) →
  missing copy = rot(colour) AND NOT colour; recover the data-dependent symmetry CENTRE offset-free by
  MINIMISING the newly-exposed set across candidate centres (max-self-overlap / rot180-size picks are NOT
  exact and silently fail genverify); rot90 needs no matrix = transpose(reverse_rows) (negative-step Slice +
  Transpose, 0 params) (task027); 4-fold REFLECTION symmetrization about a data-dependent
  axis = the double-MatMul idiom with a reflection matrix Mat[out,in]=Equal(2*b+1−in_arange,out_arange),
  the four flips OR'd via one variadic Sum(input, R@input, input@C, R@input@C)>0 (task112); apply_gravity/reflect/transpose = orientation-EQUIVARIANCE
  (same transform on input AND output) → compute both axis branches, select by structure (task341);
  "find the connected object among noise" is NOT a connectivity BAIL when the object is one clustered
  colour and noise uses other colours → object = the MINIMUM-BBOX-SPAN colour (per-channel 1-D occupancy
  → argmin of max(rowspan,colspan)), exact, no flood-fill; recover (min_row,min_col,H,W) as scalars then
  Gather-shift a small WORK×WORK window to origin (task036).
- BANDED SINGLE-CONV: pack several boolean predicates into ONE conv plane via disjoint MAGNITUDE BANDS
  recovered by thresholds (e.g. `100·center_bg + 500·center_red + 1·(#red 4-nbrs)` → in-grid=band100,
  static-red=500, olive-red=501+ all from one plane) — kills separate in-grid/mask convs. Mono-colour
  fixed-stamp centre detection needs ZERO [1,10,H,W] planes: on the colf plane a banded `10·#X-cells +
  1·#edge-cells == 50` conv proves "X-full AND edges-empty" in one pass, a `corners=+1,centre=−4 == 0` conv
  enforces mono-colour without a ×5 plane, and a Gather of per-channel counts by colf gives "count-of-my-
  colour" per cell. Caveat: `cnt==N` is NOT a shape discriminator even if the body is always N px — identify
  by SHAPE not count (task117). Center tags need
  weights large enough that neighbour-count leakage can't cross a lower threshold (task278). A single
  colour Conv can also FOLD IN a sentinel marker channel (e.g. gray weight=50 ⇒ value>9 means marker) so
  colour label and marker position both come from ONE plane; and an origin-anchored rectangular in-grid
  mask is FREE from 1-D occupancy profiles (ReduceSum of free input → 120B vecs → Greater→And) vs a
  5760B [1,10,W,W] channel-max (task206).
- OCCUPANCY/bbox over the 10-ch input must EXCLUDE channel 0: every background cell sets ch0=1 so a
  ReduceMax over all channels marks the whole grid occupied (silent bbox=full-grid bug) — use
  colf=Σ_k k·input_k (>0 ⇔ non-background) as the occupancy signal, which doubles as the value plane.
  A horizontal mirror of a cropped window = a flipped col-index ramp `min_col+(W-1)-arange(WORK)` into the
  col Gather — no reflection matrix needed (task177, task036 crop+flip).
- ring/box CENTRE detection = ONE Conv whose kernel is the ring's exact perimeter pattern (response peaks
  at the perimeter pixel-count, strictly lower elsewhere → a single Greater isolates centres, no flood-fill);
  a no-pad Conv aligns the peak to the window TOP-LEFT so add pads=[k,k,k,k] (SAME) to land it on the
  geometric centre. "8 neighbours ALL EQUAL" = `8·S2 == S1²` (Cauchy-Schwarz equality) via two 8-ring convs
  on V and V², fp16-exact when the gap ≫ fp16 step; the `S1>0` gate is LOAD-BEARING — an isolated noise
  pixel has 8 empty neighbours trivially "all equal (==0)" and gets falsely picked without it (task346).
  Independent full row+col crosshairs are SEPARABLE → is_row OR is_col, broadcast in the
  free final ops. NEGATIVE result: a 1-D ReduceSum row/col profile does NOT replace the 2-D outline Conv
  for centre detection — per-row counts are equal at edges and inner rows, and two boxes whose edges align
  can fake a phantom peak at a non-centre row; the 2-D Conv is required to bind the outline at one location.
  The real saving: run that 2-D Conv on a cheap 1-CHANNEL slice cropped to the active grid (slice the one
  relevant colour to 15×15 FIRST), not the 10-ch 30×30 input → resp 3600B→900B, kernel params 287→58 (task094).
- bounding-box-as-a-MASK (no scalar argmin/max) = boolean prefix-OR ∧ suffix-OR of the per-row/col "has"
  vector via a lower- and upper-triangular boolean MatMul + Greater (only TWO triangulars needed since
  tril.T==triu) — turns an apparent "find the rectangle component" task into closed-form tier A, beating the
  public CumSum-scan floor (task070).
- variable offset → bbox first-occupied row/col; 2-D point lookup → chained Gather(axis=2 then 3), NOT
  row∧col outer product (cross-talks); K cheap channel-Slices beat a [0..9] colour Conv for fixed small
  color sets. When each spatial REGION carries a FIXED colour, skip the 1×1 colour-index Conv entirely (it
  forces a 3600B fp32 30×30 plane) — slice each region's OWN colour channel from the FREE input for a tiny
  presence mask, then map presence→constant-colour fill via a Where PRIORITY chain (task180 16.48→17.74,
  task321 16.40→17.88). DISCRIMINATOR: use Conv-collapse to a colour-index plane ONLY when output colours
  COPY arbitrary input colours; for a FIXED known colour set always slice+Where instead. (Re-triage badly
  UNDERESTIMATES fold/overlay tasks — 180 & 321 were est gain ~0.9/0.6 but delivered +1.66/+1.48.)
- Tier S blocked if output colors are RANDOM per-instance (a fixed Conv can't route); Tier A blocked if the
  stamp/shape isn't a row⊗col separable rectangle (a 45° diagonal couples r&c → not separable).
- "remove isolated noise, keep ≥2×2 SOLID shapes" = part-of-a-filled-2×2 predicate via TWO 2×2 sum-convs
  (conv1 pad bottom/right counts each block, ==4 ⇒ full; conv2 pad top/left dilates full-blocks over their
  4 covering cells) — exact, no flood-fill. (task193)
- "spread one seed across a contiguous RUN" = iterated MaxPool(1×k) → re-gate by the run mask after each
  pool, radius=min inter-box gap, iters=max seed-to-edge distance — exact, no Scan (ORT rejects uint8
  MaxPool/int8 Max so fp16 2B is the dtype floor) (task354). DISCRIMINATOR: this lever only beats the floor
  when the active canvas is FIXED-SMALL (size-10 → planes 9× smaller); a bounded flood across a VARIABLE-size
  region on the full 30×30 canvas is a WALL — the ~16-plane fp16 fill floor (~28.8KB) pins it near ~14.2
  regardless of other optimisation, and a data-dependent crop trips the symbolic-dim trap (task198 infeasible).
Solid-rect-interior (1px-outline → erode / no-coloured-within-1, FN=0) is closed-form ONLY when shapes sit on
a noise-free or DIFFERENTLY-coloured background; when dense noise shares the shape's colour and abuts the
outline, noise+outline merge into chance-eroded blobs 8-CONNECTED to the true region — reconstruction leaks
into them, a true connectivity wall (task255 infeasible).
- [task193 cont.] Fold off-grid into the keep cond (selcond=keep OR offgrid) so
  the removed branch is just a constant [1,10,1,1] bg one-hot in the FREE Where output (task193).
- "recolour every gray stamp from the one coloured stamp" (identical solid rects at random non-overlapping
  positions) is NOT a shape-correspondence BAIL: a cell's OFFSET within its own sprite is a LOCAL run-length
  (product-chain of shifted occupancy, resets at gaps) so key=dr·4+dc is per-cell with no flood-fill; the
  colour-by-offset 4×4 histogram is learned from the single coloured sprite via a double-MatMul over offset
  one-hots, propagated by a 1-D Gather (task368).
- per-colour bbox-FILL of disjoint instance-coloured boxes = L=MatMul(A[r,c]=c·rowband_c, B[c,c']=colband_c)
  contracts the 10-ch axis into ONE [1,1,H,W] colour-index plane, no [1,10,H,W] product (disjoint ⇒ no
  double-stamp, weight-0 bg falls out); recover the in-grid bg channel as rowany⊗colany for ~120B by
  reducing the existing fp32 occupancy over the channel axis (task132). Two fp32 per-channel spatial
  reductions are a ~15.8 ceiling — don't chase tier-A past it.

## SCORING + OP FACTS
score=max(1,25−ln(mem+params)); mem=SUM over every intermediate tensor of elems×itemsize (input/output
tensors are FREE); itemsize bool/uint8=1 fp16=2 fp32/int32=4. NEVER materialize a [1,10,30,30] intermediate
(9000+B) — route the 10-ch expansion into the FREE `output` as the final op: `Equal(L_uint8, arange[1,10,1,1])`
→ BOOL (declare output value_info BOOL) or `Where(mask, onehot, input)`. For a rect output, ASSOCIATE the
three broadcasts `And(rowin[1,1,30,1], And(colin[1,1,1,30], bgbool[1,10,1,1]))` so NO [1,1,30,30] box/label
plane is ever materialised (saves ~1800B vs Where→uint8-L→Equal). Watch the bg assumption: bg = the CORNER
cell input[0][0], NOT the most-frequent colour — a line/fg colour can out-number bg (task021). opset 11 ops OK (scorer checks DOMAIN
not VERSION). BANNED: Loop/Scan/NonZero/Unique/Compress/Function. Gather/Mod allowed. fp32 exact for ints <2^24.
fp16 `Equal(diff,0)` is EXACT for integer operands and collapses a Sub+Abs+threshold chain to ONE bool op
(but fp16 Min/Max crashes ORT under ORT_DISABLE_ALL — do clipping in fp32). A +1 bias on a colour-index
plane lets a SINGLE value-carrying Where plane double as both colour-readout and occupancy mask, removing
the separate {0,1} mask plane (task205).
GOTCHAS: ORT ReduceMax/Sum reject uint8/bool (need float); ORT Mul/And/Mod reject uint8 (combines stay bool);
ORT Where/Equal implemented for uint8 but NOT int8/int16; ORT Pad rejects bool; Clip rejects int64 (clip in
float then Cast); Slice preserves the input float dtype; opset-11 has no GreaterOrEqual (use Not(Less(...)));
REMOVE unused initializers (they still count as params); calculate_params counts element COUNT not bytes
(initializer dtype is free — only element count matters). Reshape-to-scalar MUST use a `[1]`-shaped
initializer, NEVER an empty `[]` 0-dim init (a 0-dim shape makes calculate_params return None →
"performance could not be measured" scorer trap, task036). A runtime-tensor (data-dependent) Slice also
leaves SYMBOLIC dims → calculate_memory returns None (same "could not be measured" trap) — use Gather with
squeezed scalar indices instead for data-dependent border/line extraction (task161). Detection lever:
"border colour present at BOTH line-ends" = per-side ReduceMax-presence ANDed pairwise, gated by
interior=(total−ring_count>0), using only tiny [1,10] tensors — no [1,10,30] matched-pair products (task161).
To locate a UNIQUE marker, ArgMax of per-row/col ReduceMax beats Greater→Cast→Mul-ramp→ReduceSum (kills ~6
intermediates + ~60 ramp params); a 1×1 Conv with an OUTSIZED weight on the marker channel (w_marker=1000,
w_k=k) does triple duty from ONE plane — locate (ArgMax), read the ±1 sprite window (Gather), recover the
sprite colour (ReduceMax of window with marker zeroed) (task121).

## ⭐ THE 3600B PLANE FLOOR IS REAL — break it by REMOVING the plane, never by narrowing it (FLOOR_RESEARCH.md)
Rigorously ORT-measured: you CANNOT get a per-cell colour-index/value/Gather-index plane below fp32 3600B
via dtype tricks. Declaring fp32-as-uint8 → TypeInferenceError; Cast/Quantize→uint8 ADDS a plane (3600→4500);
Cast→fp16 (5400); ArgMax→int64 (7200); Gather indices reject uint8 (int32 30×30 = 3600 irreducible); a single
fp32 channel-Slice already costs 3600; full-input fp16 cast costs 18000. ORT Add/Mul/Mod reject uint8 and
ReduceSum/Max reject bool/uint8, so `Σ k·input_k` MUST be fp32. ⇒ NEVER chase dtype tricks to accumulate a
0-9 index. Instead test the THREE structure-escapes IN ORDER before accepting 3600B: (1) spatial-COPY of input
cells → Tier-S mem 0; (2) SEPARABLE row⊗col routed into the FREE bool output → ~840-3000B (also dodges
Pad-rejects-bool by never materializing a carrier); (3) small ACTIVE canvas (generator size bound) → slice the
plane below 30×30, then fp16 is cheap there. A task stuck at ~16.2-16.4 is most likely silently failing escape
(2) or (3) — re-triage it on separability + bounded-active-region before recording a verdict.
⭐ IMPORTANT CLARIFICATION (task377, contradicts a naive reading of the above): the "fp16 doesn't help" result
is ONLY about the ENTRY colour-index plane (the 10→1 reduction must output fp32). But once you HAVE that one
fp32 plane, CAST IT to fp16 (single-channel, 1800B) and run ALL downstream full-canvas ops (diffs/ring/gather-
value/Min/Max) in fp16 — the static scorer counts fp16 full planes at HALF, and fp16 Min/Max DOES work under
ORT_DISABLE_ALL. This routinely drops mem ~3-4× (task377: 95k→26k). So: pay the one 3600B fp32 entry, then go
fp16 for everything after. (Do NOT cast the 10-ch INPUT to fp16 — that's 18000B.) Generalized anchor detector
(task165): `10·S2==S1² AND S1>0` (Cauchy-Schwarz on value & value² ring-convs) finds the unique window where
all K shape-cells share one nonzero value AND recovers the colour as S1/K — works for ANY fixed multi-cell
shape, not just 8-neighbours.

## ANTI-STALL (agents have died at the 600s no-progress watchdog — obey)
- WRITE src/custom/taskNNN.py EARLY (a first working draft within a few minutes) and iterate on disk; do
  NOT think for 10+ min before writing.
- BOUND verification: fresh 200/200 (or up to 500/500 once) is ENOUGH. Do NOT run 5000/20000-sample stress
  loops or exhaustive Python brute-force — they stall the watchdog and rarely change the verdict.
- `/tmp/arc-gen/src` SHADOWS the repo `src` package (genverify does sys.path.insert(0,'/tmp/arc-gen')).
  If you hit `No module named src.custom` or a wrong `src`, run from repo root with `PYTHONPATH=.` and do NOT
  insert /tmp/arc-gen at path position 0 in your own verify script; import repo src FIRST, or replicate the
  fresh-instance check inline (load the generator by file path, don't rely on genverify's sys.path).
- For overlay/fold tasks: collapse the 10-way one-hot to a single colour-index plane with a 1×1 Conv
  (sum_k k·input_k) FIRST, then fold on tiny [1,1,H,W] tensors — leaner than slicing 9 colour channels +
  Max + rebuilding ch0 (task372 16.44 beats the task360 slice idiom at 15.98).

## Step 3 — VERIFY (authoritative) then report
evaluate() ok + ISOLATED fresh 200/200 against freshly-generated instances (see src/genverify.py +
src/harness.py; fresh_pass reads networks/taskNNN.onnx from DISK so temp-write WITHOUT touching
reports/manifest.json, or replicate the fresh comparison against your in-memory model). Double-check NO
undefined names before reporting.

## CONSTRAINTS
Write ONLY src/custom/taskNNN.py AND you MAY create/update reports/tasklog/taskNNN.md (from
reports/tasklog/_TEMPLATE.md). Do NOT adopt/pipeline/manifest/networks/commit. No sub-agents. Write the
file early and iterate.

## FINAL OUTPUT (your return text = data, not a human message)
"RESULT taskNNN: pts=X.XX mem=N params=M fresh=K/200 | tier reached: S/A/B/detection | beats P by ≥+0.3?
Y/N/MARGINAL | dominant intermediate: <what>B <why irreducible> | OPEN ANGLES untried: <list> | INSIGHT:
<transferable lesson>"  — or "INFEASIBLE taskNNN: <specific irreducible reason>"
