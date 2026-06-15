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

## Step 2 — read these
- reports/SWEEP_SYSTEM.md (tier ladder S>A>B>detection; minimal-tier mindset; MARGINAL threshold ≥+0.3)
- reports/FLOOR_BREAK_GUIDE.md (scoring model + toolkit)
- 3-4 worked task logs in reports/tasklog/ (e.g. task290 count-parametric, task188 un-duplicate-crop,
  task195 Kronecker, task061 closed-form-arithmetic, task250 double-MatMul-scatter, task035 slide-copy,
  task119 ray=closed-form-diagonals, task341 orientation-equivariance) + 1-2 src/custom/task*.py for idiom.

## PROVEN LEVERS (reuse — most wins came from spotting one of these)
- GEOMETRY BOUNDS BEAT DTYPE TRICKS: read generator coord bounds, shrink working canvas to the true active
  region, crop OUT unused color channels, slice colour channel to exact extent (fp16 can ADD bytes vs a
  tight fp32 slice). If the generator grid is a fixed full size, the canvas is all in-grid → delete the bg
  slice + in-grid Or chain (the 30×30 Pad sentinel alone zeroes off-canvas).
- REFUSE TO MATERIALIZE PLANES: rule params (color/pos/flag/size) are usually SCALARS via channel/axis
  reductions (ReduceMax(input,[2,3])→[1,10,1,1]; per-channel pixel COUNTS ReduceSum axes=[2,3]=40B; 1-D
  ReduceMin/Max occupancy) — they never need a per-cell plane. Recover SHAPE from counts, skipping the
  3600B colour-Conv. Collapse 2-D detection to 1-D ROW/COL PROFILES (tip/centroid/spans/in-grid rect =
  rowany⊗colany), zero 2-D mask planes. fp16/bool the small working planes.
- COLLAPSIBLE "DETECTION" forms (check each): copy/slide of a row/col/edge (task035); closed-form
  arithmetic of a recovered scalar — replace lookup tables, fp16 Mod is integer-exact <2048 and 4× cheaper
  than int32, fold offsets into the final Equal constant (task061); count-parametric shape rebuild
  (task290); fractal self-tiling = Kronecker kron(S,S) via macro=(u//3)*3+v//3 & micro=(u%3)*3+v%3 index
  maps — NOT naive outer product (task195); un-duplicate CROP = Where(rowmask∧colmask,input,0), dup axis
  from generator range constraints (task188); data-dependent row/col-independent coordinate remap =
  boolean double MatMul Rmat@src@CmatT, fp16 {0,1} exact (task250); ray/bounce = union of 45° diagonals
  r+c==a OR r-c==b through a vertex (task119); apply_gravity/reflect/transpose = orientation-EQUIVARIANCE
  (same transform on input AND output) → compute both axis branches, select by structure (task341).
- BANDED SINGLE-CONV: pack several boolean predicates into ONE conv plane via disjoint MAGNITUDE BANDS
  recovered by thresholds (e.g. `100·center_bg + 500·center_red + 1·(#red 4-nbrs)` → in-grid=band100,
  static-red=500, olive-red=501+ all from one plane) — kills separate in-grid/mask convs. Center tags need
  weights large enough that neighbour-count leakage can't cross a lower threshold (task278).
- variable offset → bbox first-occupied row/col; 2-D point lookup → chained Gather(axis=2 then 3), NOT
  row∧col outer product (cross-talks); K cheap channel-Slices beat a [0..9] colour Conv for fixed small
  color sets.
- Tier S blocked if output colors are RANDOM per-instance (a fixed Conv can't route); Tier A blocked if the
  stamp/shape isn't a row⊗col separable rectangle (a 45° diagonal couples r&c → not separable).

## SCORING + OP FACTS
score=max(1,25−ln(mem+params)); mem=SUM over every intermediate tensor of elems×itemsize (input/output
tensors are FREE); itemsize bool/uint8=1 fp16=2 fp32/int32=4. NEVER materialize a [1,10,30,30] intermediate
(9000+B) — route the 10-ch expansion into the FREE `output` as the final op: `Equal(L_uint8, arange[1,10,1,1])`
→ BOOL (declare output value_info BOOL) or `Where(mask, onehot, input)`. opset 11 ops OK (scorer checks DOMAIN
not VERSION). BANNED: Loop/Scan/NonZero/Unique/Compress/Function. Gather/Mod allowed. fp32 exact for ints <2^24.
GOTCHAS: ORT ReduceMax/Sum reject uint8/bool (need float); ORT Mul/And/Mod reject uint8 (combines stay bool);
ORT Where/Equal implemented for uint8 but NOT int8/int16; ORT Pad rejects bool; Clip rejects int64 (clip in
float then Cast); Slice preserves the input float dtype; opset-11 has no GreaterOrEqual (use Not(Less(...)));
REMOVE unused initializers (they still count as params); calculate_params counts element COUNT not bytes
(initializer dtype is free — only element count matters).

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
