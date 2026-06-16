# FLOOR RESEARCH — can the 3600B per-cell colour/value plane be broken? (2026-06-16)

Deep-research sweep of every angle for emitting a per-cell colour-index / value / Gather-index
plane at SUB-fp32 cost, directly from the FREE fp32 10-channel one-hot input. Every verdict below
is backed by a real ORT run under `ORT_DISABLE_ALL` with the EXACT scorer measurement path
(`src/harness.calculate_memory`: `infer_shapes(strict_mode=True)` + profiling trace,
`mem = max(static_value_info_bytes, trace_bytes)`, dtype = the *shape-inferred* dtype).

Test harness: `/tmp/floor/measure.py` (byte-exact replica of `calculate_memory`, sets
`node.name = node.output[0]` like `sanitize_model` so trace rows match). All snippets in `/tmp/floor/`.

## THE GOVERNING LAW (why nearly everything is dead)
Two independent, separately-measured facts make the fp32 plane irreducible for an
**arbitrary-copied** per-cell colour index:

1. **The scorer counts the SHAPE-INFERRED dtype, and strict inference cannot be lied to.**
   Declaring `L` as uint8 while a `Conv` actually emits fp32 →
   `TypeInferenceError: Inferred elem type differs (1) vs (2)`. So a 1-byte count requires an op
   whose *genuinely inferred* output dtype is uint8/bool. (`t_dtype_lie.py`)

2. **Every narrow-dtype op needs an fp32 plane as its INPUT, and that input is counted.**
   Cast / QuantizeLinear / DynamicQuantizeLinear all take the fp32 [1,1,30,30] (3600B) and *add*
   a uint8/fp16 plane on top — never replace it. Measured 3600+900 = 4500 and 3600+1800 = 5400.
   (`t_cast.py`, `t_quant.py`, `t_fp16.py`)

3. **There is NO integer/bool arithmetic to ACCUMULATE a 0-9 index cheaply.**
   ORT `Add`/`Mul`/`Sub`/`Mod` REJECT uint8 (`unsupported type: tensor(uint8)`), and
   `ReduceSum`/`ReduceMax` reject bool/uint8. The only way to compute `Σ_k k·input_k` (the
   colour index) is fp32 → a 3600B fp32 plane. (`t_tierB.py`)

Corollary: a per-cell plane whose VALUE is an arbitrary copied input colour (0-9) is **hard-floored
at 3600B fp32**. The escape routes below only work when the rule has *extra structure* (separable,
small active region, fixed colour set, or pure spatial copy) — none break the arbitrary case.

---

## ANGLE-BY-ANGLE VERDICTS

### Angle 1 — QuantizeLinear / Dynamic / ConvInteger / MatMulInteger → genuine uint8 intermediate?
**DEAD.** QuantizeLinear DOES emit a real uint8 plane (trace shows `(900,'uint8')`), but its input
is the fp32 colour plane = 3600B, so total = **4500** (worse than the 3600 baseline). Quantizing the
10-ch input directly is [1,10,30,30] uint8 = 9000B. ConvInteger/MatMulInteger need uint8 operands,
which would themselves be a 9000B uint8 input plane. No path feeds these from the free fp32 input
without first paying ≥3600. (`t_quant.py`)

### Angle 2 — Cast(fp32→uint8) right after a cheap op: does ORT keep it 1-byte in the trace?
**Half-true, but DEAD as a saving.** A freshly-produced uint8/fp16 plane IS counted at 1B/2B per
elem in BOTH static and trace (the "PrecisionFreeCast upcasts" lore is about the cast's *source*,
not its *destination*). BUT the fp32 source the Cast reads from is still materialized and counted.
Measured: Cast→uint8 = 3600+900 = **4500**; Cast→fp16 = 3600+1800 = **5400**. The Cast only helps if
the fp32 source never existed — which is exactly what's impossible for an accumulated index.
(`t_cast.py`, `t_fp16.py`, `t_fp16_direct.py`)
- Sub-finding: casting the WHOLE input to fp16 to make a Conv emit a fp16 plane costs **18000**
  (the [1,10,30,30] fp16 cast plane) — catastrophic. fp16-contraction only wins on a SMALL cast
  region (the existing small-canvas lever).

### Angle 3 — Reshape/Transpose the 10 channels onto the BATCH axis
**No dtype change → DEAD.** Moving channels to batch ([10,1,30,30]) does not alter the dtype floor:
the recombined per-cell plane is still fp32 [1,1,30,30] = 3600. `ArgMax` over the channel axis
produces the index in ONE op but as **int64** [1,1,30,30] = **7200B** (no dtype attribute to shrink;
casting to uint8 adds 900 on top of the 7200). Strictly worse. (`t_argmax.py`)

### Angle 4 — write the BOOL output directly, dodging the uint8 label / Pad-rejects-bool carrier
**WORKS — but only re-confirms the known Tier-A/S levers, doesn't break the arbitrary floor.**
- Pure spatial COPY (flip/translate/tile/permute) routes the free fp32 input straight into the free
  output via Gather/Slice/Transpose → **mem 0** (Tier S). No carrier at all. (`t_remap.py`)
- SEPARABLE rule: `And(rowsel[1,10,30,1], colsel[1,10,1,30])` broadcasts into the FREE bool output;
  the only cost is the two fp32 reduce outputs (1200 each) + two bool conds (300 each) = **3000**,
  and drops to ~840 if the rule is colour-agnostic (reduce the channel axis too → [1,1,30,1]=120).
  This is the documented Tier-A pattern; associating the broadcast means **no [1,1,30,30] carrier is
  ever materialized**, so the Pad-rejects-bool problem simply never arises. (`t_separable.py`)
- Building the bool output per-channel via Concat of bool planes still pays for the per-channel
  fp32 slices (3600 each — see Angle "slice cost" below), so it loses to the label-map for
  arbitrary colours. The carrier-dodge helps presentation, not the dominant cost.

### Angle 5 — make the Gather-index plane uint8 (cast to int32 only at the Gather)
**DEAD.** ORT Gather/GatherElements reject uint8/int8 indices at shape-inference
(`indices typestr Tind has unsupported type: tensor(uint8)`). Indices MUST be int32/int64, so a 2-D
per-cell index plane is int32 [1,1,30,30] = **3600B** minimum. Keeping a uint8 plane and Cast→int32
at the Gather just materializes the int32 plane anyway (3600). (`t_gather_idx.py`)
- The ONLY escape for the Gather floor is SEPARABILITY: if row-index depends only on row and col-index
  only on col, the index VECTORS are int32 [30] = 120B each and you do two 1-D Gathers on the input
  (axis 2 then axis 3). The 3600B int32 floor only bites a genuinely 2-D-coupled per-cell lookup
  (which would need GatherND with a data-dependent index plane).

---

## SUPPORTING MEASUREMENTS (irreducibility evidence)
| construct | measured bytes | note |
|---|---|---|
| Conv `Σ k·input_k` → fp32 [1,1,30,30] | **3600** | THE floor (baseline) |
| + Cast→uint8 | 4500 | fp32 source still counted |
| + Cast→fp16 | 5400 | fp32 source still counted |
| + QuantizeLinear→uint8 | 4500 | same |
| ArgMax(channel)→int64 [1,1,30,30] | 7200 | worse; no dtype attr |
| fp32 Slice of ONE 30×30 channel | **3600** | extracting a channel plane is NOT cheap |
| Greater(full input)→bool [1,10,30,30] | 9000 | comparing-then-slicing is worse than slicing-then-comparing |
| Cast whole input→fp16 | 18000 | never cast the full input |
| separable row⊗col → free bool output | 3000 (→~840 colour-agnostic) | Tier A, no carrier |
| pure spatial copy → free output | 0 | Tier S |

ORT op-dtype rejections re-confirmed under ORT_DISABLE_ALL (load these into BUILD_PROMPT's gotcha list):
`Add/Mul/Sub/Mod` reject uint8; `ReduceSum/ReduceMax` reject bool/uint8; `Gather` indices reject
uint8; `Where/Equal` DO support uint8; strict shape-inference forbids declaring a wrong dtype.

---

## VERDICT SUMMARY
- **No angle breaks the arbitrary-copied per-cell colour-index floor.** It is hard-floored at
  **3600B fp32** because (a) the index must be accumulated in fp32 (no integer/bool arithmetic),
  (b) any narrower dtype needs that fp32 plane as input, and (c) the scorer counts the genuinely
  inferred dtype (no lying). Same for the int32 [1,1,30,30] Gather-index plane (3600B, indices
  can't be uint8) and the uint8 [1,1,30,30] label (which still rides on a 3600B fp32 source).
- **The only real savings come from STRUCTURE, not dtype tricks** — exactly what BUILD_PROMPT
  already preaches. The floor is broken by *removing the plane*, never by *narrowing* it.

## SINGLE HIGHEST-IMPACT RECIPE (graduate into BUILD_PROMPT)
**"Never accumulate a 0-9 index; test the THREE structure-escapes in order before accepting 3600B."**
For any task that looks like it needs a [1,1,30,30] colour/value plane, in priority order:

1. **Spatial-copy → Tier S (mem 0):** if every output cell COPIES an input colour from a fixed
   spatial remap (flip/translate/tile/rotate/reflect/permute), route the free fp32 input straight
   into the free output with Gather/Slice/Transpose. No plane.

2. **Separable → Tier A (~840–3000B):** if output[ch,r,c] factors as rowcond[ch,r] ⊗ colcond[ch,c],
   build the two 1-D conds with channel/spatial ReduceMax (keep them [1,*,30,1]/[1,*,1,30]) and
   `And`/`Mul` them straight into the free bool output — associate the broadcast so NO [1,1,30,30]
   carrier exists (this also sidesteps Pad-rejects-bool). Reduce the channel axis too whenever the
   rule is colour-agnostic to drop the fp32 conds from 1200→120.
   For a 2-D Gather lookup, the same test applies: separable ⇒ two int32 [30] index VECTORS (120B
   each), not a [1,1,30,30] index plane.

3. **Small active canvas → shrink the plane:** if the generator bounds the active region to W×W
   (W≤~17), do ALL per-cell work at [1,1,W,W] and `Pad` to 30×30 with a sentinel only just before
   the final Equal. A fp32 plane at 12×12 is 576B, at 17×17 is 1156B — below the 3600 floor and,
   crucially, fp16-castable there for a further halving since the small fp32 source is cheap.

Only when ALL THREE fail (genuinely 2-D-coupled, full-grid, arbitrary copied colours) is 3600B the
honest irreducible floor → accept Tier B (~16.8) and record WHY (this report's governing law).
The remaining ~16.2–16.4 tasks the sweep is stuck on are almost certainly failing escape #2 or #3
silently — re-triage them for separability and bounded active region before recording a verdict.
