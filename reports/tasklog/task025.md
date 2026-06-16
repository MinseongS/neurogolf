# task025 — 1a07d186

**Rule:** Full vertical LINES sit at columns `linecols` with DISTINCT colours (a line = a
full column, colcount == grid height H). Scattered single-cell DOTS lie off the lines. A dot
of colour k that matches a line of colour k is MOVED onto the cell immediately adjacent to
that line, same row, on the side facing the dot: `out[r][lc-1]=k` if dot col < lc else
`out[r][lc+1]=k`. The original dot is erased; a dot whose colour matches NO line (the
generator's "extra" colour) is erased. `xpose=randint(0,1)` transposes BOTH grids 50/50, so
lines may instead be full ROWS — the rule is transpose-equivariant.
**Current (stored before):** 13.74 pts, gen:thbdh6332, mem 77960, params 63 (generalizes 60/60).
**Target tier:** detection (transpose-equivariant scatter). Tier B (single label plane) is
BLOCKED: xpose is 50/50 and ONNX bans data-dependent control flow (If/Loop), so BOTH
orientation branches must be materialised and selected — an irreducible ~2× doubling.

## Attempts
| # | angle | tier | mem | params | pts | fresh | outcome |
|---|---|---|---|---|---|---|---|
| 0 | prior on-disk custom (full [1,10,30,30] label-map, 2 branches) | det | 774463 | 2763 | 11.44 | 266/266 | rejected (too big) |
| 1 | per-channel batched MatMul(input,leftvec) kills the [1,10,30,30] product; label via k-contraction MatMul | det | 199663 | 3033 | 12.78 | 266/266 | correct |
| 2 | fp16 working tensors + read FREE fp32 `input` directly in BOTH branches (no transpose copy) | det | 84097 | 6033 | 13.59 | 266/266 | correct |
| 3 | share in-grid mask across branches; combine left/right/line into ONE contraction via k-axis Concat | det | 69225 | 6034 | 13.77 | 266/266 | correct |
| 4 | drop the fp16 has-cast (Greater compares the fp32 MatMul output directly) | det | 66825 | 6034 | **13.80** | **200/200** | **best** |
| 5 | replace Utri/Sr/Sl matmul matrices with arange+relpos compares (params 6034→697) | det | 77705 | 697 | 13.73 | 266/266 | WORSE — the extra rel/abs/gate intermediates cost more bytes than the params saved (params count ELEMENTS, cheaper than fp16 bytes) |

## Best achieved
13.804 @ mem 66825 params 6034 — adopted? **N (orchestrator gates).** Beats prior 13.74?
+0.064 → **MARGINAL** (< +0.3 threshold). GENERALIZES 200/200 fresh (both orientations).

## Irreducible-floor analysis
Two full orientation branches (vertical + horizontal) are mandatory (50/50 xpose, no control
flow), each carrying ~30 small 1-D tensors ([1,10,30,1]/[1,10,1,30]) plus four [1,30,30]
label-contraction tensors. The bulk: ~33 fp16 [1,10,*] (~20 KB), 8 fp16 [1,30,30] (~14 KB),
the four fp32 input-MatMul operand+output vecs (~12 KB, fp32 is forced because they MatMul the
free fp32 `input`), one fp32 [1,1,30,30] in-grid ReduceMax (3.6 KB). NO [1,10,30,30] is ever
materialised. The doubling is the structural floor; a single-plane Tier-B encoding is
unreachable while xpose is data-dependent.

## OPEN ANGLES (re-attack backlog)
- Merge left+right (and up+down) input-MatMuls into ONE MatMul per branch via a [1,10,30,2]
  vec, then Slice — same total bytes in my tally but may reduce ORT's measured peak.
- Detect orientation FIRST, then `Where`-select between `input` and a single transposed copy
  to feed ONE branch. Costs a [1,10,30,30] select (~18 KB fp16) — about break-even with the
  second branch (~30 KB); worth a measured try (could net ~10 KB).
- Build the four position/mask tensors from a SINGLE shared signed-relpos field per branch
  (cut Sub/compare count) WITHOUT the extra Abs gates that sank attempt 5.

## INSIGHT (transferable)
⭐ **Per-channel batched matvec via MatMul kills the [1,10,30,30] floor.** To compute
`has[k,r] = OR_c input[k,r,c]·mask[k,c]` without materialising the [1,10,30,30] product,
feed the FREE fp32 `input` straight into `MatMul(input, vec[1,10,30,1])` (contracts the col
axis) or `MatMul(vec[1,10,1,30], input)` (contracts the row axis) — operand order picks the
contracted axis so input is never transposed/copied. Same trick contracts the 10 colour
channels in the label sum (`L[r,c']=sum_k a[k,r]·b[k,c']`).
⭐ **Params (element COUNT) are cheaper than fp16 working bytes.** The scorer adds
`mem_bytes + param_element_count`; a 900-element fp16 NxN const = 900 in score, but the
arange-and-compare reformulation that removed it added several fp16 [1,10,30] (600 B each)
intermediates and net LOST points. Prefer fixed matmul matrices over runtime index arithmetic
when the latter spawns extra intermediates.
