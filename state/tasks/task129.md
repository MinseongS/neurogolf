---
deployed_cost: 80
logged_costs_match: match
migrated: 2026-07-09
---

# task129 — 5582e5ca

**Rule:** Input is a 3x3 grid whose 9 cells are filled with 6 sampled colours via a fixed multiplicity schedule: `colors[0]`→3 cells, `colors[1]`→2 cells, `colors[2..5]`→1 cell each (`colors[5]` may equal `colors[4]`, so [4]/[5] reach ≤2 cells). `colors[0]` is sampled distinct from `colours[1..4]`, so it is the UNIQUE colour appearing exactly 3 times. Output = solid 3x3 grid filled with `colors[0]` (the mode); off-grid is all-zero in every channel.
**Current:** 18.86 pts, ReduceSum→ArgMax→OneHot→Expand→Pad, mem ~?, params 18
**Target tier:** COUNT→FIXED-PATTERN (the whole output is a scalar mode colour → solid block) — cheapest tier.

## Attempts
| # | angle | tier | mem | params | pts | fresh | outcome |
|---|---|---|---|---|---|---|---|
| 1 | ReduceSum→Greater(>2.5)→Where([1,1,3,3])→Pad | count→pattern | 140 | 28 | 19.876 | 200/200 | adopted |

## Best achieved
19.876 @ mem 140 params 28 — adopted Y. Beats prior 18.86? Y (+1.02).

## Irreducible-floor analysis
Dominant intermediate is the `[1,10,3,3]` uint8 Where block (90B) plus the `[1,10,1,1]` fp32 counts (40B). Both are tiny; the only full-canvas tensor is the FREE output (Pad target). No 30x30 intermediate is ever materialised. Off-grid stays all-zero in `convert_to_numpy` (it does NOT set channel-0=1 off-grid), so per-channel ReduceSum over the full spatial axes gives exact 3x3-region counts with no contamination — ArgMax/ReduceMax is unnecessary because exactly-3 is generator-guaranteed, so a single `Greater(counts,2.5)` isolates the mode.

## OPEN ANGLES (re-attack backlog)
- Could drop `thr` const by comparing counts to a recovered scalar, but `Greater` already at 5 ops / 140B — diminishing returns; near tier floor for a count→solid-fill task.

## INSIGHT (transferable)
⭐ "solid-fill with the most-frequent colour of a small fixed grid" = COUNT→FIXED-PATTERN: per-channel ReduceSum counts → ONE threshold (`Greater(counts, k-0.5)`) when the winning multiplicity is generator-fixed (no ArgMax/ReduceMax/OneHot), then a `Where(modehot[1,10,1,1], one[1,1,K,K], zero[1,1,K,K])` broadcasts the channel selector across the KxK active block in ONE op and Pad routes it into the FREE output. Beats the public ArgMax+OneHot+Expand chain. Key enabler: off-grid cells are all-zero one-hots (not channel-0), so spatial ReduceSum is clean.

## 2026-07-03 S12 — UNKNOWN-bucket dossier

**Rule:** 3×3 grid filled with 6 sampled colours on a fixed multiplicity schedule; the mode colour appears exactly 3× → output = solid 3×3 block of the mode colour (off-grid all-zero).

**Cost (grader mem 80, params 0):** graph is GlobalAveragePool → Hardmax → Einsum, ZERO initializers. Counted intermediates: `counts` [1,10,1,1] fp32 40B, `winner` [1,10,1,1] fp32 40B. The [1,10,30,30] output is the FREE Pad target. Total counted ≈ 80B = two channel-vector planes.

**Blocker class:** already-at-floor. Two [1,10,1,1] reductions (80B) is the irreducible per-channel count+select for a mode→solid-fill; no 30×30 or 3×3 working plane is materialised, no params. NB the pre-existing rich log documents a stale mem=140 attempt — the LANDED net (this onnx) is already the cheaper GAP→Hardmax→Einsum at 80B, so the census UNKNOWN + 4.382 unlock is a mem*0.6 fallback artifact (physically 0).

**Lever:** no lever visible. Could recast `counts`/`winner` fp32→fp16 (mode-count ≤9, exact) for ~40B, but Hardmax/Einsum operands at [1,10,1,1] are already trivial — diminishing.
