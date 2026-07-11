# Legacy-chain structural scan (mechanized C/I discriminator, 2026-07-12)

## ⛔ LANE VERDICT: legacy-chain vein is EXHAUSTED on the deployed board

The 2026-07-11 C/I lane validated a 26/26 discriminator: a rebuild WINS iff the deployed
incumbent is a LEGACY Slice/Where/Gather/Concat chain; FLOORS iff the incumbent already
realizes the minimal free-output construction (einsum/conv + compact one-hot emission).
This scan MECHANIZES that check over ALL 400 deployed nets
(`tools/legacy_chain_scan.py`): per-node counted-byte attribution to LEGACY
(Slice/Where/Gather/Concat/Equal/Pad/Scatter) vs CONSOLIDATED (Einsum/Conv/QLinear*/MatMul)
buckets, `legacy_frac`, counted-plane count, einsum presence, pulling cost/points from
`state/manifest.json`.

**Result: 130 actionable nets (points<19.5, cost>400, not in any wall/floor/done list);
41 have legacy_frac ≥ 0.6. But the top-40 by legacy-score are ALL already per-task-rebuilt
to their documented floors** (39/41 carry `floor` + `ADOPTED`/`Best achieved` markers in
`state/tasks/taskNNN.md`; the 2 without an md — 362, 269 — are also already compact
einsum/scatter builds or flagged no-go).

**Why the classifier's legacy_frac is high yet there is no win:** on an OPTIMIZED board the
residual counted cost of a compact build IS a legacy op — the irreducible **900 B uint8
label Pad** (feeds the FREE output Equal), the **3600 B fp32 `colf` colour-index entry**
(Σk·input_k, the 1×1-Conv gather source), and the **data-dependent crop Slice**. These are
Slice/Where/Pad/Gather nodes, so they register as "legacy," but they are the FLOOR carrier,
not removable bloat. This is the "worklist is wall-residue" meta-finding, now confirmed
board-wide: **the legacy ops at the top of the deployed board ARE the label-map/colf floor.**

The 26/26 discriminator held because it was applied to the fat-middle C/I tasks that had NOT
yet been rebuilt. Those got harvested (6 wins +3.32: 267/072/217/304/275/123, all
legacy-chain incumbents). **What remains with a legacy chain is the residual floor of
already-won tasks.** True un-optimized Slice→Where→Gather chains on the actionable board: ~0.

### Refined discriminator (for future dumps)
A legacy-chain rebuild wins only if the legacy ops are **redundant/un-consolidated bloat** —
a borrowed min-merge or naive net with many full planes and NO compact one-hot emission —
NOT the irreducible 900 B/3600 B/crop carrier of an already-rebuilt net. Operational signal:
absence of a `floor`/`ADOPTED` marker in the task md AND many counted planes (>30) at a
still-high cost (>3000) AND no einsum + no compact label Pad. **Reopen trigger:** a new
public dump replacing an incumbent with a raw legacy chain (re-run this scan after every
`ng mine-public`).

## Ranked top-15 legacy-chain candidates (annotated, all NO-GO)

score = legacy_frac × (25−points). All 15 deep-read against reference oracle + task md.

| rk | task | cost | pts | lf | rule (oracle one-liner) | mechanism fit | dispatch |
|----|------|------|-----|----|-----------------------|--------------|---------|
| 1 | 351 | 1014 | 18.08 | .95 | 16×16 D2-mirror, 5×5 green cutout → reconstruct from double mirror + crop | rect-recipe einsum (DONE S8, 1632→1015 +0.487) | **NO-GO** floored: 3600 B colf entry is the floor; adopted |
| 2 | 388 | 1829 | 17.49 | .85 | 2×2 tile sparse grid + cyan lines through pixel-columns | 900 B Pad label (non-separable tile) | **NO-GO** floored: colour tile not row⊗col separable → label plane forced (task388.md) |
| 3 | 400 | 1181 | 17.93 | .87 | 24×24 D4 pattern, 5×5 blue cutout → reconstruct by symmetry + crop | rect-recipe (deployed 1181 ≪ md-best 12523; further-optimized) | **NO-GO** floored: 3600 B colf; already ≫ md floor analysis |
| 4 | 170 | 2125 | 17.34 | .79 | colour-box masked by downsampled sprite shape | chained Gather on colf (deployed 2125 ≪ md-best 11782) | **NO-GO** floored: 3×3600 B colf planes irreducible; already optimized past md |
| 5 | 272 | 402 | 19.00 | 1.00 | recolour isolated (no-4-nbr) red pixels to blue | mem-0 group-Conv (proven floor) | **NO-GO** hard floor PROVEN: no full-plane decomposition beats 18.87 (task272.md) |
| 6 | 189 | 892 | 18.21 | .86 | 2×2 legend recolours quadrant green pixels (± flips) | 900 B label-map (arbitrary legend colours → data-dep value plane) | **NO-GO** ADOPTED 16.97→further; 900 B label is floor |
| 7 | 156 | 1562 | 17.65 | .79 | two yellow rects, recolour interiors by size/height (± vflip) | compact 5-ch one-hot + 3×3 AveragePool detector | **NO-GO** floored: 900 B label + 3×3 detector is semantic floor; fp16 lever KILLED |
| 8 | 107 | 3415 | 16.86 | .70 | variable-factor kron upscale + red corner-rays | **KRON** (separable double-Gather) + fp16 diagonal rays | **NO-GO** ADOPTED (S15 +0.032, 07-09): kron already applied; 900 B label + 2×1152 diag = floor |
| 9 | 382 | 5626 | 16.36 | .65 | L-marker down-rays, right-shifted per TWO-row interval, 8-way normalized | ray/walk + dihedral normalize | **NO-GO(low-pri)** min-merge adopted S15b; complex mirror-normalize, no folding ingredient. Speculative only |
| 10 | 39 | 453 | 18.88 | .91 | nonzero bbox → crop to top-left quarter of bbox | two-axis Gather crop | **NO-GO** floored: Tier-S two-axis Gather floor (larger axis stays full); already tiny |
| 11 | 194 | 949 | 18.14 | .80 | four rotations assembled into 2×2 quadrant layout | GridSample→single-Gather (fixed-geometry scatter) | **NO-GO** FLAGGED (ci_triage): reversal-matrix params ~450-900 ≈ cost; 360 B entry floor |
| 12 | 242 | 431 | 18.93 | .90 | mirror-symmetric grid, fill 0-holes from mirror + crop | rect-recipe einsum (DONE) | **NO-GO** floored: same family as 351/400, einsum-optimized to 431 |
| 13 | 325 | 1483 | 17.70 | .75 | count connected components k → k×k diagonal of 8s | Euler-formula scalar count + 6×6 label Pad | **NO-GO** optimized (14.36→15.88 geometry-bound); component-count is scalar-reduced, at floor |
| 14 | 269 | 709 | 18.44 | .83 | tile grid A× where A=#nonzero (output shape = A·h × A·w) | — (data-dependent OUTPUT shape) | **NO-GO** FLAGGED: example-variant output shape (Resize runtime-size) — skip class |
| 15 | 288 | 958 | 18.14 | .80 | two diagonal rays up-L/up-R from topmost least-colour cell | signed-einsum / walk-einsum (rays) | **NO-GO(low-pri)** only top-15 net without a floor analysis; but rays are stamp-floored (cf 323/378 DRY). Speculative single probe at most |

## Realistic gap

**≈ 0 pt from the legacy-chain lens.** All 41 legacy_frac≥0.6 actionable nets are at their
documented floors; the residual legacy bytes are the 900 B label / 3600 B colf / crop-Slice
carriers, which no known op-type removes under onnx 1.21. The two nets without a binding
floor verdict (382 ray/mirror, 288 diagonal rays) are ray-family, which has floored
repeatedly this session (323/378/005 DRY 07-11) — worth at most ONE speculative 288 probe
if all other lanes are dry, expected ≤ +0.05 and likely a stamp floor.

**This is not a stopping verdict on the mechanism** — it is a scan result at time T against
the CURRENT deployed board. Reopen the moment a new public dump lands a raw
Slice/Where/Gather incumbent (the scan is one command: `uv run python
tools/legacy_chain_scan.py`), or if a new detection-fold primitive lets the 900 B label /
3600 B colf carrier itself collapse into a free-output contraction (the standing
`conv_fp32_arsenal` open problem in levers.yaml).

## Method / reproduction
- classifier: `tools/legacy_chain_scan.py` (`--all` for full 400, `--json OUT` to dump).
- cost model matches grader (`neurogolf.scans.minmerge.static_cost` / `scoring.calculate_memory`).
- exclusions applied: ci_triage walls+near-floor+DRY+wins, profile_compile judged, levers
  free-output-einsum-regime-crack done+floor. 15 candidates deep-read against
  `reference/arc-code-golf-solutions/taskNNN.py` + `state/tasks/taskNNN.md`.
