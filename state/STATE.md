# STATE - NeuroGolf live handoff (updated 2026-07-12; rank-machine build session + dual-session merge)
> Replace this file at session end; do not append. History lives in git and state/submissions.md.

## Confirmed State
- Confirmed BEST LB: **7316.30** (sub **54596069**, MAIN v9, pure-max, NEW RECORD). Local 7316.18, offset +0.12, no zeros.
- Session arc: MAIN v5 7305.75 → v9 **7316.30 (+10.55 LB)** across v6/v7/v8/v9. All wins fresh-gated (n≥1500–2500 A/B div0) or bit-identical.
- **⚠️ HEDGE IS STALE.** HEDGE v4 (sub 54580652 = 7298.51) is built on the OLD MAIN v4 base and is missing ALL v6–v9
  wins (~+10.7 pt of adoptions). It still carries the 9 silent-zero protections but is now ~18 pt behind MAIN.
  **A best-of-two with stale HEDGE is currently pointless** (HEDGE can only win if MAIN takes a catastrophic
  multi-zero hit larger than 18 pt — implausible under k≈1–2). **NEXT-SESSION TOP PRIORITY: rebuild HEDGE on the
  MAIN v9 base + re-apply the 9 protections (manifest below), then final selection = MAIN v9 + HEDGE-v5-on-v9-base.**
- Deadline 2026-07-15 (private decides). USER ACTION at deadline: set the two final-selection slots on the website.

## 🏗️ THIS SESSION'S BIG LEVER — rank-machine / free-output sign-decode einsum family (9 wins)
Generalized yesterday's task010 guard-row win (1002→379) across the board via the rule-wiki×cost-mismatch lens.
Template artifacts (READ THESE FIRST next session — they are the reusable recipe):
`candidates/task010/rankeinsum.py` (guard-row + sign-decode (cnt−d)(d+2−cnt)), `candidates/task254/rankeinsum.py`
(pair-slot summing, Where(nz,h,ReduceMax) constant-free min-over-nonzero, shared selector halves params),
`candidates/task239/rankeinsum.py` (TopK-float rank space + coupling-tensor einsum), `candidates/task374/rankeinsum.py`
(**thermometer-predicate affinity** — color affine in monotone-threshold predicates, no s-polynomials; **t-axis
term folding** — sum structurally-different emission terms in ONE einsum), `candidates/task050/rankeinsum.py`
(**bg=input-ch0-minus-fill** — use input's own bg channel as base so multi-read einsum factorizes; mem=0),
`candidates/task348/rankeinsum.py` (single-sign t(c)−r>0, no A·B product; 10-wide geometry via identity-embed M[30,10]).
Adoptions: 254 683→351, 239 945→697, 374 2217→1113, 348 1288→842 (codex-refined my 1288), 050 3849→1896,
018 24358→16687 (lean enumeration), 202 2817→2457 (einsum-fusion golf), 156 1562→1539 (safe-golf), 363 kernel-collapse.

## 🤝 DUAL-SESSION DYNAMIC (a Codex session runs in parallel — user OK'd duplicate submits)
Codex works its own git worktree `/Users/minseong/.codex/worktrees/56ef/neurogolf` (branch codex/neurogolf-highscore),
lane = structural factorization (`stacked_relation_ragged_factor_bank`, `sentinel_gated_panel_fold`, D4-branch).
Absorb routine (ran ~5× this session): `uv run python` diff their manifest vs ours → for each cheaper task,
gate + **fresh A/B 2500 draws** (scratchpad/fresh_ab.py pattern) → adopt only if fail=0 & div0. This caught+reverted
codex task005 (fresh-dirty 7/400) — their "fresh audited" is NOT a substitute for our gate. Absorbed this session:
19-net min-merge wave (+5.70 −005), 074 D4 (+1.24), 348 refinement (+0.43). Codex reciprocally absorbs MY wins
(they pull from shared main tree), so both boards converge upward. **Always re-check their manifest before each submit.**

## Lane closures this session (ledgered 4-field; do NOT redispatch without reopen triggers)
- **301** NO-BUILD: bg=9-level staircase (bool rank 9) forces 1B/elem threshold regime; deployed already optimal. Best alt 1065 (+0.069, below bar, constants ready).
- **212** NO-BUILD: mechanism solved 4003/4003 exact but fp32-source-read floor ~2000-2300 ≥ deployed 2249.
- **340** NO-BUILD: projection mechanism proven 2000/2000 fresh but needs ~12 fp32 planes vs deployed's double-duty index-plane (3404).
- **328** NO-BUILD + **stale-md fix**: deployed is NOT the "1717-entry bank" the md claimed — it's a compact 69-node semantic net; rule needs ~14 non-separable 2D planes (argmin/tie/parity rings), unfoldable.
- **090** NOT-ADOPTABLE + **premise fix**: exact rebuild costs 41715 (fp32 band-batch wall) vs deployed 3053; recon's "0.85% fresh-dirty" was WRONG — measured 0.023% (7/30000), negligible under k≈1-2. Dormant uint8-scalar-solver lever if idle budget.
- **199/070/177** fp16-recast DEAD: all regime-crack nets, target tensors co-bind fp32 free input in terminal Einsum (single-T reject). Scanner needs a co-bind filter.

## Durable knowledge added this session
- **rank-machine template family** = the session's engine (see artifacts above). Applies to: rank/order by scalar stat,
  recolor-by-global-scalar, histogram/count renders, separable interval/band fills, Manhattan/half-plane region fills.
- **Where it FAILS** (all NO-BUILD this session): genuine per-cell 2D non-separable planes (argmin/tie/parity, erosion,
  distance fields) can't fold — they need materialized fp32/u8 planes; and fp32-input-co-bind blocks any mixed-dtype
  Einsum fold (ORT single-T). The wall is always "how many irreducible 2D planes does the rule need," not expressiveness.
- **Grader gotcha:** `calculate_memory` counts stale `graph.value_info` even for deleted nodes — always purge value_info
  after node surgery or phantom bytes persist (task202 build).
- opset16 + ScatterElements(reduction='add') + u8 Div/Mod grader-proven (sub 54581845, still holds).

## HEDGE v4 protection manifest (9 nets — re-apply to v9 base next session)
219 exact_v1 (43%→0.30%) | 319 exact_matcher (6.9%→0.18%) | 002 rect_walk (4.8%→2.83%) | 118 varm_peel (6.8%→3.33%)
| 205 hcov_vcov_2d (1.58%→0.01%) | 233 cand_clamped_C (3.0%→0.73%) | 188 fixB (0.67%→0.057%) | 191 exact_8orient
(1.0%→0) | 048 src_rebuild (1.42%→0). Artifacts in candidates/. Rebuild: swap over overfit_nets → ng score each
isolated → pack → submit → restore from scratchpad backup → verify --hash. NOTE new MAIN-side risk carrier: **018 at
16687 is fresh 0.567%** (< prior incumbent 2.0%, so risk DROPPED, but it is the one non-clean net among the new wins —
consider a HEDGE protection port if 018 gets budget; k≈1-2 makes 0.567% ≈ 0.85% private-zero, low).

## Next Session Start
1. `uv run ng status`; confirm 54596069 = 7316.30 unchanged (changed ⇒ recompute). Re-check codex manifest for
   absorbable wins (fresh-gate 2500 before adopt). Kernel poll `uv run python tools/poll_public_dumps.py`.
2. **HEDGE rebuild on v9 base** (top priority — see stale warning above) if planning a best-of-two at deadline.
3. **Wave3 rank-machine recon**: remaining shortlist from this session's wave2 audit = **297** (header→stacked bars,
   compact-label vs 756B one-hot, contested), **377** (concentric rings, high-variance detection risk). Then a fresh
   rule-wiki×cost audit excluding all resolved tasks. Dispatch opus builders (Fable quota exhausted this session).
4. Post-deadline: validate k≈1-2 vs private zeros; writeups (six hidden 25s, 018-4850, fat-middle idiom).

## Operational Guardrails
- Fresh-gate every adoption (n≥1500 A/B; scratchpad/fresh_ab.py does file-based A/B vs backup); bit-identical exempt.
  ng gate → ng adopt only; price exceptions HEDGE-only. Source regen + src↔live reconcile after each adopt (some
  candidate .py builders write cand.onnx under different names — check `ls candidates/taskNNN/` before adopt).
- onnx==1.21.0 + onnxruntime==1.26.0 pins. TopK float/fp16/int64. submission.zip. 100/day. NO sparse initializers EVER
  (grader-ERROR). Clamp all dynamic Gather/Scatter indices. Isolated eval for knife-edge nets (220/230/294/233).
- Check `kaggle competitions submissions` before every ng submit (parallel-session guard — codex submits too).
- Model: Fable quota exhausted 2026-07-12; all builders ran opus after. Subagent recipe → opus; novel mechanism → Fable when available.
