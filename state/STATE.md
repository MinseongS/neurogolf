# STATE - NeuroGolf live handoff (updated 2026-07-12 early; C/I harvest day + insurance-portfolio rebuild)
> Replace this file at session end; do not append. History lives in git and state/submissions.md.

## Confirmed State
- Confirmed BEST LB: **7304.42** (sub **54576478**, MAIN v3, pure-max). Local 7304.3002, offset +0.12.
- HEDGE v3 confirmed **7297.67** (sub **54576509**, MAIN v3 + 8 protections — handicap −6.75
  exactly as priced; all 8 protection nets executed cleanly on the grader).
- **FINAL SELECTION (before 07-15): MAIN v3 (54576478) + HEDGE v3 (54576509).** Supersedes the
  v2 pair (54574503/54574665). Deployed tree = MAIN v3 (ng verify --hash OK at handoff).
- Deadline: 2026-07-15 (private LB decides).

## ⚖️ PORTFOLIO DOCTRINE (2026-07-12, new)
Insurance belongs EXCLUSIVELY on the HEDGE slot. Proof: every task ≥14.6pt > HEDGE-v3 public
handicap ~6.8pt, so ANY single silent-zero on MAIN already makes HEDGE the better selected
board — MAIN-side insurance changes the best-of-two in NO world and costs its price in the
lenient world. MAIN carries only strict wins (005-scale ~0.001pt repairs exempt). 188A/191
fixes were adopted on MAIN then PORTFOLIO-REVERTED to HEDGE-only (tasklogs document both).

## HEDGE v3 protection manifest (8 nets, all bundled fail=0, candidates/ has the artifacts)
219 exact (43.9%→0.30%, −1.25) | 319 exact (7.05%→0.18%, −0.91) | 002 rect-walk (5.58%→2.83%,
−0.50) | 118 varm-peel (7.83%→3.33% strict-dominant, −0.86) | 205 2-D solidity (1.39%→0.01%,
−1.94) | 233 clamped k=2 (2.69%→0.73% + crash-draw fixed, −0.16) | 188 FIX-B (1.03%→0.057%,
−1.02) | 191 8-orient (0.887%→0.000%, −0.11). Rebuild procedure: swap candidates over
overfit_nets, ng score each isolated, pack, submit, restore from scratchpad backup, verify --hash.

## Scoreboard 07-11 late → 07-12 (this session, +3.28 LB confirmed + v3 pending)
7301.10 → 7304.38: C/I triage wave = kron_fractal_einsum 3/3 (217 +1.059, 304 +0.850,
275 +0.486) + 267 einsum-fold +0.744 + 072 conv-XOR +0.137; then +123 palette-fold +0.040 (v3).
Fresh-tail program: 12 tails diagnosed (8 FIXED→hedge, 4 NO-FIX proven: 209 ambiguity-floor,
157 op-vocab plateau, 255 rebuild-scale, 187 info-destruction proof-pair, 366 exact-cover
unaffordable).

## Lane closures this session (all ledgered with reopen triggers — do NOT redispatch)
- **C/I triage lane CLOSED** (state/worklists/ci_triage_2026-07-11.md): 26 judged, 6 wins +3.32,
  20 measured floors. META-DISCRIMINATOR (26/26): win iff incumbent is a legacy Slice/Where/
  Gather chain; floor iff already an optimized einsum/conv net.
- **Legacy-chain board scan ≈0 remaining** (state/worklists/legacy_chain_scan_2026-07-12.md +
  tools/legacy_chain_scan.py): residual legacy fraction = floor carriers, not bloat.
  Wall-residue confirmed board-wide. Re-run the scanner after every mine-public.
- **int8 reverse-arbitrage bounded to 191**: 61 int8-match nets scanned; the 191 win needs
  (real int8/omission tail) AND (linear downstream fold) — conjunction holds nowhere else.
- **kron rescan dry** (3 wins captured the whole tractable class: fixed s or ≤2 size-cases).
- **deepfold rescan dry** (0 new after 07-11 adoptions); **public lane dry** (evgendvorkin 400
  loose nets 0-win; all kernels mined post-last-rerun).

## Durable physics learned (memory/insights.yaml has kron_fractal_einsum)
- **fp Conv with runtime weights is BROKEN in sequential grading** (ORT pre-packs run-0 weight)
  — exact designs must use MatMul/Einsum/QLinearConv (task191 ledger).
- Control flow fully dead: Loop/Scan EXCLUDED; If's GRAPH attr → calculate_memory returns None.
- VI-slack audits must trace the FULL 266-bundle (4-example trace gives false slack; task310).
- Bundled contains REAL-ARC draws, not only arc-gen (task131 k=10 forced by arc-agi example).
- Sparse initializers categorically unusable (sanitize_model never remaps sparse names; 372).
- .backups can hold REGRESSED nets — rebuild reverts from src/custom, not backups (233).

## Next Session Start
1. `uv run ng status` + confirm 54576478/54576509 scores == expected; **set final selection to
   these two on Kaggle** (user action on the website).
2. Kernel poll (author+slug match) → any new dump: `ng mine-public --margin 0 --apply` →
   fresh-gate grafts → `tools/legacy_chain_scan.py` + deepfold rescan (both persisted in tools/).
3. Community-intel threads (drafted, with user): 076/118/173/101 "under 9000?" — any YES
   falsifies an our-physics floor (49th-histogram bound ≥+3.9pt on 15 named sub-16 tasks stands).
4. Optional: canary-probe design (hidden-draw count k calibration — decides whether more
   −0.5~2pt exact ports pay); remaining undiagnosed tails are all ≤0.5% (101 285 133 173 105
   377 158 363) — diagnose only if k proves large.
5. Post-deadline: top-team writeups = the six hidden 25s + the fat-middle idiom (~680pt gap
   is representation-class, not tuning).

## Operational Guardrails
- Fresh-gate every adoption (n≥1500 A/B vs incumbent); bit-identical exempt. Gate via ng gate →
  ng adopt; price-exceptions are HEDGE-only now (see portfolio doctrine) and tasklog-documented.
- Keep onnx==1.21.0 + onnxruntime==1.26.0. TopK feeds float/fp16/int64. submission.zip. 100/day.
- Isolated eval for knife-edge nets (220/230/294 class, 233). Clamp every dynamic Gather/Scatter
  index (unclamped = whole-submission kill; 233 measured).
- Do not re-attempt this session's ledgered floors without their reopen triggers.
