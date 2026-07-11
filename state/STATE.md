# STATE - NeuroGolf live handoff (updated 2026-07-12; k-calibration session — no submissions spent)
> Replace this file at session end; do not append. History lives in git and state/submissions.md.

## Confirmed State
- Confirmed BEST LB: **7304.42** (sub **54576478**, MAIN v3, pure-max). Local 7304.3002, offset +0.12.
- HEDGE v3 confirmed **7297.67** (sub **54576509**, MAIN v3 + 8 protections, handicap −6.75 as priced).
- **FINAL SELECTION (before 07-15): MAIN v3 (54576478) + HEDGE v3 (54576509)** — re-verified via
  `kaggle competitions submissions` 2026-07-12; scores match expected. ⚠️ USER ACTION still open:
  confirm the two slots are actually SELECTED on the Kaggle website (cannot be checked via CLI).
- Deadline: 2026-07-15 (private LB decides). Deployed tree = MAIN v3.

## 🎲 HIDDEN-DRAW COUNT CALIBRATED: k ≈ 1–2 (2026-07-12, NO canary submission needed)
Levers.yaml `hidden-draw-k-calibration` has the full ledger. Method: the public LB scoring of MAIN
boards with ZERO task-zeros, given 31 incumbents with measured per-draw fresh-fail rates
(joint per-draw survival q=0.345), is itself a k-likelihood. Posterior (conservative fixed-set
reading): **k=1: 65.5%, k≤3: 96%, P(k≥10) < 0.01%**. The old "~260 draws" pricing (133-revert
rationale) was off by two orders of magnitude. Private-zero risk for fail-rate p ≈ 1−(1−p)^1.5.
- Data corrections made first: raw fresh_sweep_2026-07-11_raw.json PREDATES the ce8ecc7 >30-filter
  fix — **021 "70%" and 184 "38%" were oversize artifacts** (deployed binaries re-measured clean:
  021 0/446+0/626, 184 0/967+0/1256). Big rates re-confirmed REAL on deployed binaries:
  **219 41.7%, 118 7.67%, 319 6.27%, 002 5.47%**. Mid-tails isolated n=2000: 018 2.0%, 023 2.55%,
  017 1.0%, 101 0.85%, 085 0.3%, 066 0.25%, 048 0, 115 0. Do NOT cite the raw sweep json rates.
- Consequences (all decided, do not re-litigate without ledger reopen triggers):
  1. Canary probe lane CLOSED — evidence already conclusive, submissions saved.
  2. Remaining ≤0.5% tails (101/285/133/173/105/377/158/363): private-zero ~0.5–0.8% each →
     diagnosis/ports DO NOT PAY. Closed.
  3. 018/023 (2.0%/2.55%, unprotected on both boards) hedge-port EV ≈ −0.2…−0.9pt → no builders.
  4. **P(MAIN private-clean) ≈ 26–34%**, dominated by task219 (private-zero ~45–54%) → HEDGE v3
     is the LIKELY final scoring board; the selected pair is optimal as-is. No board changes.
- Fresh-GATING for adoptions is UNCHANGED (n≥1500 A/B) — k small reprices zero-RISK, not the gate.

## ⚖️ PORTFOLIO DOCTRINE (2026-07-12)
Insurance belongs EXCLUSIVELY on the HEDGE slot. Every task ≥14.6pt > HEDGE-v3 public handicap
~6.8pt, so ANY single silent-zero on MAIN already makes HEDGE the better selected board —
MAIN-side insurance changes the best-of-two in NO world. MAIN carries only strict wins.

## HEDGE v3 protection manifest (8 nets, all bundled fail=0, candidates/ has the artifacts)
219 exact (43.9%→0.30%, −1.25) | 319 exact (7.05%→0.18%, −0.91) | 002 rect-walk (5.58%→2.83%,
−0.50) | 118 varm-peel (7.83%→3.33% strict-dominant, −0.86) | 205 2-D solidity (1.39%→0.01%,
−1.94) | 233 clamped k=2 (2.69%→0.73% + crash-draw fixed, −0.16) | 188 FIX-B (1.03%→0.057%,
−1.02) | 191 8-orient (0.887%→0.000%, −0.11). Rebuild procedure: swap candidates over
overfit_nets, ng score each isolated, pack, submit, restore from scratchpad backup, verify --hash.

## Lane closures (ledgered with reopen triggers — do NOT redispatch)
- **hidden-draw-k-calibration CLOSED** (this session, see above).
- **C/I triage CLOSED** (state/worklists/ci_triage_2026-07-11.md): 26 judged, 6 wins +3.32.
  META-DISCRIMINATOR (26/26): win iff incumbent is a legacy Slice/Where/Gather chain.
- **Legacy-chain board scan ≈0 remaining**; re-run tools/legacy_chain_scan.py after every mine-public.
- **int8 reverse-arbitrage bounded to 191**; **kron rescan dry**; **deepfold rescan dry**;
  **public lane dry** (2026-07-12 poll: 75 kernels scanned, no new/updated — nothing to mine).
- Community-intel threads (076/118/173/101 "under 9000?"): checking replies BLOCKED for the agent
  (Kaggle discussions need browser login; Chrome extension not connected 2026-07-12) — USER checks
  manually. Any YES falsifies an our-physics floor; 49th-histogram ≥+3.9pt bound stands.

## Durable physics learned (memory/insights.yaml has kron_fractal_einsum)
- **Hidden eval = k≈1–2 fresh arc-gen draws per task** (2026-07-12 Bayesian; see above).
- **fp Conv with runtime weights is BROKEN in sequential grading** (ORT pre-packs run-0 weight)
  — exact designs must use MatMul/Einsum/QLinearConv (task191 ledger).
- Control flow fully dead: Loop/Scan EXCLUDED; If's GRAPH attr → calculate_memory returns None.
- VI-slack audits must trace the FULL 266-bundle (4-example trace gives false slack; task310).
- Bundled contains REAL-ARC draws, not only arc-gen (task131 k=10 forced by arc-agi example).
- Sparse initializers categorically unusable; .backups can hold REGRESSED nets (rebuild from src).

## Next Session Start
1. `uv run ng status`; re-confirm 54576478/54576509 scores unchanged (a changed score = the
   k-calibration reopen trigger #3 — recompute immediately).
2. Kernel poll: `uv run python tools/poll_public_dumps.py` → any new dump: `ng mine-public
   --margin 0 --apply` → fresh-gate grafts → legacy_chain + deepfold rescans.
3. USER: confirm final-selection slots on Kaggle website; check the 076/118/173/101 discussion
   replies (agent is blocked without browser login).
4. Post-deadline: validate k≈1–2 against the actual private zero pattern (reopen trigger #4);
   top-team writeups = the six hidden 25s + the fat-middle idiom (~680pt gap is
   representation-class, not tuning).

## Operational Guardrails
- Fresh-gate every adoption (n≥1500 A/B vs incumbent); bit-identical exempt. Gate via ng gate →
  ng adopt; price-exceptions are HEDGE-only (portfolio doctrine) and tasklog-documented.
- Keep onnx==1.21.0 + onnxruntime==1.26.0. TopK feeds float/fp16/int64. submission.zip. 100/day.
- Isolated eval for knife-edge nets (220/230/294 class, 233). Clamp every dynamic Gather/Scatter
  index. Do not re-attempt ledgered floors without their reopen triggers.
