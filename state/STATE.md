# STATE - NeuroGolf live handoff (updated 2026-07-11; poby7722 min-merge +0.0618, idea-lane audit day)
> Replace this file at session end; do not append. History lives in git and state/submissions.md.

## Confirmed State
- Confirmed BEST LB: **7300.65** (sub **54559732**, complete). Prior 7300.59 (54518734).
- Current local manifest: **7300.5290** (400/400), local↔LB offset ~+0.12 as usual.
- Deadline: 2026-07-15. Bundled fail=0 + cheaper than deployed remains the adoption gate.

## Today's arc (2026-07-11)
1. **Public mine**: poby7722 renamed kernel slug 7240→7263.52 (slug rename evades seen_kernels —
   also match by AUTHOR when polling). margin-0 → 3 wins: 285 +0.0496 / 367 +0.0111 / 396 +0.0011.
   Src regen + semantic verify 3/3. evgendvorkin EDA re-run = profiler JSONs only, no nets.
2. **public_autopsy scanner FIXED**: it preferred the unstamped stale .minmerge_backup → bogus
   fingerprints/expected_gain (396 "quantized_integer_route +0.32" was a stale-baseline artifact;
   real 396 win = one Reshape broadcast elision −32B). Now prefers latest adopt-stamped
   submission/.backups/taskNNN_*.onnx. All 3 wins = pure per-net byte tails, no board fingerprint.
3. **Idea-lane audit, 4 lanes closed with 4-field ledgers** (levers.yaml / tasks/task233.md):
   - 233 quantized-route lens: NO — red-exclusion is load-bearing on bundled (4/4 ex), u8
     per-color plane ≥2584B > savings; fp32-detection floor 3rd independent confirm.
   - 243 P2 [1,2,18,18] 2592B (deepfold board #1, dpts 1.035): NO — design-family floor
     (52/52 letters MEASURED, fp32-coupling, s-axis expressiveness). 4020 = family floor.
   - GridSample spatial embed: arithmetically DEAD (grid ≥4B/px vs bool 1B/px tails; no bool/u8
     kernel → no free-output emission; 400-net sweep for ≥3 shared-grid fp32 rearr planes = 0).
   - RNN/LSTM/GRU: **LEGAL + kernels exist (incl. fp16 LSTM) + Y omissible → 30-step recurrence
     counts as Y_h 120B** (insights.yaml rnn_lstm_gru_legal_recurrence) — but 0 paying targets
     (rank-4 one-hot input forces counted rank-3 X ≥3600B; walk-einsum reads input free).

## Boundary laws (check BEFORE any new fold attempt)
1. **Letter-budget floor**: 52 letters + ellipsis; ~8 reserved ⇒ max ~42 walk pairs/plane.
   187/286/002 alphabet-floored; 243 chain-merge needs +63 letters vs 0 headroom.
2. **fp32-coupling gate**: fold pays ONLY if incumbent already pays fp32 on the dying planes;
   fp16/bool-isolated tails get worse. (dying B) − (fp32 penalty) − (params) ≥ ~480 or don't build.
3. **Expressiveness**: AND-of-halfplanes, non-separable positioned content, data-dependent
   one-hot selectors, diagonal-p products, two-term sums (243 s-axis) don't fit one linear
   contraction + single >0 threshold.

## Active Veins
1. **New public frontier** — the only bulk lever left. Poll to 07-15
   (`kaggle kernels list --competition neurogolf-2026 --sort-by dateRun`); check seen_kernels by
   author+slug (slug renames!); on ANY new dump: `ng mine-public <dir> --margin 0 --apply`, then
   deepfold-rescan the grafts (`uv run python tools/deepfold_scan.py` — now repo-persisted).
2. **s8port fanout on future adoptions** — dry on current board; reopens with new min-merge grafts.
3. **QLinearConv detection-recast** (`ng scan qlinear_recast`) — re-run after any new Conv net.
4. Long-shot reopens (ledgered, not actionable under pins): mixed-dtype/uint8/bool Einsum or
   GridSample kernel, >52-index einsum, gated matrix-power atomic op, >42-step 1-D recurrence
   (RNN capability is proven-legal and waiting — see insights.yaml).

## Operational Guardrails
- Do not spend session time on +0.0x byte-tail cleanup unless explicitly requested.
- Do not re-attempt without reopen triggers: 233-lens list (349/366/158/173/204/018/133/285),
  s8port gate-negatives (066/264/256/250/396), letter-floor (187/286/002), 243 P2 family floor,
  GridSample lane, RNN on current board, 025 diagonal-p, 080 selector.
- Keep `onnx==1.21.0` + `onnxruntime==1.26.0`; no runtime upgrades without full 400/400 re-verify.
- Adopt via `ng gate` → `ng adopt` (+ live_to_exact_source --write-src + semantic verify); submit
  via `ng pack` → `ng submit`. Kaggle TopK feeds: float/fp16/int64 only.

## Next Session Start
1. `uv run ng status` && `kaggle competitions submissions -c neurogolf-2026 | head -4`
2. `kaggle kernels list --competition neurogolf-2026 --sort-by dateRun --page-size 20` — mine
   anything unseen; compare by AUTHOR too (poby slug-rename lesson); check seen_kernels MINED.
3. If a new dump lands: mine → adopt → deepfold-rescan grafts (tools/deepfold_scan.py).
4. Otherwise the endogenous idea board is fully ledgered as of 07-11 — new gains need either a
   new public teacher, a reopen trigger firing, or a genuinely new design family. Candidate
   unexplored directions: 8000-tier reverse engineering (top ~7982 — what representation system
   do they run?), and re-reading competition discussion for op-capability hints.
