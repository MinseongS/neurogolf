# STATE - NeuroGolf live handoff (updated 2026-07-10 PM; runtime-spend axis opened, 7300.59 confirmed)
> Replace this file at session end; do not append. History lives in git and state/submissions.md.

## Confirmed State
- Confirmed BEST LB: **7300.59** (sub **54518734**, complete). Prior 7299.25 (54518268), 7299.09 (54516745).
- Current local manifest: **7300.4672** (400/400), local↔LB offset ~+0.12 as usual.
- Deadline: 2026-07-15. Bundled fail=0 + cheaper than deployed remains the adoption gate.

## Today's arc (2026-07-10, two sessions)
1. AM: task233 match-matrix inversion +0.1305 (LB 7299.09); 233-lens 8/8 wall re-audit all negative
   (ledgers in state/tasks/).
2. PM: **runtime-timeout-spend axis OPENED** (user-directed). Runtime measured: full 400-net suite over
   ALL bundled examples = **4.26 s** (~200× headroom vs top-competitor budgets) — runtime is a free
   resource; "fatter single op" trades are always affordable.
3. **deepfold scan** (fixes old fold.py blind spots: dpts=ln(cost/(cost−B)) ranking, 600B min,
   indicator-algebra max==sum>0, Conv-as-reducer, FLOORED-list exposure) → 48 candidates →
   **5 wins, +1.495 LB**: 251 +0.588 (epsilon-J border-flood), 132 +0.391 (signed rank-K separable),
   177 +0.236 (input-as-onehot decode), 243 +0.160 (S8-port stacked chain), 085 +0.120 (rank-K punch).
   All adopted + source-owned (src/custom/) + verified (see ADOPTED entries in state/tasks/).
4. Mechanism registered: `insights.yaml: s8port_free_output_tail_fold` (5 sub-recipes + gate condition).

## The three boundary laws found today (check BEFORE any new fold attempt)
1. **Letter-budget floor**: einsum language caps absorption at 52 letters + one ellipsis; ~8 reserved
   for stack/channels/embeds ⇒ max ~42 walk pairs per plane. 187 (52/52), 286 (49-52/52), 002 (52/52)
   are alphabet-floored for chain merges.
2. **fp32-coupling gate** (066 measured): fold pays ONLY if the incumbent already pays fp32 on the dying
   planes. fp16/bool-isolated tails (066/264/256/250/396) get WORSE — binding the free fp32 input into a
   uniform-dtype einsum doubles their machinery. Arithmetic: (dying B) − (fp32 penalty) − (params, zeros
   count) ≥ ~480 or don't build.
3. **Expressiveness**: AND-of-halfplanes (256 anti-diagonal), non-separable positioned content (396
   rank-4), data-dependent one-hot selectors (080), diagonal-p products (025) don't fit one linear
   contraction + single >0 threshold.

## Active Veins
1. **New public frontier** — the only bulk lever left. Poller runs to 07-15; jonathanncoletti merged91 +
   anasriaz re-pull mined 2026-07-10 → 0 adoptable; seen_kernels backlog clear. On ANY new dump:
   `ng mine-public --margin 0`, then re-run the deepfold scan on adopted grafts (fresh grafts may carry
   fp32-paying tails = s8port food; scan at scratchpad deepfold_scan.py — port to `ng scan` if it fires again).
2. **s8port fanout on future adoptions** — vein currently dry on deployed board (wave2 0/4: all remaining
   Equal→Pad onehot tails are bool/u8, gate-condition negative). Reopens automatically with new min-merge grafts.
3. **QLinearConv detection-recast** (`ng scan qlinear_recast`) — re-run after any new Conv net.
4. Long-shot reopens (recorded in ledgers, not actionable under pins): mixed-dtype/uint8 Einsum kernel,
   >52-index einsum, 2nd free-named tensor.

## Operational Guardrails
- Do not spend session time on +0.0x byte-tail cleanup unless explicitly requested.
- Do not re-attempt: 233-lens on 349/366/158/173/204/018/133/285 (2026-07-10 ledgers), s8port on
  066/264/256/250/396 (gate-condition negatives), fold on 187/286/002 (letter floor), 025 diagonal-p,
  080 selector — reopen triggers only.
- Keep `onnx==1.21.0` + `onnxruntime==1.26.0`; no runtime upgrades without full 400/400 re-verify.
- Adopt via `ng gate` → `ng adopt` (+ live_to_exact_source --write-src + semantic verify); submit via
  `ng pack` → `ng submit`. Kaggle TopK feeds: float/fp16/int64 only.

## Next Session Start
1. `uv run ng status` && `kaggle competitions submissions -c neurogolf-2026 | head -4`
2. `kaggle kernels list --competition neurogolf-2026 --sort-by dateRun --page-size 20` — mine anything
   unseen (check seen_kernels.json actually MINED, not just seen — merged91 lesson).
3. If a new dump lands: mine → adopt → deepfold-rescan the grafts (s8port food).
4. Otherwise: runtime axis part 2 candidates = LSTM/GRU/RNN legal-unexplored (op vocabulary note in
   competition-setup memory; spatial recurrence params were the 187 blocker but 1-D scan tasks may fit);
   or GridSample as a params-free spatial embed (would reopen 264/396-class template stamps).
