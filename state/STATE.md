# STATE - NeuroGolf live handoff (updated 2026-07-13; endogenous-floor session, +0.175)
> Replace this file at session end; do not append. History lives in git, `state/tasks/`, and `state/levers.yaml`.

## Confirmed State
- 🏆 **RECORD LB 7410.67 (submission 54610908, protected — best-submission counts).** Deployed `main` =
  that config + this session's ONE bit-identical win. Local manifest **7410.80**.
- 🟢 **task239 fp16 TopK-tail recast 697→585 (+0.175), bit-identical 267/267 — ADOPTED + SUBMITTED**
  (2026-07-13, "safe sweep" submission; scoring pending at close). Cannot Kaggle-zero (identical
  outputs). If it scores as projected, new record ≈7410.9.
- ⚠️ **Submission scoring was pending at session close — check `kaggle competitions submissions -c
  neurogolf-2026` first thing.** Record 7410.67 protected regardless.

## This session's finding — ENDOGENOUS FLOOR (the +20 target is physics-blocked)
Ran 20 rigorous agents chasing +20 (no public dumps left, user directive). **Net +0.175.** Everything
else floored on TWO ORT-pinned walls (full ledger: memory `neurogolf-endogenous-floor-2026-07-13`):
1. **fp32-co-bind (Einsum uniform-T):** input is fed fp32; any Einsum on it forces all operands fp32,
   so the 900B bool [30,30] Where-mask is IRREDUCIBLE for positioned routing. Floored 2026-07-13:
   216, 202, 187, 198, 378, 397, 368, 069.
2. **Einsum no-local-shift:** can't express ±1 neighbour offsets → local completion/stamp tasks
   materialize full-grid planes ≥ deployed. Why task017's vote doesn't generalize. Floored: 361
   (argmax centre), 173 (local stamp), 080 (data-dep period), 243 (52-letter cap), 158/162/366.
- **dtype_overpay scanner OVER-COUNTS** (advertises +10.8; real +0.175): it flags fp32 tensors without
  checking they feed an Einsum co-bound to the free input. 239 was the LONE genuine dtype win (TopK-tail,
  terminal Einsum operands all fp16-exact + no raw-input operand). fp16 output-coupled lane dry (0 left).
  kernel_collapse empty. mask_dominance 8/8 fresh cracks floored (raw list = dense-recolor Equal-decode
  floors: label30→Equal→10ch = 124/354/390 class).
- **Independent-minimum confirm:** cristianoc solutions for all high-cost nets are iterative/search
  (0 loop-free short forms among cost≥3000) — the floor-signature oracle, not a self-referential claim.

## Reopen triggers (the ONLY revivers — else do not re-grind)
- (a) a permitted ORT build with mixed-dtype Einsum/Conv (fp16 carrier + fp32 input) → unlocks ~+10–15
  board-wide; every agent names this as THE top-residual lever. Currently blocked by pins (onnx 1.21 /
  ort 1.26, grader-matched — do NOT change).
- (b) fp16/uint8 input feed (grader change).
- (c) a NEW external public dump cheaper than a deployed net → grade in isolation + graft (the +86
  route; user says none are coming for the final 2 days).

## Next Session Start
1. `uv run ng status`; `kaggle competitions submissions -c neurogolf-2026` — confirm the 239 submission
   score (expect ≈7410.9; record 7410.67 protected either way).
2. **Do NOT re-run the 11 floored crack attempts above** (216/202/187/198/378/397/368/069/361/173/243/080)
   — read memory `neurogolf-endogenous-floor-2026-07-13` for the byte-math ledger + reopen triggers.
3. Endogenous levers are at a structural floor. Real upside only from the 3 reopen triggers above.
   If a new public dump appears, use the `public-minmerge` graft routine (the proven +LB lane).

## Operational Guardrails
- Adoption only via `ng adopt`; submission via `ng pack` (auto TopK-safety + refuse) then `ng submit`.
- Preserve evaluator pins: onnx==1.21.0, onnxruntime==1.26.0.
- Pre-submit uint8-TopK scan is built into `ng pack` (refuses on any offender).
- Check existing Kaggle submissions before pack/submit (parallel sessions may submit independently).
