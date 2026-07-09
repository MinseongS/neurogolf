# STATE - NeuroGolf live handoff (updated 2026-07-09, task343 + known-mechanism audit)
> Replace this file at session end; do not append. History lives in git and state/submissions.md.

## Confirmed State
- Confirmed BEST LB: **7297.46** (sub **54489856**, complete).
- Current local manifest: **7297.3437** (400/400).
- Latest completed submissions:
  - sub **54489405**: task197 self-Einsum cleanup, publicScore **7296.98**.
  - sub **54489856**: task343 fixed Slice->Conv colour reads -> free-input Einsums, publicScore **7297.46**.
- Verification after latest adoption: `ng gate` PASS, task343 fresh **1500/1500** diff=0, `uv run ng verify --hash` -> **HASH-OK**, `uv run ng pack` completed.
- Deadline: 2026-07-15. Private is the fixed dataset; bundled fail=0 + cheaper than deployed remains the adoption gate.

## Adopted In Latest Score Batch
1. `task197` - cost **852 -> 753**, points **18.2524 -> 18.3759** (**+0.1235**)
   - Candidate: `candidates/task197/task197_self_einsum.onnx`
   - Mechanism: compute first-template-row predicates directly from FREE input:
     `match[w]=dot(input[:,1,0], input[:,1,w])`, `active[w]=sum_c input[c,1,w]`.
     Deletes counted `T`/`g` Slice outputs (**-440B**) and buys a 360-element selector bank.
   - Gate PASS; fresh **1500/1500**, incumbent fail=0, candidate fail=0, diff=0.
   - Insight updated: `self_einsum_axis_activity_gate` includes `task197`.
2. `task343` - cost **1222 -> 756**, points **17.892 -> 18.372** (**+0.4802**)
   - Candidate: `candidates/task343/task343_free_input_color.onnx`
   - Builder: `candidates/task343/build_free_input_color.py`
   - Mechanism: fold fixed `Slice(input)->1x1 Conv` colour-read windows into FREE-input `Einsum`s with static row/col/channel selectors.
     Deletes `source_nonzero`/`source8_nonzero` counted windows and replaces them with cheaper selectors.
   - Gate PASS; fresh **1500/1500**, incumbent fail=0, candidate fail=0, diff=0.
   - Insight added: `fixed_slice_conv_to_free_input_einsum`.

## Audit Ledger
- User threshold for this session: skip +0.0x byte cleanup; build only plausible **single-task +0.1**, preferably **+0.3+**, or reusable mechanisms.
- Dry/no-build outcomes from this session are recorded in `state/levers.yaml` ledger entries dated **2026-07-09**:
  - `public-insight-generalize`: task343 win and fixed Slice->Conv post-scan; perm/probe/axis fanouts.
  - `free-output-einsum-regime-crack`: stamp/crop scalar-carrier re-audit (`112/062/099/163/102`, `014`).
  - `dtype-overpay`: deployed dtype residual recheck after task197.
  - `public-minmerge`: current local public dumps rechecked after task343.

## Active Veins
1. **New public frontier** - first priority if a new dump appears: `uv run ng mine-public --margin 0`, then `public_autopsy` only for +0.1+ mechanism fingerprints.
2. **Known-mechanism coverage audit** - keep using cost-split first. Build only when counted carrier/window deletion beats selector/onehot expansion by at least ~0.1 task points.
3. **True new-op / new-lowering work** - the high-value unlocks are mixed-dtype/free-input contraction legality, final-output fused routing for 900B label/mask carriers, and cheap dynamic selector construction.

## Operational Guardrails
- Do not spend session time on +0.0x byte-tail cleanup unless explicitly requested.
- Do not retry cost-1 tail nets, `350/148/233/042`, 092-profile cohort repeats, or `014/350/018` value-info crop without the reopen triggers above.
- Keep `onnx==1.21.0` and `onnxruntime==1.26.0`; no runtime upgrades without full 400/400 re-verify.
- Adopt only through `uv run ng gate` -> `uv run ng adopt`; submit through `uv run ng pack` -> `uv run ng submit`.

## Next Session Start
1. `uv run ng status`
2. `kaggle competitions submissions -c neurogolf-2026 | head -6`
3. Check for new public dumps/notebooks first.
4. If no new public frontier, continue only +0.1/+0.3 mechanism candidates with explicit cost-split proof before building.
