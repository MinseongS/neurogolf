# STATE - NeuroGolf live handoff (updated 2026-07-09, 7298.96 confirmed)
> Replace this file at session end; do not append. History lives in git and state/submissions.md.

## Confirmed State
- Confirmed BEST LB: **7298.96** (subs **54494910** and **54494981**, both complete).
- Current local manifest: **7298.8411** (400/400). This includes task367, fold-scan task174/task107, biohack44 public min-merge, kernel-collapse task044/task107, and franksunp public tail task165/task157/task096.
- Latest meaningful 0.x batch: **54493921** (biohack44 public min-merge 6 adoptions on top of task174/task107, local-at-submit **7298.8191**) -> publicScore **7298.94**.
- Latest small-tail submissions:
  - sub **54494910**: kernel-collapse on new min-merge Conv nets (`task044` 3169->3139, `task107` 3445->3415), publicScore **7298.96**.
  - sub **54494981**: public frontier full re-mine (`task165`/`task157`/`task096`, +0.0037), publicScore **7298.96**.
- Latest completed submissions:
  - sub **54490915**: task080 fp32 colour-decode -> free-input Einsum per-read, publicScore **7297.79**.
  - sub **54491406**: task161 scalar carrier tail -> free-output Einsum, publicScore **7297.90**.
  - sub **54493725**: task367 predicate-bank prune + diagonal diffusion tap, publicScore **7298.02**.
  - sub **54493869**: task174 symmetry-bitmask compaction + task107 dead-tap conv, publicScore **7298.24**.
  - sub **54493921**: biohack44 public min-merge batch, publicScore **7298.94**.
  - sub **54494910**: kernel-collapse tail, publicScore **7298.96**.
  - sub **54494981**: public frontier tail, publicScore **7298.96**.
- Verification: `uv run ng verify --hash` -> **HASH-OK** after latest manifest; `uv run ng status` -> **400/400**, local **7298.8411**.
- Deadline: 2026-07-15. Private is the fixed dataset; bundled fail=0 + cheaper than deployed remains the adoption gate.

## Latest Score Movement
1. `task367` - cost **15890 -> 14082**, points **15.3266 -> 15.4473** (**+0.1207**)
   - Mechanism: prune counted `v_main` QLinearConv predicate bank from 10 channels to 6 (`016789`), then add one diagonal tap to downstream `Wc` diffusion kernel to repair the single sparse overfill failure.
   - Candidate: `candidates/task367/cand.onnx`; source regenerated in `src/custom/task367.py`; gate PASS.
2. `task174`/`task107` fold-scan batch - local **+0.222** before public submit.
   - `task174`: symmetry-bitmask compaction, cost **4027 -> 3348**.
   - `task107`: dead-tap conv fold, cost **3576 -> 3445**, then kernel-collapse to **3415**.
3. biohack44 public min-merge batch - local **+0.6966**.
   - Main wins: `task171` **759 -> 435** (**+0.557**) and `task085` **2875 -> 2519** (**+0.132**).
   - Smaller wins: `task044`, `task035`, `task233`, `task090`.
   - Deep-lane autopsy registered `borrowed_net_redundant_branch_prune`: these were dead-node/redundant-branch cleanups in newly borrowed public nets, not a new broad representation lever.
4. post-7298.94 tails - local **+0.0218**.
   - `task044`/`task107` kernel-collapse: +0.018 local, LB stayed at **7298.96**.
   - franksunp public re-mine `task165`/`task157`/`task096`: +0.0037 local, LB stayed at **7298.96**.

## Current Audit Results
- User threshold: avoid +0.0x byte cleanup; only pursue plausible **single-task +0.1**, preferably **+0.3+**, or reusable mechanisms.
- Public frontier: latest refresh over **29 submission.zip** artifacts found **0 strictly-cheaper bundled-fail=0 wins** beyond the already adopted tails.
- `task216`: no-build. The scanner's +0.385 signal is `c12_f32` (`[1,2,20,20]`, 3200B), a load-bearing fp32 bridge for QLinearConv/crop/count logic, not a removable output mask. Current cost **8386**; cached public artifacts are worse.
- `task366`: no-build. The public-autopsy +0.306 item is stale from the already adopted **30159 -> 22219** rewrite. Current cost **22219**; further +0.1 needs ~2115B saved, and the only large tensor is load-bearing fp32 colour detection.
- Existing high-ranked cached candidates re-gated against current deployed nets: mostly already-adopted ties or fail/cost-worse. `task133` Clip-drop gives only 2B, below threshold.

## Active Veins
1. **New public frontier** - first priority if a new dump appears: `uv run ng mine-public --margin 0 <submission.zip...>`, then autopsy only for +0.1+ fingerprints.
2. **High-cost true rewrite** - remaining path to 7300 is not tail cleanup. Need a real replacement for large detection/assignment cores in `task233`, `task018`, `task133`, `task285`, `task349`, `task101`, `task076`, `task118`, or a new public teacher.
3. **Predicate-bank prune + downstream repair** - `task367` proved the mechanism, but current fanout has no PASS yet. Reopen only with a new scanner/teacher, not another blind subset sweep on `task349`.

## Operational Guardrails
- Do not spend session time on +0.0x byte-tail cleanup unless explicitly requested.
- Do not retry cost-1 tail nets, 092-profile cohort repeats, dtype boundary casts, `task216 c12_f32`, `task366 public_autopsy stale item`, or 014/350/018 value-info crop without the reopen triggers above.
- Keep `onnx==1.21.0` and `onnxruntime==1.26.0`; no runtime upgrades without full 400/400 re-verify.
- Adopt only through `uv run ng gate` -> `uv run ng adopt`; submit through `uv run ng pack` -> `uv run ng submit`.

## Next Session Start
1. `uv run ng status`
2. `kaggle competitions submissions -c neurogolf-2026 | head -6`
3. Check new public dumps/notebooks first.
4. If no public frontier, pick one high-cost true-rewrite target with explicit cost-split proof before building; do not work from stale queue deltas alone.
