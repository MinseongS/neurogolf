# STATE — NeuroGolf live handoff (updated 2026-07-16 10:05 KST)

> Replace this file at session end; do not append. Historical detail remains in git,
> `state/tasks/`, `state/submissions.md`, `state/levers.yaml`, and candidate reports.

## Current truth

- Deployed nets: **400/400**.
- Working manifest: **7455.8008**. This is not a completed full-board verification snapshot.
- Best confirmed leaderboard score remains **7431.10** (ref 54719275). No pack or submission ran
  in this session.
- Latest previously recorded immutable submission snapshot remains local **7448.024140** with
  bundled 400/400 and failures `[]`; ref 54728797 scored public 7414.54.
- The working tree has extensive concurrent/user changes. Preserve them; do not bulk-reset or
  stage blindly.

## 2026-07-16 score harvest

- Session start manifest: **7454.9756**. Current: **7455.8008**. Manifest-visible gain:
  **+0.8252**.
- **task095** adopted `208 -> 178`, score `19.662462 -> 19.818216`, **+0.155755**.
  The rank-5 learned spatial code passes bundled 265/265. Fresh5000 is intentionally diagnostic
  in 8000-overfit mode: incumbent failed23, candidate failed89, divergence102. Rank4/rank3 bounded
  searches found no exact candidate. Deployed/source SHA is `50c1bddca44b...`.
- **task198** adopted `7922 -> 7915`, **+0.000884**, bundled266/266. It rescales three rm32
  families so their thresholds reuse existing `cthr=15`. Fresh5000 candidate and incumbent had
  identical outputs (both fail2, divergence0). SHA `7c824c1442d4...`.
- **task243** adopted `4020 -> 2060`, score `16.700963 -> 17.369539`, **+0.668576**,
  bundled265/265. A 90-operand/50-label terminal Einsum deletes counted P2 and Cr, walks directly
  on padded-30 FREE input, and shares the black destination colour. Fresh500 matched incumbent
  exactly (both fail1, divergence0). SHA `e4fcc7392ced...`.
- **task187** physical deployed regression was repaired through gate/adopt: actual `4911 -> 4777`,
  bundled266/266. The manifest already credited cost4777, so this improves the real artifact but
  does not change the manifest total. SHA `013590e8945c...`.
- Exact Python source was regenerated and byte-verified against deployment for tasks095/198/243.

## New mechanism and global rescan

- `state/insights.yaml::walk_einsum_iteration_collapse` now records task243's padded-30
  shared-black FREE-input walk and the dated 400-task propagation scan.
- Full-board structural scan report:
  `candidates/task243/TRANSFER_SCAN_20260716.md`.
- No second adoption-ready transfer was found. task002 has a measured +0.492 static candidate but
  costs 338s bundled / 105.9x runtime; task187/251 change off-grid semantics; task145 loses crop/
  dtype economics.
- task286's alternative two-lane terminal candidate priced at cost16405 but failed all 265 bundled
  examples (first example wrong113). Copy/reach lanes have incompatible path-count scales and the
  signed parity contraction leaves positive zero-margin residuals. It was not adopted.

## Scanner reality

- Cached public min-merge over all 70 local submission archives: **0 adoptable tasks**.
- Current kernel-collapse scan: empty.
- canvas-crop task378 headline was false: exact/fresh-identical crop candidate costs2064 versus
  incumbent1980 because the condition must be padded back to 30x30.
- task202's +0.452 queue item is stale; its big FREE-output fold is already deployed. Remaining
  900B is a data-dependent outer equality and the tested uint8 paths are schema-invalid.
- `ng queue` public-autopsy headline gains such as task017/task305 are historical adoption gains,
  not current upside. Both tasks already cost10.

## Submission blocker / next start

- Manifest/deployed SHA mismatch remains on **31 tasks**:
  `018 023 050 080 084 096 101 107 133 134 138 139 141 145 158 191 206 217 270 280 281 286 297 323 356 367 374 381 396 398 400`.
- Do not pack or submit until those rows are isolated, rescored, and reconciled without overwriting
  concurrent work. After reconciliation run a full `uv run ng verify --hash`; prior whole-board
  verifies can time out on task047 under concurrent load.
- Start next session with `uv run ng status`, then read this file and `state/levers.yaml`.

## Invariants

- Goal 8000; default gate is bundled fail=0 plus strictly cheaper. Fresh is diagnostic.
- Every adoption must use `ng gate -> ng adopt`; never copy directly into deployment.
- Candidate work stays under `candidates/`.
- Keep `onnx==1.21.0` and `onnxruntime==1.26.0` pinned.
- Before any eventual submission: reconcile hashes, full verify, scan unsafe TopK dtypes, check
  recent Kaggle submissions, then `ng pack -> ng submit`.
