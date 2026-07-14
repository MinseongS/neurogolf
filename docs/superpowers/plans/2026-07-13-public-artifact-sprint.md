# NeuroGolf 7430 Public Artifact Sprint Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Produce and verify a completed Kaggle submission scoring at least 7430.00 by min-merging the newest public ONNX artifacts into the protected 7410.67 deployment.

**Architecture:** Use Kaggle kernel outputs as immutable teacher bundles, evaluate every task model in isolated pinned-runtime processes through the existing `ng mine-public` pipeline, and apply only bundled-clean strictly-cheaper models through the normal adoption path. Pack one all-in batch, let Kaggle arbitrate leaderboard compatibility, then reconcile the winning artifacts to project source and insight records.

**Tech Stack:** Python 3.13, `uv`, `onnx==1.21.0`, `onnxruntime==1.26.0`, Kaggle CLI, NeuroGolf `ng` CLI.

## Global Constraints

- The protected starting record is Kaggle submission 54610908 at 7410.67.
- Completion requires a completed Kaggle submission with `publicScore >= 7430.00`; a local manifest estimate is insufficient.
- Adoption requires bundled fail=0 and strictly lower measured cost. Fresh verification is diagnostic and must not block an all-in experiment.
- Every model is evaluated in an isolated process under `onnx==1.21.0` and `onnxruntime==1.26.0`.
- Never copy directly into `submission/overfit_nets/`; public wins are applied through `ng mine-public --apply`, which routes through adoption.
- Pack only through `ng pack` and submit only through `ng submit`.
- Preserve unrelated dirty files and never stage them with broad commands.
- Do not pursue task-level changes below +0.5 during this sprint unless they are free by-products of the public min-merge batch.

---

### Task 1: Acquire and fingerprint the newest public artifacts

**Files:**
- Create: `candidates/public_dumps/20260713_highroi/lucifer19_neurogolf-agi-circuit-forge/`
- Create: `candidates/public_dumps/20260713_highroi/lucifer19_neurogolf-agi-compression-core/`
- Create: `candidates/public_dumps/20260713_highroi/kaiwalyaatulraut_neurogolf-championship-solution/`
- Create: `candidates/public_dumps/20260713_highroi/lucifer19_chimera-safe-boost-caddies/`
- Create: `candidates/public_dumps/20260713_highroi/hoangvux_neurogolf/`
- Inspect: `state/seen_kernels.json`

**Interfaces:**
- Consumes: Kaggle kernel refs and authenticated Kaggle CLI.
- Produces: five source-attributed `submission.zip` paths plus SHA-256 fingerprints.

- [ ] **Step 1: Reconfirm current external state**

Run:

```bash
uv run ng status
kaggle competitions leaderboard -c neurogolf-2026 --show
kaggle competitions submissions -c neurogolf-2026
kaggle kernels list --competition neurogolf-2026 --sort-by dateRun --page-size 50
```

Expected: deployed nets are 400/400, the protected best remains at least 7410.67, and the named 2026-07-13 kernels remain visible.

- [ ] **Step 2: Download each public output into its own immutable directory**

Run:

```bash
kaggle kernels output lucifer19/neurogolf-agi-circuit-forge -p candidates/public_dumps/20260713_highroi/lucifer19_neurogolf-agi-circuit-forge
kaggle kernels output lucifer19/neurogolf-agi-compression-core -p candidates/public_dumps/20260713_highroi/lucifer19_neurogolf-agi-compression-core
kaggle kernels output kaiwalyaatulraut/neurogolf-championship-solution -p candidates/public_dumps/20260713_highroi/kaiwalyaatulraut_neurogolf-championship-solution
kaggle kernels output lucifer19/chimera-safe-boost-caddies -p candidates/public_dumps/20260713_highroi/lucifer19_chimera-safe-boost-caddies
kaggle kernels output hoangvux/neurogolf -p candidates/public_dumps/20260713_highroi/hoangvux_neurogolf
```

Expected: each successful source directory contains `submission.zip`; a source without that file is excluded without stopping the other downloads.

- [ ] **Step 3: Validate presence, archive shape, and hashes**

Run:

```bash
find candidates/public_dumps/20260713_highroi -name submission.zip -type f -print
shasum -a 256 candidates/public_dumps/20260713_highroi/*/submission.zip
unzip -l candidates/public_dumps/20260713_highroi/lucifer19_neurogolf-agi-circuit-forge/submission.zip
unzip -l candidates/public_dumps/20260713_highroi/lucifer19_neurogolf-agi-compression-core/submission.zip
unzip -l candidates/public_dumps/20260713_highroi/kaiwalyaatulraut_neurogolf-championship-solution/submission.zip
unzip -l candidates/public_dumps/20260713_highroi/lucifer19_chimera-safe-boost-caddies/submission.zip
unzip -l candidates/public_dumps/20260713_highroi/hoangvux_neurogolf/submission.zip
```

Expected: usable archives contain task ONNX files; identical SHA-256 archives are evaluated only once.

---

### Task 2: Compute the isolated all-in min-merge frontier

**Files:**
- Read: `submission/overfit_nets/task001.onnx` through `task400.onnx`
- Read: `state/manifest.json`
- Read: `src/neurogolf/scans/minmerge.py`
- Create: command output report retained in the task transcript

**Interfaces:**
- Consumes: the unique valid `submission.zip` files from Task 1.
- Produces: a deterministic list of bundled-clean strictly-cheaper task models, per-task deltas, and total projected local gain.

- [ ] **Step 1: Run dry min-merge across every unique artifact**

Run with the valid unique paths from Task 1:

```bash
uv run ng mine-public --margin 0 candidates/public_dumps/20260713_highroi/lucifer19_neurogolf-agi-circuit-forge/submission.zip candidates/public_dumps/20260713_highroi/lucifer19_neurogolf-agi-compression-core/submission.zip candidates/public_dumps/20260713_highroi/kaiwalyaatulraut_neurogolf-championship-solution/submission.zip candidates/public_dumps/20260713_highroi/lucifer19_chimera-safe-boost-caddies/submission.zip candidates/public_dumps/20260713_highroi/hoangvux_neurogolf/submission.zip
```

Expected: every candidate is evaluated in isolation and the final line reports `TOTAL adoptable:` followed by the measured point delta and task count; no deployed file changes in this dry run.

- [ ] **Step 2: Apply the sprint decision threshold**

Decision:

- Projected gain `>= +20`: proceed immediately to Task 3.
- Projected gain `>= +5 and < +20`: still proceed to Task 3, because the protected record makes the all-in experiment asymmetric.
- Projected gain `< +5`: do not apply byte-tail-only wins yet; inspect the winning public graphs against tasks 233, 366, 054, 133, 285, 158, 286, and 018 and write a separate high-yield rewrite plan requiring at least +5 batch expectation.

Expected: the next action is selected from measured total gain, not from freshgate or per-task micro-golf.

---

### Task 3: Apply and verify the public all-in frontier

**Files:**
- Modify through `ng`: `submission/overfit_nets/taskNNN.onnx` for each win
- Modify through `ng`: `state/manifest.json`
- Modify through `ng`: `state/tasks/taskNNN.md`
- Create through `ng`: timestamped task backups under `submission/.backups/`
- Create: `candidates/public_dumps/20260713_highroi/apply_started.marker`

**Interfaces:**
- Consumes: Task 2's eligible task frontier.
- Produces: one 400-model deployment whose local manifest includes every eligible public win.

- [ ] **Step 1: Re-run the same frontier with adoption enabled**

Run:

```bash
touch candidates/public_dumps/20260713_highroi/apply_started.marker
uv run ng mine-public --margin 0 --apply candidates/public_dumps/20260713_highroi/lucifer19_neurogolf-agi-circuit-forge/submission.zip candidates/public_dumps/20260713_highroi/lucifer19_neurogolf-agi-compression-core/submission.zip candidates/public_dumps/20260713_highroi/kaiwalyaatulraut_neurogolf-championship-solution/submission.zip candidates/public_dumps/20260713_highroi/lucifer19_chimera-safe-boost-caddies/submission.zip candidates/public_dumps/20260713_highroi/hoangvux_neurogolf/submission.zip
```

Expected: each reported win passes the adoption gate, creates a timestamped backup, updates its task ledger, and lowers the manifest cost.

- [ ] **Step 2: Verify deployment integrity and hashes**

Run:

```bash
uv run ng status
uv run ng verify --hash
```

Expected: nets 400/400, zero bundled failures, manifest hashes match all deployed ONNX files, and the local total is no lower than the pre-apply total.

- [ ] **Step 3: Pack and let the packer run the unsigned-TopK safety scan**

Run:

```bash
uv run ng pack
```

Expected: `submission.zip` is produced successfully; pack refuses automatically if any unsigned TopK model is present.

---

### Task 4: Submit the all-in experiment and verify the real score

**Files:**
- Create through `ng`: `submission.zip`
- Modify after result: `state/submissions.md`

**Interfaces:**
- Consumes: Task 3's packed all-in deployment.
- Produces: an authoritative Kaggle leaderboard result and either completion evidence or a cohort-split diagnosis.

- [ ] **Step 1: Prevent a duplicate concurrent submission**

Run:

```bash
kaggle competitions submissions -c neurogolf-2026
```

Expected: no newer overlapping submission from another session is pending or complete.

- [ ] **Step 2: Submit the batch**

Run:

```bash
uv run ng submit -m "7430 sprint: 7410.67 plus 20260713 public all-in min-merge"
```

Expected: Kaggle returns a new submission reference in pending or complete state.

- [ ] **Step 3: Poll until completion**

Run periodically:

```bash
kaggle competitions submissions -c neurogolf-2026
```

Expected: the new reference reaches `SubmissionStatus.COMPLETE` with a numeric public score.

- [ ] **Step 4: Evaluate the authoritative result**

Decision:

- Score `>= 7430.00`: the leaderboard goal is achieved; continue to Task 5.
- Score below 7430.00 but at least 7410.67: preserve the improvement and split the remaining public sources by large-delta cohort for the next all-in experiment.
- Score below 7410.67: restore the protected 7410 deployment using adoption backups, then isolate large-delta source cohorts; do not introduce freshgate as a blanket filter.

Expected: no completion claim is made without a completed Kaggle score at or above 7430.

---

### Task 5: Reconcile the winning deployment and record completion

**Files:**
- Modify: `src/custom/taskNNN.py` for every adopted public task
- Modify: `state/insights.yaml`
- Modify: `state/submissions.md`
- Replace: `state/STATE.md`
- Modify: `state/baseline_manifest.json`
- Modify: `state/baseline_hashes.sha256`

**Interfaces:**
- Consumes: a completed Kaggle submission at or above 7430 and its exact deployed model set.
- Produces: source-owned winning graphs, durable mechanism records, refreshed handoff state, and a focused commit.

- [ ] **Step 1: Reconstruct adopted graphs in source**

Derive the adopted task ids from backups created after the apply marker and reconstruct them:

```bash
find submission/.backups -type f -newer candidates/public_dumps/20260713_highroi/apply_started.marker -name 'task*.onnx' -print | sed -E 's#.*/task([0-9]{3})_.*#\1#' | sort -u | xargs uv run python tools/live_to_exact_source.py --write-src
```

Expected: every adopted public task has a matching `src/custom/taskNNN.py` builder and rebuild parity.

- [ ] **Step 2: Run the mandatory public insight lane**

Run:

```bash
uv run ng scan public_autopsy
```

Expected: public op-delta fingerprints are written to the public-autopsy worklist; material reusable mechanisms are added to `state/insights.yaml` and rescanned across all 400 tasks.

- [ ] **Step 3: Refresh durable state**

Update `state/submissions.md` with the submission id, message, local total, and Kaggle score. Replace `state/STATE.md` with the confirmed record, active next lever, and guardrails. Refresh baseline manifest and SHA inventory using the repository's existing baseline refresh workflow.

Expected: STATE contains only current truth and identifies the completed score as the record.

- [ ] **Step 4: Final verification**

Run:

```bash
uv run ng status
uv run ng verify --hash
uv run ng pack
git diff --check
```

Expected: 400/400, hash-clean, pack succeeds, no whitespace errors, and the packed deployment matches the leaderboard-winning model set.

- [ ] **Step 5: Commit only in-scope files**

Stage the adopted task sources, auto-stamped task logs, manifest/baseline files, insights, submissions log, and STATE explicitly. Do not stage pre-existing unrelated dirty files. Commit with:

```bash
git commit -m "optimize public frontier past 7430" -m "Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

Expected: one focused completion commit while unrelated user changes remain unstaged.
