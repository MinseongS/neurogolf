# Sweep System — targeted 1→400, minimal-tier mindset

## North star
- **#1 = 7700. Our baseline ambition = 7500** (= avg **18.75 pts/task**). We sit at ~16.1 avg.
- **Mindset rule:** if a task lands below the minimal tier it *could* reach, that is a SIGNAL that
  the structure hasn't been found yet — NOT an acceptable stopping point. Re-attack from a new
  angle before recording a verdict. Settling at the 16.8 label-map floor on a task that has
  separable / single-op structure = "생각이 부족한 것" (thinking was insufficient).
- We will not always reach Tier S, but every task gets pushed until we can articulate WHY a lower
  tier is irreducible (which exact intermediate dominates and why no reformulation removes it).

## The tier ladder (target the TOP tier that the rule admits)
| Tier | form | mem | pts | when it applies |
|---|---|---|---|---|
| **S** | single op input→output (Identity / one Conv+bias / one Gather-permute / one MatMul) | ~0 | **18–25** | output color per cell is a linear/permutation/index function of input |
| **A** | separable one-hot: rowcond[1,*,30,1] ⊗ colcond[1,*,1,30], final And/Mul→output | ~2.4–3k | **~17** | output[ch,r,c] = rowcond AND/MUL colcond |
| **B** | label-map L[1,1,30,30] uint8 → Equal(L, arange)→output(bool) | ~3.6k | **~16.8** | general per-cell deterministic rule (FALLBACK — not a finish line) |
| — | multi-plane detection (bbox/beam/count/corner-dist) | 5k–40k | 13–16 | non-local; squeeze planes via uint8/small-canvas/Pad/separable |

**Discipline:** count bytes as you design. [1,1,30,30] = 900(bool/uint8)/3600(fp32);
[1,10,30,30] = 9000/36000 — NEVER materialize; route 10-ch expansion into FREE `output`.
Small working canvas + final Pad. uint8 (not int8/int16) for ORT Where/Equal. fp32 exact <2^24.

## Per-task loop (systematized — do EVERY task this way)
1. **Read rule:** `.venv/bin/python -m src.show N --gen`. Write the exact rule into the task log.
2. **Triage tier:** which is the HIGHEST tier the rule could admit? (Look for: per-cell-local→S;
   row×col-separable→A; else B; non-local→detection.) Record the target tier.
3. **Attempt top tier first.** Only fall back when you can name the blocker.
4. **Verify (authoritative):** evaluate() ok + ISOLATED fresh 200/200 (build candidate, test vs
   freshly-generated instances — fresh_pass reads disk, so test in-memory or temp-write w/o touching
   manifest). Adopt only if BEATS current real pts AND generalizes 200/200.
5. **Record in the task log REGARDLESS of outcome** (win, loss, infeasible). Append insight.
6. Main adopts via `python -m src.adopt N`; never raw pipeline (stored keep-best reintroduces bugs).

## Per-task log format: reports/tasklog/taskNNN.md  (one file per task, append-only)
Created/updated every time the task is touched. Schema in tasklog/_TEMPLATE.md. Must contain:
- rule summary, current pts/method, **target tier + why**
- **attempts table**: angle | tier | mem | params | pts | fresh | outcome
- **best achieved** + adopted? (Y/N + new pts)
- **irreducible-floor analysis**: which intermediate dominates & why it can't be removed
- **OPEN ANGLES**: reformulations NOT yet tried (the "더 다양한 각도" backlog — never empty unless Tier S reached)
- **INSIGHT**: transferable lesson (feeds future tasks; promote big ones to project memory)

## Status ledger: reports/sweep_ledger.json  (1→400, machine-readable progress)
Fields: task, arc_id, points, memory, params, method, class, sig, status(pending|done|skip|wip),
verdict. Update status+verdict per task. `reports/sweep_wave_queue.json` = prioritized build queue.

## BAIL classes (record verdict, do not grind): flood-fill/connectivity/enclosed-region,
multi-object shape-correspondence/rotate, output-size-not-input-derivable, random-pixel noise,
non-deterministic (same input→2 outputs). These cap at the memorizer ~14 floor for everyone —
even #1. They are NOT where the 7500 gap lives; the gap is pushing feasible tasks B→A→S.
