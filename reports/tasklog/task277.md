# task277 — b230c067

## 2026-06-29 MaxPool-depth pruning probe

Rule: three cyan copies are placed on a 10x10 grid.  Two are the original
sprite (`idx=0`, output blue), one is a derived sprite with one column removed
(`idx=1`, output red).  The current graph classifies each cyan component by
diffusing a two-channel absolute-coordinate seed through the component with
six 3x3 `MaxPool` passes, then comparing the two propagated channels.

Current source/live:

- score `16.62690815255802`
- memory `4101`
- params `228`
- stored `266/266`

Probe: remove the last diffusion pass and use `AB5` instead of `AB6`.

| variant | stored | fresh | memory | params | points | decision |
|---|---:|---:|---:|---:|---:|---|
| 5 passes (`AB5`) | `266/266` | `2000/2000`, then rare fail | 3701 | 228 | 16.723860 | reject |
| 6 passes (`AB6`, incumbent) | `266/266` | known-good on rare case | 4101 | 228 | 16.626908 | keep |

The 5-pass candidate looked like a real +0.097 win and was briefly adopted by
the local gate, but a stronger fresh check found rare failures.  Example shape:
10x10.  The failing generated input has an original component whose geodesic
diameter needs the sixth 3x3 diffusion pass; the incumbent public/source graph
classifies it correctly.

Adopt decision: **rejected and reverted**.  Stored-only and 1k/2k fresh were not
strong enough for this pruning.  The sixth pass is a semantic requirement, not
dead slack.

Transferable negative insight: iterative component-labelling depth can look
overprovisioned on stored examples.  For 10x10 connected creature/box tasks,
fresh stress must include rare long components before pruning MaxPool depth.

## 2026-07-01 (S7 ref-golf) — K=5 refit HELD, NOT landed
A subagent proposed dropping the 6th masked-MaxPool CCL iteration (K=6→K=5),
mem 4101→3701 (+0.097). Passed bundled fail=0 and fresh 0-div at n=3000, BUT
re-gating vs main at n=5000 → candidate fail=1 (incumbent fail=0): a rare
8-connected component genuinely needs the 6th propagation, so K=5 is NOT
equivalent. Diverging re-fit → ~0 risk on private LB (-16) >> +0.097 public gain.
HELD at reports/candidates/task277_k5_refit.py; incumbent (K=6, 4101) kept.
Lesson: gate K-iteration drops at >=5000 fresh — the tail case is ~1/5000.

## S8 (2026-07-02) — matrix-sweep verdict: priced FLOOR (block-1/2 opus agents; occupancy/max-semiring reductions or sub-400B u8 banks). Do not re-attempt without a new mechanism.
