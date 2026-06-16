# Hidden gap-closers — fresh-rate scan (2026-06-17)

## Method (KEY: genverify n=40 misses these; this n=100 scan + adopt's isolated check is truth)
`for t in 1..400: fresh_pass(t,100)` over all adopted base nets. ~221s for 400.
Tasks with fresh<0.99 = base net does NOT generalize → scores ~0 on real LB despite high stored.
Building a GENERALIZING exact solver for these recovers ~full stored value on LB (gap-closing).
Proven by task31 (old net real=0.00 → custom 16.09 adopted; NOT in genverify's flagged-11).

## fresh=0.00 (score ~0 on LB; HIGH VALUE — each ~+18 LB if solvable):
- task219 — confirmed-infeasible 15.00 (WALL: information bottleneck)
- task255 — confirmed-infeasible 13.95 (WALL: connectivity)
- task220 — pending 18.2 (custom:task220 adopted but fails fresh?!) ← ATTACK
- task230 — pending 18.2 ← ATTACK
- task282 — pending 18.2 (ext:octavi6154) ← ATTACK
- task317 — pending 18.2 (gen:thbdh6332) ← ATTACK

## fresh 0.95-0.98 (small overcount ~0.3-1pt each, LOW priority):
- 157(0.95), 209(0.95), 23(0.96), 118(0.96), 2(0.97), 18(0.97),
  151(0.97 pending 18.19), 319(0.98), 332(0.98 pending 16.32), 363(0.98)

## Implication
The 61.27 stored-LB gap is NOT all structural. The 4 fresh=0.00 pending tasks (220/230/282/317)
@ stored 18.2 each are the bulk of recoverable headroom. If solvable (task31-style, not 219/255-style
walls), they translate to large LB gains. Re-run this scan after each batch of adoptions.
