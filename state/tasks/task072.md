---
deployed_cost: 421
logged_costs_match: stale-likely
migrated: 2026-07-09
---

# task072 — 3428a4f5

**Rule:** Input is a 13×5 grid: top panel rows 0–5, yellow(4) separator at row 6, bottom panel rows 7–12; each panel holds red(2) pixels on black(0). Output is 6×5: cell (r,c)=green(3) iff EXACTLY ONE of {top red at (r,c), bottom red at (r,c)} is set, else black(0). i.e. green = XOR of the two panels' red masks.
**Current:** 17.82 pts, ext:kojimar6275, mem 1260, params 49
**Target tier:** A (closed-form XOR on the tiny 6×5 active region; no 30×30 plane, no detection).

## Attempts
| # | angle | tier | mem | params | pts | fresh | outcome |
|---|---|---|---|---|---|---|---|
| 1 | Equal/Not XOR on 6×5 red slices, uint8 10-ch stack, Pad→output | A | 660 | 54 | 18.43 | 200/200 | adopt |

## Best achieved
18.43 @ mem 660 params 54 — adopted? Y. Beats prior 17.82? Y (+0.61).

## Irreducible-floor analysis
Dominant intermediate is the [1,10,6,5] uint8 channel-stack = 300B — the irreducible
10-channel expansion on the active 6×5 region (output is the FREE Pad result, so the
Pad input stack is the only 10-ch tensor). Two fp32 red slices cost 120B each (Slice
preserves the fp32 input dtype). The public net did the identical structure but in fp16,
making the stack 600B and each small plane 60B (mem 1260). Switching the whole post-XOR
pipeline to uint8 halves the stack to 300B and the masks to 30B. Cannot go below ~300B
for the stack without routing the 10-ch expansion into the free output, which a single
Pad can't do (Pad fills one constant; ch0=1/ch3=1 need two distinct nonzero channels).

## OPEN ANGLES (re-attack backlog)
- Drop one fp32 red slice (240B→120B): both panels share the same red channel/cols, so
  a single Slice of rows 0:13 then split — but ORT Split adds two planes, no net win.
- Concat→Pad could fuse if a single op placed green at channel 3 AND bg at channel 0;
  none exists (would need a 2-hot scatter). Stack stays the floor.

## BUILD 2026-07-11 (ci_triage builder) — conv-fold win, NOT adopted
Deployed re-measured = Slice+Slice+Equal+Where+Pad, cost 421 (mem 390 = two fp32 red slices 120+120
+ Equal-bool 30 + Where-uint8 120), 18.957 pts. The two fp32 slices (240 B) are the mem floor of the
slice route. Replaced BOTH slices with ONE negative-pad Conv W[1,10,8,1] (taps ch2@kh0 + ch2@kh7)
producing s=TR+BR in {0,1,2} at [1,1,6,5]=120 B; XOR-green = Equal(s,1). Rest unchanged (Where→Pad).
- candidate: `candidates/task072/conv_xor.onnx`
- gate PASS: mem 390→270, params 31→97 (W adds 80), **cost 421→367, pts 18.957→19.095 (+0.137)**.
- generator (3428a4f5) validate: 6/6. Fresh A/B 2500 draws: inc_fail=0, cand_fail=0, cand!=inc=0
  (behaviorally identical on fresh data). MAIN-eligible. Adoption left to adopter.
- ⭐ transferable: for a two-panel XOR/overlay, fold BOTH panel reads into one negative-pad Conv summing
  the two source rows (kernel height = panel-offset+1, single tap per source, crop to active region via
  negative pads) — converts 2×fp32-slice (2·120 B) into 1 conv plane (120 B) at the price of the W
  element count (Cin=10 forces 80). Net cheaper whenever the combine is a linear pre-step to a threshold.

## INSIGHT (transferable)
⭐ run_network thresholds the model output as `(out>0).astype(float)` before
array_equal, so the OUTPUT dtype is irrelevant to correctness — any pure {0,1}
one-hot / mask task can run its ENTIRE post-reduction pipeline in uint8 (Equal/Not/
Cast/Concat/Pad all accept uint8 under ORT_DISABLE_ALL) and declare the output uint8,
halving every working plane vs the fp16 trick. Here it took a structurally-identical
public net from 1260→660 mem (+0.61) with zero algorithmic change — pure dtype.
⭐ XOR of two binary masks = `Not(Equal(a,b))`; the `Equal` result simultaneously serves
as the bg(ch0) mask (1 where neither/both → not green), so no extra `1-green` op.

## ADOPTED 20260711T140351Z
- cost: 421 -> 367 (points 19.0946)
- source: candidates/task072/conv_xor.onnx
- note: two-panel XOR fold: both fp32 red-channel slices folded into one negative-pad Conv W[1,10,8,1] (taps ch2@kh0+kh7), s=TR+BR on [1,1,6,5]; XOR=Equal(s,1); fresh A/B 2500 cand=inc identical

## ADOPTED 20260712T140144Z
- cost: 367 -> 180 (points 19.8070)
- source: /Users/minseong/project/neurogolf/dumps/archive_extract/submission7300+/task072.onnx
- note: archive.zip submission7300+ net; fresh 2000/0 fail; mechanism-graft
