---
deployed_cost: 2503
logged_costs_match: stale-likely
migrated: 2026-07-09
---

# task019 — 10fcaaa3

## 2026-06-29 single-Conv probe

Current source score: 17.063340 @ mem 2709 params 89.

Rule: copy sparse coloured points from an HxW input into four quadrants of a
2H x 2W output and draw cyan diagonal neighbours around each copied point, with
coloured points overwriting cyan.

The tempting 20+ mechanism is a single Conv: point colour copies and cyan diagonal
stencils are both local/linear, and cyan overwrite could in principle be handled
with negative centre weights.  Ran `reports/scripts/conv_fit.py 19`; result:

- k=1 failed, channel 0 not separable on 300 fresh examples
- k=3 failed, channel 0 not separable on 300 fresh examples
- k=5 failed, channel 0 not separable on 300 fresh examples

No rewrite adopted.  The blocker is the black/background channel: the output
footprint is 2H x 2W, while H and W are data-dependent and must be recovered from
the input rectangle.  Colour/cyan are local, but the footprint mask is not a fixed
translation-invariant stencil.

## S27 (2026-07-07) — dead initializer prune micro-overlay ADOPTED

Candidate: `reports/candidates/task019/task019_prune_dead_constants.onnx`, produced by
`reports/candidates/prune_dead_constants_active_probe.py`.

Removed unused initializers `safe_name_10` and `safe_name_11`.  Bundled gate remains
fail=0.  Cost: 2507 -> 2505 (memory 2421 unchanged, params 86 -> 84), points
17.173158 -> 17.173956.  Active overlay updated in
`submission/overfit_nets/task019.onnx`; backup at
`reports/candidates/task019/task019_pre_prune_dead_constants.onnx`.

## 2026-07-08 — public lost-Pad fingerprint probe, not adopted

Tool/date: after enhancing `reports/scripts/public_win_autopsy.py` with exact lost-tensor
fingerprints, the big public-jump autopsy found that task011's win removed
`Pad:2:1x1x30x30:900B`.  Current task019 has the same final-tail fingerprint:
`Pad(label)->Equal(output)`.

Probe: `reports/candidates/task019/task019_equal_before_bool_pad.onnx` moved the final Equal before
the Pad, attempting `Equal(label, channel_codes)->Pad(bool output)` so the 900B label Pad disappears.
Result: ONNXRuntime rejects `Pad(tensor(bool))` as invalid.  The u8 workaround would require a
counted `[1,10,30,30]` carrier or post-Pad Cast and is strictly worse than the current 900B fp16
label Pad.  No overlay adopted.

Reopen trigger: a legal final-output Pad path that accepts bool tensors, or a way to route the padded
bool tensor directly into graph `output` without materializing a counted 10-channel carrier.
Falsification history: this is a public-autopsy exact-fingerprint candidate, not a broad "pad
optimization dry" verdict; public-tail dry states have been repeatedly falsified, so this remains
dormant only for task019's current tail form.

## ADOPTED 20260715T110320Z
- cost: 2503 -> 1952 (points 17.4234)
- source: candidates/task019/dynamic_qconv.onnx
- note: dynamic QLinearConv output folding: remove final idx/Pad carrier; bundled+fresh exact
