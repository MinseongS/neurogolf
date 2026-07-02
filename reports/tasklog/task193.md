# task193 — 7f4411dc

**Rule (cristianoc oracle):** keep cell colour v iff ≥2 of its 4 orthogonal neighbours == v, else 0.
Pure local rule; channel 0 = union of demotions so a single Conv can't express it (conv_fit fails).

## S5 win — uint8 shifts + boolean ≥2 counting (LANDED +0.81)
**Before (custom all-fp32):** mem 40500, params 57, total 40557, pts 14.39. Carried 4 fp32 shift-Convs
(3600B each) + 4 fp32 Casts of the match bools (3600B each) + fp32 Sum (3600B) = ~32400B of fp32 waste.
**The lever:** only `idx` (Conv channel-collapse colour read) must be fp32 (3600B floor). Everything
downstream → uint8/bool:
- shifts via **QLinearConv** on uint8 idx (scale 1, zp 0 = exact integer shift): 4×900B (was 4×3600B).
- neighbour matches = Equal(iu, shift) → bool (no fp32 cast).
- **≥2-of-4 counting in BOOLEAN logic** (opset-11 integer arith needs 4-byte types, so NO uint8
  Add/Sum): `keep = a&(b|c|d) | b&(c|d) | (c&d)` — 7 bool planes (900B) vs 4 fp32 casts + fp32 Sum (18000B).
**After: mem 18000, params 62, total 18062, pts 15.198 (+0.81).** evaluate fail 0;
`fresh_verify 193 <cand> 1500` → candidate fail 0, candidate!=incumbent 0; +3000 random in-domain 0-div.

⭐ TRANSFERABLE — **neighbour-count / local-stencil lever:** ops of the form "count how many of K
neighbours satisfy P, threshold ≥t" rendered with fp32 shift-Convs + fp32 Cast+Sum are bloated. (1)
collapse to a uint8 colour index once, shift via QLinearConv (uint8, exact), (2) compare with Equal→bool,
(3) do the threshold count with And/Or boolean logic, NOT Cast+Sum (opset-11 int arith = 4-byte min;
And/Or stay 1-byte bool). Found via the cristianoc single-pass list (task193 was the lone >1187 outlier).

## Crop optimization (S4, via cristianoc oracle cross-ref, 2026-07-01)
cristianoc/arc-code-golf-solutions task193 = single-pass neighbour-majority
(`v if >1 of 4 ortho-neighbours==v else 0`), confirming the rule. The arc-gen
generator caps grids at **20×20** (verified 3000 fresh, max dim 20), so the whole
neighbour pipeline can run at the native 20×20 instead of 30×30 (0.44× area).
- idx_f Conv (3600B, the fp32 channel-collapse floor) → Cast uint8 → **Slice to 20×20**;
  one Pad(0) + 4 Slice neighbour shifts (uint8); 4 Equal (bool); Cast fp16 + Sum count;
  Pad match→30×30 with value 9 (off-region force-kept → routes zero input → stays 0);
  Greater→keep; Where(keep, input, bg) → FREE output.
- **mem 18000 → 15284, pts 15.198 → 15.361 (+0.163)** over the S5 uint8/bool build
  (which itself fixed the 40500 fp32 incumbent). fail=0 on all 266 bundled;
  **2000–2500/2500 fresh arc-gen**, identical to the incumbent on every instance.
- Grader-safe: uint8 Equal ships in 84 scoring nets, fp16 Sum in 18; opset 11.

## S8 (2026-07-02) — matrix-sweep verdict: priced FLOOR (block-4 opus agent). Do not re-attempt without a new mechanism.

## S9 (2026-07-03) — kojimar 7184.85 teacher REJECTED (bundled-overfit, fresh 2.36%)
Teacher = ONE Conv(10,10,3,3)+bias whose output IS the graph output → mem=0, params=910,
bundled fail=0 (would be +2.693 public-LB). Fresh 2500: teacher fail 59 (2.36%) vs
incumbent 0. Trained-float weights approximate "keep v iff ≥2 ortho neighbours own-colour"
— channel 0 (unknown-centre aggregate) is NOT linearly separable in one conv; exact repair
requires counted nodes (rebuilds the incumbent). Inherent mechanism floor, not tunable.
KEEP incumbent (final scoring = private LB; fresh-gate moat).
⭐ TRANSFERABLE: kojimar mem=0 single-Conv nets on nonlinear-local-rule tasks = expect
few-% fresh floor; always fresh-gate before adopting. Works ONLY for genuinely per-channel
linearly-separable rules on one-hot input.

## S9 (2026-07-03) — single-tap 11×11 valid-Conv crop (+0.136) ADOPTED
Entry read 1×1 Conv 30×30 (3600) + Cast u8 30×30 (900) + Slice → ONE 11×11 valid Conv,
arange tap at (0,0), emits idx_f 20×20 direct. mem 12884→9984, params 558→1746 (+1210
kernel −12 dead inits + removed idx30). Bit-identical: 2500+600 uncached 0/0/0, no TopK.
FLOORS: sparse initializer REJECTED by shape_inference/check_model (harness runs both);
two-conv 11×1/1×11 split worse (re-adds 2400B intermediate); keep/keep_w 30×30 bool
mandatory for free output routing. Backup task193_pre_s9.onnx.
