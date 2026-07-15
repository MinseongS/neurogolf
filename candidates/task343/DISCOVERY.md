# task343 DISCOVERY — live cost224

## Current deployment

- Live artifact: `submission/overfit_nets/task343.onnx`
- Cost: **224 = memory160 + params64**
- Score: **19.5883539481**
- Bundled: **266/266**, fail0
- Fresh deployed-vs-candidate A/B: **1500/1500**, incumbent fail0, candidate fail0,
  divergence0, generator timeout/error0
- SHA256: `48f92513614d7445c362ff7164366dfe21d6d76db57237103a8c357da8f46d4a`
- Exact source: `src/custom/task343.py`; source rebuild is byte-identical to deployed.

This task-only run started at cost683 and reached **683 -> 252 -> 248 -> 224**. The
adopted gain is **+1.1148488077** from 683 and **+0.1177830357** from the immediately
preceding cost252 deployment. Including the older stale cost756 artifact, the full
physical chain is **756 -> 224**, **+1.2163953243**.

## Final mechanism

The output remains `output[..., c] = input[..., c mod period]`, with period in
`{3,4,6,8}` and padded output columns routed to the guaranteed-zero input column29.

1. Six negative-pad Conv scalar probes read exact three-cell vertical signatures at
   columns 0, 1, 4, 5, 8, and 11 directly from the FREE input. They share one
   `[1,10,3,1]` fp32 kernel encoding `100*top + 10*middle + bottom`.
2. Visibility comes directly from float-to-bool Cast of the column8/11 probes. This
   removes the zero initializer and reverses the corresponding Where branches.
3. For visible width at most8, period4 is selected iff both `col4==col0` and
   `col5==col1`; otherwise period3 is selected. For a generated n=4 stripe both
   equalities are exact. For n=3, both can hold only when the three repeated column
   signatures collapse to the same effective pattern, where choosing 3 or4 produces
   the same tiled output. This removes the former 31B p3 comparison subsystem.
4. For the middle branch, the exact p4/p6 test remains `col4==col0 AND col8==col0`;
   visible column11 selects period8. Singleton uint8 periods flow through Where nodes
   and one attribute-axes ReduceMax before a single int32 Cast.
5. `final_mod_inputs = [0..14] + [-1]*15`; `Mod(fmod=1)` produces the 30 int32 Gather
   indices and terminal `Gather(input, final_cols, axis=3)` routes the FREE input to
   the graph output.

## Evidence and artifacts

- Exact predicate-fold control: `build_predicate_fold.py` -> `predicate_fold.onnx`,
  official gate PASS at cost248 (memory180, params68), bundled266/266.
- Final: `build_joint_period_min.py` -> `joint_period_min.onnx`, official gate PASS at
  cost224 (memory160, params64), bundled266/266.
- Fresh diagnostic: 1500 requested/runs, both models fail0, divergence0, no generator
  error or timeout.
- Final sequence: official gate PASS -> fresh1500 exact -> official gate PASS ->
  `ng adopt`, timestamp `20260715T145956Z`.
- Exact-source rebuild: cost224/fail0 and deployed/source SHA both
  `48f92513614d7445c362ff7164366dfe21d6d76db57237103a8c357da8f46d4a`.
- Focused contracts: `test_negative_pad_color.py`, `test_exact_signature.py`,
  `test_exact_signature_min.py`, `test_predicate_fold.py`,
  `test_bottom_signature.py`, and `test_joint_period_min.py`.

## Bounded negatives

- A bottom-row-only signature priced exactly228 but failed five bundled period6 cases
  (261/266): their bottom colours were constant, so the p4 predicate falsely selected4.
- Fixed-row and mixed-row probes, single/two equality relations, and two-scalar p3
  subsets did not meet bundled+fresh exactness at the price gate.
- Sparse initializers are structurally invalid for this pinned graph: ONNX 1.21 full
  shape inference treats sparse Conv/Mod operands as unsupported sparse tensor types.
  The failed sparse route was removed and must not be repeated under the current pins.
- Wider runtime signature-basis materialization remains more expensive than the saved
  parameters.

## Residual price gate

The next +0.1 requires **cost<=202**, saving at least22 from cost224. Current counted
inventory is dominated by `final_cols` int32[30] at 120B memory, six scalar fp32 Conv
outputs at 24B memory, predicate/period intermediates at 16B memory, the 30-value
signature kernel, and the 30-value final index seed.

Reopen only with a concrete joint deletion of at least22 cost: a legal narrower/direct
terminal routing primitive, a materialization-free signature-kernel factor coupled to
predicate deletion, or a new exact period predicate using fewer than six probes. This is
a dated price gate, not a floor. Do not repeat cost684 recovery, manual manifest edits,
wider runtime bases, the bottom-row collision, or sparse Conv/Mod initializers.

## Next-session first commands

```bash
PYTHONPATH=. uv run pytest -q \
  candidates/task343/test_negative_pad_color.py \
  candidates/task343/test_exact_signature.py \
  candidates/task343/test_exact_signature_min.py \
  candidates/task343/test_predicate_fold.py \
  candidates/task343/test_bottom_signature.py \
  candidates/task343/test_joint_period_min.py
uv run ng score 343
```
