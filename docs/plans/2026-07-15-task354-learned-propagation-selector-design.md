# task354 learned propagation selector design

## Scope and success criteria

Optimize only task354 from the deployed cost-1268 graph. The default objective is the
8000-overfit gate: bundled failures must remain zero and the candidate must be cheaper than the
deployed graph. Fresh generation is diagnostic only. Every deployment change must use
`ng gate` followed by `ng adopt`.

The terminal division renderer is retained initially because it already exhaustively separates
semantic labels 0 through 9 and keeps every right/bottom padding logit at zero. The main target is
the 240-byte `x0/mp0/x1/mp1/x2/mp2/x3/Fb` horizontal propagation chain.

## Approved primary design: packed-notch single selector

The existing producer supplies:

- `Uraw[1,1,3,10]`: 255 on gray support and 0 elsewhere.
- `x0 = Min(M,Uraw)`: the encoded header colour only where a header column intersects support.

Construct one packed lane tensor:

```text
P = Uraw - x0
```

Background remains 0, ordinary support remains 255, and the seed position becomes the notch
`255-colour_code`. Search a shared one-dimensional learned integer selector that consumes `P`
and emits the exact `Fb[1,1,3,10]` code required by the incumbent. The first grammar is one
QLinearConv with odd horizontal kernels up to the full padded lane width and INT8 weights plus an
INT32 bias. Search and verification operate on the exact tensors extracted from all 266 bundled
examples, not on reconstructed ARC labels.

If it fits, the new propagation storage is `x0 + P + Fb` (90 bytes) instead of eight 30-byte
planes (240 bytes). A kernel and quantization overhead near 20--30 parameters should produce an
estimated cost around 1138--1148, enough for approximately +0.1 if the lower end is attained.

## Fallback designs

1. A two-stage selector uses a small two-channel latent QLinearConv followed by a 1x1 quantized
   decoder. It has higher fit capacity but adds a 60-byte latent carrier and is accepted only if
   its measured cost still beats 1268.
2. A three-band ranker predicts reachability from each header band separately, then performs an
   exact winner/colour selection. It is the highest-capacity and highest-cost fallback, so it is
   attempted only after the packed single- and two-stage grammars are exhausted.

No FLOAT full-input conversion, sparse Conv initializer, previously tested short MaxPool schedule,
or scalar spatial terminal renderer is repeated without a new fact.

## Search, data flow, and failure handling

The test harness extracts `Uraw`, `x0`, and incumbent `Fb` from the deployed model for every
bundled example. It first decides exact feasibility in real-valued or integer accumulator space,
then integer-repairs/quantizes any feasible solution and executes the resulting ONNX graph with
the pinned ONNX Runtime. Saturation and zero-point arithmetic are tested in runtime rather than
assumed.

A candidate is rejected immediately on any of these conditions:

- one bundled `Fb` cell differs from the incumbent target;
- final raw scorer output differs on any bundled example;
- any semantic label 0..9 or implicit padded cell violates the terminal truth table;
- strict ONNX inference or pinned ORT execution fails;
- measured cost is not below 1268.

Negative results are recorded only in the four-field `state/levers.yaml` ledger, with a bounded
grammar and a concrete reopen condition. They are not promoted to a task floor.

## Verification and adoption

Implementation follows test-first gates:

1. RED test for the candidate file, exact `Fb` reproduction, raw-output A/B, and cost reduction.
2. Build the smallest feasible selector and turn the focused test GREEN.
3. Re-run the existing complete division label/padding exhaustive test.
4. Run the full task354 focused suite, checker, strict shape inference, and isolated score.
5. Run `ng gate`; only on bundled fail=0 and lower cost run `ng adopt`.
6. Rebuild an exact source-owned `src/custom/task354.py`, verify its cost/fail result, and update
   task354 state truth sources without appending session history to `state/STATE.md`.

