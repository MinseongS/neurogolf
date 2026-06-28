# Public teacher extraction — task066

- public path: `public_candidates/urad_7174_10/extracted/task066.onnx`
- live path: `/Users/minseong/project/neurogolf/networks/task066.onnx`

## Stored evaluation

- public: ok=True pts=15.179079 mem=17763 params=652
- live: ok=True pts=15.108332 mem=19113 params=652
- delta public-live: pts=+0.070747 mem=-1350 params=0

## Structural comparison

- public nodes: 234
- live nodes: 301
- public initializer elems: 652
- live initializer elems: 652

### Op delta public-live

- `Add`: -1
- `And`: -23
- `Cast`: -10
- `Einsum`: +1
- `Equal`: -9
- `GatherElements`: -3
- `Greater`: -3
- `Less`: -1
- `MatMul`: -1
- `Not`: -1
- `Or`: -3
- `ReduceMax`: -1
- `ReduceMin`: -2
- `Squeeze`: -1
- `Sub`: -5
- `Transpose`: -1
- `Unsqueeze`: -1
- `Where`: -2

## Mechanism extraction checklist

- [ ] Identify which full-grid intermediate disappeared or changed dtype.
- [ ] Identify whether QLinear/uint8/bool/opset routing changed.
- [ ] Identify whether final one-hot expansion moved to final `Equal`.
- [ ] Decide whether this is a semantic mechanism or exact-preserve only.
- [ ] If reusable, add/update `reports/insight_registry.yaml` and rerun the recursive queue.

## Adoption status

Not adopted. This is a teacher artifact until source and fresh gates pass.
