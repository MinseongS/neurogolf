# Public teacher extraction — task157

- public path: `public_candidates/urad_7174_10/extracted/task157.onnx`
- live path: `/Users/minseong/project/neurogolf/networks/task157.onnx`

## Stored evaluation

- public: ok=True pts=15.971901 mem=7983 params=351
- live: ok=True pts=15.550170 mem=12118 params=588
- delta public-live: pts=+0.421731 mem=-4135 params=-237

## Structural comparison

- public nodes: 496
- live nodes: 1255
- public initializer elems: 351
- live initializer elems: 588

### Op delta public-live

- `Add`: -14
- `And`: -322
- `ArgMax`: -14
- `Cast`: -37
- `Concat`: -25
- `Div`: -10
- `Equal`: -34
- `Gather`: -30
- `Greater`: -4
- `Less`: -7
- `Mod`: -1
- `Mul`: -9
- `Not`: -15
- `Or`: -156
- `ReduceSum`: -16
- `Reshape`: -23
- `Split`: -9
- `Sub`: -6
- `Transpose`: -7
- `Unsqueeze`: -9
- `Where`: -11

## Mechanism extraction checklist

- [ ] Identify which full-grid intermediate disappeared or changed dtype.
- [ ] Identify whether QLinear/uint8/bool/opset routing changed.
- [ ] Identify whether final one-hot expansion moved to final `Equal`.
- [ ] Decide whether this is a semantic mechanism or exact-preserve only.
- [ ] If reusable, add/update `reports/insight_registry.yaml` and rerun the recursive queue.

## Adoption status

Not adopted. This is a teacher artifact until source and fresh gates pass.
