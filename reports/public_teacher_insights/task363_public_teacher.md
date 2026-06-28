# Public teacher extraction — task363

- public path: `public_candidates/urad_7174_10/extracted/task363.onnx`
- live path: `/Users/minseong/project/neurogolf/networks/task363.onnx`

## Stored evaluation

- public: ok=True pts=16.558177 mem=4339 params=298
- live: ok=True pts=16.457919 mem=4843 params=283
- delta public-live: pts=+0.100258 mem=-504 params=15

## Structural comparison

- public nodes: 47
- live nodes: 49
- public initializer elems: 298
- live initializer elems: 283

### Op delta public-live

- `Add`: +1
- `And`: -1
- `Cast`: +2
- `Conv`: +1
- `ConvTranspose`: -1
- `Equal`: +1
- `Greater`: +1
- `Max`: -1
- `Mul`: -1
- `Pad`: -1
- `ReduceSum`: -1
- `Sub`: -2

## Mechanism extraction checklist

- [ ] Identify which full-grid intermediate disappeared or changed dtype.
- [ ] Identify whether QLinear/uint8/bool/opset routing changed.
- [ ] Identify whether final one-hot expansion moved to final `Equal`.
- [ ] Decide whether this is a semantic mechanism or exact-preserve only.
- [ ] If reusable, add/update `reports/insight_registry.yaml` and rerun the recursive queue.

## Adoption status

Not adopted. This is a teacher artifact until source and fresh gates pass.
