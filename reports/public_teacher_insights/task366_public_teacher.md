# Public teacher extraction — task366

- public path: `public_candidates/urad_7174_10/extracted/task366.onnx`
- live path: `/Users/minseong/project/neurogolf/networks/task366.onnx`

## Stored evaluation

- public: ok=True pts=14.497209 mem=35927 params=490
- live: ok=True pts=14.469345 mem=36955 params=491
- delta public-live: pts=+0.027864 mem=-1028 params=-1

## Structural comparison

- public nodes: 640
- live nodes: 669
- public initializer elems: 490
- live initializer elems: 491

### Op delta public-live

- `And`: -14
- `Clip`: -1
- `Greater`: -2
- `Less`: -8
- `Not`: -4

## Mechanism extraction checklist

- [ ] Identify which full-grid intermediate disappeared or changed dtype.
- [ ] Identify whether QLinear/uint8/bool/opset routing changed.
- [ ] Identify whether final one-hot expansion moved to final `Equal`.
- [ ] Decide whether this is a semantic mechanism or exact-preserve only.
- [ ] If reusable, add/update `reports/insight_registry.yaml` and rerun the recursive queue.

## Adoption status

Not adopted. This is a teacher artifact until source and fresh gates pass.
