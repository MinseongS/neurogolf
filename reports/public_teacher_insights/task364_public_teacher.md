# Public teacher extraction — task364

- public path: `public_candidates/urad_7174_10/extracted/task364.onnx`
- live path: `/Users/minseong/project/neurogolf/networks/task364.onnx`

## Stored evaluation

- public: ok=True pts=14.900493 mem=24220 params=111
- live: ok=True pts=14.797371 mem=26860 params=114
- delta public-live: pts=+0.103122 mem=-2640 params=-3

## Structural comparison

- public nodes: 43
- live nodes: 34
- public initializer elems: 111
- live initializer elems: 114

### Op delta public-live

- `Concat`: -1
- `MaxPool`: +7
- `Mul`: +5
- `Slice`: -2

## Mechanism extraction checklist

- [ ] Identify which full-grid intermediate disappeared or changed dtype.
- [ ] Identify whether QLinear/uint8/bool/opset routing changed.
- [ ] Identify whether final one-hot expansion moved to final `Equal`.
- [ ] Decide whether this is a semantic mechanism or exact-preserve only.
- [ ] If reusable, add/update `reports/insight_registry.yaml` and rerun the recursive queue.

## Adoption status

Not adopted. This is a teacher artifact until source and fresh gates pass.
