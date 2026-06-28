# Public teacher extraction — task219

- public path: `public_candidates/urad_7174_10/extracted/task219.onnx`
- live path: `/Users/minseong/project/neurogolf/networks/task219.onnx`

## Stored evaluation

- public: ok=True pts=15.162385 mem=18633 params=92
- live: ok=True pts=14.825837 mem=26133 params=84
- delta public-live: pts=+0.336549 mem=-7500 params=8

## Structural comparison

- public nodes: 252
- live nodes: 255
- public initializer elems: 92
- live initializer elems: 84

### Op delta public-live

- `Gather`: -2
- `Slice`: -1

## Mechanism extraction checklist

- [ ] Identify which full-grid intermediate disappeared or changed dtype.
- [ ] Identify whether QLinear/uint8/bool/opset routing changed.
- [ ] Identify whether final one-hot expansion moved to final `Equal`.
- [ ] Decide whether this is a semantic mechanism or exact-preserve only.
- [ ] If reusable, add/update `reports/insight_registry.yaml` and rerun the recursive queue.

## Adoption status

Not adopted. This is a teacher artifact until source and fresh gates pass.
