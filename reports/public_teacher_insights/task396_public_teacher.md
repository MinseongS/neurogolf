# Public teacher extraction — task396

- public path: `public_candidates/urad_7174_10/extracted/task396.onnx`
- live path: `/Users/minseong/project/neurogolf/networks/task396.onnx`

## Stored evaluation

- public: ok=True pts=15.551194 mem=12557 params=136
- live: ok=True pts=15.516279 mem=13007 params=137
- delta public-live: pts=+0.034915 mem=-450 params=-1

## Structural comparison

- public nodes: 59
- live nodes: 61
- public initializer elems: 136
- live initializer elems: 137

### Op delta public-live

- `And`: -1
- `Greater`: -1

## Mechanism extraction checklist

- [ ] Identify which full-grid intermediate disappeared or changed dtype.
- [ ] Identify whether QLinear/uint8/bool/opset routing changed.
- [ ] Identify whether final one-hot expansion moved to final `Equal`.
- [ ] Decide whether this is a semantic mechanism or exact-preserve only.
- [ ] If reusable, add/update `reports/insight_registry.yaml` and rerun the recursive queue.

## Adoption status

Not adopted. This is a teacher artifact until source and fresh gates pass.
