# Public teacher extraction — task255

- public path: `public_candidates/urad_7174_10/extracted/task255.onnx`
- live path: `/Users/minseong/project/neurogolf/networks/task255.onnx`

## Stored evaluation

- public: ok=True pts=14.940150 mem=23123 params=262
- live: ok=True pts=14.870932 mem=24746 params=315
- delta public-live: pts=+0.069218 mem=-1623 params=-53

## Structural comparison

- public nodes: 148
- live nodes: 173
- public initializer elems: 262
- live initializer elems: 315

### Op delta public-live

- `And`: -11
- `Conv`: -12
- `Greater`: -1
- `Not`: -4
- `Or`: -4
- `QLinearConv`: +12
- `ReduceMax`: -1
- `Slice`: -4

## Mechanism extraction checklist

- [ ] Identify which full-grid intermediate disappeared or changed dtype.
- [ ] Identify whether QLinear/uint8/bool/opset routing changed.
- [ ] Identify whether final one-hot expansion moved to final `Equal`.
- [ ] Decide whether this is a semantic mechanism or exact-preserve only.
- [ ] If reusable, add/update `reports/insight_registry.yaml` and rerun the recursive queue.

## Adoption status

Not adopted. This is a teacher artifact until source and fresh gates pass.
