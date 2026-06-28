# Public teacher extraction — task286

- public path: `public_candidates/urad_7174_10/extracted/task286.onnx`
- live path: `/Users/minseong/project/neurogolf/networks/task286.onnx`

## Stored evaluation

- public: ok=True pts=14.241821 mem=46272 params=741
- live: ok=True pts=14.140713 mem=51276 params=739
- delta public-live: pts=+0.101108 mem=-5004 params=2

## Structural comparison

- public nodes: 3036
- live nodes: 3038
- public initializer elems: 741
- live initializer elems: 739

### Op delta public-live

- `ArgMax`: -1
- `BitwiseAnd`: -1
- `Cast`: -1
- `QLinearConv`: +1

## Mechanism extraction checklist

- [ ] Identify which full-grid intermediate disappeared or changed dtype.
- [ ] Identify whether QLinear/uint8/bool/opset routing changed.
- [ ] Identify whether final one-hot expansion moved to final `Equal`.
- [ ] Decide whether this is a semantic mechanism or exact-preserve only.
- [ ] If reusable, add/update `reports/insight_registry.yaml` and rerun the recursive queue.

## Adoption status

Not adopted. This is a teacher artifact until source and fresh gates pass.
