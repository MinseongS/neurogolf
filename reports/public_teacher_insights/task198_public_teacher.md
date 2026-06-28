# Public teacher extraction — task198

- public path: `public_candidates/urad_7174_10/extracted/task198.onnx`
- live path: `/Users/minseong/project/neurogolf/networks/task198.onnx`

## Stored evaluation

- public: ok=True pts=15.484236 mem=13434 params=138
- live: ok=True pts=15.484236 mem=13434 params=138
- delta public-live: pts=+0.000000 mem=0 params=0

## Structural comparison

- public nodes: 82
- live nodes: 82
- public initializer elems: 138
- live initializer elems: 138

### Op delta public-live

- no op-count delta

## Mechanism extraction checklist

- [ ] Identify which full-grid intermediate disappeared or changed dtype.
- [ ] Identify whether QLinear/uint8/bool/opset routing changed.
- [ ] Identify whether final one-hot expansion moved to final `Equal`.
- [ ] Decide whether this is a semantic mechanism or exact-preserve only.
- [ ] If reusable, add/update `reports/insight_registry.yaml` and rerun the recursive queue.

## Adoption status

Not adopted. This is a teacher artifact until source and fresh gates pass.
