# Public teacher extraction — task161

- public path: `public_candidates/urad_7174_10/extracted/task161.onnx`
- live path: `/Users/minseong/project/neurogolf/networks/task161.onnx`

## Stored evaluation

- public: ok=True pts=16.682234 mem=4055 params=41
- live: ok=True pts=16.682234 mem=4055 params=41
- delta public-live: pts=+0.000000 mem=0 params=0

## Structural comparison

- public nodes: 30
- live nodes: 30
- public initializer elems: 41
- live initializer elems: 41

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
