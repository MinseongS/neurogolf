# Public teacher extraction — task201

- public path: `public_candidates/urad_7174_10/extracted/task201.onnx`
- live path: `/Users/minseong/project/neurogolf/networks/task201.onnx`

## Stored evaluation

- public: ok=True pts=16.563150 mem=4476 params=138
- live: ok=True pts=16.529060 mem=4476 params=298
- delta public-live: pts=+0.034089 mem=0 params=-160

## Structural comparison

- public nodes: 59
- live nodes: 59
- public initializer elems: 138
- live initializer elems: 298

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
