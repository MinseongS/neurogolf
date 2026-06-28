# Public teacher extraction — task196

- public path: `public_candidates/urad_7174_10/extracted/task196.onnx`
- live path: `/Users/minseong/project/neurogolf/networks/task196.onnx`

## Stored evaluation

- public: ok=True pts=16.579758 mem=4500 params=38
- live: ok=True pts=16.531787 mem=4725 params=36
- delta public-live: pts=+0.047971 mem=-225 params=2

## Structural comparison

- public nodes: 15
- live nodes: 16
- public initializer elems: 38
- live initializer elems: 36

### Op delta public-live

- `Min`: -1

## Mechanism extraction checklist

- [ ] Identify which full-grid intermediate disappeared or changed dtype.
- [ ] Identify whether QLinear/uint8/bool/opset routing changed.
- [ ] Identify whether final one-hot expansion moved to final `Equal`.
- [ ] Decide whether this is a semantic mechanism or exact-preserve only.
- [ ] If reusable, add/update `reports/insight_registry.yaml` and rerun the recursive queue.

## Adoption status

Not adopted. This is a teacher artifact until source and fresh gates pass.
