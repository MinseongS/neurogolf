# Public teacher extraction — task165

- public path: `public_candidates/urad_7174_10/extracted/task165.onnx`
- live path: `/Users/minseong/project/neurogolf/networks/task165.onnx`

## Stored evaluation

- public: ok=True pts=16.275793 mem=5836 params=314
- live: ok=True pts=16.227235 mem=6144 params=312
- delta public-live: pts=+0.048558 mem=-308 params=2

## Structural comparison

- public nodes: 26
- live nodes: 26
- public initializer elems: 314
- live initializer elems: 312

### Op delta public-live

- `Conv`: -1
- `QLinearConv`: +1

## Mechanism extraction checklist

- [ ] Identify which full-grid intermediate disappeared or changed dtype.
- [ ] Identify whether QLinear/uint8/bool/opset routing changed.
- [ ] Identify whether final one-hot expansion moved to final `Equal`.
- [ ] Decide whether this is a semantic mechanism or exact-preserve only.
- [ ] If reusable, add/update `reports/insight_registry.yaml` and rerun the recursive queue.

## Adoption status

Not adopted. This is a teacher artifact until source and fresh gates pass.
