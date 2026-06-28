# Public teacher extraction — task004

- public path: `public_candidates/urad_7174_10/extracted/task004.onnx`
- live path: `/Users/minseong/project/neurogolf/networks/task004.onnx`

## Stored evaluation

- public: ok=True pts=16.406957 mem=5300 params=94
- live: ok=True pts=16.316615 mem=5812 params=92
- delta public-live: pts=+0.090343 mem=-512 params=2

## Structural comparison

- public nodes: 19
- live nodes: 21
- public initializer elems: 94
- live initializer elems: 92

### Op delta public-live

- `And`: -2
- `Cast`: -1
- `Greater`: +1
- `Min`: +1
- `Not`: -1

## Mechanism extraction checklist

- [ ] Identify which full-grid intermediate disappeared or changed dtype.
- [ ] Identify whether QLinear/uint8/bool/opset routing changed.
- [ ] Identify whether final one-hot expansion moved to final `Equal`.
- [ ] Decide whether this is a semantic mechanism or exact-preserve only.
- [ ] If reusable, add/update `reports/insight_registry.yaml` and rerun the recursive queue.

## Adoption status

Not adopted. This is a teacher artifact until source and fresh gates pass.
