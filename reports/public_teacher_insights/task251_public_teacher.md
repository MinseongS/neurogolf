# Public teacher extraction — task251

- public path: `public_candidates/urad_7174_10/extracted/task251.onnx`
- live path: `/Users/minseong/project/neurogolf/networks/task251.onnx`

## Stored evaluation

- public: ok=True pts=16.498936 mem=4752 params=168
- live: ok=True pts=16.362361 mem=5472 params=168
- delta public-live: pts=+0.136576 mem=-720 params=0

## Structural comparison

- public nodes: 26
- live nodes: 31
- public initializer elems: 168
- live initializer elems: 168

### Op delta public-live

- `Cast`: -3
- `Mul`: -1
- `Sub`: -1

## Mechanism extraction checklist

- [ ] Identify which full-grid intermediate disappeared or changed dtype.
- [ ] Identify whether QLinear/uint8/bool/opset routing changed.
- [ ] Identify whether final one-hot expansion moved to final `Equal`.
- [ ] Decide whether this is a semantic mechanism or exact-preserve only.
- [ ] If reusable, add/update `reports/insight_registry.yaml` and rerun the recursive queue.

## Adoption status

Not adopted. This is a teacher artifact until source and fresh gates pass.
