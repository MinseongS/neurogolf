# Public teacher extraction — task005

- public path: `public_candidates/urad_7174_10/extracted/task005.onnx`
- live path: `/Users/minseong/project/neurogolf/networks/task005.onnx`

## Stored evaluation

- public: ok=True pts=16.163335 mem=6392 params=490
- live: ok=True pts=16.132150 mem=6610 params=490
- delta public-live: pts=+0.031185 mem=-218 params=0

## Structural comparison

- public nodes: 181
- live nodes: 211
- public initializer elems: 490
- live initializer elems: 490

### Op delta public-live

- `Max`: -9
- `Slice`: -21

## Mechanism extraction checklist

- [ ] Identify which full-grid intermediate disappeared or changed dtype.
- [ ] Identify whether QLinear/uint8/bool/opset routing changed.
- [ ] Identify whether final one-hot expansion moved to final `Equal`.
- [ ] Decide whether this is a semantic mechanism or exact-preserve only.
- [ ] If reusable, add/update `reports/insight_registry.yaml` and rerun the recursive queue.

## Adoption status

Not adopted. This is a teacher artifact until source and fresh gates pass.
