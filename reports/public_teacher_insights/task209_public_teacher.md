# Public teacher extraction — task209

- public path: `public_candidates/urad_7174_10/extracted/task209.onnx`
- live path: `/Users/minseong/project/neurogolf/networks/task209.onnx`

## Stored evaluation

- public: ok=True pts=14.619906 mem=32027 params=185
- live: ok=True pts=14.157502 mem=50951 params=198
- delta public-live: pts=+0.462404 mem=-18924 params=-13

## Structural comparison

- public nodes: 181
- live nodes: 183
- public initializer elems: 185
- live initializer elems: 198

### Op delta public-live

- `Add`: +1
- `Clip`: +4
- `Conv`: -1
- `Max`: -4
- `Min`: -3
- `QLinearConv`: +1

## Mechanism extraction checklist

- [ ] Identify which full-grid intermediate disappeared or changed dtype.
- [ ] Identify whether QLinear/uint8/bool/opset routing changed.
- [ ] Identify whether final one-hot expansion moved to final `Equal`.
- [ ] Decide whether this is a semantic mechanism or exact-preserve only.
- [ ] If reusable, add/update `reports/insight_registry.yaml` and rerun the recursive queue.

## Adoption status

Not adopted. This is a teacher artifact until source and fresh gates pass.
