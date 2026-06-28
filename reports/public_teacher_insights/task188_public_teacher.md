# Public teacher extraction — task188

- public path: `public_candidates/urad_7174_10/extracted/task188.onnx`
- live path: `/Users/minseong/project/neurogolf/networks/task188.onnx`

## Stored evaluation

- public: ok=True pts=17.920816 mem=1128 params=59
- live: ok=True pts=17.904936 mem=1146 params=60
- delta public-live: pts=+0.015880 mem=-18 params=-1

## Structural comparison

- public nodes: 53
- live nodes: 71
- public initializer elems: 59
- live initializer elems: 60

### Op delta public-live

- `And`: -7
- `Equal`: -7
- `Greater`: -1
- `Not`: -2
- `Or`: -1

## Mechanism extraction checklist

- [ ] Identify which full-grid intermediate disappeared or changed dtype.
- [ ] Identify whether QLinear/uint8/bool/opset routing changed.
- [ ] Identify whether final one-hot expansion moved to final `Equal`.
- [ ] Decide whether this is a semantic mechanism or exact-preserve only.
- [ ] If reusable, add/update `reports/insight_registry.yaml` and rerun the recursive queue.

## Adoption status

Not adopted. This is a teacher artifact until source and fresh gates pass.
