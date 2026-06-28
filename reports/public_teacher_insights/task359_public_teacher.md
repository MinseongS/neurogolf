# Public teacher extraction — task359

- public path: `public_candidates/urad_7174_10/extracted/task359.onnx`
- live path: `/Users/minseong/project/neurogolf/networks/task359.onnx`

## Stored evaluation

- public: ok=True pts=16.595528 mem=4452 params=15
- live: ok=True pts=16.446089 mem=5171 params=16
- delta public-live: pts=+0.149438 mem=-719 params=-1

## Structural comparison

- public nodes: 28
- live nodes: 22
- public initializer elems: 15
- live initializer elems: 16

### Op delta public-live

- `Add`: +3
- `Cast`: +2
- `GreaterOrEqual`: -1
- `Mul`: +4
- `Not`: +1
- `Sub`: +2
- `Where`: -5

## Mechanism extraction checklist

- [ ] Identify which full-grid intermediate disappeared or changed dtype.
- [ ] Identify whether QLinear/uint8/bool/opset routing changed.
- [ ] Identify whether final one-hot expansion moved to final `Equal`.
- [ ] Decide whether this is a semantic mechanism or exact-preserve only.
- [ ] If reusable, add/update `reports/insight_registry.yaml` and rerun the recursive queue.

## Adoption status

Not adopted. This is a teacher artifact until source and fresh gates pass.
