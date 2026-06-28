# Public teacher extraction — task218

- public path: `public_candidates/urad_7174_10/extracted/task218.onnx`
- live path: `/Users/minseong/project/neurogolf/networks/task218.onnx`

## Stored evaluation

- public: ok=True pts=18.504734 mem=558 params=104
- live: ok=True pts=18.483807 mem=572 params=104
- delta public-live: pts=+0.020928 mem=-14 params=0

## Structural comparison

- public nodes: 22
- live nodes: 24
- public initializer elems: 104
- live initializer elems: 104

### Op delta public-live

- `Cast`: +2
- `Equal`: -2
- `Not`: -2

## Mechanism extraction checklist

- [ ] Identify which full-grid intermediate disappeared or changed dtype.
- [ ] Identify whether QLinear/uint8/bool/opset routing changed.
- [ ] Identify whether final one-hot expansion moved to final `Equal`.
- [ ] Decide whether this is a semantic mechanism or exact-preserve only.
- [ ] If reusable, add/update `reports/insight_registry.yaml` and rerun the recursive queue.

## Adoption status

Not adopted. This is a teacher artifact until source and fresh gates pass.
