# Public teacher extraction — task076

- public path: `public_candidates/urad_7174_10/extracted/task076.onnx`
- live path: `/Users/minseong/project/neurogolf/networks/task076.onnx`

## Stored evaluation

- public: ok=True pts=14.930913 mem=23296 params=306
- live: ok=True pts=14.847155 mem=25375 params=289
- delta public-live: pts=+0.083758 mem=-2079 params=17

## Structural comparison

- public nodes: 141
- live nodes: 144
- public initializer elems: 306
- live initializer elems: 289

### Op delta public-live

- `Cast`: -1
- `Or`: -2

## Mechanism extraction checklist

- [ ] Identify which full-grid intermediate disappeared or changed dtype.
- [ ] Identify whether QLinear/uint8/bool/opset routing changed.
- [ ] Identify whether final one-hot expansion moved to final `Equal`.
- [ ] Decide whether this is a semantic mechanism or exact-preserve only.
- [ ] If reusable, add/update `reports/insight_registry.yaml` and rerun the recursive queue.

## Adoption status

Not adopted. This is a teacher artifact until source and fresh gates pass.
