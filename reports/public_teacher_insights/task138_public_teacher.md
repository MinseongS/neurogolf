# Public teacher extraction — task138

- public path: `public_candidates/urad_7174_10/extracted/task138.onnx`
- live path: `/Users/minseong/project/neurogolf/networks/task138.onnx`

## Stored evaluation

- public: ok=True pts=15.455834 mem=13812 params=151
- live: ok=True pts=15.429750 mem=14106 params=226
- delta public-live: pts=+0.026084 mem=-294 params=-75

## Structural comparison

- public nodes: 56
- live nodes: 58
- public initializer elems: 151
- live initializer elems: 226

### Op delta public-live

- `Cast`: -3
- `Gather`: +4
- `Greater`: +1
- `Max`: -1
- `MaxPool`: -2
- `Not`: -1
- `Or`: -1
- `ReduceMax`: -1
- `Reshape`: -3
- `Slice`: -2
- `Squeeze`: +4
- `Transpose`: +2
- `Unsqueeze`: +1

## Mechanism extraction checklist

- [ ] Identify which full-grid intermediate disappeared or changed dtype.
- [ ] Identify whether QLinear/uint8/bool/opset routing changed.
- [ ] Identify whether final one-hot expansion moved to final `Equal`.
- [ ] Decide whether this is a semantic mechanism or exact-preserve only.
- [ ] If reusable, add/update `reports/insight_registry.yaml` and rerun the recursive queue.

## Adoption status

Not adopted. This is a teacher artifact until source and fresh gates pass.
