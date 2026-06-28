# Public teacher extraction — task349

- public path: `public_candidates/urad_7174_10/extracted/task349.onnx`
- live path: `/Users/minseong/project/neurogolf/networks/task349.onnx`

## Stored evaluation

- public: ok=True pts=14.826867 mem=26100 params=90
- live: ok=True pts=14.428786 mem=37800 params=1196
- delta public-live: pts=+0.398081 mem=-11700 params=-1106

## Structural comparison

- public nodes: 24
- live nodes: 16
- public initializer elems: 90
- live initializer elems: 1196

### Op delta public-live

- `Cast`: +2
- `Conv`: -1
- `ConvTranspose`: -1
- `CumSum`: -1
- `Greater`: -2
- `Max`: +1
- `MaxPool`: +6
- `QLinearConv`: +5
- `ReduceMax`: +1
- `ReduceSum`: -1
- `Relu`: -1

## Mechanism extraction checklist

- [ ] Identify which full-grid intermediate disappeared or changed dtype.
- [ ] Identify whether QLinear/uint8/bool/opset routing changed.
- [ ] Identify whether final one-hot expansion moved to final `Equal`.
- [ ] Decide whether this is a semantic mechanism or exact-preserve only.
- [ ] If reusable, add/update `reports/insight_registry.yaml` and rerun the recursive queue.

## Adoption status

Not adopted. This is a teacher artifact until source and fresh gates pass.
