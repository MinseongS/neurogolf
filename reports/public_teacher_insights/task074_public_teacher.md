# Public teacher extraction — task074

- public path: `public_candidates/urad_7174_10/extracted/task074.onnx`
- live path: `/Users/minseong/project/neurogolf/networks/task074.onnx`

## Stored evaluation

- public: ok=True pts=15.889480 mem=9000 params=50
- live: ok=True pts=15.864922 mem=7462 params=1813
- delta public-live: pts=+0.024558 mem=1538 params=-1763

## Structural comparison

- public nodes: 8
- live nodes: 13
- public initializer elems: 50
- live initializer elems: 1813

### Op delta public-live

- `Concat`: -1
- `Flatten`: -1
- `Gather`: -3
- `Max`: +2
- `ReduceMax`: -3
- `Transpose`: +1

## Mechanism extraction checklist

- [ ] Identify which full-grid intermediate disappeared or changed dtype.
- [ ] Identify whether QLinear/uint8/bool/opset routing changed.
- [ ] Identify whether final one-hot expansion moved to final `Equal`.
- [ ] Decide whether this is a semantic mechanism or exact-preserve only.
- [ ] If reusable, add/update `reports/insight_registry.yaml` and rerun the recursive queue.

## Adoption status

Not adopted. This is a teacher artifact until source and fresh gates pass.
