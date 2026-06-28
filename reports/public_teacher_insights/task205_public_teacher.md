# Public teacher extraction — task205

- public path: `public_candidates/urad_7174_10/extracted/task205.onnx`
- live path: `/Users/minseong/project/neurogolf/networks/task205.onnx`

## Stored evaluation

- public: ok=True pts=15.359697 mem=14734 params=638
- live: ok=True pts=15.103386 mem=19212 params=651
- delta public-live: pts=+0.256311 mem=-4478 params=-13

## Structural comparison

- public nodes: 70
- live nodes: 68
- public initializer elems: 638
- live initializer elems: 651

### Op delta public-live

- `Add`: +1
- `Cast`: +2
- `Conv`: -1
- `Equal`: +1
- `Greater`: -1
- `Mul`: +1
- `Not`: +1
- `Where`: -2

## Mechanism extraction checklist

- [ ] Identify which full-grid intermediate disappeared or changed dtype.
- [ ] Identify whether QLinear/uint8/bool/opset routing changed.
- [ ] Identify whether final one-hot expansion moved to final `Equal`.
- [ ] Decide whether this is a semantic mechanism or exact-preserve only.
- [ ] If reusable, add/update `reports/insight_registry.yaml` and rerun the recursive queue.

## Adoption status

Not adopted. This is a teacher artifact until source and fresh gates pass.
