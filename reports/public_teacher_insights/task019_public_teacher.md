# Public teacher extraction — task019

- public path: `public_candidates/urad_7174_10/extracted/task019.onnx`
- live path: `/Users/minseong/project/neurogolf/networks/task019.onnx`

## Stored evaluation

- public: ok=True pts=17.063340 mem=2709 params=89
- live: ok=True pts=16.967640 mem=2997 params=82
- delta public-live: pts=+0.095700 mem=-288 params=7

## Structural comparison

- public nodes: 32
- live nodes: 32
- public initializer elems: 89
- live initializer elems: 82

### Op delta public-live

- `Cast`: -2
- `Conv`: -1
- `Greater`: +2
- `QLinearConv`: +1

## Mechanism extraction checklist

- [ ] Identify which full-grid intermediate disappeared or changed dtype.
- [ ] Identify whether QLinear/uint8/bool/opset routing changed.
- [ ] Identify whether final one-hot expansion moved to final `Equal`.
- [ ] Decide whether this is a semantic mechanism or exact-preserve only.
- [ ] If reusable, add/update `reports/insight_registry.yaml` and rerun the recursive queue.

## Adoption status

Not adopted. This is a teacher artifact until source and fresh gates pass.
