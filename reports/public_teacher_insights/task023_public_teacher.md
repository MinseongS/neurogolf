# Public teacher extraction — task023

- public path: `public_candidates/urad_7174_10/extracted/task023.onnx`
- live path: `/Users/minseong/project/neurogolf/networks/task023.onnx`

## Stored evaluation

- public: ok=True pts=15.517040 mem=12843 params=291
- live: ok=True pts=14.983809 mem=22275 params=111
- delta public-live: pts=+0.533231 mem=-9432 params=180

## Structural comparison

- public nodes: 158
- live nodes: 56
- public initializer elems: 291
- live initializer elems: 111

### Op delta public-live

- `Add`: +21
- `Cast`: +4
- `Conv`: -11
- `ConvTranspose`: -8
- `Div`: +15
- `Equal`: +1
- `Greater`: -9
- `Min`: +13
- `Mul`: +22
- `Pad`: +2
- `QLinearConv`: +56
- `Sub`: +8
- `Where`: -12

## Mechanism extraction checklist

- [ ] Identify which full-grid intermediate disappeared or changed dtype.
- [ ] Identify whether QLinear/uint8/bool/opset routing changed.
- [ ] Identify whether final one-hot expansion moved to final `Equal`.
- [ ] Decide whether this is a semantic mechanism or exact-preserve only.
- [ ] If reusable, add/update `reports/insight_registry.yaml` and rerun the recursive queue.

## Adoption status

Not adopted. This is a teacher artifact until source and fresh gates pass.
