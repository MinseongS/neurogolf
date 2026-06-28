# Public teacher extraction — task018

- public path: `public_candidates/urad_7174_10/extracted/task018.onnx`
- live path: `/Users/minseong/project/neurogolf/networks/task018.onnx`

## Stored evaluation

- public: ok=True pts=14.156837 mem=48196 params=2987
- live: ok=True pts=13.898130 mem=63257 params=3038
- delta public-live: pts=+0.258707 mem=-15061 params=-51

## Structural comparison

- public nodes: 927
- live nodes: 904
- public initializer elems: 2987
- live initializer elems: 3038

### Op delta public-live

- `Add`: +24
- `And`: +3
- `ArgMax`: +1
- `Cast`: +9
- `Clip`: +56
- `Concat`: +4
- `Div`: +1
- `Equal`: +1
- `Floor`: +1
- `Gather`: -1
- `Greater`: +14
- `GreaterOrEqual`: +2
- `LessOrEqual`: +2
- `Max`: -56
- `MaxPool`: +6
- `Min`: -53
- `Mod`: +1
- `Mul`: +19
- `Slice`: +1
- `Sum`: -3
- `Where`: -9

## Mechanism extraction checklist

- [ ] Identify which full-grid intermediate disappeared or changed dtype.
- [ ] Identify whether QLinear/uint8/bool/opset routing changed.
- [ ] Identify whether final one-hot expansion moved to final `Equal`.
- [ ] Decide whether this is a semantic mechanism or exact-preserve only.
- [ ] If reusable, add/update `reports/insight_registry.yaml` and rerun the recursive queue.

## Adoption status

Not adopted. This is a teacher artifact until source and fresh gates pass.
