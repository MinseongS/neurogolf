# Public teacher extraction — task002

- public path: `public_candidates/urad_7174_10/extracted/task002.onnx`
- live path: `/Users/minseong/project/neurogolf/networks/task002.onnx`

## Stored evaluation

- public: ok=True pts=14.895737 mem=24320 params=127
- live: ok=True pts=14.400469 mem=40084 params=32
- delta public-live: pts=+0.495268 mem=-15764 params=95

## Structural comparison

- public nodes: 196
- live nodes: 91
- public initializer elems: 127
- live initializer elems: 32

### Op delta public-live

- `And`: +3
- `BitShift`: +40
- `BitwiseAnd`: +22
- `BitwiseOr`: +80
- `BitwiseXor`: +1
- `Cast`: +1
- `Gather`: +40
- `Greater`: +1
- `MatMul`: +1
- `Max`: -21
- `MaxPool`: -39
- `Mul`: -21
- `Not`: +1
- `Pad`: -1
- `Sub`: -3

## Mechanism extraction checklist

- [ ] Identify which full-grid intermediate disappeared or changed dtype.
- [ ] Identify whether QLinear/uint8/bool/opset routing changed.
- [ ] Identify whether final one-hot expansion moved to final `Equal`.
- [ ] Decide whether this is a semantic mechanism or exact-preserve only.
- [ ] If reusable, add/update `reports/insight_registry.yaml` and rerun the recursive queue.

## Adoption status

Not adopted. This is a teacher artifact until source and fresh gates pass.
