# Public teacher extraction — task008

- public path: `public_candidates/biohack_mix_20260628/_src_A/task008.onnx`
- live path: `/Users/minseong/project/neurogolf/networks/task008.onnx`

## Stored evaluation

- public: ok=True pts=16.364846 mem=5561 params=65
- live: ok=True pts=16.364846 mem=5561 params=65
- delta public-live: pts=+0.000000 mem=0 params=0

## Structural comparison

- public nodes: 57
- live nodes: 57
- public initializer elems: 65
- live initializer elems: 65

### Op delta public-live

- no op-count delta

## Mechanism extraction checklist

- [ ] Identify which full-grid intermediate disappeared or changed dtype.
- [ ] Identify whether QLinear/uint8/bool/opset routing changed.
- [ ] Identify whether final one-hot expansion moved to final `Equal`.
- [ ] Decide whether this is a semantic mechanism or exact-preserve only.
- [ ] If reusable, add/update `reports/insight_registry.yaml` and rerun the recursive queue.

## Adoption status

Not adopted. This is a teacher artifact until source and fresh gates pass.
