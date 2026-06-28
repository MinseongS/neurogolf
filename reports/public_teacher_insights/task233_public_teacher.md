# Public teacher extraction — task233

- public path: `public_candidates/biohack_mix_20260628/_src_A/task233.onnx`
- live path: `/Users/minseong/project/neurogolf/networks/task233.onnx`

## Stored evaluation

- public: ok=True pts=14.002712 mem=59147 params=565
- live: ok=True pts=14.002712 mem=59147 params=565
- delta public-live: pts=+0.000000 mem=0 params=0

## Structural comparison

- public nodes: 400
- live nodes: 400
- public initializer elems: 565
- live initializer elems: 565

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
