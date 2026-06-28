# Public teacher extraction — task388

- public path: `public_candidates/biohack_mix_20260628/_src_A/task388.onnx`
- live path: `/Users/minseong/project/neurogolf/networks/task388.onnx`

## Stored evaluation

- public: ok=True pts=17.177555 mem=2278 params=218
- live: ok=True pts=17.177555 mem=2278 params=218
- delta public-live: pts=+0.000000 mem=0 params=0

## Structural comparison

- public nodes: 44
- live nodes: 44
- public initializer elems: 218
- live initializer elems: 218

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
