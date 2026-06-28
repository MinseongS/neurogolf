# Public teacher extraction — task138

- public path: `public_candidates/lucifer_agi_circuit_20260628/extracted/task138.onnx`
- live path: `/Users/minseong/project/neurogolf/networks/task138.onnx`

## Stored evaluation

- public: ok=True pts=15.455977 mem=13789 params=172
- live: ok=True pts=15.455977 mem=13789 params=172
- delta public-live: pts=+0.000000 mem=0 params=0

## Structural comparison

- public nodes: 55
- live nodes: 55
- public initializer elems: 172
- live initializer elems: 172

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
