# Public teacher extraction — task076

- public path: `public_candidates/lucifer_agi_circuit_20260628/extracted/task076.onnx`
- live path: `/Users/minseong/project/neurogolf/networks/task076.onnx`

## Stored evaluation

- public: ok=True pts=14.931507 mem=23296 params=292
- live: ok=True pts=14.931507 mem=23296 params=292
- delta public-live: pts=+0.000000 mem=0 params=0

## Structural comparison

- public nodes: 141
- live nodes: 141
- public initializer elems: 292
- live initializer elems: 292

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
