# Public teacher extraction — task014

- public path: `public_candidates/lucifer_agi_circuit_20260628/extracted/task014.onnx`
- live path: `/Users/minseong/project/neurogolf/networks/task014.onnx`

## Stored evaluation

- public: ok=True pts=16.014680 mem=7878 params=107
- live: ok=True pts=16.014680 mem=7878 params=107
- delta public-live: pts=+0.000000 mem=0 params=0

## Structural comparison

- public nodes: 38
- live nodes: 38
- public initializer elems: 107
- live initializer elems: 107

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
