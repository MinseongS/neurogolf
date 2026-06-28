# Public teacher extraction — task010

- public path: `public_candidates/urad_7174_10/extracted/task010.onnx`
- live path: `/Users/minseong/project/neurogolf/networks/task010.onnx`

## Stored evaluation

- public: ok=True pts=18.024586 mem=730 params=340
- live: ok=True pts=18.006067 mem=750 params=340
- delta public-live: pts=+0.018519 mem=-20 params=0

## Structural comparison

- public nodes: 19
- live nodes: 21
- public initializer elems: 340
- live initializer elems: 340

### Op delta public-live

- `Unsqueeze`: -2

## Mechanism extraction checklist

- [ ] Identify which full-grid intermediate disappeared or changed dtype.
- [ ] Identify whether QLinear/uint8/bool/opset routing changed.
- [ ] Identify whether final one-hot expansion moved to final `Equal`.
- [ ] Decide whether this is a semantic mechanism or exact-preserve only.
- [ ] If reusable, add/update `reports/insight_registry.yaml` and rerun the recursive queue.

## Adoption status

Not adopted. This is a teacher artifact until source and fresh gates pass.
