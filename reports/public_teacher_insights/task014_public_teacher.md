# Public teacher extraction — task014

- public path: `public_candidates/urad_7174_10/extracted/task014.onnx`
- live path: `/Users/minseong/project/neurogolf/networks/task014.onnx`

## Stored evaluation

- public: ok=True pts=16.014555 mem=7878 params=108
- live: ok=True pts=16.003100 mem=7989 params=89
- delta public-live: pts=+0.011454 mem=-111 params=19

## Structural comparison

- public nodes: 38
- live nodes: 39
- public initializer elems: 108
- live initializer elems: 89

### Op delta public-live

- `Cast`: -1

## Mechanism extraction checklist

- [ ] Identify which full-grid intermediate disappeared or changed dtype.
- [ ] Identify whether QLinear/uint8/bool/opset routing changed.
- [ ] Identify whether final one-hot expansion moved to final `Equal`.
- [ ] Decide whether this is a semantic mechanism or exact-preserve only.
- [ ] If reusable, add/update `reports/insight_registry.yaml` and rerun the recursive queue.

## Adoption status

Not adopted. This is a teacher artifact until source and fresh gates pass.
