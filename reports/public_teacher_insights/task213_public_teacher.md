# Public teacher extraction — task213

- public path: `public_candidates/urad_7174_10/extracted/task213.onnx`
- live path: `/Users/minseong/project/neurogolf/networks/task213.onnx`

## Stored evaluation

- public: ok=True pts=17.457256 mem=1838 params=49
- live: ok=True pts=17.440962 mem=1869 params=49
- delta public-live: pts=+0.016295 mem=-31 params=0

## Structural comparison

- public nodes: 69
- live nodes: 76
- public initializer elems: 49
- live initializer elems: 49

### Op delta public-live

- `And`: -1
- `Cast`: -6

## Mechanism extraction checklist

- [ ] Identify which full-grid intermediate disappeared or changed dtype.
- [ ] Identify whether QLinear/uint8/bool/opset routing changed.
- [ ] Identify whether final one-hot expansion moved to final `Equal`.
- [ ] Decide whether this is a semantic mechanism or exact-preserve only.
- [ ] If reusable, add/update `reports/insight_registry.yaml` and rerun the recursive queue.

## Adoption status

Not adopted. This is a teacher artifact until source and fresh gates pass.
