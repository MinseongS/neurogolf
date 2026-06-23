"""Test scale? tasks for exact closed-form upscales."""
import json, sys, importlib.util
import numpy as np
sys.path.insert(0, "/tmp/arc-gen")
MAPPING = json.load(open("reports/arc_mapping.json"))

def load_gen(num):
    path = MAPPING[str(num)]["generator"]
    spec = importlib.util.spec_from_file_location(f"gen{num}", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

def fresh(num, n=30):
    gen = load_gen(num)
    exs = []
    t = 0
    while len(exs) < n and t < n * 6:
        t += 1
        try:
            ex = gen.generate()
        except Exception:
            continue
        i = np.array(ex["input"], np.int64); o = np.array(ex["output"], np.int64)
        if max(i.shape + o.shape) > 30: continue
        exs.append((i, o))
    return exs

def kron(i, kh, kw):
    return np.kron(i, np.ones((kh, kw), np.int64))

def fractal(i, kh, kw, invert=False):
    h, w = i.shape
    out = np.zeros((h * kh, w * kw), np.int64)
    for a in range(h):
        for b in range(w):
            cond = (i[a, b] != 0)
            if invert: cond = not cond
            if cond:
                out[a*kh:(a+1)*kh, b*kw:(b+1)*kw] = i
    return out

def tile_plain(i, kh, kw):
    return np.tile(i, (kh, kw))

def tile_mirror(i, kh, kw):
    # 2x2 mirror block
    if (kh, kw) != (2, 2): return None
    top = np.concatenate([i, np.fliplr(i)], axis=1)
    bot = np.concatenate([np.flipud(i), np.flipud(np.fliplr(i))], axis=1)
    return np.concatenate([top, bot], axis=0)

def test(num):
    exs = fresh(num)
    if not exs: return f"task{num}: no examples"
    kh = exs[0][1].shape[0] // exs[0][0].shape[0]
    kw = exs[0][1].shape[1] // exs[0][0].shape[1]
    forms = {
        "kron": lambda i: kron(i, kh, kw),
        "fractal": lambda i: fractal(i, kh, kw, False),
        "fractal_inv": lambda i: fractal(i, kh, kw, True),
        "tile": lambda i: tile_plain(i, kh, kw),
        "mirror": lambda i: tile_mirror(i, kh, kw),
    }
    hits = []
    for name, f in forms.items():
        ok = True
        for i, o in exs:
            try:
                p = f(i)
            except Exception:
                ok = False; break
            if p is None or p.shape != o.shape or not np.array_equal(p, o):
                ok = False; break
        if ok: hits.append(name)
    return f"task{num:03d} k=({kh},{kw}) n={len(exs)} -> EXACT: {hits if hits else 'NONE'}"

if __name__ == "__main__":
    for num in [int(x) for x in sys.argv[1:]]:
        print(test(num))

def test_stamp(num):
    """out block(a,b) depends only on in[a][b] via a fixed color->block LUT."""
    exs = fresh(num)
    if not exs: return f"task{num}: none"
    kh = exs[0][1].shape[0] // exs[0][0].shape[0]
    kw = exs[0][1].shape[1] // exs[0][0].shape[1]
    lut = {}
    ok = True
    for i, o in exs:
        h, w = i.shape
        if o.shape != (h*kh, w*kw): ok=False; break
        for a in range(h):
            for b in range(w):
                blk = o[a*kh:(a+1)*kh, b*kw:(b+1)*kw].copy()
                c = int(i[a,b])
                key = blk.tobytes()
                if c in lut:
                    if lut[c] != key: ok=False; break
                else:
                    lut[c] = key
            if not ok: break
        if not ok: break
    if ok:
        # show the blocks
        import numpy as np
        desc = {c: np.frombuffer(v,np.int64).reshape(kh,kw).tolist() for c,v in lut.items()}
        return f"task{num:03d} STAMP-LUT EXACT k=({kh},{kw}): {desc}"
    return f"task{num:03d} stamp: NO"

def test_crop(num):
    """Test exact crop forms: bbox of nonzero, and most-common-color-stripped bbox."""
    import numpy as np
    exs = fresh(num)
    if not exs: return f"task{num}: none"
    def bbox_nonzero(i):
        nz = np.argwhere(i != 0)
        if len(nz)==0: return None
        r0,c0 = nz.min(0); r1,c1 = nz.max(0)
        return i[r0:r1+1, c0:c1+1]
    def bbox_minuscommon(i):
        vals,counts = np.unique(i, return_counts=True)
        bg = vals[counts.argmax()]
        nz = np.argwhere(i != bg)
        if len(nz)==0: return None
        r0,c0 = nz.min(0); r1,c1 = nz.max(0)
        return i[r0:r1+1, c0:c1+1]
    forms = {"bbox_nz": bbox_nonzero, "bbox_common": bbox_minuscommon}
    hits=[]
    for name,f in forms.items():
        ok=True
        for i,o in exs:
            p=f(i)
            if p is None or p.shape!=o.shape or not np.array_equal(p,o): ok=False;break
        if ok: hits.append(name)
    return f"task{num:03d} n={len(exs)} crop EXACT: {hits if hits else 'NONE'}"
