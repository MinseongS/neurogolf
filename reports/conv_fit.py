"""Single-Conv auto-fitter — learn a mem-0 net for locally-linear tasks.

Usage: PYTHONPATH=. .venv/bin/python reports/conv_fit.py N [--write] [--k 1,3,5]

The NeuroGolf scorer checks sign per output channel: out[0][j] > 0  iff  the
target one-hot channel j is set. So if the rule is a per-cell function of a
KxK neighbourhood of the input one-hot, EACH output channel is a linear
threshold over the flattened KxK*10 binary patch. We fit 10 integer-weight
perceptrons (binary features + integer updates => integer separator, exact in
float32). If all 10 separate with zero error over many fresh + stored examples,
we emit a single `Conv(input,W,B) -> output` net: memory 0, params = 100*k*k+10.

  k=1 -> params ~110  -> ~20.3 pts
  k=3 -> params ~910  -> ~18.2 pts
  k=5 -> params ~2510 -> ~17.2 pts

Far above the 16.8 label-map floor. Only fits genuinely local rules (recolour,
fixed-neighbourhood paint, denoise, local pattern completion); global/structural
rules won't separate and the fitter reports failure (no file written).
"""
import importlib, sys
import numpy as np

sys.path.insert(0, "/tmp/arc-gen")
from src.harness import load_task, convert_to_numpy, evaluate, IR_VERSION
from src.genverify import load_gen

N_TRAIN = 300   # fresh examples used to build the training set
N_VERIFY = 200


def collect(gen, task, n):
    """Return list of (in_oh[10,30,30], out_oh[10,30,30]) numpy pairs."""
    pairs = []
    for ex in (task.get("train", []) + task.get("test", [])):
        bm = convert_to_numpy(ex)
        pairs.append((bm["input"][0], bm["output"][0]))
    tries = 0
    while len([p for p in pairs]) < n and tries < n * 8:
        tries += 1
        try:
            ex = gen.generate()
        except Exception:
            continue
        if max(len(ex["input"]), len(ex["input"][0]),
               len(ex["output"]), len(ex["output"][0])) > 30:
            continue
        bm = convert_to_numpy(ex)
        pairs.append((bm["input"][0], bm["output"][0]))
    return pairs


def patches(in_oh, k):
    """[10,30,30] one-hot -> [900, 10*k*k] binary patch features (zero pad)."""
    h = k // 2
    C, H, W = in_oh.shape
    pad = np.zeros((C, H + 2 * h, W + 2 * h), np.float32)
    pad[:, h:h + H, h:h + W] = in_oh
    cols = []
    for di in range(k):
        for dj in range(k):
            cols.append(pad[:, di:di + H, dj:dj + W])  # [10,30,30]
    feat = np.stack(cols, axis=0)            # [k*k, 10, 30, 30]
    feat = feat.transpose(2, 3, 1, 0).reshape(H * W, C * k * k)  # [900, 10*k*k]
    return feat


def perceptron(X, y, epochs=60):
    """Integer perceptron. X binary [n,d], y in {0,1}. Returns (w,b,errors).
    Pocket: keep best-on-training weights. Margin target: pos>=1, neg<=0."""
    n, d = X.shape
    w = np.zeros(d, np.int64)
    b = np.int64(0)
    target = np.where(y > 0, 1, -1)
    best_w, best_b, best_err = w.copy(), b, n + 1
    idx = np.arange(n)
    for ep in range(epochs):
        # shuffle deterministically by epoch (no Math.random; use roll)
        order = np.roll(idx, ep * 7 + 1)
        for i in order:
            score = int(X[i] @ w) + int(b)
            # pos must be >0 (>=1), neg must be <=0
            if y[i] > 0 and score < 1:
                w += X[i].astype(np.int64); b += 1
            elif y[i] == 0 and score > 0:
                w -= X[i].astype(np.int64); b -= 1
        scores = X @ w + b
        pred = scores > 0
        err = int(np.sum(pred != (y > 0)))
        if err < best_err:
            best_err, best_w, best_b = err, w.copy(), int(b)
        if err == 0:
            break
    return best_w, best_b, best_err


def fit_k(pairs, k):
    X = np.concatenate([patches(io, k) for io, _ in pairs], axis=0)
    Y = np.concatenate([oo.transpose(1, 2, 0).reshape(-1, 10) for _, oo in pairs], axis=0)
    d = 10 * k * k
    W = np.zeros((10, d), np.int64)
    B = np.zeros(10, np.int64)
    for j in range(10):
        wj, bj, err = perceptron(X, Y[:, j].astype(np.int64))
        if err > 0:
            return None, j, err
        W[j], B[j] = wj, bj
    return (W, B), None, 0


def build_model(W, B, k, N=0):
    import onnx
    from onnx import helper, numpy_helper, TensorProto
    h = k // 2
    # W is [10, 10*k*k] flattened as [channel, (dpos, in_ch)]? we built feat as
    # cols ordered [k*k][10] then reshape [.., C*k*k] with C fastest? check:
    # feat cols loop di,dj -> stack axis0 [k*k,10,30,30]; transpose(2,3,1,0) ->
    # [30,30,10,k*k]; reshape -> last dim = 10*k*k with k*k fastest, 10 slower.
    # So feature index = ch*(k*k) + (di*k+dj). Conv weight[outch, inch, di, dj].
    Wc = np.zeros((10, 10, k, k), np.float32)
    for j in range(10):
        wj = W[j].reshape(10, k, k)   # [inch, di, dj]
        Wc[j] = wj
    Bc = B.astype(np.float32)
    inits = [numpy_helper.from_array(Wc, "Wconv"),
             numpy_helper.from_array(Bc, "Bconv")]
    node = helper.make_node("Conv", ["input", "Wconv", "Bconv"], ["output"],
                            pads=[h, h, h, h], kernel_shape=[k, k])
    x = helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 10, 30, 30])
    yv = helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 10, 30, 30])
    g = helper.make_graph([node], f"task{N:03d}", [x], [yv], inits)
    return helper.make_model(g, ir_version=IR_VERSION,
                             opset_imports=[helper.make_opsetid("", 10)])


def fresh_verify(model, gen, nverify, collect_fail=0, N=0):
    """Run fresh examples; return (run, fail, failing_pairs[up to collect_fail])."""
    import onnx, onnxruntime as ort
    onnx.save(model, f"/tmp/_cf{N}.onnx")
    so = ort.SessionOptions()
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    sess = ort.InferenceSession(f"/tmp/_cf{N}.onnx", so)
    run = fail = tries = 0
    fails = []
    while run < nverify and tries < nverify * 8 and gen:
        tries += 1
        try:
            ex = gen.generate()
        except Exception:
            continue
        if max(len(ex["input"]), len(ex["input"][0]),
               len(ex["output"]), len(ex["output"][0])) > 30:
            continue
        bm = convert_to_numpy(ex)
        out = sess.run(None, {"input": bm["input"].astype(np.float32)})[0]
        if not ((out[0] > 0).astype(np.int8) == bm["output"][0].astype(np.int8)).all():
            fail += 1
            if len(fails) < collect_fail:
                fails.append((bm["input"][0], bm["output"][0]))
        run += 1
    return run, fail, fails


def solve(N, ks=(1, 3, 5), n_train=N_TRAIN, n_verify=N_VERIFY, rounds=8,
          write=False, verbose=True):
    """Try to fit a single-Conv net for task N. Returns dict or None on no fit.
    Result: {k, points, params, fresh_run, W, B}."""
    task = load_task(N)
    try:
        gen = load_gen(N)
    except Exception:
        gen = None
    pairs = collect(gen, task, n_train) if gen else collect(None, task, 0)
    if verbose:
        print(f"task{N:03d}: trained on {len(pairs)} examples")
    for k in ks:
        for rnd in range(rounds):
            res, badj, err = fit_k(pairs, k)
            if res is None:
                if verbose:
                    print(f"  k={k}: FAIL (channel {badj} not separable on {len(pairs)} ex, {err} err)")
                break
            W, B = res
            model = build_model(W, B, k, N)
            ev = evaluate(model, task)
            run, fail, fails = fresh_verify(model, gen, n_verify, collect_fail=12, N=N)
            if verbose:
                print(f"  k={k} r{rnd}: stored ok={ev['ok']} pts={ev.get('points'):.2f} "
                      f"params={ev.get('params')} stored_fail={ev.get('fail')} FRESH {run-fail}/{run}")
            if ev["ok"] and fail == 0 and run > 0:
                if verbose:
                    print(f"  >>> task{N:03d} SOLVED by single Conv k={k}: {ev['points']:.2f} pts, mem 0")
                if write:
                    emit_custom(W, B, k, N)
                return {"k": k, "points": ev["points"], "params": ev["params"],
                        "fresh_run": run, "W": W, "B": B}
            if not fails:
                break
            pairs.extend(fails)
    if verbose:
        print(f"task{N:03d}: no single-Conv fit found")
    return None


def main():
    N = int(sys.argv[1])
    write = "--write" in sys.argv
    ks = [1, 3, 5]
    if "--k" in sys.argv:
        ks = [int(x) for x in sys.argv[sys.argv.index("--k") + 1].split(",")]
    solve(N, ks=ks, write=write, verbose=True)


def emit_custom(W, B, k, N=0):
    h = k // 2
    Wc = np.zeros((10, 10, k, k), np.float32)
    for j in range(10):
        Wc[j] = W[j].reshape(10, k, k)
    body = f'''"""task{N:03d} — single-Conv fit (auto-generated by reports/conv_fit.py).
Per-cell rule is neighbourhood-linear; each output channel is an integer-weight
hyperplane over the {k}x{k} one-hot patch. Memory 0 (single Conv input->output)."""
import numpy as np
import onnx
from onnx import helper, numpy_helper, TensorProto
from ..harness import IR_VERSION

WC = np.array({Wc.astype(int).tolist()}, np.float32)
BC = np.array({B.astype(int).tolist()}, np.float32)


def build(task):
    inits = [numpy_helper.from_array(WC, "Wconv"),
             numpy_helper.from_array(BC, "Bconv")]
    node = helper.make_node("Conv", ["input", "Wconv", "Bconv"], ["output"],
                            pads=[{h}, {h}, {h}, {h}], kernel_shape=[{k}, {k}])
    x = helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 10, 30, 30])
    y = helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 10, 30, 30])
    g = helper.make_graph([node], "task{N:03d}", [x], [y], inits)
    return helper.make_model(g, ir_version=IR_VERSION,
                             opset_imports=[helper.make_opsetid("", 10)])
'''
    open(f"src/custom/task{N:03d}.py", "w").write(body)
    print(f"  wrote src/custom/task{N:03d}.py")


if __name__ == "__main__":
    main()
