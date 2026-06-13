"""fp16 graph surgery, gated on exactness + generalization.

Convert each fp32 net to float16 (keep_io_types). Adopt only if it still passes
stored examples AND fresh arc-gen instances AND reduces memory. Public nets are
typically already fp16 (skipped). Safe: keep-best on the real (generalizing) score.
"""
import json, os, sys, multiprocessing
import onnx
from onnxconverter_common import float16
from src.harness import load_task, evaluate
sys.path.insert(0, "/tmp/arc-gen")

def is_fp16_already(model):
    for init in model.graph.initializer:
        if init.data_type == onnx.TensorProto.FLOAT16:
            return True
    return False

def _try(num):
    from src.adopt import fresh_ok_path, load_gen
    p = f"networks/task{num:03d}.onnx"
    if not os.path.exists(p):
        return num, None
    task = load_task(num)
    cur = evaluate(p, task)
    if not cur["ok"]:
        return num, None
    model = onnx.load(p)
    if is_fp16_already(model):
        return num, None
    try:
        fp16m = float16.convert_float_to_float16(model, keep_io_types=True, disable_shape_infer=False)
    except Exception:
        return num, None
    tmp = f"/tmp/fp16s_{num}.onnx"
    onnx.save(fp16m, tmp)
    try:
        ev = evaluate(tmp, task)
    except Exception:
        return num, None
    if not ev["ok"] or ev["points"] <= cur["points"] + 1e-9:
        return num, None
    try:
        gen = load_gen(num)
    except Exception:
        gen = None
    if not fresh_ok_path(tmp, num, gen, n=50):
        return num, None
    return num, (ev["points"], ev["memory"], ev["params"], cur["points"])

def main():
    man = json.load(open("reports/manifest.json"))["tasks"]
    improved = 0; gain = 0.0
    with multiprocessing.Pool(6) as pool:
        for num, res in pool.imap_unordered(_try, range(1,401)):
            if res is None:
                continue
            pts, mem, par, old = res
            onnx.save(onnx.load(f"/tmp/fp16s_{num}.onnx"), f"networks/task{num:03d}.onnx")
            man[str(num)] = {"points":pts,"memory":mem,"params":par,
                             "method":(man[str(num)].get("method","?") if man.get(str(num)) else "?")+"+fp16"}
            improved += 1; gain += pts-old
            print(f"task{num:03d}: {old:.2f} -> {pts:.2f} (fp16)", flush=True)
    json.dump({"tasks":man}, open("reports/manifest.json","w"), indent=1)
    print(f"fp16 surgery: {improved} nets improved, +{gain:.1f} pts")

if __name__=="__main__":
    main()
