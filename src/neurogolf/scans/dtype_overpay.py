#!/usr/bin/env python3
"""dtype_overpay_scan — GLOBAL fp32 width-overpayment audit across the DEPLOYED nets.

Ported (Task 13): RETARGETED from the original `networks/` (the known false-exhaustion
root cause) to `submission/overfit_nets/` (OVERFIT_NETS). That retarget is the whole
point of the port — the networks/ scan diverged from the deployed nets.

Question: many tasks pay fp32 to convert the one-hot colour input into numeric planes
(per-cell colour-index reads, occupancy masks, count planes). How much fp32 width
overpayment remains, i.e. how many counted fp32 node-output tensors are provably integer
and narrow enough to recast to fp16 / uint8?

Grader counting model (neurogolf.scoring calculate_memory, proven on LB):
  - Only NODE OUTPUTS are counted; bytes = num_elements * dtype.itemsize (static shape,
    maxed with ORT profiler trace). Graph `input`/`output` are FREE.
  - So recasting a counted fp32 plane fp32->fp16 saves bytes/2; fp32->uint8 saves 3*bytes/4.

Classification:
  FLOOR         : not always-integer, OR integer but max|value| > 2048 (not fp16-exact).
  U8_CANDIDATE  : integer, 0<=v<=255, no einsum/topk/dilated-maxpool consumer. save 3/4.
  FP16_SAFE     : integer, max|value|<=2048, no dilated-maxpool consumer (but u8 blocked). save 1/2.
  BLOCKED       : integer & fp16-representable, but a consumer forbids ANY recast (dilated MaxPool).
  PRODUCER_BOUND: would be recastable, BUT its producing node consumes the free graph `input`
                  directly. Needs producer-replacement surgery, NOT a recast. Reported separately.

CAVEAT: observed-integer on bundled examples is NECESSARY not SUFFICIENT. Adoption still
needs per-task proof + fresh arc-gen gating.

Process safety: each task runs in its OWN subprocess (maxtasksperchild=1) — ORT 1.26
cross-session weight aliasing corrupts structurally-identical nets in one process.
"""
import json
import math
import pathlib
import subprocess
import sys

import numpy as np

from neurogolf.paths import ROOT, OVERFIT_NETS, STATE
from neurogolf.manifest import load as load_manifest

NETS = OVERFIT_NETS
TASKLOG = STATE / "tasks"

FP16_MAX_EXACT = 2048  # fp16 exactly represents integers up to 2^11


# --------------------------------------------------------------------------------------
# WORKER: runs in its own subprocess for ONE task. Prints a JSON line to stdout.
# --------------------------------------------------------------------------------------
def worker(task_num):
    import onnx
    from onnx import shape_inference, helper
    import onnxruntime as ort
    from neurogolf.scoring import load_task, convert_to_numpy

    res = {"task": task_num, "error": None, "tensors": []}
    path = NETS / f"task{task_num:03d}.onnx"
    if not path.exists():
        res["error"] = "no onnx"
        return res

    model = onnx.load(str(path))
    try:
        inferred = shape_inference.infer_shapes(model, strict_mode=True)
    except Exception as e:
        res["error"] = f"shape_infer: {e}"
        return res
    g = inferred.graph

    # graph input name(s)
    input_names = {i.name for i in g.input}
    vi = {v.name: v for v in list(g.value_info) + list(g.output) + list(g.input)}

    # producer map: tensor -> producing node ; and node consumes-input?
    producer = {}
    for n in g.node:
        for o in n.output:
            if o:
                producer[o] = n

    # consumer map: tensor -> list of (op_type, dilated_maxpool_bool)
    consumers = {}
    for n in g.node:
        dil = False
        if n.op_type == "MaxPool":
            for a in n.attribute:
                if a.name == "dilations" and any(d != 1 for d in a.ints):
                    dil = True
        for inp in n.input:
            if not inp:
                continue
            consumers.setdefault(inp, []).append((n.op_type, dil))

    # enumerate counted fp32 node-output tensors (exclude graph output "output" and inputs)
    fp32_tensors = []
    for n in g.node:
        for o in n.output:
            if not o or o == "output" or o in input_names:
                continue
            v = vi.get(o)
            if v is None or not v.type.HasField("tensor_type"):
                continue
            tt = v.type.tensor_type
            if tt.elem_type != onnx.TensorProto.FLOAT:  # fp32 only
                continue
            if not tt.HasField("shape"):
                continue
            dims = []
            ok = True
            for d in tt.shape.dim:
                if not d.HasField("dim_value") or d.dim_value <= 0:
                    ok = False
                    break
                dims.append(d.dim_value)
            if not ok:
                continue
            nelem = 1
            for d in dims:
                nelem *= d
            fp32_tensors.append((o, dims, nelem))

    if not fp32_tensors:
        return res  # nothing to recast (all counted planes already narrow / int / none)

    # build a run-only model exposing these tensors as extra outputs
    run_model = onnx.load(str(path))
    existing_out = {out.name for out in run_model.graph.output}
    for (name, dims, _) in fp32_tensors:
        if name in existing_out:
            continue
        vinfo = helper.make_tensor_value_info(name, onnx.TensorProto.FLOAT, dims)
        run_model.graph.output.append(vinfo)

    try:
        so = ort.SessionOptions()
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
        sess = ort.InferenceSession(
            run_model.SerializeToString(), so, providers=["CPUExecutionProvider"]
        )
    except Exception as e:
        res["error"] = f"session: {e}"
        return res

    out_names = [t[0] for t in fp32_tensors]
    in_name = list(input_names)[0] if input_names else "input"

    task = load_task(task_num)
    examples = (
        task.get("train", []) + task.get("test", []) + task.get("arc-gen", [])
    )

    # per-tensor accumulators
    n_t = len(fp32_tensors)
    always_int = [True] * n_t
    max_abs = [0.0] * n_t
    min_val = [math.inf] * n_t
    max_val = [-math.inf] * n_t
    only_binary = [True] * n_t
    seen_any = [False] * n_t

    ran = 0
    for ex in examples:
        bench = convert_to_numpy(ex)
        if not bench:
            continue
        try:
            outs = sess.run(out_names, {in_name: bench["input"]})
        except Exception as e:
            # ORT runtime error on this example — record and stop (net likely fine, rare)
            res["error"] = f"run: {e}"
            return res
        ran += 1
        for i, arr in enumerate(outs):
            a = np.asarray(arr)
            if a.size == 0:
                continue
            seen_any[i] = True
            fa = a.astype(np.float64)
            if not np.all(np.isfinite(fa)):
                always_int[i] = False
                only_binary[i] = False
                continue
            if not np.all(fa == np.round(fa)):
                always_int[i] = False
            mx = float(np.max(np.abs(fa)))
            if mx > max_abs[i]:
                max_abs[i] = mx
            lo = float(np.min(fa))
            hi = float(np.max(fa))
            if lo < min_val[i]:
                min_val[i] = lo
            if hi > max_val[i]:
                max_val[i] = hi
            u = np.unique(fa)
            if not np.all(np.isin(u, (0.0, 1.0))):
                only_binary[i] = False

    res["ran_examples"] = ran
    res["total_examples"] = len(examples)

    for i, (name, dims, nelem) in enumerate(fp32_tensors):
        bytes_fp32 = nelem * 4
        cons = consumers.get(name, [])
        cons_ops = sorted({c[0] for c in cons})
        einsum_cons = any(c[0] == "Einsum" for c in cons)
        topk_cons = any(c[0] == "TopK" for c in cons)
        dilmaxpool_cons = any(c[1] for c in cons)

        prod = producer.get(name)
        prod_op = prod.op_type if prod is not None else None
        producer_bound = bool(prod is not None and any(pi in input_names for pi in prod.input))

        aint = always_int[i]
        mabs = max_abs[i]
        mn = min_val[i] if seen_any[i] else None
        mx = max_val[i] if seen_any[i] else None
        binary = only_binary[i] and seen_any[i]

        # base classification
        if not seen_any[i]:
            cls, save = "FLOOR", 0
            reason = "no-observation"
        elif not aint:
            cls, save = "FLOOR", 0
            reason = "non-integer"
        elif mabs > FP16_MAX_EXACT:
            cls, save = "FLOOR", 0
            reason = f"max|v|={mabs:.0f}>2048"
        else:
            u8_ok = (mn is not None and mn >= 0 and mx is not None and mx <= 255
                     and not einsum_cons and not topk_cons and not dilmaxpool_cons)
            fp16_ok = not dilmaxpool_cons
            if dilmaxpool_cons:
                cls, save = "BLOCKED", 0
                reason = "dilated-MaxPool consumer (fp16 breaker)"
            elif u8_ok:
                cls, save = "U8_CANDIDATE", bytes_fp32 * 3 // 4
                reason = "int 0..255"
            elif fp16_ok:
                cls, save = "FP16_SAFE", bytes_fp32 // 2
                blk = []
                if einsum_cons:
                    blk.append("einsum")
                if topk_cons:
                    blk.append("topk")
                reason = "int<=2048 (u8 blocked: %s)" % ",".join(blk) if blk else "int<=2048"
            else:
                cls, save = "BLOCKED", 0
                reason = "consumer-blocked"

        # PRODUCER_BOUND override: recastable but producer eats the free input directly
        if cls in ("FP16_SAFE", "U8_CANDIDATE") and producer_bound:
            reason = f"producer {prod_op} consumes graph input directly ({reason})"
            cls = "PRODUCER_BOUND"
            # keep 'save' as the would-be saving for the separate bucket

        res["tensors"].append({
            "name": name,
            "dims": dims,
            "bytes_fp32": bytes_fp32,
            "always_integer": aint,
            "binary": binary,
            "min": mn,
            "max": mx,
            "max_abs": mabs,
            "producer_op": prod_op,
            "producer_bound": producer_bound,
            "consumer_ops": cons_ops,
            "einsum_consumer": einsum_cons,
            "topk_consumer": topk_cons,
            "dilated_maxpool_consumer": dilmaxpool_cons,
            "class": cls,
            "reason": reason,
            "would_save_bytes": save,
        })
    return res


# --------------------------------------------------------------------------------------
# SUBPROCESS ENTRY: `python dtype_overpay.py --worker N` -> prints JSON line
# --------------------------------------------------------------------------------------
def _worker_main(task_num):
    try:
        r = worker(task_num)
    except Exception as e:
        import traceback
        r = {"task": task_num, "error": f"worker-exc: {e}", "tb": traceback.format_exc()[-500:], "tensors": []}
    sys.stdout.write(json.dumps(r))
    sys.stdout.flush()


# --------------------------------------------------------------------------------------
# DRIVER
# --------------------------------------------------------------------------------------
def run_one(task_num):
    """Spawn a fresh python for one task (hard process isolation)."""
    p = subprocess.run(
        [sys.executable, str(pathlib.Path(__file__).resolve()), "--worker", str(task_num)],
        capture_output=True, text=True, cwd=str(ROOT), timeout=600,
    )
    if p.returncode != 0 or not p.stdout.strip():
        return {"task": task_num, "error": f"subproc rc={p.returncode}: {p.stderr[-300:]}", "tensors": []}
    try:
        return json.loads(p.stdout.strip().splitlines()[-1])
    except Exception as e:
        return {"task": task_num, "error": f"parse: {e}: {p.stdout[:200]}", "tensors": []}


def cur_points(cost):
    return max(1.0, 25.0 - math.log(max(1.0, cost)))


def scan_all(tasks: list[int] | None = None) -> dict:
    manifest = load_manifest()

    # which tasks already have an fp16 recast attempt logged
    fp16_logged = set()
    for f in sorted(TASKLOG.glob("task*.md")):
        try:
            if "fp16" in f.read_text().lower():
                fp16_logged.add(int(f.stem[4:]))
        except Exception:
            pass

    if tasks:
        run_tasks = [t for t in tasks
                     if manifest.get(f"{t:03d}", {}).get("cost", 0)]
    else:
        run_tasks = []
        for k, v in manifest.items():
            try:
                tn = int(k)
            except ValueError:
                continue
            cost = v.get("cost", 0)
            if not cost:
                continue  # nothing counted -> nothing to recast
            run_tasks.append(tn)
    run_tasks.sort()

    from multiprocessing import Pool
    results = {}
    with Pool(processes=8, maxtasksperchild=1) as pool:
        for r in pool.imap_unordered(run_one, run_tasks):
            results[r["task"]] = r
            done = len(results)
            if done % 20 == 0:
                sys.stderr.write(f"  ...{done}/{len(run_tasks)}\n")
                sys.stderr.flush()

    # aggregate
    per_task = []
    for tn in run_tasks:
        r = results.get(tn, {"task": tn, "error": "missing", "tensors": []})
        m = manifest.get(f"{tn:03d}", {})
        cost = m.get("cost", 0)
        cur = cur_points(cost)
        tens = r.get("tensors", [])
        headline_save = sum(t["would_save_bytes"] for t in tens
                            if t["class"] in ("FP16_SAFE", "U8_CANDIDATE"))
        pb_save = sum(t["would_save_bytes"] for t in tens if t["class"] == "PRODUCER_BOUND")
        # headline_save is a subset-sum of counted-memory tensor savings, so it can never
        # exceed total cost (memory+params) in practice; cap defensively anyway.
        headline_save = min(headline_save, cost)  # can't drop below 0
        new_cost = cost - headline_save
        new_pts = cur_points(new_cost)
        delta = new_pts - cur
        # biggest recastable tensor
        recast = [t for t in tens if t["class"] in ("FP16_SAFE", "U8_CANDIDATE")]
        recast.sort(key=lambda t: -t["would_save_bytes"])
        main_tensor = None
        if recast:
            b = recast[0]
            main_tensor = {"name": b["name"], "class": b["class"], "dims": b["dims"],
                           "save_bytes": b["would_save_bytes"], "max": b["max"]}
        eg = headline_save / cost if cost else 0.0
        per_task.append({
            "task": tn,
            "cost": cost,
            "cur_points": round(cur, 4),
            "headline_savings_bytes": headline_save,
            "producer_bound_savings_bytes": pb_save,
            "new_points": round(new_pts, 4),
            "delta_points": round(delta, 4),
            "fp16_logged": tn in fp16_logged,
            "error": r.get("error"),
            "n_fp32_tensors": len(tens),
            "main_tensor": main_tensor,
            "expected_gain": round(eg, 4),
        })

    items = [pt for pt in per_task if pt["headline_savings_bytes"] > 0]
    items.sort(key=lambda p: -p["expected_gain"])
    return {"items": items}


if __name__ == "__main__":
    if len(sys.argv) >= 3 and sys.argv[1] == "--worker":
        _worker_main(int(sys.argv[2]))
