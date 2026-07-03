#!/usr/bin/env python3
"""dtype_overpay_scan.py — GLOBAL fp32 width-overpayment audit across all 400 deployed nets.

Question: many tasks pay fp32 to convert the one-hot colour input into numeric planes
(per-cell colour-index reads, occupancy masks, count planes). How much fp32 width
overpayment remains, i.e. how many counted fp32 node-output tensors are provably integer
and narrow enough to recast to fp16 / uint8?

Grader counting model (src/harness.py calculate_memory, proven on LB):
  - Only NODE OUTPUTS are counted; bytes = num_elements * dtype.itemsize (static shape,
    maxed with ORT profiler trace). Graph `input`/`output` are FREE.
  - So recasting a counted fp32 plane fp32->fp16 saves bytes/2; fp32->uint8 saves 3*bytes/4.

Per fp32 counted tensor we OBSERVE (on ALL bundled examples: train+test+arc-gen):
  always-integer? max|value|? binary(0/1)? min value? plus static bytes + consumer ops.

Classification:
  FLOOR         : not always-integer, OR integer but max|value| > 2048 (not fp16-exact).
  U8_CANDIDATE  : integer, 0<=v<=255, no einsum/topk/dilated-maxpool consumer. save 3/4.
  FP16_SAFE     : integer, max|value|<=2048, no dilated-maxpool consumer (but u8 blocked). save 1/2.
  BLOCKED       : integer & fp16-representable, but a consumer forbids ANY recast (dilated MaxPool).
  PRODUCER_BOUND: would be recastable, BUT its producing node consumes the free graph `input`
                  directly. ORT binds producer output dtype to input dtype; a post-hoc Cast
                  leaves the fp32 plane counted AND adds a plane (net worse), and casting the
                  input first costs an ~18KB input copy. Needs producer-replacement surgery
                  (QLinearConv / narrow-emitting collapse), NOT a recast. Excluded from the
                  headline savings; reported separately.

Consumer blockers: ORT Einsum rejects uint8; uint8 TopK is a Kaggle grader-killer; fp16
breaks MaxPool-with-dilations.

CAVEAT (stated in the .md): observed-integer on bundled examples is NECESSARY not SUFFICIENT.
Adoption still needs per-task proof + fresh arc-gen gating.

Process safety: each task runs in its OWN subprocess (maxtasksperchild=1) — ORT 1.26
cross-session weight aliasing corrupts structurally-identical nets in one process.
"""
import json
import math
import os
import pathlib
import subprocess
import sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

NETWORKS = ROOT / "networks"
MANIFEST = ROOT / "reports" / "manifest.json"
TASKLOG = ROOT / "reports" / "tasklog"
OUT_JSON = ROOT / "reports" / "dtype_overpay_scan.json"
OUT_MD = ROOT / "reports" / "dtype_overpay_scan.md"

FP16_MAX_EXACT = 2048  # fp16 exactly represents integers up to 2^11


# --------------------------------------------------------------------------------------
# WORKER: runs in its own subprocess for ONE task. Prints a JSON line to stdout.
# --------------------------------------------------------------------------------------
def worker(task_num):
    import onnx
    from onnx import shape_inference, helper
    import onnxruntime as ort
    from src.harness import load_task, convert_to_numpy

    res = {"task": task_num, "error": None, "tensors": []}
    path = NETWORKS / f"task{task_num:03d}.onnx"
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
# SUBPROCESS ENTRY: `python dtype_overpay_scan.py --worker N` -> prints JSON line
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


def main():
    manifest = json.load(open(MANIFEST))["tasks"]

    def cur_points(mem, params):
        return max(1.0, 25.0 - math.log(max(1.0, mem + params)))

    # which tasks already have an fp16 recast attempt logged
    fp16_logged = set()
    for f in sorted(TASKLOG.glob("task*.md")):
        try:
            if "fp16" in f.read_text().lower():
                fp16_logged.add(int(f.stem[4:]))
        except Exception:
            pass

    tasks = []
    for k, v in manifest.items():
        try:
            tn = int(k)
        except ValueError:
            continue
        mem = v.get("memory", 0)
        if mem == 0:
            continue  # no counted intermediates -> nothing to recast
        tasks.append(tn)
    tasks.sort()

    from multiprocessing import Pool
    results = {}
    with Pool(processes=8, maxtasksperchild=1) as pool:
        for r in pool.imap_unordered(run_one, tasks):
            results[r["task"]] = r
            done = len(results)
            if done % 20 == 0:
                sys.stderr.write(f"  ...{done}/{len(tasks)}\n")
                sys.stderr.flush()

    # aggregate
    per_task = []
    errors = 0
    for tn in tasks:
        r = results.get(tn, {"task": tn, "error": "missing", "tensors": []})
        m = manifest[str(tn)]
        mem, params = m.get("memory", 0), m.get("params", 0)
        cur = cur_points(mem, params)
        if r.get("error"):
            errors += 1
        tens = r.get("tensors", [])
        headline_save = sum(t["would_save_bytes"] for t in tens
                            if t["class"] in ("FP16_SAFE", "U8_CANDIDATE"))
        pb_save = sum(t["would_save_bytes"] for t in tens if t["class"] == "PRODUCER_BOUND")
        headline_save = min(headline_save, mem)  # can't drop below 0
        new_mem = mem - headline_save
        new_pts = cur_points(new_mem, params)
        delta = new_pts - cur
        # biggest recastable tensor
        recast = [t for t in tens if t["class"] in ("FP16_SAFE", "U8_CANDIDATE")]
        recast.sort(key=lambda t: -t["would_save_bytes"])
        main_tensor = None
        if recast:
            b = recast[0]
            main_tensor = {"name": b["name"], "class": b["class"], "dims": b["dims"],
                           "save_bytes": b["would_save_bytes"], "max": b["max"]}
        per_task.append({
            "task": tn,
            "memory": mem,
            "params": params,
            "cur_points": round(cur, 4),
            "headline_savings_bytes": headline_save,
            "producer_bound_savings_bytes": pb_save,
            "new_points": round(new_pts, 4),
            "delta_points": round(delta, 4),
            "fp16_logged": tn in fp16_logged,
            "error": r.get("error"),
            "n_fp32_tensors": len(tens),
            "main_tensor": main_tensor,
            "tensors": tens,
        })

    # class counts
    class_counts = {"FP16_SAFE": 0, "U8_CANDIDATE": 0, "BLOCKED": 0, "PRODUCER_BOUND": 0, "FLOOR": 0}
    for pt in per_task:
        for t in pt["tensors"]:
            class_counts[t["class"]] = class_counts.get(t["class"], 0) + 1

    total_delta = sum(pt["delta_points"] for pt in per_task)
    total_pb_bytes = sum(pt["producer_bound_savings_bytes"] for pt in per_task)

    summary = {
        "n_tasks_scanned": len(tasks),
        "n_errors": errors,
        "total_delta_points": round(total_delta, 4),
        "class_counts": class_counts,
        "total_producer_bound_savings_bytes": total_pb_bytes,
    }
    json.dump({"summary": summary, "tasks": per_task}, open(OUT_JSON, "w"), indent=1)

    # markdown
    ranked = sorted([pt for pt in per_task if pt["delta_points"] > 0],
                    key=lambda p: -p["delta_points"])
    lines = []
    lines.append("# fp32 dtype-overpayment scan (global value-range audit)\n")
    lines.append(f"Scanned **{len(tasks)}** deployed nets with counted memory>0 "
                 f"(manifest mem==0 skipped). Errors: **{errors}**.\n")
    lines.append(f"**Total potential delta_points (headline, FP16_SAFE+U8_CANDIDATE recasts): "
                 f"{total_delta:+.3f}** across all tasks.\n")
    lines.append("Headline savings EXCLUDE `PRODUCER_BOUND` tensors (producer eats the free fp32 "
                 "graph input directly — recast is net-worse; needs producer-replacement surgery). "
                 f"PRODUCER_BOUND would-be bytes across all tasks: {total_pb_bytes}.\n")
    lines.append("> **Observed-integer on bundled examples is NECESSARY, not SUFFICIENT.** "
                 "A plane can be integer/binary on every bundled example yet exceed the range "
                 "on fresh arc-gen instances. Every adoption still needs per-task proof + fresh "
                 "arc-gen gating (`reports/scripts/fresh_verify.py`), and bit-identity vs the "
                 "incumbent `networks/taskNNN.onnx`.\n")

    lines.append("## Counted fp32 tensor classes (tensor-level tally)\n")
    lines.append("| class | count | meaning |")
    lines.append("|---|---|---|")
    lines.append(f"| FP16_SAFE | {class_counts['FP16_SAFE']} | integer, |v|<=2048, no dilated-MaxPool; save bytes/2 (u8 blocked by einsum/topk) |")
    lines.append(f"| U8_CANDIDATE | {class_counts['U8_CANDIDATE']} | integer 0..255, no einsum/topk/dilated-MaxPool; save 3*bytes/4 |")
    lines.append(f"| PRODUCER_BOUND | {class_counts['PRODUCER_BOUND']} | recastable range BUT producer consumes free input directly; excluded from headline |")
    lines.append(f"| BLOCKED | {class_counts['BLOCKED']} | integer&in-range but dilated-MaxPool consumer forbids any recast |")
    lines.append(f"| FLOOR | {class_counts['FLOOR']} | non-integer OR |v|>2048; genuinely needs fp32 |")
    lines.append("")

    lines.append(f"## Top-30 tasks by delta_points ({len([p for p in ranked])} have delta>0)\n")
    lines.append("| task | mem | params | cur pts | save B | new pts | delta | fp16 logged? | main tensor (class, dims, saveB, max) |")
    lines.append("|---|---|---|---|---|---|---|---|---|")
    for pt in ranked[:30]:
        mt = pt["main_tensor"]
        if mt:
            mtstr = f"{mt['name']} ({mt['class']}, {mt['dims']}, {mt['save_bytes']}B, max={mt['max']})"
        else:
            mtstr = "-"
        flag = "yes" if pt["fp16_logged"] else "no"
        lines.append(f"| {pt['task']} | {pt['memory']} | {pt['params']} | {pt['cur_points']} | "
                     f"{pt['headline_savings_bytes']} | {pt['new_points']} | {pt['delta_points']:+.4f} | "
                     f"{flag} | {mtstr} |")
    lines.append("")

    # producer-bound leaderboard (separate seam)
    pb = sorted([pt for pt in per_task if pt["producer_bound_savings_bytes"] > 0],
                key=lambda p: -p["producer_bound_savings_bytes"])
    lines.append(f"## PRODUCER_BOUND seam (separate — needs producer surgery, not recast) — top 15\n")
    lines.append("| task | mem | producer-bound would-be bytes | note |")
    lines.append("|---|---|---|---|")
    for pt in pb[:15]:
        pbt = [t for t in pt["tensors"] if t["class"] == "PRODUCER_BOUND"]
        ops = sorted({t["producer_op"] for t in pbt})
        lines.append(f"| {pt['task']} | {pt['memory']} | {pt['producer_bound_savings_bytes']} | producers: {','.join(str(o) for o in ops)} |")
    lines.append("")

    OUT_MD.write_text("\n".join(lines))
    # console summary
    print(json.dumps(summary, indent=1))
    print("\nTop-15 by delta_points:")
    for pt in ranked[:15]:
        mt = pt["main_tensor"]
        mts = f"{mt['name']}/{mt['class']}/{mt['save_bytes']}B" if mt else "-"
        print(f"  task{pt['task']:03d}  cur={pt['cur_points']:.3f}  saveB={pt['headline_savings_bytes']}  "
              f"delta={pt['delta_points']:+.4f}  {mts}  fp16logged={pt['fp16_logged']}")
    print(f"\nWrote {OUT_JSON}\nWrote {OUT_MD}")


if __name__ == "__main__":
    if len(sys.argv) >= 3 and sys.argv[1] == "--worker":
        _worker_main(int(sys.argv[2]))
    else:
        main()
