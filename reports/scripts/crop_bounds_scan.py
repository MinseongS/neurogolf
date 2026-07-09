#!/usr/bin/env python3
"""crop_bounds_scan.py  --  SCAN-ONLY crop-opportunity finder across all 400 tasks.

Generalizes the S9 grid-crop lever (crop a counted 30x30 plane down to the true
generator bound) to all 400 tasks AND mid-size planes (25, 20, ...).

For each task N it computes:
  1. GEN_BOUND  -- max grid H,W the generator can emit (input & output), by
                   SAMPLING the arc-gen generator ~SAMPLE_N times, plus the
                   bundled/stored max from data/taskNNN.json (train+test+arc-gen).
  2. NET_PLANES -- max spatial dim among counted intermediate tensors in
                   networks/taskNNN.onnx (via onnx.shape_inference), and the
                   counted bytes sitting in planes larger than the gen bound.

FLAG when max_net_spatial_dim > gen_bound (net computes on planes bigger than
any grid the generator can produce).  est_saving ~= sum over oversized tensors
of bytes * (1 - bound^2 / dim^2)  -- a RANKING heuristic only (only counted
ENTRY reads benefit; free-input walk-einsum planes do NOT -- task077 refutation).

Writes:  reports/crop_bounds_scan.json  (all 400 rows)
         reports/crop_bounds_scan.md    (ranked FLAGGED table)

Single-process by design (two background scans share this CPU).  Run from repo
root:  PYTHONPATH=. uv run python reports/scripts/crop_bounds_scan.py
"""
import importlib
import json
import math
import os
import signal
import sys
import time

import numpy as np
import onnx

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(ROOT)
sys.path.append(ROOT)
# arc-gen on path via append (NOT insert(0)) to avoid shadowing stdlib/common.
sys.path.append(os.path.join(ROOT, "arc-gen"))
sys.path.append(os.path.join(ROOT, "arc-gen", "tasks"))

from src.harness import load_task  # noqa: E402

SAMPLE_N = 300           # generator samples per task for the screen
PER_TASK_BUDGET = 20.0   # seconds of generator sampling per task
GRID_CAP = 30            # harness hard cap (grids >30 are dropped)
SPATIAL_MIN = 11         # dims in [SPATIAL_MIN..30] are grid-spatial (10 = channels)

# Tasks already verified / decided in S9 (grid_crop_bounds.md + NEXT_SESSION dead list).
ALREADY_HANDLED = {187, 29, 243, 198, 80, 205, 173, 138, 14, 77, 193, 192, 222, 396, 233}


class TaskTimeout(Exception):
    """Raised by SIGALRM: a single generate() call exceeded the hard budget.

    Needed because some generators rejection-sample internally and a single
    call can spin forever -- a between-samples wall-clock check never fires.
    """


def _alarm_handler(signum, frame):
    raise TaskTimeout()


signal.signal(signal.SIGALRM, _alarm_handler)


def dtype_itemsize(elem_type):
    try:
        return np.dtype(onnx.helper.tensor_dtype_to_np_dtype(elem_type)).itemsize
    except Exception:
        return 4


def spatial_dim_of(dims):
    """Effective max grid-spatial dim of a tensor shape.

    - dims in [SPATIAL_MIN..GRID_CAP] count directly (channels=10 excluded).
    - dims > GRID_CAP that are a perfect square s*s with s in range are treated
      as a flattened square plane of side s (e.g. 900 -> 30, 400 -> 20).
    Returns 0 if the tensor carries no grid-spatial axis.
    """
    best = 0
    for d in dims:
        if SPATIAL_MIN <= d <= GRID_CAP:
            best = max(best, d)
        elif d > GRID_CAP:
            s = int(round(math.isqrt(d)))
            if s * s == d and SPATIAL_MIN <= s <= GRID_CAP:
                best = max(best, s)
    return best


def analyze_net(path):
    """Return (max_spatial_dim, planes, static_mem, params, status).

    planes = counted intermediates carrying a grid-spatial axis.
    static_mem = static-shape-inference sum over ALL counted intermediates
    (the base term of harness calculate_memory, before the runtime-trace max)
    -- computed live from the net so it stays fresh vs a stale inventory.
    params = initializer + Constant-node element count (harness calculate_params).
    """
    model = onnx.load(path)
    try:
        inf = onnx.shape_inference.infer_shapes(model, strict_mode=False)
    except Exception as e:
        return None, None, None, None, f"shape_infer_error:{type(e).__name__}"
    g = inf.graph
    inits = {i.name for i in g.initializer}
    inits.update(i.name for i in g.sparse_initializer)
    planes = []
    max_dim = 0
    static_mem = 0
    seen = set()
    for vi in list(g.value_info) + list(g.output):
        name = vi.name
        if name in seen:
            continue
        seen.add(name)
        if name in inits or name in ("input", "output"):
            continue
        tt = vi.type.tensor_type
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
        elems = 1
        for d in dims:
            elems *= d
        bytes_ = elems * dtype_itemsize(tt.elem_type)
        static_mem += bytes_
        sp = spatial_dim_of(dims)
        if sp == 0:
            continue
        planes.append({"name": name, "shape": dims, "spatial": sp, "bytes": bytes_})
        max_dim = max(max_dim, sp)
    params = 0
    for init in list(g.initializer):
        e = 1
        for d in init.dims:
            e *= d
        params += e
    for si in g.sparse_initializer:
        e = 1
        for d in si.values.dims:
            e *= d
        params += e
    for node in g.node:
        if node.op_type != "Constant":
            continue
        for attr in node.attribute:
            if attr.name in ("value", "sparse_value"):
                t = attr.t if attr.name == "value" else attr.sparse_tensor.values
                e = 1
                for d in t.dims:
                    e *= d
                params += e
            elif attr.name == "value_floats":
                params += len(attr.floats)
            elif attr.name == "value_ints":
                params += len(attr.ints)
            elif attr.name == "value_strings":
                params += len(attr.strings)
    return max_dim, planes, static_mem, params, "ok"


def sample_generator(arc_id):
    """Sample the generator; return dict of max dims + count + status."""
    modname = f"task_{arc_id}"
    res = {"gen_in_h": 0, "gen_in_w": 0, "gen_out_h": 0, "gen_out_w": 0,
           "sample_n": 0, "gen_status": "ok"}
    try:
        mod = importlib.import_module(modname)
    except Exception as e:
        res["gen_status"] = f"import_error:{type(e).__name__}:{e}"[:120]
        return res
    if not hasattr(mod, "generate"):
        res["gen_status"] = "no_generate"
        return res
    t0 = time.time()
    n = 0
    fail = 0
    # HARD per-task budget via SIGALRM: a single generate() can spin forever
    # (internal rejection sampling), so a between-samples check is not enough.
    signal.alarm(int(PER_TASK_BUDGET) + 2)
    try:
        for _ in range(SAMPLE_N):
            if time.time() - t0 > PER_TASK_BUDGET:
                res["gen_status"] = f"timeout_after_{n}"
                break
            try:
                ex = mod.generate()
                gi, go = ex["input"], ex["output"]
                res["gen_in_h"] = max(res["gen_in_h"], len(gi))
                res["gen_in_w"] = max(res["gen_in_w"], len(gi[0]))
                res["gen_out_h"] = max(res["gen_out_h"], len(go))
                res["gen_out_w"] = max(res["gen_out_w"], len(go[0]))
                n += 1
            except TaskTimeout:
                res["gen_status"] = f"hard_timeout_after_{n}"
                break
            except Exception as e:
                fail += 1
                if fail > 5 and n == 0:
                    res["gen_status"] = f"generate_error:{type(e).__name__}:{e}"[:120]
                    break
    except TaskTimeout:
        res["gen_status"] = f"hard_timeout_after_{n}"
    finally:
        signal.alarm(0)
    res["sample_n"] = n
    if n == 0 and res["gen_status"] == "ok":
        res["gen_status"] = "generate_error:no_samples"
    return res


def bundled_max(task_num):
    """Max grid dim over bundled train+test+arc-gen entries in data json."""
    try:
        t = load_task(task_num)
    except Exception:
        return {"b_in": 0, "b_out": 0, "b_status": "load_error"}
    b_in = b_out = 0
    for split in ("train", "test", "arc-gen"):
        for ex in t.get(split, []) or []:
            gi, go = ex.get("input"), ex.get("output")
            if gi:
                b_in = max(b_in, len(gi), len(gi[0]))
            if go:
                b_out = max(b_out, len(go), len(go[0]))
    return {"b_in": b_in, "b_out": b_out, "b_status": "ok"}


def enrich_with_points(rows):
    """Add est_points_delta: score = 25 - ln(mem+params), so saving S bytes of
    counted mem gives delta = ln(M / (M - S)) with M = mem + params.

    Uses the LIVE static-shape-inference mem/params computed from the net
    (net_static_mem/net_params); the global inventory can be stale vs
    working-tree nets."""
    for r in rows:
        mem = r.get("net_static_mem")
        params = r.get("net_params")
        r["est_points_delta"] = 0.0
        if mem is None:
            continue
        M = mem + (params or 0)
        S = min(r["est_saving_bytes"], mem)  # can't save more than counted mem
        if M > 0 and S > 0 and M - S > 0:
            r["est_points_delta"] = round(math.log(M / (M - S)), 4)


def write_report(rows):
    flagged = [r for r in rows if r["flag"]]
    flagged.sort(key=lambda r: (-r.get("est_points_delta", 0.0), -r["est_saving_bytes"]))
    gen_fail = [r for r in rows if "error" in r["status"] or "timeout" in r["status"]]

    # Mechanism context: dominant op types per net (from the layer inventory;
    # ops rarely change even when a net is re-golfed, so slight staleness is fine).
    ops_by_task = {}
    try:
        inv = json.load(open("reports/global_layer_inventory.json"))
        for t in inv["tasks"]:
            ops = sorted((t.get("ops") or {}).items(), key=lambda kv: -kv[1])
            ops_by_task[t["task"]] = " ".join(f"{k}x{v}" if v > 1 else k
                                              for k, v in ops[:4])
    except Exception:
        pass

    with open("reports/crop_bounds_scan.md", "w") as f:
        f.write("# Crop-bounds scan (all 400 tasks)\n\n")
        f.write("SCAN-ONLY. FLAG = net carries counted intermediate planes larger than any grid "
                "the generator can emit (sampled + bundled/arc-gen max).\n\n")
        f.write("`est_saving_bytes` = sum over oversized planes of "
                "`bytes * (1 - bound^2/dim^2)`. `est_pts` = ln(M/(M-S)) with "
                "M = current mem+params, S = est saving capped at counted mem. "
                "Both are RANKING heuristics, NOT promises: only counted ENTRY reads "
                "benefit; free-input walk-einsum planes do not (task077 refutation), "
                "and shrinking a plane usually needs Slice/Pad plumbing that costs "
                "params. Verify per-hit before touching a net.\n\n")
        f.write("Sampling caveat: gen_bound from ~300 samples + bundled max; conditional "
                "branches can exceed it (see grid_crop_bounds.md) -- re-verify the bound "
                "with a large fresh sample before landing any crop.\n\n")
        f.write(f"- tasks scanned: {len(rows)}\n")
        f.write(f"- FLAGGED: {len(flagged)} "
                f"(excluding already_handled: "
                f"{sum(1 for r in flagged if 'already_handled' not in r['status'])})\n")
        f.write(f"- generator sampling failures/timeouts (bound may be underestimated): "
                f"{len(gen_fail)}\n\n")
        f.write("## Flagged tasks (ranked by est_pts, then est_saving_bytes)\n\n")
        f.write("| rank | task | arcid | net_dim | gen_bound | gen_in | gen_out | "
                "bundled(i/o) | #planes | oversized_B | est_save_B | mem+params | "
                "est_pts | top ops | status |\n")
        f.write("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|\n")
        for k, r in enumerate(flagged):
            mp = (f"{r.get('net_static_mem')}+{r.get('net_params')}"
                  if r.get("net_static_mem") is not None else "?")
            f.write(f"| {k+1} | task{r['task']:03d} | {r['arcid']} | "
                    f"{r['net_max_spatial']} | {r['gen_bound']} | "
                    f"{r['gen_in'][0]}x{r['gen_in'][1]} | {r['gen_out'][0]}x{r['gen_out'][1]} | "
                    f"{r['bundled_in']}/{r['bundled_out']} | {r['n_oversized_planes']} | "
                    f"{r['oversized_bytes']} | {int(r['est_saving_bytes'])} | {mp} | "
                    f"{r.get('est_points_delta', 0.0):.3f} | "
                    f"{ops_by_task.get(r['task'], '?')} | {r['status']} |\n")
        if gen_fail:
            f.write("\n## Generator sampling failures / timeouts (blind spots)\n\n")
            f.write("Bounds for these tasks come from partial samples + bundled data and "
                    "may be underestimates -- treat their flags with suspicion.\n\n")
            f.write("| task | arcid | sample_n | status |\n|---|---|---|---|\n")
            for r in gen_fail:
                f.write(f"| task{r['task']:03d} | {r['arcid']} | {r['sample_n']} | "
                        f"{r['status']} |\n")
    return flagged, gen_fail


def main():
    if "--report-only" in sys.argv:
        rows = json.load(open("reports/crop_bounds_scan.json"))
        enrich_with_points(rows)
        json.dump(rows, open("reports/crop_bounds_scan.json", "w"), indent=1)
        flagged, gen_fail = write_report(rows)
        print(f"report regenerated: flagged={len(flagged)} gen_fail={len(gen_fail)}")
        return

    mapping = json.load(open("reports/arc_mapping.json"))
    rows = []
    t_start = time.time()
    for i in range(1, 401):
        arc_id = mapping[str(i)]["arc_id"]
        gen = sample_generator(arc_id)
        bnd = bundled_max(i)
        netpath = f"networks/task{i:03d}.onnx"
        if os.path.exists(netpath):
            max_dim, planes, static_mem, params, net_status = analyze_net(netpath)
        else:
            max_dim, planes, static_mem, params, net_status = (None,) * 4 + ("net_missing",)

        gen_bound = max(gen["gen_in_h"], gen["gen_in_w"],
                        gen["gen_out_h"], gen["gen_out_w"],
                        bnd["b_in"], bnd["b_out"])

        flag = False
        oversized_bytes = 0
        est_saving = 0.0
        n_oversized = 0
        if max_dim is not None and planes is not None and gen_bound > 0:
            for p in planes:
                if p["spatial"] > gen_bound:
                    n_oversized += 1
                    oversized_bytes += p["bytes"]
                    ratio = (gen_bound * gen_bound) / (p["spatial"] * p["spatial"])
                    est_saving += p["bytes"] * (1.0 - ratio)
            flag = max_dim > gen_bound and n_oversized > 0

        status = "ok"
        if net_status not in ("ok",):
            status = net_status
        if gen["gen_status"] != "ok":
            status = (status + "|" if status != "ok" else "") + gen["gen_status"]
        if i in ALREADY_HANDLED:
            status = "already_handled" + ("|" + status if status != "ok" else "")

        rows.append({
            "task": i,
            "arcid": arc_id,
            "gen_in": [gen["gen_in_h"], gen["gen_in_w"]],
            "gen_out": [gen["gen_out_h"], gen["gen_out_w"]],
            "bundled_in": bnd["b_in"],
            "bundled_out": bnd["b_out"],
            "sample_n": gen["sample_n"],
            "gen_bound": gen_bound,
            "net_max_spatial": max_dim,
            "net_static_mem": static_mem,
            "net_params": params,
            "n_oversized_planes": n_oversized,
            "oversized_bytes": oversized_bytes,
            "est_saving_bytes": round(est_saving, 1),
            "flag": flag,
            "status": status,
            "top_planes": sorted(planes, key=lambda p: -p["bytes"])[:4] if planes else [],
        })
        if i % 25 == 0:
            el = time.time() - t_start
            print(f"  ...{i}/400  ({el:.0f}s)  flagged so far="
                  f"{sum(1 for r in rows if r['flag'])}", flush=True)

    enrich_with_points(rows)
    json.dump(rows, open("reports/crop_bounds_scan.json", "w"), indent=1)
    flagged, gen_fail = write_report(rows)

    print(f"\nDONE in {time.time()-t_start:.0f}s. "
          f"flagged={len(flagged)} gen_fail={len(gen_fail)}")
    print("wrote reports/crop_bounds_scan.json and reports/crop_bounds_scan.md")


if __name__ == "__main__":
    main()
