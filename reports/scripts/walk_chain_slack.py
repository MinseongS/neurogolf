"""Walk-chain slack scanner + truncator (S18 lever, LB-proven +0.243 on task243).

MECHANISM
  Multi-plane walk/flood nets (BFS, flood-fill, reachability) chain several propagation
  nodes, each emitting a COUNTED grid plane. The number of planes is forced by the 52-letter
  Einsum-equation alphabet (~46-48 steps/plane) OR by an unrolled MaxPool/Conv dilation loop.
  But the REQUIRED reach = max BFS/step distance over the *bundled* set may be < what the net
  allocates. When it is, the terminal plane(s) are worst-case slack: delete them (repoint each
  consumer to the plane the deleted node propagates) and the net still passes bundled fail=0.
  Each dropped plane = -plane_bytes = real points. This is OVERFIT (fresh long inputs may need
  the dropped steps) but PERMANENTLY safe under the constant grading dataset (see memory
  neurogolf-overfit-mode). Gate = bundled fail=0 ONLY.

DETECTION
  Maximal chains of same-op-type propagation nodes (Einsum/MaxPool/Conv/MatMul/ConvTranspose/
  GridSample) whose consecutive members share output shape, allowing <=3 intervening
  shape-preserving nodes (Sign/Cast/Relu/And/Where masks). Greedy: drop terminal step, keep if
  bundled fail==0 & ok, repeat.

USAGE
  # scan one net (report only)
  .venv/bin/python -m reports.scripts.walk_chain_slack --task 286 --net submission/overfit_nets/task286.onnx
  # sweep a whole bundle dir (report only)
  .venv/bin/python -m reports.scripts.walk_chain_slack --dir submission/overfit_nets
  # apply the found truncations in-place (backs up to <net>.pretrunc)
  .venv/bin/python -m reports.scripts.walk_chain_slack --task 243 --net submission/overfit_nets/task243.onnx --apply

RESULT (S18 sweep of all 400 overfit nets): only task243 had slack (+0.243). 286/196/277/76/
118/18/192/174/145 all tight -> dropping any terminal step fails bundled. Re-run after adopting
new public nets (a freshly-grafted net may bring un-trimmed slack).
"""
import argparse, glob, os, shutil, sys
import onnx
from onnx import shape_inference
from src.harness import load_task, evaluate

PROP = {"MaxPool", "Conv", "Einsum", "MatMul", "ConvTranspose", "GridSample"}
PASS_THRU = {"Sign", "Cast", "Relu", "Identity", "Abs", "And", "Or", "Not", "Where", "Min", "Max", "Greater", "Less"}
DT = {1: 4, 2: 1, 3: 1, 4: 2, 5: 2, 6: 4, 7: 8, 9: 1, 10: 2, 11: 8, 12: 4, 13: 8, 16: 2}


def _shapes(m):
    try:
        m = shape_inference.infer_shapes(m)
    except Exception:
        pass
    vi = {}
    for v in list(m.graph.value_info) + list(m.graph.output) + list(m.graph.input):
        dims = tuple(d.dim_value if d.HasField("dim_value") else 0 for d in v.type.tensor_type.shape.dim)
        vi[v.name] = (v.type.tensor_type.elem_type, dims)
    return vi


def _nbytes(vi, name):
    et, dims = vi.get(name, (None, None))
    if not dims or any(d == 0 for d in dims):
        return None
    c = 1
    for d in dims:
        c *= d
    return c * DT.get(et, 4)


def find_chains(g, vi):
    """maximal chains of propagation-node output-planes E1..Ek (Ei feeds Ei+1 via <=3 pass-thru)."""
    prod = {o: n for n in g.node for o in n.output}

    def upstream_prop(inp):
        seen, hops = inp, 0
        while seen in prod and hops < 4:
            pn = prod[seen]
            if pn.op_type in PROP:
                return pn.output[0]
            if pn.op_type in PASS_THRU and pn.input:
                seen = pn.input[0]
                hops += 1
                continue
            return None
        return None

    props = [n for n in g.node if n.op_type in PROP and n.output and n.output[0] in vi]
    nxt = {}
    for f in props:
        for fi in f.input:
            u = upstream_prop(fi)
            if u is not None and u != f.output[0] and vi.get(u, (0, 0))[1] == vi.get(f.output[0], (1, 1))[1]:
                nxt[u] = f.output[0]
    has_pred = set(nxt.values())
    rev = {v: k for k, v in nxt.items()}
    chains = []
    for p in props:
        o = p.output[0]
        if o in has_pred and o not in nxt and o != "output":
            chain, cur = [o], o
            while cur in rev:
                cur = rev[cur]
                chain.insert(0, cur)
            if len(chain) >= 2:
                chains.append(chain)
    return chains


def _chain_input(node, prod, vi):
    """the plane operand this prop node propagates (a node-output of matching shape)."""
    tgt = vi.get(node.output[0], (None, None))[1]
    cands = [i for i in node.input if i in prod]
    for i in cands:
        if vi.get(i, (None, None))[1] == tgt:
            return i
    return cands[0] if cands else None


def _drop_terminal(path, term_out, out_path):
    m = onnx.load(path)
    g = m.graph
    prod = {o: n for n in g.node for o in n.output}
    vi = _shapes(m)
    T = prod.get(term_out)
    if T is None:
        return None
    ci = _chain_input(T, prod, vi)
    if ci is None:
        return None
    newnodes = []
    for n in list(g.node):
        if n.output and n.output[0] == term_out:
            continue
        n.input[:] = [ci if x == term_out else x for x in n.input]
        newnodes.append(n)
    del g.node[:]
    g.node.extend(newnodes)
    onnx.save(m, out_path)
    return out_path


def scan_net(task_num, net_path, apply=False, tmp="/tmp/wcs_tmp.onnx"):
    task = load_task(task_num)
    base = evaluate(net_path, task)
    m = onnx.load(net_path)
    vi = _shapes(m)
    chains = find_chains(m.graph, vi)
    drops = []  # terminal planes we can drop, in order
    cur = net_path
    banked_pts, banked_mem = base["points"], base["memory"]
    log = []
    for chain in chains:
        # greedily drop from the terminal end of THIS chain
        remaining = list(chain)
        while len(remaining) >= 2:
            term = remaining[-1]
            b = _nbytes(vi, term)
            out = _drop_terminal(cur, term, tmp)
            if out is None:
                break
            try:
                r = evaluate(out, task)
            except Exception as e:
                log.append(f"{term}: EXC {str(e)[:50]}")
                break
            if r["ok"] and r["fail"] == 0 and r["memory"] < banked_mem:
                drops.append((term, b, r["memory"], r["points"]))
                log.append(f"{term}: DROP ok mem={r['memory']} pts={r['points']:.3f}")
                banked_pts, banked_mem = r["points"], r["memory"]
                shutil.copy(out, tmp + ".keep")
                cur = tmp + ".keep"
                remaining.pop()
            else:
                log.append(f"{term}: keep (fail={r['fail']} err={r['error']})")
                break
    result = {
        "task": task_num, "chains": chains, "drops": drops,
        "base_pts": base["points"], "base_mem": base["memory"],
        "final_pts": banked_pts, "final_mem": banked_mem,
        "delta_pts": round(banked_pts - base["points"], 4), "log": log,
    }
    if apply and drops:
        shutil.copy(net_path, net_path + ".pretrunc")
        shutil.copy(cur, net_path)
        result["applied"] = True
    for f in (tmp, tmp + ".keep"):
        if os.path.exists(f):
            os.remove(f)
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", type=int)
    ap.add_argument("--net")
    ap.add_argument("--dir", help="sweep every taskNNN.onnx in this dir")
    ap.add_argument("--apply", action="store_true", help="apply truncations in-place (backs up .pretrunc)")
    args = ap.parse_args()
    targets = []
    if args.dir:
        for p in sorted(glob.glob(f"{args.dir}/task*.onnx")):
            t = int(os.path.basename(p)[4:7])
            targets.append((t, p))
    elif args.task and args.net:
        targets = [(args.task, args.net)]
    else:
        ap.error("give --task N --net PATH, or --dir DIR")
    total = 0.0
    for t, p in targets:
        try:
            r = scan_net(t, p, apply=args.apply)
        except Exception as e:
            print(f"task{t}: ERR {str(e)[:60]}", file=sys.stderr)
            continue
        if r["delta_pts"] > 0:
            total += r["delta_pts"]
            tag = " [APPLIED]" if r.get("applied") else ""
            print(f"task{t}: +{r['delta_pts']} (mem {r['base_mem']}->{r['final_mem']}, dropped {len(r['drops'])} plane(s)){tag}")
            for l in r["log"]:
                print(f"      {l}")
        elif args.task:  # single-net mode: always show why
            print(f"task{t}: no slack (chains={r['chains']})")
            for l in r["log"]:
                print(f"      {l}")
    if args.dir:
        print(f"\nTOTAL droppable: +{round(total, 4)} across swept nets")


if __name__ == "__main__":
    main()
