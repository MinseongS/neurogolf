"""Mine public Kaggle notebook bundles for per-task nets cheaper than ours.

S15 routine (2026-07-06). Public uploaders embed full 400-net bundles as base64 in
their notebooks. Even when a dump's TOTAL is below ours, individual tasks can be
cheaper (score is a SUM). This pulls dumps, extracts nets, grader-measures the best
source per task, diffs vs our manifest, and emits an adopt candidate list. Fresh-gate
+ adopt is still manual (safe rule below).

Usage:
  # 1. find newest dumps
  kaggle kernels list -s neurogolf --sort-by dateRun | head -30
  # 2. pull the ones you want into a dir (one subdir each), then:
  uv run python -m reports.scripts.mine_public_bundles /path/to/pulled_kernels_dir

Extraction handles both embeddings seen so far:
  - split-string tuple assigned to B64 / PAYLOAD_B64 (urad, prvsiyan): exec the cell
  - single long base64 literal (franksunp, ryosuke): regex the literal
Some notebooks attach a Kaggle DATASET instead of embedding (uditjain) -> not handled;
pull the dataset separately if wanted.

SAFE ADOPT RULE (private-LB final = 2026-07-15):
  adopt source net for task T iff  bundled fail==0  AND  grader cost < ours
  AND  fresh_verify cand_fail <= our incumbent's fresh_fail  (N>=1500).
  REJECT if cand fresh-fails MORE than incumbent (public +pts but private->0).
  PERSISTENT OVERFIT TRAPS (every public cheap net fresh-fails): task 319, 48, 285. Keep ours.
"""
import base64, glob, io, json, math, os, re, sys, zipfile
import onnx
from src.harness import load_task, evaluate

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def extract_bundle(nb_path, out_dir):
    nb = json.load(open(nb_path))
    for c in nb["cells"]:
        if c["cell_type"] != "code":
            continue
        src = "".join(c["source"])
        raw = None
        # form A: exec assignment of B64/PAYLOAD_B64 tuple-of-strings
        if re.search(r"\b(B64|PAYLOAD_B64|BUNDLE\w*)\s*=", src):
            try:
                ns = {}
                exec(src, ns)
                for v in ns.values():
                    if isinstance(v, str) and len(v) > 1000:
                        try:
                            d = base64.b64decode(v)
                            if d[:2] == b"PK":
                                raw = d
                                break
                        except Exception:
                            pass
            except Exception:
                pass
        # form B: single long literal
        if raw is None:
            for lit in re.findall(r"['\"]([A-Za-z0-9+/=]{2000,})['\"]", src):
                try:
                    d = base64.b64decode(lit)
                    if d[:2] == b"PK":
                        raw = d
                        break
                except Exception:
                    pass
        if raw is not None:
            os.makedirs(out_dir, exist_ok=True)
            n = 0
            with zipfile.ZipFile(io.BytesIO(raw)) as z:
                for nm in z.namelist():
                    if nm.endswith(".onnx"):
                        open(f"{out_dir}/{os.path.basename(nm)}", "wb").write(z.read(nm))
                        n += 1
            return n
    return 0


def main():
    pulled = sys.argv[1]
    man = json.load(open("reports/manifest.json"))["tasks"]
    src_root = os.path.join(pulled, "_extracted")
    sources = {}
    for d in sorted(glob.glob(f"{pulled}/*")):
        nbf = glob.glob(f"{d}/*.ipynb")
        if not nbf:
            continue
        name = os.path.basename(d)
        out = f"{src_root}/{name}"
        n = extract_bundle(nbf[0], out)
        print(f"{name}: extracted {n} nets", file=sys.stderr)
        if n:
            sources[name] = out
    # grader-measure best source per task, diff vs manifest
    cands = []
    for t in range(1, 401):
        task = load_task(t)
        mine = man[str(t)]["memory"] + man[str(t)]["params"]
        best = None
        for s, d in sources.items():
            p = f"{d}/task{t:03d}.onnx"
            if not os.path.exists(p):
                continue
            # cheap byte prefilter before the expensive grader eval
            if os.path.getsize(p) > os.path.getsize(f"networks/task{t:03d}.onnx"):
                continue
            try:
                r = evaluate(onnx.load(p), task)
            except Exception:
                continue
            if r["fail"] != 0:
                continue
            c = r["memory"] + r["params"]
            if best is None or c < best[0]:
                best = (c, s)
        if best and best[0] < mine:
            cands.append({"task": t, "mine": mine, "cost": best[0],
                          "source": best[1], "dpts": round(math.log(mine) - math.log(best[0]), 3)})
    cands.sort(key=lambda x: -x["dpts"])
    json.dump(cands, open("reports/public_bundle_candidates.json", "w"), indent=1)
    tot = sum(c["dpts"] for c in cands)
    print(f"\n{len(cands)} candidates (bundled fail=0, cheaper), potential +{tot:.2f} pts")
    print("wrote reports/public_bundle_candidates.json  (then fresh-gate each: fresh_verify T <shim> 1500)")
    for c in cands[:40]:
        print(f"  task{c['task']:03d}: {c['mine']:>7} -> {c['cost']:>7}  +{c['dpts']:.3f}  ({c['source'].split('_')[0]})")


if __name__ == "__main__":
    main()
