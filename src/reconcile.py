"""Reconcile src/custom/*.py against the LIVE installed nets (networks/ + manifest).

The single source of truth for what is LIVE is manifest.json's `method` field
(`ext:*` = external/public net, `custom:N` = your hand-built net). A custom .py
file existing does NOT mean it is installed. Rebases / public-artifact merges can
silently DISPLACE a winning custom with a worse external net — this scan catches
that (RECOVER bucket) so you never lose a gated win by accident.

Usage:  python -m src.reconcile            # report all 4 buckets
        python -m src.reconcile --adopt    # also re-adopt every RECOVER candidate

Buckets:
  LIVE    custom is installed (method=custom:N)        — your real contribution
  RECOVER custom stored > installed points             — FREE POINTS, fresh-gate+adopt
  DEAD    external net scores higher, custom shelved    — keep as documentation
  BROKEN  custom build/eval fails                       — fix or delete
"""
import json, glob, re, importlib, contextlib, io, sys, subprocess

from src import harness


def scan():
    man = json.load(open("reports/manifest.json"))["tasks"]
    files = [f for f in sorted(glob.glob("src/custom/task*.py"))
             if re.search(r"task(\d{3})\.py$", f)]
    cats = {"LIVE": [], "RECOVER": [], "DEAD": [], "DOC": [], "BROKEN": []}
    for f in files:
        m = re.search(r"task(\d{3})\.py$", f)
        if not m:
            continue
        N = int(m.group(1))
        live = man.get(str(N)) or {}
        lp, meth = live.get("points", 0.0), live.get("method", "?")
        mod = importlib.import_module(f"src.custom.task{N:03d}")
        importlib.reload(mod)
        if not hasattr(mod, "build"):
            # intentional infeasible-declaration doc (no build), not a failure
            cats["DOC"].append((N, meth)); continue
        try:
            with contextlib.redirect_stdout(io.StringIO()), \
                 contextlib.redirect_stderr(io.StringIO()):
                ev = harness.evaluate(mod.build(None), harness.load_task(N))
            if not ev["ok"]:
                cats["BROKEN"].append((N, "eval-fail")); continue
            cp = ev["points"]
        except Exception as e:
            cats["BROKEN"].append((N, str(e)[:50])); continue
        if cp > lp + 0.01:
            cats["RECOVER"].append((N, cp, lp, cp - lp, meth))
        elif meth.startswith("custom"):
            cats["LIVE"].append((N, cp, meth))
        else:
            cats["DEAD"].append((N, cp, lp, meth))
    return cats


def main():
    cats = scan()
    for k in ("LIVE", "RECOVER", "DEAD", "DOC", "BROKEN"):
        print(f"  {k:<8}: {len(cats[k])}")
    print("\nRECOVER (custom > installed — fresh-gate & re-adopt):")
    for N, cp, lp, d, meth in sorted(cats["RECOVER"], key=lambda x: -x[3]):
        print(f"  task{N:>3}  custom {cp:5.2f} vs {lp:5.2f}  (+{d:.2f})  {meth}")
    for N, e in cats["BROKEN"]:
        print(f"  BROKEN task{N}: {e}")
    json.dump(cats, open("reports/custom_reconcile.json", "w"), default=str, indent=0)
    if "--adopt" in sys.argv:
        for N, *_ in sorted(cats["RECOVER"], key=lambda x: -x[3]):
            print(f"\n--- adopt {N} ---")
            subprocess.run([sys.executable, "-m", "src.adopt", str(N)])


if __name__ == "__main__":
    main()
