import hashlib, json
from concurrent.futures import ThreadPoolExecutor
from neurogolf import manifest
from neurogolf.gate import eval_isolated
from neurogolf.paths import OVERFIT_NETS, STATE

def _baseline_file():
    return STATE / "baseline" / "sha256.txt"

def hash_check() -> list[str]:
    m = manifest.load()
    expected: dict[str, str] = {f"task{k}.onnx": r["sha256"] for k, r in m.items() if r.get("sha256")}
    if not expected:
        expected = {line.split()[1]: line.split()[0]
                    for line in _baseline_file().read_text().splitlines() if line.strip()}
    bad = []
    for name, sha in sorted(expected.items()):
        p = OVERFIT_NETS / name
        if not p.exists() or hashlib.sha256(p.read_bytes()).hexdigest() != sha:
            bad.append(name)
    return bad

def full_verify(update: bool = False) -> dict:
    def one(n):
        return n, eval_isolated(OVERFIT_NETS / f"task{n:03d}.onnx", n)
    with ThreadPoolExecutor(max_workers=8) as ex:
        rows = dict(ex.map(one, range(1, 401)))
    failures = [n for n, r in rows.items() if not (r.get("ok") and r.get("fail") == 0)]
    total = sum(r.get("points") or 0.0 for r in rows.values())
    if update and not failures:
        import hashlib as h
        m = {}
        for n, r in rows.items():
            p = OVERFIT_NETS / f"task{n:03d}.onnx"
            m[f"{n:03d}"] = {"task": n, "cost": r["cost"], "points": r["points"], "ok": True,
                             "fail": 0, "sha256": h.sha256(p.read_bytes()).hexdigest(), "updated": "verify"}
        manifest.save(m)
    return {"n_ok": 400 - len(failures), "total_points": total, "failures": failures}
