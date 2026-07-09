import argparse, json, subprocess, sys
from pathlib import Path
from neurogolf import paths

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="ng", description="NeuroGolf lever engine CLI")
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("status")
    s = sub.add_parser("score"); s.add_argument("tasks", nargs="+", type=int)
    s = sub.add_parser("gate"); s.add_argument("onnx", type=Path); s.add_argument("--task", type=int, required=True)
    s = sub.add_parser("adopt"); s.add_argument("onnx", type=Path); s.add_argument("--task", type=int, required=True); s.add_argument("--note", default="")
    sub.add_parser("pack")
    s = sub.add_parser("submit"); s.add_argument("-m", "--message", required=True)
    s = sub.add_parser("scan"); s.add_argument("lever"); s.add_argument("--tasks", nargs="*", type=int)
    sub.add_parser("queue")
    s = sub.add_parser("mine-public"); s.add_argument("dumps", nargs="+"); s.add_argument("--margin", type=int, default=0); s.add_argument("--apply", action="store_true")
    s = sub.add_parser("verify"); s.add_argument("--hash", action="store_true"); s.add_argument("--update", action="store_true")
    return p

def main() -> None:
    paths.ensure_src_importable()
    args = build_parser().parse_args()
    if args.cmd == "status":
        from neurogolf import manifest, verify
        m = manifest.load()
        print(f"nets: {len(list(paths.OVERFIT_NETS.glob('task*.onnx')))}/400")
        if m: print(f"manifest total points: {manifest.total_points(m):.4f}")
        dirty = subprocess.run(["git", "status", "--porcelain"], capture_output=True, text=True, cwd=paths.ROOT).stdout
        if dirty: print(f"⚠ 미커밋 변경 {len(dirty.splitlines())}건")
        state = (paths.STATE / 'STATE.md')
        print(state.read_text()[:800] if state.exists() else "STATE.md 없음")
    elif args.cmd == "score":
        from neurogolf.gate import eval_isolated
        for t in args.tasks:
            print(t, json.dumps(eval_isolated(paths.OVERFIT_NETS / f"task{t:03d}.onnx", t)))
    elif args.cmd == "gate":
        from neurogolf.gate import gate
        r = gate(args.onnx, args.task)
        print(("PASS" if r.ok else "REJECT"), r.reasons or "", json.dumps(r.candidate))
        sys.exit(0 if r.ok else 1)
    elif args.cmd == "adopt":
        from neurogolf.adoption import adopt
        print(json.dumps(adopt(args.onnx, args.task, args.note)))
    elif args.cmd == "pack":
        from neurogolf.pack import pack
        print("packed:", pack())
    elif args.cmd == "submit":
        from neurogolf.submit import submit
        submit(args.message)
    elif args.cmd == "scan":
        from neurogolf.scans import run_scan
        print("worklist:", run_scan(args.lever, args.tasks))
    elif args.cmd == "queue":
        from neurogolf.scans import show_queue
        show_queue()
    elif args.cmd == "mine-public":
        from neurogolf.scans.minmerge import mine
        mine([Path(d) for d in args.dumps], margin=args.margin, apply=args.apply)
    elif args.cmd == "verify":
        from neurogolf import verify as v
        if args.hash:
            bad = v.hash_check()
            print("HASH-OK" if not bad else f"MUTATED: {bad}"); sys.exit(1 if bad else 0)
        r = v.full_verify(update=args.update)
        print(json.dumps(r)); sys.exit(0 if r["n_ok"] == 400 else 1)
