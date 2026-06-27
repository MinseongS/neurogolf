"""Public-LB probe harness for NeuroGolf.

This is intentionally different from the normal exact/fresh adoption workflow.
It manages *risky* candidate ONNX replacements and uses Kaggle publicScore as the
oracle.  The current best networks remain recoverable; every probe zip is built
from `networks/task*.onnx` plus explicit per-task overrides.

Examples:
  PYTHONPATH=. .venv/bin/python reports/public_probe.py init
  PYTHONPATH=. .venv/bin/python reports/public_probe.py add --task 10 --onnx /tmp/c.onnx --id t010_alt --note "alt source"
  PYTHONPATH=. .venv/bin/python reports/public_probe.py pack --ids t010_alt
  PYTHONPATH=. .venv/bin/python reports/public_probe.py submit --ids t010_alt --message "probe t010_alt"
  PYTHONPATH=. .venv/bin/python reports/public_probe.py poll
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
import subprocess
import time
import zipfile
from pathlib import Path

import onnx

ROOT = Path(__file__).resolve().parents[1]
REGISTRY = ROOT / "reports" / "public_probe_registry.json"
PROBE_DIR = ROOT / "reports" / "probes"
CAND_DIR = PROBE_DIR / "candidates"
ZIP_DIR = PROBE_DIR / "zips"
COMPETITION = "neurogolf-2026"
KAGGLE = Path("/opt/homebrew/Caskroom/miniconda/base/bin/kaggle")


def now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_registry() -> dict:
    if not REGISTRY.exists():
        return {
            "created_at": now(),
            "competition": COMPETITION,
            "baseline": current_baseline(),
            "candidates": {},
            "packs": {},
            "submissions": {},
        }
    return json.loads(REGISTRY.read_text())


def save_registry(reg: dict) -> None:
    REGISTRY.parent.mkdir(parents=True, exist_ok=True)
    REGISTRY.write_text(json.dumps(reg, indent=2, sort_keys=True) + "\n")


def current_baseline() -> dict:
    files = sorted((ROOT / "networks").glob("task*.onnx"))
    h = hashlib.sha256()
    for p in files:
        h.update(p.name.encode())
        h.update(b"\0")
        h.update(p.read_bytes())
        h.update(b"\0")
    anchor_path = ROOT / "reports" / "lb_anchor.json"
    anchor = json.loads(anchor_path.read_text()) if anchor_path.exists() else {}
    return {
        "task_count": len(files),
        "networks_tree_sha256": h.hexdigest(),
        "best_lb": anchor.get("lb"),
        "stored_at_submit": anchor.get("stored_at_submit"),
        "anchor_ref": anchor.get("submission_ref"),
        "captured_at": now(),
    }


def normalize_task(task: int) -> str:
    if task < 1 or task > 400:
        raise SystemExit(f"task out of range: {task}")
    return f"task{task:03d}.onnx"


def validate_onnx(path: Path) -> None:
    model = onnx.load(path)
    onnx.checker.check_model(model, full_check=True)
    if len(model.graph.input) != 1 or len(model.graph.output) != 1:
        raise SystemExit(f"{path} is not a single-input/single-output model")


def cmd_init(args: argparse.Namespace) -> None:
    PROBE_DIR.mkdir(exist_ok=True)
    CAND_DIR.mkdir(parents=True, exist_ok=True)
    ZIP_DIR.mkdir(parents=True, exist_ok=True)
    reg = load_registry()
    reg["baseline"] = current_baseline()
    save_registry(reg)
    print(f"initialized {REGISTRY}")
    print(json.dumps(reg["baseline"], indent=2))


def cmd_add(args: argparse.Namespace) -> None:
    reg = load_registry()
    src = Path(args.onnx).resolve()
    if not src.exists():
        raise SystemExit(f"candidate not found: {src}")
    validate_onnx(src)
    task_name = normalize_task(args.task)
    cid = args.id or f"t{args.task:03d}_{int(time.time())}"
    if cid in reg["candidates"] and not args.replace:
        raise SystemExit(f"candidate id exists: {cid} (use --replace)")
    CAND_DIR.mkdir(parents=True, exist_ok=True)
    dst = CAND_DIR / f"{cid}.onnx"
    shutil.copyfile(src, dst)
    base_path = ROOT / "networks" / task_name
    rec = {
        "id": cid,
        "task": args.task,
        "task_name": task_name,
        "path": str(dst.relative_to(ROOT)),
        "sha256": sha256(dst),
        "bytes": dst.stat().st_size,
        "base_sha256": sha256(base_path) if base_path.exists() else None,
        "base_bytes": base_path.stat().st_size if base_path.exists() else None,
        "note": args.note or "",
        "source": args.source or "manual",
        "status": "queued",
        "created_at": now(),
        "submissions": [],
    }
    reg["candidates"][cid] = rec
    save_registry(reg)
    print(json.dumps(rec, indent=2))


def selected_candidates(reg: dict, ids: list[str]) -> list[dict]:
    if not ids:
        ids = [cid for cid, c in reg["candidates"].items() if c.get("status") == "queued"]
    missing = [cid for cid in ids if cid not in reg["candidates"]]
    if missing:
        raise SystemExit(f"unknown candidate ids: {', '.join(missing)}")
    by_task: dict[int, dict] = {}
    for cid in ids:
        c = reg["candidates"][cid]
        task = int(c["task"])
        if task in by_task:
            raise SystemExit(f"multiple candidates for task {task}: {by_task[task]['id']} and {cid}")
        by_task[task] = c
    return [by_task[k] for k in sorted(by_task)]


def build_zip(cands: list[dict], label: str | None = None) -> tuple[Path, dict]:
    ZIP_DIR.mkdir(parents=True, exist_ok=True)
    label = label or time.strftime("%Y%m%d_%H%M%S", time.gmtime())
    out = ZIP_DIR / f"probe_{label}.zip"
    overrides = {c["task_name"]: ROOT / c["path"] for c in cands}
    files = sorted((ROOT / "networks").glob("task*.onnx"))
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as z:
        for p in files:
            src = overrides.get(p.name, p)
            z.write(src, p.name)
    meta = {
        "zip": str(out.relative_to(ROOT)),
        "sha256": sha256(out),
        "bytes": out.stat().st_size,
        "candidate_ids": [c["id"] for c in cands],
        "tasks": [c["task"] for c in cands],
        "created_at": now(),
    }
    return out, meta


def cmd_pack(args: argparse.Namespace) -> None:
    reg = load_registry()
    cands = selected_candidates(reg, args.ids)
    out, meta = build_zip(cands, args.label)
    reg["packs"][meta["sha256"]] = meta
    save_registry(reg)
    print(json.dumps(meta, indent=2))


def kaggle_cmd(*parts: str) -> str:
    cmd = [str(KAGGLE), *parts]
    return subprocess.check_output(cmd, cwd=ROOT, text=True, stderr=subprocess.STDOUT)


def cmd_submit(args: argparse.Namespace) -> None:
    reg = load_registry()
    cands = selected_candidates(reg, args.ids)
    out, meta = build_zip(cands, args.label)
    message = args.message or f"public-probe {'+'.join(meta['candidate_ids'])}"
    print(f"submitting {out} ({meta['sha256'][:12]}) message={message!r}")
    text = kaggle_cmd("competitions", "submit", "-c", COMPETITION, "-f", str(out), "-m", message)
    print(text)
    meta["message"] = message
    reg["packs"][meta["sha256"]] = meta
    # Kaggle submit output does not reliably include the ref; poll records it later.
    sid = meta["sha256"]
    reg["submissions"][sid] = {
        "pack_sha256": meta["sha256"],
        "candidate_ids": meta["candidate_ids"],
        "message": message,
        "submitted_at": now(),
        "status": "submitted",
        "raw_submit_output": text,
    }
    for c in cands:
        reg["candidates"][c["id"]].setdefault("submissions", []).append(sid)
        reg["candidates"][c["id"]]["status"] = "submitted"
    save_registry(reg)


def parse_submissions_csv() -> list[dict]:
    try:
        text = kaggle_cmd("competitions", "submissions", "-c", COMPETITION, "--csv")
    except subprocess.CalledProcessError:
        text = kaggle_cmd("competitions", "submissions", "-c", COMPETITION)
        print(text)
        return []
    rows = list(csv.DictReader(text.splitlines()))
    return rows


def cmd_poll(args: argparse.Namespace) -> None:
    reg = load_registry()
    rows = parse_submissions_csv()
    if not rows:
        print("no csv rows parsed")
        return
    print("latest submissions:")
    for r in rows[: args.limit]:
        print(r)
    # Match by message when possible.
    for sid, sub in reg.get("submissions", {}).items():
        msg = sub.get("message", "")
        for r in rows:
            if r.get("description") == msg or r.get("Description") == msg:
                sub["kaggle_row"] = r
                sub["status"] = r.get("status") or r.get("Status") or sub.get("status")
                score = r.get("publicScore") or r.get("PublicScore") or r.get("score")
                if score not in (None, ""):
                    try:
                        sub["publicScore"] = float(score)
                    except ValueError:
                        pass
                sub["polled_at"] = now()
                break
    save_registry(reg)


def cmd_list(args: argparse.Namespace) -> None:
    reg = load_registry()
    print("baseline:", json.dumps(reg.get("baseline", {}), sort_keys=True))
    for cid, c in sorted(reg.get("candidates", {}).items()):
        print(
            f"{cid:24s} task={c['task']:03d} status={c.get('status')} "
            f"bytes={c.get('bytes')} note={c.get('note','')}"
        )
    if args.submissions:
        print("\nsubmissions:")
        for sid, s in sorted(reg.get("submissions", {}).items()):
            print(f"{sid[:12]} status={s.get('status')} score={s.get('publicScore')} msg={s.get('message')}")


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)

    q = sub.add_parser("init")
    q.set_defaults(func=cmd_init)

    q = sub.add_parser("add")
    q.add_argument("--task", type=int, required=True)
    q.add_argument("--onnx", required=True)
    q.add_argument("--id")
    q.add_argument("--note")
    q.add_argument("--source")
    q.add_argument("--replace", action="store_true")
    q.set_defaults(func=cmd_add)

    q = sub.add_parser("pack")
    q.add_argument("--ids", nargs="*", default=[])
    q.add_argument("--label")
    q.set_defaults(func=cmd_pack)

    q = sub.add_parser("submit")
    q.add_argument("--ids", nargs="*", default=[])
    q.add_argument("--label")
    q.add_argument("--message")
    q.set_defaults(func=cmd_submit)

    q = sub.add_parser("poll")
    q.add_argument("--limit", type=int, default=8)
    q.set_defaults(func=cmd_poll)

    q = sub.add_parser("list")
    q.add_argument("--submissions", action="store_true")
    q.set_defaults(func=cmd_list)

    args = p.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
