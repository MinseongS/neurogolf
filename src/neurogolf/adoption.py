import hashlib, shutil
from datetime import datetime, timezone
from pathlib import Path
from neurogolf import manifest
from neurogolf.gate import gate as gate_candidate
from neurogolf.paths import OVERFIT_NETS, ROOT, STATE

BACKUPS = ROOT / "submission" / ".backups"

def adopt(
    candidate: Path,
    task_num: int,
    note: str = "",
    *,
    repair_invalid: bool = False,
    public_zero_ref: int | None = None,
) -> dict:
    res = gate_candidate(
        Path(candidate),
        task_num,
        repair_invalid=repair_invalid,
        public_zero_ref=public_zero_ref,
    )
    if not res.ok:
        raise SystemExit("gate REJECT: " + " | ".join(res.reasons))
    target = OVERFIT_NETS / f"task{task_num:03d}.onnx"
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    BACKUPS.mkdir(parents=True, exist_ok=True)
    shutil.copy2(target, BACKUPS / f"task{task_num:03d}_{ts}.onnx")
    shutil.copy2(candidate, target)
    row = {"task": task_num, "cost": res.candidate["cost"], "points": res.candidate["points"],
           "ok": True, "fail": 0, "sha256": hashlib.sha256(target.read_bytes()).hexdigest(),
           "updated": ts}
    manifest.update_row(task_num, row)
    action = (
        "REPAIRED"
        if res.repairing_invalid_topk or res.repairing_public_zero
        else "ADOPTED"
    )
    evidence = (
        f"\n- public-zero-ref: {public_zero_ref}"
        if res.repairing_public_zero
        else ""
    )
    stamp = (f"\n## {action} {ts}\n- cost: {res.incumbent_cost} -> {res.candidate['cost']}"
             f" (points {res.candidate['points']:.4f})\n- source: {candidate}\n- note: {note}\n")
    if evidence:
        stamp = stamp.rstrip() + evidence + "\n"
    log = STATE / "tasks" / f"task{task_num:03d}.md"
    log.parent.mkdir(parents=True, exist_ok=True)
    log.touch(exist_ok=True)
    log.write_text(log.read_text() + stamp)
    return row
