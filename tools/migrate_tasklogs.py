"""reports/tasklog/ -> state/tasks/ 이주. stale 휴리스틱: 배포본 cost가 본문 숫자에 없으면 stale-likely."""
import json, re, sys
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

def stale_verdict(body: str, deployed_cost: int | None) -> str:
    if deployed_cost is None:
        return "unknown"
    nums = {int(n) for n in re.findall(r"\b(\d{2,7})\b", body)}
    return "match" if deployed_cost in nums else "stale-likely"

def migrate_one(src: Path, out_dir: Path, deployed_cost: int | None) -> Path:
    body = src.read_text()
    fm = (f"---\ndeployed_cost: {deployed_cost}\n"
          f"logged_costs_match: {stale_verdict(body, deployed_cost)}\n"
          f"migrated: {date.today().isoformat()}\n---\n\n")
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / src.name
    out.write_text(fm + body)
    return out

def main() -> None:
    manifest = json.load(open(ROOT / "state" / "baseline" / "manifest.json"))
    rows = manifest["tasks"] if isinstance(manifest, dict) and "tasks" in manifest else manifest
    if isinstance(rows, dict):
        rows = list(rows.values())
    cost_by_task = {int(r["task"]): r.get("cost") for r in rows}
    src_dir, out_dir = ROOT / "reports" / "tasklog", ROOT / "state" / "tasks"
    n = 0
    for f in sorted(src_dir.glob("task*.md")):
        num = int(re.search(r"(\d+)", f.stem).group(1))
        migrate_one(f, out_dir, cost_by_task.get(num))
        n += 1
    print(f"migrated {n} tasklogs")

if __name__ == "__main__":
    main()
