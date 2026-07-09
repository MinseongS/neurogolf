import subprocess
from datetime import datetime, timezone
from pathlib import Path
from neurogolf.paths import ROOT, STATE

COMP = "neurogolf-2026"

def latest_submissions(n: int = 5) -> str:
    proc = subprocess.run(["kaggle", "competitions", "submissions", "-c", COMP],
                          text=True, capture_output=True, check=True)
    return "\n".join(proc.stdout.splitlines()[: n + 2])

def submit(message: str, zip_path: Path | None = None) -> None:
    zip_path = zip_path or (ROOT / "submission.zip")
    if zip_path.name != "submission.zip":
        raise SystemExit("제출 파일명은 submission.zip이어야 함")
    print("=== 최근 제출 (병렬 세션 확인) ===\n" + latest_submissions())
    subprocess.run(["kaggle", "competitions", "submit", "-c", COMP,
                    "-f", str(zip_path), "-m", message], check=True)
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%MZ")
    log = STATE / "submissions.md"
    log.write_text(log.read_text() + f"\n| {ts} | {message} |")
