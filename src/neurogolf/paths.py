import os, sys
from pathlib import Path


def find_root(start: Path | None = None) -> Path:
    env = os.environ.get("NEUROGOLF_ROOT")
    if env:
        return Path(env)
    cur = (start or Path.cwd()).resolve()
    for p in (cur, *cur.parents):
        if (p / "submission" / "overfit_nets").is_dir() and (p / "pyproject.toml").exists():
            return p
    raise SystemExit("neurogolf 루트를 찾을 수 없음 — repo 안에서 실행하거나 NEUROGOLF_ROOT 설정")


ROOT = find_root(Path(__file__).resolve().parent)
OVERFIT_NETS = ROOT / "submission" / "overfit_nets"
STATE = ROOT / "state"
CANDIDATES = ROOT / "candidates"
DATA = ROOT / "data"
PLAYBOOK = ROOT / "playbook"


def ensure_src_importable() -> None:
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
