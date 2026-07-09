import json
from neurogolf.paths import STATE

def _path():
    return STATE / "manifest.json"

def load() -> dict:
    return json.load(open(_path())) if _path().exists() else {}

def save(m: dict) -> None:
    json.dump(m, open(_path(), "w"), indent=1, sort_keys=True)

def update_row(task_num: int, row: dict) -> None:
    m = load(); m[f"{task_num:03d}"] = row; save(m)

def total_points(m: dict) -> float:
    return sum(r["points"] for r in m.values())
