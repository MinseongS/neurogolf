import json
from datetime import datetime, timezone
from pathlib import Path
from neurogolf.paths import CANDIDATES

WORKLISTS = CANDIDATES / "worklists"

from neurogolf.scans.mask_dominance import scan_all as _mask_dominance
from neurogolf.scans.kernel_collapse import scan_all as _kernel_collapse
from neurogolf.scans.fold import scan_all as _fold
from neurogolf.scans.dtype_overpay import scan_all as _dtype_overpay
from neurogolf.scans.public_autopsy import scan_all as _public_autopsy
from neurogolf.scans.canvas_crop_shrink import scan_all as _canvas_crop_shrink

SCANNERS = {
    "mask_dominance": _mask_dominance,
    "kernel_collapse": _kernel_collapse,
    "fold": _fold,
    "dtype_overpay": _dtype_overpay,
    "public_autopsy": _public_autopsy,
    "canvas_crop_shrink": _canvas_crop_shrink,
}


def run_scan(name: str, tasks: list[int] | None = None) -> Path:
    if name not in SCANNERS:
        raise SystemExit(f"unknown scanner '{name}' — 등록: {sorted(SCANNERS)}")
    result = SCANNERS[name](tasks)
    WORKLISTS.mkdir(parents=True, exist_ok=True)
    out = WORKLISTS / f"{name}.json"
    payload = {"lever": name, "generated": datetime.now(timezone.utc).isoformat(),
               "items": sorted(result["items"], key=lambda i: -i.get("expected_gain", 0.0))}
    out.write_text(json.dumps(payload, indent=1))
    return out


def show_queue() -> None:
    items = []
    for f in WORKLISTS.glob("*.json") if WORKLISTS.exists() else []:
        d = json.loads(f.read_text())
        items += [{**i, "lever": d["lever"]} for i in d["items"]]
    for i in sorted(items, key=lambda i: -i.get("expected_gain", 0.0))[:40]:
        print(f"task{i['task']:03d}  +{i.get('expected_gain', 0):.3f}  {i['lever']}")
