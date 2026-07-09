import zipfile
from pathlib import Path
from neurogolf.paths import OVERFIT_NETS, ROOT
from neurogolf.topk import find_unsigned_topk

def pack(nets_dir: Path | None = None, out: Path | None = None) -> Path:
    nets_dir = nets_dir or OVERFIT_NETS
    out = out or (ROOT / "submission.zip")
    files = sorted(nets_dir.glob("task*.onnx"))
    if len(files) != 400:
        raise SystemExit(f"pack REFUSED: {len(files)} nets (400 필요)")
    offenders = [f"{f.name}: {o}" for f in files for o in find_unsigned_topk(f)]
    if offenders:
        raise SystemExit("pack REFUSED, unsigned TopK:\n" + "\n".join(offenders))
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as z:
        for f in files:
            z.write(f, arcname=f.name)
    return out
