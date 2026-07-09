#!/usr/bin/env python3
"""Pack the active 8000-mode submission archive."""

from __future__ import annotations

import argparse
import zipfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_NETS = ROOT / "submission" / "overfit_nets"
DEFAULT_OUT = ROOT / "submission.zip"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--nets-dir", default=str(DEFAULT_NETS))
    parser.add_argument("--out", default=str(DEFAULT_OUT))
    args = parser.parse_args()

    nets_dir = Path(args.nets_dir).resolve()
    out = Path(args.out).resolve()
    files = sorted(nets_dir.glob("task*.onnx"))
    if len(files) != 400:
        raise SystemExit(f"expected 400 task*.onnx files in {nets_dir}, found {len(files)}")
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as archive:
        for path in files:
            archive.write(path, path.name)
    print(f"packed {len(files)} nets -> {out} ({out.stat().st_size / 1e6:.3f} MB)")


if __name__ == "__main__":
    main()
