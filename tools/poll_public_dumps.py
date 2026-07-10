#!/usr/bin/env python3
"""Poll Kaggle for NEW/updated neurogolf-2026 public notebooks and auto-mine them.

Historically our biggest gains came from public dumps (biohack44 +0.7, urad bundle,
franksunp tails). This poller does the full check -> download -> mine loop and reports
any strictly-cheaper bundled-fail=0 win against the deployed nets. It NEVER adopts or
submits automatically — a positive hit is surfaced for a human to gate/adopt/submit.

Run:  uv run python tools/poll_public_dumps.py           # one poll cycle
State: state/seen_kernels.json  (ref -> lastRunTime seen); first run seeds it silently.
Downloads: candidates/public_dumps/poll_<date>/<author_slug>/

Idempotent: only downloads a kernel whose lastRunTime changed since last seen.
Designed to be run on a schedule (cron / scheduled agent) until the 07-15 deadline.
"""
import json
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
SEEN = REPO / "state" / "seen_kernels.json"
COMP = "neurogolf-2026"
PAGES = 3  # kernels list pages to scan (25/page, sorted by dateRun)


def sh(args, timeout=300):
    return subprocess.run(args, capture_output=True, text=True, timeout=timeout)


def list_kernels():
    """Return {ref: lastRunTime} for the most-recently-run kernels."""
    out = {}
    for page in range(1, PAGES + 1):
        r = sh(["kaggle", "kernels", "list", "--competition", COMP,
                "--sort-by", "dateRun", "--page-size", "25", "-p", str(page), "--csv"])
        if r.returncode != 0:
            continue
        lines = [ln for ln in r.stdout.splitlines() if ln.strip()]
        if len(lines) < 2 or not lines[0].startswith("ref,"):
            continue
        # Columns: ref,title,author,lastRunTime,totalVotes. title/author may contain
        # commas, but ref is always col 0 and lastRunTime is always 2nd-from-last.
        for ln in lines[1:]:
            parts = ln.split(",")
            if len(parts) < 5:
                continue
            out[parts[0].strip()] = parts[-2].strip()
    return out


def mine(zips):
    r = sh(["uv", "run", "ng", "mine-public", "--margin", "0", *map(str, zips)], timeout=1800)
    return r.stdout + r.stderr


def main():
    SEEN.parent.mkdir(parents=True, exist_ok=True)
    seen = json.loads(SEEN.read_text()) if SEEN.exists() else {}
    first_run = not SEEN.exists()

    current = list_kernels()
    if not current:
        print("poll: kaggle kernels list returned nothing (auth/network?) — no-op")
        return 0

    changed = [ref for ref, t in current.items() if seen.get(ref) != t]
    if first_run:
        SEEN.write_text(json.dumps(current, indent=1))
        print(f"poll: seeded seen-list with {len(current)} kernels (no mining on first run)")
        return 0
    if not changed:
        print(f"poll: no new/updated kernels ({len(current)} scanned) — nothing to mine")
        return 0

    print(f"poll: {len(changed)} new/updated kernel(s): {changed}")
    dl_root = REPO / "candidates" / "public_dumps" / "poll_latest"
    zips = []
    for ref in changed:
        slug = ref.replace("/", "_")
        dest = dl_root / slug
        dest.mkdir(parents=True, exist_ok=True)
        r = sh(["kaggle", "kernels", "output", ref, "-p", str(dest)], timeout=600)
        z = dest / "submission.zip"
        if r.returncode != 0 and not z.exists():
            print(f"  {ref}: download failed ({r.stderr.strip()[:80]})")
            continue
        if z.exists():
            zips.append(z)
            print(f"  downloaded {ref} -> {z}")
        else:
            print(f"  {ref}: no submission.zip in output (skipped)")

    if zips:
        print("poll: mining", len(zips), "dump(s)...")
        result = mine(zips)
        print(result)
        if "TOTAL adoptable: +0.0" not in result:
            print("poll: *** POSSIBLE WIN — review above, gate/adopt manually ***")

    # commit the advanced seen-list only after a successful cycle
    SEEN.write_text(json.dumps(current, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
