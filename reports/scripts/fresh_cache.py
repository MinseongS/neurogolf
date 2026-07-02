"""Fresh-instance cache: generate arc-gen instances ONCE per task, reuse everywhere.

Generation (pure-Python generators + rejection sampling) dominates fresh-gate
wall-clock, and every re-verification regenerates from scratch. This tool
pre-generates N instances and stores the raw grids compactly; fresh_verify.py
--cache then loads them in milliseconds.

  PYTHONPATH=. .venv/bin/python reports/scripts/fresh_cache.py TASK N
  PYTHONPATH=. .venv/bin/python reports/scripts/fresh_cache.py TASK N --append

Cache: reports/fresh_cache/taskNNN.npz — int8 grids padded to 30x30 with -1
(inputs, outputs, heights, widths). A 5s SIGALRM watchdog skips pathological
generator seeds (task76-style rejection-sampling hangs).

PROTOCOL NOTE: use the cache for fast candidate iteration and re-verification;
the FINAL adoption gate should still include one uncached fresh_verify run
(smaller N is fine) so candidates can't overfit a fixed sample.
"""

import json
import os
import signal
import sys

import numpy as np

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, _ROOT)
_ARCGEN = os.path.join(_ROOT, "arc-gen")
sys.path.append(_ARCGEN)

TASK = int(sys.argv[1])
N = int(sys.argv[2]) if len(sys.argv) > 2 else 2500
APPEND = "--append" in sys.argv

mapping = json.load(open(os.path.join(_ROOT, "reports", "arc_mapping.json")))
arc = mapping[str(TASK)]["arc_id"]
import importlib

gen = importlib.import_module(f"tasks.task_{arc}")

CACHE_DIR = os.path.join(_ROOT, "reports", "fresh_cache")
os.makedirs(CACHE_DIR, exist_ok=True)
path = os.path.join(CACHE_DIR, f"task{TASK:03d}.npz")


class _Timeout(Exception):
    pass


def _alarm(signum, frame):
    raise _Timeout()


signal.signal(signal.SIGALRM, _alarm)

ins, outs, hs, ws, ohs, ows = [], [], [], [], [], []
if APPEND and os.path.exists(path):
    z = np.load(path)
    ins, outs = list(z["inputs"]), list(z["outputs"])
    hs, ws = list(z["heights"]), list(z["widths"])
    ohs = list(z["oheights"]) if "oheights" in z else list(z["heights"])
    ows = list(z["owidths"]) if "owidths" in z else list(z["widths"])

skipped = attempts = 0
while len(ins) < N:
    attempts += 1
    if attempts > N * 20:
        break
    signal.alarm(5)
    try:
        ex = gen.generate()
    except _Timeout:
        skipped += 1
        continue
    except Exception:
        continue
    finally:
        signal.alarm(0)
    gi, go = np.array(ex["input"], dtype=np.int8), np.array(ex["output"], dtype=np.int8)
    if max(gi.shape) > 30 or max(go.shape) > 30:
        continue
    if gi.shape != go.shape:
        # keep only same-shape pairs padded together; RESHAPE tasks store both shapes
        pass
    pi = np.full((30, 30), -1, np.int8)
    po = np.full((30, 30), -1, np.int8)
    pi[: gi.shape[0], : gi.shape[1]] = gi
    po[: go.shape[0], : go.shape[1]] = go
    ins.append(pi)
    outs.append(po)
    hs.append(gi.shape[0])
    ws.append(gi.shape[1])
    ohs.append(go.shape[0])
    ows.append(go.shape[1])

np.savez_compressed(
    path,
    inputs=np.stack(ins),
    outputs=np.stack(outs),
    heights=np.array(hs, np.int16),
    widths=np.array(ws, np.int16),
    oheights=np.array(ohs, np.int16),
    owidths=np.array(ows, np.int16),
)
print(f"task{TASK:03d}: cached {len(ins)} instances -> {path} (skipped {skipped} stuck seeds)")
