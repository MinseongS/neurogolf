"""canvas_crop_shrink scanner prototype.

Probe each arc-gen generator with many fresh samples, measure conservative
output extents, then cross-reference deployed ONNX graphs for counted 30x30
canvas tensors.  This is report-only: it does not build candidate nets and does
not touch submission/overfit_nets.

The intended use is the kaggloop-style crop lane from state/STATE.md:

    NG_CANVAS_CROP_SAMPLES=10000 uv run ng scan canvas_crop_shrink
    uv run python -m neurogolf.scans.canvas_crop_shrink --tasks 14 18 --samples 1000

Detailed reports are written under candidates/canvas_crop_shrink/.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import os
import random
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import onnx
import numpy as np

from neurogolf.manifest import load as load_manifest
from neurogolf.paths import CANDIDATES, OVERFIT_NETS, ROOT
from neurogolf.scoring import load_task

MAPPING = ROOT / "state" / "arc_mapping.json"
ARCGEN = ROOT / "arc-gen"
OUTDIR = CANDIDATES / "canvas_crop_shrink"


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, default))
    except ValueError:
        return default


DEFAULT_SAMPLES = _env_int("NG_CANVAS_CROP_SAMPLES", 10_000)
DEFAULT_MAX_TRIES_FACTOR = _env_int("NG_CANVAS_CROP_MAX_TRIES_FACTOR", 5)


@dataclass
class Bounds:
    bundled: int = 0
    fresh_requested: int = 0
    fresh_generated: int = 0
    skipped_large: int = 0
    generate_errors: int = 0
    input_h_max: int = 0
    input_w_max: int = 0
    output_h_max: int = 0
    output_w_max: int = 0
    occupied_h_max: int = 0
    occupied_w_max: int = 0
    occupied_bottom_max: int = -1
    occupied_right_max: int = -1
    occupied_top_min: int = 30
    occupied_left_min: int = 30
    all_zero_outputs: int = 0

    @property
    def output_area(self) -> int:
        return self.output_h_max * self.output_w_max

    @property
    def occupied_area(self) -> int:
        return self.occupied_h_max * self.occupied_w_max


def _grid_shape(grid: list[list[int]]) -> tuple[int, int]:
    if not grid:
        return 0, 0
    return len(grid), max((len(row) for row in grid), default=0)


def _update_bounds(bounds: Bounds, example: dict[str, Any], *, fresh: bool) -> None:
    inp = example["input"]
    out = example["output"]
    ih, iw = _grid_shape(inp)
    oh, ow = _grid_shape(out)
    if ih > 30 or iw > 30 or oh > 30 or ow > 30:
        bounds.skipped_large += 1
        return
    if fresh:
        bounds.fresh_generated += 1
    else:
        bounds.bundled += 1
    bounds.input_h_max = max(bounds.input_h_max, ih)
    bounds.input_w_max = max(bounds.input_w_max, iw)
    bounds.output_h_max = max(bounds.output_h_max, oh)
    bounds.output_w_max = max(bounds.output_w_max, ow)

    coords: list[tuple[int, int]] = []
    for r, row in enumerate(out):
        for c, color in enumerate(row):
            if color != 0:
                coords.append((r, c))
    if not coords:
        bounds.all_zero_outputs += 1
        return
    r0 = min(r for r, _ in coords)
    r1 = max(r for r, _ in coords)
    c0 = min(c for _, c in coords)
    c1 = max(c for _, c in coords)
    bounds.occupied_h_max = max(bounds.occupied_h_max, r1 - r0 + 1)
    bounds.occupied_w_max = max(bounds.occupied_w_max, c1 - c0 + 1)
    bounds.occupied_bottom_max = max(bounds.occupied_bottom_max, r1)
    bounds.occupied_right_max = max(bounds.occupied_right_max, c1)
    bounds.occupied_top_min = min(bounds.occupied_top_min, r0)
    bounds.occupied_left_min = min(bounds.occupied_left_min, c0)


def _generator_path(task_num: int) -> Path:
    mapping = json.loads(MAPPING.read_text())
    mapped = Path(mapping[str(task_num)]["generator"])
    path = ARCGEN / "tasks" / mapped.name
    if not path.exists():
        raise FileNotFoundError(path)
    return path


def _load_generator(task_num: int):
    if str(ARCGEN) not in sys.path:
        sys.path.insert(0, str(ARCGEN))
    path = _generator_path(task_num)
    spec = importlib.util.spec_from_file_location(
        f"canvas_crop_gen_{task_num:03d}", path
    )
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load generator for task{task_num:03d}: {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def measure_bounds(
    task_num: int,
    *,
    samples: int = DEFAULT_SAMPLES,
    seed: int = 20260709,
    max_tries_factor: int = DEFAULT_MAX_TRIES_FACTOR,
) -> tuple[Bounds | None, str | None]:
    bounds = Bounds(fresh_requested=samples)

    try:
        task = load_task(task_num)
    except Exception as e:
        return None, f"load-task:{e}"
    for group in ("train", "test", "arc-gen"):
        for ex in task.get(group, []):
            _update_bounds(bounds, ex, fresh=False)

    try:
        gen = _load_generator(task_num)
    except Exception as e:
        return None, f"load-generator:{e}"

    random.seed(seed + task_num * 1_000_003)
    tries = 0
    max_tries = max(samples * max_tries_factor, samples)
    while bounds.fresh_generated < samples and tries < max_tries:
        tries += 1
        try:
            ex = gen.generate()
        except Exception:
            bounds.generate_errors += 1
            continue
        _update_bounds(bounds, ex, fresh=True)

    if bounds.fresh_generated == 0 and samples > 0:
        return None, "no-fresh-examples"
    return bounds, None


def _shape_dims(value_info) -> tuple[int, list[int], int] | None:
    tensor_type = value_info.type.tensor_type
    if not tensor_type.HasField("shape"):
        return None
    dims = []
    for dim in tensor_type.shape.dim:
        if not dim.HasField("dim_value") or dim.dim_value <= 0:
            return None
        dims.append(dim.dim_value)
    try:
        np_dtype = onnx.helper.tensor_dtype_to_np_dtype(tensor_type.elem_type)
    except Exception:
        return None
    itemsize = int(np.dtype(np_dtype).itemsize)
    return int(tensor_type.elem_type), dims, itemsize


def _bytes_from_dims(dims: list[int], itemsize: int) -> int:
    return math.prod(dims) * itemsize


def _is_full_canvas(dims: list[int]) -> bool:
    return len(dims) >= 2 and dims[-2:] == [30, 30]


def _tensor_kind(name: str, dims: list[int], elem_type: int, bytes_: int) -> dict[str, Any]:
    return {
        "name": name,
        "shape": dims,
        "elem_type": int(elem_type),
        "bytes": int(bytes_),
    }


def analyze_model(task_num: int) -> tuple[dict[str, Any] | None, str | None]:
    path = OVERFIT_NETS / f"task{task_num:03d}.onnx"
    if not path.exists():
        return None, "missing-net"
    try:
        raw = onnx.load(path)
    except Exception as e:
        return None, f"load-net:{e}"
    try:
        model = onnx.shape_inference.infer_shapes(raw, strict_mode=False)
    except Exception:
        model = raw
    graph = model.graph
    node_outputs = {out for node in graph.node for out in node.output if out}
    io_names = {vi.name for vi in list(graph.input) + list(graph.output)}

    intermediates = []
    for vi in list(graph.value_info):
        if vi.name not in node_outputs or vi.name in io_names:
            continue
        parsed = _shape_dims(vi)
        if not parsed:
            continue
        elem_type, dims, itemsize = parsed
        if not _is_full_canvas(dims):
            continue
        intermediates.append(
            _tensor_kind(vi.name, dims, elem_type, _bytes_from_dims(dims, itemsize))
        )

    initializers = []
    for init in graph.initializer:
        dims = list(init.dims)
        if not _is_full_canvas(dims):
            continue
        try:
            np_dtype = onnx.helper.tensor_dtype_to_np_dtype(init.data_type)
            itemsize = int(np.dtype(np_dtype).itemsize)
        except Exception:
            itemsize = 1
        initializers.append(
            _tensor_kind(init.name, dims, init.data_type, _bytes_from_dims(dims, itemsize))
        )

    constants = []
    for node in graph.node:
        if node.op_type != "Constant":
            continue
        for attr in node.attribute:
            if attr.name != "value":
                continue
            dims = list(attr.t.dims)
            if not _is_full_canvas(dims):
                continue
            try:
                np_dtype = onnx.helper.tensor_dtype_to_np_dtype(attr.t.data_type)
                itemsize = int(np.dtype(np_dtype).itemsize)
            except Exception:
                itemsize = 1
            constants.append(
                _tensor_kind(
                    node.output[0] if node.output else node.name,
                    dims,
                    attr.t.data_type,
                    _bytes_from_dims(dims, itemsize),
                )
            )

    return {
        "path": str(path),
        "nodes": len(graph.node),
        "full_canvas_intermediate_count": len(intermediates),
        "full_canvas_intermediate_bytes": sum(t["bytes"] for t in intermediates),
        "full_canvas_initializer_count": len(initializers),
        "full_canvas_initializer_bytes": sum(t["bytes"] for t in initializers),
        "full_canvas_constant_count": len(constants),
        "full_canvas_constant_bytes": sum(t["bytes"] for t in constants),
        "top_intermediates": sorted(
            intermediates, key=lambda t: -t["bytes"]
        )[:12],
        "top_initializers": sorted(initializers, key=lambda t: -t["bytes"])[:12],
        "top_constants": sorted(constants, key=lambda t: -t["bytes"])[:12],
    }, None


def _points_delta(cost: int, saved: int) -> float:
    if cost <= 0 or saved <= 0 or saved >= cost:
        return 0.0
    return math.log(cost / (cost - saved))


def scan_task(
    task_num: int,
    *,
    samples: int = DEFAULT_SAMPLES,
    seed: int = 20260709,
) -> dict[str, Any]:
    bounds, berr = measure_bounds(task_num, samples=samples, seed=seed)
    model, merr = analyze_model(task_num)
    row: dict[str, Any] = {"task": task_num}
    if berr:
        row["error"] = berr
    if merr:
        row["model_error"] = merr
    if bounds:
        row["bounds"] = asdict(bounds)
    if model:
        row["model"] = model

    if not bounds or not model:
        row["expected_gain"] = 0.0
        return row

    manifest = load_manifest()
    cost = (manifest.get(f"{task_num:03d}") or {}).get("cost", 0)
    full_bytes = (
        model["full_canvas_intermediate_bytes"]
        + model["full_canvas_initializer_bytes"]
        + model["full_canvas_constant_bytes"]
    )
    counted_intermediate_bytes = model["full_canvas_intermediate_bytes"]
    output_area = max(1, bounds.output_area)
    occupied_area = max(1, bounds.occupied_area)
    visible_saved_frac = max(0.0, 1.0 - min(output_area, 900) / 900.0)
    occupied_saved_frac = max(0.0, 1.0 - min(occupied_area, 900) / 900.0)

    # High-confidence work starts with visible output bounds; occupied-only
    # shrink is useful but usually needs per-task reasoning about offsets.
    estimated_saved = int(counted_intermediate_bytes * visible_saved_frac)
    confidence = "visible-bound" if estimated_saved > 0 else "occupied-only"
    if estimated_saved == 0 and occupied_saved_frac > 0:
        estimated_saved = int(counted_intermediate_bytes * occupied_saved_frac * 0.35)

    row.update(
        {
            "deployed_cost": cost,
            "full_canvas_bytes": full_bytes,
            "visible_saved_frac": round(visible_saved_frac, 4),
            "occupied_saved_frac": round(occupied_saved_frac, 4),
            "estimated_saved_bytes": estimated_saved,
            "points_delta_est": round(_points_delta(cost, estimated_saved), 4),
            "expected_gain": round(_points_delta(cost, estimated_saved), 4),
            "confidence": confidence,
        }
    )
    return row


def _write_reports(rows: list[dict[str, Any]], *, samples: int) -> None:
    OUTDIR.mkdir(parents=True, exist_ok=True)
    generated = datetime.now(timezone.utc).isoformat()
    payload = {
        "scanner": "canvas_crop_shrink",
        "generated": generated,
        "samples_per_task": samples,
        "items": rows,
    }
    (OUTDIR / "worklist.json").write_text(json.dumps(payload, indent=1))

    ranked = [r for r in rows if r.get("estimated_saved_bytes", 0) > 0]
    ranked.sort(key=lambda r: (-r.get("expected_gain", 0), -r.get("estimated_saved_bytes", 0)))

    lines = [
        "# canvas_crop_shrink worklist",
        "",
        f"- generated: {generated}",
        f"- samples_per_task: {samples}",
        f"- ranked_candidates: {len(ranked)}",
        "",
        (
            "| task | est pts | est bytes | confidence | output max | "
            "occupied max | full 30x30 tensors | top tensors |"
        ),
        "|---:|---:|---:|---|---|---|---:|---|",
    ]
    for r in ranked[:80]:
        b = r["bounds"]
        m = r["model"]
        top = ", ".join(
            f"{t['name']}:{t['bytes']}" for t in m.get("top_intermediates", [])[:3]
        )
        lines.append(
            "| {task:03d} | {pts:.4f} | {saved} | {conf} | {oh}x{ow} | "
            "{bh}x{bw}@<=({br},{bc}) | {cnt}/{bytes_}B | {top} |".format(
                task=r["task"],
                pts=r.get("points_delta_est", 0.0),
                saved=r.get("estimated_saved_bytes", 0),
                conf=r.get("confidence", "?"),
                oh=b["output_h_max"],
                ow=b["output_w_max"],
                bh=b["occupied_h_max"],
                bw=b["occupied_w_max"],
                br=b["occupied_bottom_max"],
                bc=b["occupied_right_max"],
                cnt=m["full_canvas_intermediate_count"],
                bytes_=m["full_canvas_intermediate_bytes"],
                top=top,
            )
        )
    (OUTDIR / "report.md").write_text("\n".join(lines) + "\n")


def scan_all(tasks: list[int] | None = None, samples: int = DEFAULT_SAMPLES) -> dict:
    task_range = tasks if tasks else list(range(1, 401))
    rows = []
    for idx, task_num in enumerate(task_range, 1):
        rows.append(scan_task(task_num, samples=samples))
        if idx % 10 == 0:
            print(f"...canvas_crop_shrink scanned {idx}/{len(task_range)}", file=sys.stderr)
    _write_reports(rows, samples=samples)
    items = [
        r for r in rows
        if r.get("estimated_saved_bytes", 0) > 0 and not r.get("error")
    ]
    items.sort(key=lambda r: (-r.get("expected_gain", 0.0), -r.get("estimated_saved_bytes", 0)))
    return {"items": items}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tasks", nargs="*", type=int)
    parser.add_argument("--samples", type=int, default=DEFAULT_SAMPLES)
    parser.add_argument("--seed", type=int, default=20260709)
    args = parser.parse_args()
    task_range = args.tasks if args.tasks else list(range(1, 401))
    rows = []
    for idx, task_num in enumerate(task_range, 1):
        print(
            f"...canvas_crop_shrink task{task_num:03d} "
            f"({idx}/{len(task_range)}, samples={args.samples})",
            file=sys.stderr,
        )
        rows.append(scan_task(task_num, samples=args.samples, seed=args.seed))
    _write_reports(rows, samples=args.samples)
    items = [
        r for r in rows
        if r.get("estimated_saved_bytes", 0) > 0 and not r.get("error")
    ]
    items.sort(key=lambda r: (-r.get("expected_gain", 0.0), -r.get("estimated_saved_bytes", 0)))
    print(f"report: {OUTDIR / 'report.md'}")
    print(f"worklist: {OUTDIR / 'worklist.json'}")
    print(f"ranked candidates: {len(items)}")
    for item in items[:20]:
        b = item["bounds"]
        print(
            "task{task:03d} +{gain:.4f} est_saved={saved} "
            "out={oh}x{ow} occ={bh}x{bw} full30={full}".format(
                task=item["task"],
                gain=item.get("expected_gain", 0.0),
                saved=item.get("estimated_saved_bytes", 0),
                oh=b["output_h_max"],
                ow=b["output_w_max"],
                bh=b["occupied_h_max"],
                bw=b["occupied_w_max"],
                full=item["model"]["full_canvas_intermediate_bytes"],
            )
        )


if __name__ == "__main__":
    main()
