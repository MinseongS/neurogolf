"""Streamlit UI for inspecting NeuroGolf ONNX candidates.

Run:
    PYTHONPATH=. .venv/bin/streamlit run tools/onnx_viewer.py
"""

from __future__ import annotations

import html
import importlib.util
import math
import pathlib
import sys
import tempfile
from typing import Any

import numpy as np
import onnx
import onnxruntime
import streamlit as st

from src.harness import (
    ROOT,
    convert_to_numpy,
    evaluate,
    load_task,
    run_network,
    sanitize_model,
)


NETWORKS = ROOT / "networks"
CUSTOM = ROOT / "src" / "custom"
FRESH_CACHE = ROOT / "reports" / "fresh_cache"

ARC_COLORS = {
    -1: "#f3f4f6",
    0: "#111827",
    1: "#2563eb",
    2: "#dc2626",
    3: "#16a34a",
    4: "#facc15",
    5: "#6b7280",
    6: "#ec4899",
    7: "#f97316",
    8: "#06b6d4",
    9: "#7c3aed",
}


def color_grid_to_onehot(grid: np.ndarray) -> np.ndarray:
    arr = np.zeros((1, 10, 30, 30), dtype=np.float32)
    h, w = grid.shape
    for r in range(min(h, 30)):
        for c in range(min(w, 30)):
            color = int(grid[r, c])
            if 0 <= color <= 9:
                arr[0, color, r, c] = 1.0
    return arr


def grid_from_onehot(arr: np.ndarray) -> np.ndarray:
    x = np.asarray(arr)
    if x.shape == (1, 10, 30, 30):
        x = x[0]
    if x.shape != (10, 30, 30):
        raise ValueError(f"expected one-hot shape (1,10,30,30), got {arr.shape}")
    active = x > 0.5
    any_active = active.any(axis=0)
    grid = np.argmax(active, axis=0).astype(np.int16)
    grid[~any_active] = -1
    return grid


def normalize_arc_grid(grid: list[list[int]]) -> np.ndarray:
    out = np.full((30, 30), -1, dtype=np.int16)
    for r, row in enumerate(grid[:30]):
        for c, value in enumerate(row[:30]):
            out[r, c] = int(value)
    return out


def render_grid(grid: np.ndarray, key: str, mismatches: np.ndarray | None = None) -> None:
    rows = []
    for r in range(30):
        cells = []
        for c in range(30):
            color = ARC_COLORS.get(int(grid[r, c]), "#ffffff")
            border = "2px solid #ef4444" if mismatches is not None and mismatches[r, c] else "1px solid #d1d5db"
            title = html.escape(f"{key} ({r},{c}) = {int(grid[r, c])}")
            cells.append(
                f"<td title='{title}' style='width:13px;height:13px;"
                f"background:{color};border:{border};padding:0'></td>"
            )
        rows.append("<tr>" + "".join(cells) + "</tr>")
    st.markdown(
        "<table style='border-collapse:collapse;table-layout:fixed'>"
        + "".join(rows)
        + "</table>",
        unsafe_allow_html=True,
    )


def load_python_builder(path: pathlib.Path, task: dict) -> tuple[Any | None, str | None]:
    if not path.exists():
        return None, f"candidate path does not exist: {path}"
    module_name = f"src.custom._viewer_candidate_{abs(hash(str(path)))}"
    try:
        spec = importlib.util.spec_from_file_location(module_name, path)
        if spec is None or spec.loader is None:
            return None, f"cannot import candidate: {path}"
        module = importlib.util.module_from_spec(spec)
        module.__package__ = "src.custom"
        sys.modules[module_name] = module
        spec.loader.exec_module(module)
        build = getattr(module, "build", None)
        if build is None:
            return None, f"{path} does not define build(task)"
        return build(task), None
    except Exception as exc:
        return None, f"build failed: {exc}"


def load_candidate_model(
    task_num: int,
    task: dict,
    mode: str,
    cand_path: str,
    code: str = "",
) -> tuple[Any | None, str | None]:
    if mode == "deployed":
        path = NETWORKS / f"task{task_num:03d}.onnx"
        if not path.exists():
            return None, f"missing deployed ONNX: {path}"
        return path, None
    if mode == "source":
        return load_python_builder(CUSTOM / f"task{task_num:03d}.py", task)
    if mode == "scratch path":
        return load_python_builder(pathlib.Path(cand_path).expanduser(), task)
    if mode == "code editor":
        if "def build(" not in code:
            return None, "code must define build(task)"
        tmp = tempfile.NamedTemporaryFile("w", suffix="_viewer_cand.py", delete=False)
        tmp.write(code)
        tmp.close()
        return load_python_builder(pathlib.Path(tmp.name), task)
    return None, f"unknown candidate mode: {mode}"


def make_session(model_or_path: Any) -> tuple[onnxruntime.InferenceSession | None, str | None]:
    try:
        if isinstance(model_or_path, (str, pathlib.Path)):
            model = onnx.load(model_or_path)
        else:
            model = model_or_path
        sanitized = sanitize_model(onnx.ModelProto().FromString(model.SerializeToString()))
        if sanitized is None:
            return None, "sanitize failed"
        options = onnxruntime.SessionOptions()
        options.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_DISABLE_ALL
        return onnxruntime.InferenceSession(sanitized.SerializeToString(), options), None
    except Exception as exc:
        return None, f"ONNX Runtime load failed: {exc}"


def stored_examples(task: dict) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for split in ("train", "test", "arc-gen"):
        for idx, example in enumerate(task.get(split, [])):
            rows.append({"source": split, "index": idx, "input": example["input"], "output": example["output"]})
    return rows


def fresh_examples(task_num: int) -> tuple[list[dict[str, Any]], str | None]:
    path = FRESH_CACHE / f"task{task_num:03d}.npz"
    if not path.exists():
        return [], f"fresh cache not found: {path}"
    try:
        data = np.load(path)
        inputs = data["inputs"]
        outputs = data["outputs"]
        rows = []
        for idx in range(len(inputs)):
            rows.append({"source": "fresh-cache", "index": idx, "input": inputs[idx], "output": outputs[idx]})
        return rows, None
    except Exception as exc:
        return [], f"fresh cache load failed: {exc}"


def example_to_arrays(example: dict[str, Any]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if example.get("source") == "fresh-cache":
        input_grid = np.asarray(example["input"], dtype=np.int16)
        expected_grid = np.asarray(example["output"], dtype=np.int16)
        return input_grid, expected_grid, color_grid_to_onehot(input_grid)
    benchmark = convert_to_numpy({"input": example["input"], "output": example["output"]})
    if benchmark is None:
        raise ValueError("example exceeds 30x30")
    return normalize_arc_grid(example["input"]), normalize_arc_grid(example["output"]), benchmark["input"]


def run_examples(model_or_path: Any, examples: list[dict[str, Any]]) -> list[dict[str, Any]]:
    session, err = make_session(model_or_path)
    if err:
        raise RuntimeError(err)
    assert session is not None
    rows = []
    for example in examples:
        input_grid, expected_grid, onehot_input = example_to_arrays(example)
        onehot_output = run_network(session, onehot_input)
        output_grid = grid_from_onehot(onehot_output)
        mismatches = output_grid != expected_grid
        rows.append(
            {
                "source": example.get("source", "stored"),
                "index": example.get("index", len(rows)),
                "input_grid": input_grid,
                "expected_grid": expected_grid,
                "output_grid": output_grid,
                "mismatches": mismatches,
                "ok": not bool(mismatches.any()),
            }
        )
    return rows


def metric_text(result: dict[str, Any]) -> tuple[str, str, str, str]:
    memory = result.get("memory")
    params = result.get("params")
    points = result.get("points")
    if memory is None or params is None:
        total = "-"
    else:
        total = f"{memory + params:,}"
    return (
        f"{result.get('pass', 0)} / {result.get('fail', 0)}",
        f"{memory:,}" if isinstance(memory, int) else "-",
        f"{params:,}" if isinstance(params, int) else "-",
        f"{points:.6f}" if isinstance(points, float) else "-",
    )


def main() -> None:
    st.set_page_config(page_title="NeuroGolf ONNX Viewer", layout="wide")
    st.title("NeuroGolf ONNX Viewer")

    with st.sidebar:
        st.header("Task")
        task_num = st.number_input("Task", min_value=1, max_value=400, value=187, step=1)
        data_source = st.radio("Examples", ["stored", "fresh cache"], horizontal=True)
        max_examples = st.slider("Rows", min_value=1, max_value=200, value=30, step=1)
        st.divider()
        st.header("Candidate")
        mode = st.radio("Model source", ["deployed", "source", "scratch path", "code editor"])
        cand_path = ""
        code = ""
        if mode == "scratch path":
            cand_path = st.text_input("cand.py path", "")
        elif mode == "code editor":
            default_path = CUSTOM / f"task{int(task_num):03d}.py"
            default_code = default_path.read_text() if default_path.exists() else "def build(task):\n    return None\n"
            code = st.text_area("Python candidate code", default_code, height=360)
        run_clicked = st.button("Run", type="primary")

    task = load_task(int(task_num))
    examples = stored_examples(task)
    if data_source == "fresh cache":
        examples, fresh_err = fresh_examples(int(task_num))
        if fresh_err:
            st.warning(fresh_err)
    examples = examples[: int(max_examples)]

    if not examples:
        st.info("No examples available.")
        return

    model_or_path, model_err = load_candidate_model(int(task_num), task, mode, cand_path, code)
    if model_err:
        st.error(model_err)
        return

    assert model_or_path is not None
    eval_result = evaluate(model_or_path, task)
    pass_fail, memory, params, points = metric_text(eval_result)
    total = "-"
    if eval_result.get("memory") is not None and eval_result.get("params") is not None:
        total = f"{eval_result['memory'] + eval_result['params']:,}"
    cols = st.columns(5)
    cols[0].metric("pass / fail", pass_fail)
    cols[1].metric("memory", memory)
    cols[2].metric("params", params)
    cols[3].metric("mem + params", total)
    cols[4].metric("points", points)
    if eval_result.get("error"):
        st.error(eval_result["error"])

    try:
        rows = run_examples(model_or_path, examples)
    except Exception as exc:
        st.error(str(exc))
        return

    ok_count = sum(1 for r in rows if r["ok"])
    st.caption(f"Displayed examples: {len(rows)}; displayed pass/fail: {ok_count}/{len(rows) - ok_count}")

    for row in rows:
        label = f"{row['source']} #{row['index']} - {'PASS' if row['ok'] else 'FAIL'}"
        with st.expander(label, expanded=not row["ok"]):
            c1, c2, c3, c4 = st.columns(4)
            with c1:
                st.caption("input")
                render_grid(row["input_grid"], f"{label}-input")
            with c2:
                st.caption("expected")
                render_grid(row["expected_grid"], f"{label}-expected")
            with c3:
                st.caption("candidate output")
                render_grid(row["output_grid"], f"{label}-output", row["mismatches"])
            with c4:
                st.caption("diff")
                diff_grid = np.where(row["mismatches"], 2, -1).astype(np.int16)
                render_grid(diff_grid, f"{label}-diff", row["mismatches"])


if __name__ == "__main__":
    main()
