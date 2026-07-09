def test_scoring_module_has_grader_api():
    from neurogolf import scoring
    for name in ("load_task", "evaluate", "calculate_memory", "calculate_params",
                 "sanitize_model", "convert_to_numpy", "GRID_SHAPE"):
        assert hasattr(scoring, name)

def test_harness_shim_reexports():
    import src.harness as h
    from neurogolf import scoring
    assert h.evaluate is scoring.evaluate and h.load_task is scoring.load_task

def test_evaluate_real_deployed_net_matches_baseline():
    import json
    from neurogolf import scoring, paths
    row = next(r for r in _baseline_rows() if r["task"] == 1)
    res = scoring.evaluate(str(paths.OVERFIT_NETS / "task001.onnx"), scoring.load_task(1))
    assert res["ok"] and res["fail"] == 0
    assert res["memory"] + res["params"] == row["cost"]

def _baseline_rows():
    import json
    from neurogolf import paths
    data = json.load(open(paths.STATE / "baseline" / "manifest.json"))
    rows = data["tasks"] if isinstance(data, dict) and "tasks" in data else data
    return list(rows.values()) if isinstance(rows, dict) else rows
