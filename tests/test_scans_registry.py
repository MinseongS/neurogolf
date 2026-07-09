def test_registry_names_match_levers_yaml():
    import yaml
    from neurogolf.scans import SCANNERS
    from neurogolf.paths import STATE
    levers = yaml.safe_load(open(STATE / "levers.yaml"))["levers"]
    referenced = {l["scanner"] for l in levers if l.get("scanner")}
    assert referenced <= set(SCANNERS), f"missing scanners: {referenced - set(SCANNERS)}"


def test_run_scan_writes_worklist(tmp_path, monkeypatch):
    from neurogolf import scans
    monkeypatch.setitem(scans.SCANNERS, "dummy", lambda tasks: {"items": [{"task": 1, "expected_gain": 0.5}]})
    monkeypatch.setattr(scans, "WORKLISTS", tmp_path)
    out = scans.run_scan("dummy")
    import json
    data = json.loads(out.read_text())
    assert data["lever"] == "dummy" and data["items"][0]["task"] == 1


def test_every_scanner_smoke_one_task():
    """Each registered scanner must run cleanly on a single task and return
    {"items": [...]} (empty is fine — task 2 legitimately floors some levers)."""
    from neurogolf.scans import SCANNERS
    for name, fn in SCANNERS.items():
        result = fn([2])
        assert isinstance(result, dict), f"{name} did not return a dict"
        assert "items" in result, f"{name} result missing 'items' key"
        assert isinstance(result["items"], list), f"{name} 'items' is not a list"
