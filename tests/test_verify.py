from neurogolf import verify

def test_hash_check_detects_mutation(tmp_path, monkeypatch):
    nets = tmp_path / "nets"; nets.mkdir()
    (nets / "task001.onnx").write_bytes(b"A")
    baseline = tmp_path / "sha256.txt"
    import hashlib
    baseline.write_text(hashlib.sha256(b"B").hexdigest() + "  task001.onnx\n")
    monkeypatch.setattr(verify, "OVERFIT_NETS", nets)
    monkeypatch.setattr(verify, "_baseline_file", lambda: baseline)
    assert verify.hash_check() == ["task001.onnx"]

def test_cli_has_all_subcommands():
    from neurogolf.cli import build_parser
    subs = build_parser()._subparsers._group_actions[0].choices
    assert set(subs) >= {"status", "score", "gate", "adopt", "pack", "submit",
                         "scan", "queue", "mine-public", "verify"}
