import zipfile, pytest
from neurogolf import pack

def _mk_nets(tmp_path, n, monkeypatch):
    nets = tmp_path / "nets"; nets.mkdir()
    for i in range(1, n + 1):
        (nets / f"task{i:03d}.onnx").write_bytes(b"x")
    monkeypatch.setattr(pack, "find_unsigned_topk", lambda p: [])
    return nets

def test_pack_requires_exactly_400(tmp_path, monkeypatch):
    nets = _mk_nets(tmp_path, 3, monkeypatch)
    with pytest.raises(SystemExit):
        pack.pack(nets_dir=nets, out=tmp_path / "submission.zip")

def test_pack_refuses_on_topk_offender(tmp_path, monkeypatch):
    nets = _mk_nets(tmp_path, 400, monkeypatch)
    monkeypatch.setattr(pack, "find_unsigned_topk", lambda p: ["bad"])
    with pytest.raises(SystemExit):
        pack.pack(nets_dir=nets, out=tmp_path / "submission.zip")

def test_pack_flat_zip_400(tmp_path, monkeypatch):
    nets = _mk_nets(tmp_path, 400, monkeypatch)
    out = pack.pack(nets_dir=nets, out=tmp_path / "submission.zip")
    names = zipfile.ZipFile(out).namelist()
    assert len(names) == 400 and all("/" not in n for n in names)
