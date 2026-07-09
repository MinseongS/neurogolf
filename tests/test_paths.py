from pathlib import Path
from neurogolf import paths


def test_find_root_locates_repo():
    root = paths.find_root(Path(__file__).parent)
    assert (root / "submission" / "overfit_nets").is_dir()


def test_env_override(monkeypatch, tmp_path):
    (tmp_path / "submission" / "overfit_nets").mkdir(parents=True)
    monkeypatch.setenv("NEUROGOLF_ROOT", str(tmp_path))
    assert paths.find_root() == tmp_path
