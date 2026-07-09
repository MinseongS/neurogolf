from pathlib import Path
from tools.migrate_tasklogs import stale_verdict, migrate_one

def test_stale_when_no_logged_cost_matches_deployed():
    body = "## 분석\n cost 1604 -> 393 로 줄임. 이전 2813 이었음."
    assert stale_verdict(body, deployed_cost=999) == "stale-likely"

def test_fresh_when_deployed_cost_appears_in_log():
    body = "## 분석\n cost 1604 -> 393 로 줄임."
    assert stale_verdict(body, deployed_cost=393) == "match"

def test_migrate_one_prepends_frontmatter(tmp_path: Path):
    src = tmp_path / "task001.md"; src.write_text("# task001\ncost 500")
    out = migrate_one(src, tmp_path / "out", deployed_cost=500)
    text = out.read_text()
    assert text.startswith("---\n") and "deployed_cost: 500" in text and "logged_costs_match: match" in text
