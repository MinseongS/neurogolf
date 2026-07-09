# NeuroGolf 전면 재설계 구현 계획

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 스펙(`docs/superpowers/specs/2026-07-09-neurogolf-redesign-design.md`)대로 상태 모델(`state/`) + `ng` CLI + 레버 엔진으로 전면 재구축하고 레거시를 삭제한다.

**Architecture:** 4단계 마이그레이션 (안전화 → 상태 모델 → CLI → 레버 엔진). 채점 코어는 기존 `src/harness.py`(grader-identical)를 `src/neurogolf/scoring.py`로 이동하고 셔틀 심(shim)으로 하위호환 유지. 모든 채택은 `neurogolf.gate.gate()` 단일 관문 경유.

**Tech Stack:** Python ≥3.13, uv(hatchling package 모드), onnx/onnxruntime/numpy, pytest, kaggle CLI.

## Global Constraints

- **점수 불변식**: 의도적 `ng adopt` 없이 `submission/overfit_nets/` 400넷의 sha256이 Phase 0 기준값(`state/baseline/sha256.txt`)과 달라지면 안 된다. 각 Phase 종료 시 검증.
- 제출 파일명은 반드시 `submission.zip` (repo 루트).
- unsigned(uint8/16/32/64) 입력의 TopK는 grader-killer — pack/gate가 거부해야 한다.
- 평가는 항상 태스크별 격리 프로세스 (ORT weight-aliasing knife-edge 방지).
- 비교 기준은 항상 배포본 `submission/overfit_nets/` (networks/ 아님).
- `src/custom/taskNNN.py`는 `from ..builders import` 상대임포트를 쓰므로 `src` 패키지는 repo 루트 기준으로 임포트 가능해야 한다 (`sys.path`에 ROOT 필요).
- `load_task(n)`은 examples 리스트를 직접 반환한다: `evaluate(str(path), load_task(n))`.
- 마이그레이션 중 병렬 점수 세션 금지.
- 커밋 트레일러: `Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>`.
- 부정 판정(레버 dry/floor)은 levers.yaml ledger 4필드(`date/ran/verdict/reopen`)로만 기록. status는 `live|dormant`만 (dead 없음).

---

# Phase 0 — 안전화

### Task 1: 전체 커밋 + 태그 + 스냅샷 + 기준값 동결

**Files:**
- Modify: `.gitignore` (58-59행 `docs/superpowers/plans/`, `docs/superpowers/specs/` 제거)
- Create: `state/baseline/sha256.txt`, `state/baseline/points.json`
- Create(repo 밖): `~/neurogolf_snapshot_20260709/overfit_nets/`

**Interfaces:**
- Produces: `state/baseline/sha256.txt` (400줄, `<sha256>  <filename>` 형식), `state/baseline/points.json` (`{"total_points": float, "total_cost": int, "n_tasks": 400}`) — 이후 모든 Phase의 검증 기준.

- [ ] **Step 1: .gitignore에서 docs 제외 해제**

`.gitignore` 58-59행 삭제:
```
docs/superpowers/plans/
docs/superpowers/specs/
```

- [ ] **Step 2: repo 밖 스냅샷**

```bash
mkdir -p ~/neurogolf_snapshot_20260709
cp -R submission/overfit_nets ~/neurogolf_snapshot_20260709/
ls ~/neurogolf_snapshot_20260709/overfit_nets/*.onnx | wc -l   # 400 확인
```

- [ ] **Step 3: 기준 sha256 생성**

```bash
mkdir -p state/baseline
(cd submission/overfit_nets && shasum -a 256 task*.onnx) > state/baseline/sha256.txt
wc -l state/baseline/sha256.txt   # 400 확인
```

- [ ] **Step 4: 기준 점수 동결 (isolated-all 재측정)**

```bash
PYTHONPATH=. uv run python reports/scripts/build_overfit_manifest.py --isolated-all
```
Expected: 400/400 ok 출력. 이어서:

```bash
PYTHONPATH=. uv run python - <<'EOF'
import json
data = json.load(open("reports/overfit_manifest.json"))
rows = data["tasks"] if isinstance(data, dict) and "tasks" in data else data
if isinstance(rows, dict):
    rows = list(rows.values())
total_points = sum(r["points"] for r in rows)
total_cost = sum(r["cost"] for r in rows if r.get("cost") is not None)
assert len(rows) == 400 and all(r.get("ok") and r.get("fail") == 0 for r in rows), "400/400 아님 — 중단"
json.dump({"total_points": total_points, "total_cost": total_cost, "n_tasks": 400},
          open("state/baseline/points.json", "w"), indent=1)
print(total_points, total_cost)
EOF
cp reports/overfit_manifest.json state/baseline/manifest.json
```
Expected: 총점 ≈ 7279 근방 (직전 세션 로컬값과 대조), assert 통과.

- [ ] **Step 5: 전체 커밋 + 태그**

```bash
git add -A
git commit -m "chore: pre-redesign snapshot (uncommitted adoptions + baseline freeze)

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
git tag pre-redesign
git status --porcelain   # 비어 있어야 함
```

---

# Phase 1 — 상태 모델

### Task 2: STATE.md 증류 + NEXT_SESSION.md 삭제

**Files:**
- Create: `state/STATE.md`
- Delete: `NEXT_SESSION.md`

**Interfaces:**
- Produces: `state/STATE.md` — 세션 시작 시 읽는 유일한 live 핸드오프. **~120줄 상한, append 금지(세션 종료 시 교체).**

- [ ] **Step 1: NEXT_SESSION.md 전체를 읽고 "현재 참인 것"만 증류하여 state/STATE.md 작성**

필수 구조 (섹션 고정):

```markdown
# STATE — NeuroGolf live handoff (갱신: YYYY-MM-DD)
> 이 파일은 append 금지. 세션 종료 시 "현재 참인 것"만 남기고 교체한다. 히스토리는 git + state/submissions.md.

## 확정 상태
- BEST LB: 7279.41 (sub 54467261, LB 확인)
- 로컬 기준값: state/baseline/points.json 참조
- 마감: 2026-07-15 (private LB = 동일 고정 데이터셋; bundled fail=0 = 영구 통과)

## 활성 베인 (지금 할 일, 기대이득 순)
1. free-output-einsum regime crack — mask_dominance 잔여 ~45태스크 (16/18 crack, batch6 dry: 7/8 positioned-content floor). +20~40 LB 잠재.
2. CONV-FP32 arsenal (074/080/198/383/187) + QLinearConv(349) — regime crack의 non-mask 일반화, 미증명.
3. 공개 min-merge 모니터링 — 새 업로더가 7250+ 갱신 시 margin-0 재채굴.
4. (levers.yaml의 live 레버 순회)

## 불변식 (재검증 금지)
- submission.zip 이름 고정 / 100회/일 / unsigned TopK 금지 / isolated eval
- 비교 기준 = submission/overfit_nets/ (배포본)
- 로컬 == LB (+0.11 오프셋)

## 다음 세션 시작 절차
1. `uv run ng status` 2. 이 파일 3. `state/levers.yaml`에서 live 레버 선택 4. skills/neurogolf/SKILL.md 루프 실행
```

수용 기준: ①120줄 이하 ②위 4개 섹션 존재 ③BEST 서브미션 ID 포함 ④NEXT_SESSION.md의 미완 작업 중 levers.yaml로 갈 것(레버 ledger)과 STATE.md에 남을 것(활성 베인)이 분리되어 있음. NEXT_SESSION.md의 세션별 서사(2026-07-08 로그들)는 옮기지 않는다 — git 히스토리가 담당.

- [ ] **Step 2: NEXT_SESSION.md 삭제 + 커밋**

```bash
git rm NEXT_SESSION.md
git add state/STATE.md
git commit -m "state: distill NEXT_SESSION.md into state/STATE.md (replace-only handoff)

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

### Task 3: levers.yaml 작성

**Files:**
- Create: `state/levers.yaml`

**Interfaces:**
- Produces: `state/levers.yaml` — Task 15의 `ng scan`이 `scanner:` 필드를 스캐너 레지스트리 이름으로 해석. 스키마 (모든 엔트리 공통):
  `name(str) / status(live|dormant) / scanner(str|null) / scanner_archive(str|null, git ref 경로) / recipe(str|null) / agent_class(fable|opus) / expected_yield(str) / ledger(list of {date, ran, verdict, reopen})`

- [ ] **Step 1: 초기 레버 엔트리 작성**

아래 10개 레버를 위 스키마로 작성한다. `ledger`는 NEXT_SESSION.md·메모리의 기존 판정을 4필드로 변환해 담는다 (bare negative 금지). 첫 2개는 그대로 사용:

```yaml
levers:
  - name: free-output-einsum-regime-crack
    status: live
    scanner: mask_dominance
    scanner_archive: null
    recipe: playbook/free-output-einsum.md
    agent_class: fable
    expected_yield: "+0.2~0.9/task, 잔여 ~45태스크 (16/18 crack 실적)"
    ledger:
      - date: 2026-07-08
        ran: "batch6 8태스크 (opus)"
        verdict: "7/8 floor — positioned-content mask는 크랙 불가, 1 broken(071)"
        reopen: "새 sub-recipe 발견 시; non-mask bloat(Conv 3600B, QLinearConv)로 일반화 시"

  - name: deployed-fp16-recast
    status: live
    scanner: null           # scan_deployed_fp16.py는 리포에 실존하지 않음(문서에만 존재) — 재작성 필요
    scanner_archive: null
    recipe: playbook/fp16-recast.md
    agent_class: opus
    expected_yield: "+0.03~0.3/task; output-coupled는 winnable, input-coupled는 floor"
    ledger:
      - date: 2026-07-08
        ran: "task377/205/355 수작업 (final-Einsum 직전 fp16 Cast 삽입 패턴)"
        verdict: "3승 채택 (+0.34 LB 합); 스캐너 파일은 유실 — 문서의 input-weld/co-bind 필터로 재작성해야 함"
        reopen: "모든 채택/오버레이 후 재실행; free-output rewrite 후 상류 tail 재검사"
```

나머지 8개 (같은 스키마로, ledger는 아래 출처에서 변환):
- `kernel-collapse` — status: live, scanner: kernel_collapse, agent_class: opus. ledger: 2026-07-08 +0.553 LB 18승; reopen: "새 Conv 넷 유입 시 재실행".
- `public-minmerge` — status: live, scanner: null (CLI `ng mine-public`이 담당), agent_class: opus. ledger: 2026-07-08 margin-0 0승 (프론티어 7250.24 기준); reopen: "공개 프론티어가 현재 채굴선 위로 갱신 시 / 새 업로더".
- `reducesum-spatial-einsum` — dormant, scanner_archive: "git:pre-redesign:reports/candidates/reducesum_spatial_to_einsum_probe.py". ledger: 2026-07-08 8승 채택 후 잔여 0; reopen: "새 spatial ReduceSum 유입 시".
- `walk-chain-slack` — dormant, scanner_archive: "git:pre-redesign:reports/scripts/walk_chain_slack.py". ledger: 2026-07-07 1/10 (task243만 slack); reopen: "새 walk-chain 넷 채택 시".
- `fold-reducible-plane` — dormant, scanner: fold, ledger: 2026-07-08 top4 4/4 floor; reopen: "새 공개 teacher / monster(233/366/285) 재정식화 후".
- `topk-width-refit` — dormant, scanner: null. ledger: 후보 46/361/377 미착수; reopen: "fresh-gate 여유 있는 세션".
- `runtime-timeout-spend` — dormant, scanner: null. ledger: 미탐사 축 (공개 discussion 채굴); reopen: "regime 베인 소진 후".
- `dtype-overpay` — dormant, scanner: dtype_overpay. ledger: 2026-07-08 networks/ 겨냥 오판 이력 — 배포본 재조준 필요; reopen: "스캐너 이식(Task 15) 후 1회 재실행".

- [ ] **Step 2: yaml 파싱 확인 + 커밋**

```bash
uv run python -c "import yaml,sys; d=yaml.safe_load(open('state/levers.yaml')); assert all(l['status'] in ('live','dormant') and all(set(e)=={'date','ran','verdict','reopen'} for e in l['ledger']) for l in d['levers']); print(len(d['levers']),'levers ok')"
```
(pyyaml 미설치면 `uv add pyyaml` — Task 8에서 dependencies에 정식 추가되므로 여기선 `uv run --with pyyaml python -c ...` 사용 가능)

```bash
git add state/levers.yaml && git commit -m "state: lever registry with 4-field negative-verdict ledgers

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

### Task 4: insights/submissions/arc_mapping 이주

**Files:**
- Create(이동): `state/insights.yaml` ← `reports/insight_registry.yaml`
- Create(이동): `state/submissions.md` ← `reports/submission_log.md`
- Create(이동): `state/arc_mapping.json` ← `reports/arc_mapping.json`
- Modify: `src/genverify.py:14` 부근 `MAPPING = ROOT / "reports" / "arc_mapping.json"` → `ROOT / "state" / "arc_mapping.json"`

- [ ] **Step 1: git mv 3건**

```bash
git mv reports/insight_registry.yaml state/insights.yaml
git mv reports/submission_log.md state/submissions.md
git mv reports/arc_mapping.json state/arc_mapping.json
```

- [ ] **Step 2: genverify 경로 수정 + 기존 테스트로 검증**

`src/genverify.py`에서 `arc_mapping.json` 경로를 `state/`로 변경 후:
```bash
uv run pytest tests/test_genverify.py -v
```
Expected: PASS

- [ ] **Step 3: 커밋**

```bash
git add -A && git commit -m "state: migrate insights/submissions/arc_mapping into state/

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

### Task 5: tasklog → state/tasks/ 이주 (stale 휴리스틱 포함)

**Files:**
- Create: `tools/migrate_tasklogs.py`
- Create: `state/tasks/taskNNN.md` × (기존 tasklog 수)
- Test: `tests/test_migrate_tasklogs.py`
- Delete(이주 후): `reports/tasklog/`

**Interfaces:**
- Produces: `state/tasks/taskNNN.md` — 프론트매터 헤더 3줄(`deployed_cost / logged_costs_match / migrated`)이 붙은 tasklog. Task 11의 adopt가 이 파일에 스탬프를 append.

- [ ] **Step 1: 실패하는 테스트 작성**

```python
# tests/test_migrate_tasklogs.py
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
```

- [ ] **Step 2: 실패 확인**

Run: `uv run pytest tests/test_migrate_tasklogs.py -v` — Expected: FAIL (`No module named 'tools.migrate_tasklogs'`)

- [ ] **Step 3: 구현**

```python
# tools/migrate_tasklogs.py
"""reports/tasklog/ -> state/tasks/ 이주. stale 휴리스틱: 배포본 cost가 본문 숫자에 없으면 stale-likely."""
import json, re, sys
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

def stale_verdict(body: str, deployed_cost: int | None) -> str:
    if deployed_cost is None:
        return "unknown"
    nums = {int(n) for n in re.findall(r"\b(\d{2,7})\b", body)}
    return "match" if deployed_cost in nums else "stale-likely"

def migrate_one(src: Path, out_dir: Path, deployed_cost: int | None) -> Path:
    body = src.read_text()
    fm = (f"---\ndeployed_cost: {deployed_cost}\n"
          f"logged_costs_match: {stale_verdict(body, deployed_cost)}\n"
          f"migrated: {date.today().isoformat()}\n---\n\n")
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / src.name
    out.write_text(fm + body)
    return out

def main() -> None:
    manifest = json.load(open(ROOT / "state" / "baseline" / "manifest.json"))
    rows = manifest["tasks"] if isinstance(manifest, dict) and "tasks" in manifest else manifest
    if isinstance(rows, dict):
        rows = list(rows.values())
    cost_by_task = {int(r["task"]): r.get("cost") for r in rows}
    src_dir, out_dir = ROOT / "reports" / "tasklog", ROOT / "state" / "tasks"
    n = 0
    for f in sorted(src_dir.glob("task*.md")):
        num = int(re.search(r"(\d+)", f.stem).group(1))
        migrate_one(f, out_dir, cost_by_task.get(num))
        n += 1
    print(f"migrated {n} tasklogs")

if __name__ == "__main__":
    main()
```

- [ ] **Step 4: 테스트 통과 확인**

Run: `uv run pytest tests/test_migrate_tasklogs.py -v` — Expected: 3 PASS

- [ ] **Step 5: 실제 이주 실행 + 원본 삭제 + 커밋**

```bash
PYTHONPATH=. uv run python tools/migrate_tasklogs.py
ls state/tasks | wc -l    # reports/tasklog 파일 수와 동일 확인
grep -rl "stale-likely" state/tasks | wc -l   # stale 규모 파악(정보용)
git rm -r reports/tasklog
git add -A && git commit -m "state: migrate tasklogs with stale-detection frontmatter

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

### Task 6: AGENTS.md/README 재작성 + data.zip 삭제 + Phase 1 검증

**Files:**
- Modify: `AGENTS.md`(전면 교체), `README.md`(전면 교체)
- Delete: `data.zip`, `reports/score_modes.md`, `reports/USER_REVIEW_WORKFLOW.md`, `reports/AUTO_SOLVER_DESIGN.md`

- [ ] **Step 1: AGENTS.md 전면 교체**

```markdown
# NeuroGolf Agent Instructions

세션 시작: `uv run ng status` → `state/STATE.md` → `state/levers.yaml`에서 live 레버 선택
→ `skills/neurogolf/SKILL.md`의 표준 루프 실행. 세션 종료: STATE.md 교체(append 금지).

## 불변 규칙
- 목표 8000. 8000 overfit 모드가 기본: 게이트 = bundled fail=0 + 배포본보다 cheaper. fresh 검증은 진단용.
- 채택은 반드시 `ng adopt` 경유 (gate 우회 금지). 제출은 `ng pack` → `ng submit`.
- FREE input/output 텐서를 공격적으로 활용. 공개 INSIGHT를 채굴해 400태스크로 일반화.
- 부정 판정(소진/floor)은 state/levers.yaml ledger 4필드로만 기록. 레버는 dormant, dead 없음.
- 후보/스크래치는 `candidates/` 아래에만. 병렬 세션 시 제출 전 `kaggle competitions submissions` 확인.

## 진실 소스
- `state/STATE.md` live 핸드오프 / `state/levers.yaml` 레버 원장 / `state/tasks/` 태스크 원장
- `state/insights.yaml` 메커니즘 / `state/submissions.md` 제출 로그 / `playbook/` 레시피
- `submission/overfit_nets/` 배포본(불가침, adopt로만 변경) / `src/custom/` 빌드 소스
```

- [ ] **Step 2: README.md 교체** — 개요 + `uv sync --dev` + `uv run ng --help` + 디렉토리 맵(스펙의 목표 구조 표) + 스펙/플랜 문서 링크. 30줄 이내.

- [ ] **Step 3: 삭제 + 검증 + 커밋**

```bash
git rm data.zip reports/score_modes.md reports/USER_REVIEW_WORKFLOW.md reports/AUTO_SOLVER_DESIGN.md
(cd submission/overfit_nets && shasum -a 256 task*.onnx) | diff - state/baseline/sha256.txt && echo NETS-UNCHANGED
git add -A && git commit -m "docs: thin AGENTS/README pointers; drop data.zip and superseded docs

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```
Expected: `NETS-UNCHANGED` 출력 (불변식 유지).

---

# Phase 2 — ng CLI

### Task 7: 패키지 전환 + paths 모듈

**Files:**
- Modify: `pyproject.toml`
- Create: `src/neurogolf/__init__.py`, `src/neurogolf/paths.py`
- Test: `tests/test_paths.py`

**Interfaces:**
- Produces: `neurogolf.paths.find_root(start: Path | None = None) -> Path`; 모듈 상수 `ROOT, OVERFIT_NETS, STATE, CANDIDATES, DATA, PLAYBOOK` (모두 Path); `ensure_src_importable() -> None` (ROOT를 sys.path[0]에 삽입 — `src.custom.*`/`src.harness` 임포트용). 이후 모든 모듈이 이것만 사용 (절대경로 하드코딩 금지).

- [ ] **Step 1: pyproject.toml 교체**

```toml
[project]
name = "neurogolf"
version = "0.2.0"
description = "NeuroGolf 2026: state model + ng CLI + lever engine."
readme = "README.md"
requires-python = ">=3.13"
dependencies = ["numpy", "onnx", "onnxruntime", "pandas", "streamlit", "pyyaml"]

[project.optional-dependencies]
research = ["scipy", "torch"]

[project.scripts]
ng = "neurogolf.cli:main"

[dependency-groups]
dev = ["pytest"]

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.hatch.build.targets.wheel]
packages = ["src/neurogolf"]
```

- [ ] **Step 2: 실패하는 테스트 작성**

```python
# tests/test_paths.py
from pathlib import Path
from neurogolf import paths

def test_find_root_locates_repo():
    root = paths.find_root(Path(__file__).parent)
    assert (root / "submission" / "overfit_nets").is_dir()

def test_env_override(monkeypatch, tmp_path):
    (tmp_path / "submission" / "overfit_nets").mkdir(parents=True)
    monkeypatch.setenv("NEUROGOLF_ROOT", str(tmp_path))
    assert paths.find_root() == tmp_path
```

- [ ] **Step 3: 실패 확인** — `uv run pytest tests/test_paths.py -v` → FAIL (no module)

- [ ] **Step 4: 구현**

```python
# src/neurogolf/paths.py
import os, sys
from pathlib import Path

def find_root(start: Path | None = None) -> Path:
    env = os.environ.get("NEUROGOLF_ROOT")
    if env:
        return Path(env)
    cur = (start or Path.cwd()).resolve()
    for p in (cur, *cur.parents):
        if (p / "submission" / "overfit_nets").is_dir() and (p / "pyproject.toml").exists():
            return p
    raise SystemExit("neurogolf 루트를 찾을 수 없음 — repo 안에서 실행하거나 NEUROGOLF_ROOT 설정")

ROOT = find_root(Path(__file__).resolve().parent)
OVERFIT_NETS = ROOT / "submission" / "overfit_nets"
STATE = ROOT / "state"
CANDIDATES = ROOT / "candidates"
DATA = ROOT / "data"
PLAYBOOK = ROOT / "playbook"

def ensure_src_importable() -> None:
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
```
`src/neurogolf/__init__.py`는 빈 파일.

- [ ] **Step 5: 통과 확인 + 커밋**

```bash
uv sync --dev && uv run pytest tests/test_paths.py -v
git add pyproject.toml uv.lock src/neurogolf tests/test_paths.py
git commit -m "feat: package mode + paths module (ng entry point)

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```
주의: `NEUROGOLF_ROOT` env 테스트가 `ROOT` 모듈 상수엔 영향 없음(임포트 시점 고정) — 테스트는 `find_root()` 직접 호출로 검증.

### Task 8: scoring 이동 (harness → neurogolf.scoring + 심)

**Files:**
- Create: `src/neurogolf/scoring.py` (src/harness.py 본문 이동)
- Modify: `src/harness.py` (심으로 교체)
- Test: `tests/test_scoring.py`

**Interfaces:**
- Produces: `neurogolf.scoring`에 harness와 동일한 공개 API — `load_task(task_num)`, `evaluate(model_or_path, examples, keep_failures=False)`, `calculate_memory(model, trace_path)`, `calculate_params(model)`, `sanitize_model(model)`, `convert_to_numpy(example)`, 상수 `GRID_SHAPE, IR_VERSION, OPSET_IMPORTS, DATA_TYPE`. 기존 `src.harness` 임포트는 심을 통해 계속 동작.

- [ ] **Step 1: 실패하는 테스트 작성**

```python
# tests/test_scoring.py
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
```

- [ ] **Step 2: 실패 확인** — `uv run pytest tests/test_scoring.py -v` → FAIL

- [ ] **Step 3: 이동 구현**

1. `src/harness.py` 전체 본문을 `src/neurogolf/scoring.py`로 복사.
2. `scoring.py` 상단의 `ROOT = Path(__file__).resolve().parent.parent` 계열 정의를 삭제하고 `from neurogolf.paths import ROOT` + `DATA_DIR = ROOT / "data"`로 교체. **그 외 로직은 한 글자도 바꾸지 않는다** (grader-identical 보존).
3. `src/harness.py`를 심으로 교체:
```python
# src/harness.py — legacy shim; 실체는 neurogolf.scoring
from neurogolf.scoring import *          # noqa: F401,F403
from neurogolf.scoring import (          # noqa: F401 — star가 못 잡는 밑줄/상수 명시
    DATA_DIR, ROOT, GRID_SHAPE, IR_VERSION, OPSET_IMPORTS, DATA_TYPE,
    EXCLUDED_OP_TYPES, FILESIZE_LIMIT_IN_BYTES,
)
```
(주의: scoring.py에 `__all__`이 없으므로 star import는 공개명 전부를 가져온다. `src/builders.py`가 쓰는 `DATA_TYPE, GRID_SHAPE, IR_VERSION, OPSET_IMPORTS` 임포트가 깨지지 않는지 아래 스텝에서 확인.)

- [ ] **Step 4: 통과 확인 (스코어링 회귀 포함)**

```bash
uv run pytest tests/test_scoring.py tests/test_genverify.py -v
PYTHONPATH=. uv run python -c "import src.builders, src.pipeline, src.adopt, src.genverify; print('legacy imports ok')"
```
Expected: 전부 PASS / ok.

- [ ] **Step 5: 커밋**

```bash
git add -A && git commit -m "refactor: move grader-identical scorer to neurogolf.scoring (harness shim)

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

### Task 9: topk + gate 모듈

**Files:**
- Create: `src/neurogolf/topk.py`, `src/neurogolf/gate.py`
- Test: `tests/test_topk.py`, `tests/test_gate.py`
- 참조(이식 원본): `reports/scripts/scan_unsigned_topk.py`, `reports/scripts/build_overfit_manifest.py::measure_task_isolated`

**Interfaces:**
- Produces:
  - `neurogolf.topk.find_unsigned_topk(model_path: Path) -> list[str]` — 위반 설명 리스트(빈 리스트 = clean). 판정: TopK의 `input[0]` elem_type ∈ {UINT8,16,32,64} 또는 타입 미해결(UNKNOWN도 위반 취급). Cast `to` 전파 포함 — 원본 스크립트 로직 그대로 이식.
  - `neurogolf.gate.eval_isolated(model_path: Path, task_num: int) -> dict` — 격리 프로세스 evaluate. 반환 dict 키: `ok, pass, fail, memory, params, points, error, cost`.
  - `neurogolf.gate.deployed_cost(task_num: int) -> int | None` — 배포본 cost (state/manifest.json 캐시 우선, 없으면 eval_isolated).
  - `@dataclass neurogolf.gate.GateResult: ok: bool; reasons: list[str]; candidate: dict; incumbent_cost: int | None`
  - `neurogolf.gate.gate(candidate: Path, task_num: int) -> GateResult` — ①fail==0 ②cost < deployed ③topk clean 전부 통과 시 ok.

- [ ] **Step 1: 실패하는 테스트 작성**

```python
# tests/test_topk.py
import onnx
from onnx import TensorProto, helper
from neurogolf.topk import find_unsigned_topk

def _topk_model(elem_type):
    x = helper.make_tensor_value_info("x", elem_type, [1, 8])
    k = helper.make_tensor("k", TensorProto.INT64, [1], [3])
    v = helper.make_tensor_value_info("v", elem_type, [1, 3])
    i = helper.make_tensor_value_info("i", TensorProto.INT64, [1, 3])
    node = helper.make_node("TopK", ["x", "k"], ["v", "i"])
    graph = helper.make_graph([node], "g", [x], [v, i], initializer=[k])
    return helper.make_model(graph, opset_imports=[helper.make_opsetid("", 11)])

def test_uint8_topk_flagged(tmp_path):
    p = tmp_path / "bad.onnx"; onnx.save(_topk_model(TensorProto.UINT8), p)
    assert find_unsigned_topk(p)

def test_float_topk_clean(tmp_path):
    p = tmp_path / "ok.onnx"; onnx.save(_topk_model(TensorProto.FLOAT), p)
    assert find_unsigned_topk(p) == []
```

```python
# tests/test_gate.py
from pathlib import Path
from neurogolf import gate

def _fake_eval(fail, cost):
    return {"ok": True, "pass": 260, "fail": fail, "memory": cost, "params": 0,
            "points": 10.0, "error": None, "cost": cost}

def test_gate_rejects_bundled_failure(monkeypatch, tmp_path):
    monkeypatch.setattr(gate, "eval_isolated", lambda p, t: _fake_eval(fail=2, cost=100))
    monkeypatch.setattr(gate, "deployed_cost", lambda t: 500)
    monkeypatch.setattr(gate, "find_unsigned_topk", lambda p: [])
    r = gate.gate(tmp_path / "c.onnx", 1)
    assert not r.ok and any("fail" in s for s in r.reasons)

def test_gate_rejects_not_cheaper(monkeypatch, tmp_path):
    monkeypatch.setattr(gate, "eval_isolated", lambda p, t: _fake_eval(fail=0, cost=500))
    monkeypatch.setattr(gate, "deployed_cost", lambda t: 500)   # 같아도 거부(strictly cheaper)
    monkeypatch.setattr(gate, "find_unsigned_topk", lambda p: [])
    assert not gate.gate(tmp_path / "c.onnx", 1).ok

def test_gate_rejects_unsigned_topk(monkeypatch, tmp_path):
    monkeypatch.setattr(gate, "eval_isolated", lambda p, t: _fake_eval(fail=0, cost=100))
    monkeypatch.setattr(gate, "deployed_cost", lambda t: 500)
    monkeypatch.setattr(gate, "find_unsigned_topk", lambda p: ["TopK uint8"])
    assert not gate.gate(tmp_path / "c.onnx", 1).ok

def test_gate_passes_clean_cheaper(monkeypatch, tmp_path):
    monkeypatch.setattr(gate, "eval_isolated", lambda p, t: _fake_eval(fail=0, cost=100))
    monkeypatch.setattr(gate, "deployed_cost", lambda t: 500)
    monkeypatch.setattr(gate, "find_unsigned_topk", lambda p: [])
    r = gate.gate(tmp_path / "c.onnx", 1)
    assert r.ok and r.candidate["cost"] == 100 and r.incumbent_cost == 500
```

- [ ] **Step 2: 실패 확인** — `uv run pytest tests/test_topk.py tests/test_gate.py -v` → FAIL

- [ ] **Step 3: 구현**

`src/neurogolf/topk.py`: `reports/scripts/scan_unsigned_topk.py`의 타입테이블 구축(value_info+graph in/out+initializer, Cast `to` 전파)과 판정 로직을 함수 `find_unsigned_topk(model_path)`로 그대로 감싼다 (flat script → 함수화 외 로직 변경 금지).

```python
# src/neurogolf/gate.py
import json, subprocess, sys
from dataclasses import dataclass, field
from pathlib import Path
from neurogolf.paths import ROOT, OVERFIT_NETS, STATE
from neurogolf.topk import find_unsigned_topk

_EVAL_CODE = """
import json, sys
from neurogolf.scoring import evaluate, load_task
task, path = int(sys.argv[1]), sys.argv[2]
r = evaluate(path, load_task(task), keep_failures=False)
row = {k: r.get(k) for k in ('ok','pass','fail','memory','params','points','error')}
row['cost'] = None if row['memory'] is None or row['params'] is None else int(row['memory']) + int(row['params'])
print(json.dumps(row))
"""

def eval_isolated(model_path: Path, task_num: int) -> dict:
    proc = subprocess.run([sys.executable, "-c", _EVAL_CODE, str(task_num), str(model_path)],
                          cwd=str(ROOT), text=True, capture_output=True, timeout=600)
    if proc.returncode != 0 or not proc.stdout.strip():
        return {"ok": False, "fail": None, "cost": None, "error": (proc.stderr or "no output")[-500:]}
    return json.loads(proc.stdout.strip().splitlines()[-1])

def deployed_cost(task_num: int) -> int | None:
    mpath = STATE / "manifest.json"
    if mpath.exists():
        row = json.load(open(mpath)).get(f"{task_num:03d}")
        if row and row.get("cost") is not None:
            return int(row["cost"])
    res = eval_isolated(OVERFIT_NETS / f"task{task_num:03d}.onnx", task_num)
    return res.get("cost")

@dataclass
class GateResult:
    ok: bool
    reasons: list[str] = field(default_factory=list)
    candidate: dict = field(default_factory=dict)
    incumbent_cost: int | None = None

def gate(candidate: Path, task_num: int) -> GateResult:
    reasons: list[str] = []
    cand = eval_isolated(Path(candidate), task_num)
    if not cand.get("ok") or cand.get("fail") != 0:
        reasons.append(f"bundled fail != 0 (fail={cand.get('fail')}, error={cand.get('error')})")
    inc = deployed_cost(task_num)
    if cand.get("cost") is None or inc is None or cand["cost"] >= inc:
        reasons.append(f"not strictly cheaper (cand={cand.get('cost')}, deployed={inc})")
    offenders = find_unsigned_topk(Path(candidate))
    if offenders:
        reasons.append("unsigned TopK: " + "; ".join(offenders))
    return GateResult(ok=not reasons, reasons=reasons, candidate=cand, incumbent_cost=inc)
```

- [ ] **Step 4: 통과 확인** — `uv run pytest tests/test_topk.py tests/test_gate.py -v` → PASS. 실물 스모크: `uv run python -c "from neurogolf.gate import gate; from neurogolf.paths import OVERFIT_NETS; r = gate(OVERFIT_NETS/'task001.onnx', 1); print(r.ok, r.reasons)"` → Expected: `False ['not strictly cheaper ...']` (자기 자신은 same-cost라 거부 — 게이트 방향 실증).

- [ ] **Step 5: 커밋** — `git add -A && git commit -m "feat: gate module — the single adoption gate (isolated eval + cheaper + topk)"` (트레일러 포함)

### Task 10: manifest + adopt 모듈

**Files:**
- Create: `src/neurogolf/manifest.py`, `src/neurogolf/adoption.py`
- Test: `tests/test_adoption.py`

**Interfaces:**
- Produces:
  - `neurogolf.manifest.load() -> dict[str, dict]` (키 `"001".."400"`), `save(m) -> None` (경로 `state/manifest.json`), `update_row(task_num: int, row: dict) -> None`, `total_points(m) -> float`. row 키: `task, cost, points, ok, fail, sha256, updated`.
  - `neurogolf.adoption.adopt(candidate: Path, task_num: int, note: str = "") -> dict` — gate 재실행 → 실패 시 `SystemExit(reasons)` → 백업(`submission/.backups/taskNNN_<utc-ts>.onnx`) → 교체 → manifest row 갱신 → `state/tasks/taskNNN.md`에 스탬프 append. 반환 = 새 manifest row.
- Consumes: `neurogolf.gate.gate`, `neurogolf.paths.*`

- [ ] **Step 1: 실패하는 테스트 작성**

```python
# tests/test_adoption.py
import json
from pathlib import Path
from neurogolf import adoption, gate

def _setup(tmp_path, monkeypatch):
    nets = tmp_path / "submission" / "overfit_nets"; nets.mkdir(parents=True)
    (nets / "task001.onnx").write_bytes(b"OLD")
    state = tmp_path / "state"; (state / "tasks").mkdir(parents=True)
    (state / "tasks" / "task001.md").write_text("# task001\n")
    json.dump({}, open(state / "manifest.json", "w"))
    for mod in (adoption, adoption.manifest):
        monkeypatch.setattr(mod, "STATE", state, raising=False)
    monkeypatch.setattr(adoption, "OVERFIT_NETS", nets)
    monkeypatch.setattr(adoption, "BACKUPS", tmp_path / "submission" / ".backups")
    return nets, state

def test_adopt_replaces_backs_up_and_stamps(tmp_path, monkeypatch):
    nets, state = _setup(tmp_path, monkeypatch)
    cand = tmp_path / "cand.onnx"; cand.write_bytes(b"NEW")
    ok = gate.GateResult(ok=True, candidate={"cost": 100, "points": 20.0, "ok": True, "fail": 0}, incumbent_cost=500)
    monkeypatch.setattr(adoption, "gate_candidate", lambda c, t: ok)
    row = adoption.adopt(cand, 1, note="test win")
    assert (nets / "task001.onnx").read_bytes() == b"NEW"
    assert list((tmp_path / "submission" / ".backups").glob("task001_*.onnx"))
    assert json.load(open(state / "manifest.json"))["001"]["cost"] == 100
    assert "test win" in (state / "tasks" / "task001.md").read_text()

def test_adopt_refuses_on_gate_failure(tmp_path, monkeypatch):
    nets, _ = _setup(tmp_path, monkeypatch)
    cand = tmp_path / "cand.onnx"; cand.write_bytes(b"NEW")
    bad = gate.GateResult(ok=False, reasons=["bundled fail != 0"])
    monkeypatch.setattr(adoption, "gate_candidate", lambda c, t: bad)
    import pytest
    with pytest.raises(SystemExit):
        adoption.adopt(cand, 1)
    assert (nets / "task001.onnx").read_bytes() == b"OLD"
```

- [ ] **Step 2: 실패 확인** — `uv run pytest tests/test_adoption.py -v` → FAIL

- [ ] **Step 3: 구현**

```python
# src/neurogolf/manifest.py
import json
from neurogolf.paths import STATE

def _path():
    return STATE / "manifest.json"

def load() -> dict:
    return json.load(open(_path())) if _path().exists() else {}

def save(m: dict) -> None:
    json.dump(m, open(_path(), "w"), indent=1, sort_keys=True)

def update_row(task_num: int, row: dict) -> None:
    m = load(); m[f"{task_num:03d}"] = row; save(m)

def total_points(m: dict) -> float:
    return sum(r["points"] for r in m.values())
```

```python
# src/neurogolf/adoption.py
import hashlib, shutil
from datetime import datetime, timezone
from pathlib import Path
from neurogolf import manifest
from neurogolf.gate import gate as gate_candidate
from neurogolf.paths import OVERFIT_NETS, ROOT, STATE

BACKUPS = ROOT / "submission" / ".backups"

def adopt(candidate: Path, task_num: int, note: str = "") -> dict:
    res = gate_candidate(Path(candidate), task_num)
    if not res.ok:
        raise SystemExit("gate REJECT: " + " | ".join(res.reasons))
    target = OVERFIT_NETS / f"task{task_num:03d}.onnx"
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    BACKUPS.mkdir(parents=True, exist_ok=True)
    shutil.copy2(target, BACKUPS / f"task{task_num:03d}_{ts}.onnx")
    shutil.copy2(candidate, target)
    row = {"task": task_num, "cost": res.candidate["cost"], "points": res.candidate["points"],
           "ok": True, "fail": 0, "sha256": hashlib.sha256(target.read_bytes()).hexdigest(),
           "updated": ts}
    manifest.update_row(task_num, row)
    stamp = (f"\n## ADOPTED {ts}\n- cost: {res.incumbent_cost} -> {res.candidate['cost']}"
             f" (points {res.candidate['points']:.4f})\n- source: {candidate}\n- note: {note}\n")
    log = STATE / "tasks" / f"task{task_num:03d}.md"
    log.parent.mkdir(parents=True, exist_ok=True)
    log.touch(exist_ok=True)
    log.write_text(log.read_text() + stamp)
    return row
```

- [ ] **Step 4: 통과 확인** — `uv run pytest tests/test_adoption.py -v` → PASS
- [ ] **Step 5: 커밋** — `git add -A && git commit -m "feat: manifest + adopt (backup, replace, row update, tasklog stamp)"` (트레일러 포함)

### Task 11: pack + submit 모듈

**Files:**
- Create: `src/neurogolf/pack.py`, `src/neurogolf/submit.py`
- Test: `tests/test_pack.py`
- 참조(이식 원본): `reports/scripts/pack_overfit_submission.py`

**Interfaces:**
- Produces:
  - `neurogolf.pack.pack(nets_dir: Path | None = None, out: Path | None = None) -> Path` — 400개 정확히 아니면 `SystemExit`; 전 파일 `find_unsigned_topk` clean 아니면 `SystemExit`; flat zip을 `ROOT/submission.zip`으로.
  - `neurogolf.submit.submit(message: str, zip_path: Path | None = None) -> None` — ①`kaggle competitions submissions -c neurogolf-2026` 최신 5건 출력(병렬 세션 확인) ②`kaggle competitions submit -c neurogolf-2026 -f submission.zip -m <message>` ③`state/submissions.md`에 `| ts | message |` 행 append.
  - `neurogolf.submit.latest_submissions(n: int = 5) -> str`

- [ ] **Step 1: 실패하는 테스트 작성**

```python
# tests/test_pack.py
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
```

- [ ] **Step 2: 실패 확인** — `uv run pytest tests/test_pack.py -v` → FAIL

- [ ] **Step 3: 구현**

```python
# src/neurogolf/pack.py
import zipfile
from pathlib import Path
from neurogolf.paths import OVERFIT_NETS, ROOT
from neurogolf.topk import find_unsigned_topk

def pack(nets_dir: Path | None = None, out: Path | None = None) -> Path:
    nets_dir = nets_dir or OVERFIT_NETS
    out = out or (ROOT / "submission.zip")
    files = sorted(nets_dir.glob("task*.onnx"))
    if len(files) != 400:
        raise SystemExit(f"pack REFUSED: {len(files)} nets (400 필요)")
    offenders = [f"{f.name}: {o}" for f in files for o in find_unsigned_topk(f)]
    if offenders:
        raise SystemExit("pack REFUSED, unsigned TopK:\n" + "\n".join(offenders))
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as z:
        for f in files:
            z.write(f, arcname=f.name)
    return out
```

```python
# src/neurogolf/submit.py
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from neurogolf.paths import ROOT, STATE

COMP = "neurogolf-2026"

def latest_submissions(n: int = 5) -> str:
    proc = subprocess.run(["kaggle", "competitions", "submissions", "-c", COMP],
                          text=True, capture_output=True, check=True)
    return "\n".join(proc.stdout.splitlines()[: n + 2])

def submit(message: str, zip_path: Path | None = None) -> None:
    zip_path = zip_path or (ROOT / "submission.zip")
    if zip_path.name != "submission.zip":
        raise SystemExit("제출 파일명은 submission.zip이어야 함")
    print("=== 최근 제출 (병렬 세션 확인) ===\n" + latest_submissions())
    subprocess.run(["kaggle", "competitions", "submit", "-c", COMP,
                    "-f", str(zip_path), "-m", message], check=True)
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%MZ")
    log = STATE / "submissions.md"
    log.write_text(log.read_text() + f"\n| {ts} | {message} |")
```

- [ ] **Step 4: 통과 확인** — `uv run pytest tests/test_pack.py -v` → PASS
- [ ] **Step 5: 커밋** — `git add -A && git commit -m "feat: pack (400+topk enforced) + submit (kaggle wrapper with log)"` (트레일러 포함)

### Task 12: verify 모듈 + cli 배선

**Files:**
- Create: `src/neurogolf/verify.py`, `src/neurogolf/cli.py`
- Test: `tests/test_verify.py`

**Interfaces:**
- Produces:
  - `neurogolf.verify.hash_check() -> list[str]` — 배포본 sha256를 `state/manifest.json`(row에 sha256 있을 때) 또는 `state/baseline/sha256.txt`(부트스트랩)와 대조, 불일치 파일명 리스트 반환.
  - `neurogolf.verify.full_verify(update: bool = False) -> dict` — 400 전 태스크 `eval_isolated` 재측정, `{"n_ok": int, "total_points": float, "failures": [...]}` 반환; `update=True`면 state/manifest.json 재작성(sha256 포함).
  - `neurogolf.cli.main()` — 서브커맨드: `status, score, gate, adopt, pack, submit, scan, queue, mine-public, verify`. 시작 시 `paths.ensure_src_importable()` 호출.
- Consumes: 위 모든 모듈.

- [ ] **Step 1: 실패하는 테스트 작성**

```python
# tests/test_verify.py
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
```

- [ ] **Step 2: 실패 확인** — `uv run pytest tests/test_verify.py -v` → FAIL

- [ ] **Step 3: 구현**

```python
# src/neurogolf/verify.py
import hashlib, json
from concurrent.futures import ThreadPoolExecutor
from neurogolf import manifest
from neurogolf.gate import eval_isolated
from neurogolf.paths import OVERFIT_NETS, STATE

def _baseline_file():
    return STATE / "baseline" / "sha256.txt"

def hash_check() -> list[str]:
    m = manifest.load()
    expected: dict[str, str] = {f"task{k}.onnx": r["sha256"] for k, r in m.items() if r.get("sha256")}
    if not expected:
        expected = {line.split()[1]: line.split()[0]
                    for line in _baseline_file().read_text().splitlines() if line.strip()}
    bad = []
    for name, sha in sorted(expected.items()):
        p = OVERFIT_NETS / name
        if not p.exists() or hashlib.sha256(p.read_bytes()).hexdigest() != sha:
            bad.append(name)
    return bad

def full_verify(update: bool = False) -> dict:
    def one(n):
        return n, eval_isolated(OVERFIT_NETS / f"task{n:03d}.onnx", n)
    with ThreadPoolExecutor(max_workers=8) as ex:
        rows = dict(ex.map(one, range(1, 401)))
    failures = [n for n, r in rows.items() if not (r.get("ok") and r.get("fail") == 0)]
    total = sum(r.get("points") or 0.0 for r in rows.values())
    if update and not failures:
        import hashlib as h
        m = {}
        for n, r in rows.items():
            p = OVERFIT_NETS / f"task{n:03d}.onnx"
            m[f"{n:03d}"] = {"task": n, "cost": r["cost"], "points": r["points"], "ok": True,
                             "fail": 0, "sha256": h.sha256(p.read_bytes()).hexdigest(), "updated": "verify"}
        manifest.save(m)
    return {"n_ok": 400 - len(failures), "total_points": total, "failures": failures}
```

```python
# src/neurogolf/cli.py
import argparse, json, subprocess, sys
from pathlib import Path
from neurogolf import paths

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="ng", description="NeuroGolf lever engine CLI")
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("status")
    s = sub.add_parser("score"); s.add_argument("tasks", nargs="+", type=int)
    s = sub.add_parser("gate"); s.add_argument("onnx", type=Path); s.add_argument("--task", type=int, required=True)
    s = sub.add_parser("adopt"); s.add_argument("onnx", type=Path); s.add_argument("--task", type=int, required=True); s.add_argument("--note", default="")
    sub.add_parser("pack")
    s = sub.add_parser("submit"); s.add_argument("-m", "--message", required=True)
    s = sub.add_parser("scan"); s.add_argument("lever"); s.add_argument("--tasks", nargs="*", type=int)
    sub.add_parser("queue")
    s = sub.add_parser("mine-public"); s.add_argument("dumps", nargs="+"); s.add_argument("--margin", type=int, default=0); s.add_argument("--apply", action="store_true")
    s = sub.add_parser("verify"); s.add_argument("--hash", action="store_true"); s.add_argument("--update", action="store_true")
    return p

def main() -> None:
    paths.ensure_src_importable()
    args = build_parser().parse_args()
    if args.cmd == "status":
        from neurogolf import manifest, verify
        m = manifest.load()
        print(f"nets: {len(list(paths.OVERFIT_NETS.glob('task*.onnx')))}/400")
        if m: print(f"manifest total points: {manifest.total_points(m):.4f}")
        dirty = subprocess.run(["git", "status", "--porcelain"], capture_output=True, text=True, cwd=paths.ROOT).stdout
        if dirty: print(f"⚠ 미커밋 변경 {len(dirty.splitlines())}건")
        state = (paths.STATE / 'STATE.md')
        print(state.read_text()[:800] if state.exists() else "STATE.md 없음")
    elif args.cmd == "score":
        from neurogolf.gate import eval_isolated
        for t in args.tasks:
            print(t, json.dumps(eval_isolated(paths.OVERFIT_NETS / f"task{t:03d}.onnx", t)))
    elif args.cmd == "gate":
        from neurogolf.gate import gate
        r = gate(args.onnx, args.task)
        print(("PASS" if r.ok else "REJECT"), r.reasons or "", json.dumps(r.candidate))
        sys.exit(0 if r.ok else 1)
    elif args.cmd == "adopt":
        from neurogolf.adoption import adopt
        print(json.dumps(adopt(args.onnx, args.task, args.note)))
    elif args.cmd == "pack":
        from neurogolf.pack import pack
        print("packed:", pack())
    elif args.cmd == "submit":
        from neurogolf.submit import submit
        submit(args.message)
    elif args.cmd == "scan":
        from neurogolf.scans import run_scan
        print("worklist:", run_scan(args.lever, args.tasks))
    elif args.cmd == "queue":
        from neurogolf.scans import show_queue
        show_queue()
    elif args.cmd == "mine-public":
        from neurogolf.scans.minmerge import mine
        mine([Path(d) for d in args.dumps], margin=args.margin, apply=args.apply)
    elif args.cmd == "verify":
        from neurogolf import verify as v
        if args.hash:
            bad = v.hash_check()
            print("HASH-OK" if not bad else f"MUTATED: {bad}"); sys.exit(1 if bad else 0)
        r = v.full_verify(update=args.update)
        print(json.dumps(r)); sys.exit(0 if r["n_ok"] == 400 else 1)
```
(주: `scan/queue/mine-public` 분기는 Task 13 완료 전까지 ImportError — Task 13에서 해소.)

- [ ] **Step 4: 통과 확인 + 실기동**

```bash
uv sync --dev && uv run pytest tests/test_verify.py -v
uv run ng verify --hash    # Expected: HASH-OK (baseline과 대조)
uv run ng score 1          # Expected: task001 row JSON, cost == baseline과 동일
```

- [ ] **Step 5: 커밋** — `git add -A && git commit -m "feat: ng cli + verify (hash + full isolated rescore)"` (트레일러 포함)

### Task 13: 스캐너 이식 + 레지스트리

**Files:**
- Create: `src/neurogolf/scans/__init__.py`, `src/neurogolf/scans/mask_dominance.py`, `src/neurogolf/scans/kernel_collapse.py`, `src/neurogolf/scans/fold.py`, `src/neurogolf/scans/dtype_overpay.py`, `src/neurogolf/scans/minmerge.py`, `src/neurogolf/scans/fresh.py`
- Test: `tests/test_scans_registry.py`
- 이식 원본: `reports/scripts/{mask_dominance_scan,kernel_collapse,fold_finder,dtype_overpay_scan,mine_overfit_minmerge,fresh_verify}.py`

**Interfaces:**
- Produces:
  - `neurogolf.scans.SCANNERS: dict[str, Callable[[list[int] | None], dict]]` — 이름 = levers.yaml의 `scanner:` 값 (`mask_dominance, kernel_collapse, fold, dtype_overpay`).
  - `neurogolf.scans.run_scan(name: str, tasks: list[int] | None = None) -> Path` — 스캐너 실행, 결과를 `candidates/worklists/<name>.json`에 `{"lever": name, "generated": ts, "items": [{"task": int, "expected_gain": float, ...}]}` 형식으로 저장.
  - `neurogolf.scans.show_queue() -> None` — worklists/*.json 병합, expected_gain 내림차순 출력.
  - `neurogolf.scans.minmerge.mine(dumps: list[Path], margin: int = 0, apply: bool = False) -> list[dict]`
  - `neurogolf.scans.fresh.fresh_check(task_num: int, candidate: Path | None = None, n: int = 1500) -> tuple[int, int]` (통과수, 실행수) — 진단용.

- [ ] **Step 1: 실패하는 테스트 작성**

```python
# tests/test_scans_registry.py
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
```

- [ ] **Step 2: 실패 확인** — `uv run pytest tests/test_scans_registry.py -v` → FAIL

- [ ] **Step 3: 레지스트리 구현**

```python
# src/neurogolf/scans/__init__.py
import json
from datetime import datetime, timezone
from pathlib import Path
from neurogolf.paths import CANDIDATES

WORKLISTS = CANDIDATES / "worklists"

from neurogolf.scans.mask_dominance import scan_all as _mask_dominance
from neurogolf.scans.kernel_collapse import scan_all as _kernel_collapse
from neurogolf.scans.fold import scan_all as _fold
from neurogolf.scans.dtype_overpay import scan_all as _dtype_overpay

SCANNERS = {
    "mask_dominance": _mask_dominance,
    "kernel_collapse": _kernel_collapse,
    "fold": _fold,
    "dtype_overpay": _dtype_overpay,
}

def run_scan(name: str, tasks: list[int] | None = None) -> Path:
    if name not in SCANNERS:
        raise SystemExit(f"unknown scanner '{name}' — 등록: {sorted(SCANNERS)}")
    result = SCANNERS[name](tasks)
    WORKLISTS.mkdir(parents=True, exist_ok=True)
    out = WORKLISTS / f"{name}.json"
    payload = {"lever": name, "generated": datetime.now(timezone.utc).isoformat(),
               "items": sorted(result["items"], key=lambda i: -i.get("expected_gain", 0.0))}
    out.write_text(json.dumps(payload, indent=1))
    return out

def show_queue() -> None:
    items = []
    for f in WORKLISTS.glob("*.json") if WORKLISTS.exists() else []:
        d = json.loads(f.read_text())
        items += [{**i, "lever": d["lever"]} for i in d["items"]]
    for i in sorted(items, key=lambda i: -i.get("expected_gain", 0.0))[:40]:
        print(f"task{i['task']:03d}  +{i.get('expected_gain', 0):.3f}  {i['lever']}")
```

- [ ] **Step 4: 개별 스캐너 이식 (기계적 규칙)**

각 원본 스크립트에 대해 동일한 변환을 적용한다 — **로직 변경 금지, 다음 3가지만**:
1. 하드코딩 경로 제거: `ROOT = Path("/Users/minseong/...")` / `parents[2]` 패턴 → `from neurogolf.paths import ROOT, OVERFIT_NETS`. 스캔 대상은 항상 `OVERFIT_NETS` (dtype_overpay는 원본이 `networks/`를 겨냥 — **`OVERFIT_NETS`로 재조준**, 이것이 이식의 목적).
2. flat 스크립트/`sys.argv` 파싱 → `def scan_all(tasks: list[int] | None = None) -> dict` 함수로 감싸고 `{"items": [{"task": n, "expected_gain": g, ...원본 필드}]}` 반환. `expected_gain`은 원본이 절감 바이트를 내면 `saved_bytes` 기준 근사 `points_delta ≈ ln(cost)/…` 대신 단순히 `saved_bytes / cost` 비율로 정렬용 산출 (정확한 점수는 gate가 판정).
3. `sys.path.insert(0,"src")` / `from harness import` → `from neurogolf.scoring import`.

`minmerge.py`: `mine_overfit_minmerge.py`의 `static_cost/find_net/main`을 이식하되 `--apply`가 직접 파일을 덮지 않고 **후보별로 `neurogolf.adoption.adopt()`를 호출**하도록 변경 (게이트 단일화 — 원본의 자체 verify는 제거).
`fresh.py`: `fresh_verify.py`의 캐시 경로를 `CANDIDATES / "fresh_cache"`로 변경해 이식.
`kernel_collapse.py`: import 시 실행되는 top-level 루프를 `scan_all` 내부로 이동, 출력 후보는 `CANDIDATES / f"task{n:03d}" / "kcollapse.onnx"`.

- [ ] **Step 5: 통과 확인 + 실기동**

```bash
uv run pytest tests/test_scans_registry.py -v
uv run ng scan mask_dominance          # Expected: candidates/worklists/mask_dominance.json 생성
uv run ng queue                        # Expected: 기대이득 순 목록 (기존 mask_dominance.json의 ~45 잔여와 대조)
```

- [ ] **Step 6: 커밋** — `git add -A && git commit -m "feat: scanner registry + port live-lever scanners to neurogolf.scans"` (트레일러 포함)

### Task 14: 삭제 스윕 + 잔여 경로 정리 + Phase 2 검증

**Files:**
- Modify: `src/pipeline.py` (`MANIFEST = REPORTS/"manifest.json"` → `STATE/"safe_manifest.json"`; `REPORTS/"SCOREBOARD.md"` → `STATE/"SCOREBOARD.md"`)
- Move: `git mv reports/manifest.json state/safe_manifest.json`, `git mv reports/scripts/rebuild_networks_from_source.py tools/`
- Delete: `reports/` 잔여 전부, `tests/test_{match_insight,build_task_index,task_index_probes,coverage_lib,backfill_validation}.py`
- Modify: `.gitignore`에 `candidates/` 추가 (기존 `reports/candidates` 규칙 제거)

- [ ] **Step 1: 이동/경로 수정**

```bash
git mv reports/manifest.json state/safe_manifest.json
git mv reports/scripts/rebuild_networks_from_source.py tools/
```
`src/pipeline.py`의 두 경로 상수 수정. `tools/rebuild_networks_from_source.py` 상단 경로 상수도 `parents[2]` → `parents[1]` 보정. `src/genverify.py`의 출력 경로 `reports/genverify.json` → `state/genverify.json`으로 수정 (reports/ 삭제 대비).

- [ ] **Step 2: REBUILD_PLAYBOOK 임시 대피** (Task 15에서 분해할 원본)

```bash
mkdir -p playbook && git mv reports/REBUILD_PLAYBOOK.md playbook/_LEGACY_PLAYBOOK.md
```

- [ ] **Step 3: 삭제 실행**

```bash
git rm -r reports
git rm tests/test_match_insight.py tests/test_build_task_index.py \
       tests/test_task_index_probes.py tests/test_coverage_lib.py \
       tests/test_backfill_validation.py
```
(`reports/candidates`의 백업들은 Phase 0 스냅샷 + `pre-redesign` 태그로 복구 가능. levers.yaml의 `scanner_archive: git:pre-redesign:...` 참조가 이 태그를 가리킴.)

- [ ] **Step 4: Phase 2 종합 검증**

```bash
uv run pytest -v                                   # 남은 전체 테스트 PASS
uv run ng verify --hash                            # HASH-OK
uv run ng verify --update                          # {"n_ok": 400, "total_points": <기준값과 동일±1e-6>, ...}
uv run python -c "import json; b=json.load(open('state/baseline/points.json')); m=json.load(open('state/manifest.json')); s=sum(r['points'] for r in m.values()); assert abs(s-b['total_points'])<1e-6, (s,b); print('POINTS-INVARIANT-OK', s)"
uv run ng pack                                     # packed: .../submission.zip
```
Expected: 전부 통과. `ng verify --update`로 live manifest(state/manifest.json)가 sha256 포함으로 생성됨.

- [ ] **Step 5: 커밋**

```bash
git add -A && git commit -m "chore: legacy deletion sweep — reports/ removed, paths consolidated, invariant verified

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

# Phase 3 — 레버 엔진

### Task 15: playbook/ 분해

**Files:**
- Create: `playbook/README.md`(색인), `playbook/free-output-einsum.md`, `playbook/fp16-recast.md`, `playbook/kernel-collapse.md`, `playbook/minmerge.md`, `playbook/walk-einsum.md`, `playbook/signed-einsum-routing.md`
- Delete(분해 후): `playbook/_LEGACY_PLAYBOOK.md`

- [ ] **Step 1: `_LEGACY_PLAYBOOK.md`를 메커니즘 단위로 분해** — 각 파일 구조 고정: `## 언제 쓰나(스캐너/시그널)` `## 물리(왜 되나)` `## 레시피(단계)` `## 서브패턴` `## 함정/거부 사례`. free-output-einsum.md에는 NEXT_SESSION에서 증류한 batch3 서브레시피(input-as-Einsum-operand free counts, base-N digit factorization, ConvInteger-as-free-output, k-stacked Where carriers, residue one-hots)와 positioned-content = floor 택소노미를 반드시 포함.
- [ ] **Step 2: levers.yaml의 `recipe:` 경로가 전부 실존하는지 확인**

```bash
uv run python -c "
import yaml, pathlib
for l in yaml.safe_load(open('state/levers.yaml'))['levers']:
    if l.get('recipe'): assert pathlib.Path(l['recipe']).exists(), l['recipe']
print('recipes ok')"
```
- [ ] **Step 3: 커밋** — `git rm playbook/_LEGACY_PLAYBOOK.md` 후 커밋 (트레일러 포함)

### Task 16: 세션 운영 스킬 재작성

**Files:**
- Create: `skills/neurogolf/SKILL.md`
- Delete: `skills/neurogolf-recursive-improvement/`

- [ ] **Step 1: SKILL.md 작성** — 내용 구조 고정:
  1. **세션 시작**: `uv run ng status` → `state/STATE.md` → levers.yaml에서 live 레버 선택.
  2. **표준 루프**: `ng scan <lever>` → 워크리스트 상위 N개에 에이전트 팬아웃(agent_class 준수: opus=레시피 기계적용, fable=신규 크랙; 에이전트 프롬프트에 recipe 파일 + `state/tasks/NNN.md` + 배포 onnx 경로 포함, 산출물은 `candidates/taskNNN/`) → `ng gate`/`ng adopt` → 배치마다 `ng pack` → `ng submit` → 결과 기록(승리=adopt가 자동 스탬프, dry=levers.yaml ledger 4필드).
  3. **에피스테믹 룰** (기존 스킬에서 이관): 부정 판정 4필드 강제, floor는 독립 최소치로만, 레버는 dormant never dead.
  4. **세션 종료**: STATE.md 교체(append 금지), 커밋.
  5. **안전 레일 요약**: 게이트 우회 금지 / submission.zip / 병렬 세션 시 kaggle submissions 확인 / 100회 한도.
- [ ] **Step 2: 구 스킬 삭제 + AGENTS.md의 스킬 경로 갱신 + 커밋**

```bash
git rm -r skills/neurogolf-recursive-improvement
git add -A && git commit -m "skills: rewrite session skill around the lever-engine loop

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

### Task 17: E2E 스모크 + 마무리

**Files:**
- Modify: `state/STATE.md` (스모크 결과 반영), `~/.claude/projects/-Users-minseong-project-neurogolf/memory/MEMORY.md` + 관련 메모리 파일

- [ ] **Step 1: 엔진 E2E 스모크 (결정론적)**

```bash
uv run ng scan mask_dominance && uv run ng queue          # 워크리스트 생성·랭킹
uv run ng gate submission/overfit_nets/task001.onnx --task 1; test $? -eq 1 && echo GATE-REJECT-OK
uv run ng verify --hash && uv run ng pack                  # HASH-OK + packed
```
Expected: 워크리스트 출력, `GATE-REJECT-OK`(자기 자신 = not cheaper 거부 경로 실증), HASH-OK, packed. (pass 경로는 Task 9 단위 테스트가 보증; 실제 크랙 채택은 점수 작업 재개 시 첫 사이클이 담당.)

- [ ] **Step 2: 외부 메모리 갱신** — MEMORY.md의 리포 경로 참조를 새 구조(`state/STATE.md`, `ng` CLI, `playbook/`)로 갱신; `reports/…` 경로를 언급하는 메모리 파일들의 포인터 수정. 특히 "Live best score + queue = NEXT_SESSION.md" 라인을 `state/STATE.md`로.

- [ ] **Step 3: 최종 커밋 + 완료 보고**

```bash
uv run pytest -q && uv run ng verify --hash
git add -A && git commit -m "feat: lever engine complete — redesign done, invariant held

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
git log --oneline pre-redesign..HEAD
```
