# NeuroGolf 2026

Kaggle NeuroGolf 2026 competition repo. LB 최고 7279.41. 마감 2026-07-15.

## Setup / 사용법

```bash
uv sync --dev
uv run ng --help
```

CLI(`ng`)는 Phase 2에서 생긴다. 그 전까지는 `reports/scripts/` 잔존 스크립트를 `uv run python`으로 직접 실행.

## 디렉토리 맵

- `state/` — 단일 진실 소스 (STATE.md, levers.yaml, insights.yaml, submissions.md, tasks/)
- `playbook/` — 에이전트용 메커니즘 레시피
- `src/neurogolf/` — `ng` CLI 패키지 (Phase 2)
- `src/custom/` — 태스크별 빌드 소스 (불가침)
- `submission/overfit_nets/` — 활성 제출 아티팩트 (불가침)
- `networks/` — 소스 재생성 ONNX 아티팩트 / `data/` — 대회 데이터
- `arc-gen/` — fresh-gate 생성기 / `candidates/` — 작업용 후보 스크래치
- `tests/` — 게이트 불변식·회귀 테스트 / `tools/` — 보조 유틸리티 / `skills/` — 세션 운영 스킬

## 문서

- `docs/superpowers/specs/2026-07-09-neurogolf-redesign-design.md`
- `docs/superpowers/plans/2026-07-09-neurogolf-redesign.md`
