# NeuroGolf Agent Instructions

세션 시작: `uv run ng status` → `state/STATE.md` → `state/levers.yaml`에서 live 레버 선택
→ `skills/neurogolf/SKILL.md`의 표준 루프 실행. 세션 종료: STATE.md 교체(append 금지).

## 불변 규칙
- 목표 8000. 8000 overfit 모드가 기본: 게이트 = bundled fail=0 + 배포본보다 cheaper. fresh 검증은 진단용.
- 채택은 반드시 `ng adopt` 경유 (gate 우회 금지). 제출은 `ng pack` → `ng submit`.
- FREE input/output 텐서를 공격적으로 활용. 공개 INSIGHT를 채굴해 400태스크로 일반화.
- 부정 판정(소진/floor)은 state/levers.yaml ledger 4필드로만 기록. 레버는 dormant, dead 없음.
- 후보/스크래치는 `candidates/` 아래에만. 병렬 세션 시 제출 전 `kaggle competitions submissions` 확인.
- 채점 환경 핀 고정: onnx==1.21.0 / onnxruntime==1.26.0 (pyproject) — 400/400 재검증 없이 업그레이드 금지.

## 진실 소스
- `state/STATE.md` live 핸드오프 / `state/levers.yaml` 레버 원장 / `state/tasks/` 태스크 원장
- `state/insights.yaml` 메커니즘 / `state/submissions.md` 제출 로그 / `playbook/` 레시피
- `submission/overfit_nets/` 배포본(불가침, adopt로만 변경) / `src/custom/` 빌드 소스
