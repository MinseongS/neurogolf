# NeuroGolf 전면 재설계 — 설계 문서 (2026-07-09)

## 목적과 제약

- 목적: (1) 마감(2026-07-15)까지 남은 6일의 점수 작업 효율 극대화, (2) 마감 후에도 쓸 수 있는
  장기 기반. 사용자 지시 = "둘 다, 지금 전면 재설계".
- 진단된 병목 3개 (사용자 확인): **지식/상태 파편화** (NEXT_SESSION.md 754줄 + AGENTS.md +
  외부 메모리 + insight_registry + tasklog 400개에 분산, stale 기록이 배포넷과 어긋남),
  **도구 난립** (reports/scripts 57개 일회성 스크립트), **레버 실행 체계 부재** (확인된 레버를
  400태스크에 체계적으로 돌리는 재현 가능한 루프 없음).
- 레거시 처리: **과감 삭제 + 지식만 이주**. git 히스토리가 복구 수단.
- 실행 모델: **하이브리드** — 기계적 파이프라인(스캔→게이트→채택→패킹→제출)은 단일 Python CLI,
  창의적 부분(태스크별 크랙 설계)은 에이전트 팬아웃, 단 채택은 반드시 CLI 게이트 경유.
- uv는 이미 도입됨 (pyproject.toml + uv.lock + .venv). 남은 일 = package 모드 전환 + CLI 엔트리.

## 불가침 자산

- `submission/overfit_nets/` — 활성 제출셋 (LB 7279.41 상태, 일부 working tree에만 존재).
- `src/custom/taskNNN.py` — 태스크별 빌드 소스.
- `data/`, `arc-gen/`, `networks/`(소스 재생성 아티팩트로 강등하되 유지).
- 채점 불변식: 제출파일명 `submission.zip`, unsigned-TopK 금지, bundled fail=0 게이트,
  100회/일, 고정 데이터셋(한번 통과 = 영구 통과).

## 목표 구조

```
neurogolf/
├── pyproject.toml, uv.lock, .python-version   # uv 소유; package 모드 + [project.scripts] ng
├── README.md                                   # 얇게: 개요 + 실행법
├── AGENTS.md                                   # 얇게: "state/STATE.md 읽고 시작" + 불변 규칙만
├── state/                     # ★ 단일 진실 소스
│   ├── STATE.md               # live 핸드오프. ~100줄 상한. append 금지, 세션 종료 시 교체.
│   ├── levers.yaml            # 레버 레지스트리 (아래 스키마)
│   ├── insights.yaml          # 기존 insight_registry.yaml 이주
│   ├── submissions.md         # 제출 로그 이주
│   └── tasks/taskNNN.md       # tasklog 이주 + stale 마킹 (배포넷과 대조)
├── playbook/                  # 에이전트가 읽는 메커니즘 레시피 (REBUILD_PLAYBOOK 분해)
│   └── free-output-einsum.md, fp16-recast.md, kernel-collapse.md, minmerge.md, …
├── src/neurogolf/             # 파이썬 패키지
│   ├── cli.py                 # ng 명령
│   ├── scoring.py, gate.py, adopt.py, pack.py, submit.py
│   └── scans/                 # 살아있는 스캐너 ~12개 모듈화
├── src/custom/                # 유지 (불가침)
├── submission/overfit_nets/   # 유지 (불가침)
├── networks/, data/, arc-gen/ # 유지
├── candidates/                # 작업 영역 (gitignore; 기존 reports/candidates 대체)
├── tests/                     # 유지 + 게이트 불변식 테스트 추가
└── skills/neurogolf/          # 세션 운영 스킬 (레버 엔진 루프 매뉴얼)
```

## 상태 모델 원칙

1. **STATE.md는 교체만, append 금지.** "현재 참인 것"만 담는다. 히스토리는 git 커밋과
   state/submissions.md 담당. 754줄 핸드오프의 재발 방지.
2. **레버 판정은 levers.yaml에서만 산다.** 부정 판정(소진/floor/dry)은 반드시
   `{ran, 도구+날짜, reopen 트리거, 반증 이력}` 4필드 ledger로만 기록. status는
   live/dormant 두 값뿐 — dead 없음 (에피스테믹 룰의 스키마 강제).
3. **tasklog 이주 시 stale 청산.** 각 파일 헤더에 "이 기록이 설명하는 넷 = 현재 배포본인가?"
   필드를 추가하고, 어긋나면 stale 마크 (fat-middle backlog 오판의 구조적 재발 방지).

## ng CLI 명세

```
ng status                      # 로컬 점수, 확정 LB 최고, 배포셋 요약, 미커밋 diff 경고
ng score  <task…|--all>        # grader-동일 cost/points (태스크별 격리 프로세스)
ng gate   <onnx> --task NNN    # 채택 게이트: ①isolated bundled fail=0 ②cost < 배포본
                               #   ③unsigned-TopK 검사. 셋 다 통과 시에만 exit 0.
ng adopt  <onnx> --task NNN    # gate 재실행 → 백업 → 교체 → manifest 갱신 → tasklog 스탬프
ng pack                        # manifest --isolated-all → 400 전수 TopK 스캔 → submission.zip
                               #   (스캔 실패 시 패킹 거부)
ng submit -m "…"               # kaggle 제출 + 최신 submissions 확인 + submissions.md 기록
ng scan   <lever> [--tasks …]  # levers.yaml 등록 스캐너 실행 → 후보 워크리스트 JSON
ng queue                       # 후보 워크리스트 기대이득 순 랭킹
ng mine-public <dirs…>         # 공개덤프 margin-0 min-merge
ng verify                      # 400/400 isolated + 기준 점수 불변성 체크 (마이그레이션 검증용)
```

설계 결정:
- **gate가 유일한 관문** — adopt가 내부에서 gate를 재실행하므로 우회 불가. 평가는 항상
  격리 프로세스 (knife-edge conv-flip 방지).
- **비교 기준은 항상 배포본** `submission/overfit_nets/` — networks/ 대비 스캔이 레버를
  "소진"으로 오판한 원인의 구조적 제거.
- 이식할 스캐너 (초기 12개): mask_dominance, scan_deployed_fp16, kernel_collapse,
  mine_overfit_minmerge, reducesum→einsum, scan_unsigned_topk, dtype_overpay(배포본 기준으로
  재조준), fold_finder, walk_chain_slack, public_win_autopsy, free_output_label_rescan,
  known_insight_coverage. Phase 2에서 실사용 빈도 기준 최종 확정.

## 레버 엔진

levers.yaml 엔트리 스키마:

```yaml
- name: free-output-einsum-regime-crack
  status: live            # live | dormant (dead 없음)
  scanner: mask_dominance # → ng scan mask_dominance
  recipe: playbook/free-output-einsum.md
  agent_class: fable      # fable(신규 메커니즘 설계) | opus(검증된 레시피 기계 적용)
  expected_yield: "+0.2~0.9/task, 잔여 ~45태스크"
  ledger:
    - date: 2026-07-08
      ran: "batch6 8태스크"
      verdict: "7/8 floor (positioned-content mask)"
      reopen: "새 sub-recipe 발견 시, 또는 non-mask bloat(Conv 3600B)로 일반화 시"
```

표준 실행 루프 (한 레버 = 한 사이클):

```
① ng scan <lever>        → 후보 워크리스트 (기대이득 순)
② 에이전트 팬아웃          → 후보별 1에이전트: recipe + state/tasks/NNN.md + 배포 onnx →
                            candidates/taskNNN/에 후보 생성 (opus=레시피, fable=신규 크랙)
③ ng gate → ng adopt     → 통과분만 채택
④ ng pack → ng submit    → 배치 단위 제출
⑤ 원장 기록               → 승리는 tasklog 자동 스탬프; dry는 levers.yaml ledger에
                            reopen 트리거와 함께 기록; STATE.md 활성 베인 갱신
```

skills/neurogolf/SKILL.md = 이 루프의 운영 매뉴얼로 재작성. 세션 시작 = `ng status` +
STATE.md → live 레버 선택 → 루프 → 세션 종료 = STATE.md 교체. 기존
recursive-improvement 스킬의 에피스테믹 룰은 스키마+스킬 양쪽에 보존.

## 마이그레이션 단계와 검증

**전 단계 공통 불변식:** overfit_nets 400넷 바이트 동일 + manifest isolated-all 400/400 +
로컬 점수 == Phase 0 기준값. 위반 시 해당 단계 롤백. 재설계 중 병렬 점수 세션 금지.

- **Phase 0 — 안전화 (~30분):** working tree 전체 커밋(미커밋 7279.41 채택분 포함) +
  `pre-redesign` 태그; `build_overfit_manifest.py --isolated-all`로 기준 점수 기록;
  overfit_nets repo 밖 스냅샷 1부.
- **Phase 1 — 상태 모델 (~반나절):** state/ 구축 및 지식 이주 (tasklog 400개 stale 마킹은
  에이전트 팬아웃); NEXT_SESSION.md 삭제; AGENTS.md/README 얇게 재작성.
  검증 = 점수 관련 파일 무접촉 + 기준값 재측정 일치. 커밋.
- **Phase 2 — CLI (~1일):** package 모드 전환, src/neurogolf/ 구축, 스캐너·게이트 이식,
  게이트 불변식 테스트 (uint8-TopK 검출, isolated eval, cheaper-than-deployed).
  삭제 = reports/scripts 일회성 ~45개, 재생성 가능 스캔 JSON/MD, data.zip,
  reports/candidates stale 후보. 검증 = `ng verify` == 기준값 + `ng pack` zip 파일목록/바이트
  대조 + pytest. 커밋.
- **Phase 3 — 레버 엔진 (~반나절):** levers.yaml 배선, playbook/ 분해, 스킬 재작성.
  검증 = 실제 레버 1건 엔드투엔드 스모크 (스캔→에이전트→게이트→채택) — 성공 시 그대로
  점수 작업 재개. 커밋. 마무리로 외부 메모리 MEMORY.md를 새 구조 기준으로 갱신.

총 소요 추정 ~2일. 남은 4일 = 새 엔진으로 마스크 베인(~45태스크) + CONV-FP32 스윕.

## 삭제 목록 (요약)

- `NEXT_SESSION.md` (STATE.md로 증류 후)
- `data.zip` (data/ 중복)
- `reports/scripts/` 중 CLI로 이식되지 않는 일회성 ~45개
- `reports/` 재생성 가능 스캔 산출물 JSON/MD (~30개)
- `reports/candidates/` stale 후보 디렉토리 (Phase 0 스냅샷 확보 후; 채택 백업 포함 정리)
- 이주 완료된 reports/tasklog/, insight_registry.yaml, submission_log.md 원본

## 테스트 전략

- 게이트 불변식 pytest: 알려진 grader-killer(uint8 TopK) 검출, 격리 평가 동작,
  cheaper-than-deployed 비교 방향.
- `ng verify` = 마이그레이션 회귀 테스트이자 상시 헬스체크.
- 기존 tests/ 중 살아있는 대상 (genverify 등)은 새 모듈 경로로 갱신, 죽은 대상은 삭제.

## 에러 처리

- gate/adopt/pack은 실패 시 명확한 사유와 함께 non-zero exit; adopt는 교체 전 백업을
  candidates/ 밖 `.backup/`에 강제.
- submit은 제출 전 `kaggle competitions submissions` 최신 확인을 강제 (병렬 세션 충돌 방지).
- 평가 stall(기존 task376 사례)은 태스크별 격리 + 타임아웃으로 격리.
