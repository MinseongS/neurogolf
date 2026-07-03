# NEXT SESSION — NeuroGolf handoff (2026-07-03 S11 종료 시점)

Use this as the next-session prompt:

```text
/Users/minseong/project/neurogolf 에서 NeuroGolf 점수 개선을 이어가자.

현재 상태 (S11 종료, 2026-07-03):
- 확정 LB 최고점 = **7214.42** (54295163, S11 최종. 로컬 격리 7214.32와 정합, 오프셋 +0.10).
  마감 07-15 (최종 = private LB). 궤적은 reports/submission_log.md.
- ★ 필독: reports/REBUILD_PLAYBOOK.md (메커니즘 15 + S11 제약/종결 기록) →
  reports/blocker_census.md (400 tasklog 수요 랭킹) → 메모리 signed-einsum-routing,
  gate-policy-lb-is-arbiter, adoption-protocol.

🚨 모델 배분 (항상): Fable = 오케스트레이션·새 메커니즘 설계·채택 판단·제출만.
  나머지 전부 Agent 호출 시 model:"opus".
🚨 게이트: bundled fail=0 절대 (public LB = bundled), fresh ~98%+ → 채택 후 제출 검증.
  에이전트가 보고한 baseline은 믿지 말고 인컴번트를 직접 재측정 (S11 task117 교훈:
  에이전트 "+0.0034 승" 보고가 실측 −0.0062 손실이었음 — 정합검증이 잡아 롤백).

S11 핵심 결론 — 기존-넷 최적화 공간은 측정으로 소진됨:
- 레버 5개 전수 종결 (재탐사 금지): mech-15 전파(히트 092/234/335 후 정찰 11건 전부 kill,
  파인더 mech15_output_scan.py는 신규 넷 스크린으로 유지), dtype recast(64태스크 소진 —
  Cast 경계비용이 근본 한계), Equal-then-Pad 크로스오버(174 착지 후 종결), bool-Pad
  (32히트 0채택 — opset≤10은 attr→init 변환이 압도), int8 producer 교체(0/32 구조 반증).
- 블로커 센서스 (reports/blocker_census.md): #1 출력캐리어(111태스크/173pt) + #2 검출읽기
  (64/136) = 증명된 벽. 실행가능 수요 = 뱅크-구체화 뿌리 (#3 스탬프 51/65 + #4 flood
  28/45 + #8 구간fill 10/19). If/Loop/서브그래프 = grader 금지 op (확인) → 분기-선택은
  runtime-parameterization으로만. UNKNOWN 버킷 58태스크 = tasklog 부재/빈약.
- ⭐ 논리적 확정: 상위권 7982 > 우리 측정 천장 추정(상한 ~7918) → 우리 "플로어" 판정
  어딘가가 틀림. 구체적 단서 = task193: 손-논증으로 플로어 확정했던 걸 학습된 단일 Conv가
  뒤집음 (+2.556, fresh 17.5k 무실패). 가설: 상위권은 손-컴파일이 아니라 SGD로 컴파일한다.

이번 세션 계획 (S11 말미 사용자 합의):
1. ⭐⭐ TRAIN-TO-GOLF 팩토리 파일럿 (최대 EV, 신규 프로그램):
   - 채점 맹점: 단일-노드 넷 = mem 0 (중간 텐서 없음). Conv(input,W[10,10,k,k])→output
     단일 노드면 비용 = params만 (k=3→900→18.2pt, k=5→2500→17.2pt).
   - 파이프라인: 태스크별 tiny 아키텍처 후보(단일 Conv k∈{3,5,7}, bias 유무, 필요시
     2-노드 변형)를 arc-gen 생성 데이터로 학습 (.venv에 torch 있는지 먼저 확인, 없으면
     설치 or numpy SGD) → 게이트: bundled 100% + fresh ≥98% + 비용 < 인컴번트.
   - 학습 목표는 (out>0.0) 부호 정확도 + margin 손실 (fp32 exactness 불필요; margin을
     크게 학습하면 220/230/294류 0.0-문턱 칼끝 문제도 원천 차단 — knife-edge 메모리).
   - 파일럿 대상 = 인컴번트 비용>1000B + 룰이 국소적(스텐실/이웃)으로 보이는 20~30개
     (cristianoc oracle로 룰 판독; blocker_census.json의 검출-플로어/스탬프 버킷 우선;
     193/220/230/294가 이 클래스의 기존 성공 사례). 히트율 ≥5%면 400 전면 확장.
2. ⭐ teacher 일일 재스캔 (검증된 최대 단일 소스, S10 하루 +5.20): kaggle datasets list
   -s neurogolf --sort-by updated → 신규/갱신분만 public_teacher_scan.py → 완화 게이트.
   마감 접근할수록 상위 덤프 공개 확률 상승 — 매 세션 무조건 1순위 루틴.
3. runtime-parameterized stamp 설계 (Fable 몫, 센서스 실행가능 1순위): 선택된 dilation/
   배율을 런타임-조립 희소 커널에 임베드해 Conv 1회로. 파일럿 = task370 (4-dilation 뱅크
   1600B). 대안 프리미티브: Resize 런타임 scales(133), GridSample 런타임 grid(361 선례).
   반증 나면 싸게 종결하고 1번에 집중.
4. (필러) UNKNOWN 58태스크 도시에 배치 (blocker_census.json에서 목록, opus 배치).

죽은 길 (S11 추가분, 재탐사 금지):
- S11 정찰 kill 13건: 233/285/370/133/054/366 (mech-15 코호트) + 101/216/368/367/066/
  064/182 (파인더 적격) + 361/387 (포인터). 각 tasklog에 사유 기록.
- task294 이니셜라이저 강화 PARKED (ORT 1.26 크로스-세션 가중치 앨리어싱, task120 충돌).
  단, 1번 팩토리가 294를 재학습하면 구조가 바뀌어 앨리어싱+칼끝 동시 해결 가능성 있음.
- 구조 동일 넷 쌍은 로컬 배치 평가 오염 (120/294) → 의심건 격리 재평가.

하지 말 것 (유지):
- 검증 없이 개선 주장 금지. 채택 = 프로토콜 (백업→교체→소스재생성→정합검증(직접 재측정)→
  tasklog+⭐TRANSFERABLE). 후보는 reports/candidates/, src/custom 덮어쓰기는 채택 시에만.
- 제출: src.pipeline --report-only --pack → scan_unsigned_topk.py 전수 →
  kaggle competitions submit -c neurogolf-2026 -f submission/submission.zip -m "<msg>".
- worktree 서브에이전트 금지. fresh_verify CAND는 진짜 인컴번트 대비인지 확인 (self-trap).
```
