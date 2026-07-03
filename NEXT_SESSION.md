# NEXT SESSION — NeuroGolf handoff (2026-07-03 S10 종료 시점)

Use this as the next-session prompt:

```text
/Users/minseong/project/neurogolf 에서 NeuroGolf 점수 개선을 이어가자.

현재 상태 (S10 종료, 2026-07-03):
- 확정 LB 최고점 = **7213.63** (54288054, S10 최종. local isolated 7213.52와 정합).
  마감 07-15 (최종 = private LB). 궤적은 reports/submission_log.md.
- ★ 필독: reports/REBUILD_PLAYBOOK.md → 메모리 neurogolf-walk-einsum-mechanism,
  neurogolf-gate-policy-lb-is-arbiter (S10 게이트 완화), neurogolf-adoption-protocol.
- ⚠️ 사용자가 이번 세션에 "다른 방향성"을 제시할 예정 — 아래 큐는 그게 없을 때의 기본값.

🚨 모델 배분 규칙 (사용자 지시, 항상 적용):
- Fable(메인 루프) = 오케스트레이션·새 메커니즘 구상·채택 판단·제출만.
- 그 외 전부 = Agent 호출 시 model:"opus" 명시.

🚨 게이트 정책 (S10에서 사용자가 완화 — 메모리 gate-policy-lb-is-arbiter):
- bundled fail=0 = 절대 게이트 (public LB = bundled 채점).
- fresh ~98%+ 통과 → 채택하고 실제 제출로 검증. 100% fresh 요구는 과잉 (S9 기각분
  017/025/090/191 재심 채택으로 +0.94 회수됨). fresh-fail율은 tasklog에 기록 (private 리스크 추적).

S10에서 증명된 것:
1. ⭐ 게이트 완화 + 일일 teacher 스윕 = 28건 채택 +5.20 (kojimar 7185.95 overrides +
   신규 저자 bobmyersthesecond 7186 전체덤프). 매일 재스캔 유지 — 상대들 계속 갱신 중.
2. ⭐ task193: 학습된 단일 Conv가 S9 "본질적 플로어" 판정을 뒤집음 (17.5k fresh 무실패).
   손 분리가능성 논증으로 스텐실 룰을 플로어 확정하지 말 것.
3. ⭐ fp32 Conv→int8 QLinearConv (출력이 TopK/ArgMax 랭킹만 먹일 때) = 무손실 (264/184/365/191).
   전수 스캔 후보 레버 — "Conv 출력 → 랭킹-전용 경로" 필터로 400 스캔하면 잔존 win 가능.
4. 💀 crop-bounds 전수: 163 플래그 중 상위 11건 전부 FLOOR — 플래그 평면이 free-출력 축에
   용접됨 (메모리 crop-scan-false-positive-class). 나머지 152건도 동류 추정, 재탐사 금지.
5. 💀 15k+ 대형 넷: 273(이미 2109로 해결됨 — tasklog 정정 필요)/076(대응 WALL)/054(S8
   이중골프+3600B 라벨플로어) 전부 FLOOR. 풀 전면 팬아웃 하지 말 것. 다음 정찰은
   "단일 구체화-후-축약 마스크" 클래스만 (Einsum-vs-FREE-input 적용처).
6. 🚨 칼끝-넷 발견: 220/230/294 (mem=0 단일 Conv)는 출력이 0.0 문턱 — 같은 프로세스 내
   이전 평가가 있으면 전멸로 플립 (ORT arena float 순서). 배치 로컬 합산이 ~54.6pt
   과소평가됨 → 격리 재평가로 보정할 것. 메모리 knife-edge-conv-flip.

이번 세션 작업 큐 (사용자 새 방향성이 우선):
1. 칼끝-넷 강화 마무리: 후보 3개가 reports/harden_candidates/에 있음 (task220/230/294,
   epsilon-bias, mem/params 불변). 게이트 미완: fresh-프로세스 + dirty-프로세스(오염 평가 후)
   bundled fail=0 + fresh 2000 → 채택 (dirty_gate.py 동봉). private LB ~54pt 보험.
2. kojimar/bobmyers 일일 재스캔 (S10과 동일 절차: kaggle datasets list -s neurogolf
   --sort-by updated → public_teacher_scan.py → 완화 게이트로 채택).
3. Conv→랭킹-전용 int8 전수 스캔 (위 3번 레버).
4. walk_einsum 롱테일: 319 (Gather×16 [10,10] 1600B), 023, 370, 009, 279, 196
   (reports/walk_einsum_scan.json S10 재생성본; 204는 reject-check 확정이라 제외).

죽은 길 추가분 (S10, 재탐사 금지):
- crop-bounds 상위 11건 FLOOR: 329/150/155/289/341/239/043/345/340/335/075 (각 tasklog).
- 273/076/054 FLOOR (정찰 보고 반영; 273 tasklog는 낡은 4000-mem 서술 — 정정 필요).
- task157 teacher: UNGATEABLE + fragile, Δ-5B — 스킵 확정.
- task191 kojimar 변형(0.70%)은 bobmyers판(0.95%, -1151)에 밀림 — 재비교 불필요.

하지 말 것 (S9에서 유지):
- 검증 없이 개선 주장 금지. 채택 = 6단계 프로토콜 (백업→교체→소스재생성→정합→
  tasklog 메커니즘 기록→⭐TRANSFERABLE 라인). 메모리 adoption-protocol.
- 제출 zip = submission.zip (src.pipeline pack), 제출 전 scan_unsigned_topk.py 400 전수.
- src/custom 인컴번트 직접 덮어쓰기 금지 (live_to_exact_source --write-src 경유).
- worktree 서브에이전트 금지. 로컬 400 배치 합산은 220/230/294 격리 보정 후 해석.
```
