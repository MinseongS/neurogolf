# NEXT SESSION — NeuroGolf handoff (2026-07-03 S11 종료 시점)

Use this as the next-session prompt:

```text
/Users/minseong/project/neurogolf 에서 NeuroGolf 점수 개선을 이어가자.

현재 상태 (S11 종료, 2026-07-03):
- 로컬 격리 합계 **7214.32** (S11 제출 완료, LB 확인 대기 — reports/submission_log.md 참조.
  S10 오프셋 기준 기대 LB ~7214.4). 마감 07-15 (최종 = private LB).
- ★ 필독: reports/REBUILD_PLAYBOOK.md (메커니즘 15 + S11 제약들 추가됨) → 메모리
  neurogolf-signed-einsum-routing, neurogolf-gate-policy-lb-is-arbiter, adoption-protocol.
- 세션 운영 모드 = FAMILY-FIRST (사용자 지시): 태스크 단위 골프가 아니라
  "이 태스크가 어느 20+ family로 재컴파일되나"를 묻고, 성공 시 코호트로 전파.
  루프: 도시에(opus) → 인간 리뷰/메커니즘 감사(Fable) → 파일럿 빌드(opus) → 코호트 스윕.

🚨 모델 배분 (항상): Fable = 오케스트레이션·새 메커니즘 설계·채택 판단·제출만.
  나머지 전부 Agent 호출 시 model:"opus".
🚨 게이트: bundled fail=0 절대 (public LB = bundled), fresh ~98%+ → 채택 후 제출 검증.

S11에서 증명/확정된 것:
1. ⭐ 메커니즘 15 "signed-channel priority overlay": (out>0.0) 채널별 채점 ⇒ 겹침
   우선순위가 signed W로 선형 → [30,30] 라벨/우선순위 캐리어 삭제. 092 +0.241 착지.
   적용 경계(6/6 코호트 kill로 확정): 분리가능 축정렬 fill + 상수 색 로스터 + 캐리어에
   돈을 내는 인컴번트일 때만. 클래스 라벨 말고 출력을 렌더해서 스크리닝.
2. ⭐ 컴포지션 제약 (084 가격측정): free output에 쓰는 op는 단 하나 — 하이브리드
   (einsum+scatter) 불가, 폴드는 all-or-nothing. 비분리 데이터-의존 성분이 하나라도
   있으면 ScatterElements-into-FREE-input이 이김.
3. ⭐ dtype 함정 3종 (playbook 기록): PRODUCER_BOUND (free input 직접 소비 producer의
   출력은 recast 불가), ScatterElements updates=data dtype 바인딩, recast는 "free-input
   피연산자 없는 einsum 섬" 단위로만. dtype 상한은 번들 관측이 아니라 generator 최대치.
4. ⭐ 출력 라우팅 크로스오버: Equal-then-Pad(10·h·w) vs Pad-then-Equal(900B), 콘텐츠
   <90셀이면 전자 승. 전수 스윕 → 유일 후보 174 착지 +0.189. 이 레버 종결.
5. 💀 마른 우물 확정 (재탐사 금지): int8 QLinearConv on PRODUCER_BOUND 색번호 Conv
   (0/32, 입력 양자화 9000B가 압도 — 측정 3건), dtype recast 잔존분 (2건 착지로 수확
   완료, 나머지 dtype-bound/FLOOR), per-cell 색 읽기 플로어 8번째 확인.
6. 💀 플로어 확정 (빌드로 측정): 041 (1789B), 084 (1895B), 162 (4068B — 후보가 번들
   train#2 하나로 public-fatal, 사연은 tasklog), 177 (3621B). + signed-priority 코호트
   6개 (233/285/370/133/054/366, 각 tasklog에 비용 해부 기록).
7. 🚨 294 칼끝 원인 규명: ORT 1.26 크로스-세션 가중치 앨리어싱 (task120 가중치가 294
   세션에 주입; 이니셜라이저 강화 원천 불가) → PARKED. 220/230은 S10에 이미 강화 배포
   완료. 구조 동일 넷 쌍은 로컬 배치 평가 오염 가능 → 의심건 격리 평가.

이번 세션 작업 큐 (family-first 계속):
1. kojimar/bobmyers 일일 재스캔 (S11은 신규 없음 — 매일 반복).
2. 리뷰 루프 계속: reports/frontier_queue.md (S10-보정 완료)에서 다음 배치.
   ⚠️ 큐 휴리스틱은 출력-캐리어 플로어에 눈멀어 headroom 과대평가 (041 사례) —
   재생성 시 캐리어-플로어 추정 컬럼 추가 권장 (build_frontier_queue.py).
3. 다음 메커니즘 사냥 표적 (S11 kill들의 공통 요구, 아직 존재하지 않는 프리미티브):
   a. "runtime-parameterized stamp" — 임의 스프라이트를 데이터-의존 위치/배율로 싸게
      찍기 (370의 4-dilation 뱅크, 133의 4-배율 커널, 285/366이 공통 요구).
   b. assignment/해시매치 압축 — 233의 324-포지션 매치, 366의 복제 추출 블록 21.5KB.
   c. 177 사촌 task361 (4322B, crop+reflect) 정찰.
4. task008 사용자 수작업 채택분 (+0.136, fresh 2000/2000 확인됨) — tasklog 메커니즘
   기록이 비어 있으면 보완.

하지 말 것 (유지):
- 검증 없이 개선 주장 금지. 채택 = 프로토콜 (백업→교체→소스재생성→정합→tasklog+⭐TRANSFERABLE).
- 제출 zip = submission.zip (src.pipeline --pack), 제출 전 scan_unsigned_topk.py.
- src/custom 직접 덮어쓰기는 게이트 통과한 채택 시에만 (후보는 reports/candidates/).
- worktree 서브에이전트 금지. fresh_verify의 CAND는 진짜 인컴번트 대비인지 확인 (self-trap).
```
