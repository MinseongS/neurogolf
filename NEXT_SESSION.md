# NEXT SESSION — NeuroGolf handoff (2026-07-03 S11 종료 시점)

Use this as the next-session prompt:

```text
/Users/minseong/project/neurogolf 에서 NeuroGolf 점수 개선을 이어가자.

현재 상태 (S11 종료, 2026-07-03):
- 확정 LB 최고점 = **7214.42** (54295163, S11 최종. 로컬 격리 7214.32와 정합, 오프셋 +0.10).
  마감 07-15 (최종 = private LB). 궤적은 reports/submission_log.md.
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
   부산물 bool-Pad(opset13) 레버도 전수 반증 종결 (32히트 0채택; opset≤10 넷은
   attr→init 변환 +params가 절감 압도 — task056 측정 −0.28. reports/boolpad_scan.md).
   메커니즘 15도 전파 종결: 파인더(mech15_output_scan.py) 67 적격 → 상위 전부 kill,
   출력 술어만으론 부족 (비용이 캐리어에 있어야) — 신규 넷 스크린으로 전환.
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
3. ⭐ 블로커 센서스 완료 (reports/blocker_census.md — 400 tasklog의 수요 랭킹):
   - 실행 가능 최대 수요 = "후보/템플릿 뱅크를 전부 구체화 후 선택" 뿌리 (#3 스프라이트-
     스탬프 51태스크/65pt + #4 flood 반복 28/45 + #8 구간fill 10/19).
   - ⚠️ If/Loop/Scan/서브그래프는 grader 금지 op (확인함) — "선택 분기만 계산"은 불가.
     설계 방향 = 허용 op로 runtime-parameterization: Conv weight는 런타임 텐서 가능
     (182가 선례), Resize 런타임 scales, GridSample 런타임 grid (361 선례). 뱅크 대신
     "선택값을 커널/그리드에 임베드해서 1회 실행" — 다음 세션 Fable 설계 1순위.
   - #1+#2 (캐리어 900B + 검출 G²×4, 합 ~300pt) = 증명된 벽. UNKNOWN 버킷 58태스크
     (tasklog 부재/빈약)도 발견 — 도시에 루프의 후보군.
4. task361/387/101/216/368/367/066/064/182 = S11 정찰 kill (각 tasklog 기록) — 재탐사 금지.
5. task008 사용자 수작업 채택분 (+0.136, fresh 2000/2000 확인됨) — tasklog 메커니즘
   기록이 비어 있으면 보완.

하지 말 것 (유지):
- 검증 없이 개선 주장 금지. 채택 = 프로토콜 (백업→교체→소스재생성→정합→tasklog+⭐TRANSFERABLE).
- 제출 zip = submission.zip (src.pipeline --pack), 제출 전 scan_unsigned_topk.py.
- src/custom 직접 덮어쓰기는 게이트 통과한 채택 시에만 (후보는 reports/candidates/).
- worktree 서브에이전트 금지. fresh_verify의 CAND는 진짜 인컴번트 대비인지 확인 (self-trap).
```
