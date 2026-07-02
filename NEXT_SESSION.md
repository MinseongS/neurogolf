# NEXT SESSION — NeuroGolf handoff (2026-07-03 S9 종료 시점)

Use this as the next-session prompt:

```text
/Users/minseong/project/neurogolf 에서 NeuroGolf 점수 개선을 이어가자.

현재 상태 (S9 종료, 2026-07-03):
- 확정 LB 최고점 = **7208.43** (54270903, S9 최종. local 7208.33과 정합).
  마감 07-15 (최종 = private LB). 과거 궤적은 reports/submission_log.md 만.
- ★ 필독 순서: reports/REBUILD_PLAYBOOK.md (S9 갱신: 메커니즘 14 separable-remap einsum,
  crop 경계 reject-check, LSTM DEAD 판정) → 메모리 neurogolf-walk-einsum-mechanism.

🚨 모델 배분 규칙 (사용자 지시, 항상 적용):
- Fable(메인 루프) = 오케스트레이션·새 메커니즘 구상·채택 판단·제출만.
- 그 외 전부 = Agent 호출 시 model:"opus" 명시. 애매하면 opus로 시작 → 막히면 격상.
- 예외(fable 허용): 완전히 새로운 einsum 구성 증명, 플로어-반박 연구.

S9에서 증명된 것 (전부 LB/게이트 확정):
1. ⭐ kojimar 데이터셋의 진짜 teacher는 overrides/ 디렉토리 (base_submission은 종종 우리
   넷과 동일). 7184.85 스윕: 7건 채택 +3.9 / 5건은 비캐시 fresh-게이트가 기각 (193 2.36%,
   017 NS=9 1.27%, 191 int8 0.92%, 025 K-언더카운트, 090). fresh-게이트 = 해자 실증.
2. ⭐ 단일탭 valid-Conv crop: counted 진입 읽기에만 적용 (396/193/222/173/138 +0.5).
   free-input walk einsum엔 역효과 (task077 반박) — 단 einsum이 최종 op면 in-einsum
   index re-embed로 우회 가능 (task187 +0.153). 검증된 상한표 = reports/grid_crop_bounds.md.
3. ⭐ mechanism 14 (separable-remap einsum, task108 +1.18): 고정 분리가능 remap = 단일
   einsum mem=0. 전수 스캔 완료(reports/separable_remap_scan.md) — 적용처 16건 중 5건
   채택(+2.9), 나머지는 30-축 세금으로 기각. 스캔 투영은 ~8× 낙관이었음 (출력축은 항상
   30-폭; U 테이블 = [30,K]).
4. GridSample fp16 grid = fp32 OneHot 셀렉터 쌍의 절반 비용 (task029). moment 항등식
   판별 (hollow-rect). Range→Gather는 현 ORT에서 정상 (150/155 채택, 옛 버그 기록 정정).
5. LSTM/GRU/RNN = 카운팅 루프홀 실재하나 (Y 생략 시 스텝 무료) 진입비용 때문에
   walk-einsum 대비 전면 열세. DEAD — playbook에 가격 기록됨. 재탐사 금지.

이번 세션 작업 큐 (기대값 순):
1. kojimar 일일 재스캔: kaggle datasets list -s neurogolf --sort-by updated → 새 버전
   나오면 다운로드 → public_teacher_scan.py → 신규 win만 teacher-추출 에이전트 (반드시
   overrides/ 확인 + 비캐시 fresh 게이트 ≥2000 + uint8-TopK 감사). 상대가 +4/일 속도로
   갱신 중 — 최우선.
2. 전수 crop-bounds 스캔 확장: S9는 30×30 fp32 평면 보유 16태스크만 검증. 400 전수로
   "counted 평면 크기 > generator 상한" 스캔 (25×25/20×20 등 중간 크기 평면 포함) →
   크롭 팬아웃. 스크립트 초안: 각 넷 counted 평면 최대 spatial dim vs arc-gen 정적 상한.
3. struct_scan/walk_einsum_scan 재실행 (S9 채택 21건으로 순위 변동) + <6k 롱테일.
4. 대형 잔존 넷: 233/366/133/285/118/018 전부 S8-S9 이중 플로어 확정 — 새 메커니즘 없이
   재탐사 금지. 273/76/54/191 등 15k+ 미확정 넷 위주로.

죽은 길 (재탐사 금지 — 근거는 각 tasklog S9 항목):
- S9 플로어: 158/366/133/286/338/209/233(크롭 포함)/077(크롭)/135/053(mech-14)/
  164/172/210/311(mech-14)/025(teacher)/017(teacher)/191(teacher)/193(teacher)/090.
- LSTM/GRU/RNN 구조 베팅 (playbook 가격 기록).
- REPEAT_GROUP K-배칭 단독, sub-400B u8 뱅크, 병렬 팬아웃 MaxPool (S8 확정 유지).

하지 말 것:
- 검증 없이 개선 주장 금지. 게이트 = stored fail=0 + mem+params 감소 + 비캐시 fresh
  cand ≤ inc (≥500, 채택 시 오케스트레이터가 독립 재검증).
- 제출 zip = submission.zip (src.pipeline pack), 제출 전 scan_unsigned_topk.py 400 전수.
- src/custom 인컴번트 덮어쓰기 금지 (후보는 scratchpad → ONNX 실체화 →
  live_to_exact_source --write-src 채택; 교체 전 reports/retired_networks/ 백업).
- worktree 서브에이전트 금지 (S9 종료 시점 커밋 여부 확인: git log).
```
