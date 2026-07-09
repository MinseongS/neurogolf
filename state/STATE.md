# STATE — NeuroGolf live handoff (갱신: 2026-07-09, 공개채굴+딥레인+팬아웃 세션 종료)
> 이 파일은 append 금지. 세션 종료 시 "현재 참인 것"만 남기고 교체한다. 히스토리는 git + state/submissions.md.

## 확정 상태
- BEST LB: **7296.08 확정** (sub 54484248, batch7 완전본; 로컬 7295.96 + 오프셋 정확 일치)
- 오늘 하루: LB 7279.41 → **7296.08 (+16.67)** = min-merge 78승(+10.30) + 일반화 wave 12승(+3.14) + batch7 6/6승(+3.23)
- 로컬 == LB (+0.12 오프셋, 3회 연속 정확 일치)
- 마감: 2026-07-15 (private = 동일 고정 데이터셋; bundled fail=0 = 영구 통과)

## 이번 세션의 구조적 변화 (다음 세션이 알아야 할 것)
1. **인사이트 무기고 완성 — 팬아웃이 계속 명중한다 (batch7 6/6, wave2 12/16).** 공개 4덤프(prvsiyan 7266.72 등)
   딥레인 autopsy로 `state/insights.yaml`에 신규 메커니즘 ~20개 등록 (각 항목에 reference src/custom 구현 +
   `fanout_candidates_20260709` 리스트 내장). 주력: free-endpoint N-ary Einsum(092 +1.28), axis-code(064 +0.46),
   dynamic_kernel_stamp_conv, dead_tap_dilated_conv_crop_decode, bitpack_code_plane, maxpool_runningmax,
   identity-scale u8 QLinearConv, rank1-qconv(294 +0.35), value_info crop, perm-decode.
2. **점수분포 벤치마크 (고득점자 자기신고, 총점 ~7821+ 추정):** 갭은 몬스터가 아니라 **중간질량** — 우리 cost
   1k–8k 태스크 ~170개를 cost 55–400 밴드로 내리는 것이 갭의 대부분 (그들 모드 = 20-21pt = cost 55-148 ×
   106태스크; 우리는 그 밴드에 30개). batch7이 정확히 이 이동. cost-1 존재증명 9개 vs 우리 3개.
3. **floor 반증 연쇄 지속:** 이번 세션만 041("entry/carrier floor")·086(batch6)·092("~14.9 실용 floor")·
   283("mem-0 불가축")·333/378(entry/carrier) 반증. floor_tasks 중 stamp/crop 서브셋(062/012/112/099/163/102)
   재공격 허용 (levers.yaml 레저 갱신됨).

## 활성 베인 (기대이득 순)
1. **인사이트-무기고 팬아웃 계속** — insights.yaml 각 항목의 fanout_candidates_20260709 중 미착수분:
   - 92-스타일 profile-product einsum cohort: 054/076/101/118/133/206/233/265/285/330/342
   - stamp/crop 재개방 floors: 062/012/112/099/163/102 · probe_then_build: 138/145/328/324/379/132/004
   - compact-onehot: 209/174/177/184/025/219/310/023 · Chebyshev: 054/076/080/023 · maxpool-sweep: 110/377/270/222/077/325
   - conv sampler-decoder: 017/265/131/336/119/109/020/189 · xor-renderer: 264/237/304/297/256/250/201
   - axis-code 고위험 대형: 367 (+0.4~0.7) / 279 · task349 (074의 ScatterND shared-index 직결, 383 잔여 +0.26)
2. **cost-1 레인 종결 (0승, 플로어 증명 완료):** 하위 10넷(087~166)은 전부 단일노드 파라미터 플로어
   (RoiAlign rois=5, Gather idx=10 등 — levers.yaml 레저 참조). 방향 전환: **그들의 25점 9개는 D4
   대칭 태스크** — 무거운 넷 중 룰이 bare symmetry(transpose/identity/축 구조)인 것을 스캔하는 게 맞는 레인.
   sparse-initializer 루프홀(nnz만 과금)은 strict shape-infer가 차단 — grader 변경 시 즉시 재개.
3. **새 스캐너 2개 구축 (kaggloop 채굴, 미구현):** ① canvas_crop_shrink — arc-gen 10⁴ 샘플로 태스크별 최대
   범위 증명 → 30 상수 전면 축소 (그들 task018 +0.5; 우리 400 전체 미채굴) ② bool_tail_u8_restructure 스윕.
4. **공개 재채굴:** "Consolidated Audit" 커뮤니티 라인 소재 파악 + kaggloop 원본 회로 10개
   (014/018/021/031/106/108/193/303/330/350) 역공학. 20260709 덤프 margin-0 추가승 0 (재실행 기록됨) —
   새 프론티어/새 추출본 뜨면 ng mine-public 재실행.
5. mixed-dtype carrier / deployed-fp16-recast / kernel-collapse (새 Conv 넷 다수 유입 — 재실행 가치).

## 불변식 (재검증 금지)
- submission.zip 이름 고정 / 100회/일 / unsigned TopK 금지 / isolated eval (294 knife-edge 포함)
- 비교 기준 = submission/overfit_nets/ (배포본) · 로컬 == LB (+0.12)
- 환경 PIN: pyproject.toml의 onnx==1.21.0 + onnxruntime==1.26.0 고정 (onnx 1.22 strict shape-infer가
  negative-pad Conv 거부; ort>=1.27 task347 MaxUnpool 거부; ort<=1.23 fp16 Max/ConvInteger 커널 부재).
  400/400 재검증 없이 업그레이드 금지.
- ⚠️ 새 주의 3건: ① kaggloop은 grader가 ort **1.24.4**라 주장 — 새 dtype/op 트릭은 1.24.4 가능성 가정하고 게이트
  ② ORT buffer-reuse: shape-changing op의 declared value_info 크기가 같으면 버퍼 충돌 — unique 크기로 선언 (088 실증)
  ③ value_info 슬랙 스캔에서 Cast/TopK는 trace 미발생 phantom (진성 슬랙 = traced>0 & declared>traced만)

## 다음 세션 시작 절차
1. `uv run ng status` 2. 이 파일 3. cost-1 레인 후보 확인(candidates/) 4. 베인 #1 팬아웃 (skills/neurogolf/SKILL.md 루프)
