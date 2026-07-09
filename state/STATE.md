# STATE — NeuroGolf live handoff (갱신: 2026-07-09, kernel-collapse refresh 확정)
> 이 파일은 append 금지. 세션 종료 시 "현재 참인 것"만 남기고 교체한다. 히스토리는 git + state/submissions.md.

## 확정 상태
- BEST LB: **7296.27 확정** (sub 54484750, kernel-collapse refresh).
- 현재 로컬 manifest: **7296.1556** (400/400, `uv run ng verify` fail=0).
- 신규 제출: **sub 54484750 COMPLETE, publicScore 7296.27** — kernel-collapse refresh 14채택, 로컬 +0.193.
- 오늘 하루 확정 LB: 7279.41 → **7296.27 (+16.86)** = min-merge 78승(+10.30) + 일반화 wave 12승(+3.14) + batch7 6/6승(+3.23) + kernel-collapse refresh(+0.19).
- 마감: 2026-07-15 (private = 동일 고정 데이터셋; bundled fail=0 = 영구 통과).

## 이번 세션에서 바뀐 것
1. **kernel-collapse 재실행/채택:** public/regime overlay 이후 새 Conv 넷 대상으로 15후보 gate.
   - 채택 14개: 004/044/046/069/076/192/201/245/283/294/324/333/365/368.
   - task185는 bundled fail=267로 reject.
   - task294는 knife-edge 주의 대상이라 `uv run ng score 294` isolated 확인: fail=0, cost 612.
2. **092-style profile-product 팬아웃:** 054/076/101/118/133/206을 Fable fanout으로 재공격했으나 adoptable 0.
   - 054/076/101/118/206은 exact 후보가 incumbent보다 비쌈.
   - 133은 cheap probe가 bundled fail=267.
   - 결론: 092 메커니즘은 "two-marker/per-channel separable endpoint fill"에는 강하지만, copy/wipe/ring, sprite correspondence, cropped mask tail, creature stamp, RoiAlign+sparse ScatterND tail에는 현재 ordering이 더 싸다.
   - reopen: sub-150B sparse correction, label-pad ordering을 유지한 profile contraction, 또는 final output에 one-hot materialization 없이 직접 fuse하는 새 primitive가 나오면 재개.
3. **canvas_crop_shrink 스캐너 구축:** `src/neurogolf/scans/canvas_crop_shrink.py` 등록 완료.
   - smoke: `NG_CANVAS_CROP_SAMPLES=5 uv run ng scan canvas_crop_shrink --tasks 14 18 21 31 106 108 193 303 330 350`
   - top smoke candidates: task014(+0.156 est), task350(+0.031), task018(+0.013).
   - full proof pass는 아직 미실행: `NG_CANVAS_CROP_SAMPLES=10000 uv run ng scan canvas_crop_shrink`.
4. **공개 프론티어 감시:** 20260709 dumps margin-0 재실행 결과 추가승 0.

## 활성 베인 (기대이득 순)
1. **canvas_crop_shrink full proof → fanout**
   - 10k 샘플 전수 스캔 후 task014/350/018부터 실제 rewrite 후보 생성.
   - 목적: arc-gen 최대 output/occupied 범위를 증명하고 30x30 counted canvas를 cropped static value_info/Pad 구조로 축소.
2. **인사이트-무기고 팬아웃 계속**
   - 092-style에서 이미 reject된 054/076/101/118/133/206은 같은 방식으로 반복하지 말 것.
   - 다음 우선순위: stamp/crop 재개방 floors 062/012/112/099/163/102, probe_then_build 138/145/328/324/379, axis-code 대형 367.
   - compact-onehot, Chebyshev, maxpool-sweep, conv sampler-decoder, xor-renderer 후보는 `state/insights.yaml`의 `fanout_candidates_20260709` 기준.
3. **공개 재채굴**
   - 새 프론티어/새 업로더/새 notebook output 등장 시 `uv run ng mine-public --margin 0 ...` 즉시 재실행.
   - min-merge 채택이 나오면 반드시 public-insight-generalize deep lane으로 pairing.
4. **kernel-collapse**
   - 이번 세션 14승으로 현 overlay 기준 한 차례 수확 완료. 새 Conv 넷 유입 시 재실행.

## 하지 말 것 / 불변식
- cost-1 하위 10넷 재공격 금지(플로어 증명 완료).
- 350/148/233/042는 reopen trigger 없이 재공격 금지.
- submission.zip 이름 고정 / 100회일 제한 / unsigned TopK 금지.
- 비교 기준 = `submission/overfit_nets/` 배포본.
- 환경 PIN: pyproject.toml의 `onnx==1.21.0` + `onnxruntime==1.26.0` 고정. 400/400 재검증 없이 업그레이드 금지.
- knife-edge/aliasing 대상(특히 294)은 isolated eval로 확인.

## 다음 세션 시작 절차
1. `uv run ng status`
2. `kaggle competitions submissions -c neurogolf-2026 | head -5`로 새 병렬 제출 여부 확인
3. 이 파일 확인
4. `NG_CANVAS_CROP_SAMPLES=10000 uv run ng scan canvas_crop_shrink`
5. top 후보부터 fanout → gate → 중앙 adopt → batch submit
