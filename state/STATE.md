# STATE — NeuroGolf live handoff (갱신: 2026-07-09)
> 이 파일은 append 금지. 세션 종료 시 "현재 참인 것"만 남기고 교체한다. 히스토리는 git + state/submissions.md.

## 확정 상태
- BEST LB: 7279.41 (sub 54467261, LB 확인)
- 로컬 기준값: state/baseline/points.json 참조 (7279.2908 / cost 1042852, LB 오프셋 +0.12)
- 마감: 2026-07-15 (private LB = 동일 고정 데이터셋; bundled fail=0 = 영구 통과)

## 활성 베인 (지금 할 일, 기대이득 순)
1. free-output-einsum regime crack — mask_dominance 잔여 ~45태스크 (16/18 crack). positioned-content
   mask는 floor(pre-filter로 skip), global-state/structured(ring/block/periodic/diagonal/threshold-run/count)만 채굴. +20~40 LB 잠재.
   → 상세(DONE·FLOOR 재공격금지·crack 조건·NEXT 타깃): state/levers.yaml free-output-einsum-regime-crack 항목
2. CONV-FP32 arsenal (074/080/198/383/187) + QLinearConv(349) — regime crack의 non-mask 일반화, 미증명.
3. mixed-dtype carrier — 배포 넷 fp16 output-coupled/coordinate-tail 재캐스트(input-weld/consumer co-bind는 floor).
4. 공개 min-merge 모니터링 — 새 업로더가 7250+ 갱신 시 margin-0 재채굴(dormant, 상시 pass 가치).
5. (levers.yaml의 live 레버 순회)

## 불변식 (재검증 금지)
- submission.zip 이름 고정 / 100회/일 / unsigned TopK 금지 / isolated eval
- 비교 기준 = submission/overfit_nets/ (배포본)
- 로컬 == LB (+0.12 오프셋)
- 로컬 스코어러 환경 PIN: pyproject.toml의 onnx==1.21.0 + onnxruntime==1.26.0 고정.
  (onnx 1.22 strict shape-infer가 negative-pad Conv 넷 거부; ort >=1.27이 task347 MaxUnpool 거부;
  ort <=1.23은 fp16 Max/ConvInteger CPU 커널 부재.) 400/400 재검증 없이 절대 업그레이드 금지.

## 다음 세션 시작 절차
1. `uv run ng status` 2. 이 파일 3. `state/levers.yaml`에서 live 레버 선택 4. skills/neurogolf/SKILL.md 루프 실행
