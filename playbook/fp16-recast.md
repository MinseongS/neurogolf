# fp16-recast — output-coupled 평면을 fp16(및 uint8)로 재캐스트

> `calculate_memory`는 input/output을 dtype 무관 skip, `run_network`는 `>0` 임계 ⇒ sign-exact fp16이 통과.
> **output-coupled = winnable / input-coupled = floor.** 레버: `deployed-fp16-recast`(agent=opus, bit-identical golf).
> ⚠️ 스캐너는 배포 넷(`submission/overfit_nets/`)을 겨냥해야 함 — `networks/` 스캔은 배포본과 diverge(오판 이력).

## 언제 쓰나(스캐너/시그널)
- 최종 Einsum/Concat 직전에 fp32 텐서가 FREE graph output으로만 흐름 → 그 텐서 + graph `output`을 fp16으로.
- 시그널: `scan_dtype_and_shift_compression` 계열, 최종 output-coupled fp32 mask, coordinate/index tail.
- 실적(state/tasks/taskNNN.md 기록치): task377 4567→3967(+0.1408), task205 4652→4372(+0.0621)→3760(+0.1508),
  task355 538→518(+0.0379), task222 2733→2313(+0.166).
- **모든 채택/오버레이 후 재실행**; free-output rewrite 뒤 상류 tail 재검사(⚠️ levers.yaml: 스캐너 파일은 유실 — input-weld/co-bind 필터로 재작성 필요).

## 물리(왜 되나)
- **output-coupled(winnable):** 최종 Einsum/Concat이 평면을 graph OUTPUT에 pin → FREE `output` 자체를 fp16으로 만들면 그 평면들이 recastable(mem skip은 dtype 무관, `>0`은 sign-exact fp16 통과).
- **input-coupled(floor):** 평면이 free fp32 `input`에 welded(가장 가까운 Cast 상류가 input에 붙음) → recast가 18000B input-Cast를 강제. Einsum/Scatter가 input-welded operand와 함께 소비하는 평면은 input에 dtype-pinned = FLOOR.
- fp16 exact: 정수값 ≤2048은 fp16 정확(count 평면 안전).

## 레시피(단계)
1. 배포 넷 스캔 → output-coupled fp32 tail 후보. 각 후보에 input-weld / consumer co-bind 필터 적용(둘 다 floor).
2. 최종 op operand feeding 직전에 fp16 Cast 삽입(상류 공유 dtype 변경 회피 — input-co-bound cast-back trap).
3. graph `output`을 fp16으로 set(output-coupled 평면 pin 해제).
4. `uv run ng gate cand.onnx --task NNN` — bundled fail=0 + cost < 배포본. **채택 전 grader-side mem을 diff**(grader mem = ORT 프로파일러 trace, local과 다름).
5. `uv run ng adopt … --note "fp16 recast: output-coupled tail"` + tasklog.

## 서브패턴
- **final-op fp16 Cast 배치 (task205):** 상류 fp32 그대로, 최종 Unsqueeze/Concat operand 직전에만 fp16 Cast → input-co-bound cast-back trap 회피.
- **coordinate/index tail (task205 잔여):** 최종-output fp16 recast를 이미 한 태스크도 input-co-bound 아닌 coordinate/index tail(safe_name_30..33 등)이 남아있을 수 있음 → 별도 recast.
- **uint8 TopK compact label grid (task285/366/233/173/76/18):** full 900-cell fp16 cast를 지우고 기존 uint8 colour-label 벡터로 TopK feed. bool TopK는 ORT 무효 → bool→uint8 full-vector cast + tiny TopK-value fp16 recast. ORT TopK는 uint8 0/1 score 벡터 직접 소비 가능.
- **qlinear uint8 LUT/MatMul (task055/80/338/255):** scale=1·zero-point=0이면 fp16/fp32 one-hot LUT/MatMul selection을 uint8 QLinearMatMul/QLinearConv로. rank-8 bool row/col product = uint8 Cast+QLinearMatMul(sums in [0,8]).
- **scan dtype/shift compression (task46/216):** scan-style MaxPool/CumSum/Hillis-Steele를 lower dtype로, 또는 shift 행렬 공유; last legal cast까지 uint8/bool 유지.
- **dedupe byte-identical initializers:** overlay·graph surgery 후 바이트 동일 init 중복 제거(params golf).

## 함정/거부 사례
- **🚨 uint8-TopK grader-killer(CRITICAL):** unsigned dtype을 TopK에 먹이면 로컬 gate 전부 통과하지만 **Kaggle 서브밋 전체를 ERROR**. TopK feed는 float/fp16/int64만. 서브밋 전 `scan_unsigned_topk`로 400개 전수 스캔.
- **signed_int8_topk_compact_feeds:** signed INT8 TopK feed는 로컬 valid지만 **Kaggle-rejected**.
- **topk_k2_to_argmax_uint8:** ArgMax(uint8) binary-match 치환은 로컬 valid지만 **Kaggle-unsafe**.
- **S11 dtype long-tail — boundary Cast 비용:** fp16 region ENTER/EXIT Cast가 작은 산발 평면에선 halving보다 비쌈. 추정에 항상 cast overhead를 계상.
- **input-weld:** 가장 가까운 Cast 상류가 free fp32 input에 welded면 recast가 18000B input-Cast 강제 → skip.
- **PRODUCER_BOUND:** producer가 fp32 FREE input을 직접 소비하는 float 평면의 fp16/u8 recast는 무효(ORT가 output dtype을 input dtype에 bind; 뒤에 붙는 Cast는 평면을 ADD) → producer-replacement surgery 필요.
- **grader-mem ≠ local-mem:** 변경된 넷은 서브밋 전 GRADER-side로 diff. leak-audit false-positive 이력(task294) → leak swap 전 OLD 넷에 5000-fresh.
- **topk shrink는 별개 floor:** K 폭 축소는 생성기 extrema 증명 필요(`generator_extrema_before_topk_shrink` 173/285, `bounded_multicopy_slot_topk_floor` 101) — sample max로 줄이면 rare held-out fail.
