# kernel-collapse — Conv 커널을 최소형으로 붕괴 (params 레인, bit-identical)

> **PURE params 승 +0.553 LB/배치(18승).** single-position Conv 커널(6×6, one tap)을 1×1+pad로,
> bit-identical. memory-sweep과 **직교하는 params 레인**. 레버: `kernel-collapse`(scanner `kernel_collapse`, agent=opus).

## 언제 쓰나(스캐너/시그널)
- `uv run ng scan kernel_collapse` → `candidates/worklists/kernel_collapse.json`: 배포 Conv 넷 전수 스윕.
- 시그널: K×K Conv 커널의 실효 tap이 단일 위치(나머지 0) → 1×1 + pad로 동일 출력.
- **새 Conv 넷 유입 시(min-merge/rebuild 후) 재실행.**

## 물리(왜 되나)
- params = initializer ELEMENT count(dtype 무관). 6×6×C 커널에서 tap 하나만 nonzero면 나머지 35×C element가 순수 낭비.
- tap을 1×1 커널 + 비대칭 `pads`로 재배치하면 출력 bit-identical, params가 |kernel|만큼 감소.

## 레시피(단계)
1. `ng scan kernel_collapse`로 후보 + 각 넷의 실효 tap 위치 확보.
2. 1×1 커널 + tap 방향을 `pads`로 인코딩(방향을 pad에, input flip 금지 — flip 평면은 3200B).
3. `uv run ng gate cand.onnx --task NNN` — **bit-identical**이 게이트(bundled fail=0 + cost 하락). fresh는 진단.
4. `uv run ng adopt … --note "kernel-collapse: NxN single-tap → 1x1+pad"` + tasklog.

## 서브패턴
- **sparse_conv_single_op_floor (task15/95/230/294):** scorer가 `>0` 임계 → local stencil 규칙을 zero-memory Conv 하나로. (⚡ knife-edge: 단일-Conv 넷은 ORT weight-aliasing으로 flip — batch 400-net eval이 ~54pt 저계상. 항상 **isolated 프로세스**로 채점.)
- **parallel-plane conv-channel union (task349):** K개 평행 dilation/detector 평면 → K 출력채널 Conv 하나 + count union(Min/(>0)).
- **dihedral template match stacked Conv (task191):** 8-orientation template match를 fixed gather map으로 permute, present/no-extra predicate를 하나의 signed correlation 커널로 fold → Conv/MaxPool stamp.
- **runtime_conv_template_anchor:** runtime-추출 sprite/template mask를 **non-initializer Conv weight**로 컴파일(ORT 1.26 OK — Conv weight가 계산 텐서여도 됨).
- **runtime-parameterized stamp kernel (task370, +0.121 bit-identical):** 후보 커널 BANK + mux를 뒤집어 d(dilation/spacing)를 먼저 DETECT(clamped GatherND probe → ReduceMax) → runtime에 커널 하나 ASSEMBLE(`ScatterND(zeros, base_idx·d, ones)` → 1 QLinearConv). dilation은 static attr → CENTERED 커널(half-size C = max stamp offset), OOB tap은 update 0 trash cell로 clamp. **⚠️ mechanism 16 완전 CLOSED**: global scalar(mined out) AND per-object(refuted, task133 bmags=[1,3,4,2] = VECTOR → 단일 커널 부재).
- **spatial ReduceSum → Einsum (params −2, bit-identical):** `ReduceSum(input,axes=[2,3],keepdims=0)` → `Einsum(input,'bchw->bc')` → axes init dead(dormant 레버 `reducesum-spatial-einsum`, 재작성 쉬움; 실적 8승).
- **pad_compensated_spatial_crop:** dead Conv/QLinearConv border를 줄이고 유일 spatial consumer의 padding으로 보상.

## 함정/거부 사례
- **PRODUCER_BOUND QLinearConv dry well(074/080/383, 0/32):** QLinearConv는 QUANTIZED input 필요 → free 10ch fp32 input이 counted 9000B uint8 copy를 강제(≫ 단일채널 output saving ≤2700B). ≤2ch integer 평면을 먹일 때만 작동(264/184/365/191).
- **sparse-initializer DEAD:** ONNX가 operand를 sparse_tensor(T)로 취급, ORT/scorer가 Gather idx·Conv weight를 densify 없이 거부; scorer가 sparse_initializer 별도 체크. dense loophole 아님.
- **parallel fan-out ≠ collapsible chain(task204):** per-size anchor MaxPool/conv는 collapsible chain이 아님. sub-400B uint8 conv bank는 einsum-proof(fp32 entry ticket 1600–3600B + step params 손해).
- **radius-gated growth(task349):** window↔dilation coupling은 phase-gated shift 텐서 ≥7–9k params 필요.
