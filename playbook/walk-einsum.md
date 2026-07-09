# walk-einsum — 반복 flood/scan을 ONE multi-operand Einsum으로 붕괴

> 채점기는 **NODE OUTPUT만** 센다(정적 shape 바이트, ORT 프로파일러 trace로 max). op **내부는 무료**;
> `input`/`output`[1,10,30,30]은 dtype 무관 무료; `run_network`가 output에 무료 `>0` 임계 적용.
> ⇒ 반복 전파를 단일 Einsum 내부로 접으면 모든 중간 평면(MaxPool/Min/CumSum 스택)이 사라진다.
> 레버: `walk-chain-slack`(잔여 slack 트림) / 인사이트 `walk_einsum_iteration_collapse`.

## 언제 쓰나(스캐너/시그널)
- 반복 MaxPool/Max/Min/CumSum/And/Or 스택 = flood-fill·연결성·전파(reachability, run-length, area, distance).
- 시그널: `maxpool_scan` / `connectivity_wall` / `scan` 태그, memory ≥ 8000, K회 반복 평면(각 30×30).
- 도달성 = (#walks from seed > 0)가 **입력의 다항식**이면 무조건 대상. 13~23k iteration 넷에서 넷당 +0.7~0.9.
- 대표 적용: task187(+0.71), task110(+0.86), task243(+0.92), task077(+0.85), task313(kojimar 선례).

## 물리(왜 되나)
- reachability = seed 집합에서의 walk 수 > 0. K 전파 스텝을 하나의 Einsum으로:
  - seed rank-terms `G[m,r0]·H[m,c0]` (**비음수**)
  - per step `S[r_j,r_{j+1}]·S[c_j,c_{j+1}]·t[r_{j+1},c_{j+1}]` (S = tridiag+self, 성분 0.5; t = traversable 평면)
- 모든 operand 비음수 ⇒ 상쇄 없음 ⇒ fp32 반올림/오버플로가 `>0` 의미를 뒤집지 못함. dyadic 스케일(0.5)로
  최소 경로 가중치를 정상-float 범위(>~1e-38)에 유지(subnormal-flush-safe).
- 글자 예산 52: 8-conn = 2글자/스텝 → 23스텝/einsum; 4-conn 교대 H/V = 1글자/스텝 → 45스텝.
- ORT는 operand를 **LEFT-TO-RIGHT pairwise 수축** → walk 체인 순서대로 operand를 배치.

## 레시피(단계)
1. 생성기(`state/arc_mapping.json` → `arc-gen/tasks/…`) + 현행 `src/custom/taskNNN.py` + tasklog 읽고 counted 평면 바이트-랭크.
2. 연결성 모델 확정: **BFS4 vs BFS8 vs ground-truth를 ≥20k fresh에서 측정**(3×3 MaxPool flood = 8-connected).
3. seed·step·traversable operand 설계. 최대 BFS 거리 + 마진으로 스텝 수 결정. 거리가 크면 einsum 체인(경계에서 slot 비용 2D−1).
4. 후보를 `candidates/`(또는 워크트리)에 `build(task)`로 빌드 — **절대 `src/custom`·배포 onnx 덮어쓰지 말 것**.
5. 게이트: `uv run ng gate cand.onnx --task NNN` (bundled fail=0 + 비용 < 배포본). fresh diagnostic로 candidate ≤ incumbent 확인.
6. 채택: `uv run ng adopt cand.onnx --task NNN --note "walk-einsum: …"` → 소스 재생성·정합·tasklog 메커니즘 기록.
7. 첫 사용 시 grader 안전 위해 A/B 서브밋(74-operand Einsum도 grader-safe로 확인됨).

## 서브패턴
- **Free traversability (task243):** counted t 평면 없이 `input`을 (10,) colour selector로 반복 read; 모든
  traversability slot이 단일 `P0[q]`로 pin된 하나의 colour 글자 q를 공유(유효 경로는 monochrome).
- **Checkpoint constraints (task077):** 삼각/band operand가 내부 walk 위치를 output 인덱스에 묶음
  (row(p0)≤r, row(p4)≥r …) → bbox/interval 조건이 walk einsum 내부에 존재. 하나의 T를 subscript 교체로 재사용.
- **Gates-as-operands (task110):** 데이터 의존 선택(어느 period/template) = 작은 einsum → [K] 벡터 →
  one-hot gate `g = valid·[U·valid==0]`(strictly-lower-tri U = priority)가 큰 재구성 einsum 내부에서 곱해짐;
  이질적 fallback(identity)은 operand 평면 p=0으로 **stack**(별도 counted branch 금지).
- **Exact-count monotone machine (task145):** 3-phase {Stay, Move…}로 각 셀이 정확히 하나의 walk로 도달
  ⇒ einsum 값 = run-length/area. fp16 exact ≤2048. self-loop walk는 C(N,d) 다중도 때문에 count 불가.
- **S=1.0 exactness (task002):** 도달 ⇒ W≥1 정확 → 단일 `Greater(t,W)` epilogue. ring seeding = 4 비음수 (G,H) rank-1 쌍.
- **Batched-K placement (task219):** N개 복붙 per-band/case 블록 → 선행 dim K 하나 + placement `'kjr,ks,jsc,k->rc'`.
- **Bounded crop before scan (task187/173):** 생성기 실제 max canvas(20~25box)에서 flood — 10ch→1ch label로 줄인
  뒤 crop, 모든 반복 MaxPool 평면을 30×30→25×25로 축소, 마지막에 sentinel label로 30×30 Pad-back.
- **Single-tap valid-Conv entry crop (S9):** 30×30 평면이 **COUNTED entry read**일 때만 승리 — valid-Conv 하나로
  크롭이 in-op(무료)으로 흡수된다. 실적: **task396 +0.147, task193 +0.136, task222 +0.086**. 적용 규칙: "30×30
  평면이 COUNTED entry read일 때만 이긴다" — free-input walk-einsum의 최종 op가 아니라면 반대로 작동(아래
  backfire 참조).

## 함정/거부 사례
- **8-conn trap:** 3×3 MaxPool flood는 8-connected. 4-conn exact walk가 task187 fresh 49 vs 19로 FAIL. 항상 BFS4/BFS8/GT 측정.
- **flood_fill_iteration_count_lb_tradeoff (task187):** iteration 수는 correctness↔score 트레이드지 sample slack이 아님.
  작은 fresh에서 iteration 줄이면 rare LB fail. 거리 bound를 유도하거나 더 싼 long-range primitive를 찾을 것.
- **variable_size_mask_requires_onehot_input (task173):** in-grid black과 out-of-grid zero를 compact label로 구분 불가
  (둘 다 label 0) → `ReduceMax(input)` 차원 mask는 비싸도 의미상 필수.
- **Einsum uniform-arity:** 이질적 output 합성(drawing-copy 색 + flood 색)은 free-output einsum으로 안 접힘 →
  output이 input 색을 복사해야 하면 uint8 label epilogue(~8–12KB) 잔존. (copy/edit가 `Where(mask,onehot,input)`이면 → **free-output-einsum.md**로 이관.)
- **Single-tap valid-Conv crop backfire:** free-input walk-einsum 넷에서 crop은 무료였던 in-einsum input read를 counted로 만들고
  900B output re-Pad를 강제, 공유 square T를 직사각 쌍으로 분할(task077 cand 9167 vs 6331 floor). 예외: walk einsum이 최종 op면
  P[r',R]·P[c',C] identity-embed로 30-dim 축 재삽입(task187 +0.153).
- **Latency:** 반복 operand SIZE가 지배(9000-elem input 46회 = 330ms vs 625-elem mask 1회 = 31ms). <100ms/run 유지.
- **Bundled-vs-generator (task077/023/054):** 원본-ARC bundled 예시가 생성기 보장을 위반할 수 있음 → fresh AND bundled 둘 다 검증.
- **LSTM/GRU/RNN — priced DEAD(현 pool):** grader-accepted, Y 생략 시 per-step state 무료지만 entry ticket
  = counted 3-D X-prep 평면(≥4G²) + 소비된 Y ⇒ 표현 가능한 모든 것에서 walk-einsum의 ≥2배. 순차 phase/reset 머신 전용으로 보류, canary-submit 우선.
