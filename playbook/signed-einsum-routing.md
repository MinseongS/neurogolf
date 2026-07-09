# signed-einsum-routing — separable fill/overlap priority를 signed-weight Einsum으로

> grader가 채널당 `(out > 0.0)`을 디코드(src/harness.py:218) ⇒ separable axis-aligned fill의
> overlap/paint-order priority는 **LINEAR**. `[30,30]` label/priority carrier 평면이 필요 없다.
> 인사이트: `signed_rect_priority_overlay`(92/234/335). 레거시 playbook mechanism 15.

## 언제 쓰나(스캐너/시그널)
- output = separable rect/segment fill의 union이고 per-class overlap 순서가 FIXED.
- 시그널: `separable_rect` / `priority_overlay` / `label_map` 태그, memory ≥ 800; incumbent가 scalar-label/priority/canvas 평면을 지불.
- 스크리닝은 **output을 렌더링**해서 separability + 상수 colour roster를 보고 결정(class label로 판단 금지).

## 물리(왜 되나)
- 각 fill class q가 `fill_q(r,c)·W[q,v]`를 기여, **SIGNED per-class 채널 벡터**로:
  loser class는 overlap에서 negative로 억제(horizontal q → e_q−e_0), winner vertical q → (M+1)e_q−M·𝟙 (M ≥ max fill 다중도), bg = 여분 slot e_0.
- separable axis-aligned fill이 하나의 free 최종 Einsum `sr,sc,sv->vrc`에 올라탐. `>0` 디코드 덕에 overlap priority가 선형으로 표현됨.

## 레시피(단계)
1. output 렌더 → separable rect/segment로 분해 가능한지, colour roster가 상수인지 확인.
2. per-class signed 채널 벡터 설계(loser 억제 부호, winner (M+1)e_q−M·𝟙, bg e_0). M = 측정된 max fill 다중도.
3. 각 fill을 `sr,sc`(row/col profile) + `sv`(signed 채널)로 인코딩 → 하나의 free-output Einsum `sr,sc,sv->vrc`.
4. `reports/candidates/`에 빌드 → `uv run ng gate cand.onnx --task NNN`(bundled fail=0, cost < 배포본).
5. **모든 crossing/overlap-heavy grid를 손으로 만들어** 부호가 옳은지 검증(억제 부호 하나 틀리면 조용히 오답).
6. `uv run ng adopt … --note "signed-einsum: separable fill priority"` + tasklog.

## 서브패턴
- **uint8 presence for ArgMax (task234):** ArgMax만 먹이는 fp32 `Sign(profile)` 벡터 → `Greater(profile,0)→Cast(uint8)`. bool ArgMax는 ORT 거부, uint8은 accept. 120B fp32 → 30B bool + 30B uint8. bg 채널 signed 계수(+1 whole-grid, −1 per bg rect)로 명시적 bg-mask 평면 삭제(+0.28).
- **checkerboard/parity paint (task286):** rank-1 stacked-parity einsum `'nqrc,sr,sc,q->nq'`.
- **rank-decomposed separable (task335):** `'ntc,th,tw->nchw'`가 free output 직접 write(floor 2215).
- **threshold_linearize pairwise one-hot AND (task001):** Kronecker self-product가 곱셈처럼 보여도 verifier가 `>0` 임계 → one-hot pairwise equality/AND를 signed score로 인코딩, counted 10ch carrier 삭제(17.98→19.52, mem 0 params 240).
- **zero_compare_to_bool_cast(safe golf):** 비음수 `x>0` mask → `Cast(x→bool)`로 대체(bit-identical peephole, presence 평면 dtype 절감).

## 함정/거부 사례
- **LABEL/priority carrier cost에만 적용, detection/assignment cost에는 아님(task233 kill):** 비용이 detection read / assignment / sprite-stamp에 있으면 signed-W가 못 건드림. S11 cohort 233/285/370/133/054/366 ALL KILL. 히트는 092/234/335뿐.
- **S11 composition constraint (task084):** free output을 쓰는 op은 **단 하나**. free-einsum + residual-scatter hybrid는 합성 불가(counted [1,10,30,30] bridge = 18–36KB). fold는 ALL-OR-NOTHING. output 성분 중 하나라도 non-separable & 데이터 의존이면(e.g. A-의존 anti-diagonal) 전체가 counted [K,30,30] fp32 operand를 지불 → ScatterElements-into-FREE-input이 이김.
- **ScatterElements dtype-bound:** update가 data에 dtype-bound(fp32 input ⇒ fp32 update, recast 불가).
- **mechanism 15 = SCREEN, not mine:** `output>0` predicate만으로는 부족 — "counted mass가 삭제가능한 carrier에 있는가"도 gate해야 함(대부분 detection/selection/correlation 머신, carrier는 S8-S11에 이미 접힘). 신규 넷(teacher import) 스크리닝 용도로만.
