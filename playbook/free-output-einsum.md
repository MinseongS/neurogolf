# free-output-einsum — 900B output-mask floor을 깨는 REGIME CRACK

> **프로젝트 최대 레버(16/18 crack, LB 7264.67→7279.34 ≈ +14.7).** `[1,1,30,30] bool = 900B`
> output-welded routing mask는 벽이 아니다. `Where(mask,·,input)` 라우팅을 FREE `output`으로 쓰는
> 하나의 N-ary Einsum으로 접으면 mask가 free op 내부로 소멸한다.
> 레버: `free-output-einsum-regime-crack`(scanner `mask_dominance`, agent=fable) /
> 인사이트 `branch_einsum_copy_edit_epilogue`, `residual_spatialop_to_free_einsum_collapse`.

## 언제 쓰나(스캐너/시그널)
- `uv run ng scan mask_dominance` → `reports/candidates/worklists/mask_dominance.json`: ≥20×20 라우팅 평면이
  넷 비용의 ≥30%를 차지하는 배포 넷(약 60개, 16 done, 잔여 ~45). 각각 +0.2~0.9 후보.
- 시그널: 최종 `Where(mask, edit, input)` / `Pad(label)→Equal` / `Equal→Pad` output tail; `[1,1,30,30]` bool/u8 index 평면.
- **PRE-FILTER(필수):** positioned-content mask는 SKIP(아래 taxonomy). global-state/structured(ring·block·periodic·diagonal·threshold-run·count)만 채굴.
- agent_class = **fable**(태스크당 신규 메커니즘 설계). done/floor/타깃 목록은 `state/levers.yaml`의 레버 항목 참조(재공격 금지 리스트 포함).

## 물리(왜 되나)
- `Where(mask, edit, input)` epilogue가 하나의 free-output Einsum으로 붕괴하는 조건: per-cell 결정이
  **(작은 dynamic one-hot state) × (low-rank OR polynomial 위치 predicate)**로 인수분해될 때.
- 두 harness 사실 활용: (1) `input`/`output`[1,10,30,30]은 dtype 무관 **무료**; (2) `run_network`가 `output>0` 디코드 ⇒
  채널당 **SIGN만** 중요. off-grid 셀은 output이 one-hot `input`에 선형이라 자동 0(occupancy가 input에 실림).
- 30×30 coupling은 Einsum 내부(무료)에만 존재; counted node output = 작은 init + few-element dynamic state뿐.

## 레시피(단계)
1. `ng scan mask_dominance`로 후보 확보 → positioned-content pre-filter로 걸러냄.
2. 라우팅 predicate 분류 → 아래 서브패턴에서 factorization 선택.
3. Einsum 설계: 최종 op이 FREE `output`을 직접 쓰도록. dynamic state는 tiny operand(one-hot/count)로, 위치 구조는 static operand로.
4. `reports/candidates/taskNNN/regime.onnx` 빌드(`build_regime*.py`). **⚠️ 병렬 세션이 같은 candidate 디렉토리·`submission/overfit_nets/`를 공유** — 채택 직전 on-disk candidate 재측정, 오류/불일치면 build 재실행.
5. `uv run ng gate regime.onnx --task NNN` (bundled fail=0 + 비용 < **배포본**). unsigned TopK clean 확인.
6. `uv run ng adopt regime.onnx --task NNN --note "regime crack: <predicate>"` → tasklog에 메커니즘 + ⭐TRANSFERABLE 기록.

## 서브패턴

### batch3 서브레시피 (NEXT_SESSION 증류 — 반드시 숙지)
- **input-as-Einsum-operand for free counts (task295, 1604→393 +1.41):** `input` 자체를 최종 Einsum operand로
  (`bshw,…`) → 채널당 colour COUNT가 수축 내부에서 무료로 읽혀 ArgMax/Equal/Cast colour-추출 체인을 삭제. bounded run = rank-2 linear 곱.
- **base-N digit factorization (task398, 2813→1510 +0.62, mem 49B):** anti-diagonal `[r+j==f(param)]`을 base-5
  digit/carry로 인수분해 → 공유 small digit 텐서(~600 params); `δwv−δv0` WMAP이 배경 branch 흡수. 16-operand Einsum도 ORT에서 정상.
- **ConvInteger-as-free-output (task392, 1937→1310 +0.39):** 1×1 ConvInteger + `pads`가 small-grid→30 배치를 free op 내부에서 수행;
  runtime u8 weight + `w_zero_point=1`로 signed 채널 라우팅; 상수-1 bias 채널은 Pad로.
- **k-stacked Where carriers (task051, 1744→943 +0.32):** k-stack fp32 carrier를 하나의 `Where(bool_vec, pair11, pair10)`로 빌드
  ([2,1,1,1] pair = 240B, Cast+Pad 360B보다 싸다); mixer의 `δ_d0` bg-gate가 배경 전용 edit의 명시적 in-grid row/col gate를 제거.
- **residue one-hots via Gather(eye) (task061, 1668→1617 +0.03):** 주기적 `(r·c)%m=((r%m)(c%m))%m`을 residue one-hot으로 분리;
  k-indicator `1−(t−(k−1))²`. **ORT OneHot는 CPU 커널 없음 → 항상 `Gather(eye)` 사용.**

### predicate 유형별 factorization
- 다항 좌표 predicate(r±c, |r−a|=|c−b|, r=k) → `1−E²` multilinear form(task141: diagonal E(r,c)=0을 11-operand Einsum에 `1−E²`로).
- separable rowmask∧colmask → outer-product `(1+r⊗c)` + mixer(off-diagonal 하나 가진 I)(task341).
- dynamic per-example color → tiny `Concat(LC−e0, ones)` operand.
- in-grid gate → Einsum 내부에서 `input[0]`을 곱함(task303).
- **identity-augmented Einsum (task033):** term 축에 all-ones column + δ-slice 추가 → `Where(cond,X,input)` 인입.

### batch1/2 추가 레시피
- **orbit-SUM (task240, mem=0!):** fold-MaxPool/ArgMax를 orbit 합으로 대체(`e_color+k·e0`를 Einsum 내부 summation으로 read, 보장된 bg orbit이 뺄 e0 공급); telescoped ring/frame label = Σ nested-square rank-1.
- **mixed-radix digit tables (task075):** block-structured copy/gather → source·output 양쪽이 공유하는 mixed-radix digit 테이블 → Gather/DepthToSpace assembly로 확장.
- **product-of-diagonals (task260):** K dynamic 평행 대각선의 union = `1−Π(x−kᵢ)²`(2K reused bilinear factor, int32 Einsum — 고차수는 int 정확도가 fp보다 유리); affine bg 채널 `A·F+B`는 t∈{0,1} term 축; sentinel-k가 결여 branch gate(제어흐름 없음).
- **INIT banks + runtime one-hot (task159):** per-case geometry를 INIT bank(params=element count, dtype-free)로 두고 tiny runtime one-hot `sel`이 두 번 등장(s,t→δ)해 row/col 독립 선택.
- **threshold-run masks (task345):** rays/fills/gravity(per-column obstacle-row bounded) = signed linear-in-h 임계 `±(h−a·rho)`; integer-h interval slack으로 own+shifted column을 덮는 dyadic 스케일 하나; `A[c,c]=−1` self-flip이 이미-편집된 셀 흡수.
- **free in-grid mask (task109):** `ingrid=(ΣR)(ΣR)`로 in-grid mask를 같은 fold/permutation operand에서 무료로; two-term spatial⊗channel fill = `[m;ones]×[d;e0]` stack.

### free-input / output-tail 서브패턴 (비-mask 일반화)
- **self-Einsum axis-activity gate (task067, →25.0):** `input`을 한 Einsum에서 두 번 wiring(`bkrc,blcd->bkrc`)해 축 활성으로 최종 output을 gate. mem 39→0.
- **free-input profile 수축:** 재료 평면이 REDUCE만 되면 FREE input에 직접 수축(`einsum_vs_free_input_reduction`). moment-statistics(task158: n,Σr,Σc,Σr²,Σc² = 5개 [10]-output einsum), solid-marker profile(task008: 2×2 marker를 row/col occupancy einsum으로), band-profile 수축(task202: distinct-colour row/col profile로 연결성 회피).
- **separable-remap einsum (task108):** fixed spatial remap `out=P·in·Pᵀ`, P=U@S → 5-operand `'ra,ai,zcij,bj,sb->zcrs'`; mem=0, params=|U|+|S|(+1.175).
- **direct one-hot Gather output (task343):** label→Equal→Pad을 `Gather(input, final_cols, axis=3)`로(active col = c mod period, off-grid = all-zero col29). counted bool Equal 텐서 삭제.
- **free_final_onehot_equal (task17/95/146/381):** 10ch 확장을 최종 Equal/Where로 지연; 결과가 입력과 한 overlay class만 다르면 `Where(mask, onehot_color, input)`으로 label carrier + 최종 Equal 둘 다 제거.
- **label-pad vs onehot-pad 순서 (task308/382):** compact area < 90셀이면 `Equal(label)→Pad(output)`(=area×10), 아니면 `Pad(color_index)→Equal(output)`(=900B carrier). break-even = compact_area × 채널수 vs padded_label_area.
- **category_augmented_separable_lut (task055):** row/col-separable override 평면을 augmented LUT로 접어 여러 30×30 Where/mask 삭제(작은 row/col 벡터만 추가).
- **strided-Conv fixed-block counts (task011):** N개 고정-블록 Slice+ReduceSum branch → 원 one-hot 입력 위 하나의 strided Conv + compact Slice/Reshape(+1.52).
- **terminal GridSample (task209):** D4/affine 변환의 explicit gather+bounds-mask+zero-pad를 fp16 [1,30,30,2] sample grid의 GridSample 한 free node로.
- **sparse-edit ScatterND (task054):** line/ring 셀을 255로 write(reduction='max') → 중복 write는 idempotent union; 최종 Equal. **단조 sparse overwrite에서만 유효**(inactive 중복 write가 앞선 edit를 지우면 안 됨).

## 함정/거부 사례

### ⚠️ VEIN TAXONOMY — positioned-content = FLOOR
- **GLOBAL-STATE mask → CRACKABLE(11/12):** mask의 dynamic content가 ≤~16 element의 global state(size/color/count/diagonal offset/obstacle-row)이고 30×30 coupling이 STATIC 구조.
- **POSITIONED-CONTENT mask → FLOOR:** mask = (데이터 의존 motif CONTENT) ⊛ (데이터 의존 2-D POSITION). bool 평면은 1B/element이고 Where가 bool을 받지만 Einsum operand는 ≥2B(fp16; 종종 input co-bind로 fp32) ⇒ 모든 exact mask-free factorization이 더 비싸다(~2500B vs ~1750). 확인된 FLOOR: **task112**(GatherND positioned motif), **task163**(임의색 3×3 patch를 데이터 의존 block slot에 stamp), **task099**(14 box-config, function-space rank 64), 그리고 `state/levers.yaml floor_tasks: [112,163,99,102,62,12,348,124,354]` + `batch6_floored`.

### 날카로워진 crack 조건 (batch6 — 셋 다 참일 때만 crack, 아니면 FLOOR)
1. **GLOBAL-STATE 라우팅**(≤~16 스칼라: offset/size/color/count의 predicate), 데이터-의존-위치의 content도, 데이터-의존 run/coupling도 아님.
2. 900B 평면이 Where **ROUTING** mask여야 함, Equal **DECODE** discriminator 아님(label30→Equal→10ch = dense-recolor floor: 124/354/390).
3. mask fraction ≥~45%(저-fraction ~30%, e.g. 090은 비-mask compute에 짐). **AND** fold의 fp32 operand(free input에 co-bind)가 900B bool mask보다 작아야 함 — 진짜 low-rank/static 라우팅 필요, per-cell placement matrix면 안 됨.

### fp32-co-bind economics (깊은 floor)
- free fp32 input을 co-bind하면 모든 Einsum operand가 4B로 강제됨 → rank-2 identity-augmentation(A[2,10,10]+R[2,10,10]) ≈ 1600B fp32 > 900B bool Where-mask(bool=1B/cell, 0 params) ⇒ 구조적이어도 floor(034/168/381).
- 탈출구 = mixed-dtype Einsum(fp32 input co-bind에도 fp16 carrier) — ORT uniform-T 규칙에 막힘. **vein 전체의 top residual 레버**(task063 등 ~600B fp32 carrier 잔존).

### 기타
- **sparse-initializer DEAD:** 공식 scorer `SparseTensorProto.name` crash — 재시도 금지.
- **비-mask 확장(CONV-FP32 arsenal 074/080/198/216/162/2/349):** bool/u8 Where-mask가 아니라 NEW fold 필요(Conv/Slice를 free-output 수축으로) — 단일 태스크 최대 upside지만 미증명.
