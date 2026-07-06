# NEXT SESSION — NeuroGolf handoff (2026-07-06, S17 종료)

다음 세션 시작 프롬프트:

```text
/Users/minseong/project/neurogolf 에서 NeuroGolf 점수 개선을 이어가자.
먼저 이 파일(NEXT_SESSION.md) + 메모리 [[neurogolf-urad-7225-bundle-vein]](S17 safe best) +
[[neurogolf-overfit-mode]] + [[neurogolf-strategy-directive]] +
[[neurogolf-cristianoc-reference]] 를 읽고 시작해.
```

## 🟢 S17 확정 결과 (2026-07-06, 모두 LB 확인됨)
- **안전 최고 = LB 7245.33** (sub 54396297, private-robust). udit 7237.17 덤프에서 task206 안전채택
  (3766→1795, +0.741; RoiAlign+5Einsum, fresh inc0/cand0/div0). ← 마감 택1 헤지.
- **overfit 최고 = LB 7248.76** (sub 54396589, public-only). v1(7246.88) + udit/poby min-merge
  (377/205/076 overlay) = +1.886. 재빌드 `scratchpad/mine07/minmerge.py`, 넷=`submission/overfit_nets/`.
- 🚨 **제출한도 = 100/일 (5 아님! 미신 반증됨).** 진짜 400 에러 원인 = 파일명이 `submission.zip` 이어야 함
  (`overfit_submission.zip` → 400 "must be named submission.zip"). 제출 전 반드시 `submission.zip`으로 복사/rename.
  Kaggle API `max_daily_submissions=100` 확인. → 자유롭게 제출.

## 🎯 사용자 방향 (S17 확정): "리더보드 통과되면 오버핏이든 뭐든 상관없어. 7248.76 완전히 가져가고 더 올려."
private-LB 위험 명시적으로 수용. **public LB 최대화가 목표.** 안전/오버핏 구분 없이 점수↑면 채택.

## 🔴 다음 세션 = 오버핏 극대화 (available 레버는 소진, 남은 건 DEEP per-task 재구성)
S17에서 mechanical 오버핏 레버는 전부 소진 확인:
- 공개 leaky 넷: min-merge로 전부 포섭됨 (udit=poby=waterxiao 동일 pool). 신규 없음.
- walk-chain step-cut: task286만 유효(완료). task243=mega-Einsum(operand cut해도 counted plane 안 줄음, no pay).
- task066: 독립 contraction 5개, walk chain 아님 → floor.
**남은 유일한 상승로 = DEEP per-task leaky 재구성** (사용자 승인, 고노력·불확실):
1. ⭐ **bit-pack 인코딩 레버** (공개 task319 5852 넷에서 발견): BitShift/BitwiseAnd로 객체를 27×27 full plane
   대신 [1,10,5,5] 작은 코드로 압축 (11664B→~250B, 3× 절감). 우리 고비용 넷 중 full 30×30/27×27 객체
   plane을 materialize하는 것들에 이식 시도. 공개넷을 teacher로 op-census 비교해 mechanism 추출.
   대상 = 우리 최고비용 넷들 (285=25234, 286=23971, 233=33242, 133=21526, 018=25445...).
2. **canvas-crop 레버** (task018식, memory "narrow"): bundled 그리드가 30×30보다 작으면 캔버스 축소.
   bundled 데이터 max extent 측정 → N<30이면 넷을 N×N로.
3. **worst-case 상수 cut**: unrolled 루프 step수/TopK-K/detector bank가 이론상 max로 sized된 넷 →
   bundled-min으로 축소 (gate=bundled fail=0만). 단 S16 측정상 우리 넷은 이미 tight (5프로브 중 1승).
방법: 고비용 넷마다 (a) 공개 오버핏 넷 있으면 teacher diff, (b) 없으면 bundled-only 재구성 직접 설계.
게이트 = bundled fail=0만. 제출 파일명 반드시 `submission.zip`, 한도 100/일.

## 📌 참고
- overfit README = `submission/OVERFIT_README.md`. 넷 = `submission/overfit_nets/`. min-merge = `scratchpad/mine07/minmerge.py`(세션소멸, 재작성 필요).
- 안전본 206만 clean adopt. 377/205는 private 위험으로 safe 거부(overfit엔 포함).

## === 현재 상황 (S17 종료, 2026-07-06) ===
- **확정 LB 최고: 안전 7245.33 (sub 54396297) / overfit 7248.76 (sub 54396589).** local manifest 7245.22,
  오프셋 +0.11 일관. 마감 07-15 = private LB. S16 안전본 54394428(7244.59)도 헤지로 보존.
- S17 방법 = 신규 업로더(udit 7237.17 / poby 7235.83) 재채굴 + task206 안전채택 + overfit min-merge.
  주요 발견: 제출한도 100/일(5 미신 반증), 파일명 반드시 submission.zip.

## === (이하 S16 기록, 참고용) ===
- **S16 확정 LB 최고 = 7244.35** (sub 54393583). 재제출 7244.48 local (task387 fold 포함, pending → ~7244.59).
  로컬 manifest 7244.48, 오프셋 +0.11 일관. 마감 07-15 = private LB.
- S16 궤적: 7242.29 → 7244.14 → 7244.35 → 7244.48(local). 방법 = **매칭 엔진 실증** + 공개 unfiltered
  재채굴 + **time-for-cost einsum fold**(task387 +0.239, bit-identical).
- **🆕 time-for-cost fold 레버 부활**: `reports/scripts/fold_finder.py` → counted 평면이 SUM-축약되면
  free-input einsum으로 접음. **task387 +0.239** 성공(ReduceSum crop→Einsum('bchw,c->bh') + mask-from-scalars).
  정밀 규칙: LINEAR producer(ReduceMax/compare 아님) + SUM-contraction consumer + SINGLE-USE 평면만 접힘.
  벤 상태 = 1승 후 620B 위로 소진(064/134/025/105/216/009/364 전부 floor). 잔여 ≤620B tail(131/174) 저EV.
- **엔진 증명 완료(0순위 달성).** match_insight 랭킹 → mine_public_bundles/mine_unfiltered grader측정 →
  fresh_verify cand≤inc → 5단계 채택 → mechanism_coverage.json 기록. 대표 = task222(dropped-Slice,
  bit-identical, +0.158). 총 21태스크 채택 ~+0.41, 전부 bit-identical/cand≤inc(private-LB 안전).
- **런타임 프로파일 완료(0순위 달성).** 전체 400넷 = 229ms/pass, 최고 82ms(task001). 경쟁자 slow-태스크가
  우리는 sub-ms. `runtime_ms` 인덱스에 추가됨. **timeout 헤드룸 막대 → time-for-cost 레버 개방(단, 아래 참조).**

## === S16 핵심 판정 (재사용, 중요) ===
1. **urad 3메커니즘 자가역적용 = 둘 다 FLOOR로 판명** (엄밀한 파일럿 에이전트):
   · **value_info_crop**: 90개 crop 풀 소진. 모든 verbatim crop은 single-window fp32 floor(10ch 윈도
     불가축소, 색 랜덤). 366/233/205만 full-plane이나 compose/recolor(crop 아님). coverage `_pool` floor.
   · **qlinearconv_render**: 풀 이미 붕괴(오늘 공개프론티어 이식분). 남은 평면은 load-bearing 탐지/scatter,
     교체가능 render epilogue 아님. 신규-이식 넷의 un-collapsed one-hot만 미래 대상(mech15_output_scan.py).
   · **gridsample_warp**: 풀 전부 저비용 업스케일(≤2798), EV 낮음, 미검토. (유일하게 안 판 벤)
2. **공개 벤 수렴.** franksunp 7235.49 + llccqq624 7235.83 = per-task 우리보다 싼 min-merge. ryosuke 7230 =
   동일 계보(동일 후보셋). 모든 공개덤프 ≤7235.83. **신규 업로더 나타날 때만 재채굴** (koushik/biohack/
   yuu/poby = dataset-attached, 추출 0 — 첨부데이터셋 별도 다운로드 필요).
3. **⚠️ 8000은 압축 갭이 아니라 OVERFIT 갭일 가능성 매우 높음.** cristianoc oracle이 우리 고비용넷=알고리즘
   floor임을 독립검증; 공개프론티어 7235 상한; log-math상 7244→8000은 **400태스크 전부 ~6.7× 더 싸야** 함.
   → top~8000은 visible-case 튜닝(leaky const)일 개연성. 추격 = fresh-gate 규율(우리 private-LB 해자) 포기.
   **우리 fresh-gated 7244가 overfit 8000보다 private-LB에서 더 강할 수 있음.** overfit 전환 전 사용자 확인 필수.

## === 미완/느슨한 끝 ===
- **task076 fresh-gate 포기**(franksunp 15932→13235, +0.185): 우리 incumbent가 병적으로 느린 거대넷 →
  fresh_verify가 N=250에서도 15분+ 무출력(제너레이터+빌드 병목). 다음 세션에 다른 검증법 필요(예: bundled
  동등성 + 소량 fresh, 또는 incumbent를 먼저 골프해 빠르게). shim 재생성: mine 후 franksunp task076.onnx.
- **fold tail ≤620B**: task131(GatherND→ReduceSum 620B), task174(Cast→MatMul 600B) 미검토, 저EV(~+0.1).
- **소소 public tail** ~15태스크 Δ≤0.002 합계 ~0.02 미채택 (public/unfiltered_candidates.json). marginal.
- **GridSample 자가역적용** 미검토(유일한 미개봉 urad 벤, 단 pool 전부 저비용 업스케일 ≤2798, EV 낮음).

## === 다음 세션 우선순위 (S16 판정 반영) ===
1. **[사용자 결정 대기] 전략 방향**: (a) private-LB 안전 유지 = 신규 공개덤프 재채굴 + tail 정리(현재 거의 소진,
   ~+0.02/pass), 또는 (b) overfit 추격 = leaky-constant 튜닝으로 public 8000 시도(private 위험, 현 전략에 반).
   → 안전 grind은 마름. 큰 상승엔 (b) 승인 또는 미지-메커니즘 리서치 필요.
2. **미지-메커니즘 리서치(고위험, 구체 가설 먼저)**: 유일한 안전-상승로. 단 cristianoc가 알고리즘 floor를
   독립검증했으므로 "더 싼 알고리즘" 사냥은 금지 — 남은 건 IMPLEMENTATION golf뿐이고 그것도 대부분 소진.
   time-for-cost fold(런타임 헤드룸 막대)로 우리가 materialize하는 평면을 더 깊은 einsum으로 접는 각도가
   미탐색 후보이나, 구체 태스크·가설 없이 팬아웃 금지(floor 재확인만 됨).
3. task076 마무리 + tail 정리 (즉시 실행 가능한 잔여 안전점수).

## === 워크플로/게이트 (불변) ===
- 채택 = 백업 → networks/ 교체 → `live_to_exact_source NNN --write-src` → `measure_task NNN`(fail=0) →
  manifest → tasklog +⭐TRANSFERABLE + coverage_lib.record. 에이전트 byte추정 불신, 반드시 grader재측정.
- fresh-gate: `.venv/bin/python reports/scripts/fresh_verify.py NNN <shim.py> 1500`, cand_fail ≤ inc_fail.
  거대넷은 N=800으로 시작(느림). shim = `import onnx; def build(task): return onnx.load('<abs>')`.
- 제출: networks/*.onnx → `scan_unsigned_topk.py networks`(전수, uint8-TopK grader-killer) → zip → submit
  → `--csv`로 publicScore 폴링(CSV 파싱, descrip grep 오탐 주의).
- 도구: mine_public_bundles.py(byte-prefilter), mine_unfiltered.py(전수, 프리필터 놓친 것 잡음),
  profile_runtime.py, match_insight.py, coverage_lib.py. 덤프 추출 = extract_bundle.
- 🚨 죽은 길 재탐사 금지: value_info_crop/qlinearconv_render 자가역적용(S16 floor), train-to-golf, SGD-compile,
  task 319/48/285/188/101/090/117 공개-싼넷(overfit), int64→int32 on initializer.
- 모델: 기계적/레시피 → opus. 신규 메커니즘 설계·770 리서치만 fable. leads-only엔 SDD-건틀릿 금지.

## === 커밋 상태 ===
- 미커밋 대량(S12~S16). networks/*는 gitignore, 소스진실 = src/custom/*.py(재생성됨). S16에서 21태스크
  src/custom+tasklog+manifest, coverage, profile_runtime.py/mine_unfiltered.py 추가, 메모리 갱신.
  세션시작 시 커밋 정리 권장(사용자 지시 없으면 커밋은 물어보고). 백업 넷 = scratchpad(세션소멸) → 롤백은
  submission zip(54393046=7244.14) 또는 src/custom git 히스토리.
```
