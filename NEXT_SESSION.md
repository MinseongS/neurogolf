# NEXT SESSION — NeuroGolf handoff (2026-07-06, S15 + setup 세션 종료)

다음 세션 시작 프롬프트로 이걸 그대로 쓰면 됨:

```text
/Users/minseong/project/neurogolf 에서 NeuroGolf 점수 개선을 이어가자.
먼저 이 파일(NEXT_SESSION.md) + 메모리 [[neurogolf-strategy-directive]] +
[[neurogolf-runtime-timeout-dimension]] + [[neurogolf-urad-7225-bundle-vein]] +
reports/REBUILD_PLAYBOOK.md 를 읽고 시작해.

=== 북극성 (사용자 지침, 항상) ===
- **8000점을 향해 공격적으로.** 상위권 ~7982-8013. "바닥/소진" 판정은 공격 대상이지 정지신호가
  아님 (task133 "definitive floor"도 외부 넷이 깼음). 레버가 마르면 새 레버를 찾는다.
- **free `input`/`output` 텐서를 적극 활용해 비용을 없애라.** 둘은 0비용이고 op 내부는 미채점 —
  모든 작업을 free input에 대한 contraction / free output으로의 직접 기록 / 스팬 op의 단일 Einsum
  붕괴로 라우팅해서 counted 중간평면이 하나도 안 남게 설계.
- **딱 LB 통과 수준으로만 설계.** 게이트 = bundled fail=0 + mem 감소 + fresh ~98%+. 과최적화 금지.
- **제출은 아주 자유롭게.** 100/day, Kaggle은 최고점 보존 — 나쁜 제출은 손해 없음. 확정하려면 그냥 제출.
- **공개 인사이트를 캐내라, 넷만 복사하지 말고 — 그리고 400개 전체에 일반화하라.** 캐글
  discussion·code 노트북·넷 덤프를 적극 활용. 공개 해법이 특정 태스크에서 나보다 싸면, 그 넷 하나
  블라인드 이식으로 끝내지 말고 **왜 싼지(구조적 메커니즘/op-트릭)를 역설계** → REBUILD_PLAYBOOK +
  insight_registry에 기록 → 같은 패턴을 가진 우리 태스크 전수를 스캔해 적용. 넷 1개 = +1태스크,
  일반화된 메커니즘 = +N태스크 (진짜 상금은 여기). [[neurogolf-adoption-protocol]] ⭐TRANSFERABLE 필수.

=== 직전 세션 = 전략·구조 세팅 (완료·커밋·병합됨, 점수 변동 0) ===
이 세션은 점수 작업이 아니라 8000을 향한 세팅만 했음. 전부 main에 반영됨:
- **정리**(커밋 c0836f6, 3341038): 디스크 정크 ~150M, reports/ 12M→5.7M, arc-gen 미사용 생성기
  500개 삭제(400 유지), 메모리 40→27(바닥/천장 census 쳐냄), AGENTS/README/NEXT_SESSION+북극성.
- **🆕 인사이트→태스크 매칭 엔진 구축 + main 병합 (05ab432).** "인사이트가 400개 중 어디에
  적용되나"를 기억이 아닌 **쿼리+자동 floor제외**로 전환:
  · `reports/scripts/build_task_index.py` → `reports/task_index.json`(400행, 온디스크·gitignore,
    PROBE_VERSION 4). 태스크당 구조(op/dtype/평면/K)+의미(프로브)+경제(cost/bloat).
  · `reports/scripts/task_index_probes.py` — 확장형 의미 프로브(새 인사이트 = 함수 1개 추가).
  · `reports/scripts/coverage_lib.py` + `reports/mechanism_coverage.json` — (메커니즘×태스크) 결과
    원장 → known-floor 자동 제외.
  · `reports/scripts/match_insight.py --where "<술어>" | --mechanism X` → bloat 랭킹 후보 +
    `--emit-queue`. **엔진은 리드만 생성 — fresh-gate가 최종 심판.**
  · 백필 게이트 `tests/test_backfill_validation.py`(6개 메커니즘, gridsample_warp는 xfail).
- ⚠️ **엔진은 아직 0점 — 미검증.** 실점 루프(match→verify→adopt)를 아직 안 돌림.

=== 0순위(실점 시작): 엔진을 저비용으로 증명 + 런타임 계기판 ===
1. **엔진 실점 루프 1회** — `match_insight.py --mechanism signed_rect`(또는 urad value_info-크롭)로
   후보 상위 2~3개 → fresh_verify → 채택 → `mechanism_coverage.json`에 결과 기록. 첫 점수 뽑아 엔진 증명.
2. **400넷 런타임 프로파일링**(일회성·쌈) → timeout 헤드룸 측정 → 미탐색 **time-for-cost 레버** 개방
   ([[neurogolf-runtime-timeout-dimension]]). `runtime_ms`를 인덱스 축으로 추가.
- 🚨 토큰: leads-only 작업엔 싼 모델, SDD-건틀릿 금지. 깊은 지출은 신규 메커니즘/770 리서치에만.
  ([[neurogolf-strategy-directive]] 6번, [[model-allocation-preference]].)

=== 현재 상황 (S15 종료, 2026-07-06) ===
- 확정 LB 최고점 = **7242.29** (submission 54381272). 로컬 manifest 7242.18, 오프셋
  +0.11 일관. 마감 07-15 (최종 = private LB).
- 이번 세션 궤적: 7236.19 → 7240.17 → **7242.29** (+6.10). 방법 = 공개 노트북 번들 채굴.
- 🚨 우리는 **크게 뒤쳐져 있음**: 리더보드 top = **8013**, 우리 7242 → 갭 ~770pt.
  공개 덤프는 전부 ≤7235 (prvsiyan 7235.05가 공개 최고). 우리 7242는 공개 덤프
  여러 개를 min-merge + 우리 우위 넷 유지로 만든 것이라 사실상 "공개 파생 최고점".
  → 공개 순수-채굴은 성숙 단계(신규 덤프 재스캔은 계속). 남은 ~770pt는 상위권만 아는
  UNKNOWN 압축 메커니즘 — **이게 진짜 목표(8000)**. 독자 상승로 = 3-메커니즘 자가 역적용
  (아래 2번) + free-텐서 라우팅으로 우리 floored 태스크 재설계. S11 SGD-compile·S12
  train-to-golf는 반증됐지만, 그건 특정 접근의 실패일 뿐 "천장"이 아님(floor 증명은 틀릴 수 있음, 39번).

=== S15에서 새로 알아낸 것 (중요, 재사용) ===
1. **공개 번들 채굴 루틴이 실질 상금원이었다.** 많은 업로더가 400-net 번들을 노트북에
   base64로 임베드함. 총점이 우리보다 낮아도 개별 태스크는 더 쌀 수 있음(점수=합).
   재사용 스크립트 = `reports/scripts/mine_public_bundles.py` (추출+측정+diff 자동).
2. **prvsiyan/neurogolf-<score>-w-visualizations = 모든 공개 소스의 MIN-MERGE 노트북.**
   단일로 가장 가치 높은 덤프. 우리가 방금 채택한 넷의 더-골프된 버전까지 있었음.
3. **안전 채택 규칙 (private-LB 최종 대비, 절대 지켜):**
   태스크T의 소스넷 채택 조건 = bundled fail=0 AND grader cost < 우리 것
   AND `fresh_verify T <shim> 1500` 에서 **cand_fail ≤ 우리 incumbent_fail**.
   cand가 더 fresh-fail하면 기각(공개점수는 오르지만 private에서 0 위험).
4. **영구 overfit 함정: task 319, 48, 285.** 모든 공개 소스의 싼 넷이 fresh-fail
   (319는 85~133/1500 vs 우리 4~10). 절대 채택 금지, 우리 것 유지.
5. **우리 walk-einsum 플레이북에 없던 3개 메커니즘** (공개 넷 분석으로 확보):
   (a) value_info-legalized Slice/Pad 크롭 — free input에서 bbox 크롭 직접.
   (b) terminal GridSample — fp16 [1,30,30,2] grid = gather+mask+zero-pad 1 free node.
   (c) QLinearConv/ConvInteger signed 렌더러 — u8 codes+x_zero_point=1, i32 출력 >0 채점.
   (+ canvas-crop: 30×30→NxN 축소, 좁음.)
   → **이게 다음 진짜 상금 경로**: 남들이 안 판 우리 floored 태스크에 (a)(b)(c) 직접 역적용.
6. **task133 "DEFINITIVE FLOOR" 반증됨.** S13에서 내가 29636이 절대바닥이라 정교하게
   증명했는데 외부 넷이 21526 fail=0으로 깼음. 내 floor-증명은 틀릴 수 있다 = 외부
   프론티어 넷이 진짜 심판. 앞으로 "floor 증명했으니 재탐색 금지" 류를 과신하지 말 것.

=== 이번 세션 할 일 (우선순위) ===
1. ⭐ **공개 인사이트 채굴 = 넷 덤프 + discussion/code 둘 다** (저비용·반복, 마감 D-9라 유출↑):
   (i) 넷 덤프: `kaggle kernels list -s neurogolf --sort-by dateRun | head -30` → 신규/고점 pull
   → `.venv/bin/python -m reports.scripts.mine_public_bundles <부모dir>`
   → `reports/public_bundle_candidates.json` → 각 후보 fresh-gate → 안전규칙 채택.
   특히 prvsiyan `-w-visualizations` 최신본, uditjain(데이터셋 첨부형), yuu111111111, biohack44, franksunp.
   (ii) **discussion/code 읽기**: `kaggle kernels list -s neurogolf` + 대회 discussion 탭에서
   설명·트릭 공개글 확인. 공개 해법이 특정 태스크에서 나보다 싸면 **넷 이식으로 끝내지 말고 왜 싼지
   메커니즘을 역설계** → REBUILD_PLAYBOOK + insight_registry 기록 → **같은 패턴 우리 태스크 전수 스캔·적용**
   (넷 1개=+1, 일반화=+N). 이게 아래 2번(자가 역적용)의 상시 입력원.
2. ⭐⭐ **3개 메커니즘 자가 역적용** (남들 안 판 우리 태스크 = 유일한 독자 상승):
   우리 floored 태스크 중 (a) bbox/윈도우 크롭 여지(value_info), (b) 게더/워프 출력
   (GridSample), (c) 색 렌더링/부호가중(QLinearConv)에 맞는 걸 찾아 적용. 후보 발굴부터 —
   `reports/manifest.json`을 grader-cost 내림차순 정렬해 고비용 태스크부터 (a)(b)(c) 적합성
   판정. (구 min_stat/struct_scan 스캐너·json은 repo 정리로 제거됨; 필요하면 새 finder 작성.)
3. (연구, 고위험) ~770pt UNKNOWN 메커니즘 브레인스토밍. 상위권(8000+)의 압축을 특정
   룰클래스에서 역설계. 구체 가설 먼저, 그다음 팬아웃 (아이디어 없이 돌리면 floor만 재확인).

=== 미완료/느슨한 끝 (이어서 처리) ===
- **미완 fresh-gate 2건 (marginal, 재실행하면 됨):**
  · task157 (franksunp, 48111→46553B, +0.031) — 거대넷이라 1500 fresh 게이트가 매우 느려
    S15에서 미완. shim: urad류 로더. cand≤inc면 채택.
  · task76 (urad, 16557→15042, +0.096) — 마찬가지로 거대넷 게이트 미완.
- **uditjain13 번들 추출 실패** (base64 임베드 아님 = 데이터셋 첨부형). 원하면
  `kaggle kernels pull` 후 첨부 데이터셋 별도 다운로드. 7232.14라 신규 넷 있을 수 있음.
- **prefilter가 byte-size<0.97×ours 프록시였음.** grader-mem이 파일크기와 안 맞는
  태스크(작은 파일·높은 mem)는 놓쳤을 수 있음. 완전탐색 원하면 mine_public_bundles의
  prefilter 완화 후 전수 grader 측정(1600넷, ~10-20분).
- **소소한 tail 후보** 다수 미채택 (Δpts<0.02, 합계 ~0.1). public_bundle_candidates.json
  재생성 후 남은 것 게이트.

=== 워크플로/게이트 (불변) ===
- 🚨 모델 배분: Fable = 오케스트레이션·새 메커니즘 설계·채택판단·제출. 나머지 Agent는 model:"opus".
- 🚨 채택 프로토콜 (메모리 [[neurogolf-adoption-protocol]]): 백업 → networks/taskNNN.onnx 교체
  → `.venv/bin/python -m reports.scripts.live_to_exact_source NNN --write-src` (소스재생성)
  → `.venv/bin/python -m reports.scripts.measure_task NNN` (grader 재측정, fail=0 확인)
  → manifest 갱신 → tasklog 메커니즘 기록. **에이전트 raw-byte 추정 불신, 반드시 grader 재측정.**
- fresh-gate: `.venv/bin/python reports/scripts/fresh_verify.py TASK <shim.py> 1500`
  shim = `import onnx\ndef build(task): return onnx.load('<abs path>')`.
  (fresh_verify는 candidate가 module builder여야 함. urad류 raw onnx는 이 shim으로 감쌈.)
- 제출: 400 networks/*.onnx zip → `reports/scripts/scan_unsigned_topk.py` 전수(uint8-TopK
  grader-killer 체크, 반드시) → `kaggle competitions submit -c neurogolf-2026 -f
  submission/submission.zip -m "<msg>"` → `--csv`로 publicScore 폴링(descrip에 숫자 있으면
  grep 오탐, CSV 파싱으로 읽을 것).
- 죽은 길 재탐사 금지: train-to-golf 전체, SGD-compile, int64→int32 on initializer(Δ0),
  task 319/48/285 싼-넷(overfit), mech-16 per-object.

=== 커밋 상태 ===
- 미커밋 대량 (S12~S15 전체). networks/*는 gitignore, 소스진실 = src/custom/*.py (재생성됨).
  S15에서 ~35개 태스크의 src/custom + tasklog + manifest + 메모리 + mine_public_bundles.py
  변경/추가. 세션 시작 시 커밋 정리 권장 (사용자 지시 없으면 커밋은 물어보고).
- 백업: S15 교체 전 넷들은 scratchpad(세션소멸)에 있었음. git에 networks/ 없으니 롤백은
  이전 submission zip(54381016=7240.17, 54364720=7236.19) 또는 src/custom git 히스토리로.
```
