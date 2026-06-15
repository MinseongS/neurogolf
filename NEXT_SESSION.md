# NeuroGolf — 다음 세션 인수인계 (2026-06-15 마감)

## 현재 상태 (확정)
- **실제 Kaggle LB: 6379.07** (2026-06-15 확정·잠금).
  경로: 6338.91→6344.58→6356.04→6370.12→6374.94→6377.27→6379.07 (오늘 **+40.16**).
- 400/400 적용, 모든 적용 네트워크 fresh-gated.

## 🔑 핵심 무기: MEMORY FLOOR-BREAK (이게 메인 레버)
`reports/FLOOR_BREAK_GUIDE.md` + `memory/neurogolf-floor-break.md`에 전체 기법.
- 9k "floor"는 10채널 one-hot 중간텐서 때문. uint8 레이블맵 L[1,1,30,30] + 마지막 `Equal(L,arange)
  →무료 output`(BOOL, opset11)으로 우회. 또는 단일 Conv/Gather/separable And→output.
- **실제 LB에 1:1 반영됨 (3회 검증).** 큰 메모리 감축만 추구(작은 압축은 Kaggle 무반응).
- GOTCHA: 채점 one-hot은 in-grid 배경=ch0=1, **off-grid=전채널 0**. "검정(ch0=1)" ≠ off-grid(all-zero)
  → 레이블 sentinel≥10 또는 `Where(cond|~ingrid, input, bg_onehot)`. 색 카운트는 ch0 가중치 0인 Conv로
  (전채널 ReduceSum은 ch0 배경 포함 버그). ingrid=ReduceMax(input,axis=1)>0.5.
- 검증: `PYTHONPATH=. .venv/bin/python reports/verify_fb.py N` → STORED ok + FRESH 200/200.

## 워크플로 (검증된 정공법)
1. **own custom 재인코딩** (bail 0, 규칙 깨끗 보장): 31개 완료. 남은 건 detection-heavy floor라
   수익 점감(<0.5). 거의 소진.
2. **public-net from-scratch** (현재 메인 작업): ① triage 에이전트가 `src.show N --gen` 읽고
   per-cell 결정론 규칙만 FEASIBLE 분류(~40-65% 통과). ② build 에이전트가 from-scratch 빌드.
   완료: 97 126 008 092 265 287 328 293 222 (9개). cap 중엔 메인이 직접 빌드 가능(97/126이 예).
3. 메인이 `src.adopt N`로 직렬 채택(현재 net real을 이겨야 채택). 묶어서 제출.

## ⚠️ 동시성 교훈 (중요)
**build wave는 2-3 에이전트로 제한.** 8개 동시 → stream watchdog가 600s 무응답으로 대량 kill됨
(부분 파일은 salvage: verify_fb로 검증 후 adopt). 에이전트엔 "sub-agent 금지, 파일 일찍 쓰고 점진
반복" 명시. Sonnet도 mechanical 빌드는 충분(easy/med), 어려운 건 Opus.

## 📋 다음 세션 작업 큐 (feasible, triage 완료, 미빌드)
- **재시도(파일 있으나 미완/오류, /tmp나 src/custom에 잔재 가능)**: 132 55 110 (이번 wave 진행중이던 것),
  20 34 161 224 306 (8-agent wave에서 stall).
- **미빌드 feasible**: 70 175 198 131 228 324 333 86 51 397 (triage FEASIBLE 확정).
- **bail 확정(재시도 금지)**: 66 77 25 23 157 243 162 173 54 112 71 364 5 90 280 145 193 192 69 76
  379 9 89 4 44 118 2 387 17 383 359 368 154 148 390 (sprite/flood/correspondence/ambiguous/global).
- **triage 미실시**: same-shape public 후보 ~30개 더(reports/floorbreak_targets.json 상위 중 미분류).
  → triage 에이전트 1개 돌려 큐 보충.
- 기대: feasible 1개당 +0.5~3 (geometric clean일수록 큼). 남은 feasible ~18개 → ~+25~35 잠재.

## 천장 재추정
public-net floor-break로 ~6420-6470 현실권. 그 이상(→7000+)은 mem-0 단일Conv 패턴을 광범위 적용
해야 함(어려움). 1등 7700 = 400개 전부 ~300바이트 극한 압축(=floor-break를 전 task에).
- 2026-06-15 세션: set-aside custom 11개 중 10개 재적용(+202) → +5.67 LB. genverify 감사 완료.
  - **교훈 재확인: stored 델타는 Kaggle를 ~3.5배 과대평가.** sub-1pt 압축 win은 Kaggle에서 ~0.
    real-0 복구와 큰 압축만 실제 LB를 움직임. task204 0->13.90도 실제론 부분점수였음.
  - **genverify 감사 결론: 신규 real-0 jackpot 없음.** 구조적 real-0 = 219/255(둘 다 infeasible)뿐.
    batch가 90/251/101을 오탐(실제 200/200). 남은 lever = Phase 3 borderline 안정화뿐(저수익).
  - **borderline 위험순(격리 n=200):** 118(.920) ≫ 157(.960)~209(.965)~2(.970) > 366(.985) > 18(.995).
  - opset>10은 기존 public base(245/400)엔 정상(6338.91 채점됨). opset<=10 규칙은 내가 local-score로
    채택하는 net에만 적용.

## 핵심 규칙 (반드시 지킬 것)
1. **채택은 오직 `python -m src.adopt N`** (fresh-gate). raw `pipeline --methods custom` 금지(stored keep-best = 6505→4374 버그).
2. **로컬 stored 점수 신뢰 금지.** Kaggle은 fresh 인스턴스로 채점 → 진단은 `src.genverify`/`fresh_pass`, 진실은 실제 LB.
3. **공개 artifact 머지 금지** — 이미 소진됐고 머지하면 회귀(이번에 LB로 증명). `src.merge_external` 절대 금지.
4. **제출 한도 = 100/day** (Kaggle API maxDailySubmissions 확인, 2026-06-15. 과거 "~5/24h"는 오진단). 0점 진단은 무료 로컬 genverify로 충분(제출 아껴쓸 필요 없음), 제출 전 사용자 승인은 유지.
5. custom 파일명은 3자리 zero-pad (`task071.py`). 에이전트는 `src/custom/taskNNN.py`만 작성, adopt/commit은 메인이.

## 작업 우선순위 (번호순 ❌, 효율순 ⭕)

### 1순위 — set-aside custom 11개 재적용 (즉시 무료, 만들 필요 없음)
이미 빌드된 custom .py가 있는데 de27cbf 복원으로 공개 네트워크가 적용 중. `src.adopt`로 검증만:
```
062 156 159 166 189 204 225 240 256 288 297
```
각각 `python -m src.adopt N` → custom이 현재 공개 네트워크를 real+compact로 이기면 채택(아니면 자동 거부). 채택분 모아 커밋. (주의: 이들은 과거 6372 머지와 묶여서만 제출됐던 것 → 개별 fresh-gate가 안전.)

### 2순위 — feasibility-랭킹 신규 custom 풀이 (본진)
- 대상: **14~16점대 + 깨끗한 generator**. `reports/arc_mapping.json`의 generator로 사전 분류:
  - 풀이가능: hmirror/vmirror/rotate/transpose/tile/주기 패턴
  - 풀이불가(스킵): continuous_creature/overlaps/num_sprites/rotate-search/mega/random-scatter
- 방법: 2~4 에이전트 wave(>4는 5h Opus cap 소진). 각 에이전트는 `src.show N --gen`으로 규칙 확인 → 빌드 → 격리 `fresh_pass` 200/200 → `src/custom/taskNNN.py`만 작성. 메인이 `src.adopt N`로 직렬 채택.
- 트릭 라이브러리: SOLVING.md. 핵심 = Gather 순열, 런타임 Conv/ConvTranspose 가중치, outer-product, 부호 패킹, 단일 `Where(mask,onehot,input)` fuse(10채널 중간텐서 회피).
- 기대: task당 +1~2 (이미 통과하는 걸 더 작게). 현실 천장 ~6700.

### 3순위 — Phase 3 borderline 안정화
확률적 0점 위험(통과율 94~96%)을 정확-규칙으로 교체: `118 157 2 209` (+ 23 233 76). flip 위험 제거.

## 영구 풀이불가 (재시도 금지)
- **219** (위상 모호 = 비결정적, 200k 중 138 충돌), **255** (random_pixels 0.5)
- flood/connectivity 187 251 286 338, multi-object 96 319, size-unrecoverable 358

## 진단 도구
- `python -m src.genverify` → 400개 fresh 감사, `reports/genverify.json` (n=40 배치, Pool maxtasksperchild=1)
- 격리 정밀 확인: `fresh_pass(N, 200)` 단일 프로세스 (배치는 false-negative 가능)
- 차등 probe(제출 필요): `/tmp/nullnet.onnx`(전부 0점), `/tmp/make_probe.py`(task 묶음 zero), `/tmp/verify_cand.py`(외부 onnx eval+fresh)
- 현재 0점 진단: 확정 0 = 219·255(불가)뿐. borderline 10개(118 157 2 209 151 332 31 359 319 23)는 확률적.

## 제출 절차
```
python -c "from src.pipeline import pack; pack()"   # networks/만 zip (--pack 플래그는 solver 돌리니 금지)
kaggle competitions submit -c neurogolf-2026 -f submission/submission.zip -m "..."   # 사용자 승인 후
```
kaggle CLI: `/opt/homebrew/Caskroom/miniconda/base/bin/kaggle`
