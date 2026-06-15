# NeuroGolf — 다음 세션 인수인계 (2026-06-15 마감)

## 현재 상태 (확정)
- **실제 Kaggle LB: 6370.12** (2026-06-15 floor-break 캠페인, 잠금). 경로 6338.91→6344.58→6356.04→6370.12.
  31개 custom floor-break 재인코딩(+25.54 real, stored와 1:1 일치). 큰 win: 215(1512)/239(3318)/019(3365)/
  384(3140)/156(3507). 남은 custom 타깃은 점감(<0.5): 137 166 252 312 197 269 297 256 등 ~+3-4.
  다음 방향: custom 꼬리 마무리 또는 same-shape 깨끗-규칙 미지 public net(reports/floorbreak_targets.md,
  bail율 높음 — 깨끗 geometric만 골라 triage).
- **🔑 MEMORY FLOOR-BREAK 돌파 (reports/FLOOR_BREAK_GUIDE.md + memory/neurogolf-floor-break.md):**
  9k floor는 10채널 one-hot 중간텐서 때문 → uint8 레이블맵 L[1,1,30,30] + 마지막 Equal→무료 output
  으로 우회. **실제 LB 1:1 반영 검증됨**(13개 재인코딩 +11.46 stored = +11.46 real). 큰 메모리
  감축만 추구(작은 압축은 Kaggle 무반응). 타깃 = 우리 custom을 메모리순(규칙 깨끗 보장, bail 0);
  공개 high-mem은 memorizer라 bail. 남은 custom 타깃: 85 13 239 301 234 240 199 189 217 65 288
  215 156 384 190 63 (~+12-15 추가 예상).
- 400/400 적용, 모든 적용 네트워크 fresh-gated. 트리 클린.
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
