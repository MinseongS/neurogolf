# NeuroGolf — 다음 세션 인수인계 (2026-06-15 마감)

## 현재 상태 (확정)
- **실제 Kaggle LB: 6338.91** (신규 최고점, Kaggle은 best 제출을 유지하므로 잠긴 점수). stored manifest ~6400.18.
- 400/400 적용, 모든 적용 네트워크 fresh-gated. git HEAD = `ba37289` 계열, 트리 클린.
- 이번 세션 성과: 6372 머지 회귀(−148) 발견·되돌림 → de27cbf(6300.89) 복원 + 191/264/018 복구(+38.02).

## 핵심 규칙 (반드시 지킬 것)
1. **채택은 오직 `python -m src.adopt N`** (fresh-gate). raw `pipeline --methods custom` 금지(stored keep-best = 6505→4374 버그).
2. **로컬 stored 점수 신뢰 금지.** Kaggle은 fresh 인스턴스로 채점 → 진단은 `src.genverify`/`fresh_pass`, 진실은 실제 LB.
3. **공개 artifact 머지 금지** — 이미 소진됐고 머지하면 회귀(이번에 LB로 증명). `src.merge_external` 절대 금지.
4. **제출 한도 = rolling ~5/24h** (무제한 아님). 0점 진단은 무료 로컬 감사로, 제출은 가끔 확인용. 제출 전 사용자 승인.
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
