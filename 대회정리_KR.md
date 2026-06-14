# NeuroGolf 2026 대회 완전 정리 (한국어)

_작성일: 2026-06-14 · 현재 실제 Kaggle LB: **6300.89** (400/400 제출 완료) · 마감: **2026-07-15**_

---

## 0. 가장 핵심적인 답 (당신의 질문)

> **"이 대회는 400개의 독립적인 모델을 만드는 대회인가? 모델들끼리 연계되는 게 있나?"**

- **네, 400개의 완전히 독립적인 모델(ONNX 네트워크)을 만드는 대회입니다.**
- **채점/실행 시점에서 모델들끼리 연계되는 것은 전혀 없습니다.** 각 task(과제)마다 별도의 `.onnx` 파일 하나를 만들고, 각각 따로 채점되며, 400개의 점수가 **단순히 합산**됩니다.
- 한 모델의 입력은 그 task의 격자(grid) 하나뿐이고, 출력도 그 task의 정답 격자 하나뿐입니다. 다른 task의 데이터를 보거나 공유하지 않습니다.
- **"연계"가 존재하는 곳은 오직 "개발 과정"입니다.** 즉,
  - 같은 풀이 코드(solver)·같은 트릭 라이브러리를 여러 task에 재사용하고,
  - 공개된 남의 정답 묶음(public artifact)을 가져와 task별로 더 좋은 것을 골라 섞고(merge),
  - task들을 "유형(generator signature)"별로 분류해 전략적으로 공략합니다.
  - 하지만 이것은 **사람(우리)의 작업 방식**일 뿐, **모델 자체는 서로 독립**입니다.

한 줄 요약: **모델 400개는 런타임에서 100% 독립. 연계는 오직 우리가 만드는 과정·도구·전략에만 존재.**

---

## 1. 대회 개요

- **대회명:** The 2026 NeuroGolf Championship (Kaggle, IJCAI-ECAI 2026 주최)
- **상금:** $50,000
- **마감:** 2026-07-15
- **저장소:** `/Users/minseong/project/neurogolf` (git, 로컬 전용)
- **링크:** https://www.kaggle.com/competitions/neurogolf-2026

### 무엇을 하는 대회인가
이름이 "NeuroGolf"인 이유: **골프처럼 "가장 적은 타수(=가장 작은 신경망)"로 문제를 푸는** 대회입니다.

- 바탕 문제는 **ARC-AGI** (추상 추론 격자 퍼즐). 총 **400개의 task**가 있습니다.
- 각 task마다, 그 task의 변환 규칙을 정확히 수행하는 **ONNX 신경망**을 직접 만들어 제출합니다.
- 정확도뿐 아니라 **모델이 작을수록(메모리·파라미터 수가 적을수록) 점수가 높습니다.**

---

## 2. 점수 체계 (가장 중요한 규칙)

### task 하나당 점수
```
점수 = max(1, 25 - ln(memory_bytes + params))
```
단, **그 task의 모든 예제(train + test + arc-gen)를 전부 맞혀야만** 점수를 받습니다. 하나라도 틀리면 0점.

- `params` = 네트워크의 element 개수 (dtype 무관)
- `memory_bytes` = element 수 × 바이트 (bool/int8=1, fp16=2, fp32=4)
- **`input`/`output`이라는 이름의 텐서는 메모리 계산에서 무료**입니다.
- 모델이 작을수록 `ln(...)` 항이 작아져 점수가 25에 가까워집니다.

### 점수 감각 (직관)
| 모델 크기 | 대략 점수 |
|---|---|
| Identity (아무 비용 0) | 25점 (이 데이터셋엔 그런 task 없음) |
| < 300 (mem+params) | ~19.5점 |
| < 1000 | 18+ 점 |
| 일반적 custom 풀이 | 14.9 ~ 17점 |
| memorizer(암기형) | ~13.5점 |

### 총점
- **400개 task 점수의 합 = 최대 10,000점.**
- 이것이 각 모델이 독립적이라는 결정적 증거입니다: **합산 구조**이고, task 간 상호작용 항이 없습니다.

---

## 3. 모델(네트워크) 제약 — 각 ONNX 파일이 지켜야 할 것

각 task의 `.onnx` 파일은 다음을 만족해야 합니다:

- **opset 10, IR 10**
- **입력 1개 / 출력 1개**, 이름은 정확히 `input` / `output`
- 입력 형태: `[1, 10, 30, 30]` (10색 one-hot 격자)
- 출력 형태: `[1, 10, 30, 30]`
- **파일 크기 ≤ 1.44MB**
- 공식 채점기는 `src/harness.py`에 그대로 미러링되어 있음

### 허용/금지 연산 (실전에서 부딪힌 것들)
- **금지:** `Loop` / `Scan` / `NonZero` / `Unique` / `Compress` / `Script` / `Function`
- **허용(의외로 가능):** `Gather`, `Mod`, `Where`, `ConvTranspose`, `ConstantOfShape`(opset9), `MatMulInteger`/`ConvInteger`(opset10)
- **타입 제약:** int32 또는 float 전용 opset11; int8/int16/double 불가 (단 ORT 안에서 fp64는 일부 가능)
- **함정 사례:**
  - `Slice`를 float 데이터에 쓰면 INVALID_GRAPH (task239) → Slice 피하기
  - int32 `Equal`도 무효 처리됨 → 정수 등호 트릭 `1 - Clip(Abs(a-b),0,1)` 사용
  - ORT의 `Where`/`Mul`/`Add`는 int8 미지원 (bool/int32/int64/float만)

### 부동소수점 정확성 불변식 (이 대회의 숨은 핵심)
> **모든 채점 관련 float 연산은 정수값이고 2²⁴ 미만으로 유지해야 한다 → float32에서 합산 순서와 무관하게 정확(exact).**

이 불변식 덕분에 "로컬에서 검증됨 = Kaggle에서도 통과"가 성립합니다. (fp64는 2⁵² 미만에서 정확, fp16은 정수 ≤2048에서 정확.)

---

## 4. ⚠️ 가장 비싸게 배운 교훈 — "신선한(fresh) 인스턴스" 채점

> **Kaggle은 우리가 다운로드한 저장된 예제가 아니라, 그 자리에서 새로 생성한(freshly generated) arc-gen 인스턴스로 채점한다.**

이로 인한 결과:
- **첫 제출:** 로컬 점수 6505 → 실제 LB **4374**로 폭락.
- 이유: **memorizer(암기형) 네트워크**는 본 예제만 정확히 맞히는 lookup이라 새 인스턴스에선 0점.
- 더 나쁜 것: 저장점수 기준 keep-best가 **일반화되는 공개 네트워크를 암기형으로 덮어써 버림.**

따라서 철칙:
- **로컬 저장 manifest 점수는 절대 믿지 말 것 (vanity number).**
- **유일한 진실은 실제 LB와 `fresh_pass` (새 인스턴스 검증)뿐.**
- 모든 채택(adopt)은 반드시 `python -m src.adopt N`을 통해서만 — 이것이 `evaluate`(저장 예제) + `genverify.fresh_pass`(새 인스턴스) 두 관문을 모두 통과시킴.
- `pipeline --methods custom` 같은 raw 경로 금지(저장 keep-best 버그 재발).

---

## 5. 풀이 방법 — 모델은 어떻게 만드는가

### 5-1. 자동 solver 티어 (`src/solvers.py`)
1. **identity** — `Identity` 노드, 비용 0, 25점 (이 데이터셋엔 해당 task 없음)
2. **conv** — 단일 `Conv`+bias, 1×1~9×9 커널 사다리, 채널별 정수 퍼셉트론 fit. 정수 가중치 × 0/1 입력 → float32 정확. **16~20.4점.**
3. **memorizer** — 모든 예제에 대한 exact-match lookup (base-11 패킹 등). **~13.5점. 단 새 인스턴스에선 0점이라 사실상 무용** (위 4번 교훈).

### 5-2. task별 custom 네트워크 (`src/custom/taskNNN.py`)
손으로 그 task의 **진짜 규칙**을 ONNX 그래프로 설계. 핵심 트릭 라이브러리(여러 task에 재사용 = 개발상의 "연계"):
- `Gather(input, 계산된_인덱스, axis=2/3)` = 행/열 순열·타일·반사를 거의 공짜로 (task376 19.43@262B)
- 입력에서 가중치를 런타임 계산하는 Conv/ConvTranspose (Kronecker/zoom/fractal 스탬프): task304, 221, 217
- MatMul로 채널 순열 행렬: task203 16.64
- 행벡터⊗열벡터 outer-product + 최종 fused MatMul/Conv: task246/082/284/295/313
- `>0` 채점이므로 두 색을 부호 하나로 패킹 `v = ch_a − ch_b`: task384
- **메모리 절약 핵심:** 최종을 `Where(mask[1,1,30,30]bool, onehot[1,10,1,1], input)` 하나로 → `output`에 직접 써서 10채널 중간 텐서 회피 (task166/063)

### 5-3. 풀 수 없는(INFEASIBLE) 유형 — 시간 낭비 금지
- flood-fill / 연결성 / 닫힌 영역 (187, 251, 286, 338)
- 다중 객체 재구성 / 회전 탐색 형태 매칭 (96, 319)
- 출력 격자 크기가 입력 내용으로 복원 불가 (358)
- 조밀 무작위 픽셀 산포 (255)

이런 유형은 memorizer/공개 네트워크가 천장. 공략 전 `reports/arc_mapping.json`의 generator 시그니처로 미리 분류 (continuous_creature/overlaps/num_sprites/rotate = 위험; hmirror/vmirror/rotate/transpose = 깨끗).

---

## 6. "연계"가 실제로 존재하는 유일한 곳 — 개발 워크플로우

모델 자체는 독립이지만, **만드는 과정**에서는 다음이 task들을 가로질러 공유·연계됩니다:

1. **공통 도구·코드:** `harness.py`(채점), `builders.py`(그래프 빌더), `solvers.py`(자동 풀이), `adopt.py`(채택 관문), `genverify.py`(검증) — 모든 task가 같은 인프라 사용.
2. **트릭 라이브러리 재사용:** 위 5-2의 패턴을 여러 task에 반복 적용.
3. **공개 artifact 병합(merge):** 공개 LB 천장은 ~6372 (rajathrpai/vyanktesh 등이 400개 onnx 묶음 공개). task별로 "우리 것 vs 남의 것" 중 **fresh-gate를 통과하면서 더 좋은 것**을 골라 섞음. (단 `src.merge_external`은 저장점수 기반이라 금지 — 반드시 fresh-gate.)
4. **에이전트 wave 운영:** 여러 task를 병렬 에이전트로 동시 공략(4개씩 안전, 그 이상은 세션 한도 소진). 단 한 번에 하나의 main-loop만 manifest를 건드림(레이스 방지).

→ 다시 강조: 이 모든 "연계"는 **우리의 생산 파이프라인 차원**이며, **제출되는 400개 모델은 런타임에 서로를 전혀 모릅니다.**

---

## 7. 저장소 구조

```
data/         대회 데이터 (gitignore; kaggle CLI로 다운로드)
networks/     task별 taskNNN.onnx 1개씩 (= 최종 제출물, git에 커밋됨)
src/
  harness.py    로컬 검증+채점, 공식 neurogolf_utils.py의 정확한 미러
  builders.py   ONNX 그래프 빌더 (opset 10 / IR 10)
  solvers.py    티어형 자동 solver
  pipeline.py   전체 task에 solver 실행 → networks/ + manifest + scoreboard
  custom/       taskNNN.py — 손으로 만든 task별 custom 네트워크 (build(task) 규약)
  adopt.py      일반화 게이트 통과 시에만 채택 (유일한 안전 경로)
  genverify.py  fresh 인스턴스 검증 (배치는 Pool maxtasksperchild=1 필수)
  show.py       --gen 으로 정답 생성기(규칙) 출력
reports/      manifest.json, SCOREBOARD.md, arc_mapping.json
submission/   제출 zip (gitignore)
```

핵심 규약: `networks/`의 네트워크는 **더 높은 점수의 것으로만 교체** → 커밋 이력이 단조 개선되어 리뷰 가능.

---

## 8. 현재 상태와 전략 (2026-06-14 기준)

- **실제 Kaggle LB: 6300.89 / 10000 (400/400 task 제출).**
- LB 최상위 ~7710 (CroDoc, jacekw-Deepseek, claudex — 전부 비공개 per-task 수작업). **공개 지름길은 ~6372가 천장.**
- **현실적 목표 천장: ~6450–6650** (당초 7000보다 보수적). 7000은 질적으로 다른 레버가 필요(아직 미발견).
- **돈이 있는 곳:** 14–16점대 174개 task가 prime custom 타깃. sub-16 중 ~142개가 깨끗한 기하 규칙 → custom 가능.
- **승리 공식(반복 검증됨):**
  1. generator 시그니처로 feasibility 랭킹 (quota 쓰기 전)
  2. 2~4 에이전트 wave로 custom 빌드+검증(evaluate + isolated fresh_pass 200/200), `src/custom/taskNNN.py`만 작성
  3. main loop이 `src.adopt N`으로 직렬 채택 후 commit
  4. ~30~50 실점수마다 재검증·재제출로 LB 확정

- **이득은 점진적:** custom은 "현재 일반화되는 저장 점수를 더 작은 정확 네트워크로 이길 때만" 도움. 깨끗한 풀이당 **+1~2점** 현실적.

### 진행 단계
- **Phase 0** — 신뢰성 스윕 (비일반화 공개 네트워크가 일반화 대안을 덮은 것 복구). 일회성 정확성 회복.
- **Phase 1** — ~142개 sub-16 기하 task custom 풀이 (본진). 목표 ~6700–6800.
- **Phase 2** — 16–18점대 밀어올리기 (깨끗한 변환을 19–21점 custom으로 교체). 7000 방향.
- **Phase 3** — borderline(새 인스턴스 1–5% 실패 = Kaggle 0점 위험) 네트워크를 정확 규칙으로 교체. 알려진 대상: 23 157 76 2 209 118 233.

---

## 9. 운영 메모

- **Python:** `.venv` (3.13, onnx/onnxruntime/numpy). base conda는 3.9라 너무 구버전.
- **Kaggle CLI:** `/opt/homebrew/Caskroom/miniconda/base/bin/kaggle` (`.venv` 안에 없음). **제출은 항상 사용자에게 먼저 확인.**
- **정답 규칙 확인:** `.venv/bin/python -m src.show N --gen` (arc-gen 생성기 = 정확한 규칙).
- **세션 한도:** 5시간 롤링 + 일일 Opus 한도. 병렬 에이전트는 4개 이하 안전. 한도로 죽은 에이전트가 남긴 `src/custom/` 파일은 재검증 후 채택 시도 가능.
- **예상 처리량:** 하루 ~20–40 풀이 → 142 타깃 ≈ 1–2주. 마감(7/15)까지 여유.

---

## 10. 한 페이지 요약

| 항목 | 내용 |
|---|---|
| 무엇 | ARC-AGI 400개 task 각각에 대해 "가장 작은" ONNX 네트워크 제작 |
| 모델 독립성 | **완전 독립.** 런타임 연계 0. 점수는 단순 합산 |
| "연계"의 위치 | 오직 개발 과정 (공통 도구·트릭 재사용·공개 artifact 병합·에이전트 wave) |
| task당 점수 | `max(1, 25 - ln(mem+params))`, 모든 예제 통과 시에만 |
| 총점 | 400개 합 = 최대 10,000 |
| 결정적 함정 | Kaggle은 새로 생성한 인스턴스로 채점 → 암기형은 0점. `fresh_pass`/`adopt`로만 검증 |
| 현재 | 실제 LB 6300.89, 현실 천장 ~6450–6650, 마감 2026-07-15 |
```
