# public-insight — 공개 승리를 메커니즘으로 역공학·일반화 (딥 레인)

> **user 지시(2026-07-09): 공개가 태스크별로 우리를 이기면 onnx 채택에서 멈추지 않는다.**
> "퍼블릭은 나보다 TOTAL 점수는 낮지만, 나보다 좋은 태스크는 있다." 그 태스크의 메커니즘을 역공학·재현·등록하고
> 같은 지문을 가진 우리의 다른 태스크에 전수 적용한다.
> 레버: `public-insight-generalize`(scanner `public_autopsy`, agent=fable).
> 빠른 바이트-채택 레인은 `minmerge.md`; 이 파일은 그 위의 **깊은 레인**.

## 언제 쓰나(스캐너/시그널)
- min-merge 승리가 발생할 때마다(항상 짝으로). 공개 노트북/discussion이 새 설계 의도를 드러낼 때.
- 실적: borrowed net = +1 task; generalized 메커니즘 = +N(`free_input_einsum_substitution` +1.72/3태스크, task011 +1.52).

## 물리(왜 되나)
- 공개 넷이 어떤 태스크를 우리보다 싸게 표현했다면, 그 차이는 **재사용 가능한 표현 메커니즘**(op-delta / lost-tensor 지문)이다.
- 같은 지문을 가진 우리의 다른 넷들(`rescan_candidates`)에 그 메커니즘을 적용하면 하나의 공개 승리가 +N으로 팬아웃된다.
- 빌린 onnx를 블랙박스로 남기면 재현·일반화가 불가능 → 반드시 소스로 재현한다.

## 레시피(단계 — 4-step 딥 레인)
1. **역공학.** min-merge 승리마다 `uv run ng scan public_autopsy` → op-delta / lost-tensor 지문 추출
   (base = `.minmerge_backup/` 또는 `submission/.backups/`, winner = 배포 넷). 출력에 `learned_lost_fingerprints`,
   `fingerprint_rescan_candidates`. **공개 노트북/discussion 원문도 읽어** 설계 의도 파악
   (`kaggle kernels pull <slug>` / `kaggle kernels output <slug>`로 소스 노트북 확보).
2. **소스 재현.** 채택한 공개 onnx를 그대로 두지 말고 `src/custom/taskNNN.py`를 그 메커니즘으로 재작성 —
   `PYTHONPATH=. uv run python -m src.pipeline --tasks NNN`으로 재생성 가능해야 완료(빌린 넷 = 재현 가능 소스, 블랙박스 금지).
3. **인사이트 등록.** `state/insights.yaml`에 메커니즘 항목 추가: 지문(op-delta) / 적용 조건(`applies_when`) /
   거부 조건(`reject_when`) / 실적(`source_tasks`, 측정 gain).
4. **일반화 적용.** autopsy의 `rescan_candidates`(같은 지문의 우리 넷)에 에이전트 팬아웃 →
   각 후보 `uv run ng gate cand.onnx --task NNN` → 통과 시 `uv run ng adopt … --note "public-insight: <mechanism>"`.
   cristianoc 참조(동일 400 태스크·룰 오라클) 활용 가능 — **룰만 복사, golf 상수는 절대 복사 금지**.

## 서브패턴
- **teacher-derived 메커니즘(추출 후 광역 적용):**
  - `public_teacher_bitwise_scan_replacement`(task2/209): 반복 MaxPool/Max/Min scan stack을 BitShift/BitwiseAnd/Or/Gather·QLinearConv로(binary state일 때, +0.46~0.50).
  - `public_teacher_qlinear_conv_rewrite`(task23/349/182/233/255/364/338/4): fp32/fp16 Conv-heavy shape 로직을 QLinearConv/uint8로(task023 +0.533, task349 +0.398).
  - `gridsample_warp_render`(task209), `qlinearconv_signed_renderer`: stamp-and-compare tower → QLinearConv/ConvInteger signed uint8 renderer.
- **fingerprint 클러스터(724950 autopsy):** `free_input_einsum_substitution`(+1.720/3승), `index_or_topk_plane_removed`(+0.939 task022), `final_equal_or_output_only`(+0.595/4승). exact-fingerprint 후보 **task019/044/076**이 task011과 동일한 lost `Pad:2:1x1x30x30:900B` 지문 공유 → 재공략 대상.
- **residual_spatialop_to_free_einsum_collapse:** 잔여 spatial op(Conv/GridSample/TopK/MaxPool/wide index 평면)을 free-input Einsum 수축으로 붕괴(→ 상세는 `free-output-einsum.md`와 짝).

## 함정/거부 사례
- **🚨 golf 상수 복사 = private LB 누출.** cristianoc/공개 넷에서 **룰(알고리즘)만** 복사하고 golf 상수(fitted 폭·임계·룩업)는 절대 복사 금지 — 마감 07-15 private LB(동일 고정 데이터셋)에서 fresh-gate가 유일한 moat.
- **공개 TOTAL이 낮아도 per-task 승리는 존재한다(user 2026-07-09).** "공개가 나보다 점수 낮으니 볼 것 없다"는 잘못 — 공개가 특정 태스크에서 우리를 이기는 지점을 autopsy로 찾아 일반화하라.
- **borrowed net을 블랙박스로 남기지 말 것:** 채택 후 반드시 `src/custom/taskNNN.py` 소스 재현(step 2). 재현 없이 채택 보고 금지.
- **0-win falsification hygiene:** autopsy가 "500B+ exact lost-tensor 지문 없음"을 내도 소진 아님 — 새 공개 tail win마다 재실행. 부정 판정은 4-field ledger(date/ran/verdict/reopen)로만 기록.
- **재현 불가 판정(task133/285):** per-object scale-vector(133) / sparse reflection + TopK enumeration(285)는 단일 free-input Einsum 붕괴가 직접 적용 안 됨 — 공개가 strictly 싸거나 새 lowering이 나올 때 reopen.
