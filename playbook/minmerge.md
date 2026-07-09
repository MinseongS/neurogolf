# minmerge — 공개 ONNX min-merge (가장 빠른 레인: 바이트 채택만)

> 공개 400-net 덤프를 grader-measure, per-task로 배포본보다 싸고 bundled fail=0이면 그 onnx 바이트를 채택.
> **채점 데이터셋은 상수·rescore 없음 → bundled fail=0은 영구 통과.** public LB = local bundled 1:1.
> 레버: `public-minmerge`(CLI `ng mine-public`, agent=opus). 새 프론티어당 +0.3~4.5 LB.

## 언제 쓰나(스캐너/시그널)
- 공개 프론티어가 현재 채굴선(우리 LB) 위로 갱신될 때. 신뢰 업로더: **franksunp / prvsiyan / urad / lucifer / boristown**.
- 새 업로더 / 새 추출 ONNX 아티팩트도 트리거. margin-0 pass는 상시 가치(순수 permanent upside).
- 이것은 **빌린 넷을 그대로 채택하는 fast 레인** — 메커니즘 일반화는 `public-insight.md`.

## 물리(왜 되나)
- grader mem = ORT 프로파일러 trace 바이트; 공개 넷이 같은 태스크를 더 싸게 표현하면 그 바이트로 교체 시 우리 total이 하락.
- min-merge = per-task로 (배포본, 공개넷들) 중 bundled fail=0 & 최소비용 선택. margin-0이면 byte-level optimizer tail까지 흡수(예: 65 overlay = +0.216 local).

## 레시피(단계)
1. 공개 덤프 pull: `kaggle kernels output <slug> -p <dir>` (또는 `kaggle competitions submissions`/노트북 output). 400개 onnx 추출.
2. `uv run ng mine-public <dir1> <dir2> … --margin 0` — 각 태스크에 대해 배포본 대비 bundled fail=0 & strictly cheaper인 공개넷 식별.
3. `--apply`로 채택(또는 후보 확인 후 개별 `ng gate`/`ng adopt`). 교체된 넷은 backup(`.minmerge_backup/`).
4. `uv run ng pack` → `scan_unsigned_topk`로 400개 clean 확인 → `uv run ng submit -m "…"`.
5. **서브밋 전 항상 `kaggle competitions submissions -c neurogolf-2026`로 최신 best 재확인**(병렬 세션이 공유 디렉토리 갱신).

## 서브패턴
- **margin-0 byte tails:** default margin은 0승이어도 `--margin 0`이 optimizer byte tail을 노출(task291 +0.03, task189 +0.019 등). 새 프론티어마다 margin-0 pass.
- **min-merge ↔ autopsy 짝:** 새 min-merge 승리마다 `public-insight.md`의 autopsy를 짝으로 실행(왜 이겼는지 지문화 → 우리 다른 태스크로 일반화).
- **dynamic_bundled_cse_rewire:** bundled runtime 시그니처로 중복 active overlay 중간텐서 제거(byte tail 계열).

## 함정/거부 사례
- **Kaggle-falsified despite bundled fail=0:** task101 dynamic-CSE(13725→13721)는 bundled fail=0인데도 서브밋 시 점수를 7248로 crash시킴(자기 기여분 15.473 전손). bundled fail=0이 **항상** LB-safe는 아님 — 의심 넷은 격리 서브밋으로 확인.
- **0-win = 소진 아님:** "이 run에서 이 active set 위 더 싼 overlay 없음"일 뿐. public-tail "dry" 판정은 margin-0 remine·새 업로더로 반복 반증됨 → dormant, not closed. reopen = 프론티어 상승 / 새 업로더 / 새 추출 아티팩트.
- **동시성:** `submission/overfit_nets/`와 candidate 디렉토리를 병렬 세션과 공유 → 채택 직전 재측정, 서브밋 전 backup 확인.
- **isolated eval:** 장수 ORT 프로세스가 특정 태스크(예: task075/376)에서 stall → `--isolated-all` 매니페스트로 400/400 확인.
