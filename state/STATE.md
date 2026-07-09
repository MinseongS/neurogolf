# STATE — NeuroGolf live handoff (갱신: 2026-07-09, 092 딥팬아웃 + task265 확정)
> 이 파일은 append 금지. 세션 종료 시 "현재 참인 것"만 남기고 교체한다. 히스토리는 git + state/submissions.md.

## 확정 상태
- BEST LB: **7296.67 확정** (sub 54487441, task265 free_final_onehot tail).
- 현재 로컬 manifest: **7296.5495** (400/400).
- 신규 제출: **sub 54487441 COMPLETE, publicScore 7296.67** — task265 4357->4069 (+0.069)을 병렬 dtype-tail 배치 위에 스택.
- 직전 완료 제출: sub 54487028 (dtype-tail batch) 7296.60 / sub 54484750 7296.27.
- 로컬==LB 오프셋 +0.12 4회 연속 정확 일치 (7296.5495+0.12≈7296.67).
- 마감: 2026-07-15 (private = 동일 고정 데이터셋; bundled fail=0 = 영구 통과).

## 이번 세션에서 바뀐 것
1. **dtype-tail batch 채택/제출**
   - `task141`: output-coupled scalar/Einsum tail fp16 recast, cost **568 -> 454**, +0.2240 local.
     - 후보: `candidates/task141/task141_fp16_tail_scalar_chain.onnx`.
     - mechanism: final free-output Einsum이 graph input을 직접 먹지 않는 regime graph라 `g/V/A/B` tail을 fp16으로 내릴 수 있음.
   - `task197`: fp32 `Mul/Where` source-index tail을 `Cast(am->bool) + Where(fc_u8, 0_u8)` island로 교체, cost **904 -> 852**, +0.0592 local.
     - 후보: `candidates/task197/task197_u8_where_tail.onnx`.
     - `Sub_u8`/`Mul_u8`는 ORT invalid였고, `Where_u8` 형태만 통과.
   - `task064`: row/col presence profile count tail을 fp16으로 재캐스트, cost **5997 -> 5749**, +0.0422 local.
     - 후보: `candidates/task064/task064_fp16_profile_count_tail.onnx`.
     - `ReduceSum_u8`는 invalid, `ReduceSum_fp16`은 pass.
   - `uv run ng pack` 통과(unsigned TopK clean). `uv run ng submit` 완료, ref **54487028**, publicScore **7296.60**.
2. **deployed dtype_overpay 재스캔**
   - `uv run ng scan dtype_overpay` 완료, `candidates/worklists/dtype_overpay.json` 갱신.
   - 상위 false-positive 다수 확인:
     - `070/199/063/092/190/383/051/246/033/295/394/375`: final `Einsum`이 free fp32 `input`을 직접 먹어서 operand만 fp16/u8로 낮출 수 없음.
     - `084/126/181`: Scatter(Elements/ND) update가 data=`input` dtype에 묶임.
     - `104/310/400/351/244/039/263/242`: final Pad/Conv의 입력이 `Slice(input)`/`GridSample(input)`이라 cast는 원본 fp32 plane을 추가할 뿐.
     - `377`: residual internal fp16 시도는 `CumSum(float16)` invalid. 기존 output-coupled fp16 채택 이후 남은 tail은 custom rewrite 필요.
   - 남은 possible custom-surgery 후보: `202`, `255` (cast-balanced u8/fp16 islands; direct recast 아님).
3. **092-style profile-product cohort — 11 opus 딥 팬아웃(태스크당 전담 에이전트) 확정**
   - 결과: **1 WIN(task265 +0.069, 채택/제출) / 10 NO-WIN**. read-only 재점검의 "dry" 판정을 rigorous per-task로 확증.
   - task265: `free_final_onehot` = `Where(mask, color_onehot, input)`가 label->color30->Equal backend 전체 삭제
     (output=input except one overlay class인 태스크). +anchor2 doubling plane 제거. cost 4357->4069, fresh 0/2000.
     ⭐TRANSFERABLE: output tail이 `Pad(label)->Equal`인 배포넷 전수 스캔 가치.
   - 10 NO-WIN(076/101/118/133/206/233/285/330/342/054) 각각 `state/tasks/taskNNN.md`에 4-field 재개 레저 기록.
     지배 원인 3분류: (a) positioned-content/stamp(076/101/206/233/285/342 — data-dependent 위치의 content는
     단일 contraction으로 fold 불가), (b) non-linear 판정(118 phantom-NMS / 330 connectivity-count — einsum 표현 불가),
     (c) fp32 spatial DETECTION floor(054/133 — cost가 carrier가 아니라 detection에 있음).
   - **task054는 룰 완전 크랙**(`candidates/task054/oracle4.py` 266+3000 fresh 검증) — 배포넷이 이미 optimal이라 NO-WIN이나,
     재사용 primitive 확보: 2-level(col-then-row) cumsum segmentation = Loop/Scan 없이 overlapping 축정렬 rect 재구성.
   - **후속 2건도 NO-WIN 확정**: task342 scatter-shrink([16,4]→[4,4])는 MIRAGE(free-input bg canvas가 counted 18-36kB);
     task054 mega-einsum(+4~5 기대)은 fp32 detection floor로 26k~37k(>20131).
4. **canvas_crop_shrink 검증**
   - 10k full scan은 task018 generator에서 느려 중단.
   - smoke top hits `014/350/018` 직접 확인: 모두 final 30x30 output shape에 묶인 scalar/Pad carrier false-positive.
   - 단순 value_info crop은 shape inference에서 final `Equal/Where` output mismatch로 reject. scanner filter 보강 전에는 이 레인 우선순위 낮음.
5. **공개 프론티어 감시**
   - 20260709 dumps margin-0 재실행 결과 추가승 0.

## 활성 베인 (기대이득 순)
1. **dtype_overpay custom surgery**
   - `task202`: `den_r1/den_c1` direct u8는 Div 때문에 false-positive. 대신 `A/Am/B/Bm -> Equal` 주변의 cast-balanced island 검토.
   - `task255`: 30-wide post-Sub/Where tensors가 u8-range이나 입력-bound fp32 Einsum과 얽힘. island 단위로만 가능.
   - `task377`: `vs/d/ad/t/cum/rix` residual은 `CumSum_fp16` invalid. shift-Einsum을 narrowed gather 등으로 바꾸는 구조적 rewrite가 필요.
2. **⭐ mixed-dtype Einsum escape (CROSS-CUTTING, 최고 잠재이득) — ORT-capability 프로브**
   - 5+ 에이전트가 독립적으로 지목한 단일 최대 잔여 레버: ORT 1.26에서 free fp32 `input`을 operand로 먹으면서
     output/carrier는 **fp16**로 내리는 경로가 존재하는가? (현재 uniform-T rule로 차단 가정).
   - 풀리면 076/101/118/133/233/285/054 등 Conv/detection fp32-co-bind floor(~6700B/넷)가 fp16으로 반토막
     (233 ~+0.4, 054 ~9k까지 등). **태스크 공격이 아니라 ORT 연산 조합 프로브로 진행** (Einsum/Cast/QLinearConv/
     ConvInteger의 dtype 승격 규칙을 1.26에서 실측; 되면 arsenal 전체에 배치).
   - ⭐HEURISTIC(054에서 도출): free-output/signed einsum fold는 counted mass가 avoidable label/priority CARRIER일 때만
     이득. fp32 spatial DETECTION이면 fold가 더 나쁨 — 시도 전 detection-vs-carrier byte split을 측정하라.
3. **인사이트-무기고 팬아웃 계속**
   - 092-style 고비용 cohort는 이번 세션에서 딥 팬아웃으로 소진 확인했으니 같은 방식 반복 금지.
   - 다음 우선순위: probe_then_build 138/145/328/324/379, axis-code 대형 367, stamp/crop 재개방 floors 062/012/112/099/163/102는 새 primitive가 있을 때만.
3. **공개 재채굴**
   - 새 프론티어/새 업로더/새 notebook output 등장 시 `uv run ng mine-public --margin 0 ...` 즉시 재실행.
   - min-merge 채택이 나오면 반드시 public-insight-generalize deep lane으로 pairing.
4. **canvas_crop_shrink tooling**
   - full scan보다 먼저 false-positive filter 필요: final `Equal/Where`가 declared output 30x30을 강제하고 onehot/Pad 대체가 더 비싼 케이스 제외.
   - ⚠️ 성능도 블로커: 10k·1500 샘플 둘 다 ~10/400에서 실질 정지(태스크마다 generator 로드+ONNX+shape-infer). scanner를
     고비용 태스크 subset + 소샘플(≤300)로 재작성하거나 generator 소스 정적분석으로 extent 증명하는 방향 필요.

## 하지 말 것 / 불변식
- cost-1 하위 10넷 재공격 금지(플로어 증명 완료).
- 350/148/233/042는 reopen trigger 없이 재공격 금지.
- 014/350/018 canvas_crop_shrink 단순 value_info crop 재시도 금지(이번 세션 shape wall 확인).
- 092-style cohort(054/076/101/118/133/206/233/285/330/342) 딥 팬아웃 완료 — 반복 금지(각 tasklog 재개 레저 참조).
- task342 scatter-shrink / task054 mega-einsum 재시도 금지(둘 다 이번에 측정 확정 NO-WIN, tasklog 참조).
- submission.zip 이름 고정 / 100회일 제한 / unsigned TopK 금지.
- 비교 기준 = `submission/overfit_nets/` 배포본.
- 환경 PIN: pyproject.toml의 `onnx==1.21.0` + `onnxruntime==1.26.0` 고정. 400/400 재검증 없이 업그레이드 금지.

## 다음 세션 시작 절차
1. `uv run ng status`
2. `kaggle competitions submissions -c neurogolf-2026 | head -5`로 새 병렬 제출 여부 확인
3. 이 파일 확인
4. dtype_overpay custom surgery: `202`, `255`부터 구조 inspect -> candidate -> `ng gate`
5. (신규 최우선) mixed-dtype Einsum escape ORT-capability 프로브 — 활성 베인 #2 참조
6. 새 공개 덤프/업로더가 있으면 `uv run ng mine-public --margin 0 ...`
