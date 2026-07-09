# playbook — 메커니즘 레시피 색인

per-task 에이전트가 점수 작업 중 읽는 메커니즘 레시피. 각 파일 구조 고정:
`## 언제 쓰나(스캐너/시그널)` · `## 물리(왜 되나)` · `## 레시피(단계)` · `## 서브패턴` · `## 함정/거부 사례`.
레버 대장은 `state/levers.yaml`, 인사이트 상세는 `state/insights.yaml`. 툴링은 `ng scan/gate/adopt/mine-public`.

## 파일별 1줄 + 어느 레버가 쓰나

| 파일 | 1줄 요약 | 쓰는 레버(state/levers.yaml) |
|---|---|---|
| [free-output-einsum.md](free-output-einsum.md) | 900B output-mask floor을 free-output N-ary Einsum으로 붕괴 (프로젝트 최대 레버, 16/18 crack) | `free-output-einsum-regime-crack` |
| [walk-einsum.md](walk-einsum.md) | 반복 flood/scan을 하나의 multi-operand Einsum으로 (op 내부 무료) | `walk-chain-slack`; 인사이트 `walk_einsum_iteration_collapse` |
| [signed-einsum-routing.md](signed-einsum-routing.md) | separable fill/overlap priority를 signed-weight Einsum으로 (`out>0` 디코드) | 인사이트 `signed_rect_priority_overlay`(레거시 mech 15) |
| [fp16-recast.md](fp16-recast.md) | output-coupled 평면을 fp16/uint8로 재캐스트 (bit-identical golf) | `deployed-fp16-recast`, `dtype-overpay` |
| [kernel-collapse.md](kernel-collapse.md) | Conv 커널을 최소형(1×1+pad)으로 붕괴 (params 레인, bit-identical) | `kernel-collapse`, `topk-width-refit`, `reducesum-spatial-einsum` |
| [minmerge.md](minmerge.md) | 공개 ONNX min-merge — 배포본보다 싼 bundled fail=0 넷을 바이트 채택 (빠른 레인) | `public-minmerge` |
| [public-insight.md](public-insight.md) | 공개 승리를 메커니즘으로 역공학·재현·등록·팬아웃 (딥 레인) | `public-insight-generalize` |

## 레거시 메커니즘 → 파일 매핑

`_LEGACY_PLAYBOOK.md`(git `pre-redesign` 태그) 툴박스 16종 + 인사이트 46종의 귀속:
- 툴박스 1–6,9,13a(walk/free-input/S=1.0/checkpoint/gates/exact-count/batched-K/epilogue-fold) → **walk-einsum.md**
- 툴박스 7,8,14 + self-Einsum/free-final-onehot/direct-gather/label-pad-order/strided-conv/GridSample/moment-stats/band-profile/solid-marker/separable-LUT/sparse-edit → **free-output-einsum.md** (batch1/2/3 서브레시피 포함)
- 툴박스 15 + parity-paint(286)/threshold-linearize(1)/uint8-presence(234) → **signed-einsum-routing.md**
- 툴박스 10,11(→ 054 sparse-edit는 free-output),16(370 stamp-kernel) + dihedral(191)/runtime-anchor/sparse-conv/pad-crop/reducesum-einsum → **kernel-collapse.md**
- dtype 계열(qlinear-LUT/scan-dtype/uint8-topk/int8-topk/argmax-uint8/dedupe-init/output-coupled-fp16) → **fp16-recast.md**
- public 계열(teacher-bitwise/teacher-qlinear/autopsy fingerprint/residual-spatialop-collapse/dynamic-CSE) → **public-insight.md** + **minmerge.md**

## 미분류 (원문: git `pre-redesign` 태그의 semantic-compiler / research 인사이트)

아래는 7개 메커니즘 파일 중 하나로 깔끔히 접히지 않는 **미로워링(unlowered) semantic compiler / 메타-전략** 인사이트다.
값싼 ONNX lowering이 없어 레시피화 불가 — 대상 태스크 재정식화 시 `state/insights.yaml`에서 직접 참조:
- `high_score_frontier_final_output_only` — 20+ 프론티어는 모든 full-canvas 중간텐서를 output-only로 (메타 우선순위 규칙).
- `exact_preserve_to_semantic_rewrite` — exact-preserve 빌더는 메커니즘 추출 1순위 타깃 (메타 규칙).
- `marker_routed_hidden_path_compiler`(task066) — endpoint-pair hidden path를 marker 픽셀에서 컴파일.
- `rotation_component_template_scatter`(task233) — rotation-only 3×3 component template scatter.
- `bounded_exact_cover_without_mode_background`(task366) — sparse object 위 bounded exact-cover (mode-colour 배경 가정 폐기).

이들은 drop이 아니라 **보류**: 값싼 lowering 아이디어가 나오면 해당 파일의 서브패턴으로 승격한다.
