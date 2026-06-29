# NEXT SESSION — NeuroGolf handoff

Use this as the next-session prompt:

```text
/Users/minseong/project/neurogolf 에서 NeuroGolf 점수 개선을 이어가자.

먼저 프로젝트 로컬 스킬을 읽고 따른다:
skills/neurogolf-recursive-improvement/SKILL.md

현재 상태:
- 400개 태스크는 모두 src/custom/taskNNN.py 기준으로 source-owned 상태다.
- source/live reconcile mismatch는 0이다.
- networks/*.onnx 는 gitignore 된 로컬 build/deploy artifact다. 없거나 오래됐으면 `PYTHONPATH=. .venv/bin/python reports/scripts/rebuild_networks_from_source.py` 로 source에서 재생성한다.
- local manifest total은 약 7178.33이고 목표는 7500+다.
- public_candidates/는 참고용 teacher artifact이며 gitignore 처리되어 있다.
- public ONNX blind import 금지. public에서 얻은 것은 source-owned 코드나 insight_registry로 흡수한다.
- Kaggle 제출은 100/day라 실험 제출을 너무 아낄 필요는 없지만, 항상 재현 가능한 후보만 제출한다.

처음 할 일:
1. reports/ACTIVE_RESEARCH_STATE.md 를 읽는다.
2. reports/HIGH_SCORE_FRONTIER.md 를 읽는다.
3. reports/source_live_reconcile.md 를 확인한다.
4. reports/insight_registry.yaml 과 reports/recursive_queue.md 를 읽는다.
5. 아래 두 스크립트로 글로벌 관점을 갱신한다:
   PYTHONPATH=. .venv/bin/python reports/scripts/build_layer_inventory.py
   PYTHONPATH=. .venv/bin/python reports/scripts/find_insight_candidates.py

이번 세션의 우선순위:
1. 20+ 점수 태스크 전체를 op/mechanism/cost/semantic precondition 별로 catalog화한다.
2. 15~18점 태스크 중 high-score mechanism을 이식할 수 있는 top 후보를 뽑는다.
3. 한 후보를 source-owned semantic rewrite로 구현한다.
4. stored + fresh 검증 후 adopt 여부를 판단한다.
5. 성공/실패 인사이트를 tasklog와 insight_registry에 기록하고 전체 400개에 재스캔한다.

하지 말 것:
- public ONNX를 그대로 복사해서 점수만 올렸다고 말하지 말 것.
- 평균 점수나 최저점 순서만 보고 wall 태스크를 오래 파지 말 것.
- 검증 없이 score improvement라고 말하지 말 것.
- source/live parity를 깨뜨린 채 방치하지 말 것.
- 오래된 레거시 문서의 지시를 우선하지 말 것. ACTIVE_RESEARCH_STATE와 로컬 skill을 우선한다.
```
