import sys, json
from src.harness import evaluate, load_task
t=int(sys.argv[1])
r=evaluate(f"public_candidates/lucifer_forge/nets/task{t:03d}.onnx", load_task(t), keep_failures=False)
print(json.dumps({"task":t,"pts":r.get("points") or 0.0,"pass":r.get("pass"),"fail":r.get("fail"),"mem":r.get("memory"),"par":r.get("params"),"err":r.get("error")}))
