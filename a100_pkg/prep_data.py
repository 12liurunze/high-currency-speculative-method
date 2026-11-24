# Build RedHatAI/speculator_benchmarks subsets (one category per file, all rows kept) -> specbench.jsonl
import json, sys, urllib.request
REPO = "datasets/RedHatAI/speculator_benchmarks/resolve/main"
HOSTS = ["https://hf-mirror.com", "https://huggingface.co"]
# category -> file (writing.jsonl is byte-identical to question.jsonl, tool_call not in README: both skipped)
SUBSETS = {"humaneval": "HumanEval.jsonl", "math_reasoning": "math_reasoning.jsonl", "qa": "qa.jsonl",
           "mt_bench": "question.jsonl", "rag": "rag.jsonl", "summarization": "summarization.jsonl",
           "translation": "translation.jsonl"}

def fetch(name):
    for h in HOSTS:
        try:
            txt = urllib.request.urlopen(f"{h}/{REPO}/{name}", timeout=60).read().decode()
            return [json.loads(l) for l in txt.splitlines() if l.strip()]
        except Exception as e:
            print("fail", h, name, e, file=sys.stderr)
    sys.exit(f"no source reachable for {name}")

out = []
for cat, name in SUBSETS.items():
    rows = fetch(name)
    out += [{"category": cat, "prompt": r["prompt"]} for r in rows]
    print(cat, len(rows))
with open(sys.argv[1] if len(sys.argv) > 1 else "specbench.jsonl", "w") as f:
    for r in out:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")
