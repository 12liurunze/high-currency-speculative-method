# Send prompts one at a time (bs=1) to a dumping server; before each request write "category:index" to
# <dump-dir>/tag so the hook can label every verify step. Records per-request outputs to <dump-dir>/client.jsonl.
import argparse, json, os, time, requests
from transformers import AutoTokenizer

p = argparse.ArgumentParser()
p.add_argument("--port", type=int, required=True)
p.add_argument("--model", required=True)
p.add_argument("--data", required=True)
p.add_argument("--dump-dir", required=True)
p.add_argument("--per-cat", type=int, default=20)
p.add_argument("--max-new", type=int, default=256)
p.add_argument("--chat-kwargs", default="{}")
a = p.parse_args()

tok = AutoTokenizer.from_pretrained(a.model, trust_remote_code=True)
kw = json.loads(a.chat_kwargs)
rows = [json.loads(l) for l in open(a.data)]
cats = sorted({r["category"] for r in rows})
url = f"http://127.0.0.1:{a.port}/generate"
log = open(os.path.join(a.dump_dir, "client.jsonl"), "a")
for c in cats:
    qs = [r["prompt"] for r in rows if r["category"] == c][: a.per_cat]
    for i, q in enumerate(qs):
        with open(os.path.join(a.dump_dir, "tag"), "w") as f:
            f.write(f"{c}:{i}")
        text = tok.apply_chat_template([{"role": "user", "content": q}], tokenize=False, add_generation_prompt=True, **kw)
        t0 = time.time()
        r = requests.post(url, json={"text": text, "sampling_params": {"temperature": 0, "max_new_tokens": a.max_new}},
                          timeout=3600).json()
        m = r["meta_info"]
        rec = {"tag": f"{c}:{i}", "out_tokens": m["completion_tokens"], "verify_ct": m.get("spec_verify_ct"),
               "e2e_s": round(time.time() - t0, 3), "text": r["text"]}
        log.write(json.dumps(rec) + "\n"); log.flush()
        print(rec, flush=True)
with open(os.path.join(a.dump_dir, "tag"), "w") as f:
    f.write("done")
print("CLIENT_DONE", flush=True)
