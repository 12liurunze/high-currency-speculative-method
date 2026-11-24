# Per (category, bs): send the whole category `copies` times with client-side concurrency bs, greedy;
# a finished request is immediately replaced by the next.
# copies=auto (default) = 3 * ceil(bs / n_category): 3 full copies when bs <= n, doubled as needed for larger bs
# so that there are always >= 3 waves of bs requests (e.g. n=80: bs32/64 -> 240 req, bs128 -> 480, bs512 -> 1680).
import argparse, json, math, time, requests
from concurrent.futures import ThreadPoolExecutor
from transformers import AutoTokenizer

p = argparse.ArgumentParser()
p.add_argument("--port", type=int, default=30000)
p.add_argument("--model", required=True)
p.add_argument("--tag", required=True)
p.add_argument("--bs", default="32,64,128,256,512")
p.add_argument("--copies", default="auto", help='times each category is sent per bs: "auto" = 3*ceil(bs/n) (default) or an integer')
p.add_argument("--resume", action="store_true", help="skip (tag, rep, bs, category) already in --out without error")
p.add_argument("--max-new", type=int, default=256)
p.add_argument("--data", required=True)
p.add_argument("--out", required=True)
p.add_argument("--rep", type=int, default=1)
p.add_argument("--server-log", default=None, help="server log path, recorded in each result row")
p.add_argument("--chat-kwargs", default="{}", help='extra apply_chat_template kwargs, e.g. {"enable_thinking": false}')
a = p.parse_args()

tok = AutoTokenizer.from_pretrained(a.model, trust_remote_code=True)
rows = [json.loads(l) for l in open(a.data)]
cats = sorted({r["category"] for r in rows})
url = f"http://127.0.0.1:{a.port}/generate"

chat_kw = json.loads(a.chat_kwargs)

def fmt(q):
    return tok.apply_chat_template([{"role": "user", "content": q}], tokenize=False, add_generation_prompt=True, **chat_kw)

def gen(text):
    # a rejected request (e.g. prompt + max_new > context) returns None instead of aborting the whole category
    try:
        r = requests.post(url, json={"text": text, "sampling_params": {"temperature": 0, "max_new_tokens": a.max_new}}, timeout=3600)
        r.raise_for_status()
        return r.json()
    except requests.RequestException:
        return None

def run(texts, conc):
    # keep `conc` requests in flight; a finished one is immediately replaced by the next
    t0 = time.perf_counter()
    with ThreadPoolExecutor(conc) as ex:
        res = list(ex.map(gen, texts))
    return res, time.perf_counter() - t0

# KV capacity / running-request cap (to flag bs that cannot actually run fully concurrent)
try:
    info = requests.get(f"http://127.0.0.1:{a.port}/get_server_info", timeout=30).json()
    st0 = (info.get("internal_states") or [{}])[0]
    cap = info.get("max_total_num_tokens") or st0.get("max_total_num_tokens")
    srv_max_run = st0.get("max_running_requests") or info.get("max_running_requests")
except Exception:
    info, cap, srv_max_run = {}, None, None
spec = {k: info.get(k) for k in ("speculative_algorithm", "speculative_num_steps", "speculative_eagle_topk", "speculative_num_draft_tokens", "kv_cache_dtype", "tp_size")}
print("server config:", spec, "max_total_num_tokens:", cap, "max_running_requests:", srv_max_run, flush=True)

done = set()
if a.resume:
    try:
        for l in open(a.out):
            r = json.loads(l)
            if "error" not in r and "bs" in r: done.add((r["tag"], r.get("rep", 1), r["bs"], r["category"]))
    except FileNotFoundError:
        pass

# warmup
run([fmt(rows[0]["prompt"])] * 8, 8)
for bs in [int(x) for x in a.bs.split(",")]:
    for c in cats:
        if (a.tag, a.rep, bs, c) in done:
            print(f"resume: skip {a.tag} rep{a.rep} bs={bs} {c}", flush=True); continue
        qs = [fmt(r["prompt"]) for r in rows if r["category"] == c]
        texts = qs * (3 * math.ceil(bs / len(qs)) if a.copies == "auto" else int(a.copies))
        lens = sorted((len(tok(t).input_ids) for t in texts), reverse=True)
        # worst case in flight: the bs longest prompts sent, each at max_new
        need = sum(lens[:bs]) + bs * a.max_new
        t0 = time.time()
        try:
            res, dt = run(texts, bs)
        except Exception as e:
            rec = {"tag": a.tag, "rep": a.rep, "bs": bs, "category": c, "error": str(e)[:300]}
        else:
            nfail = sum(x is None for x in res); res = [x for x in res if x is not None]
            if not res: raise SystemExit(f"all requests failed for {c} bs={bs} (server down?)")
            ntok = sum(x["meta_info"]["completion_tokens"] for x in res)
            nver = sum(x["meta_info"].get("spec_verify_ct", 0) for x in res)
            rec = {"tag": a.tag, "rep": a.rep, "bs": bs, "category": c, "n_req": len(texts), "copies": len(texts) // len(qs), "n_fail": nfail, "e2e_s": round(dt, 3),
                   "out_tokens": ntok, "throughput_tok_s": round(ntok / dt, 1),
                   # prefix-cache hits; must stay 0 with --disable-radix-cache (repeated prompts must not skip prefill)
                   "cached_tokens": sum(x["meta_info"].get("cached_tokens") or 0 for x in res),
                   "accept_len": round(ntok / nver, 3) if nver else None,
                   "kv_need": need, "kv_cap": cap, "kv_ok": (cap is None or need <= cap), "srv_max_running": srv_max_run,
                   "t0": round(t0, 3), "t1": round(time.time(), 3), "server_log": a.server_log, **spec}
        print(json.dumps(rec), flush=True)
        with open(a.out, "a") as f:
            f.write(json.dumps(rec) + "\n")
