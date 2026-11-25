# Summarize run_accept.sh outputs: tokens per verify step = (out_tokens - 1) / verify_ct (the first output token
# comes from prefill), overall and per category; greedy losslessness check = output text identical to the chain
# run with the same steps.
# Usage: python summarize_accept.py accept/q8e3_*
import collections, json, os, re, sys

runs = {}
for d in sys.argv[1:]:
    f = os.path.join(d, "client.jsonl")
    if os.path.exists(f):
        runs[os.path.basename(d.rstrip("/"))] = {r["tag"]: r for r in map(json.loads, open(f))}


def tps(rows):
    return sum(r["out_tokens"] - 1 for r in rows) / max(sum(r["verify_ct"] for r in rows), 1)


cats = sorted({t.split(":")[0] for rs in runs.values() for t in rs})
print(f"{'run':24s} {'n':>4s} {'ALL':>6s} " + " ".join(f"{c[:8]:>8s}" for c in cats) + "  same_text_as_chain")
for name, rs in sorted(runs.items(), key=lambda kv: (re.sub(r".*_s(\d+)_.*", r"\1", kv[0]), kv[0])):
    s = re.search(r"_s(\d+)_", name).group(1)
    chain = runs.get(next((n for n in runs if n.endswith(f"chain_s{s}_k1")), ""), {})
    common = [t for t in rs if t in chain]
    same = f"{sum(rs[t]['text'] == chain[t]['text'] for t in common)}/{len(common)}" if common else "-"
    per = [tps([r for t, r in rs.items() if t.startswith(c + ":")]) for c in cats]
    print(f"{name:24s} {len(rs):4d} {tps(list(rs.values())):6.3f} " + " ".join(f"{v:8.3f}" for v in per) + f"  {same}")
