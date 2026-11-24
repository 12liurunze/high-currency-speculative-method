# Aggregate results_*.jsonl: per rep, sum tokens / sum time over all categories; then mean +- std over reps.
# Usage: python summarize.py results_eagle.jsonl [--by-cat]
import json, sys, collections, statistics as st
files = [f for f in sys.argv[1:] if not f.startswith("--")]; by_cat = "--by-cat" in sys.argv
rows = [json.loads(l) for f in files for l in open(f) if l.strip()]
agg = collections.defaultdict(lambda: [0, 0.0, []])  # (bs, tag, cat|None, rep) -> tokens, secs, accept_lens
for r in rows:
    if "error" in r: print("ERROR", r); continue
    for cat in ([r["category"]] if by_cat else [None]):
        a = agg[(r["bs"], r["tag"], cat, r.get("rep", 1))]
        a[0] += r["out_tokens"]; a[1] += r["e2e_s"]
        if r["accept_len"]: a[2].append(r["accept_len"])
reps = collections.defaultdict(list)  # (bs, tag, cat) -> [(thr, accept)]
for (bs, tag, cat, rep), (n, t, al) in agg.items():
    reps[(bs, tag, cat)].append((n / t, st.mean(al) if al else 1.0))
def ms(x): return (st.mean(x), st.stdev(x) if len(x) > 1 else 0.0)
tab = {k: (ms([v[0] for v in vs]), st.mean(v[1] for v in vs), len(vs)) for k, vs in reps.items()}
for bs in sorted({k[0] for k in tab}):
    print(f"\n== bs={bs}  (throughput tok/s mean +- std | accept_len | n_reps)")
    for (b, t, c), ((m, s), al, n) in sorted(tab.items(), key=lambda x: (str(x[0][2]), x[0][1])):
        if b != bs: continue
        note = ""
        ar = tab.get((bs, "ar", c))
        if ar and t != "ar": note += f"  x{m / ar[0][0]:.2f} vs AR"
        if t.startswith("tree"):
            ch = tab.get((bs, t.replace("tree", "chain"), c))
            if ch: note += f"  x{m / ch[0][0]:.2f} vs chain"
        print(f"{(c or 'ALL'):15s} {t:10s} {m:9.1f} +- {s:6.1f} {al:6.2f} {n:3d}{note}")
