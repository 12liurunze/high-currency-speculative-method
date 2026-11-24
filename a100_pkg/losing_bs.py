# Print bs (comma list) where tree_N<N> does not beat chain_N<N>, among bs in argv[3].
# Per rep: sum tokens / sum time over the compared categories; compare the mean over reps.
# Compared categories at a bs = those present (no error) for chain and tree in every rep.
# A bs with no comparable category (server died, tree unsupported) cannot be judged:
# it is reported on stderr and NOT returned, because a smaller budget would not fix it.
import json, sys, collections, statistics as st
out, N, cand = sys.argv[1], sys.argv[2], [int(x) for x in sys.argv[3].split(",") if x]
rows = [r for r in map(json.loads, open(out)) if "error" not in r and "bs" in r and r["tag"].split("_")[1:] == [f"N{N}"]]
cells = collections.defaultdict(set)  # bs -> {(kind, rep, cat)}
for r in rows: cells[r["bs"]].add((r["tag"].split("_")[0], r.get("rep", 1), r["category"]))

def common(bs):
    reps = {rep for _, rep, _ in cells[bs]}
    need = {(k, rep) for k in ("chain", "tree") for rep in reps}
    return {c for _, _, c in cells[bs] if all((k, rep, c) in cells[bs] for k, rep in need)}

agg = collections.defaultdict(lambda: [0, 0.0])  # (bs, kind, rep) -> tokens, secs
for r in rows:
    if r["category"] in common(r["bs"]):
        a = agg[(r["bs"], r["tag"].split("_")[0], r.get("rep", 1))]; a[0] += r["out_tokens"]; a[1] += r["e2e_s"]
thr = collections.defaultdict(list)
for (bs, kind, rep), (n, t) in agg.items(): thr[(bs, kind)].append(n / t)

lose, undecided = [], []
for b in cand:
    if (b, "tree") not in thr or (b, "chain") not in thr: undecided.append(b)
    elif st.mean(thr[(b, "tree")]) <= st.mean(thr[(b, "chain")]): lose.append(b)
if undecided:
    print(f"[losing_bs] N={N}: cannot judge bs {undecided} (no category with both chain and tree data); "
          "not shrinking these, see the error rows and server logs", file=sys.stderr)
print(",".join(map(str, lose)))
