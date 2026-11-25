# Offline replay of draft-tree policies on big-tree dumps (greedy decoding).
# Each dumped step holds the full candidate pool the drafter produced, plus the target argmax at every node,
# so the accepted length of ANY prefix-closed subtree is exact: walk from the root, following the child whose
# token equals the target's argmax at its parent, while that child is in the subtree.
# Usage: python replay.py dumps/<name> [--budgets 3,7,15] [--per-cat]
import argparse, collections, glob, json, math, os

p = argparse.ArgumentParser()
p.add_argument("dump_dir")
p.add_argument("--budgets", default="3,7,15", help="draft-node budgets B (SGLang num_draft_tokens = B + 1)")
p.add_argument("--per-cat", action="store_true")
a = p.parse_args()
budgets = [int(x) for x in a.budgets.split(",")]


def load(d):
    for f in sorted(glob.glob(os.path.join(d, "dump_*.jsonl"))):
        for l in open(f):
            r = json.loads(l)
            if r["tag"] in ("", "done"):
                continue  # warmup requests
            yield r


class Step:
    __slots__ = ("cat", "pidx", "tok", "tgt", "par", "kids", "depth", "score", "cond", "correct", "acc")

    def __init__(self, r):
        n = len(r["tok"])
        self.cat, self.pidx = r["tag"].split(":")[0], int(r["tag"].split(":")[1])
        self.tok, self.tgt = r["tok"], r["tgt"]
        self.kids = [[] for _ in range(n)]
        self.par = [-1] * n
        for j in range(n):
            c = r["nxt"][j]
            while c != -1:
                self.kids[j].append(c)
                self.par[c] = j
                c = r["sib"][c]
        self.depth = [0] * n
        order = [0]
        for j in order:
            for c in self.kids[j]:
                self.depth[c] = self.depth[j] + 1
                order.append(c)
        sc = r["score"]
        self.score = [1.0] + [float(x) for x in sc] if sc is not None else [1.0] * n  # cumulative draft prob
        self.cond = [1.0] * n
        for c in range(1, n):
            ps = self.score[self.par[c]]
            self.cond[c] = self.score[c] / ps if ps > 0 else 0.0
        for j in range(n):  # siblings sorted by conditional draft prob (rank 0 = draft's top-1)
            self.kids[j].sort(key=lambda c: -self.cond[c])
        # correct[c]: c's token is the target argmax after its parent (at most one per parent)
        self.correct = [False] + [self.tok[c] == self.tgt[self.par[c]] for c in range(1, n)]
        self.acc = r["acc"]

    def accepted(self, nodes):
        """accepted draft tokens of the prefix-closed subtree `nodes` (set of node ids, root implied)"""
        j, k = 0, 0
        while True:
            nxt = next((c for c in self.kids[j] if self.correct[c]), None)
            if nxt is None or nxt not in nodes:
                return k
            j, k = nxt, k + 1

    def full_path(self):
        return self.accepted(set(range(len(self.tok))))

    def trunk(self, B):
        """draft top-1 chain of length <= B; returns list of nodes (may be shorter if the pool lacks a child)"""
        out, j = [], 0
        while len(out) < B and self.kids[j]:
            j = self.kids[j][0]
            out.append(j)
        return out


# ---------------- policies: each returns a set of at most B draft nodes ----------------
def pol_chain(s, B):
    return set(s.trunk(B))


def pol_eagle(s, B):
    # SGLang EAGLE selection: top-B pool nodes by cumulative draft score (pool = topk x steps expansion)
    return set(sorted(range(1, len(s.tok)), key=lambda c: -s.score[c])[:B])


def pol_comb(s, B, kmax=None):
    # HedgeTree candidate set: trunk (draft top-1 chain) + leaf siblings of every trunk node, all from the
    # distributions a chain drafter already computes; pick top-B by path value = cumulative draft score
    tr = s.trunk(len(s.tok))
    cands = list(tr)
    for d, t in enumerate([0] + tr[:-1]):
        sibs = [c for c in s.kids[t] if c not in tr]
        cands += sibs[: (kmax - 1) if kmax else None]
    chosen = sorted(cands, key=lambda c: -s.score[c])[:B]
    return set(chosen)


def pol_root_fan(s, B, k):
    # first-position insurance only: k-1 root siblings + trunk of length B-(k-1)
    tr = s.trunk(max(B - (k - 1), 0))
    sibs = [c for c in s.kids[0] if c not in tr][: k - 1]
    return set(tr) | set(sibs)


# ---------------- calibration: P(node correct | its parent's prefix accepted) ----------------
# correct[c] is exactly that event: the target argmax at the parent is computed on the parent's own prefix.
def cbin(x):
    return min(9, int(-math.log2(max(x, 1e-9)))) if x < 1 else 0   # 0: p>=0.5 ... 9: p<2^-9


def feat(s, c):
    rank = s.kids[s.par[c]].index(c)
    return (min(s.depth[c], 4), min(rank, 3), cbin(s.cond[c]))


class Calib:
    """smoothed table P(correct | depth, sibling rank, draft-prob bin), backing off to (rank, bin)"""

    def __init__(self, ss):
        self.t, self.b = collections.defaultdict(lambda: [0, 0]), collections.defaultdict(lambda: [0, 0])
        for s in ss:
            for c in range(1, len(s.tok)):
                f = feat(s, c)
                for tab, key in ((self.t, f), (self.b, f[1:])):
                    tab[key][0] += s.correct[c]; tab[key][1] += 1

    def __call__(self, s, c):
        f = feat(s, c)
        k, n = self.b.get(f[1:], (0, 0))
        prior = (k + 1) / (n + 2)
        k, n = self.t.get(f, (0, 0))
        return (k + 5 * prior) / (n + 5)


def calibrated_values(s, cal):
    v = [1.0] * len(s.tok)
    order = [0]
    for j in order:
        for c in s.kids[j]:
            v[c] = v[j] * cal(s, c)
            order.append(c)
    return v


def pol_eagle_cal(s, B, cal):
    v = calibrated_values(s, cal)
    return set(sorted(range(1, len(s.tok)), key=lambda c: -v[c])[:B])


def pol_comb_cal(s, B, cal):
    v = calibrated_values(s, cal)
    tr = s.trunk(len(s.tok))
    cands = list(tr)
    for t in [0] + tr[:-1]:
        cands += [c for c in s.kids[t] if c not in tr]
    # keep prefix closure: a sibling needs its trunk parent, which has value >= sibling's only if calibrated
    # values are monotone along the trunk; enforce by taking trunk nodes greedily in order
    chosen, trunk_set = [], set()
    for c in sorted(cands, key=lambda c: -v[c]):
        if len(chosen) == B:
            break
        p = s.par[c]
        if p != 0 and p not in trunk_set:
            continue
        chosen.append(c)
        if c in tr:
            trunk_set.add(c)
    return set(chosen)


POLICIES = {
    "chain": pol_chain,
    "eagle_tree": pol_eagle,
    "comb(hedge)": pol_comb,
    "root_fan_k2": lambda s, B: pol_root_fan(s, B, 2),
    "root_fan_k3": lambda s, B: pol_root_fan(s, B, 3),
}

steps = [Step(r) for r in load(a.dump_dir)]
cats = sorted({s.cat for s in steps})
print(f"{len(steps)} verify steps from {a.dump_dir}; categories: {cats}")
bad = sum(s.full_path() != s.acc for s in steps)
print(f"sanity: replayed full-tree accepted == SGLang accepted on {len(steps) - bad}/{len(steps)} steps")

# calibration fitted on even-indexed prompts, policies compared on odd-indexed prompts (held out)
train = [s for s in steps if s.pidx % 2 == 0]
test = [s for s in steps if s.pidx % 2 == 1] or steps
cal = Calib(train)
POLICIES["eagle_tree+cal"] = lambda s, B: pol_eagle_cal(s, B, cal)
POLICIES["comb+cal"] = lambda s, B: pol_comb_cal(s, B, cal)
print(f"calibration fitted on {len(train)} steps, evaluation on {len(test)} held-out steps")

print("\n== reliability on held-out steps: mean draft conditional prob vs actual P(correct), by depth and sibling rank")
rel = collections.defaultdict(lambda: [0.0, 0, 0])
for st_ in test:
    for c in range(1, len(st_.tok)):
        rank = st_.kids[st_.par[c]].index(c)
        if rank > 2:
            continue
        x = rel[(min(st_.depth[c], 7), rank)]
        x[0] += st_.cond[c]; x[1] += st_.correct[c]; x[2] += 1
for (d, rk), (sp, k, n) in sorted(rel.items()):
    print(f"  depth {d} rank {rk}: n={n:6d}  draft prob {sp / n:.3f}  actual {k / n:.3f}  ratio {k / max(sp, 1e-9):.2f}")


# ---------------- diagnostics ----------------
def diag(ss, label):
    n = len(ss)
    p0 = sum(not any(s.correct[c] for c in s.kids[0][:1]) for s in ss) / n  # draft top-1 first token rejected
    cov = {k: sum(any(s.correct[c] for c in s.kids[0][:k]) for s in ss) / n for k in (1, 2, 3, 4, 7)}
    full = sum(s.full_path() for s in ss) / n
    print(f"  {label:15s} steps={n:5d}  P0(first token rejected, chain)={p0:.3f}  "
          f"root coverage top-k: " + " ".join(f"k{k}={v:.3f}" for k, v in cov.items()) + f"  full-pool accepted={full:.2f}")

print("\n== diagnostics")
diag(steps, "ALL")
if a.per_cat:
    for c in cats:
        diag([s for s in steps if s.cat == c], c)

# ---------------- policy table ----------------
def table(ss, label):
    print(f"\n== {label}: mean tokens per step (accepted drafts + 1 bonus), budget B draft nodes")
    print(f"{'policy':14s} " + " ".join(f"B={B:<6d}" for B in budgets))
    for name, f in POLICIES.items():
        vals = [sum(s.accepted(f(s, B)) for s in ss) / len(ss) + 1 for B in budgets]
        print(f"{name:14s} " + " ".join(f"{v:<8.3f}" for v in vals))

table(test, "held-out ALL")
if a.per_cat:
    for c in cats:
        table([s for s in test if s.cat == c], c)
