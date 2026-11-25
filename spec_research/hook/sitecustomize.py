# Spec-research dump hook. Inert unless SPEC_DUMP_DIR is set; enabled only for our own processes via PYTHONPATH.
# Right after sglang.srt.speculative.eagle_utils is imported, wraps organize_draft_results (to keep the per-node
# cumulative draft scores) and verify_tree_greedy_func (greedy tree verify), then appends one JSON line per
# (verify step, request): the whole draft tree, its scores and the target model's argmax at every node.
# The patched functions return exactly what the originals return, so decoding is unchanged.
import os
import sys

_DIR = os.environ.get("SPEC_DUMP_DIR")
if _DIR:
    import importlib.abc
    import importlib.util
    import json

    _TARGET = "sglang.srt.speculative.eagle_utils"

    def _rank0():
        try:
            import torch.distributed as dist
            return not dist.is_available() or not dist.is_initialized() or dist.get_rank() == 0
        except Exception:
            return True

    def _patch(m):
        import torch

        st = {"scores": None, "step": 0, "fh": None}
        orig_org, orig_verify = m.organize_draft_results, m.verify_tree_greedy_func

        def organize_draft_results(score_list, token_list, parents_list, num_draft_token):
            out = orig_org(score_list, token_list, parents_list, num_draft_token)
            # out = (parent_list, top_scores_index sorted, draft_tokens); verify node k>=1 is pool index top_scores_index[k-1]
            scores = torch.cat(score_list, dim=1).flatten(1)
            st["scores"] = torch.gather(scores, 1, out[1]).float().cpu()
            return out

        def dump(kw, res):
            if st["fh"] is None:
                os.makedirs(_DIR, exist_ok=True)
                st["fh"] = open(os.path.join(_DIR, f"dump_{os.getpid()}.jsonl"), "a")
            tagf = os.path.join(_DIR, "tag")
            tag = open(tagf).read().strip() if os.path.exists(tagf) else ""
            tok = kw["candidates"].cpu().tolist()           # node 0 = root (last committed token), 1.. = draft nodes
            nxt = kw["retrieve_next_token"].cpu().tolist()  # first child of each node, -1 if leaf
            sib = kw["retrieve_next_sibling"].cpu().tolist()  # next sibling, -1 if last
            tgt = kw["target_predict"].cpu().tolist()       # target argmax after each node
            acc = res[2].cpu().tolist()                     # accepted draft tokens this step
            sc = st["scores"].tolist() if st["scores"] is not None else [None] * len(tok)
            for i in range(len(tok)):
                st["fh"].write(json.dumps({"tag": tag, "step": st["step"], "i": i, "tok": tok[i], "nxt": nxt[i],
                                           "sib": sib[i], "tgt": tgt[i], "score": sc[i], "acc": acc[i]},
                                          separators=(",", ":")) + "\n")
            st["fh"].flush()
            st["step"] += 1

        def verify_tree_greedy_func(*args, **kw):
            res = orig_verify(*args, **kw)
            if _rank0():
                try:
                    dump(kw, res)
                except Exception as e:  # never break decoding because of logging
                    print(f"[spec-dump] error: {e!r}", flush=True)
            return res

        m.organize_draft_results = organize_draft_results
        m.verify_tree_greedy_func = verify_tree_greedy_func
        print(f"[spec-dump] patched {_TARGET}, dumping to {_DIR}", flush=True)

    class _Finder(importlib.abc.MetaPathFinder):
        def find_spec(self, name, path, target=None):
            if name != _TARGET:
                return None
            sys.meta_path.remove(self)
            try:
                spec = importlib.util.find_spec(name)
            finally:
                sys.meta_path.insert(0, self)
            if spec is None or spec.loader is None:
                return spec
            orig_exec = spec.loader.exec_module

            def exec_module(module, _orig=orig_exec):
                _orig(module)
                _patch(module)

            spec.loader.exec_module = exec_module
            return spec

    sys.meta_path.insert(0, _Finder())
