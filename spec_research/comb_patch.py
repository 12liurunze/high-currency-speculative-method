# Adds the comb (HedgeTree) draft tree to the SGLang runtime actually imported by envs/sglang-spec
# (experiments/runtime/sglang). Enabled per server by env SGLANG_COMB_K=K (K >= 2) with --speculative-eagle-topk 1,
# --disable-cuda-graph and --disable-overlap-schedule; with SGLANG_COMB_K unset the code paths are unchanged.
# comb: the drafter runs the plain topk=1 chain (trunk); the top-K of every draft distribution along the trunk
# is kept, trunk siblings become leaves, and the top-B nodes by cumulative draft score (B = num_steps, the same
# verify width as the chain) are verified as a tree.
# Originals are saved as *.pre-comb. Edits already present are skipped, so re-running on a runtime patched by an
# older version of this script only adds what that version did not cover (keeping the backup of the stock file).
# Nothing is written unless every remaining edit matches.
# Usage: python comb_patch.py /home/lrz/AI_infra/experiments/runtime/sglang/srt
import os, shutil, sys

SRT = sys.argv[1]


pending = []  # (rel, path, patched source, stock), written together by apply()


def patch(rel, edits):
    path = os.path.join(SRT, rel)
    src = open(path).read()
    todo = [(old, new) for old, new in edits if new not in src]
    if not todo:
        print(f"skip {rel}: already patched")
        return
    # none of the edits present: a stock file, whose backup replaces any stale .pre-comb; otherwise the file was
    # patched by an older version of this script and its .pre-comb already holds the stock original
    stock = len(todo) == len(edits)
    for old, new in todo:
        n = src.count(old)
        if n != 1:
            sys.exit(f"{rel}: expected 1 match, found {n}:\n{old}")
        src = src.replace(old, new)
    pending.append((rel, path, src, stock))


def apply():
    for rel, path, src, stock in pending:
        if stock or not os.path.exists(path + ".pre-comb"):
            shutil.copy2(path, path + ".pre-comb")
        # replace instead of writing in place: installers such as uv hard-link files to a shared cache
        shutil.copy2(path, path + ".comb-tmp")
        with open(path + ".comb-tmp", "w") as f:
            f.write(src)
        os.replace(path + ".comb-tmp", path)
        print("patched", rel)


patch("speculative/eagle_worker_v2.py", [
    ("import contextlib\nimport logging\nimport time\n",
     "import contextlib\nimport logging\nimport os\nimport time\n"),
    # mode flags (draft worker __init__)
    ('''        self.speculative_num_draft_tokens = get_spec().speculative_num_draft_tokens
        self.speculative_algorithm = SpeculativeAlgorithm.from_string(
            get_spec().speculative_algorithm
        )

        self._rebuild_topk1_chain_buffers()''',
     '''        self.speculative_num_draft_tokens = get_spec().speculative_num_draft_tokens
        self.speculative_algorithm = SpeculativeAlgorithm.from_string(
            get_spec().speculative_algorithm
        )
        # comb (HedgeTree) draft tree: SGLANG_COMB_K = siblings kept per trunk position
        self.comb_k = int(os.environ.get("SGLANG_COMB_K", "0"))
        self.comb_mode = self.comb_k > 1
        if self.comb_mode:
            assert self.topk == 1, "comb drafts a chain: use --speculative-eagle-topk 1"
            assert not get_spec().speculative_use_rejection_sampling, "comb is greedy-only"

        self._rebuild_topk1_chain_buffers()'''),
    # verify tree is laid out as a topk=K EAGLE pool
    ('''            topk=self.topk,
            num_steps=self.speculative_num_steps,
            num_draft_tokens=self.speculative_num_draft_tokens,
            tree_mask_mode=self.tree_mask_mode,''',
     '''            topk=self.comb_k if self.comb_mode else self.topk,
            num_steps=self.speculative_num_steps,
            num_draft_tokens=self.speculative_num_draft_tokens,
            tree_mask_mode=self.tree_mask_mode,'''),
    # dispatch
    ('''            return self._draft_forward_idle(forward_batch, spec_info)

        out_cache_loc = forward_batch.out_cache_loc''',
     '''            return self._draft_forward_idle(forward_batch, spec_info)
        if self.comb_mode:
            return self._comb_draft_forward(forward_batch)

        out_cache_loc = forward_batch.out_cache_loc'''),
    # comb draft forward
    ('''    def _draft_forward_idle(''',
     '''    def _comb_draft_forward(self, forward_batch: ForwardBatch):
        # comb (HedgeTree): run the topk=1 chain (trunk) and keep the top-K of each draft distribution along
        # it; the trunk's siblings become leaves. Candidates are laid out as an EAGLE topk=K pool in which
        # only parent slot 0 of every step (the trunk node) is expanded and the other slots score -1, so
        # organize_draft_results takes the top-B by cumulative draft score and build_tree_kernel_efficient
        # builds the verify tree unchanged. Pool index of step i (>=1), slot a, child c: K + K*K*(i-1) + a*K + c.
        spec_info: EagleDraftInput = forward_batch.spec_info
        K, S = self.comb_k, self.speculative_num_steps
        p, ids = spec_info.topk_p[:, :K].float(), spec_info.topk_index[:, :K]
        if self.hot_token_id is not None:
            ids = self.hot_token_id[ids]
        bs = p.shape[0]
        out_cache_loc = per_step_draft_out_cache_loc(forward_batch.out_cache_loc, bs, 1, S)
        hidden_states = spec_info.hidden_states
        decay = 1.0 - 1e-5  # child strictly below parent even when p == 1: keeps the top-B set prefix-closed
        score_list = [p.unsqueeze(1)]
        token_list = [ids]
        parents_list = [torch.arange(-1, K, dtype=torch.long, device=p.device).expand(bs, -1)]
        trunk_score = p[:, :1]
        for i in range(1, S):
            forward_batch.input_ids = ids[:, 0].contiguous()
            forward_batch.out_cache_loc = out_cache_loc[i - 1]
            spec_info.hidden_states = hidden_states
            with forward_context(
                ForwardContext(attn_backend=self.draft_attn_backend.attn_backends[i - 1])
            ):
                logits_output = self.draft_runner.forward(forward_batch).logits_output
            forward_batch.positions.add_(1)
            probs = renorm_draft_probs(
                logits_output.next_token_logits, forward_batch.sampling_info, False
            )
            p, ids = fast_topk(probs, K, dim=-1)
            if self.hot_token_id is not None:
                ids = self.hot_token_id[ids]
            hidden_states = logits_output.hidden_states
            scores = torch.full((bs, K, K), -1.0, dtype=torch.float32, device=p.device)
            scores[:, 0] = trunk_score * p.float() * decay
            trunk_score = scores[:, 0, :1]
            score_list.append(scores)
            token_list.append(ids.repeat(1, K))
            parents_list.append(
                torch.full((bs, K), K + K * K * (i - 1), dtype=torch.long, device=p.device)
            )
        parent_list, top_scores_index, draft_tokens = organize_draft_results(
            score_list, token_list, parents_list, self.speculative_num_draft_tokens
        )
        return parent_list, top_scores_index, draft_tokens, None

    def _draft_forward_idle('''),
    # draft extend after prefill: keep top-K for the root siblings
    ('''            topk_p, topk_index = fast_topk(probs, self.topk, dim=-1)
        return EagleDraftInput(''',
     '''            topk_p, topk_index = fast_topk(
                probs, self.comb_k if self.comb_mode else self.topk, dim=-1
            )
        return EagleDraftInput('''),
    # draft extend after decode
    ('''        elif self.topk == 1 and not _is_hip:
            # Gated to CUDA: see #26358''',
     '''        elif self.topk == 1 and not _is_hip and not self.comb_mode:
            # Gated to CUDA: see #26358'''),
    ('''            ret_topk_p, ret_topk_index = fast_topk(probs, self.topk, dim=-1)
            ret_draft_probs = None''',
     '''            ret_topk_p, ret_topk_index = fast_topk(
                probs, self.comb_k if self.comb_mode else self.topk, dim=-1
            )
            ret_draft_probs = None'''),
    # verify as a tree (compacts the accepted path, as for topk > 1)
    ('''            topk=self.topk,
            num_draft_tokens=self.speculative_num_draft_tokens,
            device=self.device,
            metadata_ready_pre_pad=False,''',
     '''            topk=2 if self.draft_worker.comb_mode else self.topk,
            num_draft_tokens=self.speculative_num_draft_tokens,
            device=self.device,
            metadata_ready_pre_pad=False,'''),
    # comb runs with --speculative-eagle-topk 1 but verifies a tree. Attention backends that pick the chain path from
    # that flag verify it wrongly: fa3 (the default on Hopper, chosen there precisely because topk == 1) neither fills
    # nor reads the tree mask and runs target verify as causal attention, so siblings see each other. Stop the server
    # unless the target backend is one whose tree verify is checked for comb. Inserted in front of
    # _comb_draft_forward, which runs during draft CUDA-graph capture (or on the first draft without graphs), i.e.
    # after the target backends exist and before any request is served.
    ('''    def _comb_draft_forward(self, forward_batch: ForwardBatch):
''',
     '''    # target attention backends that apply the tree mask in target verify even when the server runs with topk == 1
    COMB_TREE_VERIFY_BACKENDS = ("FlashInferAttnBackend",)

    def _comb_check_attn_backend(self):
        if getattr(self, "_comb_backend_checked", False):
            return
        # descend through wrappers: prefill/decode split, and the full-attention part of hybrid linear-attention
        # models (their linear layers are covered by the hybrid_linear_attn_backend.py edit)
        leaves, stack, seen = [], [self.target_worker.model_runner.attn_backend], set()
        while stack:
            backend = stack.pop()
            if backend is None or id(backend) in seen:
                continue
            seen.add(id(backend))
            parts = [getattr(backend, a, None) for a in ("prefill_backend", "decode_backend", "full_attn_backend")]
            parts = [b for b in parts if b is not None]
            if parts:
                stack.extend(parts)
            else:
                leaves.append(type(backend).__name__)
        unsupported = sorted({n for n in leaves if n not in self.COMB_TREE_VERIFY_BACKENDS})
        if not leaves or unsupported:
            raise RuntimeError(
                f"comb (SGLANG_COMB_K={self.comb_k}) verifies a tree under --speculative-eagle-topk 1, but the "
                f"target attention backend {unsupported or leaves} is not one whose tree verify works with "
                f"topk == 1 (fa3 skips the tree mask). Supported: {list(self.COMB_TREE_VERIFY_BACKENDS)}; "
                f"start the server with --attention-backend flashinfer."
            )
        self._comb_backend_checked = True

    def _comb_checked(fn):  # decorator: check the target attention backend before the first comb draft
        def wrapper(self, *args, **kwargs):
            self._comb_check_attn_backend()
            return fn(self, *args, **kwargs)
        return wrapper

    @_comb_checked
    def _comb_draft_forward(self, forward_batch: ForwardBatch):
'''),
])

# Draft decode CUDA graph: the captured draft_forward dispatches to _comb_draft_forward, which reads K root
# candidates from spec_info, so the static topk_p / topk_index input buffers must be K wide. Target verify is
# already captured with a custom mask and draft extend returns full logits (top-K is taken outside the graph).
patch("speculative/eagle_draft_cuda_graph_runner.py", [
    ('''            topk_p = torch.zeros((self.max_bs, self.topk), dtype=torch.float32)
            topk_index = torch.zeros((self.max_bs, self.topk), dtype=torch.int64)''',
     '''            topk_width = (
                self.eagle_worker.comb_k
                if getattr(self.eagle_worker, "comb_mode", False)
                else self.topk
            )
            topk_p = torch.zeros((self.max_bs, topk_width), dtype=torch.float32)
            topk_index = torch.zeros((self.max_bs, topk_width), dtype=torch.int64)'''),
])

# Hybrid linear-attention targets (GDN, e.g. Qwen3.5): the linear layers carry recurrent state, so a custom mask is not
# enough to verify a tree. With topk=1 they skip the parent links and run the N verify tokens as one chain, which feeds
# the trunk's siblings into the state of every later node.
patch("layers/attention/hybrid_linear_attn_backend.py", [
    ("import logging\nfrom typing import TYPE_CHECKING, Optional, Union\n",
     "import logging\nimport os\nfrom typing import TYPE_CHECKING, Optional, Union\n"),
    # self.topk only selects chain vs tree target verify in this backend
    ("        self.topk = get_spec().speculative_eagle_topk or 0\n",
     '''        self.topk = max(
            get_spec().speculative_eagle_topk or 0,
            int(os.environ.get("SGLANG_COMB_K", "0")),
        )
'''),
])

# The deduplicated conv window overlaps consecutive verify steps, which is only valid for a linear chain.
patch("mem_cache/memory_pool.py", [
    ("        and (speculative_eagle_topk is None or speculative_eagle_topk <= 1)\n",
     '''        and (speculative_eagle_topk is None or speculative_eagle_topk <= 1)
        and int(os.environ.get("SGLANG_COMB_K", "0")) <= 1
'''),
])

apply()
