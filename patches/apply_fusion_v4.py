"""Patch pristine dflash_worker_v2.py to fusion v4: NgramCorpus (C++ trie) arm.

Keeps from v3.x: non-blocking ring commit harvest, hedge gating vs model EMA,
canonical mask filler for short proposals. Replaces the hand-rolled python
dictionary with sglang's stateful C++ trie (all occurrences, depth-18
sliding-window insert/match).
"""
import ast

p = "C:/_Tools/sglang/patches/dflash_worker_v2.py"
src = open(p, encoding="utf-8").read()

a = "import logging\nimport math\n"
assert src.count(a) == 1
src = src.replace(a, "import logging\nimport math\nimport os\n")

a = (
    "def _is_all_greedy(sampling_info) -> bool:\n"
    "    return sampling_info is None or sampling_info.is_all_greedy\n"
)
assert src.count(a) == 1
src = src.replace(a, a + '''

# NGRAM x DFLASH fusion v4 (opt-in via SGLANG_DFLASH_NGRAM_FUSION=1).
# Dictionary arm = sglang's C++ ngram trie (all occurrences, stateful match);
# fires only when its shadow-scored expectation beats the draft model's own
# accept EMA. Verification is unchanged.
_NGRAM_FUSION_ENABLED = os.environ.get("SGLANG_DFLASH_NGRAM_FUSION", "0") == "1"
_NGRAM_FUSION_EMA = float(os.environ.get("SGLANG_DFLASH_NGRAM_EMA", "0.8"))
_NGRAM_FUSION_MARGIN = float(os.environ.get("SGLANG_DFLASH_NGRAM_MARGIN", "0"))
_NGRAM_FUSION_MIN_PROPOSE = int(os.environ.get("SGLANG_DFLASH_NGRAM_MIN_PROPOSE", "4"))
_NGRAM_FUSION_TRIE_DEPTH = int(os.environ.get("SGLANG_DFLASH_NGRAM_TRIE_DEPTH", "18"))
_NGRAM_FUSION_MIN_W = float(os.environ.get("SGLANG_DFLASH_NGRAM_MIN_W", "2.5"))
''')

a = "        self._draft_seq_lens_cpu_buf: Optional[torch.Tensor] = None  # [cap_bs] on CPU\n"
assert src.count(a) == 1
src = src.replace(a, a + """        self._fusion_live_rid = None
        self._fusion_live = None  # {"toks": committed history}
        self._fusion_corpus = None
        self._fusion_weight = 0.0
        self._fusion_model_ema = 4.0
        self._fusion_shadow = None  # {"cont": proposal, "chosen": str}
        self._fusion_hits = 0
        self._fusion_steps = 0
        self._fusion_ring = None
        self._fusion_ring_next = 0
        self._fusion_fail_run = 0
        self._fusion_fired = 0
        self._fusion_streak = 0
""")

a = "    def _propose_selector_block(\n"
assert src.count(a) == 1
helpers = '''    # --- NGRAM x DFLASH fusion v4 ------------------------------------------
    def _fusion_get_corpus(self):
        if self._fusion_corpus is None:
            from sglang.srt.speculative.cpp_ngram.ngram_corpus import NgramCorpus

            self._fusion_corpus = NgramCorpus(
                max_trie_depth=_NGRAM_FUSION_TRIE_DEPTH,
                min_bfs_breadth=1,
                max_bfs_breadth=1,
                draft_token_num=int(self.block_size),
                match_type="BFS",
                capacity=10_000_000,
            )
        return self._fusion_corpus

    def _fusion_drain_and_score(self):
        if not self._fusion_ring:
            return
        e = _NGRAM_FUSION_EMA
        for slot in self._fusion_ring:
            if slot["st"] is None or not slot["evt"].query():
                continue
            try:
                cl = int(slot["clen"][0])
                committed = slot["buf"][:cl].tolist()
                slot["st"]["toks"].extend(committed)
                shadow = slot["shadow"]
                if shadow is not None:
                    cont = shadow["cont"]
                    # Score the trie arm only on steps where it actually
                    # proposed: the weight is then the conditional expected
                    # accept length given a proposal, which is the fair
                    # quantity to compare against the model's accept EMA.
                    if cont is not None:
                        n = 0
                        for a_tok, b_tok in zip(cont, committed):
                            if a_tok != b_tok:
                                break
                            n += 1
                        self._fusion_weight = e * self._fusion_weight + (1 - e) * float(n)
                    if shadow["chosen"] == "model":
                        self._fusion_model_ema = e * self._fusion_model_ema + (
                            1 - e
                        ) * float(max(cl - 1, 0))
                    elif shadow["chosen"] == "trie" and cont is not None:
                        # AIMD with spurious-loss tolerance: one failed fire
                        # halves the streak (small wave inside a flood);
                        # two consecutive failures reset it (real ebb).
                        if n >= 6:
                            self._fusion_streak = min(self._fusion_streak + 1, 6)
                            self._fusion_fail_run = 0
                        else:
                            self._fusion_fail_run += 1
                            if self._fusion_fail_run >= 2:
                                self._fusion_streak = 0
                            else:
                                self._fusion_streak //= 2
            except Exception as e2:
                logger.warning("DFLASH ngram-fusion drain failed: %s", e2)
            slot["st"] = None
            slot["shadow"] = None

    def _fusion_state_for(self, req):
        rid = getattr(req, "rid", None)
        if rid is None:
            return None
        if rid == self._fusion_live_rid:
            return self._fusion_live
        toks = list(getattr(req, "origin_input_ids", None) or [])
        toks += list(getattr(req, "output_ids", None) or [])
        self._fusion_live_rid = rid
        self._fusion_live = {"toks": toks, "rid": str(rid)}
        return self._fusion_live

    def _fusion_lookup(self, st, need):
        corpus = self._fusion_get_corpus()
        toks = st["toks"]
        window = toks[-_NGRAM_FUSION_TRIE_DEPTH:]
        # Feed the trie, then match. Both are cheap C++ calls.
        corpus.batch_put([window])
        corpus.synchronize()
        drafts, mask = corpus.batch_get([st["rid"]], [window], [len(toks)])
        blk = int(self.block_size)
        row = [int(x) for x in drafts[:blk]]
        import numpy as _np

        m = _np.asarray(mask).reshape(-1)
        if m.size >= blk * blk:
            tree = m[: blk * blk].reshape(blk, blk).tolist()
            paths = corpus.leaf_paths_from_mask(row, tree)
        else:
            paths = []
        if not paths:
            # Chain fallback: with breadth=1 the row itself is the path.
            paths = [row]
        path = max(paths, key=len)
        # Root convention: drop the anchor if the path starts with it.
        cont = path[1:] if path and path[0] == toks[-1] else list(path)
        cont = [t for t in cont if t >= 0][:need]
        if len(cont) < _NGRAM_FUSION_MIN_PROPOSE:
            return None, False
        full = len(cont) == need
        if len(cont) < need:
            cont = cont + [int(self._mask_token_id)] * (need - len(cont))
        self._fusion_hits += 1
        return cont, full

    def _fusion_commit_async(self, st, out_tokens, commit_lens):
        if self._fusion_ring is None:
            self._fusion_ring = [
                {
                    "buf": torch.empty(
                        (out_tokens.shape[1],), dtype=out_tokens.dtype, pin_memory=True
                    ),
                    "clen": torch.empty(
                        (1,), dtype=commit_lens.dtype, pin_memory=True
                    ),
                    "evt": torch.cuda.Event(),
                    "st": None,
                    "shadow": None,
                }
                for _ in range(4)
            ]
        slot = self._fusion_ring[self._fusion_ring_next]
        self._fusion_ring_next = (self._fusion_ring_next + 1) % len(self._fusion_ring)
        if slot["st"] is not None:
            slot["st"] = None
            slot["shadow"] = None
        slot["buf"].copy_(out_tokens[0], non_blocking=True)
        slot["clen"].copy_(commit_lens[:1], non_blocking=True)
        slot["evt"].record()
        slot["st"] = st
        slot["shadow"] = self._fusion_shadow
        self._fusion_shadow = None

'''
src = src.replace(a, helpers + a)

start = "        noise_embedding = embed_module(block_ids)\n"
end = "            ).view(bs, int(self.block_size) - 1)\n"
i0 = src.index(start)
i1 = src.index(end, i0) + len(end)
block = src[i0:i1]
flatten = (
    "        positions = positions_2d.reshape(-1)\n"
    "        verify_out_cache_loc = verify_out_cache_loc_2d.reshape(-1)\n"
)
assert flatten in block
block = block.replace(flatten, "")
indented = "".join(("    " + ln if ln.strip() else ln) for ln in block.splitlines(keepends=True))
gate = flatten + '''
        fusion_st = None
        fusion_cont = None
        if (
            _NGRAM_FUSION_ENABLED
            and bs == 1
            and not self.use_compact_draft_cache
            and _is_all_greedy(batch.sampling_info)
            and getattr(batch, "reqs", None)
        ):
            try:
                self._fusion_drain_and_score()
                fusion_st = self._fusion_state_for(batch.reqs[0])
                if fusion_st is not None:
                    cont, full = self._fusion_lookup(fusion_st, block_size - 1)
                    chosen = "model"
                    # Regime-gated firing: a full-length continuation is the
                    # instantaneous copy signal, and the conditional-accept EMA
                    # must not have collapsed (novel-text false copies drive it
                    # toward 0, halting firing automatically).
                    bar = _NGRAM_FUSION_MIN_W * (0.5 ** self._fusion_streak)
                    if cont is not None and full and self._fusion_weight >= bar:
                        fusion_cont = cont
                        chosen = "trie"
                        self._fusion_fired += 1
                    self._fusion_shadow = {"cont": cont, "chosen": chosen}
                self._fusion_steps += 1
                if self._fusion_steps % 500 == 0:
                    logger.info(
                        "DFLASH ngram-fusion v4: %d proposals / %d fired / %d steps, "
                        "trie_w=%.2f model_ema=%.2f streak=%d",
                        self._fusion_hits,
                        self._fusion_fired,
                        self._fusion_steps,
                        self._fusion_weight,
                        self._fusion_model_ema,
                        self._fusion_streak,
                    )
            except Exception as e:
                logger.warning("DFLASH ngram-fusion v4 lookup failed: %s", e)
                fusion_st = None
                fusion_cont = None

        if fusion_cont is not None:
            self._selector_sample = None
            draft_next = torch.tensor(
                fusion_cont, dtype=torch.int64, device=device
            ).view(1, -1)
        else:
'''
src = src[:i0] + gate + indented + src[i1:]

a = "        if SIMULATE_ACC_LEN > 0:\n"
assert src.count(a) == 1
src = src.replace(a, """        if fusion_st is not None:
            try:
                self._fusion_commit_async(fusion_st, out_tokens, commit_lens)
            except Exception as e:
                logger.warning("DFLASH ngram-fusion commit failed: %s", e)

""" + a)

open(p, "w", encoding="utf-8", newline="\n").write(src)
ast.parse(src)
print("v4 patch applied, syntax OK")
