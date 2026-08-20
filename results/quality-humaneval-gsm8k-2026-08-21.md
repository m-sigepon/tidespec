# Quality Benchmark: HumanEval + GSM8K (2026-08-21)

Conditions: RadixArk/Qwen3.8-27B-NVFP4 (rev 554ebba), temp 0, server-side reasoning_effort=low
(`--default-chat-template-kwargs`); pass@1 is one greedy sample per problem, scored by executing the generated code.
fusion OFF = sglang:qwen38-27b-dflash2-fusion (SGLANG_DFLASH_NGRAM_FUSION=0);
TideSpec v4.6 = sglang:qwen38-27b-dflash2-fusionv4 (=1). Fresh container for each phase (dictionary reset).

| Benchmark | fusion OFF | TideSpec v4.6 | Diff |
|---|---|---|---|
| HumanEval pass@1 (164 problems) | **0.909** (149/164) | **0.902** (148/164) | -1 problem (within ±2.2pp statistical noise) |
| GSM8K accuracy (200 problems) | **0.985** | **0.985** | 0 |
| accept len (HumanEval) | 6.44 | 6.14 | — |
| accept len (GSM8K) | 6.26 | 6.19 | — |
| output tok/s (HumanEval, 2 concurrent) | 212.3 | 214.0 | +0.8% |

Evidence that TideSpec was active: ngram-fusion v4: 23302 proposals / 1945 fired / 24500 steps, trie_w=0.92

Verdict: no quality difference. The -1 problem on HumanEval is 0.6pp against a binomial standard error
(√(p(1-p)/164) ≈ 2.25pp) and is not statistically significant. GSM8K is an exact match. Because
speculative decoding only emits tokens that pass the target model's own accept decision, quality is
expected to track the target model, and these measurements are consistent with that.

Reproduction: [../bench/quality_suite.sh](../bench/quality_suite.sh) (run_eval requires --host 127.0.0.1 with no scheme;
on the eval client, pip install human-eval 'filelock<3.18' and enable the exec line in execution.py).
