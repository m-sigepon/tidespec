# TideSpec ("tidal speculation"): A Regime-Adaptive Draft Control Layer — Design and Implementation

**TideSpec is a control layer, not a model**:
it wraps any drafter, including DFlash2, as "one draft source among several".
The implementation is this repository's fusion v4.x (a patch to the sglang
DFLASH worker).

## In one sentence

Text generation is a tide that alternates between a "copy flood tide"
(re-emitting context: accept rate jumps) and a "generation ebb tide"
(novel content: the model's drafting is needed); TideSpec **reads the tide
every step and switches to the cheapest draft source**, raising speed while
verification still gates every emitted token (quality checked ON vs OFF on
HumanEval / GSM8K; see results/).

## Three core observations (all measured on this machine)

1. **Tidality**: on the same hardware, novel Japanese text runs at 104 tok/s
   vs. 391 tok/s for reproduction generation. The accept rate is not
   stationary — it switches by regime. The theoretical limit of a single
   drafter assumes a stationary accept rate; regime detection steps outside
   that assumption.
2. **Draft reads set the ceiling**: one cycle reads 17GB of target plus
   3.85GB of draft. In the copy tide, the trie (in RAM, 0 bytes read)
   produced comparable accept lengths in our reproduction-workload
   measurements, so eliminating the draft-model read raises the theoretical
   local ceiling from 380 to 840 tok/s.
3. **Verification is full-information feedback**: every step, the target's
   ground truth is fully revealed, so draft sources that were not used can be
   scored for free (shadow running). Each source's true skill can be tracked
   continuously at zero exploration cost.

## Mechanism (three instruments to read the tide, two steering controls)

### Instrument 1: shadow-run scoring (full-information Hedge)
Every step, the trie's hypothetical proposals are matched against the
committed tokens, and the "expected accept length when a proposal exists"
(a conditional expectation) is tracked with an EMA. The model side keeps an
EMA of its actual accepts. The comparison is made on equal footing
(conditional vs. conditional).

### Instrument 2: instantaneous copy signal
The very fact that the trie returned a full-length continuation
(block-1 tokens) is direct evidence that "the copy tide is in now". It
catches the turn of the tide faster than an average would.

### Instrument 3: AIMD streak (a transplant of TCP congestion control)
Each successful firing halves the firing bar (ride the tide); the moment one
fails, it snaps back (pull out immediately on the ebb). Borrows the golden
ratio of greed to caution.

### Steering 1: draft-free copy mode
At flood tide, the 2B draft model's forward pass is skipped entirely and the
trie's continuation is sent straight to verification. One cycle's reads
become the 17GB target only.

### Steering 2: canonical filler (error-correction layer)
When a proposal is short, the remainder is pinned with mask tokens. This
makes the verification suffix independent of the draft source (cutting off
suffix sensitivity), so partial matches can also fire safely.

## Safety structure

- Verification is unmodified: every emitted token passes the target model's
  accept decision, so a wrong draft costs speed rather than injecting tokens
  the target would not choose. (Note this does not make output bit-identical
  to a fusion-off run — see the next point — so quality is additionally
  checked by benchmark: HumanEval / GSM8K in results/.)
- Output is a deterministic function of dictionary state (proven by a
  frozen-dictionary controlled experiment). This is benign variation of the
  same class as radix-cache-state dependence; the quality probe scores 10/10
  in every configuration.

## Measurements (RTX 5090, Qwen3.8-27B NVFP4, greedy)

| Workload | No fusion | TideSpec v4.4 |
|---|---|---|
| Novel generation (10 math problems) | 142-157 | **178.8 (+15-25%)** |
| Code editing | 254 | **358.4 (+41%)** |
| Repetitive generation | 170 | 207-217 (ungated v4.3: 311) |
| Quality probe (10 problems, self-scored) | 10/10 | 10/10 |

## Roadmap

- Phase 1 (done): trie source + regime gate + draft-free firing = v4.x
- Phase 2: long blocks (L=16-24 in the copy tide; GDN verification state
  covered by K-step replay from upstream [#28010](https://github.com/sgl-project/sglang/pull/28010))
- Phase 3: tree verification of the lattice's top-W paths (recovering the
  15 paths DFlash2 discards)
- Phase 4: periodic copy source (Ramanujan period detection; a copy rule
  that does not depend on full match)
- Phase 5 (cloud training): a variable-block-length + copy-pointer-arithmetic
  drafter, and draft-read reduction via a Ramanujan block-sparse topology
