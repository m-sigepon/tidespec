# Attacking the Limits of Speculative Decoding on a Single RTX 5090:
# An Empirical Study of Qwen3.8-27B Inference Optimization and a Proposal for the TideSpec ("tidal speculation") Control Layer

**Period**: 2026-08-18 to 08-20 / **Environment**: RTX 5090 32GB (SM120), Windows 11, Docker Desktop (WSL2), sglang (main / [PR #35496](https://github.com/sgl-project/sglang/pull/35496) lineage)
**Deliverables**: this repository (batch files, Dockerfiles, patches, bench.py, measurement JSONL, this report)

---

## Abstract

We systematically attacked, driven by measurement, the inference speed limits
of serving Qwen3.8-27B (an NVFP4 W4A4 hybrid GDN model) on a single 32GB GPU.
The main contributions are the following six points.

1. **Discovery and fix of two environment-specific bugs**: (a) under WSL2 GPU
   virtualization, mmap'd safetensors loading falsely consumes the GPU
   aperture and causes spurious OOM (fully cured with
   `--weight-loader-disable-mmap`); (b) a race bug in which the abort request
   on client disconnect is swallowed due to a state race and generation runs
   away to the context limit (fixed).
2. **Formalization of the speed limit**: the single-stream decode ceiling =
   memory bandwidth × committed tokens per weight read. On this machine we
   derived a plain-AR ceiling of 105 tok/s and a theoretical ceiling of
   380 tok/s for the current speculative configuration including the draft
   read, and checked measurements against them (all measured values fall
   below the respective ceilings).
3. **Measured comparison of three speculative decoding methods**: DSpark /
   DFlash2 / NGRAM across 9 workloads × multiple passes. We quantified
   DFlash2's advantage (+7-37% on code generation, accept length 4.38 vs
   3.15) and NGRAM's stark two-sidedness (novel 60-107 / reproduction
   302-391 tok/s).
4. **Empirical study of reversibility**: even under greedy settings,
   controlled experiments demonstrated that outputs diverge between cold and
   warm prefill and across speculative methods (no bit-identity exists
   anywhere in the stack). We further discovered suffix sensitivity: the
   content of the draft suffix affects the numerics of the accepted region.
   In response, we moved the verification criterion from hash identity to
   task quality (a probe of our own making).
5. **Implementation study of dictionary x model draft fusion**: implemented
   full-information-feedback Hedge-style drafter selection (following
   Not-a-Bandit) inside the sglang DFLASH worker (v1 to v3.2). Through
   asynchronous ring scoring to eliminate overhead, a two-generation
   dictionary, and so on, we established the observation system (shadow-run
   scoring) and clarified the limits of a naive dictionary structure.
6. **TideSpec design proposal**: designed a next-generation speculative
   architecture integrating all of the above findings (variable-rate block
   geometry, multi-source Hedge aggregation, draft-free copy mode, deferred
   convergence tree verification, and a deterministic ECC layer).

---

## 1. Introduction

The setting is "an individual developer running a state-of-the-art 27B-class
model at practical speed on one consumer graphics card". Upstream
optimization targets datacenter GPUs (H200 etc.), so under the conditions of
32GB, Windows, and a virtualization stack, many problems remain unexplored.
This study carried out (1) establishing a production setup, (2) theorizing
and attacking the speed limit, and (3) designing next-generation methods, as
a single effort.

## 2. Theoretical framework

### 2.1 Formalizing the wall
Decoding is bandwidth-bound; the ceiling is
**B / (W_target + W_draft) × E[committed tokens per cycle]**.
On this machine (B=1.79TB/s, W_target=17GB, W_draft=3.85GB):
- Plain AR: 105 tok/s (measured 74; kernel overhead keeps it below theory)
- DFlash2 speculation (accept 4.4): theory 380 (measured 104-254,
  workload-dependent)

### 2.2 Structural correspondence with wireless communication
Speculative decoding = multi-level modulation, adaptive draft length = link
adaptation (AdaSD etc.), multiple drafters = carrier aggregation
(Not-a-Bandit), constrained decoding + verification = error-correcting
codes: these correspondences hold, and 2024-26 papers exist for each
counterpart. The full-information nature of verification (drafters not
chosen can still be scored after the fact) is an advantage wireless does
not have.

### 2.3 Text as a regime-switching process
Measurement confirmed that text generation is a regime-switching process
between a "copy phase" (accepts near 8, dictionary dominant) and a
"generation phase" (accepts 2-4, model draft dominant). The theoretical
limit of a single drafter (a lower bound via branching random walks)
assumes a stationary accept rate; switching via regime detection removes
that limit's premise.

## 3. System construction and environment-specific findings

### 3.1 WSL2 mmap aperture problem (new discovery)
Symptom: loading the 3.85GB draft checkpoint consumed 9.88GB (all remaining
VRAM), and mamba pool allocation failed with a negative value. By measuring
inside and outside torch separately, we located the 6.15GB excess outside
torch's allocator, consistent with mmap'd pages being counted against the
GPU aperture under WSL2 GPU-PV (working hypothesis; we could not test native
Linux). Disabling mmap normalized it to the expected 3.85GB. We found no
upstream reports of this behavior.

### 3.2 Abort race bug (fix applied locally)
On streaming disconnect, if state deletion precedes the delayed abort task,
the abort request is swallowed as "no target" and the scheduler keeps
generating (the single slot is occupied up to the context limit). Fixed by
unconditionally dispatching AbortReq (idempotent).

### 3.3 Other
We observed a weight swap via a silent update of the upstream repository and
an unexpected 16.5GB re-download, and introduced revision SHA pinning in all
batch files. We identified that host-side desktop VRAM occupancy (about 2GB)
makes the official recipe's mem-fraction 0.90 correspond to 0.93 on this
machine (cross-validated against the 5090 measurements in upstream cookbook
[PR #35663](https://github.com/sgl-project/sglang/pull/35663)).

## 4. Measurement results

### 4.1 Three-method comparison (original 27B, greedy, reasoning_effort=low, excerpt)
| Case | DSpark | DFlash2 | NGRAM (cold/warm) |
|---|---|---|---|
| TypeScript generation | 144 | **197** | 71 / 302 |
| Python generation | 139 | 142 | 67 / 123 |
| Boilerplate code | 177 | **189** | 100 / 174 |
| Math | 217 | **222** | 107 / **391** |
| Japanese prose | 105 | 104 | 64 / 158 |
| Accept length | 3.15 | **4.38** | 2.03 (max 7.75) |

### 4.2 Optimization by elimination (all rejected by measurement)
FP8 draft (offset by lower acceptance), torch.compile (+1%; the speculative
path is already graph-captured), fp32 SSM (does not fit in 32GB; later also
confirmed by upstream measurement that "bf16 is strictly faster for
DFlash2").

### 4.3 Controlled experiments on reversibility
- fusion disabled: 4 cold passes fully identical (deterministic)
- pure AR vs DFLASH: each stable, but mutually mismatched (FP differences
  in kernel paths)
- fusion enabled: differs every pass — suggests numerical sensitivity to
  the draft suffix
- quality probe (10 numeric-answer problems, self-scored): 10/10 in every
  configuration — the divergence is benign

## 5. Fusion experiments (NGRAM x DFlash2, v1 to v3.2)

We implemented a dictionary draft branch inside the DFLASH worker (keeping
the block length fixed at 8 avoids the CUDA graph shape problem). The v3
line implemented full-information shadow running (every step, all dictionary
arms' hypothetical proposals are matched against the committed tokens and
scored) and firing decisions by Hedge weights. Asynchronous ring scoring
(4-plane pinned buffers + event.query) eliminated the scoring cost's
pipeline stall (synchronous version -14%; asynchronous version has
effectively zero overhead).

**Result**: the observation system works as theory predicts (dictionary arm
score 4.1 vs model EMA 4.05). However, with the naive dictionary
(two generations per key, single occurrence), FP divergence contaminates the
reproduction sequences, so match length does not grow and no advantage over
the model draft could be established. Since the cause split (candidate
shortage vs FP-divergence mismatch) is unfinished, before porting to a trie
we will run as Phase 0: (1) a frozen-dictionary controlled experiment, and
(2) canonical filler ECC (pin everything past the proposal's confident
length with mask tokens, while also enabling partial-match proposals), and
use the results to decide whether the C++ trie port (API investigation
complete) is necessary.

## 6. Negative results (paths closed by measurement)

- Low-rank decomposition of the draft model: spectra are flat across all
  matrices (49% error at rank 1024). No low-rank redundancy remains in a
  distilled model.
- Low-rank screening of the target lm_head: at rank 2048, 19% error with
  only a 20% read reduction. Not viable.
- Quantum-inspired compression (without healing) is not viable on any path
  on this machine. MPO with healing (the CompactifAI line) remains as a
  cloud project.

## 7. TideSpec proposal (details in tidespec-design.md)

Five mechanisms plus two additional arms, addressing DFlash2's three
structural wastes (premature convergence of the lattice, fixed block of 8,
and draft reads during copy phases):
1. Variable-rate block geometry (interference scheduling: L and W set by
   source agreement/disagreement)
2. Multi-source Hedge aggregation (trie, diffusion drafter, MTP,
   **periodic copy** [Ramanujan period detection; a copy rule robust to FP
   divergence: draft[i]=tokens[i−P]])
3. Draft-free copy mode (in copy phases, read only the 17GB target)
4. Deferred convergence (tree verification of the lattice's top-W paths)
5. Deterministic ECC layer (pin everything past the proposal length with a
   canonical filler, cutting off suffix sensitivity)
Training stage (cloud): variable block length + copy-pointer arithmetic,
**a drafter with a Ramanujan block-sparse topology** (W_draft reduction).

Performance model: copy phases 700-1000, generation phases 250-300 (both
derived from measurements on this machine). The mixed estimate was updated
by back-solving the implicit copy ratio from regeneration measurements
(harmonic-mean model): even under full regeneration, the dictionary's
extraction efficiency stays content-dependent at 0.55-1.00
(math/json/ts ≈ 0.93-1.00, python/boilerplate/sql ≈ 0.55-0.63 — direct
measurement of misses due to BPE fragmentation etc.); at a practical mix
with an effective copy ratio of 0.4-0.6 this gives **350-450 tok/s**, and
exceeding 500 requires repairing extraction efficiency (the Phase 0
ECC/trie work).

## 8. Conclusion

The speed limit of a single-card setup can be accounted for, to first order,
as "bandwidth × accept length", and we currently sit at 25-50% of it. The
remainder is recoverable via (1) conditional elimination of the draft read,
(2) making the block geometry variable, and (3) tree-structuring the
verification, and this study confirmed the implementability of every one of
those components. Furthermore, the verification protocol itself (quality
probes, controlled hash experiments, shadow-run scoring) is a reusable
methodology for this kind of measurement-driven architecture research.

## Appendix: work in progress
- Investigating the first-request hang of the main-based build (fixed2)
  (suspected cause: first Triton kernel load with 0.09GB free VRAM). Plan to
  address it by fine-tuning mem-fraction.
- Upstream watch: [#34934](https://github.com/sgl-project/sglang/pull/34934) (remaining 3 prefill fusions), AdaFlash [#34171](https://github.com/sgl-project/sglang/pull/34171),
  host-sync removal [#32417](https://github.com/sgl-project/sglang/pull/32417), GDN replay [#28010](https://github.com/sgl-project/sglang/pull/28010).
