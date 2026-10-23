# adaptive-moe

![tests](https://github.com/BachDaThan/Adaptive-MoE/actions/workflows/tests.yml/badge.svg)

**English** | [Tiếng Việt](README.vi.md)

A small (experimental) PyTorch module for **adaptive** Mixture-of-Experts: the number of experts activated per token, and how strongly each one contributes, change with token difficulty instead of being fixed at top-6 or top-12.

Anyone training an MoE model can swap their MoE layer for `AdaptiveMoE`.

> **Status: experimental.** The mechanism works (the router learns token difficulty and uses more experts on harder tokens), but it does not beat simple baselines: adaptive routing was slightly worse than fixed top-k at equal budget in most settings, and in a low-budget setting it only matched a random per-token choice of k at the same average budget. See [Experimental results](#experimental-results).

## Install

```bash
pip install git+https://github.com/BachDaThan/Adaptive-MoE
```

or copy `adaptive_moe.py` into your project (PyTorch is the only dependency).

## Components

| Component | What it does |
|---|---|
| `AdaptiveRouter` | Top-p routing: take experts by descending probability until their sum reaches `top_p`. Easy tokens use fewer experts, hard tokens use more. |
| Strength head | Each token gets a learned "strength" factor in `[min_strength, 1]` that scales the contribution of the chosen experts. |
| `target_k` | Pulls the average number of experts toward a value you choose (e.g. 3.0). |
| `n_shared` | Always-on shared experts (DeepSeek style); keeps quality stable when the router picks a single expert. |
| Auxiliary losses | Load balancing, compute penalty, z-loss. Collect them with `gather_aux_loss(model)`. |
| `moe_stats(model)` | Per-layer `avg_k` and `avg_strength` after each forward pass. |
| `diff_head` (experimental) | The router gets a head that predicts token difficulty and sets `top_p` per token. See the results below. |
| `cascade_forward` | Inference: run a small tier first (e.g. 2 experts); if confidence is low, escalate to 4, then 8. Confidence: `maxprob` or `entropy`. |
| `adaptive_generate` | Text generation with per-step tier selection. |
| `max_resident` + `release()` | Keep only N experts on the GPU (LRU); `release()` frees more after a reply. |

## Usage

```python
from adaptive_moe import AdaptiveMoE, gather_aux_loss, moe_stats, adaptive_generate

# inside your model
self.moe = AdaptiveMoE(d_model=512, d_ff=1024, n_experts=16,
                       top_p=0.7, target_k=4.0, n_shared=1)
y, _, _ = self.moe(x)                      # x: [B, S, D]

# training
loss = task_loss + gather_aux_loss(model)
print(moe_stats(model))                    # {"layers.0.moe": {"avg_k": ..., "avg_strength": ...}}

# inference with per-step escalation
ids, ks = adaptive_generate(model, input_ids, max_new_tokens=64,
                            tiers=(2, 4, 8), min_confidence=0.6, confidence="entropy")
```

## Notes

- `release()` and `max_resident` are for inference (eval) only; do not enable them during training.
- `adaptive_generate` has no KV cache (each step re-runs the whole sequence). Real models should add their own KV cache; the tier-selection part stays the same.
- Cascade re-runs the forward pass when a step is hard, so those steps cost extra compute.
- `top_p`, `target_k` and the loss coefficients need tuning for your model and data.
- This is the router/layer only, not a complete model.

## Elastic budget experiments (k, KV cache, context window)

All runs below are single-seed. Run-to-run noise on the same seed (GPU nondeterminism) was about 0.002-0.003 nats, so only compare cells within one table or run. Commands are in the scripts' `--help`.

### 1. Elastic k: train once with random k, choose k at inference

8 experts/layer, 19.3M params, 7.5M-token corpus, 3000 steps. Validation loss with k forced at inference:

| k | 1 | 2 | 3 | 4 |
|---|---|---|---|---|
| Elastic (trained with k~U{1..4}) | 4.1014 | 4.0765 | 4.0734 | 4.0739 |
| Fixed k=3 / fixed k=4 (trained at that k) | | | 4.0720 | 4.0727 |

One elastic model is within about 0.0015 nats of the fixed-k models. Evaluating with random k (instead of forcing one k) gave 4.0817, so forcing a single k at inference is better.

32 experts/layer, 19.4M params, smaller 1.98M-token corpus (so heavily overfit), 3000 steps, train k~U{1..16}:

| k | 1 | 2 | 4 | 8 | 16 |
|---|---|---|---|---|---|
| Elastic, forced k | 4.4088 | 4.3110 | 4.2762 | 4.2628 | 4.2593 |
| Fixed k=4 / k=8 | | | 4.2873 | 4.2908 | |

Here the elastic model is better than fixed-k at the same k (by 0.011 at k=4 and 0.028 at k=8). With this little data random k probably acts as a regularizer, and the 8-expert run on more data shows no such gain, so this should not be read as "elastic beats fixed" in general. Quality is flat above k=8 and k=1 loses 0.15 nats versus k=8.

### 2. KV cache for generation (`elastic_kv.py`)

Cached incremental decoding matches full recomputation: max logit difference below 1e-6 at k = 1, 2, 4, 8, and greedy generation gives identical ids. With the confidence cascade, old cached tokens keep the k they were computed with, so cascade output is not identical to recomputing the whole sequence at one k (by design).

### 3. Elastic context window (`elastic_ctx.py`)

A token at position $i$ attends to positions $j$ with `j < sink` or `i - j < W` (sink = 4). At inference the KV cache keeps only `sink + W` tokens, and this gives outputs identical to re-running the whole sequence with the same mask (checked by `--selftest`). Setup: byte-level LM, 3.9M params, 8 experts, sequence length 256, 8 MB Vietnamese Wikipedia (`wikimedia/wikipedia`, `20231101.vi`), 3000 steps, k=2.

Window only, W forced at inference:

| Training | W=8 | 16 | 32 | 64 | 128 | 256 (full) |
|---|---|---|---|---|---|---|
| Full context | 1.2245 | 1.2059 | 1.1962 | 1.1913 | 1.1894 | 1.1890 |
| Random W per sample | 1.2040 | 1.1955 | 1.1904 | 1.1883 | 1.1875 | 1.1876 |

Random-W training helps most at small W (about 0.02 at W=8, 0.01 at W=16) and costs nothing at full context.

Joint elastic k and W (one model, k in {1,2,4,8}, W in {8..256}, `top_p=0.999` so the k cap binds).  
Each cell is **val loss / decode tok/s** with KV cache. KV tokens kept: 12 / 20 / 36 / 68 / 132 / 256.

| k | W=8 | 16 | 32 | 64 | 128 | 256 |
|---|---|---|---|---|---|---|
| **1** | 1.2215 / 79 | 1.2129 / 82 | 1.2083 / 81 | 1.2069 / 83 | 1.2059 / 81 | 1.2060 / 84 |
| **2** | 1.2099 / 79 | 1.2021 / 79 | 1.1973 / 80 | 1.1955 / 79 | 1.1947 / 78 | 1.1947 / 81 |
| **4** | 1.2091 / 71 | 1.2013 / 74 | 1.1965 / 73 | 1.1947 / 73 | 1.1939 / 73 | 1.1938 / 75 |
| **8** | 1.2092 / 67 | 1.2013 / 68 | 1.1964 / 66 | 1.1946 / 67 | 1.1938 / 66 | 1.1937 / 69 |

- Shrinking the cache from 256 to 68 tokens (W=64) costs about 0.0008 nats; to 36 tokens about 0.0026; to 20 tokens about 0.0074.
- k=1 costs about 0.011 nats versus k=2. k=4 and k=8 gain only about 0.001 over k=2 while decoding about 7% and 15% slower here, so k=2 is the sweet spot for this small model.
- At equal cache size, spending budget on k beats spending it on W: (k=2, W=16) is better than (k=1, W=64) with a smaller cache.
- tok/s is dominated by Python overhead at this model size and is only indicative; the batched CUDA benchmark below is the speed reference.

**Limitations:** one seed per configuration, sequence length 256, small models, byte-level tokens.

## Language-model experiment (Vietnamese text)

```bash
pip install torch datasets
python lm_experiment.py --mode adaptive --target_k 3.0   # adaptive
python lm_experiment.py --mode fixed --k 3               # fixed top-k baseline
python compare.py                                        # table of all r_*.json results
```

The script trains a small LM (syllable-level tokenization of Vietnamese Wikipedia) and prints validation loss, average experts per token, the correlation between per-token expert count and per-token loss, and the expert count per token category (punctuation, numbers, function words, proper nouns, rare words, content words). Report labels are printed in Vietnamese. Use `--text_file` to train on your own text.

Options: `--ste` (let task-loss gradients reach the expert-selection boundary) and `--diff_head` (the router predicts token difficulty, trained on the real token loss, then sets `top_p` per token within `[p_min, p_max]`). `--layer_k 1,2,2,3` sets a fixed k per layer. `--mode random` runs a random per-token k control, and `--no_renorm` keeps raw gate probabilities (Switch style) so top-1 routers still get a task gradient.

## Experimental results

Setup: 19M-parameter LM, 4 layers, 8 experts per layer, syllable-level tokenization, Vietnamese Wikipedia (30M characters, about 7.5M tokens), 3000 steps. Fixed k=2 and adaptive with `compute_coef=0` were run with 3 seeds (mean and min–max range); all other configurations used one seed. Lower validation loss is better.

| Configuration | avg_k | val loss |
|---|---|---|
| fixed top-k, k=1 (renormalized weights) | 1.00 | 4.194 |
| fixed top-1, raw gate probabilities (`--no_renorm`) | 1.00 | 4.110 |
| fixed top-2, raw gate probabilities (`--no_renorm`) | 2.00 | 4.083 |
| fixed top-k, k=2 (3 seeds) | 2.00 | **4.078** (4.077–4.080) |
| fixed per-layer k (1,2,2,3) | 2.00 | 4.115 |
| fixed per-layer k (1,2,3,3) | 2.25 | 4.115 |
| adaptive + `diff_head`, `compute_coef=0.5` | 1.90 | 4.150 |
| adaptive + `diff_head`, `diff_target=gain`, `compute_coef=0.5` | 1.88 | 4.148 |
| adaptive + `diff_head`, `compute_coef=0` (3 seeds) | 2.10 | 4.085 (4.081–4.088) |
| adaptive + `diff_head`, `compute_coef=0`, low budget (3 seeds) | 1.46 | 4.090 (4.083–4.098) |
| random per-token k in {1,2}, 46% of tokens get 2 (control, 1 seed) | 1.46 | 4.092 |
| random-k training (50/50), then run at k=1 (`--eval_cascade`) | 1.00 | 4.096 |
| random-k training (50/50), then run at k=2 | 2.00 | 4.082 |
| 32 experts (routed d_ff=128, shared d_ff=512), fixed top-8 | 8.00 | 4.113 |
| 32 experts (routed d_ff=128, shared d_ff=512), adaptive + `diff_head`, `compute_coef=0` | 7.08 | 4.126 |
| 32 experts (same), fixed top-7 | 7.00 | 4.119 |
| 32 experts (same), fixed top-6 | 6.00 | 4.126 |

![Validation loss vs average experts: adaptive routing vs fixed top-k](docs/loss_vs_k.png)

Points below the fixed top-k line are better at the same budget. The renormalized k=1 point is a weak reference (a single renormalized expert gives the router no task gradient). With raw gate probabilities (`--no_renorm`) fixed k=1 reaches 4.110 (blue square), and the adaptive and random points at k≈1.46 sit within about 0.005 nats of the dashed straight line between that point and fixed k=2, i.e. a plain linear trade-off. The right panel spans only 0.015 nats.

Takeaways:
- The router learns token difficulty (correlation with the true loss around 0.4 to 0.5) and the number of experts rises with difficulty (correlation between k and loss from +0.26 to +0.37).
- **Learned allocation does not beat random allocation at equal budget.** At about 1.46 experts per token (8 experts), adaptive routing with `compute_coef=0` reached 4.090 (3 seeds: 4.083 to 4.098), while a control that gives each token k=1 or k=2 at random with the same average budget reached 4.092 (1 seed). Fixed k=1 with raw gate probabilities (`--no_renorm`) reaches 4.110; the renormalized k=1 baseline (4.194) was handicapped because a single renormalized expert gives the router no task gradient. Against that corrected reference, the adaptive and random points at k≈1.46 lie within about 0.005 nats of the straight line between k=1 (4.110) and k=2 (4.078): a plain linear trade-off of roughly 0.03 nats per routed expert, not a special gain. (With raw gate probabilities at both ends, fixed k=2 gives 4.083, so the reference line runs from 4.110 to 4.083, about 0.027 nats per routed expert; the adaptive and random points sit about 0.005 to 0.007 nats below it, comparable to seed noise.) Escalating tokens by the model's own uncertainty (entropy of a cheap k=1 pass) beat random escalation by only 0.0014 nats at a 50% budget (`--eval_cascade`), and the whole range between all-k=1 and all-k=2 spans just 0.014 nats, so there is almost nothing for a controller to gain. (The "oracle" column in that report is not an achievable ceiling: with hindsight it picks the better of two noisy per-token losses.) At about 2 experts per token (8 experts) and at 32 experts, adaptive routing was slightly worse than fixed top-k at equal budget (about 0.007 nats).
- Finer-grained experts (32 instead of 8, similar total parameters): a first run appeared to favor adaptive routing, but that was an artifact of the shared expert shrinking along with `d_ff`. With the shared expert kept wide (`--d_ff_shared 512`), fixed top-8 (4.113) beats adaptive (4.126), which uses about 11% fewer experts and spreads k widely (4.8 on the easiest quartile to 8.3 on the hardest). The fixed top-7 (4.119) and top-6 (4.126) controls settle it: adaptive at k=7.08 matches fixed top-6 while using more experts, so it loses at equal budget by about 0.007 nats, the same gap seen with 8 experts. The fixed curve is also very flat (about 0.0065 nats per expert between k=6 and k=8), which leaves little headroom for any allocation scheme at this scale. This 32-expert setup is also worse than the 8-expert baseline (4.078) at this short training length.
- A stable observation across seeds: when the router is free, deeper layers ask for more experts (layer 1 about 1.3 to 1.6, layer 4 about 2.7 to 2.8). However, hard-coding k per layer (1,2,2,3) was worse than uniform k, so this is not a ready-made recipe.
- "Hard token" does not mean "token that benefits from more experts": a loss-based label sends more experts to rare words and function words, while content words gain the most from an extra expert. A label based on the loss reduction from adding experts (`gain`) was too noisy for the difficulty head to learn.
- `compute_coef=0.5` acts as a helpful regularizer when data is small and the model overfits (1.9M tokens), but hurts once there is enough data.

Limitations: most configurations have a single seed, the model is small, training is short, and wall-clock speed was not measured (the per-expert loop is not optimized). Results at larger scale may differ.

## Elastic k, uniform k in 1..4 (one seed)

Train once with `--mode random --uniform_k --k_lo 1 --k_hi 4 --eval_ks 1,2,3,4` (8 experts/layer, 19.3M params, 3000 steps, same data as above), then force k at inference.

| Model | k=1 | k=2 | k=3 | k=4 |
|---|---|---|---|---|
| Elastic (trained with k~U{1..4}), forced k at eval | 4.1014 | 4.0765 | 4.0734 | 4.0739 |
| Fixed top-k, trained and run at that k | 4.110\* | 4.0780 | 4.0720 | 4.0727 |

\* *fixed top-1 with raw gate probabilities (`--no_renorm`). Fixed k=2 is the 3-seed mean; k=3 and k=4 are one seed each.*

**Reading:** A single elastic model is within about 0.0015 nats of a dedicated fixed-k model at k=2..4, and at k=1 it is not worse than a dedicated top-1 model (4.101 vs 4.110). Seed-to-seed noise is about 0.002 to 0.004 nats, so only the k=1 row is clearly different from the fixed baselines' trend, and it needs more seeds. Quality is flat from k=2 to k=4 at this scale, so the budget knob matters mostly between k=1 and k=2.

Training with random k averaged 2.5 experts per token, but evaluated with random k it scored 4.082 (ppl 59.2); forcing a single k at inference is better than leaving k random.

## Elastic context window (sink + sliding window) and joint elastic k x W

**Setup:** byte-level LM, 3.9M params (d=128, 4 layers, 8 experts), sequence length 256, 8 MB of Vietnamese Wikipedia (`wikimedia/wikipedia 20231101.vi`), 3000 steps, sink = 4, top_p = 0.9. A token at position $i$ attends to positions $j$ with `j < sink` or `i - j < W`. At inference the KV cache keeps only `sink + W` tokens; this is checked to give identical outputs to re-running the whole sequence with the same mask (`--selftest`).

### Window only (k = 2): val loss when W is forced at inference

| Training | W=8 | 16 | 32 | 64 | 128 | 256 (full) |
|---|---|---|---|---|---|---|
| Full context | 1.2245 | 1.2059 | 1.1962 | 1.1913 | 1.1894 | 1.1890 |
| Random W per sample | 1.2040 | 1.1955 | 1.1904 | 1.1883 | 1.1875 | 1.1876 |

Two runs of this experiment (same seed, GPU nondeterminism) agree: random-W training helps most at small W (about 0.02 at W=8, 0.01 at W=16) and costs nothing at full context (difference within +-0.0015).

### Joint elastic k and W (one model trained with k in {1,2,4,8} and W in {8..256})

val loss (rows: k, columns: W; KV tokens kept: 12 / 20 / 36 / 68 / 132 / 256)

| k | W=8 | 16 | 32 | 64 | 128 | 256 |
|---|---|---|---|---|---|---|
| 1 | 1.2165 | 1.2095 | 1.2043 | 1.2023 | 1.2017 | 1.2018 |
| 2 | 1.2060 | 1.1992 | 1.1939 | 1.1922 | 1.1914 | 1.1915 |
| 4 | 1.2054 | 1.1988 | 1.1935 | 1.1917 | 1.1909 | 1.1910 |
| 8 | 1.2056 | 1.1989 | 1.1936 | 1.1918 | 1.1910 | 1.1911 |

A second run of the same grid was shifted up by about 0.0025 in every cell, so compare cells within one run.

**Observations:**
- Shrinking the cache from 256 to 68 tokens (W=64) costs about 0.0007 nats; to 36 tokens (W=32) about 0.0024; to 20 tokens (W=16) about 0.008.
- k=1 costs about 0.010 nats versus k=2. k=2, 4 and 8 are within 0.0005 of each other. With `top_p = 0.9` the router may already keep fewer experts than the cap, so the cap may not bind for k>=4; see "Open question".
- Spending budget on k is worth more than on W: (k=2, W=16) beats (k=1, W=64) with a smaller cache.
- Decode tok/s was flat (about 75-83) because this small model is dominated by Python overhead, so no speed claim is made here.

**Open question:** rerun with `--top_p 0.999` to check whether the k cap actually binds at k=4 and k=8.  
**Limitations:** single seed per configuration, sequence length only 256.

### Measured speed (CUDA, batch 32x128; ratio vs using all experts)

| Experts | k | inference | train |
|---|---|---|---|
| 8 | 1 / 2 / 4 | 0.42x / 0.48x / 0.66x | 0.43x / 0.50x / 0.66x |
| 32 | 1 / 8 / 16 | 0.43x / 0.48x / 0.62x | 0.18x / 0.36x / 0.56x |

Using fewer experts is faster in practice, but not proportionally at inference (32 experts, k=1 is 0.43x, not 1/32). Fixed per-step overhead (attention, embedding, dispatch) dominates at small k.

## Benchmark

`python bench.py` measures real throughput (tokens per second, inference and training) when every token is forced to use exactly k experts, so you can check whether using fewer experts is actually faster on your hardware. Example: `python bench.py --experts 32 --d_ff 128 --ks 1,4,8,16,32`.

## Tests and demo

```bash
pip install torch pytest
pytest -q                  # or: python test_adaptive_moe.py
python train_demo.py
```

## License

MIT © 2026 BachDaThan, see [LICENSE](LICENSE).
