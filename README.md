# adaptive-moe

[![tests](https://github.com/BachDaThan/Adaptive-MoE/actions/workflows/tests.yml/badge.svg)](https://github.com/BachDaThan/Adaptive-MoE/actions/workflows/tests.yml)

**English** | [Tiếng Việt](README.vi.md)

A small (experimental) PyTorch module for **adaptive** Mixture-of-Experts: the number of experts activated per token, and how strongly each one contributes, change with token difficulty instead of being fixed at top-6 or top-12.

Anyone training an MoE model can swap their MoE layer for `AdaptiveMoE`.

> **Status: experimental.** The mechanism works (the router learns token difficulty and uses more experts on harder tokens), but it does not beat simple baselines: adaptive routing was slightly worse than fixed top-k at equal budget in most settings, and in a low-budget setting it only matched a random per-token choice of k at the same average budget. See [Experimental results](#experimental-results-status-experimental).
>
> What does work is **elastic training**: training with a random k (and/or a random context window) gives one model that runs at any budget, with no loss when data is plentiful. When data is scarce it beats a plain fixed-k model; it loses to a fixed-k model with dropout if the elastic model has none, but elastic combined with dropout beats fixed-k with the same dropout by about 0.02 nats; once dropout is tuned for each, the gap is about 0.01 nats at k=2 (within noise) and 0.03 nats at k=1. See [Elastic training](#elastic-training-one-model-many-budgets).

## Install

```
pip install git+https://github.com/BachDaThan/Adaptive-MoE
```

or copy `adaptive_moe.py` into your project (PyTorch is the only dependency).

## Components

| Component                    | What it does                                                                                                                                                                              |
|------------------------------|-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `AdaptiveRouter`             | Top-p routing: take experts by descending probability until their sum reaches `top_p`. Easy tokens use fewer experts, hard tokens use more.                                               |
| Strength head                | Each token gets a learned "strength" factor in `[min_strength, 1]` that scales the contribution of the chosen experts.                                                                    |
| `target_k`                   | Pulls the average number of experts toward a value you choose (e.g. 3.0); easier to control than a blunt penalty.                                                                         |
| `n_shared`                   | Always-on shared experts (DeepSeek style); keeps quality stable when the router picks a single expert.                                                                                    |
| Auxiliary losses             | Load balancing, compute penalty, z-loss. Collect them with `gather_aux_loss(model)`.                                                                                                      |
| `moe_stats(model)`           | Per-layer `avg_k` and `avg_strength` after each forward pass.                                                                                                                             |
| `cascade_forward`            | Inference: run a small tier first (e.g. 2 experts); if confidence is low, escalate to 4, then 8. Confidence: `maxprob` or `entropy`.                                                      |
| `adaptive_generate`          | Text generation with per-step tier selection.                                                                                                                                             |
| `diff_head` (experimental)   | The router gets a head that predicts token difficulty and sets `top_p` per token. See the results below.                                                                                  |
| `max_resident` + `release()` | Keep only N experts on the GPU (LRU); `release()` frees more after a reply.                                                                                                               |
| `set_k_cap(model, k)`        | Cap the number of experts per token (at inference or training). `k` is an integer for all tokens, or an integer tensor `[B*S]` to cap each token separately (version 0.3.1).             |

## Usage

```
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
- `adaptive_generate` has no KV cache (each step re-runs the whole sequence). `elastic_kv.py` has a KV-cache version (`adaptive_generate_kv`) on a small reference model; real models can follow the same approach.
- Cascade re-runs the forward pass when a step is hard, so those steps cost extra compute.
- `top_p`, `target_k` and the loss coefficients need tuning for your model and data.
- This is the router/layer only, not a complete model.

## Vietnamese language-model experiment

```
pip install torch datasets
python lm_experiment.py --mode adaptive --target_k 3.0   # adaptive
python lm_experiment.py --mode fixed --k 3               # fixed top-k baseline
python compare.py                                        # table of all r_*.json results
```

The script trains a small LM (syllable-level tokenization of Vietnamese Wikipedia) and prints validation loss, average experts per token, the correlation between per-token expert count and per-token loss, and the expert count per token category (punctuation, numbers, function words, proper nouns, rare words, content words). Report labels are printed in Vietnamese. Use `--text_file` to train on your own text.

Options: `--ste` (let task-loss gradients reach the expert-selection boundary) and `--diff_head` (the router predicts token difficulty, trained on the real token loss, then sets `top_p` per token within `[p_min, p_max]`). `--layer_k 1,2,2,3` sets a fixed k per layer. `--mode random` runs a random per-token k control, and `--no_renorm` keeps raw gate probabilities (Switch style) so top-1 routers still get a task gradient.

## Experimental results (status: experimental)

Setup: 19M-parameter LM, 4 layers, 8 experts per layer, syllable-level tokenization, Vietnamese Wikipedia (30M characters, about 7.5M tokens), 3000 steps. Fixed k=2 and adaptive with `compute_coef=0` were run with 3 seeds (mean and min–max range); all other configurations used one seed. Lower validation loss is better.

| Configuration                                                                           | avg_k | val loss                |
|-----------------------------------------------------------------------------------------|-------|-------------------------|
| fixed top-k, k=1 (renormalized weights)                                                 | 1.00  | 4.194                   |
| fixed top-1, raw gate probabilities (`--no_renorm`)                                     | 1.00  | 4.110                   |
| fixed top-2, raw gate probabilities (`--no_renorm`)                                     | 2.00  | 4.083                   |
| fixed top-k, k=2 (3 seeds)                                                              | 2.00  | **4.078** (4.077–4.080) |
| fixed per-layer k (1,2,2,3)                                                             | 2.00  | 4.115                   |
| fixed per-layer k (1,2,3,3)                                                             | 2.25  | 4.115                   |
| adaptive + `diff_head`, `compute_coef=0.5`                                              | 1.90  | 4.150                   |
| adaptive + `diff_head`, `diff_target=gain`, `compute_coef=0.5`                          | 1.88  | 4.148                   |
| adaptive + `diff_head`, `compute_coef=0` (3 seeds)                                      | 2.10  | 4.085 (4.081–4.088)     |
| adaptive + `diff_head`, `compute_coef=0`, low budget (3 seeds)                          | 1.46  | 4.090 (4.083–4.098)     |
| random per-token k in {1,2}, 46% of tokens get 2 (control, 1 seed)                      | 1.46  | 4.092                   |
| random-k training (50/50), then run at k=1 (`--eval_cascade`)                           | 1.00  | 4.096                   |
| random-k training (50/50), then run at k=2                                              | 2.00  | 4.082                   |
| 32 experts (routed d_ff=128, shared d_ff=512), fixed top-8                              | 8.00  | 4.113                   |
| 32 experts (routed d_ff=128, shared d_ff=512), adaptive + `diff_head`, `compute_coef=0` | 7.08  | 4.126                   |
| 32 experts (same), fixed top-7                                                          | 7.00  | 4.119                   |
| 32 experts (same), fixed top-6                                                          | 6.00  | 4.126                   |

[![Validation loss vs average experts: adaptive routing vs fixed top-k](docs/loss_vs_k.png)](docs/loss_vs_k.png)

Points below the fixed top-k line are better at the same budget. The renormalized k=1 point is a weak reference (a single renormalized expert gives the router no task gradient). With raw gate probabilities (`--no_renorm`) fixed k=1 reaches 4.110 (blue square), and the adaptive and random points at k≈1.46 sit within about 0.005 nats of the dashed straight line between that point and fixed k=2, i.e. a plain linear trade-off. The right panel spans only 0.015 nats.

Takeaways:

- The router learns token difficulty (correlation with the true loss around 0.4 to 0.5) and the number of experts rises with difficulty (correlation between k and loss from +0.26 to +0.37).
- **Learned allocation does not beat random allocation at equal budget.** At about 1.46 experts per token (8 experts), adaptive routing with `compute_coef=0` reached 4.090 (3 seeds: 4.083 to 4.098), while a control that gives each token k=1 or k=2 at random with the same average budget reached 4.092 (1 seed). Fixed k=1 with raw gate probabilities (`--no_renorm`) reaches 4.110; the renormalized k=1 baseline (4.194) was handicapped because a single renormalized expert gives the router no task gradient. Against that corrected reference, the adaptive and random points at k≈1.46 lie within about 0.005 nats of the straight line between k=1 (4.110) and k=2 (4.078): a plain linear trade-off of roughly 0.03 nats per routed expert, not a special gain. (With raw gate probabilities at both ends, fixed k=2 gives 4.083, so the reference line runs from 4.110 to 4.083, about 0.027 nats per routed expert; the adaptive and random points sit about 0.005 to 0.007 nats below it, comparable to seed noise.) Escalating tokens by the model's own uncertainty (entropy of a cheap k=1 pass) beat random escalation by only 0.0014 nats at a 50% budget (`--eval_cascade`), and the whole range between all-k=1 and all-k=2 spans just 0.014 nats, so there is almost nothing for a controller to gain. (The "oracle" column in that report is not an achievable ceiling: with hindsight it picks the better of two noisy per-token losses.) At about 2 experts per token (8 experts) and at 32 experts, adaptive routing was slightly worse than fixed top-k at equal budget (about 0.007 nats).
- Finer-grained experts (32 instead of 8, similar total parameters): a first run appeared to favor adaptive routing, but that was an artifact of the shared expert shrinking along with `d_ff`. With the shared expert kept wide (`--d_ff_shared 512`), fixed top-8 (4.113) beats adaptive (4.126), which uses about 11% fewer experts and spreads k widely (4.8 on the easiest quartile to 8.3 on the hardest). The fixed top-7 (4.119) and top-6 (4.126) controls settle it: adaptive at k=7.08 matches fixed top-6 while using more experts, so it loses at equal budget by about 0.007 nats, the same gap seen with 8 experts. The fixed curve is also very flat (about 0.0065 nats per expert between k=6 and k=8), which leaves little headroom for any allocation scheme at this scale. This 32-expert setup is also worse than the 8-expert baseline (4.078) at this short training length.
- A stable observation across seeds: when the router is free, deeper layers ask for more experts (layer 1 about 1.3 to 1.6, layer 4 about 2.7 to 2.8). However, hard-coding k per layer (1,2,2,3) was worse than uniform k, so this is not a ready-made recipe.
- "Hard token" does not mean "token that benefits from more experts": a loss-based label sends more experts to rare words and function words, while content words gain the most from an extra expert. A label based on the loss reduction from adding experts (`gain`) was too noisy for the difficulty head to learn.
- `compute_coef=0.5` acts as a helpful regularizer when data is small and the model overfits (1.9M tokens), but hurts once there is enough data.

Limitations: most configurations have a single seed, the model is small, training is short, and wall-clock speed was not measured (the per-expert loop is not optimized). Results at larger scale may differ.

## Elastic training: one model, many budgets

Summary of the experiments below (small model, short training; most configurations ran 1 to 3 seeds):

- **Elastic k:** train with a random cap on the number of experts, then choose k at inference. With plentiful data, one elastic model is within about 0.0015 nats of separate fixed-k models. With scarce data it beats plain fixed-k training (about 0.03 to 0.04 nats at the early-stopping step, 0.08 nats at the final step) but loses to fixed-k with dropout by about 0.06 nats if the elastic model has no dropout; elastic combined with dropout beats fixed-k with the same dropout by about 0.02 nats; once dropout is tuned for each, the gap is about 0.01 nats at k=2 (within noise) and 0.03 nats at k=1 (section 4).
- **KV cache and context window:** generation with a KV cache matches re-running the whole sequence exactly; a sink + sliding window caps the cache at `sink + W` tokens with almost no quality loss (sections 2 and 3).
- **Smarter k allocation does not beat a single fixed k at the same average cost.** There is only a small effect in the cost range between k=1 and k=2 (section 5). A per-layer k schedule that looked good on an elastic model **does not hold up** when trained from scratch, and it does not beat uniform k with dropout either (sections 5.3 and 4.2).

Run-to-run noise at the same seed (non-deterministic GPU) is about 0.002-0.003 nats, so only compare cells within the same table or the same run.

### Scripts

| Script                                                                           | What it does                                                                                           |
|----------------------------------------------------------------------------------|--------------------------------------------------------------------------------------------------------|
| `elastic_kv.py`                                                                  | Generation with a KV cache and confidence-based cascade (`adaptive_generate_kv`), with a self-test.    |
| `elastic_ctx.py`                                                                 | Sink + sliding window with a memory-capped KV cache; `--selftest`, `--train`, `--grid`.                |
| `headroom.py`                                                                    | Token-level estimate of how much a per-token k allocator could gain.                                   |
| `tokenk.py`                                                                      | Real per-token k running through the layers (needs `adaptive_moe.py` >= 0.3.1).                        |
| `layerk.py`                                                                      | Sweeps per-layer k schedules on one elastic model.                                                     |
| `schedule_train.py`                                                              | Trains each k schedule from scratch (the verification step for `layerk.py`).                           |
| `regcheck.py`                                                                    | Early-stopping and regularization control: fixed-k (+dropout, +weight decay) vs elastic (section 4.2). |
| `summarize_reg.py`                                                               | Merges `regcheck.py` logs (mean over seeds, pairwise comparisons).                                     |
| `run_both.sh`, `run_ctx.sh`, `run_tokenk.sh`, `run_layerk.sh`, `run_schedule.sh`, `run_reg.sh`, `run_reg2.sh`, `run_reg3.sh` | Run the scripts above end to end.                                        |

```
python elastic_ctx.py --selftest
python elastic_ctx.py --grid --top_p 0.999 --hf wikimedia/wikipedia --config 20231101.vi --field text --mb 8
python headroom.py --hf wikimedia/wikipedia --config 20231101.vi --field text --mb 8 --bpe --experts 32 --k_hi 2 --L 128
bash run_tokenk.sh
bash run_schedule.sh 0
bash run_reg.sh 0         # section 4.2, round 1 (~40 min per seed)
bash run_reg2.sh 0        # section 4.2, round 2 (~40 min per seed)
bash run_reg3.sh A 0      # round 3A: dropout 0.2 and 0.3 for elastic and fixed
bash run_reg3.sh B 0      # round 3B: dropout group trained for 6000 steps
```

### 1. Elastic k: train once with random k, choose k at inference

8 experts per layer, 19.3M parameters, 7.5M-token corpus, 3000 steps (`--mode random --uniform_k --k_lo 1 --k_hi 4 --eval_ks 1,2,3,4`). Validation loss when k is forced at inference:

| k                                    | 1      | 2      | 3      | 4      |
|--------------------------------------|--------|--------|--------|--------|
| Elastic (trained with k~U{1..4})     | 4.1014 | 4.0765 | 4.0734 | 4.0739 |
| Fixed top-k, trained and run at same k | 4.110* | 4.078  | 4.0720 | 4.0727 |

\* fixed top-1 with raw gate probabilities (`--no_renorm`). Fixed k=2 is the mean of 3 seeds; k=3 and k=4 are one seed each.

One elastic model is within about 0.0015 nats of separate fixed-k models at k=2..4, and at k=1 it is no worse than a separately trained top-1 model. Seed-to-seed noise is about 0.002 to 0.004 nats. Training with random k averages 2.5 experts per token, but evaluating with random k only reaches 4.082, so forcing a fixed k at inference is better than leaving k random.

32 experts per layer, 19.4M parameters, a smaller 1.98M-token corpus (so it overfits heavily), 3000 steps, trained with k~U{1..16}:

| k                   | 1      | 2      | 4      | 8      | 16     |
|---------------------|--------|--------|--------|--------|--------|
| Elastic, forced k   | 4.4088 | 4.3110 | 4.2762 | 4.2628 | 4.2593 |
| Fixed k=4 / k=8     |        |        | 4.2873 | 4.2908 |        |

Here the elastic model beats fixed-k at the same k (by 0.011 at k=4 and 0.028 at k=8), but these fixed models have no early stopping or dropout, and the small corpus overfits heavily, so read this together with section 4.2. Quality is flat from k=8 upward, and k=1 loses 0.15 nats relative to k=8.

### 2. KV cache for generation (`elastic_kv.py`)

Incremental decoding with a cache matches full recomputation: maximum logit difference below 1e-6 at k = 1, 2, 4, 8, and greedy generation produces exactly the same ids. With confidence-based cascade, old tokens in the cache keep the k they were computed with, so cascade output is not identical to re-running the whole sequence at one fixed k (by design).

### 3. Elastic context window (`elastic_ctx.py`)

A token at position $i$ only attends to positions $j$ with `j < sink` or `i - j < W` (sink = 4). At inference the KV cache keeps only `sink + W` tokens and gives exactly the same result as re-running the whole sequence with the same mask (checked by `--selftest`, which also checks that generating 200 tokens with W=16 keeps only 20 tokens in the cache). During training, W is drawn at random per sample. Setup: byte-level LM with 3.9M parameters (d=128, 4 layers, 8 experts), sequence length 256, 8 MB of Vietnamese Wikipedia (`wikimedia/wikipedia`, `20231101.vi`), 3000 steps, k=2.

Elastic window only, W forced at inference:

| Training              | W=8    | 16     | 32     | 64     | 128    | 256 (full) |
|-----------------------|--------|--------|--------|--------|--------|------------|
| Full context          | 1.2245 | 1.2059 | 1.1962 | 1.1913 | 1.1894 | 1.1890     |
| Random W per sample   | 1.2040 | 1.1955 | 1.1904 | 1.1883 | 1.1875 | 1.1876     |

Two runs of the same experiment (same seed) agree: random-W training helps most at small W (about 0.02 at W=8, 0.01 at W=16) and loses nothing at full context (within 0.0015).

Elastic k and W together (one model trained with k in {1,2,4,8} and W in {8..256}, `top_p=0.999` so the k cap actually binds). Each cell is **validation loss / decode tok/s** with a KV cache. KV tokens kept: 12 / 20 / 36 / 68 / 132 / 256.

| k     | W=8         | 16          | 32          | 64          | 128         | 256         |
|-------|-------------|-------------|-------------|-------------|-------------|-------------|
| **1** | 1.2215 / 79 | 1.2129 / 82 | 1.2083 / 81 | 1.2069 / 83 | 1.2059 / 81 | 1.2060 / 84 |
| **2** | 1.2099 / 79 | 1.2021 / 79 | 1.1973 / 80 | 1.1955 / 79 | 1.1947 / 78 | 1.1947 / 81 |
| **4** | 1.2091 / 71 | 1.2013 / 74 | 1.1965 / 73 | 1.1947 / 73 | 1.1939 / 73 | 1.1938 / 75 |
| **8** | 1.2092 / 67 | 1.2013 / 68 | 1.1964 / 66 | 1.1946 / 67 | 1.1938 / 66 | 1.1937 / 69 |

- Shrinking the cache from 256 to 68 tokens (W=64) costs about 0.0008 nats; to 36 tokens about 0.0026; to 20 tokens about 0.0074.
- k=1 loses about 0.011 nats relative to k=2. k=4 and k=8 are only about 0.001 better than k=2 but decode about 7% and 15% slower, so k=2 is the sweet spot for this small model. (With `top_p=0.9` the grid has a similar shape; the cap may not bind at k>=4 there.)
- At equal cache size, spending budget on k pays off more than on W: (k=2, W=16) beats (k=1, W=64) with a smaller cache.
- tok/s is dominated by Python overhead at this model size and is only indicative; the proper speed measurement is the batched CUDA table in section 6.

**Limitations:** one seed per configuration, sequence length 256, small model, byte-level tokens. At this length the model barely uses long context, so truncating the cache is almost free; there are no results for long sequences yet.

### 4. Elastic vs fixed training when data is scarce

Setup: BPE with vocab 16000 trained on the training split, 1.48M training tokens (8 MB of Vietnamese Wikipedia, split by random lines), 32 experts, 4 layers, d=128, sequence length 128, batch 32, 3000 steps (about 8 passes over the data), `top_p=0.999` so the cap actually binds. The elastic model is trained with a random per-token k in {1,2} (`tokenk.py`); fixed models are trained and run at a single k.

**4.1 Validation loss at the final step, no early stopping.** All numbers are validation loss at step 3000 on the same 40 batches (81,920 tokens). Fixed models were run with `schedule_train.py`, one seed.

| Model                                    | k=1                      | k=2                      | k=4         |
|------------------------------------------|--------------------------|--------------------------|-------------|
| Fixed, trained at that k (seed 0)        | 4.6575                   | 4.7058                   | 4.8201      |
| Elastic, forced k at eval (seed 0 / 1 / 2) | 4.6613 / 4.6491 / 4.6463 | 4.6236 / 4.6120 / 4.6122 | not trained |

- At the final step, fixed training with more active experts is worse: k=4 is 0.16 nats worse than k=1. Final training loss of the elastic runs is about 3.0 to 3.3 while validation loss is above 4.6, so this regime is data-starved and overfits. Section 4.2 shows that most of this gap is final-step overfitting.
- At the final step, the elastic model at k=2 beats fixed k=2 by 0.08 nats and beats the best fixed model (k=1) by 0.03 to 0.05 nats.

**4.2 Control with early stopping and regularization (2 seeds).** Section 4.1 compares elastic against fixed models with no early stopping or regularization, so it cannot separate "random k helps" from "it just overfits less". Here we add fixed-k baselines with dropout (`+drop`) and weight decay (`+wd`), on the same data and setup, seeds 0 and 1, 3000 steps. Dropout is 0.1, applied to the embeddings and to the MoE output of every layer; `+wd` uses weight decay 0.1 (all other configurations 0.01, AdamW, lr 1e-3). Every 250 steps the validation loss (20 batches) is measured on two independent halves of the validation set: the best step (early stopping) is chosen on half A, `bestB` is the half-B loss at that step, and `finalB` is the half-B loss at step 3000. During training, the elastic model sends a random fraction f of tokens to k=2 each step (f drawn from {0, 0.25, 0.5, 0.75, 1}) and the rest use k=1. Numbers are the mean of the two seeds. Run with `run_reg.sh` (`regcheck.py`). These numbers are not directly comparable to 4.1 because the evaluation set differs. The elastic model here has no dropout and no extra weight decay.

| Configuration | bestB (early stop) | finalB (step 3000) | Best step (seed 0, 1) |
|---------------|--------------------|--------------------|-----------------------|
| fixed-k2+drop | **4.5122**         | 4.5168             | 3000, 2500            |
| fixed-k1+drop | 4.5365             | 4.5365             | 3000, 3000            |
| fixed-k2+wd   | 4.5589             | 4.6795             | 2000, 2000            |
| elastic@k2    | 4.5707             | 4.6870             | 2000, 2000            |
| elastic@k1    | 4.5980             | 4.7202             | 2000, 2000            |
| fixed-k2      | 4.6045             | 4.7722             | 1750, 2000            |
| fixed-k1      | 4.6390             | 4.7113             | 2000, 2000            |

Per-seed `bestB` differences (negative = left side is better):

| Comparison                 | seed 0  | seed 1  | mean    |
|----------------------------|---------|---------|---------|
| elastic@k2 − fixed-k2      | −0.0397 | −0.0280 | −0.0338 |
| elastic@k1 − fixed-k1      | −0.0347 | −0.0473 | −0.0410 |
| elastic@k2 − fixed-k2+wd   | +0.0112 | +0.0123 | +0.0118 |
| elastic@k2 − fixed-k2+drop | +0.0415 | +0.0753 | +0.0584 |
| elastic@k1 − fixed-k1+drop | +0.0492 | +0.0739 | +0.0615 |

- **Fixed k with dropout beats elastic without dropout at the same k.** fixed-k2+drop beats elastic@k2 by about 0.058 nats and fixed-k1+drop beats elastic@k1 by about 0.062 nats, with the same sign in both seeds. Weight decay alone also beats elastic@k2 by about 0.012 nats in both seeds. The ranking is identical across the two seeds, except that elastic@k1 and fixed-k2 swap places (they differ by less than 0.001 in seed 1).
- **Elastic still beats unregularized fixed k:** by about 0.034 nats at k=2 and 0.041 nats at k=1 at the early-stopping step. But the 0.08-nat advantage at the final step (4.1) shrinks to less than half when stopping early, so most of that gap came from fixed models overfitting for longer. At step 3000, elastic@k1 is even about 0.009 nats worse than fixed-k1 (4.7202 vs 4.7113).
- **Early stopping and dropout change the ordering of k.** At the final step, fixed k=1 beats fixed k=2 (4.7113 vs 4.7722), but at the early-stopping step k=2 beats k=1 by about 0.034 nats (4.6045 vs 4.6390), and with dropout k=2 beats k=1 by about 0.024 nats (4.5122 vs 4.5365). So the "more experts is worse" conclusion in 4.1 is mostly final-step overfitting. k=4 has not been re-run with early stopping.
- The dropout runs reach their best at step 3000 in 3 of 4 runs (one at 2500), while the other configurations peak around step 2000, so 3000 steps may not be enough for the dropout group. Longer runs have not been tried to check this.

**Round 2: elastic with dropout, stronger dropout, k=4 and per-layer k (2 seeds).** Same setup; `+drop` is dropout 0.1, `+drop2` is dropout 0.2. The elastic model in this round is trained with dropout 0.1. `sched-1124` is per-layer k (1,1,2,4), `sched-4211` is (4,2,1,1), both with average cost 2.0; `sched-1122` is (1,1,2,2), cost 1.5. Run with `run_reg2.sh`.

| Configuration   | bestB (early stop) | finalB (step 3000) | Best step (seed 0, 1) |
|-----------------|--------------------|--------------------|-----------------------|
| elastic+drop@k2 | **4.4870**         | 4.4870             | 3000, 3000            |
| fixed-k2+drop2  | 4.4923             | 4.4896             | 3000, 2750            |
| fixed-k4+drop   | 4.5040             | 4.5307             | 2500, 2500            |
| sched-1122+drop | 4.5189             | 4.5256             | 3000, 2750            |
| elastic+drop@k1 | 4.5199             | 4.5185             | 3000, 2750            |
| sched-1124+drop | 4.5234             | 4.5234             | 3000, 3000            |
| sched-4211+drop | 4.5330             | 4.5317             | 3000, 2750            |

Per-seed `bestB` differences (negative = left side is better):

| Comparison                       | seed 0  | seed 1  | mean    |
|----------------------------------|---------|---------|---------|
| elastic+drop@k2 − fixed-k2+drop  | −0.0390 | −0.0116 | −0.0253 |
| elastic+drop@k1 − fixed-k1+drop  | −0.0266 | −0.0064 | −0.0165 |
| elastic+drop@k2 − elastic@k2     | −0.0805 | −0.0869 | −0.0837 |
| elastic+drop@k2 − fixed-k2+drop2 | −0.0066 | −0.0040 | −0.0053 |
| fixed-k2+drop2 − fixed-k2+drop   | −0.0324 | −0.0076 | −0.0200 |
| fixed-k4+drop − fixed-k2+drop    | −0.0228 | +0.0063 | −0.0082 |
| sched-1124+drop − fixed-k2+drop  | +0.0077 | +0.0145 | +0.0111 |
| sched-4211+drop − fixed-k2+drop  | +0.0171 | +0.0244 | +0.0208 |

- **Elastic stacks with dropout.** elastic+drop@k2 (4.4870) is the best configuration in every table. With the same dropout of 0.1, elastic beats fixed k by about 0.025 nats at k=2 and 0.017 nats at k=1, with the same sign in both seeds (the size varies a lot, from 0.012 to 0.039 at k=2). Adding dropout to the elastic model helps by 0.084 nats. So the "fixed k with dropout beats elastic" conclusion above only holds for an elastic model without dropout.
- **Stronger dropout narrows the gap.** fixed-k2+drop2 (dropout 0.2) beats fixed-k2+drop by 0.020 nats and trails elastic+drop@k2 (dropout 0.1) by only 0.005 nats, with the same sign in both seeds but within noise. Round 3A below runs elastic with dropout 0.2 and 0.3 to answer this.
- **k=4 is not clearly better than k=2.** fixed-k4+drop beats fixed-k2+drop by 0.008 nats at bestB but with opposite signs across the two seeds, and its finalB is worse (4.5307 vs 4.5168): it peaks at step 2500 and then rises again, i.e. overfitting returns when more experts are active.
- **Per-layer k still does not beat uniform k, even with dropout.** At cost 2.0, (1,1,2,4) is 0.011 nats worse than uniform k=2 and (4,2,1,1) is 0.021 nats worse, with the same sign in both seeds. (1,1,2,2) (cost 1.5) reaches 4.5189, about 0.005 nats below the straight line between fixed-k1+drop (4.5365) and fixed-k2+drop (4.5122), which is negligible. (4,2,1,1) in particular, which beat uniform k=2 at the final step without regularization (section 5.3), is now the worst schedule.
- Most dropout runs reach their best at step 2750 to 3000 (the end of training), so 3000 steps is still not enough for this group and the gaps may change with longer runs.

**Round 3A: dropout tuned for both (2 seeds).** `+drop3` is dropout 0.3. Elastic models are trained with dropout 0.2 and 0.3. Run with `run_reg3.sh A`. The table below merges earlier rounds; `bestB` is the mean of two seeds:

| Dropout | fixed-k1 | fixed-k2 | elastic@k1 | elastic@k2 |
|---------|----------|----------|------------|------------|
| 0       | 4.6390   | 4.6045   | 4.5980     | 4.5707     |
| 0.1     | 4.5365   | 4.5122   | 4.5199     | 4.4870     |
| 0.2     | 4.5610   | 4.4923   | 4.5070     | **4.4844** |
| 0.3     | not run  | 4.5194   | 4.5561     | 4.5350     |

Per-seed `bestB` differences (negative = left side is better):

| Comparison                          | seed 0  | seed 1  | mean    |
|-------------------------------------|---------|---------|---------|
| elastic+drop2@k2 − fixed-k2+drop2   | +0.0016 | −0.0174 | −0.0079 |
| elastic+drop2@k1 − fixed-k1+drop2   | −0.0553 | −0.0527 | −0.0540 |
| elastic+drop2@k1 − fixed-k1+drop    | −0.0277 | −0.0312 | −0.0295 |
| elastic+drop3@k2 − fixed-k2+drop3   | +0.0120 | +0.0191 | +0.0156 |
| elastic+drop3@k2 − elastic+drop2@k2 | +0.0464 | +0.0548 | +0.0506 |
| fixed-k2+drop3 − fixed-k2+drop2     | +0.0360 | +0.0183 | +0.0272 |
| fixed-k1+drop2 − fixed-k1+drop      | +0.0276 | +0.0215 | +0.0246 |
| elastic+drop2@k2 − elastic+drop@k2  | +0.0082 | −0.0134 | −0.0026 |

- **The best dropout differs between the two.** fixed-k2 is best at 0.2 (both 0.1 and 0.3 are worse). elastic@k2 is nearly flat between 0.1 and 0.2 (a difference of 0.003 with opposite signs across seeds) and clearly worse at 0.3. fixed-k1 is best at 0.1 among the values tried (0.2 is 0.025 nats worse), while elastic@k1 improves at 0.2.
- **At k=2, once each side has tuned dropout, elastic and fixed are level.** 4.4844 (elastic, dropout 0.2) vs 4.4923 (fixed, dropout 0.2): ahead by 0.008 nats, but with opposite signs across the two seeds (+0.002 and −0.017), so they cannot be told apart. The 0.025-nat advantage at dropout 0.1 (round 2) mostly came from fixed k being under-regularized at that level.
- **At k=1, elastic still wins clearly.** It beats fixed k by 0.054 nats at the same dropout of 0.2, and by 0.030 nats when comparing each side's best value tried (elastic 0.2, fixed 0.1), with the same sign in both seeds. A possible reason (untested): a fixed k=1 model has little capacity and does not tolerate strong dropout, while the k=1 pass of the elastic model shares weights trained at k=2.
- **Dropout 0.3 is too strong for both**, and hurts elastic more (0.016 nats behind fixed at k=2).
- One elastic model with dropout 0.2 gives 4.4844 at k=2 and 4.5070 at k=1: level with the best fixed-k2 model and better than the best fixed-k1 model, so one model serves both budgets with no quality loss. Every run in this round is best at step 3000, except one at 2750.

Practical takeaway: when data is scarce, dropout is the much bigger lever (about 0.08 to 0.09 nats). Elastic training adds a small, budget-dependent gain on top: within noise at k=2 once dropout is tuned for both (0.008 nats, opposite signs across seeds) and about 0.03 nats at k=1. The best configuration measured is one elastic model with dropout 0.2: it matches the best fixed-k2 model and beats the best fixed-k1 model, so its main value is one model that serves many budgets (section 1), not a better regularizer than dropout. Scripted but no results yet (`run_reg3.sh B`): the dropout group trained for 6000 steps. Not done yet: elastic with k=4, intermediate dropout values (0.05 for k=1, 0.15), more seeds.

### 5. Does smart k allocation beat a single fixed k?

Same BPE setup as section 4. Cost is the average number of routed experts per token per layer (shared experts and attention not counted). Because these experiments use an elastic-trained model, they measure how an elastic model uses its experts. Section 4 shows that with scarce data, fixed k=2 training is not better than k=1 at the final step, but with early stopping or dropout k=2 beats k=1.

**5.1 Token-level estimate (`headroom.py`).** Train an elastic model, measure per-token loss at k=1 and at a higher k, then choose which tokens get the higher k under a budget (random, entropy of the k=1 output, a small MLP on early or late hidden states, or a token-frequency rule). This is a token-level estimate and ignores effects through attention.

- k=1 vs k=4 (gain 0.103 nats): at the same average cost, a single fixed k beats every allocation between k=1 and k=4. At cost 2, fixed k=2 recovers 83% of the gain versus 33% (random), 38% (entropy) and 44% to 56% (learned predictors). At cost 3, fixed k=3 recovers 97% versus 66% to 84%.
- k=1 vs k=2 (gain 0.063 nats): percentage of the k=2 gain recovered when a fraction f of tokens gets k=2:

| f    | random | entropy | rare | common | learn_emb | learn_l1 | learn_fin |
|------|--------|---------|------|--------|-----------|----------|-----------|
| 0.1  | 9%     | 10%     | 19%  | 4%     | 17%       | 16%      | 25%       |
| 0.25 | 26%    | 26%     | 36%  | 15%    | 35%       | 38%      | 45%       |
| 0.5  | 50%    | 54%     | 64%  | 36%    | 64%       | 65%      | 72%       |
| 0.75 | 76%    | 83%     | 85%  | 64%    | 88%       | 86%      | 88%       |

The `rare` rule (escalate the rarest token types, which only needs a frequency count over the training set) matches the learned predictor on input embeddings (`learn_emb`) at no cost. The entropy rule has to finish a full k=1 pass first, so it costs 1 + 2f instead of 1 + f.

**5.2 Token rules running for real in the layers (`tokenk.py`).** The per-token k is passed into every MoE layer (`set_k_cap(model, tensor)`). Models are trained with a random per-token k in {1,2}; `rare - random` at the same fraction f of tokens raised to k=2 (negative = rare is better, in nats). The total gain from k=1 to k=2 in these models is 0.034 to 0.042 nats.

| seed                          | f=0.25  | f=0.5   | f=0.75  |
|-------------------------------|---------|---------|---------|
| 0                             | -0.0061 | -0.0069 | -0.0044 |
| 1                             | -0.0075 | -0.0051 | -0.0056 |
| 2                             | -0.0043 | -0.0062 | -0.0065 |
| 0, trained with the rare rule | -0.0122 | -0.0101 | -0.0079 |

The opposite rule (escalating the most common tokens) is always worse than random (about +0.005 to +0.006), so the effect is not noise. The effect is small (about 0.006 nats) and only fills the cost range between k=1 and k=2, which fixed top-k cannot reach because k is an integer.

**5.3 Per-layer k: not confirmed when trained from scratch.** An elastic model trained with k drawn independently per layer from {1,2,4}, sweeping all 81 schedules (`layerk.py`, selected on validation half A, reported on half B), suggested that early layers do not need a large k. For example schedule (1,1,2,4) beats uniform k=2 by 0.010 to 0.011 nats at equal cost on both seeds. When each schedule is trained from scratch (`schedule_train.py`, seed 0) this result **does not replicate**:

| per-layer k | cost | val loss | train (s) | vs uniform k=2 |
|-------------|------|----------|-----------|----------------|
| (1,1,1,1)   | 1.0  | 4.6575   | 283       | -0.0483        |
| (2,2,2,2)   | 2.0  | 4.7058   | 289       | 0              |
| (4,4,4,4)   | 4.0  | 4.8201   | 327       | +0.1143        |
| (1,1,2,2)   | 1.5  | 4.6634   | 286       | -0.0425        |
| (1,1,2,4)   | 2.0  | 4.7214   | 294       | +0.0155        |
| (1,1,4,4)   | 2.5  | 4.7509   | 296       | +0.0451        |
| (4,2,1,1)   | 2.0  | 4.6873   | 306       | -0.0186        |

The schedule that looked best on the elastic model is worse than uniform k=2, while the reversed control (4,2,1,1) is better. How an elastic model uses experts per layer is not a basis for choosing which schedule to train. In the first experiment above, fixed per-layer k was also worse than uniform k, so per-layer k schedules are not recommended. Note: this table is measured at the final step in the data-scarce regime without regularization, where high k overfits. Re-run with dropout and early stopping (section 4.2, round 2), (1,1,2,4) is 0.011 nats worse than uniform k=2 and (4,2,1,1) is 0.021 nats worse, so the earlier advantage of (4,2,1,1) in the table above was an overfitting effect.

**Conclusion of this section.** A single fixed k is still the baseline to beat. The effects found are small (about 0.01 nats or less) and only lie in the cost range between k=1 and k=2. The robust result is that elastic training gives one model that runs at many budgets without losing quality (section 1); its advantage when data is scarce is much smaller than that of dropout, but it stacks with dropout at equal dropout; once dropout is tuned for each, its gain is within noise at k=2 and about 0.03 nats at k=1 (section 4.2).

### 6. Measured speed (CUDA, batch 32x128; ratio relative to using all experts)

| Experts | k          | inference             | training              |
|---------|------------|-----------------------|-----------------------|
| 8       | 1 / 2 / 4  | 0.42x / 0.48x / 0.66x | 0.43x / 0.50x / 0.66x |
| 32      | 1 / 8 / 16 | 0.43x / 0.48x / 0.62x | 0.18x / 0.36x / 0.56x |

Using fewer experts is faster in practice, but not proportionally at inference (with 32 experts, k=1 reaches 0.43x, not 1/32). The fixed cost per step (attention, embedding, dispatch) is a large share when k is small. In the 4-layer BPE runs above, real training time barely depends on k (283 s at k=1, 289 s at k=2, 327 s at k=4), so do not assume compute savings at that model size.

## Benchmark

`python bench.py` measures real throughput (tokens per second, inference and training) when every token is forced to use exactly k experts, so you can check whether using fewer experts is actually faster on your hardware. Example: `python bench.py --experts 32 --d_ff 128 --ks 1,4,8,16,32`.

## Tests and demo

```
pip install torch pytest
pytest -q                  # or: python test_adaptive_moe.py
python train_demo.py
```

## License

MIT © 2026 BachDaThan, see [LICENSE](LICENSE).
