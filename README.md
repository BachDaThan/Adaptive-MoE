# adaptive-moe

[![tests](https://github.com/BachDaThan/Adaptive-MoE/actions/workflows/tests.yml/badge.svg)](https://github.com/BachDaThan/Adaptive-MoE/actions/workflows/tests.yml)

**English** | [Tiếng Việt](README.vi.md)

A small (experimental) PyTorch module for **adaptive** Mixture-of-Experts: the number of experts activated per token, and how strongly each one contributes, change with token difficulty instead of being fixed at top-6 or top-12.

Anyone training an MoE model can swap their MoE layer for `AdaptiveMoE`.

> **Status: experimental.** The mechanism works (the router learns token difficulty and uses more experts on harder tokens), but it does not beat simple baselines: adaptive routing was slightly worse than fixed top-k at equal budget in most settings, and in a low-budget setting it only matched a random per-token choice of k at the same average budget. See [Experimental results](#experimental-results-status-experimental).
>
> What does work is **elastic training**: training with a random k (and/or a random context window) gives one model that runs at any budget, with no loss when data is plentiful. When data is scarce it beats a plain fixed-k model; it loses to a fixed-k model with dropout if the elastic model has none, but elastic combined with dropout beats fixed-k with the same dropout by about 0.02 nats; once dropout is tuned for each, elastic is still ahead by about 0.013 nats at k=2 with longer training (6000 steps, all 4 seeds; a tie at 3000 steps) and by about 0.024 nats at k=1. See [Elastic training](#elastic-training-one-model-many-budgets).

## Recommended recipe (what the experiments support)

For training a small MoE language model on limited data and a free Colab T4. The evidence comes from models of about 20M parameters; larger sizes are not validated.

1. **Train with elastic k, choose k at inference.** Each step, draw a fraction f from {0, 0.25, 0.5, 0.75, 1} of tokens to use k=2 and let the rest use k=1 (`set_k_cap(model, tensor)`). One model then serves both budgets. With plentiful data it matches separately trained fixed-k models; with scarce data it was better by about 0.013 nats at k=2 and 0.024 nats at k=1 (section 4.2).
2. **Use dropout 0.15 to 0.2** on the embeddings and the MoE outputs when data is scarce. It is the biggest lever measured (about 0.08 nats); 0.3 is too much.
3. **Keep the best checkpoint by validation loss.** Models overfit soon after 3000 steps at small data sizes.
4. **A shared byte-level BPE** for several languages costs little (Vietnamese needs only 3% more tokens when English shares a 16k vocabulary, section 7).
5. **Skip** adaptive top-p routing, per-layer k schedules and smart per-token k allocation: none of them beat a plain fixed k at equal budget (sections 5 and the experimental results).
6. `train_lm.py` implements this recipe with fp16, gradient accumulation, checkpoints on Google Drive and automatic resume.

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

- **Elastic k:** train with a random cap on the number of experts, then choose k at inference. With plentiful data, one elastic model is within about 0.0015 nats of separate fixed-k models. With scarce data it beats plain fixed-k training (about 0.03 to 0.04 nats at the early-stopping step, 0.08 nats at the final step) but loses to fixed-k with dropout by about 0.06 nats if the elastic model has no dropout; elastic combined with dropout beats fixed-k with the same dropout by about 0.02 nats; once dropout is tuned for each, elastic is still ahead by about 0.013 nats at k=2 with longer training (6000 steps, all 4 seeds; a tie at 3000 steps) and by about 0.024 nats at k=1 (section 4).
- **KV cache and context window:** generation with a KV cache matches re-running the whole sequence exactly; a sink + sliding window caps the cache at `sink + W` tokens with almost no quality loss (sections 2 and 3).
- **Smarter k allocation does not beat a single fixed k at the same average cost.** There is only a small effect in the cost range between k=1 and k=2 (section 5). A per-layer k schedule that looked good on an elastic model **does not hold up** when trained from scratch, and it does not beat uniform k with dropout either (sections 5.3 and 4.2).
- **Several languages in one model (section 7, 1 seed):** a shared 16k byte-level BPE costs Vietnamese only 3% more tokens, and the router's expert usage differs strongly between Vietnamese and English without any supervision.

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
| `langcheck.py`                                                                   | One shared byte-level BPE for several languages; per-language token counts, loss and router usage (section 7). |
| `train_lm.py`                                                                    | Train a real MoE LM with the recipe above (elastic k, dropout, fp16, Drive checkpoints, resume, `--generate`). |
| `run_both.sh`, `run_ctx.sh`, `run_tokenk.sh`, `run_layerk.sh`, `run_schedule.sh`, `run_reg.sh`, `run_reg2.sh`, `run_reg3.sh`, `run_reg4.sh` | Run the scripts above end to end.                                        |

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
bash run_reg4.sh B 2 3    # round 4B: seeds 2 and 3 at 6000 steps (see the script header for parts A, C, D)
python langcheck.py --langs vi,en   # section 7: shared tokenizer, per-language router stats
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
- Most dropout runs reach their best at step 2750 to 3000 (the end of training), so 3000 steps is still not enough for this group and the gaps may change with longer runs (tested in round 3B).

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
- **At k=2 and 3000 steps, once each side has tuned dropout, elastic and fixed are level (round 3B changes this).** 4.4844 (elastic, dropout 0.2) vs 4.4923 (fixed, dropout 0.2): ahead by 0.008 nats, but with opposite signs across the two seeds (+0.002 and −0.017), so they cannot be told apart. The 0.025-nat advantage at dropout 0.1 (round 2) mostly came from fixed k being under-regularized at that level.
- **At k=1, elastic still wins clearly.** It beats fixed k by 0.054 nats at the same dropout of 0.2, and by 0.030 nats when comparing each side's best value tried (elastic 0.2, fixed 0.1), with the same sign in both seeds. A possible reason (untested): a fixed k=1 model has little capacity and does not tolerate strong dropout, while the k=1 pass of the elastic model shares weights trained at k=2.
- **Dropout 0.3 is too strong for both**, and hurts elastic more (0.016 nats behind fixed at k=2).
- One elastic model with dropout 0.2 gives 4.4844 at k=2 and 4.5070 at k=1: level with the best fixed-k2 model and better than the best fixed-k1 model, so one model serves both budgets with no quality loss. Every run in this round is best at step 3000, except one at 2750.

**Round 3B: the dropout group trained for 6000 steps (2 seeds).** Same setup with `--steps 6000` (row names suffixed `-6k`), run with `run_reg3.sh B`. Evaluation is still every 250 steps, the best step is chosen on half A, and `finalB` is the loss at step 6000.

| Configuration       | bestB (early stop) | finalB (step 6000) | Best step (seed 0, 1) |
|---------------------|--------------------|--------------------|-----------------------|
| elastic+drop2-6k@k2 | **4.4699**         | 4.5031             | 5000, 3750            |
| elastic+drop-6k@k2  | 4.4854             | 4.6540             | 3000, 2750            |
| fixed-k2+drop2-6k   | 4.4870             | 4.5520             | 4500, 3750            |
| elastic+drop2-6k@k1 | 4.4956             | 4.5308             | 5000, 3750            |
| fixed-k2+drop-6k    | 4.5068             | 4.7560             | 2750, 2500            |
| elastic+drop-6k@k1  | 4.5122             | 4.6914             | 3000, 2750            |

Per-seed `bestB` differences (negative = left side is better):

| Comparison                                                  | seed 0  | seed 1  | mean    |
|-------------------------------------------------------------|---------|---------|---------|
| elastic+drop-6k@k2 − fixed-k2+drop-6k                       | −0.0225 | −0.0202 | −0.0213 |
| elastic+drop2-6k@k2 − fixed-k2+drop2-6k                     | −0.0201 | −0.0143 | −0.0172 |
| elastic+drop-6k@k2 − fixed-k2+drop2-6k                      | −0.0054 | +0.0021 | −0.0016 |
| elastic+drop2-6k@k2 − elastic+drop2@k2 (6000 vs 3000 steps) | −0.0281 | −0.0009 | −0.0145 |
| fixed-k2+drop2-6k − fixed-k2+drop2 (6000 vs 3000 steps)     | −0.0064 | −0.0040 | −0.0052 |

- **Dropout 0.1 has converged by about 3000 steps.** The best step stays at 2500 to 3000 and bestB barely moves compared with the 3000-step runs (fixed 4.5122 → 4.5068, elastic 4.4870 → 4.4854). After that both overfit heavily (finalB 4.65 to 4.76).
- **Dropout 0.2 benefits from longer training** (best steps 3750 to 5000), elastic more than fixed (4.4844 → 4.4699 vs 4.4923 → 4.4870). The elastic improvement comes mostly from seed 0 (0.028 vs 0.001 in seed 1), so it is not firmly established.
- **At 6000 steps elastic beats fixed k at the same dropout in both seeds:** by about 0.021 nats at dropout 0.1 and 0.017 nats at dropout 0.2. At 3000 steps the gap at dropout 0.2 was only 0.008 nats with opposite signs across seeds, so the round 3A conclusion that the two are level at k=2 holds only at 3000 steps.
- **Elastic degrades more slowly after its best step.** `finalB − bestB` at dropout 0.2 is 0.033 (elastic) vs 0.065 (fixed), and at dropout 0.1 it is 0.168 vs 0.249, smaller in both seeds. When early stopping is not available, elastic is the safer choice.
- The k=1 mode of the best elastic model (4.4956) is only 0.009 nats behind a separately trained fixed k=2 model (4.4870) while using half as many routed experts.
- Limits: 2 seeds, one elastic dropout-0.2 run still peaks at step 5000 (it may not be fully converged), round 4 adds seeds 2 and 3 and fixed k=1 at 6000 steps.

**Round 4: seeds 2 and 3, fixed k=1 at 6000 steps, dropout 0.15.** Run with `run_reg4.sh`. The tables below pool seeds 0 to 3 (4 seeds) except where marked; the 3000-step rows of the earlier rounds that had 2 seeds now have 4.

6000 steps (4 seeds):

| Configuration       | bestB (early stop) | finalB (step 6000) | Best step (seeds 0, 1, 2, 3) |
|---------------------|--------------------|--------------------|------------------------------|
| elastic+drop2-6k@k2 | **4.4637**         | 4.5043             | 5000, 3750, 3750, 4500       |
| fixed-k2+drop2-6k   | 4.4763             | 4.5409             | 4500, 3750, 3750, 4500       |
| elastic+drop2-6k@k1 | 4.4920             | 4.5323             | 5000, 3750, 4250, 4500       |
| fixed-k1+drop2-6k   | 4.5162             | 4.5221             | 5500, 5250, 5250, 5000       |
| fixed-k1+drop-6k    | 4.5335             | 4.5805             | 4750, 3750, 3750, 4000       |

Per-seed `bestB` differences at 6000 steps (negative = left side is better):

| Comparison                                              | seed 0  | seed 1  | seed 2  | seed 3  | mean    |
|---------------------------------------------------------|---------|---------|---------|---------|---------|
| elastic+drop2-6k@k2 − fixed-k2+drop2-6k                 | −0.0201 | −0.0143 | −0.0076 | −0.0084 | −0.0126 |
| elastic+drop2-6k@k1 − fixed-k1+drop2-6k                 | −0.0306 | −0.0097 | −0.0387 | −0.0179 | −0.0242 |
| elastic+drop2-6k@k1 − fixed-k1+drop-6k                  | −0.0635 | −0.0363 | −0.0368 | −0.0294 | −0.0415 |
| fixed-k1+drop2-6k − fixed-k1+drop-6k                    | −0.0329 | −0.0266 | +0.0019 | −0.0115 | −0.0173 |
| elastic+drop2-6k@k2 − elastic+drop2@k2 (6000 vs 3000)   | −0.0281 | −0.0009 | −0.0240 | −0.0080 | −0.0153 |
| fixed-k2+drop2-6k − fixed-k2+drop2 (6000 vs 3000)       | −0.0064 | −0.0040 | −0.0129 | +0.0099 | −0.0034 |

3000 steps (4 seeds), per-seed `bestB` differences:

| Comparison                         | seed 0  | seed 1  | seed 2  | seed 3  | mean    |
|------------------------------------|---------|---------|---------|---------|---------|
| elastic+drop@k2 − fixed-k2+drop    | −0.0390 | −0.0116 | −0.0258 | −0.0129 | −0.0223 |
| elastic+drop@k1 − fixed-k1+drop    | −0.0266 | −0.0064 | −0.0084 | −0.0313 | −0.0182 |
| elastic+drop2@k2 − fixed-k2+drop2  | +0.0016 | −0.0174 | +0.0035 | +0.0095 | −0.0007 |
| elastic+drop2@k1 − fixed-k1+drop2  | −0.0553 | −0.0527 | −0.0688 | −0.0512 | −0.0570 |
| elastic+drop2@k1 − fixed-k1+drop   | −0.0277 | −0.0312 | −0.0330 | −0.0374 | −0.0323 |
| fixed-k2+drop2 − fixed-k2+drop     | −0.0324 | −0.0076 | −0.0442 | −0.0313 | −0.0289 |
| fixed-k1+drop2 − fixed-k1+drop     | +0.0276 | +0.0215 | +0.0358 | +0.0138 | +0.0247 |
| elastic+drop2@k2 − elastic+drop@k2 | +0.0082 | −0.0134 | −0.0149 | −0.0089 | −0.0073 |

Dropout sweep at 3000 steps (seeds 0 and 1 only, because 0.15 was run for those seeds; `bestB`):

| Dropout | fixed-k1 | fixed-k2 | elastic@k1 | elastic@k2 |
|---------|----------|----------|------------|------------|
| 0       | 4.6390   | 4.6045   | 4.5980     | 4.5707     |
| 0.1     | 4.5365   | 4.5122   | 4.5199     | 4.4870     |
| 0.15    | 4.5416   | 4.4889   | 4.5003     | **4.4753** |
| 0.2     | 4.5610   | 4.4923   | 4.5070     | 4.4844     |
| 0.3     | not run  | 4.5194   | 4.5561     | 4.5350     |

At dropout 0.15 (seeds 0 and 1): elastic@k2 − fixed-k2 is −0.0198 and −0.0075 (mean −0.0136); elastic@k1 − fixed-k1 is −0.0371 and −0.0456 (mean −0.0413).

- **At k=1 elastic wins everywhere, in every seed.** At 3000 steps it beats fixed k=1 at the same dropout by 0.018 nats (dropout 0.1), 0.041 (0.15, 2 seeds) and 0.057 (0.2). At 6000 steps it beats fixed k=1 by 0.024 nats at fixed's best dropout (0.2) and by 0.042 at dropout 0.1. The best dropout for fixed k=1 depends on training length: 0.1 at 3000 steps (0.2 is 0.025 worse, 4 of 4 seeds), 0.2 at 6000 steps (0.017 better, 3 of 4 seeds).
- **At k=2 the gain depends on training length and dropout.** At 3000 steps it is 0.022 nats at dropout 0.1 (4 of 4 seeds), 0.014 at 0.15 (2 of 2 seeds) and essentially zero at 0.2 (mean −0.0007, mixed signs). At 6000 steps with dropout 0.2 it is 0.013 nats in all 4 seeds (range 0.008 to 0.020). The reason is that elastic keeps improving with longer training (0.015 nats from 3000 to 6000 steps, all 4 seeds) while fixed k=2 barely does (0.003, mixed signs).
- **The best dropout values** (seeds 0 and 1, 3000 steps): fixed k=2 is about level between 0.15 and 0.2 (4.4889 vs 4.4923), elastic@k2 prefers 0.15 (4.4753 vs 4.4844, 0.009 better), fixed k=1 prefers 0.1.
- **Elastic degrades more slowly after its best step.** At 6000 steps `finalB − bestB` is 0.041 for elastic@k2 against 0.065 for fixed k=2. Fixed k=1 with dropout 0.2 degrades hardly at all (0.006) because it is still improving at steps 5000 to 5500.
- The k=1 mode of the best elastic model (4.4920) is 0.016 nats behind a separately trained fixed k=2 model (4.4763) while using half as many routed experts; the 2-seed estimate of 0.009 above was optimistic.

Practical takeaway (4 seeds): when data is scarce, dropout is the much bigger lever (about 0.08 to 0.09 nats). On top of tuned dropout, elastic training gives a gain that is reliable at k=1 (about 0.024 nats against the best fixed k=1 at 6000 steps, and 0.057 at equal dropout 0.2 at 3000 steps; every seed) and smaller at k=2 (about 0.013 nats at 6000 steps in all 4 seeds, but a tie at 3000 steps with dropout 0.2). It also keeps improving with longer training and degrades more slowly after its best step. The best configuration measured is one elastic model with dropout 0.2 trained for 6000 steps (4.4637 at k=2, 4.4920 at k=1), and its main value is one model that serves several budgets (section 1). Not done yet: elastic with k=4, dropout 0.05 for k=1, training fixed k=1 longer than 6000 steps (it is still improving), more seeds.

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

**Conclusion of this section.** A single fixed k is still the baseline to beat. The effects found are small (about 0.01 nats or less) and only lie in the cost range between k=1 and k=2. The robust result is that elastic training gives one model that runs at many budgets without losing quality (section 1); its advantage when data is scarce is much smaller than that of dropout, but it stacks with dropout, and keeps a gain of about 0.013 nats at k=2 (6000 steps; a tie at 3000 steps) and about 0.024 nats at k=1 (6000 steps) once dropout is tuned for each, over 4 seeds (section 4.2).

### 6. Measured speed (CUDA, batch 32x128; ratio relative to using all experts)

| Experts | k          | inference             | training              |
|---------|------------|-----------------------|-----------------------|
| 8       | 1 / 2 / 4  | 0.42x / 0.48x / 0.66x | 0.43x / 0.50x / 0.66x |
| 32      | 1 / 8 / 16 | 0.43x / 0.48x / 0.62x | 0.18x / 0.36x / 0.56x |

Using fewer experts is faster in practice, but not proportionally at inference (with 32 experts, k=1 reaches 0.43x, not 1/32). The fixed cost per step (attention, embedding, dispatch) is a large share when k is small. In the 4-layer BPE runs above, real training time barely depends on k (283 s at k=1, 289 s at k=2, 327 s at k=4), so do not assume compute savings at that model size.

### 7. One shared tokenizer for several languages (`langcheck.py`)

Question: can one small tokenizer plus the MoE router serve several languages? Setup: Vietnamese and English Wikipedia (`wikimedia/wikipedia`, 20231101), articles of at least 1000 characters (shorter ones are mostly templated stubs, which made the Vietnamese loss meaningless in a first run), about 8 MB per language, split by article (10% validation). One ByteLevel BPE (`tokenizers` library, so no out-of-vocabulary tokens) with vocabulary 16000, trained on both languages. The model is the one from section 4 (4 layers, d=128, 32 experts, sequence length 128, dropout 0.2, 3000 steps), trained on the same number of tokens per language (1.63M each). Two variants, 1 seed: fixed k=2, and adaptive k (`top_p=0.7`, max k=8, no compute penalty). The early-stopping step is chosen on half A of each language's validation set and reported on half B. Run with `python langcheck.py --langs vi,en`.

Tokenization, on validation text: tokens per 100 characters (tokens per word in brackets).

| Language   | Vietnamese-only BPE (16k) | Shared BPE (16k) | Qwen2.5 tokenizer (151k) |
|------------|---------------------------|------------------|--------------------------|
| Vietnamese | 28.7 (1.41)               | 29.5 (1.45)      | 32.6 (1.60)              |
| English    | 34.6 (2.18)               | 26.2 (1.65)      | 24.5 (1.54)              |

Sharing costs Vietnamese 3% more tokens, while English needs 24% fewer than with a Vietnamese-only BPE and ends within 7% of Qwen's 151k-vocabulary tokenizer; Vietnamese needs about 10% fewer tokens than Qwen. A Vietnamese-only byte-level BPE spends about 2.2 tokens per English word, which is worse but far from the dozen tokens per word that a tokenizer falling back to single characters would use.

Model results (1 seed). Loss per token and bits per character should only be compared within a language, not between languages.

| Language   | Fixed k=2: loss/token | bits/char | Adaptive: loss/token | bits/char | Adaptive avg_k per layer (mean) |
|------------|-----------------------|-----------|----------------------|-----------|---------------------------------|
| Vietnamese | 4.3992                | 1.8711    | 4.3587               | 1.8539    | 6.64 4.78 4.62 3.44 (4.87)      |
| English    | 5.2137                | 1.9756    | 5.1907               | 1.9669    | 6.15 4.42 3.62 2.81 (4.25)      |

Router expert usage with fixed k=2: Jensen-Shannon divergence between the Vietnamese and English usage distributions per layer (the maximum possible is ln 2 = 0.693), next to the noise level (the same language, two halves of its validation set).

| Layer | Vietnamese vs English | noise (Vietnamese A−B) | noise (English A−B) |
|-------|-----------------------|------------------------|---------------------|
| 0     | 0.0083                | 0.0014                 | 0.0003              |
| 1     | 0.0855                | 0.0011                 | 0.0006              |
| 2     | 0.2709                | 0.0066                 | 0.0016              |
| 3     | 0.3639                | 0.0065                 | 0.0008              |

- **The router specializes by language without supervision.** In layers 1 to 3 the divergence is at least 40 times the noise level and in the last layer it reaches about half of the maximum; in layer 0 it is about 6 times the noise. In the last layer one expert takes 34% of the Vietnamese tokens (expert 14) and a different one takes 42% of the English tokens (expert 7). A caveat that has not been tested: part of this may only reflect that the two languages use different tokens (the router sees token embeddings), not a deeper language-level specialization.
- **Adaptive k differs little between languages, and not stably.** With two languages Vietnamese got 4.87 experts against 4.25 for English (13% more); with Chinese added the gap shrank to 4.45 against 4.33 (3%), so there is no reliable language effect on k. The averages are high because `top_p=0.7` with a flat router picks many experts, so they are not comparable with the k=2 budget of the elastic experiments.
- Adaptive k reaches a lower loss than fixed k=2 (by 0.04 nats in Vietnamese, 0.02 in English) but uses 2.1 to 2.4 times as many experts, so this says nothing about efficiency.
Three languages (Vietnamese, English, Chinese; same setup, 1.71M tokens per language, 1 seed):

| Language | Vietnamese-only BPE | Shared BPE (3 languages) | Qwen2.5 tokenizer |
|----------|---------------------|--------------------------|-------------------|
| Vietnamese | 28.7 (1.41)       | 30.5 (1.50)              | 32.6 (1.60)       |
| English    | 34.6 (2.18)       | 28.3 (1.78)              | 24.5 (1.54)       |
| Chinese    | 164.3             | 76.2                     | 72.7              |

(Tokens per 100 characters; Chinese has no per-word figure.) Compared with the two-language tokenizer, adding Chinese costs Vietnamese 3% and English 8% more tokens, and Chinese ends within 5% of Qwen's tokenizer. The model pays too: bits per character rise from 1.8711 to 1.9667 in Vietnamese and from 1.9756 to 2.0657 in English (about 5% more), the cost of sharing capacity with a third language. Chinese reaches 6.43 bits per character, which is not comparable to the other two.

Jensen-Shannon divergence of expert usage with fixed k=2 (noise in the last three columns):

| Layer | vi–en  | vi–zh  | en–zh  | vi A−B | en A−B | zh A−B |
|-------|--------|--------|--------|--------|--------|--------|
| 0     | 0.0070 | 0.0111 | 0.0143 | 0.0014 | 0.0003 | 0.0003 |
| 1     | 0.0998 | 0.1852 | 0.2256 | 0.0021 | 0.0005 | 0.0014 |
| 2     | 0.2119 | 0.4710 | 0.4762 | 0.0057 | 0.0006 | 0.0018 |
| 3     | 0.2566 | 0.4860 | 0.4322 | 0.0042 | 0.0012 | 0.0016 |

The router separates all three languages (at least 35 times the noise in layers 2 and 3). Pairs that involve Chinese (0.43 to 0.49) are about twice as far apart as Vietnamese and English (0.21 to 0.26), so languages with a different script separate more, which fits the token-identity caveat above. In the last layer the top expert takes 35% of Vietnamese tokens (expert 7), 42% of English (expert 23) and 39% of Chinese (expert 27). With adaptive k the average number of experts is 4.45 for Vietnamese, 4.33 for English and 4.75 for Chinese, and the loss is 0.026 to 0.031 nats lower than fixed k=2 while using 2.2 to 2.4 times as many experts.

- Not done yet: more seeds, and a control for token identity.

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

## Tự tạo data theo cỡ model (make_data.py auto) và engine C++ nhanh hơn

**Thư mục trên repo dataset** (tự tạo nếu chưa có, tên tính từ số tham số model: 49,9M → `50M`):
```
data/50M/1.txt, 2.txt, ...        văn bản train từ đầu (đoạn cách nhau bằng dòng trống)
Fine-tune/50M/1.txt, 2.txt, ...   hội thoại fine-tune, mỗi dòng {"messages":[...]}
```
`train_lm.py` tự nạp `data/<cỡ>/` (thành ngôn ngữ `gen`), `train_sft.py` tự vào `Fine-tune/<cỡ>/`; không thấy thư mục cỡ đó thì dùng cách cũ (gốc repo / `pretrain/`). Đổi tên thư mục: `--size_tag` / biến `MODEL_TAG`.

**make_data.py auto**: đặt biến môi trường rồi chạy `python make_data.py auto` (thêm `--plan` để chỉ xem kế hoạch). Nó đọc data đang có, chỉ tạo phần còn thiếu, đánh số tiếp (1.txt → 2.txt…) và up lên.
Biến: `HF_TOKEN`, `GEMINI_API_KEYS`, `HF_REPO`, `MODEL_PARAMS` (vd 50M) hoặc `MODEL_REPO` hoặc `MODEL_D/LAYERS/EXPERTS/VOCAB`, `FT_PAIRS` (tổng số cặp fine-tune; bỏ trống = params/2000), `PRETRAIN_TOKENS` (tổng token data train; bỏ trống = params×0,2; data train KHÔNG cần đặt "cặp"), `DATA_KINDS`, `WEB_SEARCH` (`off|wiki|search|both`), `WEB_TOPICS`, `MAX_FILES_PER_RUN`. Chi tiết ở đầu file `make_data.py`.
Nguồn internet: Wikipedia tiếng Việt (không cần key) và công cụ `google_search` của Gemini; data train chỉ được viết lại, không chép nguyên văn. Gemini nào không có `google_search` thì tự chuyển sang Wikipedia.

Nhiều nhà cung cấp LLM (`providers.py`): thứ tự mặc định **Cohere → OpenRouter → Mistral → Gemini** (Gemini hay quá tải nên đứng cuối). Đặt key bằng `COHERE_API_KEY1..`, `OPENROUTER_API_KEY1..`, `MISTRAL_API_KEY1..`, `GEMINI_API_KEY1..` (bao nhiêu cũng được); provider không có key tự bị bỏ qua, provider nào lỗi/hết quota/key chết thì tự rơi xuống provider sau, và provider phải chờ rate-limit lâu (> `SPILL_SECONDS`, mặc định 15s) thì nhường provider kế thay vì đứng đợi. Đổi thứ tự bằng `LLM_ORDER="cohere,openrouter,mistral,gemini"`; ép model bằng `COHERE_MODELS` / `OPENROUTER_MODELS` / `MISTRAL_MODELS` / `GEMINI_MODEL` (OpenRouter mặc định tự quét model `:free`). Cào web (`WEB_SEARCH=tavily|both`): **Tavily → Exa → Linkup → Firecrawl** (`TAVILY_API_KEY1..`, `EXA_API_KEY1..`, `LINKUP_API_KEY1..`, `FIRECRAWL_API_KEY1..`; đổi thứ tự bằng `SCRAPE_ORDER`). `google_search` vẫn chỉ có ở Gemini nên chỉ là dự phòng cuối.

**yui.cpp (engine C++)**: kernel int8 AVX2 / NEON / NEON+dotprod chọn lúc chạy, nhóm luồng riêng (`--threads N`, `--bench_threads`, `YUI_THREADS`), nạp bằng mmap, lấy mẫu top-k không sắp xếp cả từ vựng. `--exact` / `YUI_EXACT=1` tính bằng float như PyTorch để đối chiếu. `make test` chạy thử kernel; `make portable` build một file chạy mọi CPU.

**Cập nhật: .yui v2 (4-bit), key Gemini/Tavily, `--size 50M`**
- `python export_yui.py ... --quant q4` → expert 4-bit theo khối 32, attention/embedding/head giữ int8 (file nhỏ hơn int8 khoảng 1/3, chất lượng gần int8). `--quant q4all` → mọi ma trận 4-bit (nhỏ hơn int8 khoảng 44%, chất lượng giảm nhiều nhất). Lý thuyết 4-bit + hệ số fp16 mỗi 32 trọng số = 4,5 bit/trọng số nên KHÔNG thể đúng một nửa. File v1 (f32/int8) vẫn đọc được; app Y-App dùng được file v2 sau khi build lại bằng yui.cpp mới. Luôn chạy `verify_cpp.py` và chat thử trước khi dùng file q4.
- Key: `GEMINI_API_KEY1`, `GEMINI_API_KEY2`, ... (bao nhiêu cũng được, hoặc `GEMINI_API_KEYS` cách nhau dấu phẩy). Tavily (tuỳ chọn): `TAVILY_API_KEY1`, `TAVILY_API_KEY2`, ... xoay vòng; không có key vẫn chạy bằng Gemini (google_search) + Wikipedia. Trên Kaggle tự đọc Secrets cùng tên.
- Train + tạo data bằng MỘT lệnh: `train_lm.py --size 50M ...` tự chọn d/lớp/expert cho đúng ~50M tham số, tạo phần data còn thiếu vào `data/50M/` và `Fine-tune/50M/` (tối đa `--make_data_minutes` phút, mặc định 30; cần key Gemini, không có thì bỏ qua), rồi train bằng data đó. `make_data.py auto` tự nhận cỡ model từ `config.json` trong repo dataset nên không cần nhập tham số.

## Công cụ, sandbox và MCP cho model (yui_agent.py)

Model chỉ học MỘT cú pháp ngắn `<tool>tên đối_số</tool>` (mỗi công cụ một đối số chuỗi); kết quả quay lại thành lượt user `<result>...</result>` nên khuôn user/assistant của `train_sft.py` giữ nguyên. Việc nguy hiểm do **khung chạy** quyết định (không phải model): công cụ nào được dùng, sandbox, danh sách mạng cho phép (chặn IP nội bộ, kiểm tra cả khi chuyển hướng), giới hạn CPU/RAM/thời gian, xác nhận của người dùng, nhật ký `agent_audit.jsonl`. Nội dung web/MCP trả về bị coi là dữ liệu không tin cậy (thẻ `<tool>` trong đó bị vô hiệu hoá).

- Công cụ: `calc` (máy tính an toàn, không eval), `time`, `search` (Tavily nếu có key `TAVILY_API_KEY1..`, không thì Wikipedia vi), `fetch` (chỉ host trong danh sách cho phép), `read`/`write` (chỉ trong thư mục làm việc), `py` (chỉ chạy khi sandbox đủ mạnh).
- Sandbox 3 bậc tự dò: **strong** = bubblewrap (`apt install bubblewrap`): không mạng, không thấy file máy chủ, giới hạn tài nguyên; **medium** = unshare (không mạng nhưng vẫn đọc được file máy chủ); **weak** = chỉ giới hạn tài nguyên. Mặc định `py` cần bậc strong, không đủ thì tự tắt. Đây là lớp bảo vệ thực dụng, không thay container/VM khi chạy mã của người lạ.
- Co giãn: `agent.example.json` có `preset` (minimal / standard / open), `scale` (nhân mọi giới hạn tài nguyên: 0.5 = nửa, 4 = gấp bốn), `limits`, `net.allow`, `confirm`.
- MCP: kết nối server MCP (stdio `{"command":[...]}` hoặc HTTP `{"url":...}`) rồi **gán** (`bind`) công cụ MCP vào tên chuẩn để model nhỏ chỉ cần biết vài tên, vd `"bind":{"search":{"mcp":"tavily","tool":"tavily-search","args":{"query":"$arg"}}}`. `python yui_agent.py mcp-server` chạy chế độ MCP server để ứng dụng khác dùng công cụ + sandbox của Yui. MCP có sẵn để bật bằng `preset` (xem `mcp-presets`); đã thử thật với server chính thức `time` và `memory`.
- Chạy: `./yui --model m.yui --serve` rồi `python yui_agent.py chat --config agent.json`; xem trạng thái: `python yui_agent.py status`.
- Dạy model dùng công cụ: `python tool_data.py --n 1500 --tag 50M` (hoặc `DATA_KINDS=finetune,pretrain,tools` với `make_data.py auto`) tạo hội thoại calc/time/search khớp đúng cú pháp, không tốn quota Gemini. Train với `--L 256` trở lên (kết quả công cụ cần chỗ trong ngữ cảnh). Chỉ bật khi bot/app chạy qua `yui_agent.py`, nếu không model sẽ in thẻ `<tool>` thô.
- Giới hạn thật của model nhỏ: model ~50M học được một vài công cụ cố định (calc/time/search) với cú pháp cố định, không tự dùng được công cụ MCP tuỳ ý; vì vậy MCP được gán vào tên chuẩn. Muốn chắc chắn hơn nữa thì làm bước tiếp theo: giải mã ràng buộc cú pháp trong yui.cpp.

## Model lớn (1B–10B), tên model, MCP điểm cuối (cập nhật mới nhất)

- **Chạy trọn quy trình trên Kaggle**: `kaggle_pipeline.py` (tạo data → pretrain → fine-tune → xuất `.yui`), chỉnh `PARAMS`, `MODEL_REPO`, `DATA_REPO` ở đầu file. Giai đoạn data chỉ cần mạng + CPU: chạy riêng `STAGES=data` ở phiên không GPU để khỏi đốt giờ GPU. `train_sft.py` KHÔNG có `--L` (ngữ cảnh lấy từ `config.json` lúc pretrain, đặt `--L 256` ở `train_lm.py`).
- **Repo data và repo model có thể khác nhau**: data ở `BachDaThan/TrainAI-web` (`data/<cỡ>/`, `Fine-tune/<cỡ>/`), model ở repo riêng (`--hf_repo BachDaThan/TrainAI-1B`). Hai script train tự tìm data ở `--data_repo`, `HF_REPO`, `BachDaThan/TrainAI-web`, rồi `--hf_repo` (cái đầu tiên có thư mục đúng cỡ).
- **Tên model**: `--name yui-1b-v1` hoặc để trống để tự đặt `yui-<cỡ>-v<N>` (N lớn hơn mọi bản trong repo và không trùng repo nào của tài khoản HF). Lưu ở `<hf_repo>/models/<tên>/` (pretrain) và `models/<tên>-sft/` (fine-tune, có cả config + tokenizer). Tên được ghi vào `config.json` và `train_sft.py` thêm vài hội thoại để model tự biết tên mình (`--identity off` để tắt). Không đặt `--name` mà repo đã có `yui-<cỡ>-vN` thì TIẾP TỤC bản đó (phiên Kaggle mới), `--new_model` để tạo bản mới. Repo kiểu cũ (checkpoint ở gốc repo) vẫn chạy như cũ. `export_yui.py` bỏ trống `--dst` sẽ đặt tên file theo tên model.
- **Số head attention tự tăng theo cỡ** (mỗi head ≤ 128 chiều); model cũ giữ 4 head. `arch_for_params` chọn cấu hình cho 10M … 10B. Thời gian tạo data tự tính theo cỡ model và số key (`--make_data_minutes auto`).
- **Xuất model lớn**: `export_yui.py` đọc checkpoint bằng mmap và đổi từng tensor một nên không cần RAM bằng cả model. Engine C++ đã thử với file ngẫu nhiên 1B (q4, 0,57 GB, nạp 0,2 s, ~110 token/giây trên 1 lõi) và 7B (q4, 4,0 GB, chạy được trên máy 3 GB RAM nhờ mmap, chậm vì thiếu RAM).
- **MCP**: ở dự án này MCP là phần bạn tự bật; không có server bên thứ ba nào được gắn sẵn. Hai cách: (1) thiết lập sẵn `"mcp": {"memory": {"preset": "memory"}}` (`python yui_agent.py mcp-presets`), (2) nối bằng URL điểm cuối `{"ten": {"url": "https://..."}}`. Ngược lại, `python yui_agent.py mcp-http --port 8765` mở "Điểm cuối MCP" của Yui: in URL riêng (mỗi lần mở tạo địa chỉ mới, địa chỉ cũ vẫn hợp lệ), trạng thái Chưa kết nối / Đã kết nối, chỉ mở calc/time/search (muốn thêm công cụ khác phải tự bật), chống DNS rebinding, giới hạn tốc độ; thu hồi bằng `mcp-endpoints revoke`. Muốn truy cập từ ngoài máy thì dùng HTTPS (tunnel/reverse proxy).
