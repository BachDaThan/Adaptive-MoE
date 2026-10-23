# adaptive-moe

![tests](https://github.com/BachDaThan/Adaptive-MoE/actions/workflows/tests.yml/badge.svg)

**English** | [Tiếng Việt](README.vi.md)

A small (experimental) PyTorch module for **adaptive** Mixture-of-Experts: the number of experts activated per token, and how strongly each one contributes, change with token difficulty instead of being fixed at top-6 or top-12.

Anyone training an MoE model can swap their MoE layer for `AdaptiveMoE`.

> **Status: experimental.** The mechanism works (the router learns token difficulty and uses more experts on harder tokens), but in our experiments it did **not** beat a fixed top-k baseline. See [Experimental results](#experimental-results).

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

## Language-model experiment (Vietnamese text)

```bash
pip install torch datasets
python lm_experiment.py --mode adaptive --target_k 3.0   # adaptive
python lm_experiment.py --mode fixed --k 3               # fixed top-k baseline
python compare.py                                        # table of all r_*.json results
```

The script trains a small LM (syllable-level tokenization of Vietnamese Wikipedia) and prints validation loss, average experts per token, the correlation between per-token expert count and per-token loss, and the expert count per token category (punctuation, numbers, function words, proper nouns, rare words, content words). Report labels are printed in Vietnamese. Use `--text_file` to train on your own text.

Options: `--ste` (let task-loss gradients reach the expert-selection boundary) and `--diff_head` (the router predicts token difficulty, trained on the real token loss, then sets `top_p` per token within `[p_min, p_max]`). `--layer_k 1,2,2,3` sets a fixed k per layer.

## Experimental results

Setup: 19M-parameter LM, 4 layers, 8 experts per layer, syllable-level tokenization, Vietnamese Wikipedia (30M characters, about 7.5M tokens), 3000 steps. Fixed k=2 and adaptive with `compute_coef=0` were run with 3 seeds (mean and min–max range); all other configurations used one seed. Lower validation loss is better.

| Configuration | avg_k | val loss |
|---|---|---|
| fixed top-k, k=1 | 1.00 | 4.194 |
| fixed top-k, k=2 (3 seeds) | 2.00 | **4.078** (4.077–4.080) |
| fixed per-layer k (1,2,2,3) | 2.00 | 4.115 |
| fixed per-layer k (1,2,3,3) | 2.25 | 4.115 |
| adaptive + `diff_head`, `compute_coef=0.5` | 1.90 | 4.150 |
| adaptive + `diff_head`, `diff_target=gain`, `compute_coef=0.5` | 1.88 | 4.148 |
| adaptive + `diff_head`, `compute_coef=0` (3 seeds) | 2.10 | 4.085 (4.081–4.088) |
| 32 experts (routed d_ff=128, shared d_ff=512), fixed top-8 | 8.00 | 4.113 |
| 32 experts (routed d_ff=128, shared d_ff=512), adaptive + `diff_head`, `compute_coef=0` | 7.08 | 4.126 |
| 32 experts (same), fixed top-7 | 7.00 | 4.119 |
| 32 experts (same), fixed top-6 | 6.00 | 4.126 |

Takeaways:
- The router learns token difficulty (correlation with the true loss around 0.4 to 0.5) and the number of experts rises with difficulty (correlation between k and loss from +0.26 to +0.37).
- **No evidence that adaptive routing is more efficient than fixed top-k at this scale.** The best configuration (`compute_coef=0`) is about 0.007 nats worse than fixed k=2 (perplexity 59.4 vs 59.0), consistently across all 3 seeds, while using about 5% more experts.
- Finer-grained experts (32 instead of 8, similar total parameters): a first run appeared to favor adaptive routing, but that was an artifact of the shared expert shrinking along with `d_ff`. With the shared expert kept wide (`--d_ff_shared 512`), fixed top-8 (4.113) beats adaptive (4.126), which uses about 11% fewer experts and spreads k widely (4.8 on the easiest quartile to 8.3 on the hardest). The fixed top-7 (4.119) and top-6 (4.126) controls settle it: adaptive at k=7.08 matches fixed top-6 while using more experts, so it loses at equal budget by about 0.007 nats, the same gap seen with 8 experts. The fixed curve is also very flat (about 0.0065 nats per expert between k=6 and k=8), which leaves little headroom for any allocation scheme at this scale. This 32-expert setup is also worse than the 8-expert baseline (4.078) at this short training length.
- A stable observation across seeds: when the router is free, deeper layers ask for more experts (layer 1 about 1.3 to 1.6, layer 4 about 2.7 to 2.8). However, hard-coding k per layer (1,2,2,3) was worse than uniform k, so this is not a ready-made recipe.
- "Hard token" does not mean "token that benefits from more experts": a loss-based label sends more experts to rare words and function words, while content words gain the most from an extra expert. A label based on the loss reduction from adding experts (`gain`) was too noisy for the difficulty head to learn.
- `compute_coef=0.5` acts as a helpful regularizer when data is small and the model overfits (1.9M tokens), but hurts once there is enough data.

Limitations: most configurations have a single seed, the model is small, training is short, and wall-clock speed was not measured (the per-expert loop is not optimized). Results at larger scale may differ.

## Tests and demo

```bash
pip install torch pytest
pytest -q                  # or: python test_adaptive_moe.py
python train_demo.py
```

## License

MIT © 2026 BachDaThan, see [LICENSE](LICENSE).
