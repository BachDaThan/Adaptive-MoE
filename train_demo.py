"""
Demo train nhỏ trên CPU: một nửa vị trí có token kế tiếp cố định (dễ),
nửa còn lại là ngẫu nhiên có điều kiện (khó). In loss, số expert trung bình và
số expert cho token dễ / khó. Token khó ở đây là nhiễu thuần nên mức loss tối ưu
là khoảng 2.08 (= nửa ln 64), demo này kiểm tra pipeline train chứ không chứng minh tính thích ứng.

    python train_demo.py
"""
import torch
import torch.nn as nn

from adaptive_moe import AdaptiveMoE, gather_aux_loss, moe_stats

torch.manual_seed(0)
V, D, S, B = 64, 64, 32, 32


class Demo(nn.Module):
    def __init__(self):
        super().__init__()
        self.emb = nn.Embedding(V, D)
        self.moe = AdaptiveMoE(D, 128, n_experts=8, top_p=0.7, target_k=3.0,
                               compute_coef=0.5, n_shared=1, noise_std=0.1)
        self.head = nn.Linear(D, V)

    def forward(self, x):
        h = self.emb(x)
        y, _, _ = self.moe(h)
        return self.head(h + y)


def batch():
    x = torch.randint(0, V, (B, S))
    y = (x * 7 + 3) % V                          # luật cố định cho nửa token đầu
    hard = x >= V // 2
    y = torch.where(hard, torch.randint(0, V, (B, S)), y)   # nửa còn lại: nhiễu
    return x, y


model = Demo()
opt = torch.optim.AdamW(model.parameters(), lr=3e-3)
for step in range(301):
    x, y = batch()
    logits = model(x)
    task = nn.functional.cross_entropy(logits.view(-1, V), y.view(-1))
    loss = task + gather_aux_loss(model)
    opt.zero_grad()
    loss.backward()
    opt.step()
    if step % 50 == 0:
        s = moe_stats(model)["moe"]
        k = model.moe.last_info["k_per_token"].view(B, S)
        hard = x >= V // 2
        print(f"step {step:3d}  task_loss={task.item():.3f}  avg_k={s['avg_k']:.2f}  "
              f"k_easy={k[~hard].mean():.2f}  k_hard={k[hard].mean():.2f}  strength={s['avg_strength']:.2f}")
