"""
Vẽ val loss theo số expert trung bình từ các kết quả đã chạy (số liệu lấy từ bảng trong README).
    python docs/make_plot.py   ->   docs/loss_vs_k.png
"""
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))

# --- 8 expert/layer ---
ax = axes[0]
ax.plot([1, 2], [4.194, 4.078], "o-", color="#1f77b4", label="fixed top-k (k=1, k=2)")
ax.scatter([2.00, 2.25], [4.115, 4.115], marker="s", color="#7f7f7f", label="fixed per-layer k")
ax.scatter([1.42, 1.90], [4.161, 4.150], marker="^", color="#d62728", label="adaptive, compute_coef=0.5")
ax.scatter([1.41, 1.88], [4.147, 4.148], marker="v", color="#ff7f0e", label="adaptive, gain target")
ax.scatter([2.09, 2.13, 2.09], [4.0873, 4.0882, 4.0806], marker="*", s=110, color="#2ca02c",
           label="adaptive, compute_coef=0 (3 seeds)")
ax.set_title("8 experts per layer")
ax.set_xlabel("average experts per token")
ax.set_ylabel("validation loss (lower is better)")
ax.grid(alpha=0.3)
ax.legend(fontsize=8)

# --- 32 expert/layer ---
ax = axes[1]
ax.plot([6, 7, 8], [4.126, 4.119, 4.113], "o-", color="#1f77b4", label="fixed top-k (k=6, 7, 8)")
ax.scatter([7.08], [4.126], marker="*", s=130, color="#2ca02c", label="adaptive, compute_coef=0")
ax.set_ylim(4.111, 4.1285)
ax.set_title("32 experts per layer")
ax.set_xlabel("average experts per token")
ax.grid(alpha=0.3)
ax.legend(fontsize=8)

fig.suptitle("Adaptive routing vs fixed top-k (19M-parameter LM, Vietnamese Wikipedia)", fontsize=11)
fig.tight_layout()
out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "loss_vs_k.png")
fig.savefig(out, dpi=150)
print("saved", out)
