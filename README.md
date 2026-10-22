# adaptive-moe

Module PyTorch nhỏ cho MoE **thích ứng**: số chuyên gia được kích hoạt và mức đóng góp của từng chuyên gia thay đổi theo độ khó của từng token, thay vì cố định top-6 hay top-12.

Ai train model MoE cũng có thể thay layer MoE của mình bằng `AdaptiveMoE`.

## Ý tưởng

| Thành phần | Làm gì |
|---|---|
| `AdaptiveRouter` | Top-p routing: lấy expert theo xác suất giảm dần cho tới khi tổng đạt `top_p`. Token dễ dùng ít expert, token khó dùng nhiều. |
| Strength head | Mỗi token có hệ số "sức mạnh" trong `[min_strength, 1]` nhân vào đóng góp của các expert được chọn. |
| `compute_loss` | Phạt số expert kỳ vọng (có gradient) để router không luôn chọn tối đa. |
| `balance_loss` | Cân bằng tải giữa các expert (kiểu Switch Transformer). |
| `cascade_forward` | Lúc inference: chạy tầng nhỏ (vd 2 expert), nếu độ tự tin thấp thì nâng lên 4, rồi 8. |
| `max_resident` + `release()` | Chỉ giữ N expert trên GPU theo LRU; `release()` dọn bớt sau khi trả lời. |

## Dùng

```python
from adaptive_moe import AdaptiveMoE, cascade_forward

moe = AdaptiveMoE(d_model=512, d_ff=1024, n_experts=16, top_p=0.7)
out, aux_loss, info = moe(x)          # x: [B, S, D]
loss = task_loss + aux_loss           # info["avg_k"] cho biết số expert trung bình

# inference với nâng tầng
logits, k_used = cascade_forward(model, input_ids, tiers=(2, 4, 8), min_confidence=0.6)
```

## Lưu ý

- Dọn dẹp bằng `release()` chỉ dùng khi inference (eval). Không bật `max_resident` lúc train.
- Cascade chạy lại forward ở tầng cao hơn nên tốn thêm compute khi câu khó. Với model sinh text, nên áp dụng theo từng đoạn hoặc từng token thay vì cả câu.
- `top_p`, `compute_coef`, `balance_coef` cần chỉnh theo model và dữ liệu của bạn.
- Đây là phần router/layer, chưa phải model hoàn chỉnh.

## Chạy test

```bash
pip install torch
python test_adaptive_moe.py
```

## License

Chọn license thoáng (MIT hoặc Apache-2.0) để ai cũng dùng được.
