# adaptive-moe

![tests](https://github.com/BachDaThan/Adaptive-MoE/actions/workflows/tests.yml/badge.svg)

[English](README.md) | **Tiếng Việt**

Module PyTorch nhỏ (thử nghiệm) cho MoE **thích ứng**: số chuyên gia được kích hoạt và mức đóng góp của từng chuyên gia thay đổi theo độ khó của từng token, thay vì cố định top-6 hay top-12.

Ai train model MoE cũng có thể thay layer MoE của mình bằng `AdaptiveMoE`.

## Cài

```bash
pip install git+https://github.com/BachDaThan/Adaptive-MoE
```

hoặc chép `adaptive_moe.py` vào project (chỉ cần PyTorch).

## Thành phần

| Thành phần | Làm gì |
|---|---|
| `AdaptiveRouter` | Top-p routing: lấy expert theo xác suất giảm dần cho tới khi tổng đạt `top_p`. Token dễ dùng ít expert, token khó dùng nhiều. |
| Strength head | Mỗi token có hệ số "sức mạnh" trong `[min_strength, 1]` nhân vào đóng góp của expert được chọn. |
| `target_k` | Kéo số expert trung bình về gần giá trị bạn muốn (vd 3.0), dễ kiểm soát hơn phạt thô. |
| `n_shared` | Expert luôn bật (kiểu DeepSeek), giúp chất lượng ổn khi router chỉ chọn 1 expert. |
| Loss phụ | Cân bằng tải, phạt compute, z-loss. Gom bằng `gather_aux_loss(model)`. |
| `moe_stats(model)` | Xem `avg_k` và `avg_strength` của từng layer sau mỗi forward. |
| `cascade_forward` | Inference: chạy tầng nhỏ (vd 2 expert), kém tự tin thì nâng lên 4, rồi 8. Độ tự tin: `maxprob` hoặc `entropy`. |
| `adaptive_generate` | Sinh text, mỗi bước tự chọn tầng. |
| `diff_head` (thử nghiệm) | Router có đầu dự đoán độ khó của token, đặt `top_p` theo từng token. Xem mục kết quả bên dưới. |
| `max_resident` + `release()` | Chỉ giữ N expert trên GPU theo LRU; `release()` dọn bớt sau khi trả lời. |

## Dùng

```python
from adaptive_moe import AdaptiveMoE, gather_aux_loss, moe_stats, adaptive_generate

# trong model của bạn
self.moe = AdaptiveMoE(d_model=512, d_ff=1024, n_experts=16,
                       top_p=0.7, target_k=4.0, n_shared=1)
y, _, _ = self.moe(x)                      # x: [B, S, D]

# train
loss = task_loss + gather_aux_loss(model)
print(moe_stats(model))                    # {"layers.0.moe": {"avg_k": ..., "avg_strength": ...}}

# inference, mỗi bước tự nâng tầng
ids, ks = adaptive_generate(model, input_ids, max_new_tokens=64,
                            tiers=(2, 4, 8), min_confidence=0.6, confidence="entropy")
```

## Lưu ý

- `release()` và `max_resident` chỉ dùng khi inference (eval), không bật lúc train.
- `adaptive_generate` chưa có KV cache (mỗi bước chạy lại cả chuỗi). Model thật nên tự thêm KV cache; phần chọn tầng giữ nguyên.
- Cascade chạy lại forward khi câu khó, nên tốn thêm compute ở những bước đó.
- `top_p`, `target_k` và các hệ số loss cần chỉnh theo model và dữ liệu của bạn.
- Đây là phần router/layer, chưa phải model hoàn chỉnh.

## Thí nghiệm trên tiếng Việt thật

```bash
pip install torch datasets
python lm_experiment.py --mode adaptive --target_k 3.0   # adaptive
python lm_experiment.py --mode fixed --k 3               # baseline top-k cố định
```

Script train LM nhỏ (tokenize theo âm tiết) rồi in val loss, số expert trung bình, tương quan giữa k và loss của token, và k theo loại token (dấu câu, số, từ chức năng, tên riêng, từ hiếm, từ nội dung). Dùng `--text_file` nếu muốn dữ liệu riêng.

Tùy chọn: `--ste` (gradient chảy vào ranh giới chọn expert) và `--diff_head` (router có đầu dự đoán độ khó của token, huấn luyện bằng loss thật, rồi đặt `top_p` theo từng token trong `[p_min, p_max]`).

## Kết quả thí nghiệm (trạng thái: thử nghiệm)

Thiết lập: LM 19M tham số, 4 layer, 8 expert/layer, tokenize theo âm tiết, Wikipedia tiếng Việt (30M ký tự, ~7.5M token), 3000 bước. Fixed k=2 và adaptive `compute_coef=0` chạy 3 seed (trung bình, khoảng min–max), các cấu hình còn lại một seed. Val loss thấp hơn là tốt hơn.

| Cấu hình | avg_k | val loss |
|---|---|---|
| top-k cố định, k=1 | 1.00 | 4.194 |
| top-k cố định, k=2 (3 seed) | 2.00 | **4.078** (4.077–4.080) |
| k cố định theo layer (1,2,2,3) | 2.00 | 4.115 |
| k cố định theo layer (1,2,3,3) | 2.25 | 4.115 |
| adaptive + `diff_head`, `compute_coef=0.5` | 1.90 | 4.150 |
| adaptive + `diff_head`, `diff_target=gain`, `compute_coef=0.5` | 1.88 | 4.148 |
| adaptive + `diff_head`, `compute_coef=0` (3 seed) | 2.10 | 4.085 (4.081–4.088) |
| expert mịn: 32 expert (d_ff=128), top-8 cố định | 8.00 | 4.141 |
| expert mịn: 32 expert (d_ff=128), adaptive + `diff_head`, `compute_coef=0` | 7.15 | 4.123 |

Rút ra:
- Router học được độ khó của token (tương quan với loss thật khoảng 0.4 đến 0.5) và số expert tăng theo độ khó (tương quan k và loss từ +0.26 đến +0.37).
- **Chưa có bằng chứng adaptive hiệu quả hơn top-k cố định ở quy mô này.** Cấu hình tốt nhất (`compute_coef=0`) kém fixed k=2 khoảng 0.007 nat (ppl 59.4 so với 59.0), nhất quán ở cả 3 seed, trong khi dùng nhiều hơn khoảng 5% expert.
- Với expert mịn hơn (32 thay vì 8, cùng tổng tham số), adaptive thắng top-8 cố định ở lần chạy duy nhất có (4.123 so với 4.141) trong khi dùng ít hơn khoảng 11% expert, và k phân hóa rộng hơn nhiều (từ 4.9 ở nhóm token dễ nhất đến 8.4 ở nhóm khó nhất). Lưu ý: mới một seed, và cấu hình 32 expert này kém hơn bản 8 expert về tuyệt đối (4.078), một phần do expert dùng chung bị teo theo `d_ff`; dùng `--d_ff_shared` để giữ nó rộng. Đây là xu hướng đáng chú ý, chưa phải chiến thắng tuyệt đối.
- Quan sát ổn định giữa các seed: khi để router tự do, các layer sâu xin nhiều expert hơn (layer 1 khoảng 1.3 đến 1.6, layer 4 khoảng 2.7 đến 2.8). Tuy vậy gán cứng k theo layer (1,2,2,3) lại kém k đồng đều, nên chưa thể dùng điều này như một công thức.
- "Token khó" không đồng nghĩa "token được lợi khi thêm expert": nhãn dạy theo loss đưa nhiều expert cho từ hiếm và từ chức năng, trong khi từ nội dung mới là loại được lợi nhiều nhất khi thêm expert. Nhãn theo mức loss giảm được (`gain`) quá nhiễu nên đầu dự đoán không học được.
- `compute_coef=0.5` giúp như một regularizer khi dữ liệu nhỏ và model overfit (1.9M token), nhưng làm hại khi dữ liệu đủ lớn.
- Phân bổ k cố định theo layer không thắng k đồng đều.

Giới hạn: phần lớn cấu hình chỉ một seed, model nhỏ, train ngắn, chưa đo tốc độ thật (vòng lặp theo expert chưa được tối ưu). Kết quả ở model lớn có thể khác.

## Chạy test và demo

```bash
pip install torch pytest
pytest -q                  # hoặc: python test_adaptive_moe.py
python train_demo.py
```

## License

MIT © 2026 BachDaThan, xem [LICENSE](LICENSE).
