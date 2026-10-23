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

## Elastic k: train một lần, chọn k lúc suy luận

Train với k ngẫu nhiên theo token cho ra một model chạy được ở cả hai mức ngân sách, chênh lệch nhỏ (một seed: 4.096 ở k=1 và 4.082 ở k=2 trên LM tiếng Việt bên dưới, chênh 0.014 nat; model chỉ train cho top-1 với xác suất gate giữ nguyên đạt 4.110).

```python
from adaptive_moe import AdaptiveMoE, set_k_cap

moe = AdaptiveMoE(d_model=512, d_ff=1024, n_experts=8, min_k=1, max_k=2, top_p=1.0,
                  compute_coef=0.0, rand_k=(1, 2, 0.5))   # lúc train: mỗi token nhận k=1 hoặc k=2 ngẫu nhiên
# lúc suy luận: chọn ngân sách cho cả model
set_k_cap(model, 1)   # đúng 1 expert mỗi token (rẻ)
set_k_cap(model, 2)   # đúng 2 expert mỗi token
```

Khi đặt `rand_k`, `k_cap` ép đúng k (không đặt thì dùng lựa chọn ngẫu nhiên theo token). Dùng `lm_experiment.py --mode random --eval_cascade` để tái hiện số liệu. Muốn khoảng rộng hơn: `--uniform_k --k_lo 1 --k_hi 4 --eval_ks 1,2,3,4` train với k lấy đồng đều từ 1 đến 4 và báo val loss ở từng k ép cố định.

## Thí nghiệm trên tiếng Việt thật

```bash
pip install torch datasets
python lm_experiment.py --mode adaptive --target_k 3.0   # adaptive
python lm_experiment.py --mode fixed --k 3               # baseline top-k cố định
```

Script train LM nhỏ (tokenize theo âm tiết) rồi in val loss, số expert trung bình, tương quan giữa k và loss của token, và k theo loại token (dấu câu, số, từ chức năng, tên riêng, từ hiếm, từ nội dung). Dùng `--text_file` nếu muốn dữ liệu riêng.

Tùy chọn: `--ste` (gradient chảy vào ranh giới chọn expert) và `--diff_head` (router có đầu dự đoán độ khó của token, huấn luyện bằng loss thật, rồi đặt `top_p` theo từng token trong `[p_min, p_max]`). `--mode random` chạy đối chứng k ngẫu nhiên theo token, `--no_renorm` giữ nguyên xác suất gate (kiểu Switch) để top-1 vẫn có gradient cho router.

## Kết quả thí nghiệm (trạng thái: thử nghiệm)

Thiết lập: LM 19M tham số, 4 layer, 8 expert/layer, tokenize theo âm tiết, Wikipedia tiếng Việt (30M ký tự, ~7.5M token), 3000 bước. Fixed k=2 và adaptive `compute_coef=0` chạy 3 seed (trung bình, khoảng min–max), các cấu hình còn lại một seed. Val loss thấp hơn là tốt hơn.

| Cấu hình | avg_k | val loss |
|---|---|---|
| top-k cố định, k=1 (trọng số chuẩn hóa) | 1.00 | 4.194 |
| top-1 cố định, giữ nguyên xác suất gate (`--no_renorm`) | 1.00 | 4.110 |
| top-2 cố định, giữ nguyên xác suất gate (`--no_renorm`) | 2.00 | 4.083 |
| top-k cố định, k=2 (3 seed) | 2.00 | **4.078** (4.077–4.080) |
| k cố định theo layer (1,2,2,3) | 2.00 | 4.115 |
| k cố định theo layer (1,2,3,3) | 2.25 | 4.115 |
| adaptive + `diff_head`, `compute_coef=0.5` | 1.90 | 4.150 |
| adaptive + `diff_head`, `diff_target=gain`, `compute_coef=0.5` | 1.88 | 4.148 |
| adaptive + `diff_head`, `compute_coef=0` (3 seed) | 2.10 | 4.085 (4.081–4.088) |
| adaptive + `diff_head`, `compute_coef=0`, ngân sách thấp (3 seed) | 1.46 | 4.090 (4.083–4.098) |
| k ngẫu nhiên theo token trong {1,2}, 46% token nhận 2 (đối chứng, 1 seed) | 1.46 | 4.092 |
| train với k ngẫu nhiên (50/50), rồi chạy ở k=1 (`--eval_cascade`) | 1.00 | 4.096 |
| train với k ngẫu nhiên (50/50), rồi chạy ở k=2 | 2.00 | 4.082 |
| 32 expert (expert định tuyến d_ff=128, dùng chung d_ff=512), top-8 cố định | 8.00 | 4.113 |
| 32 expert (expert định tuyến d_ff=128, dùng chung d_ff=512), adaptive + `diff_head`, `compute_coef=0` | 7.08 | 4.126 |
| 32 expert (như trên), top-7 cố định | 7.00 | 4.119 |
| 32 expert (như trên), top-6 cố định | 6.00 | 4.126 |

![Val loss theo số expert trung bình: adaptive so với top-k cố định](docs/loss_vs_k.png)

Điểm nằm dưới đường top-k cố định là tốt hơn ở cùng ngân sách. Điểm k=1 chuẩn hóa là mốc tham chiếu yếu (một expert đã chuẩn hóa không cho router gradient từ task loss). Khi giữ nguyên xác suất gate (`--no_renorm`), k=1 cố định đạt 4.110 (ô vuông xanh), và các điểm adaptive, ngẫu nhiên ở k≈1.46 nằm trong khoảng 0.005 nat so với đường thẳng nét đứt nối điểm đó với k=2 cố định, tức chỉ là một đánh đổi tuyến tính thông thường. Biểu đồ bên phải chỉ trải trong 0.015 nat.

Rút ra:
- Router học được độ khó của token (tương quan với loss thật khoảng 0.4 đến 0.5) và số expert tăng theo độ khó (tương quan k và loss từ +0.26 đến +0.37).
- **Phân bổ đã học không thắng phân bổ ngẫu nhiên ở cùng ngân sách.** Ở khoảng 1.46 expert mỗi token (8 expert), adaptive với `compute_coef=0` đạt 4.090 (3 seed: 4.083 đến 4.098), còn đối chứng cho mỗi token k=1 hoặc k=2 ngẫu nhiên với cùng ngân sách trung bình đạt 4.092 (1 seed). Top-1 cố định với xác suất gate giữ nguyên (`--no_renorm`) đạt 4.110; baseline k=1 chuẩn hóa (4.194) bị thiệt vì một expert đã chuẩn hóa không cho router gradient từ task loss. So với mốc đã sửa, các điểm adaptive và ngẫu nhiên ở k≈1.46 nằm trong khoảng 0.005 nat so với đường thẳng nối k=1 (4.110) và k=2 (4.078): một đánh đổi tuyến tính khoảng 0.03 nat mỗi expert định tuyến, không phải lợi ích đặc biệt. (Khi cả hai đầu cùng giữ nguyên gate, k=2 cho 4.083, nên đường tham chiếu chạy từ 4.110 đến 4.083, khoảng 0.027 nat mỗi expert định tuyến; các điểm adaptive và ngẫu nhiên nằm thấp hơn đường đó khoảng 0.005 đến 0.007 nat, tương đương nhiễu giữa các seed.) Nâng tầng theo độ bất định của chính model (entropy của lượt k=1 rẻ) chỉ hơn nâng ngẫu nhiên 0.0014 nat ở ngân sách 50% (`--eval_cascade`), và toàn bộ khoảng từ toàn k=1 đến toàn k=2 chỉ trải 0.014 nat, nên gần như không còn gì để bộ điều khiển lấy thêm. (Cột "oracle" trong báo cáo đó không phải trần đạt được: nhìn lại với hindsight nó chọn cái tốt hơn trong hai loss nhiễu của từng token.) Ở khoảng 2 expert mỗi token (8 expert) và ở 32 expert, adaptive thua nhẹ top-k cố định ở cùng ngân sách (khoảng 0.007 nat).
- Expert mịn hơn (32 thay vì 8, tổng tham số tương đương): lần chạy đầu có vẻ ủng hộ adaptive, nhưng đó là hiệu ứng giả do expert dùng chung bị teo theo `d_ff`. Khi giữ expert dùng chung rộng (`--d_ff_shared 512`), top-8 cố định (4.113) thắng adaptive (4.126), dù adaptive dùng ít hơn khoảng 11% expert và k phân hóa rộng (4.8 ở nhóm token dễ nhất, 8.3 ở nhóm khó nhất). Hai bản đối chứng top-7 (4.119) và top-6 (4.126) cho câu trả lời: adaptive ở k=7.08 chỉ ngang top-6 cố định mà dùng nhiều expert hơn, tức thua ở cùng ngân sách khoảng 0.007 nat, đúng bằng khoảng cách ở bản 8 expert. Đường cong của bản cố định cũng rất phẳng (khoảng 0.0065 nat mỗi expert từ k=6 đến k=8), nên ở quy mô này gần như không còn chỗ cho cách phân bổ nào. Cấu hình 32 expert này cũng kém bản 8 expert (4.078) khi train ngắn như vậy.
- Quan sát ổn định giữa các seed: khi để router tự do, các layer sâu xin nhiều expert hơn (layer 1 khoảng 1.3 đến 1.6, layer 4 khoảng 2.7 đến 2.8). Tuy vậy gán cứng k theo layer (1,2,2,3) lại kém k đồng đều, nên chưa thể dùng điều này như một công thức.
- "Token khó" không đồng nghĩa "token được lợi khi thêm expert": nhãn dạy theo loss đưa nhiều expert cho từ hiếm và từ chức năng, trong khi từ nội dung mới là loại được lợi nhiều nhất khi thêm expert. Nhãn theo mức loss giảm được (`gain`) quá nhiễu nên đầu dự đoán không học được.
- `compute_coef=0.5` giúp như một regularizer khi dữ liệu nhỏ và model overfit (1.9M token), nhưng làm hại khi dữ liệu đủ lớn.
- Phân bổ k cố định theo layer không thắng k đồng đều.

Giới hạn: phần lớn cấu hình chỉ một seed, model nhỏ, train ngắn, chưa đo tốc độ thật (vòng lặp theo expert chưa được tối ưu). Kết quả ở model lớn có thể khác.

## Elastic k, phân phối đều k trong khoảng 1..4 (1 seed)

Huấn luyện một lần với k ngẫu nhiên đều trên {1..4}, sau đó ép cố định k lúc suy luận:

| Mô hình | k=1 | k=2 | k=3 | k=4 |
|---|---|---|---|---|
| Elastic, ép k lúc eval | 4.1014 | 4.0765 | 4.0734 | 4.0739 |
| Fixed top-k (train và chạy cùng k) | 4.110\* | 4.0780 | 4.0720 | 4.0727 |

\* *fixed top-1 với xác suất cổng thô (`--no_renorm`). Cố định k=2 là giá trị trung bình của 3 seed; k=3 và k=4 sử dụng 1 seed.*

**Nhận xét:** Một mô hình elastic duy nhất chỉ lệch tối đa khoảng 0.0015 nat so với mô hình fixed-k riêng biệt ở khoảng k=2..4, và ở k=1 thậm chí không tệ hơn mô hình top-1 được huấn luyện riêng (4.101 so với 4.110). Nhiễu giữa các seed rơi vào khoảng 0.002 đến 0.004 nat, vì vậy chỉ có hàng k=1 là có sự khác biệt rõ rệt so với xu hướng của baseline cố định, cấu hình này sẽ cần thêm seed để đánh giá chính xác hơn. Chất lượng mô hình gần như đi ngang từ k=2 đến k=4 ở quy mô này, do đó việc điều chỉnh chi phí tính toán (budget) chủ yếu có ý nghĩa rõ rệt trong khoảng từ k=1 đến k=2.

Quá trình huấn luyện với k ngẫu nhiên đạt trung bình 2.5 expert trên mỗi token, nhưng nếu đánh giá (eval) bằng k ngẫu nhiên thì kết quả đạt 4.082 (ppl 59.2); việc ép một mức k cố định khi suy luận mang lại hiệu quả tốt hơn là để k ngẫu nhiên.

## Cửa sổ ngữ cảnh co giãn (sink + sliding window) và đồng thời co giãn k x W

**Cấu hình thực nghiệm:** byte-level LM, 3.9M tham số (d=128, 4 layers, 8 experts), chiều dài chuỗi 256, sử dụng 8 MB dữ liệu từ tiếng Việt Wikipedia (`wikimedia/wikipedia 20231101.vi`), chạy 3000 steps, cấu hình bộ nhớ đệm cố định `sink = 4`, và `top_p = 0.9`. Một token ở vị trí $i$ sẽ chú ý (attend) đến các vị trí $j$ thỏa mãn `j < sink` hoặc `i - j < W`. Khi suy luận, KV cache chỉ giữ lại tối đa `sink + W` token; cơ chế này đã được kiểm tra chéo qua lệnh `--selftest` để đảm bảo kết quả đầu ra khớp tuyệt đối so với việc chạy lại toàn bộ chuỗi bằng cùng một mặt nạ (mask).

### Chỉ co giãn cửa sổ (cố định k = 2): val loss khi ép mức W lúc suy luận

| Cách huấn luyện | W=8 | 16 | 32 | 64 | 128 | 256 (full) |
|---|---|---|---|---|---|---|
| Ngữ cảnh đầy đủ (Full context) | 1.2245 | 1.2059 | 1.1962 | 1.1913 | 1.1894 | 1.1890 |
| W ngẫu nhiên theo từng mẫu | 1.2040 | 1.1955 | 1.1904 | 1.1883 | 1.1875 | 1.1876 |

Hai lượt chạy thử nghiệm (cùng seed, có hiện tượng bất định cấu trúc của GPU) cho ra kết quả đồng nhất: Huấn luyện với W ngẫu nhiên giúp cải thiện rõ rệt ở các giá trị W nhỏ (giảm khoảng 0.02 nat ở W=8, và 0.01 nat ở W=16) và hầu như không làm suy giảm chất lượng khi chạy ở ngữ cảnh đầy đủ (chênh lệch nằm trong khoảng sai số +-0.0015).

### Đồng thời co giãn cả k và W (một mô hình duy nhất được train với k thuộc {1,2,4,8} và W thuộc {8..256})

val loss (Hàng đại diện cho $k$, cột đại diện cho $W$; Số lượng KV token được giữ lại tương ứng: 12 / 20 / 36 / 68 / 132 / 256)

| k | W=8 | 16 | 32 | 64 | 128 | 256 |
|---|---|---|---|---|---|---|
| 1 | 1.2165 | 1.2095 | 1.2043 | 1.2023 | 1.2017 | 1.2018 |
| 2 | 1.2060 | 1.1992 | 1.1939 | 1.1922 | 1.1914 | 1.1915 |
| 4 | 1.2054 | 1.1988 | 1.1935 | 1.1917 | 1.1909 | 1.1910 |
| 8 | 1.2056 | 1.1989 | 1.1936 | 1.1918 | 1.1910 | 1.1911 |

*Lưu ý: Lượt chạy thứ hai của cùng ma trận này cho kết quả val loss bị dịch chuyển tăng lên khoảng 0.0025 ở tất cả các ô, vì vậy khuyến nghị so sánh đối chiếu giữa các ô trong cùng một lượt chạy.*

**Nhận xét & Quan sát:**
- Việc giảm kích thước bộ nhớ đệm từ 256 xuống còn 68 token (W=64) chỉ làm tốn thêm khoảng 0.0007 nat; xuống 36 token (W=32) tốn khoảng 0.0024 nat; và xuống 20 token (W=16) tốn khoảng 0.008 nat.
- Mức cấu hình k=1 làm suy giảm khoảng 0.010 nat so với k=2. Trong khi đó, các mức k=2, k=4 và k=8 cho kết quả cực kỳ sát nhau (chênh lệch dưới 0.0005 nat). Nguyên nhân có thể do cài đặt `top_p = 0.9` đã vô tình khiến router tự kích hoạt ít expert hơn mức trần, khiến cho giới hạn k>=4 không được kích hoạt triệt để (xem thêm phần "Câu hỏi mở").
- Dành tài nguyên tính toán (budget) cho expert ($k$) mang lại hiệu quả cao hơn cho cửa sổ ngữ cảnh ($W$): Cấu hình (k=2, W=16) cho kết quả tốt hơn hẳn (k=1, W=64) dù sử dụng dung lượng cache nhỏ hơn nhiều.
- Tốc độ giải mã (Decode tok/s) đi ngang (dao động khoảng 75-83) do mô hình quá nhỏ này bị nghẽn cổ chai bởi các chi phí xử lý overhead của Python, vì thế phần này không đưa ra kết luận về cải thiện tốc độ phần cứng.

**Câu hỏi mở:** Cần chạy lại thử nghiệm với thiết lập `--top_p 0.999` để kiểm tra xem giới hạn trần của k thực sự có tác dụng ràng buộc ở mức k=4 và k=8 hay không.  
**Hạn chế:** Thử nghiệm mới thực hiện trên 1 seed duy nhất cho mỗi cấu hình, độ dài chuỗi giới hạn ở mức 256.

### Tốc độ đo thực tế (CUDA, batch 32x128; tỷ lệ so với khi dùng tất cả expert)

| Số Expert | k | suy luận (inference) | huấn luyện (train) |
|---|---|---|---|
| 8 | 1 / 2 / 4 | 0.42x / 0.48x / 0.66x | 0.43x / 0.50x / 0.66x |
| 32 | 1 / 8 / 16 | 0.43x / 0.48x / 0.62x | 0.18x / 0.36x / 0.56x |

Sử dụng ít expert hơn giúp tăng tốc độ trong thực tế, nhưng không tỷ lệ thuận hoàn toàn lúc suy luận (ví dụ với 32 experts, k=1 đạt tốc độ 0.43x chứ không phải là 1/32). Chi phí cố định cho mỗi bước (attention, embedding, dispatch) chiếm tỷ trọng lớn khi k nhỏ.

## Đo tốc độ

`python bench.py` đo tốc độ thật (token mỗi giây, suy luận và train) khi mọi token bị ép dùng đúng k expert, để kiểm tra dùng ít expert hơn có nhanh hơn thật trên phần cứng của bạn không. Ví dụ: `python bench.py --experts 32 --d_ff 128 --ks 1,4,8,16,32`.

## Chạy test và demo

```bash
pip install torch pytest
pytest -q                  # hoặc: python test_adaptive_moe.py
python train_demo.py
```

## License

MIT © 2026 BachDaThan, xem [LICENSE](LICENSE).
