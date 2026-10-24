# adaptive-moe

![tests](https://github.com/BachDaThan/Adaptive-MoE/actions/workflows/tests.yml/badge.svg)

[English](README.md) | **Tiếng Việt**

Module PyTorch nhỏ (thử nghiệm) cho MoE **thích ứng**: số chuyên gia được kích hoạt và mức đóng góp của từng chuyên gia thay đổi theo độ khó của từng token, thay vì cố định top-6 hay top-12.

Ai train model MoE cũng có thể thay layer MoE của mình bằng `AdaptiveMoE`.

> **Trạng thái: thử nghiệm.** Cơ chế hoạt động (router học được độ khó của token và dùng nhiều expert hơn cho token khó), nhưng không thắng các baseline đơn giản: adaptive routing hơi kém top-k cố định ở cùng ngân sách trong phần lớn cấu hình, và ở ngân sách thấp chỉ ngang việc chọn k ngẫu nhiên theo token với cùng ngân sách trung bình. Xem [Kết quả thí nghiệm](#kết-quả-thí-nghiệm-trạng-thái-thử-nghiệm).
>
> Điều có tác dụng thật là **huấn luyện co giãn**: train với k ngẫu nhiên (và/hoặc cửa sổ ngữ cảnh ngẫu nhiên) thì cùng một model chạy được ở mọi mức ngân sách, không mất gì khi dữ liệu dồi dào và tốt hơn rõ khi dữ liệu ít. Xem [Huấn luyện co giãn](#huấn-luyện-co-giãn-một-model-nhiều-mức-ngân-sách).

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
| `set_k_cap(model, k)` | Đặt trần số expert mỗi token (khi inference hoặc train). `k` là số nguyên cho mọi token, hoặc tensor số nguyên `[B*S]` để đặt trần riêng từng token (bản 0.3.1). |

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
- `adaptive_generate` chưa có KV cache (mỗi bước chạy lại cả chuỗi). `elastic_kv.py` có bản dùng KV cache (`adaptive_generate_kv`) trên một model tham chiếu nhỏ; model thật có thể làm theo cùng cách.
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

## Huấn luyện co giãn: một model, nhiều mức ngân sách

Tóm tắt các thí nghiệm bên dưới (model nhỏ, train ngắn; đa số cấu hình chạy 1 đến 3 seed):

- **Elastic k:** train với trần số expert ngẫu nhiên rồi chọn k lúc suy luận. Khi dữ liệu dồi dào, một model elastic lệch khoảng 0.0015 nat so với các model fixed-k riêng; khi dữ liệu ít, nó tốt hơn rõ so với train fixed-k (mục 4).
- **KV cache và cửa sổ ngữ cảnh:** sinh văn bản với KV cache khớp hoàn toàn với chạy lại cả chuỗi; cửa sổ sink + trượt chặn cache ở `sink + W` token mà gần như không mất chất lượng (mục 2 và 3).
- **Phân bổ k thông minh hơn không thắng một k cố định ở cùng chi phí trung bình.** Chỉ có hiệu ứng nhỏ trong khoảng chi phí giữa k=1 và k=2 (mục 5). Lịch k theo layer từng có vẻ tốt trên model elastic **không đứng vững** khi train từ đầu (mục 5.3).

Nhiễu giữa các lần chạy cùng seed (GPU không tất định) khoảng 0.002-0.003 nat, nên chỉ so sánh các ô trong cùng một bảng hoặc một lần chạy.

### Các script

| Script | Làm gì |
|---|---|
| `elastic_kv.py` | Sinh văn bản có KV cache với cascade theo độ tự tin (`adaptive_generate_kv`), kèm self-test. |
| `elastic_ctx.py` | Cửa sổ sink + trượt với KV cache bị chặn bộ nhớ; `--selftest`, `--train`, `--grid`. |
| `headroom.py` | Ước lượng ở mức token xem bộ phân bổ k theo token có thể lợi được bao nhiêu. |
| `tokenk.py` | k theo từng token chạy thật trong các layer (cần `adaptive_moe.py` >= 0.3.1). |
| `layerk.py` | Quét các lịch k theo layer trên một model elastic. |
| `schedule_train.py` | Train từng lịch k từ đầu (bước kiểm chứng cho `layerk.py`). |
| `run_both.sh`, `run_ctx.sh`, `run_tokenk.sh`, `run_layerk.sh`, `run_schedule.sh` | Chạy trọn các script trên. |

```bash
python elastic_ctx.py --selftest
python elastic_ctx.py --grid --top_p 0.999 --hf wikimedia/wikipedia --config 20231101.vi --field text --mb 8
python headroom.py --hf wikimedia/wikipedia --config 20231101.vi --field text --mb 8 --bpe --experts 32 --k_hi 2 --L 128
bash run_tokenk.sh
bash run_schedule.sh 0
```

### 1. Elastic k: train một lần với k ngẫu nhiên, chọn k lúc suy luận

8 expert/layer, 19.3M tham số, corpus 7.5M token, 3000 bước (`--mode random --uniform_k --k_lo 1 --k_hi 4 --eval_ks 1,2,3,4`). Val loss khi ép k lúc suy luận:

| k | 1 | 2 | 3 | 4 |
|---|---|---|---|---|
| Elastic (train với k~U{1..4}) | 4.1014 | 4.0765 | 4.0734 | 4.0739 |
| Top-k cố định, train và chạy cùng k | 4.110\* | 4.078 | 4.0720 | 4.0727 |

\* top-1 cố định với xác suất gate giữ nguyên (`--no_renorm`). k=2 cố định là trung bình 3 seed; k=3 và k=4 mỗi cái một seed.

Một model elastic lệch khoảng 0.0015 nat so với model fixed-k riêng ở k=2..4, và ở k=1 không tệ hơn top-1 train riêng. Nhiễu giữa các seed khoảng 0.002 đến 0.004 nat. Train với k ngẫu nhiên có trung bình 2.5 expert mỗi token, nhưng khi đánh giá với k ngẫu nhiên chỉ đạt 4.082, nên ép một k cố định lúc suy luận tốt hơn để k ngẫu nhiên.

32 expert/layer, 19.4M tham số, corpus nhỏ hơn 1.98M token (nên overfit nặng), 3000 bước, train k~U{1..16}:

| k | 1 | 2 | 4 | 8 | 16 |
|---|---|---|---|---|---|
| Elastic, ép k | 4.4088 | 4.3110 | 4.2762 | 4.2628 | 4.2593 |
| k=4 / k=8 cố định | | | 4.2873 | 4.2908 | |

Ở đây model elastic tốt hơn fixed-k cùng k (0.011 ở k=4 và 0.028 ở k=8). Chất lượng phẳng từ k=8 trở lên và k=1 mất 0.15 nat so với k=8. Xem mục 4 để có so sánh sạch hơn khi dữ liệu ít.

### 2. KV cache cho sinh văn bản (`elastic_kv.py`)

Giải mã tăng dần có cache khớp với tính lại toàn bộ: lệch logit tối đa dưới 1e-6 ở k = 1, 2, 4, 8, và sinh greedy cho ra đúng dãy id. Với cascade theo độ tự tin, token cũ trong cache giữ k lúc nó được tính, nên kết quả cascade không y hệt việc chạy lại cả chuỗi ở một k cố định (theo thiết kế).

### 3. Cửa sổ ngữ cảnh co giãn (`elastic_ctx.py`)

Token ở vị trí $i$ chỉ nhìn các vị trí $j$ thỏa `j < sink` hoặc `i - j < W` (sink = 4). Lúc suy luận KV cache chỉ giữ `sink + W` token, và cho kết quả y hệt chạy lại cả chuỗi với cùng mặt nạ (kiểm bằng `--selftest`, cũng kiểm rằng sinh 200 token với W=16 chỉ giữ 20 token trong cache). Lúc train, W được bốc ngẫu nhiên theo từng mẫu. Thiết lập: LM byte-level 3.9M tham số (d=128, 4 layer, 8 expert), chuỗi dài 256, 8 MB Wikipedia tiếng Việt (`wikimedia/wikipedia`, `20231101.vi`), 3000 bước, k=2.

Chỉ co giãn cửa sổ, ép W lúc suy luận:

| Huấn luyện | W=8 | 16 | 32 | 64 | 128 | 256 (đầy đủ) |
|---|---|---|---|---|---|---|
| Đủ ngữ cảnh | 1.2245 | 1.2059 | 1.1962 | 1.1913 | 1.1894 | 1.1890 |
| W ngẫu nhiên theo mẫu | 1.2040 | 1.1955 | 1.1904 | 1.1883 | 1.1875 | 1.1876 |

Hai lần chạy cùng thí nghiệm (cùng seed) cho kết quả khớp: train W ngẫu nhiên giúp nhiều nhất ở W nhỏ (khoảng 0.02 ở W=8, 0.01 ở W=16) và không mất gì ở ngữ cảnh đầy đủ (lệch trong 0.0015).

Co giãn đồng thời k và W (một model train với k thuộc {1,2,4,8} và W thuộc {8..256}, `top_p=0.999` để trần k thật sự chặn). Mỗi ô là **val loss / tok/s khi giải mã** có KV cache. Số token KV giữ: 12 / 20 / 36 / 68 / 132 / 256.

| k | W=8 | 16 | 32 | 64 | 128 | 256 |
|---|---|---|---|---|---|---|
| **1** | 1.2215 / 79 | 1.2129 / 82 | 1.2083 / 81 | 1.2069 / 83 | 1.2059 / 81 | 1.2060 / 84 |
| **2** | 1.2099 / 79 | 1.2021 / 79 | 1.1973 / 80 | 1.1955 / 79 | 1.1947 / 78 | 1.1947 / 81 |
| **4** | 1.2091 / 71 | 1.2013 / 74 | 1.1965 / 73 | 1.1947 / 73 | 1.1939 / 73 | 1.1938 / 75 |
| **8** | 1.2092 / 67 | 1.2013 / 68 | 1.1964 / 66 | 1.1946 / 67 | 1.1938 / 66 | 1.1937 / 69 |

- Thu nhỏ cache từ 256 xuống 68 token (W=64) mất khoảng 0.0008 nat; xuống 36 token mất khoảng 0.0026; xuống 20 token mất khoảng 0.0074.
- k=1 mất khoảng 0.011 nat so với k=2. k=4 và k=8 chỉ hơn k=2 khoảng 0.001 nhưng giải mã chậm hơn khoảng 7% và 15%, nên k=2 là điểm ngọt của model nhỏ này. (Với `top_p=0.9` lưới có hình dạng tương tự; khi đó trần có thể không chặn ở k>=4.)
- Ở cùng kích thước cache, dồn ngân sách vào k có lợi hơn vào W: (k=2, W=16) tốt hơn (k=1, W=64) với cache nhỏ hơn.
- tok/s bị chi phối bởi overhead Python ở cỡ model này, chỉ mang tính tham khảo; số đo tốc độ chuẩn là bảng CUDA theo batch ở mục 6.

**Giới hạn:** mỗi cấu hình một seed, chuỗi dài 256, model nhỏ, token byte-level. Ở độ dài này model gần như không dùng ngữ cảnh dài nên việc cắt cache gần như miễn phí; chưa có kết quả nào cho chuỗi dài.

### 4. Huấn luyện co giãn so với huấn luyện cố định khi dữ liệu ít

Thiết lập: BPE vocab 16000 huấn luyện trên tập train, 1.48M token train (8 MB Wikipedia tiếng Việt, chia theo dòng ngẫu nhiên), 32 expert, 4 layer, d=128, chuỗi 128, batch 32, 3000 bước (khoảng 8 lượt qua dữ liệu), `top_p=0.999` để trần chặn thật. Mọi số là val loss trên cùng 40 batch (81,920 token). Model elastic train với k ngẫu nhiên theo token trong {1,2} (`tokenk.py`); model cố định train và chạy ở một k (`schedule_train.py`, một seed).

| Model | k=1 | k=2 | k=4 |
|---|---|---|---|
| Cố định, train ở đúng k đó (seed 0) | 4.6575 | 4.7058 | 4.8201 |
| Elastic, ép k lúc eval (seed 0 / 1 / 2) | 4.6613 / 4.6491 / 4.6463 | 4.6236 / 4.6120 / 4.6122 | chưa train |

- Với train cố định, nhiều expert hoạt động hơn lại tệ hơn: k=4 tệ hơn k=1 tới 0.16 nat. Ở các lần chạy elastic, loss train ở bước cuối khoảng 3.0 đến 3.3 trong khi val loss trên 4.6, nên chế độ này thiếu dữ liệu và nhiều khả năng overfit; loss train của các model cố định chưa được ghi lại.
- Model elastic ở k=2 tốt hơn k=2 cố định 0.08 nat và tốt hơn model cố định tốt nhất (k=1) từ 0.03 đến 0.05 nat.
- Giải thích khả dĩ (chưa kiểm chứng): k ngẫu nhiên hoạt động như dropout trên các expert, có lợi khi dữ liệu ít. Chưa làm: baseline k cố định có dropout hoặc dừng sớm, và thêm seed cho các model cố định. Với 7.5M token (mục 1), model elastic và model cố định ngang nhau trong 0.0015 nat.

Rút ra thực tế: khi dữ liệu hạn chế, train với k ngẫu nhiên là cách rẻ để có model ít nhất cũng tốt bằng train k cố định và về sau chạy được ở mọi k.

### 5. Phân bổ k thông minh có thắng một k cố định không?

Cùng thiết lập BPE như mục 4. Chi phí là số expert định tuyến trung bình mỗi token mỗi layer (không tính expert dùng chung và attention). Vì các thí nghiệm này dùng model train kiểu elastic, chúng đo cách một model elastic dùng expert của nó; mục 4 cho thấy khi dữ liệu ít, k=2 train cố định không tốt hơn k=1.

**5.1 Ước lượng ở mức token (`headroom.py`).** Train một model elastic, đo loss từng token ở k=1 và ở k cao hơn, rồi chọn token nào nhận k cao hơn dưới một ngân sách (ngẫu nhiên, entropy của đầu ra k=1, MLP nhỏ trên hidden state sớm hoặc muộn, hoặc luật theo tần suất token). Đây là ước lượng ở mức token, bỏ qua ảnh hưởng qua attention.

- k=1 so với k=4 (gain 0.103 nat): ở cùng chi phí trung bình, một k cố định thắng mọi cách phân bổ giữa k=1 và k=4. Ở chi phí 2, k=2 cố định lấy lại 83% gain so với 33% (ngẫu nhiên), 38% (entropy) và 44% đến 56% (bộ dự đoán đã học). Ở chi phí 3, k=3 cố định lấy lại 97% so với 66% đến 84%.
- k=1 so với k=2 (gain 0.063 nat): phần trăm gain của k=2 lấy lại được khi một tỉ lệ f token nhận k=2:

| f | ngẫu nhiên | entropy | hiếm | phổ biến | learn_emb | learn_l1 | learn_fin |
|---|---|---|---|---|---|---|---|
| 0.1 | 9% | 10% | 19% | 4% | 17% | 16% | 25% |
| 0.25 | 26% | 26% | 36% | 15% | 35% | 38% | 45% |
| 0.5 | 50% | 54% | 64% | 36% | 64% | 65% | 72% |
| 0.75 | 76% | 83% | 85% | 64% | 88% | 86% | 88% |

Luật `rare` (nâng loại token hiếm nhất, chỉ cần tra đếm tần suất trên tập train) ngang bộ dự đoán học trên embedding đầu vào (`learn_emb`) mà không tốn gì. Luật entropy phải chạy xong cả lượt k=1 trước nên tốn 1 + 2f thay vì 1 + f.

**5.2 Luật theo token chạy thật trong các layer (`tokenk.py`).** k theo từng token được truyền vào mọi layer MoE (`set_k_cap(model, tensor)`). Model train với k ngẫu nhiên theo token trong {1,2}; `rare - random` ở cùng tỉ lệ f token được nâng lên k=2 (số âm = rare tốt hơn, đơn vị nat). Tổng gain k=1 lên k=2 trong các model này là 0.034 đến 0.042 nat.

| seed | f=0.25 | f=0.5 | f=0.75 |
|---|---|---|---|
| 0 | -0.0061 | -0.0069 | -0.0044 |
| 1 | -0.0075 | -0.0051 | -0.0056 |
| 2 | -0.0043 | -0.0062 | -0.0065 |
| 0, train bằng chính luật rare | -0.0122 | -0.0101 | -0.0079 |

Luật ngược lại (nâng các token phổ biến nhất) luôn tệ hơn ngẫu nhiên (khoảng +0.005 đến +0.006), nên hiệu ứng không phải nhiễu. Hiệu ứng nhỏ (khoảng 0.006 nat) và chỉ lấp khoảng chi phí giữa k=1 và k=2, nơi top-k cố định không với tới được vì k là số nguyên.

**5.3 k theo layer: không được xác nhận khi train từ đầu.** Một model elastic train với k bốc độc lập cho từng layer từ {1,2,4}, quét cả 81 lịch (`layerk.py`, chọn trên nửa val A, báo trên nửa B), cho thấy các layer đầu có vẻ không cần k lớn. Ví dụ lịch (1,1,2,4) hơn k=2 đồng đều 0.010 đến 0.011 nat ở cùng chi phí trên cả hai seed. Khi train từng lịch từ đầu (`schedule_train.py`, seed 0) thì kết quả này **không lặp lại**:

| k theo layer | chi phí | val loss | train (giây) | so với k=2 đồng đều |
|---|---|---|---|---|
| (1,1,1,1) | 1.0 | 4.6575 | 283 | -0.0483 |
| (2,2,2,2) | 2.0 | 4.7058 | 289 | 0 |
| (4,4,4,4) | 4.0 | 4.8201 | 327 | +0.1143 |
| (1,1,2,2) | 1.5 | 4.6634 | 286 | -0.0425 |
| (1,1,2,4) | 2.0 | 4.7214 | 294 | +0.0155 |
| (1,1,4,4) | 2.5 | 4.7509 | 296 | +0.0451 |
| (4,2,1,1) | 2.0 | 4.6873 | 306 | -0.0186 |

Lịch trông tốt nhất trên model elastic lại tệ hơn k=2 đồng đều, còn đối chứng đảo ngược (4,2,1,1) lại tốt hơn. Cách một model elastic dùng expert ở từng layer không phải căn cứ để chọn lịch cần train. Trong thí nghiệm đầu tiên ở trên, k cố định theo layer cũng kém k đồng đều, nên không khuyến nghị lịch k theo layer.

**Kết luận của mục này.** Một k cố định vẫn là baseline cần vượt. Các hiệu ứng tìm được nhỏ (khoảng 0.01 nat hoặc ít hơn) và chỉ nằm trong khoảng chi phí giữa k=1 và k=2. Kết quả vững là bản thân việc huấn luyện co giãn (mục 1 và 4).

### 6. Tốc độ đo thực tế (CUDA, batch 32x128; tỷ lệ so với khi dùng tất cả expert)

| Số expert | k | suy luận (inference) | huấn luyện (train) |
|---|---|---|---|
| 8 | 1 / 2 / 4 | 0.42x / 0.48x / 0.66x | 0.43x / 0.50x / 0.66x |
| 32 | 1 / 8 / 16 | 0.43x / 0.48x / 0.62x | 0.18x / 0.36x / 0.56x |

Dùng ít expert hơn thì nhanh hơn trong thực tế, nhưng không tỷ lệ thuận lúc suy luận (32 expert, k=1 đạt 0.43x chứ không phải 1/32). Chi phí cố định mỗi bước (attention, embedding, dispatch) chiếm tỷ trọng lớn khi k nhỏ. Trong các lần chạy BPE 4 layer ở trên, thời gian train thực tế gần như không phụ thuộc k (283 giây ở k=1, 289 giây ở k=2, 327 giây ở k=4), nên không nên mặc định là tiết kiệm được compute ở cỡ model đó.

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
