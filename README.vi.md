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

## Các thực nghiệm về ngân sách co giãn (k, bộ nhớ đệm KV, cửa sổ ngữ cảnh)

Tất cả các lượt chạy dưới đây đều sử dụng cấu hình 1 seed duy nhất. Nhiễu kết quả giữa các lần chạy trên cùng một seed (do tính bất định cấu trúc của GPU) rơi vào khoảng 0.002-0.003 nat, vì vậy chỉ nên so sánh đối chiếu dữ liệu giữa các ô trong cùng một bảng hoặc cùng một lượt chạy. Các câu lệnh chi tiết có trong phần `--help` của mã nguồn script.

### 1. Elastic k: Huấn luyện một lần với k ngẫu nhiên, tùy chọn k khi suy luận

Cấu hình 8 expert/layer, 19.3M tham số, tập dữ liệu 7.5M token, chạy 3000 steps. Kết quả giá trị val loss khi ép mức k cố định lúc suy luận:

| k | 1 | 2 | 3 | 4 |
|---|---|---|---|---|
| Elastic (được train với k~U{1..4}) | 4.1014 | 4.0765 | 4.0734 | 4.0739 |
| Fixed k=3 / fixed k=4 (được train cố định) | | | 4.0720 | 4.0727 |

Mô hình elastic duy nhất này chỉ lệch khoảng 0.0015 nat so với các mô hình fixed-k độc lập. Nếu đánh giá bằng k ngẫu nhiên (thay vì ép một mức k cố định), giá trị đạt 4.0817; do đó việc ép một mức k cụ thể lúc suy luận mang lại hiệu quả tối ưu hơn.

Cấu hình 32 expert/layer, 19.4M tham số, tập dữ liệu nhỏ 1.98M token (bị hiện tượng quá khớp - overfit rất nặng), 3000 steps, huấn luyện với k~U{1..16}:

| k | 1 | 2 | 4 | 8 | 16 |
|---|---|---|---|---|---|
| Elastic, ép k cố định | 4.4088 | 4.3110 | 4.2762 | 4.2628 | 4.2593 |
| Fixed k=4 / k=8 | | | 4.2873 | 4.2908 | |

Trong trường hợp này, mô hình elastic cho kết quả tốt hơn mô hình fixed-k ở cùng cấu hình mức tính toán (giảm được 0.011 nat ở k=4 và 0.028 nat ở k=8). Với lượng dữ liệu ít như vậy, cơ chế k ngẫu nhiên nhiều khả năng đóng vai trò như một bộ điều hòa tính toán (regularizer). Khi test trên tập dữ liệu lớn hơn ở cấu hình 8 expert thì không thấy hiệu ứng này nữa, vì thế kết quả này không nên được hiểu là "elastic sẽ luôn đánh bại fixed" trên mọi bài toán. Chất lượng mô hình có xu hướng đi ngang từ mức k>=8, và cấu hình k=1 bị giảm mất 0.15 nat so với k=8.

### 2. Bộ nhớ đệm KV cache cho quá trình tạo văn bản (`elastic_kv.py`)

Cơ chế incremental decoding (giải mã tăng trưởng từng bước) có sử dụng cache cho kết quả khớp hoàn toàn so với việc tính toán lại toàn bộ chuỗi từ đầu: độ lệch logit lớn nhất nhỏ hơn 1e-6 tại các mức k = 1, 2, 4, 8, đồng thời quá trình tạo chuỗi theo thuật toán tham lam (greedy generation) cho ra các id token trùng khớp tuyệt đối. Đối với cơ chế cascade dựa trên độ tự tin, các token cũ đã lưu trong cache sẽ giữ nguyên mức k mà chúng được tính toán trước đó, do đó đầu ra của cascade theo thiết kế sẽ không trùng khớp hoàn toàn với việc chạy lại từ đầu toàn bộ chuỗi ở duy nhất một mức k.

### 3. Cửa sổ ngữ cảnh co giãn (`elastic_ctx.py`)

Một token ở vị trí $i$ sẽ chỉ chú ý (attend) đến các vị trí $j$ thỏa mãn `j < sink` hoặc `i - j < W` (với cấu hình cố định `sink = 4`). Khi suy luận, KV cache chỉ lưu lại tối đa `sink + W` token, mang lại kết quả đầu ra trùng khớp tuyệt đối so với việc chạy lại toàn bộ chuỗi với cùng một mặt nạ (được xác thực thông qua lệnh tự kiểm tra `--selftest`). Cấu hình: byte-level LM, 3.9M tham số, 8 experts, độ dài chuỗi 256, 8 MB dữ liệu lấy từ tiếng Việt Wikipedia (`wikimedia/wikipedia`, `20231101.vi`), chạy 3000 steps, cấu hình cố định k=2.

Chỉ co giãn cửa sổ ngữ cảnh, ép W lúc suy luận:

| Cách huấn luyện | W=8 | 16 | 32 | 64 | 128 | 256 (full) |
|---|---|---|---|---|---|---|
| Ngữ cảnh đầy đủ (Full context) | 1.2245 | 1.2059 | 1.1962 | 1.1913 | 1.1894 | 1.1890 |
| W ngẫu nhiên theo từng mẫu | 1.2040 | 1.1955 | 1.1904 | 1.1883 | 1.1875 | 1.1876 |

Chiến lược huấn luyện với W ngẫu nhiên hỗ trợ cải thiện chất lượng mạnh nhất ở các mức W nhỏ (giảm khoảng 0.02 nat ở W=8, và 0.01 nat ở W=16) và không làm mất đi độ chính xác khi chạy ở cấu hình toàn ngữ cảnh đầy đủ.

Đồng thời co giãn cả k và W (một mô hình duy nhất, k thuộc {1,2,4,8}, W thuộc {8..256}, sử dụng thiết lập `top_p=0.999` để giới hạn trần k có hiệu lực bắt buộc).  
Mỗi ô đại diện cho: **val loss / tốc độ giải mã (tok/s)** khi dùng KV cache. Số lượng KV token được giữ lại tương ứng: 12 / 20 / 36 / 68 / 132 / 256.

| k | W=8 | 16 | 32 | 64 | 128 | 256 |
|---|---|---|---|---|---|---|
| **1** | 1.2215 / 79 | 1.2129 / 82 | 1.2083 / 81 | 1.2069 / 83 | 1.2059 / 81 | 1.2060 / 84 |
| **2** | 1.2099 / 79 | 1.2021 / 79 | 1.1973 / 80 | 1.1955 / 79 | 1.1947 / 78 | 1.1947 / 81 |
| **4** | 1.2091 / 71 | 1.2013 / 74 | 1.1965 / 73 | 1.1947 / 73 | 1.1939 / 73 | 1.1938 / 75 |
| **8** | 1.2092 / 67 | 1.2013 / 68 | 1.1964 / 66 | 1.1946 / 67 | 1.1938 / 66 | 1.1937 / 69 |

- Giảm bộ đệm từ 256 xuống còn 68 token (W=64) làm tốn thêm khoảng 0.0008 nat; xuống 36 token tốn khoảng 0.0026 nat; và xuống 20 token tốn khoảng 0.0074 nat.
- Mức k=1 làm tốn thêm khoảng 0.011 nat so với k=2. Các mức k=4 và k=8 chỉ giúp cải thiện cực kỳ nhỏ khoảng 0.001 nat so với k=2 nhưng tốc độ giải mã lại chậm đi lần lượt khoảng 7% và 15%, do đó cấu hình k=2 chính là điểm cân bằng lý tưởng (sweet spot) cho mô hình kích thước nhỏ này.
- Khi so sánh ở cùng một mức dung lượng cache lưu giữ, việc đầu tư tài nguyên computational budget cho expert ($k$) đem lại hiệu quả vượt trội hơn so với cửa sổ ngữ cảnh ($W$): Cấu hình (k=2, W=16) cho kết quả tối ưu hơn hẳn so với (k=1, W=64) dù kích thước bộ nhớ đệm cache nhỏ hơn.
- Chỉ số tốc độ giải mã tok/s bị nghẽn phần lớn bởi các chi phí overhead xử lý của ngôn ngữ Python ở kích thước mô hình này nên các con số chỉ mang tính tham khảo sơ bộ; bảng benchmark CUDA dạng xử lý theo lô (batched) ở phía trên mới là tham chiếu chuẩn về tốc độ thực tế.

**Hạn chế:** Thử nghiệm mới thực hiện trên cấu hình 1 seed duy nhất cho mỗi phân loại, độ dài chuỗi cố định ở mức 256, kích thước mô hình nhỏ và sử dụng mã hóa dạng byte-level token.

## Thí nghiệm trên tiếng Việt thật

```bash
pip install torch datasets
python lm_experiment.py --mode adaptive --target_k 3.0   # adaptive
python lm_experiment.py --mode fixed --k 3               # baseline top-k cố định
```

Script train LM nhỏ (tokenize theo âm tiết) rồi in val loss, số expert trung bình, tương quan giữa k và loss của token, và k theo loại token (dấu câu, số, từ chức năng, tên riêng, từ hiếm, từ nội dung). Dùng `--text_file` nếu muốn dữ liệu riêng.

Tùy chọn: `--ste` (gradient chảy vào ranh giới chọn expert) và `--diff_head` (router có đầu dự đoán độ khó của token, huấn luyện bằng loss thật, rồi đặt `top_p` theo từng token trong `[p_min, p_max]`). `--mode random` chạy đối chứng k ngẫu nhiên theo token, `--no_renorm` giữ nguyên xác suất gate (kiểu Switch) để top-1 vẫn có gradient cho router.

## Các thực nghiệm về ngân sách co giãn (k, bộ nhớ đệm KV, cửa sổ ngữ cảnh)

Tất cả các lượt chạy dưới đây đều sử dụng cấu hình 1 seed duy nhất. Nhiễu kết quả giữa các lần chạy trên cùng một seed (do tính bất định cấu trúc của GPU) rơi vào khoảng 0.002-0.003 nat, vì vậy chỉ nên so sánh đối chiếu dữ liệu giữa các ô trong cùng một bảng hoặc cùng một lượt chạy. Các câu lệnh chi tiết có trong phần `--help` của mã nguồn script.

### 1. Elastic k: Huấn luyện một lần với k ngẫu nhiên, tùy chọn k khi suy luận

Cấu hình 8 expert/layer, 19.3M tham số, tập dữ liệu 7.5M token, chạy 3000 steps. Kết quả giá trị val loss khi ép mức k cố định lúc suy luận:

| k | 1 | 2 | 3 | 4 |
|---|---|---|---|---|
| Elastic (được train với k~U{1..4}) | 4.1014 | 4.0765 | 4.0734 | 4.0739 |
| Fixed k=3 / fixed k=4 (được train cố định) | | | 4.0720 | 4.0727 |

Mô hình elastic duy nhất này chỉ lệch khoảng 0.0015 nat so với các mô hình fixed-k độc lập. Nếu đánh giá bằng k ngẫu nhiên (thay vì ép một mức k cố định), giá trị đạt 4.0817; do đó việc ép một mức k cụ thể lúc suy luận mang lại hiệu quả tối ưu hơn.

Cấu hình 32 expert/layer, 19.4M tham số, tập dữ liệu nhỏ 1.98M token (bị hiện tượng quá khớp - overfit rất nặng), 3000 steps, huấn luyện với k~U{1..16}:

| k | 1 | 2 | 4 | 8 | 16 |
|---|---|---|---|---|---|
| Elastic, ép k cố định | 4.4088 | 4.3110 | 4.2762 | 4.2628 | 4.2593 |
| Fixed k=4 / k=8 | | | 4.2873 | 4.2908 | |

Trong trường hợp này, mô hình elastic cho kết quả tốt hơn mô hình fixed-k ở cùng cấu hình mức tính toán (giảm được 0.011 nat ở k=4 và 0.028 nat ở k=8). Với lượng dữ liệu ít như vậy, cơ chế k ngẫu nhiên nhiều khả năng đóng vai trò như một bộ điều hòa tính toán (regularizer). Khi test trên tập dữ liệu lớn hơn ở cấu hình 8 expert thì không thấy hiệu ứng này nữa, vì thế kết quả này không nên được hiểu là "elastic sẽ luôn đánh bại fixed" trên mọi bài toán. Chất lượng mô hình có xu hướng đi ngang từ mức k>=8, và cấu hình k=1 bị giảm mất 0.15 nat so với k=8.

### 2. Bộ nhớ đệm KV cache cho quá trình tạo văn bản (`elastic_kv.py`)

Cơ chế incremental decoding (giải mã tăng trưởng từng bước) có sử dụng cache cho kết quả khớp hoàn toàn so với việc tính toán lại toàn bộ chuỗi từ đầu: độ lệch logit lớn nhất nhỏ hơn 1e-6 tại các mức k = 1, 2, 4, 8, đồng thời quá trình tạo chuỗi theo thuật toán tham lam (greedy generation) cho ra các id token trùng khớp tuyệt đối. Đối với cơ chế cascade dựa trên độ tự tin, các token cũ đã lưu trong cache sẽ giữ nguyên mức k mà chúng được tính toán trước đó, do đó đầu ra của cascade theo thiết kế sẽ không trùng khớp hoàn toàn với việc chạy lại từ đầu toàn bộ chuỗi ở duy nhất một mức k.

### 3. Cửa sổ ngữ cảnh co giãn (`elastic_ctx.py`)

Một token ở vị trí $i$ sẽ chỉ chú ý (attend) đến các vị trí $j$ thỏa mãn `j < sink` hoặc `i - j < W` (với cấu hình cố định `sink = 4`). Khi suy luận, KV cache chỉ lưu lại tối đa `sink + W` token, mang lại kết quả đầu ra trùng khớp tuyệt đối so với việc chạy lại toàn bộ chuỗi với cùng một mặt nạ (được xác thực thông qua lệnh tự kiểm tra `--selftest`). Cấu hình: byte-level LM, 3.9M tham số, 8 experts, độ dài chuỗi 256, 8 MB dữ liệu lấy từ tiếng Việt Wikipedia (`wikimedia/wikipedia`, `20231101.vi`), chạy 3000 steps, cấu hình cố định k=2.

Chỉ co giãn cửa sổ ngữ cảnh, ép W lúc suy luận:

| Cách huấn luyện | W=8 | 16 | 32 | 64 | 128 | 256 (full) |
|---|---|---|---|---|---|---|
| Ngữ cảnh đầy đủ (Full context) | 1.2245 | 1.2059 | 1.1962 | 1.1913 | 1.1894 | 1.1890 |
| W ngẫu nhiên theo từng mẫu | 1.2040 | 1.1955 | 1.1904 | 1.1883 | 1.1875 | 1.1876 |

Chiến lược huấn luyện với W ngẫu nhiên hỗ trợ cải thiện chất lượng mạnh nhất ở các mức W nhỏ (giảm khoảng 0.02 nat ở W=8, và 0.01 nat ở W=16) và không làm mất đi độ chính xác khi chạy ở cấu hình toàn ngữ cảnh đầy đủ.

Đồng thời co giãn cả k và W (một mô hình duy nhất, k thuộc {1,2,4,8}, W thuộc {8..256}, sử dụng thiết lập `top_p=0.999` để giới hạn trần k có hiệu lực bắt buộc).  
Mỗi ô đại diện cho: **val loss / tốc độ giải mã (tok/s)** khi dùng KV cache. Số lượng KV token được giữ lại tương ứng: 12 / 20 / 36 / 68 / 132 / 256.

| k | W=8 | 16 | 32 | 64 | 128 | 256 |
|---|---|---|---|---|---|---|
| **1** | 1.2215 / 79 | 1.2129 / 82 | 1.2083 / 81 | 1.2069 / 83 | 1.2059 / 81 | 1.2060 / 84 |
| **2** | 1.2099 / 79 | 1.2021 / 79 | 1.1973 / 80 | 1.1955 / 79 | 1.1947 / 78 | 1.1947 / 81 |
| **4** | 1.2091 / 71 | 1.2013 / 74 | 1.1965 / 73 | 1.1947 / 73 | 1.1939 / 73 | 1.1938 / 75 |
| **8** | 1.2092 / 67 | 1.2013 / 68 | 1.1964 / 66 | 1.1946 / 67 | 1.1938 / 66 | 1.1937 / 69 |

- Giảm bộ đệm từ 256 xuống còn 68 token (W=64) làm tốn thêm khoảng 0.0008 nat; xuống 36 token tốn khoảng 0.0026 nat; và xuống 20 token tốn khoảng 0.0074 nat.
- Mức k=1 làm tốn thêm khoảng 0.011 nat so với k=2. Các mức k=4 và k=8 chỉ giúp cải thiện cực kỳ nhỏ khoảng 0.001 nat so với k=2 nhưng tốc độ giải mã lại chậm đi lần lượt khoảng 7% và 15%, do đó cấu hình k=2 chính là điểm cân bằng lý tưởng (sweet spot) cho mô hình kích thước nhỏ này.
- Khi so sánh ở cùng một mức dung lượng cache lưu giữ, việc đầu tư tài nguyên computational budget cho expert ($k$) đem lại hiệu quả vượt trội hơn so với cửa sổ ngữ cảnh ($W$): Cấu hình (k=2, W=16) cho kết quả tối ưu hơn hẳn so với (k=1, W=64) dù kích thước bộ nhớ đệm cache nhỏ hơn.
- Chỉ số tốc độ giải mã tok/s bị nghẽn phần lớn bởi các chi phí overhead xử lý của ngôn ngữ Python ở kích thước mô hình này nên các con số chỉ mang tính tham khảo sơ bộ; bảng benchmark CUDA dạng xử lý theo lô (batched) ở phía trên mới là tham chiếu chuẩn về tốc độ thực tế.

**Hạn chế:** Thử nghiệm mới thực hiện trên cấu hình 1 seed duy nhất cho mỗi phân loại, độ dài chuỗi cố định ở mức 256, kích thước mô hình nhỏ và sử dụng mã hóa dạng byte-level token.

### Phân bổ k theo từng token và từng layer (BPE 16000, 32 experts, 4 layers, d=128, seq 128)

Cấu hình: 8 MB Wikipedia tiếng Việt, từ vựng BPE 16000 train trên tập train, chia train/val ngẫu nhiên theo dòng, 3000 steps, `top_p=0.999` để giới hạn trần k có hiệu lực. Scripts: `headroom.py`, `tokenk.py`, `layerk.py` (`tokenk.py` và `layerk.py` yêu cầu `adaptive_moe.py` >= 0.3.1, nơi `set_k_cap` chấp nhận cả tensor theo từng token). Chi phí (cost) là số lượng expert định tuyến trung bình cho mỗi token trên mỗi layer (chưa tính shared expert và attention).

#### Quy tắc theo token: ưu tiên k=2 cho các token hiếm (tra bảng tần suất trước tính toán)

Mô hình huấn luyện với k ngẫu nhiên trong {1,2} theo từng token; val loss khi ép k theo token, thử nghiệm trên 3 seeds. Giá trị `rare - random` tương ứng với cùng tỷ lệ f token được nâng cấp lên k=2 (số âm đồng nghĩa với việc ưu tiên token hiếm cho kết quả tốt hơn, đơn vị: nats):

| seed | f=0.25 | f=0.5 | f=0.75 |
|---|---|---|---|
| 0 | -0.0061 | -0.0069 | -0.0044 |
| 1 | -0.0075 | -0.0051 | -0.0056 |
| 2 | -0.0043 | -0.0062 | -0.0065 |
| 0, train trực tiếp với quy tắc token hiếm | -0.0122 | -0.0101 | -0.0079 |

Quy tắc ngược lại (nâng cấp k=2 cho các token phổ biến nhất) luôn cho kết quả tệ hơn ngẫu nhiên (khoảng +0.005 đến +0.006 nat), chứng minh hiệu ứng này không phải do nhiễu. Tổng mức cải thiện từ k=1 lên k=2 ở các mô hình này rơi vào khoảng 0.034-0.042 nats, vì vậy lợi ích tuyệt đối khá nhỏ. Cách này giúp lấp đầy khoảng chi phí giữa k=1 và k=2 mà cơ chế fixed top-k không thể chạm tới.

#### Quy tắc k theo layer: hiệu ứng tác động lớn hơn

Một mô hình được huấn luyện với k được chọn độc lập theo từng layer từ {1,2,4} ở mỗi bước. Cả 81 lịch trình phân bổ (k1,k2,k3,k4) đều đã được đánh giá; lịch trình tối ưu nhất theo từng mức chi phí được chọn dựa trên nửa tập validation A và báo cáo val loss trên nửa tập B.

Val loss trên nửa tập B (seed 0 / seed 1):

| Cấu hình | cost | seed 0 | seed 1 |
|---|---|---|---|
| k=1 đồng đều | 1.0 | 4.7354 | 4.7081 |
| k=2 đồng đều | 2.0 | 4.6880 | 4.6571 |
| k=4 đồng đều | 4.0 | 4.6791 | 4.6519 |
| lịch trình (1,1,2,2) | 1.5 | 4.6856 | 4.6563 |
| lịch trình (1,1,2,4) | 2.0 | 4.6771 | 4.6470 |
| lịch trình (1,1,4,4) | 2.5 | 4.6691 | 4.6426 |

Ở mức chi phí cost = 2.0, lịch trình phân bổ tốt nhất vượt trội hơn k=2 đồng đều khoảng 0.011 / 0.010 nats, và lịch trình (1,1,4,4) ở chi phí 2.5 đánh bại k=4 đồng đều ở chi phí 4.0 khoảng 0.010 / 0.009 nats. Loss trung bình theo k ở mỗi layer (tập A, seed 0): Layer 0 và 1 gần như đi ngang hoặc tệ hơn nhẹ khi tăng k (4.6186 / 4.6198 / 4.6274 và 4.6197 / 4.6207 / 4.6254 tương ứng với k = 1 / 2 / 4), trong khi Layer 2 và 3 cải thiện rất mạnh (4.6415 / 4.6160 / 4.6083 và 4.6436 / 4.6154 / 4.6067). Seed 1 cho ra xu hướng hoàn toàn tương tự.

Hạn chế: Mới chỉ đánh giá (eval-only) trên một mô hình elastic duy nhất cho mỗi seed, mô hình nhỏ 4 layers, chạy trên 2 seeds. Cần được kiểm chứng thêm bằng cách train từng lịch trình từ đầu (`schedule_train.py`), và chưa rõ mô hình sâu hơn có giữ nguyên quy luật này hay không.

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
