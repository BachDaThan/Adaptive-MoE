# adaptive-moe

[![tests](https://github.com/BachDaThan/Adaptive-MoE/actions/workflows/tests.yml/badge.svg)](https://github.com/BachDaThan/Adaptive-MoE/actions/workflows/tests.yml)

[English](README.md) | **Tiếng Việt**

Module PyTorch nhỏ (thử nghiệm) cho MoE **thích ứng**: số chuyên gia được kích hoạt và mức đóng góp của từng chuyên gia thay đổi theo độ khó của từng token, thay vì cố định top-6 hay top-12.

Ai train model MoE cũng có thể thay layer MoE của mình bằng `AdaptiveMoE`.

> **Trạng thái: thử nghiệm.** Cơ chế hoạt động (router học được độ khó của token và dùng nhiều expert hơn cho token khó), nhưng không thắng các baseline đơn giản: adaptive routing hơi kém top-k cố định ở cùng ngân sách trong phần lớn cấu hình, và ở ngân sách thấp chỉ ngang việc chọn k ngẫu nhiên theo token với cùng ngân sách trung bình. Xem [Kết quả thí nghiệm](#kết-quả-thí-nghiệm-trạng-thái-thử-nghiệm).
>
> Điều có tác dụng thật là **huấn luyện co giãn**: train với k ngẫu nhiên (và/hoặc cửa sổ ngữ cảnh ngẫu nhiên) thì cùng một model chạy được ở mọi mức ngân sách, không mất gì khi dữ liệu dồi dào. Khi dữ liệu ít, nó tốt hơn train k cố định thông thường và thua k cố định có dropout nếu bản elastic không có dropout, nhưng elastic kết hợp dropout hơn k cố định cùng dropout khoảng 0.02 nat; khi mỗi bên được chỉnh dropout riêng, elastic vẫn hơn khoảng 0.013 nat ở k=2 khi train lâu hơn (6000 bước, cả 4 seed; ngang nhau ở 3000 bước) và khoảng 0.024 nat ở k=1. Xem [Huấn luyện co giãn](#huấn-luyện-co-giãn-một-model-nhiều-mức-ngân-sách).

## Công thức khuyến nghị (những gì thí nghiệm ủng hộ)

Dùng khi train LM MoE nhỏ với dữ liệu hạn chế trên Colab T4 miễn phí. Bằng chứng đến từ model khoảng 20M tham số; kích thước lớn hơn chưa được kiểm chứng.

1. **Train với k elastic, chọn k lúc suy luận.** Mỗi bước, lấy một tỉ lệ f token trong {0, 0.25, 0.5, 0.75, 1} dùng k=2, phần còn lại dùng k=1 (`set_k_cap(model, tensor)`). Một model phục vụ được cả hai ngân sách. Khi dữ liệu dồi dào nó ngang các model k cố định train riêng; khi dữ liệu ít nó tốt hơn khoảng 0.013 nat ở k=2 và 0.024 nat ở k=1 (mục 4.2).
2. **Dùng dropout 0.15 đến 0.2** trên embedding và đầu ra MoE khi dữ liệu ít. Đây là đòn bẩy lớn nhất đã đo (khoảng 0.08 nat); 0.3 là quá mạnh.
3. **Giữ checkpoint tốt nhất theo val loss.** Với dữ liệu nhỏ, model overfit ngay sau khoảng 3000 bước.
4. **Một BPE byte-level dùng chung** cho nhiều ngôn ngữ tốn rất ít (tiếng Việt chỉ cần thêm 3% token khi tiếng Anh dùng chung vocab 16k, mục 7).
5. **Bỏ qua** router top-p thích ứng, lịch k theo layer và phân bổ k thông minh theo token: không cái nào thắng k cố định đơn giản ở cùng ngân sách (mục 5 và phần kết quả thí nghiệm).
6. `train_lm.py` hiện thực công thức này với fp16, gradient accumulation, checkpoint trên Google Drive và tự tiếp tục khi bị ngắt.

## Cài

```
pip install git+https://github.com/BachDaThan/Adaptive-MoE
```

hoặc chép `adaptive_moe.py` vào project (chỉ cần PyTorch).

## Thành phần

| Thành phần                   | Làm gì                                                                                                                                                           |
|------------------------------|------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `AdaptiveRouter`             | Top-p routing: lấy expert theo xác suất giảm dần cho tới khi tổng đạt `top_p`. Token dễ dùng ít expert, token khó dùng nhiều.                                    |
| Strength head                | Mỗi token có hệ số "sức mạnh" trong `[min_strength, 1]` nhân vào đóng góp của expert được chọn.                                                                  |
| `target_k`                   | Kéo số expert trung bình về gần giá trị bạn muốn (vd 3.0), dễ kiểm soát hơn phạt thô.                                                                            |
| `n_shared`                   | Expert luôn bật (kiểu DeepSeek), giúp chất lượng ổn khi router chỉ chọn 1 expert.                                                                                |
| Loss phụ                     | Cân bằng tải, phạt compute, z-loss. Gom bằng `gather_aux_loss(model)`.                                                                                           |
| `moe_stats(model)`           | Xem `avg_k` và `avg_strength` của từng layer sau mỗi forward.                                                                                                    |
| `cascade_forward`            | Inference: chạy tầng nhỏ (vd 2 expert), kém tự tin thì nâng lên 4, rồi 8. Độ tự tin: `maxprob` hoặc `entropy`.                                                   |
| `adaptive_generate`          | Sinh text, mỗi bước tự chọn tầng.                                                                                                                                |
| `diff_head` (thử nghiệm)     | Router có đầu dự đoán độ khó của token, đặt `top_p` theo từng token. Xem mục kết quả bên dưới.                                                                   |
| `max_resident` + `release()` | Chỉ giữ N expert trên GPU theo LRU; `release()` dọn bớt sau khi trả lời.                                                                                         |
| `set_k_cap(model, k)`        | Đặt trần số expert mỗi token (khi inference hoặc train). `k` là số nguyên cho mọi token, hoặc tensor số nguyên `[B*S]` để đặt trần riêng từng token (bản 0.3.1). |

## Dùng

```
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

```
pip install torch datasets
python lm_experiment.py --mode adaptive --target_k 3.0   # adaptive
python lm_experiment.py --mode fixed --k 3               # baseline top-k cố định
python compare.py                                        # bảng tổng hợp các file r_*.json
```

Script train LM nhỏ (tokenize theo âm tiết) rồi in val loss, số expert trung bình, tương quan giữa k và loss của token, và k theo loại token (dấu câu, số, từ chức năng, tên riêng, từ hiếm, từ nội dung). Nhãn trong báo cáo được in bằng tiếng Việt. Dùng `--text_file` nếu muốn dữ liệu riêng.

Tùy chọn: `--ste` (gradient chảy vào ranh giới chọn expert) và `--diff_head` (router có đầu dự đoán độ khó của token, huấn luyện bằng loss thật, rồi đặt `top_p` theo từng token trong `[p_min, p_max]`). `--layer_k 1,2,2,3` đặt k cố định cho từng layer. `--mode random` chạy đối chứng k ngẫu nhiên theo token, `--no_renorm` giữ nguyên xác suất gate (kiểu Switch) để top-1 vẫn có gradient cho router.

## Kết quả thí nghiệm (trạng thái: thử nghiệm)

Thiết lập: LM 19M tham số, 4 layer, 8 expert/layer, tokenize theo âm tiết, Wikipedia tiếng Việt (30M ký tự, ~7.5M token), 3000 bước. Fixed k=2 và adaptive `compute_coef=0` chạy 3 seed (trung bình, khoảng min–max), các cấu hình còn lại một seed. Val loss thấp hơn là tốt hơn.

| Cấu hình                                                                                              | avg_k | val loss                |
|-------------------------------------------------------------------------------------------------------|-------|-------------------------|
| top-k cố định, k=1 (trọng số chuẩn hóa)                                                               | 1.00  | 4.194                   |
| top-1 cố định, giữ nguyên xác suất gate (`--no_renorm`)                                               | 1.00  | 4.110                   |
| top-2 cố định, giữ nguyên xác suất gate (`--no_renorm`)                                               | 2.00  | 4.083                   |
| top-k cố định, k=2 (3 seed)                                                                           | 2.00  | **4.078** (4.077–4.080) |
| k cố định theo layer (1,2,2,3)                                                                        | 2.00  | 4.115                   |
| k cố định theo layer (1,2,3,3)                                                                        | 2.25  | 4.115                   |
| adaptive + `diff_head`, `compute_coef=0.5`                                                            | 1.90  | 4.150                   |
| adaptive + `diff_head`, `diff_target=gain`, `compute_coef=0.5`                                        | 1.88  | 4.148                   |
| adaptive + `diff_head`, `compute_coef=0` (3 seed)                                                     | 2.10  | 4.085 (4.081–4.088)     |
| adaptive + `diff_head`, `compute_coef=0`, ngân sách thấp (3 seed)                                     | 1.46  | 4.090 (4.083–4.098)     |
| k ngẫu nhiên theo token trong {1,2}, 46% token nhận 2 (đối chứng, 1 seed)                             | 1.46  | 4.092                   |
| train với k ngẫu nhiên (50/50), rồi chạy ở k=1 (`--eval_cascade`)                                     | 1.00  | 4.096                   |
| train với k ngẫu nhiên (50/50), rồi chạy ở k=2                                                        | 2.00  | 4.082                   |
| 32 expert (expert định tuyến d_ff=128, dùng chung d_ff=512), top-8 cố định                            | 8.00  | 4.113                   |
| 32 expert (expert định tuyến d_ff=128, dùng chung d_ff=512), adaptive + `diff_head`, `compute_coef=0` | 7.08  | 4.126                   |
| 32 expert (như trên), top-7 cố định                                                                   | 7.00  | 4.119                   |
| 32 expert (như trên), top-6 cố định                                                                   | 6.00  | 4.126                   |

[![Val loss theo số expert trung bình: adaptive so với top-k cố định](docs/loss_vs_k.png)](docs/loss_vs_k.png)

Điểm nằm dưới đường top-k cố định là tốt hơn ở cùng ngân sách. Điểm k=1 chuẩn hóa là mốc tham chiếu yếu (một expert đã chuẩn hóa không cho router gradient từ task loss). Khi giữ nguyên xác suất gate (`--no_renorm`), k=1 cố định đạt 4.110 (ô vuông xanh), và các điểm adaptive, ngẫu nhiên ở k≈1.46 nằm trong khoảng 0.005 nat so với đường thẳng nét đứt nối điểm đó với k=2 cố định, tức chỉ là một đánh đổi tuyến tính thông thường. Biểu đồ bên phải chỉ trải trong 0.015 nat.

Rút ra:

- Router học được độ khó của token (tương quan với loss thật khoảng 0.4 đến 0.5) và số expert tăng theo độ khó (tương quan k và loss từ +0.26 đến +0.37).
- **Phân bổ đã học không thắng phân bổ ngẫu nhiên ở cùng ngân sách.** Ở khoảng 1.46 expert mỗi token (8 expert), adaptive với `compute_coef=0` đạt 4.090 (3 seed: 4.083 đến 4.098), còn đối chứng cho mỗi token k=1 hoặc k=2 ngẫu nhiên với cùng ngân sách trung bình đạt 4.092 (1 seed). Top-1 cố định với xác suất gate giữ nguyên (`--no_renorm`) đạt 4.110; baseline k=1 chuẩn hóa (4.194) bị thiệt vì một expert đã chuẩn hóa không cho router gradient từ task loss. So với mốc đã sửa, các điểm adaptive và ngẫu nhiên ở k≈1.46 nằm trong khoảng 0.005 nat so với đường thẳng nối k=1 (4.110) và k=2 (4.078): một đánh đổi tuyến tính khoảng 0.03 nat mỗi expert định tuyến, không phải lợi ích đặc biệt. (Khi cả hai đầu cùng giữ nguyên gate, k=2 cho 4.083, nên đường tham chiếu chạy từ 4.110 đến 4.083, khoảng 0.027 nat mỗi expert định tuyến; các điểm adaptive và ngẫu nhiên nằm thấp hơn đường đó khoảng 0.005 đến 0.007 nat, tương đương nhiễu giữa các seed.) Nâng tầng theo độ bất định của chính model (entropy của lượt k=1 rẻ) chỉ hơn nâng ngẫu nhiên 0.0014 nat ở ngân sách 50% (`--eval_cascade`), và toàn bộ khoảng từ toàn k=1 đến toàn k=2 chỉ trải 0.014 nat, nên gần như không còn gì để bộ điều khiển lấy thêm. (Cột "oracle" trong báo cáo đó không phải trần đạt được: nhìn lại với hindsight nó chọn cái tốt hơn trong hai loss nhiễu của từng token.) Ở khoảng 2 expert mỗi token (8 expert) và ở 32 expert, adaptive thua nhẹ top-k cố định ở cùng ngân sách (khoảng 0.007 nat).
- Expert mịn hơn (32 thay vì 8, tổng tham số tương đương): lần chạy đầu có vẻ ủng hộ adaptive, nhưng đó là hiệu ứng giả do expert dùng chung bị teo theo `d_ff`. Khi giữ expert dùng chung rộng (`--d_ff_shared 512`), top-8 cố định (4.113) thắng adaptive (4.126), dù adaptive dùng ít hơn khoảng 11% expert và k phân hóa rộng (4.8 ở nhóm token dễ nhất, 8.3 ở nhóm khó nhất). Hai bản đối chứng top-7 (4.119) và top-6 (4.126) cho câu trả lời: adaptive ở k=7.08 chỉ ngang top-6 cố định mà dùng nhiều expert hơn, tức thua ở cùng ngân sách khoảng 0.007 nat, đúng bằng khoảng cách ở bản 8 expert. Đường cong của bản cố định cũng rất phẳng (khoảng 0.0065 nat mỗi expert từ k=6 đến k=8), nên ở quy mô này gần như không còn chỗ cho cách phân bổ nào. Cấu hình 32 expert này cũng kém bản 8 expert (4.078) khi train ngắn như vậy.
- Quan sát ổn định giữa các seed: khi để router tự do, các layer sâu xin nhiều expert hơn (layer 1 khoảng 1.3 đến 1.6, layer 4 khoảng 2.7 đến 2.8). Tuy vậy gán cứng k theo layer (1,2,2,3) lại kém k đồng đều, nên chưa thể dùng điều này như một công thức.
- "Token khó" không đồng nghĩa "token được lợi khi thêm expert": nhãn dạy theo loss đưa nhiều expert cho từ hiếm và từ chức năng, trong khi từ nội dung mới là loại được lợi nhiều nhất khi thêm expert. Nhãn theo mức loss giảm được (`gain`) quá nhiễu nên đầu dự đoán không học được.
- `compute_coef=0.5` giúp như một regularizer khi dữ liệu nhỏ và model overfit (1.9M token), nhưng làm hại khi dữ liệu đủ lớn.

Giới hạn: phần lớn cấu hình chỉ một seed, model nhỏ, train ngắn, chưa đo tốc độ thật (vòng lặp theo expert chưa được tối ưu). Kết quả ở model lớn có thể khác.

## Huấn luyện co giãn: một model, nhiều mức ngân sách

Tóm tắt các thí nghiệm bên dưới (model nhỏ, train ngắn; đa số cấu hình chạy 1 đến 3 seed):

- **Elastic k:** train với trần số expert ngẫu nhiên rồi chọn k lúc suy luận. Khi dữ liệu dồi dào, một model elastic lệch khoảng 0.0015 nat so với các model fixed-k riêng. Khi dữ liệu ít, nó hơn train fixed-k thông thường (khoảng 0.03 đến 0.04 nat ở bước dừng sớm, 0.08 nat ở bước cuối) nhưng thua fixed-k có dropout khoảng 0.06 nat nếu bản elastic không có dropout; elastic kết hợp dropout lại hơn fixed-k cùng dropout khoảng 0.02 nat; khi mỗi bên được chỉnh dropout riêng, elastic vẫn hơn khoảng 0.013 nat ở k=2 khi train lâu hơn (6000 bước, cả 4 seed; ngang nhau ở 3000 bước) và khoảng 0.024 nat ở k=1 (mục 4).
- **KV cache và cửa sổ ngữ cảnh:** sinh văn bản với KV cache khớp hoàn toàn với chạy lại cả chuỗi; cửa sổ sink + trượt chặn cache ở `sink + W` token mà gần như không mất chất lượng (mục 2 và 3).
- **Phân bổ k thông minh hơn không thắng một k cố định ở cùng chi phí trung bình.** Chỉ có hiệu ứng nhỏ trong khoảng chi phí giữa k=1 và k=2 (mục 5). Lịch k theo layer từng có vẻ tốt trên model elastic **không đứng vững** khi train từ đầu, và cũng không thắng k đồng đều khi có dropout (mục 5.3 và 4.2).
- **Nhiều ngôn ngữ trong một model (mục 7, 1 seed):** BPE byte-level 16k dùng chung chỉ làm tiếng Việt tốn thêm 3% token, và mức dùng expert của router khác nhau rõ rệt giữa tiếng Việt và tiếng Anh mà không ai giám sát.

Nhiễu giữa các lần chạy cùng seed (GPU không tất định) khoảng 0.002-0.003 nat, nên chỉ so sánh các ô trong cùng một bảng hoặc một lần chạy.

### Các script

| Script                                                                           | Làm gì                                                                                       |
|----------------------------------------------------------------------------------|----------------------------------------------------------------------------------------------|
| `elastic_kv.py`                                                                  | Sinh văn bản có KV cache với cascade theo độ tự tin (`adaptive_generate_kv`), kèm self-test. |
| `elastic_ctx.py`                                                                 | Cửa sổ sink + trượt với KV cache bị chặn bộ nhớ; `--selftest`, `--train`, `--grid`.          |
| `headroom.py`                                                                    | Ước lượng ở mức token xem bộ phân bổ k theo token có thể lợi được bao nhiêu.                 |
| `tokenk.py`                                                                      | k theo từng token chạy thật trong các layer (cần `adaptive_moe.py` >= 0.3.1).                |
| `layerk.py`                                                                      | Quét các lịch k theo layer trên một model elastic.                                           |
| `schedule_train.py`                                                              | Train từng lịch k từ đầu (bước kiểm chứng cho `layerk.py`).                                  |
| `regcheck.py`                                                                    | Đối chứng dừng sớm và regularize: fixed-k (+dropout, +weight decay) so với elastic (mục 4.2).  |
| `summarize_reg.py`                                                               | Gộp log của `regcheck.py` (trung bình theo seed, so sánh theo cặp).                          |
| `langcheck.py`                                                                   | Một BPE byte-level chung cho nhiều ngôn ngữ; đếm token, loss và mức dùng expert theo từng ngôn ngữ (mục 7). |
| `train_lm.py`                                                                    | Train LM MoE thật theo công thức trên (k elastic, dropout, fp16, checkpoint lên Drive, tự tiếp tục, `--generate`). |
| `run_both.sh`, `run_ctx.sh`, `run_tokenk.sh`, `run_layerk.sh`, `run_schedule.sh`, `run_reg.sh`, `run_reg2.sh`, `run_reg3.sh`, `run_reg4.sh` | Chạy trọn các script trên.                                                |

```
python elastic_ctx.py --selftest
python elastic_ctx.py --grid --top_p 0.999 --hf wikimedia/wikipedia --config 20231101.vi --field text --mb 8
python headroom.py --hf wikimedia/wikipedia --config 20231101.vi --field text --mb 8 --bpe --experts 32 --k_hi 2 --L 128
bash run_tokenk.sh
bash run_schedule.sh 0
bash run_reg.sh 0         # mục 4.2, vòng 1 (~40 phút/seed)
bash run_reg2.sh 0        # mục 4.2, vòng 2 (~40 phút/seed)
bash run_reg3.sh A 0      # vòng 3A: dropout 0.2 và 0.3 cho elastic và fixed
bash run_reg3.sh B 0      # vòng 3B: nhóm dropout chạy 6000 bước
bash run_reg4.sh B 2 3    # vòng 4B: seed 2 và 3 ở 6000 bước (xem đầu file để biết các phần A, C, D)
python langcheck.py --langs vi,en   # mục 7: tokenizer chung, thống kê router theo ngôn ngữ
```

### 1. Elastic k: train một lần với k ngẫu nhiên, chọn k lúc suy luận

8 expert/layer, 19.3M tham số, corpus 7.5M token, 3000 bước (`--mode random --uniform_k --k_lo 1 --k_hi 4 --eval_ks 1,2,3,4`). Val loss khi ép k lúc suy luận:

| k                                   | 1      | 2      | 3      | 4      |
|-------------------------------------|--------|--------|--------|--------|
| Elastic (train với k~U{1..4})       | 4.1014 | 4.0765 | 4.0734 | 4.0739 |
| Top-k cố định, train và chạy cùng k | 4.110* | 4.078  | 4.0720 | 4.0727 |

\* top-1 cố định với xác suất gate giữ nguyên (`--no_renorm`). k=2 cố định là trung bình 3 seed; k=3 và k=4 mỗi cái một seed.

Một model elastic lệch khoảng 0.0015 nat so với model fixed-k riêng ở k=2..4, và ở k=1 không tệ hơn top-1 train riêng. Nhiễu giữa các seed khoảng 0.002 đến 0.004 nat. Train với k ngẫu nhiên có trung bình 2.5 expert mỗi token, nhưng khi đánh giá với k ngẫu nhiên chỉ đạt 4.082, nên ép một k cố định lúc suy luận tốt hơn để k ngẫu nhiên.

32 expert/layer, 19.4M tham số, corpus nhỏ hơn 1.98M token (nên overfit nặng), 3000 bước, train k~U{1..16}:

| k                 | 1      | 2      | 4      | 8      | 16     |
|-------------------|--------|--------|--------|--------|--------|
| Elastic, ép k     | 4.4088 | 4.3110 | 4.2762 | 4.2628 | 4.2593 |
| k=4 / k=8 cố định |        |        | 4.2873 | 4.2908 |        |

Ở đây model elastic tốt hơn fixed-k cùng k (0.011 ở k=4 và 0.028 ở k=8), nhưng các model cố định này không có dừng sớm hay dropout, và dữ liệu nhỏ đang overfit nặng nên cần đọc cùng mục 4.2. Chất lượng phẳng từ k=8 trở lên và k=1 mất 0.15 nat so với k=8.

### 2. KV cache cho sinh văn bản (`elastic_kv.py`)

Giải mã tăng dần có cache khớp với tính lại toàn bộ: lệch logit tối đa dưới 1e-6 ở k = 1, 2, 4, 8, và sinh greedy cho ra đúng dãy id. Với cascade theo độ tự tin, token cũ trong cache giữ k lúc nó được tính, nên kết quả cascade không y hệt việc chạy lại cả chuỗi ở một k cố định (theo thiết kế).

### 3. Cửa sổ ngữ cảnh co giãn (`elastic_ctx.py`)

Token ở vị trí $i$ chỉ nhìn các vị trí $j$ thỏa `j < sink` hoặc `i - j < W` (sink = 4). Lúc suy luận KV cache chỉ giữ `sink + W` token, và cho kết quả y hệt chạy lại cả chuỗi với cùng mặt nạ (kiểm bằng `--selftest`, cũng kiểm rằng sinh 200 token với W=16 chỉ giữ 20 token trong cache). Lúc train, W được bốc ngẫu nhiên theo từng mẫu. Thiết lập: LM byte-level 3.9M tham số (d=128, 4 layer, 8 expert), chuỗi dài 256, 8 MB Wikipedia tiếng Việt (`wikimedia/wikipedia`, `20231101.vi`), 3000 bước, k=2.

Chỉ co giãn cửa sổ, ép W lúc suy luận:

| Huấn luyện            | W=8    | 16     | 32     | 64     | 128    | 256 (đầy đủ) |
|-----------------------|--------|--------|--------|--------|--------|--------------|
| Đủ ngữ cảnh           | 1.2245 | 1.2059 | 1.1962 | 1.1913 | 1.1894 | 1.1890       |
| W ngẫu nhiên theo mẫu | 1.2040 | 1.1955 | 1.1904 | 1.1883 | 1.1875 | 1.1876       |

Hai lần chạy cùng thí nghiệm (cùng seed) cho kết quả khớp: train W ngẫu nhiên giúp nhiều nhất ở W nhỏ (khoảng 0.02 ở W=8, 0.01 ở W=16) và không mất gì ở ngữ cảnh đầy đủ (lệch trong 0.0015).

Co giãn đồng thời k và W (một model train với k thuộc {1,2,4,8} và W thuộc {8..256}, `top_p=0.999` để trần k thật sự chặn). Mỗi ô là **val loss / tok/s khi giải mã** có KV cache. Số token KV giữ: 12 / 20 / 36 / 68 / 132 / 256.

| k     | W=8         | 16          | 32          | 64          | 128         | 256         |
|-------|-------------|-------------|-------------|-------------|-------------|-------------|
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

Thiết lập: BPE vocab 16000 huấn luyện trên tập train, 1.48M token train (8 MB Wikipedia tiếng Việt, chia theo dòng ngẫu nhiên), 32 expert, 4 layer, d=128, chuỗi 128, batch 32, 3000 bước (khoảng 8 lượt qua dữ liệu), `top_p=0.999` để trần chặn thật. Model elastic train với k ngẫu nhiên theo token trong {1,2} (`tokenk.py`); model cố định train và chạy ở một k.

**4.1 Val loss ở bước cuối, không dừng sớm.** Mọi số là val loss ở bước 3000 trên cùng 40 batch (81,920 token). Model cố định chạy bằng `schedule_train.py`, một seed.

| Model                                   | k=1                      | k=2                      | k=4        |
|-----------------------------------------|--------------------------|--------------------------|------------|
| Cố định, train ở đúng k đó (seed 0)     | 4.6575                   | 4.7058                   | 4.8201     |
| Elastic, ép k lúc eval (seed 0 / 1 / 2) | 4.6613 / 4.6491 / 4.6463 | 4.6236 / 4.6120 / 4.6122 | chưa train |

- Ở bước cuối, train cố định với nhiều expert hoạt động hơn lại tệ hơn: k=4 tệ hơn k=1 tới 0.16 nat. Loss train ở bước cuối của các lần chạy elastic khoảng 3.0 đến 3.3 trong khi val loss trên 4.6, nên chế độ này thiếu dữ liệu và overfit. Mục 4.2 cho thấy phần lớn khoảng cách này là do overfit ở bước cuối.
- Ở bước cuối, model elastic ở k=2 tốt hơn k=2 cố định 0.08 nat và tốt hơn model cố định tốt nhất (k=1) từ 0.03 đến 0.05 nat.

**4.2 Đối chứng có dừng sớm và regularize (2 seed).** Mục 4.1 so elastic với model cố định không có dừng sớm hay regularize, nên chưa tách được "k ngẫu nhiên có ích" khỏi "chỉ là bớt overfit". Ở đây thêm baseline k cố định có dropout (`+drop`) và weight decay (`+wd`), cùng dữ liệu và thiết lập, seed 0 và 1, 3000 bước. Dropout 0.1 đặt trên embedding và đầu ra MoE của mỗi layer; `+wd` dùng weight decay 0.1 (các cấu hình khác 0.01, AdamW, lr 1e-3). Mỗi 250 bước đo val loss (20 batch) trên hai nửa độc lập của tập val: bước tốt nhất (dừng sớm) được chọn trên nửa A, `bestB` là loss nửa B tại bước đó, `finalB` là loss nửa B ở bước 3000. Model elastic ở đây, mỗi bước train, đưa một tỉ lệ f token ngẫu nhiên (f lấy từ {0, 0.25, 0.5, 0.75, 1}) lên k=2, phần còn lại k=1. Số là trung bình hai seed. Chạy bằng `run_reg.sh` (`regcheck.py`). Các số này không so trực tiếp được với 4.1 vì khác tập đánh giá. Model elastic ở đây không có dropout và không tăng weight decay.

| Cấu hình      | bestB (dừng sớm) | finalB (bước 3000) | Bước tốt nhất (seed 0, 1) |
|---------------|------------------|--------------------|---------------------------|
| fixed-k2+drop | **4.5122**       | 4.5168             | 3000, 2500                |
| fixed-k1+drop | 4.5365           | 4.5365             | 3000, 3000                |
| fixed-k2+wd   | 4.5589           | 4.6795             | 2000, 2000                |
| elastic@k2    | 4.5707           | 4.6870             | 2000, 2000                |
| elastic@k1    | 4.5980           | 4.7202             | 2000, 2000                |
| fixed-k2      | 4.6045           | 4.7722             | 1750, 2000                |
| fixed-k1      | 4.6390           | 4.7113             | 2000, 2000                |

Chênh lệch `bestB` theo từng seed (số âm = vế trái tốt hơn):

| So sánh                    | seed 0  | seed 1  | trung bình |
|----------------------------|---------|---------|------------|
| elastic@k2 − fixed-k2      | −0.0397 | −0.0280 | −0.0338    |
| elastic@k1 − fixed-k1      | −0.0347 | −0.0473 | −0.0410    |
| elastic@k2 − fixed-k2+wd   | +0.0112 | +0.0123 | +0.0118    |
| elastic@k2 − fixed-k2+drop | +0.0415 | +0.0753 | +0.0584    |
| elastic@k1 − fixed-k1+drop | +0.0492 | +0.0739 | +0.0615    |

- **k cố định có dropout thắng elastic không có dropout ở cùng k.** fixed-k2+drop hơn elastic@k2 khoảng 0.058 nat và fixed-k1+drop hơn elastic@k1 khoảng 0.062 nat, đúng dấu ở cả hai seed. Weight decay một mình cũng hơn elastic@k2 khoảng 0.012 nat ở cả hai seed. Thứ hạng giống nhau ở hai seed, trừ cặp elastic@k1 và fixed-k2 đổi chỗ (chênh dưới 0.001 ở seed 1).
- **Elastic vẫn hơn k cố định không regularize:** khoảng 0.034 nat ở k=2 và 0.041 nat ở k=1 tại bước dừng sớm. Nhưng lợi thế 0.08 nat ở bước cuối (4.1) co lại còn chưa tới một nửa khi dừng sớm, tức phần lớn khoảng cách đó là do model cố định overfit lâu hơn. Ở bước 3000, elastic@k1 thậm chí kém fixed-k1 khoảng 0.009 nat (4.7202 so với 4.7113).
- **Dừng sớm và dropout đổi thứ tự của k.** Ở bước cuối, k=1 cố định hơn k=2 cố định (4.7113 so với 4.7722), nhưng ở bước dừng sớm k=2 hơn k=1 khoảng 0.034 nat (4.6045 so với 4.6390), và với dropout k=2 hơn k=1 khoảng 0.024 nat (4.5122 so với 4.5365). Vậy kết luận "nhiều expert hơn thì tệ hơn" ở 4.1 chủ yếu là overfit ở bước cuối. Chưa chạy lại k=4 với dừng sớm.
- Các lần chạy có dropout đạt tốt nhất ở bước 3000 trong 3 trên 4 lần (một lần ở 2500) trong khi các cấu hình khác tốt nhất quanh bước 2000, nên 3000 bước có thể chưa đủ cho nhóm dropout. Chưa chạy dài hơn để kiểm tra.

**Vòng 2: elastic có dropout, dropout mạnh hơn, k=4 và k theo layer (2 seed).** Cùng thiết lập; `+drop` là dropout 0.1, `+drop2` là dropout 0.2. Model elastic ở vòng này được train có dropout 0.1. `sched-1124` là k theo layer (1,1,2,4), `sched-4211` là (4,2,1,1), cả hai chi phí trung bình 2.0; `sched-1122` là (1,1,2,2), chi phí 1.5. Chạy bằng `run_reg2.sh`.

| Cấu hình        | bestB (dừng sớm) | finalB (bước 3000) | Bước tốt nhất (seed 0, 1) |
|-----------------|------------------|--------------------|---------------------------|
| elastic+drop@k2 | **4.4870**       | 4.4870             | 3000, 3000                |
| fixed-k2+drop2  | 4.4923           | 4.4896             | 3000, 2750                |
| fixed-k4+drop   | 4.5040           | 4.5307             | 2500, 2500                |
| sched-1122+drop | 4.5189           | 4.5256             | 3000, 2750                |
| elastic+drop@k1 | 4.5199           | 4.5185             | 3000, 2750                |
| sched-1124+drop | 4.5234           | 4.5234             | 3000, 3000                |
| sched-4211+drop | 4.5330           | 4.5317             | 3000, 2750                |

Chênh lệch `bestB` theo từng seed (số âm = vế trái tốt hơn):

| So sánh                          | seed 0  | seed 1  | trung bình |
|----------------------------------|---------|---------|------------|
| elastic+drop@k2 − fixed-k2+drop  | −0.0390 | −0.0116 | −0.0253    |
| elastic+drop@k1 − fixed-k1+drop  | −0.0266 | −0.0064 | −0.0165    |
| elastic+drop@k2 − elastic@k2     | −0.0805 | −0.0869 | −0.0837    |
| elastic+drop@k2 − fixed-k2+drop2 | −0.0066 | −0.0040 | −0.0053    |
| fixed-k2+drop2 − fixed-k2+drop   | −0.0324 | −0.0076 | −0.0200    |
| fixed-k4+drop − fixed-k2+drop    | −0.0228 | +0.0063 | −0.0082    |
| sched-1124+drop − fixed-k2+drop  | +0.0077 | +0.0145 | +0.0111    |
| sched-4211+drop − fixed-k2+drop  | +0.0171 | +0.0244 | +0.0208    |

- **Elastic cộng dồn với dropout.** elastic+drop@k2 (4.4870) là cấu hình tốt nhất trong mọi bảng. Cùng dropout 0.1, elastic hơn k cố định khoảng 0.025 nat ở k=2 và 0.017 nat ở k=1, cùng dấu ở cả hai seed (độ lớn dao động nhiều, từ 0.012 đến 0.039 ở k=2). Thêm dropout cho elastic giúp 0.084 nat. Nên kết luận "k cố định có dropout thắng elastic" ở trên chỉ đúng với elastic không có dropout.
- **Dropout mạnh hơn thu hẹp khoảng cách.** fixed-k2+drop2 (dropout 0.2) hơn fixed-k2+drop 0.020 nat và chỉ kém elastic+drop@k2 (dropout 0.1) 0.005 nat, cùng dấu ở hai seed nhưng nằm trong mức nhiễu. Vòng 3A bên dưới chạy elastic với dropout 0.2 và 0.3 để trả lời câu hỏi này.
- **k=4 không rõ hơn k=2.** fixed-k4+drop hơn fixed-k2+drop 0.008 nat ở bestB nhưng khác dấu giữa hai seed, và finalB tệ hơn (4.5307 so với 4.5168): tốt nhất ở bước 2500 rồi tăng lại, tức overfit quay lại khi nhiều expert hoạt động hơn.
- **k theo layer vẫn không thắng k đồng đều, kể cả có dropout.** Ở chi phí 2.0, (1,1,2,4) kém k=2 đồng đều 0.011 nat và (4,2,1,1) kém 0.021 nat, cùng dấu ở cả hai seed. (1,1,2,2) (chi phí 1.5) đạt 4.5189, thấp hơn đường thẳng nối fixed-k1+drop (4.5365) và fixed-k2+drop (4.5122) khoảng 0.005 nat, không đáng kể. Riêng (4,2,1,1), từng tốt hơn k=2 đồng đều ở bước cuối không regularize (mục 5.3), giờ là lịch tệ nhất.
- Hầu hết lần chạy dropout đạt tốt nhất ở bước 2750 đến 3000 (cuối quá trình), nên 3000 bước vẫn chưa đủ cho nhóm này và các khoảng cách có thể đổi khi chạy dài hơn (đã kiểm tra ở vòng 3B).

**Vòng 3A: chỉnh dropout cho cả hai bên (2 seed).** `+drop3` là dropout 0.3. Elastic được train với dropout 0.2 và 0.3. Chạy bằng `run_reg3.sh A`. Bảng dưới gộp cả các vòng trước, `bestB` trung bình hai seed:

| Dropout | fixed-k1 | fixed-k2 | elastic@k1 | elastic@k2 |
|---------|----------|----------|------------|------------|
| 0       | 4.6390   | 4.6045   | 4.5980     | 4.5707     |
| 0.1     | 4.5365   | 4.5122   | 4.5199     | 4.4870     |
| 0.2     | 4.5610   | 4.4923   | 4.5070     | **4.4844** |
| 0.3     | chưa chạy | 4.5194  | 4.5561     | 4.5350     |

Chênh lệch `bestB` theo từng seed (số âm = vế trái tốt hơn):

| So sánh                            | seed 0  | seed 1  | trung bình |
|------------------------------------|---------|---------|------------|
| elastic+drop2@k2 − fixed-k2+drop2  | +0.0016 | −0.0174 | −0.0079    |
| elastic+drop2@k1 − fixed-k1+drop2  | −0.0553 | −0.0527 | −0.0540    |
| elastic+drop2@k1 − fixed-k1+drop   | −0.0277 | −0.0312 | −0.0295    |
| elastic+drop3@k2 − fixed-k2+drop3  | +0.0120 | +0.0191 | +0.0156    |
| elastic+drop3@k2 − elastic+drop2@k2 | +0.0464 | +0.0548 | +0.0506   |
| fixed-k2+drop3 − fixed-k2+drop2    | +0.0360 | +0.0183 | +0.0272    |
| fixed-k1+drop2 − fixed-k1+drop     | +0.0276 | +0.0215 | +0.0246    |
| elastic+drop2@k2 − elastic+drop@k2 | +0.0082 | −0.0134 | −0.0026    |

- **Mức dropout tốt nhất của hai bên khác nhau.** fixed-k2 tốt nhất ở 0.2 (0.1 và 0.3 đều kém hơn). elastic@k2 gần như phẳng giữa 0.1 và 0.2 (chênh 0.003, khác dấu giữa hai seed) và kém rõ ở 0.3. fixed-k1 tốt nhất ở 0.1 trong các mức đã thử (0.2 kém hơn 0.025 nat), còn elastic@k1 lại tốt hơn ở 0.2.
- **Ở k=2 với 3000 bước, khi mỗi bên được chỉnh dropout, elastic và cố định ngang nhau (vòng 3B thay đổi kết luận này).** 4.4844 (elastic, dropout 0.2) so với 4.4923 (cố định, dropout 0.2): hơn 0.008 nat nhưng khác dấu giữa hai seed (+0.002 và −0.017), nên chưa phân biệt được. Lợi thế 0.025 nat ở dropout 0.1 (vòng 2) phần lớn là do k cố định chưa được regularize đủ ở mức đó.
- **Ở k=1, elastic vẫn hơn rõ.** Hơn k cố định 0.054 nat ở cùng dropout 0.2, và 0.030 nat khi so mức tốt nhất đã thử của mỗi bên (elastic 0.2, cố định 0.1), cùng dấu ở cả hai seed. Lý do khả dĩ (chưa kiểm chứng): model cố định k=1 ít sức chứa nên không chịu được dropout mạnh, còn k=1 của model elastic dùng chung trọng số với lúc train ở k=2.
- **Dropout 0.3 quá mạnh cho cả hai**, nặng hơn với elastic (kém cố định 0.016 nat ở k=2).
- Một model elastic với dropout 0.2 cho 4.4844 ở k=2 và 4.5070 ở k=1: ngang model k=2 cố định tốt nhất và hơn model k=1 cố định tốt nhất, tức một model phục vụ cả hai ngân sách mà không mất chất lượng. Mọi lần chạy ở vòng này đều tốt nhất ở bước 3000, trừ một lần ở 2750.

**Vòng 3B: nhóm dropout chạy 6000 bước (2 seed).** Cùng thiết lập với `--steps 6000` (tên hàng có hậu tố `-6k`), chạy bằng `run_reg3.sh B`. Vẫn đánh giá mỗi 250 bước, bước tốt nhất chọn trên nửa A; `finalB` là loss ở bước 6000.

| Cấu hình            | bestB (dừng sớm) | finalB (bước 6000) | Bước tốt nhất (seed 0, 1) |
|---------------------|------------------|--------------------|---------------------------|
| elastic+drop2-6k@k2 | **4.4699**       | 4.5031             | 5000, 3750                |
| elastic+drop-6k@k2  | 4.4854           | 4.6540             | 3000, 2750                |
| fixed-k2+drop2-6k   | 4.4870           | 4.5520             | 4500, 3750                |
| elastic+drop2-6k@k1 | 4.4956           | 4.5308             | 5000, 3750                |
| fixed-k2+drop-6k    | 4.5068           | 4.7560             | 2750, 2500                |
| elastic+drop-6k@k1  | 4.5122           | 4.6914             | 3000, 2750                |

Chênh lệch `bestB` theo từng seed (số âm = vế trái tốt hơn):

| So sánh                                                   | seed 0  | seed 1  | trung bình |
|-----------------------------------------------------------|---------|---------|------------|
| elastic+drop-6k@k2 − fixed-k2+drop-6k                     | −0.0225 | −0.0202 | −0.0213    |
| elastic+drop2-6k@k2 − fixed-k2+drop2-6k                   | −0.0201 | −0.0143 | −0.0172    |
| elastic+drop-6k@k2 − fixed-k2+drop2-6k                    | −0.0054 | +0.0021 | −0.0016    |
| elastic+drop2-6k@k2 − elastic+drop2@k2 (6000 so với 3000 bước) | −0.0281 | −0.0009 | −0.0145    |
| fixed-k2+drop2-6k − fixed-k2+drop2 (6000 so với 3000 bước)     | −0.0064 | −0.0040 | −0.0052    |

- **Dropout 0.1 đã hội tụ ở khoảng 3000 bước.** Bước tốt nhất vẫn nằm ở 2500 đến 3000 và bestB gần như không đổi so với bản 3000 bước (fixed 4.5122 → 4.5068, elastic 4.4870 → 4.4854). Sau đó cả hai overfit nặng (finalB 4.65 đến 4.76).
- **Dropout 0.2 được lợi từ train lâu hơn** (bước tốt nhất 3750 đến 5000), elastic nhiều hơn cố định (4.4844 → 4.4699 so với 4.4923 → 4.4870). Mức cải thiện của elastic chủ yếu đến từ seed 0 (0.028 so với 0.001 ở seed 1), nên chưa chắc chắn.
- **Ở 6000 bước, elastic hơn k cố định cùng dropout ở cả hai seed:** khoảng 0.021 nat ở dropout 0.1 và 0.017 nat ở dropout 0.2. Ở 3000 bước chênh lệch ở dropout 0.2 chỉ 0.008 nat và khác dấu giữa hai seed, nên kết luận "ngang nhau ở k=2" của vòng 3A chỉ đúng ở 3000 bước.
- **Elastic xuống dốc chậm hơn sau bước tốt nhất.** `finalB − bestB` ở dropout 0.2 là 0.033 (elastic) so với 0.065 (cố định), ở dropout 0.1 là 0.168 so với 0.249, nhỏ hơn ở cả hai seed. Khi không dừng sớm được, elastic an toàn hơn.
- Chế độ k=1 của model elastic tốt nhất (4.4956) chỉ kém model k=2 cố định train riêng (4.4870) 0.009 nat, trong khi dùng một nửa số expert định tuyến.
- Giới hạn: 2 seed, một lần chạy elastic dropout 0.2 vẫn tốt nhất ở bước 5000 (có thể chưa hội tụ hẳn), vòng 4 thêm seed 2, 3 và k=1 cố định ở 6000 bước.

**Vòng 4: seed 2 và 3, k=1 cố định ở 6000 bước, dropout 0.15.** Chạy bằng `run_reg4.sh`. Các bảng dưới gộp seed 0 đến 3 (4 seed) trừ chỗ có ghi chú; các hàng 3000 bước của vòng trước vốn chỉ có 2 seed giờ có 4.

6000 bước (4 seed):

| Cấu hình            | bestB (dừng sớm) | finalB (bước 6000) | Bước tốt nhất (seed 0, 1, 2, 3) |
|---------------------|------------------|--------------------|---------------------------------|
| elastic+drop2-6k@k2 | **4.4637**       | 4.5043             | 5000, 3750, 3750, 4500          |
| fixed-k2+drop2-6k   | 4.4763           | 4.5409             | 4500, 3750, 3750, 4500          |
| elastic+drop2-6k@k1 | 4.4920           | 4.5323             | 5000, 3750, 4250, 4500          |
| fixed-k1+drop2-6k   | 4.5162           | 4.5221             | 5500, 5250, 5250, 5000          |
| fixed-k1+drop-6k    | 4.5335           | 4.5805             | 4750, 3750, 3750, 4000          |

Chênh lệch `bestB` theo từng seed ở 6000 bước (số âm = vế trái tốt hơn):

| So sánh                                                 | seed 0  | seed 1  | seed 2  | seed 3  | trung bình |
|---------------------------------------------------------|---------|---------|---------|---------|------------|
| elastic+drop2-6k@k2 − fixed-k2+drop2-6k                 | −0.0201 | −0.0143 | −0.0076 | −0.0084 | −0.0126    |
| elastic+drop2-6k@k1 − fixed-k1+drop2-6k                 | −0.0306 | −0.0097 | −0.0387 | −0.0179 | −0.0242    |
| elastic+drop2-6k@k1 − fixed-k1+drop-6k                  | −0.0635 | −0.0363 | −0.0368 | −0.0294 | −0.0415    |
| fixed-k1+drop2-6k − fixed-k1+drop-6k                    | −0.0329 | −0.0266 | +0.0019 | −0.0115 | −0.0173    |
| elastic+drop2-6k@k2 − elastic+drop2@k2 (6000 so với 3000) | −0.0281 | −0.0009 | −0.0240 | −0.0080 | −0.0153  |
| fixed-k2+drop2-6k − fixed-k2+drop2 (6000 so với 3000)   | −0.0064 | −0.0040 | −0.0129 | +0.0099 | −0.0034    |

3000 bước (4 seed), chênh lệch `bestB` theo từng seed:

| So sánh                            | seed 0  | seed 1  | seed 2  | seed 3  | trung bình |
|------------------------------------|---------|---------|---------|---------|------------|
| elastic+drop@k2 − fixed-k2+drop    | −0.0390 | −0.0116 | −0.0258 | −0.0129 | −0.0223    |
| elastic+drop@k1 − fixed-k1+drop    | −0.0266 | −0.0064 | −0.0084 | −0.0313 | −0.0182    |
| elastic+drop2@k2 − fixed-k2+drop2  | +0.0016 | −0.0174 | +0.0035 | +0.0095 | −0.0007    |
| elastic+drop2@k1 − fixed-k1+drop2  | −0.0553 | −0.0527 | −0.0688 | −0.0512 | −0.0570    |
| elastic+drop2@k1 − fixed-k1+drop   | −0.0277 | −0.0312 | −0.0330 | −0.0374 | −0.0323    |
| fixed-k2+drop2 − fixed-k2+drop     | −0.0324 | −0.0076 | −0.0442 | −0.0313 | −0.0289    |
| fixed-k1+drop2 − fixed-k1+drop     | +0.0276 | +0.0215 | +0.0358 | +0.0138 | +0.0247    |
| elastic+drop2@k2 − elastic+drop@k2 | +0.0082 | −0.0134 | −0.0149 | −0.0089 | −0.0073    |

Quét dropout ở 3000 bước (chỉ seed 0 và 1, vì dropout 0.15 mới chạy cho hai seed này; `bestB`):

| Dropout | fixed-k1  | fixed-k2 | elastic@k1 | elastic@k2 |
|---------|-----------|----------|------------|------------|
| 0       | 4.6390    | 4.6045   | 4.5980     | 4.5707     |
| 0.1     | 4.5365    | 4.5122   | 4.5199     | 4.4870     |
| 0.15    | 4.5416    | 4.4889   | 4.5003     | **4.4753** |
| 0.2     | 4.5610    | 4.4923   | 4.5070     | 4.4844     |
| 0.3     | chưa chạy | 4.5194   | 4.5561     | 4.5350     |

Ở dropout 0.15 (seed 0 và 1): elastic@k2 − fixed-k2 là −0.0198 và −0.0075 (trung bình −0.0136); elastic@k1 − fixed-k1 là −0.0371 và −0.0456 (trung bình −0.0413).

- **Ở k=1 elastic thắng ở mọi nơi, ở mọi seed.** Ở 3000 bước nó hơn k=1 cố định cùng dropout 0.018 nat (dropout 0.1), 0.041 (0.15, 2 seed) và 0.057 (0.2). Ở 6000 bước nó hơn k=1 cố định 0.024 nat ở mức dropout tốt nhất của bên cố định (0.2) và 0.042 nat ở dropout 0.1. Dropout tốt nhất của k=1 cố định đổi theo độ dài train: 0.1 ở 3000 bước (0.2 kém hơn 0.025, cả 4 seed), 0.2 ở 6000 bước (tốt hơn 0.017, 3 trên 4 seed).
- **Ở k=2, lợi ích phụ thuộc độ dài train và mức dropout.** Ở 3000 bước là 0.022 nat với dropout 0.1 (4 trên 4 seed), 0.014 với 0.15 (2 trên 2 seed) và gần như bằng 0 với 0.2 (trung bình −0.0007, dấu lẫn lộn). Ở 6000 bước với dropout 0.2 là 0.013 nat ở cả 4 seed (từ 0.008 đến 0.020). Lý do là elastic còn cải thiện khi train lâu hơn (0.015 nat từ 3000 lên 6000 bước, cả 4 seed) trong khi k=2 cố định gần như không (0.003, dấu lẫn lộn).
- **Mức dropout tốt nhất** (seed 0 và 1, 3000 bước): k=2 cố định gần như ngang nhau giữa 0.15 và 0.2 (4.4889 so với 4.4923), elastic@k2 thích 0.15 (4.4753 so với 4.4844, tốt hơn 0.009), k=1 cố định thích 0.1.
- **Elastic xuống dốc chậm hơn sau bước tốt nhất.** Ở 6000 bước `finalB − bestB` là 0.041 với elastic@k2 so với 0.065 với k=2 cố định. K=1 cố định có dropout 0.2 hầu như không xuống dốc (0.006) vì nó còn đang cải thiện ở bước 5000 đến 5500.
- Chế độ k=1 của model elastic tốt nhất (4.4920) kém model k=2 cố định train riêng (4.4763) 0.016 nat, trong khi dùng một nửa số expert định tuyến; ước lượng 0.009 từ 2 seed ở trên là hơi lạc quan.

Rút ra thực tế (4 seed): khi dữ liệu ít, dropout là đòn bẩy lớn hơn nhiều (khoảng 0.08 đến 0.09 nat). Trên nền dropout đã chỉnh, huấn luyện co giãn cho lợi ích đáng tin ở k=1 (khoảng 0.024 nat so với k=1 cố định tốt nhất ở 6000 bước, và 0.057 nat ở cùng dropout 0.2 ở 3000 bước; mọi seed) và nhỏ hơn ở k=2 (khoảng 0.013 nat ở 6000 bước, cả 4 seed, nhưng ngang nhau ở 3000 bước với dropout 0.2). Nó cũng còn cải thiện khi train lâu hơn và xuống dốc chậm hơn sau bước tốt nhất. Cấu hình tốt nhất đã đo là một model elastic với dropout 0.2 train 6000 bước (4.4637 ở k=2, 4.4920 ở k=1), và giá trị chính của nó là một model phục vụ nhiều ngân sách (mục 1). Chưa làm: elastic với k=4, dropout 0.05 cho k=1, train k=1 cố định lâu hơn 6000 bước (nó còn đang cải thiện), thêm seed.

### 5. Phân bổ k thông minh có thắng một k cố định không?

Cùng thiết lập BPE như mục 4. Chi phí là số expert định tuyến trung bình mỗi token mỗi layer (không tính expert dùng chung và attention). Vì các thí nghiệm này dùng model train kiểu elastic, chúng đo cách một model elastic dùng expert của nó. Mục 4 cho thấy khi dữ liệu ít, k=2 train cố định không tốt hơn k=1 ở bước cuối, nhưng với dừng sớm hoặc dropout thì k=2 hơn k=1.

**5.1 Ước lượng ở mức token (`headroom.py`).** Train một model elastic, đo loss từng token ở k=1 và ở k cao hơn, rồi chọn token nào nhận k cao hơn dưới một ngân sách (ngẫu nhiên, entropy của đầu ra k=1, MLP nhỏ trên hidden state sớm hoặc muộn, hoặc luật theo tần suất token). Đây là ước lượng ở mức token, bỏ qua ảnh hưởng qua attention.

- k=1 so với k=4 (gain 0.103 nat): ở cùng chi phí trung bình, một k cố định thắng mọi cách phân bổ giữa k=1 và k=4. Ở chi phí 2, k=2 cố định lấy lại 83% gain so với 33% (ngẫu nhiên), 38% (entropy) và 44% đến 56% (bộ dự đoán đã học). Ở chi phí 3, k=3 cố định lấy lại 97% so với 66% đến 84%.
- k=1 so với k=2 (gain 0.063 nat): phần trăm gain của k=2 lấy lại được khi một tỉ lệ f token nhận k=2:

| f    | ngẫu nhiên | entropy | hiếm | phổ biến | learn_emb | learn_l1 | learn_fin |
|------|------------|---------|------|----------|-----------|----------|-----------|
| 0.1  | 9%         | 10%     | 19%  | 4%       | 17%       | 16%      | 25%       |
| 0.25 | 26%        | 26%     | 36%  | 15%      | 35%       | 38%      | 45%       |
| 0.5  | 50%        | 54%     | 64%  | 36%      | 64%       | 65%      | 72%       |
| 0.75 | 76%        | 83%     | 85%  | 64%      | 88%       | 86%      | 88%       |

Luật `rare` (nâng loại token hiếm nhất, chỉ cần tra đếm tần suất trên tập train) ngang bộ dự đoán học trên embedding đầu vào (`learn_emb`) mà không tốn gì. Luật entropy phải chạy xong cả lượt k=1 trước nên tốn 1 + 2f thay vì 1 + f.

**5.2 Luật theo token chạy thật trong các layer (`tokenk.py`).** k theo từng token được truyền vào mọi layer MoE (`set_k_cap(model, tensor)`). Model train với k ngẫu nhiên theo token trong {1,2}; `rare - random` ở cùng tỉ lệ f token được nâng lên k=2 (số âm = rare tốt hơn, đơn vị nat). Tổng gain k=1 lên k=2 trong các model này là 0.034 đến 0.042 nat.

| seed                          | f=0.25  | f=0.5   | f=0.75  |
|-------------------------------|---------|---------|---------|
| 0                             | -0.0061 | -0.0069 | -0.0044 |
| 1                             | -0.0075 | -0.0051 | -0.0056 |
| 2                             | -0.0043 | -0.0062 | -0.0065 |
| 0, train bằng chính luật rare | -0.0122 | -0.0101 | -0.0079 |

Luật ngược lại (nâng các token phổ biến nhất) luôn tệ hơn ngẫu nhiên (khoảng +0.005 đến +0.006), nên hiệu ứng không phải nhiễu. Hiệu ứng nhỏ (khoảng 0.006 nat) và chỉ lấp khoảng chi phí giữa k=1 và k=2, nơi top-k cố định không với tới được vì k là số nguyên.

**5.3 k theo layer: không được xác nhận khi train từ đầu.** Một model elastic train với k bốc độc lập cho từng layer từ {1,2,4}, quét cả 81 lịch (`layerk.py`, chọn trên nửa val A, báo trên nửa B), cho thấy các layer đầu có vẻ không cần k lớn. Ví dụ lịch (1,1,2,4) hơn k=2 đồng đều 0.010 đến 0.011 nat ở cùng chi phí trên cả hai seed. Khi train từng lịch từ đầu (`schedule_train.py`, seed 0) thì kết quả này **không lặp lại**:

| k theo layer | chi phí | val loss | train (giây) | so với k=2 đồng đều |
|--------------|---------|----------|--------------|---------------------|
| (1,1,1,1)    | 1.0     | 4.6575   | 283          | -0.0483             |
| (2,2,2,2)    | 2.0     | 4.7058   | 289          | 0                   |
| (4,4,4,4)    | 4.0     | 4.8201   | 327          | +0.1143             |
| (1,1,2,2)    | 1.5     | 4.6634   | 286          | -0.0425             |
| (1,1,2,4)    | 2.0     | 4.7214   | 294          | +0.0155             |
| (1,1,4,4)    | 2.5     | 4.7509   | 296          | +0.0451             |
| (4,2,1,1)    | 2.0     | 4.6873   | 306          | -0.0186             |

Lịch trông tốt nhất trên model elastic lại tệ hơn k=2 đồng đều, còn đối chứng đảo ngược (4,2,1,1) lại tốt hơn. Cách một model elastic dùng expert ở từng layer không phải căn cứ để chọn lịch cần train. Trong thí nghiệm đầu tiên ở trên, k cố định theo layer cũng kém k đồng đều, nên không khuyến nghị lịch k theo layer. Lưu ý: bảng này đo ở bước cuối trong chế độ ít dữ liệu không regularize, nơi k cao bị overfit. Khi chạy lại có dropout và dừng sớm (mục 4.2, vòng 2), (1,1,2,4) kém k=2 đồng đều 0.011 nat và (4,2,1,1) kém 0.021 nat, nên lợi thế từng thấy của (4,2,1,1) ở bảng trên chỉ là hiệu ứng overfit.

**Kết luận của mục này.** Một k cố định vẫn là baseline cần vượt. Các hiệu ứng tìm được nhỏ (khoảng 0.01 nat hoặc ít hơn) và chỉ nằm trong khoảng chi phí giữa k=1 và k=2. Kết quả vững là huấn luyện co giãn cho một model chạy được ở nhiều ngân sách mà không mất chất lượng (mục 1); lợi thế của nó khi dữ liệu ít nhỏ hơn dropout nhiều, nhưng cộng dồn được với dropout, và vẫn giữ lợi ích khoảng 0.013 nat ở k=2 (6000 bước; ngang nhau ở 3000 bước) và khoảng 0.024 nat ở k=1 (6000 bước) khi dropout được chỉnh cho từng bên, trên 4 seed (mục 4.2).

### 6. Tốc độ đo thực tế (CUDA, batch 32x128; tỷ lệ so với khi dùng tất cả expert)

| Số expert | k          | suy luận (inference)  | huấn luyện (train)    |
|-----------|------------|-----------------------|-----------------------|
| 8         | 1 / 2 / 4  | 0.42x / 0.48x / 0.66x | 0.43x / 0.50x / 0.66x |
| 32        | 1 / 8 / 16 | 0.43x / 0.48x / 0.62x | 0.18x / 0.36x / 0.56x |

Dùng ít expert hơn thì nhanh hơn trong thực tế, nhưng không tỷ lệ thuận lúc suy luận (32 expert, k=1 đạt 0.43x chứ không phải 1/32). Chi phí cố định mỗi bước (attention, embedding, dispatch) chiếm tỷ trọng lớn khi k nhỏ. Trong các lần chạy BPE 4 layer ở trên, thời gian train thực tế gần như không phụ thuộc k (283 giây ở k=1, 289 giây ở k=2, 327 giây ở k=4), nên không nên mặc định là tiết kiệm được compute ở cỡ model đó.

### 7. Một tokenizer chung cho nhiều ngôn ngữ (`langcheck.py`)

Câu hỏi: một tokenizer nhỏ dùng chung cộng với router MoE có phục vụ được nhiều ngôn ngữ không? Thiết lập: Wikipedia tiếng Việt và tiếng Anh (`wikimedia/wikipedia`, 20231101), chỉ lấy bài dài ít nhất 1000 ký tự (bài ngắn phần lớn là bài rập khuôn, làm loss tiếng Việt vô nghĩa ở lần chạy đầu), khoảng 8 MB mỗi ngôn ngữ, chia theo bài (10% làm val). Một ByteLevel BPE (thư viện `tokenizers`, nên không có token ngoài từ điển) vocab 16000 train trên cả hai ngôn ngữ. Model như mục 4 (4 layer, d=128, 32 expert, chuỗi 128, dropout 0.2, 3000 bước), train trên cùng số token cho mỗi ngôn ngữ (1.63M mỗi bên). Hai biến thể, 1 seed: k cố định bằng 2, và k thích ứng (`top_p=0.7`, k tối đa 8, không phạt compute). Bước dừng sớm chọn trên nửa A của val từng ngôn ngữ, báo trên nửa B. Chạy bằng `python langcheck.py --langs vi,en`.

Tokenize trên văn bản val: số token trên 100 ký tự (số token trên mỗi từ trong ngoặc).

| Ngôn ngữ   | BPE chỉ Việt (16k) | BPE chung (16k) | Tokenizer Qwen2.5 (151k) |
|------------|--------------------|-----------------|--------------------------|
| Tiếng Việt | 28.7 (1.41)        | 29.5 (1.45)     | 32.6 (1.60)              |
| Tiếng Anh  | 34.6 (2.18)        | 26.2 (1.65)     | 24.5 (1.54)              |

Dùng chung làm tiếng Việt tốn thêm 3% token, còn tiếng Anh cần ít hơn 24% so với BPE chỉ Việt và chỉ kém tokenizer 151k của Qwen 7%; tiếng Việt cần ít hơn Qwen khoảng 10%. BPE byte-level chỉ train trên tiếng Việt tốn khoảng 2.2 token mỗi từ tiếng Anh: tệ hơn nhưng xa mức hàng chục token mỗi từ của tokenizer rơi về từng ký tự.

Kết quả model (1 seed). Loss mỗi token và bit mỗi ký tự chỉ nên so trong cùng một ngôn ngữ, không so giữa các ngôn ngữ.

| Ngôn ngữ   | k=2 cố định: loss/token | bit/ký tự | Thích ứng: loss/token | bit/ký tự | avg_k thích ứng theo layer (trung bình) |
|------------|-------------------------|-----------|-----------------------|-----------|-----------------------------------------|
| Tiếng Việt | 4.3992                  | 1.8711    | 4.3587                | 1.8539    | 6.64 4.78 4.62 3.44 (4.87)              |
| Tiếng Anh  | 5.2137                  | 1.9756    | 5.1907                | 1.9669    | 6.15 4.42 3.62 2.81 (4.25)              |

Mức dùng expert của router với k=2 cố định: độ phân kỳ Jensen-Shannon giữa phân phối tiếng Việt và tiếng Anh theo từng layer (tối đa là ln 2 = 0.693), kèm mức nhiễu (cùng một ngôn ngữ, hai nửa val).

| Layer | Việt so với Anh | nhiễu (Việt A−B) | nhiễu (Anh A−B) |
|-------|-----------------|------------------|-----------------|
| 0     | 0.0083          | 0.0014           | 0.0003          |
| 1     | 0.0855          | 0.0011           | 0.0006          |
| 2     | 0.2709          | 0.0066           | 0.0016          |
| 3     | 0.3639          | 0.0065           | 0.0008          |

- **Router tự chuyên môn hoá theo ngôn ngữ mà không ai dạy.** Ở layer 1 đến 3, độ phân kỳ lớn gấp ít nhất 40 lần mức nhiễu, và ở layer cuối đạt khoảng một nửa giá trị tối đa; ở layer 0 là khoảng 6 lần mức nhiễu. Ở layer cuối một expert nhận 34% token tiếng Việt (expert 14) và một expert khác nhận 42% token tiếng Anh (expert 7). Lưu ý chưa kiểm chứng: một phần có thể chỉ vì hai ngôn ngữ dùng các token khác nhau (router nhìn embedding của token), chứ chưa hẳn là chuyên môn hoá ở mức ngôn ngữ.
- **K thích ứng gần như không khác nhau giữa các ngôn ngữ, và không ổn định.** Với hai ngôn ngữ tiếng Việt nhận 4.87 expert so với 4.25 của tiếng Anh (nhiều hơn 13%); thêm tiếng Trung thì chênh lệch co lại còn 4.45 so với 4.33 (3%), nên không có hiệu ứng ngôn ngữ đáng tin lên k. Giá trị trung bình cao vì `top_p=0.7` với router còn phẳng chọn nhiều expert, nên không so được với ngân sách k=2 của các thí nghiệm elastic.
- K thích ứng đạt loss thấp hơn k=2 cố định (0.04 nat ở tiếng Việt, 0.02 ở tiếng Anh) nhưng dùng nhiều expert hơn 2.1 đến 2.4 lần, nên không nói lên điều gì về hiệu quả.
Ba ngôn ngữ (Việt, Anh, Trung; cùng thiết lập, 1.71M token mỗi ngôn ngữ, 1 seed):

| Ngôn ngữ   | BPE chỉ Việt | BPE chung (3 ngôn ngữ) | Tokenizer Qwen2.5 |
|------------|--------------|------------------------|-------------------|
| Tiếng Việt | 28.7 (1.41)  | 30.5 (1.50)            | 32.6 (1.60)       |
| Tiếng Anh  | 34.6 (2.18)  | 28.3 (1.78)            | 24.5 (1.54)       |
| Tiếng Trung | 164.3       | 76.2                   | 72.7              |

(Số token trên 100 ký tự; tiếng Trung không có số token mỗi từ.) So với tokenizer hai ngôn ngữ, thêm tiếng Trung làm tiếng Việt tốn thêm 3% và tiếng Anh thêm 8% token, còn tiếng Trung chỉ kém tokenizer của Qwen 5%. Model cũng trả giá: bit mỗi ký tự tăng từ 1.8711 lên 1.9667 ở tiếng Việt và từ 1.9756 lên 2.0657 ở tiếng Anh (khoảng 5%), đó là cái giá của việc chia sẻ sức chứa với ngôn ngữ thứ ba. Tiếng Trung đạt 6.43 bit mỗi ký tự, không so được với hai ngôn ngữ kia.

Độ phân kỳ Jensen-Shannon của mức dùng expert với k=2 cố định (ba cột cuối là mức nhiễu):

| Layer | việt–anh | việt–trung | anh–trung | nhiễu việt | nhiễu anh | nhiễu trung |
|-------|----------|------------|-----------|------------|-----------|-------------|
| 0     | 0.0070   | 0.0111     | 0.0143    | 0.0014     | 0.0003    | 0.0003      |
| 1     | 0.0998   | 0.1852     | 0.2256    | 0.0021     | 0.0005    | 0.0014      |
| 2     | 0.2119   | 0.4710     | 0.4762    | 0.0057     | 0.0006    | 0.0018      |
| 3     | 0.2566   | 0.4860     | 0.4322    | 0.0042     | 0.0012    | 0.0016      |

Router tách được cả ba ngôn ngữ (gấp ít nhất 35 lần mức nhiễu ở layer 2 và 3). Các cặp có tiếng Trung (0.43 đến 0.49) cách xa nhau gấp đôi cặp Việt–Anh (0.21 đến 0.26), nên các ngôn ngữ khác chữ viết tách nhau mạnh hơn, khớp với lưu ý về việc router chỉ nhìn theo token ở trên. Ở layer cuối, expert nhiều nhất nhận 35% token tiếng Việt (expert 7), 42% tiếng Anh (expert 23) và 39% tiếng Trung (expert 27). Với k thích ứng, số expert trung bình là 4.45 cho tiếng Việt, 4.33 cho tiếng Anh và 4.75 cho tiếng Trung, và loss thấp hơn k=2 cố định 0.026 đến 0.031 nat trong khi dùng nhiều expert hơn 2.2 đến 2.4 lần.

- Chưa làm: thêm seed, và đối chứng cho việc router chỉ phân biệt theo token.

## Đo tốc độ

`python bench.py` đo tốc độ thật (token mỗi giây, suy luận và train) khi mọi token bị ép dùng đúng k expert, để kiểm tra dùng ít expert hơn có nhanh hơn thật trên phần cứng của bạn không. Ví dụ: `python bench.py --experts 32 --d_ff 128 --ks 1,4,8,16,32`.

## Chạy test và demo

```
pip install torch pytest
pytest -q                  # hoặc: python test_adaptive_moe.py
python train_demo.py
```

## License

MIT © 2026 BachDaThan, xem [LICENSE](LICENSE).

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
