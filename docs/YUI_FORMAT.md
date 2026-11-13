# Đặc tả định dạng `.yui` (v1, v2, t2, v3)

> Trạng thái: **v1/v2/t2 = đang dùng** (mô tả đúng theo `export_yui.py::write_yui` và `yui.cpp::Model::load`).
> **v3 = ĐÃ CHỐT (6 quyết định ở mục 6.4), đã có bộ ghi (`export_yui.py --fmt 3`) và bộ đọc/kiểm (`yui_format.py`) bằng Python.**
> Engine C++ (`yui.cpp`) **chưa** đọc v3 (task P1-3); app chưa dùng (P1-4).

Quy ước: little-endian; `u8/u16/u32/u64` = số nguyên không dấu; `i32` có dấu; `f32` IEEE-754; `f16` IEEE half. "Chuỗi" ở tokenizer = `u16 độ dài` + byte UTF-8.

---

## 1. Bố cục chung (v1, v2)

```
MAGIC "YUI1"                      4 byte
HEADER                            12×u32 + 2×f32
TOKENIZER                         vocab, merges, eos, bảng khoảng Unicode
TRỌNG SỐ                          theo thứ tự cố định (mục 4)
```
Không có mục lục: muốn tới một tensor phải đi tuần tự (hoặc tự cộng kích thước, như `yui_format.py` làm).

### 1.1 HEADER

| Trường | Kiểu | Ý nghĩa |
|---|---|---|
| `ver` | u32 | 1, 2 hoặc 3 (v3: mục 6; header v3 giữ nguyên bố cục) |
| `quant` | u32 | 0 f32 · 1 int8 · 2 q4 (cả `q4` và `q4all`) · 3 t2 |
| `V` | u32 | cỡ từ vựng |
| `D` | u32 | chiều ẩn |
| `layers` | u32 | số lớp |
| `heads` | u32 | số head attention |
| `E` | u32 | số expert định tuyến mỗi lớp |
| `F` | u32 | chiều FFN của expert (exporter luôn ghi `F = 2·D`) |
| `n_shared` | u32 | số expert chung (exporter ghi 1) |
| `L` | u32 | độ dài ngữ cảnh (cửa sổ KV) |
| `min_k`, `max_k` | u32 | số expert tối thiểu / tối đa mỗi token |
| `top_p` | f32 | ngưỡng xác suất tích luỹ chọn expert |
| `min_strength` | f32 | ngưỡng độ mạnh của router |

Engine từ chối: `ver ∉ {1,2}`, `quant > 3`, `E > 256`, `D % heads ≠ 0` hoặc `D % (2·heads) ≠ 0` (head_dim phải chẵn), file cụt (`Reader::need`). `yui_format.py` kiểm tương tự và thêm trần an toàn (layers ≤ 4096, D, V ≤ 2²⁰).

### 1.2 TOKENIZER (BPE mức byte)

```
u32 nvocab (= V)        rồi nvocab × chuỗi (token, dạng ký tự byte-level kiểu GPT-2)
u32 nmerges             rồi nmerges × (chuỗi a, chuỗi b)
i32 eos                 id token <eos>, -1 nếu không có
u32 nranges             rồi nranges × (u32 từ, u32 đến, u32 lớp)   lớp 1 = chữ (\p{L}), 2 = số (\p{N})
```
Bảng khoảng Unicode để C++ tách từ giống regex GPT-2; nó phụ thuộc phiên bản `unicodedata` của Python lúc xuất, nên **không** đưa vào hash tokenizer.

**`tokenizer_sha256`** (dùng cho META và manifest): sha256 của các byte từ đầu `nvocab` đến hết `eos` (vocab + merges + eos, đúng như nằm trong file). Hai model cùng hash ⇒ cùng cách tách token.

---

## 2. Kiểu ma trận

Ma trận `rows × cols`. Từ **v2**, mỗi ma trận có `u8 kind` đứng trước; **v1** không có byte này (kind suy từ `quant`: 0 → mọi ma trận f32, 1 → mọi ma trận int8).

| kind | Tên | Bố cục sau byte kind | Số byte (không tính byte kind) |
|---|---|---|---|
| 0 | f32 | `rows×cols` f32 | `4·rows·cols` |
| 1 | int8 theo hàng | `rows` hệ số f32, rồi `rows×cols` int8 | `4·rows + rows·cols` |
| 2 | q4 (khối 32) | tất cả hệ số f16 `[rows, nb]`, rồi tất cả byte 4-bit `[rows, nb, 16]`; `nb = ⌈cols/32⌉` (hàng ngắn được đệm 0) | `2·rows·nb + 16·rows·nb` |
| 3 | t2 (ternary) | `u16 g` (cỡ nhóm, bội 32, `cols % g = 0`), hệ số f16 `[rows, cols/g]`, rồi `[rows, cols/32, 8]` byte | `2·rows·cols/g + 8·rows·cols/32` |

Chi tiết giải mã:
- **int8:** giá trị = `q · scale[hàng]`.
- **q4:** byte `j` (0–15) của khối chứa trọng số `j` ở 4 bit thấp và `j+16` ở 4 bit cao; giá trị = `(nibble − 8) · scale_f16`. Lượng tử kiểu Q4_0 (`scale = giá_trị_tuyệt_đối_lớn_nhất_có_dấu / −8`).
- **t2:** byte `j` của khối 32 chứa trọng số `j, j+8, j+16, j+24` ở các cặp bit `0-1, 2-3, 4-5, 6-7`; mã `0 → 0`, `1 → +1`, `2 → −1`; giá trị = `mã · scale_f16[nhóm]` (xem `bitnet_yui.pack_t2` / `bitnet_pack.pack_t2`). Cột không chia hết cho 32 ⇒ exporter rơi về int8 cho ma trận đó (nhờ byte kind riêng từng ma trận).

**Vector f32** (LayerNorm, router, strength_head) không có byte kind, ghi thẳng f32. **Router và LayerNorm luôn f32** để việc chọn expert giống hệt PyTorch.

---

## 3. Chọn kiểu theo `quant` của exporter (v2)

| `--quant` | `quant` ở header | emb / head | qkv, o | expert + shared |
|---|---|---|---|---|
| `f32` | 0 (ver 1) | f32 | f32 | f32 |
| `int8` | 1 (ver 1) | int8 | int8 | int8 |
| `q4` | 2 (ver 2) | int8 | int8 | q4 |
| `q4all` | 2 (ver 2) | q4 | q4 | q4 |
| `t2` | 3 (ver 2) | int8 | t2 | t2 |

Vì `q4` và `q4all` cùng `quant = 2`, muốn biết chính xác phải xem kind của từng tensor (bộ đọc trả `kind` trong bảng tensor). META v3 ghi tên đầy đủ (mục 6).

---

## 4. Thứ tự trọng số

```
emb.weight                    V × D                       (ma trận)
với mỗi lớp l = 0 … layers-1:
  blocks.l.n1.weight, n1.bias            D, D             (vector f32)
  blocks.l.attn.qkv.weight               3D × D           (ma trận)
  blocks.l.attn.o.weight                 D × D            (ma trận)
  blocks.l.n2.weight, n2.bias            D, D             (vector f32)
  blocks.l.moe.router.gate.weight        E × D            (vector f32, KHÔNG có byte kind)
  blocks.l.moe.router.strength_head.weight   1 × D        (f32)
  blocks.l.moe.router.strength_head.bias     1            (f32)
  với e = 0 … E-1:   blocks.l.moe.experts.e.{w_gate F×D, w_up F×D, w_down D×F}   (ma trận)
  với s = 0 … n_shared-1:  blocks.l.moe.shared.s.{w_gate, w_up, w_down}          (ma trận)
norm.weight, norm.bias                   D, D             (f32)
head.weight                   V × D                       (ma trận)
```
Hết `head.weight` là hết file (v1/v2): `yui_format.py` báo lỗi nếu còn byte thừa hoặc thiếu.

**Hệ quả:** mỗi lớp có router riêng và E expert, token nào cũng chọn expert ở MỌI lớp ⇒ tải lẻ expert qua mạng theo token là không khả thi (xem PLAN, mục 1). Mục lục v3 phục vụ **đọc nhanh, kiểm tra, và cấp model**, không phải tải theo token.

---

## 5. Bảng tương thích

| Người đọc \ File | v1 | v2 (q4/t2) | v3 |
|---|---|---|---|
| `yui.cpp` hiện tại (1.3) | ✅ | ✅ | ❌ báo lỗi gọn về phiên bản không hỗ trợ (đã có sẵn) |
| `yui.cpp` sau P1-3 | ✅ y hệt | ✅ y hệt | ✅ |
| `yui_format.py` (sau P1-2) | ✅ | ✅ | ✅ (kiểm cả META, mục lục, chân file) |
| App Y-App hiện tại | ✅ | ✅ | nhập được (`ModelStore.isYui` chỉ kiểm 4 byte `YUI1`) nhưng khi nạp engine cũ báo lỗi |

Nguyên tắc: **không đổi hành vi với v1/v2**; `export_yui.py` mặc định vẫn xuất v1/v2 (đã kiểm: byte-for-byte giống bản cũ ở f32/int8/q4/q4all/t2), chỉ ghi v3 khi có `--fmt 3`.

---

## 6. v3: META + mục lục (đã chốt)

Mục tiêu: file tự mô tả (tên, miền, ngôn ngữ, giấy phép…) để Kho model hiển thị mà không phải tải hết, và có mục lục để kiểm tra/đọc nhanh; vẫn **giữ magic `YUI1`**, `ver = 3`, **phần trọng số y hệt v2** (mỗi ma trận có byte kind; engine tái dùng `load_mat`). v3 luôn dùng bố cục có kind, kể cả khi `quant` là f32/int8 (khác v1).

### 6.1 Bố cục

```
MAGIC "YUI1"
HEADER             (như v2, ver = 3)
META               u32 meta_len (1…65536), rồi meta_len byte JSON UTF-8   ← mới, NGAY SAU header
TOKENIZER          (như v2)
TRỌNG SỐ           (như v2: mỗi ma trận có byte kind)
INDEX              mục lục nhị phân                                       ← mới
FOOTER             16 byte                                                ← mới
```
META đứng ngay sau header (offset 60) nên app/Kho đọc thông tin hiển thị chỉ cần vài trăm byte đầu file (`yui_format.read_meta`), không phải bỏ qua tokenizer.

**FOOTER (16 byte, cuối file):** `u64 index_offset` · `u32 index_bytes` · magic `"YUIX"`.
Ràng buộc: `index_offset` = chỗ kết thúc tensor cuối; `index_offset + index_bytes + 16 = kích thước file`.

**INDEX:** `u32 count`, rồi `count` mục:

| Trường | Kiểu | Ghi chú |
|---|---|---|
| `name_len` + `name` | u16 + UTF-8 | đúng tên ở mục 4, `1 ≤ name_len ≤ 128` |
| `offset` | u64 | tính từ đầu file, trỏ tới **byte kind** (hoặc dữ liệu f32 nếu là vector) |
| `size` | u64 | tổng byte của bản ghi (gồm byte kind / `u16 g`) |
| `kind` | u8 | 0–3 như mục 2; vector f32 ghi 0 |
| `rows`, `cols` | u32, u32 | ma trận: `rows × cols`. Vector: 2 chiều (gate router `E × D`, `strength_head.weight` `1 × D`) ghi đúng `rows × cols`; vector 1 chiều (LayerNorm, bias) ghi `rows = 1`, `cols = độ dài` |
| `group` | u16 | cỡ nhóm cho t2, ngược lại 0 |

`count` phải bằng `4 + layers·(9 + 3·(E + n_shared))` (tính từ header, không tin con số trong file). Mục lục phải **khớp đúng từng mục** với bố cục tính được từ header (thứ tự, offset liên tiếp, size, kind, rows, cols, group); lệch ⇒ file bị coi là hỏng (`yui_format.read_yui` kiểm điều này).

### 6.2 META (JSON object, UTF-8, ≤ 65536 byte)

| Khoá | Bắt buộc | Kiểu / quy tắc |
|---|---|---|
| `meta_version` | ✅ | số nguyên, hiện là `1` (bản đọc từ chối giá trị khác) |
| `name` | ✅ | chuỗi 1–128 ký tự |
| `arch_id` | ✅ | **tự suy từ header**: `amoe-d{D}-l{layers}-h{heads}-e{E}-s{n_shared}-f{F}` (vd model hiện tại: `amoe-d256-l6-h4-e16-s1-f512`); phải khớp header |
| `quant` | ✅ | `"f32"` · `"int8"` · `"q4"` · `"q4all"` · `"t2"`; phải khớp trường `quant` ở header (`q4`/`q4all` cùng ứng với 2) |
| `params` | ✅ | tổng số tham số (số nguyên > 0); phải bằng tổng tính từ bố cục |
| `tokenizer_sha256` | ✅ | hex thường 64 ký tự, thuật toán ở mục 1.2; phải khớp tokenizer thật trong file |
| `created` | ✅ | `YYYY-MM-DD` (UTC), ngày hợp lệ |
| `license` | ✅ khoá, giá trị `"unknown"` được phép | chuỗi 1–128 ký tự: mã SPDX hoặc mô tả ngắn |
| `lang` | ✅ | mảng ≤ 16 mã ngôn ngữ (`vi`, `en`, `zh-Hans`…); rỗng được |
| `domains` | ✅ | mảng ≤ 16 slug (`a-z`, `0-9`, `-`). **Khi xuất** phải thuộc danh sách chuẩn: `general, chat, code, math, vi-fiction, translate, tool, other`; khi đọc chỉ kiểm dạng slug để bản sau thêm domain mới mà bản cũ vẫn đọc được. Rỗng = tổng quát |
| `description` | tuỳ chọn | ≤ 300 ký tự, văn bản thuần |
| `base` | tuỳ chọn | tên model gốc (≤ 128) |
| `source_url` | tuỳ chọn | ≤ 512 ký tự, chỉ để hiển thị |
| `exporter` | tuỳ chọn | ≤ 128 ký tự |

`centroid` (cho bộ phân loại miền) **chưa có trong v3**; sẽ thêm như khoá tuỳ chọn ở P5 (bản đọc cũ bỏ qua khoá lạ nên không phải đổi định dạng).

Quy tắc an toàn/tương thích:
- **META là dữ liệu, không phải lệnh.** App chỉ hiển thị; không bao giờ chạy, tải, hay mở URL tự động từ META. Văn bản từ file ngoài là không tin cậy: bộ đọc thay ký tự điều khiển bằng dấu cách và từ chối chuỗi quá dài.
- Khoá lạ được **giữ nguyên và bỏ qua** (không lỗi); `meta_version` tăng khi đổi nghĩa trường cũ.
- Trần cứng: `meta_len ≤ 65536`, tên tensor ≤ 128 byte, số mục lục tính từ header.
- Hash toàn file **không** nằm trong META (tự tham chiếu được); sha256 từng file nằm ở `manifest.json` (P2-2) và app so sánh sau khi tải.

### 6.3 Cách xuất và đọc
```
python export_yui.py --out thu_muc_model --ckpt auto --dst model_q4.yui --quant q4 --fmt 3 \
       --name yui-q4 --domains chat,general --lang vi,en --license MIT [--desc "…"] [--base …] [--source-url …]
python yui_format.py model_q4.yui          # in tóm tắt + META, báo lỗi nếu file hỏng
```
Không có `--fmt 3` thì xuất v1/v2 như cũ. Trong Python: `yui_format.read_meta(path)` (nhanh, chỉ đầu file) và `yui_format.read_yui(path)` (kiểm toàn bộ).

### 6.4 Quyết định đã chốt (2026-11-13)
1. `license` là khoá bắt buộc, cho phép `"unknown"`.
2. Công thức `arch_id` như ở 6.2.
3. `domains` theo danh sách chuẩn kèm `other`.
4. `centroid` để P5.
5. Mục lục nhị phân.
6. META đặt ngay sau header.

### 6.5 Việc còn lại
- **P1-3:** `yui.cpp` đọc v3 (đọc `meta_len`, kiểm trần, nạp trọng số y như v2, trả META trong `yui_info`, tuỳ chọn kiểm mục lục); đồng bộ sang `Y-App/native/yui.cpp` (CI `sync-yui` sẽ kiểm).
- **P1-4:** app Y-App hiển thị META (qua `yui_info`).
