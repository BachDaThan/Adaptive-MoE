# APP_KB — Kho tri thức trong app (lõi C++17 `kb_core`)

> Đặc tả cho task **P3-1 + P9-12** (một đặc tả cho Python, C++ và lớp nối FFI). Quyết định nền: **Q6 = C++17 lõi chung** (bạn chọn 2026-11-16).
> Mã: `native/kb_core.h` (cổng C), `native/kb_core.cpp`. Bộ test: `tests/test_kb_core.py` + `tests/golden/`. Bản tham chiếu hành vi: `yui_rag.py` (tách từ + BM25), `yui_graph.py` (Graphify).
> Trạng thái: **`[~]` đã viết và chạy test ở máy mình (Linux x86-64, g++ 13); CHƯA build trên Android/Windows/Flutter, CHƯA nối vào app (P9-14).**

## 1. Phạm vi

Lõi chỉ làm phần **trí nhớ**: nhận chữ đã đọc sẵn từ máy chủ (Dart trong Y-App, `yuid` trong YuiVieOS), dựng chỉ mục trong RAM, và trả về **một chuỗi ngắn đúng ngân sách ký tự** để đưa cho model (công cụ `recall`). Lõi **không** đọc/ghi file, **không** mã hoá, **không** mở mạng.

| Việc | Ai làm |
|---|---|
| Đọc/ghi file `chat/ notes/ kb/`, quét thư mục, mtime, xoá file | máy chủ (Dart/`yuid`) |
| Giải mã `.enc` (P3-6, nếu làm) | máy chủ giải mã trong RAM rồi đưa chữ vào lõi |
| Tách đoạn, BM25, đồ thị, Graphify, ngân sách ký tự, từ chối khi không chắc | **lõi** |
| Gỡ thẻ `<tool>` trong nội dung tìm được (`sanitize`) | máy chủ/agent, **trước khi đưa cho model** (nội dung kho là dữ liệu không tin cậy) |

## 2. Bố cục kho (giữ nguyên `yui_vault.py`, đổi máy chỉ cần chép thư mục)

```
<gốc>/chat/    lịch sử chat, mỗi dòng .jsonl là một lượt:  {"ts":<giây>,"messages":[{"role":"user","content":"…"},{"role":"assistant","content":"…"}]}
<gốc>/notes/   ghi chú .md
<gốc>/kb/_clean/  <gốc>/kb/_tomtat/   tài liệu đã lọc/tóm tắt (file gốc trong kb/ KHÔNG nạp)
<gốc>/index/   (v1 không dùng: chỉ mục dựng lại trong RAM mỗi lần mở; xoá cũng không mất gì)
```
- **Mã cuộc hội thoại** (`conv`) của chat: `"<đường dẫn tương đối>#<k>"`, `k` = số thứ tự (từ 0) của dòng **không rỗng** trong file. Ví dụ `chat/yui-20261117.jsonl#3`.
- **Mã nguồn** (`src`) = đường dẫn tương đối dùng dấu `/`.
- Thứ tự nạp khi mở app (để khớp Python): thư mục `chat/`, rồi `notes/`, rồi `kb/`; trong mỗi thư mục sắp theo đường dẫn; **một lần `kb_commit` sau cả mẻ**.
- Tiêu đề ghi chú/tài liệu lấy từ tên file bỏ đuôi `.md/.txt/.enc` (không phân biệt hoa thường); `kb/_tomtat/…` thêm tiền tố `tóm tắt `.

## 3. Vòng đời (máy chủ gọi)

1. **Mở app/kho:** `kb_new("Yui")` → `kb_set_tz(múi giờ giây)` → với mỗi file: `kb_add_chat_turn` (từng lượt) hoặc `kb_add_doc` → `kb_commit`.
2. **Có lượt chat mới:** máy chủ ghi dòng `.jsonl` rồi gọi `kb_remember(src, conv, user, assistant, ts)` (cắt 4000 ký tự mỗi bên, gộp khoảng trắng, dựng đồ thị ngay).
3. **Sửa ghi chú / đổi file:** `kb_drop_source(src)` rồi `kb_add_doc(...)` lại, rồi `kb_commit`.
4. **Xoá chat/ghi chú:** `kb_drop_source(src)` (đoạn biến khỏi kết quả ngay; sự kiện của Graphify thành "chết" và bị lọc).
5. **Công cụ `recall`:** `kb_recall(query, chars, k, mode, out, cap)`; app đặt `chars ≤ kToolResultChars` (hiện 120). `mode = 2` (Graphify, không đủ chắc thì lùi BM25 ngắn) là mặc định đề xuất.
6. **Xoá toàn bộ kho:** máy chủ xoá file rồi `kb_clear`.

## 4. Cổng C (ABI v1, `kb_core.h`)

Chuỗi UTF-8; hàm trả chuỗi ghi vào bộ đệm của người gọi và **trả số byte cần** (như `snprintf`); `>= cap` nghĩa là bị cắt, gọi lại với bộ đệm lớn hơn. Mỗi `kb_t` có khoá riêng (gọi từ nhiều luồng được, lệnh chạy lần lượt). Chỉ xuất tên `kb_*` (build với `-fvisibility=hidden`) nên **không đụng** `libyui`/engine (P9-15).

| Hàm | Việc |
|---|---|
| `kb_abi`, `kb_new(bot)`, `kb_free`, `kb_set_tz`, `kb_clear` | vòng đời; `bot` là tên hiện trước lời trợ lý |
| `kb_add_chat_turn(src, conv, role, text, ts)` | một lượt chat (`role`: user/assistant/note); tách đoạn ≤ 500 ký tự; trả số đoạn |
| `kb_add_doc(kind, src, text, ts)` | `kind` = `"note"` (notes/) hoặc `"kb"` (kb/_clean, kb/_tomtat); tách đoạn theo đoạn văn rồi ≤ 500 ký tự |
| `kb_remember(src, conv, user, assistant, ts)` | thêm cặp hỏi–đáp + dựng đồ thị ngay; 1/0 |
| `kb_drop_source(src)` | xoá mọi đoạn của một file; trả số đoạn |
| `kb_commit` | dựng đồ thị đồng xuất hiện cho mẻ vừa nạp |
| `kb_recall(query, chars, k, mode, out, cap)` | `mode` 0 = BM25, 1 = Graphify, 2 = Graphify + lùi BM25 |
| `kb_stats` | số đoạn/file/từ/sự kiện/thực thể/sự kiện bị thay |
| `kb_dbg_*` | JSON phục vụ test vàng (không dùng trong app) |

Chuỗi "không có gì": `không tìm thấy trong lịch sử` (giống Python, để model đã học vẫn nhận ra).
Định dạng dòng: `[dd/mm] bạn: …` / `[dd/mm] Yui: …` / `ghi chú` / `tài liệu`; sự kiện bị thay thế (khi hỏi "trước đây") có thêm ` (cũ)`.

## 5. Thuật toán (tóm tắt; chi tiết là mã Python)

- **Tách từ** (`tokens`): chữ/số `[0-9A-Za-zÀ-ỹ]+`, chữ thường; mỗi từ phát ra dạng **không dấu**, thêm dạng **có dấu** nếu khác, thêm **cặp từ** `a_b`. Từ dừng không dấu chỉ bỏ khi người dùng thật sự gõ không dấu ("chó" ≠ "cho"); tên riêng giữa câu ("bạn Minh") không bị coi là đại từ "mình".
- **Đoạn:** chat/ghi chú tách theo câu, tối đa 500 ký tự. Ghi chú thêm tiêu đề vào đoạn đầu; tài liệu `kb` thêm vào mọi đoạn.
- **BM25** (k1 = 1,5, b = 0,75) trên từ + cặp từ (cặp trọng số 1,6); mở rộng truy vấn bằng 3 láng giềng mạnh nhất của đồ thị đồng xuất hiện (trọng số 0,35, cần ≥ 2 lần); ưu tiên nhẹ nội dung mới (tối đa +15%); kéo thêm câu liền kề hỏi↔đáp của kết quả tốt nhất.
- **Graphify:** ứng viên (BM25 + sự kiện chứa thực thể + lan truyền 2 bước trên đồ thị thực thể + "cầu nối" tên riêng) → chấm điểm theo **sự kiện** (từ khoá, thực thể hiếm, đồ thị, BM25, loại câu hỏi when/howmany/who/where/list/history, độ mới, vai, nơi ở, tự xưng) → **thay giá trị cũ bằng mới** cho thuộc tính bền (tên, tuổi, nơi ở, SĐT, email, địa chỉ, nghề, thích nhất) → MMR + dừng khi đủ ý → kể lại bằng câu thường theo thời gian, cắt quanh phần liên quan, vừa ngân sách → **từ chối** nếu độ phủ < 0,34 hoặc từ khoá phân biệt nhất chưa từng được kể.

## 6. Hiệu năng (đo ở máy mình, **không phải điện thoại**)

x86-64, `-O2`: 20.116 đoạn: nạp 0,16 s, `kb_commit` 0,01 s, truy vấn Graphify đầu tiên (trích sự kiện) 0,34 s, sau đó ≈ 1,4 ms/truy vấn (BM25 ≈ 0,4 ms). Điện thoại chậm hơn nhiều lần; chưa đo. Bộ nhớ: lõi lưu văn bản + chỉ mục trong RAM; chưa đo trên 100.000+ đoạn.

## 7. Cách build / test

```
make kb          # libkb_core.so  (g++/clang++ C++17, -ffp-contract=off, -fvisibility=hidden)
make test-kb     # pytest: so lõi C++ với tests/golden/kb_golden.json
make kb-tables   # sinh lại native/kb_unicode.inc + native/kb_words.inc từ Python (khi đổi yui_rag.py/yui_graph.py)
make kb-golden   # sinh lại bộ vàng (TZ=UTC PYTHONHASHSEED=0), cần Python ≥ 3.12 để số thực khớp từng bit
```
- Nguồn có chữ tiếng Việt: **MSVC cần `/utf-8`**; NDK/clang không cần. Android/Windows chưa thử.
- `-ffp-contract=off` là bắt buộc để số thực khớp Python (trên ARM compiler hay tự gộp nhân-cộng).
- Chạy dưới ASAN+UBSAN: không có lỗi (13/13 test).

### Bộ vàng phủ gì, và **không** phủ gì
Phủ: `fold`, `tokens` (có/không cặp từ), `split_text` (500 và 40), `entities`, `fact_key`, `intent_of`, `has_self`, `split_facts` trên 426 chuỗi (cố định + ngẫu nhiên có hạt giống, gồm tiếng Việt NFC/NFD, Hàn/Trung/Hy Lạp/Nga, số, ngày, đơn vị); 3 kịch bản kho (nạp, sửa file, nạp thêm, xoá file) với chunk, sự kiện, BM25 (k = 4/8, lọc vai), `recall`, Graphify (nhiều ngân sách/k, có/không từ chối, có/không lùi), gồm bộ dữ liệu của `graphify_bench.py` (1.116 và 416 đoạn) và ~620 câu hỏi sinh ngẫu nhiên.
**Thử đột biến** (cố ý sửa mã C++ để xem test có bắt không): bắt được trọng số mở rộng đồ thị 0,35, phạt sự kiện cũ 0,6, thưởng nơi ở 0,4; **lọt** 3 ngưỡng sát biên — loại trùng MMR 0,8, từ chối độ phủ 0,34, cầu nối 0,2. Nghĩa là test **chưa chứng minh** ba ngưỡng này đúng (cần thêm ca sát biên, ghi ở PLAN).

## 8. Bảo mật / quyền riêng tư

- Dữ liệu ở lại máy; lõi không có mạng. Kho **không mã hoá** ở bản đầu (P3-6 tuỳ chọn). Văn bản nằm trong RAM của tiến trình.
- Nội dung `recall` có thể chứa chữ giống lệnh (`<tool>…`): **bắt buộc** `sanitize` ở máy chủ trước khi đưa cho model.
- `kb_dbg_*` chỉ để test; bản phát hành có thể không xuất.

## 9. Khác Python có chủ ý, và lỗi nhỏ của Python được **giữ nguyên** để khớp

**Khác (lõi làm ít hơn hoặc khác):**
1. Không đọc nhiều định dạng chat (`parse_chat`: ChatGPT export, `.txt` "Bạn: …"): máy chủ đưa từng lượt vào. Bản app chỉ cần định dạng `.jsonl` ở §2.
2. Không lưu chỉ mục xuống đĩa (v1 dựng lại trong RAM). Không port `related()`, bảng `form` và liên kết `[[ghi chú]]` (không dùng bởi `recall`).
3. `\d` trong số/ngày chỉ nhận chữ số ASCII (Python nhận mọi chữ số Unicode). `str.lower()` không xử lý "ς" cuối từ tiếng Hy Lạp. `std::regex` chạy trên "khung ASCII" của chuỗi đã bỏ dấu (chữ cái không phải ASCII coi là ký tự chữ).
4. Số quá lớn (hàng trăm chữ số): Python ném `OverflowError` trong `entities`; lõi bỏ riêng thực thể số đó.
5. Giờ `[dd/mm]` dùng độ lệch múi giờ cố định đặt bằng `kb_set_tz` (không xét giờ mùa hè theo từng mốc thời gian).
6. **Đã sửa `yui_graph.py` (cần up):** chỗ phụ thuộc thứ tự `set` (hạt băm ngẫu nhiên) nay xếp ổn định (hoà thì xếp theo chữ cái; cộng số thực theo thứ tự cố định). Kết quả khác bản cũ chỉ khi hoà điểm; 8 test của `test_yui_graph.py` vẫn đạt; dòng `graphify` của `graphify_bench.py` giữ nguyên (chỉ dòng baseline `dense` trước đây dao động theo hạt băm).

**Giữ nguyên lỗi/điểm lạ của Python (lõi bắt chước từng bước):**
1. Danh sách đơn vị trong `_NUM` rẽ nhánh theo thứ tự, nên `k` nuốt `kg`/`km` ("5kg" → thực thể `n:5000`) và `tr` khớp đầu chữ "trong" ("5 trong" → `n:5000000`).
2. Ngày "14/3" cũng sinh thêm thực thể số `n:14`.
3. `stage1` trong Graphify viết `(gen) or cand` nên **không bao giờ** lùi về `cand` (rỗng thì giữ rỗng).
Hai việc trên (1, 2) là ứng viên sửa ở **cả hai bên** rồi sinh lại bộ vàng: xem **P3-8** trong kế hoạch.
