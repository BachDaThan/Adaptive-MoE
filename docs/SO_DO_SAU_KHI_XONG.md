# SƠ ĐỒ SAU KHI XONG KẾ HOẠCH (P0 → P7)

> Đây là **đích**, chưa phải hiện trạng. Nhãn `[P#]` cho biết giai đoạn tạo ra từng phần trong `PLAN.md`.
> Phần P6 (ghép Expert) vẽ nét đứt vì **chưa cam kết**: chỉ làm nếu thí nghiệm đạt tiêu chí Go.
> Sơ đồ hiện tại: `SO_DO_HIEN_TAI.md`. Sau khi xong từng giai đoạn, cập nhật file này cho đúng thực tế (task P7-3).

## 1. Bức tranh tổng sau khi xong

```mermaid
flowchart TB
  subgraph AM["Repo Adaptive-MoE"]
    direction TB
    TR["Huấn luyện + dữ liệu (như cũ)"]
    EX["export_yui.py<br/>--fmt 3 ghi META + mục lục [P1]"]
    YF["yui_format.py<br/>bộ đọc .yui thuần Python [P0-5]"]
    MM["make_manifest.py<br/>quét model + dataset → manifest.json [P2, P4]"]
    BS["bench_swarm.py<br/>đo lõi / lõi+RAG / lõi+model chuyên [P5-5]"]
    CK["tools/check_yui_sync.py<br/>CI kiểm yui.cpp 2 repo bằng nhau [P0-1]"]
    PYR["Runtime Python 'Yui' (như cũ)<br/>agent, orch, vault, RAG, Graphify, web"]
    GD["tests/golden/*<br/>dữ liệu vàng cho test chéo Python↔Dart [P3-7]"]
    TR --> EX --> MM
    YF --> EX
    YF --> MM
    PYR --> GD
  end

  subgraph HF["Hugging Face (nguồn mặc định của bạn + nguồn người dùng thêm)"]
    MF["manifest.json<br/>models[] (url, sha256, arch_id, domains, quant)<br/>datasets[] (danh sách cho phép)"]
    YM["File .yui v3 nhỏ theo miền<br/>(văn, code, toán...)"]
    DS["Datasets công khai<br/>(datasets-server /search)"]
  end
  MM -->|đẩy| MF
  EX -->|đẩy| YM

  subgraph YA["Repo Y-App"]
    direction TB
    UI["Giao diện: Chat · Lịch sử · Kho model [P2-6] · Model · Cài đặt (Hybrid) [P2-7]"]
    ST["state.dart<br/>ChatStore, ModelStore v3, Settings"]
    HUB["hub_service.dart [P2-3..5]<br/>manifest cache 24h, nhiều nguồn ưu tiên,<br/>tải tiếp tục + sha256 + LRU đĩa"]
    RT["kb/router.dart [P5]<br/>miền câu hỏi + độ chắc chắn + ngưỡng theo mạng"]
    KBA["lib/kb/* [P3]<br/>BM25 tiếng Việt, Graphify-lite,<br/>recall / note, kho trên máy"]
    DSV["dataset_service.dart [P4]<br/>từ khoá trung tính + xem trước,<br/>lọc rác, nén, ghi nguồn, lùi dần 429"]
    AGD["agent.dart<br/>calc, time, search + recall, note"]
    EN["engine.dart (FFI) + yui_probe [P5-1]"]
    NA["native/yui.cpp<br/>đọc v1/v2/v3, bounds-check [P1-3]"]
    UI --> ST
    ST --> HUB
    ST --> RT
    ST --> KBA
    ST --> DSV
    ST --> AGD
    ST --> EN --> NA
    RT --> HUB
    RT --> EN
  end

  HUB -->|"manifest + tải file (Range, /resolve/)"| MF
  HUB -->|"tải model chuyên"| YM
  DSV -->|"từ khoá đã được người dùng duyệt"| DS

  subgraph EXP["P6 — nghiên cứu, CHƯA cam kết"]
    direction TB
    EA["Thí nghiệm A: ghép cùng gốc (BTX-lite)"]
    EB["Thí nghiệm B: căn chỉnh 2 model độc lập"]
    GO{"Go / No-Go<br/>giảm perplexity ≥ 5%?"}
    AR["Expert Arena + tải lẻ theo mục lục v3<br/>bridge.bin + anchor_hash"]
    EA --> GO
    EB --> GO
    GO -. Go .-> AR
  end
  AM -.-> EXP
  AR -.-> NA

  AM <-->|"yui.cpp luôn giống nhau (CI kiểm)"| NA
  style EXP stroke-dasharray: 5 5
  style AR stroke-dasharray: 5 5
```

## 2. Định dạng `.yui` v3

```mermaid
flowchart LR
  A["MAGIC 'YUI1'<br/>ver = 3"] --> B["Cấu hình + tokenizer<br/>(như v2)"]
  B --> M["KHỐI META (JSON, có độ dài)<br/>arch_id, name, domains, lang,<br/>tokenizer_sha256, quant, params,<br/>license, centroid (tuỳ chọn)"]
  M --> W["Trọng số<br/>thứ tự y hệt v2<br/>(engine tái dùng load_mat)"]
  W --> I["MỤC LỤC ở cuối file<br/>tên tensor → offset, size, kiểu"]
  I --> Z["Chân 16 byte<br/>offset mục lục + 'YUIX'"]
```

- Engine cũ gặp `ver = 3` báo lỗi gọn thay vì chạy sai. Engine mới vẫn đọc v1/v2/t2 như cũ.
- App chỉ cần tải vài KB đầu file (hoặc `manifest.json`) để biết model giỏi gì, không phải tải cả file.
- Mục lục cho phép đọc từng phần khi cần (nền cho P6 nếu Go); ở cấp model thì chưa bắt buộc.

## 3. Luồng một câu hỏi sau khi xong

```mermaid
flowchart TD
  Q["Người dùng hỏi"] --> L1["Tầng 1: model lõi trên máy<br/>+ tra kho trên máy (recall/Graphify-lite)"]
  L1 --> C1{"Đủ chắc chắn?<br/>(yui_probe + luật)"}
  C1 -- Có --> A["Trả lời ngay<br/>(offline hoàn toàn)"]
  C1 -- Không --> H{"Hybrid bật<br/>và có mạng?"}
  H -- Không --> A
  H -- Có --> K["Trích từ khoá trung tính<br/>+ hiện 'sẽ gửi: …' cho người dùng"]
  K --> OK{"Người dùng đồng ý?"}
  OK -- Không --> A
  OK -- Có --> D["Tầng 2: tra Dataset (allowlist)<br/>lọc rác → nén ≤ ngân sách ký tự<br/>→ nạp vào prompt + ghi nguồn"]
  D --> C2{"Vẫn cần kỹ năng riêng?<br/>(miền code/toán… + ngưỡng theo mạng)"}
  C2 -- Không --> A
  C2 -- Có --> M["Tầng 3: hỏi người dùng → tải model chuyên nhỏ<br/>(LRU cache, sha256, arch_id)"]
  M --> R["Chạy như 'khách trọ': nhận ngữ cảnh nén,<br/>trả kết quả, yui_free giải phóng RAM"]
  R --> A
```

Rơi về tầng thấp hơn bất cứ khi nào mất mạng, người dùng từ chối, hoặc lỗi.

## 4. Cây thư mục mục tiêu (phần mới)

```text
Adaptive-MoE/
  docs/
    PLAN.md                 # kế hoạch + đánh dấu tiến độ
    SO_DO_HIEN_TAI.md
    SO_DO_SAU_KHI_XONG.md
    YUI_FORMAT.md           # đặc tả v1/v2/t2/v3           [P1-1]
    APP_KB.md               # đặc tả kho trong app         [P3-1]
  tools/check_yui_sync.py                                  [P0-1]
  yui_format.py, test_yui_format.py                        [P0-5]
  make_manifest.py, test_make_manifest.py                  [P2-2]
  bench_swarm.py                                           [P5-5]
  tests/golden/                                            [P3-7]
  exp_merge_same_base.py, exp_align_independent.py         [P6, chưa cam kết]

Y-App/
  tools/check_yui_sync.py                                  [P0-1]
  lib/services/hub_service.dart, dataset_service.dart      [P2, P4]
  lib/screens/hub_screen.dart                              [P2-6]
  lib/kb/ bm25.dart vi_text.dart graphify.dart
          keywords.dart clean.dart router.dart cipher.dart [P3, P4, P5]
  test/kb_golden_test.dart                                 [P3-7]
```

## 5. So sánh trước / sau (chỉ những gì kế hoạch này thật sự làm được)

| Tiêu chí | Hiện tại | Sau khi xong |
|---|---|---|
| Tìm và đổi model | Tự tìm link, nhập tay | Tab Kho: xem miền/kích cỡ, bấm tải, nhiều nguồn |
| Tự mô tả của `.yui` | Không | META (miền, tokenizer, giấy phép) + mục lục |
| An toàn khi nhận file lạ | Kiểm 4 byte đầu | `arch_id`, sha256, trần dung lượng, kiểm biên |
| Trí nhớ trong app | Chỉ lịch sử chat | Kho trên máy: BM25 + Graphify-lite + `recall`/`note` |
| Kiến thức ngoài | Chỉ Wikipedia (`search`) | Dataset allowlist, từ khoá được duyệt, có ghi nguồn |
| Gặp câu vượt tầm | Chịu thua | Gợi ý/mượn model chuyên (đo A/B rồi mới giữ) |
| Khi mất mạng | Chạy bình thường | Vẫn chạy bình thường (Hybrid chỉ là tăng cường) |
| `yui.cpp` | 2 bản chép tay | 2 bản được CI canh bằng nhau |

**Không hứa:** "RAM 3 GB giải bài toán 10–30B" hay ghép Expert từ model train độc lập. Hai điều đó chỉ được xem xét sau P6 và chỉ khi có số đo.
