# KẾ HOẠCH NÂNG CẤP Y-App + Adaptive-MoE

> File này là **bộ nhớ chung** giữa các cuộc chat. Mở chat mới chỉ cần gửi file này + nói "làm tiếp".
> Đặt tại: `Adaptive-MoE/docs/PLAN.md` (một bản duy nhất, cả 2 repo dùng chung).

**Cập nhật:** 2026-11-13 · **Phiên bản kế hoạch:** v1 · **Tiến độ:** 13/45 (P0, P1 xong; P2: P2-1, P2-2, P2-3 xong; P2-4 đã giao `[~]`, chờ bạn build)

## ▶ TIẾP THEO: bạn up 4 file của P2-4 (mục 📤 ở task P2-4), chạy GitHub Actions rồi báo kết quả build. Build xanh thì mình đánh `[x]` và làm `P2-5` (cache LRU đĩa; mình tự lấy `hub_service.dart` và `state.dart` từ repo). Build đỏ thì dán log để mình sửa

Ký hiệu: `[ ]` chưa làm · `[~]` mình đã giao file, chờ bạn up/thử · `[x]` xong (chỉ đánh khi bạn báo OK) · `[!]` bị chặn (xem ghi chú) · `[-]` bỏ (có lý do trong Nhật ký)

| Giai đoạn | Nội dung | Xong |
|---|---|---|
| P0 | Dọn nền, đo mốc ban đầu | 5/5 |
| P1 | `.yui` v3: có meta + mục lục | 5/5 |
| P2 | Kho model (Model Hub) trong Y-App | 3/8 |
| P3 | Kho tri thức ngay trong app (port RAG/Graphify) | 0/7 |
| P4 | Tra Dataset đám mây (Cloud-Edge RAG) | 0/6 |
| P5 | Mượn model chuyên theo câu hỏi (cấp model) | 0/5 |
| P6 | Nghiên cứu ghép Expert (chưa cam kết) | 0/5 |
| P7 | Phát hành + cập nhật tài liệu | 0/4 |

---

## 0. CÁCH LÀM VIỆC (đọc trước khi chat mới)

**Mở chat mới:** gửi `PLAN.md` → nói "làm tiếp". Mình sẽ đọc mục `▶ TIẾP THEO`, xem dòng `📥 Gửi mình` của task đó rồi chỉ xin đúng những file ấy. **Không cần gửi cả zip.**

**Ba cách gửi file (chọn cách nào tiện):**
1. Dán link raw GitHub (nếu repo public), vd `https://raw.githubusercontent.com/BachDaThan/Adaptive-MoE/main/export_yui.py`
2. Đính kèm đúng file được xin.
3. Dán nội dung file nếu ngắn.
Không gửi `README.md` / `README.vi.md` (rất nặng ~85 KB); nếu cần mình sẽ xin đúng mục.

**Mình tự lấy file được:** cả 2 repo đang public và môi trường của mình `git clone` được, nên thường mình tự đọc code, bạn chỉ cần up file mình giao. Việc cần dữ liệu/thiết bị của bạn thì mình vẫn xin riêng.

**Mình giao lại:** chỉ gồm file cần up, kèm đường dẫn đích ghi rõ dạng `Adaptive-MoE/…` hoặc `Y-App/…`, và luôn kèm `docs/PLAN.md` đã đánh dấu mới.

**Giới hạn thật của mình (để khỏi bất ngờ):**
- Python và C++ mình chạy thử được trong môi trường của mình (pytest, g++), nên phần đó có kết quả test thật.
- **Dart/Flutter mình không biên dịch được** (không có Flutter SDK). Code Dart giao ra ở trạng thái `[~]` cho tới khi bạn chạy GitHub Actions; nếu lỗi, dán log để mình sửa. (README Y-App cũng ghi Dart "chưa build thử".)
- Mình không thấy dữ liệu/mạng của bạn: việc cần mạng thật (đo tốc độ HF, thử điện thoại) bạn làm giúp và dán số liệu.

**Quy ước code:** comment tiếng Việt như repo hiện tại; không đổi hành vi cho file `.yui` v1/v2 cũ; mọi tính năng dùng mạng mới là **opt-in, mặc định TẮT**; mỗi task phải có test hoặc tiêu chí đo được (cột ✅).

---

## 1. ĐÁNH GIÁ TƯ VẤN CỦA GEMINI (giữ / sửa / bỏ)

Mình đối chiếu từng ý với code thật. Kết luận chung: hướng đi (kho model mở + tra dataset + giữ riêng tư) đúng và đáng làm; phần "ghép Expert từ model train độc lập" là **giả thuyết nghiên cứu**, không nên dựng cả hệ thống lên đó.

| # | Ý của Gemini | Đánh giá theo code thật | Quyết định |
|---|---|---|---|
| 1 | Tải lẻ Expert qua HTTP Range theo token | `.yui` hiện tuần tự, **mỗi lớp có router riêng + E expert**, token nào cũng chọn expert ở MỌI lớp; không có mục lục byte. Tải qua mạng theo token là bất khả thi | **Bỏ** cấp token. Làm **cấp model** (P2, P5) |
| 2 | Header 64 byte chứa arch_id, anchor_hash, centroid, bảng offset | 64 byte không đủ. Nhưng ý "meta nhỏ tải trước" đúng | **Sửa**: khối META JSON có độ dài + mục lục ở cuối file, giữ magic `YUI1` (P1) |
| 3 | `bridge.bin` Procrustes/CKA/HARC để ghép model train độc lập | Chưa có bằng chứng cho model ~50M; tên bài báo 2026 và số liệu (giảm 46–74% độ trễ…) mình **chưa xác minh được** | **Hoãn** sang P6 có tiêu chí Go/No-Go |
| 4 | "RAM 3GB giải bài 10–30B" | Overclaim: gộp expert nhỏ ≠ năng lực model lớn | **Bỏ** câu này khỏi mọi tài liệu |
| 5 | `manifest.json` tĩnh + tải qua `/resolve/` CDN | Hợp lý, đơn giản, hợp app hiện tại (đã tải HF qua link) | **Giữ** (P2) |
| 6 | Hạn mức HF 500 API / 3000 resolver mỗi 5 phút | Con số chưa kiểm chứng với tài liệu HF hiện hành | **Kiểm chứng** trước khi dựa vào (P2-1) |
| 7 | Một request nhiều khoảng byte (multi-range) | HF resolve chuyển hướng sang CDN; hỗ trợ multipart chưa chắc | **Kiểm chứng**; không được thì chạy nhiều request song song (P2-1) |
| 8 | Cache LRU đĩa 300–500 MB | Hợp lý, rẻ | **Giữ** (P2-5) |
| 9 | Hysteresis / sticky routing | Chỉ có nghĩa với expert cấp token | **Đổi** thành ngưỡng đổi model (P5-4) |
| 10 | MMR chọn expert khác nhau | Repo đã có MMR trong `yui_graph.py` | **Đổi** thành chọn model/đoạn dữ liệu đa dạng |
| 11 | Cloud-Edge RAG: datasets-server + trích từ khoá | Ý tốt, nhưng **app KHÔNG có sẵn** kho/BM25/Graphify (chỉ có bản Python); ngữ cảnh model rất ngắn (`kToolResultChars = 120`) | **Giữ** nhưng phải làm P3 trước, nén cứng theo ngân sách ký tự |
| 12 | "Tự quét mở toàn Hugging Face" | Rủi ro rác, độc hại, bản quyền | **Mặc định danh sách cho phép** (bạn + người dùng tự thêm); quét mở chỉ là tuỳ chọn nâng cao |
| 13 | Che riêng tư bằng tên file trung tính + tải mồi (decoy) | HF vẫn thấy repo/URL; decoy tốn băng thông; che giấu khó hơn minh bạch | **Bỏ decoy**; minh bạch + opt-in + xem trước truy vấn |
| 14 | Không nhận script/agent.json từ nguồn ngoài; kiểm tra biên | Đúng | **Giữ**, thêm sha256 + trần dung lượng |
| 15 | Chế độ Hybrid mặc định tắt, cập nhật chính sách | Bắt buộc: `privacy_*.md` đang cam kết chỉ nối mạng ở các mục liệt kê | **Giữ**, làm trước khi phát hành (P2-8) |

**Gemini bỏ sót:**
- `yui.cpp` (115 111 byte) đang có **2 bản y hệt** ở 2 repo → dễ lệch (P0-1).
- App chưa có kho tri thức, nhưng `privacy_*.md` đã nhắc tới "kho tri thức" → cần làm thật hoặc sửa chữ (P3, P2-8).
- File `provider-chain.patch` (84 KB) nằm ở gốc Adaptive-MoE, có vẻ là rác tồn (P0-2).

---

## 2. NGUYÊN TẮC THIẾT KẾ

1. **Offline-first:** không mạng thì app chạy y như bây giờ. Mọi thứ mới chỉ là "tăng cường".
2. **Cấp model trước, cấp expert sau (nếu có bằng chứng):** mượn **cả model nhỏ chuyên ngành** (vài chục MB, q4/t2) thay vì từng expert.
3. **Model là "khách trọ" (stateless):** bàn giao bằng văn bản nén (Graphify), không chuyển KV cache.
4. **Ngân sách ngữ cảnh cứng:** model nhỏ có L = 128–256 token; mọi thứ nhét thêm vào prompt phải nén xuống vài chục–vài trăm ký tự.
5. **Riêng tư:** không gửi nguyên văn chat ra mạng; chỉ từ khoá trung tính, hiện cho người dùng xem trước.
6. **Tương thích ngược:** engine mới đọc v1/v2/v3; engine cũ gặp v3 phải báo lỗi gọn (đã có sẵn `die("phiên bản .yui không hỗ trợ")`).
7. **Mỗi bước nhỏ, kiểm được, hoàn tác được.**

---

## 3. KẾ HOẠCH CHI TIẾT

Mỗi task: **✎** file sẽ sửa/tạo · **📥** file mình cần bạn gửi · **📤** nơi bạn up · **✅** xong khi.

### P0 — Dọn nền, đo mốc ban đầu

- [x] **P0-1 · Đồng bộ `yui.cpp` giữa 2 repo**
  ✎ tạo `tools/check_yui_sync.py` (so sha256 hai bản; vị trí: `Adaptive-MoE/yui.cpp` và `Y-App/native/yui.cpp`) + job `sync-yui` trong CI cả 2 repo · 📥 (đã lấy xong) · 📤 `Adaptive-MoE/tools/check_yui_sync.py`, `Adaptive-MoE/.github/workflows/tests.yml`, `Y-App/tools/check_yui_sync.py`, `Y-App/.github/workflows/build-app.yml`
  ✅ CI đỏ nếu 2 bản `yui.cpp` khác nhau; xanh khi bằng nhau. *Xong: bạn báo cả 2 repo xanh (2026-11-13).*

- [x] **P0-2 · Dọn `provider-chain.patch`**
  ✎ chuyển vào `docs/archive/` hoặc xoá; thêm vào `.gitignore` mẫu `*.patch` · 📥 không cần (bạn quyết: giữ hay xoá) · 📤 `Adaptive-MoE/.gitignore`
  ✅ Gốc repo không còn file patch 84 KB (nếu bạn đồng ý xoá).

- [x] **P0-3 · Chạy baseline test + build, ghi kết quả**
  ✎ ghi vào mục Nhật ký bên dưới · 📥 bạn dán kết quả `pytest -q`, `make test`, và kết quả Actions của Y-App · 📤 `Adaptive-MoE/docs/PLAN.md`
  ✅ Có số: bao nhiêu test xanh/đỏ, bản Y-App build được hay không.

- [x] **P0-4 · Xác minh Y-App thực sự build và chạy**
  ✎ sửa dòng "Dart CHƯA build thử" trong `Y-App/README.md` cho đúng thực tế · 📥 đoạn cuối README Y-App + kết quả thử trên điện thoại (bạn báo) · 📤 `Y-App/README.md`
  ✅ README phản ánh đúng; mọi lỗi build còn lại có trong Nhật ký.

- [x] **P0-5 · `yui_format.py`: bộ đọc header `.yui` thuần Python**
  ✎ tạo `yui_format.py` (đọc magic, ver, cấu hình, tokenizer-hash, vị trí các ma trận) + `test_yui_format.py` · 📥 `export_yui.py` (đã có trong zip bạn gửi nhưng chat mới cần lại) · 📤 `Adaptive-MoE/yui_format.py`, `Adaptive-MoE/test_yui_format.py`
  ✅ Test xuất file nhỏ v1/v2/t2 rồi đọc lại đúng cấu hình; dùng lại cho P1, P2.

### P1 — `.yui` v3: có META + mục lục (nền cho kho model)

Thiết kế (chốt ở P1-1): giữ magic `YUI1`, `ver = 3`. Sau cấu hình/tokenizer thêm **khối META** (độ dài + JSON UTF-8: `arch_id`, `name`, `domains[]`, `lang[]`, `tokenizer_sha256`, `quant`, `params`, `created`, `license`, tuỳ chọn `centroid` 64 số). Cuối file thêm **mục lục** (tên tensor → offset, size, kiểu) + 16 byte chân (offset mục lục + magic `YUIX`). Phần trọng số giữ nguyên thứ tự v2 nên engine tái dùng `load_mat`.

- [x] **P1-1 · Viết đặc tả `docs/YUI_FORMAT.md` (v1, v2, t2, v3)**
  ✎ tạo `docs/YUI_FORMAT.md` · 📥 không (mình đã đọc `export_yui.py`/`yui.cpp`) · 📤 `Adaptive-MoE/docs/YUI_FORMAT.md`
  ✅ Bạn duyệt đặc tả (đặc biệt các trường META) trước khi code.

- [x] **P1-2 · `export_yui.py --fmt 3` ghi META + mục lục**
  ✎ `export_yui.py`, `yui_format.py`, `test_yui_format.py` · 📥 `export_yui.py`, `yui_format.py`, `model_card.py` · 📤 `Adaptive-MoE/export_yui.py`, `yui_format.py`, `test_yui_format.py`, `docs/YUI_FORMAT.md`
  ✅ Xuất v3 từ model thử; đọc lại bằng `yui_format.py` khớp; mặc định vẫn xuất v2 (không phá thói quen). *Xong: bạn chạy trên `sft_best.pt` thật, xuất OK.*

- [x] **P1-3 · `yui.cpp` đọc v3 (bỏ qua META an toàn, trả META qua `yui_meta`)**
  ✎ `yui.cpp` (+ kiểm tra biên: trần độ dài META, trần số mục) + `test_yui_kernels.cpp` · 📥 `yui.cpp`, `test_yui_kernels.cpp` · 📤 `Adaptive-MoE/yui.cpp` **và** `Y-App/native/yui.cpp` (cùng nội dung), `Adaptive-MoE/test_yui_kernels.cpp`
  ✅ `make test` xanh; file v1/v2/t2 cũ chạy y hệt (so đầu ra); file v3 chạy đúng; file v3 bị cắt/hỏng báo lỗi, không crash.

- [x] **P1-4 · Dart đọc META**
  ✎ `lib/state.dart` (ModelStore.isYui → đọc ver + META qua FFI `yui_info` hoặc đọc đầu file), `lib/screens/models_screen.dart` hiển thị tên/miền/lượng tử · 📥 (đã nhận) · 📤 `Y-App/lib/yui_header.dart` (MỚI), `Y-App/lib/state.dart`, `Y-App/lib/screens/models_screen.dart` (không cần sửa `engine.dart`)
  ✅ Tab Model hiện miền/kích cỡ của file v3; file ver lạ bị từ chối với thông báo rõ. (`[~]` tới khi bạn build.)

- [x] **P1-5 · `verify_cpp.py` đối chiếu v3 với PyTorch**
  ✎ `verify_cpp.py` · 📥 (đã nhận; `check_load.py` không cần đổi) · 📤 `Adaptive-MoE/verify_cpp.py`, `Adaptive-MoE/test_verify_cpp.py`, `Adaptive-MoE/.github/workflows/tests.yml`
  ✅ Đầu ra engine C++ với file v3 trùng PyTorch trong sai số của v2.

### P2 — Kho model (Model Hub) trong Y-App

Mục tiêu: người dùng mở tab **Kho**, thấy danh sách model (từ `manifest.json` trên HF), bấm tải; nguồn mặc định của bạn + nguồn người dùng tự thêm. Cấp **model**, không cấp expert.

- [x] **P2-1 · Kiểm chứng giới hạn Hugging Face (tài liệu + thử thật)**
  ✎ ghi kết luận vào Sổ tay mục 4 · 📥 bạn dán kết quả lệnh `curl -I -H "Range: bytes=0-1023" <link resolve>` và thử hai khoảng byte một request (mình gửi lệnh) · 📤 `Adaptive-MoE/docs/PLAN.md`
  ✅ Biết chắc: hạn mức ẩn danh hiện hành, redirect sang CDN, có/không hỗ trợ multi-range.

- [x] **P2-2 · Đặc tả `manifest.json` + `make_manifest.py`**
  ✎ tạo `make_manifest.py` (quét `models/*/` hoặc thư mục `.yui`, đọc META bằng `yui_format.py`, ghi `manifest.json` gồm url, sha256, bytes, arch_id, domains, quant, tokenizer_sha256, license) · 📥 `model_registry.py`, `model_card.py` · 📤 `Adaptive-MoE/make_manifest.py`, `Adaptive-MoE/test_make_manifest.py`
  ✅ Chạy trên thư mục mẫu ra manifest hợp lệ; sha256 khớp file.

- [x] **P2-3 · `hub_service.dart`: lấy manifest, cache 24 giờ, nhiều nguồn có thứ tự**
  ✎ tạo `lib/services/hub_service.dart` · 📥 `lib/state.dart` (phần ModelStore/Settings) · 📤 `Y-App/lib/services/hub_service.dart`
  ✅ Hai nguồn → danh sách gộp; nguồn lỗi không làm hỏng nguồn kia; dùng cache khi mất mạng.

- [~] **P2-4 · Tải model có tiếp tục (Range), kiểm sha256, giới hạn dung lượng, kiểm `arch_id`**
  ✎ nâng `ModelStore.download` trong `lib/state.dart` · 📥 `lib/state.dart` · 📤 `Y-App/lib/state.dart`
  ✅ Ngắt mạng giữa chừng rồi tải tiếp được; sai sha256 thì xoá file; file quá trần dung lượng bị từ chối.
  📤 đã giao: `Y-App/lib/state.dart` (thêm `ModelStore.downloadHub`, `AppState.downloadHubModel`; `download()` cũ giữ nguyên), `Y-App/lib/services/hub_download.dart` (mới), `Y-App/tools/hub_download_check.dart` (mới)

- [ ] **P2-5 · Bộ nhớ đệm đĩa LRU cho model/expert tải về**
  ✎ `lib/services/hub_service.dart`, `lib/state.dart` · 📥 hai file đó (bản mới nhất) · 📤 hai file ở `Y-App/lib/…`
  ✅ Vượt trần (mặc định 500 MB, chỉnh được) thì xoá model ít dùng nhất; không bao giờ xoá model đang chạy hoặc model người dùng tự nhập.

- [ ] **P2-6 · Giao diện tab Kho + thêm/bớt nguồn**
  ✎ tạo `lib/screens/hub_screen.dart`, sửa `lib/main.dart` · 📥 `lib/main.dart`, `lib/screens/models_screen.dart` · 📤 `Y-App/lib/screens/hub_screen.dart`, `Y-App/lib/main.dart`
  ✅ Xem danh sách, tải, xoá, thêm nguồn bằng link; cảnh báo khi `arch_id` lạ.

- [ ] **P2-7 · Cài đặt "Chế độ Hybrid" (mặc định TẮT) + lưu HF token tuỳ chọn**
  ✎ `lib/state.dart` (Settings), `lib/screens/settings_screen.dart` · 📥 hai file đó · 📤 hai file ở `Y-App/lib/…`
  ✅ Tắt Hybrid thì app không gọi thêm máy chủ nào ngoài các mục đã có trong chính sách; token lưu trên máy.

- [ ] **P2-8 · Cập nhật pháp lý/quyền riêng tư cho tính năng mới**
  ✎ `assets/legal/privacy_vi.md`, `privacy_en.md`, `terms_*.md`, `legal/LEGAL_CHECKLIST.md`, `legal/THIRD_PARTY_NOTICES.md` · 📥 5 file đó · 📤 5 file ở `Y-App/…`
  ✅ Mục 3 "Dữ liệu gửi ra ngoài" liệt kê đúng: manifest, tải model, (P4) truy vấn dataset. Bản mẫu vẫn cần luật sư rà soát (như chính văn bản đã ghi).

### P3 — Kho tri thức ngay trong app (port gọn RAG/Graphify sang Dart)

Lý do: Cloud-Edge RAG cần "tra kho nội bộ trước" nhưng app hiện **không có** kho nào. Port bản **gọn**, tương thích định dạng thư mục của `yui_vault.py` để đổi máy chép là chạy.

- [ ] **P3-1 · Đặc tả kho trong app (thư mục `chat/ notes/ kb/ index/`, định dạng jsonl)**
  ✎ `docs/APP_KB.md` · 📥 `yui_vault.py` · 📤 `Adaptive-MoE/docs/APP_KB.md`
  ✅ Bạn duyệt; chốt có mã hoá ở bản đầu hay để P3-6.

- [ ] **P3-2 · Dart: tách từ + BM25 tiếng Việt không dấu (port gọn `yui_rag.py`)**
  ✎ tạo `lib/kb/bm25.dart`, `lib/kb/vi_text.dart` · 📥 `yui_rag.py` (hàm `fold`, `tokens`, `split_text`, lớp `RAG`) · 📤 hai file ở `Y-App/lib/kb/`
  ✅ Tập test chéo: cùng truy vấn, Dart và Python xếp hạng giống nhau trên dữ liệu mẫu ("ten meo" ra "tên mèo").

- [ ] **P3-3 · Dart: Graphify-lite (ứng viên rộng → chấm theo sự kiện → thay cũ bằng mới → MMR → câu thường)**
  ✎ tạo `lib/kb/graphify.dart` · 📥 `yui_graph.py`, `graphify_bench.py` (chỉ phần dữ liệu mẫu) · 📤 `Y-App/lib/kb/graphify.dart`
  ✅ Bộ test vàng sinh từ Python: đầu ra Dart ≈ Python (chấp nhận khác biệt nhỏ, ghi lại); ngân sách ký tự được tôn trọng.

- [ ] **P3-4 · Công cụ `recall` / `note` trong `agent.dart`**
  ✎ `lib/agent.dart` · 📥 `lib/agent.dart`, `tool_data.py` (phần cú pháp recall/note) · 📤 `Y-App/lib/agent.dart`
  ✅ Model đã học cú pháp gọi được `recall`; kết quả ≤ `kToolResultChars`.
  Lưu ý: model **phải được train có dữ liệu `recall`** (đã có trong `tool_data.py`, bật bằng `DATA_KINDS=...,tools`).

- [ ] **P3-5 · Tự lưu chat vào kho + màn tìm/quản lý/xoá**
  ✎ `lib/state.dart`, `lib/screens/history_screen.dart` · 📥 hai file đó · 📤 hai file ở `Y-App/lib/…`
  ✅ Chat mới được lập chỉ mục; xoá chat thì xoá cả khỏi chỉ mục; có nút xoá toàn bộ kho.

- [ ] **P3-6 · (Tuỳ chọn) Mã hoá kho tương thích `YUI_KEY`**
  ✎ `lib/kb/cipher.dart` · 📥 `yui_vault.py` (lớp `Cipher`) · 📤 `Y-App/lib/kb/cipher.dart`
  ✅ File `.enc` do Python tạo mở được trong app và ngược lại. (Bỏ nếu chưa cần; đánh `[-]`.)

- [ ] **P3-7 · Bộ test chéo Python↔Dart (golden files)**
  ✎ `tests/golden/` sinh bằng Python, Dart đọc so sánh · 📥 các file test hiện có `test_yui_graph.py` · 📤 `Adaptive-MoE/tests/golden/*`, `Y-App/test/kb_golden_test.dart`
  ✅ CI Y-App chạy `flutter test` xanh.

### P4 — Tra Dataset đám mây (Cloud-Edge RAG)

- [ ] **P4-1 · Kiểm chứng datasets-server (`/search`, `/filter`, `/rows`) và giới hạn**
  ✎ ghi vào Sổ tay mục 4 · 📥 bạn dán kết quả vài lệnh `curl` mình đưa (trên 1 dataset của bạn + 1 dataset công khai) · 📤 `Adaptive-MoE/docs/PLAN.md`
  ✅ Biết dataset nào tra được, định dạng trả về, giới hạn, độ trễ thật.

- [ ] **P4-2 · Trích từ khoá trung tính trên máy + màn xem trước truy vấn**
  ✎ tạo `lib/kb/keywords.dart`; UI hiện "sẽ gửi: …" trước khi gọi mạng · 📥 `lib/screens/chat_screen.dart` · 📤 `Y-App/lib/kb/keywords.dart`, `Y-App/lib/screens/chat_screen.dart`
  ✅ Tên riêng/số điện thoại/email/mật khẩu bị loại khỏi truy vấn; người dùng bấm huỷ được.

- [ ] **P4-3 · `dataset_service.dart`: danh sách cho phép + manifest dataset**
  ✎ tạo `lib/services/dataset_service.dart`, mở rộng `make_manifest.py` (mục `datasets[]`) · 📥 `make_manifest.py` (bản P2-2) · 📤 hai file
  ✅ Chỉ tra các dataset trong danh sách cho phép; ngưỡng uy tín (lượt tải/thích) chỉ áp cho chế độ quét mở.

- [ ] **P4-4 · Dọn rác + nén theo ngân sách ký tự (port `vi_filter`/`yui_kb.clean`, dùng Graphify-lite)**
  ✎ `lib/kb/clean.dart` · 📥 `yui_kb.py` (hàm `clean`, `distill`), `vi_filter.py` · 📤 `Y-App/lib/kb/clean.dart`
  ✅ 2–5 KB tải về còn ≤ ngân sách (mặc định ~200 ký tự cho model L=128–256).

- [ ] **P4-5 · Ghi nguồn cuối câu + chống tiêm lệnh từ dữ liệu ngoài**
  ✎ `lib/agent.dart`, `lib/state.dart` · 📥 hai file đó · 📤 hai file
  ✅ Thẻ `<tool>` trong dữ liệu ngoài bị vô hiệu (như `yui_agent.sanitize`); mỗi câu trả lời có dòng "nguồn: …".

- [ ] **P4-6 · Lùi dần khi 429 + cache kết quả tra cứu**
  ✎ `lib/services/dataset_service.dart` · 📥 `yui_web.py` (logic nghỉ 429) · 📤 `Y-App/lib/services/dataset_service.dart`
  ✅ 429 → nghỉ tăng dần; cùng truy vấn trong 24 giờ không gọi lại mạng.

### P5 — Mượn model chuyên theo câu hỏi (cấp model, "khách trọ")

- [ ] **P5-1 · API C++ đo độ chắc chắn (`yui_probe`: entropy token đầu / margin)**
  ✎ `yui.cpp`, `lib/engine.dart` · 📥 `yui.cpp`, `lib/engine.dart` · 📤 `yui.cpp` (cả 2 repo), `Y-App/lib/engine.dart`
  ✅ Gọi được từ Dart; tốn < 1 bước forward; có test C++.

- [ ] **P5-2 · Bộ phân loại miền: luật/từ khoá + `domains[]` trong manifest**
  ✎ `lib/kb/router.dart` · 📥 `lib/services/hub_service.dart` (bản mới nhất) · 📤 `Y-App/lib/kb/router.dart`
  ✅ Câu "viết code game" → gợi ý model miền `code`; câu chat thường không gợi ý.

- [ ] **P5-3 · Luồng mượn model: hỏi người dùng → tải → chạy → trả kết quả → giải phóng**
  ✎ `lib/state.dart`, `lib/engine.dart` · 📥 hai file đó (bản mới nhất) · 📤 hai file
  ✅ Bàn giao bằng ngữ cảnh nén (Graphify-lite); sau khi xong `yui_free` model mượn; RAM trở về như cũ; mất mạng thì rơi về model lõi.

- [ ] **P5-4 · Ngưỡng tự điều chỉnh theo tốc độ mạng đo thật (hysteresis)**
  ✎ `lib/kb/router.dart` · 📥 `lib/kb/router.dart` · 📤 cùng file
  ✅ Mạng chậm → chỉ gợi ý khi chắc chắn; chênh lệch nhỏ → không đổi model.

- [ ] **P5-5 · Đo A/B và quyết định giữ/bỏ**
  ✎ `bench_swarm.py` (bộ câu hỏi + chấm) · 📥 `bench.py`, `compare.py` · 📤 `Adaptive-MoE/bench_swarm.py`
  ✅ Bảng số: lõi / lõi+RAG / lõi+model chuyên. Nếu lõi+model chuyên không hơn lõi+RAG rõ rệt → đánh `[-]` phần mượn model và ghi lý do.

### P6 — Nghiên cứu ghép Expert (CHƯA CAM KẾT; chỉ làm sau khi P2–P5 ổn)

Chỉ chạy bằng Python trên Kaggle. Không đụng `yui.cpp`/app cho tới khi Go.

- [ ] **P6-1 · Thí nghiệm A: cùng gốc (BTX-lite)** — 2 nhánh SFT từ cùng checkpoint, gộp expert, đo loss/perplexity
  📥 `train_sft.py`, `adaptive_moe.py`, `lm_experiment.py` · 📤 `Adaptive-MoE/exp_merge_same_base.py`
  ✅ Có bảng loss: nhánh A, nhánh B, gộp.

- [ ] **P6-2 · Thí nghiệm B: hai model độc lập (Procrustes/CKA trên ~200 câu mẫu)**
  📥 `adaptive_moe.py`, `langcheck.py` · 📤 `Adaptive-MoE/exp_align_independent.py`, `Adaptive-MoE/anchor_vi_en.jsonl`
  ✅ Có số đo độ tương đồng trước/sau căn chỉnh và loss khi cắm expert ngoại lai.

- [ ] **P6-3 · Quyết định Go/No-Go**
  ✅ **Go** chỉ khi: gộp cùng gốc giảm perplexity ≥ 5% so với nhánh tốt nhất VÀ không làm xấu miền kia > 2%. Nếu không → đánh `[-]` P6-4, P6-5 và ghi vào Nhật ký.

- [ ] **P6-4 · (Nếu Go) Expert Arena + tải lẻ theo mục lục v3** — 📥 `yui.cpp` · 📤 `yui.cpp` (cả 2 repo)
  ✅ Vùng đệm cố định, không malloc/free liên tục; có test.

- [ ] **P6-5 · (Nếu Go) `bridge.bin` + `anchor_hash`** — 📥 `export_yui.py`, `yui_format.py` · 📤 hai file
  ✅ Chỉ cho ghép khi `anchor_hash` trùng.

### P7 — Phát hành + tài liệu

- [ ] **P7-1 · Cập nhật README hai repo (EN + VI) theo tính năng thật đã xong** — 📥 mục liên quan của README (mình xin đúng đoạn) · 📤 `README.md`, `README.vi.md`, `Y-App/README.md`
- [ ] **P7-2 · CI: test v3 + kiểm đồng bộ `yui.cpp` + `flutter test` + build** — 📥 hai file workflow · 📤 hai file workflow
- [ ] **P7-3 · Cập nhật 2 sơ đồ (`SO_DO_HIEN_TAI.md`, `SO_DO_SAU_KHI_XONG.md`) cho khớp thực tế** — 📤 `Adaptive-MoE/docs/SO_DO_*.md`
- [ ] **P7-4 · Gắn tag, nâng phiên bản engine (`yui_version`) và app** — 📥 `yui.cpp` (dòng version), `pubspec.yaml` · 📤 hai file

---

## 4. SỔ TAY KỸ THUẬT (đã đối chiếu code thật; để chat mới khỏi cần gửi lại zip)

### 4.1 Định dạng `.yui` hiện tại (v1/v2, little-endian) — từ `export_yui.py` + `yui.cpp::Model::load`
- `MAGIC = "YUI1"`; `<12I`: `ver, quant, V, D, layers, heads, E, F, n_shared, L, min_k, max_k`; `<2f`: `top_p, min_strength`.
- Tokenizer: `<I nvocab` + mỗi token (`<H len` + bytes UTF-8); `<I nmerges` + cặp chuỗi; `<i eos`; `<I nranges` + mỗi `<3I` (khoảng Unicode, lớp 1 = chữ, 2 = số).
- Trọng số theo thứ tự: `emb` (V×D) → mỗi lớp: `n1.w,n1.b` (D f32), `qkv` (3D×D), `o` (D×D), `n2.w,n2.b`, `router.gate` (E×D **f32, không có byte kiểu**), `strength_head.w` (D f32) + `.b` (1 f32), E expert (`w_gate` F×D, `w_up` F×D, `w_down` D×F), `n_shared` expert chung (cùng dạng) → `norm.w,b` → `head` (V×D). Luôn `F = 2*D`.
- Kiểu ma trận (byte đứng trước từng ma trận từ v2): 0 f32, 1 int8 theo hàng (scale f32/hàng rồi int8), 2 q4 (khối 32: fp16 scale rồi 16 byte), 3 t2 ternary (`<H` cỡ nhóm, scale fp16, 8 byte/khối 32). Router và LayerNorm luôn f32.
- Hệ quả: **không có mục lục byte**; muốn đọc 1 expert phải đi tuần tự (hoặc tự tính offset theo kiểu lượng tử) → lý do cần v3.
- Engine từ chối `ver` ngoài {1,2}, `quant > 3`, `E > 256`, cấu hình lệch (`D % heads`…). Kiểm tra biên đọc: `Reader::need` (báo "file .yui bị cụt").

### 4.2 Engine C++ (`yui.cpp`, version chuỗi `yui-engine 1.3-img`; từ P1-3 đọc `.yui` v1/v2/v3 và có thêm `yui_meta`)
- Hàm xuất (`YUI_API`): `yui_version, yui_set_threads, yui_get_threads, yui_cpu_features, yui_load, yui_free, yui_info, yui_meta (JSON), yui_chat_begin, yui_chat_begin_img, yui_complete_begin, yui_gen_next, yui_gen_tokens, yui_gen_end, yui_vision_load/run/free`.
- Biên dịch thư viện: `-DYUI_LIB -shared -fPIC -std=c++17` (Android arm64 qua NDK trong `build-app.yml`); CLI: `make` / `make portable`; test: `make test`.
- Có sẵn: SIMD (AVX2+FMA / NEON / NEON+dotprod), nhóm luồng riêng, mmap (Android; Windows đọc vào RAM), đoán trước token (`YUI_SPEC`), router cấp token theo từng lớp, KV cache trong `[0, L)`.

### 4.3 Y-App (Flutter)
- 4 tab: Chat · Lịch sử · Model · Cài đặt. `lib/engine.dart` = isolate nền + FFI. `lib/state.dart` = `ChatStore`, `ModelStore` (nhập file, tải link + token, xoá), `Settings`, `AppState`, kiểm tra cập nhật qua GitHub Releases (`kRepo = BachDaThan/Y-App`).
- `ModelStore.isYui` chỉ kiểm 4 byte đầu `"YUI1"`. Danh sách model = mọi `*.yui` trong thư mục dữ liệu app.
- `lib/agent.dart`: chỉ 3 công cụ `calc`, `time`, `search` (Wikipedia vi); `kToolResultChars = 120`; tối đa 2 lần gọi công cụ/câu. **Không có** vault/RAG/Graphify/skill/orchestrator (chỉ có bản Python).
- Dịch vụ: `voice_service`, `sherpa_stt` (STT offline), `vision_service`, `update_service`, `notifier`, `background_guard` (flutter_foreground_task 8.17.0 ghim phiên bản).
- `yui.cpp` của Y-App nằm ở **`native/yui.cpp`** (không phải gốc repo); bản Adaptive-MoE ở gốc. Cả hai đang trùng sha256 `39cc442d…c073e` (115 111 byte).
- Build: `.github/workflows/build-app.yml` (Android APK arm64 + Windows setup/portable), phát hành lên Releases tag `latest`.

### 4.4 Adaptive-MoE (Python) — module chính
- Huấn luyện/dữ liệu: `adaptive_moe.py`, `train_lm.py`, `train_sft.py`, `make_data.py`, `make_pretrain.py`, `tool_data*.py`, `kaggle_pipeline.py`, `model_registry.py` (đặt tên `models/<tên>/`), `export_yui.py`, `bitnet_*.py`, `elastic_ctx.py`, `elastic_kv.py`.
- Runtime/agent: `yui_agent.py` (ToolBox, sandbox 3 bậc, MCP, endpoint), `yui_sandbox.py`, `yui_orch.py`, `yui_task.py`, `yui_rag.py` (BM25), `yui_kb.py`, `yui_graph.py` (Graphify), `yui_vault.py` (kho + mã hoá `YUI_KEY`), `yui_skills.py`, `yui_web.py` (tìm web nhiều tầng + nghỉ 429), `providers.py`, `engine.py`/`chat.py`/`app.py` (server/Gradio).
- CI: `.github/workflows/tests.yml` chạy `pytest -q` (Python 3.10, 3.12). Có ~15 file `test_*.py`.

### 4.5 Số liệu/giả định CHƯA kiểm chứng (đừng dựa vào cho tới khi xong task tương ứng)
- ~~Hạn mức HF ẩn danh~~ → **đã đối chiếu tài liệu (P2-1)**: 500 API / 3 000 resolver / 100 trang mỗi 5 phút theo IP, cửa sổ cố định, bảng tháng 9/2025 và HF ghi "có thể thay đổi". **Chưa thấy header `RateLimit` trong lần đo thật** ⇒ app phải tự xử lý 429 (lùi dần), không được dựa vào con số.
- ~~Multi-range của CDN HF~~ → **đo thật (P2-1): CÓ.** `Range: bytes=0-99,1000-1099` trả `206` + `multipart/byteranges` từ CloudFront (506 byte cho 2×100 byte dữ liệu). Range đơn (tải tiếp khi đứt) cũng nằm trong khả năng đó. Lưu ý: P2 chỉ cần tải nguyên file + tải tiếp, **chưa cần** multi-range.
- `datasets-server /search` dùng được trên dataset nào → **P4-1**.
- Mọi tên bài báo/giảm độ trễ do Gemini nêu (Symphony-MoE, HARC, ADEPT, FineMoE, DynaExq…) → **P6**, coi là gợi ý đọc, không phải bằng chứng.

---

## 5. NHẬT KÝ QUYẾT ĐỊNH & CÂU HỎI MỞ

| Ngày | Mục | Ghi chú |
|---|---|---|
| 2026-11-12 | Khởi tạo kế hoạch v1 | Dựa trên đọc code 2 repo + tư vấn Gemini; chưa có task nào xong |
| 2026-11-13 | P0-1 xong `[x]` | Cả 2 repo xanh sau khi up. Chi tiết bên dưới: |
| 2026-11-13 | (P0-1 chi tiết) | Hiện 2 bản `yui.cpp` đang trùng nhau. Script mặc định bỏ qua nếu mất mạng, CI dùng `--strict`. Push `yui.cpp` ở cả 2 repo cùng lúc có thể đỏ tạm thời đến khi repo kia cập nhật |
| 2026-11-13 | P0-2 `[x]` | Bạn đã chuyển `provider-chain.patch` vào `docs/archive/` (xác nhận trên repo). `.gitignore` mới đã lên (chặn `/*.patch` chỉ ở gốc, không chặn `docs/archive/`). **Còn nhắc:** xoá file rác `docs/archive/Z` (1 byte), vẫn còn trên repo lúc kiểm 2026-11-13 |
| 2026-11-13 | P0-3 `[x]` baseline | `make test` (C++): **OK** (isa=avx2+fma). CI `tests` (pytest, Py 3.10 + 3.12): **xanh** (bạn báo). **pytest: 237 passed, 1 skipped, 204.91 s (0:03:24)** (bạn báo, gồm cả `test_yui_format.py`). Y-App: Actions build **xanh**. Mốc so sánh từ giờ: không được tụt dưới 237 passed / 0 failed |
| 2026-11-13 | P0-4 `[x]` | README Y-App sửa theo thực tế: build Actions OK; vẫn ghi "chưa đo trên điện thoại thật" cho tính năng chưa đo. **Chạy thật trên điện thoại Android (bạn báo):** cài APK OK; nạp `yui_q4.yui` (32,9 MB, d=256, 6 lớp, 16 expert, ngữ cảnh 128, q4) thành công; chat mượt; **≈ 186,6–240,4 token/s**; nội dung trả lời còn lan man (model nhỏ/chưa fine-tune kỹ). Mốc tốc độ để so khi làm P5. README đã ghi số này |
| 2026-11-13 | P0-5 `[x]` | `yui_format.py` (đọc header + bảng tensor + sha256 tokenizer, chỉ dùng thư viện chuẩn, kiểm biên, từ chối ver ≠ 1/2) và `test_yui_format.py` (**14 test xanh**, có thêm kiểm `D % (2·heads)` cho khớp engine, ở môi trường mình: f32/int8/q4/q4all/t2, file cụt/thừa/magic/ver/cấu hình lạ, và nạp cùng file bằng `yui.cpp` qua ctypes khớp V/D/layers/E/heads/ctx). Tokenizer hash = sha256 của vocab+merges+eos (bỏ bảng khoảng Unicode vì phụ thuộc phiên bản Python) |
| 2026-11-13 | P1-1 `[x]` (đã chốt 6/6) | Giao `docs/YUI_FORMAT.md`: đặc tả v1/v2/t2 đã đối chiếu code thật + **v3 đề xuất** (META JSON + mục lục nhị phân + footer `YUIX`). Chưa code gì. **Bạn chốt:** (1) `license` bắt buộc khoá, cho `"unknown"`; (2) `arch_id` = `amoe-d{D}-l{layers}-h{heads}-e{E}-s{n_shared}-f{F}`; (3) domains chuẩn `general, chat, code, math, vi-fiction, translate, tool, other`; (4) `centroid` để P5; (5) mục lục nhị phân; (6) **META ngay sau header** (spec đã sửa theo) |
| 2026-11-13 | P1-2 `[~]` | `export_yui.py --fmt 3` (+ `--name --domains --lang --license --desc --base --source-url`), `yui_format.py` đọc/kiểm v3 (`read_meta` nhanh + `read_yui` kiểm toàn bộ: META khớp header, mục lục khớp bố cục, chân file `YUIX`), `docs/YUI_FORMAT.md` chốt v3. **Kiểm:** xuất v1/v2 mới == bản cũ **từng byte** (f32/int8/q4/q4all/t2); trọng số v3 == v2 từng byte; `test_yui_format.py` **46 test xanh** + 136 test yui khác xanh ở môi trường mình. Chưa chạy `main()` trên checkpoint thật (môi trường mình không có torch). **Bạn chạy giúp:** `python export_yui.py --out <thư_mục_model> --ckpt auto --quant q4 --fmt 3 --dst model_v3.yui --name yui-q4 --domains chat,general --lang vi,en --license unknown` rồi `python yui_format.py model_v3.yui`, dán 1 dòng kết quả. Engine C++ **chưa** đọc v3 (P1-3), nên đừng nạp file này vào app/`yui.cpp` hiện tại |
| 2026-11-13 | P1-2 `[x]` | Bạn chạy trên checkpoint thật `sft_best.pt` (49,91M tham số): `model_v3.yui` 33,0 MB, `yui_format.py` đọc lại OK: `v3 q4, V=16000 D=256 layers=6 heads=4 E=16 L=128`, tokenizer `b3c098768a48`, META `yui-q4`, domains `chat,general`, lang `vi,en`, license `unknown`. Sai số lượng tử q4 8,58% (moe 8,58%) |
| 2026-11-13 | P1-3 `[~]` | `yui.cpp` đọc v3: kiểm `meta_len` (1…65536) và META phải là `{…}`; giữ chuỗi META không phân tích; kiểm chân file `YUIX` + mục lục (số mục, nối tiếp, phủ đúng vùng trọng số) sau khi nạp; API mới `yui_meta`. **Lệch kế hoạch có chủ ý:** META **không** nằm trong `yui_info` vì app cấp buffer 1024 byte cho hàm đó (`engine.dart:393`), META tới 64 KiB sẽ làm nó lỗi `-2`; `yui_info` chỉ thêm `"ver"`. **Kiểm (môi trường mình):** `make test` OK; engine mới sinh chữ **giống hệt** engine cũ trên file f32/int8/q4/q4all/t2 (so đầu ra); v3 sinh giống hệt v2 (q4, t2) và giống v1 (f32, int8); v3 bị cắt ở 11 vị trí, sai chân/mục lục/META/ver đều báo lỗi gọn, không crash; engine **cũ** gặp v3 báo "phiên bản .yui không hỗ trợ" (đúng dự kiến). `test_yui_format.py` **55 test**, cả nhóm yui **148 passed, 1 skipped**. Không dùng `test_yui_kernels.cpp` cho phần này (test bằng ctypes trong pytest để chạy cả trên CI Python). **Lưu ý CI:** job `sync-yui` sẽ **đỏ tạm** giữa hai lần up `yui.cpp`; up xong cả 2 repo rồi bấm Re-run là xanh |
| 2026-11-13 | P1-3 `[x]` | Kiểm trên repo: `yui.cpp` ở `Adaptive-MoE/` và `Y-App/native/` **trùng nhau và đúng bản P1-3** (so từng byte). Chưa nghe bạn báo CI, nếu `sync-yui` hoặc build đỏ thì báo mình |
| 2026-11-13 | P1-4 `[~]` | **Cách làm:** đọc META bằng **Dart thuần** từ đầu file (`yui_header.dart`: 60 byte header + `u32` độ dài + JSON) thay vì gọi FFI, vì tab Model cần hiện thông tin **trước khi nạp** model và không muốn nạp cả file chỉ để xem tên. Vì vậy `engine.dart` giữ nguyên (và `yui_meta` của engine để dành cho dùng sau). **Tab Model:** mỗi file hiện kích cỡ · lượng tử · số tham số · `.yui vN`; file v3 thêm tên model, chip **miền** (tiếng Việt: Tổng quát, Trò chuyện, Lập trình, Toán, Truyện tiếng Việt, Dịch, Công cụ, Khác) và ngôn ngữ, giấy phép; thẻ "Đang dùng" hiện mô tả + miền. **Từ chối rõ:** phiên bản `.yui` lạ (≠ 1–3), lượng tử lạ, cấu hình hỏng, META hỏng/cụt đều báo tiếng Việt ở lúc nhập, tải link, và bấm "Dùng" (không gọi engine). Header được cache theo (đường dẫn, kích thước, ngày sửa) nên vẽ lại không đọc lại file. META chỉ để hiển thị: không chạy/tải/mở link từ META, ký tự điều khiển bị thay bằng dấu cách. **Kiểm:** môi trường mình **không có Dart/Flutter** nên **chưa biên dịch**; mình đối chiếu offset đọc header với `yui_format.py` trên file v1/v2/v3 thật (khớp) và rà cú pháp tay. Đánh `[x]` sau khi Actions build xanh và bạn thấy tab Model hiện đúng với `model_v3.yui` |
| 2026-11-13 | P1-4 `[x]` | Bạn báo: Actions build **xanh 100%**. Tab Model hiện đúng `model_v3.yui`: `yui-q4`, `.yui v3`, 33,0 MB, q4, 49,9M tham số, chip "Trò chuyện" và "Tổng quát", ngôn ngữ vi/en, giấy phép unknown; file cũ nhận đúng nhãn `.yui v2`. Bấm "Dùng" nạp `model_v3.yui` ngay, chat bình thường, **≈ 174–230 token/s** (câu đầu 8 token: 49,4/s) trên Android — tương đương v2 (≈ 187–240), chưa thấy chậm đi |
| 2026-11-13 | P1-5 `[~]` | `verify_cpp.py` nâng cấp: (0) đọc và kiểm toàn bộ file bằng `yui_format` (v3: META, mục lục, chân file) + đối chiếu `config.json` với header; (1) tokenizer như cũ; (2) logits so PyTorch như cũ nhưng có **phán quyết**: f32 cần `max|diff| < 1e-3` và argmax 100%, file lượng tử chỉ cảnh báo nếu argmax < 90% (nhanh vs `--exact` cũng được so); (3) **mới `--compare FILE.yui`** (không cần torch): so logits engine giữa 2 file, cùng quant/trọng số thì phải **giống hệt** (`max|diff| = 0`) và cùng `tokenizer_sha256`. Kết thúc in `KẾT QUẢ: ĐẠT/KHÔNG ĐẠT`, mã thoát 1 khi có lỗi cứng. `test_verify_cpp.py` (8 test, cần g++ + `tokenizers`): v3 == v2 từng bit ở q4/t2/f32 (k=1,2 × nhanh/exact), trọng số khác → KHÔNG ĐẠT, `config.json` lệch → KHÔNG ĐẠT, file hỏng/cụt → KHÔNG ĐẠT; `tests.yml` thêm `tokenizers` vào bước cài để test chạy trên CI (trước đây thiếu). **Chưa chạy được phần so với PyTorch** (môi trường mình không có torch; phần đó giữ logic cũ). **Bạn chạy giúp 3 lệnh** trên máy có torch, trong thư mục repo, với `<D>` là thư mục model (có `config.json`, `tokenizer.json`, `sft_best.pt`): `make` ; `python export_yui.py --out <D> --ckpt auto --quant q4 --dst model_v2.yui` ; `python export_yui.py --out <D> --ckpt auto --quant q4 --fmt 3 --name yui-q4 --domains chat,general --lang vi,en --dst model_v3.yui` ; rồi `python verify_cpp.py --out <D> --model model_v3.yui --yui ./yui` và `python verify_cpp.py --out <D> --model model_v3.yui --compare model_v2.yui --yui ./yui --skip_logits`. Dán 2 khối kết quả (nhất là dòng `KẾT QUẢ`) |
| 2026-11-13 | P1-5 `[x]` | Bạn chạy trên `sft_best.pt` (49,91M tham số), cả 2 lệnh **ĐẠT**. **So với PyTorch** (file `model_v3.yui` q4): tokenizer lệch **0/8**; logits `max|diff|` 0,79 (k=1) và 0,77–0,78 (k=2), **argmax khớp 93%** (29 vị trí); chế độ nhanh và `--exact` cho kết quả gần như trùng nhau (k=1 giống hệt, k=2 lệch 0,78 vs 0,77) ⇒ kích hoạt int8 nhanh không thêm sai số đáng kể; mức lệch này là sai số lượng tử q4 (đo ở export: 8,58%), không phải lỗi engine/định dạng. **v3 so với v2** (cùng checkpoint, cùng q4): `max|diff| = 0,00e+00` và argmax khớp **100%** ở cả 4 tổ hợp (k=1,2 × nhanh/exact), cùng `tokenizer_sha256 b3c098768a48`. File v3 nặng hơn v2 **0,03 MB** (32,96 so với 32,93 MB: META + mục lục + chân file). **Mốc để so về sau:** q4 trên checkpoint này argmax khớp PyTorch ≥ 93% / 29 vị trí. **Hết giai đoạn P1** |
| 2026-11-13 | Trả lời câu hỏi mở số 3 (P2) | Repo model mặc định: **`BachDaThan/Yui-Models`** (đã tạo trên Hugging Face), chế độ **public** để app tải trực tiếp không cần token; vẫn giữ tuỳ chọn nhập token cho người dùng trỏ sang repo private (P2-7) |
| 2026-11-13 | P2-1 `[x]` | Tài liệu HF: 500 API / 3 000 resolver / 100 trang mỗi 5 phút (ẩn danh, theo IP, cửa sổ cố định, bảng 9/2025, *có thể thay đổi*); quá hạn → 429 + header `RateLimit`. **Đo thật** trên `Yui-Models/model_v3.yui` (bạn báo): multi-range → `HTTP/2 206`, `multipart/byteranges` từ CloudFront, `size=506` ⇒ **hỗ trợ**. Hai lệnh `HEAD` chỉ cho `HTTP/2 200` (khối 2 có `content-length: 174230`), **không thấy** `location`, `ratelimit`, `accept-ranges`, `content-range`; vì vậy redirect và header hạn mức **chưa quan sát trực tiếp** (CDN được xác nhận gián tiếp qua khối 3). **Chưa giải thích được** `174230` (file là ≈ 33,0 MB): có thể `HEAD` ở HF khác `GET`, chưa kết luận. Không chặn P2: P2-4 kiểm `sha256` + `bytes` theo manifest nên tải sai sẽ bị phát hiện. **Quyết định thiết kế:** (a) tải bằng `/resolve/` (nhóm resolver có hạn mức cao nhất), (b) 429 → lùi dần, (c) manifest cache 24 giờ nên mỗi lần mở Kho tốn ≤ 1 resolver |
| 2026-11-13 | P2-2 `[~]` | `make_manifest.py` (quét `*.yui` đệ quy, đọc + kiểm toàn bộ từng file bằng `yui_format.read_yui`, ghi `manifest.json` version 1: `id,name,file,url,bytes,sha256,yui_ver,quant,params,arch_id,tokenizer_sha256,domains,lang,license,ctx` + tuỳ chọn `description,created,base`; v1/v2 không META → id = tên file, `domains/lang` rỗng, `license = unknown`). Có `validate_manifest` (https, sha256 hex, không `..`, không trùng id, tối đa 500 model) và `--check` (so manifest với file thật). File hỏng bị bỏ qua + cảnh báo, `--strict` thì không ghi gì. Chỉ dùng thư viện chuẩn. **Kiểm (môi trường mình):** `test_make_manifest.py` **22 test xanh** (nội dung, q4all, thư mục con có dấu cách → `%20`, v2 không META, file cụt/rác, trùng id, base_url phải https, kết quả lặp lại byte-từng-byte, `--check` bắt đổi 1 byte, 13 kiểu manifest sai bị từ chối) + `test_yui_format.py` vẫn xanh (tổng 77). **Bạn chạy giúp:** đặt `model_v3.yui` vào một thư mục (vd `hub/`) rồi `python make_manifest.py hub --strict` và `python make_manifest.py hub --check hub/manifest.json`; dán 2 khối kết quả; sau đó up `hub/manifest.json` lên **gốc** repo `BachDaThan/Yui-Models` (cùng cấp `model_v3.yui`, vì `url` = base_url + đường dẫn file) |
| 2026-11-13 | P2-2 `[x]` | Bạn chạy trên `hub/` chứa `model_v3.yui`: lệnh 1 ghi `manifest.json` 1 model (`yui-q4`, .yui v3, q4, 32,96 MB, domains chat+general); lệnh 2 `--check` **KHỚP**. Bạn đã up `model_v3.yui` và `manifest.json` lên gốc `BachDaThan/Yui-Models`. **Kho hiện chỉ có 1 model; chưa có model cho các ngành khác (code, toán, dịch, truyện…)** |
| 2026-11-13 | P2-3 `[x]` | `lib/services/hub_service.dart` (Dart thuần, không phụ thuộc Flutter, chưa được app nào gọi nên **không thể làm hỏng bản build hiện tại**): `HubService.load(nguồn[], force)` gộp nhiều nguồn theo thứ tự (trùng id: nguồn trước thắng, có ghi chú), cache 24 giờ từng nguồn (đồng hồ lùi thì không tin cache), nguồn lỗi không ảnh hưởng nguồn khác, mất mạng dùng cache cũ có cờ `stale`, 429/5xx/mạng thử lại ≤ 3 lần (1s, 2s hoặc `Retry-After` tối đa 30s), chỉ nhận https (cả sau chuyển hướng), manifest ≤ 1 MB ≤ 500 mục, mục sai bị bỏ qua và đếm lại, chuỗi hiển thị bị làm sạch ký tự điều khiển. **Không tự nối mạng**: nơi gọi phải kiểm công tắc Kho (mặc định tắt, làm ở P2-6/P2-8). **Kiểm thật bằng Dart SDK 3.14 (không phải Flutter):** `dart analyze` 0 lỗi; `tools/hub_service_check.dart` **34 kiểm tra đạt** (gộp 2 nguồn, nguồn lỗi, cache TTL, mất mạng, thử lại, 18 kiểu mục sai, 5 kiểu manifest sai, trùng id, link nguồn không an toàn, cache hỏng, máy chủ cục bộ cho 200/429/quá lớn/chuyển hướng http) và **Dart đọc đúng manifest do Python sinh** (3 model, gồm cả thư mục có dấu cách `%20`). **Chưa kiểm:** build Flutter/Android thật (chờ GitHub Actions khi P2-6 nối vào app) |
| 2026-11-13 | P2-4 `[~]` | **Đã làm:** `lib/services/hub_download.dart` (Dart thuần): tải model từ Kho, tiếp tục bằng `Range` khi đứt (tự thử lại ≤ 4 lần, lùi 1s/2s/4s hoặc `Retry-After`) hoặc khi gọi lại sau khi bấm dừng; file dở tên `hub_<16 ký tự đầu sha256>.part` nên bản khác không nối nhầm; chuyển hướng xử lý bằng tay (https, ≤ 5 lần), **token chỉ gửi tới máy chủ gốc**; 416 hoặc máy chủ bỏ qua Range → tải lại từ đầu (không nối bừa); trần dung lượng (tham số `maxBytes`, tối đa 2 GB) từ chối **trước khi tải**; nhận nhiều byte hơn khai báo → dừng + xoá; SHA-256 thuần Dart tính dần khi tải; chỉ đổi tên thành `.yui` sau khi đủ byte + đúng sha256 + header đọc được + **cấu hình trong file khớp manifest** (`arch_id`: d/l/h/e, ctx, phiên bản, và `arch_id` trong META với v3). Hỏng vì mạng → giữ file dở; hỏng vì nội dung → xoá. `state.dart` chỉ **thêm** 41 dòng (`ModelStore.downloadHub`, `AppState.downloadHubModel`), `download()` link tay không đổi. **Kiểm thật (Dart SDK 3.14, máy chủ cục bộ):** `dart analyze` 0 lỗi cho 2 file mới; `tools/hub_download_check.dart` **145 kiểm tra đạt**, lặp lại 2 lần ổn định; SHA-256 khớp Python `hashlib` ở 15 độ dài (kể cả biên 55/56/57/63/64/65 byte, cắt mảnh 1–65 byte) và 3 vector chuẩn (kể cả 1 triệu chữ a); **thử đột biến**: bỏ kiểm sha, bỏ Range, bỏ kiểm cấu hình, gửi token sang mọi máy chủ, bỏ trần dung lượng → mỗi lỗi đều làm bộ kiểm báo HỎNG (bộ kiểm không "xanh giả"). **Chưa kiểm:** (a) build Flutter/Android thật cho `state.dart` (mình chỉ phân tích với các gói Flutter bị thiếu: ngoài các lỗi `notifyListeners` giống hệt code cũ do thiếu `ChangeNotifier`, không có lỗi nào ở phần mới); (b) tải thật từ Hugging Face (xem điểm lạ `174230` ở P2-1); (c) chưa có giao diện gọi `downloadHubModel` (P2-6). **Hạn chế đã biết:** SHA-256 chạy trên luồng giao diện, ≈ 1 giây/100 MB trên máy chủ của mình, điện thoại có thể chậm 3–5 lần; tải lại từ file dở lớn (vài trăm MB) phải băm lại phần đã có (vài giây); không kiểm dung lượng trống của máy (Dart không có API chuẩn) nên hết bộ nhớ chỉ báo lỗi ghi file và giữ file dở. Nếu thấy giật thì chuyển băm sang isolate (việc nhỏ, làm khi có số đo trên máy thật) |

**Câu hỏi mở (trả lời khi tới task liên quan):**
- Kho mới có 1 model (`yui-q4`, chat+general). Muốn chọn model theo ngành (P5) thì cần có model `code`/`math`/`translate`… huấn luyện và xuất trước; việc này nằm ngoài P2 (hiện chưa có task) → **quyết khi tới P2-6 (giao diện cần trạng thái "chưa có model cho ngành này") và trước P5**.
1. ~~P0-2: xoá hẳn `provider-chain.patch` hay lưu vào `docs/archive/`?~~ → **đã chốt: lưu `docs/archive/`**
2. ~~P1-1: META có cần trường `license` bắt buộc không?~~ → **đã chốt: bắt buộc có khoá, cho `"unknown"`**
3. ~~P2: repo model mặc định là repo nào, public hay private?~~ → **đã chốt: `BachDaThan/Yui-Models`, public**
4. P3-6: có cần mã hoá kho ngay bản đầu không?
5. P4-3: danh sách dataset cho phép ban đầu gồm những dataset nào?
