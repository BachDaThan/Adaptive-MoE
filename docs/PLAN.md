# KẾ HOẠCH NÂNG CẤP Y-App + Adaptive-MoE

> File này là **bộ nhớ chung** giữa các cuộc chat. Mở chat mới chỉ cần gửi file này + nói "làm tiếp".
> Đặt tại: `Adaptive-MoE/docs/PLAN.md` (một bản duy nhất, cả 2 repo dùng chung). Kế hoạch của hệ điều hành ở file riêng `YuiVieOSBeta/docs/PLAN.md` (xem mục 6).

**Cập nhật:** 2026-11-17 · **Phiên bản kế hoạch:** v3 · **Tiến độ:** 18/71 (P0, P1 xong; P2: P2-1…P2-6 xong; P8-1, P8-6 xong; P8-7 `[~]` chờ bạn chạy notebook; kế tiếp P2-7). **v2 thêm 18 task chưa làm (P5-6, P9-1…P9-17); v3 thêm 1 task (P9-18) và ghi quyết định Q6 = C++17 (P3-2, P3-3 đánh `[-]`)**; không đổi/bỏ task nào khác; tiến độ thật vẫn 18 task xong

## ▶ TIẾP THEO (cập nhật 2026-11-17, để mở chat mới làm tiếp)
1. **Bạn quyết "Chế độ Hybrid" nghĩa là gì** (Câu hỏi mở số 4 ở mục dưới; mình đã đề xuất: công tắc tổng, mặc định tắt, kèm ô Hugging Face token). Xong thì làm `P2-7`, sau đó `P2-8` (cập nhật chính sách quyền riêng tư). *(Không đổi so với 2026-11-15; vẫn là việc đứng đầu.)*
2. **Chạy notebook `yui_kaggle_4_cells.md` v6.6 với `PROFILE="thu"`** để có model đầu tiên (đặt `MODEL_LICENSE` sau khi đọc điều khoản data/teacher). Có model thì mới đo được P9-5, P9-6 và thử cả chuỗi: tải từ Kho → nạp → hỏi Yui một câu cần tra web.
3. **Việc mới, không cần chờ quyết định nào, làm song song được** (từ đợt tư vấn 2026-11-16, xem mục 1b): `P9-4` (băm SHA-256 bằng lệnh phần cứng), `P9-8` (fuzz bộ đọc `.yui`), `P9-1` (đo mật độ số 0 của model t2), `P9-16` (đo tốc độ engine trên CPU không có AVX2). Cả bốn đều nhỏ, kiểm được ở máy mình (trừ phần ARM của P9-4), và không đổi hành vi nào của app.
4. **Q6 đã chốt: C++17 (bạn trả lời 2026-11-16)** ⇒ `P3-2`, `P3-3` đánh `[-]` (thay bằng `P9-13`). Làm `P3-1` (đặc tả, gộp `P9-12`) và `P3-7` (bộ test vàng sinh từ Python) trước, rồi `P9-13`, `P9-14`. **Còn 4 câu mở (Q7–Q10)**; hai câu cần trả lời sớm là **Q8** (cách ký manifest; mình nghiêng về đường C++ vì Q6 đã là C++) và **Q9** (có cho thêm cảm biến pin/nhiệt không).
5. Chưa xác nhận (giữ từ trước): tải `yui-q4` từ tab Kho và nạp được chưa; nút "Thử tìm web" với key/SearXNG thật.
6. Sau đó: P3 (theo Q6) rồi P8-3…P8-5 (công cụ web `fetch`, chờ bạn duyệt đặc tả).
7. **Về hệ điều hành:** kế hoạch của `YuiVieOSBeta` ở file riêng `YuiVieOSBeta/docs/PLAN.md` (nâng cấp 2026-11-17: +12 task, §7.7 **chỉ dẫn cho tác tử AI thi công P5**, §13 C30–C41). Phase P5 của OS vẫn **khoá** cho tới khi bạn nói đúng câu "Tích hợp AI vào hệ điều hành"; mình chưa kích hoạt và chưa kéo mã nào vào repo OS. **Phía repo này chuẩn bị:** `P9-15` (cổng gắn), `P9-8` (fuzz) là điều kiện vào; `P9-18` (công cụ web qua broker) nếu làm P5.17 của OS; `P9-4`, `P9-7` khuyến nghị.

**Mở chat mới:** gửi file này (`PLAN.md`) là đủ; mình tự lấy code từ hai repo public `BachDaThan/Y-App` và `BachDaThan/Adaptive-MoE`, chỉ xin thêm file khi một task thật sự cần. Báo mình: "làm tiếp từ mục ▶ TIẾP THEO".

Ký hiệu: `[ ]` chưa làm · `[~]` mình đã giao file, chờ bạn up/thử · `[x]` xong (chỉ đánh khi bạn báo OK) · `[!]` bị chặn (xem ghi chú) · `[-]` bỏ (có lý do trong Nhật ký)

| Giai đoạn | Nội dung | Xong |
|---|---|---|
| P0 | Dọn nền, đo mốc ban đầu | 5/5 |
| P1 | `.yui` v3: có meta + mục lục | 5/5 |
| P2 | Kho model (Model Hub) trong Y-App | 6/8 |
| P3 | Kho tri thức ngay trong app (**lõi C++17 dùng chung**; P3-2, P3-3 bỏ `[-]`, xem P9-13) | 0/7 |
| P4 | Tra Dataset đám mây (Cloud-Edge RAG) | 0/6 |
| P5 | Mượn model chuyên theo câu hỏi (cấp model) + thí nghiệm adapter | 0/6 |
| P6 | Nghiên cứu ghép Expert (chưa cam kết) | 0/5 |
| P7 | Phát hành + cập nhật tài liệu | 0/4 |
| P8 | Phát sinh: sửa cài APK, tìm web nhiều nguồn, notebook huấn luyện, công cụ web (`fetch`) | 2/7 |
| P9 | **Mới (2026-11-16, chờ bạn duyệt; v3 thêm P9-18):** nâng cấp engine (băm phần cứng, khoá cú pháp công cụ, CAD, BITCOS), an toàn model, tiết kiệm pin, cổng nối hệ điều hành, công cụ web qua broker | 0/18 |

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

## 1b. ĐÁNH GIÁ ĐỢT TƯ VẤN THỨ HAI (2026-11-16)

Gồm: 4 gợi ý "nâng cấp" đầu của Gemini (BITCOS, LoRA-MoE, CAD, SHA-256), bản Gemini phân tích Y-App + Adaptive-MoE, hai bản phân tích hợp nhất 3 dự án, và chuỗi tư vấn về hệ điều hành (shell POSIX, phần cứng thật, game, cập nhật A/B, phát hành ISO). Mình đối chiếu với **code thật của cả 3 repo (clone 2026-11-16)** và, với BITCOS, với nguồn công khai (arXiv 2609.16338 và bài The New Stack 2026-09-17). Phần chỉ liên quan hệ điều hành nằm ở `YuiVieOSBeta/docs/PLAN.md` §13; ở đây chỉ giữ cái chạm tới 2 repo này.

**Cách đọc cột "Kiểm":** `đã kiểm` = mình tự đối chiếu code/nguồn; `tính tay` = mình tự tính, chưa đo; `chưa xác minh` = không tìm được nguồn hoặc không kiểm được, **đừng dựa vào**.

| # | Ý được đề xuất | Đối chiếu thật | Quyết định |
|---|---|---|---|
| B1 | **BITCOS** thay cách đóng gói t2 (Gemini 1) | **Đã kiểm, có thật:** arXiv 2609.16338 (Intel, preprint 14/9/2026, chưa phản biện). Công thức: **2 − z bit/trọng số** (z = tỉ lệ số 0), lossless (giải ra đúng −1/0/+1). **1.485 là kết quả của checkpoint thưa nhất** trong 29 cái (z từ 29,7% đến 51,5%), không phải mức điển hình. Gemini sai/phóng đại ở 3 chỗ: (1) "số 0 chiếm 51,5%" là mức cao nhất đo được; (2) "nhanh hơn 1,28× trên CPU": nguồn tin ghi tới 18% trên CPU và 27% trên GPU Intel Xe2, toàn bộ trên phần cứng Intel, **chưa thử Arm**, và trên CPU Lunar Lake kernel 2-bit cố định còn **nhanh hơn** BITCOS; (3) "570 → 420 MB" không suy ra được từ công thức vì chỉ phần ma trận 2-bit nhỏ đi (tối đa z/2), scale/embedding/router giữ nguyên; model 1B BitNet của bạn chưa train (P8-7 cần 8–24 phiên) nên chưa có z thật. | **Giữ, làm thí nghiệm lưu trữ/tải** (P9-1 → P9-2 → P9-3). **Không hứa tăng tốc.** NEON không có lệnh `pdep` như AVX-512 nên giải mã dòng dấu nén trên điện thoại khó nhanh; vì vậy đề xuất dùng như định dạng truyền tải, giữ 2-bit + mmap lúc chạy |
| B2 | **LoRA-MoE / adapter** thay P5/P6 (Gemini 2) | Hướng đúng và an toàn hơn P6-2 vì **cùng gốc**. Nhưng "mục lục v3 đã sẵn sàng nạp đè theo tên tensor" chỉ đúng một nửa: mục lục cho biết **vị trí**, engine hiện **không có đường nào cộng delta LoRA** (và gộp vào trọng số q4 thì phải lượng tử lại, mất chính xác ⇒ phải cộng song song lúc chạy). Kích thước "1,5–3 MB": **tính tay** cho profile hiện tại (d=256, 6 lớp, rank 16 trên qkv+o): r·(4D+2D)·6 ≈ 0,15 triệu tham số ≈ 0,3 MB fp16, nhỏ hơn Gemini nói. Tên "Route-Aware Adapter" **chưa xác minh được nguồn**. | **Thêm P5-6** (thí nghiệm Python trước, Go/No-Go so với mượn model nguyên và với lõi+RAG). Không đổi P5-3 |
| B3 | **Context-Aware Decoding (CAD)** (Gemini 3) | Kỹ thuật có thật (theo hiểu biết của mình là bài 2023 của Shi và cộng sự; chưa tra lại), công thức của Gemini tương đương (1+α)·ctx − α·base. Gemini bỏ sót: (1) mỗi token chạy **2 lượt forward + 2 KV cache** ⇒ tốc độ giảm gần một nửa (mốc 187–240 tok/s ⇒ khoảng 100–120, **tính tay**); (2) bản gốc có ràng buộc "chỉ xét token có xác suất ≥ β·max" (thiếu nó dễ ra chữ vô nghĩa); (3) chưa có bằng chứng cho model 50M; (4) "không tốn một giây huấn luyện" đúng, nhưng tốn tốc độ chạy. | **Thêm P9-6**, tuỳ chọn mặc định TẮT, chỉ bật nếu đo thật có lợi |
| B4 | **SHA-256 bằng lệnh phần cứng** (Gemini 4) | Đúng, rẻ, khớp hạn chế bạn tự ghi ở P2-4 (≈ 1 s/100 MB trên máy chủ, điện thoại chậm 3–5×). "Dưới 0,3 s cho 500 MB" hợp lý nếu có lệnh SHA2 phần cứng nhưng **sẽ bị giới hạn bởi tốc độ đọc đĩa**: cần đo trên điện thoại. Thêm: băm **sau khi tải xong** (thay vì vừa tải vừa băm bằng Dart) còn bỏ luôn việc băm lại phần dở khi tải tiếp. | **Thêm P9-4, làm sớm** |
| B5 | Nối **Elastic k/W** với pin và nhiệt (Gemini bản 2, ý 2) | Engine **đã có `kcap`** (giới hạn k theo lần gọi). **Chưa có cửa sổ trượt W** (KV cố định trong [0, L); `elastic_ctx.py` chỉ phía Python/huấn luyện). Phía OS đã có `yui_telemetry_advise` (471 check) đưa khuyến nghị k/W. Ở Y-App chưa có cảm biến pin/nhiệt (thêm gói = rủi ro build như P8-2). | **P9-9** (chính sách thuần Dart + công tắc tay), **P9-10** (cảm biến thật, cần bạn đồng ý), **P9-11** (W trong engine, ưu tiên thấp vì L hiện chỉ 128–256) |
| B6 | Model ML 5–10M tham số làm **bộ lập lịch qua `sched_ext`** (Gemini ý 3) | "09–10/2026 sched_ext đã hoàn thiện sub-scheduler": **chưa xác minh**. YuiVieOS dùng bộ lập lịch Rust tự viết (`yui-sched`), không dựa vào `sched_ext`; chưa có dữ liệu vết để huấn luyện; chạy suy luận trong đường lập lịch tăng độ trễ. | **Hoãn** (ghi ở OS §13) |
| B7 | Công cụ `sys_*` gắn vào `yui_agent.py` (Gemini ý 4) | OS đã có thiết kế đúng chỗ: 5 công cụ OS MCP nằm trong `yuid` (C++) với thẻ năng lực HMAC + xác nhận + audit. Cài thẳng `sys_power/sys_proc` trong Python sẽ **vòng qua lớp năng lực**. | `yui_agent.py` chỉ là **client** gọi `yuid` qua `/run/yuid.sock`; không tự cài công cụ hệ thống |
| B8 | **RISC-V RVV** + `libyui.a` gọi từ nhân Rust (Gemini ý 5) | riscv64 đã là target của OS; `yui.cpp` hiện có AVX2+FMA và NEON/dotprod, **còn lại là mã thuần** (đã kiểm: `detect_isa` chỉ chọn AVX2 hoặc scalar trên x86). RVV là tối ưu, không phải điều kiện chạy. Nhân Rust `no_std` **không link được** `libyui` (cần libc, luồng, mmap): engine chỉ chạy ở user-space qua `yuid`. | Chưa lập task; làm khi có QEMU `-cpu rv64,v=true` để đo |
| B9 | **Khoá cú pháp `<tool>` ngay trong vòng sinh** (bản phân tích #1) | Hay, rẻ. Nhưng ép **cú pháp**, không ép **lựa chọn đúng**: "chính xác tuyệt đối 100%" là overclaim (model vẫn có thể gọi sai công cụ/sai đối số, và vẫn cần dạy nó **khi nào** gọi: P8-5). Phải áp mặt nạ cả ở bước kiểm của đoán trước token (`YUI_SPEC`). | **Thêm P9-5** |
| B10 | **Chữ ký Ed25519 trong chân file `.yui`** (bản phân tích #2) | Mối đe doạ có thật nhưng nhỏ ở Y-App hiện tại: sha256/bytes/`arch_id` trong manifest chỉ bảo vệ khi **nguồn manifest đáng tin**, mà P2-6 cho người dùng **thêm nguồn tự chọn**. "Tiêm lệnh qua trọng số" không có cơ chế như mô tả; rủi ro thật là (1) lỗi bộ đọc file ⇒ **fuzz**, (2) model độc dùng công cụ: ở app chỉ có calc/time/search nên nhẹ, ở OS mới nặng và OS xử lý bằng thẻ năng lực **độc lập model**. **Đề xuất sửa:** chữ ký **tách rời trong manifest**, không đổi định dạng `.yui`; tái dùng Ed25519 đã qua vector RFC 8032 của YuiVieOSBeta; cùng một cách ký dùng được cho **bản cập nhật OS** (xem B17). | **Thêm P9-7** (chờ Q8) và **P9-8** (fuzz) |
| B11 | **Nén KV-cache bằng ZRAM / "Zero-reload"** (bản phân tích #3) | Với model hiện tại KV chỉ ≈ 0,8–1,6 MB (L=128, d=256, 6 lớp: 2·128·256·6 số; **tính tay**), nên nén vô nghĩa; thứ tốn khi đổi model là **nạp trọng số** (33 MB). KV của model A **không dùng được** cho model B (khác trọng số). ZRAM là tính năng của nhân, app Android không có. Chỉ có giá trị ở OS với model cỡ 7B. | **Không đưa vào kế hoạch app**; chuyển backlog OS. Giữ lõi trong RAM khi mượn model đã nằm trong P5-3 |
| B12 | **Graphify + BM25 + Vault thành thư viện C++17 dùng chung** (bản phân tích #4) | Lý do đúng (tránh duy trì 2 bản như `yui.cpp` từng bị: P0-1); C++ là thứ mình **chạy test được** (Dart/Flutter thì không); OS đã có bản tham chiếu đồ thị/vault ở `ai-bridge-yuid/reference/`. Cái giá: Graphify Python là mô-đun lớn (ứng viên → chấm sự kiện → thay cũ bằng mới → MMR), và phải sửa CI Y-App để biên dịch thêm file C++. **Bộ test vàng sinh từ Python (P3-1, P3-7) dùng chung được cho cả hai hướng nên làm trước.** | **Chờ Q6**; đã lập sẵn P9-12…P9-14 |
| B13 | **Tự chưng cất khi máy nhàn rỗi** (bản phân tích #5) | Phần "gom trí nhớ vào đồ thị, dọn nhánh trùng" làm được rẻ. Phần "tự tạo cặp SFT để fine-tune ngay trên máy 2 GB" **không khả thi** (huấn luyện cần RAM/năng lực gấp nhiều lần chạy). Rủi ro: dữ liệu web bị tiêm lệnh có thể **nhiễm vào trí nhớ dài hạn**. | Backlog OS, chưa lập task |
| B14 | **Chuyển Qwen 0.8B / Gemma 4 E2B / Phi-4-mini sang `.yui` bằng một "converter"** (Gemini, chuỗi cuối) | **Đánh giá thấp độ khó rất nhiều.** Engine chỉ chạy kiến trúc Adaptive-MoE: LayerNorm có bias, qkv gộp, router theo từng lớp, **`F = 2·D` cố định** (đã kiểm ở Sổ tay 4.1). Qwen/Gemma/Phi là họ dense kiểu Llama (RMSNorm, RoPE, GQA, SwiGLU, từ vựng ~150k, tokenizer khác): không phải việc của một script chuyển trọng số, mà là **thêm kiến trúc mới vào `yui.cpp`** (kèm kiểm đối chiếu PyTorch như P1-5). Tên/cỡ model Gemini nêu **chưa xác minh**; giấy phép từng model phải đọc. | **Không làm "converter".** Lập P9-17: khảo sát 3 đường (A chưng cất vào kiến trúc của bạn, đã có pipeline; B thêm kiến trúc dense vào engine, việc lớn; C backend GGUF thứ hai, ít việc nhất) rồi **bạn chọn** (Q11) |
| B15 | "Máy yếu đời cũ": HP Compaq 800/8200 (Gemini) | Haswell (đời 4, ví dụ 800 G1) **có** AVX2+FMA ⇒ engine dùng đường nhanh. Sandy Bridge (đời 2, ví dụ 8200) **không có AVX2** ⇒ engine rơi xuống **mã thuần (`scalar`)**, chưa có đường SSE/AVX1. Chưa ai đo tốc độ đó. | **Thêm P9-16** (đo; chỉ viết đường SSE4.1/AVX1 nếu số đo cho thấy không đủ dùng) |
| B16 | Shell POSIX lai AI, driver thật, TPM/Secure Boot, PQC, game/upscaler, A/B update, ISO bằng GitHub Actions | Thuộc hệ điều hành. Đã đối chiếu từng ý với `YuiVieOSBeta` (mã + kế hoạch): xem `YuiVieOSBeta/docs/PLAN.md` §13 (Đánh giá) và §14 (Phase P6 + backlog). Con số "độ hoàn thiện 45%/75%/90%/100%" và "sẵn sàng 50%/40%/10%/15%" của Gemini **không có phương pháp đo**, bỏ. | Xem file OS |
| B17 | Ký manifest bản cập nhật OS bằng Ed25519, chung định dạng `make_manifest.py` (Gemini, chuỗi cuối) | Ý hay: **một cách ký cho cả Kho model lẫn bản cập nhật OS.** Lưu ý an ninh: Gemini đề xuất để khoá ký trong GitHub Secrets; kế hoạch OS (P4.7) nói khoá **ngoại tuyến**. Với một hệ điều hành đặt nặng bảo mật nên giữ khoá ngoại tuyến (xem OS §13). | P9-7 thiết kế **tổng quát** (không gắn cứng với model) để OS dùng lại |

**Việc mình đã tự kiểm lần này (để bạn khỏi nghi):** clone 3 repo; `docs/PLAN.md` trên GitHub **trùng từng byte** file bạn gửi; biên dịch `yui.cpp` thành thư viện và liệt kê hàm xuất (**17 hàm** `yui_*`); tìm trong `yui.cpp` các từ khoá sha256/riscv/rvv/grammar/cad/sink: **không có**; đọc `detect_isa`; đọc `ARCHITECTURE`/`PLAN` và C-ABI của OS. **Chưa làm được:** chạy Flutter, chạy trên điện thoại, đo tốc độ trên CPU cụ thể.

---

## 2. NGUYÊN TẮC THIẾT KẾ

1. **Offline-first:** không mạng thì app chạy y như bây giờ. Mọi thứ mới chỉ là "tăng cường".
2. **Cấp model trước, cấp expert sau (nếu có bằng chứng):** mượn **cả model nhỏ chuyên ngành** (vài chục MB, q4/t2) thay vì từng expert.
3. **Model là "khách trọ" (stateless):** bàn giao bằng văn bản nén (Graphify), không chuyển KV cache.
4. **Ngân sách ngữ cảnh cứng:** model nhỏ có L = 128–256 token; mọi thứ nhét thêm vào prompt phải nén xuống vài chục–vài trăm ký tự.
5. **Riêng tư:** không gửi nguyên văn chat ra mạng; chỉ từ khoá trung tính, hiện cho người dùng xem trước.
6. **Tương thích ngược:** engine mới đọc v1/v2/v3; engine cũ gặp v3 phải báo lỗi gọn (đã có sẵn `die("phiên bản .yui không hỗ trợ")`).
7. **Mỗi bước nhỏ, kiểm được, hoàn tác được.**
8. **Đo trước, cam kết sau:** con số từ tư vấn bên ngoài (BITCOS, CAD, tốc độ băm…) phải có ô "đo trên máy bạn" và ngưỡng Go/No-Go **trước khi** vào README/app. Tính năng engine mới đều **mặc định TẮT** và khi tắt phải cho đầu ra **y hệt** bản cũ (so từng token).
9. **Cú pháp công cụ có thể ép bằng engine; quyền công cụ do lớp hệ thống quyết, không do model.** Dù model tin cậy đến đâu, nội dung web/tệp có thể tiêm lệnh vào nó; vì vậy quyền gọi `sys_*` luôn đến từ thẻ năng lực của `yuid`, không từ việc model "muốn" gọi.

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

- [x] **P2-4 · Tải model có tiếp tục (Range), kiểm sha256, giới hạn dung lượng, kiểm `arch_id`**
  ✎ nâng `ModelStore.download` trong `lib/state.dart` · 📥 `lib/state.dart` · 📤 `Y-App/lib/state.dart`
  ✅ Ngắt mạng giữa chừng rồi tải tiếp được; sai sha256 thì xoá file; file quá trần dung lượng bị từ chối.
  📤 đã giao: `Y-App/lib/state.dart` (thêm `ModelStore.downloadHub`, `AppState.downloadHubModel`; `download()` cũ giữ nguyên), `Y-App/lib/services/hub_download.dart` (mới), `Y-App/tools/hub_download_check.dart` (mới)

- [x] **P2-5 · Bộ nhớ đệm đĩa LRU cho model/expert tải về**
  ✎ `lib/services/hub_service.dart`, `lib/state.dart` · 📥 hai file đó (bản mới nhất) · 📤 hai file ở `Y-App/lib/…`
  ✅ Vượt trần (mặc định 500 MB, chỉnh được) thì xoá model ít dùng nhất; không bao giờ xoá model đang chạy hoặc model người dùng tự nhập.
  📤 đã giao: `Y-App/lib/services/hub_cache.dart` (mới; thay cho việc sửa `hub_service.dart` như dự kiến, vì tách riêng dễ kiểm hơn), `Y-App/lib/state.dart`, `Y-App/tools/hub_cache_check.dart`, `Y-App/tools/hub_flow_check.dart` (mới)

- [x] **P2-6 · Giao diện tab Kho + thêm/bớt nguồn**
  ✎ tạo `lib/screens/hub_screen.dart`, sửa `lib/main.dart` · 📥 `lib/main.dart`, `lib/screens/models_screen.dart` · 📤 `Y-App/lib/screens/hub_screen.dart`, `Y-App/lib/main.dart`
  ✅ Xem danh sách, tải, xoá, thêm nguồn bằng link; cảnh báo khi `arch_id` lạ.
  📤 đã giao (khác dự kiến: thêm sửa `state.dart`, `hub_service.dart`, `hub_download.dart`, `models_screen.dart` và 2 file kiểm): `Y-App/lib/screens/hub_screen.dart` (mới), `Y-App/lib/main.dart`, `Y-App/lib/state.dart`, `Y-App/lib/screens/models_screen.dart`, `Y-App/lib/services/hub_service.dart`, `Y-App/lib/services/hub_download.dart`, `Y-App/tools/hub_service_check.dart`, `Y-App/tools/hub_download_check.dart`, `Adaptive-MoE/docs/PLAN.md`
  📤 bản sửa lỗi quét Kho (2026-11-15): `Y-App/lib/services/hub_service.dart`, `Y-App/tools/hub_service_check.dart`

- [ ] **P2-7 · Cài đặt "Chế độ Hybrid" (mặc định TẮT) + lưu HF token tuỳ chọn**
  ✎ `lib/state.dart` (Settings), `lib/screens/settings_screen.dart` · 📥 hai file đó · 📤 hai file ở `Y-App/lib/…`
  ✅ Tắt Hybrid thì app không gọi thêm máy chủ nào ngoài các mục đã có trong chính sách; token lưu trên máy.

- [ ] **P2-8 · Cập nhật pháp lý/quyền riêng tư cho tính năng mới**
  ✎ `assets/legal/privacy_vi.md`, `privacy_en.md`, `terms_*.md`, `legal/LEGAL_CHECKLIST.md`, `legal/THIRD_PARTY_NOTICES.md` · 📥 5 file đó · 📤 5 file ở `Y-App/…`
  ✅ Mục 3 "Dữ liệu gửi ra ngoài" liệt kê đúng: manifest, tải model, (P4) truy vấn dataset. Bản mẫu vẫn cần luật sư rà soát (như chính văn bản đã ghi).

### P3 — Kho tri thức ngay trong app (port gọn RAG/Graphify sang Dart)

Lý do: Cloud-Edge RAG cần "tra kho nội bộ trước" nhưng app hiện **không có** kho nào. Port bản **gọn**, tương thích định dạng thư mục của `yui_vault.py` để đổi máy chép là chạy.

> **✔ Q6 đã chốt (2026-11-16): C++17 lõi chung** (P9-12…P9-14), dùng cho cả Y-App (qua FFI) và `yuid` của hệ điều hành. Hệ quả: **P3-2 và P3-3 đánh `[-]`** (thay bằng P9-13); phần Dart của P3-4/P3-5 chỉ còn là lớp nối FFI (P9-14); **P3-1 (đặc tả) và P3-7 (bộ test vàng sinh từ Python) làm trước**, dùng chung cho Python, C++ và (nếu cần) Dart. Phải sửa CI Y-App để biên dịch thêm file C++ (nằm trong P9-14). Các mô tả P3-2/P3-3 bên dưới giữ nguyên để tham khảo.

- [ ] **P3-1 · Đặc tả kho trong app (thư mục `chat/ notes/ kb/ index/`, định dạng jsonl)**
  ✎ `docs/APP_KB.md` · 📥 `yui_vault.py` · 📤 `Adaptive-MoE/docs/APP_KB.md`
  ✅ Bạn duyệt; chốt có mã hoá ở bản đầu hay để P3-6.

- [-] **P3-2 · Dart: tách từ + BM25 tiếng Việt không dấu (port gọn `yui_rag.py`)**  ← **Bỏ 2026-11-16 theo Q6 (C++17): thay bằng P9-13.**
  ✎ tạo `lib/kb/bm25.dart`, `lib/kb/vi_text.dart` · 📥 `yui_rag.py` (hàm `fold`, `tokens`, `split_text`, lớp `RAG`) · 📤 hai file ở `Y-App/lib/kb/`
  ✅ Tập test chéo: cùng truy vấn, Dart và Python xếp hạng giống nhau trên dữ liệu mẫu ("ten meo" ra "tên mèo").

- [-] **P3-3 · Dart: Graphify-lite (ứng viên rộng → chấm theo sự kiện → thay cũ bằng mới → MMR → câu thường)**  ← **Bỏ 2026-11-16 theo Q6 (C++17): thay bằng P9-13.**
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

- [ ] **P5-6 · Thí nghiệm adapter chuyên ngành cùng gốc (LoRA), thay thế/bổ sung P5-3 nếu Go** *(đề xuất 2026-11-16, mục B2)*
  ✎ `exp_lora_domain.py` (adapter rank 8/16 trên qkv+o của checkpoint nền; huấn luyện trên một ngành nhỏ; đo loss/perplexity ngành đó và ngành chung) · 📥 `train_sft.py`, `adaptive_moe.py` · 📤 `Adaptive-MoE/exp_lora_domain.py`
  ✅ Python/Kaggle thôi, không đụng `yui.cpp`/app. **Go (đề xuất)** khi adapter ≤ 1 MB đạt ≥ 90% mức cải thiện của model chuyên ngành nguyên (cùng bộ câu của P5-5) VÀ ngành chung xấu đi ≤ 2%. Nếu Go mới lập task mới: định dạng adapter riêng (magic `YUIA`, ràng buộc `tokenizer_sha256` + `arch_id` + hash checkpoint nền để không áp nhầm; engine **cộng song song** `x·Aᵀ·Bᵀ` ở f32, không gộp vào q4). Nếu No-Go đánh `[-]` và ghi lý do. Cần có model đầu tiên (P8-7) trước.

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

### P8 — Việc phát sinh (bạn nêu giữa chừng, 2026-11-13)

- [x] **P8-1 · Sửa "tải ngầm xong mà không cài được": khai báo quyền `REQUEST_INSTALL_PACKAGES`**
  Nguyên nhân (đã đối chiếu code + tài liệu gói): CI sinh `AndroidManifest.xml` mới chỉ thêm `INTERNET` và `RECORD_AUDIO`; gói `open_filex` (mở APK trong `update_service.dart`) **cố ý bỏ** quyền này. Android 8+ chỉ cho cài từ app và chỉ hiện mục "Cài app không rõ nguồn" cho app **đã khai báo** quyền ⇒ trình cài mở ra nhưng bị chặn.
  ✎ `.github/workflows/build-app.yml` (thêm 1 dòng `sed` + 2 dòng ghi chú, YAML đã kiểm hợp lệ, đã chạy thử đúng lệnh `sed` trên manifest mẫu) · 📤 `Y-App/.github/workflows/build-app.yml`
  ✅ Build xong → **cài tay APK mới một lần** (bản đang cài không có quyền nên không tự cài đè được chính nó) → vào app tải ngầm một bản cập nhật bất kỳ → bấm Cài đặt: Android hiện nút/trang cho phép "cài app không rõ nguồn" cho Yui (nếu chưa bật) rồi cài được. **Chưa kiểm trên máy thật** (mình không chạy được Android), chờ bạn báo.
- [ ] **P8-2 · (tuỳ chọn) Nút "Cấp quyền cài app" trong Cài đặt của app**: kiểm đã được phép chưa và mở thẳng trang cấp quyền, thay cho dòng chữ báo lỗi hiện có (`UpdateService.install`). Cần thêm gói (vd `permission_handler`) hoặc viết MethodChannel trong CI (vì `MainActivity` do `flutter create` sinh) ⇒ có rủi ro build, **chỉ làm nếu P8-1 chưa đủ**.
- [ ] **P8-3 · Đặc tả công cụ web `fetch` (đọc/cào trang) cho app + model** · 📥 `lib/agent.dart` (đã có), `yui_agent.py` (`tool_fetch`/`safe_http`, đã đọc) · 📤 `Adaptive-MoE/docs/APP_WEB_TOOL.md`
  Hiện trạng thật: app **cố ý** chỉ có `calc`, `time`, `search` (Wikipedia vi); đầu `agent.dart` ghi rõ `fetch/read/write/py` không đưa vào app điện thoại. Model: dữ liệu huấn luyện `tool_data*.py` **chưa có** `fetch` (chỉ calc/time/search/py/recall/note/skill/mkskill/chain) nên model hiện tại sẽ không biết gọi.
  Đặc tả phải chốt: (a) chỉ https; chặn localhost/IP riêng/nội bộ ở **mỗi bước chuyển hướng** (như `safe_http` của `yui_agent.py`: ghim IP đã kiểm tra để chống DNS rebinding; ≤ 3 chuyển hướng); (b) tối đa 1 trang/câu hỏi, ≤ 256 KB, thời gian chờ, chỉ nhận `text/*`/json/xml, bỏ `script/style`, tách chữ; (c) nội dung trang là **dữ liệu không tin cậy** (không được coi là lệnh); (d) công tắc riêng, **mặc định TẮT**, có thể hỏi xác nhận mỗi lần như profile `open` (`confirm=[py, fetch]`); (e) **kết quả trả lại cho model chỉ ≈ 120 ký tự** (`kToolResultChars`, do ngữ cảnh 128 token) ⇒ cào cả trang gần như vô dụng nếu thiếu bước chọn đoạn liên quan (BM25 ở P3) → quyết định làm sau P3 hay làm bản tối giản "trích đoạn quanh từ khoá"; (f) model phải được dạy `<tool>fetch URL</tool>` (thêm kind vào `tool_data_x.py`) rồi **huấn luyện lại** (việc của bạn trên Colab/Kaggle).
  ✅ Bạn duyệt đặc tả và chốt thứ tự (trước hay sau P3).
- [ ] **P8-4 · `lib/services/web_fetch.dart` + `tools/web_fetch_check.dart`** (Dart thuần, kiểm bằng máy chủ cục bộ + đột biến như các mục P2; **chưa nối vào app**) · ✅ bộ kiểm xanh gồm SSRF (localhost, 10.x, 192.168.x, 169.254.x, IPv6 nội bộ, chuyển hướng sang IP nội bộ), quá lớn, sai loại nội dung, quá nhiều chuyển hướng
- [ ] **P8-5 · Dữ liệu huấn luyện `fetch` + nối `AgentTools` + công tắc Cài đặt + README** · ✎ `tool_data_x.py`, `lib/agent.dart`, `lib/screens/settings_screen.dart`, `lib/state.dart` · ✅ build xanh; **model chỉ dùng được sau khi bạn huấn luyện lại** (mình không huấn luyện được ở đây)

- [x] **P8-6 · Tìm web nhiều tầng trong app (điện thoại + Windows), không chỉ Wikipedia** · ✎ `lib/agent.dart`, `lib/state.dart`, `lib/screens/settings_screen.dart` · 📤 `Y-App/lib/services/web_search.dart` (mới), `Y-App/tools/web_search_check.dart` (mới), `Y-App/lib/agent.dart`, `Y-App/lib/state.dart`, `Y-App/lib/screens/settings_screen.dart`, `Adaptive-MoE/docs/PLAN.md`
  Port gọn `yui_web.py` sang Dart: công cụ `search` thử lần lượt **API có key của bạn** (tavily, exa, linkup, firecrawl, braveapi, gcse; nhiều key/dịch vụ, xoay vòng, key hỏng/hết hạn mức tự nghỉ) → **DuckDuckGo** (không cần key) → **SearXNG** (các link https bạn nhập) → **Wikipedia** (vi rồi en). Cú pháp model gọi vẫn `<tool>search …</tool>` nên **không cần huấn luyện lại**. Trong Cài đặt: công tắc "Tìm web nhiều nguồn" (mặc định BẬT; tắt = chỉ Wikipedia như cũ) + mục "Key và SearXNG (nâng cao)" để dán key (mỗi dòng `tên=key1,key2`).
  ✅ Build xanh; hỏi Yui một câu cần tra web thì có trả lời (không còn chỉ Wikipedia); thử thêm khi tắt mạng/tắt công tắc.

- [~] **P8-7 · Notebook huấn luyện v6.6 (chỉnh theo yêu cầu) + nối app khớp với model** · ✎ `yui_kaggle_4_cells.md`, `lib/agent.dart`, `lib/state.dart`, `lib/screens/settings_screen.dart` · 📤 `yui_kaggle_4_cells.md` (thư mục repo Kaggle của bạn), `Y-App/lib/agent.dart`, `Y-App/lib/state.dart`, `Y-App/lib/screens/settings_screen.dart`, `Y-App/tools/web_search_check.dart`
  (1) `PROFILE` ở Ô1: `thu` (50M, 3 giờ) để có model đầu tiên sớm; `chinh` (150M, 12 giờ); `lon` (1B + BitNet, nhiều phiên). (2) Ô4 xuất `.yui` v3 có META + `manifest.json` (gộp model đang có trên repo Kho), tự đẩy lên Kho chỉ khi `PUBLISH_TO_HUB = True` (mặc định tắt, không tự tạo repo). (3) Chống tạo nhầm model mới khi repo đã có checkpoint (`FORCE_NEW` để ép). (4) Batch lấy từ profile thay vì cố định B=2. (5) App: độ dài kết quả công cụ theo ngữ cảnh model đang nạp (`toolResultCharsForCtx`: ≤128 → 120, lớn hơn → 300, khớp `TOOL_RESULT_CHARS` lúc tạo data). (6) Cài đặt: nút **Thử tìm web** chạy công cụ tìm web ngay trên máy, không cần model.
  ✅ Chạy Ô1→Ô4 với `PROFILE="thu"` ra file trong `hub/`; up lên Kho; trong app tải và nạp được; hỏi một câu cần tra web thì Yui gọi công cụ.

### P9 — Nâng cấp engine, an toàn model, cổng nối hệ điều hành (đề xuất 2026-11-16)

Phát sinh từ đợt tư vấn thứ hai (mục 1b). **Mọi task đều `[ ]` và chờ bạn duyệt.** Quy tắc chung: (1) không task nào đổi hành vi file `.yui` v1/v2/v3 cũ; (2) tính năng engine mới **mặc định TẮT** và khi tắt thì đầu ra **y hệt** bản cũ; (3) task nào sửa `yui.cpp` phải up **cả 2 repo cùng nội dung** (`Adaptive-MoE/yui.cpp` và `Y-App/native/yui.cpp`; job `sync-yui` đỏ tạm giữa hai lần up, up xong bấm Re-run); (4) ngưỡng Go/No-Go ghi "đề xuất" là mình tự đặt, bạn chỉnh được.

**Nhóm A, làm ngay được, không chờ quyết định (nhỏ, an toàn):**

- [ ] **P9-1 · Đo mật độ số 0 của model t2** (nền để quyết BITCOS)
  ✎ `tools/t2_zero_stats.py` + `test_t2_zero_stats.py` (đọc `.yui` t2 bằng `yui_format.py` và `bitnet_pack.unpack_t2`; in z từng ma trận, z tổng, và số byte tiết kiệm ước tính ≈ z/2 trên phần 2-bit so với cả file) · 📥 `bitnet_pack.py`, `yui_format.py` · 📤 `Adaptive-MoE/tools/t2_zero_stats.py`, `Adaptive-MoE/test_t2_zero_stats.py`
  ✅ Test với ma trận tổng hợp có z biết trước ra đúng z; bạn chạy trên **một model t2 thật** (kể cả model nhỏ thử) và dán bảng. **Go P9-2 khi** file nhỏ đi ≥ 10% sau khi trừ phần không đổi (đề xuất).

- [ ] **P9-4 · `yui_sha256_file(path, out_hex)` bằng lệnh phần cứng (ARMv8 SHA2 / SHA-NI) + gọi qua FFI** *(mục B4)*
  ✎ `yui.cpp` (cả 2 repo), `Y-App/lib/engine.dart` (binding), `Y-App/lib/services/hub_download.dart` (gọi trong isolate nền, băm **sau khi tải xong**), `Y-App/tools/hub_download_check.dart` · 📥 `yui.cpp`, `lib/engine.dart`, `lib/services/hub_download.dart` · 📤 các file trên
  ✅ Khớp `hashlib` ở các độ dài biên (0, 1, 55, 56, 57, 63, 64, 65 byte; 1 triệu chữ `a`; file ≥ 100 MB) trên đường **thuần C** và (nếu máy mình có SHA-NI) đường SHA-NI. **Đường ARMv8 mình không chạy được** (không có phần cứng ở đây): chỉ kiểm trên CI/điện thoại của bạn, ghi rõ. Dò lệnh lúc chạy, không có thì dùng bản thuần C; không băm trên luồng giao diện; bạn đo 500 MB trên điện thoại (mốc hiện tại ≈ 15–20 s theo ghi chú P2-4 nhân 3–5). Giữ `Sha256` Dart làm dự phòng nếu FFI lỗi. **Chưa hứa con số "dưới 0,3 s"**: bị giới hạn bởi đọc đĩa.

- [ ] **P9-8 · Fuzz bộ đọc `.yui`** *(mục B10)*
  ✎ `tools/fuzz_yui.py` (+ test nhỏ cho CI): lấy file v1/v2/v3/t2 nhỏ, đột biến byte/cắt cụt/đổi độ dài/phá chân file `YUIX`, nạp bằng `yui.cpp` dựng với ASan+UBSan (g++) · 📥 `yui.cpp`, `yui_format.py` · 📤 `Adaptive-MoE/tools/fuzz_yui.py`, `Adaptive-MoE/test_fuzz_yui.py`, và `yui.cpp` (cả 2 repo) nếu tìm ra lỗi
  ✅ 20 000 đột biến: không crash, không lỗi sanitizer (file hỏng chỉ được báo lỗi gọn); lỗi nào tìm ra thì sửa kèm test hồi quy. Mình chạy được ở máy mình. Đây là cách **trực tiếp** nhất để trả lời nỗi lo "file `.yui` giả khai thác tràn bộ đệm".

- [ ] **P9-16 · Đo tốc độ engine trên CPU không có AVX2 (đường `scalar`)** *(mục B15)*
  ✎ `tools/bench_isa.py` (ép `isa` bằng biến môi trường hoặc cờ thử nghiệm, chạy cùng model q4 ở `scalar` và `avx2+fma`, in tok/s) · 📥 `yui.cpp` (hàm `detect_isa`) · 📤 `Adaptive-MoE/tools/bench_isa.py`
  ✅ Có bảng tok/s `scalar` so với `avx2+fma` trên máy mình, và **bạn chạy 1 lần trên máy thật không có AVX2** (ví dụ Sandy Bridge). **Chỉ khi** `scalar` < 20 tok/s trên model 50M q4 (đề xuất) thì lập task viết đường SSE4.1/AVX1; nếu không thì ghi `[-]`.

**Nhóm B, engine (mặc định tắt):**

- [ ] **P9-5 · Ràng buộc cú pháp `<tool>` ngay trong vòng sinh** *(mục B9)*
  ✎ `yui.cpp` (cả 2 repo) + `test_yui_kernels.cpp`; `opts_json` thêm `"grammar":"tool"` + danh sách tên công cụ được phép (lấy từ `AgentTools`); `Y-App/lib/agent.dart` truyền danh sách · 📥 `yui.cpp`, `lib/agent.dart`, `test_yui_kernels.cpp` · 📤 các file trên
  ✅ Tắt ⇒ đầu ra **y hệt** cũ (so từng token); bật ⇒ với logits ngẫu nhiên 10 000 lượt luôn ra chuỗi `<tool>tên đối số</tool>` hợp lệ hoặc không mở thẻ; chạy đúng khi đoán trước token bật; chậm thêm ≤ 5%. Ghi rõ trong tài liệu: chỉ bảo đảm **cú pháp**, không bảo đảm chọn đúng công cụ.

- [ ] **P9-6 · Context-aware decoding (CAD) tuỳ chọn** *(mục B3)*
  ✎ `yui.cpp` (cả 2 repo): `opts_json` thêm `cad_alpha` (mặc định 0) và `cad_beta` (mặc định 0,1; chỉ xét token có xác suất ≥ β·max), `bench_cad.py` (bộ ~50 câu tiếng Việt "đọc kết quả công cụ rồi trả lời" + chấm tự động xem câu trả lời có chứa đúng sự kiện không) · 📥 `yui.cpp`, `bench.py` · 📤 các file trên
  ✅ α=0 ⇒ y hệt cũ; bảng α ∈ {0; 0,25; 0,5; 1} × tỉ lệ bám đúng sự kiện × tốc độ. **Go (đề xuất):** tăng ≥ 10 điểm % số câu bám đúng kết quả công cụ và tốc độ ≥ 0,45× mốc; không thì đánh `[-]`. **Cần model thật (sau P8-7)** thì số đo mới có nghĩa.

- [ ] **P9-2 · (Nếu Go ở P9-1) Đặc tả `t2c` (BITCOS) trong `docs/YUI_FORMAT.md`** *(mục B1)*
  ✎ `docs/YUI_FORMAT.md`: kiểu ma trận mới = bitmap vị trí khác 0 + dòng dấu nén, scale giữ nguyên. **Chốt:** dùng làm **định dạng truyền tải** (Kho: nhỏ hơn; app giải về t2 2-bit sau khi tải, giữ mmap) hay **định dạng chạy** (engine giải lúc nạp, mất mmap cho phần t2). Mình đề xuất truyền tải trước. Manifest có thể cần thêm `sha256_local` nếu file lưu trên máy khác file trên Kho. · 📤 `Adaptive-MoE/docs/YUI_FORMAT.md`
  ✅ Bạn duyệt đặc tả.

- [ ] **P9-3 · (Nếu Go) Cài `t2c`**
  ✎ `bitnet_pack.py`/`export_yui.py` (`--quant t2c`), hàm chuyển t2c→t2 nằm trong `yui.cpp` (`yui_expand_t2c`), test · 📥 `bitnet_pack.py`, `export_yui.py`, `yui.cpp`, `yui_format.py` · 📤 các file trên (cả 2 repo cho `yui.cpp`)
  ✅ Chuyển ngược ra **đúng từng bit** t2 gốc (nạp ra logits `max|diff| = 0`); dung lượng khớp 2 − z; file v1/v2/v3 cũ không đổi. Tốc độ chỉ **đo**, không cam kết.

**Nhóm C, an toàn model và tiết kiệm pin:**

- [ ] **P9-7 · Chữ ký Ed25519 tách rời trong manifest** *(mục B10, B17; chờ Q8)*
  ✎ `make_manifest.py` (`--sign KEY`, trường `sig` = chữ ký trên chuỗi chuẩn hoá `sha256+bytes+arch_id`; thiết kế **tổng quát** để OS dùng lại cho bản cập nhật), `Y-App/lib/services/hub_service.dart` và `hub_download.dart` (kiểm), khoá công khai gắn trong app + cho người dùng thêm khoá của nguồn tự chọn; model không chữ ký vẫn tải được nhưng **cảnh báo rõ trên thẻ** · 📥 `make_manifest.py`, `hub_service.dart`, `hub_download.dart`; tái dùng mã Ed25519 đã qua RFC 8032 của YuiVieOSBeta (`kernel-services/ed25519.cpp`) nếu chọn đường C++ (thêm file vào build native ⇒ sửa `build-app.yml`); đường Dart thêm gói mật mã = rủi ro build · 📤 các file trên
  ✅ Vector RFC 8032 khớp; manifest sửa 1 byte ⇒ từ chối; khoá lạ ⇒ cảnh báo; bản không ký dùng được khi người dùng xác nhận. **Khoá bí mật không bao giờ commit lên repo**, giữ ngoại tuyến.

- [ ] **P9-9 · Chính sách "Tiết kiệm" cho k (theo pin/nhiệt) thuần Dart + công tắc tay** *(mục B5)*
  ✎ `lib/services/power_policy.dart` (mới, Dart thuần), `lib/state.dart`, `lib/screens/settings_screen.dart`, `tools/power_policy_check.dart`; dùng `kcap` sẵn có · 📥 `lib/state.dart`, `lib/screens/settings_screen.dart` · 📤 các file trên
  ✅ Bảng chân trị (pin < 20% **hoặc** nóng > 75 °C **hoặc** bật "Tiết kiệm" ⇒ k = 1; cắm sạc và mát ⇒ k tối đa của model) khớp ngưỡng của `ARCHITECTURE.md` §6.3 bên YuiVieOSBeta (cùng 20%/75 °C); đầu vào là tham số nên kiểm được bằng Dart SDK; chưa có cảm biến thì dùng công tắc tay.

- [ ] **P9-10 · Cảm biến pin/nhiệt thật (Android + Windows)** — **chỉ khi bạn đồng ý (Q9)**: cần thêm gói hoặc MethodChannel trong CI (rủi ro build như P8-2). Nối vào `power_policy.dart` của P9-9.

- [ ] **P9-11 · (ưu tiên thấp) Cửa sổ trượt W (sink + W) trong `yui.cpp`** — engine hiện giữ KV cố định trong [0, L); chỉ đáng làm nếu bạn huấn luyện model có L lớn hơn 256. 📥 `elastic_ctx.py`, `yui.cpp`. ✅ mô phỏng Python = engine từng logit.

**Nhóm D, kho tri thức C++ (Q6 = C++17 đã chốt 2026-11-16 ⇒ **làm**; thứ tự: P3-1/P9-12 → P3-7 → P9-13 → P9-14):**

- [ ] **P9-12 · Đặc tả lõi kho tri thức C++17 (`kb_core`)**: gộp với P3-1 (một đặc tả cho cả Python, C++ và Dart) · 📤 `Adaptive-MoE/docs/APP_KB.md`
- [ ] **P9-13 · `kb_core.cpp`: fold tiếng Việt + BM25 + Graphify** (thay P3-2/P3-3) · ✅ chạy đúng với bộ vàng sinh từ Python (P3-7), sai khác nhỏ được ghi lại; `make test` + pytest (ctypes) xanh; ngân sách ký tự được tôn trọng. Có thể tham khảo bản `fold` và đồ thị C++ đã có ở YuiVieOSBeta (`userspace-vi/text.cpp`, `ai-bridge-yuid/reference/graph.cpp`) nhưng **không giả định cùng định dạng**.
- [ ] **P9-14 · Nối FFI Dart + `recall`/`note`** · ✎ `Y-App/native/…`, `lib/kb/kb_ffi.dart`, `lib/agent.dart`, `build-app.yml` (biên dịch thêm file) · ✅ `[~]` tới khi bạn build.

**Nhóm E, cổng nối hệ điều hành và model lớn hơn:**

- [ ] **P9-15 · Cổng gắn hệ điều hành (phía engine)** *(mục 6)*
  ✎ `Makefile` (thêm mục tiêu `libyui_engine.so`/`.a`: đổi **tên file**, không đổi tên hàm, để không đụng `libyui` của OS), `yui.cpp` (cả 2 repo: thêm `yui_features()` trả JSON `{"sha256":…,"grammar":…,"cad":…,"abi":N}` để `yuid` biết engine có gì), `docs/OS_BRIDGE.md` · 📥 `Makefile`, `yui.cpp` · 📤 các file trên
  ✅ `nm -D` liệt kê đủ **17 hàm hiện có** + `yui_features`; test C nhỏ nạp **cả hai thư viện** (`libyui` của OS và `libyui_engine`) trong một tiến trình bằng `dlopen(RTLD_LOCAL)` mà không đụng nhau (2 tên đang trùng: `yui_version`, `yui_free`; **cùng chữ ký, khác ngữ nghĩa** nên lỗi sẽ âm thầm).

- [ ] **P9-17 · Khảo sát đường lên model lớn hơn** *(mục B14)*
  ✎ `docs/BIG_MODEL_PATHS.md`: so 3 đường (A chưng cất vào kiến trúc Adaptive-MoE; B thêm kiến trúc dense vào `yui.cpp`; C backend GGUF thứ hai) theo: việc phải viết, rủi ro, RAM/tốc độ ước lượng cho máy 2–4 GB, giấy phép từng model ứng viên (phải đọc điều khoản, không đoán) · 📤 `Adaptive-MoE/docs/BIG_MODEL_PATHS.md`
  ✅ Bạn chọn (Q11). Chưa viết dòng mã nào trước khi chọn.

- [ ] **P9-18 · `yui_agent.py`/`yui_web.py`: chế độ `broker` (không tự mở socket; mọi HTTP đi qua `yuid`)** *(đề xuất 2026-11-17, mục 6; cặp với OS P3.16 + P5.17)*
  ✎ `yui_web.py`, `yui_agent.py` (biến `YUI_NET=broker` + hàm vận chuyển cắm được), tạo `test_yui_broker.py` · 📥 `yui_web.py`, `yui_agent.py`, `test_yui_extra.py` · 📤 các file trên
  ✅ Mặc định **không đổi** (pytest không tụt dưới mốc 237 passed của P0-3). Ở `broker`: chưa có thẻ ⇒ công cụ `search`/`fetch` trả câu "chưa được cấp quyền mạng" (model không bị câm, không ném lỗi); có broker giả ⇒ chạy và đi đúng một đường; test chạy với socket bị chặn (vá `socket.socket`) chứng minh **không có kết nối trực tiếp**; nội dung trả về vẫn qua `sanitize` (thẻ `<tool>` bị vô hiệu). Giao thức thật với `yuid` (tên phương thức RPC) chỉ chốt khi P5 của OS được kích hoạt: ở đây dùng giao diện trừu tượng, **không đoán tên RPC**. Python thuần, mình chạy được test ở máy mình.


---

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
- **Chưa có** (đã grep `yui.cpp` ngày 2026-11-16): SHA-256 (băm đang làm bằng Dart ở `hub_download.dart`), nhánh RISC-V Vector, ràng buộc cú pháp lúc sinh, context-aware decoding, cửa sổ trượt W. **Đã có:** `kcap` (giới hạn k theo lần gọi). Trên x86 `detect_isa` chỉ chọn **AVX2+FMA hoặc `scalar`** (không có đường SSE/AVX1): CPU đời 2 trở về trước chạy mã thuần.
- **Hàm xuất: đúng 17 hàm `yui_*`** (đã liệt kê bằng `nm -D` trên bản biên dịch thật): `version, set_threads, get_threads, cpu_features, load, free, info, meta, chat_begin, chat_begin_img, complete_begin, gen_next, gen_tokens, gen_end, vision_load, vision_run, vision_free`.
- **Va chạm tên với hệ điều hành:** `libyui` của YuiVieOSBeta (`include/yui/yui.h`: pack/graph/telex/vault/telemetry/sandbox) là một thư viện **khác**, nhưng trùng hai tên hàm với engine này (`yui_version`, `yui_free`). Hai tên đó **cùng chữ ký nhưng khác ngữ nghĩa** (`yui_free` của OS giải phóng bộ nhớ malloc, của engine xoá `Model`), nên nếu hai thư viện cùng nằm trong một không gian ký hiệu thì **không có lỗi liên kết nào báo trước**, chỉ sai âm thầm. Xem P9-15.

### 4.3 Y-App (Flutter)
- 4 tab: Chat · Lịch sử · Model · Cài đặt. `lib/engine.dart` = isolate nền + FFI. `lib/state.dart` = `ChatStore`, `ModelStore` (nhập file, tải link + token, xoá), `Settings`, `AppState`, kiểm tra cập nhật qua GitHub Releases (`kRepo = BachDaThan/Y-App`).
- `ModelStore.isYui` chỉ kiểm 4 byte đầu `"YUI1"`. Danh sách model = mọi `*.yui` trong thư mục dữ liệu app.
- `lib/agent.dart`: chỉ 3 công cụ `calc`, `time`, `search` (Wikipedia vi); `kToolResultChars = 120`; tối đa 2 lần gọi công cụ/câu. **Không có** vault/RAG/Graphify/skill/orchestrator (chỉ có bản Python).
- Dịch vụ: `voice_service`, `sherpa_stt` (STT offline), `vision_service`, `update_service`, `notifier`, `background_guard` (flutter_foreground_task 8.17.0 ghim phiên bản).
- `yui.cpp` của Y-App nằm ở **`native/yui.cpp`** (không phải gốc repo); bản Adaptive-MoE ở gốc. Cả hai đang trùng sha256 `9e33a935…0e4d6` (118 651 byte; kiểm lại 2026-11-17 — con số cũ `39cc442d…` là trước P1-3).
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
- **BITCOS (cập nhật 2026-11-16, đã kiểm nguồn):** có thật (arXiv 2609.16338, Intel, preprint chưa phản biện). Công thức 2 − z bit/trọng số; z đo được 29,7%–51,5% trên 29 checkpoint; 1.485 là mức của checkpoint thưa nhất. Tăng tốc ghi nhận: tới 18% CPU / 27% GPU, **chỉ trên phần cứng Intel**, chưa thử Arm; trên một CPU Intel kernel 2-bit cố định còn nhanh hơn. **z của model của bạn: chưa đo (P9-1).**
- Kích thước adapter LoRA ≈ 0,3 MB fp16 (d=256, 6 lớp, rank 16, qkv+o): **tính tay**, chưa đo (P5-6).
- Kích thước KV cache ≈ 0,8 MB fp16 / 1,6 MB f32 (L=128, d=256, 6 lớp): **tính tay**.
- Tốc độ khi bật CAD ≈ 0,5× (2 lượt forward): **ước lượng**, chưa đo (P9-6).
- "`sched_ext` có sub-scheduler hoàn thiện 09–10/2026", "Route-Aware Adapter chứng minh trên 50–150M", tên/cỡ model Gemma 4 E2B / Qwen 3 Small / Phi-4-mini: **chưa xác minh**.

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
| 2026-11-13 | P2-4 `[x]` | Bạn báo GitHub Actions build **xanh** với `state.dart` + `hub_download.dart` mới ⇒ phần biên dịch Flutter đã được xác nhận. Còn chưa kiểm: tải thật từ Hugging Face trên điện thoại (chờ có giao diện P2-6) |
| 2026-11-13 | P2-5 `[~]` | **Đã làm:** `lib/services/hub_cache.dart`: sổ `hub_index.json` trong thư mục model ghi các model **tải từ Kho** (tên file, id, sha256, dung lượng, nguồn, lúc tải, lần dùng cuối). `trim(trần, protected)` xoá model **lâu chưa dùng nhất trước**, dừng ngay khi tổng ≤ trần. **Chỉ model có trong sổ mới có thể bị xoá**: file người dùng tự nhập hoặc tải bằng link tay không nằm trong sổ nên không bao giờ bị đụng; model đang chạy/đang chọn/vừa tải được truyền vào `protected`. Sổ luôn đối chiếu với đĩa: file mất hoặc đã bị thay (dung lượng khác lúc ghi) thì bỏ khỏi sổ và **không xoá**; sổ hỏng = rỗng (không xoá gì); tên trong sổ có `/`, `\\`, `..` hoặc bắt đầu bằng `.` bị loại (chặn xoá file ngoài thư mục). Thao tác đọc-sửa-ghi xếp hàng nên nạp model trong lúc đang tải không làm mất mục. `cleanPartials()` xoá `hub_<16 hex>.part` quá 7 ngày. **Nối vào `state.dart` (+30/−3 dòng):** `Settings.hubCacheMb` (mặc định **500**, tối thiểu 50, lưu trong SharedPreferences, **chưa có giao diện chỉnh**); `ModelStore.downloadHub` ghi sổ sau khi tải xong; `ModelStore.delete` gọi `forget`; `loadModel` gọi `touch` (nạp = vừa dùng); `AppState.downloadHubModel` truyền trần làm `maxBytes` (model lớn hơn cả trần bị từ chối trước khi tải) và **sau khi tải xong** mới `trim`, lưu kết quả ở `AppState.lastEvicted`; khởi động gọi `cleanPartials`. **Kiểm thật (Dart SDK 3.14):** `hub_cache_check` **19 đạt** (LRU, touch, bảo vệ, file tự nhập, file bị thay, forget, sổ hỏng, tên độc hại, 20 thao tác song song, dọn .part); `hub_flow_check` **11 đạt** chạy cả dòng thật (tải qua máy chủ cục bộ → ghi sổ → xoá theo LRU: tải A,B,C với trần 250 KB thì xoá A; nạp B rồi tải D thì xoá C chứ không xoá B; model đang chạy được giữ; tải hỏng hay model quá trần không làm mất model cũ; file tự nhập 500 KB còn nguyên); thử đột biến 5 kiểu ở `hub_cache` + 1 kiểu ở dòng tải, mọi lỗi đều bị bắt (lần đầu 1 đột biến lọt vì test tên độc hại quá yếu, đã siết lại). `hub_service_check` 32 đạt (34 khi truyền manifest Python), `hub_download_check` 145 đạt. `dart analyze` 0 lỗi cho các file mới; `state.dart` không phát sinh lỗi mới (số cảnh báo do thiếu gói Flutter ở máy mình y hệt trước khi sửa). **Chưa kiểm:** build Flutter thật (chờ bạn). **Thiết kế cần biết:** (a) trần là **mềm**: kiểm sau khi tải xong nên tạm vượt tối đa một model, đổi lại tải hỏng không làm mất model cũ; (b) nếu chỉ còn model được bảo vệ thì dừng và vẫn vượt trần (đúng thiết kế, ghi trong test); (c) "dùng" = lúc nạp model, chưa tính từng lượt chat; (d) model Kho đã tải **trước** bản này (nếu bạn từng thử) không có trong sổ nên **không bị xoá tự động** (an toàn, và bạn xoá tay được); (e) P2-6 phải hiển thị `lastEvicted` để người dùng biết model nào vừa bị dọn và cho đặt trần |
| 2026-11-13 | P8 mới (bạn nêu) | Bạn thấy cập nhật ngầm tải xong, bấm Cài đặt thì hiện màn cài APK nhưng không cài được, nghi app chưa xin quyền cài APK. **Xác nhận bằng code:** CI chỉ thêm `INTERNET` + `RECORD_AUDIO` vào manifest; `open_filex` cố ý không khai báo `REQUEST_INSTALL_PACKAGES` (theo mô tả gói). → làm ngay **P8-1** (sửa CI). Bạn hỏi về "cào web chưa nối vào app và model": **đúng**, và là chủ ý thiết kế cũ (app chỉ có calc/time/search Wikipedia; dữ liệu huấn luyện không có `fetch`). Vì kết quả công cụ chỉ ~120 ký tự nên cào cả trang không hữu ích nếu thiếu bước chọn đoạn (P3) → **ghi vào kế hoạch thành P8-3…P8-5**, chưa code, chờ bạn duyệt đặc tả |
| 2026-11-13 | P8-1 `[~]` | Giao `.github/workflows/build-app.yml`: thêm `sed` khai báo `android.permission.REQUEST_INSTALL_PACKAGES` ngay sau dòng `RECORD_AUDIO` (+ 2 dòng ghi chú). **Đã kiểm:** YAML hợp lệ (4 job: sync-yui, android, windows, release), chạy đúng 3 lệnh `sed` trên manifest mẫu ra XML hợp lệ có đủ 3 quyền. **Chưa kiểm:** cài thật trên điện thoại. **Lưu ý quan trọng:** bản đang cài trên máy bạn **không có quyền** nên không tự cài đè được bản mới; cài tay APK mới một lần (tải từ Releases rồi mở), từ lần sau nút Cài đặt trong app mới dùng được. Cài đè vẫn giữ dữ liệu/model vì cùng khoá ký |
| 2026-11-14 | P2-5 `[x]` | Bạn báo GitHub Actions build **xanh** với `state.dart` + `hub_cache.dart` |
| 2026-11-14 | P8-6 `[~]` | Bạn hỏi vì sao có nhiều công cụ cào web (trong `yui_web.py`, bản PC/Python) mà app chỉ dùng Wikipedia, và muốn nối thêm cho cả điện thoại lẫn PC. **Nguyên nhân thật:** app mới chỉ port công cụ `search` ở dạng tối thiểu (1 lần hỏi Wikipedia); các tầng khác chưa bao giờ được port. **Đã làm:** `lib/services/web_search.dart` (Dart thuần, chạy được trên Android + Windows): tầng key (6 dịch vụ, xoay vòng + nghỉ key 1–30 phút khi hết hạn mức, 6 giờ khi key sai), DuckDuckGo HTML, SearXNG, Wikipedia vi→en, móc `kb` cho kho tri thức P3, bộ nhớ đệm 10 phút, ngân sách 25 giây, câu báo khi hết cách (không ném lỗi, không để model "câm"). **An toàn:** chỉ https; chỉ nói chuyện với các tên miền dịch vụ đã biết + host SearXNG bạn tự nhập; host SearXNG phải phân giải ra địa chỉ **công cộng** (chặn localhost, 10.x, 172.16–31.x, 192.168.x, 169.254.x, 100.64/10, IPv6 nội bộ, mapped IPv4); chuyển hướng ≤ 3 lần và kiểm lại từng bước; giới hạn dung lượng phản hồi; key không xuất hiện trong kết quả hay trạng thái (che còn 4 ký tự). **Cố ý KHÔNG port:** các tầng đào HTML Brave/Bing/Google của `yui_web.py` (bản parse chưa thử với trang thật, và điều khoản các trang đó cấm truy vấn tự động); muốn dùng Brave/Google thì dùng bản API chính thức (`braveapi`, `gcse`) đã có. **Nối app:** `AgentTools.search` dùng `AgentTools.web` (đặt từ Cài đặt bởi `AppState.applyWebConfig()`, gọi lúc khởi động và mỗi lần đổi); `web == null` thì chạy Wikipedia như cũ. `Settings.webMulti` (mặc định bật) và `Settings.webCfg`. **Kiểm thật (Dart SDK 3.14, máy chủ giả cục bộ, không cần mạng/key thật):** `web_search_check` **114 đạt**, chạy lặp 25 lần đều đạt (lúc đầu có 1 bài phụ thuộc thứ tự xáo trộn ngẫu nhiên, đã sửa): parser 8 dịch vụ, hình dạng yêu cầu của từng dịch vụ (header/key/thân JSON), xoay vòng và nghỉ key, rơi xuống tầng sau, DuckDuckGo bị chặn 202/xác minh, SearXNG lỗi/khoá JSON/nhãn sai, Wikipedia, cache, ngân sách thời gian, 23 địa chỉ riêng/công cộng, chuyển hướng ra host lạ/vòng lặp/307 giữ POST, `defaultWebHttp` cắt dung lượng và không tự chuyển hướng, 8 lượt tìm song song, và đường `AgentTools.run('search')` (cắt 120 ký tự, gỡ thẻ `<tool>`). **Thử đột biến 12 lỗi cố ý** (bỏ chặn địa chỉ riêng, bỏ kiểm chuyển hướng, bỏ nghỉ key, bỏ xoay vòng, lộ key, nghỉ key sai quá ngắn, bỏ ngân sách, tự chuyển hướng, bỏ cache, cho userinfo…): đều bị bắt; 2 đột biến (bỏ kiểm Content-Type; bỏ ngắt khi quá dung lượng) ban đầu lọt vì tương đương về kết quả, mình đã thêm bài kiểm cho Content-Type và ghi chú phần còn lại. `dart analyze` 0 lỗi cho 3 file Dart thuần; `state.dart`/`settings_screen.dart` chỉ phát sinh lỗi do thiếu gói Flutter ở máy mình (không có lỗi nào liên quan code mới). **Chưa kiểm:** (a) build Flutter thật (chờ bạn); (b) **chưa thử với DuckDuckGo thật hay key thật của dịch vụ nào** (máy mình không ra được các trang đó; bản parse viết theo tài liệu/`yui_web.py`, thử bằng dữ liệu giả); (c) DuckDuckGo có thể chặn/đổi giao diện: khi đó app tự sang tầng sau (Wikipedia). **Cần biết:** (1) câu bạn tra sẽ được gửi tới DuckDuckGo/dịch vụ bạn nhập (công tắc để tắt); (2) key lưu trong bộ nhớ riêng của app, **không mã hoá**; (3) kết quả đưa cho model vẫn bị cắt ở 120 ký tự (`kToolResultChars`), nên chất lượng trả lời còn tuỳ model và sẽ tốt hơn khi có P3; (4) chưa chống DNS-rebinding bằng cách ghim IP cho host SearXNG tự thêm (ghi trong đầu file) |
| 2026-11-14 | P8-6 `[x]` | Bạn báo build **xanh** với tìm web nhiều nguồn. **Chưa có xác nhận** đã hỏi Yui một câu cần tra web và thấy trả lời qua tầng mới (DuckDuckGo/SearXNG/key thật) |
| 2026-11-14 | P2-6 `[~]` | **Đã làm:** tab **Kho** (biểu tượng cửa hàng) là tab thứ 4; Cài đặt dời thành tab thứ 5 (`chat_screen.dart` chỉ nhảy sang tab Model = 2 nên không ảnh hưởng). `lib/screens/hub_screen.dart`: **mặc định TẮT** (thẻ giải thích gửi gì đi đâu + nút "Bật Kho model"; chưa bật thì không nối mạng; `Settings.hubOn` mặc định false); khi bật chỉ tải danh sách lúc mở tab Kho (vì `IndexedStack` dựng sẵn mọi tab lúc khởi động nên màn nghe `app.tab`), cache 24 giờ, kéo xuống/nút Làm mới để tải lại; hiện tình trạng từng nguồn (số model, "từ bộ nhớ đệm", "bản đã lưu X giờ trước", lỗi tiếng Việt, số mục lỗi bị bỏ qua); **lọc theo ngành** với đủ 8 ngành kể cả ngành 0 model (hiện "Chưa có model nào cho ngành X trong các nguồn đang bật"); thẻ model hiện mô tả, dung lượng, lượng tử, tham số, phiên bản .yui, ngữ cảnh, miền/ngôn ngữ, giấy phép, nguồn; **cảnh báo arch_id lạ** (không đúng dạng `amoe-d…-l…-h…-e…-s…-f…`) ngay trên thẻ và trong hộp xác nhận; Tải (hộp xác nhận nêu máy chủ tải, dung lượng, giấy phép; từ chối nếu lớn hơn trần), thanh tiến độ + nút **Dừng (giữ phần đã tải)**, Dùng, Xoá; nút **Nguồn của Kho**: thêm nguồn bằng link (`Tên = https://…/manifest.json`, https, tối đa 8) hoặc xoá dòng để bớt; nút **Trần dung lượng** (thanh trượt 50–4000 MB, hạ trần thì dọn ngay model Kho ít dùng nhất và báo tên). Thông báo sau tải nêu model nào bị dọn (`lastEvicted`). Logic mới đặt ở phần Dart thuần để kiểm được: `parseHubSources` (hub_service.dart) và `hubArchWarning` (hub_download.dart). **Kiểm thật (Dart SDK 3.14):** `hub_service_check` **38 đạt** (thêm 6 cho `parseHubSources`: https, trùng, trùng nguồn mặc định, link có `=` trong query, tên dài, tối đa 8), `hub_download_check` **148 đạt** (thêm 3), `hub_cache_check` 19, `hub_flow_check` 11, `web_search_check` 114 đều đạt; `dart analyze` 0 lỗi cho các file Dart thuần; với `hub_screen.dart`/`state.dart`/`main.dart` mình chỉ phân tích khi thiếu gói Flutter nên **không kiểm được giao diện**: ngoài các lỗi do thiếu `ChangeNotifier`/`Icons`/widget, không có lỗi nào liên quan code mới (không có ký hiệu của HubService/HubModel/hubArchWarning… bị báo sai). **Chưa kiểm:** (a) build Flutter thật; (b) bố cục/giao diện trên máy thật (mình không chạy được app); (c) tải thật từ Hugging Face. **Quyết định cần bạn xem lại:** Kho mặc định TẮT (theo mục "opt-in" của kế hoạch), bạn bật ngay trong tab Kho; chưa có chỗ nhập HF token cho repo riêng (P2-7) |
| 2026-11-15 | P8-1 `[x]` | Bạn báo cài APK mới và cập nhật ngầm **đã cài được** |
| 2026-11-15 | P2-6 lỗi quét Kho | Ảnh chụp tab Kho: "Yui-Models: nguồn chuyển hướng sang địa chỉ không phải https, đã chặn", dù repo có model và `manifest.json`. **Nguyên nhân (đối chiếu code):** `defaultHubFetch` kiểm `scheme` của header `Location` THÔ; khi máy chủ chuyển hướng bằng đường dẫn **tương đối** (kiểu `/api/resolve-cache/…`, Hugging Face hay trả) thì `scheme` rỗng nên bị chặn nhầm. (Chưa xác nhận bằng cách gọi Hugging Face thật vì máy mình không ra được; kết luận dựa trên đúng câu thông báo trong ảnh khớp với đúng đoạn code đó và ca lỗi này tái hiện được.) **Sửa:** `hubRedirectTarget` ghép `Location` với địa chỉ hiện tại rồi mới kiểm (chỉ https, có tên máy, không userinfo), `defaultHubFetch` tự đi theo chuyển hướng tối đa 5 lần, thông báo lỗi nêu rõ đích bị chặn. **Vì sao test cũ không bắt được:** bài kiểm cũ dùng đúng Location tương đối và coi việc bị chặn là "đúng", tức là bài kiểm đã mã hoá luôn lỗi này. Đã viết lại: Location tương đối phải được đi theo, 12 ca cho `hubRedirectTarget` (tương đối, `../`, `//host`, CDN khác máy chủ, http/ftp/javascript/data/userinfo bị chặn), vòng lặp, thiếu Location; **đột biến** quay lại cách cũ thì bộ kiểm báo hỏng. `hub_service_check` **53 đạt**. `hub_download` vốn đã ghép địa chỉ tương đối đúng nên không bị |
| 2026-11-15 | P8-7 `[~]` | Bạn nhờ chỉnh notebook train (`yui_kaggle_4_cells.md` v6.5 → v6.6) vì chưa có model để thử tìm web. **Phát hiện chính:** tham số 1B tổng có ~219M tham số hoạt động/token; với 2×T4 trong 12 giờ chỉ học được ≈0,18–0,54 tỷ token (≈1–2 token/tham số hoạt động; mốc tham khảo ~20) nên 1B cần khoảng 8–24 phiên 12 giờ; 150M đạt trong một phiên (**ước lượng, giả định hiệu suất GPU 5–15%, hãy thay bằng tok/s thật trong log**). **Đã chỉnh:** xem task P8-7. **Kiểm (dry-run trên máy mình, bản giả của Kaggle/torch/HF):** 4 ô đều biên dịch; Ô1 ra đúng giá trị cho cả 3 profile; Ô3 sinh đúng lệnh (DDP 2 GPU, 1 GPU, BitNet bỏ fast_moe) và **mọi cờ đều có trong `train_lm.py`**; Ô4 tạo file `.yui` v3 thật có META, `make_manifest.py` chạy thật ra manifest gộp model cũ + 2 model mới, khớp sha256, `PUBLISH_TO_HUB=False` không đẩy gì, `=True` chỉ đẩy file mới + manifest và không tự tạo repo; khi thiếu `make_manifest.py` thì báo rõ, không sập; chạy lặp 3 lần đều đạt. **Chưa kiểm:** chạy thật trên Kaggle (GPU, quota LLM, HF thật), OOM với `TRAIN_B` mới (hạ B nếu gặp), chất lượng model. **Cần bạn quyết:** giấy phép `MODEL_LICENSE` (để `unknown` tới khi bạn đọc điều khoản FineWeb-2 và các LLM teacher); có đổi cỡ thì Ô2 tạo data mới cho cỡ đó (tốn quota LLM), data 1B cũ vẫn giữ |
| 2026-11-15 | P2-6 `[x]` | Bạn báo build xanh và **tab Kho quét được danh sách** sau bản sửa chuyển hướng tương đối |
| 2026-11-15 | P8-6b lỗi chữ "Th ủ đô" | Bạn gửi kết quả "Thử tìm web": "Th ủ đô Việt Nam hiện nay là thành phố Hà Nội. Sau đây là danh sách… (vi.wikipedia.org)" (**tầng DuckDuckGo đã chạy thật**, vì đoạn trích kiểu "… hiểu theo n…" là của DuckDuckGo). **Nguyên nhân:** DuckDuckGo tô đậm từ khoá bằng `<b>` và có thể cắt giữa chữ (`Th<b>ủ</b> đô`); hàm làm sạch thay MỌI thẻ bằng dấu cách nên thành "Th ủ đô". Lỗi này mình mang nguyên từ `yui_web._clean` của Python sang Dart nên **cả bản PC lẫn app đều bị**. **Sửa cả hai:** thẻ định dạng chữ (`b, strong, i, em, u, mark, span, a, abbr, …`) được gỡ KHÔNG chèn dấu cách; thẻ khối/xuống dòng (`br, p, li…`) và thẻ lạ vẫn thành dấu cách; `<` chỉ coi là thẻ khi đi liền chữ cái/`/`/`!`/`?` nên chữ như `x < y và y > z` không còn bị nuốt (lỗi phụ mình tìm thấy khi viết test). **Kiểm:** Dart `web_search_check` **126 đạt** (thêm 9 ca, có ca đúng chuỗi tô đậm của DuckDuckGo), Python `test_yui_extra.py` **22 đạt** (thêm 1 test); thử đột biến quay lại cách cũ thì cả hai bộ kiểm báo hỏng. **Chưa kiểm:** trên máy bạn sau khi build (hãy hỏi lại cùng câu). 📤 `Y-App/lib/services/web_search.dart`, `Y-App/tools/web_search_check.dart`, `Adaptive-MoE/yui_web.py`, `Adaptive-MoE/test_yui_extra.py`, `Adaptive-MoE/docs/PLAN.md` |
| 2026-11-15 | P8-6b `[x]` | Bạn xác nhận sau khi build: "Thử tìm web" trả "Thủ đô Việt Nam hiện nay là thành phố Hà Nội. Sau đây là danh sách các kinh đô/thủ đô - hiểu theo ng… (vi.wikipedia.org)", **chữ liền, hết lỗi "Th ủ"**. Bạn tạm dừng chat này để mở chat mới làm tiếp |
| 2026-11-16 | Kế hoạch v2: đánh giá đợt tư vấn thứ hai + P5-6 + P9 + mục 6 | Đối chiếu 4 gợi ý của Gemini + 2 bản phân tích hợp nhất + chuỗi tư vấn hệ điều hành với code thật của 3 repo (xem mục 1b, 17 dòng đánh giá). **Kết luận chính:** (1) BITCOS **có thật** nhưng 1.485 bit là mức tốt nhất của 29 model và tăng tốc chỉ đo trên Intel, nên chỉ làm thí nghiệm sau khi **đo z của model bạn** (P9-1); (2) "converter Qwen/Gemma → .yui" **không phải việc của một script**: engine chỉ chạy kiến trúc Adaptive-MoE (`F = 2·D`, LayerNorm có bias…), cần chọn đường (P9-17, Q11); (3) nén KV-cache bằng ZRAM **vô nghĩa ở app** (KV ≈ 1 MB); (4) khoá cú pháp công cụ: tốt nhưng chỉ bảo đảm **cú pháp**; (5) **phát hiện thật ở code:** engine không có đường SSE/AVX1, máy CPU đời 2 chạy mã thuần (P9-16); và `libyui` của OS **trùng tên** (và 2 tên hàm, cùng chữ ký, khác ngữ nghĩa) với engine này (P9-15). **Thêm:** P5-6 (1 task), P9 (17 task); tổng 52 ⇒ 70 task. Không task nào cũ bị đổi mã hay bỏ; **P3-2/P3-3 treo chờ Q6**. 📤 `Adaptive-MoE/docs/PLAN.md`, `YuiVieOSBeta/docs/PLAN.md` |
| 2026-11-17 | Kế hoạch v3: nâng cấp 3 repo | Đối chiếu đợt tư vấn thứ ba của Gemini (chủ yếu về hệ điều hành) và log test của bạn (Kaggle: C++/C11 xanh; Rust `kernel-cap` 23, `yui-sched` 13, `yui-zram` 8; 3 kernel build + QEMU boot; IME Direct mode `e885ac9`: `test_ime` 459, `YUI_IME_MODE=direct` chạy) với code thật 3 repo (clone 2026-11-17). **Mình tự chạy lại** `make test` + `make check` của YuiVieOSBeta: 2994 + 61 = **3055** check, 0 fail; `scripts/verify.sh` xanh với file kế hoạch OS mới. **Không làm được:** `cargo`, QEMU, `make ibus` (thiếu công cụ), và API CI của GitHub (hết hạn mức) ⇒ CI chưa được đối chiếu từ phía mình. **Thay đổi ở file này:** (1) Q6 = C++17 ⇒ P3-2, P3-3 `[-]`; (2) thêm P9-18; (3) sửa sha256/dung lượng `yui.cpp` ở §4.3 (số cũ lệch từ P1-3; hai bản vẫn trùng nhau: kiểm 2026-11-17); (4) §6 cập nhật. **Không đổi mã hay trạng thái task nào khác.** OS: +12 task (P0.14, P1.11–P1.12, P3.13–P3.16, P4.14, P5.17, P6.14–P6.16) + §5.2 + §7.7 + §13 C30–C41, xem `YuiVieOSBeta/docs/PLAN.md`. 📤 `Adaptive-MoE/docs/PLAN.md`, `YuiVieOSBeta/docs/PLAN.md` |

**Câu hỏi mở (trả lời khi tới task liên quan):**
- **(4) "Chế độ Hybrid" của P2-7 nghĩa là gì?** Kho (`Settings.hubOn`) và "Tìm web nhiều nguồn" (`Settings.webMulti`) hiện đã có công tắc riêng. Đề xuất của mình: Hybrid là **một công tắc tổng** "Cho phép app dùng dịch vụ ngoài máy" ở Cài đặt (mặc định TẮT cho người mới; người dùng hiện có được giữ nguyên các lựa chọn đã bật): tắt = không gọi máy chủ nào ngoài mục đã nêu trong chính sách (cập nhật app, Wikipedia); bật = cho phép bật riêng Kho, tìm web nhiều nguồn và các mục sau này (P4). Kèm ô nhập Hugging Face token (tuỳ chọn, chỉ gửi tới máy chủ huggingface.co, lưu trên máy). **Chờ bạn xác nhận hoặc sửa.**
- Kho mới có 1 model (`yui-q4`, chat+general). Muốn chọn model theo ngành (P5) thì cần có model `code`/`math`/`translate`… huấn luyện và xuất trước; việc này nằm ngoài P2 (hiện chưa có task) → **quyết khi tới P2-6 (giao diện cần trạng thái "chưa có model cho ngành này") và trước P5**.
1. ~~P0-2: xoá hẳn `provider-chain.patch` hay lưu vào `docs/archive/`?~~ → **đã chốt: lưu `docs/archive/`**
2. ~~P1-1: META có cần trường `license` bắt buộc không?~~ → **đã chốt: bắt buộc có khoá, cho `"unknown"`**
3. ~~P2: repo model mặc định là repo nào, public hay private?~~ → **đã chốt: `BachDaThan/Yui-Models`, public**
4. P3-6: có cần mã hoá kho ngay bản đầu không?
5. P4-3: danh sách dataset cho phép ban đầu gồm những dataset nào?

**Câu hỏi mới (2026-11-16, trả lời khi tới task liên quan; mình đã nêu nghiêng về đâu):**
- ~~(6) P3: kho tri thức bằng Dart hay C++17 lõi chung?~~ **→ đã chốt C++17 (bạn trả lời 2026-11-16).** Hệ quả ghi ở đầu mục P3; CI Y-App phải biên dịch thêm file C++ (P9-14).
- **(7) P5-6: có làm thí nghiệm adapter LoRA không?** Đề xuất: có, nhưng **sau** khi có model đầu tiên (P8-7); chỉ Python.
- **(8) Ký model Kho (P9-7):** chữ ký **tách rời trong manifest** (mình đề xuất, không đổi `.yui`) hay trong chân file `.yui` (như bản phân tích)? Và Ed25519 đường **C++** (tái dùng mã đã qua RFC 8032 của OS) hay **Dart** (thêm gói)? Mình nghiêng về C++ nếu Q6 cũng là C++.
- **(9) Cảm biến pin/nhiệt (P9-10):** cho phép thêm gói/MethodChannel không? (rủi ro build như P8-2). Nếu chưa, chỉ làm công tắc tay (P9-9).
- **(10) BITCOS:** đồng ý chỉ làm tiếp (P9-2, P9-3) **sau khi** P9-1 cho thấy file nhỏ đi ≥ 10%?
- **(11) Model lớn hơn (P9-17):** sau khi đọc bản khảo sát mới chọn A (chưng cất vào kiến trúc của bạn), B (thêm kiến trúc dense vào engine) hay C (backend GGUF thứ hai). Chưa cần trả lời ngay.

---

## 6. CẦU NỐI VỚI HỆ ĐIỀU HÀNH YuiVieOS

- **Repo OS:** `BachDaThan/YuiVieOSBeta`, nhánh `arena/17570fa8-yuivieos`. Kế hoạch riêng: `YuiVieOSBeta/docs/PLAN.md` (phase P0–P6). Hai file không trộn vào nhau; file này chỉ ghi phần chạm tới Y-App/Adaptive-MoE.
- **Phase P5 của OS ("Tích hợp AI vào hệ điều hành") đang KHOÁ** theo quy định của chính file đó: chỉ chạy khi bạn nói đúng câu **"Tích hợp AI vào hệ điều hành"**. Mình chưa kích hoạt, chưa kéo mã nào vào repo OS; ở đây chỉ **chuẩn bị phía engine** để khi bạn ra lệnh thì không vướng.
- **OS sẽ lấy gì từ 2 repo này:** `yui.cpp` (thành engine), định dạng `.yui` v3 (mmap), `yui_agent.py` (làm **client** gọi `yuid`), và (nếu Q6 = C++) `kb_core`.
- **Hai phát hiện cần xử lý trước khi OS cắm engine vào (đã kiểm bằng code):**
  1. **Hai thư viện cùng tên `libyui`.** Của OS (graph/vault/telex/telemetry/sandbox) và engine này (load/chat/gen) trùng tên file và trùng hai tên hàm (cùng chữ ký, **khác ngữ nghĩa**, nên sai sẽ âm thầm). Cách xử lý nhẹ nhất: engine build ra **`libyui_engine.so`** (đổi tên file, không đổi tên hàm, nên Y-App không phải sửa) và `yuid` nạp nó bằng `dlopen(RTLD_LOCAL)` + `dlsym` theo tên. Là task **P9-15**.
  2. **Chạy tệp mô hình lạ ở dịch vụ nền.** Trong app, file `.yui` lỗi chỉ làm app sập; trong `yuid` (dịch vụ nền) hệ quả nặng hơn. Vì vậy **P9-8 (fuzz)** nên xong trước khi P5 của OS chạy.
- **Điều kiện đề xuất từ phía repo này trước khi OS kích hoạt P5:** P9-15 xong; P9-8 xong; (khuyến nghị) P9-4 và P9-7 xong; ghim commit SHA của cả 2 repo vào `vendor/VERSIONS.lock` (đã có trong P5.2 của OS).
- **Công cụ hệ thống (`sys_power`, `sys_proc`…) không nằm trong repo này.** Chúng ở `yuid` (C++) với thẻ năng lực; `yui_agent.py` chỉ gọi qua `/run/yuid.sock` (nguyên tắc 9).
- **Cập nhật 2026-11-17 (nâng cấp cả 3 kế hoạch):**
  - *Trạng thái OS (đã kiểm):* mình chạy lại `make test` + `make check` trên `main` của YuiVieOSBeta: **3055 check** (2994 C++17 + 61 C11), 0 fail, số theo từng file trùng log Kaggle của bạn. Direct mode của bộ gõ (`e885ac9`) đã vào `main`. Kernel 3 kiến trúc boot trong QEMU **theo log của bạn** (mình không chạy được QEMU/cargo và không đọc được CI). Chi tiết: `YuiVieOSBeta/docs/PLAN.md` §12.
  - *Chỉ dẫn cho tác tử AI thi công P5:* OS `docs/PLAN.md` §7.7 (vẫn khoá tới khi có đúng câu kích hoạt). Điều kiện vào phía repo này: P9-15, P9-8; khuyến nghị P9-4, P9-7; nếu làm P5.17 thì thêm P9-18.
  - *Công cụ web trong OS không tự mở mạng:* OS thêm thẻ phiên (P3.16, mặc định 15 phút, có ô nhập thời gian) và P5.17; phía agent là **P9-18**.
  - *Đợt tư vấn thứ ba của Gemini* (dịch .exe/.apk sang định dạng gốc, warm pool micro-VM, cửa sổ liền mạch, shader pre-cache, Box64/FEX, sinh policy chặn, bộ gõ/USB token trong hộp cát) **không chạm tới 2 repo này**; đánh giá ở OS §13 (C30–C41). Điểm đáng nhớ: ý "trả HTTP 200 rỗng cho HTTPS" không làm được nếu không MITM; thiết kế đúng là sinkhole DNS + thẻ phiên.
