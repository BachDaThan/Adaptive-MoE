# SỔ ĐĂNG KÝ PHÁP LÝ & ĐIỀU KHOẢN (dùng chung Y-App · Adaptive-MoE · YuiVieOSBeta)

> **Mình không phải luật sư.** File này là sổ theo dõi kỹ thuật: văn bản nào có liên quan, mình đã kiểm gì, còn **chưa** kiểm gì, việc nào phải hỏi luật sư. Không file nào trong dự án (kể cả file này) bảo đảm "không vi phạm pháp luật". Một kế hoạch kỹ thuật chỉ giảm rủi ro; việc tuân thủ còn phụ thuộc cách bạn vận hành và phát hành.
>
> Đặt tại `Adaptive-MoE/docs/LEGAL_REGISTER.md` (một bản duy nhất, như `PLAN.md`). Kế hoạch: `docs/PLAN.md` nhóm P9-F (phía AI) và `YuiVieOSBeta/docs/PLAN.md` Phase P7 (phía OS).
>
> **Tạo:** 2026-11-17 · **Quy tắc tuổi bản kiểm:** luật ở Việt Nam đổi nhanh (Nghị định xử phạt 330/2026 ra đời tháng 8/2026, sau hai nghị định hướng dẫn). Một dòng "đã kiểm" chỉ có giá trị nếu **ngày đọc ≤ 90 ngày** tại thời điểm bấm Public; quá hạn thì đọc lại.

Ký hiệu trạng thái: ✅ đã kiểm (có nguồn + ngày) · 🟡 đã thấy văn bản tồn tại nhưng **chưa đọc điều khoản gốc** · ❓ chưa kiểm · ⚖️ phải hỏi luật sư.

---

## A. Văn bản pháp luật liên quan

| Văn bản | Thông tin đã xác nhận (2026-11-17) | Trạng thái | Ảnh hưởng tới dự án | Task |
|---|---|---|---|---|
| **Luật Trí tuệ nhân tạo 134/2025/QH15** | Quốc hội thông qua 10/12/2025, hiệu lực 01/03/2026. Phân hệ thống AI theo 3 mức rủi ro (cao/trung bình/thấp), Điều 7 liệt kê hành vi bị cấm (gồm cản trở cơ chế giám sát, can thiệp và kiểm soát của con người). Nguồn: `thuvienphapluat.vn` (số hiệu 134/2025/QH15) | 🟡 | Yui/Yui OS là "hệ thống AI"; cần tự phân loại mức rủi ro, minh bạch nội dung AI, giữ con người kiểm soát hành động quan trọng | P9-20, OS P7.6, P7.7 |
| **Nghị định 142/2026/NĐ-CP** | Hướng dẫn Luật AI, ban hành 30/04/2026. Nêu việc phân loại hệ thống AI theo rủi ro và đánh giá sự phù hợp với hệ thống rủi ro cao. Nguồn: `vietq.vn`, `thuviennhadat.vn` | 🟡 (chưa đọc tiêu chí phân loại) | Quyết định Yui thuộc mức nào, có nghĩa vụ gì | OS P7.7 |
| **Thông tư 05/2026/TT-BKHCN** (Khung đạo đức AI quốc gia) | Có văn bản trên cổng thông tin Chính phủ | ❓ (chưa đọc nội dung) | Có thể có yêu cầu đạo đức/quản trị cho nhà phát triển | OS P7.7 |
| **Luật Bảo vệ dữ liệu cá nhân 91/2025/QH15** | Có hiệu lực 01/01/2026; **thay thế hoàn toàn Nghị định 13/2023** (theo VCCI, 15/01/2026) | 🟡 | Cơ sở cho đồng ý rõ ràng, quyền xoá, chuyển dữ liệu ra nước ngoài | P9-22, OS P7.5, P7.9 |
| **Nghị định 356/2025/NĐ-CP** | Ban hành 31/12/2025, hiệu lực 01/01/2026; hướng dẫn Luật BVDLCN. Khi chuyển dữ liệu ra ngoài lãnh thổ phải ghi nhận và giải trình được mục đích (theo bài trên `baodautu.vn`) | 🟡 | Như trên | P9-22 |
| **Nghị định 330/2026/NĐ-CP** | Ban hành 19/08/2026, quy định **xử phạt** vi phạm hành chính về an ninh mạng và BVDLCN; hiệu lực ngay, **không có thời gian chuyển tiếp** (theo bản tin EY). Có nội dung riêng về dữ liệu cá nhân của trẻ em từ đủ 7 tuổi (theo EY; chưa đọc điều khoản gốc). **Gemini không nhắc văn bản này** | 🟡 | Từ nay vi phạm BVDLCN có chế tài cụ thể ⇒ kiểm lại mọi chỗ gửi dữ liệu ra ngoài | P9-21, P9-22 |
| **Nghị định 13/2023/NĐ-CP** | Đã bị thay thế từ 01/01/2026. **Bản mẫu pháp lý của Y-App vẫn dẫn văn bản này** (xem §E) | ✅ lỗi thời | Phải sửa | P9-22 |
| **Luật Dữ liệu 60/2024/QH15** | Được ghi trong phần căn cứ của NĐ 356/2025. Ngày hiệu lực Gemini nêu (01/07/2025): **chưa kiểm** | ❓ | Chưa rõ áp dụng cho dự án cá nhân phi thương mại | OS P7.7 |
| **Mật mã dân sự**: NĐ 58/2016/NĐ-CP (sửa đổi bởi NĐ 53/2018, NĐ 32/2023; trang tra cứu ghi NĐ 32/2023 đã được thay thế bởi NĐ 211/2025/NĐ-CP) | Chế độ cấp phép kinh doanh sản phẩm/dịch vụ mật mã dân sự và xuất nhập khẩu **có tồn tại**. **Chưa xác định** có áp dụng cho phần mềm mã nguồn mở miễn phí có Ed25519/SHA/scrypt/`yui_vault` hay không | ⚖️ | Chỉ nêu để hỏi luật sư; **không** kết luận | OS P7.8 |
| Quy định xuất khẩu mật mã ở nước đặt máy chủ mã nguồn (GitHub) | Chưa kiểm | ❓ ⚖️ | Như trên | OS P7.8 |

**Gemini viết "hoàn thành cả hai kế hoạch thì không vi phạm pháp luật": mình không đồng ý.** Lý do ở OS `PLAN.md` §13 mục C42–C44: (1) không ai bảo đảm được điều đó; (2) nhiều task Gemini viện dẫn (P2-7, P2-8, P3-5, P4-2, P4-5, P9-7, P1.9, P4.13) hiện **chưa làm**; (3) nó bỏ sót NĐ 330/2026 và điều khoản của chính các dịch vụ AI bạn dùng (§C).

---

## B. Nội dung bắt buộc trong Điều khoản sử dụng và Chính sách quyền riêng tư

Dùng làm danh sách đối chiếu khi sửa `assets/legal/*.md` (Y-App, P9-22) và soạn bản cho OS (P7.9). Mỗi dòng phải **khớp hành vi thật của mã** (xem §E: hiện có một chỗ lệch).

1. Dữ liệu chat/kho tri thức nằm trên máy; nhà phát triển không nhận bản sao. Liệt kê **đủ** mọi đích gửi ra ngoài: Wikipedia (khi model gọi `search`), kho model (Hugging Face), cập nhật app (GitHub), nhà cung cấp tìm kiếm có key (Tavily, Exa, Linkup, Firecrawl, Brave API, Google PSE), DuckDuckGo (nếu bật), SearXNG người dùng liệt kê, máy chủ MCP người dùng kết nối. Mỗi mục ghi: gửi gì, kèm IP, bên nhận ở nước ngoài.
2. Mọi kết nối ngoài **mặc định tắt**, bật bằng hành động rõ ràng của người dùng; im lặng không phải đồng ý.
3. Quyền xem/sửa/xuất/xoá/rút lại đồng ý; nút xoá toàn bộ kho (P3-5); thời hạn xử lý yêu cầu nếu có người liên hệ (NĐ 356 nêu thời hạn xử lý yêu cầu của chủ thể dữ liệu, PwC ghi "lên đến 30 ngày"; chưa đọc điều khoản gốc).
4. Nội dung do AI tạo: dán nhãn "do AI tạo" trong giao diện; có thể sai (ảo giác); **không dùng** cho chẩn đoán/điều trị y tế, tư vấn pháp lý, quyết định tài chính hay quyết định có rủi ro cao khác.
5. Model/dữ liệu/gói do người dùng tự tải hoặc tự thêm nguồn: thuộc nhà cung cấp tương ứng, theo giấy phép của họ; nhà phát triển không chịu trách nhiệm với nguồn không đáng tin; model/gói **không có chữ ký hợp lệ** sẽ có cảnh báo (P9-7, OS P3.12).
6. Tính năng đọc kết quả công cụ tìm kiếm không dùng API (DuckDuckGo, nếu có): chạy từ IP của người dùng, có thể trái điều khoản của trang đó, người dùng tự chọn bật và tự chịu trách nhiệm; khuyến nghị dùng API chính thức hoặc Wikipedia.
7. Ghi công nguồn: Wikipedia CC BY-SA 4.0 (hiển thị tên miền nguồn); giấy phép các thư viện (`THIRD_PARTY_NOTICES.md`).
8. Người dưới tuổi: **ngưỡng tuổi và đồng ý của cha mẹ phải xác nhận lại theo luật mới** (bản mẫu hiện ghi 16 tuổi, dựa trên khung cũ).
9. Giới hạn trách nhiệm "nguyên trạng" và luật áp dụng: có thể không có hiệu lực với phần luật bắt buộc (bản mẫu đã ghi "trong phạm vi pháp luật cho phép"); ⚖️ hỏi luật sư, nhất là khi AI điều khiển hệ thống (OS P5) và gây mất dữ liệu.
10. Với OS: `audit.log` lưu gì, lưu bao lâu, xoá thế nào (OS P7.5); cảnh báo SmartScreen/Android khi dùng khoá tự ký (không vi phạm luật, nhưng phải giải thích trên trang tải).
11. Thông tin liên hệ chủ sở hữu và địa chỉ công khai của chính sách (Google Play/App Store yêu cầu URL công khai).

---

## C. Điều khoản của dịch vụ bên thứ ba bạn đang/sắp dùng

Cột "Đã biết" chỉ ghi điều mình **thực sự đọc được hôm nay**. Mọi dòng còn ❓ phải do bạn đọc bản **hiện hành** và điền ngày + kết luận (task P9-19).

| Dịch vụ | Dùng ở đâu | Đã biết | Trạng thái | Việc cần làm |
|---|---|---|---|---|
| **Gemini API** (9 khoá của bạn) | Sinh/lọc/chưng cất dữ liệu train (`make_data.py`, notebook Kaggle); (ngoài 3 repo) bot Yui | Bản **lưu trữ** của *Gemini API Additional Terms* (hiệu lực 17/10/2025): *"không dùng Dịch vụ để phát triển model cạnh tranh với Dịch vụ"* (`ai.google.dev/gemini-api/terms-archive/terms_10_17_25`). Bản 25/09/2025: không dùng Dịch vụ trong ứng dụng **hướng tới hoặc có thể được người dưới 18 tuổi truy cập** (`.../terms_09_25_25`). Dịch vụ miễn phí: Google dùng nội dung bạn gửi để cải thiện sản phẩm, người duyệt có thể đọc; **không gửi thông tin cá nhân/nhạy cảm** (`.../terms_04_03_25`) | 🟡 bản cũ; **chưa đọc bản hiện hành** | Đọc bản hiện hành; nếu vẫn cấm ⇒ **không** dùng đầu ra Gemini làm dữ liệu train cho model bạn phát hành, hoặc đánh `MODEL_LICENSE` = không công bố; ⚖️ hỏi "model chat tiếng Việt nhỏ" có bị coi là cạnh tranh không |
| Cohere · OpenRouter (+ từng model cụ thể) · Mistral | Chuỗi nhà cung cấp LLM của `make_data.py` | Chưa đọc | ❓ | Đọc điều khoản về **dùng đầu ra để huấn luyện model khác**; OpenRouter: đọc theo **từng model** |
| Tavily · Exa · Linkup · Firecrawl | Tìm/cào web bằng khoá của bạn | Chưa đọc | ❓ | Đọc điều khoản về lưu cache, huấn luyện trên kết quả, tái phân phối |
| Brave Search API · Google PSE (`gcse`) | Tầng 1 tìm web trong Y-App | Chưa đọc | ❓ | Như trên; Google: điều khoản PSE |
| **DuckDuckGo HTML** (`html.duckduckgo.com`) | Tầng 2 tìm web trong Y-App, **chạy không cần key** | Y-App tự ghi chú: điều khoản các trang này cấm truy vấn tự động (`web_search.dart` đầu file; `terms_*.md`) | ⚠️ rủi ro điều khoản **đang bật mặc định** (xem §E) | P9-21 |
| Google/Bing/Brave HTML không key; `curl_cffi` giả dấu vân tay TLS | `yui_web.py` (bản PC) | `web_search.dart` ghi rõ **không port**; `THIRD_PARTY_NOTICES.md` nêu `curl_cffi` dùng "để giả dấu vân tay TLS" | ⚠️ | P9-23: bản công khai không bật; giả vân tay để né chặn bot nặng hơn một vi phạm điều khoản thông thường |
| Hugging Face (Space free tier, repo model/dataset) | Kho `Yui-Models`, `TrainAI*` | Chưa đọc | ❓ | Đọc điều khoản dịch vụ + nội dung được phép lưu trữ/phân phối |
| Kaggle · GitHub (Actions, Releases) | Huấn luyện, CI, phát hành | Chưa đọc | ❓ | Đọc về giờ GPU/tự động hoá/Releases (OS P4.10) |
| Wikipedia (vi) | `search`, dữ liệu train | CC BY-SA 4.0, ghi công bằng tên miền nguồn (đã có) | ✅ | Giấy phép **chia sẻ tương tự** có thể ảnh hưởng nhãn giấy phép của model/dữ liệu dẫn xuất: ⚖️ |
| fineweb-2 / Common Crawl | Dữ liệu pre-train | Chưa đọc giấy phép dataset và điều khoản nguồn | ❓ | Ghi vào `DATA_SOURCES.md` từng model (P9-19) |
| Model giáo viên/nền (Qwen, Gemma, Phi…) nếu dùng | Chưng cất (P9-17, chưa chọn) | Chưa đọc | ❓ | **Đọc điều khoản trước khi chọn đường**, không đoán |

---

## D. Giấy phép thành phần mã

| Thành phần | Trạng thái |
|---|---|
| Ba repo: MIT (`LICENSE`, © 2026 BachDaThan) | ✅ file có thật (đã xem 2026-11-17) |
| Lõi OS Tầng 1–2: không dependency ngoài, **máy giữ** (`scripts/verify.py::check_kernel_dependencies`, OS P0.14) | ✅ |
| Thư viện Dart/Flutter của Y-App | `legal/THIRD_PARTY_NOTICES.md` có; đối chiếu với `pubspec.lock` khi phát hành (P9-22) |
| Thành phần ngoài dự kiến ở OS Phase P3/P6: Wine, VKD3D-Proton, DXVK, Waydroid, Box64, FEX-Emu, MangoHud | ❓ **chưa đọc giấy phép** (OS P7.2). Quy tắc: chưa có dòng trong `THIRD-PARTY.md` thì không được vào bản build nào. Không đóng gói bản Windows, Google Mobile Services hay thành phần đóng nguồn không được phép phân phối lại |
| Phần mềm gọt bởi Smart App Stripper | Chỉ gọt **cục bộ trên bản người dùng đã có**; không phân phối lại (OS §3 lưu ý P1) |

---

## E. Kết quả kiểm kỹ thuật đã làm (bằng chứng)

**E1. Quét bí mật trong lịch sử git (2026-11-17, sơ bộ).** Quét `git log --all -p` trên bản clone đầy đủ của 3 repo: YuiVieOSBeta 55 commit/10 nhánh từ xa, Adaptive-MoE 190 commit/2 nhánh, Y-App 119 commit/2 nhánh. Mẫu: khối khoá riêng (RSA/EC/OpenSSH/PGP), `hf_…`, `AIza…`, `sk-…`, `ghp_/github_pat_…`, `tvly-…`, token Discord, `AKIA…`, `xox…`, gán `api_key/secret/token/password = "<chuỗi dài>"`; và tên tệp `.jks .keystore .p12 .pfx .pem .key .env key.properties google-services.json`. **Kết quả: 0 trúng mẫu thật, 0 tệp tên nhạy cảm.** Một dòng khớp ở OS là chuỗi giả trong test (`super-secret-vault-master-password-12345`). **Giới hạn:** chỉ là regex tự viết, không thay `gitleaks`/`trufflehog`; không quét Issues/PR/Actions logs/Releases/repo Hugging Face/notebook Kaggle. Khoá Gemini/Tavily… của bạn nằm ở Kaggle Secrets nên không thuộc phạm vi quét này. → P9-24.

**E2. Kiểm test độc lập.** `YuiVieOSBeta` `main` (HEAD `179e5d9`): 11 457 check C++17 + 61 C11 = **11 518, 0 fail**, sau khi `make` trước (chạy riêng `make test` từ cây sạch thì `test_tools` lỗi 20/309 vì chưa có `yui-update`; ghi thành OS P0.15). `Adaptive-MoE`: `pytest tests/test_kb_core.py` = **13 passed** (bạn: 16,45 s; mình: 13,53 s, HEAD `c9c79d0`).

**E3. Mặc định mạng của Y-App lệch với chính sách (phát hiện mới).** `lib/state.dart`: `bool webMulti = true` và `p.getBool('webMulti') ?? true`; `lib/services/web_search.dart`: `kWebDefaultOrder` có `'ddg'` (không cần khoá) ⇒ cài mới, chưa cấu hình gì, từ khoá tìm kiếm + IP của người dùng đã đi tới `html.duckduckgo.com` khi model gọi `search`. Trong khi `assets/legal/privacy_vi.md` dòng 22 ghi tính năng mở rộng **"chỉ khi bạn cấu hình"**. Vừa lệch chính sách (minh bạch/đồng ý), vừa là rủi ro điều khoản của DuckDuckGo. → P9-21 (cùng P2-7).

**E4. Căn cứ pháp lý trong bản mẫu Y-App đã lỗi thời.** `assets/legal/privacy_vi.md:30`, `privacy_en.md:24`, `legal/LEGAL_CHECKLIST.md:18` dẫn **Nghị định 13/2023/NĐ-CP** (đã bị thay từ 01/01/2026). Không văn bản nào dẫn Luật BVDLCN 91/2025, NĐ 356/2025, NĐ 330/2026, Luật AI 134/2025, NĐ 142/2026. Ngưỡng 16 tuổi ở `terms_vi.md:13`, `terms_en.md:12`, `privacy_vi.md:33`, `LEGAL_CHECKLIST.md:19` cần xác nhận lại. → P9-22.

**E5. Chỗ làm đúng (giữ nguyên).** `web_search.dart` chỉ nói chuyện với tên miền đã biết, chặn địa chỉ nội bộ khi người dùng nhập SearXNG, không port các tầng Google/Bing/Brave HTML; `hubOn` mặc định tắt; Y-App đã có `legal_gate` buộc đồng ý khi `kLegalVersion` đổi; `LEGAL_CHECKLIST.md` đã nêu việc Google Play cấm tự cập nhật bằng APK.

---

## F. Câu hỏi cụ thể để hỏi luật sư (đưa nguyên danh sách này)

1. Yui (app chạy model cục bộ) và Yui OS (có tác tử điều khiển hệ thống, P5) thuộc mức rủi ro nào theo Điều 9 Luật AI 134/2025 và NĐ 142/2026? Người phát triển cá nhân, phát hành miễn phí mã nguồn mở có nghĩa vụ đăng ký/thông báo/đánh giá sự phù hợp nào không?
2. Xử lý hoàn toàn trên thiết bị có phải lập hồ sơ đánh giá tác động hoặc thông báo cơ quan theo Luật BVDLCN 91/2025 và NĐ 356/2025 không? Khi người dùng tự bật dịch vụ tìm kiếm ở nước ngoài, ai là bên chuyển dữ liệu?
3. Dữ liệu cá nhân của trẻ em: ngưỡng tuổi, đồng ý của cha mẹ theo khung mới (NĐ 330/2026 có nội dung về trẻ từ đủ 7 tuổi).
4. Phân phối miễn phí phần mềm mã nguồn mở có Ed25519/SHA-2/scrypt/`yui_vault`, và ISO/thiết bị chạy YuiVieOS: có chịu NĐ 58/2016/NĐ-CP (và văn bản sửa đổi/thay thế) không? Nghĩa vụ khi đặt mã ở GitHub (Mỹ)?
5. Dùng đầu ra của Gemini/Cohere/OpenRouter/Mistral để sinh dữ liệu huấn luyện một model nhỏ tiếng Việt rồi công bố công khai: điều khoản của từng bên có cho phép không?
6. Model huấn luyện trên dữ liệu CC BY-SA/ODC-By: nên gắn giấy phép nào, có nghĩa vụ chia sẻ tương tự không?
7. Gọt ứng dụng (Smart App Stripper) trên máy người dùng có hợp lệ với giấy phép phần mềm đóng nguồn không; phân phối lại bản gọt: không.
8. Đóng gói/liên kết thành phần (L)GPL (Wine, VKD3D-Proton, Waydroid) cùng ISO: nghĩa vụ cung cấp mã nguồn.
9. Điều khoản miễn trừ trách nhiệm có hiệu lực đến đâu khi AI điều khiển hệ thống và gây mất dữ liệu?
10. Nhãn hiệu "Yui", "YuiVieOS" (tra cứu trước khi dùng thương mại).

---

## G. Cổng "Public" (nối OS P7.4)

Chỉ bấm Public (repo, Releases, Kho model, store) khi **tất cả** đúng và có bằng chứng:

- [ ] Bản mẫu điều khoản/quyền riêng tư đã sửa theo §B/§E, **luật sư đã rà** (ghi ngày + tên người rà).
- [ ] Kết nối ngoài mặc định tắt cho người mới (P9-21); chính sách khớp hành vi (có test).
- [ ] Mọi dòng ❓ trong §C của dịch vụ **đang dùng** đã được đọc, ngày đọc ≤ 90 ngày.
- [ ] Mọi model trong Kho có `license` ≠ `unknown` và `DATA_SOURCES.md` (P9-19).
- [ ] Quét bí mật sạch bằng công cụ thật, mọi nhánh (P9-24, OS P7.3); khoá ký phát hành **không** nằm trên GitHub.
- [ ] Nhãn "do AI tạo" + cảnh báo rủi ro cao hiển thị (P9-20, OS P7.6).
- [ ] `THIRD-PARTY.md`/`THIRD_PARTY_NOTICES.md` khớp thành phần thật trong bản phát hành (OS P7.2).
- [ ] Trang tải giải thích cảnh báo SmartScreen/Android do khoá tự ký.
- [ ] Câu hỏi §F đã có câu trả lời bằng văn bản hoặc được ghi rõ "chấp nhận rủi ro".

---

## H. Ngoài phạm vi 3 repo nhưng cùng rủi ro (bot Yui trên Discord, HF free tier)

Bot Yui là dự án riêng, không thuộc kế hoạch này, nhưng: (1) nếu dùng Gemini API thì điều khoản ở §C (người dưới 18, model cạnh tranh, dịch vụ miễn phí dùng nội dung để cải thiện sản phẩm) áp dụng nguyên si, và Discord có người dùng nhỏ tuổi; (2) lịch sử chat nằm trên máy chủ ở nước ngoài ⇒ chuyển dữ liệu ra ngoài lãnh thổ; (3) NĐ 330/2026 có hiệu lực ngay. Nếu bạn muốn, mình lập kế hoạch pháp lý riêng cho bot.
