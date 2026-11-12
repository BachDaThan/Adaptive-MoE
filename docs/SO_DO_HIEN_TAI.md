# SƠ ĐỒ HIỆN TẠI — Adaptive-MoE + Y-App

> Chụp ngày 2026-11-12, từ code thật trong 2 repo. Xem sơ đồ trên GitHub (hỗ trợ Mermaid) hoặc dán vào https://mermaid.live.
> Sơ đồ SAU KHI XONG KẾ HOẠCH: xem `SO_DO_SAU_KHI_XONG.md`.

## 1. Bức tranh tổng

```mermaid
flowchart TB
  subgraph AM["Repo Adaptive-MoE  (Python + C++)  -  nhà máy + chạy thử"]
    direction TB
    D1["Tạo dữ liệu<br/>make_data.py, make_pretrain.py<br/>tool_data*.py, vi_filter.py, teencode.py"]
    T1["Huấn luyện<br/>train_lm.py, train_sft.py<br/>adaptive_moe.py, fast_moe.py<br/>elastic_ctx.py, elastic_kv.py, bitnet_*.py"]
    X1["export_yui.py<br/>f32 / int8 / q4 / q4all / t2"]
    K1["kaggle_pipeline.py<br/>data → pretrain → SFT → .yui"]
    E1["yui.cpp  (CLI)<br/>engine C++ + --serve"]
    subgraph PY["Runtime Python 'Yui'  (chỉ chạy ở PC/server)"]
      direction TB
      AG["yui_agent.py<br/>ToolBox, sandbox 3 bậc, MCP, endpoint"]
      OR["yui_orch.py + yui_task.py<br/>nhiều bước, tác vụ nền"]
      KB["Bộ nhớ: yui_vault.py (mã hoá YUI_KEY)<br/>yui_rag.py (BM25), yui_kb.py<br/>yui_graph.py (Graphify), yui_skills.py"]
      WB["yui_web.py + providers.py<br/>tìm web nhiều tầng, xoay key, nghỉ 429"]
      SB["yui_sandbox.py<br/>mini-sandbox Python tự viết"]
    end
    D1 --> T1 --> X1
    K1 -.điều phối.-> D1
    K1 -.điều phối.-> T1
    K1 -.điều phối.-> X1
    E1 --> AG
    AG --> OR
    AG --> KB
    AG --> WB
    AG --> SB
  end

  subgraph HF["Hugging Face"]
    HD["Dataset repo (data train)<br/>vd BachDaThan/TrainAI-web"]
    HM["Model repo<br/>models/&lt;tên&gt;/ (config, tokenizer, .pt)<br/>+ file .yui"]
  end

  subgraph YA["Repo Y-App  (Flutter, Android + Windows)"]
    direction TB
    UI["Giao diện: Chat · Lịch sử · Model · Cài đặt"]
    ST["lib/state.dart<br/>ChatStore, ModelStore, Settings, AppState"]
    EN["lib/engine.dart<br/>isolate nền + Dart FFI"]
    AGD["lib/agent.dart<br/>3 công cụ: calc, time, search"]
    SV["services: voice, sherpa_stt, vision,<br/>update, notifier, background_guard"]
    NA["native/yui.cpp → libyui.so / yui.dll"]
    UI --> ST --> EN --> NA
    ST --> AGD
    UI --> SV
  end

  CI["GitHub Actions build-app.yml<br/>APK arm64 + Windows setup/portable"]
  GH["GitHub Releases (tag latest)"]
  WK["Wikipedia vi (công cụ search)"]

  D1 -->|đọc/ghi| HD
  T1 -->|đẩy checkpoint| HM
  X1 -->|"file .yui (v1/v2/t2)"| HM
  HM -->|"người dùng nhập file hoặc dán link (+token)"| ST
  AGD -.mạng.-> WK
  CI --> GH
  GH -.kiểm tra cập nhật mỗi 12 giờ.-> ST
  E1 <-->|"yui.cpp GIỐNG HỆT, chép tay"| NA
```

## 2. Định dạng `.yui` hiện tại (tuần tự, không có mục lục)

```mermaid
flowchart LR
  A["MAGIC 'YUI1'<br/>ver 1 hoặc 2"] --> B["Cấu hình 12 số nguyên<br/>+ top_p, min_strength"]
  B --> C["Tokenizer<br/>vocab, merges, eos,<br/>khoảng Unicode"]
  C --> D["emb  V×D"]
  D --> E["Lặp mỗi lớp"]
  E --> E1["LayerNorm1, qkv, o, LayerNorm2"]
  E1 --> E2["Router: gate E×D, strength<br/>(luôn f32)"]
  E2 --> E3["E expert (gate/up/down)<br/>+ expert chung"]
  E3 -->|lớp kế| E
  E3 --> F["norm cuối + head V×D"]
```

Mỗi ma trận (từ v2) có 1 byte kiểu: 0 f32 · 1 int8 · 2 q4 · 3 t2 (ternary). Không có bảng offset → muốn đọc riêng một expert phải tính tuần tự.

## 3. Luồng một câu hỏi trong Y-App hiện tại

```mermaid
sequenceDiagram
  participant U as Người dùng
  participant S as state.dart
  participant G as yui.cpp (FFI, isolate)
  participant T as agent.dart
  participant W as Wikipedia vi
  U->>S: gõ câu hỏi
  S->>G: lịch sử vài lượt gần nhất (contextTurns)
  G-->>S: stream chữ (có thể kèm thẻ tool)
  alt model viết lệnh công cụ
    S->>T: calc / time / search
    T->>W: chỉ khi là search (cần mạng)
    W-->>T: đoạn ngắn
    T-->>S: kết quả ≤ 120 ký tự
    S->>G: tiếp tục với kết quả (tối đa 2 lần)
  end
  S-->>U: câu trả lời (nhả chữ mượt)
```

Chạy 100% offline trừ công cụ `search`, tải model từ link và kiểm tra cập nhật.

## 4. Có gì ở đâu (ma trận tính năng)

| Tính năng | Adaptive-MoE (Python) | Y-App (Flutter) |
|---|---|---|
| Engine C++ chạy `.yui` | ✔ (CLI, `--serve`) | ✔ (FFI) |
| `yui.cpp` | bản gốc | **bản sao y hệt** (115 111 byte) |
| Công cụ calc / time / search | ✔ | ✔ |
| Công cụ fetch / read / write / py / MCP | ✔ (sandbox, danh sách cho phép) | ✘ (cố ý) |
| Kho tri thức, RAG, Graphify, skill, vault mã hoá | ✔ | **✘ (không có trong `lib/`)** |
| Làm việc nhiều bước, tác vụ nền | ✔ | ✘ |
| Đoán trước token (speculative) | ✔ trong `yui.cpp` | ✔ (Cài đặt) |
| Hiểu ảnh | ✔ | ✔ |
| Giọng nói (STT offline, TTS) | — | ✔ |
| Tải model từ link HF (+token) | ✔ (script) | ✔ từng file một |
| Danh sách/kho model, manifest | ✘ | ✘ |
| META trong `.yui`, mục lục byte | ✘ | ✘ |
| Cập nhật app | — | ✔ (GitHub Releases) |

## 5. Điểm yếu hiện tại (nguồn gốc của kế hoạch)

1. Muốn đổi model phải tự tìm file/link rồi nhập tay (chưa có kho).
2. `.yui` không tự mô tả (không biết giỏi gì, tương thích gì) và không có mục lục.
3. App không nhớ dài hạn (kho tri thức chỉ có bản Python).
4. `yui.cpp` có 2 bản chép tay → dễ lệch.
5. Ngữ cảnh model nhỏ rất ngắn (L = 128–256 token; kết quả công cụ ≤ 120 ký tự) nên mọi thứ thêm vào prompt phải nén.
