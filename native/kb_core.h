/* kb_core.h - LÕI KHO TRI THỨC DÙNG CHUNG (C++17, không phụ thuộc thư viện ngoài) cho Y-App (qua FFI) và yuid của YuiVieOS.
 * Cổng C (ABI ổn định, tiền tố kb_): bám sát yui_rag.py (tách từ + BM25) và yui_graph.py (Graphify). Đặc tả: docs/APP_KB.md.
 *
 * Quy ước:
 *  - Mọi chuỗi là UTF-8. Hàm trả chuỗi ghi vào bộ đệm của người gọi (out, cap) và trả về SỐ BYTE CẦN (như snprintf, chưa tính '\0');
 *    nếu trả về >= cap thì kết quả bị cắt và gọi lại với bộ đệm lớn hơn. out luôn kết thúc bằng '\0' khi cap > 0.
 *  - Lõi KHÔNG đọc/ghi file: máy chủ (Dart/yuid) tự đọc chat/ notes/ kb/ rồi đưa chữ vào; chỉ mục dựng lại trong RAM (như Python: "xoá index cũng được").
 *  - Mỗi kb_t có khoá riêng: gọi từ nhiều luồng được nhưng các lệnh trên cùng một kb_t chạy lần lượt.
 *  - Giờ hiển thị [dd/mm]: dùng múi giờ do kb_set_tz() đặt (giây lệch UTC), mặc định 0.
 */
#ifndef KB_CORE_H
#define KB_CORE_H
#include <stdint.h>
#ifdef __cplusplus
extern "C" {
#endif
#if defined(_WIN32)
#define KB_API __declspec(dllexport)
#else
#define KB_API __attribute__((visibility("default")))
#endif

#define KB_ABI 1

typedef struct kb_t kb_t;

KB_API int kb_abi(void);                                  /* KB_ABI */
KB_API kb_t* kb_new(const char* bot_name);                /* bot_name: tên hiện trước lời trợ lý ("Yui"); NULL = "Yui" */
KB_API void kb_free(kb_t* kb);
KB_API void kb_set_tz(kb_t* kb, int offset_seconds);
KB_API void kb_clear(kb_t* kb);                           /* xoá sạch chỉ mục (RAM) */

/* ---- nạp dữ liệu (src = đường dẫn tương đối của file nguồn trong kho, vd "chat/a.jsonl", "notes/meo.md", "kb/_clean/x.md") ---- */
/* Một lượt chat. conv = mã cuộc hội thoại ("chat/a.jsonl#0"). role: "user" | "assistant" | "note". Trả số đoạn đã thêm. */
KB_API int kb_add_chat_turn(kb_t* kb, const char* src, const char* conv, const char* role, const char* text, int64_t ts);
/* Tài liệu: kind "note" (notes/) hoặc "kb" (kb/_clean, kb/_tomtat). Tiêu đề lấy từ tên file như yui_rag._ingest. Trả số đoạn. */
KB_API int kb_add_doc(kb_t* kb, const char* kind, const char* src, const char* text, int64_t ts);
/* Ghi nhớ một cặp hỏi-đáp mới (cắt 4000 ký tự, gộp khoảng trắng) và cập nhật đồ thị ngay. 1 = đã thêm, 0 = bỏ (rỗng). */
KB_API int kb_remember(kb_t* kb, const char* src, const char* conv, const char* user, const char* assistant, int64_t ts);
/* Xoá mọi đoạn của một file nguồn (file bị xoá/đổi). Trả số đoạn đã xoá. */
KB_API int kb_drop_source(kb_t* kb, const char* src);
/* Dựng đồ thị đồng xuất hiện cho các đoạn thêm từ lần kb_commit trước (gọi sau khi nạp xong một mẻ, như RAG.sync). */
KB_API void kb_commit(kb_t* kb);

/* ---- tìm lại: chuỗi ngắn đưa cho model, vừa khít `chars` ký tự ---- */
/* mode 0 = BM25 (RAG.recall); 1 = Graphify (không lùi); 2 = Graphify, không đủ chắc thì lùi về BM25 ngắn (160 ký tự, 2 đoạn). k <= 0: mặc định. */
KB_API int kb_recall(kb_t* kb, const char* query, int chars, int k, int mode, char* out, int cap);

typedef struct { int chunks; int files; int terms; int facts; int entities; int superseded; } kb_stats_t;
KB_API void kb_stats(kb_t* kb, kb_stats_t* out);

/* ---- hàm kiểm thử/gỡ lỗi (JSON), dùng cho bộ test vàng so với Python ---- */
KB_API int kb_dbg_fold(const char* text, char* out, int cap);
KB_API int kb_dbg_tokens(const char* text, int bigrams, char* out, int cap);                 /* JSON ["a","b",...] */
KB_API int kb_dbg_split_text(const char* text, int maxlen, char* out, int cap);              /* JSON ["...",...] */
KB_API int kb_dbg_entities(const char* text, char* out, int cap);                            /* JSON ["p:lan","n:5000000",...] */
KB_API int kb_dbg_fact_key(const char* text, char* out, int cap);                            /* JSON chuỗi hoặc null */
KB_API int kb_dbg_intent(const char* text, char* out, int cap);                              /* tên ý định, vd "when" */
KB_API int kb_dbg_has_self(const char* text);
KB_API int kb_dbg_split_facts(const char* text, char* out, int cap);                         /* JSON ["...",...] */
KB_API int kb_dbg_search_json(kb_t* kb, const char* query, int k, const char* role, char* out, int cap);   /* BM25: [{cid,score,src,conv,i,p,r,t,ts},...] */
KB_API int kb_dbg_query_json(kb_t* kb, const char* query, int chars, int k, int abstain, char* out, int cap); /* Graphify: {context,facts,conf,intent,ids} */
KB_API int kb_dbg_chunks_json(kb_t* kb, char* out, int cap);                                 /* mọi đoạn còn sống: [{cid,src,conv,i,p,r,t,ts},...] */
KB_API int kb_dbg_facts_json(kb_t* kb, char* out, int cap);                                  /* chạy sync rồi liệt kê sự kiện còn sống: [{cid,text,who,ts,ents,key,stale},...] */

#ifdef __cplusplus
}
#endif
#endif
