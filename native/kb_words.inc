// TỰ SINH bởi tools/gen_kb_words.py từ yui_rag.py + yui_graph.py. KHÔNG sửa tay.
static const int KB_RAG_VERSION = 4;
static const char* const KB_STOP[] = {
  "a", "an", "and", "anh", "are", "ay", "ban", "be",
  "bi", "boi", "cac", "chi", "cho", "co", "con", "cua",
  "cung", "da", "dang", "day", "den", "do", "duoc", "em",
  "for", "gi", "hay", "ho", "in", "is", "it", "khi",
  "khong", "la", "lai", "len", "luon", "ma", "mik", "minh",
  "mot", "nao", "nay", "ne", "nen", "neu", "nha", "nhe",
  "nhu", "nhung", "no", "o", "of", "oi", "on", "or",
  "ra", "rat", "roi", "sao", "se", "that", "the", "thi",
  "this", "thoi", "to", "toi", "trong", "tu", "va", "van",
  "vao", "ve", "vi", "voi", "was", "were", "xuong",
};
static const size_t KB_STOP_N = sizeof(KB_STOP) / sizeof(KB_STOP[0]);
static const char* const KB_STOP_ACC[] = {
  "anh", "bạn", "bị", "bởi", "cho", "chị", "các", "còn",
  "có", "cũng", "của", "em", "gì", "hay", "họ", "khi",
  "không", "luôn", "là", "lên", "lại", "mà", "mình", "một",
  "nhé", "như", "nhỉ", "những", "nào", "này", "nè", "nên",
  "nó", "nếu", "ra", "rất", "rồi", "sao", "sẽ", "thì",
  "thôi", "thế", "trong", "tôi", "từ", "và", "vào", "vì",
  "vẫn", "về", "với", "xuống", "đang", "đây", "đã", "đó",
  "được", "đến", "ơi", "ấy", "ở",
};
static const size_t KB_STOP_ACC_N = sizeof(KB_STOP_ACC) / sizeof(KB_STOP_ACC[0]);
static const char* const KB_COMMON_CAP[] = {
  "Chiều", "Chúc", "Cũng", "Cảm", "Hôm", "Khi", "Mik", "Mình",
  "Ngày", "Nhưng", "Năm", "Nếu", "Ok", "Rồi", "Sáng", "Tháng",
  "Thì", "Thứ", "Trưa", "Tôi", "Tối", "Tớ", "Và", "Vì",
  "Ừ",
};
static const size_t KB_COMMON_CAP_N = sizeof(KB_COMMON_CAP) / sizeof(KB_COMMON_CAP[0]);
static const char* const KB_SELF_FORMS[] = {
  "mik", "minh", "mình", "toi", "tui", "tôi", "tớ",
};
static const size_t KB_SELF_FORMS_N = sizeof(KB_SELF_FORMS) / sizeof(KB_SELF_FORMS[0]);
static const char* const KB_QWORDS[] = {
  "ai", "ban", "bao", "bay", "biet", "ca", "cac", "con",
  "cu", "dai", "dau", "day", "gi", "gia", "gio", "hien",
  "hom", "ke", "khi", "khong", "la", "liet", "luc", "may",
  "moi", "nam", "nao", "ngay", "nhac", "nhieu", "nho", "nhu",
  "nhung", "noi", "phai", "sao", "tai", "tat", "thang", "the",
  "tien", "truoc", "tung", "xua",
};
static const size_t KB_QWORDS_N = sizeof(KB_QWORDS) / sizeof(KB_QWORDS[0]);
static const char* const KB_TIME_WORDS[] = {
  "bay", "con", "cung", "dao", "dau", "do", "gio", "hom",
  "luc", "ma", "moi", "nay", "nua", "roi", "sau", "thi",
  "truoc", "van", "vua",
};
static const size_t KB_TIME_WORDS_N = sizeof(KB_TIME_WORDS) / sizeof(KB_TIME_WORDS[0]);
struct KbPat { const char* name; const char* rx; };
static const KbPat KB_DURABLE[] = {
  {"ten", R"KBX(\b(ten la|doi ten|goi la|ten (con|cua))\b)KBX"},
  {"tuoi", R"KBX(\b(\d+\s*tuoi|tuoi)\b)KBX"},
  {"noi o", R"KBX(\b(song o|dang o|o tai|chuyen (den|ve|sang|toi)|don (den|sang|ve))\b)KBX"},
  {"so dien thoai", R"KBX(\bso dien thoai\b|\bsdt\b)KBX"},
  {"email", R"KBX(\bemail\b|\bgmail\b)KBX"},
  {"dia chi", R"KBX(\bdia chi\b)KBX"},
  {"cong viec", R"KBX(\b(lam nghe|lam viec (o|tai)|cong viec)\b)KBX"},
  {"thich nhat", R"KBX(\bthich nhat\b|\byeu thich nhat\b)KBX"},
};
static const size_t KB_DURABLE_N = sizeof(KB_DURABLE) / sizeof(KB_DURABLE[0]);
static const KbPat KB_INTENT[] = {
  {"history", R"KBX(\b(truoc day|luc dau|ban dau|tung|truoc kia|ngay xua|cu|hoi do)\b)KBX"},
  {"when", R"KBX(\b(khi nao|ngay nao|luc nao|hom nao|thang may|nam nao|bao gio|ngay may)\b)KBX"},
  {"howmany", R"KBX(\b(bao nhieu|may (nguoi|cai|lan|con|mon|nam|tuoi)|so luong|het bao nhieu|gia)\b)KBX"},
  {"who", R"KBX(\b(ai|ten gi|ten la gi|la ai)\b)KBX"},
  {"where", R"KBX(\b(o dau|noi nao|di dau|tai dau)\b)KBX"},
  {"list", R"KBX(\b(nhung|liet ke|tat ca|co nhung|cac)\b)KBX"},
};
static const size_t KB_INTENT_N = sizeof(KB_INTENT) / sizeof(KB_INTENT[0]);
struct KbKi { const char* name; int k; };
static const KbKi KB_K_BY_INTENT[] = {
  {"list", 6},
  {"history", 4},
  {"when", 3},
  {"howmany", 3},
  {"who", 3},
  {"where", 3},
  {"other", 4},
};
static const size_t KB_K_BY_INTENT_N = sizeof(KB_K_BY_INTENT) / sizeof(KB_K_BY_INTENT[0]);
struct KbMult { const char* unit; long long mul; };
static const KbMult KB_MULT[] = {
  {"triệu", 1000000LL},
  {"trieu", 1000000LL},
  {"tr", 1000000LL},
  {"nghìn", 1000LL},
  {"nghin", 1000LL},
  {"ngàn", 1000LL},
  {"ngan", 1000LL},
  {"k", 1000LL},
};
static const size_t KB_MULT_N = sizeof(KB_MULT) / sizeof(KB_MULT[0]);
