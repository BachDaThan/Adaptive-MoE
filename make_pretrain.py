"""
make_pretrain.py - tạo văn bản tiếng Việt DÀI, SÂU, sạch để pre-train (tách riêng khỏi make_data.py, dùng lại key/provider/cào web của nó).

Khác `make_data.py pretrain` (đoạn rời 120-250 từ):
  - mỗi lần gọi LLM ra MỘT bài hoàn chỉnh ~1200 từ (nhiều đoạn, có mở - thân - kết) => ít tốn quota hơn nhiều trên mỗi token;
  - cây chủ đề nhiều tầng (lĩnh vực -> chủ đề con) x hình thức x đối tượng x góc nhìn, tự cân bằng để không lặp một lĩnh vực;
  - 2 nguồn: VIẾT LẠI tài liệu thật (Tavily/Exa/Linkup/Firecrawl/Wikipedia vi) và SÁNG TÁC từ chủ đề (kiểu sách giáo khoa);
  - lọc nhiều tầng: độ dài, markdown/lời dẫn, lộ tên AI, thiếu dấu, câu lặp, chép nguyên văn nguồn, MinHash bỏ bài gần giống
    (so với CẢ data đang có trên repo) rồi LLM chấm điểm theo lô (bỏ bài dưới ngưỡng);
  - ghi vào data/<cỡ>/N.txt, ĐÁNH SỐ TIẾP sau các file đang có (cùng định dạng: bài cách nhau dòng trống, trong bài xuống dòng đơn)
    nên train_lm.py tự nạp chung với data cũ, không cần đổi lệnh train. File cũ không bị đụng tới.

Chạy (đọc cùng biến môi trường như make_data.py: HF_TOKEN, *_API_KEY1.., LLM_ORDER, SCRAPE_ORDER, MODEL_PARAMS, HF_REPO...):
  python make_pretrain.py                       # tạo phần còn thiếu để đạt PRETRAIN_TOKENS (mặc định params x 0,2), up lên HF
  python make_pretrain.py --plan                # chỉ in kế hoạch
  python make_pretrain.py --new_tokens 5M       # tạo THÊM đúng ngần này token
  python make_pretrain.py --words 1500 --min_score 8 --synth 0.4

Biến/tuỳ chọn chính:
  PRETRAIN_TOKENS / --target_tokens   tổng token muốn có trong data/<cỡ>/ (đã có bao nhiêu thì chỉ tạo phần thiếu)
  PRETRAIN_NEW_TOKENS / --new_tokens  tạo thêm đúng ngần này (ưu tiên hơn target)
  --words 1200       độ dài mục tiêu mỗi bài (từ)
  --synth 0.35       tỉ lệ bài sáng tác từ chủ đề (phần còn lại viết lại từ tài liệu thật khi cào được)
  --min_score 7      điểm tối thiểu (1-10) khi LLM chấm; --no_judge để tắt chấm điểm (đỡ tốn quota)
  --file_chars 100000  số ký tự mỗi file N.txt;  --max_minutes  dừng sau ngần này phút (phần đã tạo vẫn lưu)
  WEB_SEARCH (off|wiki|search|tavily|both, mặc định both), WEB_TOPICS (chủ đề riêng, cách nhau dấu ;)
"""
import argparse
import os
import random
import re
import sys
import time
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import make_data as md  # noqa: E402  (dùng lại key, provider, cào web, Publisher, HFRepo)
from data_layout import data_prefix, size_tag  # noqa: E402
from providers import Empty, ModelBusy, ModelGone, QuotaOut, SearchUnavailable, Transient  # noqa: E402

# ============================ cây chủ đề: lĩnh vực -> chủ đề con ============================
TREE = {
    "khoa học tự nhiên": ["vật lý đời thường", "hoá học trong bếp và nhà", "sinh học tế bào và di truyền", "thiên văn và vũ trụ", "địa chất, núi lửa, động đất",
                          "khí hậu và thời tiết", "năng lượng và điện", "toán học trong đời sống", "đại dương và sinh vật biển"],
    "công nghệ": ["máy tính hoạt động thế nào", "internet và mạng", "trí tuệ nhân tạo cơ bản", "lập trình cho người mới", "điện thoại và viễn thông",
                  "an toàn thông tin cá nhân", "robot và tự động hoá", "lịch sử máy tính", "dữ liệu và cơ sở dữ liệu"],
    "lịch sử Việt Nam": ["thời dựng nước và Bắc thuộc", "các triều đại Lý - Trần - Lê", "nhà Nguyễn và thời Pháp thuộc", "hai cuộc kháng chiến",
                         "đổi mới và hội nhập", "danh nhân và anh hùng dân tộc", "di tích và di sản", "kinh tế - xã hội các thời kỳ"],
    "địa lý và thiên nhiên Việt Nam": ["sông ngòi và đồng bằng", "núi non và cao nguyên", "biển đảo", "rừng và hệ sinh thái", "khí hậu bốn mùa các vùng",
                                      "vườn quốc gia", "động vật quý hiếm", "thiên tai và phòng chống"],
    "văn hoá và đời sống Việt Nam": ["ẩm thực ba miền", "lễ hội và phong tục", "Tết cổ truyền", "làng nghề truyền thống", "nhạc cụ và dân ca", "trang phục truyền thống",
                                     "các dân tộc thiểu số", "chợ, phố, nếp sống đô thị", "nông thôn và nghề nông"],
    "văn học và ngôn ngữ": ["ca dao, tục ngữ, thành ngữ", "truyện Kiều và văn học trung đại", "thơ mới và văn học hiện đại", "từ nguyên và từ Hán Việt",
                            "ngữ pháp và chính tả tiếng Việt", "nghệ thuật kể chuyện", "dịch thuật", "lịch sử chữ Quốc ngữ"],
    "sức khoẻ và cơ thể": ["giấc ngủ", "dinh dưỡng cân bằng", "vận động và thể thao", "sức khoẻ tinh thần thường thức", "sơ cứu cơ bản", "bệnh thường gặp và phòng ngừa",
                           "tuổi dậy thì và lão hoá", "vệ sinh và thói quen tốt"],
    "kinh tế và tài chính cá nhân": ["tiết kiệm và chi tiêu", "lạm phát và giá cả", "ngân hàng và lãi suất", "khởi nghiệp nhỏ", "thương mại điện tử",
                                     "thuế và bảo hiểm cơ bản", "đầu tư thận trọng", "nghề nghiệp và kỹ năng làm việc"],
    "giáo dục và kỹ năng sống": ["phương pháp học hiệu quả", "tư duy phản biện", "quản lý thời gian", "giao tiếp và lắng nghe", "viết và thuyết trình",
                                 "làm việc nhóm", "học ngoại ngữ", "đọc sách có chiến lược"],
    "xã hội và tâm lý": ["gia đình và các mối quan hệ", "bạn bè và tuổi trẻ", "đô thị hoá", "truyền thông và tin giả", "đạo đức và ứng xử", "cảm xúc và cách điều chỉnh",
                         "người cao tuổi", "tình nguyện và cộng đồng"],
    "nghệ thuật và giải trí": ["âm nhạc Việt Nam", "điện ảnh và phim ảnh", "hội hoạ và điêu khắc", "nhiếp ảnh", "game và văn hoá game", "sân khấu: chèo, tuồng, cải lương",
                               "thể thao và các môn phổ biến", "kiến trúc"],
    "môi trường và đời sống xanh": ["rác thải và tái chế", "biến đổi khí hậu", "nước sạch", "năng lượng tái tạo", "nông nghiệp bền vững", "bảo tồn đa dạng sinh học",
                                    "ô nhiễm không khí đô thị", "sống tối giản"],
    "đời sống hằng ngày": ["nấu ăn và dụng cụ bếp", "sửa chữa đồ dùng trong nhà", "đi lại và giao thông", "nhà cửa và nội thất", "mua sắm thông minh", "chăm sóc thú cưng",
                           "làm vườn trên ban công", "du lịch tự túc"],
    "tư duy và triết lý": ["triết lý sống Á Đông", "logic và lập luận", "đạo đức học thường thức", "lịch sử tư tưởng", "tự do và trách nhiệm", "ý nghĩa của công việc",
                           "cách nhìn nhận thất bại", "hạnh phúc là gì"],
    "kể chuyện và sáng tác": ["truyện ngắn đời thường", "truyện cổ tích và truyền thuyết kể lại", "tuỳ bút về nơi chốn", "ký ức tuổi thơ", "truyện khoa học viễn tưởng nhẹ nhàng",
                              "truyện trinh thám nhỏ", "thư gửi người thân", "nhật ký hành trình"],
}
FORMS = [
    ("bài giảng kiểu sách giáo khoa: giải thích khái niệm từ cơ bản đến sâu, có ví dụ cụ thể, tổng kết cuối bài", "expository"),
    ("bài báo chuyên đề (feature) có bối cảnh, diễn biến, nhân vật và phân tích", "expository"),
    ("bài tuỳ bút / nghị luận có luận điểm rõ, dẫn chứng, phản biện và kết luận", "expository"),
    ("hướng dẫn chi tiết từng bước, kèm lý do đằng sau mỗi bước và lỗi thường gặp (viết thành văn xuôi, không gạch đầu dòng)", "expository"),
    ("bài giải thích dạng hỏi - đáp được viết liền mạch thành văn xuôi, từng câu hỏi gợi mở dẫn tới phần trả lời", "expository"),
    ("truyện ngắn có mở đầu, xung đột, cao trào, kết; lời thoại xen kẽ dẫn truyện", "story"),
    ("bài phân tích so sánh hai quan điểm / hai cách làm, nêu ưu nhược và kết luận cân bằng", "expository"),
    ("bài tiểu luận kể về lịch sử phát triển của một vấn đề theo trình tự thời gian", "expository"),
    ("chương sách phổ biến kiến thức cho bạn đọc phổ thông, giọng ấm áp, nhiều hình ảnh so sánh", "expository"),
    ("bài viết trải nghiệm cá nhân (ngôi thứ nhất) pha kiến thức, giọng tự nhiên, chân thực", "personal"),
]
AUDIENCES = ["học sinh cấp 2", "học sinh cấp 3", "sinh viên năm đầu", "người đi làm bận rộn", "người lớn tuổi mới làm quen chủ đề", "bạn đọc phổ thông tò mò", "người trong nghề muốn hiểu sâu hơn"]
ANGLES = ["nguồn gốc và lịch sử", "cơ chế hoạt động bên trong", "những hiểu lầm phổ biến", "ví dụ thực tế ở Việt Nam", "so sánh với cách làm/nơi khác",
          "tác động đến đời sống", "câu chuyện đằng sau", "những câu hỏi còn bỏ ngỏ", "bài học rút ra"]

LEAK = md.LEAK
META_START = re.compile(r"^\s*(dưới đây|tất nhiên|chắc chắn|đây là|sau đây|bài viết\s*:|tiêu đề\s*:|okay|sure|here)", re.I)
VI_MARKS = set("àáảãạăằắẳẵặâầấẩẫậèéẻẽẹêềếểễệìíỉĩịòóỏõọôồốổỗộơờớởỡợùúủũụưừứửữựỳýỷỹỵđ")
END_OK = tuple(".!?…”\")")


class Stop(Exception):
    pass


# ============================ lọc heuristic ============================
def clean_doc(raw, words, src=None):
    """-> (văn bản bài: các đoạn cách nhau bằng xuống dòng đơn, None) hoặc (None, lý do loại)."""
    t = re.sub(r"^```\w*\s*|\s*```$", "", raw.strip())
    t = re.sub(r"[*_`#>]+", "", t)
    lines = [re.sub(r"^\s*(?:[-•–]\s+|\d+[.)]\s+)", "", ln).strip() for ln in t.splitlines()]
    lines = [ln for ln in lines if ln]
    while lines and META_START.match(lines[0]) and len(lines[0]) < 160:
        lines.pop(0)
    if lines and len(lines[0]) < 70 and not lines[0].endswith(END_OK) and len(lines) > 3:
        lines.pop(0)   # tiêu đề cụt
    if len(lines) < 4:
        return None, "quá ít đoạn"
    text = "\n".join(lines)
    n = len(text.split())
    if n < 0.55 * words or n > 2.4 * words:
        return None, ("quá ngắn" if n < words else "quá dài")
    if not text.rstrip().endswith(END_OK):   # bị cắt giữa chừng: lùi về câu cuối trọn vẹn nếu còn >= 85%
        cut = max(text.rfind(c) for c in ".!?…")
        if cut < 0.85 * len(text):
            return None, "bị cắt cụt"
        text = text[:cut + 1]
    letters = [c for c in text if c.isalpha()]
    if not letters:
        return None, "không có chữ"
    if sum(c.isascii() for c in letters) / len(letters) > 0.88 or sum(c.lower() in VI_MARKS for c in letters) / len(letters) < 0.15:
        return None, "không phải tiếng Việt có dấu"
    if LEAK.search(text):
        return None, "lộ tên AI/nguồn"
    sents = [s.strip().lower() for s in re.split(r"(?<=[.!?…])\s+", text.replace("\n", " ")) if len(s.strip()) > 15]
    if sents and len(set(sents)) / len(sents) < 0.92:
        return None, "lặp câu"
    w = md.norm_key(text).split()
    if len(w) > 40:
        grams = Counter(" ".join(w[i:i + 4]) for i in range(len(w) - 3))
        if grams.most_common(1)[0][1] > max(4, len(w) // 150):
            return None, "lặp cụm từ"
    if src and md.too_close(text, src, n=10):
        return None, "chép nguyên văn nguồn"
    return text, None


# ============================ bỏ bài gần giống (MinHash bottom-k, thuần Python) ============================
class Dedup:
    K, SHINGLE, INDEX, THRESH = 128, 5, 8, 0.4

    def __init__(self):
        self.sigs, self.index, self.keys = [], {}, set()

    def sketch(self, text):
        w = md.norm_key(text).split()
        if len(w) < self.SHINGLE + 5:
            return None
        hs = {hash(" ".join(w[i:i + self.SHINGLE])) & 0xFFFFFFFFFFFF for i in range(len(w) - self.SHINGLE + 1)}
        return sorted(hs)[:self.K]

    def near(self, sk):
        cands = set()
        for h in sk[:self.INDEX]:
            cands.update(self.index.get(h, ()))
        sa = set(sk)
        for i in cands:
            sb = self.sigs[i]
            u = sorted(sa | set(sb))[:self.K]
            sb_set = set(sb)
            if sum(1 for x in u if x in sa and x in sb_set) / max(1, len(u)) >= self.THRESH:
                return True
        return False

    def add(self, text):
        self.keys.add(md.norm_key(text[:80]))
        sk = self.sketch(text)
        if sk is None:
            return
        self.sigs.append(sk)
        for h in sk[:self.INDEX]:
            self.index.setdefault(h, []).append(len(self.sigs) - 1)

    def is_dup(self, text):
        if md.norm_key(text[:80]) in self.keys:
            return True
        sk = self.sketch(text)
        return bool(sk) and self.near(sk)


def load_existing(remote, prefix, limit):
    d = Dedup()
    if remote is None:
        return d, 0
    try:
        names = [n for n in remote.list() if re.fullmatch(re.escape(prefix) + r"\d+\.txt", n)]
    except Exception as e:  # noqa: BLE001
        print(f"  ⚠ không đọc được repo để bỏ trùng ({str(e)[:80]})", flush=True)
        return d, 0
    names.sort(key=lambda n: int(re.search(r"(\d+)\.txt$", n).group(1)))
    docs = 0
    for n in names[-limit:]:
        try:
            for blk in remote.read(n).decode("utf-8", "ignore").split("\n\n"):
                if len(blk.strip()) >= 300:
                    d.add(blk.strip()); docs += 1
        except Exception:  # noqa: BLE001
            pass
    return d, docs


# ============================ prompt ============================
SYSTEM = ("Bạn là tác giả sách và biên tập viên tiếng Việt giàu kinh nghiệm, viết văn bản sạch, chính xác, tự nhiên để làm dữ liệu học cho mô hình ngôn ngữ. "
          "Chỉ trả về nội dung bài viết: không lời dẫn, không tiêu đề, không markdown, không gạch đầu dòng, không đánh số.")
RULES = ("Yêu cầu: tiếng Việt chuẩn, có dấu đầy đủ, đúng chính tả và ngữ pháp; khoảng {w} từ, chia 6 đến 12 đoạn, các đoạn cách nhau bằng một dòng trống; "
         "mạch lạc có mở bài, phát triển ý, kết; mỗi đoạn mang thông tin hoặc ý mới, không lặp ý, không câu sáo rỗng; "
         "chỉ nêu số liệu, năm tháng, tên người khi chắc chắn đúng, nếu không chắc thì diễn đạt chung hơn; "
         "không nhắc tới AI, mô hình ngôn ngữ, 'tài liệu', Wikipedia hay nguồn; không chào hỏi người đọc; viết trọn vẹn đến hết bài.")


def seed_prompt(words, domain, sub, form, aud, angle, search=False):
    p = (f"Viết MỘT bài hoàn chỉnh về «{sub}» (lĩnh vực: {domain}).\nHình thức: {form}.\nĐối tượng bạn đọc: {aud}.\nGóc tiếp cận chính: {angle}.\n" + RULES.format(w=words))
    if search:
        p = f"Hãy tra cứu thông tin chính xác về «{sub}» rồi viết dựa trên đó.\n" + p
    return SYSTEM, p


def rewrite_prompt(words, doc, form, aud):
    p = (f"TÀI LIỆU thật (chủ đề «{doc['title']}»):\n\"\"\"\n{doc['text'][:6000]}\n\"\"\"\n\n"
         f"Dựa trên thông tin trong tài liệu, viết MỘT bài MỚI hoàn toàn bằng lời văn của bạn (không chép nguyên câu, không chép quá 8 từ liên tiếp, "
         f"có thể sắp xếp lại ý, bổ sung giải thích và ví dụ không mâu thuẫn tài liệu, không thêm số liệu ngoài tài liệu).\n"
         f"Hình thức: {form}.\nĐối tượng bạn đọc: {aud}.\n" + RULES.format(w=words))
    return SYSTEM, p


def judge_prompt(docs):
    parts = []
    for i, d in enumerate(docs, 1):
        parts.append(f"[{i}] ({len(d.split())} từ)\n{d[:700]}\n...\n{d[-250:]}")
    p = ("Chấm điểm từng bài tiếng Việt dưới đây để dùng làm dữ liệu pre-train, thang 1-10, xét: văn tự nhiên đúng chính tả ngữ pháp; mạch lạc; "
         "không lặp ý; nội dung có giá trị và không có dấu hiệu bịa; không có lời dẫn kiểu chatbot. Chấm nghiêm: bài chỉ ổn thì 5-6, bài tốt 7-8, xuất sắc 9-10.\n\n"
         + "\n\n".join(parts) + f"\n\nChỉ trả về một mảng JSON gồm đúng {len(docs)} số nguyên theo thứ tự, ví dụ [7, 5, 8].")
    return "Bạn là biên tập viên tiếng Việt khó tính. Chỉ trả về mảng JSON.", p


# ============================ vòng chạy ============================
class Balancer:
    """Chọn lĩnh vực ít được dùng nhất (có ngẫu nhiên) để dữ liệu phủ đều."""

    def __init__(self, tree):
        self.tree, self.used = tree, Counter()

    def pick(self):
        doms = list(self.tree)
        weights = [1.0 / (1 + self.used[d]) for d in doms]
        d = random.choices(doms, weights)[0]
        self.used[d] += 1
        return d, random.choice(self.tree[d])


def parse_scores(raw, n):
    nums = re.findall(r"\d+(?:\.\d+)?", md.json_slice(raw))
    if len(nums) != n:
        return None
    return [float(x) for x in nums]


def build_args(a):
    argv = ["pretrain", "--web", a.web, "--tag", a.tag, "--repo", a.repo, "--repo_type", a.repo_type, "--out_dir", a.out_dir]
    if a.repo_dir:
        argv += ["--repo_dir", a.repo_dir]
    if a.no_upload:
        argv += ["--no_upload"]
    if a.model:
        argv += ["--model", a.model]
    return md.build_parser().parse_args(argv)


def run(a):
    random.seed(a.seed)
    md.load_kaggle_secrets()
    token = md.env("HF_TOKEN")
    margs = build_args(a)
    gem, llm_parts = md.build_llm(margs)
    if gem is None:
        sys.exit("Thiếu key LLM: đặt ít nhất một loại COHERE_API_KEY1 / OPENROUTER_API_KEY1 / MISTRAL_API_KEY1 / GEMINI_API_KEY1")
    scraper, scrape_parts = md.build_scrapers(margs)
    print("🔑 " + ", ".join(llm_parts + scrape_parts), flush=True)

    remote = md.LocalRepo(a.repo_dir) if a.repo_dir else (None if a.no_upload else md.HFRepo(a.repo, a.repo_type, token))
    params, src = md.model_params(token, remote)
    tag = md.env("MODEL_TAG") or size_tag(params)
    prefix = data_prefix(tag)
    cpt = float(md.env("CHARS_PER_TOKEN", "3.0"))
    have_chars = 0
    if remote is not None:
        try:
            _, have_chars, _ = md.count_existing(remote, tag)
        except Exception as e:  # noqa: BLE001
            sys.exit(f"Không đọc được repo dữ liệu ({str(e)[:120]}): kiểm tra HF_TOKEN / HF_REPO")
    have_tokens = int(have_chars / cpt)
    if a.new_tokens:
        need = md.parse_count(a.new_tokens)
    else:
        target = md.parse_count(a.target_tokens) if a.target_tokens else int(params * float(md.env("PRETRAIN_TOKENS_PER_PARAM", "0.2")))
        need = max(0, target - have_tokens)
    est_docs = int(need * cpt / (a.words * 5.2))   # ~5,2 ký tự/từ tiếng Việt (gồm dấu cách)
    print(f"🧭 Model {md.fmt_n(params)} (từ {src}) -> {prefix} | đang có ~{md.fmt_n(have_tokens)} token | cần thêm ~{md.fmt_n(need)} token "
          f"≈ {est_docs} bài {a.words} từ (mỗi bài ~{md.fmt_n(a.words * 5.2 / cpt)} token)", flush=True)
    if a.plan:
        return
    if need <= 0:
        print("✅ Đã đủ so với mục tiêu, không tạo thêm.")
        return

    pool = md.build_pool(gem, a.model)
    print("📋 Thứ hạng model (cao → thấp):", ", ".join(pool[:10]) or "(không có)", flush=True)
    if not pool:
        sys.exit("Không có model nào dùng được")
    dedup, n_old = load_existing(remote, prefix, a.dedup_files)
    print(f"📚 Đã nạp {n_old} bài cũ để bỏ trùng/gần giống", flush=True)

    topics = [t.strip() for t in re.split(r"[;\n]", md.env("WEB_TOPICS", "") or "") if t.strip()] or [s for subs in TREE.values() for s in subs]
    webber = None
    if a.web != "off":
        mode = a.web
        if mode == "tavily" and scraper is None:
            mode = "both"
        webber = md.WebPicker(mode, md.Wiki(margs.wiki_base), topics, scraper)
    bal = Balancer(TREE)
    pub = md.Publisher(remote, a.out_dir, prefix, upload=not a.no_upload)
    deadline = time.time() + a.max_minutes * 60 if a.max_minutes > 0 else float("inf")

    st = dict(files=0, docs=0, chars=0, rej=Counter(), fails=0, busy_since=None, streak=0)
    buf, buf_chars, pending_judge, pending = [], 0, [], None
    stop_chars = need * cpt

    def flush(force=False):
        nonlocal buf, buf_chars
        if buf and (buf_chars >= a.file_chars or (force and len(buf) >= 2)):
            pub.publish("\n\n".join(buf) + "\n")
            st["files"] += 1
            buf, buf_chars = [], 0

    def accept(doc):
        nonlocal buf_chars
        dedup.add(doc)
        buf.append(doc); buf_chars += len(doc)
        st["docs"] += 1; st["chars"] += len(doc)
        flush()

    def do_judge(force=False):
        nonlocal pending_judge
        if not pending_judge or (len(pending_judge) < a.judge_batch and not force):
            return
        batch, pending_judge = pending_judge, []
        scores = None
        for m in pool[:2]:
            try:
                raw = gem.generate(m, *judge_prompt(batch), temperature=0.0, max_tokens=200, json_mode=False)
                scores = parse_scores(raw, len(batch))
                break
            except (ModelBusy, ModelGone, Empty, Transient, QuotaOut, SearchUnavailable):
                continue
        if scores is None:
            print(f"  ⚖ không chấm được lô {len(batch)} bài, nhận theo lọc heuristic", flush=True)
            scores = [10.0] * len(batch)
        kept = 0
        for d, s in zip(batch, scores):
            if s >= a.min_score:
                accept(d); kept += 1
            else:
                st["rej"]["điểm thấp"] += 1
        print(f"  ⚖ chấm lô {len(batch)} bài: {scores} -> giữ {kept}", flush=True)

    try:
        while st["chars"] + sum(len(d) for d in pending_judge) < stop_chars:
            if time.time() > deadline:
                print(f"⏱ Hết thời gian cho phép ({a.max_minutes:g} phút); phần đã tạo vẫn được lưu.", flush=True)
                break
            now = time.time()
            cur = next((m for m in pool if gem.busy.get(m, 0.0) <= now), None)
            if cur is None:
                st["busy_since"] = st["busy_since"] or now
                if now - st["busy_since"] > 20 * 60:
                    print("❌ mọi model đều quá tải hơn 20 phút, dừng (phần đã tạo vẫn được lưu)", flush=True)
                    break
                wait = max(5.0, min(60.0, min(gem.busy.get(m, 0.0) for m in pool) - now))
                print(f"  ⏳ mọi model đang quá tải, chờ {wait:.0f}s...", flush=True)
                time.sleep(wait)
                continue
            st["busy_since"] = None

            if pending is None:   # chọn đề bài MỚI (thử lại sau lỗi thì giữ nguyên đề)
                form, kind = random.choice(FORMS)
                aud = random.choice(AUDIENCES)
                doc, use_search, domain, sub = None, False, None, None
                if webber and random.random() >= a.synth:
                    doc, topic, use_search = webber.pick(cur, gem)
                if doc:
                    pending = ("rewrite", doc, None, None, form, aud, False)
                else:
                    domain, sub = bal.pick()
                    pending = ("seed", None, domain, sub, form, aud, use_search)
            mode, doc, domain, sub, form, aud, use_search = pending
            if mode == "rewrite":
                system, prompt = rewrite_prompt(a.words, doc, form, aud)
                print(f"  📖 viết lại «{doc['title']}» ({doc.get('source', 'web')}) -> {cur}", flush=True)
            else:
                if use_search and not gem.can_search(cur):
                    use_search = False
                system, prompt = seed_prompt(a.words, domain, sub, form, aud, random.choice(ANGLES), search=use_search)
                print(f"  ✍ sáng tác «{sub}» ({domain}) -> {cur}", flush=True)
            try:
                raw = gem.generate(cur, system, prompt, temperature=a.temperature, max_tokens=min(8192, int(a.words * 5)), json_mode=False, search=use_search)
            except SearchUnavailable:
                pending = (mode, doc, domain, sub, form, aud, False)
                continue
            except ModelBusy:
                continue
            except ModelGone as e:
                print(f"  ⚠ bỏ model {cur}: {e}", flush=True)
                pool.remove(cur)
                if not pool:
                    print("❌ hết model để thử")
                    break
                continue
            except (Empty, Transient) as e:
                st["fails"] += 1
                print(f"  ⚠ lỗi tạm ({str(e)[:80]}), lần {st['fails']}", flush=True)
                if st["fails"] >= 8:
                    print("❌ lỗi liên tục, dừng (phần đã tạo vẫn được lưu)")
                    break
                time.sleep(min(30, 3 * st["fails"]))
                continue
            except QuotaOut as e:
                print(f"⛔ {e}. Dừng; phần đã tạo vẫn được lưu.", flush=True)
                break
            st["fails"], pending = 0, None
            text, why = clean_doc(raw, a.words, doc)
            if text is None or dedup.is_dup(text):
                why = why or "trùng/gần giống"
                st["rej"][why] += 1; st["streak"] += 1
                print(f"  ✗ loại: {why}", flush=True)
                if st["streak"] >= a.max_reject_streak:
                    print(f"❌ {st['streak']} bài liên tiếp bị loại, dừng để khỏi đốt quota (xem lý do ở trên; phần đã tạo vẫn được lưu)", flush=True)
                    break
                continue
            st["streak"] = 0
            dedup.add(text)   # giữ chỗ ngay để bài kế không trùng bài này (kể cả khi đang chờ chấm)
            print(f"  + bài {len(text.split())} từ ({len(text)} ký tự)", flush=True)
            if a.no_judge:
                accept(text)
            else:
                pending_judge.append(text)
                do_judge()
        do_judge(force=True)
    finally:
        flush(force=True)
    print(f"\n🎉 Xong: {st['files']} file mới, {st['docs']} bài (~{md.fmt_n(st['chars'] / cpt)} token). Loại: {dict(st['rej']) or 'không'}", flush=True)


def build_parser():
    ap = argparse.ArgumentParser(description="Tạo văn bản tiếng Việt dài để pre-train (xem đầu file)")
    ap.add_argument("--plan", action="store_true", help="chỉ in kế hoạch")
    ap.add_argument("--target_tokens", default=md.env("PRETRAIN_TOKENS"), help="tổng token muốn có (vd 20M); mặc định params x 0,2")
    ap.add_argument("--new_tokens", default=md.env("PRETRAIN_NEW_TOKENS"), help="tạo THÊM đúng ngần này token")
    ap.add_argument("--words", type=int, default=int(md.env("PRETRAIN_WORDS", "1200")))
    ap.add_argument("--synth", type=float, default=float(md.env("PRETRAIN_SYNTH", "0.35")), help="tỉ lệ bài sáng tác từ chủ đề (còn lại viết lại từ web khi có)")
    ap.add_argument("--min_score", type=float, default=float(md.env("PRETRAIN_MIN_SCORE", "7")))
    ap.add_argument("--no_judge", action="store_true", help="không dùng LLM chấm điểm (đỡ tốn quota)")
    ap.add_argument("--max_reject_streak", type=int, default=25, help="dừng nếu ngần này bài liên tiếp bị loại (tránh đốt quota vô ích)")
    ap.add_argument("--judge_batch", type=int, default=4, help="số bài mỗi lần chấm")
    ap.add_argument("--file_chars", type=int, default=int(md.env("PRETRAIN_FILE_CHARS", "100000")))
    ap.add_argument("--dedup_files", type=int, default=300, help="đọc tối đa ngần này file gần nhất để bỏ trùng")
    ap.add_argument("--web", choices=["off", "wiki", "search", "tavily", "both"], default=md.env("WEB_SEARCH", "both"))
    ap.add_argument("--temperature", type=float, default=0.9)
    ap.add_argument("--max_minutes", type=float, default=0)
    ap.add_argument("--model", default=md.env("LLM_MODEL"))
    ap.add_argument("--repo", default=md.env("HF_REPO", "BachDaThan/TrainAI-web"))
    ap.add_argument("--repo_type", default=md.env("HF_REPO_TYPE", "dataset"))
    ap.add_argument("--repo_dir", default=None, help="dùng thư mục máy làm 'repo' (để thử)")
    ap.add_argument("--no_upload", action="store_true")
    ap.add_argument("--out_dir", default="data_out")
    ap.add_argument("--tag", default=md.env("MODEL_TAG", ""))
    ap.add_argument("--seed", type=int, default=int(time.time()) % 100000)
    return ap


def main():
    a = build_parser().parse_args()
    lock = md.RunLock(os.path.join(a.out_dir, ".lock_make_pretrain"))
    if not lock.acquire():
        print("⚠ đang có một make_pretrain.py khác chạy trong thư mục này; bỏ qua để không chạy chồng.", flush=True)
        return
    try:
        run(a)
    finally:
        lock.release()


if __name__ == "__main__":
    main()
