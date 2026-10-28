#!/usr/bin/env python3
"""
tool_data.py - tạo data fine-tune DẠY MODEL DÙNG CÔNG CỤ (khớp yui_agent.py), không tốn quota Gemini.

Mỗi hội thoại có dạng (đúng khuôn user/assistant của train_sft.py):
    user: 23 nhân 17 bằng mấy?           assistant: <tool>calc 23*17</tool>
    user: <result>391</result>           assistant: 23 x 17 = 391 nha
Gồm: calc (kết quả do chính máy tính an toàn của yui_agent tính => luôn đúng), time, search (từ file sự kiện --facts, có sẵn ~30 sự kiện chắc chắn đúng),
và hội thoại thường KHÔNG dùng công cụ (--direct file hội thoại có sẵn) để model biết khi nào KHÔNG cần gọi.

  python tool_data.py --n 1500 --out_dir tool_out                         # chỉ tạo file ở máy
  python tool_data.py --n 1500 --tag 50M --repo BachDaThan/TrainAI-web     # tạo rồi up vào Fine-tune/50M/N.txt (số kế tiếp), cần HF_TOKEN
  --facts facts.jsonl : mỗi dòng {"q":"câu hỏi","query":"từ khoá","snippet":"đoạn ngắn tra được","answer":"câu trả lời ngắn"}
  --direct Fine-tune/50M/1.txt : (tuỳ chọn) hội thoại thường trộn vào làm mẫu "không cần công cụ"
LƯU Ý khi train: (1) ngữ cảnh --L phải đủ chứa hội thoại (calc/time ~50 token, search ~100-180 token) -> dùng --L 256 trở lên; đặt result_chars của
yui_agent nhỏ tương ứng (vd 120 cho L=128). (2) train_sft.py gộp xuống dòng thành dấu cách nên đối số của công cụ chỉ được ở MỘT dòng: dữ liệu này
chỉ dạy calc/time/search; công cụ py/fetch/write dành cho model lớn hơn hoặc để người dùng tự gọi.
"""
import argparse, json, os, random, re, sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from yui_agent import safe_calc, _DOW  # noqa: E402

FACTS = [
    ("thủ đô của Nhật Bản là gì", "thủ đô Nhật Bản", "Tokyo là thủ đô của Nhật Bản.", "thủ đô của Nhật Bản là Tokyo á"),
    ("thủ đô nước Pháp là thành phố nào", "thủ đô Pháp", "Paris là thủ đô của Pháp.", "là Paris đó bạn"),
    ("thủ đô của Hàn Quốc", "thủ đô Hàn Quốc", "Seoul là thủ đô của Hàn Quốc.", "Seoul nha"),
    ("thủ đô Thái Lan là gì vậy", "thủ đô Thái Lan", "Bangkok là thủ đô của Thái Lan.", "là Bangkok nè"),
    ("thủ đô nước Anh", "thủ đô Anh", "London là thủ đô của Vương quốc Anh.", "London đó bạn"),
    ("thủ đô của Đức là gì", "thủ đô Đức", "Berlin là thủ đô của nước Đức.", "Berlin nha"),
    ("núi cao nhất thế giới là núi nào", "núi cao nhất thế giới", "Everest cao khoảng 8.849 mét, là đỉnh núi cao nhất thế giới so với mực nước biển.", "là Everest, cao khoảng 8.849 m á"),
    ("hành tinh lớn nhất hệ mặt trời", "hành tinh lớn nhất hệ Mặt Trời", "Sao Mộc là hành tinh lớn nhất trong Hệ Mặt Trời.", "là Sao Mộc đó"),
    ("hành tinh nào được gọi là hành tinh đỏ", "hành tinh đỏ", "Sao Hỏa được gọi là hành tinh đỏ vì bề mặt có nhiều oxit sắt.", "là Sao Hỏa nha"),
    ("đại dương lớn nhất thế giới", "đại dương lớn nhất", "Thái Bình Dương là đại dương lớn nhất thế giới.", "là Thái Bình Dương á"),
    ("nước sôi ở bao nhiêu độ", "nhiệt độ sôi của nước", "Nước tinh khiết sôi ở 100 độ C ở áp suất mực nước biển.", "100 độ C ở mực nước biển nè"),
    ("ngày quốc khánh việt nam là ngày nào", "ngày Quốc khánh Việt Nam", "Quốc khánh Việt Nam là ngày 2 tháng 9.", "là ngày 2/9 nha"),
    ("vịnh hạ long được unesco công nhận năm nào", "Vịnh Hạ Long UNESCO", "Vịnh Hạ Long được UNESCO công nhận là Di sản thiên nhiên thế giới năm 1994.", "năm 1994 á"),
    ("thành phố nào đông dân nhất việt nam", "thành phố đông dân nhất Việt Nam", "Thành phố Hồ Chí Minh là thành phố đông dân nhất Việt Nam.", "là TP.HCM đó"),
    ("trái đất quay quanh mặt trời mất bao lâu", "Trái Đất quay quanh Mặt Trời", "Trái Đất quay quanh Mặt Trời một vòng mất khoảng 365,25 ngày.", "khoảng 365 ngày á"),
    ("mặt trăng quay quanh trái đất mất bao lâu", "chu kỳ Mặt Trăng quay quanh Trái Đất", "Mặt Trăng quay quanh Trái Đất một vòng mất khoảng 27,3 ngày.", "khoảng 27 ngày nha"),
    ("số nguyên tố nhỏ nhất là số mấy", "số nguyên tố nhỏ nhất", "Số nguyên tố nhỏ nhất là 2, và cũng là số nguyên tố chẵn duy nhất.", "là số 2 đó"),
    ("số pi xấp xỉ bao nhiêu", "số pi xấp xỉ", "Số pi xấp xỉ bằng 3,14159.", "khoảng 3,14159 nè"),
    ("một năm có bao nhiêu tháng", "một năm có bao nhiêu tháng", "Một năm dương lịch có 12 tháng.", "12 tháng nha"),
    ("một tuần có mấy ngày", "một tuần có mấy ngày", "Một tuần có 7 ngày.", "7 ngày á"),
    ("kim loại nào nhẹ nhất", "kim loại nhẹ nhất", "Liti là kim loại nhẹ nhất, khối lượng riêng khoảng 0,53 gam trên cm khối.", "là liti á"),
    ("khí nào chiếm nhiều nhất trong không khí", "thành phần không khí", "Nitơ chiếm khoảng 78 phần trăm thể tích không khí.", "là nitơ, khoảng 78% đó"),
    ("châu lục nào lớn nhất", "châu lục lớn nhất", "Châu Á là châu lục lớn nhất thế giới về diện tích và dân số.", "là châu Á nha"),
    ("sông nào dài nhất thế giới", "sông dài nhất thế giới", "Sông Nile (sông Nin) và sông Amazon thường được nêu là hai con sông dài nhất thế giới, tùy cách đo.", "thường là sông Nin hoặc Amazon, tùy cách đo á"),
    ("hà nội là thủ đô của nước nào", "Hà Nội thủ đô", "Hà Nội là thủ đô của nước Cộng hòa Xã hội Chủ nghĩa Việt Nam.", "của Việt Nam nha"),
    ("tiếng việt có bao nhiêu thanh điệu", "thanh điệu tiếng Việt", "Tiếng Việt phổ thông có 6 thanh điệu.", "6 thanh điệu á"),
    ("1 km bằng bao nhiêu mét", "1 km bằng bao nhiêu mét", "1 kilômét bằng 1.000 mét.", "1.000 mét nè"),
    ("1 giờ có bao nhiêu phút", "một giờ bao nhiêu phút", "Một giờ có 60 phút.", "60 phút nha"),
    ("con vật nào cao nhất thế giới", "động vật cao nhất", "Hươu cao cổ là loài động vật trên cạn cao nhất thế giới.", "là hươu cao cổ đó"),
    ("cá voi xanh là động vật gì", "cá voi xanh", "Cá voi xanh là động vật có vú sống dưới biển và là loài lớn nhất từng được biết đến.", "là động vật có vú, to nhất luôn á"),
]
SIT_USER = ["", "", "", "ê ", "cho mik hỏi ", "bạn ơi ", "xin hỏi ", "mik thắc mắc "]


def fmt_num(x):
    return f"{x:,}".replace(",", ".") if isinstance(x, int) and abs(x) >= 10000 else str(x)


def gen_calc(rng):
    k = rng.choice(["add", "sub", "mul", "div", "mix", "pct", "pow"])
    a, b, c = rng.randint(2, 99), rng.randint(2, 99), rng.randint(2, 20)
    if k == "add":
        a, b = rng.randint(10, 9999), rng.randint(10, 9999); ex, say = f"{a}+{b}", rng.choice([f"{a} cộng {b}", f"{a} + {b}", f"{a} với {b} cộng lại"])
    elif k == "sub":
        a, b = rng.randint(50, 9999), rng.randint(2, 49); ex, say = f"{a}-{b}", rng.choice([f"{a} trừ {b}", f"{a} - {b}", f"lấy {a} bớt đi {b}"])
    elif k == "mul":
        ex, say = f"{a}*{b}", rng.choice([f"{a} nhân {b}", f"{a} x {b}", f"{a} × {b}"])
    elif k == "div":
        a = b * rng.randint(2, 60); ex, say = f"{a}/{b}", rng.choice([f"{a} chia {b}", f"{a} / {b}", f"{a} chia cho {b}"])
    elif k == "mix":
        ex, say = f"({a}+{b})*{c}", f"({a} + {b}) × {c}"
    elif k == "pct":
        p = rng.choice([5, 10, 15, 20, 25, 30, 50]); a = rng.randint(2, 200) * 10; ex, say = f"{a}*{p}/100", f"{p}% của {a}"
    else:
        a, c = rng.randint(2, 12), rng.randint(2, 5); ex, say = f"{a}**{c}", f"{a} mũ {c}"
    r = safe_calc(ex)
    q = rng.choice(SIT_USER) + rng.choice([f"{say} bằng mấy?", f"tính {say} giúp mik", f"{say} ra bao nhiêu vậy", f"{say} = ?", f"giúp mik tính: {say}"])
    ans = rng.choice([f"{say} = {r} nha", f"ra {r} á", f"kết quả là {r} đó", f"{r} nè bạn", f"được {r} nha"])
    return [("user", q), ("assistant", f"<tool>calc {ex}</tool>"), ("user", f"<result>{r}</result>"), ("assistant", ans)]


def gen_time(rng):
    import datetime
    d = datetime.datetime(2025, 1, 1) + datetime.timedelta(days=rng.randint(0, 900), minutes=rng.randint(0, 1439))
    res = f"{_DOW[d.weekday()]}, {d:%d/%m/%Y %H:%M} (UTC+7)"
    kind = rng.choice(["hour", "dow", "date"])
    if kind == "hour":
        q, a = rng.choice(["mấy giờ rồi", "bây giờ là mấy giờ", "giờ này là mấy giờ vậy"]), rng.choice([f"bây giờ là {d:%H:%M} nha", f"{d:%H:%M} rồi đó", f"đang {d:%H:%M} nè"])
    elif kind == "dow":
        q, a = rng.choice(["hôm nay thứ mấy", "nay là thứ mấy vậy", "hôm nay là thứ mấy nhỉ"]), rng.choice([f"hôm nay là {_DOW[d.weekday()]} nha", f"{_DOW[d.weekday()]} đó bạn"])
    else:
        q, a = rng.choice(["hôm nay ngày mấy", "nay là ngày bao nhiêu", "hôm nay ngày bao nhiêu vậy"]), rng.choice([f"hôm nay là ngày {d:%d/%m/%Y} nha", f"ngày {d:%d/%m/%Y} đó"])
    return [("user", rng.choice(SIT_USER) + q + rng.choice(["?", "", " nhỉ?"])), ("assistant", "<tool>time</tool>"), ("user", f"<result>{res}</result>"), ("assistant", a)]


def gen_search(rng, facts, max_snip):
    q, query, snip, ans = rng.choice(facts)
    snip = snip[:max_snip].rsplit(" ", 1)[0] if len(snip) > max_snip else snip
    return [("user", rng.choice(SIT_USER) + q + rng.choice(["?", "", " vậy?"])), ("assistant", f"<tool>search {query}</tool>"), ("user", f"<result>{snip}</result>"), ("assistant", ans)]


def load_facts(path):
    out = list(FACTS)
    if path:
        for ln in open(path, encoding="utf-8"):
            if ln.strip():
                o = json.loads(ln)
                out.append((o["q"], o["query"], o["snippet"], o["answer"]))
    return out


def load_direct(path):
    pairs = []
    if path:
        sys.path.insert(0, HERE)
        from sft_format import parse_text
        convs, _ = parse_text(open(path, encoding="utf-8").read())
        pairs = [c for c in convs if all(len(u) < 200 and len(a) < 200 and "<tool>" not in a for u, a in c)]
    return pairs


def make(n, seed, facts, direct, max_snip, mix):
    rng = random.Random(seed)
    kinds = ["calc", "time", "search"] + (["direct"] if direct else [])
    w = [mix[k] for k in kinds]
    out, seen = [], set()
    while len(out) < n:
        k = rng.choices(kinds, w)[0]
        if k == "calc":
            conv = gen_calc(rng)
        elif k == "time":
            conv = gen_time(rng)
        elif k == "search":
            conv = gen_search(rng, facts, max_snip)
        else:
            conv = [m for u, a in rng.choice(direct) for m in (("user", u), ("assistant", a))]
        key = json.dumps(conv, ensure_ascii=False)
        if key in seen and k != "time":
            continue
        seen.add(key)
        out.append(conv)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n", type=int, default=1500, help="tổng số hội thoại")
    ap.add_argument("--per_file", type=int, default=100)
    ap.add_argument("--facts", default=None)
    ap.add_argument("--direct", default=None)
    ap.add_argument("--result_chars", type=int, default=300, help="độ dài tối đa kết quả công cụ (khớp result_chars của yui_agent)")
    ap.add_argument("--mix", default="calc=3,time=1,search=3,direct=3", help="tỉ lệ các loại")
    ap.add_argument("--out_dir", default="tool_out")
    ap.add_argument("--tag", default=None, help="cỡ model (vd 50M): up vào Fine-tune/<tag>/ theo số kế tiếp")
    ap.add_argument("--repo", default=os.environ.get("HF_REPO", "BachDaThan/TrainAI-web"))
    ap.add_argument("--repo_type", default="dataset")
    ap.add_argument("--repo_dir", default=None, help="repo là một thư mục (để thử)")
    ap.add_argument("--seed", type=int, default=1)
    a = ap.parse_args()
    mix = {k: float(v) for k, v in (kv.split("=") for kv in a.mix.split(","))}
    convs = make(a.n, a.seed, load_facts(a.facts), load_direct(a.direct), a.result_chars, mix)
    chunks = ["\n".join(json.dumps({"messages": [{"role": r, "content": t} for r, t in c]}, ensure_ascii=False) for c in convs[i:i + a.per_file]) + "\n"
              for i in range(0, len(convs), a.per_file)]
    if a.tag:
        import make_data
        from data_layout import finetune_prefix
        remote = make_data.LocalRepo(a.repo_dir) if a.repo_dir else make_data.HFRepo(a.repo, a.repo_type, os.environ.get("HF_TOKEN"))
        pub = make_data.Publisher(remote, a.out_dir, finetune_prefix(a.tag), upload=True)
        for ch in chunks:
            pub.publish(ch)
    else:
        os.makedirs(a.out_dir, exist_ok=True)
        for i, ch in enumerate(chunks, 1):
            open(os.path.join(a.out_dir, f"{i}.txt"), "w", encoding="utf-8").write(ch)
    print(f"đã tạo {len(convs)} hội thoại dùng công cụ ({len(chunks)} file)")


if __name__ == "__main__":
    main()
