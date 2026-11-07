#!/usr/bin/env python3
"""
tool_data_x.py - phần MỞ RỘNG của tool_data.py: dạy model dùng py (mini-sandbox), recall/note (RAG), skill/mkskill và CHUỖI nhiều bước.
Mọi kết quả công cụ trong data đều do CHÍNH sandbox tính (py) hoặc do khuôn định dạng thật của hệ thống sinh ra (recall, [tiến độ], [tóm tắt]) => model học đúng thứ
nó sẽ thấy lúc chạy thật. Mỗi lệnh gọi nằm trên MỘT dòng (train_sft.py gộp xuống dòng thành dấu cách).

  python tool_data_x.py --write_skills ~/chat-yui/skills     # chép các skill mẫu (khớp những gì model được dạy) vào thư mục skills của bạn
"""
import os
import random
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import yui_sandbox as ys  # noqa: E402
from yui_orch import fmt_progress, fmt_summary  # noqa: E402
from yui_vault import slug  # noqa: E402

EXTRA_KINDS = ["py", "recall", "note", "skill", "mkskill", "chain"]
SIT = ["", "", "", "ê ", "cho mik hỏi ", "bạn ơi ", "xin hỏi ", "giúp mik: "]
OK_A = ["{r} nha", "ra {r} á", "kết quả là {r} đó", "{r} nè Boss", "được {r} nha", "là {r} đó"]


def py(code):
    ok, out = ys.execute(code, {"wall_s": 3, "steps": 400000})
    return out if ok else None


def money(n):
    return f"{n:,}".replace(",", ".")


def _pick(rng, xs):
    return xs[rng.randrange(len(xs))]


# ---------------------------------------------------------------- py
def gen_py(rng, max_snip):
    for _ in range(30):
        k = rng.randrange(20)
        a, b, n = rng.randint(2, 9), rng.randint(12, 9999), rng.randint(2, 200)
        if k == 0:
            n = rng.randint(10, 150); q, code = f"{a} mũ {n} bằng bao nhiêu", f"print({a}**{n})"
        elif k == 1:
            n = rng.randint(8, 40); q, code = f"{n} giai thừa là bao nhiêu", f"print(factorial({n}))"
        elif k == 2:
            x, y = rng.randint(100, 99999), rng.randint(100, 99999); q, code = f"ước chung lớn nhất của {x} và {y}", f"print(gcd({x}, {y}))"
        elif k == 3:
            x, y = rng.randint(6, 999), rng.randint(6, 999); q, code = f"bội chung nhỏ nhất của {x} và {y}", f"print(lcm({x}, {y}))"
        elif k == 4:
            x = _pick(rng, [rng.randint(10 ** 6, 10 ** 10), rng.choice([1000000007, 998244353, 999999937, 1000003, 104729, 7919, 2147483647])])
            q, code = f"{x} có phải số nguyên tố không", f"print(isprime({x}))"
        elif k == 5:
            x = rng.randint(10 ** 3, 10 ** 12); q, code = f"phân tích {x} ra thừa số nguyên tố", f"print(factor({x}))"
        elif k == 6:
            n = rng.randint(50, 20000); q, code = f"tổng các số từ 1 đến {n}", f"print(sum(range(1, {n + 1})))"
        elif k == 7:
            n = rng.randint(20, 100000); q, code = f"tổng các số nhỏ hơn {n} chia hết cho 3 hoặc 5", f"print(sum(i for i in range(1, {n}) if i % 3 == 0 or i % 5 == 0))"
        elif k == 8:
            n = rng.randint(50, 200000); q, code = f"có bao nhiêu số nguyên tố không vượt quá {n}", f"print(len(primes({n})))"
        elif k == 9:
            n = rng.randint(10, 150); q, code = f"số Fibonacci thứ {n} là bao nhiêu", f"print(fib({n}))"
        elif k == 10:
            n = rng.randint(50, 3000); q, code = f"{a} mũ {n} có bao nhiêu chữ số", f"print(len(str({a}**{n})))"
        elif k == 11:
            x = rng.randint(10 ** 7, 10 ** 15); q, code = f"tổng các chữ số của {x}", f"print(sum(digits({x})))"
        elif k == 12:
            x = rng.randint(20, 100000); f = _pick(rng, ["bin", "hex"]); q = f"đổi {x} sang " + ("nhị phân" if f == "bin" else "hệ 16"); code = f"print({f}({x})[2:])"
        elif k == 13:
            x, e, m = rng.randint(2, 999), rng.randint(10 ** 3, 10 ** 12), rng.randint(7, 99999); q, code = f"{x} mũ {e} chia {m} dư bao nhiêu", f"print(pow({x}, {e}, {m}))"
        elif k == 14:
            x, y = rng.randint(2, 30), rng.randint(2, 30); q, code = f"1/{x} + 1/{y} bằng bao nhiêu (phân số)", f"print(Fraction(1, {x}) + Fraction(1, {y}))"
        elif k == 15:
            x, y, d = rng.randint(1, 99), rng.randint(2, 99), rng.randint(5, 30); q, code = f"{x} chia {y} lấy {d} chữ số thập phân", f"print(dec(Fraction({x}, {y}), {d}))"
        elif k == 16:
            y1, m1, d1, y2, m2, d2 = rng.randint(2020, 2026), rng.randint(1, 12), rng.randint(1, 28), rng.randint(2026, 2030), rng.randint(1, 12), rng.randint(1, 28)
            if rng.random() < .5:
                q, code = f"từ {d1}/{m1}/{y1} đến {d2}/{m2}/{y2} là bao nhiêu ngày", f"print((date({y2}, {m2}, {d2}) - date({y1}, {m1}, {d1})).days)"
            else:
                q, code = f"ngày {d1}/{m1}/{y1} là thứ mấy", f"print(dow(date({y1}, {m1}, {d1})))"
        elif k == 17:
            xs = [rng.randint(1, 99) for _ in range(rng.randint(5, 9))]; s = ", ".join(map(str, xs)); w = _pick(rng, ["sort", "max", "sum"])
            q, code = {"sort": (f"sắp xếp tăng dần: {s}", f"print(sorted([{s}]))"), "max": (f"số lớn nhất trong: {s}", f"print(max([{s}]))"), "sum": (f"tổng của: {s}", f"print(sum([{s}]))")}[w]
        elif k == 18:
            w = _pick(rng, ["strawberry", "banana", "mississippi", "xin chào các bạn", "trí tuệ nhân tạo", "committee", "bookkeeper"]); c = _pick(rng, sorted(set(w.replace(" ", ""))))
            if rng.random() < .5:
                q, code = f"chữ '{c}' xuất hiện mấy lần trong '{w}'", f"print('{w}'.count('{c}'))"
            else:
                q, code = f"đảo ngược chuỗi '{w}'", f"print('{w}'[::-1])"
        elif k == 19:
            nn, kk = rng.randint(5, 60), rng.randint(2, 6)
            if rng.random() < .5:
                q, code = f"chọn {kk} người từ {nn} người có bao nhiêu cách", f"print(comb({nn}, {kk}))"
            else:
                q, code = f"xếp {kk} người trong {nn} người vào hàng có bao nhiêu cách", f"print(perm({nn}, {kk}))"
        res = py(code)
        if res is None or len(res) > max_snip:
            continue
        if k == 4:
            ans = f"có, {q.split()[0]} là số nguyên tố" if res == "True" else f"không, {q.split()[0]} không phải số nguyên tố"
        elif k == 5:
            ans = f"{q.split()[2]} = " + " × ".join(res.strip("[]").split(", "))
        else:
            ans = _pick(rng, OK_A).format(r=res)
        return [("user", _pick(rng, SIT) + q + _pick(rng, ["?", "", " vậy?"])), ("assistant", f"<tool>py {code}</tool>"), ("user", f"<result>{res}</result>"), ("assistant", ans)]
    return None


# ---------------------------------------------------------------- recall / note
NAMES = ["Mướp", "Bơ", "Tèo", "Na", "Kem", "Sushi", "Đậu", "Mít", "Lu", "Bông"]
PLACES = ["Đà Lạt", "Đà Nẵng", "Huế", "Nha Trang", "Hà Giang", "Phú Quốc", "Sa Pa", "Vũng Tàu"]
FOODS = ["phở bò", "bún chả", "cà phê sữa đá", "trà sữa", "bánh mì", "cơm tấm", "chè đậu xanh"]
FRIENDS = ["Lan", "Minh", "Hùng", "Thảo", "Nam", "Vy", "Khang", "Trang"]


def _date(rng):
    return f"{rng.randint(1, 28):02d}/{rng.randint(1, 12):02d}"


def _fact(rng):
    """(câu bạn từng nói, truy vấn tìm lại, câu hỏi hiện tại, câu trả lời kiểu mẫu, giá trị số hoặc None)"""
    k = rng.randrange(7)
    if k == 0:
        n, y = _pick(rng, NAMES), rng.randint(1, 12)
        return f"mèo nhà mik tên là {n}, nó {y} tuổi", "tên mèo", _pick(rng, ["mèo nhà mik tên gì ta", "mik từng kể tên con mèo chưa nhỉ", "bạn còn nhớ mèo của mik tên gì không"]), f"mèo của Boss tên {n}, {y} tuổi đó", None
    if k == 1:
        v = rng.randint(2, 40) * 500000; return f"tháng này mik tiết kiệm được {money(v)} đồng", "tiết kiệm", _pick(rng, ["tháng trước mik tiết kiệm được bao nhiêu nhỉ", "mik từng nói tiết kiệm bao nhiêu vậy"]), f"bạn tiết kiệm được {money(v)} đồng đó", v
    if k == 2:
        p = _pick(rng, PLACES); return f"cuối tuần rồi mik đi {p} chơi, vui lắm", "đi chơi", _pick(rng, ["lần trước mik đi chơi ở đâu ta", "mik từng đi chơi đâu nhỉ"]), f"bạn đi {p} chơi đó", None
    if k == 3:
        f = _pick(rng, FOODS); return f"món mik thích nhất là {f}", "món thích nhất", _pick(rng, ["mik thích ăn gì nhất nhỉ", "bạn nhớ món mik thích không"]), f"bạn thích nhất là {f} nha", None
    if k == 4:
        fr, d = _pick(rng, FRIENDS), rng.randint(1, 28); m = rng.randint(1, 12); return f"sinh nhật bạn {fr} là ngày {d}/{m}", f"sinh nhật {fr}", f"sinh nhật bạn {fr} là ngày nào nhỉ", f"sinh nhật {fr} là {d}/{m} đó", None
    if k == 5:
        v = rng.randint(5, 80) * 100000; return f"mik mới mua cái điện thoại giá {money(v)} đồng", "mua điện thoại giá", _pick(rng, ["cái điện thoại mik mua giá bao nhiêu nhỉ", "mik mua điện thoại hết bao nhiêu"]), f"điện thoại bạn mua giá {money(v)} đồng", v
    v = rng.randint(10, 120); return f"hôm nay mik chạy bộ được {v} phút", "chạy bộ", "mik chạy bộ bao lâu nhỉ", f"bạn chạy bộ {v} phút đó", v


def gen_recall(rng, max_snip):
    said, query, ask, ans, _ = _fact(rng)
    d = _date(rng)
    if rng.random() < 0.12:   # không có trong lịch sử: dạy model nói thật, không bịa
        res = "không tìm thấy trong lịch sử"
        return [("user", ask), ("assistant", f"<tool>recall {query}</tool>"), ("user", f"<result>{res}</result>"), ("assistant", _pick(rng, ["mik không thấy bạn kể chuyện này trong lịch sử nè, bạn nói lại cho mik nhé", "mik tìm không ra á, chắc bạn chưa kể mik nghe"]))]
    res = f"[{d}] bạn: {said}"
    if rng.random() < 0.4 and len(res) < max_snip - 40:
        res += f"\n[{d}] Yui: {_pick(rng, ['mik nhớ rồi nha', 'ok mik ghi nhớ nè', 'nghe hay quá'])}"
    if len(res) > max_snip:
        return None
    return [("user", ask), ("assistant", f"<tool>recall {query}</tool>"), ("user", f"<result>{res}</result>"), ("assistant", ans)]


def gen_note(rng, max_snip):
    k = rng.randrange(5)
    if k == 0:
        h, what = rng.randint(6, 20), _pick(rng, ["họp nhóm", "đi khám", "học tiếng Anh", "đá bóng", "gặp bạn Lan"]); title, text, q = f"lịch {what}", f"mai {h}h {what}", f"ghi nhớ giúp mik: mai {h}h {what}"
    elif k == 1:
        f = _pick(rng, FOODS); title, text, q = "món thích", f"mik thích {f}", f"lưu lại là mik thích {f} nha"
    elif k == 2:
        fr = _pick(rng, FRIENDS); d = rng.randint(1, 28); m = rng.randint(1, 12); title, text, q = f"sinh nhật {fr}", f"sinh nhật {fr} ngày {d}/{m}", f"nhớ giùm mik sinh nhật {fr} ngày {d}/{m}"
    elif k == 3:
        v = rng.randint(1, 30) * 1000000; title, text, q = "mục tiêu tiết kiệm", f"mục tiêu tiết kiệm {money(v)} đồng", f"ghi chú: mục tiêu tiết kiệm của mik là {money(v)} đồng"
    else:
        p = _pick(rng, PLACES); title, text, q = "kế hoạch du lịch", f"muốn đi {p} vào dịp tới", f"note lại là mik muốn đi {p} dịp tới"
    return [("user", q), ("assistant", f"<tool>note {title}|{text}</tool>"), ("user", f"<result>đã ghi chú: {slug(title)}</result>"), ("assistant", _pick(rng, ["mik ghi lại rồi nha", "xong rồi, mik lưu vào ghi chú nè", "ok mik nhớ giúp bạn rồi đó"]))]


# ---------------------------------------------------------------- skill
# (tên, mô tả, các bước đúng như file skill, hàm sinh tình huống)
SKILL_LIB = {
    "lai-kep": ("tính tiền gửi sau n năm với lãi kép", ["py print(round(Fraction(P) * Fraction(100 + r, 100) ** n))", "trả lời số tiền đã làm tròn"]),
    "so-nguyen-to": ("kiểm tra và phân tích một số lớn", ["py print(isprime(n), factor(n))", "nói số đó có phải nguyên tố không và các thừa số"]),
    "so-ngay": ("đếm số ngày giữa hai mốc thời gian", ["py print((date(y2, m2, d2) - date(y1, m1, d1)).days)", "trả lời số ngày"]),
    "nho-lai-gia": ("tìm giá đã mua rồi tính phần trăm", ["recall giá đã mua", "calc giá * phần trăm / 100", "trả lời kết quả"]),
    "ghi-chu-ngay": ("ghi lại việc quan trọng để nhớ sau này", ["note tiêu đề|nội dung", "báo đã ghi lại"]),
}


def skill_text(name):
    d, steps = SKILL_LIB[name]
    return f"# {name}\nmô tả: {d}\nbước:\n" + "\n".join(f"{i}. {s}" for i, s in enumerate(steps, 1)) + "\n"


def export_skills(dst):
    os.makedirs(dst, exist_ok=True)
    for n in SKILL_LIB:
        open(os.path.join(dst, n + ".md"), "w", encoding="utf-8").write(skill_text(n))
    return len(SKILL_LIB)


def _skill_line(name):
    from yui_skills import render   # đúng định dạng SkillStore.load() sẽ trả lúc chạy thật
    return render(skill_text(name), name, 700)


def gen_skill(rng, max_snip):
    k = rng.randrange(3)
    if k == 0:
        P, r, n = rng.randint(1, 50) * 1000000, rng.randint(3, 12), rng.randint(2, 10)
        q = f"gửi {money(P)} đồng lãi {r}% một năm, sau {n} năm được bao nhiêu"
        code = f"print(round(Fraction({P}) * Fraction({100 + r}, 100) ** {n}))"; res = py(code)
        skill_q, ans = "lãi kép", f"sau {n} năm bạn có {money(int(res))} đồng nha"
    elif k == 1:
        x = rng.randint(10 ** 5, 10 ** 11)
        q = f"{x} có phải số nguyên tố không, nếu không thì phân tích giúp mik"
        code = f"print(isprime({x}), factor({x}))"; res = py(code)
        skill_q, ans = "số nguyên tố", f"kết quả: {res}"
    else:
        y1, m1, d1, y2, m2, d2 = rng.randint(2022, 2026), rng.randint(1, 12), rng.randint(1, 28), rng.randint(2026, 2029), rng.randint(1, 12), rng.randint(1, 28)
        q = f"còn bao nhiêu ngày từ {d1}/{m1}/{y1} tới {d2}/{m2}/{y2}"
        code = f"print((date({y2}, {m2}, {d2}) - date({y1}, {m1}, {d1})).days)"; res = py(code)
        skill_q, ans = "số ngày", f"là {res} ngày nha"
    name = {"lãi kép": "lai-kep", "số nguyên tố": "so-nguyen-to", "số ngày": "so-ngay"}[skill_q]
    sk = _skill_line(name)
    if res is None or len(sk) > max_snip or len(res) > max_snip:
        return None
    return [("user", q), ("assistant", f"<tool>skill {skill_q}</tool>"), ("user", f"<result>{sk}</result>"), ("assistant", f"<tool>py {code}</tool>"), ("user", f"<result>{res}</result>"), ("assistant", ans)]


MK_IDEAS = [
    ("phan-tram", "tính phần trăm của một số", "py print(so * p / 100); trả lời kèm kết quả", "tính phần trăm"),
    ("doi-nhi-phan", "đổi số sang nhị phân", "py print(bin(n)[2:]); trả lời kết quả", "đổi sang nhị phân"),
    ("dem-chu", "đếm số lần một chữ xuất hiện trong chuỗi", "py print(chuoi.count(chu)); trả lời số lần", "đếm chữ trong chuỗi"),
    ("ngay-thu-may", "xem một ngày là thứ mấy", "py print(dow(date(y, m, d))); trả lời thứ", "xem ngày là thứ mấy"),
    ("nho-viec", "ghi việc cần làm vào ghi chú", "note việc cần làm|nội dung; báo đã ghi", "ghi việc cần làm"),
]


def gen_mkskill(rng, max_snip):
    name, desc, steps, what = _pick(rng, MK_IDEAS)
    q = _pick(rng, [f"lưu cách {what} thành skill giúp mik", f"tạo skill {what} đi", f"mik muốn bạn nhớ cách {what}, lưu thành skill nha"])
    return [("user", q), ("assistant", f"<tool>mkskill {name}|{desc}|{steps}</tool>"), ("user", f"<result>đã tạo skill {name}</result>"),
            ("assistant", _pick(rng, [f"xong rồi, lần sau bạn gọi skill {name} là được", f"mik lưu skill {name} rồi nha"]))]


# ---------------------------------------------------------------- chuỗi nhiều bước
def _st(tool, arg, res):
    return {"tool": tool, "arg": arg, "res": res, "ok": True}


def gen_chain(rng, max_snip):
    k = rng.randrange(4)
    if k in (0, 1):   # recall -> calc/py -> trả lời
        d = _date(rng)
        v = rng.randint(2, 40) * 500000
        mult = rng.choice([2, 3, 6, 12])
        said = f"tháng này mik tiết kiệm được {money(v)} đồng"
        res1 = f"[{d}] bạn: {said}"
        expr = f"{v}*{mult}"
        tool, code = ("calc", expr) if k == 0 else ("py", f"print({v} * {mult})")
        res2 = py(f"print({expr})")
        q = f"tháng trước mik tiết kiệm bao nhiêu, nhân {mult} lên giúp mik"
        conv = [("user", q), ("assistant", "<tool>recall tiết kiệm</tool>"), ("user", f"<result>{res1}</result>"), ("assistant", f"<tool>{tool} {code}</tool>"),
                ("user", f"<result>{res2}</result>"), ("assistant", f"{money(v)} nhân {mult} là {money(int(res2))} đồng nha")]
        if len(res1) > max_snip:
            return None
        return conv
    if k == 2:   # dạng "tiến độ": model nhỏ chỉ thấy câu hỏi + bước gần nhất kèm tiến độ rút gọn
        a, b = rng.randint(2, 50) * 100000, rng.randint(2, 50) * 100000
        steps = [_st("recall", "tiền ăn", f"[{_date(rng)}] bạn: mik tiêu {money(a)} đồng tiền ăn"), _st("recall", "tiền xăng", f"[{_date(rng)}] bạn: tiền xăng hết {money(b)} đồng")]
        prog = f"[tiến độ: {fmt_progress(steps[:1], 220)}]"
        s3 = f"{a}+{b}"; r3 = py(f"print({s3})")
        q = "tổng tiền ăn với tiền xăng của mik là bao nhiêu"
        # model đã gọi recall #2 (kết quả kèm tiến độ), rồi tính tổng
        c1 = [("user", q), ("assistant", "<tool>recall tiền xăng</tool>"), ("user", f"<result>{prog} {steps[1]['res']}</result>"), ("assistant", f"<tool>calc {s3}</tool>"),
              ("user", f"<result>{r3}</result>"), ("assistant", f"tổng là {money(int(r3))} đồng nha")]
        return c1
    # k == 3: tóm tắt cuối (hệ thống tổng hợp rồi đưa model viết câu trả lời)
    x = rng.randint(10 ** 4, 10 ** 8)
    steps = [_st("recall", "số may mắn", f"[{_date(rng)}] bạn: số may mắn của mik là {x}"), _st("py", f"print(isprime({x}))", py(f"print(isprime({x}))")),
             _st("py", f"print(factor({x}))", py(f"print(factor({x}))"))]
    note = fmt_summary(steps, 300)
    q = "số may mắn của mik có phải số nguyên tố không, phân tích luôn nha"
    last = steps[-1]
    return [("user", q), ("assistant", f"<tool>{last['tool']} {last['arg']}</tool>"), ("user", f"<result>{note}</result>"),
            ("assistant", f"số may mắn {x} " + ("là số nguyên tố" if steps[1]["res"] == "True" else f"không nguyên tố, phân tích: {last['res']}"))]


GENS = {"py": gen_py, "recall": gen_recall, "note": gen_note, "skill": gen_skill, "mkskill": gen_mkskill, "chain": gen_chain}
DEFAULT_MIX = "calc=3,time=1,search=3,direct=3,py=4,recall=3,note=1,skill=1,mkskill=1,chain=2"


def make_one(kind, rng, max_snip, max_chars=None):
    for _ in range(20):
        conv = GENS[kind](rng, max_snip)
        if conv and (max_chars is None or sum(len(t) + 8 for _, t in conv) <= max_chars):
            return conv
    return None


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--write_skills", default=None, help="chép skill mẫu vào thư mục này")
    ap.add_argument("--demo", type=int, default=0)
    a = ap.parse_args()
    if a.write_skills:
        print(f"đã chép {export_skills(os.path.expanduser(a.write_skills))} skill mẫu")
    r = random.Random(1)
    for i in range(a.demo):
        c = make_one(list(GENS)[i % len(GENS)], r, 300)
        print("\n".join(f"  {role}: {t}" for role, t in c) + "\n")
