#!/usr/bin/env python3
"""
tool_data_os.py - hội thoại dạy model nhỏ dùng công cụ ĐIỀU KHIỂN HỆ ĐIỀU HÀNH (YuiVieOS) bằng tiếng Việt: có dấu, không dấu và teencode.
Sinh bằng mã xác định, KHÔNG gọi LLM. Dùng chung khuôn với tool_data.py:  assistant: <tool>tên đối-số</tool>   user: <result>…</result>

Công cụ (mỗi lệnh gọi nằm trên MỘT dòng; đối số luôn là token không dấu dù người dùng gõ kiểu nào):
  sys_power <tat_may|khoi_dong_lai|ngu|ngu_dong|khoa_man_hinh> [HH:MM]     sys_scheduler <lavd|throughput|server-rt>
  sys_app_stripper <ten_app> <bat|tat>                                       sys_proc <top|tim TÊN|dung PID>
  droid_apps | droid_launch <ten>                                            store_search <từ khoá> | store_install <id>
  server_preset init <minecraft|terraria> <RAM> <cổng>
  sh_goi_y <y_dinh>      (gợi ý lệnh POSIX từ DANH SÁCH ĐÃ DUYỆT của hệ điều hành — model không tự bịa lệnh shell; yêu cầu phá hoại -> "từ chối")

NGUYÊN TẮC AN TOÀN ĐƯỢC DẠY (khớp PLAN P4.13/P5.18 của YuiVieOS):
  * Hành động nhạy cảm (tắt/khởi động lại/ngủ, dừng tiến trình, đổi lập lịch, cài ứng dụng, mở cổng) -> công cụ chỉ trả "chờ xác nhận vật lý: …".
    Model phải nói rõ việc CHƯA xảy ra và nhắc người dùng bấm xác nhận; KHÔNG BAO GIỜ nói "xong rồi" trước khi có xác nhận.
  * Từ chối đụng vào dịch vụ lõi (yuid, nhân). Lệnh POSIX chỉ lấy từ công cụ sh_goi_y (danh sách đã duyệt); công cụ trả "từ chối" thì model từ chối và không tự đưa lệnh.
  * Cài ứng dụng: luôn nêu nhãn nguồn (chính thức/cộng đồng/chưa ký) và HỎI trước khi gửi yêu cầu cài.
ĐỊNH DẠNG KẾT QUẢ LÀ HỢP ĐỒNG DỰ KIẾN: khi MCP thật phía hệ điều hành có, sinh lại data cho khớp từng chữ.

  python tool_data_os.py --demo 12      # in thử 12 hội thoại
"""
import random
import re
import unicodedata

OS_TOOL_NAMES = ("sys_power", "sys_scheduler", "sys_app_stripper", "sys_proc", "droid_apps", "droid_launch", "store_search", "store_install", "server_preset", "sh_goi_y")
OS_TOOL_RE = r"sys_\w+|droid_\w+|store_\w+|server_preset|sh_goi_y"

TEEN = [("không", "ko"), ("được", "dc"), ("mình", "mik"), ("bạn", "bn"), ("gì", "j"), ("vậy", "v"), ("rồi", "r"), ("với", "vs"), ("biết", "bt"),
        ("bây giờ", "bh"), ("nhiều", "nhiu"), ("luôn", "lun"), ("giúp", "gjup")]
SIT = ["", "", "", "ê ", "bạn ơi ", "cho mik nhờ ", "giúp mik: ", "Yui ơi "]


def strip_accents(s):
    s = unicodedata.normalize("NFD", s.replace("đ", "d").replace("Đ", "D"))
    return "".join(c for c in s if unicodedata.category(c) != "Mn")


def variant(rng, s):
    """Câu của người dùng: 60% có dấu, 25% không dấu, 15% teencode (viết tắt + không dấu)."""
    p = rng.random()
    if p < 0.60:
        return s
    if p < 0.85:
        return strip_accents(s)
    for a, b in TEEN:
        s = re.sub(rf"\b{a}\b", b, s)
    return strip_accents(s)


def _u(rng, s):
    return ("user", variant(rng, rng.choice(SIT) + s))


def _tool(text):
    return ("assistant", f"<tool>{text}</tool>")


def _res(text):
    return ("user", f"<result>{text}</result>")


def _a(text):
    return ("assistant", text)


# ------------------------------------------------------------------ nguồn tình huống
ACTIONS = {
    "tat_may": ("tắt máy", ["tắt máy đi", "tắt máy giúp mik", "tắt máy luôn đi", "shutdown máy giùm"]),
    "khoi_dong_lai": ("khởi động lại máy", ["khởi động lại máy đi", "restart máy giúp mik", "khởi động lại giùm mik"]),
    "ngu": ("cho máy ngủ", ["cho máy ngủ đi", "đưa máy vào chế độ ngủ", "ngủ máy giúp mik"]),
    "ngu_dong": ("cho máy ngủ đông", ["cho máy ngủ đông đi", "hibernate máy giúp mik"]),
}
APPS_DROID = ["zalo", "facebook", "shopee", "tiktok", "momo", "grab"]
APPS_STRIP = [("trinh_duyet", "trình duyệt"), ("youtube", "YouTube"), ("facebook", "Facebook"), ("zalo", "Zalo")]
PROCS = [("firefox", 1.2), ("code", 0.8), ("zalo", 0.45), ("chrome", 1.5), ("obs", 0.6), ("steam", 0.7), ("discord", 0.5), ("telegram", 0.3)]
STORE = [
    ("trình duyệt nhẹ", [("firefox", "Firefox", "Nguồn chính thức: nhà phát hành", "đã ký"), ("falkon", "Falkon", "Nguồn cộng đồng: kho mở", "đã ký")]),
    ("trình xem pdf", [("sumatrapdf", "Sumatra PDF", "Nguồn chính thức: GitHub Releases", "đã ký"), ("okular", "Okular", "Nguồn cộng đồng: Flathub", "đã ký")]),
    ("trình phát nhạc", [("vlc", "VLC", "Nguồn chính thức: nhà phát hành", "đã ký"), ("mpv", "mpv", "Nguồn cộng đồng: kho mở", "đã ký")]),
    ("soạn thảo văn bản", [("libreoffice", "LibreOffice", "Nguồn chính thức: nhà phát hành", "đã ký"), ("abiword", "AbiWord", "Nguồn cộng đồng: kho mở", "chưa ký")]),
]
POSIX_OK = [
    ("liệt kê file lớn hơn 100MB trong thư mục này", "file_lon", "find . -type f -size +100M", "chỉ đọc", "lệnh này chỉ liệt kê file lớn hơn 100 MB, không xóa gì hết"),
    ("xem thư mục nào chiếm nhiều dung lượng nhất", "dung_luong", "du -sh * | sort -h", "chỉ đọc", "lệnh này tính dung lượng từng mục rồi xếp từ nhỏ đến lớn, dòng cuối là nặng nhất"),
    ("tìm file tên báo cáo trong thư mục hiện tại", "tim_file", "find . -iname '*baocao*'", "chỉ đọc", "lệnh này tìm file có tên chứa baocao, không phân biệt hoa thường"),
    ("đếm số dòng của file ghi_chu.txt", "dem_dong", "wc -l ghi_chu.txt", "chỉ đọc", "lệnh này in số dòng của file"),
    ("tìm chữ lỗi trong file nhat_ky.log", "tim_chu", "grep -n 'lỗi' nhat_ky.log", "chỉ đọc", "lệnh này in các dòng chứa chữ lỗi kèm số dòng"),
    ("nén thư mục anh thành file anh.tar.gz", "nen_thu_muc", "tar -czf anh.tar.gz anh", "ghi dữ liệu, kiểm tra đường dẫn trước", "lệnh này tạo file anh.tar.gz, thư mục gốc vẫn còn nguyên"),
    ("xem những cổng mạng đang mở trên máy", "cong_mo", "ss -tuln", "chỉ đọc", "lệnh này liệt kê cổng TCP/UDP đang lắng nghe"),
    ("xem 50 dòng cuối của file nhat_ky.log", "duoi_log", "tail -n 50 nhat_ky.log", "chỉ đọc", "lệnh này in 50 dòng cuối của file"),
    ("copy thư mục tai_lieu sang thư mục sao_luu", "sao_chep", "cp -r tai_lieu sao_luu", "ghi dữ liệu, kiểm tra đường dẫn trước", "lệnh này sao chép cả thư mục, bản gốc vẫn còn"),
    ("xem tiến trình tên firefox", "tim_tien_trinh", "pgrep -a firefox", "chỉ đọc", "lệnh này in pid và dòng lệnh của các tiến trình tên firefox"),
]
POSIX_BAD = ["xóa sạch toàn bộ dữ liệu trên ổ cứng", "format luôn ổ đĩa chính", "xóa hết mọi thứ trong thư mục gốc cho nhanh", "tắt hết bảo vệ rồi xóa file hệ thống"]
SYSTEM_PROCS = ["yuid", "nhân hệ điều hành", "dịch vụ lõi của hệ thống"]


def _clock(rng):
    return f"{rng.randint(0, 23):02d}:{rng.choice(['00', '15', '30', '45'])}"


def _say_time(hhmm):
    h, m = hhmm.split(":")
    return f"{int(h)}h" + (m if m != "00" else "")


# ------------------------------------------------------------------ các kiểu hội thoại
def gen_power(rng, _max):
    key = rng.choice(list(ACTIONS))
    name, says = ACTIONS[key]
    if rng.random() < 0.5:
        return [_u(rng, rng.choice(says)), _tool(f"sys_power {key}"), _res(f"chờ xác nhận vật lý: {name} ngay"),
                _a(rng.choice([f"mik đã gửi yêu cầu {name} rồi nhưng CHƯA làm đâu nha Boss — Boss bấm xác nhận trên màn hình thì máy mới {name.replace('cho máy ', '')}",
                               f"yêu cầu {name} đang chờ Boss bấm xác nhận trên màn hình nha, chưa xong đâu"]))]
    t = _clock(rng)
    return [_u(rng, f"hẹn {_say_time(t)} {name} giúp mik"), _tool(f"sys_power {key} {t}"), _res(f"đã đặt yêu cầu hẹn giờ, chờ xác nhận vật lý: {name} lúc {t}"),
            _a(f"mik đã đặt yêu cầu {name} lúc {t}, nhưng cần Boss bấm xác nhận trên màn hình thì lịch mới có hiệu lực nha")]


def gen_lock(rng, _max):
    return [_u(rng, rng.choice(["khóa màn hình giúp mik", "khóa máy lại đi", "lock màn hình"])), _tool("sys_power khoa_man_hinh"), _res("đã khóa màn hình"),
            _a(rng.choice(["khóa rồi nha Boss", "đã khóa màn hình nè"]))]


def gen_sched(rng, _max):
    mode, say, ans = rng.choice([
        ("throughput", "máy đang biên dịch lớn, chuyển sang chế độ chạy tối đa đa nhân đi", "chế độ throughput (ưu tiên tổng công việc, giao diện có thể chậm hơn)"),
        ("server-rt", "máy chủ game bị giật, chuyển sang chế độ ưu tiên độ trễ cho dịch vụ", "chế độ server-rt (ưu tiên độ trễ cho dịch vụ, có trần CPU để giao diện không bị đói)"),
        ("lavd", "đưa lập lịch về chế độ mặc định giúp mik", "chế độ mặc định lavd"),
    ])
    return [_u(rng, say), _tool(f"sys_scheduler {mode}"), _res(f"chờ xác nhận vật lý: chuyển lập lịch sang {mode}"),
            _a(f"mik đã gửi yêu cầu chuyển sang {ans}, Boss bấm xác nhận trên màn hình thì mới đổi nha")]


def gen_strip(rng, _max):
    app, name = rng.choice(APPS_STRIP)
    on = rng.random() < 0.4
    say = f"bật lại gọt quảng cáo cho {name}" if on else rng.choice([f"{name} bị lỗi, tắt gọt quảng cáo đi", f"tắt gọt cho {name} giúp mik, trang hiển thị sai"])
    res = f"đã {'bật' if on else 'tắt'} gọt cho {app} (đổi lại được bất cứ lúc nào)"
    return [_u(rng, say), _tool(f"sys_app_stripper {app} {'bat' if on else 'tat'}"), _res(res),
            _a(f"xong rồi nha Boss, mik đã {'bật' if on else 'tắt'} gọt cho {name}" + ("" if on else " — nếu hết lỗi mà muốn bật lại cứ nói mik"))]


def gen_top(rng, _max):
    ps = rng.sample(PROCS, 3)
    ps.sort(key=lambda x: -x[1])
    res = " | ".join(f"{i + 1}) {n} {g:.1f} GB" for i, (n, g) in enumerate(ps))
    return [_u(rng, rng.choice(["máy chậm quá, cái gì đang ăn RAM vậy", "xem ứng dụng nào đang dùng nhiều RAM nhất", "cái nào nặng máy nhất"])),
            _tool("sys_proc top"), _res(res),
            _a(f"đang ngốn RAM nhiều nhất là {ps[0][0]} ({ps[0][1]:.1f} GB), kế đến {ps[1][0]} và {ps[2][0]} nha Boss — muốn mik dừng cái nào thì nói, mik sẽ gửi yêu cầu để Boss xác nhận")]


def gen_kill(rng, _max):
    n, g = rng.choice(PROCS)
    pid = rng.randint(1000, 9999)
    return [_u(rng, f"tắt {n} đang bị treo giúp mik"), _tool(f"sys_proc tim {n}"), _res(f"{n} (pid {pid}) {g:.1f} GB"), _tool(f"sys_proc dung {pid}"),
            _res(f"chờ xác nhận vật lý: dừng tiến trình {n} (pid {pid})"),
            _a(f"mik đã gửi yêu cầu dừng {n} nhưng CHƯA dừng đâu nha — Boss bấm xác nhận trên màn hình thì mới dừng (nhớ lưu việc đang làm trước)")]


def gen_refuse_core(rng, _max):
    n, pid = rng.choice([("yuid", 1), ("nhân hệ điều hành", 0), ("dịch vụ lõi của hệ thống", 2)])
    key = "yuid" if n == "yuid" else ("nhan" if pid == 0 else "dich_vu_loi")
    return [_u(rng, f"tắt luôn {n} đi cho nhẹ máy"), _tool(f"sys_proc tim {key}"), _res(f"{n} (pid {pid}) — dịch vụ lõi, không được dừng"),
            _a(f"mik không tắt {n} được nha Boss — đó là phần lõi của hệ thống, tắt là máy mất điều khiển. Nếu máy nặng thì mik xem ứng dụng nào ngốn RAM rồi gửi yêu cầu dừng cái đó cho Boss xác nhận nhé")]


def gen_droid(rng, _max):
    if rng.random() < 0.35:
        apps = rng.sample(APPS_DROID, 3)
        return [_u(rng, "trong hộp cát Android đang có app nào"), _tool("droid_apps"), _res(", ".join(apps)),
                _a(f"đang có {apps[0]}, {apps[1]} và {apps[2]} nha Boss, muốn mở cái nào nói mik")]
    app = rng.choice(APPS_DROID)
    return [_u(rng, f"mở {app} bản android giúp mik"), _tool(f"droid_launch {app}"), _res(f"đã dựng hộp cát Android và mở {app}"),
            _a(f"mik mở {app} trong hộp cát Android rồi nha Boss — đóng cửa sổ là hộp cát tự tắt cho đỡ tốn RAM")]


def gen_store_search(rng, _max):
    q, items = rng.choice(STORE)
    res = " | ".join(f"{i + 1}) {n} — {src}, {sg}" for i, (_id, n, src, sg) in enumerate(items))
    first = items[0]
    return [_u(rng, f"tìm {q} trong kho ứng dụng"), _tool(f"store_search {q}"), _res(res),
            _a(f"mik thấy {first[1]} ({first[2].lower()}, {first[3]}) và {items[1][1]} ({items[1][2].lower()}, {items[1][3]}). Boss muốn mik gửi yêu cầu cài cái nào không?")]


def gen_store_install(rng, _max):
    q, items = rng.choice(STORE)
    _id, n, src, sg = rng.choice(items)
    unsigned = sg == "chưa ký"
    res = f"chờ xác nhận vật lý: cài {n} — {src}, " + ("tệp chưa ký, sẽ chạy trong hộp cát hạn chế không mạng" if unsigned else "đã ký")
    ans = (f"mik đã gửi yêu cầu cài {n}, Boss bấm xác nhận trên màn hình thì mới cài nha. Lưu ý {n} chưa có chữ ký nên sẽ chạy trong hộp cát hạn chế, không có mạng"
           if unsigned else f"mik đã gửi yêu cầu cài {n} ({src.lower()}), chờ Boss bấm xác nhận trên màn hình nha")
    return [_u(rng, f"cài {n} giúp mik"), _tool(f"store_install {_id}"), _res(res), _a(ans)]


def gen_server(rng, _max):
    game, port, eula = rng.choice([("minecraft", 25565, " Boss nhớ tự đọc và chấp nhận EULA của Mojang trước khi mở cho người khác vào nha."), ("terraria", 7777, "")])
    ram = rng.choice(["2G", "4G", "6G"])
    return [_u(rng, f"dựng server {game} {ram.replace('G', ' G')} ram cổng {port} giúp mik"), _tool(f"server_preset init {game} {ram} {port}"),
            _res(f"đã tạo cấu hình {game}; yêu cầu mở cổng {port} đang chờ xác nhận vật lý — cổng vẫn ĐÓNG cho tới khi bạn xác nhận"),
            _a(f"mik tạo cấu hình {game} rồi nhưng cổng {port} vẫn đóng nha — Boss bấm xác nhận trên màn hình thì mới mở cho người ngoài vào.{eula}")]


def gen_chain_recall(rng, max_snip):
    key = rng.choice(list(ACTIONS))
    name = ACTIONS[key][0]
    t = _clock(rng)
    d = f"2026-{rng.randint(9, 11):02d}-{rng.randint(1, 28):02d}"
    res1 = f"[{d}] bạn: tối nay {name} lúc {t} nha"
    if len(res1) > max_snip:
        return None
    return [_u(rng, f"lần trước mik dặn {name} lúc mấy giờ, hẹn lại giúp mik"), _tool(f"recall hẹn {name}"), _res(res1), _tool(f"sys_power {key} {t}"),
            _res(f"đã đặt yêu cầu hẹn giờ, chờ xác nhận vật lý: {name} lúc {t}"),
            _a(f"hôm {d[8:]}/{d[5:7]} Boss dặn {name} lúc {t}, mik đã đặt lại yêu cầu rồi — Boss bấm xác nhận trên màn hình thì lịch mới có hiệu lực nha")]


def gen_chain_search(rng, max_snip):
    snip = "Sumatra PDF, MuPDF và Okular là các trình xem PDF nhẹ, hợp với máy cấu hình thấp."
    if len(snip) > max_snip:
        return None
    return [_u(rng, "tìm xem trình xem pdf nào nhẹ rồi tìm trong kho giúp mik"), _tool("search trình xem pdf nhẹ cho máy yếu"), _res(snip),
            _tool("store_search sumatra pdf"), _res("1) Sumatra PDF — Nguồn chính thức: GitHub Releases, đã ký"),
            _a("trên web người ta hay khuyên Sumatra PDF cho máy yếu, và mik thấy nó trong kho (nguồn chính thức, đã ký). Boss muốn mik gửi yêu cầu cài không?")]


def gen_posix(rng, _max):
    if rng.random() < 0.8:
        say, intent, cmd, kind, why = rng.choice(POSIX_OK)
        return [_u(rng, say), _tool(f"sh_goi_y {intent}"), _res(f"lệnh gợi ý ({kind}): {cmd}"), _a(f"`{cmd}` nha Boss — {why}" + ("" if kind == "chỉ đọc" else ". Boss kiểm tra kỹ đường dẫn trước khi chạy nhé"))]
    return [_u(rng, rng.choice(POSIX_BAD)), _tool("sh_goi_y pha_hoai"), _res("từ chối: yêu cầu phá hoại dữ liệu, không có lệnh nào được duyệt"),
            _a("mik không đưa lệnh đó nha Boss, nó sẽ xóa sạch dữ liệu và rất khó lấy lại. Nếu Boss muốn dọn dẹp, nói rõ thư mục hoặc file nào, mik chỉ lệnh an toàn và nhắc kiểm tra đường dẫn trước")]


GENS_OS = [(gen_power, 3), (gen_lock, 1), (gen_sched, 2), (gen_strip, 2), (gen_top, 2), (gen_kill, 2), (gen_refuse_core, 1), (gen_droid, 2),
           (gen_store_search, 2), (gen_store_install, 2), (gen_server, 2), (gen_chain_recall, 2), (gen_chain_search, 1), (gen_posix, 3)]


def gen_os(rng, max_snip=300):
    fns, w = zip(*GENS_OS)
    return rng.choices(fns, w)[0](rng, max_snip)


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--demo", type=int, default=10)
    ap.add_argument("--seed", type=int, default=1)
    a = ap.parse_args()
    r = random.Random(a.seed)
    shown = 0
    while shown < a.demo:
        c = gen_os(r)
        if c:
            print("\n".join(f"  {role}: {t}" for role, t in c) + "\n")
            shown += 1
