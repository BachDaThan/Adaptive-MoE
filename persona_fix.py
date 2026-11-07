"""
persona_fix.py - chuẩn hoá cách xưng hô của Yui: gọi người dùng là "Boss", tự xưng "Yui" hoặc "mình" (không "anh/chị/em/bạn").

Dùng ở 2 nơi:
  1) make_data.py gọi fix() cho MỌI câu trả lời của assistant trước khi lưu (LLM đôi khi vẫn lỡ gọi "anh").
  2) Chuyển data CŨ đã có trên HF (Fine-tune/<cỡ>/N.txt):
       python persona_fix.py --repo BachDaThan/TrainAI-web --tag 1B            # chạy thử (dry-run): chỉ báo số dòng sẽ đổi + vài ví dụ
       python persona_fix.py --repo BachDaThan/TrainAI-web --tag 1B --apply    # đổi thật; bản gốc được sao lưu vào backup/persona_anh/ trên cùng repo
       python persona_fix.py --local_dir ./Fine-tune/1B --apply                # hoặc sửa file trong thư mục máy

Chỉ đụng lượt "assistant". Lượt "user" giữ nguyên (người dùng muốn gọi gì tuỳ họ).
Quy tắc là heuristic (không phải NLP thật): tránh các từ ghép như "anh hùng", "anh chị em", "Anh" (nước Anh), "bạn bè", "bạn thân"...
Xem thử bằng dry-run trước khi --apply.
"""
import argparse
import json
import os
import re
import sys

BOSS = "Boss"

# "anh" KHÔNG phải đại từ khi theo sau là các từ này (anh hùng, anh em, anh chị, anh ấy = ngôi thứ ba, anh trai, anh ta...)
_ANH_NEXT_BLOCK = r"(?!\s*(?:hùng|em\b|chị\b|trai\b|ấy\b|ta\b|ruột\b|họ\b|kia\b|cả\b|hai\b|dũng|minh|tuấn|đào|thư|ngữ|văn|hào|linh|tú\b))"
# ...hoặc đứng sau các từ này (chị anh, em anh, các anh, người anh, của anh ấy không xét): giữ nguyên
_ANH_PREV_BLOCK = r"(?<!chị )(?<!em )(?<!các )(?<!người )(?<!ông )(?<!bà )(?<!nhà )(?<!Nhà )(?<!anh )(?<!Anh )"
_ANH_LOWER = re.compile(_ANH_PREV_BLOCK + r"\banh\b" + _ANH_NEXT_BLOCK)
# "Anh" viết hoa: chỉ đổi khi ở đầu câu và theo sau là động từ/phó từ hay gặp khi nói với người dùng (tránh "Anh" = nước Anh)
_ANH_VERBS = ("muốn", "thử", "ôn", "hay", "xem", "cứ", "đang", "có", "nên", "đừng", "cần", "chọn", "làm", "biết", "nghĩ", "thích", "hỏi", "nhớ", "ăn",
              "uống", "ngủ", "đi", "nói", "đọc", "viết", "kể", "tìm", "dùng", "mới", "đã", "sẽ", "không", "chưa", "vừa", "hãy", "phải", "được", "bị",
              "tự", "cũng", "còn", "rất", "quá", "mệt", "buồn", "vui", "khỏe", "khoẻ", "ổn", "ơi", "ơi,", "yên", "bình", "chờ", "đợi", "giỏi", "hỏi")
_ANH_UPPER = re.compile(r"(^|[.!?…]\s+|\n)Anh(?=\s+(?:" + "|".join(re.escape(v) for v in _ANH_VERBS) + r")\b)")

# "bạn" = đại từ gọi người dùng; không đổi khi là danh từ (bạn bè, bạn thân, bạn trai/gái, bạn học, người bạn, các bạn...)
_BAN_NEXT_BLOCK = r"(?!\s*(?:bè|thân|trai|gái|học|cùng|đời|đồng|tốt|cũ|mới|nhỏ|bên|ấy|ta\b|mình|cậu|đọc|nghe))"
_BAN_PREV_BLOCK = r"(?<!người )(?<!Người )(?<!các )(?<!Các )(?<!những )(?<!Những )(?<!một )(?<!Một )(?<!mấy )(?<!nhiều )(?<!hai )(?<!ba )(?<!kết )(?<!làm )(?<!của một )"
_BAN = re.compile(_BAN_PREV_BLOCK + r"\b([Bb])ạn\b" + _BAN_NEXT_BLOCK)


def fix(text):
    """Đổi cách gọi người dùng trong một câu trả lời của assistant -> 'Boss'. Trả về (text mới)."""
    if not text:
        return text
    t = _ANH_UPPER.sub(lambda m: m.group(1) + BOSS, text)
    t = _ANH_LOWER.sub(BOSS, t)
    t = _BAN.sub(BOSS, t)
    return t


def changed(text):
    return fix(text) != text


def fix_conv_line(line):
    """1 dòng JSONL {"messages": [...]} -> (dòng mới, có đổi không). Dòng không phải JSON hội thoại giữ nguyên."""
    s = line.strip()
    if not s.startswith("{") or '"messages"' not in s:
        return line, False
    try:
        obj = json.loads(s)
        msgs = obj["messages"]
    except Exception:  # noqa: BLE001
        return line, False
    ch = False
    for m in msgs:
        if m.get("role") == "assistant" and isinstance(m.get("content"), str):
            n = fix(m["content"])
            if n != m["content"]:
                m["content"], ch = n, True
    return (json.dumps(obj, ensure_ascii=False) if ch else line), ch


def fix_text(blob):
    out, n_changed, n_conv, examples = [], 0, 0, []
    for ln in blob.split("\n"):
        new, ch = fix_conv_line(ln)
        if '"messages"' in ln:
            n_conv += 1
        if ch:
            n_changed += 1
            if len(examples) < 4:
                examples.append((ln, new))
        out.append(new)
    return "\n".join(out), n_changed, n_conv, examples


def _show(examples):
    for old, new in examples:
        try:
            o, n = json.loads(old)["messages"], json.loads(new)["messages"]
            for a, b in zip(o, n):
                if a["role"] == "assistant" and a["content"] != b["content"]:
                    print("    - " + a["content"][:140]); print("    + " + b["content"][:140]); break
        except Exception:  # noqa: BLE001
            pass


def main():
    ap = argparse.ArgumentParser(description="Đổi xưng hô data SFT cũ sang 'Boss' (mặc định chạy thử, không ghi gì)")
    ap.add_argument("--repo", default=os.environ.get("HF_REPO", "BachDaThan/TrainAI-web"))
    ap.add_argument("--repo_type", default="dataset")
    ap.add_argument("--tag", default=os.environ.get("MODEL_PARAMS", "1B"))
    ap.add_argument("--local_dir", default=None, help="sửa các file N.txt trong thư mục máy thay vì HF")
    ap.add_argument("--apply", action="store_true", help="ghi thật (HF: sao lưu bản gốc rồi ghi đè)")
    a = ap.parse_args()

    if a.local_dir:
        tot_c = tot_ch = 0
        for f in sorted(os.listdir(a.local_dir)):
            if not re.fullmatch(r"\d+\.txt", f):
                continue
            p = os.path.join(a.local_dir, f)
            blob = open(p, encoding="utf-8").read()
            new, ch, n, ex = fix_text(blob)
            tot_c += n; tot_ch += ch
            if ch and a.apply:
                open(p + ".bak", "w", encoding="utf-8").write(blob)
                open(p, "w", encoding="utf-8").write(new)
        print(f"{'ĐÃ ĐỔI' if a.apply else 'sẽ đổi'} {tot_ch}/{tot_c} hội thoại")
        return

    from huggingface_hub import HfApi, hf_hub_download
    tok = os.environ.get("HF_TOKEN")
    api = HfApi(token=tok)
    files = [f for f in api.list_repo_files(a.repo, repo_type=a.repo_type) if re.fullmatch(rf"Fine-tune/{re.escape(a.tag)}/\d+\.txt", f)]
    print(f"{len(files)} file Fine-tune/{a.tag}/ trong {a.repo}")
    tot_c = tot_ch = 0
    for f in sorted(files, key=lambda x: int(re.search(r"(\d+)\.txt$", x).group(1))):
        p = hf_hub_download(a.repo, f, repo_type=a.repo_type, token=tok, force_download=True)
        blob = open(p, encoding="utf-8").read()
        new, ch, n, ex = fix_text(blob)
        tot_c += n; tot_ch += ch
        if ch:
            print(f"  {f}: {ch}/{n} hội thoại đổi")
            if not a.apply and tot_ch <= 8:
                _show(ex)
        if ch and a.apply:
            api.upload_file(path_or_fileobj=blob.encode("utf-8"), path_in_repo=f"backup/persona_anh/{f}", repo_id=a.repo, repo_type=a.repo_type,
                            commit_message=f"backup {f} trước khi đổi xưng hô")
            api.upload_file(path_or_fileobj=new.encode("utf-8"), path_in_repo=f, repo_id=a.repo, repo_type=a.repo_type,
                            commit_message=f"persona: gọi người dùng là Boss ({f})")
    print(f"{'ĐÃ ĐỔI' if a.apply else 'Sẽ đổi (chạy lại với --apply để ghi)'}: {tot_ch}/{tot_c} hội thoại")


if __name__ == "__main__":
    sys.exit(main())
