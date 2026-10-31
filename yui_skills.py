#!/usr/bin/env python3
"""
yui_skills.py - SKILL cho model Yui: công thức làm việc ngắn (các bước + công cụ nên dùng) lưu trong thư mục skills/ của kho người dùng.

  * Người dùng chép file .md vào skills/ là dùng được (như chép lịch sử chat). Kho có mã hoá thì file chưa mã hoá vẫn đọc được, seal() sẽ mã hoá sau.
  * Mỗi skill là một file  skills/<tên>.md  (hoặc skills/<tên>/SKILL.md):
        # tên-skill
        mô tả: tóm tắt tính kiệm một dòng
        khi nào: gợi ý lúc nào dùng (tuỳ chọn)
        bước:
        1. recall <từ khoá>
        2. py <mã tính toán>
        3. trả lời ngắn gọn từ kết quả
  * Model có 3 thao tác (cú pháp một đối số, hợp model nhỏ):
        <tool>skill</tool>                         liệt kê skill (tên: mô tả)
        <tool>skill tên-hoặc-từ-khoá</tool>        nạp công thức (khớp tên, nếu không thì tìm theo mô tả)
        <tool>mkskill tên|mô tả|bước 1; bước 2</tool>   tạo skill mới (không ghi đè skill đã có, đánh dấu nguồn: model)
  * Skill chỉ là LỜI KHUYÊN dạng chữ: không cấp thêm quyền. Mọi tool nó nhắc tới vẫn bị khung chạy (profile, sandbox, xác nhận) kiểm soát như thường.
    Nội dung skill đi qua sanitize() trước khi đưa cho model nên không thể giả lệnh <tool>.
"""
import math
import os
import re

from yui_rag import fold, tokens
from yui_vault import VaultError, slug

MAX_SKILL_CHARS = 4000
MAX_SKILLS = 300


def render(text, name, chars=700):
    """Dạng gọn đưa cho model: bỏ tiêu đề + dòng mô tả/nguồn/khi nào, giữ các bước.  'lai-kep: 1. py ... 2. trả lời ...'"""
    keep = [ln.strip() for ln in text.splitlines()
            if ln.strip() and not ln.lstrip().startswith("#") and not re.match(r"^\s*(mô tả|mo ta|description|nguồn|nguon|khi nào|khi nao|when|bước|buoc|steps)\s*[:：]\s*$", ln, re.I)
            and not re.match(r"^\s*(mô tả|mo ta|description|nguồn|nguon|khi nào|khi nao|when)\s*[:：]", ln, re.I)]
    return f"{name}: {' '.join(' '.join(keep).split())}"[:chars]


class SkillStore:
    def __init__(self, ws):
        self.ws = ws

    def _files(self):
        out = {}
        for p, _, _ in self.ws.list_files("skills", exts=(".md", ".txt")):
            base = os.path.basename(p)
            base = base[:-4] if base.endswith(".enc") else base
            name = os.path.basename(os.path.dirname(p)) if base.lower().startswith("skill.") else re.sub(r"\.(md|txt)$", "", base, flags=re.I)
            out.setdefault(slug(name), p)
        return out

    @staticmethod
    def parse(text, name):
        desc, when = "", ""
        for ln in text.splitlines():
            m = re.match(r"^\s*(mô tả|mo ta|description)\s*[:：]\s*(.+)$", ln, re.I)
            if m and not desc:
                desc = m.group(2).strip()
            m = re.match(r"^\s*(khi nào|khi nao|when)\s*[:：]\s*(.+)$", ln, re.I)
            if m and not when:
                when = m.group(2).strip()
        if not desc:
            for ln in text.splitlines():
                if ln.strip() and not ln.lstrip().startswith("#"):
                    desc = ln.strip()[:120]
                    break
        return {"name": name, "desc": desc[:160], "when": when[:160], "text": text}

    def list(self):
        out = []
        for name, p in sorted(self._files().items())[:MAX_SKILLS]:
            try:
                out.append(self.parse(self.ws.read_text(p), name))
            except (VaultError, OSError) as e:
                self.ws.log_error("skills.read", f"{name}: {e}")
        return out

    def listing(self, chars=300):
        sk = self.list()
        if not sk:
            return "chưa có skill nào"
        line, out = "", []
        for s in sk:
            item = f"{s['name']}: {s['desc']}" if s["desc"] else s["name"]
            if len(line) + len(item) + 3 > chars:
                out.append("…")
                break
            out.append(item)
            line += item + "; "
        return "; ".join(out)

    def find(self, query):
        """Khớp tên chính xác (không dấu), nếu không thì xếp hạng theo mô tả. Trả skill hoặc None."""
        q = slug(query)
        sk = self.list()
        for s in sk:
            if s["name"] == q:
                return s
        qt = set(tokens(query, bigrams=False))
        best, bs = None, 0.0
        for s in sk:
            dt = set(tokens(f"{s['name'].replace('-', ' ')} {s['desc']} {s['when']}", bigrams=False))
            sc = sum(1.0 / math.log(2 + len(t)) for t in qt & dt)
            if sc > bs:
                best, bs = s, sc
        return best

    def load(self, query, chars=700):
        s = self.find(query)
        if not s:
            return "không có skill phù hợp. " + self.listing(200)
        return render(s["text"], s["name"], chars)

    def create(self, name, desc, steps, source="model"):
        name = slug(name, "skill")
        desc = " ".join(str(desc).split())[:160]
        steps = [x.strip() for x in re.split(r"[;\n]+", str(steps)) if x.strip()][:12]
        if not desc or not steps:
            raise VaultError("skill cần tên|mô tả|các bước (cách nhau bằng dấu ;)")
        if name in self._files():
            raise VaultError(f"đã có skill '{name}' (không ghi đè)")
        if len(self._files()) >= MAX_SKILLS:
            raise VaultError("đã đủ số skill tối đa")
        text = f"# {name}\nmô tả: {desc}\nnguồn: {source}\nbước:\n" + "\n".join(f"{i}. {s}" for i, s in enumerate(steps, 1)) + "\n"
        if len(text) > MAX_SKILL_CHARS:
            raise VaultError("skill quá dài")
        self.ws.write_text(f"{name}.md", text, sub="skills")
        return name
