# -*- coding: utf-8 -*-
# clean_finetune.py — quét Fine-tune/<cỡ>/N.txt đã có trên HF, BỎ hội thoại trùng / gần trùng / mở đầu bị lặp quá nhiều, rồi ghi lại.
# Mặc định chỉ XEM TRƯỚC (không sửa gì). Thêm --apply để ghi thật; bản gốc luôn được nén lên Fine-tune-backup/ trước khi sửa.
# Hội thoại dùng <tool> chỉ bị bỏ khi trùng y hệt (khuôn tool lặp là bình thường, make_data v6.2 lo phần đa dạng của nó).
# Sau khi dọn, chạy lại Ô2: make_data auto thấy thiếu so với mục tiêu (FT_PAIRS) sẽ tự tạo bù bằng bộ sinh mới, đa dạng hơn.
#   python clean_finetune.py            # xem trước
#   python clean_finetune.py --apply    # ghi thật
#   python clean_finetune.py --local thu_muc   # thử trên thư mục máy (với --apply: ghi đè file trong đó)
import os, re, io, json, sys, glob, time, argparse, zipfile
from collections import Counter

ap = argparse.ArgumentParser()
ap.add_argument("--repo", default=os.environ.get("HF_REPO", "BachDaThan/TrainAI-web"))
ap.add_argument("--tag", default=os.environ.get("MODEL_PARAMS", "1B"))
ap.add_argument("--local", default=None)
ap.add_argument("--apply", action="store_true")
ap.add_argument("--cap", type=float, default=0.05, help="một kiểu mở đầu (2 từ đầu câu trả lời) tối đa ngần này của tổng")
ap.add_argument("--near", type=float, default=0.7, help="bỏ nếu >= ngần này số 3-gram đã có ở các hội thoại giữ trước đó")
a = ap.parse_args()

W = lambda s: re.sub(r"[^\w]+", " ", s.lower()).split()
def grams(c):
    w = W(" ".join(u + " " + x for u, x in c)); return {hash(tuple(w[i:i + 3])) for i in range(max(0, len(w) - 2))}

def load():
    """-> {tên file: [dòng...]}"""
    if a.local:
        return {os.path.relpath(f, a.local): open(f, encoding="utf-8", errors="ignore").read().splitlines()
                for f in sorted(glob.glob(os.path.join(a.local, "*.txt")))}
    from huggingface_hub import HfApi, hf_hub_download
    tok = os.environ.get("HF_TOKEN"); api = HfApi(token=tok); rt, fl = "dataset", []
    for rt in ("dataset", "model"):
        try: fl = api.list_repo_files(a.repo, repo_type=rt); break
        except Exception: fl = []
    a.rt = rt
    names = sorted((f for f in fl if re.match(rf"^Fine-tune/{re.escape(a.tag)}/\d+\.txt$", f)), key=lambda f: int(re.search(r"(\d+)\.txt$", f).group(1)))
    return {f: open(hf_hub_download(a.repo, f, repo_type=rt, token=tok), encoding="utf-8", errors="ignore").read().splitlines() for f in names}

files = load()
rows = []   # (file, line, conv|None, is_tool)
for f, lines in files.items():
    for ln in lines:
        if '"messages"' not in ln: rows.append((f, ln, None, False)); continue
        try: m = json.loads(ln)["messages"]
        except Exception: rows.append((f, ln, None, False)); continue
        conv = [(m[i].get("content", ""), m[i + 1].get("content", "")) for i in range(0, len(m) - 1, 2)
                if m[i].get("role") == "user" and m[i + 1].get("role") == "assistant"]
        rows.append((f, ln, conv or None, "<tool>" in ln))
total = sum(1 for r in rows if r[2])
print(f"Đọc {len(files)} file, {total} hội thoại.")
maxop = max(20, int(a.cap * total))

seen_exact, seen_g, opn = set(), set(), Counter()
why = Counter(); keep = {}
for f, ln, conv, is_tool in rows:
    ok = True
    if conv:
        k = " ".join(W(" ".join(u + "|" + x for u, x in conv)))
        if k in seen_exact: ok = False; why["trùng y hệt"] += 1
        elif not is_tool:
            op = " ".join(W(conv[0][1])[:2]); g = grams(conv)
            if op and opn[op] >= maxop: ok = False; why["mở đầu bị lặp quá nhiều"] += 1
            elif len(g) >= 6 and len(g & seen_g) / len(g) >= a.near: ok = False; why["gần trùng"] += 1
            else: opn[op] += 1; seen_g |= g
        if ok: seen_exact.add(k)
    if ok: keep.setdefault(f, []).append(ln)

removed = sum(why.values())
print(f"Sẽ bỏ {removed} hội thoại ({removed / max(1, total):.1%}): {dict(why)}")
print("Kiểu mở đầu hay gặp nhất (sau khi lọc):", ", ".join(f"«{o}» {n}" for o, n in opn.most_common(5)))
changed = [f for f in files if keep.get(f, []) != files[f]]
print(f"{len(changed)}/{len(files)} file sẽ thay đổi.")
if not a.apply:
    print("👀 Xem trước thôi, chưa sửa gì. Chạy lại với --apply để ghi thật."); sys.exit(0)

if a.local:
    for f in changed: open(os.path.join(a.local, f), "w", encoding="utf-8").write("\n".join(keep.get(f, [])) + "\n")
    print("✅ Đã ghi lại trong", a.local); sys.exit(0)

from huggingface_hub import HfApi
api = HfApi(token=os.environ.get("HF_TOKEN"))
buf = io.BytesIO()
with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
    for f, lines in files.items(): z.writestr(f, "\n".join(lines) + "\n")
bk = f"Fine-tune-backup/{a.tag}_{time.strftime('%Y%m%d_%H%M%S')}.zip"
api.upload_file(path_or_fileobj=buf.getvalue(), path_in_repo=bk, repo_id=a.repo, repo_type=a.rt)
print("💾 Đã sao lưu bản gốc:", bk)
for f in changed:
    body = "\n".join(keep.get(f, [])) + "\n"
    if keep.get(f): api.upload_file(path_or_fileobj=body.encode("utf-8"), path_in_repo=f, repo_id=a.repo, repo_type=a.rt)
    else: api.delete_file(f, repo_id=a.repo, repo_type=a.rt)
print(f"✅ Xong. Chạy lại Ô2 để make_data bù phần thiếu; Ô4 nhớ đặt FORCE_SFT=\"1\" nếu muốn train lại ngay.")
