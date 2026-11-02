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
ap.add_argument("--cap1", type=float, default=0.06, help="một TỪ mở đầu (vd «có») tối đa ngần này của tổng, gộp mọi biến thể: có chứ / có á / có nè...")
ap.add_argument("--capq", type=float, default=0.04, help="một kiểu mở đầu CÂU HỎI (2 từ đầu, vd «làm sao») tối đa ngần này của tổng")
ap.add_argument("--show", default=None, help="in 8 hội thoại mẫu có câu trả lời bắt đầu bằng từ này (vd --show có) rồi thoát, để xem tận mắt")
ap.add_argument("--capyn", type=float, default=0.12, help="câu hỏi dạng yes/no «... có ... không/hông?» tối đa ngần này của tổng (hỏi tự nhiên thường ~10-15%)")
ap.add_argument("--near", type=float, default=0.7, help="bỏ nếu >= ngần này số 3-gram đã có ở các hội thoại giữ trước đó")
a = ap.parse_args()

YN = re.compile(r"\bcó\b.*\b(không|hông|hong|hem|hôn|ko|chưa|k)\b\W*$")   # «X có Y không?»
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
if a.show:
    import random
    hit = [r[2] for r in rows if r[2] and not r[3] and W(r[2][0][1])[:1] == [a.show.lower()]]
    print(f"{len(hit)} hội thoại có câu trả lời bắt đầu bằng «{a.show}». Mẫu ngẫu nhiên:")
    for c in random.sample(hit, min(8, len(hit))): print("  Hỏi:", c[0][0][:110], "\n  Đáp:", c[0][1][:110], "\n")
    sys.exit(0)
print(f"Đọc {len(files)} file, {total} hội thoại.")
maxop = max(20, int(a.cap * total)); maxfw = max(30, int(a.cap1 * total)); maxq = max(30, int(a.capq * total)); maxyn = max(30, int(a.capyn * total))
n_yn = sum(1 for r in rows if r[2] and YN.search(r[2][0][0].lower()))
print(f"Câu hỏi dạng «có ... không?»: {n_yn} ({n_yn / max(1, total):.1%}) — giữ tối đa {maxyn}")
print("Câu HỎI hay bắt đầu bằng:", ", ".join(f"«{o}» {n}" for o, n in Counter(' '.join(W(r[2][0][0])[:2]) for r in rows if r[2]).most_common(6)))
print("Câu TRẢ LỜI hay bắt đầu bằng từ:", ", ".join(f"«{o}» {n}" for o, n in Counter(' '.join(W(r[2][0][1])[:1]) for r in rows if r[2] and not r[3]).most_common(6)))

seen_exact, seen_g, opn, opn1, opnq, opnyn = set(), set(), Counter(), Counter(), Counter(), 0
why = Counter(); keep = {}
for f, ln, conv, is_tool in rows:
    ok = True
    if conv:
        k = " ".join(W(" ".join(u + "|" + x for u, x in conv)))
        if k in seen_exact: ok = False; why["trùng y hệt"] += 1
        elif not is_tool:
            op = " ".join(W(conv[0][1])[:2]); fw = " ".join(W(conv[0][1])[:1]); g = grams(conv)
            qo = " ".join(W(conv[0][0])[:2]); is_yn = bool(YN.search(conv[0][0].lower()))
            if is_yn and opnyn >= maxyn: ok = False; why["câu hỏi yes/no «có...không» quá nhiều"] += 1
            elif qo and opnq[qo] >= maxq: ok = False; why["câu hỏi mở đầu bị lặp quá nhiều"] += 1
            elif fw and opn1[fw] >= maxfw: ok = False; why["từ mở đầu bị lặp quá nhiều"] += 1
            elif op and opn[op] >= maxop: ok = False; why["mở đầu bị lặp quá nhiều"] += 1
            elif len(g) >= 6 and len(g & seen_g) / len(g) >= a.near: ok = False; why["gần trùng"] += 1
            else: opn[op] += 1; opn1[fw] += 1; opnq[qo] += 1; opnyn += is_yn; seen_g |= g
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
