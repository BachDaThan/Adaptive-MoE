# -*- coding: utf-8 -*-
# diversity_audit.py — đo độ ĐA DẠNG của data Fine-tune/<cỡ>/ và data/<cỡ>/ (không cần GPU, chạy được ở Ô2 sau make_data).
# Dùng:  python diversity_audit.py                  (kéo thẳng từ HF theo HF_REPO / MODEL_PARAMS / HF_TOKEN trong môi trường)
#        python diversity_audit.py --local ./thu_muc (đọc file .txt/.jsonl local)
# Trong Kaggle: dán nội dung file vào 1 ô mới (sau Ô2) rồi chạy, hoặc !python diversity_audit.py
import os, re, sys, json, math, glob, hashlib, argparse, random
from collections import Counter

ap = argparse.ArgumentParser()
ap.add_argument("--repo", default=os.environ.get("HF_REPO", "BachDaThan/TrainAI-web"))
ap.add_argument("--tag", default=os.environ.get("MODEL_PARAMS", "1B"))
ap.add_argument("--local", default=None)
ap.add_argument("--max_files", type=int, default=60)
ap.add_argument("--sample", type=int, default=20000)
a = ap.parse_args()
random.seed(0)

def get_files(prefix):
    """Trả về list nội dung file text."""
    if a.local:
        sub = os.path.join(a.local, prefix)
        base = sub if os.path.isdir(sub) else (a.local if prefix == "Fine-tune" else None)
        if base is None: return []
        fs = sorted(glob.glob(os.path.join(base, "**", "*.*"), recursive=True))
        fs = [f for f in fs if f.endswith((".txt", ".jsonl", ".json"))]
        return [open(f, encoding="utf-8", errors="ignore").read() for f in fs[:a.max_files]]
    from huggingface_hub import HfApi, hf_hub_download
    tok = os.environ.get("HF_TOKEN"); fl = []; rt = "dataset"
    for rt in ("dataset", "model"):
        try: fl = HfApi(token=tok).list_repo_files(a.repo, repo_type=rt); break
        except Exception: fl = []
    fl = [f for f in fl if re.match(rf"^{re.escape(prefix)}/{re.escape(a.tag)}/[^/]+\.(txt|jsonl)$", f)]
    random.shuffle(fl)
    return [open(hf_hub_download(a.repo, f, repo_type=rt, token=tok), encoding="utf-8", errors="ignore").read() for f in fl[:a.max_files]]

def norm(s): return re.sub(r"\s+", " ", re.sub(r"\d+", "#", s.lower())).strip()
def toks(s): return re.findall(r"\w+", s.lower())
def distinct_n(texts, n):
    tot, seen = 0, set()
    for t in texts:
        w = toks(t)
        for i in range(len(w) - n + 1): seen.add(hash(tuple(w[i:i + n]))); tot += 1
    return len(seen) / tot if tot else 0.0
def entropy(c):
    t = sum(c.values()); return -sum(v / t * math.log2(v / t) for v in c.values()) if t else 0.0
def pct(xs, p): xs = sorted(xs); return xs[min(len(xs) - 1, int(len(xs) * p))] if xs else 0
def top(c, k=8):
    t = sum(c.values()) or 1
    return ", ".join(f"«{w}» {v / t:.1%}" for w, v in c.most_common(k))

def parse_convs(blobs):
    convs = []
    for blob in blobs:
        for line in blob.splitlines():
            line = line.strip()
            if '"messages"' not in line: continue
            try: m = json.loads(line).get("messages", [])
            except Exception: continue
            m = [(x.get("role", ""), str(x.get("content", ""))) for x in m if isinstance(x, dict)]
            if m: convs.append(m)
    return convs

print(f"== Repo {a.repo} | cỡ {a.tag} ==")
# ---------------- FINE-TUNE ----------------
convs = parse_convs(get_files("Fine-tune"))
print(f"\n📘 FINE-TUNE: {len(convs)} hội thoại đọc được")
if convs:
    if len(convs) > a.sample: convs = random.sample(convs, a.sample)
    users = [" ".join(c for r, c in m if r == "user") for m in convs]
    firsts = [next((c for r, c in m if r == "user"), "") for m in convs]
    asst = [" ".join(c for r, c in m if r == "assistant") for m in convs]
    is_tool = [any("<tool>" in c for r, c in m) for m in convs]
    exact = 1 - len(set(hashlib.md5(norm(u + "|" + s).encode()).hexdigest() for u, s in zip(users, asst))) / len(convs)
    q_dup = 1 - len(set(norm(f) for f in firsts)) / len(convs)
    print(f"  Trùng y hệt (cả hỏi+đáp, đã bỏ số): {exact:.1%} | trùng câu hỏi đầu: {q_dup:.1%}")
    for nm, tx in (("câu hỏi", firsts), ("câu trả lời", asst)):
        print(f"  distinct-1/2/3 của {nm}: {distinct_n(tx, 1):.3f} / {distinct_n(tx, 2):.3f} / {distinct_n(tx, 3):.3f}   (càng gần 1 càng đa dạng; distinct-3 < 0.5 là lặp nhiều)")
    op_u = Counter(" ".join(toks(f)[:3]) for f in firsts); op_a = Counter(" ".join(toks(s)[:3]) for s in asst)
    print(f"  Mở đầu câu hỏi hay gặp: {top(op_u)}")
    print(f"  Mở đầu câu trả lời hay gặp: {top(op_a)}")
    print(f"  Entropy mở đầu hỏi/đáp: {entropy(op_u):.2f} / {entropy(op_a):.2f} bit (thấp = rập khuôn)")
    L = [len(s) for s in asst]; T = [len(m) for m in convs]
    print(f"  Độ dài trả lời (ký tự) p10/p50/p90: {pct(L, .1)}/{pct(L, .5)}/{pct(L, .9)} | số lượt/hội thoại p50/p90: {pct(T, .5)}/{pct(T, .9)}")
    multi = sum(1 for m in convs if sum(1 for r, _ in m if r == "user") > 1) / len(convs)
    print(f"  Hội thoại nhiều lượt (≥2 câu hỏi): {multi:.1%}")
    nt = sum(is_tool)
    print(f"  Hội thoại dùng tool: {nt} ({nt / len(convs):.1%})")
    if nt:
        tc = [t for t, f in zip(users, is_tool) if f]
        tpl = Counter(norm(re.sub(r"<tool>.*?</tool>|<result>.*?</result>", "", u + " " + s, flags=re.S)) [:80] for u, s in zip(tc, [s for s, f in zip(asst, is_tool) if f]))
        kinds = Counter(k for m in convs for _, c in m for k in re.findall(r"<tool>(\w+)", c))
        print(f"  Số 'khuôn' tool khác nhau (80 ký tự đầu, đã bỏ số): {len(tpl)}/{nt} = {len(tpl) / nt:.1%} (thấp = sinh từ template)")
        print(f"  Theo loại tool: {dict(kinds)}")
    nd = sum(1 for u in users if not re.search(r"[ăâđêôơưáàảãạéèẻẽẹíìỉĩịóòỏõọúùủũụýỳỷỹỵ]", u.lower()))
    print(f"  Câu hỏi KHÔNG có dấu tiếng Việt: {nd / len(convs):.1%} (0% = chưa có kiểu gõ không dấu/teencode)")
    short = sum(1 for f in firsts if len(f) < 25) / len(convs)
    print(f"  Câu hỏi ngắn (<25 ký tự): {short:.1%}")

# ---------------- PRE-TRAIN ----------------
print("\n📗 PRE-TRAIN (data/<cỡ>/ do make_data tạo; FineWeb-2 tải riêng trong Ô3 không nằm ở đây)")
blobs = get_files("data")
lines = [l.strip() for b in blobs for l in b.splitlines() if len(l.strip()) > 40]
print(f"  {len(lines)} dòng/đoạn đọc được")
if lines:
    if len(lines) > a.sample: lines = random.sample(lines, a.sample)
    dup = 1 - len(set(norm(l) for l in lines)) / len(lines)
    print(f"  Trùng đoạn: {dup:.1%} | distinct-1/2/3: {distinct_n(lines, 1):.3f} / {distinct_n(lines, 2):.3f} / {distinct_n(lines, 3):.3f}")
    print(f"  Mở đầu hay gặp: {top(Counter(' '.join(toks(l)[:3]) for l in lines))}")
    L = [len(l) for l in lines]
    print(f"  Độ dài đoạn p10/p50/p90: {pct(L, .1)}/{pct(L, .5)}/{pct(L, .9)}")
else:
    print("  (trống — đúng với cấu hình hiện tại: DATA_KINDS không có 'pretrain', Ô3 chỉ dùng FineWeb-2 vie_Latn)")
print("\nXong. Gợi ý ngưỡng: trùng câu hỏi < 5%, distinct-3 câu trả lời > 0.6, entropy mở đầu > 5 bit, khuôn tool > 30%.")
