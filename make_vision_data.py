"""
make_vision_data.py - tự tạo dữ liệu ảnh tiếng Việt cho train_vision.py: lấy ảnh rồi nhờ Gemini viết MÔ TẢ + vài CÂU HỎI-ĐÁP ngắn bằng tiếng Việt.

Nguồn ảnh (chọn một):
  --hf_dataset nlphuji/flickr30k [--split test]   ảnh đời thường công khai trên Hugging Face (cột ảnh mặc định "image"; tự kiểm tra giấy phép)
  --images_dir thu_muc_anh                        ảnh của chính bạn (jpg/png/webp)
Khoá Gemini: biến môi trường GEMINI_API_KEY (nhiều khoá cách nhau dấu phẩy -> xoay vòng khi bị giới hạn tốc độ).
Chạy:   python make_vision_data.py --hf_dataset nlphuji/flickr30k --n 1500 --out vdata
Kết quả: vdata/images/*.jpg (đã thu nhỏ) + vdata/data.jsonl  -> dùng:  python train_vision.py --out <model> --stage 1 --data vdata/data.jsonl --images_dir vdata
Chạy lại thì làm tiếp từ chỗ dừng (bỏ qua ảnh đã xong). Gemini bản miễn phí có hạn mức/ngày: đặt --n vừa phải, chia nhiều ngày.
Lưu ý chất lượng: câu trả lời cố ý NGẮN (<= --max_chars) vì model nhỏ; hãy đọc lướt vài chục dòng data.jsonl để chắc mô tả đúng trước khi train.
"""
import argparse, base64, io, json, os, random, sys, time, urllib.error, urllib.request

CAPTION_Q = ["", "", "Mô tả ảnh này.", "Trong ảnh có gì?", "Hãy mô tả bức ảnh."]
PROMPT = """Bạn là người viết dữ liệu huấn luyện. Nhìn ảnh rồi trả về DUY NHẤT một JSON (không markdown) dạng:
{"caption": "...", "qa": [{"q": "...", "a": "..."}, {"q": "...", "a": "..."}, {"q": "...", "a": "..."}]}
Quy tắc: toàn bộ bằng tiếng Việt tự nhiên. "caption" là 1-2 câu mô tả đúng những gì NHÌN THẤY (không đoán, không bịa tên riêng). "qa" gồm 3 cặp hỏi-đáp
về nội dung ảnh (đối tượng, màu sắc, số lượng, hành động, bối cảnh), câu trả lời ngắn gọn 1 câu, đúng với ảnh. Mỗi câu không quá {max_chars} ký tự."""


def keys():
    ks = [k.strip() for k in os.environ.get("GEMINI_API_KEY", "").split(",") if k.strip()]
    if not ks: sys.exit("thiếu GEMINI_API_KEY (Kaggle: Add-ons > Secrets, rồi os.environ[...] = ...)")
    return ks


def ask_gemini(jpeg_bytes, model, key, max_chars):
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={key}"
    body = {"contents": [{"parts": [{"text": PROMPT.replace("{max_chars}", str(max_chars))},
                                    {"inline_data": {"mime_type": "image/jpeg", "data": base64.b64encode(jpeg_bytes).decode()}}]}],
            "generationConfig": {"temperature": 0.4, "responseMimeType": "application/json"}}
    req = urllib.request.Request(url, json.dumps(body).encode(), {"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=60) as r:
        out = json.loads(r.read())
    return out["candidates"][0]["content"]["parts"][0]["text"]


def parse(text, max_chars):
    """-> list[(q, a)] đã lọc, hoặc []."""
    t = text.strip()
    if t.startswith("```"): t = t.strip("`").removeprefix("json").strip()
    try:
        j = json.loads(t)
    except Exception:  # noqa: BLE001
        return []
    rows = []
    cap = str(j.get("caption", "")).strip()
    if 8 <= len(cap) <= max_chars * 2: rows.append((random.choice(CAPTION_Q), cap))
    for p in j.get("qa", []) or []:
        q, a = str(p.get("q", "")).strip(), str(p.get("a", "")).strip()
        if 3 <= len(q) <= max_chars and 2 <= len(a) <= max_chars: rows.append((q, a))
    return rows


def to_jpeg(img, side):
    img = img.convert("RGB")
    img.thumbnail((side, side))
    b = io.BytesIO(); img.save(b, "JPEG", quality=90)
    return b.getvalue()


def iter_images(a):
    from PIL import Image
    if a.images_dir:
        for fn in sorted(os.listdir(a.images_dir)):
            if fn.lower().endswith((".jpg", ".jpeg", ".png", ".webp")):
                try: yield fn, Image.open(os.path.join(a.images_dir, fn))
                except Exception: continue  # noqa: BLE001
    else:
        from datasets import load_dataset
        ds = load_dataset(a.hf_dataset, split=a.split, streaming=True)
        for i, row in enumerate(ds): yield f"{i:07d}", row[a.image_col]


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--hf_dataset", default=None); ap.add_argument("--split", default="test"); ap.add_argument("--image_col", default="image")
    ap.add_argument("--images_dir", default=None)
    ap.add_argument("--out", default="vdata"); ap.add_argument("--n", type=int, default=1000, help="số ảnh tối đa")
    ap.add_argument("--model", default=os.environ.get("GEMINI_MODEL", "gemini-2.5-flash"))
    ap.add_argument("--max_chars", type=int, default=160); ap.add_argument("--side", type=int, default=448)
    ap.add_argument("--sleep", type=float, default=4.0, help="giây nghỉ giữa các ảnh (hạn mức miễn phí ~ 10-15 lượt/phút)")
    a = ap.parse_args(argv)
    if not a.hf_dataset and not a.images_dir: sys.exit("cần --hf_dataset hoặc --images_dir")
    ks = keys()
    os.makedirs(os.path.join(a.out, "images"), exist_ok=True)
    jl, prog = os.path.join(a.out, "data.jsonl"), os.path.join(a.out, "done.txt")
    done = set(open(prog).read().split()) if os.path.exists(prog) else set()
    ok = bad = 0; ki = 0
    with open(jl, "a", encoding="utf-8") as fo, open(prog, "a") as fp:
        for name, im in iter_images(a):
            if len(done) >= a.n: break
            if name in done: continue
            jpeg = to_jpeg(im, a.side); text = None; wait = 10
            for _ in range(6):
                try:
                    text = ask_gemini(jpeg, a.model, ks[ki % len(ks)], a.max_chars); break
                except urllib.error.HTTPError as e:
                    if e.code in (429, 503): ki += 1; print(f"giới hạn/quá tải ({e.code}), nghỉ {wait}s"); time.sleep(wait); wait = min(wait * 2, 120)
                    else: print("lỗi HTTP", e.code); break
                except Exception as e:  # noqa: BLE001
                    print("lỗi:", e); time.sleep(3)
            rows = parse(text, a.max_chars) if text else []
            if rows:
                rel = f"images/{name}.jpg"
                with open(os.path.join(a.out, rel), "wb") as f: f.write(jpeg)
                for q, ans in rows: fo.write(json.dumps({"image": rel, "q": q, "a": ans}, ensure_ascii=False) + "\n")
                fo.flush(); ok += 1
            else: bad += 1
            fp.write(name + "\n"); fp.flush(); done.add(name)
            if (ok + bad) % 20 == 0: print(f"đã xử lý {ok + bad} ảnh (tốt {ok}, bỏ {bad})", flush=True)
            time.sleep(a.sleep)
    print(f"xong: {ok} ảnh có dữ liệu, {bad} bỏ. File: {jl}")


if __name__ == "__main__":
    main()
