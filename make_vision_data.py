"""
make_vision_data.py - tự tạo dữ liệu ảnh tiếng Việt cho train_vision.py: lấy ảnh rồi nhờ Gemini viết MÔ TẢ + vài CÂU HỎI-ĐÁP ngắn bằng tiếng Việt.

Nguồn ảnh (chọn một):
  --hf_dataset nlphuji/flickr30k [--split test]   ảnh đời thường công khai trên Hugging Face (cột ảnh mặc định "image"; tự kiểm tra giấy phép)
  --images_dir thu_muc_anh                        ảnh của chính bạn (jpg/png/webp)
Khoá Gemini: biến môi trường GEMINI_API_KEY (nhiều khoá cách nhau dấu phẩy -> xoay vòng khi bị giới hạn tốc độ).
Chọn model Gemini TỰ ĐỘNG: hỏi API danh sách model đang có, xếp phiên bản CAO -> THẤP (3.10 > 3.8 > 3.7 ...; chỉ dòng flash, bản ổn định trước preview),
dùng model cao nhất; hết quota NGÀY / model biến mất / quá tải thì tự chuyển xuống model kế tiếp. Gemini ra bản mới hơn vẫn tự dùng được, không phải sửa code.
Muốn ép model cụ thể: --model ten1,ten2 (hoặc biến GEMINI_MODEL); muốn cho cả dòng pro: --allow_pro.
Chạy:   python make_vision_data.py --hf_dataset nlphuji/flickr30k --n 1500 --out vdata
Kết quả: vdata/images/*.jpg (đã thu nhỏ) + vdata/data.jsonl  -> dùng:  python train_vision.py --out <model> --stage 1 --data vdata/data.jsonl --images_dir vdata
Chạy lại thì làm tiếp từ chỗ dừng (bỏ qua ảnh đã xong). Gemini bản miễn phí có hạn mức/ngày: đặt --n vừa phải, chia nhiều ngày.
Lưu ý chất lượng: câu trả lời cố ý NGẮN (<= --max_chars) vì model nhỏ; hãy đọc lướt vài chục dòng data.jsonl để chắc mô tả đúng trước khi train.
"""
import argparse, base64, io, json, os, random, re, sys, time, urllib.error, urllib.request

CAPTION_Q = ["", "", "Mô tả ảnh này.", "Trong ảnh có gì?", "Hãy mô tả bức ảnh."]
PROMPT = """Bạn là người viết dữ liệu huấn luyện. Nhìn ảnh rồi trả về DUY NHẤT một JSON (không markdown) dạng:
{"caption": "...", "qa": [{"q": "...", "a": "..."}, {"q": "...", "a": "..."}, {"q": "...", "a": "..."}]}
Quy tắc: toàn bộ bằng tiếng Việt tự nhiên. "caption" là 1-2 câu mô tả đúng những gì NHÌN THẤY (không đoán, không bịa tên riêng). "qa" gồm 3 cặp hỏi-đáp
về nội dung ảnh (đối tượng, màu sắc, số lượng, hành động, bối cảnh), câu trả lời ngắn gọn 1 câu, đúng với ảnh. Mỗi câu không quá {max_chars} ký tự."""


def keys():
    ks = [k.strip() for k in os.environ.get("GEMINI_API_KEY", "").split(",") if k.strip()]
    if not ks: sys.exit("thiếu GEMINI_API_KEY (Kaggle: Add-ons > Secrets, rồi os.environ[...] = ...)")
    return ks


BAN_TOKENS = {"exp", "embedding", "image", "vision", "audio", "tts", "asr", "imagen", "veo", "lyria", "live", "robotics", "aqa", "learnlm",
              "gemma", "thinking", "customtools", "dialog", "native", "computer", "deep", "research", "banana", "nano"}
BASE = "https://generativelanguage.googleapis.com/v1beta"


def rank_models(names):
    """Chỉ model họ gemini. Phiên bản CAO -> THẤP (so từng số: 3.10 > 3.8); cùng phiên bản thì bản ổn định trước preview, flash > flash-lite > pro.
    Bí danh '-latest' chỉ để dự phòng nên xếp CUỐI. (Cùng cách xếp với make_data.py.)"""
    scored = []
    for n in names:
        low = n.lower()
        toks = set(re.split(r"[-_.]", low))
        if not low.startswith("gemini") or toks & BAN_TOKENS:
            continue
        m = re.match(r"^gemini-(\d+(?:\.\d+)*)", low)
        if m:
            ver, alias = tuple(int(x) for x in m.group(1).split(".")), 0
        elif low.endswith("-latest"):
            ver, alias = (0,), 1
        else:
            continue
        tier = 2 if "flash" in low and "lite" not in low else (1 if "flash" in low else 0)
        stable = 0 if "preview" not in toks else 1
        neg = tuple(-x for x in (ver + (0, 0, 0))[:3])
        scored.append(((alias, neg, stable, -tier, n), n))
    return [n for _, n in sorted(scored)]


def http_get(url, timeout=30):
    with urllib.request.urlopen(urllib.request.Request(url), timeout=timeout) as r:
        return json.loads(r.read())


def list_models(ks):
    """Tên các model hỗ trợ generateContent (thử lần lượt từng key tới khi có key chạy được)."""
    for k in ks:
        found, tok = [], None
        try:
            while True:
                j = http_get(f"{BASE}/models?pageSize=1000&key={k}" + (f"&pageToken={tok}" if tok else ""))
                found += [m["name"].replace("models/", "") for m in j.get("models", []) if "generateContent" in m.get("supportedGenerationMethods", [])]
                tok = j.get("nextPageToken")
                if not tok: break
            return found
        except Exception as e:  # noqa: BLE001
            print("không liệt kê được model bằng một key:", str(e)[:100])
    return []


def model_chain(a, ks):
    forced = [m for m in re.split(r"[,;\s]+", a.model or "") if m]
    if forced: return forced
    ranked = [n for n in rank_models(list_models(ks)) if a.allow_pro or "flash" in n.lower()]
    if not ranked:
        print("⚠ không lấy được danh sách model, dùng tạm gemini-2.5-flash"); return ["gemini-2.5-flash"]
    print("Model Gemini sẽ dùng (cao -> thấp):", ", ".join(ranked[:6]) + (" ..." if len(ranked) > 6 else ""), flush=True)
    return ranked


class Caller:
    """Gọi Gemini với chuỗi model (cao -> thấp) x nhiều key. Hết quota NGÀY của (model,key) thì khoá cặp đó; model 404 thì bỏ model; quá tải thì tạm né."""

    def __init__(self, chain, ks, max_chars):
        self.chain, self.ks, self.max_chars = chain, ks, max_chars
        self.dead_model, self.dead_pair, self.cool_until, self.rot, self.over = set(), set(), {}, 0, {}

    def pick(self):
        now = time.time()
        for m in self.chain:
            if m in self.dead_model or self.cool_until.get(m, 0) > now: continue
            for j in range(len(self.ks)):
                i = (self.rot + j) % len(self.ks)
                if (m, i) not in self.dead_pair: return m, i
        return None

    def call(self, jpeg):
        wait = 8
        for _ in range(40):
            pk = self.pick()
            if pk is None:
                if any(self.cool_until.get(m, 0) > time.time() for m in self.chain if m not in self.dead_model):
                    time.sleep(15); continue
                raise SystemExit("⛔ Hết quota NGÀY (hoặc không còn model dùng được) trên mọi key. Tiến độ đã lưu: chạy lại lệnh cũ vào ngày mai.")
            m, i = pk
            try:
                return ask_gemini(jpeg, m, self.ks[i], self.max_chars)
            except urllib.error.HTTPError as e:
                body = (e.read() or b"").decode("utf-8", "ignore")[:500].lower().replace(" ", "")
                if e.code == 429 and ("perday" in body or "daily" in body):
                    print(f"  quota NGÀY hết: {m} (key {i + 1}) -> thử cặp khác/model thấp hơn", flush=True); self.dead_pair.add((m, i))
                elif e.code == 429:
                    self.rot += 1; print(f"  giới hạn tốc độ ({m}), xoay key, nghỉ {wait}s", flush=True); time.sleep(wait); wait = min(wait * 2, 90)
                elif e.code in (500, 503, 504):
                    self.over[m] = self.over.get(m, 0) + 1
                    if self.over[m] >= 2: self.cool_until[m] = time.time() + 600; self.over[m] = 0; print(f"  {m} quá tải, né 10 phút", flush=True)
                    time.sleep(5)
                elif e.code in (404, 400) and any(w in body for w in ("notfound", "notsupported", "isnotfound", "deprecated")):
                    print(f"  model không còn dùng được: {m}", flush=True); self.dead_model.add(m)
                elif e.code in (401, 403):
                    print(f"  key {i + 1} bị từ chối ({e.code}), bỏ key này", flush=True)
                    for mm in self.chain: self.dead_pair.add((mm, i))
                else:
                    print("  lỗi HTTP", e.code, body[:120]); return None
            except Exception as e:  # noqa: BLE001
                print("  lỗi:", str(e)[:100]); time.sleep(3)
        return None


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
    ap.add_argument("--model", default=os.environ.get("GEMINI_MODEL", ""), help="để trống = tự chọn model Gemini mới nhất còn dùng được")
    ap.add_argument("--allow_pro", action="store_true", help="cho phép cả dòng pro (quota miễn phí thấp)")
    ap.add_argument("--max_chars", type=int, default=160); ap.add_argument("--side", type=int, default=448)
    ap.add_argument("--sleep", type=float, default=4.0, help="giây nghỉ giữa các ảnh (hạn mức miễn phí ~ 10-15 lượt/phút)")
    a = ap.parse_args(argv)
    if not a.hf_dataset and not a.images_dir: sys.exit("cần --hf_dataset hoặc --images_dir")
    ks = keys()
    caller = Caller(model_chain(a, ks), ks, a.max_chars)
    os.makedirs(os.path.join(a.out, "images"), exist_ok=True)
    jl, prog = os.path.join(a.out, "data.jsonl"), os.path.join(a.out, "done.txt")
    done = set(open(prog).read().split()) if os.path.exists(prog) else set()
    ok = bad = 0
    with open(jl, "a", encoding="utf-8") as fo, open(prog, "a") as fp:
        for name, im in iter_images(a):
            if len(done) >= a.n: break
            if name in done: continue
            jpeg = to_jpeg(im, a.side)
            text = caller.call(jpeg)
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
