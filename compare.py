"""
compare.py - gom các file kết quả (r_*.json hoặc results_*.json) thành một bảng.

    python compare.py                 # tự tìm r_*.json
    python compare.py r_fixed.json r_diff.json
"""
import glob
import json
import sys


def label(a):
    name = a["mode"] + ("+ste" if a.get("ste") else "") + ("+diff" if a.get("diff_head") else "") + ("+gain" if a.get("diff_target") == "gain" else "")
    budget = (f"L{a['layer_k']}" if a.get("layer_k") else f"k={a['k']}") if a["mode"] == "fixed" else f"tk={a['target_k']}"
    return name, budget


def main():
    paths = sys.argv[1:] or sorted(glob.glob("r_*.json"))
    if not paths:
        print("Không tìm thấy file kết quả.")
        return
    head = f"{'file':<22}{'chế độ':<16}{'ngân sách':<10}{'seed':>5}{'data':>6}{'E':>4}{'cc':>5}{'val_loss':>10}{'ppl':>8}{'avg_k':>7}{'corr(k,loss)':>14}{'k Q4-Q1':>9}"
    print(head)
    print("-" * len(head))
    for p in paths:
        with open(p, encoding="utf-8") as f:
            d = json.load(f)
        a, r = d["args"], d["result"]
        name, budget = label(a)
        q = r["k_by_loss_quartile"]
        print(f"{p[-21:]:<22}{name:<16}{budget:<10}{a['seed']:>5}{a.get('max_chars', 0) // 1_000_000:>5}M{a.get('experts', 8):>4}{(0 if a['mode'] == 'fixed' else a.get('compute_coef', 0)):>5}{r['val_loss']:>10.4f}"
              f"{r['val_ppl']:>8.1f}{r['avg_k']:>7.2f}{r['corr_k_loss']:>+14.3f}{q[3] - q[0]:>+9.2f}")


if __name__ == "__main__":
    main()
