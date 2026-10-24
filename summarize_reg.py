"""Gộp log của regcheck.py:  python summarize_reg.py log_reg_s*.txt"""
import re, sys, glob, collections
pat = re.compile(r"RESULT seed=(\d+) row=(\S+) best_step=(\d+) bestB=([\d.]+) finalB=([\d.]+)")
rows = collections.defaultdict(list)
files = [f for a in sys.argv[1:] for f in glob.glob(a)]
for f in files:
    for line in open(f, encoding="utf-8", errors="ignore"):
        m = pat.search(line)
        if m:
            rows[m.group(2)].append((int(m.group(1)), int(m.group(3)), float(m.group(4)), float(m.group(5))))
if not rows:
    sys.exit("không thấy dòng RESULT nào")
print("cấu hình".ljust(16) + "seed".rjust(5) + "bestB (dừng sớm)".rjust(18) + "finalB".rjust(10) + "  bước tốt nhất")
mean = {}
for name, r in sorted(rows.items(), key=lambda kv: sum(x[2] for x in kv[1]) / len(kv[1])):
    n = len(r)
    mb = sum(x[2] for x in r) / n
    mf = sum(x[3] for x in r) / n
    mean[name] = mb
    print(name.ljust(16) + str(n).rjust(5) + f"{mb:18.4f}" + f"{mf:10.4f}" + "  " + ",".join(str(x[1]) for x in sorted(r)))
fixed = {k: v for k, v in mean.items() if k.startswith("fixed")}
if fixed and "elastic@k2" in mean:
    bf = min(fixed, key=fixed.get)
    print(f"\nfixed tốt nhất (có dừng sớm / regularize): {bf} = {fixed[bf]:.4f}")
    print(f"elastic@k2 - {bf}: {mean['elastic@k2'] - fixed[bf]:+.4f} nat   (số âm = elastic vẫn thắng)")
    if "fixed-k2" in mean:
        print(f"elastic@k2 - fixed-k2 (thường): {mean['elastic@k2'] - mean['fixed-k2']:+.4f} nat")
