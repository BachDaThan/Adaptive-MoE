"""Gộp log của regcheck.py:  python summarize_reg.py "log_reg*_s*.txt" """
import re, sys, glob, collections
pat = re.compile(r"RESULT seed=(\d+) row=(\S+) best_step=(\d+) bestB=([\d.]+) finalB=([\d.]+)")
COST = {"fixed-k1": 1, "fixed-k1+drop": 1, "fixed-k2": 2, "fixed-k2+drop": 2, "fixed-k2+wd": 2, "fixed-k2+drop2": 2,
        "fixed-k4+drop": 4, "sched-1124+drop": 2, "sched-4211+drop": 2, "sched-1122+drop": 1.5,
        "elastic@k1": 1, "elastic@k2": 2, "elastic+drop@k1": 1, "elastic+drop@k2": 2,
        "fixed-k2+drop3": 2, "fixed-k1+drop2": 1, "elastic+drop2@k1": 1, "elastic+drop2@k2": 2,
        "elastic+drop3@k1": 1, "elastic+drop3@k2": 2}
rows = collections.defaultdict(list)
files = sorted({f for a in sys.argv[1:] for f in glob.glob(a)})
for f in files:
    for line in open(f, encoding="utf-8", errors="ignore"):
        m = pat.search(line)
        if m:
            rows[m.group(2)].append((int(m.group(1)), int(m.group(3)), float(m.group(4)), float(m.group(5))))
if not rows:
    sys.exit("không thấy dòng RESULT nào")
print("cấu hình".ljust(18) + "cost".rjust(5) + "seed".rjust(5) + "bestB (dừng sớm)".rjust(18) + "finalB".rjust(10) + "  bước tốt nhất")
mean = {}
for name, r in sorted(rows.items(), key=lambda kv: sum(x[2] for x in kv[1]) / len(kv[1])):
    n = len(r)
    mb = sum(x[2] for x in r) / n
    mf = sum(x[3] for x in r) / n
    mean[name] = mb
    c = COST.get(re.sub(r"-6k(?=@|$)", "", name))
    print(name.ljust(18) + (f"{c:5.1f}" if c else "".rjust(5)) + str(n).rjust(5) + f"{mb:18.4f}" + f"{mf:10.4f}" + "  " +
          ",".join(str(x[1]) for x in sorted(r)))
pairs = [("elastic@k2", "fixed-k2"), ("elastic@k2", "fixed-k2+drop"), ("elastic+drop@k2", "fixed-k2+drop"),
         ("elastic+drop@k1", "fixed-k1+drop"), ("fixed-k2+drop2", "fixed-k2+drop"), ("fixed-k4+drop", "fixed-k2+drop"),
         ("sched-1124+drop", "fixed-k2+drop"), ("sched-4211+drop", "fixed-k2+drop"), ("sched-1122+drop", "fixed-k1+drop"),
         ("sched-1122+drop", "fixed-k2+drop"),
         # vòng 3
         ("elastic+drop2@k2", "fixed-k2+drop2"), ("elastic+drop2@k1", "fixed-k1+drop2"), ("elastic+drop3@k2", "fixed-k2+drop3"),
         ("elastic+drop2@k2", "elastic+drop@k2"), ("fixed-k2+drop3", "fixed-k2+drop2"), ("elastic+drop3@k2", "elastic+drop2@k2"),
         # 6000 bước
         ("elastic+drop-6k@k2", "fixed-k2+drop-6k"), ("elastic+drop2-6k@k2", "fixed-k2+drop2-6k"),
         ("elastic+drop-6k@k2", "fixed-k2+drop2-6k"), ("elastic+drop-6k@k2", "elastic+drop@k2"),
         ("fixed-k2+drop2-6k", "fixed-k2+drop2")]
print("\nso sánh theo cặp (bestB; số âm = hàng đầu tốt hơn, đơn vị nat):")
for a, b in pairs:
    if a in mean and b in mean:
        print(f"  {a} - {b}: {mean[a] - mean[b]:+.4f}")
