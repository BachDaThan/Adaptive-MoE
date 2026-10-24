"""Gộp log của regcheck.py:  python summarize_reg.py "log_reg*_s*.txt"
So sánh theo cặp chỉ dùng các seed mà cả hai hàng đều có, in kèm chênh lệch từng seed."""
import re, sys, glob, collections
pat = re.compile(r"RESULT seed=(\d+) row=(\S+) best_step=(\d+) bestB=([\d.]+) finalB=([\d.]+)")
COST = {"fixed-k1": 1, "fixed-k1+drop": 1, "fixed-k2": 2, "fixed-k2+drop": 2, "fixed-k2+wd": 2, "fixed-k2+drop2": 2,
        "fixed-k4+drop": 4, "sched-1124+drop": 2, "sched-4211+drop": 2, "sched-1122+drop": 1.5,
        "elastic@k1": 1, "elastic@k2": 2, "elastic+drop@k1": 1, "elastic+drop@k2": 2,
        "fixed-k2+drop3": 2, "fixed-k1+drop2": 1, "elastic+drop2@k1": 1, "elastic+drop2@k2": 2,
        "elastic+drop3@k1": 1, "elastic+drop3@k2": 2,
        "fixed-k1+drop15": 1, "fixed-k2+drop15": 2, "elastic+drop15@k1": 1, "elastic+drop15@k2": 2}
by = collections.defaultdict(dict)  # hàng -> seed -> (bước, bestB, finalB); chạy lại thì lấy lần cuối, không đếm đôi
for f in sorted({f for a in sys.argv[1:] for f in glob.glob(a)}):
    for line in open(f, encoding="utf-8", errors="ignore"):
        m = pat.search(line)
        if m:
            by[m.group(2)][int(m.group(1))] = (int(m.group(3)), float(m.group(4)), float(m.group(5)))
if not by:
    sys.exit("không thấy dòng RESULT nào")
print("cấu hình".ljust(22) + "cost".rjust(5) + "  n".rjust(4) + "bestB (dừng sớm)".rjust(18) + "finalB".rjust(10) + "  seed: bước tốt nhất")
for name, d in sorted(by.items(), key=lambda kv: sum(x[1] for x in kv[1].values()) / len(kv[1])):
    n = len(d)
    mb = sum(x[1] for x in d.values()) / n
    mf = sum(x[2] for x in d.values()) / n
    c = COST.get(re.sub(r"-6k(?=@|$)", "", name))
    print(name.ljust(22) + (f"{c:5.1f}" if c else "".rjust(5)) + str(n).rjust(4) + f"{mb:18.4f}" + f"{mf:10.4f}" + "  " +
          " ".join(f"{s}:{x[0]}" for s, x in sorted(d.items())))
pairs = [("elastic@k2", "fixed-k2"), ("elastic@k2", "fixed-k2+drop"), ("elastic+drop@k2", "fixed-k2+drop"),
         ("elastic+drop@k1", "fixed-k1+drop"), ("fixed-k2+drop2", "fixed-k2+drop"), ("fixed-k4+drop", "fixed-k2+drop"),
         ("sched-1124+drop", "fixed-k2+drop"), ("sched-4211+drop", "fixed-k2+drop"), ("sched-1122+drop", "fixed-k1+drop"),
         ("sched-1122+drop", "fixed-k2+drop"),
         # vòng 3
         ("elastic+drop2@k2", "fixed-k2+drop2"), ("elastic+drop2@k1", "fixed-k1+drop2"), ("elastic+drop3@k2", "fixed-k2+drop3"),
         ("elastic+drop2@k2", "elastic+drop@k2"), ("fixed-k2+drop3", "fixed-k2+drop2"), ("elastic+drop3@k2", "elastic+drop2@k2"),
         ("elastic+drop2@k1", "fixed-k1+drop"), ("fixed-k1+drop2", "fixed-k1+drop"),
         # 6000 bước
         ("elastic+drop-6k@k2", "fixed-k2+drop-6k"), ("elastic+drop2-6k@k2", "fixed-k2+drop2-6k"),
         ("elastic+drop-6k@k2", "fixed-k2+drop2-6k"), ("elastic+drop2-6k@k2", "elastic+drop2@k2"),
         ("fixed-k2+drop2-6k", "fixed-k2+drop2"),
         ("elastic+drop2-6k@k1", "fixed-k1+drop2-6k"), ("elastic+drop2-6k@k1", "fixed-k1+drop-6k"),
         ("elastic+drop-6k@k1", "fixed-k1+drop-6k"), ("fixed-k1+drop2-6k", "fixed-k1+drop-6k"),
         # vòng 4: dropout 0.15
         ("elastic+drop15@k2", "fixed-k2+drop15"), ("elastic+drop15@k1", "fixed-k1+drop15"),
         ("fixed-k2+drop15", "fixed-k2+drop2"), ("fixed-k2+drop15", "fixed-k2+drop"),
         ("fixed-k1+drop15", "fixed-k1+drop"), ("elastic+drop15@k2", "elastic+drop2@k2")]
print("\nso sánh theo cặp (bestB; số âm = hàng đầu tốt hơn, đơn vị nat; chỉ dùng seed có ở cả hai hàng):")
for a, b in pairs:
    if a in by and b in by:
        seeds = sorted(set(by[a]) & set(by[b]))
        if seeds:
            diffs = [by[a][s][1] - by[b][s][1] for s in seeds]
            print(f"  {a} - {b}: {sum(diffs) / len(diffs):+.4f}  (n={len(seeds)}; " +
                  ", ".join(f"s{s} {x:+.4f}" for s, x in zip(seeds, diffs)) + ")")
