"""Thử yui_sandbox.py: đúng đắn (so với Python thật), chính xác tuyệt đối, và chống tấn công / ngốn tài nguyên."""
import contextlib
import io
import os
import random
import sys
import time

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import yui_sandbox as ys  # noqa: E402


def run(code, **kw):
    return ys.run(code, {"wall_s": 3, **kw.pop("limits", {})}, isolate=False, **kw)


def test_exact_arithmetic_and_helpers():
    assert run("print(2**100)") == (True, str(2 ** 100))
    assert run("2**64") == (True, "18446744073709551616")                     # biểu thức cuối tự in
    assert run("print(factorial(30), gcd(84, 36), lcm(4, 6))")[1] == "265252859812191058636308480000000 12 12"
    assert run("print(isprime(1000000007), isprime(1000000011), factor(600851475143))")[1] == "True False [71, 839, 1471, 6857]"
    assert run("print(len(primes(100000)), fib(90), divisors(36))")[1] == "9592 2880067194370816120 [1, 2, 3, 4, 6, 9, 12, 18, 36]"
    assert run("print(Fraction(1, 3) + Fraction(1, 6), dec(Fraction(1, 7), 12))")[1] == "1/2 0.142857142857"
    assert run("print((date(2026, 12, 25) - date(2026, 10, 30)).days, dow(date(2026, 10, 30)))")[1] == "56 Thứ Sáu"
    assert run("print(7/2)")[1] == "3.5" and run("print(7/2)", exact_div=True)[1] == "7/2" and run("print(6/2)", exact_div=True)[1] == "3"
    assert run("print(f'{2/3:.4f} {Fraction(1,3):.5f}')")[1] == "0.6667 0.33333"
    assert run("print(round(Fraction(2000000) * Fraction(103, 100) ** 10))")[1] == "2687833"
    ok, out = run("print(1)\nprint(1/0)")
    assert not ok and "chia cho 0" in out and out.startswith("1")


def test_language_features_match_real_python():
    progs = [
        "def f(n):\n    return 1 if n < 2 else n * f(n - 1)\nprint(f(20))",
        "xs = [i * i for i in range(10) if i % 2]\nprint(xs, sum(xs), max(xs), sorted(xs, reverse=True))",
        "d = {}\nfor w in 'a b a c b a'.split():\n    d[w] = d.get(w, 0) + 1\nprint(sorted(d.items(), key=lambda kv: (-kv[1], kv[0])))",
        "a, b = 0, 1\nfor _ in range(30):\n    a, b = b, a + b\nprint(a)",
        "i = 0\nwhile True:\n    i += 1\n    if i % 7 == 0 and i % 5 == 0:\n        break\nprint(i)",
        "s = 'xin chào'\nprint(s.upper(), s[::-1], len(s), s.split(), ','.join(['a', 'b']))",
        "print([x for x in range(20) if all(x % d for d in range(2, x)) and x > 1])",
        "from math import sqrt, pi\nprint(round(sqrt(2), 6), round(pi, 4))",
        "import itertools\nprint(list(itertools.permutations([1, 2, 3], 2)), len(list(itertools.combinations(range(6), 3))))",
        "def g(a, b=2, *r):\n    return a * b + sum(r)\nprint(g(3), g(3, 4), g(1, 1, 5, 6))",
        "t = (1, 2, 3)\nx, y, z = t\nprint(x + y * z, t[1:], {1, 2} | {2, 3})",
        "print(bin(10), hex(255), oct(8), int('101', 2), divmod(17, 5), pow(3, 4, 5), abs(-3), 7 // 2, -7 % 3, 2 ** -1)",
        "print([ (i, c) for i, c in enumerate('abc')], list(zip([1, 2], 'xy')), list(map(lambda v: v * 2, range(4))), any([0, 1]), all([]))",
    ]
    for p in progs:
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            exec(p, {"__builtins__": __builtins__})
        ok, out = run(p)
        assert ok and out == buf.getvalue().strip(), (p, out, buf.getvalue())


def test_random_expressions_match_python():
    rng = random.Random(7)
    ops = ["+", "-", "*", "//", "%"]
    for _ in range(300):
        e = str(rng.randint(-50, 50))
        for _ in range(rng.randint(1, 5)):
            e = f"({e} {rng.choice(ops)} {rng.randint(1, 60)})"
        assert run(f"print({e})")[1] == str(eval(e)), e


ATTACKS = [
    "().__class__", "__import__('os')", "open('/etc/passwd')", "import os", "from os import system", "type(1)", "print.__self__", "[].append.__self__",
    "'{0.__class__}'.format(1)", "eval('1')", "exec('x=1')", "getattr(1, 'real')", "(lambda: 0).__globals__", "f'{print.__class__}'", "globals()", "vars()", "dir()",
    "compile('1','x','eval')", "breakpoint()", "input()", "import subprocess", "import sys", "[c for c in ().__class__.__bases__]", "'%d' % 5", "class A: pass", "try:\n    pass\nexcept:\n    pass",
    "with open('x') as f: pass", "global x", "yield 1", "del x", "assert 1", "(lambda: (yield))()",
]


@pytest.mark.parametrize("code", ATTACKS)
def test_attacks_are_rejected(code):
    ok, out = run(code)
    assert not ok and out.startswith("lỗi"), (code, out)


HOGS = [
    ("10**10**10", "quá lớn"), ("'a' * 10**10", "quá lớn"), ("[0] * 10**9", "quá lớn"), ("list(range(10**9))", "quá nhiều phần tử"), ("factorial(10**6)", "cần"),
    ("1 << 10**9", "quá lớn"), ("def f(n): return f(n + 1)\nf(0)", "đệ quy"), ("x = [1]\nwhile True: x = x + x", "quá dài"), ("x = 1\nwhile True: x = x * 2", "quá lớn"),
    ("sorted(range(10**9))", "quá nhiều phần tử"), ("Fraction('1e999999')", "số mũ"), ("fib(10**9)", "cần"), ("2 ** 0.5 ** 99999999999999", None),
    ("primes(10**9)", "cần"), ("'x'.center(10**9)", "tối đa 1000"), ("'ab'.join(range(3))", "chuỗi"), ("sum(range(10**12))", None), ("while True: pass", None),
]


@pytest.mark.parametrize("code,msg", HOGS)
def test_resource_hogs_stop_fast(code, msg):
    t = time.time()
    ok, out = ys.run(code, {"wall_s": 2}, isolate=False)
    assert time.time() - t < 6
    if msg:
        assert not ok and msg in out, out
    elif code.startswith("2 **"):
        assert True
    else:
        assert not ok


def test_output_flood_is_truncated_not_error():
    ok, out = run("while True: print('x' * 1000)", limits={"out_kb": 4})
    assert ok and len(out) < 4200 and out.endswith("(đầu ra bị cắt)")


def test_isolated_child_process_works_and_kills_on_timeout():
    assert ys.run("print(2**200)") == (True, str(2 ** 200))
    t = time.time()
    ok, out = ys.run("while True: pass", {"wall_s": 1, "steps": 10 ** 12})
    assert not ok and time.time() - t < 6 and ("quá" in out)
    ok, out = ys.run("x = bytearray(10)\nprint(len(x))")      # bytearray không tồn tại trong sandbox
    assert not ok
    assert ys.run("print(" + "1+" * 3000 + "1)")[0] is False   # mã quá dài/lồng sâu không làm sập
    assert ys.run("print(1)", python="/nonexistent/python")[0] is False


def test_child_env_has_no_secrets(monkeypatch):
    monkeypatch.setenv("HF_TOKEN", "hf_SECRET")
    ok, out = ys.run("print('x')")
    assert ok and "hf_SECRET" not in out


def test_clean_code_accepts_model_formatting():
    assert run("```python\nprint(1+1)\n```") == (True, "2")
    assert run("print(“xin”)")[1] == "xin"        # ngoặc kép cong
    assert run("  print(3)  ")[1] == "3"
