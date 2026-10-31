#!/usr/bin/env python3
"""
yui_sandbox.py - mini-sandbox TỰ CHẾ cho model Yui: chạy một tập con của Python để tính KẾT QUẢ CHÍNH XÁC TUYỆT ĐỐI (số nguyên lớn tuỳ ý,
phân số, số nguyên tố, ngày tháng...) mà không cần bubblewrap/Docker, nên chạy được trên Kaggle, Colab, Windows, Android/Termux. Chỉ dùng thư viện chuẩn.

AN TOÀN theo 3 lớp
  1) NGÔN NGỮ: mã được phân tích bằng ast rồi do CHÍNH trình thông dịch này duyệt cây; KHÔNG bao giờ gọi eval/exec/compile.
     Không có import tuỳ ý (chỉ math, fractions, datetime, itertools, functools, collections.Counter, statistics, string), không file, không mạng,
     không class, không try, không truy cập thuộc tính ngoài danh sách cho phép, cấm thuộc tính bắt đầu bằng "_" và tên bắt đầu bằng "__" (chặn __class__, __globals__ ...).
     Tên như open/eval/exec/type/getattr/__import__ đơn giản là KHÔNG TỒN TẠI.
  2) TÀI NGUYÊN (đếm ngay trong trình thông dịch): số bước, thời gian, số bit của số nguyên, độ dài chuỗi/danh sách, độ sâu đệ quy, dung lượng đầu ra.
     Mọi thứ có thể phình to (2**10**9, 'a'*10**10, [0]*10**9, factorial(10**6), list(range(10**12))...) bị chặn TRƯỚC khi tính.
  3) TIẾN TRÌNH: mặc định chạy trong một tiến trình con riêng (python -S -B -s), môi trường rỗng, giới hạn RAM/CPU/file bằng rlimit (Linux/macOS),
     hết giờ thì giết cả nhóm tiến trình. Tiến trình con không nhận gì ngoài mã và giới hạn.
  Đây là lớp bảo vệ thực dụng; khi chạy mã của người lạ ở quy mô lớn vẫn nên đặt thêm container/VM.

DÙNG
  python yui_sandbox.py "print(2**100)"            # chạy một đoạn mã
  python yui_sandbox.py --help-vi                   # bảng tóm tắt cú pháp + hàm có sẵn
  from yui_sandbox import run;  ok, text = run("print(factorial(20))")
Dòng cuối là biểu thức thì được tự in (như REPL):  run("2**64")  ->  18446744073709551616
"""
import sys

if __name__ == "__main__" and "--child" in sys.argv:   # tiến trình con: bỏ thư mục chứa file này khỏi sys.path để tên file trong repo không che thư viện chuẩn
    import os as _os
    _h = _os.path.dirname(_os.path.abspath(__file__))
    sys.path[:] = [p for p in sys.path if _os.path.abspath(p or _os.getcwd()) != _h]

import argparse
import ast
import collections
import datetime
import functools
import itertools
import json
import math
import os
import re
import string as _string
import subprocess
import textwrap
import time
from fractions import Fraction

try:
    import resource
except ImportError:  # Windows
    resource = None

DEFAULT_LIMITS = dict(steps=3_000_000, wall_s=5, mem_mb=256, out_kb=16, max_bits=200_000, max_items=200_000, max_str=1_000_000,
                      depth=100, code_chars=4000, nodes=3000)
_DOW_VI = ["Thứ Hai", "Thứ Ba", "Thứ Tư", "Thứ Năm", "Thứ Sáu", "Thứ Bảy", "Chủ Nhật"]
_MISSING = object()


class SandboxError(Exception):
    """Lỗi do sandbox báo (thông điệp ngắn, tiếng Việt, đưa thẳng cho người dùng/model)."""


class Unsupported(SandboxError):
    pass


class _Stop(Exception):
    pass


class _Break(Exception):
    pass


class _Continue(Exception):
    pass


class _Return(Exception):
    def __init__(self, value):
        self.value = value


# ============================ hiển thị giá trị ============================
def show(v, nested=False):
    """Chuỗi hiển thị của một giá trị: Fraction -> 3/4, ngày -> 2026-10-30, set được sắp xếp để kết quả ổn định."""
    t = type(v)
    if t is str:
        return repr(v) if nested else v
    if t is bool or v is None:
        return str(v)
    if t is int:
        return str(v)
    if t is float:
        return repr(v)
    if t is Fraction:
        return str(v.numerator) if v.denominator == 1 else f"{v.numerator}/{v.denominator}"
    if t is list:
        return "[" + ", ".join(show(x, True) for x in v) + "]"
    if t is tuple:
        return "(" + ", ".join(show(x, True) for x in v) + ("," if len(v) == 1 else "") + ")"
    if t in (set, frozenset):
        try:
            items = sorted(v)
        except TypeError:
            items = list(v)
        return "{" + ", ".join(show(x, True) for x in items) + "}" if items else "set()"
    if t in (dict, collections.Counter):
        body = "{" + ", ".join(f"{show(k, True)}: {show(x, True)}" for k, x in v.items()) + "}"
        return body if t is dict else f"Counter({body})"
    if t is datetime.date:
        return v.isoformat()
    if t is datetime.timedelta:
        return f"{v.days} ngày" if not v.seconds and not v.microseconds else str(v)
    if t is range:
        return f"range({v.start}, {v.stop})" if v.step == 1 else f"range({v.start}, {v.stop}, {v.step})"
    if isinstance(v, UFunc):
        return f"<hàm {v.name}>"
    if isinstance(v, SbxModule):
        return f"<module {v.name}>"
    return "<đối tượng lặp>" if hasattr(v, "__next__") else f"<{t.__name__}>"


class SbxModule:
    def __init__(self, name, members):
        self.name, self.members = name, members


class Scope:
    __slots__ = ("vars", "parent")

    def __init__(self, parent=None):
        self.vars, self.parent = {}, parent

    def lookup(self, name):
        s = self
        while s is not None:
            if name in s.vars:
                return s.vars[name]
            s = s.parent
        raise SandboxError(f"chưa có tên '{name}'")


class UFunc:
    """Hàm do người dùng định nghĩa (def/lambda) bên trong sandbox."""

    def __init__(self, it, name, params, defaults, vararg, body, scope, is_lambda):
        self.it, self.name, self.params, self.defaults, self.vararg, self.body, self.scope, self.is_lambda = it, name, params, defaults, vararg, body, scope, is_lambda

    def __call__(self, *args, **kw):
        return self.it.call_user(self, args, kw)


_NUM = (int, float, bool, Fraction)
_SEQ = (str, list, tuple)
_OPERANDS = (int, float, bool, Fraction, str, list, tuple, set, frozenset, datetime.date, datetime.timedelta, collections.Counter)
_SPEC = re.compile(r"^(.?[<>=^])?[+\- ]?#?0?(\d{1,3})?[,_]?(\.\d{1,3})?[bcdeEfFgGnosxX%]?$")
_FRAC_F = re.compile(r"^\.(\d{1,3})f$")
_CMP = {ast.Eq: lambda a, b: a == b, ast.NotEq: lambda a, b: a != b, ast.Lt: lambda a, b: a < b, ast.LtE: lambda a, b: a <= b,
        ast.Gt: lambda a, b: a > b, ast.GtE: lambda a, b: a >= b, ast.Is: lambda a, b: a is b, ast.IsNot: lambda a, b: a is not b,
        ast.In: lambda a, b: a in b, ast.NotIn: lambda a, b: a not in b}

# thuộc tính / phương thức được phép gọi theo kiểu giá trị (mọi thứ khác bị chặn)
_ATTRS = {
    str: set("capitalize casefold count endswith find index isalnum isalpha isdecimal isdigit islower isnumeric isspace isupper join lower lstrip "
             "partition replace rfind rindex rpartition rsplit rstrip split splitlines startswith strip swapcase title upper zfill ljust rjust center".split()),
    list: set("append clear copy count extend index insert pop remove reverse sort".split()),
    tuple: {"count", "index"},
    dict: set("copy get items keys pop popitem setdefault update values".split()),
    set: set("add clear copy difference discard intersection isdisjoint issubset issuperset pop remove symmetric_difference union update".split()),
    frozenset: set("copy difference intersection isdisjoint issubset issuperset symmetric_difference union".split()),
    int: {"bit_length", "bit_count", "conjugate"},
    float: {"is_integer", "as_integer_ratio"},
    Fraction: {"numerator", "denominator", "limit_denominator", "as_integer_ratio"},
    datetime.date: {"year", "month", "day", "weekday", "isoweekday", "isoformat", "replace", "toordinal"},
    datetime.timedelta: {"days", "seconds", "total_seconds"},
    collections.Counter: {"most_common", "items", "keys", "values", "get"},
    range: {"start", "stop", "step", "index", "count"},
}


class Interp:
    def __init__(self, lim, exact_div=False, cancel=None):
        self.lim, self.exact_div, self.cancel = lim, exact_div, cancel
        self.steps, self._chk, self.depth = 0, 256, 0
        self.t_end = time.monotonic() + lim["wall_s"]
        self.out, self.out_len = [], 0
        self._eh = {getattr(ast, k[2:]): getattr(self, k) for k in dir(self) if k.startswith("e_")}
        self._sh = {getattr(ast, k[2:]): getattr(self, k) for k in dir(self) if k.startswith("s_")}
        self.wrappers = self._make_wrappers()
        self.modules = {}
        self.base = Scope()
        self.g = Scope(self.base)
        self._install()

    # ---------------- giới hạn ----------------
    def _slow(self):
        self._chk = self.steps + 256
        if self.steps > self.lim["steps"]:
            raise SandboxError("quá nhiều bước tính (vòng lặp quá dài?)")
        if time.monotonic() > self.t_end:
            raise SandboxError(f"quá {self.lim['wall_s']}s nên bị dừng")
        if self.cancel is not None and self.cancel.is_set():
            raise SandboxError("đã bị dừng")

    def tick(self, n=1):
        self.steps += n
        if self.steps >= self._chk:
            self._slow()

    def check_num(self, r):
        t = type(r)
        mx = self.lim["max_bits"]
        if t is int:
            if r.bit_length() > mx:
                raise SandboxError(f"số quá lớn (giới hạn khoảng {int(mx * 0.301)} chữ số)")
        elif t is Fraction:
            if r.numerator.bit_length() > mx or r.denominator.bit_length() > mx:
                raise SandboxError("phân số quá lớn")
        return r

    def chk(self, c):
        if len(c) > self.lim["max_items"]:
            raise SandboxError(f"quá nhiều phần tử (tối đa {self.lim['max_items']})")

    def take(self, it):
        out = []
        cap = self.lim["max_items"]
        for v in it:
            self.tick()
            out.append(v)
            if len(out) > cap:
                raise SandboxError(f"quá nhiều phần tử (tối đa {cap})")
        return out

    def ticked(self, it):
        for v in it:
            self.tick()
            yield v

    def emit(self, text):
        cap = self.lim["out_kb"] * 1024
        self.out.append(text)
        self.out_len += len(text)
        if self.out_len > cap:
            joined = "".join(self.out)[:cap] + "…(đầu ra bị cắt)"
            self.out = [joined]
            raise _Stop()

    # ---------------- chạy ----------------
    def run(self, code):
        code = clean_code(code)
        if len(code) > self.lim["code_chars"]:
            raise SandboxError(f"mã quá dài (tối đa {self.lim['code_chars']} ký tự)")
        try:
            tree = ast.parse(code, mode="exec")
        except SyntaxError as e:
            raise SandboxError(f"sai cú pháp: {e.msg} (dòng {e.lineno})")
        except (RecursionError, MemoryError, ValueError):
            raise SandboxError("mã không hợp lệ hoặc lồng nhau quá sâu")
        self._static(tree)
        body = tree.body
        try:
            for i, st in enumerate(body):
                if i == len(body) - 1 and isinstance(st, ast.Expr):
                    v = self.ev(st.value, self.g)
                    if v is not None:
                        self.emit(show(v) + "\n")
                else:
                    self.stmt(st, self.g)
        except (_Break, _Continue, _Return):
            raise SandboxError("break/continue/return đặt sai chỗ")

    def _static(self, tree):
        n = 0
        for x in ast.walk(tree):
            n += 1
            if n > self.lim["nodes"]:
                raise SandboxError("mã quá phức tạp")
            if isinstance(x, ast.Attribute) and x.attr.startswith("_"):
                raise SandboxError("không được dùng tên bắt đầu bằng _")
            if isinstance(x, ast.Name) and x.id.startswith("__"):
                raise SandboxError("không được dùng tên bắt đầu bằng __")
            if isinstance(x, ast.arg) and x.arg.startswith("__"):
                raise SandboxError("không được dùng tên bắt đầu bằng __")
            if isinstance(x, ast.FunctionDef) and x.name.startswith("__"):
                raise SandboxError("không được dùng tên bắt đầu bằng __")

    # ---------------- biểu thức ----------------
    def ev(self, n, sc):
        self.steps += 1
        if self.steps >= self._chk:
            self._slow()
        h = self._eh.get(type(n))
        if h is None:
            raise Unsupported(f"không hỗ trợ cú pháp {type(n).__name__}")
        return h(n, sc)

    def e_Constant(self, n, sc):
        v = n.value
        if type(v) in (int, float, str, bool) or v is None:
            return v
        raise Unsupported("không hỗ trợ kiểu hằng số này")

    def e_Name(self, n, sc):
        return sc.lookup(n.id)

    def e_BinOp(self, n, sc):
        return self.binop(type(n.op), self.ev(n.left, sc), self.ev(n.right, sc))

    def e_UnaryOp(self, n, sc):
        v, op = self.ev(n.operand, sc), type(n.op)
        if op is ast.Not:
            return not v
        if type(v) not in _NUM:
            raise SandboxError(f"không áp dụng được cho kiểu {type(v).__name__}")
        if op is ast.USub:
            return self.check_num(-v)
        if op is ast.UAdd:
            return +v
        if op is ast.Invert and type(v) in (int, bool):
            return ~v
        raise Unsupported("toán tử này")

    def e_BoolOp(self, n, sc):
        is_and = isinstance(n.op, ast.And)
        v = is_and
        for x in n.values:
            v = self.ev(x, sc)
            if is_and and not v:
                return v
            if not is_and and v:
                return v
        return v

    def e_Compare(self, n, sc):
        left = self.ev(n.left, sc)
        for op, c in zip(n.ops, n.comparators):
            right = self.ev(c, sc)
            if type(op) in (ast.In, ast.NotIn):
                try:
                    self.tick(len(right) >> 5)
                except TypeError:
                    pass
            if not _CMP[type(op)](left, right):
                return False
            left = right
        return True

    def e_IfExp(self, n, sc):
        return self.ev(n.body if self.ev(n.test, sc) else n.orelse, sc)

    def seq(self, elts, sc):
        out = []
        for e in elts:
            if isinstance(e, ast.Starred):
                out.extend(self.take(self.ev(e.value, sc)))
            else:
                out.append(self.ev(e, sc))
        self.chk(out)
        return out

    def e_Call(self, n, sc):
        f = self.ev(n.func, sc)
        args = self.seq(n.args, sc)
        kw = {}
        for k in n.keywords:
            if k.arg is None:
                raise Unsupported("không hỗ trợ **kwargs")
            kw[k.arg] = self.ev(k.value, sc)
        if not callable(f):
            raise SandboxError("giá trị này không gọi được như hàm")
        self.tick(2)
        return f(*args, **kw)

    def e_Attribute(self, n, sc):
        obj, name = self.ev(n.value, sc), n.attr
        if name.startswith("_"):
            raise SandboxError("không được dùng tên bắt đầu bằng _")
        if isinstance(obj, SbxModule):
            if name not in obj.members:
                raise SandboxError(f"module {obj.name} không có '{name}'")
            return obj.members[name]
        t = type(obj)
        t = int if t is bool else t
        allowed = _ATTRS.get(t)
        if allowed is None or name not in allowed:
            raise SandboxError(f"không có '.{name}' cho kiểu {t.__name__}")
        w = self.wrappers.get((t, name))
        return functools.partial(w, obj) if w else getattr(obj, name)

    def e_Subscript(self, n, sc):
        c = self.ev(n.value, sc)
        if type(c) not in (list, tuple, str, dict, range, collections.Counter):
            raise SandboxError(f"không lấy phần tử bằng [] với kiểu {type(c).__name__}")
        return c[self.ev(n.slice, sc)]

    def e_Slice(self, n, sc):
        return slice(*(self.ev(x, sc) if x is not None else None for x in (n.lower, n.upper, n.step)))

    def e_List(self, n, sc):
        return self.seq(n.elts, sc)

    def e_Tuple(self, n, sc):
        return tuple(self.seq(n.elts, sc))

    def e_Set(self, n, sc):
        return set(self.seq(n.elts, sc))

    def e_Dict(self, n, sc):
        d = {}
        for k, v in zip(n.keys, n.values):
            if k is None:
                raise Unsupported("không hỗ trợ **")
            d[self.ev(k, sc)] = self.ev(v, sc)
        self.chk(d)
        return d

    def comp(self, gens, sc, elt):
        def rec(i, s):
            if i == len(gens):
                yield elt(s)
                return
            g = gens[i]
            if g.is_async:
                raise Unsupported("async")
            for item in self.ev(g.iter, s):
                self.tick()
                self.assign(g.target, item, s)
                if all(self.ev(c, s) for c in g.ifs):
                    yield from rec(i + 1, s)
        return rec(0, Scope(sc))

    def e_ListComp(self, n, sc):
        return self.take(self.comp(n.generators, sc, lambda s: self.ev(n.elt, s)))

    def e_SetComp(self, n, sc):
        return set(self.take(self.comp(n.generators, sc, lambda s: self.ev(n.elt, s))))

    def e_DictComp(self, n, sc):
        return dict(self.take(self.comp(n.generators, sc, lambda s: (self.ev(n.key, s), self.ev(n.value, s)))))

    def e_GeneratorExp(self, n, sc):
        return self.comp(n.generators, sc, lambda s: self.ev(n.elt, s))

    def _params(self, a, sc, name):
        if a.kwonlyargs or a.kwarg or a.posonlyargs:
            raise Unsupported("chỉ hỗ trợ đối số thường, giá trị mặc định và *args")
        names = [x.arg for x in a.args]
        defaults = {}
        for nm, d in zip(names[len(names) - len(a.defaults):], a.defaults):
            defaults[nm] = self.ev(d, sc)
        return names, defaults, (a.vararg.arg if a.vararg else None)

    def e_Lambda(self, n, sc):
        names, defaults, va = self._params(n.args, sc, "lambda")
        return UFunc(self, "lambda", names, defaults, va, n.body, sc, True)

    def fmt_value(self, x, spec):
        if spec:
            if not _SPEC.match(spec):
                raise SandboxError("định dạng không hợp lệ")
            if type(x) is Fraction:
                m = _FRAC_F.match(spec)
                if m:
                    return self.dec(x, int(m.group(1)))
            return format(x, spec)
        return show(x)

    def e_JoinedStr(self, n, sc):
        parts = []
        for v in n.values:
            if isinstance(v, ast.Constant):
                parts.append(v.value)
                continue
            x = self.ev(v.value, sc)
            spec = self.ev(v.format_spec, sc) if v.format_spec is not None else ""
            if v.conversion == 114:
                x = show(x, True)
            elif v.conversion == 115:
                x = show(x)
            parts.append(self.fmt_value(x, spec))
        s = "".join(parts)
        if len(s) > self.lim["max_str"]:
            raise SandboxError("chuỗi quá dài")
        return s

    # ---------------- toán tử ----------------
    def binop(self, op, a, b):
        ta, tb = type(a), type(b)
        if ta not in _OPERANDS or tb not in _OPERANDS:
            raise SandboxError(f"không tính được {ta.__name__} và {tb.__name__}")
        mx = self.lim["max_bits"]
        ints = ta in (int, bool) and tb in (int, bool)
        if op is ast.Add:
            if ta is str and tb is str:
                if len(a) + len(b) > self.lim["max_str"]:
                    raise SandboxError("chuỗi quá dài")
                return a + b
            if ta in (list, tuple) and ta is tb:
                if len(a) + len(b) > self.lim["max_items"]:
                    raise SandboxError("danh sách quá dài")
                return a + b
            r = a + b
        elif op is ast.Sub:
            r = a - b
        elif op is ast.Mult:
            if ta in _SEQ or tb in _SEQ:
                seq, k = (a, b) if ta in _SEQ else (b, a)
                if type(k) not in (int, bool):
                    raise SandboxError("chỉ nhân chuỗi/danh sách với số nguyên")
                cap = self.lim["max_str"] if type(seq) is str else self.lim["max_items"]
                if len(seq) * max(k, 0) > cap:
                    raise SandboxError("kết quả quá lớn")
                return seq * k
            if ints and a.bit_length() + b.bit_length() > mx + 64:
                raise SandboxError("số quá lớn")
            r = a * b
        elif op is ast.Div:
            if self.exact_div and ints:
                if b == 0:
                    raise ZeroDivisionError("chia cho 0")
                f = Fraction(a, b)
                r = f.numerator if f.denominator == 1 else f
            else:
                r = a / b
        elif op is ast.FloorDiv:
            r = a // b
        elif op is ast.Mod:
            if ta is str:
                raise SandboxError("không dùng % với chuỗi, hãy dùng f-string")
            r = a % b
        elif op is ast.Pow:
            if tb is Fraction and b.denominator == 1:
                b, tb = int(b), int
            if ints:
                if b >= 0 and abs(a) > 1 and a.bit_length() * b > mx:
                    raise SandboxError("luỹ thừa quá lớn")
            elif ta is Fraction and tb in (int, bool):
                if abs(b) > 2000 or (a.numerator.bit_length() + a.denominator.bit_length()) * abs(b) > mx:
                    raise SandboxError("luỹ thừa quá lớn")
            r = a ** b
        elif op in (ast.BitAnd, ast.BitOr, ast.BitXor):
            r = a & b if op is ast.BitAnd else a | b if op is ast.BitOr else a ^ b
        elif op is ast.LShift:
            if not ints or b < 0 or b > mx or a.bit_length() + b > mx:
                raise SandboxError("dịch bit quá lớn")
            r = a << b
        elif op is ast.RShift:
            if not ints or b < 0:
                raise SandboxError("dịch bit không hợp lệ")
            r = a >> min(b, mx + 64)
        else:
            raise Unsupported("toán tử này")
        return self.check_num(r)

    # ---------------- câu lệnh ----------------
    def block(self, stmts, sc):
        for s in stmts:
            self.stmt(s, sc)

    def stmt(self, n, sc):
        self.steps += 1
        if self.steps >= self._chk:
            self._slow()
        h = self._sh.get(type(n))
        if h is None:
            raise Unsupported(f"không hỗ trợ cú pháp {type(n).__name__} (sandbox chỉ hỗ trợ gán, if/for/while, def, return, import)")
        h(n, sc)

    def s_Expr(self, n, sc):
        self.ev(n.value, sc)

    def s_Pass(self, n, sc):
        pass

    def s_Break(self, n, sc):
        raise _Break()

    def s_Continue(self, n, sc):
        raise _Continue()

    def s_Return(self, n, sc):
        raise _Return(self.ev(n.value, sc) if n.value is not None else None)

    def assign(self, t, v, sc):
        if isinstance(t, ast.Name):
            sc.vars[t.id] = v
        elif isinstance(t, (ast.Tuple, ast.List)):
            items = self.take(v)
            if any(isinstance(e, ast.Starred) for e in t.elts):
                raise Unsupported("không hỗ trợ gán có dấu *")
            if len(items) != len(t.elts):
                raise SandboxError(f"gán {len(t.elts)} biến nhưng có {len(items)} giá trị")
            for e, x in zip(t.elts, items):
                self.assign(e, x, sc)
        elif isinstance(t, ast.Subscript):
            c, k = self.ev(t.value, sc), self.ev(t.slice, sc)
            if type(c) not in (list, dict):
                raise SandboxError("chỉ gán được vào phần tử của list/dict")
            c[k] = v
            if type(c) is dict:
                self.chk(c)
        else:
            raise Unsupported("kiểu gán này")

    def s_Assign(self, n, sc):
        v = self.ev(n.value, sc)
        for t in n.targets:
            self.assign(t, v, sc)

    def s_AugAssign(self, n, sc):
        op, t = type(n.op), n.target
        rhs = self.ev(n.value, sc)
        if isinstance(t, ast.Name):
            cur = sc.lookup(t.id)
            if type(cur) is list and op is ast.Add and type(rhs) is list:
                cur.extend(rhs)
                self.chk(cur)
                return
            sc.vars[t.id] = self.binop(op, cur, rhs)
        elif isinstance(t, ast.Subscript):
            c, k = self.ev(t.value, sc), self.ev(t.slice, sc)
            if type(c) not in (list, dict, collections.Counter):
                raise SandboxError("chỉ gán được vào phần tử của list/dict")
            c[k] = self.binop(op, c[k], rhs)
        else:
            raise Unsupported("kiểu gán này")

    def s_If(self, n, sc):
        self.block(n.body if self.ev(n.test, sc) else n.orelse, sc)

    def s_For(self, n, sc):
        broke = False
        for item in self.ev(n.iter, sc):
            self.tick()
            self.assign(n.target, item, sc)
            try:
                self.block(n.body, sc)
            except _Break:
                broke = True
                break
            except _Continue:
                continue
        if not broke and n.orelse:
            self.block(n.orelse, sc)

    def s_While(self, n, sc):
        broke = False
        while self.ev(n.test, sc):
            self.tick()
            try:
                self.block(n.body, sc)
            except _Break:
                broke = True
                break
            except _Continue:
                continue
        if not broke and n.orelse:
            self.block(n.orelse, sc)

    def s_FunctionDef(self, n, sc):
        if n.decorator_list:
            raise Unsupported("không hỗ trợ decorator")
        names, defaults, va = self._params(n.args, sc, n.name)
        sc.vars[n.name] = UFunc(self, n.name, names, defaults, va, n.body, sc, False)

    def call_user(self, f, args, kw):
        self.tick(3)
        self.depth += 1
        try:
            if self.depth > self.lim["depth"]:
                raise SandboxError("đệ quy quá sâu")
            sc = Scope(f.scope)
            n = len(f.params)
            if len(args) > n and not f.vararg:
                raise SandboxError(f"{f.name}() nhận tối đa {n} đối số")
            for p, a in zip(f.params, args):
                sc.vars[p] = a
            if f.vararg:
                sc.vars[f.vararg] = tuple(args[n:])
            for k, v in kw.items():
                if k not in f.params:
                    raise SandboxError(f"{f.name}() không có đối số '{k}'")
                if k in sc.vars:
                    raise SandboxError(f"{f.name}() nhận đối số '{k}' hai lần")
                sc.vars[k] = v
            for p in f.params:
                if p not in sc.vars:
                    if p not in f.defaults:
                        raise SandboxError(f"{f.name}() thiếu đối số '{p}'")
                    sc.vars[p] = f.defaults[p]
            if f.is_lambda:
                return self.ev(f.body, sc)
            try:
                self.block(f.body, sc)
            except _Return as r:
                return r.value
            except (_Break, _Continue):
                raise SandboxError("break/continue ngoài vòng lặp")
            return None
        finally:
            self.depth -= 1

    def s_Import(self, n, sc):
        for a in n.names:
            mod = self.modules.get(a.name)
            if mod is None:
                raise SandboxError(f"không được import '{a.name}' (chỉ có: {', '.join(sorted(self.modules))})")
            sc.vars[a.asname or a.name] = mod

    def s_ImportFrom(self, n, sc):
        mod = self.modules.get(n.module or "") if not n.level else None
        if mod is None:
            raise SandboxError(f"không được import từ '{n.module}' (chỉ có: {', '.join(sorted(self.modules))})")
        for a in n.names:
            if a.name == "*":
                sc.vars.update(mod.members)
            elif a.name in mod.members:
                sc.vars[a.asname or a.name] = mod.members[a.name]
            else:
                raise SandboxError(f"module {mod.name} không có '{a.name}'")

    # ---------------- phương thức có kiểm soát kích thước ----------------
    def _make_wrappers(self):
        def w_join(s, it):
            parts = self.take(it)
            if not all(type(p) is str for p in parts):
                raise SandboxError("join cần danh sách chuỗi")
            if sum(map(len, parts)) + len(s) * len(parts) > self.lim["max_str"]:
                raise SandboxError("chuỗi quá dài")
            return s.join(parts)

        def w_replace(s, old, new, *a):
            if (len(s) + 1) * max(1, len(new)) > self.lim["max_str"]:
                raise SandboxError("chuỗi quá dài")
            return s.replace(old, new, *a)

        def w_pad(name):
            def w(s, width, *a):
                if type(width) is not int or width > 1000:
                    raise SandboxError("độ rộng tối đa 1000")
                return getattr(s, name)(width, *a)
            return w

        def w_append(c, x):
            c.append(x)
            self.chk(c)

        def w_insert(c, i, x):
            c.insert(i, x)
            self.chk(c)

        def w_extend(c, it):
            c.extend(self.take(it))
            self.chk(c)

        def w_add(c, x):
            c.add(x)
            self.chk(c)

        def w_supdate(c, *its):
            for it in its:
                c.update(self.take(it))
            self.chk(c)

        def w_dupdate(c, *a, **kw):
            c.update(*[x if isinstance(x, dict) else self.take(x) for x in a], **kw)
            self.chk(c)

        def w_setdefault(c, k, v=None):
            r = c.setdefault(k, v)
            self.chk(c)
            return r

        def w_setop(name):
            def w(c, *its):
                return getattr(c, name)(*[self.take(i) for i in its])
            return w

        w = {(str, "join"): w_join, (str, "replace"): w_replace, (list, "append"): w_append, (list, "insert"): w_insert, (list, "extend"): w_extend,
             (set, "add"): w_add, (set, "update"): w_supdate, (dict, "update"): w_dupdate, (dict, "setdefault"): w_setdefault}
        for nm in ("zfill", "ljust", "rjust", "center"):
            w[(str, nm)] = w_pad(nm)
        for t in (set, frozenset):
            for nm in ("union", "intersection", "difference", "symmetric_difference", "issubset", "issuperset", "isdisjoint"):
                w[(t, nm)] = w_setop(nm)
        return w

    # ---------------- hàm dựng sẵn ----------------
    def _int(self, x, name="đối số"):
        if type(x) not in (int, bool):
            raise SandboxError(f"{name} phải là số nguyên")
        return int(x)

    def isprime(self, n):
        n = self._int(n)
        if n < 2:
            return False
        small = (2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41)
        for p in small:
            if n % p == 0:
                return n == p
        if n.bit_length() > 4096:
            raise SandboxError("số quá lớn cho isprime")
        d, s = n - 1, 0
        while d % 2 == 0:
            d //= 2
            s += 1
        bases = small if n < 3317044064679887385961981 else small + (43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97)
        self.tick(max(1, n.bit_length() // 4))
        for a in bases:
            x = pow(a, d, n)
            if x in (1, n - 1):
                continue
            for _ in range(s - 1):
                x = x * x % n
                if x == n - 1:
                    break
            else:
                return False
        return True

    def _rho(self, n):
        if n % 2 == 0:
            return 2
        c = 1
        while True:
            x = y = 2
            d = 1
            while d == 1:
                self.tick()
                x = (x * x + c) % n
                y = (y * y + c) % n
                y = (y * y + c) % n
                d = math.gcd(abs(x - y), n)
            if d != n:
                return d
            c += 1

    def factor(self, n):
        n = self._int(n)
        if n < 1 or n.bit_length() > 128:
            raise SandboxError("factor(n) cần 1 <= n < 2**128")
        out, p = [], 2
        while p < 10000 and p * p <= n:
            self.tick()
            while n % p == 0:
                out.append(p)
                n //= p
            p += 1 if p == 2 else 2
        stack = [n] if n > 1 else []
        while stack:
            m = stack.pop()
            if m < 10 ** 8 or self.isprime(m):   # sau khi vét ước < 10^4, số < 10^8 chắc chắn là nguyên tố
                out.append(m)
            else:
                d = self._rho(m)
                stack += [d, m // d]
        return sorted(out)

    def primes(self, n):
        n = self._int(n)
        if n < 0 or n > 2_000_000:
            raise SandboxError("primes(n) cần 0 <= n <= 2000000")
        if n < 2:
            return []
        self.tick(n >> 6)
        sieve = bytearray([1]) * (n + 1)
        sieve[0] = sieve[1] = 0
        for i in range(2, int(n ** 0.5) + 1):
            if sieve[i]:
                sieve[i * i::i] = bytearray(len(range(i * i, n + 1, i)))
        return [i for i in range(n + 1) if sieve[i]]

    def nextprime(self, n):
        n = self._int(n) + 1
        while not self.isprime(n):
            self.tick()
            n += 1
        return max(n, 2)

    def fib(self, n):
        n = self._int(n)
        if n < 0 or n > 300_000:
            raise SandboxError("fib(n) cần 0 <= n <= 300000")

        def fd(k):
            if k == 0:
                return 0, 1
            a, b = fd(k >> 1)
            c, d = a * (2 * b - a), a * a + b * b
            return (d, c + d) if k & 1 else (c, d)
        return self.check_num(fd(n)[0])

    def divisors(self, n):
        n = self._int(n)
        if n < 1:
            raise SandboxError("divisors(n) cần n >= 1")
        divs = [1]
        fs = self.factor(n)
        for p in sorted(set(fs)):
            e = fs.count(p)
            divs = [d * p ** k for d in divs for k in range(e + 1)]
            self.chk(divs)
        return sorted(divs)

    def digits(self, n):
        n = self._int(n)
        return [int(c) for c in str(abs(n))]

    def dec(self, x, n=10):
        """Khai triển thập phân CHÍNH XÁC của x tới n chữ số (làm tròn nửa lên), trả chuỗi: dec(Fraction(1,7), 10) -> '0.1428571429'."""
        n = self._int(n, "n")
        if n < 0 or n > 2000:
            raise SandboxError("dec(x, n) cần 0 <= n <= 2000")
        f = Fraction(x)
        neg, f = f < 0, abs(f)
        scaled = (f.numerator * 10 ** n * 2 + f.denominator) // (2 * f.denominator)
        s = str(scaled).zfill(n + 1)
        res = s[:len(s) - n] + ("." + s[len(s) - n:] if n else "")
        return ("-" if neg and scaled else "") + res

    def mean(self, it):
        xs = self.take(it)
        if not xs:
            raise SandboxError("mean của danh sách rỗng")
        s = sum(xs)
        return s // len(xs) if all(type(x) is int for x in xs) and s % len(xs) == 0 else s / len(xs)

    def median(self, it):
        xs = sorted(self.take(it))
        if not xs:
            raise SandboxError("median của danh sách rỗng")
        m = len(xs) // 2
        return xs[m] if len(xs) % 2 else (xs[m - 1] + xs[m]) / 2

    def _install(self):
        T = self.take

        def b_print(*a, sep=" ", end="\n"):
            self.emit(sep.join(show(x) for x in a) + end)

        def b_sum(it, start=0):
            r = start
            for v in self.ticked(it):
                r = self.binop(ast.Add, r, v)
            return r

        def b_minmax(f):
            def g(*a, **kw):
                return f(T(a[0]) if len(a) == 1 else list(a), **kw)
            return g

        def b_pow(a, b, m=None):
            if m is None:
                return self.binop(ast.Pow, a, b)
            if type(b) in (int, bool) and b.bit_length() > self.lim["max_bits"]:
                raise SandboxError("số mũ quá lớn")
            return pow(a, b, m)

        def b_int(x=0, base=None):
            r = int(x) if base is None else int(x, base)
            return self.check_num(r)

        def b_fraction(*a):
            if a and type(a[0]) is str and (len(a[0]) > 200 or "e" in a[0].lower()):
                raise SandboxError("Fraction(chuỗi) không nhận số mũ hoặc chuỗi quá dài")
            return self.check_num(Fraction(*a))

        def b_fact(n):
            n = self._int(n)
            if n < 0 or n > 20000:
                raise SandboxError("factorial(n) cần 0 <= n <= 20000")
            self.tick(n >> 4)
            return self.check_num(math.factorial(n))

        def b_comb(n, k):
            n, k = self._int(n), self._int(k)
            if n < 0 or k < 0 or n > 20000:
                raise SandboxError("comb(n, k) cần 0 <= k, n <= 20000")
            self.tick(n >> 4)
            return self.check_num(math.comb(n, k))

        def b_perm(n, k=None):
            n = self._int(n)
            if n < 0 or n > 20000:
                raise SandboxError("perm(n, k) cần 0 <= n <= 20000")
            self.tick(n >> 4)
            return self.check_num(math.perm(n) if k is None else math.perm(n, self._int(k)))

        def b_prod(it, start=1):
            r = start
            for v in self.ticked(it):
                r = self.binop(ast.Mult, r, v)
            return r

        def b_format(v, spec=""):
            return self.fmt_value(v, spec)

        def b_td(days=0, weeks=0, hours=0, minutes=0, seconds=0):
            if any(abs(x) > 10 ** 7 for x in (days, weeks, hours, minutes, seconds)):
                raise SandboxError("khoảng thời gian quá lớn")
            return datetime.timedelta(days=days, weeks=weeks, hours=hours, minutes=minutes, seconds=seconds)

        def b_days(n):
            return b_td(days=n)

        def b_dow(d):
            if type(d) is not datetime.date:
                raise SandboxError("dow(d) cần một ngày: dow(date(2026, 10, 30))")
            return _DOW_VI[d.weekday()]

        def b_reduce(f, it, *init):
            r = _MISSING if not init else init[0]
            for v in self.ticked(it):
                r = v if r is _MISSING else f(r, v)
            if r is _MISSING:
                raise SandboxError("reduce của danh sách rỗng")
            return r

        def b_counter(it=()):
            return collections.Counter(T(it))

        def b_product(*its, repeat=1):
            return itertools.product(*[T(i) for i in its], repeat=repeat)

        def b_perms(it, r=None):
            return itertools.permutations(T(it), r)

        def b_combs(it, r):
            return itertools.combinations(T(it), r)

        def b_combs_r(it, r):
            return itertools.combinations_with_replacement(T(it), r)

        def b_chain(*its):
            return itertools.chain(*its)

        def b_accumulate(it, func=None, initial=None):
            return itertools.accumulate(it, func, initial=initial)

        def b_islice(it, *a):
            return itertools.islice(it, *a)

        def b_count(start=0, step=1):
            return itertools.count(start, step)

        def b_round(x, nd=None):
            if type(x) not in _NUM:
                raise SandboxError("round cần một số")
            return round(x) if nd is None else round(x, nd)

        def b_chr(i):
            if type(i) not in (int, bool) or not 0 <= i < 0x110000:
                raise SandboxError("chr(i) cần 0 <= i < 1114112")
            return chr(i)

        def b_range(*a):
            if not a or any(type(x) not in (int, bool) for x in a):
                raise SandboxError("range chỉ nhận số nguyên")
            return range(*a)

        def b_bits(f):
            def g(x):
                if type(x) not in (int, bool):
                    raise SandboxError("cần số nguyên")
                return f(x)
            return g

        b = {
            "print": b_print, "sum": b_sum, "min": b_minmax(min), "max": b_minmax(max), "pow": b_pow, "int": b_int, "Fraction": b_fraction,
            "len": len, "abs": abs, "divmod": divmod, "round": b_round, "sorted": lambda it, key=None, reverse=False: sorted(T(it), key=key, reverse=reverse),
            "reversed": reversed, "enumerate": enumerate, "zip": zip, "map": map, "filter": filter, "range": b_range,
            "list": lambda it=(): T(it), "tuple": lambda it=(): tuple(T(it)), "set": lambda it=(): set(T(it)),
            "frozenset": lambda it=(): frozenset(T(it)), "dict": lambda *a, **kw: dict(*[x if isinstance(x, dict) else T(x) for x in a], **kw),
            "str": lambda x="": show(x), "repr": lambda x: show(x, True), "float": lambda x=0.0: float(x), "bool": lambda x=False: bool(x),
            "chr": b_chr, "ord": lambda c: ord(c), "bin": b_bits(bin), "hex": b_bits(hex), "oct": b_bits(oct),
            "any": lambda it: any(self.ticked(it)), "all": lambda it: all(self.ticked(it)), "format": b_format,
            "True": True, "False": False, "None": None,
            # tiện cho model nhỏ: gọi thẳng không cần import
            "factorial": b_fact, "comb": b_comb, "perm": b_perm, "gcd": math.gcd, "lcm": math.lcm, "isqrt": math.isqrt,
            "isprime": self.isprime, "factor": self.factor, "primes": self.primes, "nextprime": self.nextprime, "fib": self.fib,
            "divisors": self.divisors, "digits": self.digits, "dec": self.dec, "mean": self.mean, "median": self.median,
            "date": datetime.date, "days": b_days, "dow": b_dow, "Counter": b_counter,
        }
        mathm = {"sqrt": math.sqrt, "isqrt": math.isqrt, "gcd": math.gcd, "lcm": math.lcm, "factorial": b_fact, "comb": b_comb, "perm": b_perm,
                 "floor": math.floor, "ceil": math.ceil, "trunc": math.trunc, "fabs": math.fabs, "log": math.log, "log2": math.log2, "log10": math.log10,
                 "exp": math.exp, "sin": math.sin, "cos": math.cos, "tan": math.tan, "asin": math.asin, "acos": math.acos, "atan": math.atan,
                 "atan2": math.atan2, "sinh": math.sinh, "cosh": math.cosh, "tanh": math.tanh, "hypot": math.hypot, "degrees": math.degrees,
                 "radians": math.radians, "fsum": lambda it: math.fsum(self.ticked(it)), "prod": b_prod, "pow": math.pow, "copysign": math.copysign,
                 "isclose": math.isclose, "isfinite": math.isfinite, "isnan": math.isnan, "pi": math.pi, "e": math.e, "tau": math.tau,
                 "inf": math.inf, "nan": math.nan}
        for k in ("sqrt", "floor", "ceil", "log", "log2", "log10", "exp", "sin", "cos", "tan", "pi", "e"):
            b[k] = mathm[k]
        self.base.vars.update(b)
        self.modules = {
            "math": SbxModule("math", mathm),
            "fractions": SbxModule("fractions", {"Fraction": b_fraction}),
            "datetime": SbxModule("datetime", {"date": datetime.date, "timedelta": b_td}),
            "itertools": SbxModule("itertools", {"product": b_product, "permutations": b_perms, "combinations": b_combs,
                                                  "combinations_with_replacement": b_combs_r, "chain": b_chain, "accumulate": b_accumulate,
                                                  "islice": b_islice, "count": b_count}),
            "functools": SbxModule("functools", {"reduce": b_reduce}),
            "collections": SbxModule("collections", {"Counter": b_counter}),
            "statistics": SbxModule("statistics", {"mean": self.mean, "median": self.median}),
            "string": SbxModule("string", {"ascii_lowercase": _string.ascii_lowercase, "ascii_uppercase": _string.ascii_uppercase,
                                            "ascii_letters": _string.ascii_letters, "digits": _string.digits}),
        }


# ============================ chạy ============================
def clean_code(code):
    """Làm sạch mã do model viết: bỏ hàng rào ```, thụt lề thừa, ngoặc kép cong, khoảng trắng lạ."""
    code = str(code).replace("\u00a0", " ").replace("\u200b", "")
    code = code.replace("“", '"').replace("”", '"').replace("‘", "'").replace("’", "'")
    code = re.sub(r"^\s*```[A-Za-z0-9_-]*[ \t]*\n?", "", code)
    code = re.sub(r"\n?[ \t]*```\s*$", "", code)
    return textwrap.dedent(code).strip("\n").strip()


def _err_text(e):
    if isinstance(e, SandboxError):
        return str(e)
    if isinstance(e, ZeroDivisionError):
        return "chia cho 0"
    if isinstance(e, OverflowError):
        return "số quá lớn để tính"
    if isinstance(e, RecursionError):
        return "đệ quy quá sâu"
    if isinstance(e, MemoryError):
        return "hết bộ nhớ"
    return f"{type(e).__name__}: {str(e)[:120]}"


def execute(code, limits=None, exact_div=False, cancel=None):
    """Chạy NGAY TRONG tiến trình này (không cách ly bằng tiến trình con). cancel: threading.Event, set() thì dừng. Trả (ok, văn bản)."""
    lim = dict(DEFAULT_LIMITS)
    lim.update(limits or {})
    it = Interp(lim, exact_div, cancel)
    old_rec = sys.getrecursionlimit()
    sys.setrecursionlimit(max(old_rec, 12000))
    old_digits = sys.get_int_max_str_digits() if hasattr(sys, "get_int_max_str_digits") else None
    if old_digits is not None:
        sys.set_int_max_str_digits(70000)
    ok, err = True, ""
    try:
        it.run(code)
    except _Stop:
        pass
    except Exception as e:  # noqa: BLE001 - mọi lỗi của mã người dùng đều thành thông báo ngắn
        ok, err = False, _err_text(e)
    finally:
        sys.setrecursionlimit(old_rec)
        if old_digits is not None:
            sys.set_int_max_str_digits(old_digits)
    out = "".join(it.out).strip("\n")
    if ok:
        return True, out or "(không in gì: hãy dùng print(...))"
    return False, (out + "\n" if out else "") + f"lỗi: {err}"


def _apply_rlimits(lim):
    if resource is None:
        return
    mem = (lim["mem_mb"] + 64) * 1024 * 1024
    for res, val in ((resource.RLIMIT_AS, mem), (resource.RLIMIT_CPU, int(lim["wall_s"]) + 2), (resource.RLIMIT_FSIZE, 0),
                     (resource.RLIMIT_NOFILE, 32), (resource.RLIMIT_CORE, 0)):
        try:
            resource.setrlimit(res, (val, val))
        except (ValueError, OSError):
            pass


def _child_main():
    data = json.loads(sys.stdin.buffer.read(1 << 20).decode("utf-8"))
    _apply_rlimits(data["limits"])
    ok, text = execute(data["code"], data["limits"], data.get("exact_div", False))
    sys.stdout.write(json.dumps({"ok": ok, "text": text}, ensure_ascii=False))
    sys.stdout.flush()


def run(code, limits=None, exact_div=False, isolate=True, python=None, cancel=None):
    """Chạy mã trong sandbox. isolate=True (mặc định): tiến trình con riêng có rlimit + giết khi quá giờ hoặc khi cancel.set(). Trả (ok, văn bản)."""
    if not isolate:
        return execute(code, limits, exact_div, cancel)
    lim = dict(DEFAULT_LIMITS)
    lim.update(limits or {})
    if len(str(code)) > lim["code_chars"] * 3:
        return False, "lỗi: mã quá dài"
    payload = json.dumps({"code": str(code), "limits": lim, "exact_div": bool(exact_div)}).encode("utf-8")
    cmd = [python or sys.executable or "python3", "-S", "-B", "-s", os.path.abspath(__file__), "--child"]
    env = {"PYTHONHASHSEED": "0", "PYTHONUTF8": "1", "PYTHONIOENCODING": "utf-8", "LANG": "C.UTF-8"}
    if os.environ.get("SYSTEMROOT"):
        env["SYSTEMROOT"] = os.environ["SYSTEMROOT"]
    try:
        p = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, env=env, cwd=os.path.dirname(os.path.abspath(__file__)) or None,
                             start_new_session=(os.name != "nt"))
    except OSError as e:
        return False, f"lỗi: không chạy được sandbox ({e})"
    def kill():
        try:
            os.killpg(p.pid, 9)
        except (ProcessLookupError, PermissionError, OSError, AttributeError):
            p.kill()
        p.communicate()

    deadline, first = time.monotonic() + lim["wall_s"] + 3, True
    while True:
        try:
            out, _ = p.communicate(payload if first else None, timeout=0.1)
            break
        except subprocess.TimeoutExpired:
            first = False
            if cancel is not None and cancel.is_set():
                kill()
                return False, "lỗi: đã bị dừng"
            if time.monotonic() > deadline:
                kill()
                return False, f"lỗi: quá {lim['wall_s']}s nên bị dừng"
    try:
        r = json.loads(out.decode("utf-8", "replace").strip().splitlines()[-1])
        return bool(r["ok"]), str(r["text"])
    except (ValueError, IndexError, KeyError):
        why = "hết bộ nhớ hoặc bị hệ điều hành dừng" if p.returncode and p.returncode < 0 else "thoát bất thường"
        return False, f"lỗi: sandbox {why}"


HELP_VI = """Mini-sandbox Yui: Python rút gọn, chạy an toàn, kết quả chính xác tuyệt đối.
Viết: print(...)  hoặc để biểu thức ở dòng cuối (tự in). Mỗi tool nên là MỘT dòng, các lệnh cách nhau bằng dấu ;
Có sẵn (không cần import): factorial comb perm gcd lcm isqrt sqrt floor ceil log log2 log10 exp sin cos tan pi e
  isprime(n) factor(n) primes(n) nextprime(n) fib(n) divisors(n) digits(n) dec(x, n) mean median Fraction(a, b)
  date(y, m, d) days(n) dow(d) Counter sum min max sorted len range enumerate zip map filter any all abs round divmod pow bin hex oct int str float
Cú pháp: gán, if/elif/else, for, while, def, lambda, return, list/dict/set comprehension, f-string, import math/fractions/datetime/itertools/functools/collections/statistics/string
Số nguyên lớn tuỳ ý (đến ~60000 chữ số). Chia chính xác: dùng Fraction(a, b) hoặc dec(Fraction(a, b), 20); chạy với exact_div=True thì a/b của hai số nguyên ra phân số.
KHÔNG có: file, mạng, class, try, import khác, thuộc tính bắt đầu bằng _, tên bắt đầu bằng __, open/eval/exec/getattr/type."""


def main(argv=None):
    ap = argparse.ArgumentParser(description="Mini-sandbox Yui", add_help=True)
    ap.add_argument("code", nargs="*")
    ap.add_argument("--child", action="store_true", help=argparse.SUPPRESS)
    ap.add_argument("--no-isolate", action="store_true", help="chạy trong chính tiến trình này (không dùng tiến trình con)")
    ap.add_argument("--exact-div", action="store_true")
    ap.add_argument("--wall", type=float, default=None)
    ap.add_argument("--mem", type=int, default=None)
    ap.add_argument("--steps", type=int, default=None)
    ap.add_argument("--help-vi", action="store_true")
    a = ap.parse_args(argv)
    if a.child:
        return _child_main()
    if a.help_vi or not a.code:
        print(HELP_VI)
        return
    lim = {k: v for k, v in (("wall_s", a.wall), ("mem_mb", a.mem), ("steps", a.steps)) if v is not None}
    ok, text = run(" ".join(a.code), lim, a.exact_div, isolate=not a.no_isolate)
    print(text)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
